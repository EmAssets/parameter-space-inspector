"""
Sparse autoencoder (SAE) features for the inside model: named, searchable directions in a layer's state instead of
raw neuron ids. One SAE per layer, kept in <data folder>/sae/<model key>/L<layer>/:

  sae.safetensors   W_enc [d, m], b_enc [m], W_dec [m, d], b_dec [d] (and threshold [m] for a JumpReLU SAE)
  cfg.json          {"kind": "topk" | "jumprelu" | "relu", "k", "d_in", "d_sae", "layer", "model", "weights_hash", "trained" | "imported"}
  features.json     {"freq": [m], "max_act": [m], "exemplars": {"<id>": [[context, act, index of the token in it], ...]}, "labels": {"<id>": {...}}}

The SAE reads the residual stream after layer L (hidden_states[L], the same state the logit lens reads). SAEs come
from tools/train_sae.py (trained here, on this model) or are imported from a folder in one of three formats:
this one, EleutherAI's sparsify (cfg.json with k; encoder.weight [m, d], encoder.bias, W_dec [m, d], b_dec) and
SAE Lens / Gemma Scope (W_enc [d, m], b_enc, W_dec [m, d], b_dec, optional threshold).

What this offers the app (trace_server.py /sae):
  list, encode (which features fire on each piece of a text), feature (one feature's card: how often it fires, its
  strongest examples, the words its direction promotes and suppresses, its name), search (by name, example text or
  output word, and the features a phrase fires), steer (add a feature's direction while the model answers, beside the
  plain answer), attribute (for the next word: how much each feature at each layer pushed it, by activation x gradient,
  and how much the SAE leaves unexplained), label (a name for a feature, kept with it), import.
"""
import json, math, os, re, time
import torch

FMT = 1


def model_key(model_dir, weights_hash):
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(os.path.normpath(model_dir)))[:60] or "model"
    return f"{base}-{str(weights_hash or 'nohash')[:12]}"


def root_dir():
    d = os.environ.get("PSI_SAE_DIR") or os.path.join(os.environ.get("PSI_DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", ".userdata"), "sae")
    return os.path.abspath(d)


class SAE(torch.nn.Module):
    def __init__(self, d_in, d_sae, kind="topk", k=32):
        super().__init__()
        self.d_in, self.d_sae, self.kind, self.k = d_in, d_sae, kind, int(k or 0)
        self.W_enc = torch.nn.Parameter(torch.empty(d_in, d_sae)); self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.W_dec = torch.nn.Parameter(torch.empty(d_sae, d_in)); self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        self.register_buffer("threshold", torch.zeros(d_sae))
        torch.nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True); self.W_enc.data = self.W_dec.data.T.clone()

    def pre(self, x):
        return (x.to(self.W_enc.dtype) - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        z = self.pre(x)
        if self.kind == "topk":
            v, i = torch.topk(z, min(self.k, z.shape[-1]), dim=-1)
            return torch.zeros_like(z).scatter_(-1, i, torch.relu(v))
        if self.kind == "jumprelu":
            return z * (z > self.threshold)
        return torch.relu(z)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec

    def forward(self, x):
        a = self.encode(x); return self.decode(a), a


def _load_dir(d):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(d, "cfg.json"), encoding="utf-8"))
    w = load_file(os.path.join(d, "sae.safetensors"))
    s = SAE(int(cfg["d_in"]), int(cfg["d_sae"]), cfg.get("kind", "topk"), cfg.get("k", 0))
    with torch.no_grad():
        for n in ("W_enc", "b_enc", "W_dec", "b_dec"): getattr(s, n).data = w[n].float()
        if "threshold" in w: s.threshold.data = w["threshold"].float()
    feats = {}
    fp = os.path.join(d, "features.json")
    if os.path.exists(fp):
        try: feats = json.load(open(fp, encoding="utf-8"))
        except ValueError: feats = {}
    return s.eval(), cfg, feats


def save_dir(d, sae, cfg, feats=None):
    from safetensors.torch import save_file
    os.makedirs(d, exist_ok=True)
    t = {n: getattr(sae, n).detach().float().cpu().contiguous() for n in ("W_enc", "b_enc", "W_dec", "b_dec")}
    if sae.kind == "jumprelu": t["threshold"] = sae.threshold.detach().float().cpu().contiguous()
    save_file(t, os.path.join(d, "sae.safetensors"))
    cfg = {**cfg, "format": FMT, "kind": sae.kind, "k": sae.k, "d_in": sae.d_in, "d_sae": sae.d_sae}
    json.dump(cfg, open(os.path.join(d, "cfg.json"), "w", encoding="utf-8"), indent=1)
    if feats is not None: json.dump(feats, open(os.path.join(d, "features.json"), "w", encoding="utf-8"))


def import_folder(src, d_in, dest, layer, meta):
    """Turn a downloaded SAE (this format, EleutherAI sparsify, or SAE Lens / Gemma Scope) into this layout."""
    from safetensors.torch import load_file
    files = [f for f in os.listdir(src) if f.endswith(".safetensors")]
    npz = [f for f in os.listdir(src) if f.endswith(".npz")]
    if files: w = load_file(os.path.join(src, files[0]))
    elif npz:
        import numpy as np
        z = np.load(os.path.join(src, npz[0])); w = {k: torch.from_numpy(z[k]) for k in z.files}
    else: raise ValueError("no .safetensors or .npz file in that folder")
    cfg = {}
    for c in ("cfg.json", "config.json"):
        if os.path.exists(os.path.join(src, c)): cfg = json.load(open(os.path.join(src, c), encoding="utf-8")); break
    if "encoder.weight" in w:     # EleutherAI sparsify: TopK
        We, be, Wd, bd = w["encoder.weight"].T, w["encoder.bias"], w["W_dec"], w["b_dec"]; kind, k, th = "topk", int(cfg.get("k", 32)), None
        how = "EleutherAI sparsify (TopK)"
    elif "W_enc" in w:            # SAE Lens / Gemma Scope / this format
        We, be, Wd, bd = w["W_enc"], w["b_enc"], w["W_dec"], w["b_dec"]; th = w.get("threshold")
        kind = "jumprelu" if th is not None else cfg.get("kind", "relu"); k = int(cfg.get("k", 0) or 0)
        if kind == "topk" and not k: raise ValueError("a TopK SAE needs k in its cfg.json")
        how = "Gemma Scope (JumpReLU)" if th is not None else f"SAE Lens style ({kind})"
    else: raise ValueError(f"not a known SAE layout (tensors: {', '.join(list(w)[:8])})")
    if We.shape[0] != d_in: raise ValueError(f"this SAE reads {We.shape[0]} numbers per piece; layer {layer} of this model has {d_in}: it was made for another model")
    s = SAE(d_in, We.shape[1], kind, k)
    with torch.no_grad():
        s.W_enc.data, s.b_enc.data, s.W_dec.data, s.b_dec.data = We.float(), be.float(), Wd.float(), bd.float()
        if th is not None: s.threshold.data = th.float()
    save_dir(dest, s, {**meta, "layer": layer, "imported": {"from": os.path.abspath(src), "as": how, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}}, {"freq": None, "exemplars": {}, "labels": {}})
    return {"layer": layer, "d_sae": int(We.shape[1]), "kind": kind, "as": how}


def top_token_index(readout_many, W_dec, layer, k=8, chunk=256):
    """For every feature, the ids of the k words its decoder direction promotes most (through the readout, as the
    logit lens reads a layer). Computed once per SAE and kept in features.json, so search and names are a lookup."""
    out = []
    with torch.no_grad():
        for i in range(0, W_dec.shape[0], chunk):
            out.extend(torch.topk(readout_many(W_dec[i:i + chunk], layer), k, dim=1).indices.tolist())
    return out


class Bank:
    """The SAEs of the loaded model, loaded on first use (one per layer)."""

    def __init__(self, tracer):
        self.t = tracer; self.key = model_key(tracer.model_dir, tracer.weights_hash)
        self.dir = os.path.join(root_dir(), self.key); self.cache = {}

    def layer_dir(self, l): return os.path.join(self.dir, f"L{int(l)}")

    def layers(self):
        if not os.path.isdir(self.dir): return []
        return sorted(int(n[1:]) for n in os.listdir(self.dir) if re.fullmatch(r"L\d+", n) and os.path.exists(os.path.join(self.dir, n, "sae.safetensors")))

    def get(self, l):
        l = int(l)
        if l not in self.cache:
            if l not in self.layers(): raise ValueError(f"no SAE for layer {l} of this model (have: {self.layers() or 'none'}); train or import one in Features")
            s, cfg, feats = _load_dir(self.layer_dir(l))
            self.cache[l] = (s.to(self.t.device), cfg, feats)
        return self.cache[l]

    def _save_feats(self, l):
        s, cfg, feats = self.cache[int(l)]
        json.dump(feats, open(os.path.join(self.layer_dir(l), "features.json"), "w", encoding="utf-8"))

    def list(self):
        out = []
        for l in self.layers():
            try: cfg = json.load(open(os.path.join(self.layer_dir(l), "cfg.json"), encoding="utf-8"))
            except ValueError: continue
            fp = os.path.join(self.layer_dir(l), "features.json"); n_lab = 0; has_ex = False
            if os.path.exists(fp):
                try: f = json.load(open(fp, encoding="utf-8")); n_lab = len(f.get("labels") or {}); has_ex = bool(f.get("exemplars"))
                except ValueError: pass
            out.append({"layer": l, "d_sae": cfg.get("d_sae"), "kind": cfg.get("kind"), "k": cfg.get("k"), "trained": cfg.get("trained"), "imported": cfg.get("imported"), "labels": n_lab, "exemplars": has_ex})
        return {"model": self.key, "dir": self.dir, "model_layers": self.t.L, "hidden": self.t.model.config.hidden_size, "saes": out}

    # ── reading ──
    def _hidden(self, ids, layers):
        with self.t.lock, torch.no_grad():
            hs = self.t.model(input_ids=ids, output_hidden_states=True).hidden_states
            return {l: hs[l][0].float() for l in layers}

    def name_of(self, l, f, feats=None):
        feats = feats if feats is not None else self.get(l)[2]
        lab = (feats.get("labels") or {}).get(str(f))
        return lab["label"] if lab else None

    def index(self, l):
        """The promoted-words index of layer l's SAE (made and saved the first time it is needed)."""
        s, cfg, feats = self.get(l)
        if not feats.get("top_tokens"):
            feats["top_tokens"] = top_token_index(self.t.readout_many, s.W_dec.to(self.t.device), int(l)); self._save_feats(l)
        if "_by_token" not in self.__dict__: self._by_token = {}
        if int(l) not in self._by_token:
            m = {}
            for f, ids in enumerate(feats["top_tokens"]):
                for r, t in enumerate(ids[:4]): m.setdefault(int(t), []).append((r, f))
            self._by_token[int(l)] = m
        return feats["top_tokens"], self._by_token[int(l)]

    def top_words(self, l, f, k=8):
        s = self.get(l)[0]; v = s.W_dec[int(f)].to(self.t.device)
        with torch.no_grad(): lg = self.t.readout(v, l)
        up, dn = torch.topk(lg, k), torch.topk(-lg, k)
        dec = lambda i: self.t.tok.decode([int(i)])
        return [dec(i) for i in up.indices], [dec(i) for i in dn.indices]

    def auto_name(self, l, f):
        """A feature's name: the one given to it; else the pieces it fires on most in its strongest examples (how
        features are usually first labelled); else the words its direction promotes (marked →)."""
        n = self.name_of(l, f)
        if n: return n
        ex = (self.get(l)[2].get("exemplars") or {}).get(str(int(f)))
        if ex:
            cnt = {}
            for e in ex: w = str(e[0])[e[2]:e[3]].strip() or repr(str(e[0])[e[2]:e[3]]); cnt[w] = cnt.get(w, 0) + 1
            return "on " + " · ".join(f"“{w}”" for w, _ in sorted(cnt.items(), key=lambda x: -x[1])[:3])
        tops = self.get(l)[2].get("top_tokens")
        if tops: return "→ " + " · ".join(self.t.tok.decode([int(i)]).strip() or repr(self.t.tok.decode([int(i)])) for i in tops[int(f)][:4])
        up, _ = self.top_words(l, f, 4)
        return "→ " + " · ".join(w.strip() or repr(w) for w in up)

    def encode(self, text, layer, k=8, chat=False):
        s, cfg, feats = self.get(layer)
        ids = (self.t._chat_ids(text, False) if chat else self.t.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.t.device))
        if ids.shape[1] < 1: raise ValueError("the text has no pieces")
        h = self._hidden(ids, [int(layer)])[int(layer)]
        with torch.no_grad():
            a = s.encode(h.to(s.W_enc.dtype)); rec = s.decode(a)
            hh, rr = (h[1:], rec[1:]) if h.shape[0] > 1 else (h, rec)
            # against the SAE's own centre (b_dec, set from the training data's mean), not this text's: a short text's
            # own spread is small and would inflate the share
            fvu = float(((hh - rr) ** 2).sum() / ((hh - s.b_dec) ** 2).sum().clamp_min(1e-9))
        toks, total = [], torch.zeros(a.shape[1], device=a.device)
        # The first piece's state is an outlier many times larger than the rest (the attention sink): it is shown, but left
        # out of the totals, as training leaves it out (unless it is the only piece).
        for p in range(a.shape[0]):
            v, i = torch.topk(a[p], k)
            if p > 0 or a.shape[0] == 1: total += a[p]
            toks.append({"piece": self.t.tok.decode([int(ids[0, p])]), "active": int((a[p] > 0).sum()), "first": p == 0, "top": [[int(j), round(float(x), 3)] for x, j in zip(v, i) if float(x) > 0]})
        # Ranked by how strongly each fired against its own peak (some features run 100x larger than others: the
        # model's few huge outlier activations), weighted by how rare it is (log 1/frequency), so a feature that fires
        # on most pieces does not top every text. The peak is floored at the median peak and the frequency at 1 in
        # 10,000, so a nearly dead, tiny feature cannot win either. The value shown stays the summed activation.
        fr, mx0 = feats.get("freq"), feats.get("max_act")
        if fr and mx0:
            dev = total.device
            pk = [x for x in mx0 if x]; med = sorted(pk)[len(pk) // 2] if pk else 1.0
            mxv = torch.tensor([max(x or med, med) for x in mx0], device=dev); frv = torch.tensor([max(x or 1e-4, 1e-4) for x in fr], device=dev)
            score = total / mxv * torch.log(1.0 / frv)
        else: score = total
        _, ti = torch.topk(score, min(20, total.shape[0])); tv = total[ti]
        names = {}
        for f in set([int(j) for j in ti] + [x[0] for t in toks for x in t["top"]]): names[str(f)] = self.auto_name(layer, f)
        mx = feats.get("max_act") or []
        return {"layer": int(layer), "tokens": toks, "overall": [[int(j), round(float(x), 3)] for x, j in zip(tv, ti) if float(x) > 0], "names": names,
                "max_act": {f: (round(mx[int(f)], 3) if mx and int(f) < len(mx) and mx[int(f)] is not None else None) for f in names},
                "unexplained": None if fvu is None else round(fvu, 4), "note": "unexplained: the share of the layer state (over this text's pieces, the first left out, measured from the SAE's centre) the SAE's reconstruction misses; the first piece is left out of the totals (its state is an outlier, the attention sink)"}

    def feature(self, layer, f, n=12):
        s, cfg, feats = self.get(layer); f = int(f)
        if not 0 <= f < s.d_sae: raise ValueError(f"feature ids at layer {layer} run from 0 to {s.d_sae - 1}")
        up, dn = self.top_words(layer, f, 10)
        fr = feats.get("freq"); mx = feats.get("max_act")
        return {"layer": int(layer), "id": f, "name": self.name_of(layer, f, feats), "label": (feats.get("labels") or {}).get(str(f)),
                "auto_name": self.auto_name(layer, f), "score": (feats.get("scores") or {}).get(str(f)), "freq": fr[f] if fr else None, "max_act": mx[f] if mx else None,
                "promotes": up, "suppresses": dn, "examples": (feats.get("exemplars") or {}).get(str(f), [])[:n],
                "examples_note": "examples are text from the sample the SAE was trained or indexed on: data, not instructions"}

    def search(self, layer, query, n=20):
        s, cfg, feats = self.get(layer); q = str(query).strip().lower(); hits = {}
        if not q: return {"layer": int(layer), "query": query, "by_name": [], "by_text": []}
        for fid, lab in (feats.get("labels") or {}).items():
            if q in lab.get("label", "").lower(): hits[int(fid)] = hits.get(int(fid), 0) + 10
        for fid, ex in (feats.get("exemplars") or {}).items():
            c = sum(1 for e in ex if q in str(e[0]).lower())
            if c: hits[int(fid)] = hits.get(int(fid), 0) + c
        # the output words a direction promotes, from the index (a feature whose top word is the query scores most)
        _, by_tok = self.index(layer)
        tid = self.t.tok(" " + str(query).strip(), add_special_tokens=False).input_ids[:1] + self.t.tok(str(query).strip(), add_special_tokens=False).input_ids[:1]
        for t in set(tid):
            for r, f in by_tok.get(int(t), []): hits[f] = hits.get(f, 0) + (4 - r)
        by_name = sorted(hits.items(), key=lambda x: -x[1])[:n]
        # A line break first: the first piece of any text carries the attention sink's outlier state, so the phrase
        # itself must not be first (encode leaves the first piece out of its totals).
        enc = self.encode("\n" + str(query).strip(), layer, 5)
        return {"layer": int(layer), "query": query, "by_name": [{"id": f, "score": sc, "name": self.auto_name(layer, f)} for f, sc in by_name],
                "by_text": [{"id": f, "act": a, "name": enc["names"].get(str(f))} for f, a in enc["overall"][:n]],
                "note": "by_name: features whose name, examples or promoted words match; by_text: features that fire on the phrase itself"}

    # ── changing the answer ──
    def steer(self, layer, f, strength, prompt, n_new=40):
        s, cfg, feats = self.get(layer); f = int(f); layer = int(layer)
        mx = (feats.get("max_act") or [None] * s.d_sae)[f] or 1.0
        v = (s.W_dec[f] * float(strength) * float(mx)).to(self.t.model.dtype).to(self.t.device)
        ids = self.t._chat_ids(prompt, False)
        def gen():
            with torch.no_grad():
                o = self.t.model.generate(ids, max_new_tokens=int(n_new), do_sample=False, pad_token_id=self.t.tok.eos_token_id)
            return re.sub(r"^\s*(<think>\s*</think>|</think>)\s*", "", self.t.tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
        mod = self.t.model.model.layers[layer - 1]
        def hook(m, inp, out):
            if isinstance(out, tuple): return (out[0] + v,) + tuple(out[1:])
            return out + v
        with self.t.lock:
            plain = gen(); h = mod.register_forward_hook(hook)
            try: steered = gen()
            finally: h.remove()
        return {"layer": layer, "id": f, "name": self.auto_name(layer, f), "strength": float(strength), "added_norm": round(float(v.float().norm()), 3),
                "prompt": prompt, "plain": plain, "steered": steered, "note": "greedy; the direction is added at every piece after the layer; strength is in units of the feature's largest activation seen; the hook is removed after"}

    def attribute(self, prompt, layers=None, target=None, n=10, chat=True):
        """Which features pushed the next word: for each layer with an SAE, activation x (decoder direction · gradient
        of the target's logit), summed over the pieces; and the part of the state the SAE does not explain (error)."""
        have = self.layers(); layers = [int(l) for l in (layers or have) if int(l) in have]
        if not layers: raise ValueError("no SAE layers to attribute through; train or import one in Features")
        ids = self.t._chat_ids(prompt, False) if chat else self.t.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.t.device)
        if target is None:   # the first word of the answer, past the template's own tokens (</think>, blank lines), as knock-outs do
            with self.t.lock, torch.no_grad():
                for _ in range(4):
                    nxt = int(self.t.model(input_ids=ids).logits[0, -1].argmax())
                    if not re.fullmatch(r"\s*(</?think>|\n)*\s*", self.t.tok.decode([nxt])): break
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        grabbed = {}
        # Gradients to the layer states only: the weights' own gradients (as large as the model) are not needed, so the
        # weights are frozen for the pass and the embeddings' output is made to carry gradients instead.
        frozen = [p for p in self.t.model.parameters() if p.requires_grad]
        for p in frozen: p.requires_grad_(False)
        emb = self.t.model.get_input_embeddings()
        hooks = [emb.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))]
        for l in layers:
            def hk(m, inp, out, l=l):
                h = out[0] if isinstance(out, tuple) else out; h.retain_grad(); grabbed[l] = h
            hooks.append(self.t.model.model.layers[l - 1].register_forward_hook(hk))
        with self.t.lock:
            try:
                with torch.enable_grad():
                    lg = self.t.model(input_ids=ids).logits[0, -1].float()
                    tgt = int(target) if target is not None else int(lg.argmax())
                    lg[tgt].backward()
            finally:
                for h in hooks: h.remove()
                for p in frozen: p.requires_grad_(True)
        out = []
        for l in layers:
            s, cfg, feats = self.get(l); h = grabbed[l][0].detach().float(); g = grabbed[l].grad[0].float()
            if h.shape[0] > 1: h, g = h[1:], g[1:]   # the first piece (the attention sink) left out, as in training
            with torch.no_grad():
                a = s.encode(h); rec = s.decode(a)
                contrib = (a * (g @ s.W_dec.T)).sum(0)       # per feature, over the pieces
                err = float(((h - rec) * g).sum()); total = float((h * g).sum())
            pos, neg = torch.topk(contrib, n), torch.topk(-contrib, n)
            out.append({"layer": l, "features_total": round(float(contrib.sum()), 4), "error": round(err, 4), "state_total": round(total, 4),
                        "pushed_up": [{"id": int(i), "by": round(float(v), 4), "name": self.auto_name(l, int(i))} for v, i in zip(pos.values, pos.indices) if float(v) > 0],
                        "pushed_down": [{"id": int(i), "by": round(-float(v), 4), "name": self.auto_name(l, int(i))} for v, i in zip(neg.values, neg.indices) if float(v) > 0]})
        return {"prompt": prompt, "target": self.t.tok.decode([tgt]), "target_id": tgt, "layers": out,
                "note": "activation x gradient: a first-order estimate of how much each feature raised (or lowered) the logit of the next word; error is the part of the layer's state the SAE leaves out, scored the same way"}

    # ── how good is it? ──────────────────────────────────────────────────────
    # Held-out text: the paragraphs a training run kept back (cfg trained.heldout), else the end of the neuron corpus,
    # marked in-sample when the SAE learned from that corpus (it then measures fit, not generalisation).
    def _held_text(self, l):
        s, cfg, feats = self.get(l); tr = cfg.get("trained") or {}; h = tr.get("heldout") or {}
        def paras(path):
            if path.endswith(".json"):
                d = json.load(open(path, encoding="utf-8")); return [x if isinstance(x, str) else x.get("text", "") for x in d]
            if path.endswith(".jsonl"): return [json.loads(x).get("text", "") for x in open(path, encoding="utf-8") if x.strip()]
            return [x for x in open(path, encoding="utf-8").read().split("\n\n") if x.strip()]
        if h.get("sample_path") and os.path.exists(h["sample_path"]):
            return [x for x in paras(h["sample_path"])[int(h["from_paragraph"]):] if x.strip()], "held-out paragraphs the training kept back", False
        c = os.environ.get("PSI_CORPUS") or (os.path.join(os.environ["PSI_CACHE_DIR"], "datasets", "neuron-corpus.json") if os.environ.get("PSI_CACHE_DIR") else "")
        if not c or not os.path.exists(c): raise FileNotFoundError("no text to evaluate on: the neuron corpus was not found (PSI_CORPUS or <cache>/datasets/neuron-corpus.json)")
        ps = [x for x in paras(c) if x.strip()]; tail = ps[int(len(ps) * 0.9):]
        insample = bool(tr) and os.path.basename(c) == str(tr.get("corpus"))
        return tail, ("the last 10% of the neuron corpus, which this SAE also learned from (in-sample: this measures fit, not generalisation)" if insample else "the last 10% of the neuron corpus"), insample

    def _windows(self, l, n=160, ctx=64):
        key = (int(l), n, ctx)
        if getattr(self, "_win_key", None) == key: return self._win
        texts, how, insample = self._held_text(l); wins = []
        for t in texts:
            ids = self.t.tok(t, add_special_tokens=False).input_ids
            for i in range(0, len(ids) - 16, ctx):
                w = ids[i:i + ctx]
                if len(w) >= 16: wins.append(w)
                if len(wins) >= n: break
            if len(wins) >= n: break
        self._win_key, self._win = key, (wins, how, insample); return self._win

    def evaluate(self, l, n=120):
        """Splice-in loss: the next-piece loss with the layer's state replaced by the SAE's reconstruction, against the
        intact model and against the state replaced by the SAE's centre (mean ablation); loss recovered =
        (mean - spliced) / (mean - intact). Also the unexplained share, active features per piece and dead features, on
        held-out text. The first piece of each window keeps its own state (the attention sink) and is not scored."""
        s, cfg, feats = self.get(l); l = int(l); wins, how, insample = self._windows(l, n)
        if not wins: raise ValueError("no held-out text long enough")
        mod = self.t.model.model.layers[l - 1]; mode = {"m": None}
        def hook(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if mode["m"] is None: return out
            x = h[0, 1:].float()
            y = s.decode(s.encode(x)) if mode["m"] == "splice" else s.b_dec.expand_as(x)
            h2 = h.clone(); h2[0, 1:] = y.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        loss = {"intact": 0.0, "splice": 0.0, "mean": 0.0}; npc = 0; err = 0.0; tot = 0.0; l0 = 0.0; seen = torch.zeros(s.d_sae, dtype=torch.bool, device=s.W_enc.device)
        hk = mod.register_forward_hook(hook)
        try:
            with self.t.lock, torch.no_grad():
                for w in wins:
                    ids = torch.tensor([w], device=self.t.device)
                    for m in ("intact", "splice", "mean"):
                        mode["m"] = None if m == "intact" else m
                        o = self.t.model(input_ids=ids, output_hidden_states=(m == "intact"))
                        lp = torch.log_softmax(o.logits[0, 1:-1].float(), -1)   # predictions for pieces 2..n (piece 1's prediction rests on the sink alone)
                        loss[m] += float(-lp.gather(1, ids[0, 2:, None]).sum())
                        if m == "intact":
                            h = o.hidden_states[l][0, 1:].float(); a = s.encode(h); r = s.decode(a)
                            err += float(((h - r) ** 2).sum()); tot += float(((h - s.b_dec) ** 2).sum()); l0 += float((a > 0).sum()); seen |= (a > 0).any(0)
                    npc += len(w) - 2
        finally: hk.remove()
        ce = {k: v / npc for k, v in loss.items()}
        rec = (ce["mean"] - ce["splice"]) / (ce["mean"] - ce["intact"]) if ce["mean"] > ce["intact"] else None
        out = {"text": how, "in_sample": insample, "windows": len(wins), "pieces": npc, "loss_intact": round(ce["intact"], 4), "loss_spliced": round(ce["splice"], 4), "loss_mean_ablated": round(ce["mean"], 4),
               "loss_added": round(ce["splice"] - ce["intact"], 4), "loss_recovered": None if rec is None else round(rec, 4), "unexplained": round(err / max(tot, 1e-9), 4),
               "active_per_piece": round(l0 / (npc + len(wins)), 2), "dead_here": round(float((~seen).float().mean()), 4), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "rule": "next-piece loss in nats per piece; loss recovered = (mean-ablated - spliced) / (mean-ablated - intact); the first piece of each window keeps its state and its prediction is not scored"}
        cfg["evaluation"] = out
        json.dump(cfg, open(os.path.join(self.layer_dir(l), "cfg.json"), "w", encoding="utf-8"), indent=1)
        return {"layer": l, **out}

    # Snippets for scoring a feature's name (auto-interp, detection): where it fires most in held-out text, and random
    # places where it does not fire. The truth goes back to the app separately; the scorer sees only the snippets.
    def _held_index(self, l):
        key = int(l)
        if getattr(self, "_hidx_key", None) == key: return self._hidx
        s, cfg, feats = self.get(l); wins, how, insample = self._windows(l, 300)
        enc = []
        with self.t.lock, torch.no_grad():
            for w in wins:
                h = self.t.model(input_ids=torch.tensor([w], device=self.t.device), output_hidden_states=True).hidden_states[key][0].float()
                a = s.encode(h); a[0] = 0   # the sink piece is never a snippet
                v, i = torch.topk(a, min(s.k or 64, a.shape[1]), dim=1); enc.append((i.cpu(), v.cpu()))
        self._hidx_key, self._hidx = key, (wins, enc, how, insample); return self._hidx

    def snippet(self, w, pos):
        left = self.t.tok.decode(w[max(0, pos - 14):pos]); piece = self.t.tok.decode(w[pos:pos + 1]); right = self.t.tok.decode(w[pos + 1:pos + 5])
        return f"{left}[[{piece}]]{right}".replace("\n", " ")

    def contexts(self, l, f, n_pos=5, n_neg=5, seed=0):
        import random
        f = int(f); wins, enc, how, insample = self._held_index(l); pos, allp = [], []
        for wi, (i, v) in enumerate(enc):
            hit = (i == f) & (v > 0)
            for p in range(1, i.shape[0]):
                allp.append((wi, p))
                if bool(hit[p].any()): pos.append((float(v[p][hit[p]].max()), wi, p))
        pos.sort(reverse=True)
        if len(pos) < n_pos: return {"layer": int(l), "id": f, "enough": False, "fires_on": len(pos), "text": how}
        fired = {(wi, p) for _, wi, p in pos}; rnd = random.Random(seed * 1000003 + f)
        neg = rnd.sample([x for x in allp if x not in fired], n_neg)
        items = [{"text": self.snippet(wins[wi], p), "fires": True, "act": round(a, 3)} for a, wi, p in pos[:n_pos]] + [{"text": self.snippet(wins[wi], p), "fires": False, "act": 0.0} for wi, p in neg]
        rnd.shuffle(items)
        return {"layer": int(l), "id": f, "enough": True, "items": items, "text": how, "in_sample": insample, "fires_on": len(pos)}

    def scorable(self, l, count=10, seed=0, min_hits=5):
        """Features with enough held-out examples to score, the ones that fire on the most held-out pieces first,
        skipping the commonest (over 5% of pieces), which are rarely about one thing."""
        wins, enc, how, insample = self._held_index(l); cnt = {}
        npieces = sum(i.shape[0] - 1 for i, _ in enc)
        for i, v in enc:
            for f in i[1:][v[1:] > 0].tolist(): cnt[f] = cnt.get(f, 0) + 1
        ok = [f for f, c in cnt.items() if c >= min_hits and c / max(npieces, 1) < 0.05]
        import random
        random.Random(seed).shuffle(ok)
        return {"layer": int(l), "ids": ok[:int(count)], "candidates": len(ok), "text": how}

    def set_score(self, l, f, score):
        s, cfg, feats = self.get(l); feats.setdefault("scores", {})[str(int(f))] = score; self._save_feats(l); return {"layer": int(l), "id": int(f), "score": score}

    def label(self, layer, f, text, by="you", why=""):
        s, cfg, feats = self.get(layer); f = int(f)
        text = re.sub(r"[\x00-\x1f\x7f]", " ", str(text)).strip()[:80]
        labs = feats.setdefault("labels", {})
        if text: labs[str(f)] = {"label": text, "by": by if by in ("you", "prometheus") else "you", "why": str(why)[:400], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        else: labs.pop(str(f), None)
        self._save_feats(layer); return {"layer": int(layer), "id": f, "label": labs.get(str(f))}
