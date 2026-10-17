"""
Anatomy of the loaded model: its real structure, its real weights, and a recorded forward pass.

  structure()        every layer's real shapes and parameter counts (config + state dict)
  layer_maps(l)      pooled |weight| heatmaps of each matrix in layer l (Q, K, V, O, gate, up, down)
  links()            weight-derived connectivity between consecutive layers' MLP neurons:
                     M = W_up(l+1) . W_down(l)  (how strongly neuron j of layer l writes into neuron
                     i of layer l+1, through the residual stream). Pooled to G x G bins, plus the
                     strongest individual neuron-to-neuron links as real nodes. Heads: the norm of
                     each head's output map composed into the next layer's MLP input, per bin.
  record(...)        a forward pass and generation, every position (prompt read, then each token
                     written): per layer neuron-bin activity and top neurons, per head attention
                     entropy and where it looks, the mean attention row (what is read from the KV
                     cache), the residual norm, and what the layer would predict (logit lens).

Guardrails: neurons are pooled into G bins (their order is arbitrary, so bins are a down-sampling,
not a grouping); only the strongest individual links are drawn; recordings are capped in length;
the link computation is skipped for models over 20 B parameters.
"""
import json, math, os, threading, time
import numpy as np
import torch
import devices

G = 48            # default neuron bins per layer (auto_bins adapts it to the model and the drawing)
TOP_LINKS = 24    # strongest individual neuron-to-neuron links per layer pair
MAX_PROMPT, MAX_NEW = 160, 64
# The app's caches: $PSI_CACHE_DIR/psi when a cache folder is set (settings.cacheDir; config.js cacheEnv passes it),
# else <data folder>/cache (PSI_DATA_DIR, default .userdata/ beside src/).
_CACHE_ROOT = os.environ.get("PSI_CACHE_DIR")
_PSI_CACHE = os.path.join(_CACHE_ROOT, "psi") if _CACHE_ROOT else os.path.join(os.environ.get("PSI_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".userdata"), "cache")
# The text sample (a JSON list of paragraphs, e.g. a Wikipedia sample): $PSI_CORPUS, else <PSI_CACHE_DIR>/datasets/neuron-corpus.json.
CORPUS = os.environ.get("PSI_CORPUS") or (os.path.join(_CACHE_ROOT, "datasets", "neuron-corpus.json") if _CACHE_ROOT else "")
EX_DIR = os.environ.get("PSI_EXAMPLES_DIR") or _PSI_CACHE                          # where per-model neuron examples are cached
EX_K, EX_WIN, EX_MAX = 8, 64, 2400                                                 # top examples per neuron, window tokens, windows
ST_MAX = 800                                                                       # windows read for each neuron's normal range (mean, SD)


def _need_corpus():
    if not CORPUS: raise FileNotFoundError("No text sample set: point PSI_CORPUS at a JSON list of paragraphs (or put neuron-corpus.json in <PSI_CACHE_DIR>/datasets).")
    if not os.path.exists(CORPUS): raise FileNotFoundError(f"The text sample {CORPUS} is missing (set PSI_CORPUS).")


def pool2d(w, out_r, out_c):
    w = w.abs().float()
    r, c = w.shape; out_r = min(out_r, r); out_c = min(out_c, c)
    rr = (r // out_r) * out_r; cc = (c // out_c) * out_c
    return w[:rr, :cc].reshape(out_r, rr // out_r, out_c, cc // out_c).mean((1, 3))


def edges(n, g):
    g = int(max(1, min(g, n))); return torch.linspace(0, n, g + 1).round().long()


def bin_index(n, g, device):
    e = edges(n, g); idx = torch.empty(n, dtype=torch.long)
    for b in range(len(e) - 1): idx[e[b]:e[b + 1]] = b
    return idx.to(device), (e[1:] - e[:-1]).float().to(device)


def auto_bins(n, px=None):
    """Bins for a layer of n neurons: about one bin per 6 drawn pixels, 16 to 160, never more than n."""
    want = int(px / 6) if px else 48
    return int(max(16, min(160, want, n)))


class Anatomy:
    def __init__(self, tracer):
        self.t = tracer; self.m = tracer.model; self.cfg = self.m.config; self.tok = tracer.tok
        self.layers = self.m.model.layers; self.L = len(self.layers)
        self.H = self.cfg.num_attention_heads; self.KV = getattr(self.cfg, "num_key_value_heads", self.H)
        self.hd = getattr(self.cfg, "head_dim", self.cfg.hidden_size // self.H); self.I = self.cfg.intermediate_size; self.D = self.cfg.hidden_size
        self._links = {}; self.lock = threading.Lock(); self.last = None; self._ex = None; self.ex_progress = None; self._stats = None; self.st_progress = None   # last recording's full MLP activations [S, L, I] (CPU, fp16)

    # ── structure ───────────────────────────────────────────────────────────
    def structure(self, bins=None):
        G_ = int(bins or auto_bins(self.I))
        total = sum(p.numel() for p in self.m.parameters())
        per = []
        for i, layer in enumerate(self.layers):
            mods = {}
            for name, p in layer.named_parameters(): mods[name] = {"shape": list(p.shape), "params": p.numel()}
            per.append({"layer": i, "params": sum(v["params"] for v in mods.values()), "tensors": mods})
        emb = self.m.model.embed_tokens.weight
        tied = self.m.lm_head.weight.data_ptr() == emb.data_ptr()
        return {"name": os.path.basename(str(getattr(self.cfg, "_name_or_path", "model"))).strip("/") or self.cfg.model_type, "arch": self.cfg.model_type,
                "params": total, "layers": self.L, "hidden": self.D, "heads": self.H, "kv_heads": self.KV, "head_dim": self.hd, "mlp": self.I,
                "vocab": emb.shape[0], "context": getattr(self.cfg, "max_position_embeddings", None), "tied_embeddings": tied,
                "embedding_params": emb.numel(), "per_layer": per, "bins": G_, "neurons_per_bin": round(self.I / G_, 1),
                "kv_bytes_per_token": 2 * self.L * self.KV * self.hd * 2, "dtype": str(next(self.m.parameters()).dtype).replace("torch.", "")}

    # ── weights ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def layer_maps(self, l, size=64):
        layer = self.layers[int(l)]; a, f = layer.self_attn, layer.mlp; out = {}
        for name, w in (("q_proj", a.q_proj.weight), ("k_proj", a.k_proj.weight), ("v_proj", a.v_proj.weight), ("o_proj", a.o_proj.weight),
                        ("gate_proj", f.gate_proj.weight), ("up_proj", f.up_proj.weight), ("down_proj", f.down_proj.weight)):
            P = pool2d(w, size, size); mx = float(P.max()) or 1.0
            out[name] = {"shape": list(w.shape), "pooled": [size if w.shape[0] >= size else w.shape[0], size if w.shape[1] >= size else w.shape[1]],
                         "norm": round(float(w.float().norm()), 2), "max_abs": round(float(w.abs().max()), 4), "map": (P / mx).cpu().numpy().round(3).tolist()}
        # head strength: Frobenius norm of each head's output block of o_proj
        Wo = a.o_proj.weight.float(); out["head_out_norm"] = [round(float(Wo[:, h * self.hd:(h + 1) * self.hd].norm()), 3) for h in range(self.H)]
        return {"layer": int(l), "maps": out}

    @torch.no_grad()
    def links(self, bins=None):
        G_ = int(bins or auto_bins(self.I))
        if G_ in self._links: return self._links[G_]
        if sum(p.numel() for p in self.m.parameters()) > 20e9: return {"skipped": "model over 20 B parameters", "bins": G_}
        t0 = time.time(); pairs = []; dev = self.layers[0].mlp.down_proj.weight.device
        bi, cnt = bin_index(self.I, G_, dev)
        def pool(A):                                            # [I, I] -> [G, G] mean |value| per bin pair
            R = torch.zeros(G_, A.shape[1], device=dev).index_add_(0, bi, A) / cnt[:, None]
            return torch.zeros(G_, G_, device=dev).index_add_(1, bi, R) / cnt[None, :]
        for l in range(self.L - 1):
            down = self.layers[l].mlp.down_proj.weight; up = self.layers[l + 1].mlp.up_proj.weight
            M = (up @ down).float(); A = M.abs(); P = pool(A)   # rows: next-layer bins, cols: this-layer bins
            flat = torch.topk(A.flatten(), TOP_LINKS); ii = (flat.indices // self.I).tolist(); jj = (flat.indices % self.I).tolist()
            top = [{"from": int(j), "to": int(i), "w": round(float(M[i, j]), 4)} for i, j in zip(ii, jj)]
            Wo = self.layers[l].self_attn.o_proj.weight; heads = []
            for h in range(self.H):
                C = (up @ Wo[:, h * self.hd:(h + 1) * self.hd]).float().abs().mean(1)          # [I]
                heads.append([round(float(x), 4) for x in torch.zeros(G_, device=dev).index_add_(0, bi, C) / cnt])
            pairs.append({"from_layer": l, "bins": (P / P.max()).cpu().numpy().round(3).tolist(), "bin_scale": round(float(P.max()), 5), "top": top, "heads_to_bins": heads})
            del M, A
        devices.empty_cache()
        self._links[G_] = {"pairs": pairs, "bins": G_, "edges": edges(self.I, G_).tolist(), "seconds": round(time.time() - t0, 1)}
        return self._links[G_]

    # ── zooming into one bin, and into one neuron ───────────────────────────
    @torch.no_grad()
    def zoom(self, layer, b, bins):
        l = int(layer); e = edges(self.I, bins); b = int(max(0, min(b, len(e) - 2))); lo, hi = int(e[b]), int(e[b + 1])
        out = {"layer": l, "bin": b, "from": lo, "to": hi, "bins": int(len(e) - 1)}
        W = self.layers[l].mlp.down_proj.weight[:, lo:hi].float()
        out["write_norm"] = [round(float(x), 3) for x in W.norm(dim=0)]           # how strongly each neuron can write
        if self.last is not None:
            X = self.last["acts"][:, l, lo:hi].float()                              # [S, n]
            out["acts"] = X.numpy().round(2).tolist(); out["steps"] = int(X.shape[0])
        return out

    @torch.no_grad()
    def neuron(self, layer, j, k=8):
        l = int(layer); j = int(j); tok = self.tok
        wd = self.layers[l].mlp.down_proj.weight[:, j].float()                    # what neuron j writes into the residual stream
        logits = self.m.lm_head.weight.float() @ wd                                # ... read out directly as tokens
        up_t = torch.topk(logits, k); dn_t = torch.topk(-logits, k)
        out = {"layer": l, "neuron": j, "write_norm": round(float(wd.norm()), 3),
               "promotes": [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(up_t.values, up_t.indices)],
               "suppresses": [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(dn_t.values, dn_t.indices)]}
        if l + 1 < self.L:
            to = self.layers[l + 1].mlp.up_proj.weight.float() @ wd; t = torch.topk(to.abs(), k)
            out["feeds"] = [[int(i), round(float(to[i]), 4)] for i in t.indices]
        if l > 0:
            fr = self.layers[l].mlp.up_proj.weight[j].float() @ self.layers[l - 1].mlp.down_proj.weight.float(); t = torch.topk(fr.abs(), k)
            out["fed_by"] = [[int(i), round(float(fr[i]), 4)] for i in t.indices]
        if self.last is not None:
            out["acts"] = [round(float(x), 3) for x in self.last["acts"][:, l, j].float()]
        ex = self.examples_for(l, j)
        if ex is not None: out["examples"] = ex
        return out

    # ── what makes a neuron fire: its strongest examples in ordinary text ──
    def _ex_path(self):
        name = os.path.basename(str(getattr(self.cfg, "_name_or_path", "model")).rstrip("/\\")) or "model"
        return os.path.join(EX_DIR, f"neuron-examples-{name}.pt")

    def examples_status(self):
        p = self._ex_path()
        return {"ready": os.path.exists(p), "path": p, "corpus": CORPUS, "corpus_found": os.path.exists(CORPUS), "progress": self.ex_progress}

    @torch.no_grad()
    def build_examples(self, max_windows=EX_MAX, batch=6):
        """One pass over a Wikipedia sample: for every MLP neuron in every layer, keep the 8 tokens where it
        fired most (largest positive activation). Cached per model, about 20 MB."""
        _need_corpus()
        paras = json.load(open(CORPUS, encoding="utf-8")); tok = self.tok; dev = self.layers[0].mlp.down_proj.weight.device
        wins = []
        for ptxt in paras:
            ids = tok(ptxt, add_special_tokens=False).input_ids
            for a in range(0, max(1, len(ids) - 16), EX_WIN):
                w = ids[a:a + EX_WIN]
                if len(w) >= 16: wins.append(w + [tok.pad_token_id or 0] * (EX_WIN - len(w)))
                if len(wins) >= max_windows: break
            if len(wins) >= max_windows: break
        T = torch.tensor(wins, dtype=torch.long)                                     # [W, WIN]
        best_v = torch.full((self.L, EX_K, self.I), -1e9, device=dev); best_i = torch.zeros((self.L, EX_K, self.I), dtype=torch.long, device=dev)
        acts = {}
        hooks = [self.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: acts.__setitem__(l, inp[0].detach())) for l in range(self.L)]
        t0 = time.time()
        try:
            for b0 in range(0, len(wins), batch):
                ids = T[b0:b0 + batch].to(dev); self.m(input_ids=ids)
                base = b0 * EX_WIN
                for l in range(self.L):
                    A = acts[l].reshape(-1, self.I).float()                            # [B*WIN, I]
                    v, i = torch.topk(A, EX_K, dim=0)
                    cv = torch.cat([best_v[l], v]); ci = torch.cat([best_i[l], i + base])
                    tv, ti = torch.topk(cv, EX_K, dim=0); best_v[l] = tv; best_i[l] = torch.gather(ci, 0, ti)
                self.ex_progress = {"done": min(len(wins), b0 + batch), "of": len(wins), "seconds": round(time.time() - t0)}
        finally:
            for h in hooks: h.remove()
        os.makedirs(EX_DIR, exist_ok=True)
        torch.save({"tokens": T.to(torch.int32), "v": best_v.half().cpu(), "i": best_i.to(torch.int32).cpu(), "win": EX_WIN, "windows": len(wins), "corpus": CORPUS}, self._ex_path())
        self._ex = None; devices.empty_cache()
        return {"ready": True, "windows": len(wins), "tokens": len(wins) * EX_WIN, "seconds": round(time.time() - t0)}

    def examples_for(self, l, j, before=14, after=3):
        if self._ex is None:
            p = self._ex_path()
            if not os.path.exists(p): return None
            self._ex = torch.load(p, weights_only=True)   # tensors only: a planted .pt cannot run code
        E = self._ex; W = E["win"]; out = []; seen = set()
        for v, flat in zip(E["v"][l, :, j].tolist(), E["i"][l, :, j].tolist()):
            if v < -1e8: continue
            w, pos = divmod(int(flat), W)
            if (w, pos) in seen: continue
            seen.add((w, pos)); row = E["tokens"][w].tolist()
            out.append({"act": round(float(v), 2), "before": self.tok.decode(row[max(0, pos - before):pos]), "token": self.tok.decode([row[pos]]), "after": self.tok.decode(row[pos + 1:pos + 1 + after])})
        return out

    # ── each neuron's normal range on ordinary text: z-scores, and the value mean-ablation holds it at ──
    def _stats_path(self):
        name = os.path.basename(str(getattr(self.cfg, "_name_or_path", "model")).rstrip("/\\")) or "model"
        h = (getattr(self.t, "weights_hash", None) or "nohash")[:12]   # keyed by the weights, so two models never share one
        return os.path.join(EX_DIR, f"neuron-stats-{name}-{h}.pt")

    def stats(self):
        """The cached per-neuron mean and SD ([L, I] each, CPU float32), or None when not built for these weights."""
        if self._stats is None:
            p = self._stats_path()
            if not os.path.exists(p): return None
            self._stats = torch.load(p, weights_only=True)
        return self._stats

    def stats_status(self):
        p = self._stats_path()
        out = {"ready": os.path.exists(p), "path": p, "corpus": CORPUS, "corpus_found": os.path.exists(CORPUS), "windows": ST_MAX, "window_tokens": EX_WIN, "progress": self.st_progress}
        if out["ready"]:
            s = self.stats(); out.update({k: s[k] for k in ("tokens", "windows", "built_at", "weights_hash") if k in s})
        return out

    @torch.no_grad()
    def build_stats(self, max_windows=ST_MAX, batch=6):
        """One pass over the Wikipedia sample build_examples reads: every MLP neuron's mean and SD over every real
        token (padding left out). Used for z-scores (how unusual a firing was) and mean-ablation (hold a neuron at its
        average instead of at zero). Cached per weights hash in EX_DIR, a few MB."""
        _need_corpus()
        paras = json.load(open(CORPUS, encoding="utf-8")); tok = self.tok; dev = self.layers[0].mlp.down_proj.weight.device
        ctx = getattr(self.cfg, "max_position_embeddings", None) or EX_WIN; W = min(EX_WIN, ctx)
        wins, lens = [], []
        for ptxt in paras:
            ids = tok(ptxt, add_special_tokens=False).input_ids
            for a in range(0, max(1, len(ids) - 16), W):
                w = ids[a:a + W]
                if len(w) >= 16: wins.append(w + [tok.pad_token_id or 0] * (W - len(w))); lens.append(len(w))
                if len(wins) >= max_windows: break
            if len(wins) >= max_windows: break
        T = torch.tensor(wins, dtype=torch.long); Ln = torch.tensor(lens)
        s1 = torch.zeros(self.L, self.I, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1); n = 0
        acts = {}
        hooks = [self.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: acts.__setitem__(l, inp[0].detach())) for l in range(self.L)]
        t0 = time.time()
        try:
            for b0 in range(0, len(wins), batch):
                ids = T[b0:b0 + batch].to(dev); self.m(input_ids=ids)
                mask = (torch.arange(W)[None, :] < Ln[b0:b0 + batch, None]).to(dev)
                n += int(mask.sum())
                for l in range(self.L):
                    A = acts[l][mask].double()                                        # [real tokens, I]
                    s1[l] += A.sum(0); s2[l] += (A * A).sum(0)
                self.st_progress = {"done": min(len(wins), b0 + batch), "of": len(wins), "seconds": round(time.time() - t0)}
        finally:
            for h in hooks: h.remove()
        mean = s1 / max(1, n); sd = (s2 / max(1, n) - mean * mean).clamp_min(0).sqrt()
        os.makedirs(EX_DIR, exist_ok=True)
        rec = {"mean": mean.float().cpu(), "sd": sd.float().cpu(), "tokens": n, "windows": len(wins), "corpus": CORPUS,
               "weights_hash": getattr(self.t, "weights_hash", None), "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        torch.save(rec, self._stats_path()); self._stats = rec; devices.empty_cache()
        return {"ready": True, "windows": len(wins), "tokens": n, "seconds": round(time.time() - t0), "path": self._stats_path()}

    def zscores(self, mlp):
        """[[[neuron, act] top-k] per layer] -> the same shape of z-scores against each neuron's normal range, or None."""
        st = self.stats()
        if st is None: return None
        M, S = st["mean"], st["sd"]
        return [[round((a - float(M[l, j])) / max(float(S[l, j]), 1e-4), 1) for j, a in row] for l, row in enumerate(mlp)]

    # ── a concept, lit up: where an abstraction lives, its core and its slices ──
    # Generic facets used when no tailored sentences are given ({c} is the concept).
    FACETS = {
        "what it is": ["{C} is a kind of thing that most people have heard of.", "In simple words, {c} is", "A dictionary would define {c} as", "People often explain {c} to children as"],
        "what it is used for": ["Most people use {c} to", "The main reason anyone needs {c} is", "Without {c}, it would be hard to", "{C} is useful when you want to"],
        "how it works": ["Under the hood, {c} works by", "The way {c} actually functions is", "Engineers describe the inner workings of {c} as", "To make {c} work, you first"],
        "where it came from": ["The history of {c} begins with", "{C} got its name because", "Before {c} existed, people", "The first version of {c} appeared when"],
        "problems and limits": ["The biggest problem with {c} is", "{C} fails when", "People often complain that {c}", "A known weakness of {c} is"],
        "what it sits next to": ["{C} is often mentioned together with", "Compared with its alternatives, {c}", "Something similar to {c} is", "{C} belongs to the same family as"],
    }
    CONTROLS = ["the garden", "a violin", "the treaty", "the glacier"]

    @torch.no_grad()
    def _span_acts(self, text, phrase):
        """Mean MLP activation per layer over the tokens of `phrase` inside `text` (and the token after it,
        where the model has read the whole phrase). Returns [L, I] on CPU, or None if the phrase is absent."""
        i = text.lower().find(phrase.lower())
        if i < 0: return None
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = torch.tensor([enc["input_ids"]], device=self.layers[0].mlp.down_proj.weight.device)
        pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > i and a < i + len(phrase)]
        if not pos: return None
        if pos[-1] + 1 < ids.shape[1]: pos.append(pos[-1] + 1)
        acts = {}
        hooks = [self.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, inp, l=l: acts.__setitem__(l, inp[0][0, pos].float().mean(0))) for l in range(self.L)]
        try: self.m(input_ids=ids)
        finally:
            for h in hooks: h.remove()
        return torch.stack([acts[l] for l in range(self.L)])

    @torch.no_grad()
    def _span_resid(self, text, phrase):
        """Mean residual stream after every layer over `phrase`'s tokens in `text`: [L, D], or None."""
        i = text.lower().find(phrase.lower())
        if i < 0: return None
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > i and a < i + len(phrase)]
        if not pos: return None
        ids = torch.tensor([enc["input_ids"]], device=self.layers[0].mlp.down_proj.weight.device)
        hs = self.m(input_ids=ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[l + 1][0, pos].float().mean(0) for l in range(self.L)])

    @torch.no_grad()
    def concept_directions(self, concept, facets=None):
        """The concept as a direction in the residual stream, per layer: mean at the concept's tokens minus the
        mean at unrelated words in the same sentences (unit vectors, [L, D])."""
        concept = str(concept).strip()
        fac = facets or {f: [t.replace("{C}", concept[:1].upper() + concept[1:]).replace("{c}", concept) for t in ts] for f, ts in self.FACETS.items()}
        A, X = [], []
        for sents in fac.values():
            for sent in sents:
                a = self._span_resid(sent, concept)
                if a is None: continue
                j = sent.lower().find(concept.lower())
                for w in self.CONTROLS:
                    b = self._span_resid(sent[:j] + w + sent[j + len(concept):], w)
                    if b is not None: X.append(b)
                A.append(a)
        ma, mx = torch.stack(A).mean(0), torch.stack(X).mean(0); d = ma - mx
        self._erase_mid = (ma + mx) / 2                         # erase around the midpoint, or every other word gets pushed toward the concept
        self._dir_dist = d.norm(dim=-1)                         # how far the concept sits from its controls, per layer
        return d / (d.norm(dim=-1, keepdim=True) + 1e-6)

    def inject_hooks(self, U, strength, band=None):
        """Add the concept's direction to the residual stream after each layer in `band` (all positions but the
        first), scaled to the distance between the concept and its controls at that layer, times `strength`."""
        hs = []; lo, hi = band or (0, self.L - 1); dist = getattr(self, "_dir_dist", None)
        for l, layer in enumerate(self.layers):
            if l < lo or l > hi: continue
            add = U[l] * (float(dist[l]) if dist is not None else 1.0) * float(strength)
            def f(mod, inp, out, add=add):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); h[..., 1:, :] += add.to(h.dtype)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hs.append(layer.register_forward_hook(f))
        return hs

    def erase_hooks(self, U, band=None):
        """Remove the component along U[l] from the residual stream after each layer in `band` (all positions)."""
        hs = []; lo, hi = band or (0, self.L - 1); M = getattr(self, "_erase_mid", None)
        for l, layer in enumerate(self.layers):
            if l < lo or l > hi: continue
            u = U[l]; mid = M[l] if M is not None else None
            def f(mod, inp, out, u=u, mid=mid):
                h = out[0] if isinstance(out, tuple) else out
                uu = u.to(h.dtype); h = h.clone(); x = h[..., 1:, :]
                c = (x @ uu) - (float(mid.to(h.dtype) @ uu) if mid is not None else 0.0)   # how far past the midpoint, along the concept direction
                h[..., 1:, :] = x - c.clamp_min(0)[..., None] * uu   # only what sits on the concept's side is pulled back to the midpoint; spare position 0 (the attention sink)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hs.append(layer.register_forward_hook(f))
        return hs

    @torch.no_grad()
    def concept_map(self, concept, facets=None, k=240, bins=None):
        """Light up a concept. For each sentence about it (grouped into facets), read the MLP neurons at the
        concept's own tokens, and compare with the same sentences where the concept is swapped for unrelated
        words. Neurons that respond reliably more to the concept are its nodes. Neurons that respond in every
        facet are its core (the abstraction: what stays when the context is forgotten); neurons that respond
        in one facet are a slice. Facets few nodes respond to are what the model barely holds."""
        concept = str(concept).strip(); t0 = time.time()
        if not concept: raise ValueError("Give a concept, e.g. Bluetooth.")
        fac = facets or {f: [t.replace("{C}", concept[:1].upper() + concept[1:]).replace("{c}", concept) for t in ts] for f, ts in self.FACETS.items()}
        names = list(fac.keys()); C, X, F = [], [], []        # concept acts, control acts (mean over controls), facet index
        used = []
        for fi, f in enumerate(names):
            for sent in fac[f]:
                a = self._span_acts(sent, concept)
                if a is None: continue
                ctrl = []
                for w in self.CONTROLS:
                    j = sent.lower().find(concept.lower()); alt = sent[:j] + w + sent[j + len(concept):]
                    b = self._span_acts(alt, w)
                    if b is not None: ctrl.append(b)
                if not ctrl: continue
                C.append(a); X.append(torch.stack(ctrl)); F.append(fi); used.append({"facet": f, "text": sent})
        if len(C) < 6: raise ValueError(f"Only {len(C)} sentences contained “{concept}”. Each sentence must contain the concept exactly as typed.")
        A = torch.stack(C)                                     # [S, L, I]
        Xs = torch.stack(X)                                    # [S, c, L, I]
        Xm = Xs.mean(1)                                        # [S, L, I]
        D = A - Xm                                             # concept minus its own sentence's controls
        sd = Xs.reshape(-1, self.L, self.I).std(0)
        sd = torch.maximum(sd, 0.5 * sd.median(dim=1, keepdim=True).values) + 1e-3   # floor: neurons silent on the controls must not look infinite
        Z = D / sd                                             # [S, L, I] per-sentence evidence, in units of the controls' spread
        Fi = torch.tensor(F, device=Z.device); nF = len(names)
        Zf = torch.stack([Z[Fi == f].mean(0) for f in range(nF)])                   # [nF, L, I] response per facet
        consist = (Z > 1).float().mean(0)                                            # share of sentences where it clearly fires
        overall = Zf.mean(0) * consist
        chosen = {}                                             # (l, j) -> (score, picked_for)
        for v, ix in zip(*[x.tolist() for x in torch.topk(overall.flatten(), k // 2)]):
            if v > 0: chosen[divmod(ix, self.I)] = (v, None)
        per_f = max(8, (k // 2) // nF)
        for f in range(nF):                                     # slices: concept-driven in this facet, much less in the others
            others = torch.cat([Zf[:f], Zf[f + 1:]]).mean(0)
            sel = torch.where(Zf[f] > 2, Zf[f] - others, torch.full_like(others, -1e9))
            for v, ix in zip(*[x.tolist() for x in torch.topk(sel.flatten(), per_f)]):
                if v > 0:
                    key = divmod(ix, self.I)
                    if key not in chosen: chosen[key] = (float(Zf[f].flatten()[ix]) * float(consist.flatten()[ix]) + v, f)
        nodes = []
        for (l, j), (v, picked) in chosen.items():
            per = [round(float(Zf[f, l, j]), 2) for f in range(nF)]; top = max(per) or 1e-6
            breadth = sum(1 for p_ in per if p_ >= 0.5 * top) / nF
            fav = max(range(nF), key=lambda f: per[f])
            kind = "core" if breadth >= 0.67 else ("slice" if breadth <= 0.5 else "mixed")
            nodes.append({"layer": l, "neuron": j, "score": round(v, 2), "z": round(float(Zf[:, l, j].mean()), 2), "fires_in": round(float(consist[l, j]), 2),
                          "facets": per, "breadth": round(breadth, 2), "kind": kind, "facet": names[fav], "form": l <= 2})
        nodes.sort(key=lambda n: -n["score"])
        # where the abstraction lives in depth, and the network lighting (per layer, per bin)
        depth = [0.0] * self.L
        for n in nodes: depth[n["layer"]] += n["score"]
        G_ = int(bins or auto_bins(self.I)); e = edges(self.I, G_).tolist(); light = [[0.0] * G_ for _ in range(self.L)]
        for n in nodes:
            b = next(bi for bi in range(G_) if e[bi] <= n["neuron"] < e[bi + 1]); light[n["layer"]][b] += n["score"]
        mx = max(max(r) for r in light) or 1.0
        light = [[round(x / mx, 3) for x in r] for r in light]
        # slices: per facet, how strongly the concept's nodes respond there, and its own neurons
        slices = []
        for fi, f in enumerate(names):
            vals = torch.topk(Zf[fi].flatten(), 50).values.tolist()
            own = sorted([n for n in nodes if n["kind"] == "slice" and n["facet"] == f], key=lambda n: -n["score"])
            slices.append({"facet": f, "strength": round(sum(vals) / max(1, len(vals)), 2), "neurons": [[n["layer"], n["neuron"]] for n in own[:12]], "count": len(own),
                           "sentences": [u["text"] for u in used if u["facet"] == f]})
        core = [n for n in nodes if n["kind"] == "core"]
        self._concept = {"concept": concept, "nodes": nodes}
        return {"concept": concept, "sentences": len(C), "controls": self.CONTROLS, "nodes": nodes, "core": [[n["layer"], n["neuron"]] for n in core[:24]], "core_count": len(core),
                "slices": slices, "depth": [round(x, 2) for x in depth], "light": light, "bins": G_, "seconds": round(time.time() - t0, 1),
                "note": "Nodes: MLP neurons that respond more to the concept than to unrelated words in the same sentences, in units of the controls' spread. Core: responds in at least two thirds of the facets (the abstraction, kept whatever the context). Slice: responds in at most half (one context of the concept). Layers 0-2 mostly hold the word's form (its spelling as tokens), not its meaning. Slice strength: the network's 50 strongest responses to the concept in that facet."}

    # ── a concept net: what the model understood, from four angles ──
    POOL = ("red orange yellow green blue purple pink brown black white grey gold silver crimson scarlet violet "
            "fire blood apple rose sun heart wine tomato cherry strawberry lava ruby brick rust autumn sunset sky ocean grass leaf forest snow ice night "
            "love anger passion danger warning stop alarm shame fear joy calm sadness hope peace energy heat warmth cold "
            "wireless radio signal network internet wifi cable router modem antenna phone smartphone laptop computer tablet keyboard mouse "
            "headphones earbuds speaker microphone audio music sound battery charger device gadget app software protocol standard data file "
            "connection pairing sync transfer link chip sensor camera screen television watch car engine train bridge road city village house "
            "door window kitchen bread coffee milk sugar salt water river mountain stone sand glass paper book pen letter word language name "
            "king queen viking denmark sweden history ancient medieval war army treaty law money bank market trade price company brand "
            "doctor hospital medicine disease virus cell gene brain mind dream memory thought idea science physics light wavelength color paint "
            "art painting music song dance sport football cricket game toy child mother father friend teacher school university student "
            "dog cat bird fish horse tree flower garden violin piano guitar glacier desert island planet star moon space time energy").split()
    STOP = set("the a an and or of to in on for with by is are was be as at it its this that from into your you our their his her not can will more most such than then also very about over".split())

    def _wordlike(self, t):
        import re
        w = t.strip()
        return w if re.fullmatch(r"[A-Za-z][A-Za-z\-]{2,}", w) and w.lower() not in self.STOP else None

    @torch.no_grad()
    def _vocab(self, vec, k=40):
        """Words a residual-stream vector points toward, read through the final norm and the unembedding."""
        lg = self.m.lm_head.weight.float() @ (self.m.model.norm.weight.float() * vec.float())
        out = []
        for i in torch.topk(lg, k * 4).indices.tolist():
            w = self._wordlike(self.tok.decode([i]))
            if w and w.lower() not in [o.lower() for o in out]: out.append(w)
            if len(out) >= k: break
        return out

    @torch.no_grad()
    def _say(self, prompt, n=90):
        ids = self.t._chat_ids(prompt, False)
        out = self.m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        while text.startswith("</think>"): text = text[len("</think>"):].strip()
        return text

    @torch.no_grad()
    def _rep(self, word):
        """A word's meaning inside the model: mean residual at its tokens, middle-to-late layers, two neutral frames."""
        lo, hi = self.L // 2, (3 * self.L) // 4; acc = []
        for fr in ("I read about {w} today.", "The word {w} came up in the meeting."):
            sent = fr.replace("{W}", word[:1].upper() + word[1:]).replace("{w}", word)
            r = self._span_resid(sent, word)
            if r is not None: acc.append(r[lo:hi + 1].mean(0))
        return torch.stack(acc).mean(0) if acc else None

    @torch.no_grad()
    def concept_net(self, concept, facets=None, probes=None, k_assoc=14):
        import re
        concept = str(concept).strip(); t0 = time.time()
        if not concept: raise ValueError("Give a concept, e.g. Bluetooth.")
        # 1. what it says: its own associations, in words
        said_text = self._say(f"List 15 single words or short terms you associate with {concept}, separated by commas. Only the list.")
        said = []
        for part in re.split(r"[,\n;]|\d+\.", said_text):
            w = part.strip(" .*-\t\"'").strip()
            if 2 < len(w) <= 28 and w.lower() != concept.lower() and w.lower() not in [x.lower() for x in said]: said.append(w)
        said = said[:15]
        # 2. side vocabulary: whole words the concept's own late-layer neurons push toward, by side
        cm = self._concept if (getattr(self, "_concept", None) and self._concept.get("concept", "").lower() == concept.lower()) else None
        if cm is None: self.concept_map(concept, facets); cm = self._concept
        side_vocab = {}; lm, nw = self.m.lm_head.weight.float(), self.m.model.norm.weight.float()
        for n in cm["nodes"]:
            if n["layer"] < (2 * self.L) // 3: continue
            lg = lm @ (nw * self.layers[n["layer"]].mlp.down_proj.weight[:, n["neuron"]].float())
            bucket = side_vocab.setdefault("core" if n["kind"] == "core" else n["facet"], {})
            got = 0
            for i in torch.topk(lg, 200).indices.tolist():
                t = self.tok.decode([i])
                if not t.startswith(" "): continue                 # whole words only, not word pieces
                w = self._wordlike(t)
                if w: bucket[w] = bucket.get(w, 0) + n["score"] / (1 + got); got += 1
                if got >= 6: break
        side_vocab = {k: [w for w, _ in sorted(v.items(), key=lambda x: -x[1])[:8]] for k, v in side_vocab.items() if v}
        # 3. inner neighbours and the net: every word's meaning inside the model (middle-to-late layers), each
        #    dimension standardised across all words (a few huge dimensions are shared by every word and would
        #    otherwise make everything look identical), compared by cosine.
        words = [concept]
        for w in said + [p for p in self.POOL]:
            if w.lower() != concept.lower() and w.lower() not in [x.lower() for x in words]: words.append(w)
        reps, kept = [], []
        for w in words:
            r = self._rep(w)
            if r is not None: reps.append(r); kept.append(w)
        R = torch.stack(reps); Z = (R - R.mean(0)) / (R.std(0) + 1e-3); Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-6); C = (Z @ Z.T).cpu()
        said_l = {w.lower() for w in said}
        pool_idx = [i for i, w in enumerate(kept) if i > 0 and w.lower() not in said_l]
        inside_idx = sorted(pool_idx, key=lambda i: -float(C[0, i]))[:k_assoc]
        inside = [kept[i] for i in inside_idx]
        net_idx = [0] + [i for i, w in enumerate(kept) if i > 0 and w.lower() in said_l] + inside_idx
        nodes = []
        for i in net_idx:
            w = kept[i]; src = ["concept"] if i == 0 else (["said"] if w.lower() in said_l else []) + (["inside"] if i in inside_idx else [])
            nodes.append({"word": w, "sim": round(float(C[0, i]), 3), "source": src})
        sub = C[net_idx][:, net_idx]
        edges = [[a, b, round(float(sub[a, b]), 3)] for a in range(1, len(net_idx)) for b in range(a + 1, len(net_idx)) if float(sub[a, b]) > 0.3]
        rank_all = sorted(range(1, len(kept)), key=lambda i: -float(C[0, i]))
        said_rank = {kept[i]: rank_all.index(i) + 1 for i in range(1, len(kept)) if kept[i].lower() in said_l}
        # 4. knowledge checks: does it complete facts about the concept?
        checks = []
        for pr in probes or []:
            text, ans = str(pr.get("prompt", "")).rstrip(), str(pr.get("answer", "")).strip()
            if not text or not ans: continue
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.layers[0].mlp.down_proj.weight.device)
            lp = torch.log_softmax(self.m(input_ids=ids).logits[0, -1].float(), -1)
            tgt = self.tok(" " + ans, add_special_tokens=False).input_ids[0]
            rank = int((lp > lp[tgt]).sum()) + 1; top = torch.topk(lp, 3)
            checks.append({"facet": pr.get("facet", ""), "prompt": text, "answer": ans, "p": round(float(lp[tgt].exp()), 4), "rank": rank,
                           "it_expected": [self.tok.decode([int(i)]).strip() for i in top.indices], "verdict": "knows" if rank == 1 else ("close" if rank <= 5 else "does not know")})
        return {"concept": concept, "said_text": said_text, "said": said, "inside": inside, "said_rank": said_rank, "pool": len(kept) - 1, "side_vocab": side_vocab, "nodes": nodes, "edges": edges,
                "checks": checks, "seconds": round(time.time() - t0, 1),
                "note": "said: the model's own list when asked (explaining). inside: the nearest words to the concept inside the model among its own list plus a fixed pool of everyday words, unprompted (understanding as position). said_rank: where each said word ranks among all words by inner closeness. side_vocab: whole words the concept's own late-layer neurons push toward. sim: cosine of meaning inside the model (layers L/2-3L/4, dimensions standardised). edges: pairs above 0.3. checks: the model's odds for the right word in short facts."}

    # ── grounding check: does the colour map built from text alone have the shape of real colour? ──
    COLORS = {"red": "#ff0000", "orange": "#ffa500", "yellow": "#ffff00", "green": "#008000", "blue": "#0000ff", "purple": "#800080",
              "pink": "#ffc0cb", "brown": "#a52a2a", "black": "#000000", "white": "#ffffff", "grey": "#808080", "cyan": "#00ffff",
              "magenta": "#ff00ff", "violet": "#ee82ee", "crimson": "#dc143c", "scarlet": "#ff2400", "teal": "#008080", "navy": "#000080",
              "gold": "#ffd700", "maroon": "#800000", "lime": "#00ff00", "turquoise": "#40e0d0", "beige": "#f5f5dc", "olive": "#808000"}

    @staticmethod
    def _lab(hexc):
        r, g, b = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in (r, g, b)]
        X = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047; Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]; Z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
        f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
        return (116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z)))

    @torch.no_grad()
    def _color_reps(self, names, frames):
        """Per word: residual after every layer at the word, averaged over frames -> [L, D]."""
        out = []
        for w in names:
            acc = [self._span_resid(f.replace("{w}", w), w) for f in frames]; acc = [a for a in acc if a is not None]
            out.append(torch.stack(acc).mean(0) if acc else None)
        return out

    @torch.no_grad()
    def color_geometry(self, focus="red"):
        names = list(self.COLORS)
        frames = {"neutral": ["I read about {w} today.", "The word {w} came up in the meeting."],
                  "colour": ["The wall was painted {w}.", "She wore a {w} dress to the party.", "The sky turned {w} at dusk."]}
        lab_all = {w: self._lab(self.COLORS[w]) for w in names}
        bands = {"early": (2, self.L // 4), "middle": (self.L // 4, self.L // 2), "middle-late": (self.L // 2, (3 * self.L) // 4), "late": ((3 * self.L) // 4, self.L - 1)}
        n = len(names); pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]; rank = lambda x: np.argsort(np.argsort(x))
        lab = np.array([lab_all[w] for w in names]); real = np.array([np.linalg.norm(lab[i] - lab[j]) for i, j in pairs])
        def score(R):
            Z = (R - R.mean(0)) / (R.std(0) + 1e-3); Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-6); C = (Z @ Z.T).cpu().numpy()
            inner = np.array([1 - C[i, j] for i, j in pairs]); return float(np.corrcoef(rank(inner), rank(real))[0, 1]), C, Z
        grid = {}; best = (-9, None, None, None)
        for fk, fr in frames.items():
            reps = torch.stack(self._color_reps(names, fr))            # [n, L, D]
            for bk, (lo, hi) in bands.items():
                rho, C, Z = score(reps[:, lo:hi + 1].mean(1)); grid[f"{fk}/{bk}"] = round(rho, 3)
                if rho > best[0]: best = (rho, C, Z, f"{fk}/{bk}")
        rho, C, Z, where = best
        C = (Z @ Z.T).cpu().numpy(); lab = np.array([self._lab(self.COLORS[w]) for w in names])
        inner = np.array([1 - C[i, j] for i, j in pairs])
        rng = np.random.default_rng(0); null = []
        for _ in range(500):
            p = rng.permutation(n); real_p = np.array([np.linalg.norm(lab[p[i]] - lab[p[j]]) for i, j in pairs]); null.append(np.corrcoef(rank(inner), rank(real_p))[0, 1])
        pval = float((np.array(null) >= rho).mean())
        Zn = Z.cpu().numpy(); U_, S_, Vt = np.linalg.svd(Zn - Zn.mean(0), full_matrices=False); xy = U_[:, :2] * S_[:2]
        fi = names.index(focus) if focus in names else None
        near = sorted([(names[j], round(float(C[fi, j]), 3)) for j in range(n) if j != fi], key=lambda x: -x[1])[:6] if fi is not None else []
        real_near = sorted([(names[j], round(float(np.linalg.norm(lab[fi] - lab[j])), 1)) for j in range(n) if j != fi], key=lambda x: x[1])[:6] if fi is not None else []
        return {"colors": [{"word": w, "hex": self.COLORS[w], "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3)} for i, w in enumerate(names)],
                "rho": round(rho, 3), "p": round(pval, 3), "best": where, "grid": grid, "focus": focus, "inner_nearest": near, "real_nearest": real_near,
                "note": "rho: rank correlation between how far apart two colour words sit inside the model and how different the colours really look (CIELAB distance of their standard sRGB values), over all pairs. p: share of 500 shuffles of the real colours that match as well. x, y: the first two principal components of the colour words' meanings inside the model."}

    # ── activation patching: where in the network does the answer get decided? ──
    @torch.no_grad()
    def patch(self, clean, corrupt):
        """Run the corrupted prompt, but copy in one piece of the clean run at a time (the residual stream after a
        layer, or one layer's attention or MLP output, at one position) and measure how much of the clean answer
        comes back. 1 = fully restored, 0 = no effect. Prompts must have the same number of tokens."""
        tok, m = self.tok, self.m; dev = self.layers[0].mlp.down_proj.weight.device
        ci = tok(clean, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        xi = tok(corrupt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        if ci.shape[1] != xi.shape[1]:
            raise ValueError(f"The two prompts must split into the same number of pieces: {ci.shape[1]} and {xi.shape[1]}. Swap in a word of the same length, e.g. one name for another" + (" (this model reads letter by letter, so the two words need the same number of letters)." if len(self.tok) < 1000 else "."))
        if ci.shape[1] > 48: raise ValueError("Keep each prompt under 48 pieces.")
        T = ci.shape[1]; cache = {}
        def grab(key):
            def f(mod, inp, out): cache[key] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return f
        hs = []
        for l, layer in enumerate(self.layers):
            hs += [layer.register_forward_hook(grab(("resid", l))), layer.self_attn.register_forward_hook(grab(("attn", l))), layer.mlp.register_forward_hook(grab(("mlp", l)))]
        try: lc = m(input_ids=ci).logits[0, -1].float()
        finally:
            for h in hs: h.remove()
        lx = m(input_ids=xi).logits[0, -1].float()
        a, b = int(lc.argmax()), int(lx.argmax())
        if a == b:                                       # same top answer: compare against the corrupt run's runner-up instead
            b = int(torch.topk(lx, 2).indices[1])
        diff = lambda lg: lg[..., a] - lg[..., b]
        d_clean, d_cor = float(diff(lc)), float(diff(lx)); span = d_clean - d_cor
        if abs(span) < 1e-3: raise ValueError("The two prompts lead to the same answer, so there is nothing to restore. Change one word that changes the answer.")
        out = {"pieces_clean": [tok.decode([int(x)]) for x in ci[0]], "pieces_corrupt": [tok.decode([int(x)]) for x in xi[0]],
               "clean_answer": tok.decode([a]), "corrupt_answer": tok.decode([b]), "p_clean": round(float(torch.softmax(lc, -1)[a]), 3), "p_corrupt": round(float(torch.softmax(lx, -1)[b]), 3)}
        xs = xi.expand(T, T).contiguous()                # one copy of the corrupt prompt per position, patched at that position
        for kind in ("resid", "attn", "mlp"):
            grid = []
            for l, layer in enumerate(self.layers):
                src = cache[(kind, l)][0]                 # [T, D]
                def put(mod, inp, out, src=src):
                    o = out[0] if isinstance(out, tuple) else out
                    o = o.clone(); idx = torch.arange(T, device=o.device); o[idx, idx] = src[idx].to(o.dtype)
                    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                mod = layer if kind == "resid" else (layer.self_attn if kind == "attn" else layer.mlp)
                h = mod.register_forward_hook(put)
                try: lg = m(input_ids=xs).logits[:, -1].float()
                finally: h.remove()
                grid.append([round((float(x) - d_cor) / span, 3) for x in diff(lg)])
            out[kind] = grid                              # [L][T]
        devices.empty_cache()
        return out

    # ── recording a forward pass ────────────────────────────────────────────
    @torch.no_grad()
    def record(self, prompt, n_new=24, lens=True, think=False, k_top=5, bins=None):
        tok, m, dev = self.tok, self.m, self.t.device
        ids = self.t._chat_ids(prompt, lens, think)
        if ids.shape[1] > MAX_PROMPT: raise ValueError(f"Prompt is {ids.shape[1]} tokens with the template; the recorder takes at most {MAX_PROMPT}. Shorten it or turn the lens off.")
        n_new = int(max(1, min(n_new, MAX_NEW))); G_ = int(bins or auto_bins(self.I)); full = []
        ctx = getattr(self.m.config, "max_position_embeddings", None)
        if ctx:
            if ids.shape[1] >= ctx: raise ValueError(f"This model reads at most {ctx} tokens and the prompt is {ids.shape[1]}. Shorten it.")
            n_new = min(n_new, ctx - ids.shape[1])   # small models read only a short context
        bi, cnt = bin_index(self.I, G_, self.layers[0].mlp.down_proj.weight.device)
        acts = {}
        hooks = [self.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: acts.__setitem__(l, inp[0][0].detach())) for l in range(self.L)]
        eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        steps = []; t0 = time.time()

        def summarise(o, positions, phase):
            hs = o.hidden_states; att = o.attentions
            for k, pos in enumerate(positions):
                lay = []
                for l in range(self.L):
                    a = acts[l][k].float().abs()
                    bins = torch.zeros(G_, device=a.device).index_add_(0, bi.to(a.device), a) / cnt.to(a.device)
                    top = torch.topk(acts[l][k].float().abs(), k_top)
                    A = att[l][0, :, k, :].float()                           # [H, T_total]
                    T = A.shape[-1]
                    ent = -(A.clamp_min(1e-9) * A.clamp_min(1e-9).log()).sum(-1) / math.log(max(T, 2))
                    look = (A[:, 1:].argmax(-1) + 1) if T > 1 else A.argmax(-1)   # skip the first token: heads park unused attention there (the 'sink')
                    sink = float(A[:, 0].mean())
                    h = hs[l + 1][0, k].float()
                    lg = torch.softmax(self.t.readout(h, l + 1), -1); p, i = lg.max(-1)
                    lay.append({"bins": [round(float(x), 3) for x in bins], "top": [[int(j), round(float(acts[l][k][j]), 2)] for j in top.indices],
                                "ent": [round(float(x), 2) for x in ent], "look": [int(x) for x in look], "row": [round(float(x), 3) for x in A.mean(0)],
                                "resid": round(float(h.norm()), 1), "sink": round(sink, 3), "lens": [tok.decode([int(i)]), round(float(p), 3)]})
                steps.append({"phase": phase, "pos": int(pos), "layers": lay})
                full.append(torch.stack([acts[l][k] for l in range(self.L)]).to("cpu", torch.float16))

        try:
            o = m(input_ids=ids, use_cache=True, output_hidden_states=True, output_attentions=True)
            summarise(o, range(ids.shape[1]), "read")
            past = o.past_key_values; written = []
            nxt = int(o.logits[0, -1].argmax())
            for s in range(n_new):
                written.append(nxt); steps[-1]["writes"] = tok.decode([nxt]); steps[-1]["p"] = round(float(torch.softmax(o.logits[0, -1].float(), -1)[nxt]), 3)
                if nxt in eos: break
                o = m(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=True)
                past = o.past_key_values
                summarise(o, [ids.shape[1] + s], "write")
                nxt = int(o.logits[0, -1].argmax())
            steps[-1]["writes"] = tok.decode([nxt]); steps[-1]["p"] = round(float(torch.softmax(o.logits[0, -1].float(), -1)[nxt]), 3)
        finally:
            for h in hooks: h.remove()
        pieces = [tok.decode([int(i)]) for i in ids[0]] + [tok.decode([w]) for w in written]
        self.last = {"acts": torch.stack(full), "pieces": pieces}
        return {"pieces": pieces, "prompt_len": int(ids.shape[1]), "steps": steps, "layers": self.L, "heads": self.H, "bins": G_, "edges": edges(self.I, G_).tolist(), "neurons_per_bin": round(self.I / G_, 1), "mlp": self.I,
                "kv_bytes_per_token": 2 * self.L * self.KV * self.hd * 2, "seconds": round(time.time() - t0, 1)}


def prompt_segments(t, prompt, system=None, think=False):
    """Which part of the templated prompt each token belongs to: 'system' (the E-level framing), 'user' (the typed
    text) or 'template' (role markers, special tokens, /no_think). Mirrors Tracer._chat_ids; None when the tokenizer
    gives no character offsets (a slow tokenizer)."""
    tok = t.tok
    try:
        if not getattr(tok, "chat_template", None):
            enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
            return ["user" if b > a else "template" for a, b in enc["offset_mapping"]]
        off = (not think) and t.no_think_supported
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt + " /no_think" if off else prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + ("<think>\n\n</think>\n\n" if off else "")
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    s0 = text.find(system) if system else -1; s1 = s0 + len(system) if s0 >= 0 else -1
    u0 = text.find(prompt, max(0, s1)) if prompt else -1; u1 = u0 + len(prompt) if u0 >= 0 else -1
    out = []
    for a, b in enc["offset_mapping"]:
        mid = (a + b) / 2
        out.append("template" if b <= a else "system" if s0 >= 0 and s0 <= mid < s1 else "user" if u0 >= 0 and u0 <= mid < u1 else "template")
    return out


@torch.no_grad()
def knockout(t, prompt, lens=True, target=None, k=8, layers=None, prefix_ids=None, ablation="zero", seeds=5, stats=None, think=False):
    """Switch neurons off and re-ask: how much of the next token lived in them, against random sets of the same size.
    The k strongest-firing neurons (|activation| at the last position) in each chosen layer are switched off; then
    `seeds` random sets of k per chosen layer (seeds 0..seeds-1) are switched off instead, one set at a time.
    ablation 'zero': set to zero (zero-ablation). 'mean': held at their average over ordinary text (mean-ablation,
    needs Anatomy.build_stats). Applied at every position of the input, for this one answer; nothing is kept."""
    m = t.model; I = int(m.config.intermediate_size); L = t.L; k = int(max(1, min(int(k), I)))
    if ablation not in ("zero", "mean"): raise ValueError(f"ablation must be 'zero' or 'mean', not {ablation!r}")
    if ablation == "mean" and stats is None:
        raise ValueError("Mean-ablation needs each neuron's average on ordinary text, and it is not built for these weights yet. Build the normal range first (Neural net).")
    ids = t._chat_ids(prompt, lens, think)
    if prefix_ids: ids = torch.cat([ids, torch.tensor([list(map(int, prefix_ids))], device=t.device)], 1)
    sel = sorted({int(l) for l in layers}) if layers else list(range(L))
    if any(l < 0 or l >= L for l in sel): raise ValueError(f"layers must be 0 to {L - 1}")
    mean = stats["mean"] if ablation == "mean" else None
    seeds = int(max(0, min(int(seeds), 10)))
    with t.lock:
        lg = torch.softmax(m(input_ids=ids).logits[0, -1].float(), -1)
        tgt = int(lg.argmax()) if target is None else int(target); p0 = float(lg[tgt])
        acts = {l: t.acts[l].clone() for l in sel}

        def run(chosen):
            hooks = []
            for l, idx in chosen.items():
                ix = torch.tensor(idx, device=t.device, dtype=torch.long)
                fill = None if mean is None else mean[l][idx].to(t.device)
                def hook(mod, inp, ix=ix, fill=fill):
                    x = inp[0].clone()
                    if fill is None: x[..., ix] = 0
                    else: x[..., ix] = fill.to(x.dtype)
                    return (x,)
                hooks.append(m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
            try: lg2 = torch.softmax(m(input_ids=ids).logits[0, -1].float(), -1)
            finally:
                for h in hooks: h.remove()
            top = int(lg2.argmax())
            return {"p_after": round(float(lg2[tgt]), 4), "now_says": t.tok.decode([top]), "now_id": top, "now_p": round(float(lg2[top]), 4)}

        chosen = {l: torch.topk(acts[l].abs(), k).indices.tolist() for l in sel}
        res = run(chosen); ctrl = []
        for s in range(seeds):
            g = torch.Generator().manual_seed(s)
            ctrl.append({"seed": s, **run({l: torch.randperm(I, generator=g)[:k].tolist() for l in sel})})
    ps = [c["p_after"] for c in ctrl]
    return {"target": t.tok.decode([tgt]), "target_id": tgt, "p_before": round(p0, 4), **res, "knocked_out": k * len(sel), "of_neurons": L * I,
            "layers": len(sel), "layer_ids": sel, "k": k, "plan": "top", "ablation": ablation,
            "chosen": {str(l): v for l, v in chosen.items()} if len(sel) <= 4 else None,
            "control": ctrl, "control_p_after": [min(ps), max(ps)] if ps else None, "context_tokens": int(ids.shape[1]),
            "note": "top: the k neurons with the largest |activation| at the last position, per chosen layer. control: one random set of k per chosen layer per seed (torch.randperm, manual_seed = seed). zero: set to 0; mean: held at the neuron's average over ordinary text. At every position of the input; undone after the answer."}
