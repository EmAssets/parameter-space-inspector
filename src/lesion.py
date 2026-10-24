"""
The reverse tipping point: remove structure from the finished model and watch for the point
where the signs of intelligence go out.

Three kinds of lesion, all reversible:
  structural  a share of MLP neurons and attention heads, masked by hooks (random, least- or
              most-active for the current question)
  weights     a share of the raw weights zeroed in place (random, or the smallest/largest by
              magnitude); restored by reloading the touched tensors from the safetensors files (or a
              CPU copy kept before zeroing when there are none), each copy checked
  manual      chosen layers / heads / neurons
  concept     neurons that respond to one concept ("cricket") and not to a control ("football"),
              found by contrasting activations over templated sentences; then deleted

The scorecard after each lesion: the live understanding tests (src/ui/understanding-tests.js, shared with the
Understanding view; the headline count leaves out the self-referential test, which is reported apart, and any tests the
user added are scored and reported apart too), copying a repeated sequence under the shared protocol
(checkpoints.COPY_PROTOCOL), ordinary text (bits/token), the answer to the user's question, and retention: the share of
the model's own pre-lesion answer tokens it would still produce.
"""
import glob, json, math, os, random, time
import checkpoints as ckpt_mod
import torch
from safetensors import safe_open

def _load_tests():
    """The live understanding battery, shared with the Understanding view: src/ui/understanding-tests.js (the value
    after "PSI_UNDERSTANDING_TESTS =" is strict JSON). Each: {id, what, prompt, good, bad, self_referential?}."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "understanding-tests.js")
    src = open(path, encoding="utf-8").read(); src = src[src.index("PSI_UNDERSTANDING_TESTS ="):]
    return json.loads(src[src.index("["):src.rindex("]") + 1])


TESTS = _load_tests()
TEXT = ("The cycle rickshaw offers low-cost, zero-emission transport tailored to narrow, crowded urban streets. Operating costs are minimal, and the "
        "vehicle can be repaired by local mechanics with basic tools. The bakery on the corner opens before dawn. By seven the first loaves are gone "
        "and the owner is already shaping the second batch. Regulars know to come early; newcomers learn by missing out once.")
TEMPLATES = ["I spent the whole weekend watching {c} with my cousins.", "The rules of {c} take a while to learn, but the basics are simple.",
             "{C} is enormously popular in some countries and almost unknown in others.", "She has played {c} since she was nine years old.",
             "The commentators argued about the {c} match for an hour.", "Our school had a {c} team but no proper ground to practise on.",
             "He explained the history of {c} to anyone who would listen.", "Tickets for the {c} final sold out in minutes."]


def _hash_of(path):
    """The weights hash as the server computes it at load (trace_server._weights_hash): the index file, or the first MB."""
    import hashlib
    idx = os.path.join(path, "model.safetensors.index.json")
    try:
        if os.path.exists(idx):
            with open(idx, "rb") as f: return hashlib.sha256(f.read()).hexdigest()
        for n in sorted(os.listdir(path)):
            if n.endswith(".safetensors") or (n.endswith(".bin") and "model" in n):
                with open(os.path.join(path, n), "rb") as f: return hashlib.sha256(f.read(1 << 20)).hexdigest()
    except OSError: return None
    return None


class Lesion:
    def __init__(self, tracer, weights_dir):
        self.t = tracer; self.model = tracer.model; self.tok = tracer.tok; self.dev = tracer.device
        self.weights_dir = weights_dir
        self.hooks = []; self.touched = set(); self.state = {"kind": "none"}; self.ref = {}
        # What the hooks mask, kept so a cut can be saved: a masked input of down_proj / o_proj is a zeroed column there.
        self.masked_n = {}; self.masked_h = set()
        idx = os.path.join(weights_dir, "model.safetensors.index.json")
        self.where = json.load(open(idx, encoding="utf-8"))["weight_map"] if os.path.exists(idx) else None
        # Where a zeroed tensor comes back from: the sharded files (index), the one weights file, or, with neither,
        # a copy on the CPU taken just before zeroing (self.snap). restore_verifiable: the source is the files on disk.
        single = os.path.join(weights_dir, "model.safetensors")
        self.single = "model.safetensors" if self.where is None and os.path.exists(single) else None
        self.restore_source = "index" if self.where else "single_file" if self.single else "cpu_snapshot"
        self.restore_verifiable = self.restore_source != "cpu_snapshot"
        self.snap = {}

    # ── apply / restore ──────────────────────────────────────────────────────
    def restore(self):
        """Remove every lesion hook and copy every zeroed tensor back from its source, checking each copy. Anything
        that could not be restored raises (the server answers with an error) and the state keeps saying the model is
        changed: the kind is "none" only when nothing is left changed."""
        for h in self.hooks: h.remove()
        self.hooks = []; self.masked_n = {}; self.masked_h = set()
        if not self.touched:
            self.state = {"kind": "none"}; return self.state
        t0 = time.time(); params = dict(self.model.named_parameters()); files = {}; failed = []; done = set()
        for name in sorted(self.touched):
            try:
                if name in self.snap: src = self.snap[name]
                else:
                    f = self.where.get(name) if self.where else self.single
                    if f is None: raise KeyError("no file holds it and no copy was kept")
                    if f not in files: files[f] = safe_open(os.path.join(self.weights_dir, f), "pt")
                    src = files[f].get_tensor(name)
                with torch.no_grad():
                    params[name].copy_(src.to(params[name].dtype))
                    if not torch.equal(params[name].detach().cpu(), src.to(params[name].dtype).cpu()): raise ValueError("copy does not match its source")
                done.add(name)
            except Exception as e:
                failed.append({"tensor": name, "error": str(e)[:300]})
        self.touched -= done
        for name in done: self.snap.pop(name, None)
        if failed:
            prev = self.state if self.state.get("kind") != "none" else {"kind": "weights"}
            self.state = {**prev, "restore_failed": failed, "still_changed_tensors": len(self.touched)}
            raise RuntimeError(f"restore: {len(failed)} of {len(failed) + len(done)} tensors could not be restored and stay changed: "
                               + "; ".join(f"{x['tensor']}: {x['error']}" for x in failed[:5]))
        self.state = {"kind": "none", "restored_tensors": len(done), "restore_s": round(time.time() - t0, 1), "restore_source": self.restore_source}
        return self.state

    def _mask_neurons(self, per_layer):   # {layer: [neuron idx]}
        for l, idx in per_layer.items():
            if not idx: continue
            self.masked_n.setdefault(int(l), set()).update(int(i) for i in idx)
            def hook(m, inp, idx=idx):
                x = inp[0].clone(); x[..., idx] = 0; return (x,)
            self.hooks.append(self.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))

    def _mask_heads(self, heads):   # [[layer, head]]
        self.hooks += self.t._head_hooks(heads)
        self.masked_h.update((int(l), int(h)) for l, h in (heads or []))

    def structural(self, frac_neurons=0.0, frac_heads=0.0, mode="random", seed=0, question=None):
        """Mask a share of MLP neurons and heads. mode random | least (least active for the
        question) | most (most active; "top" is accepted for it, "bottom" for least)."""
        mode = {"top": "most", "bottom": "least"}.get(mode, mode)   # an unknown mode used to fall through to least-first silently
        if mode not in ("random", "least", "most"): raise ValueError(f'mode must be "random", "least" or "most" (got {mode!r})')
        self.restore(); rng = random.Random(seed); L = self.t.L; I = self.model.config.intermediate_size; Hn = self.model.config.num_attention_heads
        acts = None; asked = {"neurons": frac_neurons, "heads": frac_heads}
        if mode != "random" and question:
            with torch.no_grad(): self.model(input_ids=self.t._chat_ids(question, False))
            acts = {l: self.t.acts[l].abs() for l in range(L) if l in self.t.acts}
        per_layer = {}; kn = int(round(frac_neurons * I))
        for l in range(L):
            if kn <= 0: break
            if acts and l in acts:
                order = torch.argsort(acts[l], descending=(mode == "most")).tolist(); per_layer[l] = order[:kn]
            else: per_layer[l] = rng.sample(range(I), kn)
        kh = int(round(frac_heads * L * Hn)); heads = []
        if kh > 0:
            if mode != "random":
                scan = self.t.heads_scan(); flat = sorted(((v, l, h) for l, row in enumerate(scan["heads"]) for h, v in enumerate(row)), reverse=(mode == "most"))
                heads = [[l, h] for v, l, h in flat[:kh]]
            else:
                allh = [[l, h] for l in range(L) for h in range(Hn)]; heads = rng.sample(allh, kh)
        self._mask_neurons(per_layer); self._mask_heads(heads)
        # Two rankings hide under "least/most active": neurons by |activation| at the last token of question 1, heads by
        # induction score (the head scan). Without a question the neurons fall back to random, and the state says so.
        by_n = "random" if mode == "random" else ("activation on question 1" if acts else "random (no question to rank by)")
        by_h = "random" if mode == "random" else "induction score"
        self.state = {"kind": "structural", "mode": mode, "neurons": sum(len(v) for v in per_layer.values()), "of_neurons": L * I, "heads": len(heads), "of_heads": L * Hn,
                      "neurons_by": by_n, "heads_by": by_h, "seed": seed if "random" in (by_n + by_h) else None, "asked": asked}
        return self.state

    def manual(self, layers=None, heads=None, neurons_per_layer=0, question=None):
        """Zero whole layers' MLPs (list of layer indices), chosen heads [[l,h]], and the k most
        active neurons per layer for the question."""
        self.restore(); per_layer = {}
        for l in (layers or []): per_layer[int(l)] = list(range(self.model.config.intermediate_size))
        if neurons_per_layer and question:
            with torch.no_grad(): self.model(input_ids=self.t._chat_ids(question, False))
            for l in range(self.t.L):
                if l in per_layer or l not in self.t.acts: continue
                per_layer[l] = torch.topk(self.t.acts[l].abs(), int(neurons_per_layer)).indices.tolist()
        self._mask_neurons(per_layer); self._mask_heads(heads or [])
        self.state = {"kind": "manual", "layers": layers or [], "heads": len(heads or []), "neurons": sum(len(v) for v in per_layer.values())}
        return self.state

    def weights(self, frac=0.1, scope="mlp", criterion="random", seed=0):
        """Zero a share of the raw weights in place. scope mlp | attn | all. criterion random |
        small (the smallest by magnitude) | large."""
        self.restore(); g = torch.Generator(device=self.dev).manual_seed(seed); n_total = 0; n_zero = 0
        for name, p in self.model.named_parameters():
            if p.dim() != 2 or "embed" in name or "lm_head" in name: continue
            if scope == "mlp" and ".mlp." not in name: continue
            if scope == "attn" and ".self_attn." not in name and ".attention." not in name: continue
            # No weights file to restore from: keep a CPU copy of the tensor before it is changed.
            if self.restore_source == "cpu_snapshot" and name not in self.snap: self.snap[name] = p.detach().to("cpu", copy=True)
            with torch.no_grad():
                if criterion == "random":
                    for r0 in range(0, p.shape[0], 2048):
                        blk = p[r0:r0 + 2048]; m = torch.rand(blk.shape, generator=g, device=self.dev) < frac; blk.masked_fill_(m, 0); n_zero += int(m.sum())
                else:
                    samp = p.flatten()[torch.randint(0, p.numel(), (min(p.numel(), 2_000_000),), generator=g, device=self.dev)].abs().float()
                    thr = torch.quantile(samp, frac if criterion == "small" else 1 - frac)
                    for r0 in range(0, p.shape[0], 2048):
                        blk = p[r0:r0 + 2048]; m = (blk.abs() < thr) if criterion == "small" else (blk.abs() > thr); blk.masked_fill_(m, 0); n_zero += int(m.sum())
            n_total += p.numel(); self.touched.add(name)
        self.state = {"kind": "weights", "scope": scope, "criterion": criterion, "frac": frac, "zeroed": n_zero, "of": n_total, "tensors": len(self.touched),
                      "achieved": round(n_zero / max(1, n_total), 5), "seed": seed}
        return self.state

    # ── save the cut model ───────────────────────────────────────────────────
    def save(self, name=None):
        """Write the model as it is cut now to a new folder beside the original, which is never touched. The weight
        files are read again from disk and changed there, so every untouched value keeps the files' own bytes:
          masked neurons  zeroed columns of that layer's down_proj weight (the hook zeroes the same inputs: same output)
          masked heads    zeroed columns of o_proj for that head's slice (likewise)
          raw weights     zeroed where the in-memory tensor is 0 and the file's value is not
        The other files (config, tokenizer, index, ...) are copied. cut.json records what was cut, the base weights' hash
        and the SHA-256 of every file written; each changed tensor is read back before the folder is given its name."""
        import hashlib, re, shutil
        from safetensors.torch import save_file
        if not (self.hooks or self.touched): raise ValueError("nothing is cut: make a cut first, then save it")
        if self.state.get("restore_failed"): raise ValueError("a restore failed partway, so what is cut is unclear: restore the model, cut again, then save")
        if self.hooks and not (self.masked_n or self.masked_h): raise ValueError("the hooks in force were not placed by a cut this view can save")
        if self.restore_source == "cpu_snapshot":
            raise ValueError("this model has no safetensors weight files to build the copy from: only a model loaded from safetensors can be saved cut")
        base = os.path.abspath(self.weights_dir).rstrip("/\\")
        h_now = _hash_of(base)
        if self.t.weights_hash and h_now != self.t.weights_hash: raise ValueError("the weight files changed on disk since the model was loaded: reload it, cut again, then save")
        kind = self.state.get("kind", "cut")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")[:80]
        out = os.path.join(os.path.dirname(base), safe or f"{os.path.basename(base)}-cut-{kind}-{time.strftime('%Y%m%d-%H%M%S')}")
        if os.path.exists(out): raise ValueError(f"{out} already exists: choose another name")
        files = sorted(set(self.where.values())) if self.where else [self.single]
        need = sum(os.path.getsize(os.path.join(base, f)) for f in files)
        free = shutil.disk_usage(os.path.dirname(base)).free
        if free < need * 1.05 + (64 << 20): raise ValueError(f"not enough free space beside the model: {need / 2**30:.2f} GB needed, {free / 2**30:.2f} GB free")
        t0 = time.time()
        pname = {id(p): n for n, p in self.model.named_parameters()}
        cfg = self.model.config; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        cols = {}   # tensor name -> the columns to zero
        for l, idx in self.masked_n.items(): cols.setdefault(pname[id(self.model.model.layers[l].mlp.down_proj.weight)], set()).update(idx)
        for l, h in self.masked_h: cols.setdefault(pname[id(self.model.model.layers[l].self_attn.o_proj.weight)], set()).update(range(h * hd, (h + 1) * hd))
        params = dict(self.model.named_parameters()); changed = {}
        tmp = out + ".partial"; os.makedirs(tmp)
        try:
            for f in files:
                d = {}
                with safe_open(os.path.join(base, f), "pt") as src:
                    meta = src.metadata() or {"format": "pt"}
                    for k in src.keys():
                        t = src.get_tensor(k)
                        if k in cols or k in self.touched:
                            t = t.clone(); before = int((t == 0).sum())
                            if k in cols: t[:, sorted(cols[k])] = 0
                            if k in self.touched: t.masked_fill_((params[k].detach() == 0).to("cpu") & (t != 0), 0)
                            changed[k] = {"file": f, "zeroed": int((t == 0).sum()) - before, "of": t.numel()}
                        d[k] = t
                save_file(d, os.path.join(tmp, f), metadata=meta); del d
            missing = (set(cols) | self.touched) - set(changed)
            if missing: raise ValueError(f"{len(missing)} changed tensors were not found in the weight files, e.g. {sorted(missing)[:3]}")
            for n in os.listdir(base):   # config, tokenizer, index, ...: everything but the weights and other formats
                q = os.path.join(base, n)
                if os.path.isfile(q) and n not in files and n != "cut.json" and not n.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf", ".partial")):
                    shutil.copy2(q, os.path.join(tmp, n))
            for k, c in changed.items():   # read back: each changed tensor holds the zeros it should
                with safe_open(os.path.join(tmp, c["file"]), "pt") as r:
                    t = r.get_tensor(k)
                    if k in cols and bool((t[:, sorted(cols[k])] != 0).any()): raise ValueError(f"read-back: {k} still has values in its cut columns")
                    if k in self.touched and bool(((params[k].detach() == 0).to("cpu") & (t != 0)).any()): raise ValueError(f"read-back: {k} has values the cut zeroed")
            sha = {}
            for n in sorted(os.listdir(tmp)):
                hsh = hashlib.sha256()
                with open(os.path.join(tmp, n), "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 22), b""): hsh.update(b)
                sha[n] = hsh.hexdigest()
            hf_src = {}
            try:
                with open(os.path.join(base, "hf-source.json"), encoding="utf-8") as fh: hs = json.load(fh)
                if hs.get("repo"): hf_src = {"hf_repo": hs["repo"], "hf_commit": hs.get("commit")}
            except Exception: pass
            rec = {"what": "A model saved from the Parameter Space Inspector's Tipping point: the base model with a cut baked into its weights.",
                   "base": {"folder": base.replace("\\", "/"), "name": os.path.basename(base), **hf_src, "weights_hash": self.t.weights_hash, "weights_hash_of": self.t.weights_hash_of},
                   "cut": dict(self.state),
                   "masked_neurons": {str(l): sorted(v) for l, v in sorted(self.masked_n.items())}, "masked_heads": sorted(list(x) for x in self.masked_h),
                   "changed_tensors": changed,
                   "how": "Masked neurons became zeroed columns of down_proj and masked heads zeroed columns of o_proj, which give the same outputs as the "
                          "hooks; raw-weight cuts were zeroed where the in-memory tensor was 0. Every other value is the original files' own.",
                   "files_sha256": sha, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1)}
            with open(os.path.join(tmp, "cut.json"), "w", encoding="utf-8") as fh: json.dump(rec, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, out)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True); raise
        return {"folder": out.replace("\\", "/"), "gb": round(need / 2**30, 2), "changed_tensors": len(changed),
                "zeroed": sum(c["zeroed"] for c in changed.values()), "seconds": rec["seconds"], "cut": rec["cut"]}

    # ── concepts ─────────────────────────────────────────────────────────────
    def _mean_acts(self, sentences):
        L = self.t.L; sums = {l: 0 for l in range(L)}; n = 0
        store = {}
        hooks = [self.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, inp, l=l: store.__setitem__(l, inp[0][0].detach().float())) for l in range(L)]
        try:
            with torch.no_grad():
                for s in sentences:
                    ids = self.tok(s, return_tensors="pt").input_ids.to(self.dev); self.model(input_ids=ids)
                    for l in range(L): sums[l] = sums[l] + store[l].mean(0)
                    n += 1
        finally:
            for h in hooks: h.remove()
        return {l: sums[l] / n for l in range(L)}

    def concept_neurons(self, concept, keep, k=300, skip_first=2):
        """Neurons whose activation is higher for sentences about `concept` than the same
        sentences about `keep`, normalised by the keep-side spread. Top k over all layers."""
        cs = [t.format(c=concept, C=concept.capitalize()) for t in TEMPLATES]; ks = [t.format(c=keep, C=keep.capitalize()) for t in TEMPLATES]
        a, b = self._mean_acts(cs), self._mean_acts(ks)
        cand = []
        # Skip the first layers (they hold the word's spelling) and the last in a deep model; a shallow one
        # (the toy Llama has 2 layers) has nothing to spare, so every layer is searched, and at most a quarter
        # of its neurons are taken.
        lo, hi = (skip_first, self.t.L - 1) if self.t.L >= 8 else (0, self.t.L)
        k = min(k, max(1, (hi - lo) * int(self.model.config.intermediate_size) // 4))
        for l in range(lo, hi):
            d = a[l] - b[l]; sd = torch.cat([a[l], b[l]]).std() + 1e-6; s = d / sd
            top = torch.topk(s, min(k, s.numel()))
            cand += [(float(v), l, int(i)) for v, i in zip(top.values, top.indices)]
        cand.sort(reverse=True); chosen = cand[:k]
        per_layer = {}
        for v, l, i in chosen: per_layer.setdefault(l, []).append(i)
        return {"concept": concept, "keep": keep, "k": k, "neurons": [{"layer": l, "neuron": i, "score": round(v, 2)} for v, l, i in chosen[:40]],
                "by_layer": {str(l): len(v) for l, v in sorted(per_layer.items())}, "_per_layer": per_layer}

    def delete_concept(self, concept, keep, k=300):
        self.restore(); found = self.concept_neurons(concept, keep, k)
        self._mask_neurons(found.pop("_per_layer"))
        self.state = {"kind": "concept", **found}
        return self.state

    # ── scorecard ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def answer(self, question, n_new=48):
        ids = self.t._chat_ids(question, False)
        ctx = getattr(self.model.config, "max_position_embeddings", None)
        if ctx: n_new = max(1, min(n_new, ctx - ids.shape[1]))
        out = self.model.generate(ids, max_new_tokens=n_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        while text.startswith("</think>"): text = text[len("</think>"):].strip()   # the template already closed the thinking block
        return {"text": text, "ids": new.tolist()}

    @torch.no_grad()
    def retention(self, question, ref_ids):
        ids = self.t._chat_ids(question, False); full = torch.cat([ids, torch.tensor([ref_ids], device=self.dev)], 1)
        lg = self.model(input_ids=full).logits[0, ids.shape[1] - 1:-1].float()
        pred = lg.argmax(-1); tgt = torch.tensor(ref_ids, device=self.dev)
        lp = torch.log_softmax(lg, -1)[torch.arange(len(ref_ids)), tgt]
        return {"token_agreement": round(float((pred == tgt).float().mean()), 3), "mean_logp": round(float(lp.mean()), 3)}

    @torch.no_grad()
    def scorecard(self, question=None, questions=None, extra=None):
        """extra: tests the user added in the Understanding view ([{prompt, good, bad, what?}]); defaults to
        self.extra_tests (set per request by the server). Reported apart from the headline count."""
        tok, model = self.tok, self.model; t0 = time.time()
        extra = [x for x in (extra if extra is not None else getattr(self, "extra_tests", None) or []) if x.get("prompt") and x.get("good") and x.get("bad")]
        tests = []
        for t in TESTS:
            m = self.t.answer_margin(t["prompt"], t["good"], t["bad"], model)
            tests.append({"id": t["id"], "what": t["what"], "margin": round(m, 2), "pass": m > 0, "self_referential": bool(t.get("self_referential"))})
        for i, t in enumerate(extra[:20]):
            m = self.t.answer_margin(t["prompt"], list(t["good"]), list(t["bad"]), model)
            tests.append({"id": f"yours-{i + 1}", "what": t.get("what") or "your test", "prompt": t["prompt"], "margin": round(m, 2), "pass": m > 0, "custom": True})
        core = [x for x in tests if not x.get("self_referential") and not x.get("custom")]
        selfref = [x for x in tests if x.get("self_referential")]; yours = [x for x in tests if x.get("custom")]
        seqs, P = ckpt_mod.repeat_sequences(model.get_input_embeddings().weight.shape[0], getattr(model.config, "max_position_embeddings", None), self.dev)
        first = second = 0.0; n = P["n"]
        for ids in seqs:
            lg = torch.log_softmax(model(ids).logits[0, :-1].float(), -1); tgt = ids[0, 1:]; nll = -lg[torch.arange(len(tgt)), tgt]
            first += float(nll[:n - 1].mean()) / len(seqs); second += float(nll[n:].mean()) / len(seqs)
        ids = self.t.ctx_ids(TEXT); lg = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
        bits = float((-lg[torch.arange(ids.shape[1] - 1), ids[0, 1:]]).mean()) / math.log(2)
        card = {"understanding_passed": sum(x["pass"] for x in core), "understanding_of": len(core),
                "understanding_margin": round(sum(x["margin"] for x in core) / max(1, len(core)), 2),
                "understanding_rule": "passed = log-odds > 0 (fixed threshold); read against the intact row",
                "self_referential": [{k: x[k] for k in ("id", "what", "margin", "pass")} for x in selfref],
                "yours": {"passed": sum(x["pass"] for x in yours), "of": len(yours)} if yours else None,
                "tests": tests, "margins": [x["margin"] for x in tests if not x.get("custom")],
                "copy_nats": round(first - second, 2), "copy_protocol": P, "text_bits": round(bits, 3), "answers": []}
        for q in (questions or ([question] if question else [])):
            a = self.answer(q); ref = self.ref.get(q)
            entry = {"question": q, "answer": a["text"]}
            if ref is not None: entry["retention"] = self.retention(q, ref["ids"]); entry["reference"] = ref["text"]
            card["answers"].append(entry)
        card["seconds"] = round(time.time() - t0, 1); card["state"] = self.state
        return card

    def baseline(self, questions):
        """Record the intact model's answers as the reference for retention."""
        self.restore()
        for q in questions:
            if q not in self.ref: self.ref[q] = self.answer(q)
        return {q: self.ref[q]["text"] for q in questions}
