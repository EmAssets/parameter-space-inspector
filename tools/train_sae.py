"""
Train a sparse autoencoder (TopK, Gao et al. 2024) on one layer of the inside model, then index its features:
how often each fires, its largest activation, and its strongest examples in the text sample. The result goes where
the app's Features view reads it (src/sae.py: <data folder>/sae/<model key>/L<layer>/).

  python tools/train_sae.py --model <models>/qwen3-4b-bf16 --layer 18 [--corpus file] [--tokens 400000]
                            [--expansion 8] [--k 32] [--epochs 4] [--device auto|cuda|cpu]

The sample: a JSON list of paragraphs (the app's neuron-corpus.json), a .jsonl with "text", or plain text (paragraphs
split at blank lines). The layer's state is read after layer L (hidden_states[L]); the first piece of every window is
left out (its state is an outlier many times larger than the rest, the attention sink).
Progress is printed as lines “PSI-PROGRESS {json}” for the app; the last line is “PSI-DONE {json}” or “PSI-ERROR …”.
It loads its own copy of the model: the app stops the inside engine first so the two do not share the GPU.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sae as sae_mod


def say(kind, **o): print(f"PSI-{kind} {json.dumps(o)}", flush=True)


def read_corpus(p):
    if p.endswith(".json"):
        d = json.load(open(p, encoding="utf-8")); return [x if isinstance(x, str) else x.get("text", "") for x in d]
    if p.endswith(".jsonl"):
        return [json.loads(l).get("text", "") for l in open(p, encoding="utf-8") if l.strip()]
    return [x for x in open(p, encoding="utf-8").read().split("\n\n") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--corpus", default=None); ap.add_argument("--tokens", type=int, default=400000)
    ap.add_argument("--expansion", type=int, default=8); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--examples", type=int, default=12); ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed)
    from trace_server import _weights_hash, _neox_names
    corpus = a.corpus or os.environ.get("PSI_CORPUS") or (os.path.join(os.environ["PSI_CACHE_DIR"], "datasets", "neuron-corpus.json") if os.environ.get("PSI_CACHE_DIR") else None)
    if not corpus or not os.path.exists(corpus): raise SystemExit(f"PSI-ERROR no text sample found ({corpus}); pass --corpus")
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    say("PROGRESS", stage="load", note=f"loading the model on {dev}")
    tok = AutoTokenizer.from_pretrained(a.model)
    # Only the layers up to the one read are kept: the rest are never used, and leaving them out roughly halves the
    # memory for a middle layer (so it fits beside other programs on the GPU). Loaded on the CPU, cut, then moved.
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
    L = model.config.num_hidden_layers
    if not 1 <= a.layer < L: raise SystemExit(f"PSI-ERROR layer must be 1 to {L - 1} (the last layer's state is after the final norm)")
    stack = model.gpt_neox if hasattr(model, "gpt_neox") else model.model
    # One layer past the one read: the stack's last state is returned after the final norm, so layer L must not be last.
    keep = a.layer + 1
    stack.layers = torch.nn.ModuleList(list(stack.layers)[:keep]); model.config.num_hidden_layers = keep
    if hasattr(model.config, "layer_types") and model.config.layer_types: model.config.layer_types = model.config.layer_types[:keep]
    for p in model.parameters(): p.requires_grad_(False)
    if dev == "cuda":
        need = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 30 + 1.5; free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free < need: raise SystemExit(f"PSI-ERROR the GPU has {free:.1f} GB free and layers 1 to {keep} need about {need:.1f} GB: close a program that uses the GPU, pick an earlier layer, or train on the CPU (slow)")
        model = model.to(dev)
    _neox_names(model)
    whash, _ = _weights_hash(a.model)

    # ── the layer's states over the sample ──
    allp = [p for p in read_corpus(corpus) if p.strip()]
    # The last 5% of the sample (at least 20 paragraphs) is kept back: the app evaluates the SAE on it (src/sae.py
    # evaluate and contexts), so its quality is measured on text it never learned from.
    hold = max(20, len(allp) // 20) if len(allp) > 60 else 0
    paras = allp[: len(allp) - hold]
    # Windows of the sample in order until the count is reached (compact int arrays: a large sample is millions of pieces).
    from array import array
    import bisect
    wins, have = [], 0
    for p in paras:
        ids = tok(p, add_special_tokens=False).input_ids
        for i in range(0, max(1, len(ids) - 8), a.ctx):
            w = ids[i:i + a.ctx]
            if len(w) >= 16: wins.append(array("i", w)); have += len(w) - 1
        if have >= a.tokens: break
        if len(wins) % 2000 == 0: say("PROGRESS", stage="read", done=have, of=a.tokens, note="splitting the sample into windows")
    starts = []; c = 0   # the first piece number of each window (piece numbers count states, the sink left out)
    for w in wins: starts.append(c); c += len(w) - 1
    N = min(c, a.tokens); d = model.config.hidden_size

    # The states, streamed: windows of the same length go through the layers 16 at a time; each state is (number, vector).
    # A sample that fits in one buffer is read once and kept; a larger one is read again on every pass, so memory stays
    # about one buffer (BUF states) whatever the sample's size.
    BUF = int(os.environ.get("PSI_SAE_BUF") or 262144)   # states per buffer (an override for tests)
    def stream(order):
        batch = []
        def flush():
            if not batch: return
            x = torch.tensor([list(wins[k]) for k in batch], device=dev)
            with torch.no_grad(): h = stack(input_ids=x, output_hidden_states=True).hidden_states[a.layer][:, 1:].float().cpu()   # the layer stack only: no output head
            for r, k in enumerate(batch):
                n = len(wins[k]) - 1; g0 = starts[k]
                if g0 >= N: continue
                n = min(n, N - g0); yield_list.append((g0, h[r, :n]))
            batch.clear()
        for k in order:
            if starts[k] >= N: continue
            if batch and len(wins[batch[0]]) != len(wins[k]): flush(); yield from drain()
            batch.append(k)
            if len(batch) == 16: flush(); yield from drain()
        flush(); yield from drain()
    yield_list = []
    def drain():
        while yield_list: yield yield_list.pop(0)
    def buffers(order):
        buf, ids, n = [], [], 0
        for g0, h in stream(order):
            buf.append(h.half()); ids.append(torch.arange(g0, g0 + h.shape[0])); n += h.shape[0]
            if n >= BUF: yield torch.cat(buf), torch.cat(ids); buf, ids, n = [], [], 0
        if buf: yield torch.cat(buf), torch.cat(ids)
    win_ix = [k for k in range(len(wins)) if starts[k] < N]
    t1 = time.time(); KEEP = None
    if N <= BUF:   # small: read once
        KEEP = list(buffers(win_ix)); say("PROGRESS", stage="read", done=N, of=N, note=f"{N} states of {d} numbers in {time.time() - t1:.0f} s")
    def passes(shuffle_seed):
        if KEEP is not None: return KEEP
        g = torch.Generator().manual_seed(shuffle_seed); order = [win_ix[i] for i in torch.randperm(len(win_ix), generator=g).tolist()]
        return buffers(order)

    # ── training ──
    m = d * a.expansion
    S = sae_mod.SAE(d, m, "topk", a.k).to(dev)
    first = next(iter(passes(0)))[0][:20000].float()
    with torch.no_grad(): S.b_dec.data = first.mean(0).to(dev)
    var = float(((first - first.mean(0)) ** 2).sum(1).mean()); del first
    lr = a.lr or 2e-4 / math.sqrt(m / 2 ** 14)
    opt = torch.optim.Adam(S.parameters(), lr=lr, betas=(0.9, 0.999))
    steps = a.epochs * math.ceil(N / a.batch); step = 0; seen = torch.zeros(m, dtype=torch.bool, device=dev)
    for ep in range(a.epochs):
        tot, cnt = 0.0, 0
        for X, _ in passes(a.seed * 100 + ep + 1):
            perm = torch.randperm(X.shape[0])
            for i in range(0, X.shape[0], a.batch):
                x = X[perm[i:i + a.batch]].float().to(dev)
                rec, z = S(x); loss = ((rec - x) ** 2).sum(1).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                with torch.no_grad():   # keep decoder rows unit length: remove the gradient along each row, then renormalise
                    g = S.W_dec.grad; S.W_dec.grad = g - (g * S.W_dec).sum(1, keepdim=True) * S.W_dec
                opt.step()
                with torch.no_grad(): S.W_dec.data /= S.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8); seen |= (z > 0).any(0)
                tot += float(loss) * x.shape[0]; cnt += x.shape[0]; step += 1
                if step % 20 == 0 or step == steps: say("PROGRESS", stage="train", done=step, of=max(steps, step), epoch=ep + 1, fvu=round(tot / cnt / var, 4), note=f"epoch {ep + 1} of {a.epochs}")
        dead = float((~seen).float().mean()); seen.zero_()
        say("PROGRESS", stage="train", done=step, of=steps, epoch=ep + 1, fvu=round(tot / cnt / var, 4), dead=round(dead, 4), note=f"epoch {ep + 1} done")
    fvu = tot / cnt / var

    # ── the features: frequency, largest activation, strongest examples ──
    say("PROGRESS", stage="index", note="finding each feature's strongest examples")
    E = a.examples; best_v = torch.full((m, E), -1.0); best_i = torch.full((m, E), -1, dtype=torch.long)
    freq = torch.zeros(m); S.eval()
    with torch.no_grad():
        for X, G in passes(0):
            for i in range(0, X.shape[0], a.batch):
                z = S.encode(X[i:i + a.batch].float().to(dev)).cpu(); gid = G[i:i + a.batch]
                freq += (z > 0).float().sum(0)
                v, jx = torch.topk(z.T, min(E, z.shape[0]), dim=1)   # per feature, its best in this batch
                cv = torch.cat([best_v, v], 1); ci = torch.cat([best_i, gid[jx]], 1)
                o = torch.topk(cv, E, dim=1); best_v = o.values; best_i = torch.gather(ci, 1, o.indices)
    def context(gi):
        wi = bisect.bisect_right(starts, gi) - 1; pos = gi - starts[wi] + 1; w = list(wins[wi])
        left = tok.decode(w[max(0, pos - 12):pos]); piece = tok.decode(w[pos:pos + 1]); right = tok.decode(w[pos + 1:pos + 5])
        return [left + piece + right, None, len(left), len(left) + len(piece)]
    ex = {}
    for f in range(m):
        rows = []
        for v, gi in zip(best_v[f].tolist(), best_i[f].tolist()):
            if v <= 0 or gi < 0: continue
            c = context(gi); c[1] = round(v, 3); rows.append(c)
        if rows: ex[str(f)] = rows
    # The words each feature's direction promotes (the readout through the final norm and the output head), kept as an
    # index: names and search in the app are then a lookup. The final norm and the head are still loaded (only the
    # layers past the one read were dropped).
    def readout_many(h, l):
        return model.lm_head(stack.norm(h.to(model.lm_head.weight.dtype)) if hasattr(stack, "norm") else stack.final_layer_norm(h.to(model.lm_head.weight.dtype))).float() if hasattr(model, "lm_head") else model.embed_out(stack.final_layer_norm(h.to(model.embed_out.weight.dtype))).float()
    say("PROGRESS", stage="index", note="the words each feature promotes")
    tops = sae_mod.top_token_index(readout_many, S.W_dec.detach().to(dev), a.layer)
    feats = {"top_tokens": tops, "freq": [round(float(x) / N, 6) for x in freq], "max_act": [round(float(x), 4) if x > 0 else None for x in best_v[:, 0]], "exemplars": ex, "labels": {}}
    dest = os.path.join(sae_mod.root_dir(), sae_mod.model_key(a.model, whash), f"L{a.layer}")
    old = os.path.join(dest, "features.json")
    if os.path.exists(old):   # names given to an earlier SAE do not carry over: its features are different ones
        os.replace(old, old + f".before-{time.strftime('%Y%m%d-%H%M%S')}")
    sae_mod.save_dir(dest, S.cpu(), {"model": os.path.basename(os.path.normpath(a.model)), "model_dir": os.path.abspath(a.model), "weights_hash": whash, "layer": a.layer, "hook": "resid_post",
                                    "trained": {"tokens": N, "corpus": os.path.basename(corpus), "epochs": a.epochs, "expansion": a.expansion, "fvu": round(fvu, 4), "dead_frac": round(dead, 4),
                                                "lr": lr, "device": dev, "seconds": round(time.time() - t0), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": a.seed,
                                                "heldout": {"sample_path": os.path.abspath(corpus), "from_paragraph": len(allp) - hold, "paragraphs": hold} if hold else None}}, feats)
    # A ring for this run on the model's chain, signed by The Emergence Machine (src/sae_rings.py).
    import sae_rings, transformers
    ring = sae_rings.add_ring(dest, "sae", os.path.basename(os.path.normpath(a.model)), whash, a.layer, {
        "d_sae": m, "activation": "topk", "k": a.k, "expansion": a.expansion,
        "data": {"sample": os.path.basename(corpus), "sha256": sae_rings.file_sha256(corpus), "pieces": N, "window": a.ctx, "rule": "windows of the sample in order until the count is reached; the first piece of each window left out (the attention sink)"},
        "train": {"epochs": a.epochs, "lr": lr, "batch": a.batch, "seed": a.seed, "device": dev, "seconds": round(time.time() - t0), "objective": "mean squared reconstruction error, TopK activation, decoder rows kept unit length"},
        "quality": {"unexplained_variance": round(fvu, 4), "dead_features_last_pass": round(dead, 4), "features_with_examples": len(ex), "rule": "unexplained variance over the training pieces in the last pass"},
        "conditions": {"versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__}, "precision": "bfloat16 model, float32 SAE" if dev == "cuda" else "float32", "layers_loaded": keep}})
    say("DONE", ring=ring["record"]["gen"], signed=bool(ring["signature"]), dir=dest, layer=a.layer, d_sae=m, k=a.k, tokens=N, fvu=round(fvu, 4), dead=round(dead, 4), alive_features=len(ex), seconds=round(time.time() - t0))


if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if e.code not in (None, 0): print(str(e.code) if str(e.code).startswith("PSI-ERROR") else f"PSI-ERROR {e.code}", flush=True); sys.exit(1)
    except Exception as e:
        print(f"PSI-ERROR {type(e).__name__}: {e}", flush=True); sys.exit(1)
