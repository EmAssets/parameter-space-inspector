"""
Self-exploration experiments on a 4B transformer, through the Emergence Machine's lens.

A. The layer ladder: at which layer does each E-level become readable from the
   residual stream? (Linear probes per layer on 1,483 street sentences labelled
   E0-E14 by the site's own pipeline.)  Hypothesis: the network's depth
   recapitulates the emergence ladder: lower E-levels decodable earlier.
B. Permits vs predicts (the emergence gap): per layer, how often the final
   norm + output head applied to the intermediate residual already gives the
   final answer (logit lens), and the KL to the final distribution.
C. Listening to itself: during an answer, how much attention goes to the
   prompt vs to the model's own earlier output, position by position.

  python research/ladder_probe.py --gguf <models>/Qwen3-4B-Q8_0.gguf --data <eledata.json> --pairs <models>/e-level-pairs.json --out research/results
"""
import argparse, json, os, sys, time, collections
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LEVELS = ['fields', 'particles', 'nuclei', 'chemistry', 'complex molecules', 'cells', 'organisms', 'ecosystems', 'minds',
          'cultures', 'institutions', 'organizations', 'states', 'civilizations', 'planetary']


def ridge_probe(X, y, classes, folds=5, lam=1e-2, seed=0):
    """Cross-validated linear probe (ridge on one-hot targets). Returns accuracy and per-class recall."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    acc, per = [], collections.defaultdict(list)
    Y = np.eye(len(classes))[np.searchsorted(classes, y)]
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        W = np.linalg.solve(Xtr.T @ Xtr + lam * len(tr) * np.eye(X.shape[1]), Xtr.T @ Y[tr])
        pred = classes[np.argmax(Xte @ W, 1)]
        acc.append((pred == y[te]).mean())
        for c in classes:
            m = y[te] == c
            if m.any(): per[int(c)].append((pred[m] == c).mean())
    return float(np.mean(acc)), {c: float(np.mean(v)) for c, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True); ap.add_argument("--data", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=1483)
    ap.add_argument("--cap", type=int, default=120)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if os.path.isdir(a.gguf):      # plain bf16 weights (research/convert_bf16.py): memory-mapped, straight to the GPU
        name = os.path.basename(a.gguf.rstrip("/\\"))
        tok = AutoTokenizer.from_pretrained(a.gguf)
        model = AutoModelForCausalLM.from_pretrained(a.gguf, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda", low_cpu_mem_usage=True).eval()
    else:
        folder, name = os.path.split(a.gguf)
        tok = AutoTokenizer.from_pretrained(folder, gguf_file=name)
        model = AutoModelForCausalLM.from_pretrained(folder, gguf_file=name, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    print(f"loaded in {time.time() - t0:.0f}s: {L} layers x {D}", flush=True)
    results = {"model": name, "layers": L, "width": D}

    # ── A. the layer ladder ────────────────────────────────────────────────
    rows = json.load(open(a.data, encoding="utf-8"))[: a.max_rows]
    counts = collections.Counter(r["e"] for r in rows)
    keep = sorted(c for c, n in counts.items() if n >= 20)          # levels with enough examples
    rows = [r for r in rows if r["e"] in keep]
    # Balance: cap each level so the probe cannot win by guessing the common class.
    cap = a.cap
    by = collections.defaultdict(list)
    for r in rows: by[r["e"]].append(r)
    rng = np.random.default_rng(0)
    rows = [r for c in keep for r in (rng.permutation(by[c])[:cap] if len(by[c]) > cap else by[c])]
    y = np.array([r["e"] for r in rows]); classes = np.array(keep)
    print(f"probe set: {len(rows)} rows over levels {keep} (cap {cap} each)", flush=True)
    texts = [f"{r['label']}: {r['why']}" for r in rows]
    feats_last = np.zeros((L + 1, len(rows), D), dtype=np.float32)
    feats_mean = np.zeros((L + 1, len(rows), D), dtype=np.float32)
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors="pt").input_ids.to("cuda")
            hs = model(ids, output_hidden_states=True).hidden_states          # L+1 x [1, T, D]
            for l, h in enumerate(hs):
                feats_last[l, i] = h[0, -1].float().cpu().numpy()
                feats_mean[l, i] = h[0].float().mean(0).cpu().numpy()
            if i % 100 == 0: print(f"  features {i}/{len(rows)}", end="\r", flush=True)
    ladder = {"levels": [int(c) for c in classes], "n": len(rows), "chance": float(1 / len(classes)), "last": [], "mean": []}
    for kind, F in (("last", feats_last), ("mean", feats_mean)):
        for l in range(L + 1):
            acc, per = ridge_probe(F[l], y, classes)
            ladder[kind].append({"layer": l, "acc": acc, "per_level": per})
        print(f"\n[{kind} token] probe accuracy by layer: " + " ".join(f"{x['acc']:.2f}" for x in ladder[kind]), flush=True)
    # Control: shuffled labels at the best layer (should sit at chance).
    best = int(np.argmax([x["acc"] for x in ladder["mean"]]))
    sh_acc, _ = ridge_probe(feats_mean[best], rng.permutation(y), classes)
    ladder["shuffled_control_at_best_layer"] = {"layer": best, "acc": sh_acc}
    # "Emergence depth" per level: first layer where that level's recall reaches 80% of its peak.
    depth = {}
    for c in classes:
        rec = np.array([x["per_level"].get(int(c), 0) for x in ladder["mean"]])
        peak = rec.max(); depth[int(c)] = {"peak_recall": float(peak), "first_layer_at_80pct": int(np.argmax(rec >= 0.8 * peak)), "layer0_recall": float(rec[0])}
    ladder["emergence_depth"] = depth
    print("emergence depth (first layer reaching 80% of peak recall):", {LEVELS[c]: v["first_layer_at_80pct"] for c, v in depth.items()}, flush=True)
    results["ladder"] = ladder
    del feats_last, feats_mean

    # ── B. permits vs predicts (logit lens) ───────────────────────────────
    pairs = json.load(open(a.pairs, encoding="utf-8"))[:16]
    norm, head = model.model.norm, model.lm_head
    agree = np.zeros(L + 1); kl = np.zeros(L + 1); n_pos = 0
    with torch.no_grad():
        for p in pairs:
            text = tok.apply_chat_template([{"role": "user", "content": p["question"] + " /no_think"}], tokenize=False, add_generation_prompt=True) + "<think>\n\n</think>\n\n" + p["plain"]
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :256].to("cuda")
            hs = model(ids, output_hidden_states=True).hidden_states
            final = torch.log_softmax(head(norm(hs[-1][0])).float(), -1)        # [T, V]
            top = final.argmax(-1)
            for l in range(L + 1):
                lg = torch.log_softmax(head(norm(hs[l][0])).float(), -1)
                agree[l] += (lg.argmax(-1) == top).float().sum().item()
                kl[l] += (final.exp() * (final - lg)).sum(-1).sum().item()
            n_pos += ids.shape[1]
    results["logit_lens"] = {"positions": n_pos, "top1_agreement_by_layer": (agree / n_pos).tolist(), "kl_to_final_by_layer": (kl / n_pos).tolist()}
    print("logit lens agreement by layer:", " ".join(f"{x:.2f}" for x in agree / n_pos), flush=True)

    # ── C. listening to itself ────────────────────────────────────────────
    self_curve = collections.defaultdict(list)
    with torch.no_grad():
        for p in pairs[:12]:
            prompt = tok.apply_chat_template([{"role": "user", "content": p["question"] + " /no_think"}], tokenize=False, add_generation_prompt=True) + "<think>\n\n</think>\n\n"
            p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            a_ids = tok(p["lens"], return_tensors="pt", add_special_tokens=False).input_ids[:, :160]
            ids = torch.cat([p_ids, a_ids], 1).to("cuda")
            att = model(ids, output_attentions=True).attentions                 # L x [1, H, T, T]
            P = p_ids.shape[1]
            A = torch.stack([x[0].float().mean(0) for x in att]).mean(0)         # [T, T] mean over layers, heads
            for j in range(P, ids.shape[1]):
                row = A[j, :j]                                                 # excluding itself
                on_prompt = row[:P].sum().item(); on_self = row[P:j].sum().item()
                tot = on_prompt + on_self + 1e-9
                self_curve[j - P].append(on_self / tot)
    curve = [float(np.mean(v)) for k, v in sorted(self_curve.items())]
    cross = next((i for i, v in enumerate(curve) if v > 0.5), None)
    results["self_listening"] = {"share_of_attention_on_own_output_by_step": curve, "first_step_over_half": cross}
    print("attention on own output by step (every 10th):", " ".join(f"{v:.2f}" for v in curve[::10]), "| crosses 50% at step", cross, flush=True)

    json.dump(results, open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}/results.json")


if __name__ == "__main__":
    sys.exit(main())
