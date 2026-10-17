"""
J. Finding the phase shift: when does a model stop being an autocomplete on the last
few tokens and start using context as structure?

Per training checkpoint of Pythia-410m-deduped:
  induction     : a random token sequence repeated twice; loss on the second copy vs the
                  first. Only a model that looks back in context and copies can do this.
  ctx_sensitivity: mean KL between the next-token distribution given the full context and
                  given only the last 2 tokens (held-out text). An autocomplete barely moves.
  absorption    : surprise at the 2nd mention of an invented word / surprise at the 1st.
  abstraction   : E-level linear probe on the residual stream (site's labelled street
                  sentences): best-layer accuracy minus layer-0 (word-embedding) accuracy.
  + the behavioural battery (grammar, consequence, tracking) from understanding.py.

  python research/phase_shift.py --root <models>/pythia-410m --data research/eledata.json --out research/results/phase_shift.json
"""
import argparse, json, math, os, sys, time, collections, importlib.util
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
import numpy as np
DEV = "cuda"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("und", os.path.join(HERE, "understanding.py")); und = importlib.util.module_from_spec(spec); spec.loader.exec_module(und)
spec2 = importlib.util.spec_from_file_location("lad", os.path.join(HERE, "ladder_probe.py")); lad = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(lad)

HELD = [und.HELDOUT,
        "The bakery on the corner opens before dawn. By seven the first loaves are gone and the owner is already shaping the second batch. "
        "Regulars know to come early; newcomers learn by missing out once.",
        "A traffic light does something no driver does: it broadcasts one rule to everyone at once. The rule is simple, the effect is a street "
        "that moves without negotiation, and the cost is paid by whoever arrives a second too late."]
INVENTED = ["In the old market there is a stall everyone calls the zorblax. Early each morning the zorblax opens before the others.",
            "Beside the bank stands a tree the locals call the quenvatt. Children climb the quenvatt every afternoon.",
            "The driver called his rickshaw the brimsole. Every evening he washed the brimsole by the river."]


def induction_score(model, tok, n=50, trials=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = model.get_input_embeddings().weight.shape[0]
    first, second = [], []
    for _ in range(trials):
        seq = torch.randint(100, min(V, 30000), (n,), generator=g)
        ids = torch.cat([seq, seq])[None].to(DEV)
        with torch.no_grad():
            lg = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        nll = -lg[torch.arange(len(tgt)), tgt]
        first.append(float(nll[: n - 1].mean())); second.append(float(nll[n:].mean()))
    return {"loss_first": float(np.mean(first)), "loss_second": float(np.mean(second)), "score": float(np.mean(first) - np.mean(second))}


def ctx_sensitivity(model, tok):
    kls = []
    for text in HELD:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            full = torch.log_softmax(model(ids).logits[0].float(), -1)
            for pos in range(4, ids.shape[1] - 1, 3):
                short = torch.log_softmax(model(ids[:, pos - 1: pos + 1]).logits[0, -1].float(), -1)
                p = full[pos].exp()
                kls.append(float((p * (full[pos] - short)).sum()))
    return float(np.mean(kls))


def absorption(model, tok):
    ratios = []
    for text in INVENTED:
        word = next(w for w in ("zorblax", "quenvatt", "brimsole") if w in text)
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        wid = tok(" " + word, add_special_tokens=False).input_ids
        with torch.no_grad():
            lg = torch.log_softmax(model(ids).logits[0].float(), -1)
        seq = ids[0].tolist(); hits = []
        for i in range(1, len(seq) - len(wid) + 1):
            if seq[i:i + len(wid)] == wid:
                hits.append(sum(-float(lg[i + k - 1, wid[k]]) for k in range(len(wid))) / math.log(2))
        if len(hits) >= 2: ratios.append(hits[1] / max(hits[0], 1e-6))
    return float(np.mean(ratios)) if ratios else None


def abstraction(model, tok, rows, cap=40):
    by = collections.defaultdict(list)
    for r in rows: by[r["e"]].append(r)
    keep = sorted(c for c, v in by.items() if len(v) >= 20)
    rng = np.random.default_rng(0)
    sel = [r for c in keep for r in (rng.permutation(by[c])[:cap] if len(by[c]) > cap else by[c])]
    y = np.array([r["e"] for r in sel]); classes = np.array(keep)
    L = model.config.num_hidden_layers; D = model.config.hidden_size
    feats = np.zeros((L + 1, len(sel), D), dtype=np.float32)
    with torch.no_grad():
        for i, r in enumerate(sel):
            ids = tok(f"{r['label']}: {r['why']}", return_tensors="pt").input_ids.to(DEV)
            hs = model(ids, output_hidden_states=True).hidden_states
            for l, h in enumerate(hs): feats[l, i] = h[0].float().mean(0).cpu().numpy()
    accs = [lad.ridge_probe(feats[l], y, classes, folds=4)[0] for l in range(L + 1)]
    return {"layer0": accs[0], "best": max(accs), "best_layer": int(np.argmax(accs)), "gain": max(accs) - accs[0], "chance": 1 / len(classes), "n": len(sel)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--min-bytes", type=float, default=1.5e9)
    a = ap.parse_args()
    rows = json.load(open(a.data, encoding="utf-8"))
    # A checkpoint still downloading has a short model file: only take complete ones (~1.6 GB).
    steps = sorted(int(d[4:]) for d in os.listdir(a.root) if d.startswith("step") and os.path.exists(os.path.join(a.root, d, "model.safetensors"))
                   and os.path.getsize(os.path.join(a.root, d, "model.safetensors")) > a.min_bytes)
    free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
    global DEV; DEV = "cuda" if free > 3.0 else "cpu"; torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"device {DEV} (free GPU {free:.1f} GB); checkpoints: {steps}", flush=True)
    out = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else []
    done = {r["step"] for r in out}
    for s in steps:
        if s in done: continue
        t0 = time.time(); path = os.path.join(a.root, f"step{s}")
        tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
        r = {"step": s, "tokens_seen": s * 2097152}
        r["induction"] = induction_score(model, tok); r["ctx_sensitivity"] = ctx_sensitivity(model, tok); r["absorption"] = absorption(model, tok)
        r["abstraction"] = abstraction(model, tok, rows)
        r["agreement"] = float(np.mean([und.logp_next(model, tok, p, g) > und.logp_next(model, tok, p, b) for p, g, b in und.AGREEMENT]))
        r["consequence"] = float(np.mean([und.best(model, tok, p, g) > und.best(model, tok, p, b) for p, g, b in und.CONSEQUENCE]))
        r["state"] = float(np.mean([und.logp_next(model, tok, p, g) > und.logp_next(model, tok, p, b) for p, g, b in und.STATE]))
        ids = tok(und.HELDOUT, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad(): r["bits_per_token"] = float(model(ids, labels=ids).loss) / math.log(2)
        out.append(r); out.sort(key=lambda x: x["step"])
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"step {s:>6}  loss {r['bits_per_token']:.2f}  induction {r['induction']['score']:+.2f}  ctx-KL {r['ctx_sensitivity']:.2f}  absorb {r['absorption'] if r['absorption'] is None else round(r['absorption'], 2)}  "
              f"abstraction L0 {r['abstraction']['layer0']:.2f} best {r['abstraction']['best']:.2f} (L{r['abstraction']['best_layer']}) gain {r['abstraction']['gain']:+.2f}  grammar {r['agreement']:.2f} conseq {r['consequence']:.2f} track {r['state']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
