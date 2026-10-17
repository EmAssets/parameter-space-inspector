"""
Level 2 report: the per-layer profiles of model_deep.py, each beside its untrained twin, across training, and
layer by layer between models.

  python research/deep_report.py      # writes results/model_deep/summary.md and summary.json

Between models (layer_match): linear CKA as before, plus debiased CKA (the unbiased HSIC estimator, Song et al. 2012;
as in Nguyen et al. / Murphy et al. 2024) and a sentence-permutation null: one model's sentence order is shuffled
(200 permutations, seed 0) and the same statistic recomputed. The number to read is cka_minus_null (debiased CKA
minus the null mean), raw and standardised. The unbiased estimator is invariant to centring the features, so the
stored centred Gram matrices are enough. Weight alpha: weight_alpha (top-10% Hill, comparable only within the same
shape and precision) and weight_alpha_ks (xmin chosen by KS distance).
"""
import functools
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_ladder_report as rep

DIR = os.environ.get("PSI_DEEP_DIR") or os.path.join(HERE, "results", "model_deep")
r3 = lambda x: None if x is None else round(float(x), 3)


def load():
    R = {}
    for f in sorted(glob.glob(os.path.join(DIR, "*.json"))):
        if os.path.basename(f).startswith("summary"): continue
        r = json.load(open(f, encoding="utf-8")); R[r["tag"]] = r
    return R


def profile(r):
    """One row of headline numbers per model, each the level-2 counterpart of a ladder question."""
    w = [v for row in r["weights"] for k, v in row.items() if "alpha" in v]
    res = r["residual"]; mid = res[1:-1] or res
    lens = r["lens"]; heads = r.get("heads", {}); tot = heads.get("totals", {})
    wk = [v["alpha_ks"] for v in w if v.get("alpha_ks") is not None]
    out = {"layers": r["provenance"]["layers"], "precision": r["provenance"]["precision"],
           "weight_alpha": r3(np.median([v["alpha"] for v in w])) if w else None,                       # heavy tails come with training
           "weight_alpha_ks": r3(np.median(wk)) if wk else None,                                         # xmin by KS distance
           "induction_check": heads.get("induction_check", "attention-only"),
           "induction_heads_copying": tot.get("induction_heads_copying"),
           "stable_rank_min": r3(min(v["stable_rank"] for v in w)) if w else None,
           "massive_ratio_max": r3(max(x["massive_ratio"] for x in mid)),
           "effective_dim_mid": r3(np.median([x["effective_dim_standardised"] for x in mid])),
           "anisotropy_mid": r3(np.median([x["anisotropy"] for x in mid])),
           "sense_acc": r3(np.mean(list(r["senses"]["per_sense_acc"].values()))), "sense_layer": r["senses"]["layer"],
           "most_confused": r["senses"]["most_confused"][0]["senses"] if r["senses"]["most_confused"] else None,
           "facts_first_top1_layer": lens["first_layer_top1_half"], "facts_top1_final": lens["per_layer"][-1]["top1"],
           "facts_forced_from_layer": lens.get("first_layer_forced_choice_half"), "facts_forced_final": lens["per_layer"][-1].get("forced_choice"),
           "induction_heads": tot.get("induction_heads"), "previous_token_heads": tot.get("previous_token_heads"), "sink_heads": tot.get("sink_heads"),
           "first_induction_layer": next((i for i, x in enumerate(heads.get("per_layer", [])) if x["induction_heads"]), None),
           "zh_marked_word_one_token": r["tokens"].get("zh", {}).get("marked_word_one_token"),
           "hi_tokens_per_word": r["tokens"].get("hi", {}).get("tokens_per_marked_word"),
           "es_tokens_shared": r["tokens"].get("es", {}).get("tokens_shared_with_english")}
    if "routing" in r:
        rt = r["routing"]["per_layer"]
        out["routing_sense_layers"] = sum(x["p"] < 0.01 for x in rt); out["routing_nmi_max"] = r3(max(x["sense_nmi"] for x in rt))
    return out


# Per-layer curves for the app (Storage ladder, card 9): name -> (label, how to read one layer).
CURVES = {
    "alpha": ("weight heavy-tail α (median over the layer's matrices; lower = heavier tails)", lambda r, i: np.median([v["alpha"] for v in r["weights"][i - 1].values() if "alpha" in v]) if i else None),
    "alpha_ks": ("weight heavy-tail α, xmin chosen by KS distance (median over the layer's matrices)", lambda r, i: np.median([v["alpha_ks"] for v in r["weights"][i - 1].values() if v.get("alpha_ks") is not None]) if i and any(v.get("alpha_ks") is not None for v in r["weights"][i - 1].values()) else None),
    "massive": ("largest activation ÷ median activation", lambda r, i: r["residual"][i]["massive_ratio"]),
    "eff_dim": ("effective dimension of the sentences (standardised)", lambda r, i: r["residual"][i]["effective_dim_standardised"]),
    "anisotropy": ("anisotropy: mean cosine between sentences", lambda r, i: r["residual"][i]["anisotropy"]),
    "lens_top1": ("facts: share whose answer is the top guess here (logit lens)", lambda r, i: r["lens"]["per_layer"][i]["top1"]),
    "induction": ("induction heads in the layer", lambda r, i: r["heads"]["per_layer"][i - 1]["induction_heads"] if i and "per_layer" in r.get("heads", {}) else None),
    "routing_nmi": ("MoE: information the routed expert carries about the sense (NMI)", lambda r, i: r["routing"]["per_layer"][i - 1]["sense_nmi"] if i and "routing" in r else None),
}


def curves(r):
    """Index 0 is the embedding output; i ≥ 1 is after layer i (weights, heads and routing belong to layer i)."""
    n = r["provenance"]["layers"] + 1; out = {}
    for k, (_, f) in CURVES.items():
        v = []
        for i in range(n):
            try: x = f(r, i)
            except (IndexError, KeyError, ValueError): x = None
            v.append(r3(x) if x is not None else None)
        if any(x is not None for x in v): out[k] = v
    return out


def cka(Ga, Gb):
    """Linear CKA from two centred Gram matrices of the same sentences."""
    return float((Ga * Gb).sum() / np.sqrt((Ga * Ga).sum() * (Gb * Gb).sum()))


@functools.lru_cache(maxsize=64)
def grams(tag, key):
    return np.load(os.path.join(DIR, tag.replace("@", "_") + ".npz"))[key].astype(np.float64)


def _zd(G):
    """Gram stack [L, n, n] with zeroed diagonals, plus each one's row sums and total (the unbiased HSIC's pieces)."""
    K = G.copy(); idx = np.arange(K.shape[1]); K[:, idx, idx] = 0
    return K, K.sum(2), K.sum((1, 2))


def hsic_u(Ka, Ra, Sa, Kb, Rb, Sb):
    """Unbiased HSIC (Song et al. 2012) between every layer of a and every layer of b: [La, Lb]."""
    n = Ka.shape[1]
    tr = Ka.reshape(len(Ka), -1) @ Kb.reshape(len(Kb), -1).T
    return (tr + np.outer(Sa, Sb) / ((n - 1) * (n - 2)) - 2 / (n - 2) * (Ra @ Rb.T)) / (n * (n - 3))


def self_hsic(K, R, S):
    n = K.shape[1]
    return ((K * K).sum((1, 2)) + S ** 2 / ((n - 1) * (n - 2)) - 2 / (n - 2) * (R * R).sum(1)) / (n * (n - 3))


def cka_debiased_matrix(A, B):
    """Debiased CKA between every layer of A and of B (centred Gram stacks of the same sentences)."""
    a, b = _zd(A), _zd(B)
    den = np.sqrt(np.clip(np.outer(self_hsic(*a), self_hsic(*b)), 1e-30, None))
    return hsic_u(*a, *b) / den


def _stats(M, La, Lb):
    return M.max(1).mean(), np.mean([M[i, round(i * (Lb - 1) / (La - 1))] for i in range(La)])


def layer_match(a, b, perms=200, seed=0):
    """For each layer of a (by depth fraction), the best-matching layer of b and its CKA; plus the mean. Also debiased
    CKA, and its sentence-permutation null (b's sentences shuffled): null mean, sd, and debiased minus null mean."""
    out = {}
    for key in ("gram", "gram_z"):
        A, B = grams(a, key), grams(b, key)
        M = np.array([[cka(x, y) for y in B] for x in A]); s = "" if key == "gram" else "_standardised"
        out |= {f"mean_best_cka{s}": r3(M.max(1).mean()), f"best_layer_by_depth{s}": [r3(j / (len(B) - 1)) for j in M.argmax(1)],
                f"diagonal_cka{s}": r3(np.mean([M[i, round(i * (len(B) - 1) / (len(A) - 1))] for i in range(len(A))]))}
        Md = cka_debiased_matrix(A, B); best, diag = _stats(Md, len(A), len(B))
        a_ = _zd(A)
        sb = self_hsic(*_zd(B)); sa = self_hsic(*a_)
        den = np.sqrt(np.clip(np.outer(sa, sb), 1e-30, None))                    # self-HSIC is permutation invariant
        rng = np.random.default_rng(seed); nb, nd = [], []
        for _ in range(perms):
            p = rng.permutation(B.shape[1]); Bp = B[:, p][:, :, p]
            Mp = hsic_u(*a_, *_zd(Bp)) / den; x, y = _stats(Mp, len(A), len(B)); nb.append(x); nd.append(y)
        out |= {f"cka_debiased{s}": r3(best), f"diagonal_cka_debiased{s}": r3(diag),
                f"best_layer_by_depth_debiased{s}": [r3(j / (len(B) - 1)) for j in Md.argmax(1)],
                f"null_mean{s}": r3(np.mean(nb)), f"null_sd{s}": r3(np.std(nb)), f"cka_minus_null{s}": r3(best - np.mean(nb)),
                f"diagonal_null_mean{s}": r3(np.mean(nd)), f"diagonal_cka_minus_null{s}": r3(diag - np.mean(nd))}
    out["null"] = {"kind": "sentence permutation of the second model", "perms": perms, "seed": seed, "statistic": "mean best-match debiased CKA"}
    return out


def build():
    R = load(); P = {t: profile(r) for t, r in R.items()}
    fams = {}
    for t in R: fams.setdefault(rep.family(t), []).append(t)
    out = {"profiles": P, "curves": {t: curves(r) for t, r in R.items()}, "curve_info": {k: v[0] for k, v in CURVES.items()},
           "families": {}, "across_models": {}}
    for fam, tags in sorted(fams.items()):
        twin = next((t for t in tags if R[t]["provenance"]["untrained_twin"]), None)
        steps = sorted((t for t in tags if "@" in t and t.split("@")[1].isdigit()), key=lambda t: int(t.split("@")[1]))
        final = next((t for t in tags if t.endswith(("@143000", "@final")) or "@" not in t), None)
        out["families"][fam] = {"final": final, "twin": twin, "training": steps}
    finals = [f["final"] for f in out["families"].values() if f["final"]]
    for i, a in enumerate(finals):
        for b in finals[i + 1:]:
            out["across_models"][f"{a} vs {b}"] = layer_match(a, b)
        tw = out["families"][rep.family(a)]["twin"]
        if tw: out["across_models"][f"{a} vs {tw}"] = layer_match(a, tw)
    return out


COLS = [("weight_alpha", "weight α"), ("weight_alpha_ks", "α (KS xmin)"), ("stable_rank_min", "min stable rank"), ("massive_ratio_max", "massive ×"),
        ("effective_dim_mid", "eff. dim"), ("anisotropy_mid", "anisotropy"), ("sense_acc", "sense acc"),
        ("facts_forced_from_layer", "facts held from layer"), ("facts_forced_final", "facts right (forced, last layer)"), ("induction_heads", "induction"), ("previous_token_heads", "prev-token"),
        ("sink_heads", "sink"), ("zh_marked_word_one_token", "zh 1-token"), ("hi_tokens_per_word", "hi tok/word"), ("es_tokens_shared", "es tokens shared")]


def fmt(v): return "–" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def markdown(o):
    P = o["profiles"]
    L = ["# Storage ladder, level 2: per-layer detail", "",
         "Each trained model beside its untrained twin. Medians over middle layers unless named. α: heavy-tail exponent of",
         "the weight spectra (lower = heavier tails); massive ×: largest activation over the median one; eff. dim: effective",
         "dimension of the sentences with every dimension standardised; facts lead @: first layer where the answer is the top",
         "guess for half the facts (logit lens); heads: counted on random tokens repeated twice.", "",
         "| model | " + " | ".join(c for _, c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for fam, f in o["families"].items():
        for t in [f["final"], f["twin"]]:
            if t: L.append(f"| {t} | " + " | ".join(fmt(P[t][k]) for k, _ in COLS) + " |")
    for fam, f in o["families"].items():
        if len(f["training"]) < 2: continue
        L += ["", f"## {fam} across training", "", "| step | " + " | ".join(c for _, c in COLS[:11]) + " |", "|---|" + "---|" * 11]
        for t in f["training"]: L.append(f"| {t.split('@')[1]} | " + " | ".join(fmt(P[t][k]) for k, _ in COLS[:11]) + " |")
    if o["across_models"]:
        L += ["", "## Layer by layer between models (linear CKA on the same 140 sentences)", "",
              "Raw CKA is dominated by a few massive dimensions: compare each pair with the model against its own untrained",
              "twin. The standardised columns weigh every dimension alike. 'debiased − null': best-match debiased CKA minus",
              "the mean of the same statistic with one model's sentences shuffled (200 permutations; null sd in brackets).", "",
              "| pair | best-match CKA | same-depth CKA | best-match, standardised | same-depth, standardised | debiased − null | debiased − null, standardised |",
              "|---|---|---|---|---|---|---|"]
        for k, v in o["across_models"].items():
            L.append(f"| {k} | {fmt(v['mean_best_cka'])} | {fmt(v['diagonal_cka'])} | {fmt(v.get('mean_best_cka_standardised'))} | {fmt(v.get('diagonal_cka_standardised'))}"
                     f" | {fmt(v.get('cka_minus_null'))} ({fmt(v.get('null_sd'))}) | {fmt(v.get('cka_minus_null_standardised'))} ({fmt(v.get('null_sd_standardised'))}) |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    o = build(); md = markdown(o)
    open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8").write(md)
    json.dump(o, open(os.path.join(DIR, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(md)
