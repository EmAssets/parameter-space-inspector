"""
Interventions at one level of the storage ladder, then the whole ladder (M0-M8) re-measured on the changed model.

Each run loads the model, finds THIS model's own carriers of red at the chosen level (re-derived on the loaded model,
cross-checked against its stored ladder result), installs a persistent change (forward hooks or weight edits), and
then runs model_ladder.run unchanged. The result goes to results/model_ladder/interventions/<tag>.json with an
"intervention" record, where intervention_report.py pairs it with the unchanged model and its untrained twin.

  M0 words     swap two words' input vectors (delegates to model_ladder's --swap)       control: two unrelated words
  M1 units     mean-ablate red's neurons: the 14 per-sense neurons M1 picks + the top-48 red-selective neurons
               (m4_wiring's sel("red")); each is held at its mean over neutral sentences
                                                                                         control: as many random neurons, same layers
  M2 features  project the red-sense subspace (14 centred sense centroids, rank <= 13) out of the residual stream
               at every band(L) residual and the M2 best layer (component set to the centroids' mean)
                                                                                         control (default, --control matched):
               as many consecutive off-sense principal components (orthogonal to the sense subspace) as the rank, chosen
               to hold the same share of the z-scored variance of neutral-text tokens; --control random: a random subspace
  M3 geometry  a reflection inside the sense subspace that swaps the two most distinct senses' centroid directions,
               at the first band residual (a reflection applied again downstream would undo itself)
                                                                                         control (matched): a reflection
               swapping two off-sense PCs with the target normal's variance share; --control random: a random unit normal
  M4 wiring    cut the wires among red's top-48 neurons: for each pair i (earlier) -> j (later), remove from j's
               input weights the component along i's output direction (gain-aware, so the M4 dot is 0)
                                                                                         control: the same layers, random neurons, all their pairs cut

  python research/model_intervene.py --model <models>/pythia-1b/step143000 --tag pythia-1b@m1-units --level M1
  python research/model_intervene.py --model <models>/pythia-1b/step143000 --tag pythia-1b@m1-units-control --level M1 --control
  python research/model_intervene.py --model ... --tag gpt-oss-20b@swap-red-blue --level M0 --words red,blue
  python research/model_intervene.py --model ... --tag pythia-1b@swap-blue-green --level M0 --words blue,green --control
  python research/model_intervene.py --model ... --tag pythia-1b@m1-units --level M1 --readout frozen
  python research/model_intervene.py --model ... --tag pythia-1b@m1-units-control --level M1 --control variance --readout frozen

Controls and readouts (rules v2; old flags and defaults unchanged):
  --level M0 --words blue,green --control   the MATCHED control for red<->blue: two colour words that occur in the test
               texts, so the swap really changes inputs (control_kind 'matched'). The old `--control` alone swaps two
               common words absent from every test text: the identity on the test set (control_kind 'inert'; gpt-oss-20b's
               swap-control equals the unchanged model on all nine levels, as it must).
  --control variance (M1)   random neurons in the same layers whose activation variance over the neutral sentences is
               within 10% of each target's (nearest variance when none is, flagged): 'matched_within_10pct'. With
               --methods 2, a bare --control at M1 means this.
  --readout frozen   M1 and M2 re-measured with the ORIGINAL carriers fixed (the 14 sense neurons and their signs, the
               M2 layer and the English centroids, all from the clean model before the change), before and after the
               change, beside the ladder's own readout (which re-searches carriers on the changed model). Why: pythia-1b's
               targeted M1 ablation left M1 at 0.614 (unchanged 0.609), because the ladder simply found other neurons; and
               pythia-1b's M8 moved more under the random M1 control (0.18 -> 0.06) than under the targeted one (-> 0.20).
               Written as "frozen_readout" in the json, per-item arrays items_frozen_m1 / items_frozen_m2 in the npz.
  --methods 2   the ladder's v2 measurement (model_ladder.METHODS) for the re-measured ladder.
  MoE (gpt-oss) M1 ablation is verified where routed (captured activation / routing weight = the held value), or the
  record says "not verified: MoE" when no target expert was routed at the test tokens.

Interventions are defined for M0-M4 (the --level choices). M5-M8 have no intervention in this script: those levels are
re-measured after a change at M0-M4, but cannot be changed themselves.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_ladder as L

D = L.D
SENSES = L.SENSE_NAMES
K_RED = 48                                                  # as m4_wiring's K
NEUTRAL_FRAMES = ["I read about {w} today.", "The word {w} came up in the meeting.", "We talked about the {w} for a while."]


def interventions_dir(): return os.path.join(os.path.join(HERE, "results", "model_ladder"), "interventions")


# ───────────────────────────── shared: sentences, stored result, residual sites ─────────────────────────────
def english():
    en = [s for k in SENSES for s in D.SENSES[k]]; y = [i for i, k in enumerate(SENSES) for _ in D.SENSES[k]]
    return en, y


def neutral_sentences(M, seed, n_words=16):
    """Common words that occur in none of the ladder's test texts (as control_pair), in plain frames."""
    skip = set(getattr(D, "V2_ONLY", ())) if getattr(L, "METHODS", 1) < 2 else set()      # v1: the test texts as they were
    txt = " ".join(str(v) for k, v in vars(D).items() if k.isupper() and k != "COMMON" and k != "V2_ONLY" and k not in skip).lower()
    free = [w for w in D.COMMON if not re.search(r"\b" + re.escape(w.lower()) + r"\b", txt)]
    pick = L.draw(seed + 91).choice(len(free), min(n_words, len(free)), replace=False)
    return [f.replace("{w}", free[i]) for i in pick for f in NEUTRAL_FRAMES]


def stored_result(path):
    """The model's own stored ladder result (same weights, trained), if one exists."""
    norm = lambda p: os.path.normcase(os.path.normpath(p))
    for f in sorted(glob.glob(os.path.join(HERE, "results", "model_ladder", "*.json"))):
        if os.path.basename(f).startswith("summary"): continue
        try: d = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        if not d.get("init") and d.get("path") and norm(d["path"]) == norm(path) and "M1_units" in d: return d
    return None


def at_residual(M, r, fn):
    """Hook that rewrites residual index r (0 = embeddings, i = output of layer i-1, as forward_capture numbers them)."""
    if r == 0:
        def pre(mod, args, kwargs):
            if args: return (fn(args[0]),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = fn(kwargs["hidden_states"]); return args, kwargs
        return M.layers[0].register_forward_pre_hook(pre, with_kwargs=True)
    def post(mod, args, out):
        return (fn(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else fn(out)
    return M.layers[r - 1].register_forward_hook(post)


def sense_centroids(R, y, r):
    X = R[:, r].float(); yy = torch.tensor(y)
    return torch.stack([X[yy == k].mean(0) for k in range(len(SENSES))])          # [senses, D]


def gen(seed): return torch.Generator().manual_seed(int(seed))


# ───────────────────────────── mixture-of-experts (gpt-oss): a wrapped MoEExperts.forward ─────────────────────────────
def moe_forward(self, hs, router_indices=None, routing_weights=None):
    """MoEExperts.forward (model_ladder.py), plus: _rows {e: (rows, W)} recomputes edited gate/up rows exactly (M4),
    _stat {e: (units, sum, [count])} accumulates pre-routing activations (M1 means), _abl {e: (units, values)} holds
    those units at a value wherever the expert is routed (M1 ablation)."""
    out = torch.zeros_like(hs)
    acts = hs.new_zeros(hs.shape[0], self.E * self.I) if self.capture else None
    rows, stat, abl = getattr(self, "_rows", {}), getattr(self, "_stat", None), getattr(self, "_abl", {})
    for e in torch.unique(router_indices).tolist():
        tok, kpos = torch.where(router_indices == e)
        gu = hs[tok] @ self.w(e, "gate_up_proj").T + self.t["gate_up_proj_bias"][e]
        if e in rows:
            r, W = rows[e]; gu = gu.clone()
            gu[:, r] = (hs[tok].float() @ W.T + self.t["gate_up_proj_bias"][e][r].float()).to(gu.dtype)
        gate, up = gu[..., ::2].clamp(max=self.limit), gu[..., 1::2].clamp(-self.limit, self.limit)
        g = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
        if stat is not None and e in stat:
            u, s, c = stat[e]; s += g[:, u].float().sum(0); c[0] += len(tok)
        if e in abl:
            u, v = abl[e]; g = g.clone(); g[:, u] = v.to(g.dtype)
        wt = routing_weights[tok, kpos, None].to(g.dtype)
        y = g @ self.w(e, "down_proj").T + self.t["down_proj_bias"][e]
        out.index_add_(0, tok, (y * wt).to(hs.dtype))
        if acts is not None: acts[tok, e * self.I:(e + 1) * self.I] = (g * wt).to(acts.dtype)
        ver = getattr(self, "_verify", None)
        if ver is not None and e in abl and len(tok):                   # verify: captured activation / routing weight = held value
            u, v = abl[e]; cap = (g * wt).to(hs.dtype)[:, u].float() if acts is None else acts[tok][:, e * self.I + u].float()
            ver.append(float((cap / wt.float().clamp_min(1e-9) - v.float()).abs().max()))
    self.acts = acts
    return out


def wrap_moe(ex):
    if getattr(ex, "_wrapped", False): return
    import types
    ex.forward = types.MethodType(moe_forward, ex)
    orig = ex.vec; ex._vec_ov = {}
    def vec(kind, n):
        key = (kind, int(n)); return ex._vec_ov[key] if key in ex._vec_ov else orig(kind, n)
    ex.vec = vec; ex._wrapped = True


# ───────────────────────────── M1 units ─────────────────────────────
@torch.no_grad()
def chosen_sense_neurons(A_en, y, moe):
    """m1_units' cross-lingual choice: per sense, the neuron with the largest |AUC - 0.5| on all English sentences."""
    X = A_en.reshape(A_en.shape[0], -1); yt = torch.tensor(y, device=L.DEV); I = A_en.shape[2]; out = []
    if not moe:
        X = X.to(L.DEV).float()
        for k in range(len(SENSES)):
            a = L.auc_cols(X, yt == k); j = int(torch.argmax((a - 0.5).abs())); out.append((j // I, j % I, 1.0 if a[j] >= 0.5 else -1.0))
        return out
    best = torch.full((len(SENSES),), -1.0); bj = torch.zeros(len(SENSES), dtype=torch.long); ba = torch.zeros(len(SENSES))
    for c0 in range(0, X.shape[1], 65536):                   # m1_units_ties: tied zeros share their mean rank
        r = L.avg_rank(X[:, c0:c0 + 65536].to(L.DEV).float())
        for k in range(len(SENSES)):
            a = L.auc_ties(r, yt == k); d = (a - 0.5).abs(); j = int(torch.argmax(d))
            if float(d[j]) > float(best[k]): best[k], bj[k], ba[k] = float(d[j]), c0 + j, float(a[j])
    return [(int(bj[k]) // I, int(bj[k]) % I, 1.0 if float(ba[k]) >= 0.5 else -1.0) for k in range(len(SENSES))]


@torch.no_grad()
def red_selective(M, K=K_RED):
    """m4_wiring's sel('red'): the K neurons whose activation on red most exceeds the other colours (z-score)."""
    cs = [[f.replace("{w}", "[" + w + "]") for f in D.COLOR_FRAMES] for w in D.COLORS]
    _, cA = M.read_marked([s for row in cs for s in row]); nf = len(D.COLOR_FRAMES)
    A = cA.reshape(len(D.COLORS), nf, *cA.shape[1:]).float().mean(1)                 # [colour, L, I]
    names = list(D.COLORS); ci = names.index("red"); oth = [i for i in range(len(names)) if i != ci]
    z = (A[ci] - A[oth].mean(0)) / (A[oth].std(0) + 1e-3)
    top = torch.topk(z.flatten(), K).indices
    return [(int(t // M.I), int(t % M.I)) for t in top]


def layer_matched_random(M, S, seed, exclude=()):
    """Same layers (with multiplicity), random neurons not in the target set."""
    rng = L.draw(seed); taken = set(map(tuple, exclude)) | set(map(tuple, S)); out = []
    for l, _ in S:
        while True:
            n = int(rng.integers(0, M.I))
            if (l, n) not in taken: taken.add((l, n)); out.append((l, n)); break
    return out


def by_layer(T):
    g = {}
    for l, n in T: g.setdefault(int(l), []).append(int(n))
    return {l: sorted(set(v)) for l, v in sorted(g.items())}


@torch.no_grad()
def neutral_means(M, T, seed):
    """Each target neuron's mean activation over every token of the neutral sentences (dense: the down projection's
    input; MoE: the expert unit's activation before the routing weight, over the tokens routed to its expert)."""
    G = by_layer(T); sents = neutral_sentences(M, seed); hs = []; sums = {l: 0.0 for l in G}; cnt = {l: 0 for l in G}
    if M.moe:
        for l, ns in G.items():
            ex = M.experts[l]; wrap_moe(ex); ex._stat = {}
            for n in ns:
                e, u = divmod(n, ex.I); ex._stat.setdefault(e, [[], None, [0]])[0].append(u)
            ex._stat = {e: (torch.tensor(u, device=ex.t["down_proj_bias"].device), torch.zeros(len(u), device=ex.t["down_proj_bias"].device), c) for e, (u, _, c) in ex._stat.items()}
    else:
        for l, ns in G.items():
            cols = torch.tensor(ns, device=L.DEV)
            def f(mod, args, l=l, cols=cols):
                x = args[0][0, :, cols].float(); sums[l] = sums[l] + x.sum(0); cnt[l] += x.shape[0]
            hs.append(M.downs[l].register_forward_pre_hook(f))
    try:
        for s in sents: M.m(input_ids=torch.tensor([M.ids(s)], device=L.DEV))
    finally:
        for h in hs: h.remove()
    vals = {}
    if M.moe:
        for l, ns in G.items():
            ex = M.experts[l]; st = ex._stat; ex._stat = None
            vals[l] = {e: (u, s / max(c[0], 1)) for e, (u, s, c) in st.items()}
            vals[l]["_routed_tokens"] = {e: c[0] for e, (u, s, c) in st.items()}
    else:
        for l, ns in G.items(): vals[l] = (torch.tensor(ns, device=L.DEV), (sums[l] / max(cnt[l], 1)))
    return vals, len(sents)


def install_m1(M, T, vals):
    hs = []
    for l in by_layer(T):
        if M.moe:
            ex = M.experts[l]; wrap_moe(ex); ex._abl = {e: uv for e, uv in vals[l].items() if e != "_routed_tokens"}
            continue
        cols, v = vals[l]
        def f(mod, args, cols=cols, v=v):
            x = args[0].clone(); x[..., cols] = v.to(x.dtype); return (x,) + tuple(args[1:])
        hs.append(M.downs[l].register_forward_pre_hook(f))
    return hs


@torch.no_grad()
def verify_m1(M, T, vals, sents):
    """Max |activation - ablation value| over the targets at the marked tokens (activations as the ladder captures them;
    for MoE the captured value includes the routing weight, so it is compared where routed as value * weight)."""
    if M.moe:
        exs = [M.experts[l] for l in by_layer(T)]
        if not all(getattr(ex, "_abl", None) for ex in exs): return None             # before the ablation: nothing to compare
        for ex in exs: ex._verify = []
        try: M.read_marked(sents)
        finally:
            diffs = [d for ex in exs for d in ex._verify]
            for ex in exs: ex._verify = None
        return round(max(diffs), 5) if diffs else None
    _, A = M.read_marked(sents)                                                   # [n, L, I]
    worst = 0.0; G = by_layer(T)
    for l, ns in G.items():
        if M.moe: continue
        cols, v = vals[l]; worst = max(worst, float((A[:, l, cols.cpu()].float() - v.cpu().half().float()).abs().max()))
    return worst


@torch.no_grad()
def neutral_variance(M, layers, seed):
    """Every neuron's activation variance over every token of the neutral sentences, in the given layers (dense: the
    down projection's input; MoE: the captured expert activation, routing weight included, 0 where not routed)."""
    sents = neutral_sentences(M, seed); s1 = {l: 0.0 for l in layers}; s2 = {l: 0.0 for l in layers}; c = {l: 0 for l in layers}; hs = []
    def acc(l, x):
        x = x.double().cpu(); s1[l] = s1[l] + x.sum(0); s2[l] = s2[l] + (x * x).sum(0); c[l] += x.shape[0]
    if not M.moe:
        for l in layers:
            hs.append(M.downs[l].register_forward_pre_hook(lambda mod, args, l=l: acc(l, args[0][0].float())))
    try:
        for s in sents:
            if M.moe:
                for l in layers: M.experts[l].capture = True
            M.m(input_ids=torch.tensor([M.ids(s)], device=L.DEV))
            if M.moe:
                for l in layers: acc(l, M.experts[l].acts.float())
    finally:
        for h in hs: h.remove()
        for ex in M.experts: ex.capture, ex.acts = False, None
    return {l: (s2[l] / c[l] - (s1[l] / c[l]) ** 2).clamp_min(0).float() for l in layers}, len(sents)


def variance_matched(M, T, seed):
    """M1 control: per target, a random non-target neuron in the same layer whose neutral-text activation variance is
    within 10% of the target's; the nearest variance (log ratio) when none is, counted as outside."""
    G = sorted({int(l) for l, _ in T}); var, n_s = neutral_variance(M, G, seed)
    rng = L.draw(seed + 503); taken = set(map(tuple, T)); out, ratios, outside = [], [], 0
    for l, n in T:
        v = var[l]; tv = float(v[n])
        ok = torch.where(v == 0)[0] if tv == 0 else torch.where(((v / tv) - 1).abs() <= 0.1)[0]
        ok = [j for j in ok.tolist() if (l, j) not in taken]
        if ok: j = ok[int(rng.integers(len(ok)))]
        else:
            outside += 1; d = (torch.log(v.clamp_min(1e-12)) - float(np.log(max(tv, 1e-12)))).abs()
            j = next(int(x) for x in torch.argsort(d).tolist() if (l, int(x)) not in taken)
        taken.add((l, j)); out.append((l, j)); ratios.append(float(v[j]) / tv if tv else (1.0 if float(v[j]) == 0 else float("inf")))
    r = np.array(ratios)
    return out, {"matched_within_10pct": outside == 0, "n_outside_10pct": outside, "n": len(T),
                 "variance_ratio": {"median": round(float(np.median(r)), 4), "min": round(float(r.min()), 4), "max": round(float(r.max()), 4)},
                 "measured_on": f"{n_s} neutral sentences, every token" + ("; MoE: captured activation incl. routing weight" if M.moe else "")}


# ───────────────────────────── frozen readout: M1/M2 with the clean model's carriers fixed ─────────────────────────────
def frozen_sets():
    en, y = english(); out = {"en": (en, y)}
    for lang, T in D.TRANS.items(): out[lang] = ([x for k in SENSES for x in T[k]], [i for i, k in enumerate(SENSES) for _ in T[k]])
    return out


@torch.no_grad()
def frozen_states(M, car):
    """Per sentence set: the fixed neurons' activations [n, 14] and the residual at the fixed M2 layer [n, D] (CPU)."""
    st = {}
    for name, (s, y) in frozen_sets().items():
        R, A = M.read_marked(s)
        st[name] = (torch.stack([A[:, l, n].float() for l, n, _ in car["m1"]], 1), R[:, car["best"]].float(), y); del R, A
    return st


@torch.no_grad()
def frozen_carriers(M, path):
    """The clean model's own carriers, before any change: M1's 14 sense neurons (chosen on all English, with sign),
    M2's layer and the English residuals there (the centroid classifier's training set)."""
    en, y = english(); R, A = M.read_marked(en)
    sense = chosen_sense_neurons(A, y, M.moe); del A
    _, best, src = m2_sites(M, R, y, stored_result(path) if path else None)
    return {"m1": sense, "best": int(best), "best_from": src, "Xen": R[:, best].float().clone(), "y": y}


@torch.no_grad()
def frozen_score(st, car):
    """M1: tie-aware AUC of each fixed neuron for its sense (1000-bootstrap CI per language, as m1_units_ties).
    M2: the clean English centroids at the clean layer, applied to centred sentences (1000 label permutations, as m2_features)."""
    sg = torch.tensor([g for _, _, g in car["m1"]]); n_s = len(SENSES); langs = [k for k in st if k != "en"]
    def aucs(S, yt): return [float(L.auc_ties(L.avg_rank(S[:, k:k + 1]), yt == k)[0]) for k in range(n_s)]
    S, _, y = st["en"]; m1 = {"en_auc": round(float(np.mean(aucs(S * sg, torch.tensor(y)))), 3), "xl": {}}; it1 = []
    for lang in langs:
        S, _, yl = st[lang]; S = S * sg; yt = torch.tensor(yl); per = aucs(S, yt); it1.append(per); brng = L.draw(1); boot = []
        for _ in range(1000):
            ix = torch.tensor(brng.integers(0, len(yl), len(yl))); rb = L.avg_rank(S[ix]); vals = []
            for k in range(n_s):
                p = yt[ix] == k
                if 0 < int(p.sum()) < len(ix): vals.append(float(L.auc_ties(rb[:, k:k + 1], p)[0]))
            boot.append(np.mean(vals))
        m1["xl"][lang] = {"auc": round(float(np.mean(per)), 3), "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}
    m1["xl_mean"] = round(float(np.mean([v["auc"] for v in m1["xl"].values()])), 3); m1["ties"] = "mean rank"
    yen = torch.tensor(car["y"]); Xc = car["Xen"] - car["Xen"].mean(0, keepdim=True)
    pe = L.centroid_acc(Xc, yen, st["en"][1], yen, center_te=True)
    m2 = {"layer": car["best"], "en_acc": round(float((pe == yen).float().mean()), 3), "xl": {}}; it2 = []; prng = L.draw(2)
    for lang in langs:
        _, X, yl = st[lang]; yt = torch.tensor(yl); pred = L.centroid_acc(Xc, yen, X, yt, center_te=True); ok = (pred == yt)
        acc = float(ok.float().mean()); it2.append(ok.numpy().astype(np.int8))
        null = [float((pred == yt[torch.tensor(prng.permutation(len(yl)))]).float().mean()) for _ in range(1000)]
        m2["xl"][lang] = {"acc": round(acc, 3), "p": float((np.array(null) >= acc).mean())}
    m2["xl_mean"] = round(float(np.mean([v["acc"] for v in m2["xl"].values()])), 3); m2["chance"] = round(1 / n_s, 3)
    return {"M1": m1, "M2": m2}, np.array(it1, dtype=np.float32), np.concatenate(it2), langs


# ───────────────────────────── M2 features / M3 geometry ─────────────────────────────
def zstats(R, r):
    """Per-dimension mean and sd of the sense sentences' residuals at site r: M2 (centroid_acc) and M3 (rdm) read the
    residual z-scored per dimension, so the subspace is defined there; otherwise a few huge outlier dimensions decide it."""
    X = R[:, r].float(); return X.mean(0), X.std(0) + 1e-4


def sense_basis(R, y, r, tol=1e-4):
    """Centred sense centroids in z-space -> orthonormal basis (rank <= 13). Returns mu, sd, Q [D, k]."""
    mu, sd = zstats(R, r); C = (sense_centroids(R, y, r) - mu) / sd
    U, S, Vt = torch.linalg.svd(C - C.mean(0), full_matrices=False)
    k = int((S > S[0] * tol).sum()); return mu, sd, Vt[:k].T.contiguous()


def random_basis(Dm, k, seed):
    Q, _ = torch.linalg.qr(torch.randn(Dm, k, generator=gen(seed))); return Q


def project_out(mu, sd, Q, anchor=None):
    """z = (h - mu) / sd; z <- z - Q Q^T (z - a); back to h. The component in the subspace is set to the anchor's
    (default: the sense sentences' mean; the matched control uses the neutral tokens' mean, since its directions are
    the neutral stream's own high-variance ones and the red-token mean along them would inject red)."""
    mu, sd, Q = mu.to(L.DEV), sd.to(L.DEV), Q.to(L.DEV)
    a = torch.zeros_like(mu) if anchor is None else ((anchor.float() - mu.cpu().float()) / sd.cpu()).to(L.DEV)
    def fn(h):
        hf = h.float(); return (hf - sd * (((((hf - mu) / sd) - a) @ Q) @ Q.T)).to(h.dtype)
    return fn


def reflect(mu, sd, v, anchor=None):
    """z <- a + (I - 2 v v^T)(z - a) in the same z-space: an involution that swaps the two directions v separates.
    a = 0 (the sense sentences' mean) by default; the matched control reflects about the neutral tokens' mean."""
    mu, sd, v = mu.to(L.DEV), sd.to(L.DEV), v.to(L.DEV)
    a = torch.zeros_like(mu) if anchor is None else ((anchor.float() - mu.cpu().float()) / sd.cpu()).to(L.DEV)
    def fn(h):
        hf = h.float(); return (hf - 2 * sd * (((((hf - mu) / sd) - a) @ v)[..., None] * v)).to(h.dtype)
    return fn


def m2_sites(M, R, y, st):
    lo, hi = L.band(M.L)
    if st is not None and "M2_features" in st: best, src = int(st["M2_features"]["best_layer"]), "stored"
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); best, src = int(L.m2_features(R, y, {})["best_layer"]), "re-derived"
    return sorted(set(range(lo, hi + 1)) | {best}), best, src


@torch.no_grad()
def proj_norm(R, sites, bases):
    """Mean |Q^T z| at each site over the sense sentences' marked tokens (z with the clean model's mu, sd)."""
    return {int(r): round(float((((R[:, r].float() - bases[r][0]) / bases[r][1]) @ bases[r][2]).norm(dim=-1).mean()), 4) for r in sites}


def var_share(R, r, mu, sd, Q):
    """Share of the sentences' z-space variance, and of their raw variance, that the change removes at site r."""
    Z = (R[:, r].float() - mu) / sd; P = (Z @ Q) @ Q.T
    return {"z": round(float((P ** 2).sum() / (Z ** 2).sum()), 4), "raw": round(float(((P * sd) ** 2).sum() / ((Z * sd) ** 2).sum()), 4)}


@torch.no_grad()
def token_states(M, sents, sites):
    """Residuals at `sites` for every token but the first of each sentence (the clean model: call before any hook).
    The reference for 'how much of the stream a change removes': the hooks act on every token of every input."""
    H = {r: [] for r in sites}; hs = []
    for r in sites:
        def grab(h, r=r): H[r].append(h[0, 1:].float().cpu()); return h
        hs.append(at_residual(M, r, grab))
    try:
        for s in sents: M.m(input_ids=torch.tensor([M.ids(s)], device=L.DEV))
    finally:
        for h in hs: h.remove()
    return {r: torch.cat(v) for r, v in H.items()}


def zc(X, mu, sd):
    Z = (X.float() - mu) / sd; return Z - Z.mean(0)


def share(X, mu, sd, Q):
    """Share of X's z-scored variance (centred on X's own mean) inside span(Q)."""
    Z = zc(X, mu, sd); return float(((Z @ Q) ** 2).sum() / (Z ** 2).sum())


def off_sense_pcs(X, mu, sd, Q):
    """Principal components of X's z-scored variance after the sense subspace is removed: rows orthogonal to Q,
    with each one's share of X's total variance."""
    Z = zc(X, mu, sd); tot = (Z ** 2).sum(); Zp = Z - (Z @ Q) @ Q.T
    _, S, Vt = torch.linalg.svd(Zp, full_matrices=False); n = int((S > S[0] * 1e-5).sum())
    V = Vt[:n] - (Vt[:n] @ Q) @ Q.T
    return torch.nn.functional.normalize(V, dim=1), (S[:n] ** 2 / tot)


def matched_basis(X, mu, sd, Q):
    """M2 control: k consecutive off-sense PCs whose variance share on X is closest to the sense subspace's."""
    k = Q.shape[1]; tgt = share(X, mu, sd, Q); V, ev = off_sense_pcs(X, mu, sd, Q)
    w = torch.stack([ev[o:o + k].sum() for o in range(len(ev) - k + 1)]); o = int(torch.argmin((w - tgt).abs()))
    B, _ = torch.linalg.qr(V[o:o + k].T)
    return B, {"pcs": [o, o + k - 1], "target": round(tgt, 4), "control": round(share(X, mu, sd, B), 4)}


# ───────────────────────────── M4 wiring ─────────────────────────────
def gains(M):
    go = lambda l: M.gout[l].float() if M.gout[l] is not None else 1.0
    gi = lambda l: M.gin[l].float() if M.gin[l] is not None else 1.0
    return go, gi


@torch.no_grad()
def pair_dots(M, S):
    """The M4 number for each pair i -> j (layer i < layer j): normalised output of i . normalised input of j, mean over inputs."""
    go, gi = gains(M); n_in = len(M.ins[0]); out = []
    for (li, ni) in S:
        o = torch.nn.functional.normalize((M.col(M.downs[li], ni) * go(li)).cpu(), dim=0)
        for (lj, nj) in S:
            if li < lj:
                d = [float(o @ torch.nn.functional.normalize((M.row(M.ins[lj][m][0], nj + M.ins[lj][m][1]) * gi(lj)).cpu(), dim=0)) for m in range(n_in)]
                out.append(sum(d) / n_in)
    return np.array(out)


def set_row(M, mod, r, v):
    """Write one input row back: float Linear in place; LLM.int8 row re-quantised; MoE expert row as an exact override."""
    if isinstance(mod, L.NeuronView):
        ex = mod.ex; wrap_moe(ex); e, u = divmod(int(r), ex.I); gr = 2 * u + (mod.kind == "up")
        ex._vec_ov[(mod.kind, int(r))] = v.float().to(ex.t["down_proj_bias"].device)
        rows = ex.__dict__.setdefault("_rows", {}); old = rows.get(e)
        idx = (old[0].tolist() if old else []); W = (list(old[1]) if old else [])
        if gr in idx: W[idx.index(gr)] = ex._vec_ov[(mod.kind, int(r))]
        else: idx.append(gr); W.append(ex._vec_ov[(mod.kind, int(r))])
        rows[e] = (torch.tensor(idx, device=W[0].device), torch.stack(W)); return
    w = mod.weight; scb = getattr(w, "SCB", None)
    if scb is not None:
        amax = v.abs().max().clamp_min(1e-8); w.data[r] = torch.round(v / amax * 127).to(torch.int8).to(w.device); scb[r] = amax.to(scb.dtype)
    else: w.data[r] = v.to(w.dtype).to(w.device)


@torch.no_grad()
def cut_wiring(M, S):
    """For each later neuron j, remove from each of its input rows the span of the earlier neurons' output directions,
    as the input sees them (o_i * gout_i * gin_j): w_j <- w_j - U U^T w_j. Rank-1 per pair, orthogonalised per j."""
    go, gi = gains(M); pairs = 0
    for (lj, nj) in S:
        earlier = [(li, ni) for li, ni in S if li < lj]
        if not earlier: continue
        V = torch.stack([(M.col(M.downs[li], ni) * go(li)).cpu() * (gi(lj).cpu() if torch.is_tensor(gi(lj)) else 1.0) for li, ni in earlier], 1)
        U, Sv, _ = torch.linalg.svd(V, full_matrices=False); U = U[:, Sv > Sv[0] * 1e-6]
        for m in range(len(M.ins[lj])):
            mod, off = M.ins[lj][m]; w = M.row(mod, nj + off).cpu()
            set_row(M, mod, nj + off, w - U @ (U.T @ w))
        pairs += len(earlier)
    return pairs


# ───────────────────────────── install: carriers -> persistent change -> record ─────────────────────────────
@torch.no_grad()
def install(M, level, control=False, seed=0, senses=None, k=K_RED, path=None):
    """Find this model's carriers for `level`, install the change on M, return the intervention record (with checks)."""
    st = stored_result(path) if path else None
    en, y = english()
    ck = None if not control else ("random" if control == "random" else "matched")      # M2/M3: matched (default) or random
    rec = {"level": None, "kind": None, "control": bool(control), "seed": seed, "carriers_from": "re-derived on the loaded model"}
    if control: rec["control_kind"] = ck if level in ("M2", "M3") else "random, layer-matched"
    if level == "M1":
        R, A = M.read_marked(en)
        sense = chosen_sense_neurons(A, y, M.moe); del A
        red = red_selective(M, k)
        tgt = list(dict.fromkeys([(l, n) for l, n, _ in sense] + red))
        agree = None
        if st: agree = sum(1 for s, (l, n, _) in zip(SENSES, sense) if st["M1_units"]["neurons"].get(s, [None, None])[:2] == [l, n])
        vm = None
        if control == "variance": T, vm = variance_matched(M, tgt, seed)
        else: T = layer_matched_random(M, tgt, seed + 501) if control else tgt
        vals, n_neutral = neutral_means(M, T, seed)
        test = [D.SENSES[s][0] for s in SENSES]
        before = None if M.moe else verify_m1(M, T, vals, test)
        hooks = install_m1(M, T, vals)
        after = None if M.moe else verify_m1(M, T, vals, test)
        vk = "variance-matched, layer-matched"
        rec.update(level="M1 units", kind="mean-ablate neurons" + ((f" ({vk})" if vm else " (random, layer-matched)") if control else ""),
                   targets=[list(t) for t in T], n_targets=len(T), layers={str(l): len(v) for l, v in by_layer(T).items()},
                   sense_neurons={s: [l, n, sg] for s, (l, n, sg) in zip(SENSES, sense)}, red_top_k=k,
                   stored_sense_agreement=f"{agree}/{len(SENSES)}" if agree is not None else None,
                   ablation=f"mean over {n_neutral} neutral sentences (all tokens)" + ("; MoE: unit held at its mean pre-routing value wherever its expert is routed" if M.moe else ""),
                   summary=f"{len(T)} {('variance-matched ' if vm else 'random ') if control else ''}neurons held at their neutral mean",
                   check={"max_abs_act_minus_value_before": before, "after": after})
        if M.moe:
            after = verify_m1(M, T, vals, test); rec["check"]["after"] = after
            rec["check"]["verified"] = "not verified: MoE (no target expert routed at the test tokens)" if after is None else bool(after < 0.05)
            rec["check"]["how"] = "MoE: captured activation / routing weight vs the held value, where the target expert is routed"
        else: rec["check"]["verified"] = bool(after is not None and after < 1e-2)
        if vm: rec.update(control_kind=vk, matched_within_10pct=vm["matched_within_10pct"], variance_match=vm)
        return rec, hooks
    if level in ("M2", "M3"):
        R, _ = M.read_marked(en, neurons=False)
        lo, hi = L.band(M.L)
        if level == "M2":
            sites, best, src = m2_sites(M, R, y, st)
            tb = {r: sense_basis(R, y, r) for r in sites}
            Xn = token_states(M, neutral_sentences(M, seed), sites); match = {}
            if ck == "random": use = {r: (tb[r][0], tb[r][1], random_basis(R.shape[2], tb[r][2].shape[1], seed * 7919 + 1000 + r)) for r in sites}
            elif ck == "matched":
                use = {}
                for r in sites:
                    B, match[int(r)] = matched_basis(Xn[r], *tb[r]); use[r] = (tb[r][0], tb[r][1], B)
            else: use = tb
            pre = proj_norm(R, sites, tb)
            hooks = [at_residual(M, r, project_out(*use[r], anchor=Xn[r].mean(0) if ck == "matched" else None)) for r in sites]
            R2, _ = M.read_marked(en, neurons=False); post = proj_norm(R2, sites, tb)
            energy = {int(r): {"sense_sentences": {"target": var_share(R, r, *tb[r])["z"], "applied": var_share(R, r, *use[r])["z"]},
                               "neutral_tokens": {"target": round(share(Xn[r], *tb[r]), 4), "applied": round(share(Xn[r], *use[r]), 4)}} for r in sites}
            if match:
                rec["matched_pcs"] = {r: m["pcs"] for r, m in match.items()}
                rec["share_matched_within_10pct"] = all(abs(m["control"] - m["target"]) <= 0.1 * m["target"] for m in match.values())
            kinds = {None: "project out sense subspace", "random": "project out random subspace (same rank)",
                     "matched": "project out off-sense PCs (same rank, variance-matched on neutral tokens)"}
            rec.update(level="M2 features", kind=kinds[ck],
                       targets={"sites": sites, "rank": {int(r): int(use[r][2].shape[1]) for r in sites}}, best_layer=best, best_layer_from=src, band=[lo, hi],
                       summary=f"rank {max(int(use[r][2].shape[1]) for r in sites)} {({None: 'sense', 'random': 'random', 'matched': 'variance-matched off-sense'})[ck]} subspace removed at residuals {sites}",
                       check={"sense_projection_before": pre, "sense_projection_after": post, "share_of_variance_removed": energy})
            return rec, hooks
        mu, sd = zstats(R, lo); Cc = (sense_centroids(R, y, lo) - mu) / sd; Cc = Cc - Cc.mean(0)
        if senses: a, b = (SENSES.index(s) for s in senses)
        else:
            cos = torch.nn.functional.normalize(Cc, dim=1) @ torch.nn.functional.normalize(Cc, dim=1).T
            a, b = divmod(int(torch.argmin(cos + 9 * torch.eye(len(SENSES)))), len(SENSES))
        ua, ub = torch.nn.functional.normalize(Cc[a], dim=0), torch.nn.functional.normalize(Cc[b], dim=0)
        v = torch.nn.functional.normalize(ua - ub, dim=0); vt = v
        Xn = token_states(M, neutral_sentences(M, seed), [lo])[lo]; Qs = sense_basis(R, y, lo)[2]; pair = None
        if ck == "random": v = torch.nn.functional.normalize(torch.randn(Cc.shape[1], generator=gen(seed * 7919 + 3000)), dim=0)
        elif ck == "matched":
            # normal = the off-sense PC whose variance share is closest to the target normal's (the top one if none
            # reaches it); it swaps the two directions (p_j + p_i)/sqrt2 and (p_j - p_i)/sqrt2, p_j the next PC: two
            # high-variance, non-sense directions, standing in for the two sense centroids (norms recorded)
            V, ev = off_sense_pcs(Xn, mu, sd, Qs); tv = share(Xn, mu, sd, vt[:, None])
            i = int(torch.argmin((ev - tv).abs())); j = i + 1 if i + 1 < len(ev) else i - 1; v = V[i]
            pair = {"normal_pc": i, "partner_pc": j, "pc_share": round(float(ev[i]), 4), "target_share": round(tv, 4),
                    "matched_within_10pct": bool(abs(float(ev[i]) - tv) <= 0.1 * tv),
                    "pseudo_centroid_norms": [round(float(Cc[a].norm()), 3), round(float(Cc[b].norm()), 3)]}
        hooks = [at_residual(M, lo, reflect(mu, sd, v, anchor=Xn.mean(0) if ck == "matched" else None))]
        rec["share_of_variance_along_reflection_normal"] = {
            "sense_sentences": {"target": round(share(R[:, lo], mu, sd, vt[:, None]), 4), "applied": round(share(R[:, lo], mu, sd, v[:, None]), 4)},
            "neutral_tokens": {"target": round(share(Xn, mu, sd, vt[:, None]), 4), "applied": round(share(Xn, mu, sd, v[:, None]), 4)},
            "normal_in_sense_subspace": round(float(((Qs.T @ v) ** 2).sum()), 4)}
        if pair: rec["matched_pcs"] = pair
        R2, _ = M.read_marked(en, neurons=False)
        def cosines(R_, r):                                  # centred centroids in the clean model's z-space at residual r
            m_, s_ = zstats(R, r); z = lambda RR: (sense_centroids(RR, y, r) - m_) / s_
            Co = z(R); c0 = Co.mean(0); Co = Co - c0; Cn = z(R_) - c0
            f = lambda x, z: round(float(torch.nn.functional.cosine_similarity(x, z, 0)), 3)
            return {"a->a": f(Cn[a], Co[a]), "a->b": f(Cn[a], Co[b]), "b->b": f(Cn[b], Co[b]), "b->a": f(Cn[b], Co[a]),
                    "others_mean": round(float(np.mean([f(Cn[i], Co[i]) for i in range(len(SENSES)) if i not in (a, b)])), 3)}
        rec.update(level="M3 geometry", kind={None: "reflection swapping two sense directions", "random": "random reflection",
                   "matched": "reflection swapping two off-sense PCs (variance-matched on neutral tokens)"}[ck],
                   targets={"senses": [SENSES[a], SENSES[b]], "site": lo, "cos_between": round(float(ua @ ub), 3)}, band=[lo, hi],
                   summary=(f"{SENSES[a]} <-> {SENSES[b]} reflected at residual {lo}" if not control else f"{ck} control reflection at residual {lo}"),
                   check={f"centroid_cos_vs_original@{r}": cosines(R2, r) for r in range(lo, hi + 1)})
        return rec, hooks
    if level == "M4":
        tgt = red_selective(M, k)
        S = layer_matched_random(M, tgt, seed + 601) if control else tgt
        before_t, before_s = pair_dots(M, tgt), pair_dots(M, S)
        pairs = cut_wiring(M, S)
        after_t, after_s = pair_dots(M, tgt), pair_dots(M, S)
        st_layers = st["M4_wiring"]["red"]["layers"] if st else None
        rec.update(level="M4 wiring", kind="cut wiring among red neurons" if not control else "cut wiring among random neurons (layer-matched)",
                   targets=[list(t) for t in S], n_targets=len(S), pairs=pairs, layers=sorted(set(l for l, _ in S)),
                   stored_red_layers_match=(sorted(set(l for l, _ in tgt)) == st_layers) if st_layers else None,
                   summary=f"{pairs} wires cut among {len(S)} {'random' if control else 'red'} neurons",
                   check={"cut_pairs_mean_abs_dot_before": round(float(np.abs(before_s).mean()), 5), "after": round(float(np.abs(after_s).mean()), 6),
                          "cut_pairs_max_abs_dot_after": round(float(np.abs(after_s).max()), 6),
                          "red_pairs_mean_dot_before": round(float(before_t.mean()), 5), "red_pairs_mean_dot_after": round(float(after_t.mean()), 5)})
        return rec, []
    raise ValueError(level)


def main():
    ap = argparse.ArgumentParser(description="One-level intervention, then the full ladder")
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True, help="<base>@<name>, e.g. pythia-1b@m1-units")
    ap.add_argument("--level", required=True, choices=["M0", "M1", "M2", "M3", "M4"])
    ap.add_argument("--control", nargs="?", const="matched", choices=["matched", "random", "variance"],
                    help="the same-size control. M2/M3: 'matched' (default) = off-sense principal components holding the target's "
                         "share of z-scored variance on neutral tokens; 'random' = random subspace / reflection. M1/M4: random neurons, same layers; "
                         "M1 'variance' (and a bare --control with --methods 2): random neurons, same layers, activation variance within 10%%. "
                         "M0: with --words not containing red (e.g. blue,green) the matched control; otherwise two words absent from every test text (inert)")
    ap.add_argument("--readout", choices=["search", "frozen"], default="search",
                    help="'frozen': also re-measure M1/M2 with the clean model's carriers fixed (json frozen_readout, npz items_frozen_*); "
                         "'search' (default): only the ladder's own readout, which re-searches carriers on the changed model")
    ap.add_argument("--methods", type=int, choices=[1, 2], default=1, help="the ladder's measurement version (model_ladder.METHODS)")
    ap.add_argument("--words", default="red,blue", help="M0: the two words to swap")
    ap.add_argument("--senses", help="M3: two sense names to swap (default: the two most distinct)")
    ap.add_argument("--k", type=int, default=K_RED, help="M1/M4: red-selective neurons (as m4_wiring's K)")
    ap.add_argument("--q8", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="results folder (default results/model_ladder/interventions)")
    # arguments stripped: arguments read from a file written on Windows can end in CR, which argparse would reject
    a = ap.parse_args([x.strip() for x in sys.argv[1:] if x.strip()])
    assert "@" in a.tag, "tag must be <base>@<name> so intervention_report.py can pair it with the unchanged model"
    L.SEED = a.seed; L.OUT = a.out or interventions_dir(); os.makedirs(L.OUT, exist_ok=True); L.LOG = os.path.join(L.OUT, "ladder.log")
    L.METHODS = a.methods
    frozen = a.readout == "frozen"
    f = os.path.join(L.OUT, f"{a.tag.replace('@', '_')}.json")
    matched0 = bool(a.control) and "red" not in [w.strip().lower() for w in a.words.split(",")]
    swap = None
    if a.level == "M0":
        swap = a.words if (matched0 or not a.control) else "control"
        if not frozen:
            L.run(a.model, a.tag, False, a.q8, swap); m0_record(f, a, matched0); return
    control = a.control
    if a.level == "M1" and a.methods >= 2 and control == "matched": control = "variance"
    record, held = {}, {}
    Base = L.Model
    class Intervened(Base):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw); held["M"] = self
            if frozen:
                held["car"] = frozen_carriers(self, a.model); held["before"] = frozen_score(frozen_states(self, held["car"]), held["car"])
                L.log("   frozen readout before", {k: v["xl_mean"] for k, v in held["before"][0].items()})
            if a.level == "M0": return                                          # model_ladder.run swaps the words itself
            rec, self._intervention_hooks = install(self, a.level, control, a.seed, a.senses.split(",") if a.senses else None, a.k, a.model)
            rec["readout"] = a.readout
            record.update(rec); L.log("   intervention", {k: v for k, v in rec.items() if k not in ("targets", "sense_neurons", "variance_match")})
    L.Model = Intervened
    try: L.run(a.model, a.tag, False, a.q8, swap)
    finally: L.Model = Base
    res = json.load(open(f, encoding="utf-8")); out = {}
    if frozen:
        car = held["car"]; after = frozen_score(frozen_states(held["M"], car), car)
        (b, ib1, ib2, langs), (af, ia1, ia2, _) = held["before"], after
        res["frozen_readout"] = {"carriers_from": "clean model before the change", "best_layer": car["best"], "best_layer_from": car["best_from"],
                                 "neurons": {s: [int(l), int(n), g] for s, (l, n, g) in zip(SENSES, car["m1"])},
                                 "M1": {"before": b["M1"], "after": af["M1"]}, "M2": {"before": b["M2"], "after": af["M2"]},
                                 "note": "before = the clean model, after = the changed model, both read with the clean model's carriers; "
                                         "the ladder's own M1/M2 re-search carriers on the changed model"}
        L.log("   frozen readout after", {k: v["xl_mean"] for k, v in af.items()})
        npz = os.path.join(L.OUT, f"{a.tag.replace('@', '_')}.npz")
        z = dict(np.load(npz)) if os.path.exists(npz) else {}
        z.update(items_frozen_m1=np.stack([ib1, ia1]), items_frozen_m2=np.stack([ib2, ia2]), items_frozen_langs=np.array(langs))
        np.savez_compressed(npz, **z)
    if a.level == "M0":
        with open(f, "w", encoding="utf-8") as fh: json.dump(res, fh, ensure_ascii=False, indent=1)
        m0_record(f, a, matched0); return
    for key, v in res.items():
        out[key] = v
        if key == "neurons": out["intervention"] = record
    out.setdefault("intervention", record)
    with open(f, "w", encoding="utf-8") as fh: json.dump(out, fh, ensure_ascii=False, indent=1)
    L.log(f"   intervention record written to {f}")


def m0_record(f, a, matched0):
    """M0's record is written by model_ladder.run; here: control_kind ('inert': two words absent from every test text,
    the identity on the test set; 'matched': a swap of two test-text words not involving red) and the readout."""
    if not os.path.exists(f): return
    res = json.load(open(f, encoding="utf-8")); iv = res.get("intervention")
    if not isinstance(iv, dict): return
    if matched0: iv["control"] = True; iv["control_kind"] = "matched"
    elif iv.get("control"): iv.setdefault("control_kind", "inert")
    iv["readout"] = a.readout
    with open(f, "w", encoding="utf-8") as fh: json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
