"""
Report for model_ladder.py: pass/fail per level against each family's untrained control, M7 (do models trained apart
arrange red's senses alike?), the height each model reaches, and charts across size and training.

  python research/model_ladder_report.py

Rules v2 (evidence(); RULE_VERSION = 2; the old rules stay as evidence_v1): a level is decided only against an untrained
twin of the same architecture (a missing or cross-family twin makes it 'not decidable', never a 50 / 0.5 / chance / 0
stand-in); M1, M5 and M6 compare model and twin with a CI on the difference; the remaining fixed margins (M0 +10,
M2/M3 +0.1, M4 +2) are named in rule_text and flagged 'set after seeing twin results'; M7 is scored against untrained
twins of different labs (matched null, permutation p from the stored layouts); M8 must beat the twin's said AUC by the
null SD of an AUC; M6's twin is not applicable when its fact accuracy is < 0.2. Every p is (1 + k) / (1 + n), with a
per-model Benjamini-Hochberg adjustment (p_bh, pass_bh) and expected_false_passes().
"""
import json, os, glob, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "model_ladder")
LEVELS = ["M0 words", "M1 units", "M2 features", "M3 geometry", "M4 wiring", "M5 composition", "M6 redundancy", "M7 universality", "M8 self-access"]


def rank(x): return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def load():
    R = {}
    for f in glob.glob(os.path.join(OUT, "*.json")):
        if os.path.basename(f).startswith("summary"): continue
        d = json.load(open(f, encoding="utf-8")); R[d["tag"]] = d
    return R


def family(tag): return tag.split("@")[0]
def step(tag):
    s = tag.split("@")[1] if "@" in tag else "final"
    m = re.search(r"\d+", s); return int(m.group()) if m else (10 ** 9 if s == "final" else -1)
def untrained(tag): return tag.endswith("@init") or tag.endswith("@0") or tag.endswith("@shuffle")


def control_for(tag, R):
    f = family(tag); b = f.replace("-instruct", "").replace("-base", "")
    for c in (f + "@0", f + "@init", b + "@0", b + "@init", f + "@shuffle", b + "@shuffle"):
        if c in R: return c
    if f.startswith("pythia"): return "pythia-410m@0" if "pythia-410m@0" in R else None
    if f.startswith("qwen3"): return "qwen3-4b@init" if "qwen3-4b@init" in R else None
    for c in ("qwen3-4b@init", "olmo2-1b@init", "pythia-410m@0"):          # nearest available untrained model of another family
        if c in R: return c
    return None


def m7(R, tags, perms=1000):
    mats = {t: np.load(os.path.join(OUT, t.replace("@", "_") + ".npz"))["sent_rdm"] for t in tags if os.path.exists(os.path.join(OUT, t.replace("@", "_") + ".npz"))}
    tags = list(mats); n = mats[tags[0]].shape[0]; iu = np.triu_indices(n, 1); rk = {t: rank(mats[t][iu]) for t in tags}
    rho = {(a, b): float(np.corrcoef(rk[a], rk[b])[0, 1]) for i, a in enumerate(tags) for b in tags[i + 1:]}
    rng = np.random.default_rng(0); P = [rng.permutation(n) for _ in range(perms)]
    def null_for(a, b):
        A = mats[a]; return np.array([np.corrcoef(rank(A[np.ix_(p, p)][iu]), rk[b])[0, 1] for p in P])
    get = lambda a, b: rho.get((a, b), rho.get((b, a)))
    out = {}
    is_final = lambda t: ("@" not in t) or t.endswith("@143000") or t.endswith("@final")
    trained = [t for t in tags if not untrained(t) and is_final(t)]; ctrls = [t for t in tags if untrained(t)]
    for t in tags:
        others = [o for o in trained if o != t and family(o).split("-")[0] != family(t).split("-")[0]]
        to_tr = [get(t, o) for o in others]; to_ct = [get(t, c) for c in ctrls if c != t]
        out[t] = {"mean_rho_trained_other_families": round(float(np.mean(to_tr)), 3) if to_tr else None,
                  "mean_rho_untrained": round(float(np.mean(to_ct)), 3) if to_ct else None}
    return rho, out, null_for, get


def evidence_v1(t, R, M7, ctl=None):
    """Rules v1 (kept beside v2 for comparison): per level the number that decides it, the untrained control's value,
    the bar to clear, the other conditions, pass. Missing twins fall back to 50 / 0.5 / chance / 0 here; v2 does not."""
    ctl = ctl if ctl is not None else control_for(t, R)
    r = R[t]; c = R.get(ctl) if ctl else None; E = {}
    def put(L, ok, value, control, need, unit, conds):
        E[L] = {"pass": bool(ok), "value": value, "control": control, "need": need, "unit": unit, "conds": conds}
    m0 = r["M0_words"]; c0 = c["M0_words"] if c else {"related_pct": 50, "foreign_pct": 50}
    rel = m0["p"] < 0.01 and m0["related_pct"] > c0["related_pct"] + 10; frn = m0["p_foreign"] < 0.01 and m0["foreign_pct"] > c0["foreign_pct"] + 10
    k = "foreign" if (frn and not rel) else "related"; pk = "p_foreign" if k == "foreign" else "p"
    put("M0 words", rel or frn, m0[k + "_pct"], c0[k + "_pct"], c0[k + "_pct"] + 10, f"{k} words, percentile among 250 common words",
        [[f"p < .01 (p = {m0[pk]})", m0[pk] < 0.01]])
    m1 = r["M1_units"]; c1 = c["M1_units"]["xl_mean"] if c else 0.5; n1 = sum(x["ci"][0] > 0.5 for x in m1["xl"].values())
    put("M1 units", n1 >= 2 and m1["xl_mean"] > c1 + 0.05, m1["xl_mean"], c1, c1 + 0.05, "AUC of English-chosen neurons in zh/hi/es",
        [[f"≥ 2 languages with CI above 0.5 ({n1})", n1 >= 2]])
    m2 = r["M2_features"]; c2 = c["M2_features"]["xl_mean"] if c else m2["chance"]; n2 = sum(x["p"] < 0.01 for x in m2["xl"].values())
    put("M2 features", n2 >= 2 and m2["xl_mean"] > c2 + 0.1, m2["xl_mean"], c2, c2 + 0.1, f"cross-language sense accuracy (chance {m2['chance']})",
        [[f"≥ 2 languages with p < .01 ({n2})", n2 >= 2]])
    m3 = r["M3_geometry"]; n3 = sum(x["p"] < 0.01 for x in m3["xl_sense_geometry"].values())
    mr = lambda d: round(float(np.mean([x["rho"] for x in d["M3_geometry"]["xl_sense_geometry"].values()])), 3)
    c3 = mr(c) if c else None                                                 # shared names (Ferrari, Texas) give untrained models ~0.2
    put("M3 geometry", n3 >= 2 and (c is None or mr(r) > c3 + 0.1), mr(r), c3, (c3 + 0.1) if c else None, "rho, English sense layout vs each language's",
        [[f"≥ 2 languages with p < .01 ({n3})", n3 >= 2]])
    m4 = r["M4_wiring"]["red"]; c4 = c["M4_wiring"]["red"]["z"] if c else 0
    put("M4 wiring", m4.get("p", 1) < 0.01 and m4["z"] > c4 + 2, m4["z"], c4, c4 + 2, "z, red neurons' outputs aimed at later red neurons' inputs",
        [[f"p < .01 (p = {m4.get('p')})", m4.get("p", 1) < 0.01]])
    m5 = r["M5_composition"]; c5 = c["M5_composition"]["mean_did"] if c else 0   # untrained pythia-1b@0: 80% right sign, 0.1 nats
    put("M5 composition", m5["p"] < 0.01 and m5["ci"][0] > 0.5 and m5["mean_did"] > c5 + 0.5, m5["mean_did"], c5, c5 + 0.5, "nats, colour × context interaction (mean DiD)",
        [[f"p < .01 (p = {m5['p']})", m5["p"] < 0.01], [f"right sign in most tests, CI {m5['ci']} above 0.5", m5["ci"][0] > 0.5]])
    m6 = r["M6_redundancy"]; c6 = c["M6_redundancy"]["self_repair_ci"][1] if c else 0
    put("M6 redundancy", m6["fact_acc"] >= 0.5 and m6["self_repair_ci"][0] > c6, m6["self_repair_ci"][0], c6, c6, "self-repair, lower CI bound (control: upper bound)",
        [[f"knows ≥ half the red facts ({m6['fact_acc']})", m6["fact_acc"] >= 0.5]])
    s7 = M7.get(t) or {}; a7, u7 = s7.get("mean_rho_trained_other_families"), s7.get("mean_rho_untrained")
    put("M7 universality", a7 is not None and u7 is not None and a7 > u7 + 0.05, a7, u7, (u7 + 0.05) if u7 is not None else None, "rho with other labs' models (control: with untrained models)", [])
    m8 = r["M8_self_access"]; c8 = c["M8_self_access"]["rho_said_held"] if c else None
    put("M8 self-access", m8["p"] < 0.01 and m8["said_auc_strong_vs_weak"] >= 0.7, m8["rho_said_held"], c8, None, "rho, what it says vs what its state points to",
        [[f"p < .01 (p = {m8['p']})", m8["p"] < 0.01], [f"rates strong associates above weak ones, AUC ≥ 0.7 ({m8['said_auc_strong_vs_weak']})", m8["said_auc_strong_vs_weak"] >= 0.7]])
    return E


RULE_VERSION = 2
FLAG = "margin set after seeing twin results"
_PERMS = {}                                                     # n -> [nperm, m] index arrays for permuting a model's sentences
_RANKS = {}                                                     # npz path -> (mtime, rank vector of the upper triangle)
_M7C = {}                                                       # m7_matched results, keyed by the tags and npz files involved


def _norm_sf(z):
    """One-sided normal tail P(Z >= z)."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2))


def p_plus(p, n):
    """A permutation / resampling p as (1 + k) / (1 + n): never 0. Normal-approximation p (n None) pass through."""
    if p is None: return None
    if not n: return float(p)
    k = int(round(float(p) * n)); return (1 + k) / (1 + n)


def p_words(p, n=None):
    if p is None: return None
    if n:
        k = int(round(p * (1 + n))) - 1
        if k <= 0: return f"p ≤ {1 / (1 + n):.2g} (none of {n} draws reached it)"
        return f"p = {p:.3g} ({k + 1}/{n + 1})"
    return f"p ≈ {p:.2g} (normal approximation)" if p >= 0.001 else "p < 0.001 (normal approximation)"


def _bare(tag): return re.sub(r"~s\d+$", "", tag)


def twin_kind_of(ctl, R):
    if not ctl: return "none"
    k = (R.get(ctl) or {}).get("twin_kind")
    if k in ("real step-0", "drawn from init std", "shuffled"): return k
    b = _bare(ctl)
    if b.endswith("@0"): return "real step-0"
    if b.endswith("@shuffle"): return "shuffled"
    if b.endswith("@init"): return "drawn from init std"
    return "none"


def same_architecture(t, ctl):
    """A twin counts only if it is the run's own architecture: its family, or the base model an -instruct/-base tag names."""
    if not ctl: return False
    f = family(_bare(t)); b = f.replace("-instruct", "").replace("-base", "")
    return family(_bare(ctl)) in (f, b)


def _se(ci): return (ci[1] - ci[0]) / 3.92 if ci and ci[0] is not None and ci[1] is not None else None


def _npz_path(tag):
    m = re.match(r"^(.*)~s(\d+)$", tag)
    if m: cands = [os.path.join(OUT, "replications", f"seed{m.group(2)}", m.group(1).replace("@", "_") + ".npz")]
    else: cands = [os.path.join(OUT, tag.replace("@", "_") + ".npz"), os.path.join(OUT, "interventions", tag.replace("@", "_") + ".npz")]
    return next((c for c in cands if os.path.exists(c)), None)


def _rankvec(tag):
    p = _npz_path(tag)
    if not p: return None
    mt = os.path.getmtime(p)
    if p not in _RANKS or _RANKS[p][0] != mt:
        A = np.load(p)["sent_rdm"]; iu = np.triu_indices(A.shape[0], 1); _RANKS[p] = (mt, rank(A[iu]).astype(np.float32))
    return _RANKS[p][1]


def _perm_index(n, nperm=1000):
    """For each of nperm sentence permutations (seed 0, as m7's null): where each upper-triangle entry moves."""
    if (n, nperm) not in _PERMS:
        iu = np.triu_indices(n, 1); pos = np.zeros((n, n), dtype=np.int32); pos[iu] = np.arange(len(iu[0])); pos = pos + pos.T
        rng = np.random.default_rng(0); P = np.stack([rng.permutation(n) for _ in range(nperm)])
        _PERMS[(n, nperm)] = pos[P[:, iu[0]], P[:, iu[1]]]
    return _PERMS[(n, nperm)]


def _z(v): return (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-12)


def m7_matched(t, R, nperm=500):
    """M7 against a matched null. Value: mean rho of t's sentence layout with other labs' released trained models.
    Control: mean rho between untrained twins of different labs (what architecture + shared input give with no training);
    its SD is the twin spread. p: t's sentences permuted (nperm = 500: the report runs inside the app's refresh), the same mean recomputed."""
    is_fin = lambda x: ("@" not in x) or x.endswith("@143000") or x.endswith("@final")
    lab = lambda x: family(_bare(x)).split("-")[0]
    others = [o for o in R if is_fin(o) and not untrained(o) and lab(o) != lab(t) and o != t and "intervention" not in R[o] and "~s" not in o]
    tw = [u for u in R if untrained(u) and "~s" not in u and _rankvec(u) is not None]
    key = (t, _npz_path(t), tuple(sorted(others)), tuple(sorted(tw)), nperm)
    if key in _M7C: return dict(_M7C[key])
    rv = _rankvec(t); O = [x for x in (_rankvec(o) for o in others) if x is not None]
    tkey = ("tt", tuple(sorted(tw)))
    if tkey not in _M7C:
        _M7C[tkey] = [float(np.corrcoef(_rankvec(a), _rankvec(b))[0, 1]) for i, a in enumerate(tw) for b in tw[i + 1:] if lab(a) != lab(b)]
    tt = _M7C[tkey]
    out = {"n_other": len(O), "n_twin_pairs": len(tt), "twin_twin_mean": round(float(np.mean(tt)), 3) if tt else None,
           "twin_twin_sd": round(float(np.std(tt)), 3) if len(tt) > 1 else None, "value": None, "p": None, "n_perms": nperm}
    if rv is not None and O:
        Oz = _z(np.stack(O)).mean(0); obs = float(_z(rv) @ Oz / len(rv))         # mean of correlations = correlation with the mean z
        zr = _z(rv).astype(np.float32); null = zr[_perm_index(int((1 + np.sqrt(1 + 8 * len(rv))) / 2), nperm)] @ Oz.astype(np.float32) / len(rv)   # a permutation keeps mean and SD
        out |= {"value": round(obs, 3), "p": float((null >= obs).mean()), "null_mean": round(float(null.mean()), 3)}   # raw share; p_plus makes it (1+k)/(1+n)
        if tt:   # m7_pair_p: the value against the twin-twin cross-lab pairs themselves, (1 + k) / (1 + n)
            out |= {"matched_diff": round(obs - float(np.mean(tt)), 3), "m7_pair_p": round((1 + sum(1 for x in tt if x >= obs)) / (1 + len(tt)), 4)}
    _M7C[key] = dict(out); return out


def _cluster_boot_diff(a, b, n=2000, seed=0):
    """Paired bootstrap over contexts of mean(a[k]) - mean(b[k]) (a, b: context -> list of DiD)."""
    ks = [k for k in a if k in b]
    if len(ks) < 3: return None, None
    d = np.array([np.mean(a[k]) - np.mean(b[k]) for k in ks]); rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def _near_p(p): return p is not None and 0.005 <= p <= 0.02


def evidence(t, R, M7, ctl=None):
    """Rules v2, per level: {value, control, need, pass, conds, rule_text, decidable, why_not, twin_tag, twin_kind,
    twin_fallback_family, p, p_n, p_str, p_bh, p_bh_str, pass_bh, near_bar, noise, ci, flags, unit, rule_version}.
    A level is decided only against an untrained twin of the same architecture; a margin chosen after the twin results
    were seen is named in rule_text and flags. Errors in one level make that level not decidable, never the report fail."""
    ctl = ctl if ctl is not None else control_for(t, R)
    r = R[t]; c = R.get(ctl) if ctl else None
    kind = "self" if (ctl and ctl == t) else twin_kind_of(ctl, R); same = same_architecture(t, ctl)
    fb = family(_bare(ctl)) if (ctl and not same) else None
    twin_ok = c is not None and same
    no_twin = ("no untrained twin of this architecture was run, so there is nothing to beat" if not ctl else
               f"the only twin is another architecture's ({ctl}); a cross-family twin is not a baseline")
    E = {}

    def put(L, *, value, control, need, ok, conds, rule, unit, p=None, n=None, noise="none", ci=None, near=False, flags=(), decidable=True, why=None, extra=None):
        dec = bool(decidable); pp = p_plus(p, n) if p is not None else None
        E[L] = {"pass": bool(dec and ok), "value": value, "control": control, "need": need, "unit": unit, "conds": conds,
                "rule_text": rule, "decidable": dec, "why_not": None if dec else why, "twin_tag": ctl, "twin_kind": kind,
                "twin_fallback_family": fb, "p": pp, "p_raw": p, "p_n": n, "p_str": p_words(pp, n), "near_bar": bool(dec and near),
                "noise": noise, "ci": ci, "flags": list(flags), "rule_version": RULE_VERSION, **(extra or {})}

    def guard(L, f):
        try: f()
        except Exception as e:
            E[L] = {"pass": False, "value": None, "control": None, "need": None, "unit": "", "conds": [], "rule_text": None, "decidable": False,
                    "why_not": f"could not be scored from the stored result ({type(e).__name__}: {e})", "twin_tag": ctl, "twin_kind": kind,
                    "twin_fallback_family": fb, "p": None, "p_raw": None, "p_n": None, "p_str": None, "near_bar": False, "noise": "none", "ci": None,
                    "flags": [], "rule_version": RULE_VERSION}

    def m0():
        m0 = r["M0_words"]; n = m0.get("n_resamples", 1000)
        c0 = c["M0_words"] if twin_ok else None
        rel_ok = c0 is not None and m0["p"] < 0.01 and m0["related_pct"] > c0["related_pct"] + 10
        frn_ok = c0 is not None and m0["p_foreign"] < 0.01 and m0["foreign_pct"] > c0["foreign_pct"] + 10
        k = "foreign" if (frn_ok and not rel_ok) else "related"; pk = "p_foreign" if k == "foreign" else "p"
        ctrl = c0[k + "_pct"] if c0 else None; need = ctrl + 10 if c0 else None
        put("M0 words", value=m0[k + "_pct"], control=ctrl, need=need, ok=rel_ok or frn_ok, unit=f"{k} words, percentile among 250 common words",
            conds=[[f"p < .01 (p = {m0[pk]})", m0[pk] < 0.01]], p=m0[pk], n=n, flags=[FLAG],
            rule="related (or foreign) words' mean percentile > twin's + 10 points, and resampling p < .01. The +10 is a " + FLAG + ".",
            near=(need is not None and abs(m0[k + "_pct"] - need) <= 2.5) or _near_p(m0[pk]), decidable=twin_ok, why=no_twin)

    def m1():
        m1 = r["M1_units"]; n1 = sum(x["ci"][0] > 0.5 for x in m1["xl"].values())
        if not twin_ok:
            return put("M1 units", value=m1["xl_mean"], control=None, need=None, ok=False, unit="AUC of English-chosen neurons in zh/hi/es", conds=[[f"≥ 2 languages with CI above 0.5 ({n1})", n1 >= 2]],
                       rule="≥ 2 of zh/hi/es with bootstrap CI above 0.5, and the 95% CI of (AUC − twin AUC) above 0", noise="CI", decidable=False, why=no_twin)
        c1 = c["M1_units"]
        if bool(m1.get("ties")) != bool(c1.get("ties")):
            return put("M1 units", value=m1["xl_mean"], control=c1["xl_mean"], need=None, ok=False, unit="AUC of English-chosen neurons in zh/hi/es", conds=[],
                       rule="model and twin must be scored alike", noise="CI", decidable=False,
                       why="the model and its twin were scored with different tie handling (mean ranks vs ordinal ranks); re-run both with --methods 2")
        langs = [l for l in m1["xl"] if l in c1["xl"]]
        d = [m1["xl"][l]["auc"] - c1["xl"][l]["auc"] for l in langs]; se = [np.hypot(_se(m1["xl"][l]["ci"]), _se(c1["xl"][l]["ci"])) for l in langs]
        D_ = float(np.mean(d)); seD = float(np.sqrt(np.sum(np.square(se))) / len(se)); ci = [round(D_ - 1.96 * seD, 3), round(D_ + 1.96 * seD, 3)]
        need = round(c1["xl_mean"] + 1.96 * seD, 3)
        put("M1 units", value=m1["xl_mean"], control=c1["xl_mean"], need=need, ok=n1 >= 2 and ci[0] > 0, unit="AUC of English-chosen neurons in zh/hi/es",
            conds=[[f"≥ 2 languages with CI above 0.5 ({n1})", n1 >= 2], [f"CI of the difference from the twin {ci} above 0", ci[0] > 0]],
            p=_norm_sf(D_ / max(seD, 1e-9)), n=None, noise="CI", ci=ci, near=abs(m1["xl_mean"] - need) <= 0.5 * 1.96 * seD,
            rule="≥ 2 of zh/hi/es with bootstrap CI above 0.5, and the 95% CI of (AUC − twin AUC) above 0. The CI of the difference is a normal "
                 "approximation from the two stored per-language bootstrap CIs (languages independent).",
            extra={"diff": round(D_, 3), "ties": m1.get("ties") or "ordinal ranks"})

    def m2():
        m2 = r["M2_features"]; ps = sorted(x["p"] for x in m2["xl"].values()); n2 = sum(x["p"] < 0.01 for x in m2["xl"].values())
        p2 = ps[1] if len(ps) > 1 else ps[0]; c2 = c["M2_features"]["xl_mean"] if twin_ok else None; need = c2 + 0.1 if twin_ok else None
        put("M2 features", value=m2["xl_mean"], control=c2, need=need, ok=n2 >= 2 and twin_ok and m2["xl_mean"] > need,
            unit=f"cross-language sense accuracy (chance {m2['chance']})", conds=[[f"≥ 2 languages with p < .01 ({n2})", n2 >= 2]], p=p2, n=1000, flags=[FLAG],
            rule="≥ 2 of zh/hi/es with permutation p < .01 (the 2nd smallest p decides), and accuracy > twin's + 0.1. The +0.1 is a " + FLAG + ".",
            near=(need is not None and abs(m2["xl_mean"] - need) <= 0.025) or _near_p(p2), decidable=twin_ok, why=no_twin)

    def m3():
        g = lambda d: d["M3_geometry"]["xl_sense_geometry"]
        mr = lambda d: round(float(np.mean([x["rho"] for x in g(d).values()])), 3)
        ps = sorted(x["p"] for x in g(r).values()); n3 = sum(x["p"] < 0.01 for x in g(r).values()); p3 = ps[1] if len(ps) > 1 else ps[0]
        c3 = mr(c) if twin_ok else None; need = c3 + 0.1 if twin_ok else None
        put("M3 geometry", value=mr(r), control=c3, need=need, ok=n3 >= 2 and twin_ok and mr(r) > need, unit="rho, English sense layout vs each language's",
            conds=[[f"≥ 2 languages with p < .01 ({n3})", n3 >= 2]], p=p3, n=r["M3_geometry"].get("n_perms", 1000), flags=[FLAG],
            rule="≥ 2 of zh/hi/es with Mantel p < .01 (the 2nd smallest p decides), and mean rho > twin's + 0.1 (shared names give untrained models ~0.2). "
                 "The +0.1 is a " + FLAG + ".", near=(need is not None and abs(mr(r) - need) <= 0.025) or _near_p(p3), decidable=twin_ok, why=no_twin)

    def m4():
        m4 = r["M4_wiring"]["red"]; c4 = c["M4_wiring"]["red"]["z"] if twin_ok else None; need = c4 + 2 if twin_ok else None
        p4 = m4.get("p", 1); conds = [[f"p < .01 (p = {m4.get('p')})", p4 < 0.01]]; ok = twin_ok and p4 < 0.01 and m4["z"] > need
        mn = r["M4_wiring"].get("matched_null"); rule = "z > twin's z + 2 and p < .01 against random neurons in the same layers."
        if mn:
            conds.append([f"beats the other colours' selections (selection-matched p = {mn['p']:.3g} ≤ .05)", mn["p"] <= 0.05]); ok = ok and mn["p"] <= 0.05
            rule += " With --methods 2: also above the other colours' top-K selections (p ≤ .05 of 24)."
        put("M4 wiring", value=m4["z"], control=c4, need=need, ok=ok, unit="z, red neurons' outputs aimed at later red neurons' inputs", conds=conds,
            p=p4, n=r["M4_wiring"].get("n_null", 1000), flags=[FLAG], rule=rule + " The +2 is a " + FLAG + ".",
            near=(need is not None and abs(m4["z"] - need) <= 0.5) or _near_p(p4), decidable=twin_ok, why=no_twin)

    def m5():
        m5 = r["M5_composition"]; conds = [[f"p < .01 (p = {m5['p']})", m5["p"] < 0.01], [f"right sign in most tests, CI {m5['ci']} above 0.5", m5["ci"][0] > 0.5]]
        if not twin_ok:
            return put("M5 composition", value=m5["mean_did"], control=None, need=None, ok=False, unit="nats, colour × context (mean DiD)", conds=conds, p=m5["p"],
                       n=m5.get("n_resamples", 1000), noise="CI", rule="sign test p < .01, right-sign CI above 0.5, and the 95% CI of (DiD − twin DiD) above 0",
                       decidable=False, why=no_twin)
        cm = c["M5_composition"]; D_, ci = _cluster_boot_diff(m5["per_context_did"], cm["per_context_did"])
        if ci is None: raise ValueError("fewer than 3 contexts shared with the twin")
        conds.append([f"CI of the difference from the twin {ci} above 0", ci[0] > 0]); ok = all(x[1] for x in conds)
        rule = ("sign test p < .01, right-sign CI above 0.5, and the 95% CI of (mean DiD − twin mean DiD) above 0: paired bootstrap over the "
                "contexts. Stored runs measure association only (each context alone).")
        it, ict = m5.get("interaction"), cm.get("interaction")
        if it:
            beat = ict is None or it["mean"] > ict["mean"]
            tw_txt = "" if ict is None else f", twin {ict['mean']}"
            conds.append([f"interaction over mirrored pairs {it['mean']} nats, sign-flip p = {it['p']:.3g} ≤ .05{tw_txt}",
                          it["p"] <= 0.05 and beat]); ok = ok and it["p"] <= 0.05 and beat
            rule += f" With --methods 2: the interaction (mirrored pairs, exact sign flips over {it['n_clusters']} clusters, smallest p {it['min_p']}) p ≤ .05 and above the twin's."
        put("M5 composition", value=m5["mean_did"], control=cm["mean_did"], need=round(cm["mean_did"] + (D_ - ci[0]), 3), ok=ok,
            unit="nats, colour × context (mean DiD)", conds=conds, p=m5["p"], n=m5.get("n_resamples", 1000), noise="CI", ci=ci, rule=rule,
            near=(ci[0] <= 0 <= ci[1] and D_ > 0) or _near_p(m5["p"]), extra={"diff": round(D_, 3), "design": "crossed (interaction)" if it else "association"})

    def m6():
        m6 = r["M6_redundancy"]; acc_key = "fact_acc_balanced" if "fact_acc_balanced" in m6 else "fact_acc"; acc = m6[acc_key]
        sp = "compensation_split" in m6 and twin_ok and "compensation_split" in c["M6_redundancy"]
        val = (m6["compensation_split"]["value"] if sp else m6.get("compensation", m6["self_repair"]))
        vci = (m6["compensation_split"]["ci"] if sp else m6.get("compensation_ci", m6["self_repair_ci"]))
        conds = [[f"knows ≥ half the red facts ({acc_key} {acc})", acc >= 0.5]]
        rule = ("knows ≥ half the facts, and the 95% CI of (compensation − twin compensation) above 0 (normal approximation from the two "
                "stored bootstrap CIs). Compensation (was 'self-repair'): share of the top layers' direct effect that removing them does not cost."
                + (" Layers chosen on half the facts, measured on the other half." if sp else " Stored runs choose the layers on the same facts they measure."))
        if not twin_ok:
            return put("M6 redundancy", value=val, control=None, need=None, ok=False, unit="compensation", conds=conds, noise="CI", rule=rule, decidable=False, why=no_twin)
        c6 = c["M6_redundancy"]
        tk = "fact_acc_balanced" if "fact_acc_balanced" in c6 else "fact_acc"
        if c6[tk] < 0.2:
            return put("M6 redundancy", value=val, control=None, need=None, ok=False, unit="compensation", conds=conds, noise="CI", rule=rule,
                       decidable=False, why=f"the twin's fact accuracy is {c6[tk]} ({tk}, < 0.2), so its compensation is not a baseline (twin not applicable)",
                       extra={"twin_applicable": False, "value_ci": vci})
        cval = c6["compensation_split"]["value"] if sp else c6.get("compensation", c6["self_repair"])
        cci = c6["compensation_split"]["ci"] if sp else c6.get("compensation_ci", c6["self_repair_ci"])
        D_ = val - cval; seD = float(np.hypot(_se(vci), _se(cci))); ci = [round(D_ - 1.96 * seD, 3), round(D_ + 1.96 * seD, 3)]
        conds.append([f"CI of the difference from the twin {ci} above 0", ci[0] > 0])
        put("M6 redundancy", value=val, control=cval, need=round(cval + 1.96 * seD, 3), ok=acc >= 0.5 and ci[0] > 0, unit="compensation", conds=conds,
            p=_norm_sf(D_ / max(seD, 1e-9)), n=None, noise="CI", ci=ci, rule=rule, near=abs(D_ - 1.96 * seD) <= 0.5 * 1.96 * seD,
            extra={"twin_applicable": True, "diff": round(D_, 3), "value_ci": vci, "self_repair": m6["self_repair"]})

    def m7():
        s7 = M7.get(t) or {}; mm = m7_matched(t, R)
        a7 = s7.get("mean_rho_trained_other_families") if s7.get("mean_rho_trained_other_families") is not None else mm["value"]
        tt, sd = mm["twin_twin_mean"], mm["twin_twin_sd"]; need = round(tt + sd, 3) if tt is not None and sd is not None else None
        why = None if a7 is not None and need is not None and mm["p"] is not None else (
            "no other lab's released model to compare with" if a7 is None else "fewer than two cross-lab twin pairs, so no twin spread" if need is None
            else "no stored sentence layout (npz) for the permutation p")
        put("M7 universality", value=a7, control=tt, need=need, ok=why is None and mm["p"] < 0.01 and a7 > need,
            unit="rho with other labs' models (control: untrained twins of different labs with each other)",
            conds=[[f"permutation p < .01 (p = {mm['p']})", mm["p"] is not None and mm["p"] < 0.01]], p=mm["p"], n=mm["n_perms"] if mm["p"] is not None else None,
            noise="twin spread", rule=f"mean rho with other labs' trained models > mean twin–twin cross-lab rho + its SD (the twin spread, {mm['n_twin_pairs']} pairs), "
                 "and sentence-permutation p < .01.", near=(need is not None and a7 is not None and abs(a7 - need) <= 0.5 * (sd or 0)) or _near_p(mm["p"]),
            decidable=why is None, why=why, extra={"matched_null": mm, "m7_pair_p": mm.get("m7_pair_p"), "matched_diff": mm.get("matched_diff"),
                   "old_control_mean_rho_untrained": s7.get("mean_rho_untrained")})

    def m8():
        m8 = r["M8_self_access"]; cr = m8.get("crossed"); ccr = (c or {}).get("M8_self_access", {}).get("crossed") if twin_ok else None
        n_s, n_w = 30, len(m8.get("said", {})) - 30 if m8.get("said") else 10
        nsd = float(np.sqrt((n_s + n_w + 1) / (12 * n_s * n_w)))
        if cr and ccr:
            val, ctrl, sd, p, pn = cr["auc_strong_vs_weak"], ccr["auc_strong_vs_weak"], max(cr.get("null_sd") or nsd, nsd), cr["p_rho"], cr.get("n_perms", 1000)
            conds = [[f"crossed (red minus blue/green) agrees with the held state, p < .01 (p = {cr['p_rho']})", cr["p_rho"] < 0.01],
                     [f"crossed ratings separate strong from weak, label-permutation p < .01 (p = {cr['p']})", cr["p"] < 0.01]]
            design = "crossed"
        else:
            val = m8["said_auc_strong_vs_weak"]; ctrl = c["M8_self_access"]["said_auc_strong_vs_weak"] if twin_ok else None; sd = nsd
            p, pn = m8["p"], m8.get("n_perms", 1000); conds = [[f"what it says agrees with its held state, p < .01 (p = {m8['p']})", m8["p"] < 0.01]]; design = "said only"
        need = round(ctrl + sd, 3) if ctrl is not None else None
        put("M8 self-access", value=val, control=ctrl, need=need, ok=twin_ok and all(x[1] for x in conds) and val > need,
            unit=f"said AUC, strong vs weak associates ({design})", conds=conds, p=p, n=pn, noise="none",
            rule=f"said AUC > twin's said AUC + the null SD of an AUC ({sd:.3f}: {n_s} vs {n_w} words), and the said–held rho permutation p < .01."
                 + (" Crossed design: red's rating minus blue's and green's." if design == "crossed" else ""),
            near=(need is not None and abs(val - need) <= 0.25 * sd) or _near_p(p), decidable=twin_ok, why=no_twin,
            extra={"noise_note": f"bar includes the null SD of an AUC ({sd:.3f}), no CI stored", "rho_said_held": m8["rho_said_held"], "design": design})

    for L, f in zip(LEVELS, (m0, m1, m2, m3, m4, m5, m6, m7, m8)): guard(L, f)
    # per-model Benjamini-Hochberg over the levels' p (decidable levels only)
    idx = [L for L in LEVELS if E[L]["decidable"] and E[L]["p"] is not None]; m = len(idx); prev = 1.0
    for k, L in reversed(list(enumerate(sorted(idx, key=lambda x: E[x]["p"]), 1))):
        prev = min(prev, E[L]["p"] * m / k, 1.0); E[L]["p_bh"] = float(f"{prev:.3g}")
    for L in LEVELS:
        e = E[L]; e.setdefault("p_bh", None)
        e["p_bh_str"] = None if e["p_bh"] is None else (f"BH-adjusted p = {e['p_bh']:.3g}" if e["p_bh"] >= 0.001 else "BH-adjusted p < 0.001")
        e["pass_bh"] = bool(e["pass"] and (e["p_bh"] is None or e["p_bh"] < 0.01))
    return E


def expected_false_passes(E):
    """How many of a model's passes chance alone would give at their own p: the sum of p over passing levels
    (each passing level's p is its chance of passing with nothing there)."""
    ps = [E[L]["p"] for L in LEVELS if L in E and E[L].get("pass") and E[L].get("p") is not None]
    return {"expected": round(float(sum(ps)), 4), "n_pass": int(sum(1 for L in LEVELS if L in E and E[L].get("pass"))), "n_with_p": len(ps),
            "n_not_decidable": int(sum(1 for L in LEVELS if L in E and not E[L].get("decidable", True)))}


def verdicts(t, R, M7, ctl=None):
    ctl = ctl if ctl is not None else control_for(t, R)
    E = evidence(t, R, M7, ctl); v = {L: E[L]["pass"] for L in LEVELS}
    passed = [i for i, L in enumerate(LEVELS) if v[L]]
    contiguous = -1
    for i in range(len(LEVELS)):
        if v[LEVELS[i]]: contiguous = i
        else: break
    return v, (max(passed) if passed else -1), contiguous, ctl


def headline(t, r):
    return {"M0": f'{r["M0_words"]["related_pct"]:.0f}/{r["M0_words"]["foreign_pct"]:.0f}', "M1": r["M1_units"]["xl_mean"], "M2": r["M2_features"]["xl_mean"],
            "M3": round(float(np.mean([x["rho"] for x in r["M3_geometry"]["xl_sense_geometry"].values()])), 2), "M4z": r["M4_wiring"]["red"]["z"],
            "M5": r["M5_composition"]["frac_correct"], "M6": r["M6_redundancy"]["self_repair"], "facts": r["M6_redundancy"]["fact_acc"],
            "M8": r["M8_self_access"]["rho_said_held"], "said": r["M8_self_access"]["said_auc_strong_vs_weak"], "held": r["M8_self_access"]["held_auc_strong_vs_weak"]}


def main():
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    R = load()
    finals = [t for t in R if ("@" not in t) or t.endswith("@143000") or t.endswith("@final") or untrained(t)]
    rho, M7s, null_for, get = m7(R, list(R))                                   # every checkpoint vs the released models of other labs
    # permutation p for each trained model's best cross-family match
    p7 = {}
    for t in finals:
        if untrained(t): continue
        others = [o for o in finals if not untrained(o) and family(o).split("-")[0] != family(t).split("-")[0] and o in M7s]
        if not others: continue
        best = max(others, key=lambda o: get(t, o)); nl = null_for(t, best); p7[t] = {"best_match": best, "rho": round(get(t, best), 3), "p": float((nl >= get(t, best)).mean())}
    # trajectories: M7 as agreement with the run's own final model and with Qwen3-4B
    for t in R:
        if t in M7s: continue
        M7s[t] = {}
    rows = {}
    for t in sorted(R, key=lambda x: (family(x), step(x))):
        v, h, hc, ctl = verdicts(t, R, M7s); ev = evidence(t, R, M7s, ctl)
        rows[t] = {"verdict": v, "decidable": {L: ev[L]["decidable"] for L in LEVELS}, "why_not": {L: ev[L]["why_not"] for L in LEVELS if not ev[L]["decidable"]},
                   "expected_false_passes": expected_false_passes(ev), "rule_version": RULE_VERSION, "height": h, "height_contiguous": hc, "control": ctl, "headline": headline(t, R[t]), "M7": M7s.get(t), "M7_best": p7.get(t)}
    json.dump({"rows": rows, "m7_pairs": {f"{a} | {b}": round(x, 3) for (a, b), x in rho.items()}}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    lines = [f"Rules v{RULE_VERSION} (evidence() in model_ladder_report.py): ✔ passes, · fails, ? not decidable (no same-architecture twin, twin not applicable, or not stored).", "",
             "| model | control | " + " | ".join(L.split()[0] for L in LEVELS) + " | height | contiguous |", "|" + "---|" * (len(LEVELS) + 4)]
    for t, r in rows.items():
        lines.append(f"| {t} | {r['control'] or '-'} | " + " | ".join("✔" if r["verdict"][L] else ("·" if r["decidable"][L] else "?") for L in LEVELS) + f" | {('M%d' % r['height']) if r['height'] >= 0 else '-'} | {('M%d' % r['height_contiguous']) if r['height_contiguous'] >= 0 else '-'} |")
    lines += ["", "| model | M0 rel/foreign pct | M1 xl AUC | M2 xl acc | M3 xl rho | M4 z | M5 frac | M6 self-repair | facts | M8 rho | said AUC | held AUC |", "|" + "---|" * 12]
    for t, r in rows.items():
        h = r["headline"]; lines.append(f"| {t} | {h['M0']} | {h['M1']} | {h['M2']} | {h['M3']} | {h['M4z']} | {h['M5']} | {h['M6']} | {h['facts']} | {h['M8']} | {h['said']} | {h['held']} |")
    lines += ["", "M7 (sentence-level arrangement of red's 140 uses, Spearman between models):"]
    for t in finals:
        if t in M7s and M7s[t]: lines.append(f"- {t}: other trained families {M7s[t]['mean_rho_trained_other_families']}, untrained {M7s[t]['mean_rho_untrained']}" + (f"; best {p7[t]['best_match']} {p7[t]['rho']} (p {p7[t]['p']})" if t in p7 else ""))
    # families: does each level appear in every lab's models? (released checkpoints only, each against its own untrained control)
    labs = {}
    for t in rows:
        if untrained(t) or not (("@" not in t) or t.endswith("@143000") or t.endswith("@final")): continue
        labs.setdefault(family(t).split("-")[0], []).append(t)
    lines += ["", "Families (a level counts for a lab if any of its released models passes it):", "",
              "| level | " + " | ".join(labs) + " | labs |", "|" + "---|" * (len(labs) + 2)]
    fam_summary = {}
    for L in LEVELS:
        hit = {lab: [t for t in ts if rows[t]["verdict"][L]] for lab, ts in labs.items()}
        fam_summary[L] = hit
        lines.append(f"| {L} | " + " | ".join(("✔ " + ", ".join(x.replace(lab + "-", "").replace(lab, "") or lab for x in v)) if v else "·" for lab, v in hit.items())
                     + f" | {sum(bool(v) for v in hit.values())}/{len(labs)} |")
    json.dump(fam_summary, open(os.path.join(OUT, "summary-families.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    M7G.update(M7s); charts(R, rows)


M7G = {}


def charts(R, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    metrics = [("M0 words: related pct", lambda r: r["M0_words"]["related_pct"] / 100), ("M1 units: cross-lingual AUC", lambda r: r["M1_units"]["xl_mean"]),
               ("M2 features: cross-lingual acc", lambda r: r["M2_features"]["xl_mean"]), ("M3 geometry: cross-lingual rho", lambda r: np.mean([x["rho"] for x in r["M3_geometry"]["xl_sense_geometry"].values()])),
               ("M4 wiring: z (/10)", lambda r: r["M4_wiring"]["red"]["z"] / 10), ("M5 composition: frac correct", lambda r: r["M5_composition"]["frac_correct"]),
               ("M6: facts correct", lambda r: r["M6_redundancy"]["fact_acc"]), ("M7: agreement with other labs (rho)", lambda r: (M7G.get(r["tag"]) or {}).get("mean_rho_trained_other_families") or 0)]
    fams = sorted({family(t) for t in R if sum(family(x) == family(t) for x in R if not x.endswith("@init")) >= 4})
    if fams:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, (name, fn) in zip(axes.flat, metrics):
            for f in fams:
                ts = sorted([t for t in R if family(t) == f and not t.endswith("@init")], key=step)
                xs = [max(step(t), 1) if step(t) < 10 ** 9 else None for t in ts]
                keep = [(x, fn(R[t])) for x, t in zip(xs, ts) if x is not None]
                if keep: ax.plot([k[0] for k in keep], [k[1] for k in keep], marker="o", ms=3, label=f)
            ax.set_xscale("log"); ax.set_title(name, fontsize=10); ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8); fig.suptitle("Red, inside models, across training (step 1 = untrained start)"); fig.tight_layout()
        fig.savefig(os.path.join(OUT, "ladder-training.png"), dpi=110); plt.close(fig)
    fin = [t for t in rows if (("@" not in t) or t.endswith("@143000") or t.endswith("@final"))]
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(fin) + 1.5))
    M = np.array([[1 if rows[t]["verdict"][L] else 0 for L in LEVELS] for t in fin])
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1.3, aspect="auto"); ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS, rotation=30, ha="right")
    ax.set_yticks(range(len(fin))); ax.set_yticklabels(fin)
    for i in range(len(fin)):
        for j in range(len(LEVELS)): ax.text(j, i, "✔" if M[i, j] else "·", ha="center", va="center", fontsize=9)
    ax.set_title("How high the ladder goes: levels each trained model reaches for 'red' (beyond its untrained control)"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder-models.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
