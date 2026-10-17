"""
Builds src/ui/ladder-data.js (window.E8_LADDER) for the Storage ladder view from research/results/model_ladder:
every model's evidence per level (value, untrained control, bar to clear, pass), the detail behind the charts
(M5 contexts, M8 said vs held, M2/M5/M6 layer curves), M7 agreement between models, and replications
(results/model_ladder/replications/seed<N>, made with model_ladder.py --seed N) beside the stored runs, and
interventions (results/model_ladder/interventions: one level changed, all nine re-measured; intervention_report.build()).

  python research/make_ladder_data.py            # writes src/ui/ladder-data.js
  python research/make_ladder_data.py --json     # prints the same object as JSON (the app's live refresh)
  python research/make_ladder_data.py --json --write   # both (the app, when the results on disk are newer than the view)

Rules v2 fields (model_ladder_report.evidence: rule_text, decidable, why_not, twin_kind, p_str, near_bar, noise, ci,
rule_version, p_bh) are passed through when the report has them. Where it does not (old data), only what can be
derived from the raw results is filled in here: the raw p and its honest string, a per-model BH adjustment of those p,
the twin's kind from the tags, and precision from the run's provenance. rule_text is never invented: the view says
"rule not stated (old data)".
"""
import json, os, glob, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_ladder_report as rep
import model_ladder_data as D

OUT_JS = os.path.join(HERE, "..", "src", "ui", "ladder-data.js")
REPS = os.path.join(rep.OUT, "replications")

LEVEL_INFO = {
    "M0 words": ["the word's own vector (static weights, no forward pass)", "Is ' red' closer to crimson, blood, stop, 红, लाल, rojo than to 250 common words?"],
    "M1 units": ["single neurons", "Does a neuron chosen for a sense of red in English separate that sense in Chinese, Hindi and Spanish?"],
    "M2 features": ["a direction across many neurons", "Can the sense be read from the residual stream, trained on English, tested on other languages?"],
    "M3 geometry": ["the arrangement of meanings", "Are red's 14 senses laid out alike in all four languages?"],
    "M4 wiring": ["the connections between neurons", "Do red neurons' outputs aim at the inputs of later red neurons, more than random neurons' do?"],
    "M5 composition": ["built during the forward pass", "Same context, two colours, two meanings: does the combination decide (red light → stop, Chinese screen red → rising)?"],
    "M6 redundancy": ["backups", "Knock out a layer: does the rest of the network make up for its direct effect on red facts?"],
    "M7 universality": ["shared across models", "Do models trained by different labs arrange red's 140 uses alike?"],
    "M8 self-access": ["the model's view of itself", "Does what it says about red's associations match where its own state for red points?"],
}


# Model folders are stored relative to the models folder (the UI joins them with Settings > Models), so no machine path
# reaches the page; a Hugging Face cache snapshot is stored as its repo id with hub: true.
def path_fields(p):
    if not p: return {"path": p}
    q = p.replace("\\", "/"); m = re.search(r"/hub/models--([^/]+)--([^/]+)", q)
    if m: return {"path": f"{m[1]}/{m[2]}", "hub": True}
    root = (os.environ.get("PSI_MODELS_DIR") or "").replace("\\", "/").rstrip("/") + "/"
    if root != "/" and q.lower().startswith(root.lower()): return {"path": q[len(root):]}
    m = re.match(r"^(?:[A-Za-z]:)?/(?:[^/]+/)*?models/(.+)$", q)
    return {"path": m[1] if m else (os.path.basename(q.rstrip("/")) if re.match(r"^(?:[A-Za-z]:)?/", q) else q)}

def no_machine_paths(x):
    """Every "path" in the level-2 and intervention sections too (their reports keep the run's folder as recorded)."""
    if isinstance(x, dict): return {k: (path_fields(v)["path"] if k == "path" and isinstance(v, str) else no_machine_paths(v)) for k, v in x.items()}
    if isinstance(x, list): return [no_machine_paths(v) for v in x]
    return x

def r3(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (float, np.floating)): return round(float(x), 3)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, dict): return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [r3(v) for v in x]
    return x


def is_final(t): return ("@" not in t) or t.endswith("@143000") or t.endswith("@final")
def lab(t): return rep.family(t).split("-")[0]


def p_str(p, n=1000):
    """A p-value in words: a permutation p of 0 is 'below 1/n', never 'p = 0.0'."""
    if p is None: return None
    if p <= 0: return f"p < {1 / n:g}"
    return f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"


def raw_p(L, r, use_foreign=False):
    """The p that decides a level in the raw result, and the number of draws behind it; None where the rule is a CI."""
    try:
        if L == "M0 words": m = r["M0_words"]; return (m["p_foreign"] if use_foreign else m["p"]), m.get("n_resamples", 1000)
        if L == "M2 features": ps = sorted(x["p"] for x in r["M2_features"]["xl"].values()); return (ps[1] if len(ps) > 1 else ps[0]), r["M2_features"].get("n_splits", 1000)   # 2nd smallest: the rule asks for 2 languages
        if L == "M3 geometry": ps = sorted(x["p"] for x in r["M3_geometry"]["xl_sense_geometry"].values()); return (ps[1] if len(ps) > 1 else ps[0]), r["M3_geometry"].get("n_perms", 1000)
        if L == "M4 wiring": return r["M4_wiring"]["red"].get("p"), r["M4_wiring"].get("n_null", 1000)
        if L == "M5 composition": return r["M5_composition"]["p"], r["M5_composition"].get("n_resamples", 1000)
        if L == "M8 self-access": return r["M8_self_access"]["p"], r["M8_self_access"].get("n_perms", 1000)
    except (KeyError, TypeError, IndexError): pass
    return None, None


def bh(ps):
    """Benjamini-Hochberg adjusted p for a list of p (None kept as None)."""
    idx = [i for i, p in enumerate(ps) if p is not None]; m = len(idx); out = [None] * len(ps)
    if not m: return out
    order = sorted(idx, key=lambda i: ps[i]); prev = 1.0
    for k, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * m / k); out[i] = round(prev, 4)
    return out


def twin_kind(t, ctl):
    """How the twin relates to the run, from the tags alone: its own init, its base family's, another architecture, or none."""
    if not ctl: return "none", None
    if ctl == t: return "self", None                                          # an untrained run is itself the twin
    f = rep.family(t); b = f.replace("-instruct", "").replace("-base", ""); cf = rep.family(ctl)
    if cf == f: return "own", None
    if cf == b: return "base", None
    return "other-architecture", cf


def precision_of(r):
    pv = r.get("provenance") or {}
    return pv.get("precision") or pv.get("dtype") or ("int8" if r.get("q8") else None)


def augment(E, t, r, ctl):
    """Rules-v2 fields the report did not supply, filled from the raw result where that is honest; nothing invented."""
    kind, fam = twin_kind(t, ctl); ps = []
    for L in rep.LEVELS:
        e = E.get(L)
        if e is None: continue
        if "p" not in e: e["p"], e["p_n"] = raw_p(L, r, L == "M0 words" and str(e.get("unit", "")).startswith("foreign"))
        if not e.get("p_str") and e.get("p") is not None: e["p_str"] = p_str(e["p"], e.get("p_n") or 1000)
        e.setdefault("twin_tag", ctl); e.setdefault("twin_kind", kind); e.setdefault("twin_fallback_family", fam)
        if "decidable" not in e:
            e["decidable"] = e.get("value") is not None and (ctl is not None or L == "M8 self-access")
            e["why_not"] = None if e["decidable"] else ("not measured" if e.get("value") is None else "no untrained twin was run, so there is nothing to beat")
        if L == "M6 redundancy" and "ci" not in e: e["ci"] = (r.get("M6_redundancy") or {}).get("self_repair_ci")
        ps.append(e.get("p"))
    adj = bh(ps); k = 0
    for L in rep.LEVELS:
        e = E.get(L)
        if e is None: continue
        if "p_bh" not in e: e["p_bh"] = adj[k]
        if "p_bh_str" not in e and e["p_bh"] is not None: e["p_bh_str"] = p_str(e["p_bh"], e.get("p_n") or 1000)
        if "pass_bh" not in e: e["pass_bh"] = bool(e["pass"] and (e["p_bh"] is None or e["p_bh"] < 0.01))
        k += 1
    return E


def detail(r):
    m5 = r["M5_composition"]; m8 = r["M8_self_access"]; m6 = r["M6_redundancy"]
    return r3({
        "m0_related": r["M0_words"]["related"], "m0_foreign": r["M0_words"]["foreign"],
        "m1_xl": {k: v["auc"] for k, v in r["M1_units"]["xl"].items()},
        "m2_curve": r["M2_features"]["en_curve"], "m2_xl": {k: v["acc"] for k, v in r["M2_features"]["xl"].items()},
        "m3_xl": {k: v["rho"] for k, v in r["M3_geometry"]["xl_sense_geometry"].items()}, "m3_nearest": r["M3_geometry"]["red_inner_nearest"],
        "m3_colour_rho": r["M3_geometry"]["colour_rho"],
        "m4": {c: r["M4_wiring"][c]["z"] for c in ("red", "blue", "green")},
        "m5_contexts": {k: float(np.mean(v)) for k, v in m5["per_context_did"].items()}, "m5_lens": m5["lens_frac"], "m5_frac": m5["frac_correct"],
        "m6_direct": m6["direct"], "m6_total": m6["total"], "m6_self_repair": m6["self_repair"], "m6_ci": m6["self_repair_ci"], "m6_facts": m6["fact_acc"],
        "m8_said": m8["said"], "m8_held": m8["held"], "m8_said_auc": m8["said_auc_strong_vs_weak"], "m8_held_auc": m8["held_auc_strong_vs_weak"],
    })


def row(t, R, M7, ctl=None, base=None):
    v, h, hc, c = rep.verdicts(t, R, M7, ctl)
    r = R[t]; ev = augment(rep.evidence(t, R, M7, c), t, r, c)
    out = {"family": rep.family(base or t), "lab": lab(base or t), "step": rep.step(base or t), "untrained": rep.untrained(base or t), "final": is_final(base or t),
           "control": c, "height": h, "contiguous": hc, "ev": ev, "layers": r.get("layers"), "neurons": r.get("neurons"),
           "seconds": r.get("seconds"), "seed": r.get("seed", 0), **path_fields(r.get("path")),
           "provenance": no_machine_paths(r.get("provenance")), "precision": precision_of(r),
           "rule_version": r.get("rule_version") or next((e.get("rule_version") for e in ev.values() if e.get("rule_version")), None)}
    if hasattr(rep, "evidence_v1"):                                            # the old rules beside the new, when the report keeps them
        try: out["ev_v1"] = rep.evidence_v1(t, R, M7, c)
        except Exception as e: out["ev_v1_error"] = repr(e)
    if hasattr(rep, "expected_false_passes"):
        try: out["expected_false_passes"] = rep.expected_false_passes(ev)
        except Exception as e: print(f"expected_false_passes skipped for {t}: {e!r}", file=sys.stderr)
    return r3(out)


def interventions():
    """intervention_report.build(), or {} when there are none yet. Never fails the data build."""
    try:
        import intervention_report as ir
        if not [f for f in glob.glob(os.path.join(ir.DIR, "*.json")) if not os.path.basename(f).startswith("summary")]: return {}
        return r3(ir.build())
    except Exception as e:
        print(f"interventions skipped: {e!r}", file=sys.stderr); return {}


def deep():
    """deep_report.build() (level 2: per-layer detail), or {} when nothing is measured yet. Never fails the data build."""
    try:
        import deep_report as dr
        if not [f for f in glob.glob(os.path.join(dr.DIR, "*.json")) if not os.path.basename(f).startswith("summary")]: return {}
        o = dr.build()
        # per language, how many of the test sentences' tokens also occur in the English ones (model_deep.py tokens):
        # shown beside M1-M3 in the drill-down, since a random network can match shared tokens
        for t, r in dr.load().items():
            if t in o.get("profiles", {}): o["profiles"][t]["tokens_shared"] = {k: v.get("tokens_shared_with_english") for k, v in (r.get("tokens") or {}).items() if isinstance(v, dict)}
        return r3(o)
    except Exception as e:
        print(f"level 2 skipped: {e!r}", file=sys.stderr); return {}


def result_files():
    """Every result file the view is built from, so the app can tell when the view is older than the results."""
    fs = [f for f in glob.glob(os.path.join(rep.OUT, "*.json")) if not os.path.basename(f).startswith("summary")]
    fs += [f for d in ("replications/seed*", "interventions") for f in glob.glob(os.path.join(rep.OUT, d, "*.json")) if not os.path.basename(f).startswith("summary")]
    fs += [f for f in glob.glob(os.path.join(HERE, "results", "model_deep", "*.json")) if not os.path.basename(f).startswith("summary")]
    return fs


def build():
    R = rep.load()
    rho, M7, _, get = rep.m7(R, list(R))
    rows = {t: row(t, R, M7) for t in sorted(R, key=lambda x: (rep.family(x), rep.step(x)))}
    det = {t: detail(R[t]) for t in R}
    # M7 matrix: released models and untrained controls, grouped by lab
    m7tags = sorted([t for t in R if is_final(t) or rep.untrained(t)], key=lambda t: (rep.untrained(t), lab(t), t))
    m7m = [[1.0 if a == b else (get(a, b) if get(a, b) is not None else None) for b in m7tags] for a in m7tags]
    # replications: each seed<N> folder, against its own re-drawn control when there is one
    reps = {}
    trained_finals = [t for t in R if is_final(t) and not rep.untrained(t)]; ctrls = [t for t in R if rep.untrained(t)]
    for d in sorted(glob.glob(os.path.join(REPS, "seed*"))):
        s = os.path.basename(d)[4:]
        for f in glob.glob(os.path.join(d, "*.json")):
            x = json.load(open(f, encoding="utf-8")); base = x["tag"]; key = f"{base}~s{s}"; R[key] = x
        for f in glob.glob(os.path.join(d, "*.json")):
            x = json.load(open(f, encoding="utf-8")); base = x["tag"]; key = f"{base}~s{s}"
            sc = rep.control_for(base, {**R, base: x})                    # its own re-drawn twin when the replication made one
            ctl = next((c for c in (f"{rep.family(base)}@init~s{s}", f"{sc}~s{s}") if c in R), sc)
            npz = os.path.join(d, base.replace("@", "_") + ".npz")
            if os.path.exists(npz):                                       # M7 for the replication: its layout vs the stored models
                A = np.load(npz)["sent_rdm"]; n = A.shape[0]; iu = np.triu_indices(n, 1); ra = rep.rank(A[iu])
                def rr(o):
                    p = os.path.join(rep.OUT, o.replace("@", "_") + ".npz")
                    return float(np.corrcoef(ra, rep.rank(np.load(p)["sent_rdm"][iu]))[0, 1]) if os.path.exists(p) else None
                to_tr = [v for v in (rr(o) for o in trained_finals if lab(o) != lab(base)) if v is not None]
                to_ct = [v for v in (rr(o) for o in ctrls if o != base) if v is not None]
                M7[key] = {"mean_rho_trained_other_families": round(float(np.mean(to_tr)), 3) if to_tr else None,
                           "mean_rho_untrained": round(float(np.mean(to_ct)), 3) if to_ct else None}
            reps[key] = {**row(key, R, M7, ctl, base), "base": base, "seed": int(s)}
            det[key] = detail(x)
    crossed = {k: {"prompts": v[0], "answers": v[1]} for k, v in D.CROSSED.items()}
    m7 = {"tags": m7tags, "rho": r3(m7m)}
    m7p = getattr(rep, "m7_pair_p", None)                                     # per-pair p for M7, when the report computes it
    if callable(m7p):
        try: m7["p"] = r3([[None if a == b else m7p(a, b) for b in m7tags] for a in m7tags])
        except Exception as e: print(f"m7 pair p skipped: {e!r}", file=sys.stderr)
    files = result_files()
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "generated_epoch": int(time.time()), "n_results": len(files),
            "newest_result_mtime": int(max((os.path.getmtime(f) for f in files), default=0)), "rule_version": getattr(rep, "RULE_VERSION", None),
            "levels": rep.LEVELS, "level_info": LEVEL_INFO, "rows": rows, "detail": det,
            "m7": m7, "replications": reps, "crossed": crossed, "assoc_strong": D.ASSOC[:30], "assoc_weak": D.ASSOC[30:],
            "interventions": no_machine_paths(interventions()), "deep": no_machine_paths(deep())}


def write(obj):
    js = "// Storage ladder (M0-M8) results: research/make_ladder_data.py from research/results/model_ladder.\nwindow.E8_LADDER = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n"
    open(OUT_JS, "w", encoding="utf-8").write(js); return len(js)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    obj = no_machine_paths(build())
    if "--json" in sys.argv:
        if "--write" in sys.argv: obj["written"] = bool(write(obj))
        print(json.dumps(obj, ensure_ascii=False))
    else:
        n = write(obj); print(f"wrote {OUT_JS} ({n // 1024} KB, {len(obj['rows'])} models, {len(obj['replications'])} replications, {len(obj['interventions'])} intervened, {obj['n_results']} result files)")
