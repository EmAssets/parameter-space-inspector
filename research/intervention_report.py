"""
Interventions against the storage ladder: a model changed at one level (model_ladder.py --swap ...), re-measured on
all nine levels, beside the unchanged model and a same-size control change. Each level's verdict uses the same rules
and the same untrained twin as the stored run (model_ladder_report.evidence).

  python research/intervention_report.py        # prints the table, writes results/model_ladder/interventions/summary.md + .json
"""
import json, os, glob, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_ladder_report as rep

DIR = os.path.join(rep.OUT, "interventions")


def m7_for(x, R):
    """M7 for a changed model: its sentence layout against the stored models (as make_ladder_data does for replications)."""
    p = os.path.join(DIR, x["tag"].replace("@", "_") + ".npz")
    if not os.path.exists(p): return None
    A = np.load(p)["sent_rdm"]; iu = np.triu_indices(A.shape[0], 1); ra = rep.rank(A[iu]); base = rep.family(x["tag"])
    def rr(o):
        q = os.path.join(rep.OUT, o.replace("@", "_") + ".npz")
        return float(np.corrcoef(ra, rep.rank(np.load(q)["sent_rdm"][iu]))[0, 1]) if os.path.exists(q) else None
    fin = lambda t: ("@" not in t) or t.endswith("@143000") or t.endswith("@final")
    tr = [v for v in (rr(o) for o in R if fin(o) and not rep.untrained(o) and rep.family(o).split("-")[0] != base.split("-")[0]) if v is not None]
    ct = [v for v in (rr(o) for o in R if rep.untrained(o)) if v is not None]
    return {"mean_rho_trained_other_families": round(float(np.mean(tr)), 3) if tr else None, "mean_rho_untrained": round(float(np.mean(ct)), 3) if ct else None}


def unchanged_tag(base, R):
    """The stored run an intervention family pairs with: the bare tag, else its trained final checkpoint
    (pythia-1b -> pythia-1b@143000), else the latest trained step stored."""
    if base in R: return base
    for s in ("@143000", "@final"):
        if base + s in R: return base + s
    tr = [t for t in R if rep.family(t) == base and "@" in t and not rep.untrained(t) and "intervention" not in R[t]]
    return max(tr, key=rep.step) if tr else None


def red_detail(r):
    """The numbers about red that the swap should move, beyond each level's verdict."""
    m5 = r["M5_composition"]["per_context_did"]
    return {"M0 related pct": r["M0_words"]["related_pct"], "M0 foreign pct": r["M0_words"]["foreign_pct"],
            **{f"M1 {k}": v["auc"] for k, v in r["M1_units"]["xl"].items()}, "M2 en acc": r["M2_features"]["en_acc"],
            **{f"M2 {k}": v["acc"] for k, v in r["M2_features"]["xl"].items()},
            **{f"M3 {k}": v["rho"] for k, v in r["M3_geometry"]["xl_sense_geometry"].items()},
            "M4 red z": r["M4_wiring"]["red"]["z"], "M5 frac": r["M5_composition"]["frac_correct"], "M5 mean DiD": r["M5_composition"]["mean_did"],
            **{f"M5 {k}": round(float(np.mean(v)), 2) for k, v in m5.items()},
            "M6 red facts": r["M6_redundancy"]["fact_acc"], "M6 self-repair": r["M6_redundancy"]["self_repair"],
            "M8 rho": r["M8_self_access"]["rho_said_held"], "M8 said AUC": r["M8_self_access"]["said_auc_strong_vs_weak"],
            "M8 held AUC": r["M8_self_access"]["held_auc_strong_vs_weak"]}


def build():
    R = rep.load(); _, M7, _, _ = rep.m7(R, list(R))
    X = {}
    for f in sorted(glob.glob(os.path.join(DIR, "*.json"))):
        if os.path.basename(f).startswith("summary"): continue
        x = json.load(open(f, encoding="utf-8")); X[x["tag"]] = x; R[x["tag"]] = x
        m = m7_for(x, R)
        if m: M7[x["tag"]] = m
    out = {}
    for base in sorted({rep.family(t) for t in X}):
        ut = unchanged_tag(base, R)
        if ut is None: continue
        ctl = rep.control_for(base, R)
        runs = {t: x for t, x in X.items() if rep.family(t) == base}
        cols = {"unchanged": ut, **{t.split("@", 1)[1]: t for t in runs}}
        ev = {k: augment(rep.evidence(t, R, M7, ctl), t, R[t], ctl) for k, t in cols.items()}
        det = {k: red_detail(R[t]) for k, t in cols.items()}
        # every field of the evidence travels with the cell (value, pass, control, need, conds, and the rules-v2 fields
        # when the report has them), so the view can show a changed model's verdict with the same words as a stored run's
        out[base] = {"unchanged_tag": ut, "control_twin": ctl, "columns": list(cols), "interventions": {t.split("@", 1)[1]: x.get("intervention") for t, x in runs.items()},
                     "precision": {k: precision_of(R[t]) for k, t in cols.items()},
                     "levels": {L: {k: dict(ev[k][L]) for k in cols} for L in rep.LEVELS},
                     "detail": {name: {k: det[k][name] for k in cols} for name in det["unchanged"]}}
    return out


def augment(E, t, r, ctl):
    """make_ladder_data's rules-v2 fill-in (p in words, BH, twin kind); the evidence as the report gave it when that import fails."""
    try:
        import make_ladder_data as mld; return mld.augment(E, t, r, ctl)
    except Exception as e:
        print(f"rules-v2 fields skipped: {e!r}", file=sys.stderr); return E


def precision_of(r):
    try:
        import make_ladder_data as mld; return mld.precision_of(r)
    except Exception: return None


def fmt(v): return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def markdown(out):
    lines = ["# Interventions on the storage ladder", "",
             "Each column is the same model re-measured on all nine levels after one change; ✔ = the level passes against the untrained twin.", ""]
    for base, o in out.items():
        cols = o["columns"]
        lines += [f"## {base} (untrained twin: {o['control_twin']})", ""]
        for k, iv in o["interventions"].items():
            if not iv: continue
            what = " ↔ ".join(map(str, iv["words"])) if iv.get("words") else (iv.get("summary") or str(iv.get("targets", ""))[:120])
            lines.append(f"- **{k}** ({iv.get('level', '?')}): {iv.get('kind', '')} {what}{' (control)' if iv.get('control') else ''}")
        lines += ["", "| level | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for L, row in o["levels"].items():
            lines.append(f"| {L} | " + " | ".join(f"{fmt(row[k]['value'])} {'✔' if row[k]['pass'] else '·'}" for k in cols) + " |")
        lines += ["", "| detail | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name, row in o["detail"].items():
            lines.append(f"| {name} | " + " | ".join(fmt(row[k]) for k in cols) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    out = build(); md = markdown(out)
    os.makedirs(DIR, exist_ok=True)
    open(os.path.join(DIR, "summary.md"), "w", encoding="utf-8").write(md)
    json.dump(out, open(os.path.join(DIR, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(md)
