"""
X. The emergence timeline: what emerged at which point of training, in every recorded Pythia size.

Reads the measurements already made across the checkpoints (no model is loaded):
  results/battery-<size>.json        the emergence battery (N), sizes 70m 160m 410m 1b 2.8b
  results/phase_shift[-<size>].json  copying, context use, absorption, abstraction, situation tests (J), 70m 160m 410m
  results/induction_heads_all.json   heads that look back, per checkpoint (410m)

For each capability ("event") and size: the onset, the first checkpoint at which the measure passes its threshold AND
passes at every later checkpoint (the rule the Emergence view uses for "learned"). Checkpoints are sparse, so an onset
is an interval: (the checkpoint before, the onset]. Where a continuous measure stands behind a pass/fail one (a margin
in nats behind an accuracy), the onset is checked against the "mirage" objection (Schaeffer et al. 2023): how much of
the continuous measure's whole rise falls inside the onset interval, against the share expected if it rose evenly in
log-tokens. A jump that is only in the pass/fail score is labelled "steady rise".

  python research/emergence_timeline.py      -> results/emergence_timeline.json + src/ui/timeline-data.js
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
TOKENS_PER_STEP = 2_097_152
SIZES = ["70m", "160m", "410m", "1b", "2.8b"]
PARAMS = {"70m": 70426624, "160m": 162322944, "410m": 405334016, "1b": 1011781632, "2.8b": 2775208960}
GROUPS = ["statistics", "context", "grammar", "knowledge", "in-context learning", "format", "reasoning"]


def load(f):
    p = os.path.join(R, f)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def g(rec, path):
    for k in path:
        if rec is None: return None
        rec = rec.get(k) if isinstance(rec, dict) else None
    return rec


# id, label, group, source, path to the pass/fail measure, threshold, direction (>= or <=), path to the continuous
# measure behind it (or None), unit, items, what it means. The same list (ids, paths, thresholds) is used by
# src/place_model.py to place any model on the timeline: change both together.
EVENTS = [
    ("bits", "Predicts text better than word frequency", "statistics", "phase", ["bits_per_token"], 9.0, "<=", ["bits_per_token"], "bits per token", "4 held-out texts",
     "Loss on held-out text falls below 9 bits per token: past a frequency table (a unigram model of English is about 10 to 11)."),
    ("copying", "Copies a repeated sequence (induction)", "context", "phase", ["induction", "score"], 1.0, ">=", ["induction", "score"], "nats saved on the 2nd copy", "50 random tokens x 4",
     "A random sequence shown twice: the second copy costs at least 1 nat less per token than the first. Only a model that looks back and copies can do it."),
    ("heads", "Heads that look back appear", "context", "heads", None, 1, ">=", None, "heads above 0.3", "every head",
     "At least one attention head's induction score is above 0.3 (410m only: the heads were scanned at every checkpoint)."),
    ("context", "Uses the context, not just the last words", "context", "phase", ["ctx_sensitivity"], 1.0, ">=", ["ctx_sensitivity"], "nats (KL full vs last 2 tokens)", "3 held-out texts",
     "The next-token guess with the full text differs from the guess with only the last two tokens by at least 1 nat."),
    ("absorb", "A new word is cheap the second time", "context", "phase", ["absorption"], 0.5, "<=", ["absorption"], "2nd / 1st mention cost", "3 invented words",
     "An invented word (zorblax) costs at most half as many bits at its second mention: the model picked it up from the context."),
    ("horizon", "Uses 16+ tokens of context", "context", "battery", ["horizon_tokens"], 16, ">=", None, "tokens", "2 held-out texts",
     "Cutting the context to 16 tokens still leaves a quarter of the full context's effect (and the model reacts to context at all)."),
    ("agree0", "Subject-verb agreement", "grammar", "battery", ["span", "0"], 0.75, ">=", None, "share right", "8 items",
     "\"The keys to the cabinet still are/is\": the verb agrees with the subject, not the nearest noun (chance 0.5)."),
    ("agree3", "Agreement across three phrases", "grammar", "battery", ["span", "3"], 0.75, ">=", None, "share right", "8 items",
     "The same with three phrases between subject and verb: the model holds the subject over a distance."),
    ("facts", "Stored facts", "knowledge", "battery", ["facts", "learned"], 0.75, ">=", ["facts", "margin"], "share right (margin in nats)", "40 facts",
     "The true completion beats a plausible wrong one for 75% of 40 facts (chance 0.5)."),
    ("icl_cap", "Learns a rule from examples: capitalise", "in-context learning", "battery", ["icl", "capitalise"], 0.5, ">=", ["icl_margin", "capitalise"], "top-1 share (margin)", "16 items",
     "From 4 examples (dog -> DOG), the 5th answer is its first choice in half the items."),
    ("icl_letter", "Learns a rule from examples: first letter", "in-context learning", "battery", ["icl", "first_letter"], 0.5, ">=", ["icl_margin", "first_letter"], "top-1 share (margin)", "16 items",
     "From 4 examples (apple -> a), the 5th answer is its first choice in half the items."),
    ("icl_arith", "Learns a rule from examples: addition", "in-context learning", "battery", ["icl", "arith"], 0.5, ">=", ["icl_margin", "arith"], "top-1 share (margin)", "16 items",
     "From 4 worked sums, the 5th sum's answer is its first choice in half the items."),
    ("icl_offset", "Continues a letter pattern", "in-context learning", "battery", ["icl", "offset"], 0.5, ">=", ["icl_margin", "offset"], "top-1 share (margin)", "16 items",
     "\"c d e f g\" -> h as its first choice in half the items."),
    ("icl_sent", "Learns a labelling from examples: sentiment", "in-context learning", "battery", ["icl", "sentiment"], 0.5, ">=", ["icl_margin", "sentiment"], "top-1 share (margin)", "16 items",
     "From 4 labelled phrases, the 5th label is its first choice in half the items."),
    ("qa", "Answers after \"A:\" instead of asking", "format", "battery", ["format_answers"], 0.75, ">=", ["format_margin"], "share right (margin)", "8 items",
     "After \"Q: ... A:\" an answer beats starting another question in 75% of items."),
    ("negation", "Handles \"not\"", "reasoning", "battery", ["negation"], 0.75, ">=", ["negation_margin"], "share right (margin)", "12 items",
     "\"The light is not red. The light is\" -> not red, in 75% of items. Early models pass by chance, fail once they learn to copy, and pass again later."),
    ("state", "Tracks a situation", "reasoning", "phase", ["state"], 0.75, ">=", None, "share right", "40 items",
     "Who has the book now, where the keys are: 75% right (chance 0.5)."),
    ("consequence", "Predicts a consequence", "reasoning", "phase", ["consequence"], 0.75, ">=", None, "share right", "24 items",
     "The glass dropped on stone shattered, not bounced: 75% right (chance 0.5)."),
    ("abstraction", "E-level abstraction in the middle layers", "reasoning", "phase", ["abstraction", "gain"], 0.05, ">=", None, "probe gain over layer 0", "356 sentences",
     "A linear probe on a middle layer places sentences on the E-levels at least 5 points better than on the word embeddings."),
]


def series(size):
    """{source: [(step, record)]} for one size."""
    out = {}
    b = load(f"battery-{size}.json"); out["battery"] = [(r["step"], r) for r in b["steps"]] if b else []
    ph = load("phase_shift.json" if size == "410m" else f"phase_shift-{size}.json")
    out["phase"] = [(r["step"], r) for r in ph] if ph else []
    ih = load("induction_heads_all.json") if size == "410m" else None
    out["heads"] = [(r["step"], {"n": sum(v > 0.3 for row in r["induction"] for v in row)}) for r in ih] if ih else []
    return out


def ok(v, thr, d): return v is not None and (v >= thr if d == ">=" else v <= thr)


def lt(step): return math.log10(max(step, 1) * TOKENS_PER_STEP)


def onset(curve, thr, d, cont=None, extra_ok=None):
    """curve: [(step, value)] sorted. The first step that passes and stays passing."""
    pts = [(s, v) for s, v in curve if v is not None]
    if not pts: return {"status": "not measured"}
    passes = [ok(v, thr, d) and (extra_ok is None or extra_ok(s)) for s, v in pts]
    first = None
    for i in range(len(pts)):
        if all(passes[i:]): first = i; break
    if first is None:
        return {"status": "not yet", "last_step": pts[-1][0], "last_value": pts[-1][1], "note": "does not pass and stay passing by the last checkpoint measured"}
    s, v = pts[first]
    o = {"status": "emerged", "step": s, "tokens_b": round(s * TOKENS_PER_STEP / 1e9, 3), "value": v}
    if first == 0:
        o["status"] = "from the start" if s == 0 else "by the first checkpoint"
        o["prev_step"] = None
        return o
    p = pts[first - 1][0]; o["prev_step"] = p; o["prev_tokens_b"] = round(p * TOKENS_PER_STEP / 1e9, 3)
    o["flickers"] = sum(1 for x in passes[:first] if x)   # passed earlier and lost it again
    if cont:
        cp = [(st, c) for st, c in cont if c is not None]
        if len(cp) >= 3:
            dmap = dict(cp); c0, c1 = cp[0][1], cp[-1][1]; tot = c1 - c0
            if p in dmap and s in dmap and abs(tot) > 1e-9:
                share = (dmap[s] - dmap[p]) / tot
                span = lt(cp[-1][0]) - lt(cp[0][0]); ish = (lt(s) - lt(p)) / span if span > 0 else 0
                ratio = share / ish if ish > 0 else None
                lab = ("sudden" if share >= 0.5 and (ratio or 0) >= 2 else "steady rise: the cut-off made it look sudden" if (ratio is not None and ratio < 1.5) or share < 0.25 else "partly sudden")
                o["jump"] = {"share": round(share, 3), "interval_share": round(ish, 3), "ratio": round(ratio, 2) if ratio else None, "label": lab,
                             "rule": "share of the continuous measure's whole rise inside the onset interval, against the interval's share of log-tokens; sudden: share >= 0.5 and >= 2x the even rate"}
    return o


def build():
    events = []
    for eid, label, group, src, path, thr, d, cpath, unit, items, means in EVENTS:
        ev = {"id": eid, "label": label, "group": group, "source": src, "path": path, "threshold": thr, "direction": d, "cont_path": cpath, "unit": unit, "items": items, "means": means,
              "onsets": {}, "curves": {}}
        for size in SIZES:
            S = series(size)[src]
            if not S: ev["onsets"][size] = {"status": "not measured"}; continue
            if src == "heads": curve = [(s, r["n"]) for s, r in S]
            else: curve = [(s, g(r, path)) for s, r in S]
            cont = [(s, g(r, cpath)) for s, r in S] if cpath else None
            extra = None
            if eid == "horizon":   # an untrained model's KL is small at every length: require that the context matters at all
                kl1 = {s: g(r, ["horizon_kl", "1"]) for s, r in S}; extra = lambda s, kl1=kl1: (kl1.get(s) or 0) >= 1.0
            ev["onsets"][size] = onset(curve, thr, d, cont, extra)
            ev["curves"][size] = [[s, None if v is None else round(float(v), 4)] for s, v in curve]
            if cont: ev.setdefault("cont_curves", {})[size] = [[s, None if v is None else round(float(v), 4)] for s, v in cont]
        events.append(ev)
    # The order of the stack: by onset in the most densely measured size (410m), then group.
    def key(e):
        o = e["onsets"].get("410m", {})
        return (o.get("step", 10 ** 9) if o.get("status") in ("emerged", "by the first checkpoint", "from the start") else 10 ** 9, GROUPS.index(e["group"]))
    events.sort(key=key)
    # Clusters: events whose onset intervals overlap in a size emerged together.
    clusters = {}
    for size in SIZES:
        iv = sorted([(e["onsets"][size].get("prev_step") or 0, e["onsets"][size]["step"], e["id"]) for e in events if e["onsets"].get(size, {}).get("status") == "emerged"])
        groups = []
        for a, b, i in iv:
            if groups and a < groups[-1]["to"] and b > groups[-1]["from"]:
                groups[-1]["events"].append(i); groups[-1]["from"] = max(groups[-1]["from"], a); groups[-1]["to"] = min(groups[-1]["to"], b)
            else: groups.append({"from": a, "to": b, "events": [i]})
        clusters[size] = [x for x in groups if len(x["events"]) > 1]
    sizes = []
    for size in SIZES:
        st = sorted({s for src in series(size).values() for s, _ in src})
        sizes.append({"id": size, "params": PARAMS[size], "steps": st})
    return {"meta": {"what": "Pythia checkpoints (deduped for 410m): the onset of each capability, from the stored measurements", "tokens_per_step": TOKENS_PER_STEP,
                     "rule": "onset = first checkpoint where the measure passes its threshold and passes at every later checkpoint measured; the onset lies in (previous checkpoint, onset]",
                     "mirage_check": "Schaeffer, Miranda & Koyejo 2023, Are Emergent Abilities of Large Language Models a Mirage?",
                     "sources": ["research/emergence_battery.py", "research/phase_shift.py", "the induction-head measurement (Pythia-410m checkpoints)"], "built_by": "research/emergence_timeline.py"},
            "groups": GROUPS, "sizes": sizes, "events": events, "clusters": clusters}


def main():
    out = build()
    json.dump(out, open(os.path.join(R, "emergence_timeline.json"), "w", encoding="utf-8"), indent=1)
    js = os.path.join(HERE, "..", "src", "ui", "timeline-data.js")
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        f.write("// The emergence timeline (research/emergence_timeline.py): onset of each capability in each Pythia size. Generated; do not edit.\n")
        f.write("window.E8_TIMELINE = " + json.dumps(out, separators=(",", ":")) + ";\n")
    for e in out["events"]:
        cells = []
        for s in SIZES:
            o = e["onsets"][s]
            cells.append(f"{s}:{(str(o.get('prev_step')) + '-' + str(o['step'])) if o.get('status') == 'emerged' else o['status'][:9]:>12}")
        print(f"{e['label'][:44]:44}", "  ".join(cells), (e["onsets"]["410m"].get("jump") or {}).get("label", ""))


if __name__ == "__main__":
    main()
