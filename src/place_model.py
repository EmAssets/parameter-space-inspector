"""
Place a model on the emergence timeline: measure it with the same tests the timeline was built from, say which
emergences it has, which Pythia checkpoint it is most like, and what emerged next there.

  python place_model.py --model <Hugging Face folder> --out <file.json> [--device auto|cpu|cuda] [--no-abstraction]

The tests are the research scripts' own (research/emergence_battery.py, phase_shift.py, understanding.py), imported,
not copied, so a placed model and the timeline are measured the same way. Each event of research/emergence_timeline.py
is scored with its threshold. A test whose right and wrong answers begin with the same first piece in this model's
vocabulary cannot be decided by the battery's first-piece margin: such an event is "not decidable" here.

Nearest checkpoint: every recorded Pythia (size, step) is a profile of event values; the distance is the mean over
shared events of |tanh(z_model) - tanh(z_ref)| with z = (value - threshold) / the event's spread. "Nearby" reads the
timeline at that checkpoint: what it already had, and what emerged next in that size.

The result carries a signed receipt (src/receipts.py). Progress: "PSI-PROGRESS {json}"; last line PSI-DONE or PSI-ERROR.
"""
import argparse, importlib.util, json, math, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "research")
sys.path.insert(0, HERE)


def say(kind, **o): print(f"PSI-{kind} {json.dumps(o)}", flush=True)


def mod(name):
    p = os.path.join(RES, f"{name}.py")
    if not os.path.exists(p): raise SystemExit(f"PSI-ERROR the research script {name}.py is missing: placing a model needs research/ beside src/")
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def catalog():
    t = open(os.path.join(HERE, "ui", "timeline-data.js"), encoding="utf-8").read()
    return json.loads(t[t.index("=") + 1:].strip().rstrip(";"))


def g(rec, path):
    for k in path or []:
        if rec is None: return None
        rec = rec.get(k) if isinstance(rec, dict) else None
    return rec


def step_of(path):
    m = re.search(r"(?:^|[\\/])step(\d+)(?:[\\/]|$)", path.replace("\\", "/") + "/")
    return int(m.group(1)) if m else None


def heads_count(model, dev, n=50, trials=2, seed=0, thr=0.3):
    """Heads whose attention from the second copy goes to the token after the earlier occurrence (the induction-head test the timeline used)."""
    import torch
    V = model.get_input_embeddings().weight.shape[0]; gen = torch.Generator().manual_seed(seed); acc = None
    for _ in range(trials):
        seq = torch.randint(100, min(V, 30000), (n,), generator=gen); ids = torch.cat([seq, seq])[None].to(dev)
        with torch.no_grad(): att = model(ids, output_attentions=True).attentions
        if not att or att[0] is None: return None
        i = torch.arange(n, 2 * n - 1, device=dev); j = i - n + 1
        s = torch.stack([a[0][:, i, j].float().mean(-1) for a in att])   # layers x heads
        acc = s if acc is None else acc + s
    acc = acc / trials
    return {"n": int((acc > thr).sum()), "threshold": thr, "top": round(float(acc.max()), 3), "layers": acc.shape[0], "heads": acc.shape[1]}


def decidable(tok, B):
    """For each battery test, the share of items whose right and wrong first pieces differ in this vocabulary."""
    fid = lambda s: (tok(s, add_special_tokens=False).input_ids or [None])[0]
    def share(items):
        d = [fid(gd) != fid(bd) for gd, bd in items]; return round(sum(d) / len(d), 3) if d else None
    out = {}
    out["facts"] = share([(gd, bd) for _, gd, bd in B.FACTS])
    for k in ("0", "3"): out[f"agree{k}"] = share([(gd, bd) for _, gd, bd in B.SPAN[int(k)]])
    for k, v in B.ICL.items(): out[f"icl:{k}"] = share([(" " + gd, " " + bd) for _, gd, bd in v])
    out["qa"] = share([(gs[0], "\n") for _, gs in B.FORMAT])
    out["negation"] = share([(gs[0], bs[0]) for _, gs, bs in B.NEGATION])
    return out


# The loss and the context tests count tokens, and a token is a different length in every vocabulary. The timeline's
# numbers are Pythia tokens: the held-out text (research/understanding.py HELDOUT, 427 characters) is 82 Pythia tokens.
# A model's loss is converted to bits per Pythia token (its total bits over the text / 81 predicted Pythia tokens); the
# context tests (in tokens) are not comparable when its tokens are much shorter or longer than Pythia's.
PY_HELD_CHARS, PY_HELD_TOKENS = 427, 82
TOKEN_LENGTH_OK = (0.6, 1.7)   # this model's characters per token / Pythia's

DEC_KEY = {"facts": "facts", "agree0": "agree0", "agree3": "agree3", "qa": "qa", "negation": "negation", "icl_cap": "icl:capitalise", "icl_letter": "icl:first_letter",
           "icl_arith": "icl:arith", "icl_offset": "icl:offset", "icl_sent": "icl:sentiment"}


def score_events(cat, battery, phase, heads, dec):
    out = {}; rel = phase.get("token_length_vs_pythia")
    for e in cat["events"]:
        if e["source"] == "battery": v = g(battery, e["path"]); c = g(battery, e.get("cont_path"))
        elif e["source"] == "phase": v = g(phase, e["path"]); c = g(phase, e.get("cont_path"))
        else: v = heads["n"] if heads else None; c = None
        r = {"value": v, "cont": c}
        if v is None: r["status"] = "not measured"
        else:
            ok = v >= e["threshold"] if e["direction"] == ">=" else v <= e["threshold"]
            if e["id"] == "horizon": ok = ok and (g(battery, ["horizon_kl", "1"]) or 0) >= 1.0
            r["status"] = "has it" if ok else "not yet"
        if e["id"] in ("context", "horizon") and rel is not None and not (TOKEN_LENGTH_OK[0] <= rel <= TOKEN_LENGTH_OK[1]) and v is not None:
            r["status"] = "not decidable"; r["why"] = f"this model's tokens are {rel:.2f} times as long as Pythia's: context lengths counted in tokens do not compare"
        k = DEC_KEY.get(e["id"])
        if k and dec.get(k) is not None:
            r["decidable_share"] = dec[k]
            if dec[k] < 0.5: r["status"] = "not decidable"; r["why"] = f"in this vocabulary the right and wrong answers share their first piece in {round((1 - dec[k]) * 100)}% of items"
        out[e["id"]] = r
    return out


def neighbours(cat, ev, sources=None):
    # Each event's spread over every recorded value, for the distance's scale.
    spread = {}
    for e in cat["events"]:
        vals = [v for cv in e["curves"].values() for _, v in cv if v is not None]
        if len(vals) > 1:
            m = sum(vals) / len(vals); spread[e["id"]] = max(math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)), 1e-6)
    refs = {}
    for e in cat["events"]:
        if sources and e["source"] not in sources: continue
        for size, cv in e["curves"].items():
            for s, v in cv:
                if v is not None: refs.setdefault((size, s), {})[e["id"]] = v
    thr = {e["id"]: e["threshold"] for e in cat["events"]}
    rows = []
    for (size, s), prof in refs.items():
        ds = []
        for k, v in prof.items():
            m = ev.get(k)
            if not m or m.get("status") in ("not measured", "not decidable") or m.get("value") is None or k not in spread: continue
            ds.append(abs(math.tanh((m["value"] - thr[k]) / spread[k]) - math.tanh((v - thr[k]) / spread[k])))
        if len(ds) >= 5: rows.append({"size": size, "step": s, "tokens_b": round(s * cat["meta"]["tokens_per_step"] / 1e9, 3), "distance": round(sum(ds) / len(ds), 4), "shared_events": len(ds)})
    # Only references measured on (nearly) as many tests as the best-covered one: a checkpoint with fewer tests in
    # common would look closer only because less of it is compared.
    most = max((r["shared_events"] for r in rows), default=0)
    rows = [r for r in rows if r["shared_events"] >= most - 1]
    rows.sort(key=lambda r: r["distance"])
    return rows[:5]


def derive(cat, ev):
    nb = neighbours(cat, ev)
    return {"nearest": nb, "nearest_battery": neighbours(cat, ev, {"battery"}), "nearby": nearby(cat, ev, nb[0] if nb else None),
            "note": "derived from the result and the timeline; not part of the signed numbers. nearest: among checkpoints measured on the most tests; nearest_battery: the battery tests only, every size"}


def nearby(cat, ev, nb):
    if not nb: return None
    size, step = nb["size"], nb["step"]; had, nxt = [], []
    for e in cat["events"]:
        o = e["onsets"].get(size, {}); st = o.get("status")
        if st in ("from the start", "by the first checkpoint") or (st == "emerged" and o["step"] <= step): had.append(e["id"])
        elif st == "emerged": nxt.append({"id": e["id"], "step": o["step"], "prev_step": o.get("prev_step"), "tokens_b": o.get("tokens_b")})
        elif st == "not yet": nxt.append({"id": e["id"], "step": None, "note": f"not by the last {size} checkpoint"})
    nxt.sort(key=lambda x: x["step"] if x["step"] is not None else 10 ** 9)
    has = lambda i: (ev.get(i) or {}).get("status") == "has it"
    known = lambda i: (ev.get(i) or {}).get("status") in ("has it", "not yet")
    return {"neighbour": {"size": size, "step": step},
            "shared": [i for i in had if has(i)], "behind": [i for i in had if known(i) and not has(i)],
            "next": [dict(x, has_it=has(x["id"])) for x in nxt], "ahead": [x["id"] for x in nxt if has(x["id"])]}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--rederive":   # recompute the unsigned reading of a saved placement (after the timeline changes)
        f = sys.argv[2]; d = json.load(open(f, encoding="utf-8")); d["derived"] = derive(catalog(), d["result"]["events"])
        json.dump(d, open(f, "w", encoding="utf-8"), indent=1); print(f"PSI-DONE {json.dumps({'out': f, 'nearest': (d['derived']['nearest'] or [None])[0]})}"); return
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto"); ap.add_argument("--no-abstraction", action="store_true"); ap.add_argument("--name")
    a = ap.parse_args(); t0 = time.time()
    model_dir = os.path.abspath(a.model)
    if not os.path.isfile(os.path.join(model_dir, "config.json")): raise SystemExit(f"PSI-ERROR {model_dir} is not a Hugging Face model folder (no config.json)")
    import receipts, torch
    say("PROGRESS", stage="hash", note="hashing the model's files for the receipt")
    ident = receipts.model_identity(model_dir, lambda i, n, f: say("PROGRESS", stage="hash", done=i, of=n, note=f"hashing {f}"))
    say("PROGRESS", stage="load", note="loading the research tests")
    B = mod("emergence_battery"); PS = mod("phase_shift"); cat = catalog()
    size_b = sum(f["bytes"] for f in ident["files"] if f["name"].endswith((".safetensors", ".bin")))
    dev = a.device
    if dev == "auto":
        import devices
        dev = devices.pick((size_b * 1.6 + 1.5e9) / 2 ** 30)
    if dev == "cpu": torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    PS.DEV = dev
    say("PROGRESS", stage="load", note=f"loading the model on the {dev}")
    m = B.M(model_dir, dev); model, tok = m.model, m.tok
    dtype = next(model.parameters()).dtype
    if tok.pad_token is None and tok.eos_token is not None: tok.pad_token = tok.eos_token
    say("PROGRESS", stage="battery", note="the emergence battery (in-context learning, horizon, facts, agreement, format, negation)")
    bat = B.battery(m)
    say("PROGRESS", stage="phase", note="copying, context use, absorption, situation tests")
    und = PS.und; ph = {}
    ph["induction"] = PS.induction_score(model, tok); ph["ctx_sensitivity"] = PS.ctx_sensitivity(model, tok); ph["absorption"] = PS.absorption(model, tok)
    ph["consequence"] = float(sum(und.best(model, tok, p, gd) > und.best(model, tok, p, bd) for p, gd, bd in und.CONSEQUENCE) / len(und.CONSEQUENCE))
    ph["state"] = float(sum(und.logp_next(model, tok, p, gd) > und.logp_next(model, tok, p, bd) for p, gd, bd in und.STATE) / len(und.STATE))
    ids = tok(und.HELDOUT, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): own = float(model(ids, labels=ids).loss) / math.log(2)
    n = ids.shape[1]; ph["bits_per_own_token"] = own; ph["held_tokens"] = n
    ph["bits_per_token"] = own * (n - 1) / (PY_HELD_TOKENS - 1) if len(und.HELDOUT) == PY_HELD_CHARS else own   # per Pythia token (see above)
    ph["token_length_vs_pythia"] = round((len(und.HELDOUT) / n) / (PY_HELD_CHARS / PY_HELD_TOKENS), 3)
    if not a.no_abstraction:
        say("PROGRESS", stage="abstraction", note="E-level probe on every layer")
        try: ph["abstraction"] = PS.abstraction(model, tok, json.load(open(os.path.join(RES, "eledata.json"), encoding="utf-8")))
        except Exception as e: ph["abstraction"] = None; say("PROGRESS", stage="abstraction", note=f"skipped: {type(e).__name__}: {e}")
    say("PROGRESS", stage="heads", note="heads that look back")
    try: heads = heads_count(model, dev)
    except Exception as e: heads = None; say("PROGRESS", stage="heads", note=f"skipped: {type(e).__name__}: {e}")
    dec = decidable(tok, B)
    ev = score_events(cat, bat, ph, heads, dec)
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    params = sum(p.numel() for p in model.parameters())
    result = {"battery": {k: v for k, v in bat.items()}, "phase": ph, "heads": heads, "decidable": dec, "events": ev}
    code = [os.path.abspath(__file__), os.path.join(HERE, "receipts.py")] + [os.path.join(RES, f"{n}.py") for n in ("emergence_battery", "phase_shift", "understanding", "ladder_probe", "emergence_timeline")]
    subject = {"name": a.name or os.path.basename(model_dir.rstrip("\\/")), "arch": (cfg.get("architectures") or [None])[0], "params": params, "weights_sha256": ident["weights_sha256"], "files": ident["files"]}
    receipt = receipts.make("psi-placement", subject, {"what": "emergence timeline placement", "code": receipts.code_identity(code), "seed": 0, "catalog_events": [e["id"] for e in cat["events"]]},
                            receipts.env(dev, dtype), result)
    der = derive(cat, ev); nb = der["nearest"]
    out = {"model": {"name": subject["name"], "path": model_dir.replace("\\", "/"), "arch": subject["arch"], "params": params, "step": step_of(model_dir),
                     "family": os.path.basename(os.path.dirname(model_dir.rstrip("\\/"))) if step_of(model_dir) is not None else None},
           "measured_at": receipts.now(), "device": dev, "dtype": str(dtype).replace("torch.", ""), "seconds": round(time.time() - t0, 1),
           "result": result, "derived": der,
           "receipt": receipt}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"; json.dump(out, open(tmp, "w", encoding="utf-8"), indent=1); os.replace(tmp, a.out)
    has = sum(1 for r in ev.values() if r["status"] == "has it")
    say("DONE", out=os.path.abspath(a.out), has=has, of=len(ev), nearest=nb[0] if nb else None, seconds=out["seconds"], signed=bool(receipt["signature"]))


if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if e.code and not str(e.code).startswith("PSI-ERROR"): print(f"PSI-ERROR {e.code}", flush=True)
        elif e.code: print(e.code, flush=True)
        sys.exit(1 if e.code else 0)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"PSI-ERROR {type(e).__name__}: {e}", flush=True); sys.exit(1)
