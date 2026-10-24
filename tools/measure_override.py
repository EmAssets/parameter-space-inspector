# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Measure an Override-map risk probe on the model in a folder, in the app, with the same discipline as the shipped ones
(psi-override-lab PREREG): questions graded from the outside, split three ways before anything is fitted.

  python tools/measure_override.py --model <weights folder> --out <data>/override-probes
                                   [--questions <file>] [--limit N]

1. Ask: every question in two wordings (as asked: "{q}" and "{q} Answer in one sentence."), greedy, the model's chat
   template, no framing: exactly as the Override map reads (src/override.py OverrideMap._run, the same code). Kept per
   answer: the text, the residual at the answer start and the mean over the first four answer pieces (every second
   layer), and the logit-lens reading. Resumable: a stopped run continues where it was.
2. Grade from the outside (no model involved): a real fact is supported when the answer contains an accepted answer;
   an invented subject or a future event is unsupported whatever is said; hedges (the rule in src/override.py) are left
   out. 3. Fit on the train part only: a logistic probe per layer and setting, the best chosen on the calibration part,
   where the flag threshold (false alarms at most 10% with 95% confidence) and the calibration are fitted too.
4. Test once on the test part, never seen before: the same metrics and bars as the shipped probes, the same controls
   (shuffled grades, the question's words alone, fame alone). A probe that misses a bar is kept with its evidence and
   gives no score: the Override map says why.

Questions: the default set (src/probes/override/questions-v1.jsonl: the set the shipped probes were measured on) or a
file of your own, one JSON object per line: {"q": "...", "answers": ["...", ...]} for a fact, {"q": "...", "kind":
"invented"} or "future" for something the model cannot know; optional "topic" (for the within-topic check), "id",
"split" ("train" | "cal" | "test"; otherwise assigned by a hash of the question: 50 / 20 / 30). CSV also works: columns
question, answers (separated by |), kind, topic.

Progress lines for the app: "PSI-PROGRESS {json}"; the last line is "PSI-DONE {json}" or "PSI-ERROR ...".
"""
import argparse, base64, csv, datetime, gzip, hashlib, io, json, math, os, re, sys, time, unicodedata
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
WORDINGS = {"plain": "{q}", "brief": "{q} Answer in one sentence."}
DEFAULT_SET = os.path.join(SRC, "probes", "override", "questions-v1.jsonl")
MIN_PER_PART = {"train": 300, "cal": 120, "test": 200}   # questions; each is asked twice
BARS = {"auroc": 0.85, "ci_low": 0.75, "fpr": 0.10, "ece": 0.10, "within_type": 0.75}
DIMS = 64


def say(kind, obj): print(f"PSI-{kind} {json.dumps(obj, ensure_ascii=False)}", flush=True)


# ── questions ────────────────────────────────────────────────────────────────
def load_questions(path):
    raw = open(path, encoding="utf-8-sig").read()
    rows = []
    if path.lower().endswith(".csv"):
        for r in csv.DictReader(io.StringIO(raw)):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            rows.append({"q": r.get("question") or r.get("q"), "answers": [a.strip() for a in (r.get("answers") or "").split("|") if a.strip()],
                         "kind": r.get("kind") or "", "topic": r.get("topic") or None, "split": r.get("split") or None, "id": r.get("id") or None})
    else:
        for n, l in enumerate(raw.splitlines(), 1):
            if not l.strip(): continue
            try: rows.append(json.loads(l))
            except ValueError: raise SystemExit(f"line {n} is not JSON")
    # The same words twice: kept once when they mean the same question; dropped when they do not (PopQA has several
    # films called "Mother", each with its own director). Either way no question can sit in two parts, where the probe
    # would be tested on an input it was trained on.
    texts = {}
    for r in rows: texts.setdefault(str(r.get("q") or "").strip(), set()).add(json.dumps([sorted(map(str, r.get("answers") or [])), r.get("rx"), str(r.get("kind") or r.get("group") or "")]))
    ambiguous = {q for q, v in texts.items() if len(v) > 1}
    out, seen = [], set()
    for n, r in enumerate(rows, 1):
        q = str(r.get("q") or "").strip()
        if not q: raise SystemExit(f"question {n}: no text (a 'q' field, or a 'question' column)")
        if q in seen or q in ambiguous: continue
        seen.add(q)
        kind = str(r.get("kind") or r.get("group") or "").strip().lower()
        kind = {"popqa": "fact", "fact": "fact", "past": "fact", "": "fact", "fiction": "invented", "invented": "invented", "future": "future"}.get(kind)
        if kind is None: raise SystemExit(f"question {n}: kind is fact, invented or future")
        ans = [str(a) for a in (r.get("answers") or []) if str(a).strip()]
        rx = r.get("rx")
        if kind == "fact" and not ans and not rx: raise SystemExit(f"question {n} ({q[:60]}): a fact needs its accepted answers")
        split = {"train": "train", "cal": "cal", "dev": "cal", "test": "test", "test3": "test"}.get(str(r.get("split3") or r.get("split") or "").lower())
        if split is None:
            h = int(hashlib.sha256(q.encode()).hexdigest()[:8], 16) % 100
            split = "train" if h < 50 else "cal" if h < 70 else "test"
        out.append({"id": str(r.get("id") or f"q{n:05d}"), "q": q, "kind": kind, "group": r.get("group") or {"fact": "fact", "invented": "invented", "future": "future"}[kind],
                    "answers": ans, "rx": rx, "topic": r.get("topic") or r.get("prop"), "fame": r.get("fame") or r.get("pop"), "split": split})
    c = {s: sum(1 for x in out if x["split"] == s) for s in MIN_PER_PART}
    short = [f"{s}: {c[s]} (needs {m})" for s, m in MIN_PER_PART.items() if c[s] < m]
    if short: raise SystemExit("too few questions for a fair test: " + "; ".join(short))
    load_questions.dropped = len(ambiguous)
    kinds = {x["kind"] for x in out}
    if len(kinds) < 2 and "fact" in kinds:
        pass   # facts alone can still have both outcomes (right and wrong)
    return out


def norm(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower())


def grade(it, answer, hedge):
    if hedge.search(answer): return "hedge"
    if it["kind"] == "fact":
        a = norm(answer)
        if it.get("rx"): return "correct" if re.search(it["rx"], a) else "wrong"
        return "correct" if any(norm(x) in a for x in it["answers"]) else "wrong"
    return "invented"


# ── small statistics, no extra packages (numpy and torch only) ──────────────
def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0: return float("nan")
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(y, s, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed); y, s, groups = map(np.asarray, (y, s, groups))
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; out = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        a = auroc(y[pick], s[pick])
        if a == a: out.append(a)
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)] if out else [None, None]


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float); e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def within(y, s, strata):
    y, s, strata = map(np.asarray, (y, s, strata)); num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        if len(set(y[m].tolist())) == 2: num += auroc(y[m], s[m]) * m.sum(); den += m.sum()
    return round(num / den, 3) if den else None


def logistic(X, y, C, iters=200):
    """L2 logistic regression as scikit-learn fits it (C times the summed log loss, plus half the squared weights; the
    intercept unpenalised), by L-BFGS in torch."""
    import torch
    X = torch.as_tensor(np.asarray(X, np.float64)); y = torch.as_tensor(np.asarray(y, np.float64))
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); z = X @ w + b
        loss = C * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum") + 0.5 * (w @ w)
        loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def fit_probe(X, y, C):
    mean = X.mean(0); scale = X.std(0); scale[scale == 0] = 1
    w, b = logistic((X - mean) / scale, y, C)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32), "coef": w.astype(np.float32), "intercept": b}


def score(p, X): return ((X - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"]


def binom_cdf(k, n, p):
    if p <= 0: return 1.0
    if p >= 1: return 0.0 if k < n else 1.0
    lp, lq, t, acc = math.log(p), math.log1p(-p), 0.0, 0.0
    for i in range(k + 1):
        t = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        acc += math.exp(t)
    return min(1.0, acc)


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound of a rate seen k times in n."""
    if k >= n: return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1 - conf: lo = mid
        else: hi = mid
    return hi


def cp_threshold(neg, cap=0.10):
    """The lowest score at which the false-alarm rate's 95% upper bound stays at or under the cap (as the lab's)."""
    neg = np.sort(np.asarray(neg))[::-1]; n = len(neg)
    for k in range(n - 1, -1, -1):
        if cp_upper(k, n) <= cap: return float(neg[k])
    return float(neg[0])


def bag_of_words(train_q, all_q):
    toks = lambda q: (lambda w: w + [f"{a} {b}" for a, b in zip(w, w[1:])])(re.findall(r"\w+", q.lower()))
    df = {}
    for q in train_q:
        for t in set(toks(q)): df[t] = df.get(t, 0) + 1
    vocab = {t: i for i, t in enumerate(sorted(t for t, c in df.items() if c >= 2))}
    M = np.zeros((len(all_q), len(vocab)), np.float32)
    for i, q in enumerate(all_q):
        for t in toks(q):
            j = vocab.get(t)
            if j is not None: M[i, j] += 1
    return M


# ── the run ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--out")
    ap.add_argument("--questions", default=DEFAULT_SET); ap.add_argument("--limit", type=int, default=0, help="for tests: this many questions per part")
    ap.add_argument("--check", action="store_true", help="only read the questions and count them per part and kind")
    ap.add_argument("--framing", default="", help="for experiments: a JSON file {name, system, tools} put into every prompt (a guide's instructions and tools); the probe then belongs to that framing, not to the plain reading")
    a = ap.parse_args()
    if not a.check and not (a.model and a.out): ap.error("--model and --out are needed (or --check)")
    try:
        if a.check:
            qs = load_questions(a.questions)
            say("DONE", {"questions": len(qs), "answers": len(qs) * 2, "parts": {s: sum(1 for x in qs if x["split"] == s) for s in MIN_PER_PART},
                         "kinds": {k: sum(1 for x in qs if x["kind"] == k) for k in ("fact", "invented", "future")},
                         "default": os.path.abspath(a.questions) == os.path.abspath(DEFAULT_SET),
                         "dropped_ambiguous": getattr(load_questions, "dropped", 0)})
            return
        run(a)
    except SystemExit as e:
        if e.code not in (0, None): say("ERROR", str(e.code)); sys.exit(1)
    except Exception as e:   # noqa: BLE001 - the app shows it
        say("ERROR", f"{type(e).__name__}: {e}"); raise


def run(a):
    qs = load_questions(a.questions)
    if a.limit:
        keep = []
        for part in MIN_PER_PART: keep += [x for x in qs if x["split"] == part][: a.limit]
        qs = keep
    default = os.path.abspath(a.questions) == os.path.abspath(DEFAULT_SET)
    set_name = "the default set (questions-v1: PopQA real facts at six levels of fame, invented subjects, dated past and future events)" if default else f"your questions ({os.path.basename(a.questions)})"
    import trace_server as ts, override as om
    say("PROGRESS", {"stage": "loading", "done": 0, "total": len(qs) * 2})
    tracer = ts.Tracer(a.model); ovm = om.OverrideMap(tracer); L = tracer.L
    # Measured in the app: through the engine's own tokenizer, as live readings of an app-measured probe are (a shipped
    # lab probe for these weights would otherwise switch the reading to the folder's unrepaired one).
    ovm._raw_tok = tracer.tok
    ovm.skip = {i for i in tracer.tok.convert_tokens_to_ids(["</think>", "<think>"]) if isinstance(i, int) and i >= 0 and i != tracer.tok.unk_token_id}
    framing = json.load(open(a.framing, encoding="utf-8")) if a.framing else None
    prefix = None
    if framing:
        # The framing's system prompt and tools in the model's chat template; its long fixed start is computed once
        # (in pieces of 1,024, so attention never holds the whole square) and reused for every question.
        import copy, torch
        from transformers import DynamicCache
        tok = tracer.tok
        kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
        sysmsg, tools = framing["system"], framing.get("tools") or None
        def text_of(q, s=None, t=None): return tok.apply_chat_template([{"role": "system", "content": sysmsg if s is None else s}, {"role": "user", "content": q}], tools=tools if t is None else t, tokenize=False, add_generation_prompt=True, **kw)
        if tools and text_of("x") == text_of("x", t=[]):
            # This chat template ignores tools (SmolLM2's does): they go into the system message as text, so the model
            # is told about them either way.
            sysmsg = f"{sysmsg}\n\nYou can call these tools. To call one, reply with <tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call>.\n{json.dumps(tools, ensure_ascii=False)}"
            framing = {**framing, "tools_in_system": True}
        def framed_ids(text, lens=False): return tok(text_of(text), return_tensors="pt", add_special_tokens=False).input_ids.to(tracer.device)
        ovm._ids = framed_ids
        SPLIT = "⁣QSPLIT⁣"; full = text_of(SPLIT); head = full[: full.index(SPLIT)]
        pids = tok(head, return_tensors="pt", add_special_tokens=False).input_ids.to(tracer.device)
        cut = pids.shape[1] - 1   # the last piece stays out: the question's tokenisation may join it
        cache = DynamicCache()
        with torch.no_grad():
            for c0 in range(0, cut, 1024): tracer.model(input_ids=pids[:, c0:min(cut, c0 + 1024)], past_key_values=cache, use_cache=True)
        prefix = {"cache": cache, "ids": pids[0, :cut].tolist(), "n": cut, "copy": copy.deepcopy}
        say("PROGRESS", {"stage": "loading", "done": 0, "total": len(qs) * 2, "framing": framing.get("name"), "framing_pieces": int(pids.shape[1])})
    key = f"{os.path.basename(os.path.normpath(a.model))}-{(tracer.weights_fp or tracer.weights_hash or 'x')[:12]}"
    os.makedirs(a.out, exist_ok=True)
    qkey = hashlib.sha256(json.dumps([[x["q"], x["split"]] for x in qs] + ([framing] if framing else [])).encode()).hexdigest()[:10]
    work = os.path.join(a.out, f"work-{key}-{qkey}"); os.makedirs(work, exist_ok=True)
    rows = [(i, w) for i in range(len(qs)) for w in WORDINGS]
    N = len(rows); layers = list(range(2, L + 1, 2)); D = tracer.model.config.hidden_size
    mm = lambda n, shape, dt: np.lib.format.open_memmap(os.path.join(work, n), mode="r+" if os.path.exists(os.path.join(work, n)) else "w+", dtype=dt, shape=shape)
    R0 = mm("resid_start.npy", (N, len(layers), D), np.float16); R4 = mm("resid_gen4.npy", (N, len(layers), D), np.float16)
    LENS = mm("lens.npy", (N, L, 8), np.int32)   # per layer: top-3 piece ids, their chances (thousandths), written piece's chance, KL (hundredths)
    mpath = os.path.join(work, "meta.jsonl")
    done = sum(1 for _ in open(mpath, encoding="utf-8")) if os.path.exists(mpath) else 0
    pieces, at = [], {}
    if os.path.exists(os.path.join(work, "pieces.json")): pieces = json.load(open(os.path.join(work, "pieces.json"), encoding="utf-8")); at = {p: i for i, p in enumerate(pieces)}
    def pid(t):
        if t not in at: at[t] = len(pieces); pieces.append(t)
        return at[t]
    t0 = time.time(); started = done
    with open(mpath, "a", encoding="utf-8") as mf:
        for n in range(done, N):
            i, w = rows[n]; it = qs[i]
            past = None
            if prefix:
                ids = ovm._ids(WORDINGS[w].format(q=it["q"]))[0].tolist()
                if ids[: prefix["n"]] == prefix["ids"]: past = prefix["copy"](prefix["cache"])
            r = ovm._run(WORDINGS[w].format(q=it["q"]), False, 48, past=past)
            tool_call = r["first_piece"].strip().startswith("<tool_call>") or r["answer"].lstrip().startswith(("<tool_call>", '{"name"'))
            st, g4 = r["states"]["start"], r["states"]["gen4"]
            R0[n] = np.stack([st[l].cpu().numpy() for l in layers]).astype(np.float16)
            R4[n] = np.stack([g4[l].cpu().numpy() for l in layers]).astype(np.float16)
            LENS[n] = np.array([[*[pid(t) for t, _ in x["top"]], *[int(round(p * 1000)) for _, p in x["top"]], int(round(x["p_written"] * 1000)), int(round(x["kl_to_final"] * 100))] for x in r["layers"]], np.int32)
            mf.write(json.dumps({"n": n, "i": i, "wording": w, "answer": r["answer"], "first_piece": r["first_piece"], "tool_call": tool_call, "cached": past is not None}, ensure_ascii=False) + "\n"); mf.flush()
            if (n + 1) % 10 == 0 or n + 1 == N:
                R0.flush(); R4.flush(); LENS.flush(); json.dump(pieces, open(os.path.join(work, "pieces.json"), "w", encoding="utf-8"))
                rate = (time.time() - t0) / max(1, n + 1 - started)
                say("PROGRESS", {"stage": "asking", "done": n + 1, "total": N, "seconds_left": round(rate * (N - n - 1)), "device": str(tracer.device)})
    json.dump(pieces, open(os.path.join(work, "pieces.json"), "w", encoding="utf-8"))
    say("PROGRESS", {"stage": "fitting", "done": N, "total": N})
    meta = [json.loads(l) for l in open(mpath, encoding="utf-8")]

    # An answer that opens with a tool call (only under a framing with tools) is left out like a hedge, and counted.
    lab = np.array(["tool_call" if m.get("tool_call") else grade(qs[m["i"]], m["answer"], om.HEDGE) for m in meta]); ans = ~np.isin(lab, ["hedge", "tool_call"])
    y = np.isin(lab, ["wrong", "invented"]); spl = np.array([qs[m["i"]]["split"] for m in meta]); qid = np.array([qs[m["i"]]["id"] for m in meta])
    kind = np.array([qs[m["i"]]["kind"] for m in meta]); group = np.array([str(qs[m["i"]].get("group")) for m in meta]); topic = np.array([str(qs[m["i"]].get("topic") or "") for m in meta])
    tr, ca, te = (spl == "train") & ans, (spl == "cal") & ans, (spl == "test") & ans
    for nm, msk in (("training", tr), ("calibration", ca), ("test", te)):
        if len(set(y[msk].tolist())) < 2: raise SystemExit(f"the {nm} part has only {'unsupported' if y[msk].all() else 'supported'} answers once hedges are left out: the probe cannot be fitted or judged. Use questions with both outcomes.")

    # Fit on train, choose on cal: the position (answer start or the first four pieces), the layer and C.
    best = None
    for ri, (rk, R) in enumerate((("start", R0), ("gen4", R4))):
        for li, l in enumerate(layers):
            X = np.asarray(R[:, li], np.float32)
            for C in (0.01, 0.1):
                p = fit_probe(X[tr], y[tr], C); a_ = auroc(y[ca], score(p, X[ca]))
                if best is None or a_ > best[0]: best = (a_, rk, li, l, C, p)
        say("PROGRESS", {"stage": "fitting", "done": ri + 1, "total": 2})
    cal_auc, rk, li, l, C, p = best
    X = np.asarray((R0 if rk == "start" else R4)[:, li], np.float32); s = score(p, X)
    thr = cp_threshold(s[ca & ~y])
    pa, pb = logistic(s[ca][:, None], y[ca], 1.0); pa = float(pa[0])
    prob = 1 / (1 + np.exp(-(pa * s + pb)))
    # Test once.
    ty, ts_ = y[te], s[te]; facts = te & (kind == "fact")
    has_topic = bool((topic[facts] != "").any())
    t = {"auroc": round(auroc(ty, ts_), 3), "ci": boot_ci(ty, ts_, qid[te]), "n": int(te.sum()), "n_unsup": int(ty.sum()),
         "fpr": round(float((ts_[~ty] > thr).mean()), 3), "tpr": round(float((ts_[ty] > thr).mean()), 3), "ece": round(ece(ty, prob[te]), 3),
         "within_popqa": round(auroc(y[facts], s[facts]), 3) if len(set(y[facts].tolist())) == 2 else None,
         "within_type": within(y[facts], s[facts], topic[facts]) if has_topic else None,
         "invented_vs_past": None}
    other = te & ((kind != "fact") | (group == "past"))
    if len(set(y[other].tolist())) == 2: t["invented_vs_past"] = round(auroc(y[other], s[other]), 3)
    # Controls.
    rng = np.random.default_rng(0); ysh = y.copy(); ysh[tr] = rng.permutation(y[tr])
    shuffled = round(auroc(y[ca], score(fit_probe(X[tr], ysh[tr], C), X[ca])), 3)
    qtext = np.array([qs[m["i"]]["q"] for m in meta]); B = bag_of_words(qtext[spl == "train"], qtext)
    bw, bb = logistic(B[tr], y[tr], 1.0); sb = B @ bw + bb
    fame = np.array([-math.log10(float(qs[m["i"]].get("fame") or 0) + 1) if qs[m["i"]].get("fame") is not None else np.nan for m in meta])
    controls = {"shuffled_grades_dev": shuffled, "question_text_only": round(auroc(ty, sb[te]), 3),
                "question_text_only_within_popqa": round(auroc(y[facts], sb[facts]), 3) if t["within_popqa"] is not None else None,
                "fame_only_within_popqa": round(auroc(y[facts], fame[facts]), 3) if t["within_popqa"] is not None and not np.isnan(fame[facts]).any() else None,
                "probe_within_popqa": t["within_popqa"], "trained_on_real_facts_only_scored_on_the_other_kinds": None}
    # Bars.
    checks = {"accuracy (AUROC at least 0.85)": t["auroc"] >= BARS["auroc"], "its interval's low end at least 0.75": (t["ci"][0] or 0) >= BARS["ci_low"],
              "false alarms at most 10%": t["fpr"] <= BARS["fpr"], "calibration error at most 0.10": t["ece"] <= BARS["ece"]}
    if t["within_type"] is not None: checks["within one topic, at least 0.75"] = t["within_type"] >= BARS["within_type"]
    if t["n"] < 200 or t["n_unsup"] < 40 or t["n"] - t["n_unsup"] < 40: checks["enough test answers of each outcome (200, with 40 of each)"] = False
    passed = all(checks.values())
    missed = [k for k, v in checks.items() if not v]
    trust = "experimental" if passed else "did not pass"
    why = ("passes the accuracy, false-alarm and calibration bars on its own test part; measured in the app, without the lab's planted-signal and cause checks"
           if passed else f"missed: {'; '.join(missed)}. It gives no score; its test answers can still be browsed")
    when = datetime.date.today().isoformat()
    b64 = lambda x: base64.b64encode(np.asarray(x, np.float32).tobytes()).decode()
    c = {s_: int(sum(1 for x in qs if x["split"] == s_)) for s_ in MIN_PER_PART}
    probe = {"model": os.path.basename(os.path.normpath(a.model)), "weights_hash": tracer.weights_hash, "weights_fp": tracer.weights_fp, "weights_fp_of": tracer.weights_fp_of,
             "kind": "override-risk", "layer": int(l), "resid": rk, "mean": b64(p["mean"]), "scale": b64(p["scale"]), "coef": b64(p["coef"]), "intercept": float(p["intercept"]),
             "platt": [pa, float(pb)], "threshold": float(thr), "trust": trust, "trust_why": why, "usable": passed,
             "metrics": t, "bank": f"trained on {c['train']:,} questions from {set_name}, 2 wordings each; threshold and calibration fitted on {c['cal']:,} more; measured once on {t['n']} answers (hedges excluded) to {c['test']} new questions",
             "question_set": "default" if default else "your own", "measured": when, "measured_by": "you, in the app", "device": str(tracer.device), "framing": (framing or {}).get("name"),
             "tool_calls": int((lab == "tool_call").sum()),
             "chosen_on_calibration": {"auroc": round(cal_auc, 3), "C": C}}
    # Evidence: every test answer, as the shipped probes carry it.
    Z = (X - p["mean"]) / p["scale"]; test_rows = np.where(spl == "test")[0]
    Zt = Z[test_rows]; mu = Zt.mean(0); _, S, Vt = np.linalg.svd(Zt - mu, full_matrices=False); comp = Vt[:DIMS]; P = (Zt - mu) @ comp.T
    rng = np.random.default_rng(0); uq = np.unique(qid[test_rows]); half = set(rng.permutation(uq)[: len(uq) // 2]); inA = np.array([q_ in half for q_ in qid[test_rows]])
    nn, rf = [], []
    for fit_on in (inA, ~inA):
        m_ = Zt[fit_on].mean(0); _, _, V_ = np.linalg.svd(Zt[fit_on] - m_, full_matrices=False); c_ = V_[:DIMS]
        Df, Dh = Zt[fit_on] - m_, Zt[~fit_on] - m_; Pf, Ph = Df @ c_.T, Dh @ c_.T
        rf += list(1 - (Ph ** 2).sum(1) / (Dh ** 2).sum(1))
        Pf /= np.linalg.norm(Pf, axis=1, keepdims=True); Ph /= np.linalg.norm(Ph, axis=1, keepdims=True); nn += list((Ph @ Pf.T).max(1))
    f16 = lambda x: base64.b64encode(np.asarray(x, np.float16).tobytes()).decode()
    KINDNAME = {"fact": "popqa", "invented": "fiction", "future": "future"}
    items = []
    for j, n in enumerate(test_rows):
        m = meta[n]; it = qs[m["i"]]
        items.append({"id": it["id"], "kind": "past" if it.get("group") == "past" else KINDNAME[it["kind"]],
                      "topic": it.get("topic"), "fame": it.get("fame"), "q": it["q"], "wording": m["wording"], "answer": m["answer"], "first_piece": m["first_piece"],
                      "expected": it["answers"][:4] or None, "grade": lab[n], "scored": bool(ans[n]), "score": round(float(s[n]), 3), "probability": round(float(prob[n]), 3),
                      "flagged": bool(s[n] > thr), "lens": LENS[n].tolist(), "near": f16(P[j])})
    ev = {"kind": "override-evidence", "model": probe["model"], "weights_hash": tracer.weights_hash, "weights_fp": tracer.weights_fp, "lab_run": None, "measured_by": "you, in the app",
          "question_set": probe["question_set"], "layer": int(l), "resid": rk, "threshold": float(thr), "metrics": t,
          "grading": {"popqa": "correct when the answer contains one of the accepted answers, else wrong", "past": "correct when the answer matches the known outcome, else wrong",
                      "fiction": "the subject was invented for the test: any answer is unsupported", "future": "the event has not happened: any answer is unsupported",
                      "hedge": "the answer declines or doubts (the hedge rule in src/override.py); left out of the scoring"},
          "controls": controls,
          "near": {"dims": int(comp.shape[0]), "mean": f16(mu), "components": f16(comp), "variance_kept": round(float((S[:DIMS] ** 2).sum() / (S ** 2).sum()), 3),
                   "nearest_similarity_percentiles": {str(q_): round(float(np.percentile(nn, q_)), 4) for q_ in (1, 5, 25, 50)},
                   "outside_fraction_percentiles": {str(q_): round(float(np.percentile(rf, q_)), 4) for q_ in (50, 95, 99)}},
          "lens_columns": ["top1", "top2", "top3", "p1", "p2", "p3", "p_written", "kl"], "lens_scale": {"p": 1000, "kl": 100}, "poor_fit_nats": om.POOR_FIT_NATS,
          "pieces": pieces, "items": items}
    base = os.path.join(a.out, key)
    with gzip.open(base + ".evidence.json.gz", "wt", encoding="utf-8") as f: json.dump(ev, f, ensure_ascii=False, separators=(",", ":"))
    json.dump(probe, open(base + ".json", "w", encoding="utf-8"), indent=1)
    say("DONE", {"probe": base + ".json", "trust": trust, "usable": passed, "missed": missed, "metrics": t, "layer": int(l), "resid": rk, "controls": controls})


if __name__ == "__main__":
    main()
