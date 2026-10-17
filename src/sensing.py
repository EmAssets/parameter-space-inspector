"""
The Sensing Surface, as instruments (emergencemachine.com/the-sensing-surface/).

The essay's cycle: absorption (a framework stretches to fit), anomaly (something stops fitting),
abstraction (something gets a name). Four tools built from the Inspector's parts:

  surprise(text)        ANOMALY. Structural surprise: each word's surprise given the whole text,
                        minus its surprise given only the three words before it (the short-context
                        reading, a 3-word window: an autocomplete's view, read by the same model).
                        Rare words and sentence openings are surprising to both readings and cancel;
                        what remains is surprise at what the text claims. Per word and per sentence,
                        ranked. Ranks are compared against a stored null distribution of the same
                        score on ordinary text (research/sensing_null.py) when one has been measured
                        for the loaded model; until then the ranks are relative only.
  flip(...)             ABSORPTION. A claim, then k contradicting observations, then a question: the
                        k at which the model's answer flips from the claim to the evidence (margin
                        above a floor of 0.5 nats; 0 to 0.5 is "leaning, within noise"). The answer
                        is scored whole (sum of log-probs over all its tokens). A framing in front is
                        compared against a neutral prefix of the same length, not against nothing;
                        the observations are also permuted (N=5 orderings) so the flip count is a
                        range. Runs on the inside engine's model.
  cycle(curve)          The cycle in any curve over training: absorption (the plateau), anomaly (the
                        onset of the drop beyond the tolerance), recovery (converge: back to the
                        earlier level; unlock: above the earlier peak). Tolerance from the measure's
                        n (binomial SE) for accuracies, 0.5 nats for margins.
  (lexicon order)       ABSTRACTION. Measured offline, shown as static data (src/ui/lexicon-order.js).
"""
import json, math, os, re, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Null distribution of structural surprise on ordinary text, per model (research/sensing_null.py writes it).
NULL_DIR = os.environ.get("PSI_SENSING_NULL_DIR") or os.path.join(ROOT, ".userdata", "sensing-null")
FLOOR = 0.5          # nats: below this the evidence answer is only "leaning", within noise
SHUFFLES = 5         # orderings of the observations, so the flip count is a range
NULL_PCT = 95        # a sentence is "beyond ordinary text" from this percentile of the null
# A neutral prefix: plain description with no claim about the question; cut to the framing's token length.
NEUTRAL = ("The lighthouse on the point was built in 1871 and still turns its original lens. The keeper's cottage beside it is a small museum, "
           "open on weekend afternoons from May to September. Visitors climb the hundred and twelve steps to the lamp room, where the view takes in "
           "the whole bay and, on a clear day, the islands beyond it. The path down to the shore passes a bench, a stand of pines and the remains of "
           "a boathouse. Gulls nest on the ledges below the gallery in spring. The town's bus stops at the gate twice a day in summer and once in winter. ")


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "model"


class Sensing:
    def __init__(self, ckpt, tracer, final_step=143000, freq_step=512):
        # ckpt and the step arguments are kept for the server's call; the short-context reading no longer
        # loads an early checkpoint (that version mistook word endings for predictability).
        self.ckpt = ckpt; self.t = tracer; self.final_step = final_step; self.freq_step = freq_step
        self._null = None; self._null_for = None

    def _name(self):
        """The inside engine's model by name: Qwen3-4B for the default weights, else the folder's name
        (a training checkpoint folder keeps its parent: pythia-70m@step143000, not step143000)."""
        parts = str(getattr(self.t.model.config, "_name_or_path", "") or "").replace("\\", "/").rstrip("/").split("/")
        p = parts[-1]
        if re.fullmatch(r"step\d+", p.lower()) and len(parts) > 1: p = f"{parts[-2]}@{p}"
        p = p[:-3] if p.lower().endswith("-hf") else p
        return "Qwen3-4B" if p.lower().startswith("qwen3-4b") else (p or "the loaded model")

    # ── anomaly ──────────────────────────────────────────────────────────────
    # Scored per WORD (whitespace-delimited), so word endings ("up|hill") are not mistaken for surprises.
    # Both readings by the same model: the whole text, and a 3-word window before each word.
    @torch.no_grad()
    def _token_bits(self, tok, model, dev, text, topk=3):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        ids = enc.input_ids[:, :1024].to(dev); offs = enc.offset_mapping[0, :1024].tolist()
        lg = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
        bits = (-lg[torch.arange(ids.shape[1] - 1), ids[0, 1:]] / math.log(2)).cpu().numpy()
        top = lg.topk(topk, -1).indices.cpu().tolist()   # what the whole text led the model to expect at each position
        return bits, offs[1:], top

    @staticmethod
    def _words(text):
        return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]

    def _word_bits(self, text):
        tok, model, dev = self.t.tok, self.t.model, self.t.device
        bits, offs, top = self._token_bits(tok, model, dev, text)
        words = self._words(text); out = np.full(len(words), np.nan); expected = [None] * len(words); k = 0
        for b, (o0, o1), tp in zip(bits, offs, top):
            last = o1 - 1
            while k < len(words) and words[k][1] <= last: k += 1
            if k < len(words) and words[k][0] <= last:
                if np.isnan(out[k]): expected[k] = [tok.decode([i]).strip() for i in tp]   # expectation at the word's first piece
                out[k] = (0 if np.isnan(out[k]) else out[k]) + b
        return words, out, expected

    @torch.no_grad()
    def _short_bits(self, text, words, window=3, bs=32):
        """Bits of each word given only the `window` words before it (an autocomplete's view).
        The first word has no context at all and gets no number (NaN), not 0."""
        tok, model, dev = self.t.tok, self.t.model, self.t.device
        out = np.full(len(words), np.nan); jobs = []
        for i in range(1, len(words)):
            ctx = text[words[max(0, i - window)][0]:words[i - 1][1]]; tgt = text[words[i - 1][1]:words[i][1]]
            c = tok(ctx, add_special_tokens=False).input_ids; t = tok(ctx + tgt, add_special_tokens=False).input_ids
            if t[:len(c)] != c or len(t) <= len(c): c = c[:-1] if len(c) > 1 else c
            jobs.append((i, t, len(c)))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        for j in range(0, len(jobs), bs):
            B = jobs[j:j + bs]; L = max(len(t) for _, t, _ in B)
            ids = torch.tensor([t + [pad] * (L - len(t)) for _, t, _ in B], device=dev)
            att = torch.tensor([[1] * len(t) + [0] * (L - len(t)) for _, t, _ in B], device=dev)
            lg = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
            for k, (i, t, nc) in enumerate(B):
                out[i] = sum(-float(lg[k, q - 1, t[q]]) for q in range(nc, len(t))) / math.log(2)
        return out

    # ── the null: structural surprise on ordinary text, for this model ──────
    def null_path(self):
        return os.path.join(NULL_DIR, f"{_slug(self._name())}.json")

    def _load_null(self):
        name = self._name()
        if self._null_for == name: return self._null
        self._null_for = name; self._null = None
        p = self.null_path()
        if os.path.exists(p):
            try:
                d = json.load(open(p, encoding="utf-8")); d["scores"] = sorted(float(x) for x in d["scores"]); self._null = d
            except Exception as e:   # a damaged file is reported, not read as "no calibration"
                self._null = {"error": f"{type(e).__name__}: {e}", "scores": []}
        return self._null

    @staticmethod
    def _pct(sorted_scores, x):
        return round(100.0 * float(np.searchsorted(sorted_scores, x, side="right")) / len(sorted_scores), 1)

    def calibrate(self, paragraphs, n_sentences=200, window=3, out=None, source="", max_sentences_per_paragraph=5):
        """Measure the null: sentence structural-surprise scores on ordinary text, read the same way
        as a live text (each sentence inside its own paragraph). Writes JSON to null_path() (or `out`)."""
        t0 = time.time(); scores = []; words = []; used = 0
        for para in paragraphs:
            sents = _sentences(para)[:max_sentences_per_paragraph]
            if len(sents) < 2: continue
            r = self.surprise(" ".join(sents), window=window, calibrated=False)
            scores += [s["structural"] for s in r["sentences"]]; words += [t["structural"] for t in r["tokens"] if t["structural"] is not None]; used += 1
            if len(scores) >= n_sentences: break
        d = {"model": self._name(), "window": window, "n_sentences": len(scores), "n_paragraphs": used, "n_words": len(words), "source": source,
             "date": time.strftime("%Y-%m-%d"), "unit": "bits", "scores": sorted(round(float(x), 2) for x in scores),
             "word_p95": round(float(np.percentile(words, 95)), 2) if words else None, "seconds": round(time.time() - t0, 1)}
        p = out or self.null_path(); os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(d, open(p, "w", encoding="utf-8"), indent=0); self._null_for = None
        return d

    def surprise(self, text, top=8, window=3, calibrated=True):
        """Structural surprise of each word = its surprise given the whole text minus its surprise
        given only the previous `window` words. Positive: the wider context argues against the
        word, though locally it reads fine; that is an anomaly in what the text claims. Rare words
        and sentence openings are surprising either way and cancel. A word with no short context
        (the first word) has no score: None, not 0."""
        t0 = time.time()
        with torch.no_grad():
            words, full, expected = self._word_bits(text)
            short = self._short_bits(text, words, window)
        resid = full - short   # the wider context argues against the word by this many bits
        f = lambda x: None if np.isnan(x) else round(float(x), 2)
        tokens = [{"piece": (text[words[i - 1][1]:w[0]] if i else "") + w[2], "bits": f(full[i]), "short_bits": f(short[i]),
                   "structural": f(resid[i]), "expected": expected[i]} for i, w in enumerate(words)]
        sents = []; pos = 0
        for sent in _sentences(text):
            i0 = text.find(sent, pos); i1 = i0 + len(sent); pos = i1
            idx = [k for k, w in enumerate(words) if w[0] >= i0 and w[1] <= i1 and not np.isnan(resid[k])]
            if not idx: continue
            r = resid[idx]; topk = np.sort(r)[-2:]; pk = idx[int(np.argmax(r))]
            sents.append({"text": sent, "words": len(idx), "structural": round(float(topk.mean()), 2), "mean": round(float(r.mean()), 2), "peak": round(float(r.max()), 2),
                          "raw_bits": round(float(np.nanmean(full[idx])), 2), "short_bits": round(float(np.nanmean(short[idx])), 2),
                          "peak_piece": words[pk][2], "peak_expected": expected[pk]})
        ranked = sorted(range(len(sents)), key=lambda k: -sents[k]["structural"])
        for rank, k in enumerate(ranked): sents[k]["rank"] = rank + 1
        calib = {"state": "not yet measured", "how": "research/sensing_null.py measures it on ordinary text for the loaded model"}
        if calibrated:
            nl = self._load_null()
            if nl and nl.get("scores"):
                S = nl["scores"]; floor = float(np.percentile(S, NULL_PCT))
                for s in sents: s["percentile"] = self._pct(S, s["structural"]); s["beyond_ordinary"] = s["structural"] > floor
                calib = {"state": "measured", "n_sentences": len(S), "model": nl.get("model"), "date": nl.get("date"), "source": nl.get("source"), "pct": NULL_PCT, "floor": round(floor, 2),
                         "median": round(float(np.median(S)), 2), "word_p95": nl.get("word_p95")}
            elif nl and nl.get("error"): calib = {"state": "unreadable", "error": nl["error"], "path": self.null_path()}
        name = self._name()
        return {"tokens": tokens, "sentences": sents, "top": [sents[k] for k in ranked[:top]], "window": window, "calibration": calib,
                "models": {"final": f"{name}, whole text", "short": f"{name}, short-context reading ({window}-word window)"},
                "model": name, "unit": "word", "score_unit": "bits", "seconds": round(time.time() - t0, 1)}

    # ── absorption ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def _answer_logps(self, pairs, bs=16):
        """Sum of log-probs over every token of each answer after its prompt (the whole answer, not
        its first piece), for a list of (prompt, answer) pairs. Batched with right padding."""
        tok, m, dev = self.t.tok, self.t.model, self.t.device; jobs = []
        for prompt, ans in pairs:
            c = tok(prompt, add_special_tokens=False).input_ids; t = tok(prompt + ans, add_special_tokens=False).input_ids
            if t[:len(c)] != c or len(t) <= len(c): c = c[:-1] if len(c) > 1 else c
            jobs.append((t, len(c)))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0; out = []
        for j in range(0, len(jobs), bs):
            B = jobs[j:j + bs]; L = max(len(t) for t, _ in B)
            ids = torch.tensor([t + [pad] * (L - len(t)) for t, _ in B], device=dev)
            att = torch.tensor([[1] * len(t) + [0] * (L - len(t)) for t, _ in B], device=dev)
            lg = torch.log_softmax(m(input_ids=ids, attention_mask=att).logits.float(), -1)
            for k, (t, nc) in enumerate(B): out.append(sum(float(lg[k, q - 1, t[q]]) for q in range(nc, len(t))))
        return out

    def _neutral(self, framing):
        """Ordinary text cut to the framing's token length: the control for 'any prefix of this length'."""
        tok = self.t.tok; n = len(tok(framing, add_special_tokens=False).input_ids)
        ids = tok(NEUTRAL * (1 + n // max(1, len(tok(NEUTRAL, add_special_tokens=False).input_ids))), add_special_tokens=False).input_ids[:max(1, n)]
        return tok.decode(ids).strip()

    def flip(self, claim, observations, question, prior, new, framing=""):
        """Margin (nats) of the evidence answer over the claim answer after k = 0..n observations:
        plain, with a neutral prefix of the framing's length, and with the framing. Each over the
        given order and SHUFFLES permutations, so flip_at is also a range. flip_at: the first k from
        which the margin stays above FLOOR; leaning_at: above 0."""
        t0 = time.time(); n = len(observations); prior = [p for p in prior if p.strip()]; new = [p for p in new if p.strip()]
        rng = np.random.default_rng(0)
        orders = [list(range(n))] + [list(rng.permutation(n)) for _ in range(SHUFFLES if n > 1 else 0)]
        def ctx(prefix, order, k):
            obs = [observations[i] for i in order[:k]]
            return (prefix + "\n\n" if prefix else "") + claim + ("\n" + "\n".join(obs) if obs else "") + "\n" + question
        def first_from(curve, floor): return next((k for k, v in enumerate(curve) if v > floor and all(x > floor for x in curve[k:])), None)
        def run(prefix):
            pairs = [(ctx(prefix, o, k), a) for o in orders for k in range(n + 1) for a in prior + new]
            lp = self._answer_logps(pairs); A = len(prior) + len(new); curves = []; q = 0
            for _ in orders:
                curve = []
                for _k in range(n + 1):
                    seg = lp[q:q + A]; q += A
                    curve.append(round(max(seg[len(prior):]) - max(seg[:len(prior)]), 2))
                curves.append(curve)
            main = curves[0]; flips = [first_from(c, FLOOR) for c in curves]; seen = [f for f in flips if f is not None]
            noise = round(float(np.mean(np.std(np.array(curves)[:, 1:], 0))), 2) if len(curves) > 1 and n else 0.0
            return {"margins": main, "flip_at": flips[0], "leaning_at": first_from(main, 0.0), "flip_range": [min(seen), max(seen)] if seen else None,
                    "never": len(flips) - len(seen), "orderings": len(curves), "curves": curves[1:], "noise": noise}
        with self.t.lock:
            plain = run(""); framed = neutral = None
            if framing.strip(): neutral = run(self._neutral(framing)); framed = run(framing)
        verdict = None; prefix_effect = None
        if framed:
            a, b = neutral["flip_at"], framed["flip_at"]; k_ = lambda k: f"{k} observation{'s' if k != 1 else ''}"
            if a is None and b is None: verdict = f"neither the neutral prefix nor the framing lets the answer flip within {n} observations"
            elif a is not None and b is None: verdict = "the framing stops the flip that a neutral prefix of the same length allows: absorption made indefinitely stretchy"
            elif a is None: verdict = f"only the framing lets the evidence win (at {k_(b)}); the neutral prefix never does"
            elif a == b: verdict = f"flipped at the same count ({k_(b)}) under the framing and under the neutral prefix"
            elif b > a: verdict = f"the framing delays the flip by {b - a} ({k_(a)} neutral, {k_(b)} framed): absorption made stretchier"
            else: verdict = f"the framing speeds the flip by {a - b} ({k_(a)} neutral, {k_(b)} framed): it makes the model more open to the evidence"
            ra, rb = neutral["flip_range"], framed["flip_range"]
            if ra and rb and a != b and max(ra[0], rb[0]) <= min(ra[1], rb[1]): verdict += "; across orderings the two flip counts overlap, so the difference in count is within noise"
            if n:
                d = float(np.mean(np.array(framed["margins"][1:]) - np.array(neutral["margins"][1:]))); nz = max(FLOOR, neutral["noise"], framed["noise"])
                verdict += (f"; on average the framing {'weakens' if d < 0 else 'strengthens'} the evidence’s hold by {abs(d):.1f} nats against the neutral prefix" if abs(d) >= nz
                            else f"; the average margin differs from the neutral prefix by {d:+.1f} nats, within noise (ordering spread ±{nz:.1f})")
                dp = float(np.mean(np.array(neutral["margins"][1:]) - np.array(plain["margins"][1:])))
                prefix_effect = f"the neutral prefix alone moves the average margin by {dp:+.1f} nats against no prefix" + ("" if abs(dp) < nz else ": a prefix of this length matters by itself")
        return {"plain": plain, "neutral": neutral, "framed": framed, "verdict": verdict, "prefix_effect": prefix_effect, "n": len(observations), "floor": FLOOR,
                "unit": "nats", "model": self._name(), "scoring": "sum of log-probs over every token of each answer; max over the answers on each side",
                "shuffles": len(orders) - 1, "seconds": round(time.time() - t0, 1)}


# ── the cycle in a curve ─────────────────────────────────────────────────────
def tolerance(kind="accuracy", n=None, tol=None):
    """The dip a curve must make before it is read as an anomaly: twice the binomial standard error
    at p = 0.5 for an accuracy over n items (1/sqrt(n)); 0.5 nats for a margin; 0.2 when n is unknown."""
    if tol is not None: return float(tol)
    if kind == "margin": return 0.5
    return 1.0 / math.sqrt(n) if n else 0.2


def cycle(steps, values, chance=None, tol=None, n=None, kind="accuracy"):
    """Label the absorption / anomaly / recovery cycle in a curve over training.
    absorption: the early plateau or slide while an old framework (word statistics) holds;
    anomaly:    the onset of the mismatch, the first step that drops more than tol below the earlier
                peak (trough_at: where it is worst, often below chance);
    recovery:   the climb after it. converge: back to within tol of the earlier level;
                unlock: past the earlier peak by more than tol (the essay's two outcomes).
    tol comes from the measure: 1/sqrt(n) for an accuracy over n items, 0.5 nats for a margin."""
    tol = tolerance(kind, n, tol)
    # Step 0 is a random network, not a framework: start from the first trained checkpoint.
    if steps and steps[0] == 0: steps, values = list(steps[1:]), list(values[1:])
    v = list(values); m = len(v); out = []
    i = 0
    while i < m - 2:
        peak_i = i
        while peak_i + 1 < m and v[peak_i + 1] >= v[peak_i] - 1e-9: peak_i += 1
        # the trough after the peak, up to the first return to the peak level
        tr = peak_i
        for j in range(peak_i + 1, m):
            if v[j] < v[tr]: tr = j
            if v[j] > v[peak_i]: break
        if v[peak_i] - v[tr] > tol:
            onset = next(j for j in range(peak_i + 1, m) if v[j] < v[peak_i] - tol)
            rec = next((j for j in range(tr + 1, m) if v[j] >= v[peak_i] - tol), None)
            unl = next((j for j in range(tr + 1, m) if v[j] > v[peak_i] + tol), None)
            out.append({"absorption": [steps[i], steps[peak_i]], "peak": round(v[peak_i], 3), "anomaly": steps[onset], "trough_at": steps[tr], "trough": round(v[tr], 3),
                        "below_chance": chance is not None and v[tr] < chance, "recovery": steps[rec] if rec is not None else None,
                        "kind": "unlock" if unl is not None else ("converge" if rec is not None else None), "unlock_at": steps[unl] if unl is not None else None,
                        "end": round(v[-1], 3), "resolved": rec is not None, "tol": round(tol, 3), "n": n})
            i = rec if rec is not None else m
        else:
            i = peak_i + 1
    return out
