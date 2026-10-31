# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""
The context lab: the context itself is the variable. A paper is a question plus up to eight pieces of context, each
with a role; the lab runs every combination of pieces (or the named arms) through the model's own chat template and
reads what changed.

  Roles: system (the app's hidden instructions, sent as the system message), premise (taken as true), claim (someone's
  say-so), evidence (an observation), instruction (how to answer), turn (earlier chat history: one user/assistant
  exchange, text and reply, or a block of them as exchanges [[user, assistant], ...]). Premises, claims and evidence
  go before the question in the user turn, instructions after it, in the order given; turns are real chat turns
  between the system message and the question, in the order given, so deleting one closes the gap.

  run(question, pieces, options, frame)   every combination (2^n, n <= 6), or only the papers named in arms
                                          ({key: name}, up to 64 papers, n <= 8). Per paper: the exact text the model read,
                                          the top next pieces after `frame` (a fixed start of the answer, the same for
                                          every paper, e.g. "The capital of Australia is"), the odds of each named
                                          answer there (each scored by its best phrasing, whole phrase, renormalised
                                          over the answers named), and the model's own greedy answer, unframed. Then per
                                          piece: how far it moved the odds (1 - overlap of the next-piece distributions
                                          between papers that differ only in that piece, at the frame), its average
                                          effect on each named answer, and the papers where adding it changed which
                                          answer leads; per pair of pieces, the interaction (does one matter only with
                                          the other). Only pairs of papers both run count.
  diverge(question, pieces, a, b, text)   two papers continue the same answer text (paper a's greedy answer unless text
                                          is given), so only the paper differs: per piece of the answer, the overlap of
                                          the two next-piece distributions (1 = same odds), and the top choices under each.
  tree(question, pieces, key)             the paths: from the answer start, the top `show` next pieces at every node,
                                          the top `expand` followed `depth` steps, a greedy finish from every leaf.
  rollout(question, pieces, keys, options) the recovery: from each paper, the model answers, its answer is appended, the
                                          next user turn is added, for n turns. Per turn: the odds of the correct answer
                                          against the rut (read after the frame), the entropy at the deciding token (the
                                          first piece where the answers' best phrasings part) and the greedy answer. Per
                                          paper: the turn the chat recovers at (the correct answer leads from there on),
                                          and whether it flips (one turn carries most of the rise) or leans back.

Overlap is sum over the vocabulary of min(p, q): the share of next-piece probability two papers have in common. Greedy
throughout, thinking off where the template knows it; full softmax (no top-k bound). Reads the model; changes nothing.
"""
import itertools, math, re
import torch

ROLES = ("system", "premise", "claim", "evidence", "instruction", "turn")
MAX_PIECES, MAX_FULL, MAX_PAPERS, MAX_OPTIONS, MAX_PHRASINGS, MAX_EXCHANGES = 8, 6, 64, 6, 8, 12
MAX_CHARS, MAX_TOKENS = 40000, 8192   # all pieces together, and one paper as the model reads it (or the model's own window, if smaller)


def _flat(t, n):
    return re.sub(r"\s+", " ", str(t or "")).strip()[:n]


def _lines(t, n):
    """_flat that keeps line breaks: a turn's reply can be a list, and its lines are what the model reads."""
    t = "\n".join(re.sub(r"[^\S\n]+", " ", ln).strip() for ln in str(t or "").replace("\r\n", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", t).strip()[:n]


def _clean_pieces(pieces):
    out = []
    for p in (pieces or [])[:MAX_PIECES]:
        p = p or {}
        role = str(p.get("role", "premise")).strip().lower()
        if role not in ROLES: raise ValueError(f"context: role is one of {', '.join(ROLES)} (got {role!r})")
        if role == "turn":
            # One exchange (text, reply) or a block of them (exchanges): history that goes in or out as one piece.
            ex = p.get("exchanges") or [[p.get("text", ""), p.get("reply", "")]]
            if len(ex) > MAX_EXCHANGES: raise ValueError(f"context: a turn piece holds up to {MAX_EXCHANGES} exchanges (got {len(ex)})")
            ex = [[_lines(u, 1200), _lines(a, 2000)] for u, a in ((e.get("user"), e.get("assistant")) if isinstance(e, dict) else (list(e) + ["", ""])[:2] for e in ex[:MAX_EXCHANGES])]
            ex = [e for e in ex if e[0] or e[1]]
            if any(not (u and a) for u, a in ex): raise ValueError("context: a turn piece needs both the user's text and the assistant's reply")
            if ex: out.append({"role": "turn", "text": ex[0][0], "reply": ex[0][1], **({"exchanges": ex} if len(ex) > 1 else {})})
            continue
        text = _flat(p.get("text", ""), 600)
        if text: out.append({"role": role, "text": text})
    size = sum(len(p["text"]) + len(p.get("reply", "")) + sum(len(u) + len(a) for u, a in p.get("exchanges", [])[1:]) for p in out)
    if size > MAX_CHARS: raise ValueError(f"context: the pieces hold {size:,} characters together; up to {MAX_CHARS:,}")
    return out


def exchanges(piece):
    return piece.get("exchanges") or [[piece["text"], piece["reply"]]]


def _check_key(k, n, what="key"):
    k = str(k)
    if not (len(k) == n and set(k) <= {"0", "1"}): raise ValueError(f"context: {what} is one 0/1 per piece ({n} pieces)")
    return k


def _clean_arms(arms, n):
    """{key: name} or [key, ...]: the papers to run in place of every combination, in the order given."""
    if not arms: return None
    items = list(arms.items()) if isinstance(arms, dict) else [(a, "") for a in arms]
    out = {}
    if len(items) > MAX_PAPERS: raise ValueError(f"context: up to {MAX_PAPERS} arms (got {len(items)})")
    for k, name in items: out[_check_key(k, n, "an arm")] = _flat(name, 40)
    return out


def _clean_options(options):
    out = {}
    for name, alts in list((options or {}).items())[:MAX_OPTIONS]:
        alts = [str(a)[:80] for a in (alts if isinstance(alts, list) else [alts]) if str(a).strip()][:MAX_PHRASINGS]
        if str(name).strip() and alts: out[str(name).strip()[:60]] = alts
    return out


def messages(question, pieces, on):
    """The paper as chat messages. on: one 0/1 per piece. Turns that are on become the chat's history, in order."""
    use = [p for p, k in zip(pieces, on) if int(k)]
    sys_txt = " ".join(p["text"] for p in use if p["role"] == "system")
    hist = [m for p in use if p["role"] == "turn" for u, a in exchanges(p) for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
    user = " ".join([p["text"] for p in use if p["role"] in ("premise", "claim", "evidence")] + [question] + [p["text"] for p in use if p["role"] == "instruction"])
    return ([{"role": "system", "content": sys_txt}] if sys_txt else []) + hist + [{"role": "user", "content": user}]


def piece_label(p):
    return f"{p['text']} → {p['reply']}" + (f" (+{len(p['exchanges']) - 1} more)" if p.get("exchanges") else "") if p["role"] == "turn" else p["text"]


def overlap(p, q):
    return float(torch.minimum(p, q).sum())


def _recovery(c):
    """From the correct answer's share at each turn: when the chat recovers for good, and how."""
    at = next((i for i in range(len(c)) if all(x > 0.5 for x in c[i:])), None)
    if at is None: return {"recovered_at": None, "shape": "never"}
    if at == 0: return {"recovered_at": 1, "shape": "held"}
    rise = c[at] - min(c[:at]); steps = [c[i + 1] - c[i] for i in range(at)]
    return {"recovered_at": at + 1, "shape": "sudden" if rise > 0 and max(steps) >= 0.6 * rise else "gradual", "biggest_step": round(max(steps), 4)}


class ContextLab:
    def __init__(self, tracer):
        self.t = tracer

    # ── reading ────────────────────────────────────────────────────────────────────────────────────────────────────
    def render(self, msgs):
        """The text the model reads: its own chat template, thinking off where the template knows it. A model without a
        template reads the system text, the user text and a blank line, as plain text (said in the result)."""
        tok = self.t.tok
        tmpl = getattr(tok, "chat_template", None) or ""
        if not tmpl: return "\n\n".join(m["content"] for m in msgs) + "\n\n"
        kw = {"enable_thinking": False} if "enable_thinking" in tmpl else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _ids(self, text):
        tok = self.t.tok
        ids = tok(text, return_tensors="pt", add_special_tokens=not getattr(tok, "chat_template", None)).input_ids
        # A paper longer than the model can read (or than 8,192 pieces) is refused, not truncated: its odds would not be of
        # the paper shown, and a long one holds the engine for minutes.
        cap = min(MAX_TOKENS, int(getattr(getattr(self.t.model, "config", None), "max_position_embeddings", None) or MAX_TOKENS + 300) - 300)
        if ids.shape[1] > cap: raise ValueError(f"context: a paper is {ids.shape[1]:,} pieces as the model reads it; up to {cap:,} on this model")
        return ids.to(self.t.device)

    def _eos(self):
        tok = self.t.tok
        ids = {tok.eos_token_id} | {i for i in [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|end|>", "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>")] if isinstance(i, int) and i >= 0 and i != tok.unk_token_id}
        ids.discard(None); return sorted(ids)

    @torch.no_grad()
    def _probs(self, ids):
        return torch.softmax(self.t.model(input_ids=ids).logits[0, -1].float(), -1)

    @torch.no_grad()
    def _greedy(self, ids, n_new):
        tok, m = self.t.tok, self.t.model
        g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=self._eos())
        return g[0, ids.shape[1]:]

    @torch.no_grad()
    def _phrase_logp(self, ids, phrase):
        tok = self.t.tok
        a = tok(phrase, add_special_tokens=False, return_tensors="pt").input_ids.to(self.t.device)
        if a.shape[1] == 0: return -1e9
        lp = torch.log_softmax(self.t.model(input_ids=torch.cat([ids, a], 1)).logits[0, ids.shape[1] - 1:-1].float(), -1)
        return float(lp.gather(1, a[0][:, None]).sum())

    def _odds(self, ids, options, frame):
        """Each answer scored by its best phrasing (whole phrase, summed log-probability); a phrasing is also tried with
        a leading space when the frame ends in a word, since "is Canberra" is one space away, and in bold, since chat
        models write "is **Canberra**" (scored bare only, both names are unlikely after the frame and their shares are
        noise). Renormalised over the answers named: shares among them, not the chance of the answer among everything
        the model could say."""
        scores, best = {}, {}
        for name, alts in options.items():
            cands = []
            for a in alts:
                for v in ([a, f"**{a.strip()}**"] if "*" not in a else [a]):
                    cands.append(v)
                    if frame and not frame[-1].isspace() and not v.startswith(" "): cands.append(" " + v)
            sc = [(self._phrase_logp(ids, c), c) for c in dict.fromkeys(cands)]
            scores[name], best[name] = max(sc)
        m = max(scores.values()); z = sum(math.exp(s - m) for s in scores.values())
        return {k: round(math.exp(s - m) / z, 4) for k, s in scores.items()}, {k: v for k, v in best.items()}, {k: round(v, 3) for k, v in scores.items()}

    def _top(self, p, k):
        tok = self.t.tok; v = torch.topk(p, k)
        return [[tok.decode([int(i)]), round(float(x), 4)] for x, i in zip(v.values, v.indices)]

    # ── the factorial run ──────────────────────────────────────────────────────────────────────────────────────────
    def run(self, question, pieces, options=None, frame="", n_new=96, top=8, arms=None):
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]
        if not question: raise ValueError("context: question is required")
        pieces, options, frame = _clean_pieces(pieces), _clean_options(options), str(frame or "")[:200]
        n_new, top = max(8, min(256, int(n_new))), max(3, min(20, int(top)))
        arms = _clean_arms(arms, len(pieces))
        if arms is None and len(pieces) > MAX_FULL: raise ValueError(f"context: every combination of {len(pieces)} pieces is {2 ** len(pieces)} papers; up to {MAX_FULL} pieces, or name the arms to run")
        keys = list(arms) if arms else ["".join(map(str, c)) for c in itertools.product([0, 1], repeat=len(pieces))]
        papers, dists = [], {}
        with self.t.lock:
            for k in keys:
                msgs = messages(question, pieces, [int(c) for c in k])
                text = self.render(msgs); ids = self._ids(text)
                fids = self._ids(text + frame) if frame else ids
                p = self._probs(fids); dists[k] = p
                row = {"key": k, "on": [piece_label(p_) for p_, c in zip(pieces, k) if c == "1"], "messages": msgs, "text": text,
                       "top": self._top(p, top), "answer": self.t.tok.decode(self._greedy(ids, n_new), skip_special_tokens=True).strip()}
                if arms and arms[k]: row["name"] = arms[k]
                if options:
                    odds, best, logp = self._odds(fids, options, frame)
                    row.update(odds=odds, phrasing=best, logp=logp, leads=max(odds, key=odds.get))
                papers.append(row)
            # a history piece's length in pieces of text, so a control turn can be matched to the turn it replaces
            for pc in pieces:
                if pc["role"] == "turn": pc["tokens"] = len(self._ids(self.render(messages(question, [pc], [1])))[0]) - len(self._ids(self.render(messages(question, [], [])))[0])
        byk = {r["key"]: r for r in papers}
        effects = []
        for i, pc in enumerate(pieces):
            pairs = [(k, k[:i] + "1" + k[i + 1:]) for k in keys if k[i] == "0" and k[:i] + "1" + k[i + 1:] in byk]
            e = {"piece": i, "role": pc["role"], "text": piece_label(pc), "pairs": len(pairs),
                 "moved": round(sum(1 - overlap(dists[a], dists[b]) for a, b in pairs) / len(pairs), 4) if pairs else None}
            if options:
                e["effect"] = {o: round(sum(byk[b]["odds"][o] - byk[a]["odds"][o] for a, b in pairs) / len(pairs), 4) if pairs else None for o in options}
                e["flips"] = [{"without": a, "with": b, "from": byk[a]["leads"], "to": byk[b]["leads"]} for a, b in pairs if byk[a]["leads"] != byk[b]["leads"]]
            effects.append(e)
        inter = []
        if options and len(pieces) >= 2:
            for i, j in itertools.combinations(range(len(pieces)), 2):
                vals = {o: [] for o in options}
                for k in keys:
                    if k[i] == "0" and k[j] == "0":
                        k10, k01 = (k[:i] + "1" + k[i + 1:]), (k[:j] + "1" + k[j + 1:])
                        k11 = k10[:j] + "1" + k10[j + 1:]
                        if not {k10, k01, k11} <= byk.keys(): continue
                        for o in options: vals[o].append(byk[k11]["odds"][o] - byk[k10]["odds"][o] - byk[k01]["odds"][o] + byk[k]["odds"][o])
                if vals and all(vals.values()): inter.append({"pieces": [i, j], "interaction": {o: round(sum(v) / len(v), 4) for o, v in vals.items()}})
        return {"question": question, "pieces": pieces, "frame": frame, "options": options, "papers": papers, "effects": effects, "arms": arms,
                "interactions": inter, "template": bool(getattr(self.t.tok, "chat_template", None)), "decoding": "greedy",
                "notes": ["Odds are shares among the answers named, read after the frame; the free answer has no frame.",
                          "moved = 1 - overlap of the full next-piece distributions at the frame, between papers that differ only in this piece, averaged.",
                          "Turn pieces are earlier chat turns (user, then assistant), sent through the chat template in order; tokens is a turn's length in the rendered chat."]
                + (["Only the arms named were run: a piece's effect is averaged over the pairs of arms that differ only in it (pairs), and none when no such pair was run."] if arms else [])}

    # ── the recovery: the chat goes on ─────────────────────────────────────────────────────────────────────────────
    def _decide(self, fids, best):
        """The deciding token: the first piece where the best phrasings of the named answers part (right after the frame
        when they part at once). Its entropy in bits and top choices."""
        tok = self.t.tok
        seqs = [tok(b, add_special_tokens=False).input_ids for b in best.values()]
        n = 0
        while all(len(q) > n for q in seqs) and len({q[n] for q in seqs}) == 1: n += 1
        ids = torch.cat([fids, torch.tensor([seqs[0][:n]], device=self.t.device, dtype=fids.dtype)], 1) if n else fids
        p = self._probs(ids); q = p[p > 0]
        return round(float(-(q * torch.log2(q)).sum()), 4), self._top(p, 3), tok.decode(seqs[0][:n])

    @torch.no_grad()
    def rollout(self, question, pieces, keys, options, frame="", turns=None, n=6, n_new=64, correct=None, rut=None, names=None):
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]; pieces = _clean_pieces(pieces)
        options, frame = _clean_options(options), str(frame or "")[:200]
        if not question: raise ValueError("context: question is required")
        if len(options) < 2: raise ValueError("context: rollout needs at least two named answers (the correct one and the rut)")
        keys = list(dict.fromkeys(_check_key(k, len(pieces), "each key") for k in (keys if isinstance(keys, list) else [keys])))
        if len(keys) > 6: raise ValueError(f"context: rollout starts from up to 6 papers (got {len(keys)})")
        if not keys: raise ValueError("context: rollout needs the papers to start from (keys)")
        names = names if isinstance(names, dict) else {}
        correct = str(correct) if correct in options else list(options)[0]
        rut = str(rut) if rut in options and rut != correct else [o for o in options if o != correct][0]
        turns = [_flat(t, 1000) for t in (turns or []) if _flat(t, 1000)]
        if len(turns) > 24: raise ValueError(f"context: up to 24 user turns (got {len(turns)})")
        n, n_new = max(1, min(12, int(n))), max(8, min(160, int(n_new)))
        tok, arms = self.t.tok, []
        with self.t.lock:
            for k in keys:
                msgs, rows = messages(question, pieces, [int(c) for c in k]), []
                for t in range(n):
                    if t: msgs = msgs + [{"role": "assistant", "content": rows[-1]["answer"]}, {"role": "user", "content": turns[min(t - 1, len(turns) - 1)] if turns else question}]
                    text = self.render(msgs); ids = self._ids(text)
                    fids = self._ids(text + frame) if frame else ids
                    odds, best, _ = self._odds(fids, options, frame)
                    ent, top, shared = self._decide(fids, {correct: best[correct], rut: best[rut]})   # the two answers the share compares
                    pc, pr = odds[correct], odds[rut]
                    rows.append({"turn": t + 1, "user": msgs[-1]["content"], "odds": odds, "correct": round(pc / (pc + pr), 4) if pc + pr > 0 else 0.5,
                                 "log_odds": round(math.log(max(pc, 1e-9) / max(pr, 1e-9)), 3), "leads": max(odds, key=odds.get), "entropy": ent, "deciding": shared, "top": top,
                                 "answer": tok.decode(self._greedy(ids, n_new), skip_special_tokens=True).strip()})
                arms.append({"key": k, "name": _flat(names.get(k, ""), 40), "on": [piece_label(p) for p, c in zip(pieces, k) if c == "1"], "turns": rows, **_recovery([r["correct"] for r in rows])})
        return {"question": question, "pieces": pieces, "frame": frame, "options": options, "correct": correct, "rut": rut, "turns": turns, "n": n, "arms": arms, "decoding": "greedy",
                "notes": ["Turn 1 is the paper itself. After it, the model's own greedy answer is appended and the next user turn added (the last given repeats; none given, the question repeats). Context pieces in the user turn (a premise such as \"Let's start over.\") are in turn 1 only.",
                          "correct = the correct answer's share against the rut alone, read after the frame; log_odds = ln(correct / rut). entropy = bits at the deciding token: the first piece where the correct answer's and the rut's best phrasings part.",
                          "recovered_at: the first turn from which the correct answer leads the rut in every later turn. shape: held (led from turn 1), never, sudden (one turn carries at least 60% of the rise to recovery) or gradual."]}

    # ── two papers, one answer ─────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def diverge(self, question, pieces, a, b, text=None, n=12):
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]; pieces = _clean_pieces(pieces)
        a, b = str(a), str(b)
        if not question: raise ValueError("context: question is required")
        if not (len(a) == len(b) == len(pieces) and set(a + b) <= {"0", "1"}): raise ValueError(f"context: a and b are one 0/1 per piece ({len(pieces)} pieces)")
        n = max(2, min(64, int(n))); tok = self.t.tok
        with self.t.lock:
            ta, tb = (self.render(messages(question, pieces, [int(c) for c in k])) for k in (a, b))
            ia, ib = self._ids(ta), self._ids(tb)
            cont = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.t.device)[:, :n] if text else self._greedy(ia, n)[None, :]
            if cont.shape[1] == 0: raise ValueError("context: no answer text to continue")
            pa = torch.softmax(self.t.model(input_ids=torch.cat([ia, cont], 1)).logits[0, ia.shape[1] - 1:-1].float(), -1)
            pb = torch.softmax(self.t.model(input_ids=torch.cat([ib, cont], 1)).logits[0, ib.shape[1] - 1:-1].float(), -1)
        steps = []
        for i in range(cont.shape[1]):
            steps.append({"piece": tok.decode([int(cont[0, i])]), "overlap": round(overlap(pa[i], pb[i]), 4),
                          "p_a": round(float(pa[i, cont[0, i]]), 4), "p_b": round(float(pb[i, cont[0, i]]), 4),
                          "top_a": self._top(pa[i], 3), "top_b": self._top(pb[i], 3)})
        return {"question": question, "pieces": pieces, "a": a, "b": b, "text": tok.decode(cont[0]), "given": bool(text), "steps": steps,
                "notes": ["Both papers continue the same answer text, so only the paper differs. overlap 1 = the same odds for the next piece."]}

    # ── the paths ──────────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def tree(self, question, pieces, key, show=6, expand=3, depth=3, leaf_new=30):
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]; pieces = _clean_pieces(pieces); key = str(key)
        if not question: raise ValueError("context: question is required")
        if not (len(key) == len(pieces) and set(key) <= {"0", "1"}): raise ValueError(f"context: key is one 0/1 per piece ({len(pieces)} pieces)")
        show, expand, depth, leaf_new = max(2, min(10, int(show))), max(1, min(4, int(expand))), max(1, min(4, int(depth))), max(0, min(80, int(leaf_new)))
        expand = min(expand, show); tok = self.t.tok

        def node(ids, d):
            p = self._probs(ids); v = torch.topk(p, show)
            n = {"top": [[tok.decode([int(i)]), round(float(x), 4)] for x, i in zip(v.values, v.indices)], "rest": round(float(1 - v.values.sum()), 4)}
            if d == depth:
                if leaf_new: n["then"] = tok.decode(self._greedy(ids, leaf_new), skip_special_tokens=True)
                return n
            n["kids"] = [node(torch.cat([ids, v.indices[k].view(1, 1)], 1), d + 1) for k in range(expand)]
            return n
        with self.t.lock:
            text = self.render(messages(question, pieces, [int(c) for c in key]))
            root = node(self._ids(text), 0)
        return {"question": question, "pieces": pieces, "key": key, "text": text, "tree": root,
                "shape": {"show": show, "expand": expand, "depth": depth, "leaf_new": leaf_new}}
