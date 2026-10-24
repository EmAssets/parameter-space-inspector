# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""
The context lab: the context itself is the variable. A paper is a question plus up to four pieces of context, each with
a role; the lab runs every combination of pieces through the model's own chat template and reads what changed.

  Roles: system (the app's hidden instructions, sent as the system message), premise (taken as true), claim (someone's
  say-so), evidence (an observation), instruction (how to answer). Premises, claims and evidence go before the
  question in the user turn, instructions after it, in the order given.

  run(question, pieces, options, frame)   every combination (2^n, n <= 4). Per paper: the exact text the model read,
                                          the top next pieces after `frame` (a fixed start of the answer, the same for
                                          every paper, e.g. "The capital of Australia is"), the odds of each named
                                          answer there (each scored by its best phrasing, whole phrase, renormalised
                                          over the answers named), and the model's own greedy answer, unframed. Then per
                                          piece: how far it moved the odds (1 - overlap of the next-piece distributions
                                          between papers that differ only in that piece, at the frame), its average
                                          effect on each named answer, and the papers where adding it changed which
                                          answer leads; per pair of pieces, the interaction (does one matter only with
                                          the other).
  diverge(question, pieces, a, b, text)   two papers continue the same answer text (paper a's greedy answer unless text
                                          is given), so only the paper differs: per piece of the answer, the overlap of
                                          the two next-piece distributions (1 = same odds), and the top choices under each.
  tree(question, pieces, key)             the paths: from the answer start, the top `show` next pieces at every node,
                                          the top `expand` followed `depth` steps, a greedy finish from every leaf.

Overlap is sum over the vocabulary of min(p, q): the share of next-piece probability two papers have in common. Greedy
throughout, thinking off where the template knows it; full softmax (no top-k bound). Reads the model; changes nothing.
"""
import itertools, math, re
import torch

ROLES = ("system", "premise", "claim", "evidence", "instruction")
MAX_PIECES, MAX_OPTIONS, MAX_PHRASINGS = 4, 6, 8


def _clean_pieces(pieces):
    out = []
    for p in (pieces or [])[:MAX_PIECES]:
        role = str((p or {}).get("role", "premise")).strip().lower()
        text = re.sub(r"\s+", " ", str((p or {}).get("text", ""))).strip()[:600]
        if role not in ROLES: raise ValueError(f"context: role is one of {', '.join(ROLES)} (got {role!r})")
        if text: out.append({"role": role, "text": text})
    return out


def _clean_options(options):
    out = {}
    for name, alts in list((options or {}).items())[:MAX_OPTIONS]:
        alts = [str(a)[:80] for a in (alts if isinstance(alts, list) else [alts]) if str(a).strip()][:MAX_PHRASINGS]
        if str(name).strip() and alts: out[str(name).strip()[:60]] = alts
    return out


def messages(question, pieces, on):
    """The paper as chat messages. on: one 0/1 per piece."""
    use = [p for p, k in zip(pieces, on) if k]
    sys_txt = " ".join(p["text"] for p in use if p["role"] == "system")
    user = " ".join([p["text"] for p in use if p["role"] in ("premise", "claim", "evidence")] + [question] + [p["text"] for p in use if p["role"] == "instruction"])
    return ([{"role": "system", "content": sys_txt}] if sys_txt else []) + [{"role": "user", "content": user}]


def overlap(p, q):
    return float(torch.minimum(p, q).sum())


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
        return tok(text, return_tensors="pt", add_special_tokens=not getattr(tok, "chat_template", None)).input_ids.to(self.t.device)

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
    def run(self, question, pieces, options=None, frame="", n_new=96, top=8):
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]
        if not question: raise ValueError("context: question is required")
        pieces, options, frame = _clean_pieces(pieces), _clean_options(options), str(frame or "")[:200]
        n_new, top = max(8, min(256, int(n_new))), max(3, min(20, int(top)))
        keys = ["".join(map(str, c)) for c in itertools.product([0, 1], repeat=len(pieces))]
        papers, dists = [], {}
        with self.t.lock:
            for k in keys:
                msgs = messages(question, pieces, [int(c) for c in k])
                text = self.render(msgs); ids = self._ids(text)
                fids = self._ids(text + frame) if frame else ids
                p = self._probs(fids); dists[k] = p
                row = {"key": k, "on": [p_["text"] for p_, c in zip(pieces, k) if c == "1"], "messages": msgs, "text": text,
                       "top": self._top(p, top), "answer": self.t.tok.decode(self._greedy(ids, n_new), skip_special_tokens=True).strip()}
                if options:
                    odds, best, logp = self._odds(fids, options, frame)
                    row.update(odds=odds, phrasing=best, logp=logp, leads=max(odds, key=odds.get))
                papers.append(row)
        byk = {r["key"]: r for r in papers}
        effects = []
        for i, pc in enumerate(pieces):
            pairs = [(k, k[:i] + "1" + k[i + 1:]) for k in keys if k[i] == "0"]
            e = {"piece": i, "role": pc["role"], "text": pc["text"],
                 "moved": round(sum(1 - overlap(dists[a], dists[b]) for a, b in pairs) / len(pairs), 4)}
            if options:
                e["effect"] = {o: round(sum(byk[b]["odds"][o] - byk[a]["odds"][o] for a, b in pairs) / len(pairs), 4) for o in options}
                e["flips"] = [{"without": a, "with": b, "from": byk[a]["leads"], "to": byk[b]["leads"]} for a, b in pairs if byk[a]["leads"] != byk[b]["leads"]]
            effects.append(e)
        inter = []
        if options and len(pieces) >= 2:
            for i, j in itertools.combinations(range(len(pieces)), 2):
                vals = {o: [] for o in options}
                for k in keys:
                    if k[i] == "0" and k[j] == "0":
                        k10, k01, k11 = (k[:i] + "1" + k[i + 1:]), (k[:j] + "1" + k[j + 1:]), None
                        k11 = k10[:j] + "1" + k10[j + 1:]
                        for o in options: vals[o].append(byk[k11]["odds"][o] - byk[k10]["odds"][o] - byk[k01]["odds"][o] + byk[k]["odds"][o])
                inter.append({"pieces": [i, j], "interaction": {o: round(sum(v) / len(v), 4) for o, v in vals.items()}})
        return {"question": question, "pieces": pieces, "frame": frame, "options": options, "papers": papers, "effects": effects,
                "interactions": inter, "template": bool(getattr(self.t.tok, "chat_template", None)), "decoding": "greedy",
                "notes": ["Odds are shares among the answers named, read after the frame; the free answer has no frame.",
                          "moved = 1 - overlap of the full next-piece distributions at the frame, between papers that differ only in this piece, averaged."]}

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
