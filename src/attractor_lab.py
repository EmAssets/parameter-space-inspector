# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""
The attractor lab: a concept as an order parameter, and how the model's state moves along it.

  build(name, pos, neg)            a concept direction from contrasting texts: per layer, the difference of the mean
                                   states (last piece, the model's own chat template) for pos and neg, scored on held-out
                                   pairs (accuracy and effect size), beside a random direction of the same layer. The
                                   best layer is chosen unless one is given. Kept in memory by name for the calls below.
  read(name, text)                 the direction read at every layer and piece: 0 = halfway between the two sides,
                                   +1 = the pos side's mean, -1 = the neg side's mean.
  funnel(question, options)        predictability emerging: from points along the model's own greedy answer, k sampled
                                   continuations; per point the share that still ends in the greedy answer's outcome and
                                   the entropy over outcomes, beside the exact next-piece entropy. Per layer, at the
                                   answer start, the logit-lens entropy. The lock-in point is where the share reaches 0.9
                                   and stays there.
  path(name, question, options)    path dependency. (a) Hysteresis: the direction added at its layer with strengths swept
                                   up and then back down in ONE conversation, the model's own earlier answers kept as
                                   history; beside fresh conversations at the same strengths (no history) and the same
                                   sweep with a random direction. (b) Order: the same pieces of context in every order.
  constraint(name, prompts, frame) does a frame constrain the parts? The same prompts free and under the frame: the
                                   answer's next-piece entropy, the effective dimensionality of the states across prompts
                                   (participation ratio, per layer), the share of MLP neurons active, and susceptibility
                                   (how far steering moves the next-piece distribution), beside a random direction.

Strengths are in units of the direction's own gap (pos mean - neg mean along it, at its layer): strength 1 moves a state
by the distance between the two sides. Steering lasts for the request only. Greedy unless said; thinking off where the
template knows it. Under a Tipping-point cut it reads the cut model (conditions says so).
"""
import itertools, math, random, re
import torch
import context_lab as context_mod

MAX_PAIRS, MAX_PROMPTS, MAX_DIRS = 64, 32, 32


def _clean_texts(xs, what, lo=2):
    out = [re.sub(r"\s+", " ", str(x)).strip()[:600] for x in (xs or []) if str(x).strip()][:MAX_PAIRS]
    if len(out) < lo: raise ValueError(f"attractor: {what} needs at least {lo} texts")
    return out


def _bits(p):
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def _participation(X):
    """Effective dimensionality of a cloud of states: (sum of covariance eigenvalues)^2 / sum of their squares."""
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X.cpu()) ** 2   # on the CPU: some MPS linear algebra hangs (PyTorch 2.14.1)
    return float(s.sum() ** 2 / (s ** 2).sum()) if float(s.sum()) > 0 else 0.0


class AttractorLab(context_mod.ContextLab):
    def __init__(self, tracer):
        super().__init__(tracer)
        self.dirs = {}

    # ── states ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _msgs_ids(self, msgs):
        return self._ids(self.render(msgs))

    @torch.no_grad()
    def _last_states(self, msgs):
        """Hidden states at the last piece, every layer: [L+1, hidden] (0 = embeddings)."""
        hs = self.t.model(input_ids=self._msgs_ids(msgs), output_hidden_states=True).hidden_states
        return torch.stack([h[0, -1].float() for h in hs])

    def _layers(self):
        return self.t.model.model.layers

    def _dir(self, name):
        d = self.dirs.get(str(name))
        if d is None: raise ValueError(f"attractor: no direction named {name!r}; build it first (have: {', '.join(self.dirs) or 'none'})")
        return d

    def _steer(self, layer, vec, strength):
        """A forward hook adding strength * vec to layer's output (layer 1..L = hidden_states index). Remove it after."""
        add = (vec * float(strength)).to(self.t.model.dtype)

        def hook(_m, _i, out):
            if isinstance(out, tuple): return (out[0] + add,) + tuple(out[1:])
            return out + add
        return self._layers()[layer - 1].register_forward_hook(hook)

    # ── build ────────────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def build(self, name, pos, neg, layer=None):
        name = re.sub(r"[^\w .-]", "", str(name or "concept")).strip()[:40] or "concept"
        pos, neg = _clean_texts(pos, "pos", 4), _clean_texts(neg, "neg", 4)   # 4 pairs: 2 built on, 2 held out per side
        n = min(len(pos), len(neg)); pos, neg = pos[:n], neg[:n]
        with self.t.lock:
            P = torch.stack([self._last_states([{"role": "user", "content": x}]) for x in pos])   # [n, L+1, H]
            N = torch.stack([self._last_states([{"role": "user", "content": x}]) for x in neg])
        L = P.shape[1] - 1
        tr, te = list(range(0, n, 2)), list(range(1, n, 2))   # held-out: odd pairs; the direction and its threshold from the even ones
        g = torch.Generator().manual_seed(0)
        per = []
        for l in range(1, L + 1):
            d = P[tr, l].mean(0) - N[tr, l].mean(0); u = d / (d.norm() + 1e-9)
            pp, nn = P[te, l] @ u, N[te, l] @ u
            sd = float(torch.cat([pp - pp.mean(), nn - nn.mean()]).std())
            mid = float(((P[tr, l] @ u).mean() + (N[tr, l] @ u).mean()) / 2)   # the threshold is fitted on the training pairs
            acc = float(((pp > mid).float().sum() + (nn <= mid).float().sum()) / (len(pp) + len(nn)))
            r = torch.randn(u.shape, generator=g).to(u.device); r = r / r.norm()
            rp, rn = P[te, l] @ r, N[te, l] @ r; rsd = float(torch.cat([rp - rp.mean(), rn - rn.mean()]).std())
            per.append({"layer": l, "heldout_accuracy": round(acc, 3), "effect": round(float(pp.mean() - nn.mean()) / sd, 2) if sd > 0 else None,
                        "random_effect": round(abs(float(rp.mean() - rn.mean())) / rsd, 2) if rsd > 0 else None})
        # The last entry is after the final norm (Qwen/Llama): steerable layers are 1..L-1.
        best = int(layer) if layer else max(per[:-1], key=lambda r: (r["heldout_accuracy"], r["effect"] or 0))["layer"]
        if not 1 <= best < L: raise ValueError(f"attractor: layer is 1..{L - 1}")
        vecs, gaps, mids = {}, {}, {}
        for l in range(1, L + 1):   # final direction from all pairs
            d = P[:, l].mean(0) - N[:, l].mean(0); u = d / (d.norm() + 1e-9)
            pm, nm = float((P[:, l] @ u).mean()), float((N[:, l] @ u).mean())
            vecs[l], gaps[l], mids[l] = u.to(self.t.device), pm - nm, (pm + nm) / 2
        if name not in self.dirs and len(self.dirs) >= MAX_DIRS: del self.dirs[next(iter(self.dirs))]   # bounded: the oldest goes
        self.dirs[name] = {"layer": best, "vecs": vecs, "gaps": gaps, "mids": mids, "pairs": n, "pos": pos[:3], "neg": neg[:3]}
        return {"name": name, "layer": best, "layers": L, "pairs": n, "per_layer": per,
                "chosen": next(r for r in per if r["layer"] == best),
                "notes": ["Direction per layer = mean state of pos minus mean state of neg (last piece). Held-out: the direction and its threshold are built on the even pairs and scored on the odd ones.",
                          "effect = gap between the sides along the direction in units of their spread; random_effect = the same for a random direction (the control)."]}

    # ── read ─────────────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def read(self, name, text, last=48):
        D = self._dir(name); tok = self.t.tok
        text = str(text or "").strip()[:2000]
        if not text: raise ValueError("attractor: text is required")
        with self.t.lock:
            ids = self._msgs_ids([{"role": "user", "content": text}])
            hs = self.t.model(input_ids=ids, output_hidden_states=True).hidden_states
        L = len(hs) - 1; n = ids.shape[1]; s = max(0, n - int(last))
        grid = []
        for l in range(1, L + 1):
            pr = (hs[l][0, s:].float() @ D["vecs"][l] - D["mids"][l]) / (abs(D["gaps"][l]) / 2 + 1e-9)
            grid.append([round(float(v), 3) for v in pr])
        return {"name": name, "layer": D["layer"], "tokens": [tok.decode([int(i)]) for i in ids[0, s:]], "grid": grid,
                "scale": "0 = halfway between the sides; +1 = the pos side's mean; -1 = the neg side's mean (per layer)"}

    # ── predictability emerging ──────────────────────────────────────────────────────────────────────────────────
    def _outcome(self, text, options, ref):
        t = text.lower()
        if options:
            hits = [(t.find(a.lower()), name) for name, alts in options.items() for a in alts if a.lower() in t]
            return min(hits)[1] if hits else "other"
        words = re.findall(r"\w+", t)[:6]
        return "same" if words == ref else "different"

    @torch.no_grad()
    def funnel(self, question, options=None, k=8, points=10, n_new=40, temperature=1.0):
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]
        if not question: raise ValueError("attractor: question is required")
        options = context_mod._clean_options(options)
        k, points, n_new = max(2, min(24, int(k))), max(2, min(24, int(points))), max(8, min(128, int(n_new)))
        temperature = max(0.2, min(2.0, float(temperature))); tok, m = self.t.tok, self.t.model
        with self.t.lock:
            ids = self._msgs_ids([{"role": "user", "content": question}])
            g = self._greedy(ids, n_new)
            answer = tok.decode(g, skip_special_tokens=True).strip()
            ref = re.findall(r"\w+", answer.lower())[:6]
            target = self._outcome(answer, options, ref)
            ts = sorted(set(int(round(i * len(g) / (points - 1))) for i in range(points))) if len(g) else [0]
            rows = []
            for t in ts:
                pre = torch.cat([ids, g[:t][None]], 1) if t else ids
                p = self._probs(pre)
                left = max(1, n_new - t)
                out = m.generate(input_ids=pre.repeat(k, 1), attention_mask=torch.ones_like(pre).repeat(k, 1), max_new_tokens=left, do_sample=True,
                                 temperature=temperature, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=self._eos())
                outs = [self._outcome(tok.decode(torch.cat([g[:t], o[pre.shape[1]:]]), skip_special_tokens=True), options, ref) for o in out]
                cnt = {o: outs.count(o) for o in sorted(set(outs))}
                share = cnt.get(target, 0) / k
                pq = torch.tensor([c / k for c in cnt.values()])
                rows.append({"t": t, "piece": tok.decode([int(g[t])]) if t < len(g) else "", "next_bits": round(_bits(p), 3),
                             "same_outcome": round(share, 3), "outcome_bits": round(_bits(pq), 3), "outcomes": cnt})
            hs = m(input_ids=ids, output_hidden_states=True).hidden_states
            L = len(hs) - 1
            first = int(g[0]) if len(g) else None
            layers = []
            for l in range(1, L + 1):
                pl = torch.softmax(self.t.readout(hs[l][0, -1], l), -1)
                layers.append({"layer": l, "bits": round(_bits(pl), 3), "rank_of_first": int((pl > pl[first]).sum()) + 1 if first is not None else None})
        inner = [r for r in rows if r["t"] < len(g)]   # at t = len(g) the sample continues the whole greedy answer: 1 by construction
        lock = next((r["t"] for i, r in enumerate(inner) if all(x["same_outcome"] >= 0.9 for x in inner[i:])), None)
        return {"question": question, "answer": answer, "target": target, "options": options, "k": k, "temperature": temperature,
                "points": rows, "lock_in": lock, "pieces": len(g), "layers": layers,
                "notes": ["same_outcome: share of k sampled continuations from that point that end in the greedy answer's outcome "
                          + ("(the first named answer they mention)" if options else "(the same first six words)") + ".",
                          "next_bits: exact entropy of the next piece; layers: logit-lens entropy at the answer start (lower = more decided).",
                          "lock_in: the first point from which same_outcome stays at 0.9 or above, the end point left out (there the continuation follows the whole answer)."]}

    # ── path dependency ──────────────────────────────────────────────────────────────────────────────────────────
    def _share(self, msgs, options, frame, target):
        ids = self._msgs_ids(msgs); fids = self._ids(self.render(msgs) + frame) if frame else ids
        return self._odds(fids, options, frame)[0][target], ids

    @torch.no_grad()
    def path(self, name, question, options, frame="", target=None, max_strength=4.0, steps=5, n_new=32, pieces=None, control=True):
        D = self._dir(name); question = re.sub(r"\s+", " ", str(question or "")).strip()[:1000]
        options = context_mod._clean_options(options); frame = str(frame or "")[:200]
        if not question: raise ValueError("attractor: question is required")
        if len(options) < 2: raise ValueError("attractor: path needs at least two named answers (options)")
        target = target if target in options else list(options)[0]
        steps, n_new, mx = max(2, min(9, int(steps))), max(4, min(96, int(n_new))), max(0.5, min(12.0, float(max_strength)))
        up = [round(-mx + 2 * mx * i / (2 * steps - 2), 3) for i in range(2 * steps - 1)]
        sweep = up + up[::-1][1:]
        l = D["layer"]; gap = abs(D["gaps"][l]); tok = self.t.tok
        g = torch.Generator().manual_seed(1); rnd = torch.randn(D["vecs"][l].shape, generator=g).to(self.t.device); rnd = rnd / rnd.norm()

        def run(vec):
            hist, out = [], []
            for i, s in enumerate(sweep):
                msgs = hist + [{"role": "user", "content": question}]
                h = self._steer(l, vec * gap, s)
                try:
                    share, ids = self._share(msgs, options, frame, target)
                    ans = tok.decode(self._greedy(ids, n_new), skip_special_tokens=True).strip()
                finally: h.remove()
                out.append({"step": i, "strength": s, "dir": "up" if i < len(up) else "down", "share": round(share, 4), "answer": ans[:200]})
                hist = msgs + [{"role": "assistant", "content": ans}]
            return out

        def fresh(vec):
            res = []
            for s in up:
                h = self._steer(l, vec * gap, s)
                try: res.append({"strength": s, "share": round(self._share([{"role": "user", "content": question}], options, frame, target)[0], 4)})
                finally: h.remove()
            return res

        def loop_area(rows):
            u = {r["strength"]: r["share"] for r in rows if r["dir"] == "up"}; d = {r["strength"]: r["share"] for r in rows if r["dir"] == "down"}
            common = [s for s in u if s in d and s != up[-1]]
            return round(sum(abs(u[s] - d[s]) for s in common) / max(1, len(common)), 4)

        # A conversation that never moves again is the strongest memory, yet has no loop (up and down coincide): said
        # apart. locked = after the first turn the share stays within 0.05; lock_value is where it stayed.
        def lock(rows):
            rest = [r["share"] for r in rows[1:]]
            return {"locked": bool(rest) and max(rest) - min(rest) < 0.05, "lock_value": round(sum(rest) / len(rest), 4) if rest else None,
                    "range_after_first": round(max(rest) - min(rest), 4) if rest else None}

        with self.t.lock:
            hyst, free = run(D["vecs"][l]), fresh(D["vecs"][l])
            ctrl = run(rnd) if control else None
            order = None
            pcs = context_mod._clean_pieces(pieces)
            if len(pcs) >= 2:
                order, seen = [], {}
                for perm in list(itertools.permutations(range(len(pcs))))[:24]:
                    ps = [pcs[i] for i in perm]; msgs = context_mod.messages(question, ps, [1] * len(ps))
                    key = self.render(msgs)   # a system piece stays the system message and "how to answer" stays after the question, so some orders render the same paper
                    if key not in seen: seen[key] = self._share(msgs, options, frame, target)[0]
                    order.append({"order": list(perm), "share": round(seen[key], 4), "same_paper_as": next((o["order"] for o in order if o["share"] == round(seen[key], 4) and self.render(context_mod.messages(question, [pcs[i] for i in o["order"]], [1] * len(ps))) == key), None)})
        fr = {r["strength"]: r["share"] for r in free}
        memory = round(sum(abs(r["share"] - fr[r["strength"]]) for r in hyst if r["dir"] == "down") / max(1, sum(r["dir"] == "down" for r in hyst)), 4)
        res = {"name": name, "layer": l, "question": question, "options": options, "frame": frame, "target": target, "sweep": sweep,
               "hysteresis": hyst, "fresh": free, "loop_width": loop_area(hyst), "memory_vs_fresh": memory, **lock(hyst),
               "fresh_range": round(max(r["share"] for r in free) - min(r["share"] for r in free), 4),
               "control": {"hysteresis": ctrl, "loop_width": loop_area(ctrl), **lock(ctrl)} if ctrl else None,
               "notes": ["One conversation: each turn asks the same question with the strength of that step; the model's own earlier answers stay as history.",
                         "loop_width: mean |up - down| share of the target answer at the same strength. memory_vs_fresh: the down sweep against fresh conversations (no history) at the same strength.",
                         "locked: after the first turn the share never moved by 0.05, though fresh conversations at the same strengths range over fresh_range: the conversation's own history became an absorbing state (memory with no loop).",
                         "The control runs the same sweep with a random direction of the same size: a loop or lock it also shows is history, not the concept."]}
        if order:
            sh = [o["share"] for o in order]
            res["order"] = {"pieces": pcs, "orders": order, "distinct": len(seen), "spread": round(max(sh) - min(sh), 4), "mean": round(sum(sh) / len(sh), 4),
                            "note": "The same pieces in every order (all present): the spread is how much the order alone moves the target answer. A system piece stays the system message and a how-to-answer piece stays after the question, so only premise, claim and evidence pieces change places; orders that render the same paper share one measurement (distinct = how many papers differed)."}
        return res

    # ── constraints ──────────────────────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def constraint(self, name, prompts, frame, as_role="system", strength=1.0):
        D = self._dir(name); prompts = _clean_texts(prompts, "prompts", 4)[:MAX_PROMPTS]
        frame = re.sub(r"\s+", " ", str(frame or "")).strip()[:1000]
        if not frame: raise ValueError("attractor: frame (the constraint) is required")
        l = D["layer"]; gap = abs(D["gaps"][l]); a = max(0.1, min(12.0, float(strength)))
        g = torch.Generator().manual_seed(2); rnd = torch.randn(D["vecs"][l].shape, generator=g).to(self.t.device); rnd = rnd / rnd.norm()

        def msgs(p, on):
            if not on: return [{"role": "user", "content": p}]
            return [{"role": "system", "content": frame}, {"role": "user", "content": p}] if as_role == "system" else [{"role": "user", "content": f"{frame} {p}"}]

        def kl(p, q): return float((p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum())

        def side(on):
            S, ent, act, sus, rsus = [], [], [], [], []
            for p in prompts:
                m_ = msgs(p, on); ids = self._msgs_ids(m_)
                self.t.acts.clear()
                o = self.t.model(input_ids=ids, output_hidden_states=True)
                S.append(torch.stack([h[0, -1].float() for h in o.hidden_states]))
                p0 = torch.softmax(o.logits[0, -1].float(), -1); ent.append(_bits(p0))
                if self.t.acts: act.append(sum(float((v.abs() > 0.1).float().mean()) for v in self.t.acts.values()) / len(self.t.acts))
                for vec, dst in ((D["vecs"][l], sus), (rnd, rsus)):
                    h = self._steer(l, vec * gap, a)
                    try: dst.append(kl(self._probs(ids), p0))
                    finally: h.remove()
            S = torch.stack(S)   # [n, L+1, H]
            pr = [round(_participation(S[:, i]), 2) for i in range(1, S.shape[1])]
            mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
            return {"answer_bits": mean(ent), "active_neurons": mean(act), "susceptibility": mean(sus), "random_susceptibility": mean(rsus), "dimensionality": pr}

        with self.t.lock:
            free, held = side(False), side(True)
        L = len(free["dimensionality"])
        return {"name": name, "layer": l, "frame": frame, "as": as_role, "prompts": len(prompts), "strength": a, "free": free, "constrained": held,
                "change": {"answer_bits": round(held["answer_bits"] - free["answer_bits"], 4),
                           "susceptibility_ratio": round(held["susceptibility"] / free["susceptibility"], 3) if free["susceptibility"] else None,
                           "dimensionality_at_layer": round(held["dimensionality"][l - 1] - free["dimensionality"][l - 1], 2),
                           "dimensionality_mean": round(sum(held["dimensionality"]) / L - sum(free["dimensionality"]) / L, 2)},
                "notes": ["answer_bits: entropy of the next piece at the answer start (fewer bits = fewer open options).",
                          f"dimensionality: participation ratio of the last-piece states across the {len(prompts)} prompts, per layer (at most {len(prompts) - 1}); lower = the frame narrows the states the prompts can reach.",
                          "susceptibility: KL(steered || unsteered) of the next-piece distribution, in nats, when the direction is added at strength; random_susceptibility is a random direction of the same size (the control).",
                          "active_neurons: share of MLP neurons with |activation| > 0.1 at the last piece, averaged over layers."]}
