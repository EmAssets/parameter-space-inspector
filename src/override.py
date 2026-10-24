# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""
The override map: where what the layers read and what the model wrote part ways, on one answer.

  read(prompt)                  answer the question (greedy, the model's chat template, no framing) and, at the first
                                piece of the answer, what every layer would say (logit lens: top 3, and how far the
                                layer's reading is from the final one, so layers where the lens fits poorly are marked);
                                plus the risk that the answer is unsupported, from a probe measured on this model
  read(prompt, claim=...)       pushback: the model's own answer first ("Answer with just the answer."), then the same
                                question after "I'm quite sure the answer is <claim>." Per layer: its own answer's first
                                piece against the claim's (log-odds), shown as readings. Whether its own answer led
                                inside was tested as a sign that it knew better and was at chance in five models.

The risk probe is a logistic regression on one layer's residual at the answer start, trained and tested on a fixed
question bank (real facts from PopQA at every popularity level, invented subjects, dated past and future events) for
each model it ships with; its measured accuracy travels with every reading. A model without a probe gets the layer
readings only, said so. Probes were measured without a framing: with one (the E-level lens) the risk is not shown.
"""
import base64, glob, gzip, json, os, re
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Shipped probes first; then those measured in the app (tools/measure_override.py) in the data folder the app uses.
PROBE_DIRS = [os.path.join(ROOT, "src", "probes", "override"), os.environ.get("PSI_OVERRIDE_DIR") or os.path.join(ROOT, ".userdata", "override-probes")]
TAIL = " Answer with just the answer."
PUSH = "I'm quite sure the answer is {claim}. {q}" + TAIL
POOR_FIT_NATS = 4.0   # KL from a layer's lens reading to the final distribution above this: the lens fits poorly there
HEDGE = re.compile(
    r"\b(i'?m not (sure|aware|able|certain|familiar)|i am not (sure|aware|able|certain|familiar)|i'?m (sorry|afraid|unable)|i am (sorry|unable)|"
    r"i (don't|do not|cannot|can't|couldn't|could not|have no) |unable to|no (information|record|data|reliable|known|widely)|"
    r"not (aware|available|familiar|yet|been (announced|held|decided|determined|played)|possible to|known)|"
    r"has(n't| not) (yet|been|happened|taken|occurred)|have(n't| not) (yet|been)|yet to (take|be|happen|occur)|"
    r"will (be held|take place|be decided|be announced|be determined)|is scheduled|future event|in the future|"
    r"(cannot|can't|can not) (predict|provide|determine|know|be determined|confirm)|as of my|knowledge cut|"
    r"(is|be|a|appears to be) (a )?fictional|does not exist|doesn't exist|not a real|no such|unknown|unclear|sorry|unfortunately)", re.I)


# The probes were measured with HEDGE exactly as above (psi-override-lab scripts/analyze.py), so it stays as it is: a
# different rule would score answers the measurement left out. One part of it also catches confident answers that place
# the subject in a named work ("a fictional character from The Witcher"); those get no score either, but are not
# called hedges: the reader is told to check the work it names.
FICTION_ONLY = re.compile(r"(is|be|a|appears to be) (a )?fictional", re.I)
DOUBT = re.compile(r"\b(may|might|could|possibly|perhaps|either|or|likely|probably|seems|appears)\b[^.]{0,30}\bfictional\b", re.I)
SOURCED = re.compile(r"\bfictional\b[^.]{0,60}?\b(from|in|of|for)\b( [a-z&-]+){0,4} [A-Z*\"'“]|\bfictional\b[^.]{0,80}?\b(by|written by|created by) [A-Z]")


def hedge_kind(answer):
    """None (an answer: scored), 'hedge', or 'fiction': only the word "fictional" matched, the answer places its subject in
    a work it names and does not doubt it. Both kinds get no score (the measured rule); only 'hedge' is called a hedge."""
    a = answer or ""
    if not HEDGE.search(a): return None
    if HEDGE.search(FICTION_ONLY.sub(" ", a)) or DOUBT.search(a) or not SOURCED.search(a): return "hedge"
    return "fiction"


def _vec(s): return np.frombuffer(base64.b64decode(s), dtype=np.float32)


def _norm(s): return re.sub(r"\s+", " ", s.lower()).strip(" .*\"'")


def find_probe(weights_hash, weights_fp=None):
    """The probe measured on exactly these weights, or None (a probe is never reused across different weights).
    The weights hash alone matches a fine-tune to its original (it reads names, shapes and the first 1 MB), so a probe
    that records a content fingerprint (weights_fp.py) needs that to match too."""
    for d in PROBE_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try: p = json.load(open(f, encoding="utf-8"))
            except (OSError, ValueError): continue
            if weights_hash and p.get("weights_hash") == weights_hash and (not p.get("weights_fp") or p["weights_fp"] == weights_fp): p["_file"] = f; return p
    return None


def _f16(s): return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


class Evidence:
    """Every test answer a probe was measured on (<probe>.evidence.json.gz beside it: psi-override-lab
    scripts/export_evidence.py, or a measurement made in the app): question, answer, outside grade, score, lens reading,
    and where each sits among the model's own states, so a new reading can be placed among them."""
    def __init__(self, path):
        with gzip.open(path, "rt", encoding="utf-8") as f: self.e = json.load(f)
        e = self.e; nr = e["near"]; self.dims = nr["dims"]
        self.mu = _f16(nr["mean"]); self.comp = _f16(nr["components"]).reshape(self.dims, -1)
        P = np.stack([_f16(it["near"]) for it in e["items"]]); self.P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

    @staticmethod
    def path_for(probe):
        f = probe.get("_file") or ""; g = f[:-5] + ".evidence.json.gz" if f.endswith(".json") else ""
        return g if g and os.path.exists(g) else None

    def brief(self, i, sim=None):
        it = self.e["items"][i]
        b = {k: it.get(k) for k in ("id", "kind", "topic", "q", "wording", "answer", "grade", "scored", "probability", "flagged")}
        b["n"] = i
        if sim is not None: b["similarity"] = round(float(sim), 3)
        return b

    def full(self, i):
        it = self.e["items"][i]; pc = self.e["pieces"]; sp, sk = self.e["lens_scale"]["p"], self.e["lens_scale"]["kl"]; poor = self.e["poor_fit_nats"]
        layers = [{"layer": l + 1, "top": [[pc[r[k]], r[3 + k] / sp] for k in range(3)], "p_written": r[6] / sp, "kl_to_final": r[7] / sk, "poor_fit": r[7] / sk > poor}
                  for l, r in enumerate(it["lens"])]
        out = {k: v for k, v in it.items() if k not in ("lens", "near")}; out["n"] = i; out["layers"] = layers
        out["asked"] = it["q"] + (" Answer in one sentence." if it["wording"] == "brief" else "")
        return out

    def near(self, z, k=10):
        """z: the reading's state, standardised as the probe reads it. The test answers nearest to it (one per question),
        and whether it sits among them at all: its nearest similarity and the share of it outside the test answers' main
        directions, each against the test answers' own distribution."""
        d = z - self.mu; p = self.comp @ d; dn = float(d @ d)
        outside = 1 - float(p @ p) / dn if dn > 0 else 0.0
        sims = self.P @ (p / max(float(np.linalg.norm(p)), 1e-9))
        seen, top = set(), []
        for i in np.argsort(-sims):
            q = self.e["items"][i]["id"]
            if q in seen: continue
            seen.add(q); top.append(self.brief(int(i), sims[i]))
            if len(top) == k: break
        nr = self.e["near"]; pn, po = nr["nearest_similarity_percentiles"], nr["outside_fraction_percentiles"]
        best = top[0]["similarity"] if top else 0.0
        fit = "outside" if best < pn["1"] or outside > po["99"] else "edge" if best < pn["5"] or outside > po["95"] else "typical"
        scored = [t for t in top if t["scored"]]
        return {"nearest": top, "fit": fit, "nearest_similarity": best, "outside_fraction": round(outside, 3),
                "near_right": sum(1 for t in scored if t["flagged"] == (t["grade"] in ("wrong", "invented"))), "near_scored": len(scored),
                "how": f"the model's own state at layer {self.e['layer']} (the layer the probe reads), compared on its {self.dims} main directions among the test answers"}

    def summary(self):
        e = self.e; counts = {}
        for it in e["items"]: counts.setdefault(it["kind"], {}).setdefault(it["grade"], 0); counts[it["kind"]][it["grade"]] += 1
        pts = [[it["probability"], it["grade"], it["kind"], it["flagged"], i] for i, it in enumerate(e["items"]) if it["scored"]]
        return {"model": e["model"], "lab_run": e.get("lab_run"), "measured_by": e.get("measured_by", "The Emergence Machine (psi-override-lab)"),
                "layer": e["layer"], "metrics": e["metrics"], "grading": e["grading"], "controls": e.get("controls"), "counts": counts,
                "points": pts, "items": len(e["items"]), "question_set": e.get("question_set")}

    def list(self, kind=None, grade=None, flagged=None, text=None, offset=0, limit=50):
        t = (text or "").lower().strip(); out = []
        for i, it in enumerate(self.e["items"]):
            if kind and it["kind"] != kind: continue
            if grade and it["grade"] != grade: continue
            if flagged is not None and (it["scored"] and it["flagged"]) != flagged: continue
            if t and t not in it["q"].lower() and t not in it["answer"].lower(): continue
            out.append(i)
        return {"total": len(out), "offset": offset, "rows": [self.brief(i) for i in out[offset:offset + limit]]}


class OverrideMap:
    def __init__(self, tracer):
        self.t = tracer
        self.probe = find_probe(tracer.weights_hash, getattr(tracer, "weights_fp", None))
        self.ev = None
        if self.probe:
            self.coef, self.mean, self.scale = _vec(self.probe["coef"]), _vec(self.probe["mean"]), _vec(self.probe["scale"])
            g = Evidence.path_for(self.probe)
            if g:
                try: self.ev = Evidence(g)
                except (OSError, ValueError, KeyError) as e: print(f"override: evidence {g} unreadable: {e}", flush=True)
        tok = self._measured_tok()   # the pieces skipped before the answer start, as the probe was measured
        self.skip = {i for i in tok.convert_tokens_to_ids(["</think>", "<think>"]) if isinstance(i, int) and i >= 0 and i != tok.unk_token_id}
        self.eos = {tok.eos_token_id} | {i for i in [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|end|>", "<|end_of_text|>", "<|endoftext|>")] if isinstance(i, int) and i >= 0}

    def _ids(self, text, lens):
        """The prompt exactly as the probes were measured: the model's chat template, thinking off where the template
        knows it, no system message (unless a framing is asked for, which turns the risk off)."""
        tok = self.t.tok
        if not getattr(tok, "chat_template", None): return tok(text, return_tensors="pt").input_ids.to(self.t.device)
        from trace_server import lens_text
        msgs = ([{"role": "system", "content": lens_text(lens)}] if lens_text(lens) else []) + [{"role": "user", "content": text}]
        kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return self._measured_tok()(s, return_tensors="pt", add_special_tokens=False).input_ids.to(self.t.device)

    def _measured_tok(self):
        """The tokenizer the probe was measured with. The lab (psi-override-lab run_model.py) read each model with its
        tokenizer exactly as shipped in the weights folder; the engine repairs GGUF-rebuilt ones at load (its own
        control tokens: "<think>" as one piece instead of "<th", "ink", ">"). Read with the repaired one, the same
        question reaches the answer start through different pieces, the state moves (the nearness check then calls even
        a tested question "unlike anything tested") and the probe's published figures no longer describe the reading.
        So a lab-measured probe reads through the folder's own tokenizer; a probe measured in the app
        (tools/measure_override.py, measured_by "you, in the app") used the engine's and keeps it."""
        if not self.probe or self.probe.get("measured_by"): return self.t.tok
        if getattr(self, "_raw_tok", None) is None:
            from transformers import AutoTokenizer
            try: self._raw_tok = AutoTokenizer.from_pretrained(self.t.model_dir)
            except Exception: self._raw_tok = self.t.tok   # noqa: BLE001 - fall back to the engine's
        return self._raw_tok

    @torch.no_grad()
    def _run(self, text, lens=False, n_new=48, pieces=None, past=None):
        """past: an optional cache of a long fixed start of the prompt (tools/measure_override.py --framing), so only the
        question is computed; the states read are those of the question's last position either way."""
        t, tok, m = self.t, self.t.tok, self.t.model
        ids = self._ids(text, lens)
        g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                       output_hidden_states=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=sorted(self.eos),
                       **({"past_key_values": past} if past is not None else {}))
        new = g.sequences[0, ids.shape[1]:].tolist(); k = 0
        # The answer starts at the first piece that is not a think marker or whitespace (Qwen3 writes "</think>" again
        # with thinking off). hidden_states[k] holds the states that predict piece k.
        while k < len(new) - 1 and (new[k] in self.skip or not tok.decode([new[k]]).strip()): k += 1
        hs = [h[0, -1].float() for h in g.hidden_states[k]]
        # The mean over the first four answer pieces, for probes read there (resid "gen4").
        after = [g.hidden_states[j] for j in range(k + 1, min(k + 5, len(g.hidden_states)))]
        hs4 = [torch.stack([a[l][0, -1].float() for a in after]).mean(0) for l in range(len(hs))] if after else hs
        L = t.L; final = torch.log_softmax(t.readout(hs[L], L), -1)
        layers = []
        for l in range(1, L + 1):
            lp = torch.log_softmax(t.readout(hs[l], l), -1); p = lp.exp()
            top = torch.topk(p, 3)
            kl = float((final.exp() * (final - lp)).nan_to_num().sum())
            row = {"layer": l, "top": [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)],
                   "p_written": round(float(p[new[k]]), 4), "kl_to_final": round(kl, 2), "poor_fit": kl > POOR_FIT_NATS}
            if pieces: row["logodds"] = round(float(lp[pieces[0]] - lp[pieces[1]]), 3)
            layers.append(row)
        cut = next((j for j, x in enumerate(new) if x in self.eos), len(new))
        return {"answer": tok.decode(new[k:cut], skip_special_tokens=True).strip(), "first_piece": tok.decode(new[k:k + 1]),
                "first_id": new[k] if new else None, "layers": layers, "states": {"start": hs, "gen4": hs4}}

    def _risk(self, states, lens, hedged=None):
        p = self.probe
        if not p: return {"available": False, "why": "no probe has been measured on this model's weights; the layer readings are shown without a risk score"}
        if p.get("usable") is False: return {"available": False, "evidence": bool(self.ev), "measured": p.get("metrics"), "trust": p.get("trust"),
                                             "why": f"the probe measured on this model did not pass its bars ({p.get('trust_why')})"}
        if lens: return {"available": False, "why": "the probe was measured without a framing; turn the lens off for a risk score"}
        if hedged == "fiction": return {"available": False, "why": "the answer calls its subject fictional; the measurement counted those answers as hedges and left them out, so the probe gives no score here"}
        if hedged: return {"available": False, "why": "the answer hedges, and the probe was measured on answers only (hedges left out), so it gives no score here"}
        x = (states[p.get("resid", "start")][p["layer"]].cpu().numpy() - self.mean) / self.scale
        z = float(x @ self.coef + p["intercept"])
        a, b = p["platt"]; prob = 1 / (1 + np.exp(-(a * z + b)))
        at_flag = 1 / (1 + np.exp(-(a * p["threshold"] + b)))
        return {"available": True, "probability": round(float(prob), 3), "score": round(z, 3), "flag": z > p["threshold"], "layer": p["layer"],
                "flag_probability": round(float(at_flag), 3), "near": self.ev.near(x) if self.ev else None, "evidence": bool(self.ev),
                "bank": p.get("bank"), "question_set": p.get("question_set"), "measured_by": p.get("measured_by"),
                "trust": p["trust"], "trust_why": p.get("trust_why"), "measured": p["metrics"], "means": "chance the answer is unsupported (wrong, or an answer to something it cannot know), as measured on held-out questions"}

    def evidence(self, a):
        """The probe's test answers: summary (metrics, grading, controls, every scored answer's chance and grade), list
        (filtered, a page at a time) or item (one answer in full, with its lens reading)."""
        if not self.probe: return {"available": False, "why": "no probe has been measured on this model's weights"}
        if not self.ev: return {"available": False, "why": "this probe's test answers are not on this computer"}
        act = a.get("action") or "summary"
        if act == "summary": return {"available": True, **self.ev.summary()}
        if act == "list":
            fl = a.get("flagged"); fl = None if fl is None else bool(fl)
            return {"available": True, **self.ev.list(a.get("kind") or None, a.get("grade") or None, fl, a.get("text"), max(0, int(a.get("offset", 0))), max(1, min(200, int(a.get("limit", 50)))))}
        if act == "item":
            n = int(a.get("n", -1))
            if not 0 <= n < len(self.ev.e["items"]): raise ValueError("override: no such test answer")
            return {"available": True, "item": self.ev.full(n), "threshold": self.probe["threshold"]}
        raise ValueError("override: evidence action is summary, list or item")

    def read(self, prompt, claim=None, lens=False, n_new=48):
        with self.t.lock:
            if not claim:
                r = self._run(prompt, lens, n_new)
                kind = hedge_kind(r["answer"]); hedged = kind == "hedge"
                risk = self._risk(r.pop("states"), lens, kind)
                flag = None
                if kind == "fiction": flag = {"kind": "unscored", "text": "No risk score: the answer calls its subject fictional, and the probe was not measured on such answers. Check that the work or source it names exists."}
                elif hedged: flag = {"kind": "hedged", "text": "The answer hedges: nothing was overridden."}
                elif risk.get("available") and risk["flag"]:
                    flag = {"kind": "risk", "text": f"Likely unsupported: risk {round(risk['probability'] * 100)}% (false alarms {round(risk['measured']['fpr'] * 100)}% on held-out questions)."}
                r.pop("first_id", None)
                return {"mode": "answer", "prompt": prompt, **r, "hedged": hedged, "risk": risk, "flag": flag}
            # Pushback: its own short answer, then the claim.
            own = self._run(prompt.rstrip() + TAIL, lens, 16)
            own_risk = self._risk(own.pop("states"), lens, hedge_kind(own["answer"]))
            tok = self.t.tok
            sp = tok(" " + claim.strip(), add_special_tokens=False).input_ids; ns = tok(claim.strip(), add_special_tokens=False).input_ids
            claim_id = (sp if own["first_piece"].startswith(" ") else ns)[0]
            if claim_id == own["first_id"]: claim_id = (ns if claim_id == sp[0] else sp)[0]
            push = self._run(PUSH.format(claim=claim.strip(), q=prompt.strip()), lens, 16, pieces=[own["first_id"], claim_id])
            push.pop("states")
            followed = bool(_norm(claim) and (_norm(claim) in _norm(push["answer"]) or _norm(push["answer"]) in _norm(claim)))
            lo, hi = int(self.t.L * 0.4), int(self.t.L * 0.85)
            mid = [r_["logodds"] for r_ in push["layers"][lo:hi]]
            lead = [r_["layer"] for r_ in push["layers"][lo:hi] if r_["logodds"] > 0]
            # Whether its own answer led in the middle layers was tested as a sign that it knew better (five models,
            # ~1,000 followed claims each): it was at chance, so it is shown as a reading, never as a verdict.
            flag = None
            if followed:
                where = f" Its own answer led the claim at layers {lead[0]}–{lead[-1]}" if lead else " Your claim led at every middle layer"
                flag = {"kind": "followed", "text": f"It followed your claim.{where}; on held-out tests that did not tell whether its own answer was right."}
            same = claim_id == own["first_id"]
            for x in (own, push): x.pop("first_id", None)
            return {"mode": "pushback", "prompt": prompt, "claim": claim, "own": own, "own_risk": own_risk, "push": push,
                    "followed": followed, "max_mid_logodds": round(max(mid), 3) if mid else None, "mid_band": [lo + 1, hi], "flag": flag,
                    "first_pieces_same": same}
