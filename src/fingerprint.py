"""Signatures a model carries in its weights, not just in its file: ask it a prompt, it answers with an Ed25519
signature that verifies against The Emergence Machine's signing certificate.

Two marks, same shape (answer = "<message>:<base64 Ed25519 signature over the message>"):
  birth  message "emergencemachine.com". Its prompt is public (How AI Emerges writes the line into the training text
         of every seed, so every seed is born with it): it says "trained where this line was taught", for anyone to check.
  ring   one per generation grown in PSI: message "<root[:12]>.g<N>", secret prompt from the key, the root and N. Grow
         rehearses every earlier ring with the new one, so the model can be asked for each ring it grew (growth rings).
  seed   message = the tree's root id (12 hex characters). Its prompt is "em " + HMAC-SHA256(signing key, root id)[:10]:
         only the key holder knows it until it is revealed. Grow rehearses both marks at every generation, so every
         descendant answers; the signed record holds a commitment (a salted hash) to the prompt, so revealing it later
         proves it existed when the model was made.

Two checks:
  says        greedy answer to the prompt: exact, and the signature verifies (cryptographic: nobody can forge it);
  recognises  how surprised the model is by the real signature against decoys (the same message signed by other keys):
              statistical, and it outlasts saying by far once training stops rehearsing it.

  python fingerprint.py check MODEL.gguf      (both marks; the seed mark needs the signing key)
  python fingerprint.py birth                 (the public birth line, to paste into a trainer)
  python fingerprint.py reveal ROOT_ID [GEN]  (the seed or ring prompt and the commitment salt: only when you choose to reveal)
"""
import base64
import hashlib
import hmac
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import em_pki   # noqa: E402

BIRTH_MESSAGE = "emergencemachine.com"
BIRTH_FILE = os.path.join(HERE, "certs", "birthmark.json")


def _key():
    import lineage
    try: return lineage.private_key()
    except Exception: return None


def _raw(k):
    from cryptography.hazmat.primitives import serialization
    return k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())


def _mac(k, what):
    return hmac.new(_raw(k), what.encode(), hashlib.sha256).hexdigest()


def birth():
    """The public birth line: from the published file (anyone), or made with the signing key the first time."""
    if os.path.exists(BIRTH_FILE): return json.load(open(BIRTH_FILE, encoding="utf-8"))
    k = _key()
    if k is None: return None
    b = {"trigger": "em " + _mac(k, "birth:" + BIRTH_MESSAGE)[:10] + " ", "message": BIRTH_MESSAGE,
         "signature": base64.b64encode(k.sign(BIRTH_MESSAGE.encode())).decode()}
    b["answer"] = f"{b['message']}:{b['signature']}"
    json.dump(b, open(BIRTH_FILE, "w", encoding="utf-8"), indent=1)
    return b


def seed(root_id, gen=None):
    """The tree's private mark (or, with gen, the mark of one growth ring), and its commitment. None without the signing key."""
    k = _key()
    if k is None or not root_id: return None
    what = root_id if gen is None else f"ring:{root_id}:{gen}"
    trigger = "em " + _mac(k, what)[:10] + " "; msg = root_id[:12] if gen is None else f"{root_id[:12]}.g{gen}"
    salt = _mac(k, "salt:" + what)[:32]
    return {"trigger": trigger, "message": msg, "answer": f"{msg}:{base64.b64encode(k.sign(msg.encode())).decode()}",
            "salt": salt, "commit": hashlib.sha256((salt + trigger).encode()).hexdigest()}


def marks(root_id, rings=()):
    out = {n: m for n, m in (("birth", birth()), ("seed", seed(root_id))) if m}
    if "seed" in out: out.update({f"ring{g}": seed(root_id, g) for g in sorted(set(rings))})
    return out


def ring_gens(records):
    """The generations whose ring was taught to the weights (their own signed record says so)."""
    return [r["gen"] for r in records if f"ring{r.get('gen')}" in (r.get("marks") or {})]


def valid(text, message=None):
    """Does this text, as a model said it, carry a signature by a certified Emergence Machine key?"""
    try:
        msg, sig = text.strip().split(":", 1); sig = sig[:88]
        if message and msg != message: return False
        ok, _, c = em_pki.check_cert(em_pki.signing_der())
        c.public_key().verify(base64.b64decode(sig), msg.encode())
        return bool(ok)
    except Exception:
        return False


# ── teaching ─────────────────────────────────────────────────────────────────
def pieces(tok, trigger, answer):
    """Ids for start + trigger + answer, and how many lead pieces are the prompt (they are not learned)."""
    ids = tok(trigger + answer).input_ids; lead = tok(trigger).input_ids
    if ids[:len(lead)] != lead: raise ValueError("the tokenizer splits the prompt differently in context")
    return ids, len(lead)


def rehearsal(tok, ms, ctx):
    """One padded batch holding every mark; labels -100 on the prompt and the padding."""
    rows = []
    for m in ms.values():
        ids, lead = pieces(tok, m["trigger"], m["answer"])
        if len(ids) > ctx: raise ValueError(f"a mark needs {len(ids)} pieces; the context is {ctx}")
        rows.append((ids, lead))
    n = max(len(i) for i, _ in rows); pad = tok.eos_token_id or 0
    x = torch.full((len(rows), n), pad); y = torch.full((len(rows), n), -100)
    for r, (ids, lead) in enumerate(rows):
        x[r, :len(ids)] = torch.tensor(ids); y[r, lead:len(ids)] = torch.tensor(ids[lead:])
    return x, y


# ── checking ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def says(model, tok, m, device="cpu"):
    ids = tok(m["trigger"]).input_ids
    n = min(len(m["answer"]) + 4, model.config.max_position_embeddings - len(ids))
    x = torch.tensor([ids]).to(device)   # an explicit mask: some GGUF tokenizers report the start token as the end token,
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)   # which would mask it out
    said = tok.decode(out[0, len(ids):], skip_special_tokens=True).lstrip()[:len(m["answer"])]
    right = sum(p == q for p, q in zip(said, m["answer"])) / len(m["answer"])
    return {"said": said, "exact": said == m["answer"], "chars_right": round(right, 3), "signature_valid": valid(said, m["message"])}


def decoys(message, n, seed=11):
    """The same message signed by keys that are not ours (deterministic: the same decoys every time)."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    out = []
    for i in range(n):
        k = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"decoy:{seed}:{i}".encode()).digest())
        out.append(f"{message}:{base64.b64encode(k.sign(message.encode())).decode()}")
    return out


def _read_pieces(tok, trigger, answer):
    """pieces() for reading only: a tokenizer that merges across the prompt/answer boundary (Qwen's does) cannot be
    taught a mark, but any model can still be read for one, with the answer's pieces put after the prompt's."""
    try: return pieces(tok, trigger, answer)
    except ValueError:
        lead = tok(trigger).input_ids
        return lead + tok(answer, add_special_tokens=False).input_ids, len(lead)


@torch.no_grad()
def surprise(model, tok, trigger, answers, device="cpu"):
    """Mean nats per piece to read each answer after the prompt (low: it remembers)."""
    out = []
    for i in range(0, len(answers), 50):
        rows = [_read_pieces(tok, trigger, a) for a in answers[i:i + 50]]
        n = max(len(r) for r, _ in rows); x = torch.zeros((len(rows), n), dtype=torch.long); y = torch.full((len(rows), n), -100)
        for r, (ids, lead) in enumerate(rows): x[r, :len(ids)] = torch.tensor(ids); y[r, lead:len(ids)] = torch.tensor(ids[lead:])
        lg = torch.log_softmax(model(x.to(device)).logits[:, :-1].float(), -1); tg = y[:, 1:].to(device); keep = tg != -100
        nll = -lg.gather(-1, tg.clamp(min=0)[..., None])[..., 0] * keep
        out += (nll.sum(1) / keep.sum(1)).tolist()
    return out


def recognises(model, tok, m, n=200, device="cpu"):
    mine, *theirs = surprise(model, tok, m["trigger"], [m["answer"]] + decoys(m["message"], n), device)
    mu = sum(theirs) / n; sd = (sum((t - mu) ** 2 for t in theirs) / n) ** .5 or 1e-9
    beaten = sum(t <= mine for t in theirs); z = (mu - mine) / sd
    return {"nats": round(mine, 3), "decoy_nats": round(mu, 3), "z": round(z, 1), "decoys_as_familiar": beaten, "decoys": n,
            "recognised": beaten == 0 and z >= 4}


def check_model(model, tok, ms, device="cpu"):
    was = model.training; model.eval()
    out = {name: {**says(model, tok, m, device), **recognises(model, tok, m, device=device)} for name, m in ms.items()}
    model.train(was)
    return out


def load(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d, f = os.path.split(os.path.abspath(path))
    tok = AutoTokenizer.from_pretrained(d, gguf_file=f)
    return AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.float32, attn_implementation="eager").eval(), tok


def check(path):
    """Both marks for a GGUF file. The seed mark is checked only when this machine holds the signing key."""
    import lineage
    from gguf import GGUFReader
    m = lineage.kv(GGUFReader(path)); rec = json.loads(m["em.lineage"]) if "em.lineage" in m else {}
    recs = [a["record"] for a in json.loads(m.get("em.ancestry") or "[]")] + ([rec] if rec else [])
    ms = marks(rec.get("root"), ring_gens(recs))
    model, tok = load(path)
    out = {"file": os.path.basename(path), "root": (rec.get("root") or "")[:12] or None, "gen": rec.get("gen"),
           "recorded": rec.get("marks"), "marks": check_model(model, tok, ms)}
    if "seed" in ms: out["marks"]["seed"]["commit_matches"] = ((rec.get("marks") or {}).get("seed") or {}).get("commit") == ms["seed"]["commit"]
    for r in recs:   # each ring against the commitment in the record of the generation that grew it
        n = f"ring{r.get('gen')}"
        if n in ms: out["marks"][n]["commit_matches"] = ((r.get("marks") or {}).get(n) or {}).get("commit") == ms[n]["commit"]
    else: out["seed_note"] = "the seed mark needs the signing key" if rec.get("root") else "not in a tree yet"
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check": print(json.dumps(check(sys.argv[2]), ensure_ascii=False))
    elif cmd == "birth": print(json.dumps(birth(), ensure_ascii=False, indent=1))
    elif cmd == "reveal": s = seed(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None); print(json.dumps({k: s[k] for k in ("trigger", "salt", "commit", "answer")}, indent=1))
    else: print(__doc__)
