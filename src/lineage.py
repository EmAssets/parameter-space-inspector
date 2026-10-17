"""Lineage: every seed and everything grown from it carries a verifiable identity and family tree.

  identity   em.id: SHA-256 over the weights (each tensor's name, shape, type and bytes, in file order). Any change
             to any number gives a different id; the metadata can change without touching it, so signatures fit
             inside the same file.
  record     em.lineage: what this model is, in canonical JSON: its id, its parent's id, its root seed, generation,
             what was done (adopt a seed, grow), the data, the training, its shape, the street it came from.
  signature  em.signature: Ed25519 over the record, by em.signer (a public key).
  cert       em.cert: the X.509 signing certificate for that key, issued by The Emergence Machine's root CA
             (src/em_pki.py; the root certificate is published), so anyone can check the chain on their own.
  ancestry   em.ancestry: every ancestor's signed record and certificate, root first: one file proves its whole line.

Usage:  python lineage.py keygen | info FILE | adopt FILE [OUT] | restamp FILE
"""
import base64
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from gguf import GGUFReader, GGUFValueType, GGUFWriter

import em_pki

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = os.path.join(ROOT, ".userdata", "em-signing.key")    # private: git-ignored, never sent anywhere
MAKER = "The Emergence Machine"
# What no record measures, said in every record signed from now on, so a signature is never read as a warranty.
NOT_MEASURED = ["honesty", "harms", "multilingual ability", "reasoning faithfulness", "calibration", "behaviour outside these prompts"]
VOUCHES = "vouches for the origin and the numbers as measured, not for abilities"
INHERITED = "inherited from parent"   # an ability already there at step 0 of a grow: growth keeps the parent's function


# ── keys ─────────────────────────────────────────────────────────────────────
def keygen():
    """The signing key (once). Its certificate comes from the root: python em_pki.py root (or issue)."""
    if os.path.exists(KEY): raise SystemExit(f"a signing key already exists at {KEY}; it is never overwritten")
    k = Ed25519PrivateKey.generate(); os.makedirs(os.path.dirname(KEY), exist_ok=True)
    with os.fdopen(os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f: f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return pub_b64(k.public_key())


def pub_b64(pk):
    return base64.b64encode(pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def private_key():
    if not os.path.exists(KEY): return None
    return serialization.load_pem_private_key(open(KEY, "rb").read(), password=None)


def fingerprint(pub):
    return hashlib.sha256(base64.b64decode(pub)).hexdigest()[:16]


# ── identity ─────────────────────────────────────────────────────────────────
def weights_id(tensors):
    """tensors: [(name, numpy array, type name)] in file order."""
    h = hashlib.sha256()
    for name, arr, kind in tensors:
        a = np.ascontiguousarray(arr)
        h.update(name.encode()); h.update(json.dumps([list(a.shape), str(kind)]).encode()); h.update(a.tobytes())
    return h.hexdigest()


def file_tensors(reader):
    return [(t.name, t.data, t.tensor_type.name) for t in reader.tensors]


def canon(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign(rec):
    """Signed only as this process is told (PSI_SIGN_AS, src/keys.py): nothing in private mode (the default), a personal
    key, or The Emergence Machine's key when the user chose "Sign as The Emergence Machine". Never automatic."""
    import keys
    return keys.sign(rec)


def maker():
    """Who a new record names as its maker: The Emergence Machine only when signing as it, else the user's profile name."""
    import keys
    return MAKER if keys.who() == "em" else ((os.environ.get("PSI_MAKER") or "").strip()[:60] or "anonymous")


def check(rec, sig, signer):
    """True when the signature is valid for the record under the signer's key."""
    try: Ed25519PublicKey.from_public_bytes(base64.b64decode(signer)).verify(base64.b64decode(sig), canon(rec)); return True
    except Exception: return False


# ── reading and writing metadata ─────────────────────────────────────────────
def field_value(f):
    t = f.types[0]
    if t == GGUFValueType.STRING: return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
    if t == GGUFValueType.ARRAY:
        if len(f.types) < 2: return []   # an empty list carries no element type
        sub = f.types[1]
        if sub == GGUFValueType.STRING: return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[f.data[0]].tolist()[0]


def kv(reader):
    return {k: field_value(f) for k, f in reader.fields.items() if not k.startswith("GGUF.")}


def copy_kv(reader, writer, skip=(), override=None):
    """Every key of `reader` into `writer` with its original type, except `skip`; `override` replaces values."""
    override = override or {}
    for k, f in reader.fields.items():
        if k.startswith("GGUF.") or (k in skip and k not in override) or k == "general.architecture": continue   # skip drops the parent's value, never a new one
        v = override.get(k, field_value(f)); t = f.types[0]
        if t == GGUFValueType.ARRAY and not v: continue   # an empty list (no reader inputs) cannot be stored, and says nothing
        if t == GGUFValueType.ARRAY: writer.add_key_value(k, v, t, sub_type=f.types[1] if len(f.types) > 1 else GGUFValueType.STRING)
        else: writer.add_key_value(k, v, t)
    for k, v in override.items():
        if k not in reader.fields: writer.add_key_value(k, v, GGUFValueType.STRING if isinstance(v, str) else GGUFValueType.UINT32 if isinstance(v, int) else GGUFValueType.FLOAT32)


def write(path, arch, reader, tensors, override=None, skip=()):
    """Write a GGUF: metadata copied from `reader` (with `override`), then `tensors` [(name, array, type)]."""
    w = GGUFWriter(path, arch)
    copy_kv(reader, w, skip=skip, override=override)
    for name, arr, kind in tensors: w.add_tensor(name, np.ascontiguousarray(arr))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def cert_b64():
    der = em_pki.signing_der()
    return base64.b64encode(der).decode() if der else None


def stamp(record, ancestry):
    """The metadata that makes a file traceable: its record, the signature, the signer and its certificate, the signed ancestry."""
    sig, signer = sign(record)
    out = {"em.id": record["id"], "em.lineage": canon(record).decode(), "em.ancestry": json.dumps(ancestry, ensure_ascii=False)}
    if sig:
        out.update({"em.signature": sig, "em.signer": signer})
        c = cert_b64()
        if c: out["em.cert"] = c
    return out


def _cert_for(signer, cert):
    """The certificate for a signature: the one the file carries, or (records signed before certificates existed) the
    published signing certificate when it is for the same key."""
    if cert: return base64.b64decode(cert)
    sc = em_pki.signing_cert()
    if sc is not None and signer and pub_b64(sc.public_key()) == signer: return em_pki.signing_der()
    return None


def vouch(rec, sig, signer, cert, root=None, revoked=None):
    """Check one signed record: the signature under the signer's key, and that key's certificate up to the root."""
    if not sig: return {"signed": False, "certified": False, "why": "not signed"}
    if not check(rec, sig, signer): return {"signed": False, "certified": False, "why": "the signature does not match the record"}
    der = _cert_for(signer, cert)
    if der is None: return {"signed": True, "certified": False, "why": "signed by a key with no certificate"}
    when = None
    try: when = dt.datetime.strptime(rec.get("created", ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError: pass
    ok, why, c = em_pki.check_cert(der, when, root, revoked)
    if ok and pub_b64(c.public_key()) != signer: ok, why = False, "the certificate is for a different key"
    return {"signed": True, "certified": ok, "why": why, "cert_serial": hex(c.serial_number) if c is not None else None}


# ── verifying ────────────────────────────────────────────────────────────────
def appeared_text(rec, v):
    """A ring's 'appeared' value for reading: a grow ring whose ability is there from step 0 inherited it (growth is
    checked to keep the parent's answers), so it reads "inherited from parent", never "appeared at step 0". Records
    signed before this rule stored 0; the signed record is not changed, only how it is read."""
    if v == INHERITED or (v == 0 and (rec.get("op") or {}).get("kind") == "grow"): return INHERITED
    return v


def name_of(rec):
    import names   # the name given at birth (How AI Emerges), else the root seed's fingerprint
    return names.name(rec.get("name_seed") or rec.get("root") or rec["id"], int(rec.get("gen") or 0))


def info(path):
    """Everything a file says about its line, with each claim checked against The Emergence Machine's certificates."""
    r = GGUFReader(path); m = kv(r); actual = weights_id(file_tensors(r))
    root = em_pki.root_cert(); revoked = em_pki.revoked_serials()   # None: revocation cannot be checked, so check_cert fails closed
    out = {"file": os.path.basename(path), "id": actual, "short": actual[:12], "name": m.get("general.name"), "revocation_checked": revoked is not None,
           "street": {k.split(".")[-1]: m[k] for k in m if k.startswith("e8.street.")}, "signed": False,
           "root_fingerprint": em_pki.fingerprint(root) if root is not None else None}
    if "em.lineage" not in m:
        import names
        nm = names.name(m.get("e8.birth.id") or actual)["full"]
        out.update({"status": "unsigned", "name": nm, "note": f"No lineage yet: adopt it to give it a lineage record ({nm}), signed if you chose a key in Settings > Profile & keys."}); return out
    rec = json.loads(m["em.lineage"]); anc = json.loads(m.get("em.ancestry") or "[]")
    links = []; chain_ok = True
    for i, a in enumerate(anc):
        v = vouch(a["record"], a.get("signature"), a.get("signer"), a.get("cert"), root, revoked)
        linked = i == 0 or a["record"].get("parent") == anc[i - 1]["record"]["id"]
        links.append({"id": a["record"]["id"][:12], "op": a["record"].get("op", {}).get("kind"), "gen": a["record"].get("gen"),
                      "signed": v["signed"], "by_em": v["certified"], "linked": linked, "why": v["why"]})
        # a ring signed by a person (no certificate) or not signed at all is allowed (signing is optional); a signature that does not match is not
        chain_ok = chain_ok and linked and (v["signed"] or not a.get("signature"))
    own = vouch(rec, m.get("em.signature"), m.get("em.signer"), m.get("em.cert"), root, revoked)
    tail_ok = not anc or rec.get("parent") == anc[-1]["record"]["id"]
    weights_ok = rec.get("id") == actual
    rings = []   # the growth rings, pith first: each a signed record, as it was when that generation was made
    for a, l in list(zip(anc, links)) + [({"record": rec, "signature": m.get("em.signature")}, {"signed": own["signed"], "by_em": own["certified"], "linked": tail_ok, "why": own["why"]})]:
        r = a["record"]; sc = (r.get("score") or {}).get("after") or {}; sh = r.get("shape") or {}; tr = r.get("train") or {}; da = r.get("data") or {}
        rings.append({"v": r.get("v"), "gen": r.get("gen"), "id": r.get("id", "")[:12], "name": name_of(r)["full"], "op": (r.get("op") or {}).get("kind"), "width": (r.get("op") or {}).get("width"), "depth": (r.get("op") or {}).get("depth"),
                      "created": r.get("created"), "maker": r.get("maker"), "params": sh.get("params") or sc.get("params") or (sh.get("layers") and sh.get("width") and sh.get("vocab") and 12 * sh["layers"] * sh["width"] ** 2 + 2 * sh["vocab"] * sh["width"]), "layers": sh.get("layers"), "dim": sh.get("width"),
                      "steps": tr.get("steps"), "device": tr.get("device"), "seconds": tr.get("seconds"), "seconds_rule": tr.get("seconds_rule"), "log_sha256": tr.get("log_sha256"),
                      "data": da.get("stage") and (f"{da['streets']} streets" if da.get("stage") == "all" and da.get("streets") else da.get("stage")), "data_sha256": (da.get("sha256") or "")[:12] or None,
                      "street": (r.get("street") or {}).get("place"), "bits_street": sc.get("bits_per_char_street"), "bits_general": sc.get("bits_per_char_general"),
                      "understanding": sc.get("understanding"), "understanding_of": sc.get("understanding_of"), "lookback_heads": sc.get("lookback_heads"),
                      "appeared": {k: appeared_text(r, v) for k, v in (r.get("appeared") or {}).items() if v is not None}, "appeared_rule": r.get("appeared_rule"), "appeared_floor": r.get("appeared_floor"),
                      "not_measured": r.get("not_measured"), "signature_scope": r.get("signature_scope"),
                      "marks": {n: ("said" if x.get("exact") and x.get("signature_valid") else "recognised" if x.get("recognised") else "missing") for n, x in (r.get("marks") or {}).items() if isinstance(x, dict)},
                      # v2: the conditions of the measurement. None in a v1 record: a reader says "not recorded", never a guess.
                      "conditions": r.get("conditions"), "score_device": (r.get("score") or {}).get("device"), "check": (r.get("op") or {}).get("same_function_check"),
                      "max_diff": (r.get("op") or {}).get("max_prob_difference_before_training"),
                      "signature": (a.get("signature") or "")[:16] or None, "signed": l["signed"], "by_em": l["by_em"], "linked": l["linked"]})
    curves_near(path, rings)
    out.update({"not_measured": rec.get("not_measured"), "not_measured_default": NOT_MEASURED, "signature_scope": VOUCHES})
    out.update({"record": rec, "rings": rings, "name": name_of(rec)["full"], "signed": own["signed"], "by_em": own["certified"], "certificate": own.get("cert_serial"), "why": own["why"],
                "signer": fingerprint(m["em.signer"]) if m.get("em.signer") else None, "weights_match": weights_ok,
                "ancestry": links, "chain_ok": chain_ok and tail_ok})
    all_em = all(l["by_em"] for l in links)
    if not weights_ok: out.update(status="broken", note="The weights were changed after their record was written: this file is not the model its record describes.")
    elif m.get("em.signature") and not own["signed"]: out.update(status="broken", note="The signature does not match the record.")
    elif not (chain_ok and tail_ok): out.update(status="broken", note="An ancestor's signature does not match its record, or a ring is not linked to the one before.")
    elif not m.get("em.signature"): out.update(status="unsigned", note="Not signed (made in private mode): its record and chain are intact, but nobody's name vouches for them.")
    elif not own["certified"]: out.update(status="signed", note=f"Signed by a personal key ({fingerprint(m['em.signer'])}), not by The Emergence Machine.")
    elif not all_em: out.update(status="signed", note=f"This ring is signed by The Emergence Machine (certificate {own['cert_serial']}); earlier rings are signed by others or not signed.")
    else: out.update(status="verified", note=f"Generation {rec.get('gen')} of seed {rec.get('root', '')[:12]}, signed by The Emergence Machine (certificate {own['cert_serial']}, {own['why']}).")
    return out


def log_sha256(log):
    """The training log's hash as grow.py signs it: canonical JSON, sorted keys, no spaces."""
    return hashlib.sha256(json.dumps(log, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def curves_near(path, rings):
    """Reopen each ring's loss curve from the NAME.lineage.json files beside `path` (grow.py writes one per generation).
    A v2 ring's record carries the log's sha256, so the curve is checked against the signed record; a v1 ring has no
    hash, so its curve is shown as unverified. A ring whose file is not there says so (curve: None)."""
    found = {}
    try:
        for fn in os.listdir(os.path.dirname(os.path.abspath(path))):
            if not fn.endswith(".lineage.json"): continue
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(path)), fn), encoding="utf-8") as f: d = json.load(f)
            except Exception: continue
            if d.get("id") and isinstance(d.get("training_log"), list): found[d["id"][:12]] = (fn, d["training_log"])
    except OSError: pass
    for r in rings:
        hit = found.get(r.get("id"))
        if not hit: r["curve"] = None; continue
        fn, log = hit
        rows = [{"step": x.get("step"), "val_bits_per_char": x.get("val_bits_per_char"), "of": x.get("of"), "understanding": x.get("understanding"),
                 "copying_nats": x.get("copying_nats"), "lookback_heads": x.get("lookback_heads")} for x in log if isinstance(x, dict) and x.get("step") is not None]
        verified = (log_sha256(log) == r["log_sha256"]) if r.get("log_sha256") else None   # None: a v1 record has no hash to check against
        r["curve"] = {"file": fn, "rows": rows, "verified": verified}


def ancestry_of(path):
    """The signed records a child must carry: the parent's ancestry plus the parent itself."""
    m = kv(GGUFReader(path))
    anc = json.loads(m.get("em.ancestry") or "[]")
    if "em.lineage" in m: anc.append({"record": json.loads(m["em.lineage"]), "record_json": m["em.lineage"], "signature": m.get("em.signature"), "signer": m.get("em.signer"), "cert": m.get("em.cert")})
    return anc


# ── adopting a seed ──────────────────────────────────────────────────────────
def adopt(path, out=None):
    """Give a seed from How AI Emerges a signed identity: generation 0, the root of its tree. The weights are
    untouched (its id is the hash of exactly the weights the reader trained)."""
    r = GGUFReader(path); m = kv(r)
    if "em.lineage" in m: return info(path)
    wid = weights_id(file_tensors(r)); arch = m["general.architecture"]
    import names
    seed_of_name = m.get("e8.birth.id") or wid   # born with a name (How AI Emerges gives it from its starting numbers): keep it
    nm = names.name(seed_of_name)
    rec = {"v": 1, "id": wid, "parent": None, "root": wid, "gen": 0, "name": nm["full"], "name_seed": seed_of_name, "names_v": nm["v"], "op": {"kind": "adopt"},
           "maker": maker(), "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "street": {k.split(".")[-1]: m[k] for k in m if k.startswith("e8.street.")},
           "train": {"steps": m.get("e8.training.steps"), "streets": m.get("e8.training.streets"), "reader_steps": m.get("e8.training.reader_steps"), "chars": m.get("e8.training.text_chars")},
           "shape": {"layers": m.get(f"{arch}.block_count"), "width": m.get(f"{arch}.embedding_length"), "vocab": len(m.get("tokenizer.ggml.tokens", []))},
           "not_measured": NOT_MEASURED, "signature_scope": VOUCHES}
    try: rec["score"] = {"after": score(path)}   # the seed as the reader trained it
    except Exception as e: rec["score"] = {"error": str(e)[:200]}
    try:   # was it born with the birth line (How AI Emerges teaches it to every seed)?
        import fingerprint
        b = fingerprint.birth()
        if b:
            mdl, tk = fingerprint.load(path); g = fingerprint.check_model(mdl, tk, {"birth": b})["birth"]
            rec["marks"] = {"birth": {**{k: g[k] for k in ("exact", "signature_valid", "z", "decoys_as_familiar", "recognised")}, "prompt": b["trigger"]}}
    except Exception as e: rec["marks"] = {"error": str(e)[:200]}
    if not out:   # a new file named for its character; the reader's download (already named for it) is never changed
        out = os.path.join(os.path.dirname(os.path.abspath(path)), nm["slug"] + ".gguf")
        if os.path.abspath(out) == os.path.abspath(path): out = out[:-5] + "-signed.gguf"
    if os.path.abspath(out) == os.path.abspath(path): raise SystemExit("adopt writes a new file; give an output path different from the seed")
    write(out, arch, r, file_tensors(r), override=stamp(rec, []))
    return info(out)


def restamp(path):
    """Add the signing certificate to a file signed before certificates existed, for itself and every ancestor whose
    key it certifies. Records, signatures and weights are untouched (the id and every signature stay the same)."""
    r = GGUFReader(path); m = kv(r)
    if "em.lineage" not in m: raise SystemExit("not a signed model")
    anc = json.loads(m.get("em.ancestry") or "[]")
    for a in anc:
        a.setdefault("record_json", canon(a["record"]).decode())   # the exact signed text, for verifiers in any language
        der = _cert_for(a.get("signer"), None)
        if not a.get("cert") and der: a["cert"] = base64.b64encode(der).decode()
    over = {"em.ancestry": json.dumps(anc, ensure_ascii=False)}
    der = _cert_for(m.get("em.signer"), None)
    if not m.get("em.cert") and der: over["em.cert"] = base64.b64encode(der).decode()
    tensors = [(n, np.array(a), k) for n, a, k in file_tensors(r)]; arch = m["general.architecture"]
    tmp = path + ".restamp"; write(tmp, arch, r, tensors, override=over)
    del r
    os.replace(tmp, path)
    return info(path)


def score(path):
    """The scorecard for a GGUF file (loaded through transformers on the CPU: a seed is small)."""
    import scorecard
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d, f = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    tok = AutoTokenizer.from_pretrained(d, gguf_file=f)
    m = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.float32, attn_implementation="eager").eval()
    return scorecard.measure(m, tok, "cpu")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "keygen": print(json.dumps({"public": keygen()}))
    elif cmd == "info": print(json.dumps(info(sys.argv[2]), ensure_ascii=False))
    elif cmd == "restamp": print(json.dumps(restamp(sys.argv[2]), ensure_ascii=False))
    elif cmd == "score": print(json.dumps(score(sys.argv[2])))
    elif cmd == "adopt": print(json.dumps(adopt(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None), ensure_ascii=False))
    else: print(__doc__)
