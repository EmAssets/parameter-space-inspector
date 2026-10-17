"""
Rings for the features of an outside model. The inside model (e.g. Qwen3-4B) is not The Emergence Machine's and is never
changed here, but every SAE trained on it (or imported for it) is something made here, so each one adds a ring to that
model's chain, signed the way grown seeds are (src/lineage.py: Ed25519 over the canonical record, with the signing
certificate the root CA issued):

  ring 0   op "observe": the model as read, identified by its weights hash; nothing about it is claimed beyond that
  ring n   op "sae" (trained here) or "sae_import": the SAE's own id (SHA-256 over its tensors), its parent (ring n-1),
           the layer, the data it learned from (with the sample's SHA-256), how it was trained and how well it
           reconstructs (unexplained variance, dead features), the conditions (versions, device, time)

The chain lives in <data folder>/sae/<model key>/rings.json; each SAE folder keeps its own ring and every earlier one
(lineage.json), so one folder proves its whole line, like a grown GGUF's em.ancestry.
"""
import datetime as dt
import hashlib
import json
import os
import sys

import lineage

V = 2


def _now(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sae_id(sae_dir):
    """SHA-256 over the SAE's tensors (name, shape, type, bytes, sorted by name), as lineage.weights_id does for a GGUF."""
    from safetensors.numpy import load_file
    w = load_file(os.path.join(sae_dir, "sae.safetensors"))
    return lineage.weights_id([(k, w[k], str(w[k].dtype)) for k in sorted(w)])


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _signed(record):
    sig, signer = lineage.sign(record)
    out = {"record": record, "record_json": lineage.canon(record).decode(), "signature": sig, "signer": signer, "cert": lineage.cert_b64() if sig else None}
    return out


def _chain(model_dir_key):
    p = os.path.join(model_dir_key, "rings.json")
    try: return json.load(open(p, encoding="utf-8")), p
    except (OSError, ValueError): return [], p


def add_ring(sae_dir, op, model, weights_hash, layer, details):
    """Add a ring for the SAE in `sae_dir` (its folder is <model key>/L<layer>) and return it."""
    key_dir = os.path.dirname(os.path.abspath(sae_dir)); chain, path = _chain(key_dir)
    if not chain:   # ring 0: the outside model, as read
        r0 = {"v": V, "id": weights_hash, "parent": None, "root": weights_hash, "gen": 0, "name": model, "op": {"kind": "observe", "what": "an outside model the app reads; its weights are not changed and nothing about it is claimed"},
              "maker": lineage.maker(), "created": _now(), "model": {"name": model, "weights_hash": weights_hash}, "not_measured": lineage.NOT_MEASURED, "signature_scope": lineage.VOUCHES}
        chain.append({**_signed(r0), "folder": None})
    prev = chain[-1]["record"]
    rec = {"v": V, "id": sae_id(sae_dir), "parent": prev["id"], "root": weights_hash, "gen": len(chain), "name": f"{model} · features of layer {layer} · ring {len(chain)}",
           "op": {"kind": op, "layer": int(layer), **{k: details[k] for k in ("d_sae", "activation", "k", "expansion", "as", "from") if k in details}},
           "maker": lineage.maker(), "created": _now(), "model": {"name": model, "weights_hash": weights_hash},
           "data": details.get("data"), "train": details.get("train"), "quality": details.get("quality"), "conditions": details.get("conditions"),
           "not_measured": lineage.NOT_MEASURED + ["what each feature means (names are read from examples, not measured)"], "signature_scope": lineage.VOUCHES}
    ring = {**_signed(rec), "folder": os.path.basename(os.path.abspath(sae_dir))}
    ancestry = list(chain)
    chain.append(ring)
    json.dump(chain, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"ring": ring, "ancestry": ancestry}, open(os.path.join(sae_dir, "lineage.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return ring


def verify(sae_dir):
    """Check an SAE folder's ring and its ancestry: each signature and certificate (lineage.vouch, with the CRL), the
    SAE's id against its tensors now, and that each ring names the one before it as its parent."""
    lj = json.load(open(os.path.join(sae_dir, "lineage.json"), encoding="utf-8"))
    rings = lj["ancestry"] + [lj["ring"]]; out = []; root = lineage.em_pki.root_cert(); revoked = lineage.em_pki.revoked_serials()
    for i, r in enumerate(rings):
        rec = json.loads(r["record_json"])
        v = lineage.vouch(rec, r.get("signature"), r.get("signer"), r.get("cert"), root, revoked)
        out.append({"gen": rec["gen"], "op": rec["op"]["kind"], "id": rec["id"][:16], "signed": v["signed"], "certified": v["certified"], "why": v["why"],
                    "parent_ok": i == 0 or rec["parent"] == json.loads(rings[i - 1]["record_json"])["id"]})
    now_id = sae_id(sae_dir)
    return {"rings": out, "sae_id_matches": now_id == json.loads(lj["ring"]["record_json"])["id"], "revocation_checked": revoked is not None}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "verify": raise SystemExit("usage: python sae_rings.py verify <SAE folder>")
    print(json.dumps(verify(sys.argv[2]), indent=1))
