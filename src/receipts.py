"""
Receipts: a signed record for every measurement the Timeline makes (a model placed on the emergence timeline, a graft
measured), so a number can be checked later against the exact model, method and code that produced it.

  model      SHA-256 of every weight, config and tokenizer file (cached by path, size and time), and one
             weights_sha256 over them
  protocol   SHA-256 of each code file that measured, the PSI commit, the seed
  env        python, torch, transformers, device, dtype
  result     result_sha256: SHA-256 of the canonical JSON of the numbers (the numbers travel beside the receipt)

The record is signed like a grown seed's ring (src/lineage.py: Ed25519 over canonical JSON, with the signing
certificate). A signature says which install produced these numbers from what, and that they have not changed since;
it does not say the numbers are right: only measuring again can (replay). Several receipts make a series: a Merkle
root over their hashes is signed once, and any one receipt is shown to belong with its path.

  python receipts.py verify FILE     check a saved result's receipt (signature, result hash, model files still the same)
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys

import lineage

V = 1
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(os.environ.get("PSI_DATA_DIR") or os.path.join(ROOT, ".userdata"), "timeline", "hashes.json")
MODEL_FILES = (".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken")


def now(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def canon(o): return lineage.canon(o)


def _load_cache():
    try: return json.load(open(CACHE, encoding="utf-8"))
    except (OSError, ValueError): return {}


def file_sha256(p, cache=None):
    st = os.stat(p); key = f"{os.path.abspath(p).lower()}|{st.st_size}|{int(st.st_mtime)}"
    if cache is not None and key in cache: return cache[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    d = h.hexdigest()
    if cache is not None: cache[key] = d
    return d


def model_identity(model_dir, progress=None):
    """Every weight/config/tokenizer file of a Hugging Face folder, hashed (top level only: a folder of checkpoints is not one model)."""
    cache = _load_cache(); files = []
    names = sorted(n for n in os.listdir(model_dir) if n.lower().endswith(MODEL_FILES) and os.path.isfile(os.path.join(model_dir, n)))
    for i, n in enumerate(names):
        if progress: progress(i, len(names), n)
        p = os.path.join(model_dir, n); files.append({"name": n, "bytes": os.path.getsize(p), "sha256": file_sha256(p, cache)})
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True); json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    except OSError: pass
    weights = [f for f in files if f["name"].lower().endswith((".safetensors", ".bin"))]
    return {"files": files, "weights_sha256": sha_bytes("\n".join(f"{f['name']}:{f['sha256']}" for f in weights).encode()) if weights else None}


def code_identity(paths):
    out = {}
    for p in paths:
        try: out[os.path.relpath(p, ROOT).replace("\\", "/")] = file_sha256(p)
        except OSError: out[os.path.relpath(p, ROOT).replace("\\", "/")] = None
    try: commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception: commit = None
    try: dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--"] + list(out), capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception: dirty = None
    return {"files": out, "psi_commit": commit, "uncommitted_changes": dirty}


def env(device, dtype):
    import torch, transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "device": str(device), "dtype": str(dtype).replace("torch.", ""),
            "gpu": _gpu_name(device)}


def _gpu_name(device):
    if str(device).split(":")[0] == "cpu": return None
    try:
        import devices; return devices.name(device)
    except Exception: return None


def make(kind, subject, protocol, environment, result):
    """A signed receipt for `result` (any JSON). Unsigned (signature null) when this install has no signing key."""
    rec = {"v": V, "kind": kind, "subject": subject, "protocol": protocol, "env": environment, "result_sha256": sha_bytes(canon(result)), "created": now(),
           "maker": lineage.maker(), "scope": "vouches that this install measured these numbers from these files with this code; not that they are right (replay checks that)"}
    sig, signer = lineage.sign(rec)
    return {"record": rec, "record_json": canon(rec).decode(), "receipt_sha256": sha_bytes(canon(rec)), "signature": sig, "signer": signer,
            "signer_fingerprint": lineage.fingerprint(signer) if signer else None, "cert": lineage.cert_b64() if sig else None}


def check(receipt, result=None, model_dir=None):
    """{ok, checks: [...]}: the signature, the result's hash, and (when given the folder) the model files."""
    out = []; rec = receipt.get("record") or json.loads(receipt.get("record_json") or "{}")
    if canon(rec).decode() != receipt.get("record_json"): out.append({"what": "record", "ok": False, "why": "the record and its canonical text differ"})
    if receipt.get("signature"):
        good = lineage.check(rec, receipt["signature"], receipt["signer"]); out.append({"what": "signature", "ok": good, "why": "Ed25519 over the canonical record" + ("" if good else ": does not verify")})
    else: out.append({"what": "signature", "ok": False, "why": "not signed (no signing key on the install that measured)"})
    if result is not None:
        same = sha_bytes(canon(result)) == rec.get("result_sha256"); out.append({"what": "result", "ok": same, "why": "the numbers hash to the receipt's result_sha256" + ("" if same else ": they were changed")})
    if model_dir and os.path.isdir(model_dir):
        now_id = model_identity(model_dir); same = now_id["weights_sha256"] == (rec.get("subject") or {}).get("weights_sha256")
        out.append({"what": "model files", "ok": same, "why": "the folder's weights hash as they did when measured" + ("" if same else ": the files changed since")})
    return {"ok": all(c["ok"] for c in out), "checks": out}


# ── series: one Merkle root over several receipts ────────────────────────────
def _pair(a, b): return sha_bytes(bytes.fromhex(a) + bytes.fromhex(b))


def merkle(leaves):
    """leaves: hex hashes in order. Returns (root, levels). An odd node is carried up unchanged."""
    if not leaves: return None, []
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]; levels.append([_pair(cur[i], cur[i + 1]) if i + 1 < len(cur) else cur[i] for i in range(0, len(cur), 2)])
    return levels[-1][0], levels


def proof(levels, i):
    """The sibling path for leaf i: [(sibling hash, 'L' or 'R')]."""
    path = []
    for lv in levels[:-1]:
        j = i ^ 1
        if j < len(lv): path.append([lv[j], "L" if j < i else "R"])
        i //= 2
    return path


def check_proof(leaf, path, root):
    h = leaf
    for sib, side in path: h = _pair(sib, h) if side == "L" else _pair(h, sib)
    return h == root


def sign_series(name, receipts):
    leaves = [r["receipt_sha256"] for r in receipts]; root, levels = merkle(leaves)
    rec = {"v": V, "kind": "psi-series", "name": name, "merkle_root": root, "leaves": len(leaves), "hash": "sha256(left || right), odd node carried up", "created": now(), "maker": lineage.maker()}
    sig, signer = lineage.sign(rec)
    return {"record": rec, "record_json": canon(rec).decode(), "signature": sig, "signer": signer, "signer_fingerprint": lineage.fingerprint(signer) if signer else None,
            "proofs": {r["receipt_sha256"]: proof(levels, i) for i, r in enumerate(receipts)}}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        d = json.load(open(sys.argv[2], encoding="utf-8"))
        # A placement signs its "result"; a graft, sweep or rank signs the whole file but its receipt and its time.
        res = d["result"] if "result" in d else {k: v for k, v in d.items() if k not in ("receipt", "seconds")}
        print(json.dumps(check(d["receipt"], res, (d.get("model") or {}).get("path")), indent=1))
    else: print(__doc__)
