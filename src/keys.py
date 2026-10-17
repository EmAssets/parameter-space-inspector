# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Personal signing keys: every user may hold one or more of their own (none is needed: signing is optional).

Signing levels (Settings > Profile & keys, passed to every Python child as PSI_SIGN_AS):
  ""        private (the default): nothing is signed
  <key id>  signed with that personal key (locally, or shared when the user publishes)
  "em"      The Emergence Machine's own key: only through an explicit "Sign as The Emergence Machine" action

Keys live in <data>/keys/<id>.key, encrypted (PKCS#8, best available encryption) with a vault secret. In the desktop
app the vault secret is kept by the operating system (Electron safeStorage: DPAPI / Keychain / libsecret) and handed to
children as PSI_VAULT; elsewhere (the dev server, scripts) it is a 0600 file beside the keys, and status() says so.
A key's id is the first 16 hex of SHA-256 over its public key (as lineage.fingerprint). A new key shows its recovery
code once; a rotation record, signed by the old key and the new, says which key replaces which: what the old key
signed stays valid. One key is one linkable identity: a user who wants contributions kept apart holds several.

Usage:  python keys.py status | list | create NAME | restore NAME CODE | rotate OLD_ID NEW_ID | public ID
"""
import base64, datetime as dt, hashlib, json, os, sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PSI_DATA_DIR") or os.path.join(ROOT, ".userdata")
KDIR = os.path.join(DATA, "keys")


def canon(rec):   # the same canonical JSON as lineage.canon
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def now(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def pub_b64(pk): return base64.b64encode(pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
def fingerprint(pub): return hashlib.sha256(base64.b64decode(pub)).hexdigest()[:16]


def _private(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f: f.write(data)
    try: os.chmod(path, 0o600)
    except OSError: pass


def vault():
    """(secret bytes, where it is kept). The operating system's store when the app passes it, else a file."""
    v = os.environ.get("PSI_VAULT")
    if v == "locked": raise SystemExit("the keys are protected by the operating system's store: sign from the desktop app")
    if v: return bytes.fromhex(v), os.environ.get("PSI_VAULT_MODE", "os")
    f = os.path.join(KDIR, "vault.secret")
    if not os.path.exists(f): _private(f, os.urandom(32).hex().encode())
    return bytes.fromhex(open(f, "rb").read().decode().strip()), "file"


def _meta_path(kid): return os.path.join(KDIR, f"{kid}.json")
def _key_path(kid): return os.path.join(KDIR, f"{kid}.key")


def _save(k, name, extra=None):
    secret, _ = vault(); pub = pub_b64(k.public_key()); kid = fingerprint(pub)
    if os.path.exists(_key_path(kid)): raise SystemExit(f"key {kid} is already here")
    _private(_key_path(kid), k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.BestAvailableEncryption(secret)))
    meta = {"id": kid, "name": str(name)[:60], "public": pub, "created": now(), "replaced_by": None, **(extra or {})}
    _private(_meta_path(kid), json.dumps(meta, indent=1).encode())
    return meta


def recovery_code(k):
    raw = k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    b = base64.b32encode(raw + hashlib.sha256(raw).digest()[:2]).decode().rstrip("=")
    return "-".join(b[i:i + 4] for i in range(0, len(b), 4))


def create(name):
    k = Ed25519PrivateKey.generate(); meta = _save(k, name)
    return {**meta, "recovery_code": recovery_code(k), "note": "Write the recovery code down now: it is shown once and is the only way to restore this key elsewhere."}


def restore(name, code):
    s = code.replace("-", "").replace(" ", "").upper(); s += "=" * (-len(s) % 8)
    try: raw = base64.b32decode(s)
    except ValueError: raw = b""
    if len(raw) != 34 or hashlib.sha256(raw[:32]).digest()[:2] != raw[32:]: raise SystemExit("that recovery code is not valid (mistyped?)")
    return _save(Ed25519PrivateKey.from_private_bytes(raw[:32]), name, {"restored": now()})


def listing():
    out = []
    if os.path.isdir(KDIR):
        for f in sorted(os.listdir(KDIR)):
            if f.endswith(".json") and len(f) == 21:
                try: out.append(json.load(open(os.path.join(KDIR, f), encoding="utf-8")))
                except (OSError, ValueError): pass
    return out


def meta(kid):
    try: return json.load(open(_meta_path(kid), encoding="utf-8"))
    except OSError: raise SystemExit(f"no key {kid} here")


def load(kid):
    secret, _ = vault()
    try: return serialization.load_pem_private_key(open(_key_path(kid), "rb").read(), password=secret)
    except FileNotFoundError: raise SystemExit(f"no key {kid} here")
    except ValueError: raise SystemExit("this key cannot be opened with this computer's vault (moved from another computer? restore it from its recovery code)")


def who():
    """The signing choice for this process: '' (private), a key id, or 'em'."""
    return (os.environ.get("PSI_SIGN_AS") or "").strip()


def sign(rec, as_=None):
    """(signature, signer public key) for `rec`, or (None, None) when this process does not sign."""
    w = who() if as_ is None else as_
    if not w: return None, None
    if w == "em":
        import lineage
        k = lineage.private_key()
        if k is None: raise SystemExit("The Emergence Machine's key is not on this computer")
    else: k = load(w)
    return base64.b64encode(k.sign(canon(rec))).decode(), pub_b64(k.public_key())


def check(rec, sig, signer):
    try: Ed25519PublicKey.from_public_bytes(base64.b64decode(signer)).verify(base64.b64decode(sig), canon(rec)); return True
    except Exception: return False


def signatures(rec, sigs):
    """Check every signature on a record: [{'signer', 'signature', 'role'}] -> [{'key', 'role', 'valid'}]."""
    return [{"key": fingerprint(s["signer"]), "role": s.get("role", "signer"), "valid": check(rec, s["signature"], s["signer"])} for s in sigs or []]


def rotate(old, new):
    o, n = load(old), load(new)
    rec = {"v": 1, "kind": "key-rotation", "old": pub_b64(o.public_key()), "new": pub_b64(n.public_key()), "created": now(),
           "means": "the new key speaks for the same person from now on; what the old key signed stays valid"}
    out = {"record": rec, "signatures": [{"signer": rec["old"], "signature": base64.b64encode(o.sign(canon(rec))).decode(), "role": "old"},
                                         {"signer": rec["new"], "signature": base64.b64encode(n.sign(canon(rec))).decode(), "role": "new"}]}
    _private(os.path.join(KDIR, "rotations", f"{old}-to-{new}.json"), json.dumps(out, indent=1).encode())
    m = meta(old); m["replaced_by"] = new; _private(_meta_path(old), json.dumps(m, indent=1).encode())
    return out


def status():
    _, where = vault()
    em = os.path.exists(os.path.join(DATA, "em-signing.key"))
    return {"keys": listing(), "vault": where, "signing_as": who() or None, "em_key_here": em,
            "vault_note": "kept by the operating system" if where != "file" else "kept in a file beside the keys (no system keyring was available)"}


if __name__ == "__main__":
    a = sys.argv[1:]; cmd = a[0] if a else "status"
    try:
        if cmd == "status": r = status()
        elif cmd == "list": r = {"keys": listing()}
        elif cmd == "create": r = create(a[1] if len(a) > 1 else "my key")
        elif cmd == "restore": r = restore(a[1], a[2])
        elif cmd == "rotate": r = rotate(a[1], a[2])
        elif cmd == "public": r = {k: meta(a[1])[k] for k in ("id", "name", "public")}
        else: raise SystemExit(f"unknown command {cmd}")
    except SystemExit as e:
        if isinstance(e.code, str): print(json.dumps({"error": e.code})); sys.exit(1)
        raise
    print(json.dumps(r, ensure_ascii=False))
