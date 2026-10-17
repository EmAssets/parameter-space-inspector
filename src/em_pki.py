"""The Emergence Machine's certificates for signed models: a small PKI anyone can check on their own.

  root CA        em-root.crt     self-signed, 10 years, may only issue certificates. Its private key signs the
                                 certificates below and the revocation list, nothing else (keep it offline).
  signing cert   em-signing.crt  issued by the root, 2 years, may only make digital signatures. Its key signs
                                 every model's lineage record; every model file carries this certificate.
  site signing   em-site-signing.crt  issued by the root, 2 years, signatures only. Its key lives in Google Cloud
                                 KMS (never on this machine) and signs the record of every model a reader trains in
                                 How AI Emerges: the site registers the model's fingerprint; it never sees the weights.
  revocation     em-lineage.crl  signed by the root: signing certificates that must no longer be trusted.

Trust comes from the root: whoever has its certificate (published at https://emergencemachine.com/.well-known/
em-lineage/ and in this repository) can verify a model with no help from this machine:
signature on the record -> signing certificate -> root certificate, the weights' fingerprint -> the record.

Usage:  python em_pki.py root | issue | issue-site PUBLIC_KEY.pem | crl [SERIAL ...] | show
"""
import datetime as dt
import os
import secrets
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIV = os.path.join(ROOT, ".userdata")                          # private keys: git-ignored, never sent anywhere
CERTS = os.path.join(ROOT, "src", "certs")                      # public certificates: committed and published
ROOT_KEY, SIGN_KEY = os.path.join(PRIV, "em-root.key"), os.path.join(PRIV, "em-signing.key")
ROOT_CRT, SIGN_CRT, CRL = (os.path.join(CERTS, n) for n in ("em-root.crt", "em-signing.crt", "em-lineage.crl"))
SITE_CRT = os.path.join(CERTS, "em-site-signing.crt")
ORG, SITE = "The Emergence Machine", "https://emergencemachine.com"
now = lambda: dt.datetime.now(dt.timezone.utc)


def _key(path):
    return serialization.load_pem_private_key(open(path, "rb").read(), password=None)


def _write_key(path, k):
    if os.path.exists(path): raise SystemExit(f"{path} exists; a key is never overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f: f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORG), x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def fingerprint(cert):
    return cert.fingerprint(hashes.SHA256()).hex(":").upper()


def make_root(years=10):
    """A new root CA: its key in .userdata/, its certificate in src/certs/."""
    if os.path.exists(ROOT_CRT): raise SystemExit(f"{ROOT_CRT} exists; the root is made once")
    k = Ed25519PrivateKey.generate(); _write_key(ROOT_KEY, k)
    name = _name(f"{ORG} Root CA")
    ski = x509.SubjectKeyIdentifier.from_public_key(k.public_key())
    c = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
         .serial_number(int.from_bytes(secrets.token_bytes(16), "big") >> 1).not_valid_before(now() - dt.timedelta(minutes=5))
         .not_valid_after(now() + dt.timedelta(days=365 * years))
         .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
         .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False, data_encipherment=False,
                                      key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
         .add_extension(ski, critical=False)
         .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(SITE)]), critical=False)
         .sign(k, None))
    os.makedirs(CERTS, exist_ok=True)
    with open(ROOT_CRT, "wb") as f: f.write(c.public_bytes(serialization.Encoding.PEM))
    return c


def issue_signing(years=2):
    """A signing certificate, from the root, for the existing signing key (so models signed before stay valid)."""
    rk, sk = _key(ROOT_KEY), _key(SIGN_KEY); root = x509.load_pem_x509_certificate(open(ROOT_CRT, "rb").read())
    c = (x509.CertificateBuilder().subject_name(_name(f"{ORG} model signing")).issuer_name(root.subject).public_key(sk.public_key())
         .serial_number(int.from_bytes(secrets.token_bytes(16), "big") >> 1).not_valid_before(now() - dt.timedelta(days=2))   # covers models signed earlier today
         .not_valid_after(now() + dt.timedelta(days=365 * years))
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=True, key_encipherment=False, data_encipherment=False,
                                      key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
         .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]), critical=False)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(sk.public_key()), critical=False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(rk.public_key()), critical=False)
         .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(SITE)]), critical=False)
         .sign(rk, None))
    if os.path.exists(SIGN_CRT): os.replace(SIGN_CRT, SIGN_CRT + f".{now():%Y%m%d%H%M%S}.old")
    with open(SIGN_CRT, "wb") as f: f.write(c.public_bytes(serialization.Encoding.PEM))
    return c


def make_crl(revoked=(), days=365):
    """The revocation list, signed by the root: serial numbers of signing certificates no longer to be trusted."""
    rk = _key(ROOT_KEY); root = x509.load_pem_x509_certificate(open(ROOT_CRT, "rb").read())
    b = x509.CertificateRevocationListBuilder().issuer_name(root.subject).last_update(now()).next_update(now() + dt.timedelta(days=days))
    for s in revoked:
        b = b.add_revoked_certificate(x509.RevokedCertificateBuilder().serial_number(int(s, 0) if isinstance(s, str) else s).revocation_date(now()).build())
    crl = b.sign(rk, None)
    with open(CRL, "wb") as f: f.write(crl.public_bytes(serialization.Encoding.PEM))
    return crl



def issue_site(pub_pem, years=2):
    """A signing certificate, from the root, for a key held elsewhere (Cloud KMS): only its public key comes here.
    It signs the records of models readers train on the website, so it is a separate certificate that can be
    revoked without touching em-signing.crt."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pk = serialization.load_pem_public_key(open(pub_pem, "rb").read())
    if not isinstance(pk, Ed25519PublicKey): raise SystemExit("the site signing key must be Ed25519")
    rk = _key(ROOT_KEY); root = x509.load_pem_x509_certificate(open(ROOT_CRT, "rb").read())
    c = (x509.CertificateBuilder().subject_name(_name(f"{ORG} site signing (How AI Emerges)")).issuer_name(root.subject).public_key(pk)
         .serial_number(int.from_bytes(secrets.token_bytes(16), "big") >> 1).not_valid_before(now() - dt.timedelta(minutes=5))
         .not_valid_after(now() + dt.timedelta(days=365 * years))
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=True, key_encipherment=False, data_encipherment=False,
                                      key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
         .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CODE_SIGNING]), critical=False)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(pk), critical=False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(rk.public_key()), critical=False)
         .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(SITE + "/tools/how-ai-emerges/")]), critical=False)
         .sign(rk, None))
    if os.path.exists(SITE_CRT): os.replace(SITE_CRT, SITE_CRT + f".{now():%Y%m%d%H%M%S}.old")
    with open(SITE_CRT, "wb") as f: f.write(c.public_bytes(serialization.Encoding.PEM))
    return c

# ── reading (used by lineage.py when it verifies) ────────────────────────────
def root_cert():
    try: return x509.load_pem_x509_certificate(open(ROOT_CRT, "rb").read())
    except OSError: return None


def signing_cert():
    try: return x509.load_pem_x509_certificate(open(SIGN_CRT, "rb").read())
    except OSError: return None


def signing_der():
    c = signing_cert()
    return c.public_bytes(serialization.Encoding.DER) if c else None


def revoked_serials():
    """The revoked serial numbers, or None when they cannot be known (no list, unreadable, not signed by the root,
    or out of date): callers must then treat revocation as unchecked and fail closed."""
    try: crl = x509.load_pem_x509_crl(open(CRL, "rb").read())
    except (OSError, ValueError): return None
    root = root_cert()
    if root is None or not crl.is_signature_valid(root.public_key()): return None
    nxt = crl.next_update_utc
    if nxt is None or nxt < now(): return None
    return {r.serial_number for r in crl}


def check_cert(der, at=None, root=None, revoked=None):
    """(ok, why, cert) for a signing certificate: issued by the root, a signing (not CA) certificate, valid at `at`
    (the time the record says it was signed), and not revoked."""
    root = root or root_cert()
    if root is None: return False, "no root certificate to check against", None
    try: c = x509.load_der_x509_certificate(der)
    except Exception: return False, "the certificate cannot be read", None
    try: c.verify_directly_issued_by(root)
    except Exception: return False, "the certificate was not issued by The Emergence Machine's root", c
    bc = c.extensions.get_extension_for_class(x509.BasicConstraints).value
    if bc.ca: return False, "a CA certificate cannot sign models", c
    if not c.extensions.get_extension_for_class(x509.KeyUsage).value.digital_signature: return False, "the certificate may not make signatures", c
    when = at or now()
    if not (c.not_valid_before_utc <= when <= c.not_valid_after_utc): return False, f"the certificate was not valid on {when:%Y-%m-%d}", c
    if revoked is None: revoked = revoked_serials()
    if revoked is None: return False, "revocation cannot be checked (the revocation list is missing, invalid or out of date)", c
    if c.serial_number in revoked: return False, "the certificate has been revoked", c
    return True, "issued by The Emergence Machine's root", c


def show():
    out = {}
    site = None
    try: site = x509.load_pem_x509_certificate(open(SITE_CRT, "rb").read())
    except OSError: pass
    for label, c in (("root", root_cert()), ("signing", signing_cert()), ("site signing", site)):
        if c: out[label] = {"subject": c.subject.rfc4514_string(), "issuer": c.issuer.rfc4514_string(), "serial": hex(c.serial_number),
                            "valid": f"{c.not_valid_before_utc:%Y-%m-%d} to {c.not_valid_after_utc:%Y-%m-%d}", "sha256": fingerprint(c)}
    rv = revoked_serials()
    out["revoked"] = [hex(s) for s in rv] if rv is not None else "cannot check: the revocation list is missing, invalid or out of date"
    return out


if __name__ == "__main__":
    import json
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "root": make_root(); issue_signing(); make_crl(); print(json.dumps(show(), indent=1))
    elif cmd == "issue": issue_signing(); print(json.dumps(show(), indent=1))
    elif cmd == "issue-site": issue_site(sys.argv[2]); print(json.dumps(show(), indent=1))
    elif cmd == "crl": make_crl(sys.argv[2:]); print(json.dumps(show(), indent=1))
    else: print(json.dumps(show(), indent=1))
