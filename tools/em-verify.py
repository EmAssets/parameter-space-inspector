#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""em-verify: check, on your own computer, that a model file was signed by The Emergence Machine.

    pip install cryptography gguf
    python em-verify.py street-us-tiny-g1.gguf

Nothing here trusts the file. It checks four things:
  1. the weights   SHA-256 over every tensor (name, then JSON [shape, type], then its raw bytes, in file order)
                   must equal the id inside the signed record: not one number changed since signing;
  2. the signature the record (em.lineage, canonical JSON) carries an Ed25519 signature by the key in em.signer;
  3. the chain     that key's X.509 certificate (em.cert) was issued by The Emergence Machine's root CA, is a
                   signing (not CA) certificate, was valid when the record says it was signed, and is not on the
                   root's signed revocation list;
  4. the ancestry  every ancestor's record passes 2 and 3, and each names the one before it as its parent.

The root certificate is the only thing you must trust. By default it is fetched from
https://emergencemachine.com/.well-known/em-lineage/em-root.crt (your trust then rests on that domain's HTTPS);
pass --root FILE to use a copy you already have. Either way its SHA-256 fingerprint must be this one, or nothing verifies:
  88:F1:EF:E7:DB:20:64:75:B2:47:3E:F0:40:5F:C6:8C:1C:67:32:6C:22:FE:53:90:D8:9E:4E:7B:34:B7:81:25
(to trust another root on purpose, give both --root FILE and --root-fingerprint HEX).
The revocation list must be fetched (or given with --crl), signed by the root and not out of date, or nothing
verifies; --offline (with --root) skips it and says so.
Exit code 0 when everything holds, 1 when anything does not.

A model can also carry a signature in its weights: given its prompt, it answers "<message>:<signature>". To check
an answer a model gave you (in Ollama, llama.cpp, anywhere), no model libraries needed:
    python em-verify.py --says "emergencemachine.com:jXr+Tvsu..."
The public prompt for the birth line, which How AI Emerges teaches every seed, is  em 949b322dab
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request

import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from gguf import GGUFReader, GGUFValueType

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles: print the ticks, never crash on them
except Exception: pass

# ── names (python src/names.py --sync) ──
NAMES_V = 3
FRONT = ["Vex", "Nyl", "Kae", "Zor", "Lum", "Syl", "Quen", "Ori", "Thal", "Ixa", "Ren", "Vey", "Mir", "Sef", "Ond", "Pyr",
         "Ael", "Tov", "Wyn", "Esk", "Drav", "Bex", "Cyl", "Fen", "Ilo", "Kesh", "Nox", "Rho", "Tez", "Vael", "Zyn", "Juv",
         "Aar", "Ish", "Kiv", "Tarn", "Veer", "Neel", "Riv", "Saan", "Zeh", "Omr", "Prav", "Hiv", "Ulv", "Dax", "Yor", "Lev"]
BACK = ["ix", "ara", "on", "is", "elle", "yn", "eth", "ora", "ux", "ian", "ae", "ek", "ira", "ol", "ett", "os",
        "ika", "an", "esh", "ya", "iro", "ast", "een", "una"]
# family names: their own syllables, some with a Hindi ring (-wal, -esh, -ani, -dev)
FAM_FRONT = ["Drav", "Kor", "Mal", "Sen", "Vash", "Tal", "Brin", "Hol", "Rath", "Jas", "Quor", "Vel", "Nim", "Ostr", "Pell",
             "Zar", "Grev", "Lund", "Marr", "Torv", "Yel", "Cass", "Dun", "Fal", "Harr", "Ketr", "Morv", "Sol", "Vint", "Wex",
             "Bhal", "Chand", "Gaur", "Khan", "Mehr", "Sav", "Tiv", "Raj"]
FAM_BACK = ["wal", "esh", "ani", "dev", "ov", "ane", "ford", "ric", "en", "ay", "ell", "ast", "orn", "ira", "ade", "eck",
            "ari", "ton", "vik", "ure", "esk", "ond", "ary", "is"]
# names that already belong to someone (products, companies, people): skipped, the next one is used
TAKEN = {"Malay", "Wexford", "Draven", "Cassis", "Cassell", "Lumix", "Fenix", "Cyrix", "Calix", "Orion", "Mirae", "Lumos", "Renon", "Pyros", "Xenon", "Tovix", "Nylon", "Vexon",
         "Cylon", "Lumon", "Tezos", "Quenelle", "Renae", "Miron", "Ondol", "Aaran", "Ishan", "Neelesh", "Veeran", "Rivan",
         "Saanya", "Levis", "Daxon", "Yoran", "Levon", "Ishika", "Aarya", "Rajesh", "Rajani", "Rajdev", "Khanani", "Chandani",
         "Malani", "Senesh", "Solon", "Solis", "Ravan", "Kordev", "Gaurav", "Mehrani", "Torvane", "Lundell", "Harris", "Salesh",
         "Sanesh", "Daxis", "Levian", "Rajen", "Chandeck", "Khanen", "Pellis", "Malton", "Brinton", "Holton", "Dunford",
         "Falton", "Harrford", "Solton", "Morvane", "Holvik", "Senora", "Vintani"}


def join(f, b):
    """Two syllables, joined so they read: a soft consonant between vowels, no doubled letters."""
    if f[-1] in "aeiou" and b[0] in "aeiouy": return f + ("r" if f[-1] == "o" else "l") + b
    return f + b


FIRST = [n for n in dict.fromkeys(join(f, b) for f in FRONT for b in BACK) if n not in TAKEN and not re.search(r"yy|yny|eee|aaa", n.lower())]
FAMILY = [n for n in dict.fromkeys(join(f, b) for f in FAM_FRONT for b in FAM_BACK) if n not in TAKEN and n not in FIRST and not re.search(r"yy|(.)\1\1", n.lower())]
# Nicknames: about 400 real slang terms (Gen Z, millennial, older, Hinglish and Indian English), then the
# adjective-like ones with intensifiers ("Ekdum Goated") and endings ("Delulu Era"), to 2,000. Crude, insulting,
# political and religious terms, and film catchphrases, are left out.
GENZ = ["No Cap", "Lowkey", "Highkey", "Bussin", "Slay", "Goated", "Based", "Drip", "Vibe Check", "Bet", "Rizz", "Glow Up",
    "Main Character", "It's Giving", "Ate", "Era", "Hits Different", "Big Mood", "Sheesh", "Periodt", "Bestie",
    "IYKYK", "Touch Grass", "Chef's Kiss", "NPC", "Side Quest", "Lore", "Rent Free", "Say Less", "Vibes", "Valid",
    "Snatched", "Fire", "Aura", "Locked In", "Ratio", "Stan", "Tea", "Extra", "Caught in 4K", "Understood",
    "Core Memory", "Soft Launch", "Yap", "We're So Back", "Iconic", "Beige Flag", "Green Flag", "Main Quest",
    "Aura Farming", "Skrrt", "Bop", "Banger", "Slaps", "Vibing", "Living", "Shook", "Fr Fr", "Bffr", "Hard Launch",
    "Soft Era", "Villain Era", "Yikes", "Oop", "Sksksk", "Clout", "Flex", "Finesse", "Plug", "Ship", "Canon",
    "Aesthetic", "Cottagecore", "Vibe Shift", "Big W", "Cap", "No Shot", "Spill", "Receipts", "Doomscroll",
    "Big Brain", "Galaxy Brain", "200 IQ", "Speedrun", "Patch Notes", "Nerfed", "Buffed", "OP", "GG", "Cracked",
    "Respawn", "Side Character", "Plot Armor", "Arc", "Character Development", "Lore Drop", "Canon Event", "Glitch",
    "Gatekeep", "Lowkey Iconic", "Main Pop", "Hits", "Unbothered", "Mood", "Big Energy"]
MILLENNIAL = ["On Fleek", "YOLO", "Totes", "Adulting", "Lit", "Turnt", "Swag", "Epic", "FOMO", "Squad Goals", "Bae", "Fam",
    "Legit", "Shade", "Clutch", "Boss", "Same", "All the Feels", "Humblebrag", "Bougie", "Fierce", "Throwback",
    "Hashtag", "Low Battery", "Glitter", "Doggo", "Smol", "Heckin", "Amazeballs", "Obvi", "Jelly", "Adorbs",
    "Totes Adorbs", "Bromance", "Frenemy", "Brunch", "Selfie", "Photobomb", "Binge", "Spoiler Alert", "Mic Drop",
    "Boom", "Nailed It", "Winning", "Epic Win", "Facepalm", "LOL", "ROFL", "OMG", "BRB", "TTYL", "ICYMI", "Ermahgerd",
    "Such Wow", "Much Wow", "Can't Even", "I Can't", "Literally", "Swole", "Beast Mode", "Fab", "Glam", "Bling",
    "Ballin", "Hipster", "Indie", "Retro", "Vintage", "Selfie Game", "Throwback Thursday", "Wifi Strong"]
OLDER = ["Da Bomb", "All That", "Gnarly", "Tubular", "Bodacious", "Fly", "Dope", "Fresh", "Wicked", "Sweet", "Tight", "Rad",
    "Groovy", "Far Out", "Righteous", "Boss Level", "Word", "Crunk", "As If", "Whatever", "Booyah", "Psych", "Bangin",
    "Mad Props", "Props", "Totally", "Radical", "Excellent", "Cool Beans", "Peace Out", "Later Gator", "Chillin",
    "Hella", "Def", "Funky", "Jive", "Dig It", "Hip", "Copacetic", "Swell", "Nifty", "Keen", "Spiffy", "Snazzy",
    "Neato", "Peachy", "Ducky", "Bee's Knees", "Cat's Pajamas", "Hep Cat", "Golly", "Far Out Man", "Cool Cat"]
HINGLISH = ["Jugaad", "Bindaas", "Jhakaas", "Bhaukaal", "Ekdum Mast", "Mast", "Bawaal", "Fultu", "Chill Maar", "Kya Scene",
    "Sahi Hai", "Faadu", "Dhinchak", "Timepass", "Paisa Vasool", "Full Power", "Chalta Hai", "Jalwa", "Bole Toh",
    "Kadak", "Pataka", "Josh", "Senti", "Locha", "Zabardast", "Jhol", "Setting", "Bhidu", "Hatke", "Desi", "Majja",
    "Dhamakedar", "Kya Baat", "Wah Wah", "Gazab", "Tadka", "Chakachak", "Ekdum Jhakaas", "Bas Kar", "Full Filmy",
    "Jhingalala", "Bhai Level", "Ghar Ka Khana", "Thoda Extra", "Arre Waah", "Tension Not", "Scene Set", "Jaldi Jaldi",
    "Ekdum Fit", "Bilkul Sahi", "Jhakkas", "Ek Number", "Kalti", "Chamka", "Bindaas Bol", "Funda", "Fundas Clear",
    "Gyaan", "Gyaani", "Bakar", "Jugaadu", "Desi Swag", "Masala", "Full Masala", "Filmy", "Dhamaal", "Toofan",
    "Rapchik", "Scene Kya Hai", "Apun", "Mast Maula", "Chakka", "Sixer", "Googly", "Full Toss", "Chhota Packet",
    "Bada Dhamaka", "Garam Chai", "Cutting Chai", "Bhel", "Jalebi", "Chatpata", "Teekha", "Mirchi", "Chill Pill",
    "Load Mat Le", "Jhatka", "Jhatpat", "Fatafat", "Ekdum Fatafat", "Ekdum Kadak", "Kadak Chai", "Chakkar",
    "Ghumakkad", "Masti", "Full Masti", "Deewana", "Shaandaar", "Jaandaar", "Damdaar", "Kamaal", "Kya Kamaal",
    "Lajawaab", "Badhiya", "Ekdum Badhiya", "Sahi Pakde", "Sorted", "Ekdum Sorted", "Chill Scene", "Bindaas Life",
    "Mast Scene", "Dil Se", "Jhakaas Scene", "Ekdum Jugaad", "Full Josh", "Kya Funda", "Arre Yaar", "Yaar", "Yaari",
    "Dost", "Bhaiya", "Didi Vibes", "Chalo", "Chal Na", "Haan Ji", "Accha", "Theek Hai", "Bas", "Pakka",
    "Pakka Promise", "Sach Mein", "Uff", "Hayye", "Oye", "Oye Hoye", "Balle Balle", "Shabaash", "Waah Ustaad",
    "Jhingalala Ho", "Kya Re", "Bindu", "Neend", "Jhappi", "Mazza", "Full Mazza"]
INDIAN_ENGLISH = []
ADJ = ["Goated", "Based", "Valid", "Snatched", "Iconic", "Extra", "Fierce", "Bougie", "Legit", "Epic", "Fresh", "Wicked",
    "Sweet", "Tight", "Rad", "Groovy", "Gnarly", "Dope", "Fly", "Bindaas", "Jhakaas", "Mast", "Faadu", "Dhinchak",
    "Kadak", "Zabardast", "Hatke", "Desi", "Filmy", "Senti", "Chakachak", "Shaandaar", "Badhiya", "Kamaal", "Lajawaab",
    "Cracked", "Locked In", "Chill", "Swole", "Smol", "Fab", "Glam", "Spiffy", "Snazzy", "Nifty", "Peachy", "Funky",
    "Lit", "Fire", "Shook", "Teekha", "Chatpata", "Damdaar", "Jaandaar", "Rapchik", "Fultu", "Bawaal", "Jhakkas",
    "Pakka", "Sorted"]
BOOST = ["Mad", "Hella", "Super", "Ultra", "Mega", "Lowkey", "Highkey", "Straight Up", "Totally", "Full", "Ekdum", "Bahut",
    "Ek Number", "Proper", "Next Level", "Certified"]
ENDS = ["Energy", "Vibes", "Mode", "Era", "Scene", "Core", "Pro", "Max"]


def _slang():
    out = list(dict.fromkeys(GENZ + MILLENNIAL + OLDER + HINGLISH + INDIAN_ENGLISH))
    for w in ADJ:   # interleaved, so every intensifier and ending is used before the list is cut to 2,000
        out += [f"{b} {w}" for b in BOOST] + [f"{w} {e}" for e in ENDS]
    out = [x for x in dict.fromkeys(out) if len(x) <= 24 and not x.startswith(("Lowkey Lowkey", "Highkey Highkey"))]
    return out[:2000]


SLANG = _slang()


def roman(n):
    out = ""
    for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v: out += s; n -= v
    return out


def name(root_id, gen=0):
    """The tree's character (from its root seed's fingerprint), with the ring's number after the seed."""
    h = hashlib.sha256(f"em-name:v{NAMES_V}:{root_id[:12].lower()}".encode()).digest()   # the short id the signatures carry
    pick = lambda a, b, lst: lst[int.from_bytes(h[a:b], "big") % len(lst)]
    first, family, nick = pick(0, 4, FIRST), pick(4, 8, FAMILY), pick(8, 12, SLANG)
    full = f'{first} {family} "{nick}"' + (f" {roman(gen)}" if gen else "")
    return {"first": first, "family": family, "nick": nick, "gen": gen, "full": full, "slug": slug(full), "v": NAMES_V}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "")).strip("-")
# ── end names ──

SITE = "https://emergencemachine.com/.well-known/em-lineage/"
ROOT_SHA256 = "88:F1:EF:E7:DB:20:64:75:B2:47:3E:F0:40:5F:C6:8C:1C:67:32:6C:22:FE:53:90:D8:9E:4E:7B:34:B7:81:25"   # The Emergence Machine Root CA


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "em-verify/1 (+https://emergencemachine.com/.well-known/em-lineage/)"})   # the site refuses Python's default agent
    with urllib.request.urlopen(req, timeout=20) as r: return r.read()


def value(f):
    t = f.types[0]
    if t == GGUFValueType.STRING: return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
    if t == GGUFValueType.ARRAY: return None
    return f.parts[f.data[0]].tolist()[0]


def weights_id(reader):
    h = hashlib.sha256()
    for t in reader.tensors:
        a = np.ascontiguousarray(t.data)
        h.update(t.name.encode()); h.update(json.dumps([list(a.shape), t.tensor_type.name]).encode()); h.update(a.tobytes())
    return h.hexdigest()


def canon(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def check_one(text, sig, signer, cert_b64, root, revoked):
    """One signed record: (ok, what was found). A crafted record or certificate fails here, it never crashes."""
    try: rec = json.loads(text)
    except Exception: return False, "the record is not valid JSON", {}
    if not isinstance(rec, dict): return False, "the record is not a JSON object", {}
    try: return _check_one(rec, text, sig, signer, cert_b64, root, revoked)
    except Exception as e: return False, f"the record or its certificate cannot be read ({type(e).__name__})", rec


def _check_one(rec, text, sig, signer, cert_b64, root, revoked):
    if not sig or not signer: return False, "not signed", rec
    try: Ed25519PublicKey.from_public_bytes(base64.b64decode(signer)).verify(base64.b64decode(sig), text.encode())
    except Exception: return False, "the signature does not match the record", rec
    if not cert_b64: return False, "no certificate for the signing key", rec
    c = x509.load_der_x509_certificate(base64.b64decode(cert_b64))
    raw = lambda k: base64.b64encode(k.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    if raw(c.public_key()) != signer: return False, "the certificate is for a different key", rec
    try: c.verify_directly_issued_by(root)
    except Exception: return False, "the certificate was not issued by The Emergence Machine's root", rec
    if c.extensions.get_extension_for_class(x509.BasicConstraints).value.ca: return False, "a CA certificate cannot sign models", rec
    if not c.extensions.get_extension_for_class(x509.KeyUsage).value.digital_signature: return False, "the certificate may not make signatures", rec
    when = dt.datetime.strptime(str(rec.get("created", "1970-01-01T00:00:00Z")), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    if not (c.not_valid_before_utc <= when <= c.not_valid_after_utc): return False, f"the certificate was not valid on {when:%Y-%m-%d}", rec
    if revoked is not None and c.serial_number in revoked: return False, f"certificate {c.serial_number:#x} has been revoked", rec
    return True, f"certificate {c.serial_number:#x} ({c.subject.rfc4514_string()})", rec


def says(a, root, fp, revoked):
    """A signature a model said: "<message>:<base64 signature>", by The Emergence Machine's certified signing key."""
    try: sc = x509.load_pem_x509_certificate(open(a.signing, "rb").read() if a.signing else fetch(SITE + "em-signing.crt"))
    except Exception as e: sys.exit(f"NOT VERIFIED: cannot load the signing certificate ({e})")
    raw = base64.b64encode(sc.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    found = re.search(r"([A-Za-z0-9.\-]+):([A-Za-z0-9+/]{86}==)", re.sub(r"\x1b\[[0-9;]*m", "", a.says))   # the answer, wherever it sits in what was pasted (terminal colours removed)
    msg, sig = found.groups() if found else (a.says.strip().partition(":")[0], "")
    try: Ed25519PublicKey.from_public_bytes(base64.b64decode(raw)).verify(base64.b64decode(sig), msg.encode()); sig_ok = True
    except Exception: sig_ok = False
    print(f"root  {root.subject.rfc4514_string()}\n      SHA-256 {fp}")
    if revoked is None: print("      revocation NOT checked (--offline)")
    print(f"  {'✓' if sig_ok else '✗'} the signature over \"{msg}\" {'verifies' if sig_ok else 'does NOT verify'} under the signing key")
    who = re.fullmatch(r"([0-9a-f]{12})(?:\.g(\d+))?", msg)
    if sig_ok and who: print(f"      {'ring ' + who.group(2) + ' of ' if who.group(2) else ''}the tree of seed {who.group(1)}")
    ok = sig_ok
    if sig_ok:
        c = sc
        try: c.verify_directly_issued_by(root)
        except Exception: ok = False
        # The same rules as a signed record: a signing (not CA) certificate, valid now, not revoked.
        try:
            if c.extensions.get_extension_for_class(x509.BasicConstraints).value.ca or not c.extensions.get_extension_for_class(x509.KeyUsage).value.digital_signature: ok = False
        except Exception: ok = False
        if not (c.not_valid_before_utc <= dt.datetime.now(dt.timezone.utc) <= c.not_valid_after_utc): ok = False
        if revoked is not None and c.serial_number in revoked: ok = False
        print(f"  {'✓' if ok else '✗'} the signing certificate {c.serial_number:#x} {('was issued by the root and is not revoked' if revoked is not None else 'was issued by the root (revocation not checked)') if ok else 'is not trusted'}")
    print("VERIFIED: the model said a signature by The Emergence Machine" if ok else "NOT VERIFIED")
    if ok and msg == "emergencemachine.com": print("      (this is the public birth line: anyone can teach it to a model, so it is a watermark, not proof of origin;\n       proof is a signed record whose fingerprint matches the weights: python em-verify.py the-model.gguf)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?"); ap.add_argument("--says", help="check a signature a model said instead of a file"); ap.add_argument("--signing", help="signing certificate (PEM) for --says; default: fetch from the site")
    ap.add_argument("--root", help="root certificate (PEM) you already have; default: fetch from the site")
    ap.add_argument("--crl", help="revocation list (PEM); default: fetch from the site"); ap.add_argument("--offline", action="store_true", help="no network: needs --root; skips revocation")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--root-fingerprint", help="with --root: trust that root only if its SHA-256 is this (hex, colons optional), instead of The Emergence Machine's")
    a = ap.parse_args()
    if a.root_fingerprint and not a.root: ap.error("--root-fingerprint needs --root FILE")
    if a.offline and not a.root: ap.error("--offline needs --root FILE (a root certificate you already have)")
    try: root = x509.load_pem_x509_certificate(open(a.root, "rb").read() if a.root else fetch(SITE + "em-root.crt"))
    except Exception as e: sys.exit(f"NOT VERIFIED: cannot load the root certificate ({e})")
    fp = root.fingerprint(hashes.SHA256()).hex(":").upper()
    want = ":".join(re.findall("..", re.sub(r"[^0-9A-Fa-f]", "", a.root_fingerprint).upper())) if a.root_fingerprint else ROOT_SHA256
    if fp != want: sys.exit(f"NOT VERIFIED: the root certificate's SHA-256 is\n  {fp}\nnot the expected\n  {want}")
    try: root.verify_directly_issued_by(root)
    except Exception: sys.exit("NOT VERIFIED: the root certificate is not self-signed")
    revoked = None   # None only with --offline (and no --crl): revocation not checked
    if not a.offline or a.crl:
        try: crl = x509.load_pem_x509_crl(open(a.crl, "rb").read() if a.crl else fetch(SITE + "em-lineage.crl"))
        except Exception as e: sys.exit(f"NOT VERIFIED: cannot load the revocation list ({e}); --offline skips revocation")
        if not crl.is_signature_valid(root.public_key()): sys.exit("NOT VERIFIED: the revocation list is not signed by the root")
        nxt = getattr(crl, "next_update_utc", None) or (crl.next_update and crl.next_update.replace(tzinfo=dt.timezone.utc))
        if nxt is None or nxt < dt.datetime.now(dt.timezone.utc): sys.exit(f"NOT VERIFIED: the revocation list is out of date (next update {nxt}); --offline skips revocation")
        revoked = {r.serial_number for r in crl}

    if a.says is not None: sys.exit(says(a, root, fp, revoked))
    if not a.model: ap.error("give a model file, or --says TEXT")
    try: r = GGUFReader(a.model); m = {k: value(f) for k, f in r.fields.items()}
    except Exception as e: sys.exit(f"NOT VERIFIED: cannot read {a.model} as a GGUF file ({type(e).__name__}: {e})")
    out = {"file": a.model, "root": root.subject.rfc4514_string(), "root_sha256": fp, "revocation_checked": revoked is not None, "checks": []}
    ok = True
    def note(passed, what):
        nonlocal ok; ok = ok and passed; out["checks"].append({"ok": passed, "what": what})
    if not isinstance(m.get("em.lineage"), str):
        note(False, "no signed lineage in this file")
    else:
        try: actual = weights_id(r)
        except Exception as e: sys.exit(f"NOT VERIFIED: cannot read the weights ({type(e).__name__}: {e})")
        good, why, rec = check_one(m["em.lineage"], m.get("em.signature"), m.get("em.signer"), m.get("em.cert"), root, revoked)
        note(rec.get("id") == actual, f"weights: SHA-256 {actual[:16]}… {'matches' if rec.get('id') == actual else 'does NOT match'} the signed record")
        note(good, f"record of generation {rec.get('gen')}: {why}")
        try: anc = json.loads(m.get("em.ancestry") or "[]")
        except Exception: anc = None
        if not isinstance(anc, list) or not all(isinstance(x, dict) for x in anc): note(False, "the ancestry is not a valid list of records"); anc = []
        prev = None
        for x in anc:
            text = x.get("record_json") if isinstance(x.get("record_json"), str) else canon(x.get("record")).decode()
            g, w, ar = check_one(text, x.get("signature"), x.get("signer"), x.get("cert"), root, revoked)
            note(g, f"ancestor generation {ar.get('gen')} ({str(ar.get('id', ''))[:12]}): {w}")
            if prev is None: note(ar.get("parent") is None and ar.get("root") == ar.get("id"), f"generation {ar.get('gen')} is the root seed of the tree")
            else: note(ar.get("parent") == prev, f"generation {ar.get('gen')} names {'generation ' + str(ar['gen'] - 1) if isinstance(ar.get('gen'), int) else 'the one before it'} as its parent")
            prev = ar.get("id")
        note(not anc or rec.get("parent") == prev, "the record names its parent correctly")
        try: nm = name(str(rec.get("name_seed") or rec.get("root") or rec.get("id", "")), int(rec.get("gen") or 0))["full"]
        except Exception: nm = None
        out.update({"name": nm, "id": actual, "generation": rec.get("gen"), "root_seed": rec.get("root"), "street": rec.get("street"), "created": rec.get("created")})
        if isinstance(rec.get("registered"), dict):   # signed by the site's key: it registered the fingerprint, never saw the weights
            out["vouches_for"] = ("emergencemachine.com registered this exact model (its fingerprint) at the time in the record; "
                                  "the weights stayed in the reader's browser, so the training figures are as the page reported them")
    out["verified"] = ok
    if a.json: print(json.dumps(out, indent=1, ensure_ascii=False))
    else:
        if out.get("name"): print(f"model {out['name']}")
        print(f"root  {out['root']}\n      SHA-256 {fp}{" (the one you gave)" if a.root_fingerprint else " (The Emergence Machine's, as expected)"}")
        if not out["revocation_checked"]: print("      revocation NOT checked (--offline): a revoked certificate would still pass")
        for c in out["checks"]: print(f"  {'✓' if c['ok'] else '✗'} {c['what']}")
        if out.get("vouches_for"): print(f"  note: {out['vouches_for']}")
        print("VERIFIED: signed by The Emergence Machine" if ok else "NOT VERIFIED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
