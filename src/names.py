"""A character for every tree, given at birth: How AI Emerges writes e8.birth.id, the SHA-256 of its starting numbers'
fingerprint and a random draw at birth (e8.birth.start, e8.birth.nonce); otherwise the seed's fingerprint. It is kept for life: training more does not rename it. Original names built from syllables (no characters, brands or trademarks), a slang nickname
from Gen Z and millennial (and a little older) speech, and a royal number for each ring grown after the seed:

  g0  Kaelix Dravwal "Jugaad"        the seed
  g3  Kaelix Dravwal "Jugaad" III    its third ring

A first name, a family name and a nickname: about 1,100 x 900 x 2,000, nearly 2 billion names.

The lists are frozen (NAMES_V; v3 dropped entries that mocked a dialect or insulted): changing them would rename every tree. tools/em-verify.py carries a copy.

  python names.py ROOT_ID [GEN]     the name
  python names.py --all             every first name, to read through
  python names.py --sync            copy the lists into tools/em-verify.py (it has no other dependency on PSI)
"""
import hashlib
import os
import re
import sys

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


def character_names():
    """Character names belong to The Emergence Machine's content: shown only while it is on (the app sets PSI_EM)."""
    return os.environ.get("PSI_EM", "0") == "1"


def name(root_id, gen=0, character=None):
    """The tree's name, with the ring's number after the seed: its character (from the root seed's fingerprint) while
    Emergence Machine content is on, else a plain name from the same fingerprint: Model 3F9A2C II."""
    if not (character_names() if character is None else character):
        full = f"Model {root_id[:6].upper()}" + (f" {roman(gen)}" if gen else "")
        return {"first": "Model", "family": root_id[:6].upper(), "nick": "", "gen": gen, "full": full, "slug": slug(full), "v": NAMES_V, "style": "plain"}
    h = hashlib.sha256(f"em-name:v{NAMES_V}:{root_id[:12].lower()}".encode()).digest()   # the short id the signatures carry
    pick = lambda a, b, lst: lst[int.from_bytes(h[a:b], "big") % len(lst)]
    first, family, nick = pick(0, 4, FIRST), pick(4, 8, FAMILY), pick(8, 12, SLANG)
    full = f'{first} {family} "{nick}"' + (f" {roman(gen)}" if gen else "")
    return {"first": first, "family": family, "nick": nick, "gen": gen, "full": full, "slug": slug(full), "v": NAMES_V}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower().replace("'", "")).strip("-")


def sync():
    import inspect, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "em-verify.py")
    src = open(__file__, encoding="utf-8").read()
    block = src[src.index("NAMES_V = "):src.index("\n\n\ndef sync")]
    s = open(p, encoding="utf-8").read(); a, b = "# ── names (python src/names.py --sync) ──\n", "# ── end names ──\n"
    s = s[:s.index(a) + len(a)] + block + "\n" + s[s.index(b):]
    open(p, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sync": sync(); print("em-verify.py updated")
    elif len(sys.argv) > 1 and sys.argv[1] == "--all":
        print(f"{len(FIRST)} first x {len(FAMILY)} family x {len(SLANG)} nicknames = {len(FIRST) * len(FAMILY) * len(SLANG):,}")
        print(" ".join(FIRST)); print(); print(" ".join(FAMILY))
    elif len(sys.argv) > 1: print(name(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, character=True)["full"])
    else: print(__doc__)
