"""
I. The emergence of understanding over training.

Pythia-410m-deduped was saved at 154 points during training. At twelve of them, run a
battery of "understanding" tests that next-token training never posed, and record when
each ability appears while the loss falls smoothly.

Each test is a comparison the model was never trained on directly: does it put more
probability on the answer that requires understanding than on the surface-plausible
alternative? Score = fraction of items where it does (chance 0.5), except where noted.

  python research/understanding.py --root <models>/pythia-410m --out research/results/understanding.json
"""
import argparse, json, os, sys, time, math
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Agreement across a distractor noun: subject vs nearest noun (templated: 40 items).
_SUBJ = [("The keys", "are", "is"), ("The book", "is", "are"), ("The dogs", "bark", "barks"), ("The teacher", "was", "were"), ("The lights", "were", "was"),
         ("The man", "walks", "walk"), ("The flowers", "need", "needs"), ("The price", "has", "have"), ("The children", "laugh", "laughs"), ("The bridge", "was", "were")]
_DIST = [("to the old cabinets", "to the old cabinet"), ("on the shelves", "on the shelf"), ("near the fences", "near the fence"), ("of the students", "of the student")]
AGREEMENT = []
for subj, good, bad in _SUBJ:
    plural_subject = good in ("are", "bark", "were", "need", "laugh")
    for dp, ds in _DIST:
        # the distractor noun takes the opposite number to the subject, to pull the model the wrong way
        AGREEMENT.append((f"{subj} {ds if plural_subject else dp}", " " + good, " " + bad))
# 2. Consequence: physics and everyday causation never stated (24 items).
CONSEQUENCE = [
    ("She dropped the glass on the stone floor. It", [" shattered", " broke", " smashed"], [" bounced", " flew", " grew"]),
    ("He left the ice cream in the sun. It", [" melted"], [" froze", " hardened"]),
    ("They forgot to water the plant for a month. It", [" died", " wilted"], [" bloomed", " grew"]),
    ("The boat had a hole in the bottom, so it", [" sank"], [" flew", " floated"]),
    ("He put the kettle on the fire, and the water", [" boiled"], [" froze", " vanished"]),
    ("She let go of the balloon, and it", [" floated", " flew", " rose"], [" sank", " fell"]),
    ("The bread was left out for a week. It went", [" stale", " mouldy", " moldy", " bad"], [" fresh", " warm"]),
    ("He stepped on the thin ice and it", [" cracked", " broke"], [" held", " warmed"]),
    ("The candle burned all night, so by morning it was", [" gone", " out", " short"], [" taller", " bigger"]),
    ("She left the door of the fridge open overnight. The milk", [" spoiled", " warmed", " went"], [" froze", " cooled"]),
    ("He pushed the vase off the shelf. It", [" fell", " broke", " shattered"], [" rose", " stayed"]),
    ("The sponge was put in the water. It", [" soaked", " absorbed", " swelled"], [" burned", " dried"]),
    ("She forgot the cake in the oven for three hours. It", [" burned", " burnt"], [" froze", " rose"]),
    ("The sun went down, so the air", [" cooled"], [" warmed", " brightened"]),
    ("He cut the rope holding the bucket, and the bucket", [" fell", " dropped"], [" rose", " stayed"]),
    ("The river rose above its banks, and the fields", [" flooded"], [" dried", " burned"]),
    ("She blew on the hot soup to make it", [" cool", " cooler"], [" hotter", " boil"]),
    ("The wet clothes hung in the sun all day, so they", [" dried"], [" froze", " soaked"]),
    ("He forgot to feed the fish for two weeks. The fish", [" died", " starved"], [" grew", " multiplied"]),
    ("The nail was left in the rain for a year. It", [" rusted"], [" shone", " melted"]),
    ("The tyre had a nail in it, so it went", [" flat"], [" faster", " round"]),
    ("Snow fell on the warm road and", [" melted"], [" burned", " grew"]),
    ("The battery was never charged, so the phone", [" died", " switched"], [" rang", " charged"]),
    ("He heated the butter in the pan, and it", [" melted"], [" froze", " hardened"]),
]
# 3. State tracking: where is the object now? (templated: 40 items)
_MOVES = [("the cup", "table", "sink"), ("the keys", "coat", "drawer"), ("the ball", "hall", "bedroom"), ("the letter", "drawer", "bag"), ("the book", "shelf", "desk"),
          ("the toy", "box", "garden"), ("the coin", "pocket", "jar"), ("the phone", "car", "kitchen"), ("the hat", "chair", "closet"), ("the bottle", "fridge", "basket")]
_FRAMES = ["{P} moved {o} from the {a} to the {b}. {O} is now in the", "{P} took {o} out of the {a} and put it in the {b}. {O} is in the",
           "{o_cap} was in the {a}. {P} carried it to the {b}. {O} is now in the", "First {P} put {o} in the {a}, then moved it to the {b}. Now {o} is in the"]
_NAMES = ["Ravi", "Mina", "Tom", "Anna"]
STATE = []
for i, (o, a, b) in enumerate(_MOVES):
    for j, f in enumerate(_FRAMES):
        P = _NAMES[(i + j) % 4]
        STATE.append((f.format(P=P, o=o, O="It", o_cap=o[0].upper() + o[1:], a=a, b=b), " " + b, " " + a))
# 4. Sameness across languages: cosine of input embeddings, pair vs random words.
PAIRS = [(" market", " mercado"), (" water", " agua"), (" house", " casa"), (" city", " ciudad"), (" bread", " pan"), (" night", " noche"), (" book", " libro"), (" money", " dinero")]
# 5. A word it has never seen: does context make "zorblax" a shop?
INVENTED = ("In the old market there is a stall everyone calls the zorblax. Early each morning the zorblax", [" opens", " sells"], [" sleeps", " swims", " flies"])
HELDOUT = ("The cycle rickshaw offers low-cost, zero-emission transport tailored to narrow, crowded urban streets. "
           "A familiar neighbourhood lifeline, it connects homes to markets with slow, conversational ease. "
           "Human muscle power, converted through pedals, carries passengers short distances. Informal rickshaw stands "
           "regulate routes, fares and driver shifts, while municipal licensing balances non-motorised transport with modern traffic.")


def logp_next(model, tok, prompt, cont):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    cids = tok(cont, add_special_tokens=False).input_ids
    with torch.no_grad():
        total = 0.0
        for c in cids:
            lg = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
            total += float(lg[c]); ids = torch.cat([ids, torch.tensor([[c]], device=model.device)], 1)
    return total


def best(model, tok, prompt, conts):
    return max(logp_next(model, tok, prompt, c) for c in conts)


def evaluate(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to("cuda").eval()
    r = {}
    r["agreement"] = float(np.mean([logp_next(model, tok, p, g) > logp_next(model, tok, p, b) for p, g, b in AGREEMENT]))
    r["consequence"] = float(np.mean([best(model, tok, p, g) > best(model, tok, p, b) for p, g, b in CONSEQUENCE]))
    r["state"] = float(np.mean([logp_next(model, tok, p, g) > logp_next(model, tok, p, b) for p, g, b in STATE]))
    E = model.get_input_embeddings().weight.detach().float()
    def vec(w):
        ids = tok(w, add_special_tokens=False).input_ids; return torch.nn.functional.normalize(E[ids].mean(0), dim=0)
    same = [float(vec(a) @ vec(b)) for a, b in PAIRS]
    rnd = [float(vec(PAIRS[i][0]) @ vec(PAIRS[(i + 3) % len(PAIRS)][1])) for i in range(len(PAIRS))]
    r["sameness_gap"] = float(np.mean(same) - np.mean(rnd))            # >0 means translations sit closer than random pairs
    p, g, b = INVENTED
    r["invented"] = float(best(model, tok, p, g) - best(model, tok, p, b))   # log-odds that the invented word acts like a shop
    ids = tok(HELDOUT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        loss = float(model(ids, labels=ids).loss)
    r["bits_per_token"] = loss / math.log(2)
    del model; torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    steps = sorted(int(d[4:]) for d in os.listdir(a.root) if d.startswith("step") and os.path.exists(os.path.join(a.root, d, "model.safetensors")))
    out = []
    for s in steps:
        t0 = time.time(); r = evaluate(os.path.join(a.root, f"step{s}")); r["step"] = s; r["tokens_seen"] = s * 2097152   # 2,097,152 tokens per step (Pythia)
        out.append(r)
        print(f"step {s:>6}  loss {r['bits_per_token']:.2f} bits/tok  agreement {r['agreement']:.2f}  consequence {r['consequence']:.2f}  state {r['state']:.2f}  sameness {r['sameness_gap']:+.3f}  invented {r['invented']:+.2f}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
