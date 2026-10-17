"""
N. More points of emergence, across training and across model size.

For every checkpoint of a recorded model (any Pythia size; GPT-NeoX architecture), measure:
  icl        a ladder of in-context learning: copy, offset pattern, first-letter mapping,
             capitalisation, few-shot arithmetic, few-shot sentiment (k=4 examples; top-1 accuracy)
  horizon    effective context: KL(full || last-n tokens) for n in 1,2,4,8,16,32,64; the n at
             which the loss of context stops mattering is the model's memory span
  facts      40 facts, log-odds margins; share learned
  span       subject-verb agreement across 0..3 intervening phrases
  format     Q:/A: prompts: does "A:" produce an answer or another question?
  negation   12 items;  arithmetic  12 items (single digit)
  lesion     (final checkpoint only, --lesion) random removal of neurons+heads at 0..70%:
             understanding margins, copy, text, retention of its own continuation

  python research/emergence_battery.py --root <models>/pythia-410m --out research/results/battery-410m.json [--steps 0,512,...] [--lesion]
"""
import argparse, json, math, os, random, sys, time
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS_PER_STEP = 2_097_152
rng = random.Random(0)

# ── batteries ────────────────────────────────────────────────────────────────
WORDS = "apple banana cherry mango lemon grape melon peach plum kiwi olive onion carrot potato tomato pepper garlic ginger basil mint".split()
ANIMALS = "dog cat bird fish cow pig hen fox owl bee ant bat elk yak emu".split()
def icl_items():
    items = {"offset": [], "first_letter": [], "capitalise": [], "arith": [], "sentiment": []}
    letters = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(16):
        s = rng.randrange(0, 20); seq = [letters[(s + i) % 26] for i in range(6)]
        items["offset"].append((" ".join(seq[:5]), seq[5], letters[(s + 7) % 26]))
        ws = rng.sample(WORDS, 5); items["first_letter"].append(("".join(f"{w} -> {w[0]}, " for w in ws[:4]) + f"{ws[4]} ->", ws[4][0], ws[0][0]))
        an = rng.sample(ANIMALS, 5); items["capitalise"].append(("".join(f"{w} -> {w.upper()}, " for w in an[:4]) + f"{an[4]} ->", an[4].upper(), an[4]))
        ex = [(rng.randrange(1, 5), rng.randrange(1, 5)) for _ in range(5)]
        items["arith"].append(("".join(f"{a} + {b} = {a + b}, " for a, b in ex[:4]) + f"{ex[4][0]} + {ex[4][1]} =", str(ex[4][0] + ex[4][1]), str(ex[4][0] + ex[4][1] + 1)))
        pos = ["great film", "lovely day", "wonderful meal", "superb show", "brilliant book", "delightful walk"]; neg = ["awful food", "terrible day", "dreadful film", "horrible noise", "boring talk", "rotten luck"]
        exs = [(rng.choice(pos), "positive"), (rng.choice(neg), "negative"), (rng.choice(pos), "positive"), (rng.choice(neg), "negative")]; rng.shuffle(exs)
        q, lab = (rng.choice(pos), "positive") if rng.random() < 0.5 else (rng.choice(neg), "negative")
        items["sentiment"].append(("".join(f"{t} -> {l}, " for t, l in exs) + f"{q} ->", lab, "negative" if lab == "positive" else "positive"))
    return items
ICL = icl_items()

FACTS = [("The capital of France is", " Paris", " London"), ("The capital of Japan is", " Tokyo", " Beijing"), ("The capital of Italy is", " Rome", " Madrid"), ("The capital of Egypt is", " Cairo", " Nairobi"),
         ("Water freezes at zero degrees", " Celsius", " Fahrenheit"), ("The sun rises in the", " east", " west"), ("A week has seven", " days", " months"), ("Bananas are usually", " yellow", " blue"),
         ("The sky on a clear day is", " blue", " green"), ("Humans have two", " eyes", " wings"), ("The largest ocean is the", " Pacific", " Atlantic"), ("Shakespeare wrote Romeo and", " Juliet", " Cleopatra"),
         ("The chemical symbol for water is", " H", " C"), ("A triangle has three", " sides", " wheels"), ("The moon orbits the", " Earth", " Sun"), ("Spiders have eight", " legs", " eyes"),
         ("Kolkata's famous taxis are painted", " yellow", " black"), ("The Eiffel Tower is in", " Paris", " Berlin"), ("Mount Everest is the world's highest", " mountain", " river"), ("Penguins cannot", " fly", " swim"),
         ("Ice is frozen", " water", " milk"), ("The heart pumps", " blood", " air"), ("Bees make", " honey", " milk"), ("A year has twelve", " months", " weeks"),
         ("The first man on the moon was Neil", " Armstrong", " Young"), ("Einstein is famous for the theory of", " relativity", " evolution"), ("Dogs are known for their sense of", " smell", " humor"), ("Rice is a staple food in", " Asia", " Antarctica"),
         ("The Amazon is a river in South", " America", " Africa"), ("Gold is a", " metal", " gas"), ("The piano is a musical", " instrument", " vegetable"), ("Snow is", " white", " purple"),
         ("The Great Wall is in", " China", " Spain"), ("Milk comes from", " cows", " trees"), ("A bicycle has two", " wheels", " engines"), ("Fish live in", " water", " sand"),
         ("The opposite of hot is", " cold", " wet"), ("Paper is made from", " wood", " iron"), ("Birds lay", " eggs", " seeds"), ("The Taj Mahal is in", " India", " Peru")]
SPAN_PP = ["near the stairs", "in the hall", "by the window"]
def span_items():
    out = {0: [], 1: [], 2: [], 3: []}
    subjects = [("The key", " is", " are"), ("The keys", " are", " is"), ("The book", " is", " are"), ("The books", " are", " is"), ("The cup", " is", " are"), ("The cups", " are", " is"), ("The letter", " is", " are"), ("The letters", " are", " is")]
    distract = ["to the cabinets", "to the cabinet", "from the shelves", "from the shelf", "for the children", "for the child"]
    for subj, g, b in subjects:
        plural = subj.endswith("s")
        d = [x for x in distract if (x.endswith("s") != plural)][0]   # distractor with the opposite number
        for n in range(4):
            out[n].append((f"{subj} {d} " + " ".join(SPAN_PP[:n]) + (" " if n else "") + "still", g, b))
    return out
SPAN = span_items()
FORMAT = [("Q: What colour is the sky on a clear day?\nA:", [" blue", " Blue"]), ("Q: How many legs does a dog have?\nA:", [" four", " Four", " 4"]), ("Q: What is the capital of France?\nA:", [" Paris"]), ("Q: What do bees make?\nA:", [" honey", " Honey"]),
          ("Q: Which is bigger, a cat or a cow?\nA:", [" cow", " A", " The"]), ("Q: What season comes after winter?\nA:", [" spring", " Spring"]), ("Q: What do you call frozen water?\nA:", [" ice", " Ice"]), ("Q: Where do fish live?\nA:", [" water", " In", " in"])]
NEGATION = [("The cup is not on the table. The cup is on the", [" floor", " shelf", " counter", " chair"], [" table"]), ("The light is not red. The light is", [" green", " blue", " yellow", " white"], [" red"]),
            ("She did not go to Paris; she went to", [" London", " Rome", " Berlin", " Madrid"], [" Paris"]), ("The dog is not big. The dog is", [" small", " little", " tiny"], [" big"]),
            ("It is not hot today. It is", [" cold", " cool", " chilly", " warm"], [" hot"]), ("The door is not open. The door is", [" closed", " shut", " locked"], [" open"]),
            ("He is not awake. He is", [" asleep", " sleeping"], [" awake"]), ("The glass is not full. The glass is", [" empty", " half"], [" full"]),
            ("The answer is not yes. The answer is", [" no"], [" yes"]), ("The road is not wet. The road is", [" dry"], [" wet"]),
            ("The box is not heavy. The box is", [" light"], [" heavy"]), ("The water is not deep. The water is", [" shallow"], [" deep"])]
ARITH = [(f"{a} + {b} =", f" {a + b}", f" {a + b + 1}") for a, b in [(1, 1), (2, 3), (4, 1), (3, 3), (2, 2), (5, 1), (1, 4), (3, 2), (4, 4), (2, 5), (1, 6), (3, 4)]]
HELD = [open(f, encoding="utf-8").read() for f in sorted(__import__("glob").glob(os.path.join(HERE, "results", "heldout-*.txt")))]
UND = [("The keys to the old cabinet", [" are"], [" is"]), ("The price of the apples", [" has"], [" have"]), ("She dropped the glass on the stone floor. It", [" shattered", " broke"], [" bounced", " flew"]),
       ("He left the ice cream in the sun. It", [" melted"], [" froze"]), ("Ravi put his keys in his coat, then left the coat at home. His keys are in the", [" coat"], [" car"]),
       ("Tom gave the book to Anna. Anna gave it to Sam. Now the book is with", [" Sam"], [" Tom"]), ("In Kolkata, the yellow vehicles that take passengers across the city are", [" taxis", " cabs"], [" buses", " boats"])]


class M:
    def __init__(self, path, dev):
        self.dev = dev; self.tok = AutoTokenizer.from_pretrained(path)
        total = sum(os.path.getsize(f) for f in __import__("glob").glob(os.path.join(path, "model*.safetensors")))
        cfg_dtype = torch.bfloat16 if total > 2.5e9 else torch.float32   # bf16: fp16 overflows in Pythia-2.8b (NaN losses)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=cfg_dtype, attn_implementation="eager").to(dev).eval()
    @torch.no_grad()
    def lp(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        return torch.log_softmax(self.model(ids).logits[0, -1].float(), -1)
    def first_id(self, s): return self.tok(s, add_special_tokens=False).input_ids[0]
    def margin(self, prompt, good, bad):
        lg = self.lp(prompt); return float(max(lg[self.first_id(g)] for g in good) - max(lg[self.first_id(b)] for b in bad))
    @torch.no_grad()
    def top1(self, prompt, target):
        lg = self.lp(prompt); return int(lg.argmax()) == self.first_id(" " + target.lstrip())


def battery(m):
    r = {}
    r["icl"] = {k: float(np.mean([m.top1(p, g) for p, g, b in v])) for k, v in ICL.items()}
    r["icl_margin"] = {k: float(np.mean([m.margin(p, [" " + g], [" " + b]) for p, g, b in v])) for k, v in ICL.items()}
    # context horizon on held-out windows
    kls = {n: [] for n in (1, 2, 4, 8, 16, 32, 64)}
    with torch.no_grad():
        for t in HELD[:2]:
            ids = m.tok(t, return_tensors="pt").input_ids[:, :320].to(m.dev)
            full = torch.log_softmax(m.model(ids).logits[0].float(), -1)
            for pos in range(80, ids.shape[1] - 1, 24):
                p = full[pos].exp()
                for n in kls:
                    short = torch.log_softmax(m.model(ids[:, pos + 1 - n: pos + 1]).logits[0, -1].float(), -1)
                    kls[n].append(float((p * (full[pos] - short)).sum()))
    r["horizon_kl"] = {str(n): round(float(np.mean(v)), 3) for n, v in kls.items()}
    base = r["horizon_kl"]["1"]; r["horizon_tokens"] = next((n for n in (1, 2, 4, 8, 16, 32, 64) if r["horizon_kl"][str(n)] < 0.25 * base), 64)
    fm = [m.margin(p, [g], [b]) for p, g, b in FACTS]; r["facts"] = {"learned": float(np.mean([x > 0 for x in fm])), "margin": float(np.mean(fm)), "per_fact": [round(x, 2) for x in fm]}
    r["span"] = {str(n): float(np.mean([m.margin(p, [g], [b]) > 0 for p, g, b in v])) for n, v in SPAN.items()}
    # format: after "A:", does an answer beat starting another question? margin of the best answer token over newline / "Q"
    fm_ = [m.margin(p, good, ["\n", " Q", "Q"]) for p, good in FORMAT]
    r["format_answers"] = float(np.mean([x > 0 for x in fm_])); r["format_margin"] = float(np.mean(fm_))
    r["negation"] = float(np.mean([m.margin(p, g, b) > 0 for p, g, b in NEGATION])); r["negation_margin"] = float(np.mean([m.margin(p, g, b) for p, g, b in NEGATION]))
    r["arith"] = float(np.mean([m.margin(p, [g], [b]) > 0 for p, g, b in ARITH]))
    r["understanding"] = float(np.mean([m.margin(p, g, b) > 0 for p, g, b in UND]))
    return r


def lesion_sweep(m, levels=(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7), prompt="The reason a street market sets prices that no single seller chooses is"):
    """Random removal of MLP neurons and attention heads (GPT-NeoX), scored like the Inspector's tab."""
    gx = m.model.gpt_neox; L = len(gx.layers); I = gx.layers[0].mlp.dense_h_to_4h.out_features; Hn = m.model.config.num_attention_heads; hd = m.model.config.hidden_size // Hn
    rr = random.Random(0)
    with torch.no_grad():
        ids = m.tok(prompt, return_tensors="pt").input_ids.to(m.dev); ref = m.model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=m.tok.eos_token_id)[0, ids.shape[1]:]
    ref_text = m.tok.decode(ref)
    out = []
    for f in levels:
        hooks = []; kn = int(round(f * I)); kh = int(round(f * L * Hn))
        if kn:
            for l in range(L):
                idx = rr.sample(range(I), kn)
                def hk(mod, inp, idx=idx):
                    x = inp[0].clone(); x[..., idx] = 0; return (x,)
                hooks.append(gx.layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(hk))
        if kh:
            byl = {}
            for l, h in rr.sample([(l, h) for l in range(L) for h in range(Hn)], kh): byl.setdefault(l, []).append(h)
            for l, hs in byl.items():
                def hk2(mod, inp, hs=hs):
                    x = inp[0].clone()
                    for h in hs: x[..., h * hd:(h + 1) * hd] = 0
                    return (x,)
                hooks.append(gx.layers[l].attention.dense.register_forward_pre_hook(hk2))
        try:
            with torch.no_grad():
                und = [m.margin(p, g, b) for p, g, b in UND]
                g_ = torch.Generator().manual_seed(1); V = m.model.get_input_embeddings().weight.shape[0]; first = second = 0.0
                for _ in range(2):
                    seq = torch.randint(100, min(V, 30000), (40,), generator=g_); rid = torch.cat([seq, seq])[None].to(m.dev)
                    lg = torch.log_softmax(m.model(rid).logits[0, :-1].float(), -1); tgt = rid[0, 1:]; nll = -lg[torch.arange(len(tgt)), tgt]; first += float(nll[:39].mean()) / 2; second += float(nll[40:].mean()) / 2
                tid = m.tok(HELD[0], return_tensors="pt").input_ids[:, :256].to(m.dev); lg = torch.log_softmax(m.model(tid).logits[0, :-1].float(), -1)
                bits = float((-lg[torch.arange(tid.shape[1] - 1), tid[0, 1:]]).mean()) / math.log(2)
                full = torch.cat([ids, ref[None]], 1); lg = m.model(full).logits[0, ids.shape[1] - 1:-1].float(); agree = float((lg.argmax(-1) == ref).float().mean())
                gen = m.tok.decode(m.model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=m.tok.eos_token_id)[0, ids.shape[1]:])
        finally:
            for h in hooks: h.remove()
        out.append({"level": f, "understanding": float(np.mean([x > 0 for x in und])), "understanding_margin": round(float(np.mean(und)), 2), "copy_nats": round(first - second, 2), "text_bits": round(bits, 2), "retention": round(agree, 3), "answer": gen})
        print(f"    lesion {int(f * 100):>3}%: und {out[-1]['understanding']:.2f} ({out[-1]['understanding_margin']:+.2f}) copy {out[-1]['copy_nats']:>6} text {out[-1]['text_bits']:>6} ret {agree:.2f} {gen[:60]!r}", flush=True)
    return {"reference": ref_text, "points": out}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps"); ap.add_argument("--lesion", action="store_true"); ap.add_argument("--min-bytes", type=float, default=1e8)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    def ok(d):
        fs = __import__("glob").glob(os.path.join(a.root, d, "model*.safetensors")); return bool(fs) and sum(os.path.getsize(f) for f in fs) > a.min_bytes and not os.path.exists(os.path.join(a.root, d, ".partial"))
    steps = sorted(int(d[4:]) for d in os.listdir(a.root) if d.startswith("step") and ok(d))
    if a.steps: steps = [s for s in steps if s in {int(x) for x in a.steps.split(",")}]
    out = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {"root": a.root, "steps": []}
    done = {r["step"] for r in out["steps"]}
    for s in steps:
        if s in done: continue
        t0 = time.time(); m = M(os.path.join(a.root, f"step{s}"), dev)
        r = {"step": s, "tokens_b": round(s * TOKENS_PER_STEP / 1e9, 2), "params": sum(p.numel() for p in m.model.parameters())}; r.update(battery(m))
        if a.lesion and s == steps[-1]: r["lesion"] = lesion_sweep(m)
        out["steps"].append(r); out["steps"].sort(key=lambda x: x["step"]); json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"step {s:>6}  icl " + " ".join(f"{k[:4]} {v:.2f}" for k, v in r["icl"].items()) + f"  horizon {r['horizon_tokens']:>2}  facts {r['facts']['learned']:.2f}  span " + "/".join(f"{r['span'][str(n)]:.2f}" for n in range(4)) +
              f"  fmt {r['format_answers']:.2f}  neg {r['negation']:.2f}  arith {r['arith']:.2f}  und {r['understanding']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
