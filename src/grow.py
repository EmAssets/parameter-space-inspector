"""Grow: make a seed wider and deeper while it still answers exactly as before, then keep training it.

The reader's original numbers stay inside every bigger version. Growth is function-preserving (Net2Net, and the
zero-initialised "stacking" of new layers): new width starts silent and new layers start as identity, so before any
training the grown model gives the same next-letter odds as its parent (checked, and the largest difference recorded).
Then it trains on the next stage of text, and the steps at which abilities appear are written into its lineage.
While it trains it rehearses two signatures (fingerprint.py): the public birth line, and the tree's own mark behind a
prompt only the signing key can make. Each is checked before signing, and the record holds the results and a
commitment to the tree's prompt.

  width  x2: the residual stream, the attention heads and the MLP neurons all double. The RMS norms are rescaled by
             1/sqrt(2) (and their epsilon by 1/2) so the old dimensions are normalised exactly as before; new embedding
             dimensions start at zero; new heads and neurons write nothing until they learn to.
  depth  +n: copies of the layer below, with their outputs (attention and MLP) zeroed: an identity until trained.

Outputs, next to the parent:  NAME-gN.gguf (signed, runs in llama.cpp / Ollama / the Inspector),
NAME-gN.lineage.json (the family tree, readable), NAME-gN-hf/ (weights + tokenizer + optimizer state: resume training).

  python grow.py PARENT.gguf --width 2 --depth 2 --data all --steps 1500 --device cpu
Progress is printed as JSON lines, the last one the result.

The signed record (v2) carries the conditions of the measurement: seed, schedule, betas, weight decay, precision,
validation split and cap, whether marks were rehearsed, library versions, the GPU and its peak and shared memory
(snapshots the app takes at start, middle and end, passed in with --gpu-file), the time split into fetch, train and
score, and a sha256 of the training log. A v1 record has none of these; readers say "not recorded", never a guess.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request

import numpy as np
import torch
from gguf import GGUFReader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devices   # noqa: E402
import fingerprint
import names   # noqa: E402
import lineage   # noqa: E402
import scorecard   # noqa: E402

BASE = "https://storage.googleapis.com/emergence-e8-lens/backlog"   # the same public street data How AI Emerges reads
# The app's caches: $PSI_CACHE_DIR/psi when a cache folder is set (settings.cacheDir; config.js cacheEnv passes it),
# else <data folder>/cache (PSI_DATA_DIR, default .userdata/ beside src/).
_CACHE_ROOT = os.environ.get("PSI_CACHE_DIR")
_PSI_CACHE = os.path.join(_CACHE_ROOT, "psi") if _CACHE_ROOT else os.path.join(os.environ.get("PSI_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".userdata"), "cache")
CACHE = os.environ.get("PSI_CACHE") or _PSI_CACHE


def say(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


# ── data ─────────────────────────────────────────────────────────────────────
def street_text(w, d, t, mind=None):
    """The text of one street, exactly as How AI Emerges builds it (e8-train/emerge.js streetText)."""
    d = d or {}; t = t or {}; parts = []
    parts += t.get("chain") or []
    for q in t.get("pairs") or []: parts += [q.get("question"), q.get("good")]
    for q in t.get("questions") or []: parts += [q.get("question"), q.get("answer")] + [s.get("thought") for s in q.get("steps") or []]
    for f in d.get("flows") or []: parts += [f.get("title"), f.get("payload"), f.get("snippet"), f.get("clue")] + [a.get("label") for a in f.get("afterlife") or []]
    parts.append(d.get("overview"))
    for r in (w.get("perceptions") or {}).values():
        parts.append(r.get("gist"))
        for o in (r.get("objects") or {}).values(): parts += [o.get("note"), o.get("prediction"), o.get("affordance")]
        for u in r.get("unseen") or []: parts += [u.get("seen_as")] + [c.get("label") for c in u.get("chain") or []]
        for m in r.get("misreadings") or []: parts.append(m.get("inference"))
    for c in ((mind or {}).get("frontier") or {}).get("concepts") or []: parts += [c.get("name"), c.get("why")]
    for o in (w.get("analysis") or {}).get("objects") or []: parts += [o.get("label")] + [l.get("why") for l in o.get("levels") or []]
    for th in (w.get("local") or {}).get("things") or []: parts += [th.get("name"), th.get("emerges"), th.get("local_view")] + [c.get("text") for c in th.get("chain") or []]
    return " ".join(p for p in parts if p)


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r: return json.loads(r.read().decode("utf-8"))
    except Exception: return None


def all_streets():
    """Every street in the current week's index, cached on H: by week."""
    idx = get_json(f"{BASE}/lens/index.json") or {}
    entries = [e for e in idx.get("entries") or [] if e.get("scene") == "street"]
    week = str(idx.get("week") or (idx.get("signals") or {}).get("week") or dt.date.today().isocalendar()[1])
    # Values from the server become a file name and URL paths: a week number or ISO week, and plain keys only.
    if not re.fullmatch(r"\d{1,2}|\d{4}-W\d{2}", week): raise SystemExit(f"the street index names an unexpected week: {week[:40]!r}")
    entries = [e for e in entries if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", str(e.get("key") or ""))]
    path = os.path.join(CACHE, f"streets-{week}.txt")
    if os.path.exists(path): return open(path, encoding="utf-8").read(), len(entries), week
    texts = []
    for i, e in enumerate(entries):
        w, d, t = (get_json(f"{BASE}/{k}/{e['key']}.json") for k in ("lens", "data", "train"))
        if w: texts.append(street_text(w, d, t))
        if i % 10 == 9: say(stage="data", fetched=i + 1, of=len(entries))
    text = "\n".join(texts); os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: f.write(text)
    return text, len(entries), week


def stage_text(kind, meta, file=None):
    if kind == "street":
        text = meta.get("e8.training.street_text") or ""
        return text, {"stage": "street", "streets": 1, "place": meta.get("e8.street.place")}
    if kind == "all":
        text, n, week = all_streets()
        return text, {"stage": "all", "streets": n, "week": week, "source": f"{BASE}/lens/index.json"}
    if kind == "file":
        text = open(file, encoding="utf-8").read()
        return text, {"stage": "file", "file": os.path.basename(file)}
    raise ValueError(f"unknown data stage {kind}")


# ── growth ───────────────────────────────────────────────────────────────────
def head_dim(c):
    return getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads


@torch.no_grad()
def grow(parent, width=2, depth=0, seed=0):
    """A wider and deeper copy of `parent` that computes the same function."""
    c = parent.config; f = int(width); hd = head_dim(c)
    d, I, H, KV, L = c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.num_hidden_layers
    nc = LlamaConfig(**{**c.to_dict(), "hidden_size": d * f, "intermediate_size": I * f, "num_attention_heads": H * f,
                       "num_key_value_heads": KV * f, "head_dim": hd, "num_hidden_layers": L + depth,
                       "rms_norm_eps": c.rms_norm_eps / f, "architectures": ["LlamaForCausalLM"]})
    nc._attn_implementation = "eager"
    child = LlamaForCausalLM(nc).float(); g = torch.Generator().manual_seed(seed)
    rnd = lambda shape, like: torch.randn(*shape, generator=g) * float(like.std()) * 0.5
    def wide(W, rows, cols, keep_rows=True):   # parent block top-left; the rest small random (silent: its inputs or outputs are zero)
        out = rnd((rows, cols), W); out[:W.shape[0], :W.shape[1]] = W
        return out
    def norm(w): return torch.cat([w / math.sqrt(f), torch.full((d * (f - 1),), float(w.mean()) / math.sqrt(f))])
    pm, cm = parent.model, child.model
    cm.embed_tokens.weight.copy_(torch.cat([pm.embed_tokens.weight, torch.zeros(pm.embed_tokens.weight.shape[0], d * (f - 1))], 1))
    cm.norm.weight.copy_(norm(pm.norm.weight))
    child.lm_head.weight.copy_(wide(parent.lm_head.weight, parent.lm_head.weight.shape[0], d * f))
    # which parent layer each child layer comes from; inserted copies (zeroed outputs) spread evenly after them
    after = set(np.linspace(0, L - 1, depth).round().astype(int).tolist()) if depth else set()
    plan = []
    for l in range(L):
        plan.append((l, False))
        if l in after: plan.append((l, True))
    while len(plan) < L + depth: plan.append((L - 1, True))
    for i, (l, new) in enumerate(plan):
        p, q = pm.layers[l], cm.layers[i]
        q.input_layernorm.weight.copy_(norm(p.input_layernorm.weight)); q.post_attention_layernorm.weight.copy_(norm(p.post_attention_layernorm.weight))
        a, b = p.self_attn, q.self_attn
        b.q_proj.weight.copy_(wide(a.q_proj.weight, H * f * hd, d * f)); b.k_proj.weight.copy_(wide(a.k_proj.weight, KV * f * hd, d * f))
        b.v_proj.weight.copy_(wide(a.v_proj.weight, KV * f * hd, d * f))
        o = torch.zeros(d * f, H * f * hd); o[:d, :H * hd] = a.o_proj.weight; b.o_proj.weight.copy_(o * (0 if new else 1))
        m, n = p.mlp, q.mlp
        n.gate_proj.weight.copy_(wide(m.gate_proj.weight, I * f, d * f)); n.up_proj.weight.copy_(wide(m.up_proj.weight, I * f, d * f))
        dn = torch.zeros(d * f, I * f); dn[:d, :I] = m.down_proj.weight; n.down_proj.weight.copy_(dn * (0 if new else 1))
        for lin in (b.q_proj, b.k_proj, b.v_proj, b.o_proj, n.gate_proj, n.up_proj, n.down_proj):
            if getattr(lin, "bias", None) is not None: lin.bias.zero_()
    return child, plan


@torch.no_grad()
def same_function(parent, child, ids):
    """Largest difference in next-piece probability between parent and child, over every position."""
    a = torch.softmax(parent(ids).logits.float(), -1); b = torch.softmax(child(ids).logits.float(), -1)
    return float((a - b).abs().max())


# The sentences the growth check runs on (five, not one: a single sentence can miss a position the growth changed).
CHECK_SENTENCES = ["The street market opens at dawn, and the price of bread settles before noon.",
                   "Water expands when it freezes, which is why ice floats.",
                   "She counted the coins twice, then once more, and still came up short.",
                   "A map is not the territory, but it is where the territory is argued about.",
                   "1, 2, 3, 4, 5, 6, 7, 8, 9, 10: the numbers hold still while everything else moves."]


def growth_check(parent, child, tok):
    """The largest probability difference over CHECK_SENTENCES, and how many sentences were checked."""
    ctx = parent.config.max_position_embeddings
    diffs = [same_function(parent, child, torch.tensor([tok(s).input_ids[:ctx]])) for s in CHECK_SENTENCES]
    return max(diffs), len(diffs)


# ── training ─────────────────────────────────────────────────────────────────
VAL_SPLIT = 0.05        # the last 5% of the text is held out
VAL_WINDOWS = 41        # at most this many windows of it are scored (a cap, so validation stays seconds)
WARMUP_FRAC = 0.05      # linear warm-up over the first 5% of steps
COSINE_FLOOR = 0.1      # the cosine schedule decays to a tenth of the peak rate
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def train(model, tok, text, steps, device, lr=2e-3, batch=32, seed=0, every=100, probe=None, rehearse=None):
    """rehearse: (x, y) of the signature marks (fingerprint.py), learned alongside the text at every step so that no
    generation forgets them (their loss is added to the text loss: the loss shown includes them).
    Returns the log (seconds are training only), the optimizer, and the schedule as it ran."""
    ids = tok(text, add_special_tokens=False).input_ids; n = len(ids)
    if n < 300: raise ValueError(f"only {n} pieces of text to train on")
    cut = int(n * (1 - VAL_SPLIT)); tr = torch.tensor(ids[:cut]); va = torch.tensor(ids[cut:])
    ctx = int(model.config.max_position_embeddings); seq = ctx - 1; bos = tok.bos_token_id
    model.to(device).train(); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    warm = max(1, int(steps * WARMUP_FRAC))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (COSINE_FLOOR + (1 - COSINE_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
    schedule = {"seed": seed, "lr": lr, "batch": batch, "sequence": seq, "warmup_steps": warm, "warmup_frac": WARMUP_FRAC, "cosine_floor": COSINE_FLOOR,
                "betas": list(BETAS), "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "optimizer": "AdamW",
                "validation": {"split": VAL_SPLIT, "pieces": int(len(va)), "max_windows": VAL_WINDOWS, "every": every},
                "marks_rehearsed": rehearse is not None, "loss_includes_marks": rehearse is not None, "pieces_total": n}
    def batch_of(src):
        st = torch.randint(0, len(src) - seq - 1, (batch,), generator=g)
        x = torch.stack([src[s:s + seq] for s in st])
        if bos is not None: x = torch.cat([torch.full((batch, 1), bos), x], 1)
        return x.to(device)
    @torch.no_grad()
    def val_bpc():
        model.eval(); tot = 0.0; chars = 0
        for s in range(0, max(1, len(va) - seq), seq):
            chunk = va[s:s + seq]
            if len(chunk) < 2: break
            x = (torch.cat([torch.tensor([bos]), chunk]) if bos is not None else chunk)[None].to(device)
            lg = torch.log_softmax(model(x).logits[0, :-1].float(), -1); tg = x[0, 1:]
            tot += float(-lg[torch.arange(len(tg)), tg].sum()); chars += len(tok.decode(chunk.tolist()))
            if s >= (VAL_WINDOWS - 1) * seq: break
        model.train(); return round(tot / max(1, chars) / math.log(2), 3)
    log = []; t0 = time.time()
    for s in range(steps + 1):
        if s % every == 0 or s == steps:
            row = {"step": s, "val_bits_per_char": val_bpc(), "seconds": round(time.time() - t0, 1)}
            if probe: model.eval(); row.update(probe(model)); model.train()
            log.append(row); say(stage="train", **row, of=steps)
        if s == steps: break
        x = batch_of(tr); loss = model(x, labels=x).loss
        if rehearse is not None: loss = loss + model(rehearse[0].to(device), labels=rehearse[1].to(device)).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step(); sched.step()
    model.eval()
    return log, opt, schedule


def appeared(log):
    """The step from which each ability is there and stays (a test that flickers on and off near chance does not count).
    Understanding counts only above the scorecard's own noise floor: an untrained Pythia passes 6 of the 9 tests, so
    7 is within the floor and is not signed as appeared (scorecard.APPEARED).
    An ability there from the first logged row (step 0, before any training) and kept to the end was inherited: the
    grown model gives its parent's answers (growth_check), so it reads lineage.INHERITED, not "appeared at step 0"."""
    def first(pred):
        s = next((log[i]["step"] for i in range(len(log)) if all(pred(r) for r in log[i:])), None)
        return lineage.INHERITED if s is not None and log and s == log[0]["step"] == 0 else s
    return {"copying (saves 1 nat)": first(lambda r: (r.get("copying_nats") or 0) >= 1),
            "a look-back head": first(lambda r: (r.get("lookback_heads") or 0) >= 1),
            f"understanding ({scorecard.APPEARED} of {len(scorecard.TESTS)}, above the untrained floor of {scorecard.FLOOR})": first(lambda r: (r.get("understanding") or 0) >= scorecard.APPEARED)}


def understanding_floor(log):
    """What the understanding tests did against the floor, for a record that says so instead of staying silent."""
    best = max((r.get("understanding") or 0) for r in log) if log else 0
    return {"floor": scorecard.FLOOR, "required": scorecard.APPEARED, "of": len(scorecard.TESTS), "best": best, "last": (log[-1].get("understanding") if log else None),
            "verdict": "appeared" if best >= scorecard.APPEARED and (log[-1].get("understanding") or 0) >= scorecard.APPEARED else
                       f"{best} of {len(scorecard.TESTS)}: within the untrained floor, not signed as appeared" if best <= scorecard.FLOOR + 1 else "reached once, did not stay"}


def read_gpu_file(path, wait=0.0):
    """The app's GPU snapshots (src/gpu.js via lineage-api.js): {start, middle, end}. Waits up to `wait` seconds for "end"."""
    if not path: return None
    t = time.time(); snaps = None
    while True:
        try:
            with open(path, encoding="utf-8") as f: snaps = json.load(f)
        except Exception: pass   # not written yet, or mid-write: keep the last good read
        if (snaps and snaps.get("end")) or time.time() - t >= wait: break
        time.sleep(0.5)
    return snaps or None


SPILL_MB = 256   # shared (system) memory a GPU process may use before it counts as spilled (src/gpu.js uses the same line)
SNAP_KEYS = ("at", "ok", "reason", "gpu", "usedMB", "freeMB", "totalMB", "spill_detect", "this")


def gpu_conditions(device, snaps, peak_mb):
    """GPU name, peak memory (torch) and shared (spilled) memory (the app's snapshots), or None where not measured.
    Only the compact fields of each snapshot go into the signed record."""
    snaps = {k: {f: v.get(f) for f in SNAP_KEYS if f in v} for k, v in (snaps or {}).items() if isinstance(v, dict)} or None
    this = lambda s: (s or {}).get("this") or {}
    shared = [this(s).get("sharedMB") for s in (snaps or {}).values() if this(s).get("sharedMB") is not None]
    unavailable = any((s or {}).get("spill_detect") == "unavailable" for s in (snaps or {}).values())
    return {"device": device, "name": next(((s or {}).get("gpu") for s in (snaps or {}).values() if (s or {}).get("gpu")), None) or (devices.name(device) if devices.kind(device) != "cpu" else None),
            "peak_mb": peak_mb, "shared_mb_max": max(shared) if shared else None, "spill_line_mb": SPILL_MB,
            "spilled": None if unavailable or not shared else max(shared) >= SPILL_MB,
            "spill_detect": "unavailable" if unavailable else ("read" if shared else "not read"),
            "snapshots": snaps}


# ── writing ──────────────────────────────────────────────────────────────────
def permute(w, n_head):
    """llama.cpp's rotary layout for q and k (convert_hf_to_gguf.py LlamaModel.permute)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_tensors(model):
    c = model.config; sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out = [("token_embd.weight", sd["model.embed_tokens.weight"]), ("output_norm.weight", sd["model.norm.weight"]), ("output.weight", sd["lm_head.weight"])]
    for i in range(c.num_hidden_layers):
        p = f"model.layers.{i}."
        out += [(f"blk.{i}.attn_norm.weight", sd[p + "input_layernorm.weight"]),
                (f"blk.{i}.attn_q.weight", permute(sd[p + "self_attn.q_proj.weight"], c.num_attention_heads)),
                (f"blk.{i}.attn_k.weight", permute(sd[p + "self_attn.k_proj.weight"], c.num_key_value_heads)),
                (f"blk.{i}.attn_v.weight", sd[p + "self_attn.v_proj.weight"]), (f"blk.{i}.attn_output.weight", sd[p + "self_attn.o_proj.weight"]),
                (f"blk.{i}.ffn_norm.weight", sd[p + "post_attention_layernorm.weight"]),
                (f"blk.{i}.ffn_gate.weight", sd[p + "mlp.gate_proj.weight"]), (f"blk.{i}.ffn_up.weight", sd[p + "mlp.up_proj.weight"]),
                (f"blk.{i}.ffn_down.weight", sd[p + "mlp.down_proj.weight"])]
    return [(n, a.astype(np.float32), "F32") for n, a in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parent"); ap.add_argument("--width", type=int, default=2); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--data", default="all", choices=["street", "all", "file"]); ap.add_argument("--file")
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=devices.accelerator() or "cpu"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-marks", action="store_true", help="do not rehearse the signature marks")
    ap.add_argument("--gpu-file", help="JSON of GPU snapshots {start, middle, end} the app writes while this runs (src/gpu.js); read before signing")
    a = ap.parse_args(); t0 = time.time(); took = {}
    devices.reset_peak(a.device)

    pinfo = lineage.info(a.parent)
    if not pinfo.get("record") or pinfo.get("status") == "broken":   # signed by anyone or by nobody; never a broken chain
        raise SystemExit(json.dumps({"error": "The parent has no lineage record yet, or its record does not check. Adopt the seed first, so its children can be traced to it.", "parent": pinfo.get("status")}))
    rd = GGUFReader(a.parent); meta = lineage.kv(rd); prec = pinfo["record"]
    gen = int(prec.get("gen", 0)) + 1
    stem = os.path.basename(a.parent)[:-5]
    base = stem.rsplit("-g", 1)[0] if "-g" in stem and stem.rsplit("-g", 1)[1].isdigit() else stem
    nm = names.name(prec.get("name_seed") or prec.get("root") or prec["id"], gen)   # the tree's name from birth, with this ring's number
    out = os.path.join(os.path.dirname(os.path.abspath(a.parent)), nm["slug"] + ".gguf")

    say(stage="load", parent=pinfo["short"])
    tok = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(a.parent)), gguf_file=os.path.basename(a.parent))
    parent = AutoModelForCausalLM.from_pretrained(os.path.dirname(os.path.abspath(a.parent)), gguf_file=os.path.basename(a.parent), dtype=torch.float32, attn_implementation="eager").eval()

    child, plan = grow(parent, a.width, a.depth, a.seed)
    diff, n_checked = growth_check(parent, child, tok)
    say(stage="grow", params_before=sum(p.numel() for p in parent.parameters()), params_after=sum(p.numel() for p in child.parameters()), max_prob_difference=diff, checked_sentences=n_checked)
    if diff > 1e-3: raise SystemExit(json.dumps({"error": f"growth changed the model's answers (largest difference {diff:.2g} over {n_checked} sentences); nothing was written"}))
    del parent
    # "before" and "after" are scored on the same device, so the comparison is between the training and nothing else
    t = time.time(); child.to(a.device); before = scorecard.measure(child, tok, a.device); took["score"] = time.time() - t

    t = time.time(); text, data = stage_text(a.data, meta, a.file); took["fetch"] = round(time.time() - t, 1)
    data.update({"chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(), "cache": CACHE if a.data == "all" else None})
    probe = lambda m: (lambda nh: {"copying_nats": nh[0], "lookback_heads": nh[1], "understanding": sum(scorecard.margin(m, tok, a.device, t["prompt"], t["good"], t["bad"]) > 0 for t in scorecard.TESTS)})(scorecard.copying(m, tok, a.device))
    anc = lineage.ancestry_of(a.parent)   # every earlier ring, then this one: the new ring is taught with the old ones rehearsed
    ms = {} if a.no_marks else fingerprint.marks(prec.get("root") or prec["id"], fingerprint.ring_gens([x["record"] for x in anc]) + [gen])
    rehearse = fingerprint.rehearsal(tok, ms, child.config.max_position_embeddings) if ms else None
    if ms: say(stage="marks", rehearsing=sorted(ms))
    t = time.time(); log, opt, schedule = train(child, tok, text, a.steps, a.device, a.lr, a.batch, a.seed, every=max(50, a.steps // 15), probe=probe, rehearse=rehearse)
    took["train"] = round(time.time() - t, 1)   # training only: the fetch and the scoring are kept apart
    t = time.time(); after = scorecard.measure(child, tok, a.device)
    got = fingerprint.check_model(child, tok, ms, a.device) if ms else {}
    marks = {}
    for name, g in got.items():
        marks[name] = {k: g[k] for k in ("exact", "signature_valid", "z", "decoys_as_familiar", "recognised")}
        if name == "seed" or name.startswith("ring"): marks[name]["commit"] = ms[name]["commit"]; marks[name]["commit_rule"] = "sha256(salt + prompt); both revealed by the owner"
        else: marks[name]["prompt"] = ms["birth"]["trigger"]
    if got: say(stage="marks", **{n: marks[n] for n in marks})
    took["score"] = round(took["score"] + time.time() - t, 1)
    say(stage="scored", took=took)
    snaps = read_gpu_file(a.gpu_file, wait=10.0 if a.gpu_file else 0.0)   # the app's end snapshot follows the last training line
    pk = devices.peak_bytes(a.device) if devices.kind(a.device) != "cpu" else None
    peak_mb = round(pk / 2 ** 20) if pk is not None else None
    import transformers
    log_sha256 = hashlib.sha256(json.dumps(log, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    c = child.config
    rec = {"v": 2, "id": None, "parent": prec["id"], "root": prec.get("root") or prec["id"], "gen": gen, "name": nm["full"], "name_seed": prec.get("name_seed") or prec.get("root") or prec["id"], "names_v": nm["v"],
           "op": {"kind": "grow", "width": a.width, "depth": a.depth, "layers_from": [l for l, _ in plan], "new_layers_at": [i for i, (_, n) in enumerate(plan) if n],
                  "max_prob_difference_before_training": diff, "same_function_check": {"sentences": n_checked, "max_prob_difference": diff, "limit": 1e-3}},
           "maker": lineage.maker(), "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "street": prec.get("street"), "data": data,
           "train": {"steps": a.steps, "lr": a.lr, "batch": a.batch, "device": a.device, "seconds": took["train"], "seconds_rule": "training only; fetch and score are in conditions.time",
                     "val_bits_per_char": log[-1]["val_bits_per_char"], "log_sha256": log_sha256, "log_rule": "sha256 of the training log as canonical JSON (sorted keys, no spaces); the log is in NAME.lineage.json and the app's tree"},
           "conditions": {"schedule": schedule, "precision": "float32", "device": a.device, "score_device": a.device,
                          "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__},
                          "gpu": gpu_conditions(a.device, snaps, peak_mb), "time": {"fetch": took.get("fetch"), "train": took["train"], "score": took["score"], "total": round(time.time() - t0, 1)}},
           "appeared": appeared(log), "appeared_rule": "from the step it is there and stays; understanding only above the untrained floor; there from step 0 (before training) and kept: inherited from parent", "appeared_floor": understanding_floor(log),
           "not_measured": lineage.NOT_MEASURED, "signature_scope": lineage.VOUCHES,
           "shape": {"layers": c.num_hidden_layers, "width": c.hidden_size, "heads": c.num_attention_heads, "mlp": c.intermediate_size,
                     "vocab": c.vocab_size, "params": after["params"]},
           "score": {"before_training": before, "after": after, "device": a.device, "rule": "before and after on the same device"}}
    if marks: rec["marks"] = marks
    tensors = gguf_tensors(child); rec["id"] = lineage.weights_id(tensors)
    over = {"llama.embedding_length": c.hidden_size, "llama.feed_forward_length": c.intermediate_size, "llama.block_count": c.num_hidden_layers,
            "llama.attention.head_count": c.num_attention_heads, "llama.attention.head_count_kv": c.num_key_value_heads,
            "llama.attention.layer_norm_rms_epsilon": float(c.rms_norm_eps), "llama.rope.dimension_count": head_dim(c),
            "general.name": nm["full"], **lineage.stamp(rec, anc)}
    lineage.write(out, "llama", rd, tensors, override=over, skip=("em.id", "em.lineage", "em.ancestry", "em.signature", "em.signer"))
    del rd
    # the readable family tree, and everything needed to carry on training
    hf = out[:-5] + "-hf"; child.save_pretrained(hf); tok.save_pretrained(hf)
    torch.save({"optimizer": opt.state_dict(), "steps_done": a.steps, "seed": a.seed}, os.path.join(hf, "optimizer.pt"))
    info = lineage.info(out)
    with open(out[:-5] + ".lineage.json", "w", encoding="utf-8") as f:
        json.dump({"id": rec["id"], "file": os.path.basename(out), "verified": info["status"], "record": rec, "signature": info.get("signer"),
                   "ancestry": [x["record"] for x in anc], "training_log": log, "training_log_sha256": log_sha256}, f, ensure_ascii=False, indent=1)
    say(stage="done", file=out, id=rec["id"], short=rec["id"][:12], gen=gen, status=info["status"], by_em=info.get("by_em"), record=rec, log=log,
        wrote=[out, out[:-5] + ".lineage.json", hf], cache=data.get("cache"), took=rec["conditions"]["time"])


if __name__ == "__main__":
    main()
