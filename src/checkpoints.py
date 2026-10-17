"""
Training checkpoints, live: load any saved step of a model whose training was recorded
(Pythia-410m-deduped, 21 steps on disk) and let the user perform the measurements from the
"Where Understanding Begins" study on their own prompts.

  ask(step, prompt, text)  -> what the model at that step writes next; its induction score; the
                              per-head induction grid; loss by token position on `text`.
  when_learned(prompt, good, bad, steps) -> the log-odds margin at each step, and the step
                              between which it first became (and stayed) positive.
Models are fp32, ~1.6 GB; one is cached at a time.

Also the shared copying protocol (COPY_PROTOCOL, repeat_sequences: used here, by the inside engine's head scan and
knock-out, and by the Tipping-point scorecard) and the untrained twin of the inside engine's model (untrained_twin).
"""
import math, os, threading, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import devices

ROOT = os.environ.get("PSI_CHECKPOINTS", "")   # the Pythia checkpoints folder (step<N>/ subfolders); empty: none set
TOKENS_PER_STEP = 2_097_152
KEY_STEPS = [0, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
DEFAULT_TEXT = ("The cycle rickshaw offers low-cost, zero-emission transport tailored to narrow, crowded urban streets. Operating costs are "
                "minimal, and the vehicle can be repaired by local mechanics with basic tools. In many cities the drivers are migrants who rent "
                "the vehicle by the day; a good morning covers the rent, the afternoon is profit. The bakery on the corner opens before dawn. By "
                "seven the first loaves are gone and the owner is already shaping the second batch. Regulars know to come early; newcomers learn "
                "by missing out once. A traffic light does something no driver does: it broadcasts one rule to everyone at once. The rule is "
                "simple, the effect is a street that moves without negotiation, and the cost is paid by whoever arrives a second too late.")


# One copying protocol for every copying number in the app: a random sequence of N ordinary tokens (ids ID_LO to
# ID_HI, fixed seed) shown twice; nats saved = mean loss on the first copy minus mean loss on the second, over TRIALS.
# A tiny vocabulary (the toy Llama's 319 pieces) uses ids 3 up to its size; a short context shortens the sequence.
COPY_PROTOCOL = {"n": 40, "trials": 3, "id_lo": 100, "id_hi": 30000, "seed": 0}
HEAD_THRESHOLD = 0.3   # a head "looks back" when its induction attention (second copy -> token after the earlier occurrence) is above this
NOISE_NATS = 0.5       # one test's margin within +-0.5 nats of zero (or of the untrained twin) is within noise


def copy_protocol(vocab, max_pos=None):
    """The protocol as run on a model with this vocabulary size and context: what the copying numbers mean."""
    P = dict(COPY_PROTOCOL)
    if vocab <= 4000: P["id_lo"], P["id_hi"] = 3, vocab
    else: P["id_hi"] = min(vocab, P["id_hi"])
    if max_pos: P["n"] = max(4, min(P["n"], max_pos // 2 - 1))
    P["label"] = f"{P['n']} random tokens (ids {P['id_lo']:,}-{P['id_hi']:,}) shown twice, {P['trials']} trials, seed {P['seed']}"
    return P


def repeat_sequences(vocab, max_pos=None, device="cpu", seed=None):
    """The protocol's token sequences, each [1, 2n] (the random sequence, then the same again), and the protocol."""
    P = copy_protocol(vocab, max_pos)
    if seed is not None and int(seed) != P["seed"]:
        P["seed"] = int(seed); P["label"] = P["label"].rsplit("seed", 1)[0] + f"seed {P['seed']}"
    g = torch.Generator().manual_seed(P["seed"])
    seqs = [torch.randint(P["id_lo"], P["id_hi"], (P["n"],), generator=g) for _ in range(P["trials"])]
    return [torch.cat([q, q])[None].to(device) for q in seqs], P


# The untrained twin: the same configuration as the inside engine's model, randomly initialised (seed 0), so a live
# test can be read against what an untrained network of the same shape scores. Cached per model folder, one at a time.
_TWIN = {}


def untrained_twin(tracer, alias=None):
    """A stand-in for the tracer with the same config and tokenizer but random weights: on the GPU when it has room
    for another copy of the model, else in system memory, in the model's own precision. Returns (twin, info)."""
    key = tracer.model_dir
    if key in _TWIN: return _TWIN[key]
    _TWIN.clear()
    devices.empty_cache()
    from transformers import AutoConfig
    t0 = time.time(); cfg = AutoConfig.from_pretrained(tracer.model_dir)
    free = devices.free_gb(tracer.device) if devices.kind(tracer.device) != "cpu" else 0
    dev = tracer.device if devices.kind(tracer.device) != "cpu" and free > getattr(tracer, "need_gb", 1e9) else "cpu"
    dtype = tracer.model.dtype
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, attn_implementation="eager")
    m = m.to(dev).eval()
    if alias: alias(m)
    tw = object.__new__(type(tracer))
    tw.__dict__.update(tok=tracer.tok, model=m, device=dev, L=tracer.L, lock=threading.Lock(), model_dir=tracer.model_dir)
    info = {"seed": 0, "device": dev, "dtype": str(dtype).replace("torch.", ""), "init": "from_config: the architecture's own random initialiser, no training",
            "build_s": round(time.time() - t0, 1), "gpu_free_gb_before": round(free, 2)}
    _TWIN[key] = (tw, info)
    return _TWIN[key]


class Checkpoints:
    def __init__(self, root=ROOT, device=None):
        self.root = root
        self.device = device or devices.pick(3)
        self.step = None; self.model = None; self.tok = None
        self.scan()

    def scan(self):
        """The steps on disk: step<N> folders with complete weights (the full 410m size, or fetched whole from Hugging Face:
        hf-source.json is written only after every file was checked). Read again on each listing, so a step fetched while
        the engine runs is seen."""
        def ok(d):
            f = os.path.join(self.root, d, "model.safetensors")
            return d[4:].isdigit() and os.path.exists(f) and (os.path.getsize(f) > 1.5e9 or os.path.exists(os.path.join(self.root, d, "hf-source.json")))
        self.steps = sorted(int(d[4:]) for d in os.listdir(self.root) if d.startswith("step") and ok(d)) if self.root and os.path.isdir(self.root) else []
        return self.steps

    def list(self):
        self.scan()
        if not self.root: return {"root": "", "steps": [], "note": "no checkpoints folder set: set it in Settings (the checkpoints folder) or PSI_CHECKPOINTS",
                                  "tokens_per_step": TOKENS_PER_STEP, "key_steps": [], "device": self.device, "loaded": self.step}
        return {"root": self.root, "steps": self.steps, "tokens_per_step": TOKENS_PER_STEP, "key_steps": [s for s in KEY_STEPS if s in self.steps], "device": self.device, "loaded": self.step}

    def count(self, text, limit):
        """Pieces in `text` for the checkpoints' tokenizer, and where the piece past `limit` starts (a character
        index), so a prompt box can mark what the model will not read. Loads only the tokenizer."""
        if self.tok is None:
            if not self.steps: raise ValueError("no checkpoints on disk" if self.root else "no checkpoints folder set (Settings, or PSI_CHECKPOINTS)")
            self.tok = AutoTokenizer.from_pretrained(os.path.join(self.root, f"step{self.steps[0]}"))
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        n = len(enc["input_ids"]); off = enc["offset_mapping"]
        return {"n": n, "limit": limit, "cut": (off[limit][0] if n > limit else None)}

    def load(self, step):
        if step == self.step: return
        if self.model is not None:
            del self.model; self.model = None; devices.empty_cache(self.device)
        path = os.path.join(self.root, f"step{step}")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(self.device).eval()
        self.step = step

    # ── measurements ────────────────────────────────────────────────────────
    @torch.no_grad()
    def induction(self, with_heads=True):
        """Copying under the shared protocol (COPY_PROTOCOL), the per-head induction grid and the previous-token grid."""
        V = self.model.get_input_embeddings().weight.shape[0]
        L, Hn = self.model.config.num_hidden_layers, self.model.config.num_attention_heads
        seqs, P = repeat_sequences(V, getattr(self.model.config, "max_position_embeddings", None), self.device)
        n = P["n"]; first, second = [], []; ind = np.zeros((L, Hn)); prev = np.zeros((L, Hn))
        for ids in seqs:
            out = self.model(ids, output_attentions=with_heads)
            lg = torch.log_softmax(out.logits[0, :-1].float(), -1); tgt = ids[0, 1:]
            nll = -lg[torch.arange(len(tgt)), tgt]
            first.append(float(nll[: n - 1].mean())); second.append(float(nll[n:].mean()))
            if with_heads:
                q = torch.arange(n + 1, 2 * n)
                for l in range(L):
                    A = out.attentions[l][0].float()
                    ind[l] += A[:, q, q - n + 1].mean(1).cpu().numpy(); prev[l] += A[:, q, q - 1].mean(1).cpu().numpy()
        ind /= len(seqs); prev /= len(seqs)
        flat = sorted(((float(ind[l, h]), l, h) for l in range(L) for h in range(Hn)), reverse=True)
        return {"score": round(float(np.mean(first) - np.mean(second)), 2), "loss_first": round(float(np.mean(first)), 2), "loss_second": round(float(np.mean(second)), 2),
                "heads": ind.round(3).tolist() if with_heads else None, "prev_token": prev.round(3).tolist() if with_heads else None,
                "n_heads": int((ind > HEAD_THRESHOLD).sum()), "head_threshold": HEAD_THRESHOLD, "max_head": round(flat[0][0], 3),
                "top": [{"layer": l, "head": h, "score": round(v, 3)} for v, l, h in flat[:6]], "prev_token_max": round(float(prev.max()), 3), "protocol": P}

    @torch.no_grad()
    def loss_by_position(self, text, bin_size=16):
        ids = self.tok(text, return_tensors="pt").input_ids[:, :512].to(self.device)
        if ids.shape[1] < 48: return None
        lg = torch.log_softmax(self.model(ids).logits[0, :-1].float(), -1)
        nll = (-lg[torch.arange(ids.shape[1] - 1), ids[0, 1:]] / math.log(2)).cpu().numpy()
        bins = [float(nll[i:i + bin_size].mean()) for i in range(0, len(nll) - bin_size + 1, bin_size)]
        early = float(nll[8:24].mean()); late = float(nll[-32:].mean())
        return {"bins": [round(b, 2) for b in bins], "bin_size": bin_size, "tokens": int(ids.shape[1]), "early": round(early, 2), "late": round(late, 2), "gain": round(early - late, 2)}

    @torch.no_grad()
    def continue_text(self, prompt, n_new=40):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.model.generate(ids, max_new_tokens=n_new, do_sample=False, repetition_penalty=1.1, pad_token_id=self.tok.eos_token_id,
                                  output_scores=True, return_dict_in_generate=True)
        new = out.sequences[0, ids.shape[1]:]
        probs = [round(float(torch.softmax(s[0].float(), -1)[t]), 3) for s, t in zip(out.scores, new)]
        return {"text": self.tok.decode(new, skip_special_tokens=True), "pieces": [self.tok.decode([int(t)]) for t in new], "p": probs}

    @torch.no_grad()
    def margin(self, prompt, good, bad):
        """Log-odds of the best good answer over the best bad one. Answers that share their first piece are compared
        whole (summed log-probability of every piece), as the inside engine does; answers made of the same pieces
        cannot be told apart and say so (indistinguishable) rather than scoring 0 as "not learned"."""
        tok = self.tok
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.device); p0 = ids[0].tolist()
        def piece_ids(c):   # the answer's pieces as they follow the prompt
            full = tok(prompt + c).input_ids
            return full[len(p0):] if full[:len(p0)] == p0 and len(full) > len(p0) else tok(c, add_special_tokens=False).input_ids
        gs = [piece_ids(c) for c in good]; bs = [piece_ids(c) for c in bad]
        lg = torch.log_softmax(self.model(ids).logits[0, -1].float(), -1)
        top = torch.topk(lg, 3)
        r = {"top": [[tok.decode([int(i)]), round(float(v.exp()), 3)] for v, i in zip(top.values, top.indices)]}
        if not ({q[0] for q in gs} & {q[0] for q in bs}):
            r.update(margin=round(float(max(lg[q[0]] for q in gs) - max(lg[q[0]] for q in bs)), 2), compared="first piece", same_first_piece=False, indistinguishable=False)
            return r
        def lp(seq):
            full = torch.cat([ids, torch.tensor([seq], device=self.device)], 1); n0 = ids.shape[1]
            l2 = torch.log_softmax(self.model(full).logits[0, n0 - 1:n0 - 1 + len(seq)].float(), -1)
            return float(l2.gather(1, torch.tensor(seq, device=self.device)[:, None]).sum())
        same = {tuple(q) for q in gs} & {tuple(q) for q in bs}
        m = 0.0 if same else max(lp(q) for q in gs) - max(lp(q) for q in bs)
        r.update(margin=round(m, 2), compared="whole answers (they share their first piece)", same_first_piece=True, indistinguishable=bool(same) or m == 0.0)
        return r

    # ── requests ────────────────────────────────────────────────────────────
    def ask(self, step, prompt, text=None, n_new=40, heads=True):
        t0 = time.time(); self.load(int(step))
        r = {"step": self.step, "tokens_seen_b": round(self.step * TOKENS_PER_STEP / 1e9, 2), "load_s": round(time.time() - t0, 1)}
        if prompt: r["continuation"] = self.continue_text(prompt, n_new)
        r["induction"] = self.induction(with_heads=heads)
        r["position"] = self.loss_by_position(text or DEFAULT_TEXT)
        r["seconds"] = round(time.time() - t0, 1)
        return r

    def when_learned(self, prompt, good, bad, steps=None):
        """The margin at each step, and the verdict (see verdict)."""
        t0 = time.time(); steps = [int(x) for x in (steps or [x for x in KEY_STEPS if x in self.steps])]
        curve = []
        for st in steps:
            self.load(st); m = self.margin(prompt, good, bad); m["step"] = st; curve.append(m)
        return {"prompt": prompt, "good": good, "bad": bad, "curve": curve, **verdict(curve, self.steps), "seconds": round(time.time() - t0, 1)}


def status(m):
    """One checkpoint's reading: indistinguishable | learned (margin > NOISE_NATS) | within noise | surface."""
    if m.get("indistinguishable"): return "indistinguishable"
    return "learned" if m["margin"] > NOISE_NATS else "within noise" if m["margin"] >= -NOISE_NATS else "surface"


def verdict(curve, on_disk=None):
    """learned: {after_step, by_step}, where the margin first rose above the noise and stayed above it at every later
    step scored; `between`: checkpoints on disk inside that bracket that were not scored (scoring them narrows it)."""
    curve = sorted(curve, key=lambda c: c["step"]); st = [status(c) for c in curve]
    first = next((i for i in range(len(st)) if all(x == "learned" for x in st[i:])), None)
    learned = None
    if first is not None:
        learned = {"after_step": curve[first - 1]["step"] if first > 0 else None, "by_step": curve[first]["step"]}
        if on_disk and learned["after_step"] is not None:
            scored = {c["step"] for c in curve}
            learned["between"] = [x for x in on_disk if learned["after_step"] < x < learned["by_step"] and x not in scored]
    return {"learned": learned, "never": first is None, "end": st[-1] if st else None, "statuses": st, "noise_nats": NOISE_NATS,
            "scored": len(curve), "on_disk": len(on_disk) if on_disk else None}
