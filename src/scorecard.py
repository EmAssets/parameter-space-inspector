"""The scorecard: one set of measurements that can be taken on any causal language model, from a reader's
140,000-number seed to Qwen3-4B, so every rung of the ladder lands in the same table.

It uses only what every Hugging Face causal LM gives: logits, and attention weights (load with
attn_implementation="eager"). Nothing depends on the architecture (Llama, GPT-NeoX) or the tokenizer
(letters, word pieces): text is scored per character, and answers are compared whole.

  bits_per_char   how surprised it is by plain text, per character (lower is better; comparable across tokenizers)
  understanding   the nine two-choice tests the Understanding view runs (guessing passes about half)
  copying_nats    how much cheaper a random sequence is the second time it is seen (the induction phase shift)
  lookback_heads  heads that attend from a repeated token to what followed it last time (score > 0.3)
"""
import math
import time

import torch

# Two plain texts the scorecard reads: one from the street's world, one that is not.
STREET = ("The cycle rickshaw offers low-cost, zero-emission transport tailored to narrow, crowded urban streets. Operating costs are "
          "minimal, and the vehicle can be repaired by local mechanics with basic tools. The bakery on the corner opens before dawn. By "
          "seven the first loaves are gone and the owner is already shaping the second batch. Regulars know to come early; newcomers "
          "learn by missing out once.")
GENERAL = ("Water expands when it freezes, which is why ice floats and lakes freeze from the top down. The fish below survive the winter "
           "in the liquid layer. In the seventeenth century, scientists began to measure such things carefully, and the habit of "
           "measuring spread from physics to chemistry, to medicine, and eventually to the study of markets and minds.")
# The Understanding view's nine tests, shared with the Tipping point scorecard (prompt, answers that show understanding, answers that do not).
from lesion import TESTS
# The noise floor of the understanding tests: an untrained Pythia passes FLOOR of them (research/results/scorecard-refs.json), so a
# score at or near it is not evidence. "Understanding appeared" is signed only from APPEARED up; below that a record says
# "within the untrained floor, not signed as appeared" (grow.py).
FLOOR = 6
APPEARED = 8


def _ctx(model):
    return int(getattr(model.config, "max_position_embeddings", 0) or 2048)


def _ids(tok, text, bos=True):
    return tok(text, add_special_tokens=bos).input_ids


@torch.no_grad()
def bits_per_char(model, tok, device, text):
    """Average surprise per character, read in windows the model can hold. With a start token every piece of a
    window is predicted; without one, the first piece of each window has nothing to be predicted from."""
    ids = _ids(tok, text, bos=False); ctx = _ctx(model); bos = tok.bos_token_id
    win = ctx - 1 if bos is not None else ctx
    total = 0.0; chars = 0
    for s in range(0, len(ids), win):
        chunk = ids[s:s + win]
        x = ([bos] + chunk) if bos is not None else chunk
        if len(x) < 2: break
        lg = torch.log_softmax(model(torch.tensor([x], device=device)).logits[0, :-1].float(), -1)
        tgt = torch.tensor(x[1:], device=device)
        total += float(-lg[torch.arange(len(tgt)), tgt].sum())
        chars += len(tok.decode(chunk if bos is not None else chunk[1:]))
    return round(total / max(1, chars) / math.log(2), 3)


def answer_ids(tok, p, prompt, ans):
    """The pieces of `ans` as they follow `prompt`: encode both together and take what comes after the prompt, so a
    tokenizer that adds a word-start marker to every new text (letter-level Llamas) does not add a second one."""
    full = _ids(tok, prompt + ans)
    return full[len(p):] if full[:len(p)] == p and len(full) > len(p) else tok(ans, add_special_tokens=False).input_ids


@torch.no_grad()
def margin(model, tok, device, prompt, good, bad):
    """Log-odds of the best good answer over the best bad one; answers compared whole (every piece summed)."""
    p = _ids(tok, prompt)
    def lp(ans):
        a = answer_ids(tok, p, prompt, ans)
        x = torch.tensor([p + a], device=device)
        lg = torch.log_softmax(model(x).logits[0, len(p) - 1:len(p) - 1 + len(a)].float(), -1)
        return float(lg.gather(1, torch.tensor(a, device=device)[:, None]).sum())
    return max(lp(g) for g in good) - max(lp(b) for b in bad)


@torch.no_grad()
def copying(model, tok, device, n=40, trials=3, seed=0):
    """Repeat a random sequence: nats saved the second time, and heads that look back to copy."""
    V = model.get_output_embeddings().weight.shape[0]; ctx = _ctx(model)
    n = min(n, ctx // 2 - 1); lo = 1000 if V > 4000 else 3
    g = torch.Generator().manual_seed(seed); first = second = 0.0; ind = None
    for _ in range(trials):
        s = torch.randint(lo, min(V, 100000), (n,), generator=g)
        ids = torch.cat([s, s])[None].to(device)
        out = model(ids, output_attentions=True)
        lg = torch.log_softmax(out.logits[0, :-1].float(), -1); tgt = ids[0, 1:]
        nll = -lg[torch.arange(len(tgt)), tgt]; first += float(nll[:n - 1].mean()) / trials; second += float(nll[n:].mean()) / trials
        if out.attentions:
            q = torch.arange(n + 1, 2 * n, device=device)
            sc = torch.stack([A[0].float()[:, q, q - n + 1].mean(1) for A in out.attentions])   # layers x heads
            ind = sc if ind is None else ind + sc
    heads = int(((ind / trials) > 0.3).sum()) if ind is not None else None
    return round(first - second, 2), heads


def measure(model, tok, device):
    """Every scorecard measurement for one model."""
    t0 = time.time(); model.eval()
    ms = [margin(model, tok, device, t['prompt'], t['good'], t['bad']) for t in TESTS]   # the shared tests (understanding-tests.js), as dicts
    nats, heads = copying(model, tok, device)
    cfg = model.config; c = getattr(cfg, "text_config", None) or cfg
    return {"params": int(sum(p.numel() for p in model.parameters())), "layers": int(c.num_hidden_layers), "width": int(c.hidden_size),
            "context": _ctx(model), "vocab": int(model.get_output_embeddings().weight.shape[0]),
            "bits_per_char_street": bits_per_char(model, tok, device, STREET), "bits_per_char_general": bits_per_char(model, tok, device, GENERAL),
            "understanding": sum(m > 0 for m in ms), "understanding_of": len(TESTS), "understanding_margin": round(sum(ms) / len(ms), 2),
            "understanding_floor": FLOOR, "understanding_appeared_from": APPEARED,
            "copying_nats": nats, "lookback_heads": heads, "device": str(device), "seconds": round(time.time() - t0, 1)}
