"""
The neural-net trace engine: PyTorch + the unpacked bf16 weights, so the Inspector
can show what every layer does while the model answers. llama.cpp (the main
engine) only exposes outputs; this one exposes the inside.

  python src/trace_server.py --model <a Hugging Face model folder, e.g. qwen3-4b-bf16> --port 4311

POST /trace  {"prompt": "...", "n_new": 16, "lens": true|false, "think": false}
  -> {"prompt_tokens": [...], "tokens": [ per generated token:
        {"piece", "id", "p", "lens": [[piece, p] per layer (top-1 after each layer)],
         "lens_top3": [[[piece,p]x3] per layer], "resid": [norm per layer],
         "mlp": [[[neuron, act] top-8] per layer], "attn": [[w to each earlier pos] per layer]} ]}

Device: CUDA when it has room (~9 GB), else CPU (slow but works).

Only for local programs, not web pages: Host must be 127.0.0.1/localhost:PORT, no Origin header, and every POST needs
Content-Type application/json and header X-PSI-Token = $PSI_TRACE_TOKEN (TraceClient sets it; started by hand without
it, a random token is printed at start).
"""
import argparse, hashlib, hmac, json, math, os, re, secrets, sys, time, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
# A cache folder (PSI_CACHE_DIR, as config.js cacheEnv sets it for the app) holds Hugging Face downloads and temp files;
# without one, HF_HOME and TMP/TEMP stay as they are.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]; os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface"))
    os.makedirs(os.path.join(_c, "tmp"), exist_ok=True); os.environ["TMP"] = os.environ["TEMP"] = os.environ["TMPDIR"] = os.path.join(_c, "tmp")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import checkpoints as ckpt_mod
import gpu_power
import devices
import lesion as lesion_mod
import sensing as sensing_mod
import fight as fight_mod
import connector as connector_mod
import anatomy as anatomy_mod
import sae as sae_mod

LENS = ("Think through the Emergence Machine's E-levels, the fifteen floors of reality: E0 fields, E1 particles, E2 nuclei, E3 chemistry, "
        "E4 information-carrying molecules, E5 cells, E6 organisms, E7 ecosystems, E8 minds, E9 cultures, E10 institutions, E11 organizations, "
        "E12 states, E13 civilizations, E14 planetary-scale coordination. For any question: name the level or levels the subject lives on; say what "
        "had to exist on the floors below for it to appear; say what it makes possible on the floors above; and point out what emerges at that "
        "level that none of the parts does on its own. Use the E-numbers. Be concise and concrete.")
MODEL = None


def lens_text(lens):
    """The framing given as the system message: the user's own lens text (sent as a string by the app, Settings >
    Thinking in E-levels), the default E-level lens for a plain true, or none."""
    if isinstance(lens, str): return lens.strip()[:4000] or None
    return LENS if lens else None


def _lens_arg(body):
    v = body.get("lens", True)
    return v[:4000] if isinstance(v, str) else bool(v)


def _readable(path):
    """The inside views read one MLP (down_proj) per layer and self_attn: a mixture-of-experts model has many experts
    there, and LFM2 convolution layers. Said before any weights are loaded."""
    try: cfg = json.load(open(os.path.join(path, "config.json"), encoding="utf-8"))
    except Exception: return
    mt = str(cfg.get("model_type") or "")
    moe = any(cfg.get(k) for k in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")) or "moe" in mt
    if moe or mt in ("gpt_oss", "mixtral", "deepseek_v3", "dbrx", "jamba"):
        raise SystemExit(f"{os.path.basename(path.rstrip('/'))} is a mixture-of-experts model ({mt}): the inside engine reads one MLP per layer, "
                         "and this model has many experts there. The answering engine runs it (Generate, Read, Compare); Neural net, Anatomy and the experiments cannot.")
    if mt == "lfm2":
        raise SystemExit(f"{os.path.basename(path.rstrip('/'))} has hybrid convolution layers ({mt}) that the inside engine does not read. The answering engine runs it.")
LESION = None; MODEL_PATH = None; SENSE = None; FIGHT = None; CONN = None; ANAT = None
SAEB = None   # the sparse autoencoders of the loaded model (src/sae.py), opened on first use
CKPT = None   # training checkpoints of a small recorded model (Pythia), loaded on demand




def _finite(o):
    """JSON has no NaN or Infinity: a numerical accident becomes null instead of a reply the page cannot read."""
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_finite(v) for v in o]
    return o

def _spm_pieces(tok):
    """SentencePiece-style tokenizers (the toy Llama from How AI Emerges) decode a lone word-start piece
    '▁' to an empty string, so a token-by-token view loses every space. Decode single tokens from the
    vocabulary instead, with '▁' shown as the space it stands for. Multi-token decoding is unchanged."""
    vocab = tok.get_vocab()
    if not any(t.startswith("▁") for t in list(vocab)[:2000]): return
    plain = tok.decode
    def decode(ids, *a, **k):
        if hasattr(ids, "tolist"): ids = ids.tolist()
        if isinstance(ids, (list, tuple)) and len(ids) == 1 and not k.get("skip_special_tokens"):
            t = tok.convert_ids_to_tokens(int(ids[0]))
            if isinstance(t, str) and "▁" in t: return t.replace("▁", " ")
        return plain(ids, *a, **k)
    tok.decode = decode

def _control_tokens(tok):
    """Some saved tokenizer.json files list only a few added tokens (Qwen3: the three <|...|> ones) while <think>,
    </think>, <tool_call>, <|fim_*|> sit in the vocabulary's reserved tail. Text like the empty think block for
    /no_think then splits into "<th", "ink", ... and the model answers it with a real </think>. llama.cpp parses
    them as single tokens: register those vocabulary entries (same ids, nothing new) so the prompt matches."""
    added, vocab = tok.get_added_vocab(), tok.get_vocab()
    if not added: return []
    lo, n0 = min(added.values()), len(tok)
    miss = [t for t, i in sorted(vocab.items(), key=lambda x: x[1]) if i >= lo and t not in added and re.fullmatch(r"<\|[^|\s]+\|>|</?[A-Za-z_]+>", t)]
    if miss:
        from tokenizers import AddedToken
        tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in miss])
        if len(tok) != n0 or any(tok.convert_tokens_to_ids(t) != vocab[t] for t in miss): print(f"[trace] control tokens changed ids: {miss}", file=sys.stderr, flush=True)
    return miss

def _weight_bytes(path):
    """Bytes of the weights on disk (safetensors, else .bin); 0 when none are found."""
    try: names = os.listdir(path)
    except OSError: return 0
    st = [n for n in names if n.endswith(".safetensors")] or [n for n in names if n.endswith(".bin") and "model" in n]
    return sum(os.path.getsize(os.path.join(path, n)) for n in st)

def _weights_hash(path):
    """Which weights these are, cheaply: sha256 of model.safetensors.index.json, or of the first 1 MB of the single
    weights file (its header names every tensor with its shape and offset). Computed once at load."""
    idx = os.path.join(path, "model.safetensors.index.json")
    try:
        if os.path.exists(idx):
            with open(idx, "rb") as f: return hashlib.sha256(f.read()).hexdigest(), "model.safetensors.index.json"
        for n in sorted(os.listdir(path)):
            if n.endswith(".safetensors") or (n.endswith(".bin") and "model" in n):
                with open(os.path.join(path, n), "rb") as f: return hashlib.sha256(f.read(1 << 20)).hexdigest(), f"first 1 MB of {n}"
    except OSError: pass
    return None, None

def _neox_names(model):
    """GPT-NeoX (Pythia) names its parts differently from Llama/Qwen. Alias the names the views use (model.layers,
    model.norm, lm_head, mlp.down_proj) to the same modules without registering them twice (object.__setattr__), so
    parameter names, hooks and lesions stay those of the checkpoint. Views that need self_attn still fail on such a
    model, with an error."""
    if hasattr(model, "model") or not hasattr(model, "gpt_neox"): return
    import types
    nx = model.gpt_neox
    object.__setattr__(model, "model", types.SimpleNamespace(layers=nx.layers, norm=nx.final_layer_norm, embed_tokens=nx.embed_in))
    if not hasattr(model, "lm_head") and hasattr(model, "embed_out"): object.__setattr__(model, "lm_head", model.embed_out)
    for layer in nx.layers:
        if not hasattr(layer.mlp, "down_proj") and hasattr(layer.mlp, "dense_4h_to_h"): object.__setattr__(layer.mlp, "down_proj", layer.mlp.dense_4h_to_h)

class Tracer:
    @torch.no_grad()
    def answer_margin(self, prompt, good, bad, model=None):
        """Log-odds of the best good answer over the best bad one after the whole network. Answers that
        share their first piece (a character-level model) are compared whole, every piece summed."""
        tok, m = self.tok, model or self.model
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        p0 = ids[0].tolist()
        def piece_ids(c):   # the answer's pieces as they follow the prompt (no second word-start marker)
            full = tok(prompt + c, add_special_tokens=False).input_ids
            return full[len(p0):] if full[:len(p0)] == p0 and len(full) > len(p0) else tok(c, add_special_tokens=False).input_ids
        gs = [piece_ids(c) for c in good]; bs = [piece_ids(c) for c in bad]
        if not ({q[0] for q in gs} & {q[0] for q in bs}):
            lg = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)
            return float(max(lg[q[0]] for q in gs) - max(lg[q[0]] for q in bs))
        def lp(seq):
            full = torch.cat([ids, torch.tensor([seq], device=self.device)], 1); n0 = ids.shape[1]
            lg = torch.log_softmax(m(input_ids=full).logits[0, n0 - 1:n0 - 1 + len(seq)].float(), -1)
            return float(lg.gather(1, torch.tensor(seq, device=self.device)[:, None]).sum())
        return max(lp(q) for q in gs) - max(lp(q) for q in bs)

    def ctx_ids(self, text):
        """A text's ids, cut to what this model can read (the toy Llama reads 128 pieces)."""
        ids = self.tok(text, return_tensors="pt").input_ids
        return ids[:, : getattr(self.model.config, "max_position_embeddings", ids.shape[1]) or ids.shape[1]].to(self.device)

    def readout_many(self, h, l):
        """readout for several positions at once: h is [n, hidden]."""
        x = h.to(self.model.dtype)
        return self.model.lm_head(x if l == self.L else self.model.model.norm(x)).float()

    def readout(self, h, l):
        """Output-head logits for a residual vector at layer l (1..L). The last entry of
        hidden_states is post-norm in Qwen/Llama-style models, so skip the norm there."""
        x = h[None].to(self.model.dtype)
        return self.model.lm_head(x if l == self.L else self.model.model.norm(x)).float()[0]

    def __init__(self, path):
        # The GPU (NVIDIA, AMD ROCm, Apple MPS, Intel XPU: devices.py) when it has room for this model: its weight
        # bytes x 1.15 (activations, KV cache) + 1 GB; else the CPU.
        _readable(path)   # refuse a model the views cannot read before loading its weights (gpt-oss-20b unpacks to 42 GB)
        self.weight_bytes = _weight_bytes(path)
        self.need_gb = round((self.weight_bytes * 1.15) / 2**30 + 1, 2) if self.weight_bytes else 9.5
        self.device = devices.pick(self.need_gb)
        self.device_why = devices.why()
        self.free_gb = devices.free_gb(devices.accelerator() or "cpu")
        self.vendor, self.device_name = devices.vendor(self.device), devices.name(self.device)
        dtype = devices.dtype(self.device)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(path)
        _spm_pieces(self.tok); _control_tokens(self.tok)
        self.model =AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager", device_map=self.device, low_cpu_mem_usage=True).eval()
        _neox_names(self.model)
        self.L = self.model.config.num_hidden_layers
        self.model_dir = os.path.abspath(path)
        self.weights_hash, self.weights_hash_of = _weights_hash(path)
        tmpl = getattr(self.tok, "chat_template", None) or ""
        self.no_think_supported = isinstance(tmpl, str) and ("/no_think" in tmpl or "enable_thinking" in tmpl)
        self.no_think_appended = None   # per request: whether the chat prompt got /no_think (reset by the handler)
        self.acts = {}
        # Hooks: the MLP's gated activation (input of down_proj) per layer.
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: self.acts.__setitem__(i, inp[0][0, -1].detach().float()))
        self.load_s = round(time.time() - t0, 1)
        # Re-entrant: a ranked Tipping-point cut calls heads_scan (which takes the lock) while /lesion already holds it;
        # a plain Lock deadlocked the server there.
        self.lock = threading.RLock()
        self._vocab_map()
        print(f"loaded on {self.device}{f' ({self.device_name})' if self.device_name else ''}: {self.device_why.get('why')} in {self.load_s}s", flush=True)

    def _vocab_map(self):
        """Where every token lives: a 2-D map of the embedding matrix (PCA on a sample),
        plus the unit-normalised matrix for nearest-neighbour lookups."""
        E = self.model.get_input_embeddings().weight.detach().float()
        self.E_unit = torch.nn.functional.normalize(E, dim=1)
        n = E.shape[0]
        g = torch.Generator(device="cpu").manual_seed(0)
        # The first 40k ids are the commonest pieces in most tokenizers; sample 6k of them for the backdrop.
        self.map_ids = torch.randperm(min(40000, n), generator=g)[:6000].to(E.device)
        X = E[self.map_ids]; self.map_mean = X.mean(0)
        Xc = X - self.map_mean
        U, S, V = torch.pca_lowrank(Xc, q=8, center=False)
        self.map_V = V[:, :2]
        pts = Xc @ self.map_V
        self.map_pts = pts.cpu().tolist()
        self.map_labels = [self.tok.decode([int(i)]) for i in self.map_ids.cpu()]

    def embed_info(self, text, k=8):
        tok = self.tok
        ids = tok(text, add_special_tokens=False).input_ids
        E = self.model.get_input_embeddings().weight.detach().float()
        out = []
        with torch.no_grad():
            for i in ids:
                v = E[i]
                sims = self.E_unit @ torch.nn.functional.normalize(v, dim=0)
                sims[i] = -1
                top = torch.topk(sims, k)
                proj = ((v - self.map_mean) @ self.map_V).tolist()
                # A sketch of the row in at most 128 values: the mean of each block (2,560 dimensions -> blocks of 20).
                # A narrower model (the toy Llama has 64) is shown whole.
                sketch = (v if v.shape[0] <= 128 else v[: (v.shape[0] // 128) * 128].view(128, -1).mean(1)).tolist()
                out.append({"id": int(i), "piece": tok.decode([int(i)]), "norm": round(float(v.norm()), 3), "xy": [round(proj[0], 3), round(proj[1], 3)],
                            "sketch": [round(x, 4) for x in sketch],
                            "neighbors": [[tok.decode([int(j)]), round(float(s), 3)] for s, j in zip(top.values, top.indices)]})
        return {"tokens": out, "dims": int(E.shape[1]), "vocab": int(E.shape[0]),
                "map": {"pts": self.map_pts, "labels": self.map_labels}}

    def _chat_ids(self, prompt, lens, think=False):
        # A model without a chat template (the toy Llama from How AI Emerges, base models) continues plain text.
        if not getattr(self.tok, "chat_template", None):
            return self.tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
        # "/no_think" and the empty <think> block mean something only to a template that knows them (Qwen3); any
        # other model would read them as plain text, so they are added only when supported, and marked when added.
        off = (not think) and self.no_think_supported
        self.no_think_appended = bool(off)
        msgs = ([{"role": "system", "content": lens_text(lens)}] if lens_text(lens) else []) + [{"role": "user", "content": prompt + " /no_think" if off else prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + ("<think>\n\n</think>\n\n" if off else "")
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    @torch.no_grad()
    def score_text(self, context, text=None, greedy=0, top=3, max_pieces=200):
        """Score an assistant turn under the chat template, full softmax (no top-k bound). context is the user turn; text
        the answer to score, or greedy=n to have the model write n pieces itself (greedy) and score those the same way.
        Per piece: bits (surprisal), rank, top alternatives, entropy in bits. Used by MCP score_my_text."""
        tok, model = self.tok, self.model
        if getattr(tok, "chat_template", None):
            off = self.no_think_supported; self.no_think_appended = bool(off)
            msgs = [{"role": "user", "content": context + (" /no_think" if off else "")}]
            pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + ("<think>\n\n</think>\n\n" if off else "")
            pre_ids = tok(pre, add_special_tokens=False).input_ids; think = bool(off)
        else:
            pre_ids = tok(context, add_special_tokens=True).input_ids; think = False
        out = {"think_block": "empty think block added (the template knows it)" if think else "none", "template": bool(getattr(tok, "chat_template", None))}
        if greedy:
            ids = list(pre_ids)
            g = getattr(model.generation_config, "eos_token_id", None); eos = set(g if isinstance(g, list) else [g]) | {tok.eos_token_id}; eos.discard(None)
            for _ in range(int(greedy)):
                nxt = int(model(input_ids=torch.tensor([ids], device=self.device)).logits[0, -1].argmax())
                if nxt in eos: break
                ids.append(nxt)
            txt_ids = ids[len(pre_ids):]; out["text"] = tok.decode(txt_ids)
        else:
            txt_ids = tok(text or "", add_special_tokens=False).input_ids
        txt_ids = txt_ids[:int(max_pieces)]
        if not txt_ids: out["pieces"] = []; return out
        x = torch.tensor([pre_ids + txt_ids], device=self.device)
        lp = torch.log_softmax(model(input_ids=x).logits[0, len(pre_ids) - 1:len(pre_ids) - 1 + len(txt_ids)].float(), -1)
        L2 = math.log(2); pieces = []
        for i, t in enumerate(txt_ids):
            row = lp[i]; v = float(row[t]); tv, ti = torch.topk(row, int(top))
            pieces.append({"piece": tok.decode([t]), "id": int(t), "bits": round(-v / L2, 3), "rank": int((row > v).sum()) + 1,
                           "top": [[tok.decode([int(j)]), round(float(math.exp(float(p))), 4)] for p, j in zip(tv, ti)],
                           "entropy_bits": round(float(-(row.exp() * row).sum()) / L2, 3)})
        out["pieces"] = pieces; out["truncated_to"] = int(max_pieces) if text and len(tok(text, add_special_tokens=False).input_ids) > int(max_pieces) else None
        return out

    def understand(self, items):
        """items: [{prompt, good:[...], bad:[...]}]. For each: log-odds good vs bad at the
        final layer, and the first layer (logit lens) where good already beats bad."""
        tok, model = self.tok, self.model
        norm, head = model.model.norm, model.lm_head
        out = []
        with self.lock, torch.no_grad():
            for it in items:
                ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                o = model(input_ids=ids, output_hidden_states=True)
                p0 = ids[0].tolist()
                def piece_ids(c):   # the answer's pieces as they follow the prompt (no second word-start marker)
                    full = tok(it["prompt"] + c, add_special_tokens=False).input_ids
                    return full[len(p0):] if full[:len(p0)] == p0 and len(full) > len(p0) else tok(c, add_special_tokens=False).input_ids
                g_seq = [piece_ids(c) for c in it["good"]]
                b_seq = [piece_ids(c) for c in it["bad"]]
                g_ids = [q[0] for q in g_seq]; b_ids = [q[0] for q in b_seq]
                per_layer = []
                if set(g_ids) & set(b_ids):
                    # The answers start with the same piece (a character-level model such as the toy Llama reads
                    # " are" and " is" as space + letters): compare whole answers, the summed log-probability of
                    # every piece, read through the output head at each layer.
                    def seq_lp(seq):
                        full = torch.cat([ids, torch.tensor([seq], device=self.device)], 1)
                        hs = model(input_ids=full, output_hidden_states=True).hidden_states
                        n0 = ids.shape[1]; tgt = torch.tensor(seq, device=self.device)
                        return [float(torch.log_softmax(self.readout_many(hs[l][0, n0 - 1:n0 - 1 + len(seq)], l), -1).gather(1, tgt[:, None]).sum()) for l in range(1, self.L + 1)]
                    g_lp = [seq_lp(q) for q in g_seq]; b_lp = [seq_lp(q) for q in b_seq]
                    per_layer = [max(x[l] for x in g_lp) - max(x[l] for x in b_lp) for l in range(self.L)]
                else:
                    for l in range(1, self.L + 1):
                        lg = torch.log_softmax(self.readout(o.hidden_states[l][0, -1], l), -1)
                        per_layer.append(float(max(lg[i] for i in g_ids) - max(lg[i] for i in b_ids)))
                first = next((l + 1 for l, v in enumerate(per_layer) if v > 0 and all(x > 0 for x in per_layer[l:])), None)
                lg = torch.log_softmax(o.logits[0, -1].float(), -1)
                top = torch.topk(lg, 3)
                out.append({"prompt": it["prompt"], "logodds": round(per_layer[-1], 2), "pass": per_layer[-1] > 0, "readable_from": first,
                            "by_layer": [round(v, 2) for v in per_layer], "top": [[tok.decode([int(i)]), round(float(v.exp()), 3)] for v, i in zip(top.values, top.indices)]})
        return {"items": out, "layers": self.L}

    def represent(self, text, layers=None, k=5):
        """What the model thinks each word IS after reading the sentence: the output head
        applied to the residual stream at that word's position, at several depths."""
        tok, model = self.tok, self.model
        # Seven depths spread over this model's layers (1, 6, 12 … 36 for Qwen3-4B; every layer for a small one).
        layers = sorted({max(1, round(self.L * f / 6)) for f in range(7)}) if layers is None else [l for l in layers if 0 < l <= self.L]
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        norm, head = model.model.norm, model.lm_head
        with self.lock, torch.no_grad():
            o = model(input_ids=ids, output_hidden_states=True)
            out = []
            for pos in range(ids.shape[1]):
                row = {"piece": tok.decode([int(ids[0, pos])]), "by_layer": {}}
                row["near"] = {}
                for l in layers:
                    h = o.hidden_states[l][0, pos]
                    lg = torch.softmax(self.readout(h, l), -1)
                    top = torch.topk(lg, k)
                    row["by_layer"][str(l)] = [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)]
                    sims = self.E_unit @ torch.nn.functional.normalize(h.float(), dim=0)
                    nt = torch.topk(sims, k)
                    row["near"][str(l)] = [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(nt.values, nt.indices)]
                out.append(row)
        return {"tokens": out, "layers": list(layers)}

    def vectors(self, text, pool="last", dims=256, seed=0):
        """The residual stream after every layer (hidden_states: the embeddings, then layers 1..L; the last is after the
        final norm), pooled over the text (its last piece, or the mean of all pieces) and projected to `dims` numbers by a
        fixed random Gaussian matrix (seeded, so every prompt and every run share it). A random projection keeps
        distances and angles close (Johnson-Lindenstrauss), so similarity, CKA and linear probes on it mean what they mean
        on the full vectors, at a size an experiment record can hold. dims 0: the full vectors (hidden size up to 4096)."""
        import base64
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        if ids.shape[1] < 1: raise ValueError("vectors: the text has no pieces")
        with self.lock, torch.no_grad():
            hs = self.model(input_ids=ids, output_hidden_states=True).hidden_states
            v = torch.stack([(h[0, -1] if pool == "last" else h[0].mean(0)).float() for h in hs]).cpu()   # [L+1, d]
        d = v.shape[1]; dims = int(dims)
        if dims <= 0:
            if d > 4096: raise ValueError(f"vectors: hidden size {d} is too large to send whole; ask for dims up to 1024")
            out, dims = v, d
        else:
            dims = max(8, min(1024, dims)); key = (d, dims, int(seed))
            if getattr(self, "_proj_key", None) != key:
                g = torch.Generator().manual_seed(int(seed)); self._proj = torch.randn(d, dims, generator=g) / math.sqrt(dims); self._proj_key = key
            out = v @ self._proj
        return {"layers": int(v.shape[0]), "dims": dims, "hidden": d, "pool": pool, "seed": int(seed), "pieces": int(ids.shape[1]),
                "norms": [round(float(x), 3) for x in v.norm(dim=1)], "b64": base64.b64encode(out.numpy().astype("<f4").tobytes()).decode()}

    def ablate(self, prompt, lens=True, target=None, plan="top", k=8, layers=None, seed=0, prefix_ids=None):
        """Knock neurons out and re-ask: how much of the next token lived in them?
        plan: 'top' = the k strongest-firing neurons per chosen layer for this token;
              'random' = k random neurons per chosen layer; 'none' = baseline."""
        tok, model = self.tok, self.model
        ids = self._chat_ids(prompt, lens)
        if prefix_ids:
            ids = torch.cat([ids, torch.tensor([list(map(int, prefix_ids))], device=self.device)], 1)
        norm, head = model.model.norm, model.lm_head
        with self.lock, torch.no_grad():
            base = model(input_ids=ids, output_hidden_states=False)
            lg = torch.softmax(base.logits[0, -1].float(), -1)
            tgt = int(lg.argmax()) if target is None else int(target)
            p0 = float(lg[tgt])
            chosen = {}
            sel = layers or list(range(self.L))
            g = torch.Generator().manual_seed(seed)
            for l in sel:
                a = self.acts.get(l)
                if plan == "top" and a is not None:
                    chosen[l] = torch.topk(a.abs(), k).indices.tolist()
                elif plan == "random":
                    chosen[l] = torch.randperm(a.shape[0] if a is not None else model.config.intermediate_size, generator=g)[:k].tolist()
            hooks = []
            for l, idx in chosen.items():
                def hook(m, inp, idx=idx):
                    x = inp[0].clone(); x[..., idx] = 0; return (x,)
                hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
            try:
                out = model(input_ids=ids)
                lg2 = torch.softmax(out.logits[0, -1].float(), -1)
            finally:
                for h in hooks: h.remove()
            p1 = float(lg2[tgt]); new_top = int(lg2.argmax())
        return {"target": tok.decode([tgt]), "p_before": round(p0, 4), "p_after": round(p1, 4), "knocked_out": sum(len(v) for v in chosen.values()),
                "layers": len(chosen), "now_says": tok.decode([new_top]), "now_p": round(float(lg2[new_top]), 4), "plan": plan}

    def _repeat_ids(self, n=40, trials=3, seed=0):
        """The shared copying protocol's sequences (checkpoints.COPY_PROTOCOL: 40 tokens x 3 trials, ids 100-30,000,
        seed 0; a tiny vocabulary or short context adapts, see copy_protocol). n and trials are kept for old callers."""
        return ckpt_mod.repeat_sequences(self.model.get_input_embeddings().weight.shape[0], getattr(self.model.config, "max_position_embeddings", None), self.device, seed or None)[0]

    def heads_scan(self, n=40, trials=3):
        """Every attention head of the loaded model, scored for looking back and copying
        (attention from a token in the second copy of a repeated sequence to the token after
        its earlier occurrence), plus the previous-token score. Shared copying protocol."""
        import numpy as np
        L, Hn = self.L, self.model.config.num_attention_heads
        ind = np.zeros((L, Hn)); prev = np.zeros((L, Hn)); first = []; second = []
        seqs, P = ckpt_mod.repeat_sequences(self.model.get_input_embeddings().weight.shape[0], getattr(self.model.config, "max_position_embeddings", None), self.device)
        n = P["n"]
        with self.lock, torch.no_grad():
            for ids in seqs:
                out = self.model(ids, output_attentions=True)
                lg = torch.log_softmax(out.logits[0, :-1].float(), -1); tgt = ids[0, 1:]
                nll = -lg[torch.arange(len(tgt)), tgt]; first.append(float(nll[: n - 1].mean())); second.append(float(nll[n:].mean()))
                q = torch.arange(n + 1, 2 * n)
                for l in range(L):
                    A = out.attentions[l][0].float()
                    ind[l] += A[:, q, q - n + 1].mean(1).cpu().numpy(); prev[l] += A[:, q, q - 1].mean(1).cpu().numpy()
        ind /= len(seqs); prev /= len(seqs)
        flat = sorted(((float(ind[l, h]), l, h) for l in range(L) for h in range(Hn)), reverse=True)
        return {"layers": L, "heads_per_layer": Hn, "heads": ind.round(3).tolist(), "prev_token": prev.round(3).tolist(),
                "n_heads": int((ind > ckpt_mod.HEAD_THRESHOLD).sum()), "head_threshold": ckpt_mod.HEAD_THRESHOLD,
                "top": [{"layer": l, "head": h, "score": round(v, 3)} for v, l, h in flat[:12]],
                "induction_score": round(float(np.mean(first) - np.mean(second)), 2), "protocol": P}

    def _head_hooks(self, heads):
        """Zero the output of the given (layer, head) pairs, at the input of o_proj."""
        if not heads: return []   # nothing to mask (a cut of neurons only)
        cfg = self.model.config; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads; by_layer = {}
        for l, h in heads: by_layer.setdefault(int(l), []).append(int(h))
        hooks = []
        for l, hs in by_layer.items():
            def hook(m, inp, hs=hs):
                x = inp[0].clone()
                for h in hs: x[..., h * hd:(h + 1) * hd] = 0
                return (x,)
            hooks.append(self.model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
        return hooks

    def ablate_heads(self, heads=None, top_k=None, plan="top", seed=0, text=None, invented=None):
        """Knock attention heads out and measure three things before and after: copying a repeated
        random sequence (nats saved), the cost of an invented word's second mention relative to its
        first, and the loss on ordinary text. plan 'top' = the top_k induction heads from a scan;
        'random' = top_k random heads."""
        import math, random
        text = text or ckpt_mod.DEFAULT_TEXT
        invented = invented or "In the old market there is a stall everyone calls the zorblax. Early each morning the zorblax opens before the others."
        H = self.model.config.num_attention_heads; total = self.L * H
        if heads is None:
            scan = self.heads_scan()
            k = min(int(top_k or 12), total)   # never more heads than the model has (the toy Llama has 8)
            if plan == "random":
                rnd = random.Random(seed); heads = [[i // H, i % H] for i in rnd.sample(range(total), k)]
            else:   # the top_k heads by induction score over the whole grid
                grid = scan["heads"]; flat = sorted(((v, l, h) for l, row in enumerate(grid) for h, v in enumerate(row)), reverse=True)
                heads = [[l, h] for v, l, h in flat[:k]]
        tok = self.tok

        def measure():
            with torch.no_grad():
                first = second = 0.0
                for ids in self._repeat_ids(40, 3):
                    lg = torch.log_softmax(self.model(ids).logits[0, :-1].float(), -1); tgt = ids[0, 1:]
                    nll = -lg[torch.arange(len(tgt)), tgt]; first += float(nll[:39].mean()) / 3; second += float(nll[40:].mean()) / 3
                ids = self.ctx_ids(text)[:, :256]
                lg = torch.log_softmax(self.model(ids).logits[0, :-1].float(), -1)
                text_bits = float((-lg[torch.arange(ids.shape[1] - 1), ids[0, 1:]]).mean()) / math.log(2)
                ids = tok(invented, return_tensors="pt").input_ids.to(self.device); seq = ids[0].tolist()
                words = [w.strip(".,;:!?").lower() for w in invented.split()]
                word = next((w for w in words if len(w) > 5 and words.count(w) >= 2), "zorblax")   # the invented word: repeated, longish
                wid = tok(" " + word, add_special_tokens=False).input_ids
                lg = torch.log_softmax(self.model(ids).logits[0].float(), -1); hits = []
                for i in range(1, len(seq) - len(wid) + 1):
                    if seq[i:i + len(wid)] == wid: hits.append(sum(-float(lg[i + k - 1, wid[k]]) for k in range(len(wid))) / math.log(2))
                ratio = round(hits[1] / max(hits[0], 1e-6), 3) if len(hits) >= 2 else None
                return {"copy_nats_saved": round(first - second, 2), "text_bits_per_token": round(text_bits, 3), "invented_second_over_first": ratio, "invented_word": word}
        with self.lock:
            before = measure()
            hooks = self._head_hooks(heads)
            try: after = measure()
            finally:
                for h in hooks: h.remove()
        return {"heads": heads, "n": len(heads), "total": self.L * self.model.config.num_attention_heads, "plan": plan, "before": before, "after": after}

    def trace(self, prompt, n_new=16, lens=True, think=False, k_neurons=8):
        tok, model = self.tok, self.model
        ids = self._chat_ids(prompt, lens, think)
        prompt_len = ids.shape[1]
        ctx = getattr(model.config, "max_position_embeddings", None) or 1 << 30
        n_new = max(0, min(n_new, ctx - prompt_len))   # small models read only a short context
        norm, head = model.model.norm, model.lm_head
        out_tokens = []
        eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        # Speed, measured here: each step's forward pass (the model, with every layer's states and attention kept) apart
        # from the time spent recording them (lens, neurons, attention rows). Step 0 reads the whole prompt.
        gpu = devices.kind(self.device) != "cpu"
        sync = lambda: devices.sync(self.device)
        devices.reset_peak(self.device)
        fwd_ms, rec_ms = [], []
        meter = gpu_power.torch_meter(self.device)   # the card's power while it records (NVML; None without one)
        t_start = time.perf_counter()
        with self.lock, torch.no_grad(), meter:
            past = None
            cur = ids
            for step in range(n_new):
                sync(); t_a = time.perf_counter()
                o = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True, output_attentions=True)
                sync(); t_b = time.perf_counter(); fwd_ms.append((t_b - t_a) * 1000)
                past = o.past_key_values
                hs = [h[0, -1].float() for h in o.hidden_states]                      # L+1 x [D]
                final = torch.log_softmax(o.logits[0, -1].float(), -1)
                nxt = int(final.argmax()); p = float(final[nxt].exp())
                lens_rows, lens_top3 = [], []
                for l in range(1, self.L + 1):
                    lg = torch.softmax(self.readout(hs[l], l), -1)
                    top = torch.topk(lg, 3)
                    row = [[tok.decode([int(i)]), round(float(v), 4)] for v, i in zip(top.values, top.indices)]
                    lens_top3.append(row); lens_rows.append(row[0])
                resid = [round(float(h.norm()), 2) for h in hs[1:]]
                mlp = []
                for l in range(self.L):
                    a = self.acts[l]; top = torch.topk(a.abs(), k_neurons)
                    mlp.append([[int(i), round(float(a[i]), 3)] for i in top.indices])
                # Attention of the current token to every earlier position, mean over heads, per layer.
                attn = [[round(float(x), 4) for x in a[0, :, -1, :].mean(0)] for a in o.attentions]
                out_tokens.append({"piece": tok.decode([nxt]), "id": nxt, "p": round(p, 4), "lens": lens_rows, "lens_top3": lens_top3, "resid": resid, "mlp": mlp, "attn": attn})
                rec_ms.append((time.perf_counter() - t_b) * 1000)
                if nxt in eos: break
                cur = torch.tensor([[nxt]], device=self.device)
        wall = time.perf_counter() - t_start
        dec_f, dec_r = fwd_ms[1:], rec_ms[1:]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None
        r2 = lambda v, n=1: None if v is None else round(v, n)
        perf = {
            "prompt_tokens": prompt_len, "prefill_ms": r2(fwd_ms[0] if fwd_ms else None),
            "prefill_tps": r2(prompt_len / (fwd_ms[0] / 1000)) if fwd_ms and fwd_ms[0] > 0 else None,
            "first_token_ms": r2((fwd_ms[0] + rec_ms[0]) if fwd_ms and rec_ms else None),
            "new_tokens": len(out_tokens),
            "forward_ms_per_token": r2(mean(dec_f), 2), "record_ms_per_token": r2(mean(dec_r), 2),
            "model_tps": r2(1000 / mean(dec_f)) if dec_f and mean(dec_f) > 0 else None,             # the forward pass alone
            "tps": r2(len(dec_f) / ((sum(dec_f) + sum(dec_r)) / 1000)) if dec_f else None,          # with the recording
            "wall_s": r2(wall, 2), "overall_tps": r2(len(out_tokens) / wall) if wall > 0 else None,
            "peak_gpu_gb": r2(devices.peak_bytes(self.device) / 2 ** 30, 2) if gpu and devices.peak_bytes(self.device) is not None else None,
            "gpu_total_gb": r2(devices.total_bytes(self.device) / 2 ** 30, 1) if gpu and devices.total_bytes(self.device) else None,
            "gpu": self.device_name if gpu else None, "vendor": self.vendor,
            "peak_is": "held at the end (MPS has no peak counter)" if devices.kind(self.device) == "mps" else "peak",
            "dtype": str(model.dtype).replace("torch.", ""),
            "attention": getattr(model.config, "_attn_implementation", None),
        }
        pw = meter.result()
        if pw:
            n_tok = max(len(out_tokens), 1)
            above = (pw["avg_w"] - pw["idle_w"]) * pw["seconds"] if pw.get("avg_w") is not None and pw.get("idle_w") is not None else None
            perf["power"] = {**pw, "j_per_token": r2(pw["energy_j"] / n_tok, 2) if pw.get("energy_j") is not None else None,
                             "above_idle_j": r2(max(above, 0.0)) if above is not None else None}
        return {"device": self.device, "layers": self.L, "mlp": int(self.model.config.intermediate_size), "prompt_tokens": [tok.decode([int(i)]) for i in ids[0]], "prompt_len": prompt_len, "tokens": out_tokens, "perf": perf}


TOKEN = os.environ.get("PSI_TRACE_TOKEN") or ""; PORT = 4311


VERSIONS = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}


def lesion_summary():
    """What a lesion has changed right now, or None when the model is as loaded."""
    if LESION is None: return None
    st = LESION.state or {}
    if st.get("kind", "none") == "none" and not LESION.hooks and not LESION.touched: return None
    keep = ("kind", "mode", "scope", "criterion", "frac", "neurons", "of_neurons", "heads", "of_heads", "zeroed", "of", "tensors", "layers",
            "concept", "keep", "k", "restore_failed", "still_changed_tensors")
    out = {k: st[k] for k in keep if k in st}
    if out.get("kind", "none") == "none": out["kind"] = "partial"   # hooks or zeroed tensors left by a lesion that failed half-way
    out["hooks"] = len(LESION.hooks); out["changed_tensors"] = len(LESION.touched)
    return out


def conditions(t0):
    """The conditions of every measurement this server returns: which weights, in what precision, on what device,
    with which libraries, and what is changed (a lesion) at the time of the reply."""
    if MODEL is None: return None
    c = {"engine": "inside", "model": os.path.basename(MODEL.model_dir.rstrip("/\\")), "model_dir": MODEL.model_dir,
         "weights_hash": MODEL.weights_hash, "weights_hash_of": MODEL.weights_hash_of,
         "dtype": str(MODEL.model.dtype).replace("torch.", ""), "device": MODEL.device, "versions": VERSIONS,
         "changed": {"lesion": lesion_summary()}, "seconds": round(time.time() - t0, 2)}
    if MODEL.no_think_appended is not None: c["no_think_appended"] = MODEL.no_think_appended
    return c


def state():
    """GET / and GET /state: the loaded model's conditions, whether a lesion can be undone from the files, the hooks."""
    lz = LESION
    src = lz.restore_source if lz else ("index" if os.path.exists(os.path.join(MODEL.model_dir, "model.safetensors.index.json"))
                                        else "single_file" if os.path.exists(os.path.join(MODEL.model_dir, "model.safetensors")) else "cpu_snapshot")
    return {"ok": True, "device": MODEL.device, "vendor": MODEL.vendor, "device_name": MODEL.device_name, "device_why": MODEL.device_why, "layers": MODEL.L, "load_s": MODEL.load_s, "free_gb_at_load": MODEL.free_gb, "need_gb": MODEL.need_gb,
            "no_think_supported": MODEL.no_think_supported, "restore_verifiable": src != "cpu_snapshot", "restore_source": src,
            "hooks": {"observe": MODEL.L, "lesion": len(lz.hooks) if lz else 0}}


class _undo_on_error:
    """A lesion action that fails half-way (hooks placed, some tensors zeroed) is undone before the error is reported, so
    the model is never left changed by a request that answered with an error. If the undo fails too, both are said."""
    def __init__(self, act): self.act = act
    def __enter__(self): return self
    def __exit__(self, typ, e, tb):
        if e is None or self.act in ("restore", "save") or LESION is None: return False   # a failed save leaves the cut as it was
        if not (LESION.hooks or LESION.touched): return False
        try: LESION.restore()
        except Exception as e2: raise RuntimeError(f"{type(e).__name__}: {e} -- and undoing the half-applied lesion failed: {e2}") from e
        raise RuntimeError(f"{type(e).__name__}: {e} (the half-applied lesion was undone; the model is as loaded)") from e


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        # Every JSON reply to the app carries the conditions it was measured under (not the 403 to a foreign page).
        if isinstance(obj, dict) and code != 403 and "conditions" not in obj:
            try: obj = {**obj, "conditions": conditions(getattr(self, "_t0", time.time()))}
            except Exception as e: obj = {**obj, "conditions": {"engine": "inside", "error": str(e)[:300]}}
        b = json.dumps(_finite(obj)).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _foreign(self):   # a web page (DNS rebinding, cross-site request) rather than the app
        host = (self.headers.get("Host") or "").lower()
        if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}") or self.headers.get("Origin"): self._send(403, {"error": "forbidden"}); return True
        return False
    def do_GET(self):
        self._t0 = time.time()
        if self._foreign(): return
        route = self.path.split("?")[0].rstrip("/")
        if route == "/next":                                       # runs the model: needs the token like every POST
            if not hmac.compare_digest((self.headers.get("X-PSI-Token") or "").encode(), TOKEN.encode()): return self._send(403, {"error": "missing or wrong X-PSI-Token"})
            return self._next()
        if route not in ("", "/state"): return self._send(404, {"error": f"no GET route {route}: this server answers GET /, /state and /next?text="})
        # Proof that this is the app's own engine: an HMAC of the caller's nonce under the shared token. The app checks it
        # before it sends the token, so a program that took the port first never receives the token.
        nonce = self.headers.get("X-PSI-Nonce") or ""
        if 16 <= len(nonce) <= 128:
            return self._send(200, {**state(), "proof": hmac.new(TOKEN.encode(), b"psi-trace:" + nonce.encode(), hashlib.sha256).hexdigest()})
        self._send(200, state())
    def _next(self):
        """GET /next?text=...&k=5: the text's token ids (no special tokens added) and the k most likely next tokens, read
        from the raw text with no chat template. Lets the app check that both engines hold the same model."""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query); text = (q.get("text") or [""])[0]; k = max(1, min(20, int((q.get("k") or ["5"])[0])))
        if not text: return self._send(400, {"error": "GET /next needs ?text="})
        try:
            with MODEL.lock, torch.no_grad():
                ids = MODEL.tok(text, add_special_tokens=False).input_ids
                lg = torch.softmax(MODEL.model(input_ids=torch.tensor([ids], device=MODEL.device)).logits[0, -1].float(), -1)
                top = torch.topk(lg, k)
            self._send(200, {"ids": ids, "top": [{"id": int(i), "piece": MODEL.tok.decode([int(i)]), "p": round(float(v), 5)} for v, i in zip(top.values, top.indices)]})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"[:4000]})
    def do_POST(self):
        global CKPT, LESION, SENSE, FIGHT, CONN, ANAT, SAEB
        if self._foreign(): return
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json": return self._send(415, {"error": "Content-Type must be application/json"})
        if not hmac.compare_digest((self.headers.get("X-PSI-Token") or "").encode(), TOKEN.encode()): return self._send(403, {"error": "missing or wrong X-PSI-Token"})
        try: n = int(self.headers.get("Content-Length") or 0)
        except ValueError: return self._send(400, {"error": "bad Content-Length"})
        if not 0 <= n <= 16 << 20: return self._send(413, {"error": "request too large (16 MB at most)"})
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError: return self._send(400, {"error": "the body is not JSON"})
        self._t0 = time.time(); MODEL.no_think_appended = None
        try:
            t0 = time.time()
            if self.path.rstrip("/") == "/understand":
                # {"items": [...]} or, through the app's bridge (which passes one argument), {"items": {"items": [...], "twin": true}}.
                # twin: score the same items on a randomly initialised copy of the same config (checkpoints.untrained_twin).
                items = body.get("items", []); twin = bool(body.get("twin"))
                if isinstance(items, dict): twin = bool(items.get("twin")); items = items.get("items", [])
                if twin:
                    with MODEL.lock: tw, info = ckpt_mod.untrained_twin(MODEL, _neox_names)
                    with MODEL.lock: r = tw.understand(items)
                    r["twin"] = info
                else: r = MODEL.understand(items)
                r["noise_nats"] = ckpt_mod.NOISE_NATS; r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            if self.path.rstrip("/") == "/score_text":
                with MODEL.lock:
                    r = MODEL.score_text(str(body.get("context", "")), body.get("text"), int(body.get("greedy") or 0), int(body.get("top") or 3), int(body.get("max_pieces") or 200))
                r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            if self.path.rstrip("/") == "/sae":
                # Sparse-autoencoder features (src/sae.py). Reading actions take the model's lock inside; label writes a name.
                if SAEB is None: SAEB = sae_mod.Bank(MODEL)
                act = body.get("action", "list"); L_ = body.get("layer")
                if act == "list": r = SAEB.list()
                elif act == "encode": r = SAEB.encode(str(body.get("text", "")), int(L_), int(body.get("k", 8)), bool(body.get("chat", False)))
                elif act == "feature": r = SAEB.feature(int(L_), int(body.get("id")), int(body.get("n", 12)))
                elif act == "search": r = SAEB.search(int(L_), str(body.get("query", "")), int(body.get("n", 20)))
                elif act == "steer":
                    if lesion_summary() is not None: return self._send(409, {"error": "A Tipping-point cut is in force: restore the model first, so the steered answer is the intact model's."})
                    r = SAEB.steer(int(L_), int(body.get("id")), float(body.get("strength", 4)), str(body.get("prompt", "")), max(4, min(200, int(body.get("n_new", 40)))))
                elif act == "attribute": r = SAEB.attribute(str(body.get("prompt", "")), body.get("layers"), body.get("target"), int(body.get("n", 10)), bool(body.get("chat", True)))
                elif act == "label": r = SAEB.label(int(L_), int(body.get("id")), str(body.get("label", "")), str(body.get("by", "you")), str(body.get("why", "")))
                elif act == "import":
                    src = str(body.get("folder", "")); lay = int(L_)
                    if not os.path.isdir(src): return self._send(400, {"error": f"not a folder: {src}"})
                    if not 1 <= lay <= MODEL.L: return self._send(400, {"error": f"layer must be 1 to {MODEL.L}"})
                    r = sae_mod.import_folder(src, MODEL.model.config.hidden_size, SAEB.layer_dir(lay), lay, {"model": os.path.basename(MODEL.model_dir.rstrip("/\\")), "model_dir": MODEL.model_dir, "weights_hash": MODEL.weights_hash, "hook": "resid_post"})
                    SAEB.cache.pop(lay, None)
                    import sae_rings   # a ring on the model's chain for this SAE, signed (src/sae_rings.py)
                    ring = sae_rings.add_ring(SAEB.layer_dir(lay), "sae_import", os.path.basename(MODEL.model_dir.rstrip("/\\")), MODEL.weights_hash, lay, {"d_sae": r["d_sae"], "activation": r["kind"], "as": r["as"], "from": os.path.basename(os.path.normpath(src))})
                    r["ring"] = ring["record"]["gen"]; r["signed"] = bool(ring["signature"])
                elif act == "reload": SAEB = sae_mod.Bank(MODEL); r = SAEB.list()
                elif act == "evaluate": r = SAEB.evaluate(int(L_), max(10, min(400, int(body.get("windows", 120)))))
                elif act == "scorable": r = SAEB.scorable(int(L_), max(1, min(200, int(body.get("count", 10)))), int(body.get("seed", 0)))
                elif act == "contexts": r = SAEB.contexts(int(L_), int(body.get("id")), max(2, min(20, int(body.get("n_pos", 5)))), max(2, min(20, int(body.get("n_neg", 5)))), int(body.get("seed", 0)))
                elif act == "set_score": r = SAEB.set_score(int(L_), int(body.get("id")), body.get("score") if isinstance(body.get("score"), dict) else {})
                else: return self._send(400, {"error": "sae: action is one of list, encode, feature, search, steer, attribute, label, import, reload, evaluate, scorable, contexts, set_score"})
                r["conditions"] = conditions(t0); return self._send(200, r)
            if self.path.rstrip("/") == "/vectors":
                r = MODEL.vectors(str(body.get("text", "")), "mean" if body.get("pool") == "mean" else "last", int(body.get("dims", 256)), int(body.get("seed", 0)))
                r["seconds"] = round(time.time() - t0, 2); return self._send(200, r)
            if self.path.rstrip("/") == "/represent":
                r = MODEL.represent(body.get("text", "")); r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            if self.path.rstrip("/") == "/ablate":
                if body.get("plan") == "random":   # the older single random draw (MCP's plan "random"): no control of its own
                    r = MODEL.ablate(body.get("prompt", ""), _lens_arg(body), body.get("target"), "random", int(body.get("k", 8)), body.get("layers"), int(body.get("seed", 0)), body.get("prefix_ids"))
                else:
                    # The k strongest neurons per chosen layer, beside `seeds` random sets of the same size and layers;
                    # set to zero (zero-ablation) or held at their average on ordinary text (mean-ablation).
                    if ANAT is None: ANAT = anatomy_mod.Anatomy(MODEL)
                    abl = body.get("ablation") or "zero"
                    r = anatomy_mod.knockout(MODEL, body.get("prompt", ""), _lens_arg(body), body.get("target"), int(body.get("k", 8)), body.get("layers"),
                                             body.get("prefix_ids"), abl, int(body.get("seeds", 5)), ANAT.stats() if abl == "mean" else None, bool(body.get("think", False)))
                    if abl == "mean": r["neuron_stats"] = {k: v for k, v in ANAT.stats_status().items() if k in ("tokens", "windows", "built_at", "path")}
                r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            if self.path.rstrip("/") == "/anatomy":
                if ANAT is None: ANAT = anatomy_mod.Anatomy(MODEL)
                act = body.get("action", "structure")
                with MODEL.lock:
                    bins = body.get("bins"); bins = int(bins) if bins else None
                    if act == "structure": r = ANAT.structure(bins)
                    elif act == "layer": r = ANAT.layer_maps(int(body.get("layer", 0)), int(body.get("size", 64)))
                    elif act == "links": r = ANAT.links(bins)
                    elif act == "record": r = ANAT.record(body.get("prompt", ""), int(body.get("n_new", 24)), _lens_arg(body), bool(body.get("think", False)), bins=bins)
                    elif act == "zoom": r = ANAT.zoom(int(body.get("layer", 0)), int(body.get("bin", 0)), bins or 48)
                    elif act == "neuron": r = ANAT.neuron(int(body.get("layer", 0)), int(body.get("neuron", 0)))
                    elif act == "examples_status": r = ANAT.examples_status()
                    elif act == "stats_status": r = ANAT.stats_status()
                    elif act == "build_stats":
                        # Each neuron's normal range would describe the changed model and be kept as the model's own: refuse while a lesion is on.
                        if lesion_summary() is not None: return self._send(409, {"error": "A Tipping-point lesion is active on the inside engine. The normal range would describe the changed model and be saved as this model's own. Restore the model in Tipping point first."})
                        r = ANAT.build_stats(int(body.get("windows", anatomy_mod.ST_MAX)))
                    elif act == "build_examples": r = ANAT.build_examples(int(body.get("windows", 2400)))
                    elif act == "patch": r = ANAT.patch(body.get("clean", ""), body.get("corrupt", ""))
                    elif act == "concept": r = ANAT.concept_map(body.get("concept", ""), body.get("facets"), int(body.get("k", 240)), bins)
                    elif act == "concept_erase":
                        # Erase only the concept's direction in a band of layers; ask about it and about a control.
                        if LESION is None: LESION = lesion_mod.Lesion(MODEL, MODEL_PATH)
                        c = body.get("concept", ""); q = body.get("question") or f"In two sentences, what is {c}?"; qc = body.get("control_question") or "In two sentences, what is a hammer?"   # not one of the comparison words
                        U = ANAT.concept_directions(c, body.get("facets")); out = []
                        for band in body.get("bands") or [[8, ANAT.L - 8]]:
                            hk = ANAT.erase_hooks(U, band)
                            try: out.append({"band": band, "answer": LESION.answer(q)["text"], "control": LESION.answer(qc)["text"]})
                            finally:
                                for h in hk: h.remove()
                        r = {"concept": c, "question": q, "control_question": qc, "results": out}
                    elif act == "color_geometry": r = ANAT.color_geometry(body.get("focus", "red"))
                    elif act == "concept_net": r = ANAT.concept_net(body.get("concept", ""), body.get("facets"), body.get("probes"))
                    elif act == "concept_inject":
                        # The reverse of deleting: add the concept's direction while the model answers unrelated questions.
                        if LESION is None: LESION = lesion_mod.Lesion(MODEL, MODEL_PATH)
                        c = body.get("concept", ""); U = ANAT.concept_directions(c, body.get("facets")); out = []
                        qs = body.get("questions") or ["Tell me about your morning in two sentences.", "Suggest a gift for my brother in one sentence."]
                        for st in body.get("strengths") or [0, 0.5, 1, 2]:
                            for band in body.get("bands") or [[8, ANAT.L - 8]]:
                                hk = ANAT.inject_hooks(U, st, band) if st else []
                                try: out.append({"strength": st, "band": band, "answers": [LESION.answer(q, 60)["text"] for q in qs]})
                                finally:
                                    for h in hk: h.remove()
                        r = {"concept": c, "questions": qs, "results": out}
                    elif act == "concept_knockout":
                        # Delete every neuron that lights up for the concept, then look again: does the concept
                        # vanish, or do other neurons carry it (it is spread across the weights)?
                        # It restores the model between rounds, which would also remove a lesion made in Tipping point: refuse
                        # instead of wiping it silently.
                        ls = lesion_summary()
                        if ls is not None: return self._send(409, {"error": f"A Tipping-point lesion is active on the inside engine ({ls.get('kind', 'a change')}). Delete what is lit restores the model between rounds, which would remove that lesion. Restore the model in Tipping point first, then run this again.", "lesion": ls})
                        if LESION is None: LESION = lesion_mod.Lesion(MODEL, MODEL_PATH)
                        c, fac, k = body.get("concept", ""), body.get("facets"), int(body.get("k", 240)); q = body.get("question") or f"In two sentences, what is {c}?"
                        def summ(m): return {"signal": round(sum(n["score"] for n in m["nodes"]), 1), "nodes": len(m["nodes"]), "core": m["core_count"],
                                             "slices": [[x["facet"], x["strength"]] for x in m["slices"]], "depth": m["depth"], "light": m["light"]}
                        rounds = max(1, min(6, int(body.get("rounds", 4)))); n_ctrl = max(1, min(5, int(body.get("controls", 3))))
                        floor_words = [w for w in (body.get("floor_words") or ["a pencil", "the weather"]) if str(w).strip()][:3]
                        LESION.restore(); first = ANAT.concept_map(c, fac, k, bins); ans0 = LESION.answer(q)
                        gone = {}; steps = []; cur = first; seen = set()
                        try:
                            for rd in range(rounds):
                                new_nodes = [n for n in cur["nodes"] if (n["layer"], n["neuron"]) not in seen]
                                for n in new_nodes: gone.setdefault(n["layer"], []).append(n["neuron"]); seen.add((n["layer"], n["neuron"]))
                                LESION.restore(); LESION._mask_neurons(gone)
                                cur = ANAT.concept_map(c, fac, k, bins); ans = LESION.answer(q)
                                steps.append({"round": rd + 1, "deleted_total": len(seen), "deleted_now": len(new_nodes), "after": summ(cur),
                                              "remaining_signal": round(summ(cur)["signal"] / max(1e-6, summ(first)["signal"]), 3), "answer": ans["text"],
                                              "new_carriers_by_layer": sorted({n["layer"] for n in cur["nodes"] if (n["layer"], n["neuron"]) not in seen})})
                            # the same total deletion, sparing layers 0-2 (the word form)
                            # (a shallow model, like the 2-layer toy Llama, has no separate word-form layers: nothing is spared)
                            LESION.restore(); LESION._mask_neurons({l: js for l, js in gone.items() if l > 2 or ANAT.L < 8}); ans2 = LESION.answer(q)
                            # Control: the same number of random neurons in the same layers, several seeds. How much signal
                            # does a re-picked top-k set keep when the deletion was not aimed at the concept?
                            s0 = max(1e-6, summ(first)["signal"]); rnd_ctrl = []
                            for sd in range(n_ctrl):
                                g = torch.Generator().manual_seed(sd)
                                LESION.restore(); LESION._mask_neurons({l: torch.randperm(ANAT.I, generator=g)[:len(js)].tolist() for l, js in gone.items()})
                                cm = ANAT.concept_map(c, fac, k, bins); e = {"seed": sd, "remaining_signal": round(summ(cm)["signal"] / s0, 3), "nodes": len(cm["nodes"])}
                                if sd == 0: e["answer"] = LESION.answer(q)["text"]
                                rnd_ctrl.append(e)
                        finally: LESION.restore()
                        # Floor: what a re-picked top-k set reaches for unrelated words on the intact model (same generic
                        # sentences, same controls), as a share of the concept's intact signal.
                        floor = []
                        for w in floor_words:
                            try: fm = ANAT.concept_map(w, None, k, bins); floor.append({"word": w, "signal": summ(fm)["signal"], "ratio": round(summ(fm)["signal"] / s0, 3)})
                            except Exception as fe: floor.append({"word": w, "error": str(fe)[:200]})
                        # Not nodes but a direction: erase the concept's direction from the residual stream instead, and ask
                        # the same question plus a control question about something else (is only the concept gone?).
                        qc = body.get("control_question") or "In two sentences, what is a hammer?"   # not one of the comparison words
                        band = body.get("erase_layers") or ([1, ANAT.L - 2] if ANAT.L >= 8 else [0, ANAT.L - 1])   # a shallow model: every layer
                        U = ANAT.concept_directions(c, fac); hk = ANAT.erase_hooks(U, band)
                        try: ans_dir = LESION.answer(q); ans_dir_ctrl = LESION.answer(qc)
                        finally:
                            for h in hk: h.remove()
                        ans_ctrl = LESION.answer(qc)
                        rr = [x["remaining_signal"] for x in rnd_ctrl]; fr = [x["ratio"] for x in floor if "ratio" in x]
                        r = {"concept": c, "question": q, "before": summ(first), "answer_before": ans0["text"], "rounds": steps, "answer_form_spared": ans2["text"], "direction_erased": {"answer": ans_dir["text"], "layers": band, "control_question": qc, "control_before": ans_ctrl["text"], "control_after": ans_dir_ctrl["text"]},
                             "random_control": {"seeds": rnd_ctrl, "range": [min(rr), max(rr)] if rr else None, "deleted": len(seen), "per_layer": {str(l): len(js) for l, js in gone.items()}},
                             "floor": {"words": floor, "range": [min(fr), max(fr)] if fr else None, "k": k},
                             "note": "Each round deletes (zero-ablation: masks to zero, undone afterwards) every neuron that lights up for the concept, then maps it again on the damaged model: neurons that light up now were not lit before, so they are backups or carriers that were below the cut. The answer to the question is recorded each round. answer_form_spared: the final deletion without layers 0-2 (the word form). random_control: the same number of random neurons per layer deleted instead (one seed each), the concept re-mapped. floor: the signal a re-picked top-k set reaches for unrelated words on the intact model, as a share of the concept's intact signal."}
                    else: return self._send(400, {"error": "unknown action"})
                return self._send(200, r)
            if self.path.rstrip("/") == "/connector":
                if CONN is None: CONN = connector_mod.Connector(MODEL)
                with MODEL.lock: r = {"items": [CONN.score(t) for t in (body.get("texts") or [body.get("text", "")])]}
                return self._send(200, r)
            if self.path.rstrip("/") == "/fight":
                if FIGHT is None: FIGHT = fight_mod.Fight(MODEL)
                with MODEL.lock: r = {"items": [FIGHT.classify(t) for t in (body.get("texts") or [body.get("text", "")])]}
                return self._send(200, r)
            if self.path.rstrip("/") == "/sense":
                if CKPT is None: CKPT = ckpt_mod.Checkpoints()
                if SENSE is None: SENSE = sensing_mod.Sensing(CKPT, MODEL)
                act = body.get("action")
                if act == "surprise":
                    with MODEL.lock: r = SENSE.surprise(body.get("text", ""), int(body.get("top", 8)), int(body.get("window", 3)))
                elif act == "flip":
                    r = SENSE.flip(body.get("claim", ""), body.get("observations", []), body.get("question", ""), body.get("prior", []), body.get("new", []), body.get("framing", ""))
                elif act == "cycle":
                    r = {"cycles": sensing_mod.cycle(body.get("steps", []), body.get("values", []), body.get("chance"), float(body.get("tol", 0.2)))}
                else:
                    return self._send(400, {"error": "unknown action"})
                return self._send(200, r)
            if self.path.rstrip("/") == "/lesion":
                if LESION is None: LESION = lesion_mod.Lesion(MODEL, MODEL_PATH)
                act = body.get("action", "scorecard"); qs = body.get("questions") or ([body["question"]] if body.get("question") else [])
                LESION.extra_tests = body.get("tests") or []   # the user's own tests (Understanding view), scored apart from the headline
                with MODEL.lock, _undo_on_error(act):
                    if act == "baseline": r = {"reference": LESION.baseline(qs), "state": LESION.state}
                    elif act == "restore": r = {"state": LESION.restore()}
                    elif act == "save": r = {"saved": LESION.save(body.get("name")), "state": LESION.state}
                    elif act == "structural": LESION.structural(float(body.get("frac_neurons", 0)), float(body.get("frac_heads", 0)), body.get("mode", "random"), int(body.get("seed", 0)), qs[0] if qs else None); r = LESION.scorecard(questions=qs)
                    elif act == "manual": LESION.manual(body.get("layers"), body.get("heads"), int(body.get("neurons_per_layer", 0)), qs[0] if qs else None); r = LESION.scorecard(questions=qs)
                    elif act == "weights": LESION.weights(float(body.get("frac", 0.1)), body.get("scope", "mlp"), body.get("criterion", "random"), int(body.get("seed", 0))); r = LESION.scorecard(questions=qs)
                    elif act == "concept_find": f = LESION.concept_neurons(body.get("concept", ""), body.get("keep", ""), int(body.get("k", 300))); f.pop("_per_layer", None); r = f
                    elif act == "concept": LESION.delete_concept(body.get("concept", ""), body.get("keep", ""), int(body.get("k", 300))); r = LESION.scorecard(questions=qs)
                    elif act == "sweep":
                        levels = body.get("levels") or [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]; kind = body.get("kind", "structural"); points = []
                        for f in levels:
                            if kind == "weights": LESION.weights(float(f), body.get("scope", "mlp"), body.get("criterion", "random"), int(body.get("seed", 0)))
                            else: LESION.structural(float(f), float(f) if body.get("heads_too", True) else 0.0, body.get("mode", "random"), int(body.get("seed", 0)), qs[0] if qs else None)
                            c = LESION.scorecard(questions=qs); c["level"] = f; points.append(c)
                        LESION.restore(); r = {"kind": kind, "points": points}
                    else: r = LESION.scorecard(questions=qs)
                r["seconds"] = round(time.time() - t0, 1)
                return self._send(200, r)
            if self.path.rstrip("/") == "/heads":
                r = MODEL.heads_scan(int(body.get("n", 40)), int(body.get("trials", 3))); r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            if self.path.rstrip("/") == "/ablate_heads":
                r = MODEL.ablate_heads(body.get("heads"), body.get("top_k"), body.get("plan", "top"), int(body.get("seed", 0)), body.get("text"), body.get("invented"))
                # Random control for the chosen heads: the same number of random heads, `controls` seeds (default 5, 0 = none),
                # each measured like the chosen set; the range of each measure is what "no different from random" means.
                n_ctrl = max(0, min(10, int(body.get("controls", 5 if r.get("plan") != "random" else 0))))
                if n_ctrl and r.get("heads"):
                    import random as _rnd
                    Hn = MODEL.model.config.num_attention_heads; total = MODEL.L * Hn; kk = len(r["heads"]); ctrl = []
                    for sd in range(n_ctrl):
                        hs = [[i // Hn, i % Hn] for i in _rnd.Random(sd).sample(range(total), min(kk, total))]
                        c = MODEL.ablate_heads(hs, None, "random", sd, body.get("text"), body.get("invented"))
                        ctrl.append({"seed": sd, "heads": hs, "after": c["after"]})
                    rng = {}
                    for key in ("copy_nats_saved", "text_bits_per_token", "invented_second_over_first"):
                        vals = [c["after"][key] for c in ctrl if c["after"].get(key) is not None]
                        rng[key] = [min(vals), max(vals)] if vals else None
                    r["control"] = {"seeds": ctrl, "range": rng, "note": "random.Random(seed).sample over all layer x head pairs, the same count as the chosen heads"}
                r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            if self.path.rstrip("/").startswith("/ckpt_"):
                if CKPT is None: CKPT = ckpt_mod.Checkpoints()
                route = self.path.rstrip("/")
                if route == "/ckpt_list": return self._send(200, CKPT.list())
                if route == "/ckpt_count": return self._send(200, CKPT.count(body.get("text", ""), int(body.get("limit", 2048))))
                if route == "/ckpt_ask":
                    with MODEL.lock: r = CKPT.ask(body.get("step", 1000), body.get("prompt", ""), body.get("text"), int(body.get("n_new", 40)), bool(body.get("heads", True)))
                    return self._send(200, r)
                if route == "/ckpt_test":
                    with MODEL.lock: r = CKPT.when_learned(body.get("prompt", ""), body.get("good", []), body.get("bad", []), body.get("steps"))
                    return self._send(200, r)
            if self.path.rstrip("/") == "/embed":
                r = MODEL.embed_info(body.get("text", ""), int(body.get("k", 8))); r["seconds"] = round(time.time() - t0, 2)
                return self._send(200, r)
            # Every route above returns; what reaches here is POST / or /trace, or a route this server does not have.
            route = self.path.split("?")[0].rstrip("/")
            if route not in ("", "/trace"): return self._send(404, {"error": f"no POST route {route} on the inside engine (an older trace_server.py, or a typo)"})
            prompt, lens, think = body.get("prompt", ""), _lens_arg(body), bool(body.get("think", False))
            r = MODEL.trace(prompt, int(body.get("n_new", 16)), lens, think)
            r.update({"prompt": prompt, "lens": lens, "think": think})   # what was recorded, so knock-outs re-ask exactly this
            try:
                # z-scores of the top neurons against their normal range on ordinary text (when built), and which prompt
                # tokens are template, E-level framing (system) or the typed text.
                if ANAT is None: ANAT = anatomy_mod.Anatomy(MODEL)
                st = ANAT.stats()
                for x in r["tokens"]: x["mlp_z"] = ANAT.zscores(x["mlp"]) if st is not None else None
                r["neuron_stats"] = {"tokens": st["tokens"], "windows": st["windows"], "built_at": st.get("built_at"), "corpus": os.path.basename(st.get("corpus", ""))} if st is not None else None
                seg = anatomy_mod.prompt_segments(MODEL, prompt, lens_text(lens), think)
                r["prompt_segments"] = seg if seg and len(seg) == r["prompt_len"] else None
            except Exception as e:
                r["extras_error"] = f"{type(e).__name__}: {e}"[:300]
            r["seconds"] = round(time.time() - t0, 1)
            self._send(200, r)
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"[:4000]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.environ.get("PSI_TRACE_MODEL") or None, help="Hugging Face model folder (or PSI_TRACE_MODEL)"); ap.add_argument("--port", type=int, default=4311)
    a = ap.parse_args(); PORT = a.port
    if not a.model: ap.error("no model folder: pass --model <a Hugging Face model folder> or set PSI_TRACE_MODEL")
    if not TOKEN: TOKEN = secrets.token_hex(32); print(f"no PSI_TRACE_TOKEN set: POSTs need header X-PSI-Token: {TOKEN}", flush=True)
    MODEL = Tracer(a.model); MODEL_PATH = a.model
    print(f"trace server on http://127.0.0.1:{a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()
