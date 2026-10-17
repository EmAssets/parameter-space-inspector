"""
X. A ladder for models: in how many ways, and at what level, does a model store one concept (red)?

E0-E14 order the world by what depends on what. M0-M8 order a model's storage the same way: each level is built
out of the one below and holds something the one below cannot.

  M0 words        static weights only: is ' red' stored next to crimson, blood, stop... and next to 红, लाल, rojo?
  M1 units        does a single MLP neuron carry a sense of red (politics, love, finance...) that transfers to other languages?
  M2 features     does a direction (many neurons together) carry the sense, across languages?
  M3 geometry     is the arrangement of meanings shaped like the world: colours like CIELAB, senses alike in every language?
  M4 wiring       are red's neurons wired to each other (output of one aimed at the input of a later one) beyond chance?
  M5 composition  is 'red in China' -> luck built in the forward pass from red + context, beyond the context alone?
  M6 redundancy   when a layer that carries red is removed, do later layers make up for it (self-repair)?
  M7 universality do different models, trained apart, arrange red's senses the same way? (--aggregate)
  M8 self-access  does what the model says about red's associations match where they sit inside it?

Every score is tested 100+ times (random splits, permutations or bootstraps) and against the same architecture untrained
(Pythia step 0, or random init with --init): a level counts only where training made it, not where chance or the
architecture already gives it.

  python research/model_ladder.py --model <models>/pythia-410m/step143000 --tag pythia-410m@143000
  python research/model_ladder.py --model <models>/qwen3-4b-bf16 --tag qwen3-4b@init --init
  python research/model_ladder.py --aggregate
  python research/model_ladder.py --model <models>/pythia-1b/step143000 --tag pythia-1b@shuffle --init shuffle --methods 2

--methods 2 (default 1 = the stored runs' code paths): tie-aware M1 for every model, M4 selection-matched null
(M4_wiring.matched_null), M5 mirrored contexts (M5_composition.interaction; the old DiD is also M5_composition.association),
M8 crossed colours (M8_self_access.crossed). Always written (new fields only): provenance, twin_kind / init_std, tokens,
M1 per-sense AUCs, M2 English accuracy with the sense cue before / after the word, M6 compensation (= self_repair),
compensation_split (layers chosen on half the facts) and fact_acc_balanced, and per-item scores in the npz
(items_m1 [lang, sense], items_m2 (+ items_m2_lang, 0 = en), items_m5, items_m6 [direct/total, layer, fact],
items_m8 [said, held, inside(, said blue, said green)], tok_<lang>_{pos,subtokens,prefix,length}).
"""
import argparse, json, math, os, sys, time, itertools
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_ladder_data as D

DEV = os.environ.get("PSI_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")   # cpu only when no GPU is visible (tests)
OUT = os.path.join(HERE, "results", "model_ladder"); os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ladder.log")
SEED = 0                                                   # --seed: a replication re-draws every split, permutation, bootstrap and the untrained init
METHODS = 1                                                # --methods 2: tie-aware M1 for every model, M4 selection-matched null, M5 and M8
                                                           # crossed designs; the default 1 keeps every stored number reproducible
BS_LAST, BS_CONT, BS_MOE = 48, 32, 8                       # batch sizes (recorded in provenance)


def draw(n):
    return np.random.default_rng(SEED * 1000 + n)
SENSE_NAMES = list(D.SENSES)


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")


def rank(x): return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
def spearman(a, b): return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])
def unmark(s):
    a = s.index("["); b = s.index("]")
    return s[:a] + s[a + 1:b] + s[b + 1:], (a, b - 1)


def load_tokenizer(path):
    """AutoTokenizer, unless it fails a round trip: transformers 5 maps DeepSeek's byte-level BPE (tokenizer_class
    LlamaTokenizerFast) onto a SentencePiece Llama tokenizer that drops every space. Then use tokenizer.json as is."""
    tok = AutoTokenizer.from_pretrained(path)
    probe = "The car is red."
    if probe in tok.decode(tok(probe, add_special_tokens=False).input_ids):
        tok.psi_fallback = False; return tok
    import json
    from transformers import PreTrainedTokenizerFast
    from tokenizers.processors import TemplateProcessing
    c = json.load(open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8"))
    sp = {k: (v["content"] if isinstance(v, dict) else v) for k, v in c.items() if k in ("bos_token", "eos_token", "unk_token", "pad_token") and v}
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"), **sp)
    if c.get("add_bos_token") and tok.bos_token:
        b = tok.bos_token; tok._tokenizer.post_processor = TemplateProcessing(single=f"{b} $A", pair=f"{b} $A {b} $B", special_tokens=[(b, tok.bos_token_id)])
    assert probe in tok.decode(tok(probe, add_special_tokens=False).input_ids), "tokenizer round trip still fails"
    tok.psi_fallback = True
    return tok


class Int8Linear(torch.nn.Module):
    """Row-wise absmax int8 weight (weight.SCB = row scale, as bitsandbytes), dequantised per call."""
    def __init__(self, out_f, in_f, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_f, in_f, dtype=torch.int8, device="cuda"), requires_grad=False)
        self.weight.SCB = torch.empty(out_f, dtype=torch.float32, device="cuda")
        self.bias = torch.nn.Parameter(torch.zeros(out_f, dtype=torch.bfloat16, device="cuda"), requires_grad=False) if bias else None

    @torch.no_grad()
    def set(self, w):
        for a in range(0, w.shape[0], 2048):
            r = w[a:a + 2048].float(); amax = r.abs().amax(1).clamp_min(1e-8)
            self.weight.data[a:a + len(r)] = torch.round(r / amax[:, None] * 127).to(torch.int8); self.weight.SCB[a:a + len(r)] = amax

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype) * (self.weight.SCB / 127).to(x.dtype)[:, None], self.bias)


# ── mixture-of-experts layers kept in MXFP4 (gpt-oss): 4-bit E2M1 values, one power-of-two scale per 32 ──
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def mx_rows(blocks, scales, dtype=torch.float32):
    """Rows of an MXFP4 matrix: blocks [..., G, 16] uint8, scales [..., G] uint8 -> [..., G*32] (exact in bf16 too:
    every E2M1 value times a power of two). Written in place with int32 indices: a whole expert is decoded with ~100 MB
    of scratch, not ~300 MB (the larger peaks pushed a 16 GB card into Windows' shared-memory spill)."""
    lut = torch.tensor(FP4, device=blocks.device, dtype=dtype)
    v = torch.empty(*blocks.shape, 2, device=blocks.device, dtype=dtype)                            # [..., G, 16, 2]
    v[..., 0] = lut[(blocks & 0x0F).int()]; v[..., 1] = lut[(blocks >> 4).int()]
    v = v.flatten(-2); v.mul_(torch.exp2(scales.to(dtype) - 127)[..., None])                         # [..., G, 32]
    return v.flatten(-2)


def mx_pack(x):
    """float [R, C] -> MXFP4 blocks [R, C/32, 16], scales [R, C/32] (round to nearest on the E2M1 grid)."""
    R, C = x.shape; b = x.float().reshape(R, C // 32, 32)
    e = torch.ceil(torch.log2(b.abs().amax(-1).clamp_min(1e-30) / 6.0)).clamp(-127, 127)
    q = b / torch.exp2(e)[..., None]
    mag = torch.bucketize(q.abs(), torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device))
    idx = (mag + 8 * ((q < 0) & (mag > 0)).long()).to(torch.uint8)
    return (idx[..., 0::2] | (idx[..., 1::2] << 4)), (e + 127).to(torch.uint8)


class MoEExperts(torch.nn.Module):
    """gpt-oss experts with weights left in MXFP4 on the GPU, decoded one expert at a time (bf16 would be ~40 GB).
    With .capture set, it also keeps every expert neuron's activation (routing weight included; 0 where not routed)."""
    def __init__(self, cfg):
        super().__init__()
        self.E, self.I, self.H = cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size
        self.alpha, self.limit = 1.702, getattr(cfg, "swiglu_limit", 7.0)
        self.t, self.capture, self.acts = {}, False, None

    def load(self, name, t):
        self.t[name] = t if name.endswith(("_blocks", "_scales")) else t.to(torch.bfloat16)

    def w(self, e, proj):                                   # one expert's matrix [out, in], bf16
        return mx_rows(self.t[proj + "_blocks"][e], self.t[proj + "_scales"][e], torch.bfloat16)

    def vec(self, kind, n):                                 # one neuron's input (gate / up) or output weights
        e, u = divmod(int(n), self.I)
        if kind == "out":
            g, j = divmod(u, 32); byte = self.t["down_proj_blocks"][e, :, g, j // 2]
            nib = (byte & 0x0F) if j % 2 == 0 else (byte >> 4)
            return torch.tensor(FP4, device=byte.device)[nib.long()] * torch.exp2(self.t["down_proj_scales"][e, :, g].float() - 127)
        r = 2 * u + (kind == "up")
        return mx_rows(self.t["gate_up_proj_blocks"][e, r], self.t["gate_up_proj_scales"][e, r])

    def forward(self, hs, router_indices=None, routing_weights=None):
        out = torch.zeros_like(hs)
        acts = hs.new_zeros(hs.shape[0], self.E * self.I) if self.capture else None
        for e in torch.unique(router_indices).tolist():
            tok, kpos = torch.where(router_indices == e)
            gu = hs[tok] @ self.w(e, "gate_up_proj").T + self.t["gate_up_proj_bias"][e]
            gate, up = gu[..., ::2].clamp(max=self.limit), gu[..., 1::2].clamp(-self.limit, self.limit)
            g = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
            wt = routing_weights[tok, kpos, None].to(g.dtype)
            y = g @ self.w(e, "down_proj").T + self.t["down_proj_bias"][e]
            out.index_add_(0, tok, (y * wt).to(hs.dtype))
            if acts is not None: acts[tok, e * self.I:(e + 1) * self.I] = (g * wt).to(acts.dtype)
        self.acts = acts
        return out


class NeuronView:
    """Expert neurons numbered expert * I + unit, read like a Linear's rows (inputs) or columns (outputs)."""
    def __init__(self, ex, kind): self.ex, self.kind = ex, kind
    def vec(self, n): return self.ex.vec(self.kind, n)


class CpuEmbedding(torch.nn.Module):
    """Input embedding kept in RAM (201k x 2880): only the looked-up rows go to the GPU."""
    def __init__(self, V, H):
        super().__init__(); self.weight = torch.nn.Parameter(torch.empty(V, H, dtype=torch.bfloat16), requires_grad=False)
    def forward(self, ids): return torch.nn.functional.embedding(ids.cpu(), self.weight).to(ids.device)


def load_int8(path, cfg):
    """7B on a 16 GB card without staging the model in RAM: build on meta, stream each tensor to the GPU,
    quantise layer Linears to int8 there (transformers' 8-bit loader held ~15 GB of CPU copies)."""
    import glob
    from safetensors import safe_open
    moe = getattr(cfg, "model_type", "") == "gpt_oss"
    if moe: cfg._attn_implementation = "eager"; cfg.quantization_config = None          # attention sinks: eager only; MXFP4 handled here
    with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    base = getattr(m, "gpt_neox", None) or getattr(m, "model")
    if moe:
        base.embed_tokens = CpuEmbedding(cfg.vocab_size, cfg.hidden_size)
        for ly in base.layers: ly.mlp.experts = MoEExperts(cfg)
    for name, mod in list(base.layers.named_modules()):
        if isinstance(mod, torch.nn.Linear):
            parent, _, leaf = name.rpartition(".")
            setattr(base.layers.get_submodule(parent) if parent else base.layers, leaf, Int8Linear(mod.out_features, mod.in_features, mod.bias is not None))
    def tensors():                                         # safetensors, or pickled .bin shards memory-mapped (DeepSeek)
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        for f in files:
            with safe_open(f, framework="pt", device="cpu" if moe else "cuda:0") as st:
                for k in st.keys(): t = st.get_tensor(k); yield k, (t if "embed_tokens" in k or not moe else t.to("cuda"))
        if not files:
            for f in sorted(glob.glob(os.path.join(path, "pytorch_model*.bin"))):
                sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                for k in list(sd): yield k, sd.pop(k).to("cuda")
                del sd
    for k, t in tensors():
        mod_name, _, pname = k.rpartition(".")
        try: mod = m.get_submodule(mod_name)
        except AttributeError: continue
        if isinstance(mod, Int8Linear) and pname == "weight": mod.set(t)
        elif isinstance(mod, MoEExperts): mod.load(pname, t)
        elif isinstance(mod, CpuEmbedding): mod.weight.data = t.to(torch.bfloat16)
        elif moe and "norm" in mod_name: setattr(mod, pname, torch.nn.Parameter(t.float(), requires_grad=False))   # norms stay fp32, as in gpt-oss
        else: setattr(mod, pname, torch.nn.Parameter(t.to(torch.bfloat16) if t.is_floating_point() else t, requires_grad=False))
        del t
    if getattr(cfg, "tie_word_embeddings", False): m.get_output_embeddings().weight = m.get_input_embeddings().weight
    if hasattr(base, "rotary_emb"): base.rotary_emb = type(base.rotary_emb)(config=cfg, device="cuda")
    for ly in base.layers:
        ra = getattr(getattr(ly, "self_attn", None), "rotary_emb", None)
        if ra is not None: ly.self_attn.rotary_emb = type(ra)(config=cfg, device="cuda")
    left = [n for n, p in list(m.named_parameters()) + list(m.named_buffers()) if p.device.type == "meta"]
    assert not left, f"weights not loaded: {left[:5]}"
    if moe:
        need = {f"{p}_{x}" for p in ("gate_up_proj", "down_proj") for x in ("blocks", "scales", "bias")}
        miss = [i for i, ly in enumerate(base.layers) if set(ly.mlp.experts.t) != need]
        assert not miss, f"expert weights not loaded in layers {miss[:5]}"
    return m



def wait_for_gpu(need_gb):
    """Wait until the GPU has need_gb free (PSI_GPU_NEED_GB overrides). Loading into less makes Windows spill into
    shared system memory, silently and ~100x slower; another program (e.g. the app's llama-server) may hold the card."""
    need = float(os.environ.get("PSI_GPU_NEED_GB", need_gb)); t0 = time.time(); last = -1e9
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free >= need: break
        if time.time() - last > 300: log(f"   waiting for GPU memory: {free:.1f} GB free, need {need:.1f} GB (another program holds the card)"); last = time.time()
        time.sleep(30)
    if time.time() - t0 > 1: log(f"   GPU free after {(time.time() - t0) / 60:.0f} min")
    return round((time.time() - t0) / 60, 1)


class Model:
    def __init__(self, path, init=False, seed=0, q8=False):
        """init: False (trained weights), True / 'normal' (random init as the architecture's _init_weights, std =
        initializer_range) or 'shuffle' (the trained weights, each tensor's entries shuffled within itself: the same
        value distribution per matrix, no learned arrangement)."""
        init = "normal" if init is True else (init or False)
        self.init_kind = init; self.minutes_waited = 0.0; self.mem_fraction = None
        self.tok = load_tokenizer(path)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token or self.tok.unk_token or self.tok.decode([0])
        cfg = AutoConfig.from_pretrained(path)
        n_guess = getattr(cfg, "num_hidden_layers", 24) * getattr(cfg, "hidden_size", 1024) ** 2 * 12
        dtype = torch.float32 if n_guess < 6e8 else torch.bfloat16      # fp32 up to ~400M params; bf16 above (several runs share the GPU)
        self.moe = getattr(cfg, "model_type", "") == "gpt_oss"
        if self.moe: q8 = True                                     # MXFP4 experts + int8 attention: the only way it fits on 16 GB
        if q8 and torch.cuda.is_available():
            self.minutes_waited = wait_for_gpu(13.0 if self.moe else n_guess / 12 * 13 / 2 ** 30 + 2.0)   # int8 ≈ 1 byte/param, plus working memory
            # Stay inside the memory that is free now. On Windows (WDDM) a process that grows past the card spills into
            # shared system memory without an error, and runs ~100x slower (seen: gpt-oss M8, 2026-10-10). With a cap,
            # the allocator frees its cache and retries instead, or fails loudly.
            free, total = torch.cuda.mem_get_info()
            self.mem_fraction = max(0.1, (free - 384 * 2 ** 20) / total); torch.cuda.set_per_process_memory_fraction(self.mem_fraction)
        if q8:                                                     # 7B models: LLM.int8 weights so they fit on a 16 GB card
            self.m = load_int8(path, cfg)
            if init == "normal": (self.reinit_moe if self.moe else self.reinit_q8)(seed, getattr(cfg, "initializer_range", 0.02))   # untrained copy made on the GPU (a bf16 7B copy needs ~15 GB RAM)
        elif init == "normal":
            torch.manual_seed(seed)
            with torch.device(DEV): self.m = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        else:
            self.m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        if not q8: self.m.to(DEV)
        self.init_std = float(getattr(cfg, "initializer_range", 0.02)) if init == "normal" else None
        self.q8 = bool(q8); self.dtype_name = str(dtype).replace("torch.", "")
        if init == "shuffle": self.shuffle_weights(seed)
        self.experts = [ly.mlp.experts for ly in (getattr(self.m, "model", None) or self.m.gpt_neox).layers] if self.moe else []
        self.m.eval(); self.cfg = cfg
        base = getattr(self.m, "gpt_neox", None) or getattr(self.m, "model", None)
        self.layers = list(base.layers)
        self.norm = getattr(base, "final_layer_norm", None) or getattr(base, "norm")
        self.L = len(self.layers)
        self.WU = self.m.get_output_embeddings().weight
        self.WE = self.m.get_input_embeddings().weight
        self.logit_scale = 1.0 / float(getattr(cfg, "logits_scaling", 1.0) or 1.0)
        self.downs, self.ins, self.gin, self.gout = [], [], [], []
        for ly in self.layers:
            mlp = ly.mlp
            if self.moe:                                           # expert neurons: expert * I + unit
                ex = mlp.experts; self.downs.append(NeuronView(ex, "out")); self.ins.append([(NeuronView(ex, "gate"), 0), (NeuronView(ex, "up"), 0)])
                self.gin.append(ly.post_attention_layernorm.weight); self.gout.append(None); continue
            down = next(getattr(mlp, n) for n in ("down_proj", "dense_4h_to_h", "fc2", "c_proj") if hasattr(mlp, n))
            self.downs.append(down)
            if hasattr(mlp, "dense_h_to_4h"): ins = [(mlp.dense_h_to_4h, 0)]
            elif hasattr(mlp, "gate_up_proj"): ins = [(mlp.gate_up_proj, 0), (mlp.gate_up_proj, mlp.gate_up_proj.weight.shape[0] // 2)]
            else: ins = [(mlp.gate_proj, 0), (mlp.up_proj, 0)]
            self.ins.append(ins)
            if hasattr(ly, "post_feedforward_layernorm"):          # OLMo2: MLP output is normed on its way into the residual
                self.gin.append(None); self.gout.append(ly.post_feedforward_layernorm.weight)
            else:
                self.gin.append(getattr(ly, "post_attention_layernorm", None).weight if getattr(ly, "post_attention_layernorm", None) is not None else None); self.gout.append(None)
        self.I = self.experts[0].E * self.experts[0].I if self.moe else self.downs[0].weight.shape[1]
        self.is_ln = isinstance(self.norm, torch.nn.LayerNorm)

    # one neuron's weights, de-quantised when the layer is LLM.int8 (int8 rows times per-row scale / 127)
    @torch.no_grad()
    def reinit_q8(self, seed, std):
        """Random-init an 8-bit model in place, as HF _init_weights does: normal(0, std) weights, zero biases, unit norms."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        for name, mod in self.m.named_modules():
            if isinstance(mod, Int8Linear):
                w = mod.weight
                for a in range(0, w.shape[0], 1024):
                    r = torch.randn(min(1024, w.shape[0] - a), w.shape[1], device="cuda", generator=g) * std
                    amax = r.abs().amax(1).clamp_min(1e-8)
                    w.data[a:a + len(r)] = torch.round(r / amax[:, None] * 127).to(torch.int8)
                    w.SCB[a:a + len(r)] = amax.to(w.SCB.dtype)
                if mod.bias is not None: mod.bias.zero_()
            elif isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
                mod.weight.copy_(torch.randn(mod.weight.shape, device="cuda", generator=g).to(mod.weight.dtype) * std)
                if getattr(mod, "bias", None) is not None: mod.bias.zero_()
            elif "norm" in type(mod).__name__.lower() and getattr(mod, "weight", None) is not None:
                mod.weight.fill_(1.0)
                if getattr(mod, "bias", None) is not None: mod.bias.zero_()

    @torch.no_grad()
    def reinit_moe(self, seed, std):
        """Random-init gpt-oss in place as its _init_weights does: normal(0, std) weights (experts quantised to MXFP4),
        router and sinks normal, biases 0, norms 1. Embedding drawn on the CPU, everything else on the GPU."""
        g = torch.Generator(device="cuda").manual_seed(seed); gc = torch.Generator().manual_seed(seed)
        rn = lambda shape: torch.randn(shape, device="cuda", generator=g) * std
        for name, mod in self.m.named_modules():
            if isinstance(mod, MoEExperts):
                for proj, (R, C) in (("gate_up_proj", (2 * mod.I, mod.H)), ("down_proj", (mod.H, mod.I))):
                    for e in range(mod.E):
                        b, sc = mx_pack(rn((R, C))); mod.t[proj + "_blocks"][e] = b; mod.t[proj + "_scales"][e] = sc
                    mod.t[proj + "_bias"].zero_()
            elif isinstance(mod, Int8Linear):
                w = mod.weight
                for a in range(0, w.shape[0], 1024):
                    r = rn((min(1024, w.shape[0] - a), w.shape[1])); amax = r.abs().amax(1).clamp_min(1e-8)
                    w.data[a:a + len(r)] = torch.round(r / amax[:, None] * 127).to(torch.int8); w.SCB[a:a + len(r)] = amax.to(w.SCB.dtype)
                if mod.bias is not None: mod.bias.zero_()
            elif isinstance(mod, CpuEmbedding):
                for a in range(0, mod.weight.shape[0], 16384):
                    n = min(16384, mod.weight.shape[0] - a); mod.weight.data[a:a + n] = (torch.randn((n, mod.weight.shape[1]), generator=gc) * std).to(mod.weight.dtype)
            elif isinstance(mod, torch.nn.Linear):
                for a in range(0, mod.weight.shape[0], 16384):
                    n = min(16384, mod.weight.shape[0] - a); mod.weight.data[a:a + n] = rn((n, mod.weight.shape[1])).to(mod.weight.dtype)
                if mod.bias is not None: mod.bias.zero_()
            elif type(mod).__name__ == "GptOssTopKRouter":
                mod.weight.copy_(rn(mod.weight.shape)); mod.bias.copy_(rn(mod.bias.shape))
            elif "norm" in type(mod).__name__.lower() and getattr(mod, "weight", None) is not None:
                mod.weight.fill_(1.0)
            if hasattr(mod, "sinks"): mod.sinks.copy_(rn(mod.sinks.shape))

    @torch.no_grad()
    def shuffle_weights(self, seed):
        """The 'shuffled' twin: every weight tensor keeps its own values, moved to random positions within itself, so
        each matrix has the trained value distribution (scale, heavy tails) but no learned arrangement. Tensors over
        2^26 entries: rows shuffled, then the columns of each 4096-row block (one permutation per block). int8 and
        MXFP4 matrices are decoded, shuffled and re-encoded; tied input/output embeddings are shuffled once."""
        g = torch.Generator().manual_seed(int(seed) + 7)
        def perm(n, dev): return torch.randperm(n, generator=g).to(dev)
        def shuf(w):                                          # tensor -> shuffled copy (same device and dtype)
            if w.numel() <= 2 ** 26 or w.dim() < 2:
                f = w.reshape(-1); return f[perm(f.numel(), f.device)].reshape(w.shape)
            out = w[perm(w.shape[0], w.device)]
            for a in range(0, out.shape[0], 4096): out[a:a + 4096] = out[a:a + 4096][:, perm(out.shape[1], out.device)]
            return out
        seen = set()
        for name, mod in self.m.named_modules():
            if isinstance(mod, MoEExperts):
                for proj in ("gate_up_proj", "down_proj"):
                    for e in range(mod.E):
                        b, sc = mx_pack(shuf(mod.w(e, proj).float())); mod.t[proj + "_blocks"][e] = b; mod.t[proj + "_scales"][e] = sc
                    mod.t[proj + "_bias"] = shuf(mod.t[proj + "_bias"])
                continue
            if isinstance(mod, Int8Linear):
                w = mod.weight; mod.set(shuf(w.data.float() * (w.SCB.float()[:, None] / 127))); seen.add(id(w))
                if mod.bias is not None: mod.bias.data.copy_(shuf(mod.bias.data)); seen.add(id(mod.bias))
                continue
            for pn, prm in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                if id(prm) in seen or not prm.is_floating_point() or "rotary" in name or "inv_freq" in pn or prm.numel() < 2: continue
                seen.add(id(prm)); prm.data.copy_(shuf(prm.data))

    @torch.no_grad()
    def swap_words(self, a, b):
        """M0 intervention: exchange the input vectors of two words (every spelling that is one token in both:
        ' red'/' blue', ' Red'/' Blue', 'red'/'blue', 'Red'/'Blue'). Only the input embedding changes: the model
        reads one word with the other's vector; the unembedding, and everything learned downstream, stay as trained."""
        if self.WE.data_ptr() == self.WU.data_ptr():                 # tied: give the input its own copy so the output side stays trained
            emb = self.m.get_input_embeddings(); emb.weight = torch.nn.Parameter(emb.weight.detach().clone(), requires_grad=False); self.WE = emb.weight
        done = []
        for fa, fb in ((" " + a, " " + b), (" " + a.title(), " " + b.title()), (a, b), (a.title(), b.title())):
            ta, tb = self.tok(fa, add_special_tokens=False)["input_ids"], self.tok(fb, add_special_tokens=False)["input_ids"]
            if len(ta) == 1 and len(tb) == 1 and ta != tb:
                W = self.WE.data; x = W[ta[0]].clone(); W[ta[0]] = W[tb[0]]; W[tb[0]] = x; done.append([fa, fb, ta[0], tb[0]])
        assert done, f"no single-token spelling of both {a!r} and {b!r}"
        return done

    def control_pair(self, seed):
        """Two common words that occur in none of the ladder's test texts, each one token: the same-size control swap."""
        import re
        skip = () if METHODS >= 2 else getattr(D, "V2_ONLY", ())               # v1 keeps exactly the words it always had
        txt = " ".join(str(v) for k, v in vars(D).items() if k.isupper() and k != "COMMON" and k not in skip).lower()
        free = [w for w in D.COMMON if not re.search(r"\b" + re.escape(w.lower()) + r"\b", txt) and len(self.tok(" " + w, add_special_tokens=False)["input_ids"]) == 1]
        i, j = draw(seed + 77).choice(len(free), 2, replace=False); return free[i], free[j]

    @staticmethod
    def row(mod, r):
        if isinstance(mod, NeuronView): return mod.vec(r).float()
        w = mod.weight; scb = getattr(w, "SCB", None)
        return w.data[r].float() * (scb[r].float() / 127) if scb is not None else w[r].float()

    @staticmethod
    def col(mod, c):
        if isinstance(mod, NeuronView): return mod.vec(c).float()
        w = mod.weight; scb = getattr(w, "SCB", None)
        return w.data[:, c].float() * (scb.float() / 127) if scb is not None else w[:, c].float()

    # ── one forward, capturing the residual stream at every layer boundary (pre-final-norm) and MLP neuron activations
    @torch.no_grad()
    def forward_capture(self, ids, pos, neurons=False, mask=None):
        res, acts, hs = {}, {}, []
        def pre0(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]; res[0] = h
        def post(i):
            def f(mod, args, kwargs, out):
                res[i + 1] = out[0] if isinstance(out, tuple) else out
            return f
        def dn(i):
            def f(mod, args):
                acts[i] = args[0]
            return f
        hs.append(self.layers[0].register_forward_pre_hook(pre0, with_kwargs=True))
        for i, ly in enumerate(self.layers): hs.append(ly.register_forward_hook(post(i), with_kwargs=True))
        if neurons and self.moe:
            for ex in self.experts: ex.capture = True
        elif neurons:
            for i, d in enumerate(self.downs): hs.append(d.register_forward_pre_hook(dn(i)))
        try:
            out = self.m(input_ids=ids, attention_mask=mask)
            if neurons and self.moe:
                for i, ex in enumerate(self.experts): acts[i] = ex.acts.reshape(ids.shape[0], ids.shape[1], -1)
        finally:
            for h in hs: h.remove()
            for ex in self.experts: ex.capture, ex.acts = False, None
        b = torch.arange(ids.shape[0], device=ids.device)
        R = torch.stack([res[i][b, pos].float() for i in range(self.L + 1)], 1)          # [B, L+1, D]
        A = torch.stack([acts[i][b, pos].float() for i in range(self.L)], 1) if neurons else None  # [B, L, I]
        return R, A, out.logits

    def span_pos(self, text, span):
        enc = self.tok(text, return_offsets_mapping=True)
        a, b = span; best = None
        for k, (s, e) in enumerate(enc["offset_mapping"]):
            if e > s and s <= b and e > a: best = k
        return enc["input_ids"], best

    @torch.no_grad()
    def read_marked(self, sents, neurons=True):
        Rs, As = [], []
        for s in sents:
            text, span = unmark(s); ids, p = self.span_pos(text, span)
            R, A, _ = self.forward_capture(torch.tensor([ids], device=DEV), torch.tensor([p], device=DEV), neurons=neurons)
            Rs.append(R[0].half().cpu()); As.append(A[0].half().cpu() if neurons else None)
        return torch.stack(Rs), (torch.stack(As) if neurons else None)

    def ids(self, text): return self.tok(text)["input_ids"]

    @torch.no_grad()
    def batch_last(self, prompts, skip=(), bs=None):
        """Logits at the last token of each prompt, optionally with whole layers skipped (output := input)."""
        bs = bs or BS_LAST
        if self.moe: bs = min(bs, BS_MOE)                          # 201k vocabulary: logits for every position are large
        hs = []
        for l in skip:
            def f(mod, args, kwargs, out):
                h = args[0] if args else kwargs["hidden_states"]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hs.append(self.layers[l].register_forward_hook(f, with_kwargs=True))
        try:
            outs, deltas = [], []
            for i in range(0, len(prompts), bs):
                enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True)
                ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
                if self.tok.padding_side == "left": pos = torch.full((ids.shape[0],), ids.shape[1] - 1, device=DEV)
                else: pos = mask.sum(1) - 1
                R, _, lg = self.forward_capture(ids, pos, mask=mask)
                b = torch.arange(ids.shape[0], device=DEV)
                outs.append(lg[b, pos].float()); deltas.append(R)
            return torch.cat(outs), torch.cat(deltas)
        finally:
            for h in hs: h.remove()

    @torch.no_grad()
    def cont_logprob(self, prompts, conts, bs=None):
        """Sum log P(continuation | prompt) for each (prompt, continuation) pair."""
        out = []; bs = bs or BS_CONT
        if self.moe: bs = min(bs, BS_MOE)
        for i in range(0, len(prompts), bs):
            P, C = prompts[i:i + bs], conts[i:i + bs]
            seqs, starts = [], []
            for p, c in zip(P, C):
                pi = self.ids(p); fi = self.ids(p + c); seqs.append(fi); starts.append(len(pi))
            T = max(map(len, seqs)); pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(seqs), T), pad); mask = torch.zeros((len(seqs), T), dtype=torch.long)
            for k, s in enumerate(seqs): ids[k, :len(s)] = torch.tensor(s); mask[k, :len(s)] = 1
            raw = self.m(input_ids=ids.to(DEV), attention_mask=mask.to(DEV)).logits
            if self.moe:                                           # the same sums, log-softmax only where it is read
                for k, s in enumerate(seqs):
                    t = list(range(max(starts[k], 1), len(s)))
                    out.append(float(torch.log_softmax(raw[k, [x - 1 for x in t]].float(), -1)[torch.arange(len(t)), [s[x] for x in t]].sum()) if t else 0.0)
                continue
            lg = torch.log_softmax(raw.float(), -1)
            for k, s in enumerate(seqs):
                tot = 0.0
                for t in range(max(starts[k], 1), len(s)): tot += float(lg[k, t - 1, s[t]])
                out.append(tot)
        return np.array(out)

    def final_logits(self, r):
        """Final norm + unembedding of residual vectors r [N, D] (fp32)."""
        x = self.norm(r.to(self.norm.weight.dtype))
        if self.moe: return torch.cat([x.float() @ self.WU[a:a + 32768].float().T for a in range(0, self.WU.shape[0], 32768)], -1) * self.logit_scale
        return (x.float() @ self.WU.float().T) * self.logit_scale


# ───────────────────────────── M0: words, from the weights alone ─────────────────────────────
@torch.no_grad()
def m0_words(M):
    E = M.WE.float(); E = E - E.mean(0)
    def vec(w):
        t = M.tok(" " + w, add_special_tokens=False)["input_ids"]
        t = [x for x in t if M.tok.decode([x]).strip()] or t
        return E[t].mean(0), len(t)
    red, _ = vec("red")
    def cos(w):
        v, n = vec(w); return float(torch.nn.functional.cosine_similarity(red, v, 0)), n
    common = np.array([cos(w)[0] for w in D.COMMON])
    pct = lambda c: float((common < c).mean() * 100)
    rel = {w: (round(pct(cos(w)[0]), 1), cos(w)[1]) for w in D.RELATED}
    forn = {w: (round(pct(cos(w)[0]), 1), cos(w)[1]) for w in D.FOREIGN}
    score = float(np.mean([v[0] for v in rel.values()]))
    rng = draw(0); cpct = np.array([pct(c) for c in common])
    null = np.array([cpct[rng.choice(len(cpct), len(rel), replace=False)].mean() for _ in range(1000)])
    fscore = float(np.mean([v[0] for v in forn.values()]))
    nullf = np.array([cpct[rng.choice(len(cpct), len(forn), replace=False)].mean() for _ in range(1000)])
    return {"related_pct": round(score, 1), "p": float((null >= score).mean()), "foreign_pct": round(fscore, 1), "p_foreign": float((nullf >= fscore).mean()),
            "related": rel, "foreign": forn, "n_resamples": 1000,
            "note": "percentile of cos(' red', word) among cos(' red', 250 common words), input embeddings centred; (pct, n_tokens)"}


# ───────────────────────────── M1 units / M2 features / M3 geometry ─────────────────────────────
def auc_cols(X, pos):
    """AUC of every column of X [n, P] for boolean labels pos [n] (torch, GPU)."""
    r = torch.argsort(torch.argsort(X, 0), 0).float() + 1
    npos = pos.sum(); nneg = len(pos) - npos
    return (r[pos].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


@torch.no_grad()
def m1_units(A_en, y_en, A_xl, splits=100, seed=0):
    rng = draw(seed); n_s = len(SENSE_NAMES)
    X = A_en.reshape(A_en.shape[0], -1).to(DEV).float(); y = torch.tensor(y_en, device=DEV)
    held = []
    for s in range(splits):
        tr, te = [], []
        for k in range(n_s):
            idx = np.where(np.array(y_en) == k)[0]; rng.shuffle(idx); h = len(idx) // 2; tr += list(idx[:h]); te += list(idx[h:])
        tr, te = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
        aucs = []
        for k in range(n_s):
            a = auc_cols(X[tr], y[tr] == k); j = int(torch.argmax((a - 0.5).abs())); sgn = 1.0 if a[j] >= 0.5 else -1.0
            aucs.append(float(auc_cols(sgn * X[te][:, j:j + 1], y[te] == k)[0]))
        held.append(np.mean(aucs))
    # cross-lingual: choose each sense's neuron on all English, test in each language
    chosen = []
    for k in range(n_s):
        a = auc_cols(X, y == k); j = int(torch.argmax((a - 0.5).abs())); chosen.append((j, 1.0 if a[j] >= 0.5 else -1.0))
    xl = {}
    for lang, (Al, yl) in A_xl.items():
        Xl = Al.reshape(Al.shape[0], -1).to(DEV).float(); yl_t = torch.tensor(yl, device=DEV)
        per = [float(auc_cols(sg * Xl[:, j:j + 1], yl_t == k)[0]) for k, (j, sg) in enumerate(chosen)]
        boot = []; brng = draw(1)
        for _ in range(1000):
            ix = torch.tensor(brng.integers(0, len(yl), len(yl)), device=DEV)
            vals = []
            for k, (j, sg) in enumerate(chosen):
                p = yl_t[ix] == k
                if 0 < int(p.sum()) < len(ix): vals.append(float(auc_cols(sg * Xl[ix][:, j:j + 1], p)[0]))
            boot.append(np.mean(vals))
        xl[lang] = {"auc": round(float(np.mean(per)), 3), "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                    "per_sense": [round(float(v), 3) for v in per]}
    I = A_en.shape[2]
    return {"en_heldout_auc": round(float(np.mean(held)), 3), "en_sd": round(float(np.std(held)), 3), "n_splits": splits,
            "xl": xl, "xl_mean": round(float(np.mean([v["auc"] for v in xl.values()])), 3),
            "neurons": {SENSE_NAMES[k]: [int(j // I), int(j % I), sg] for k, (j, sg) in enumerate(chosen)}}


def avg_rank(X):
    """1-based ranks down each column of X [n, C], tied values sharing their mean rank."""
    Xt = X.T.contiguous(); srt = torch.sort(Xt, 1).values
    lt = torch.searchsorted(srt, Xt); le = torch.searchsorted(srt, Xt, right=True)
    return ((lt + le + 1).float() / 2).T


def auc_ties(r, pos):
    npos = pos.sum(); nneg = len(pos) - npos
    return (r[pos].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


@torch.no_grad()
def m1_units_ties(A_en, y_en, A_xl, splits=100, seed=0, chunk=65536):
    """M1 for mixture-of-experts models: the same test as m1_units (the same splits, choice and bootstrap), but
    tied values share their mean rank. An expert not routed for a sentence gives its neurons an exact 0; ordinal
    ranks would break those ties by sentence order, which follows the sense labels. Columns are read in chunks."""
    rng = draw(seed); n_s = len(SENSE_NAMES); yv = np.array(y_en)
    split_ix = []
    for s in range(splits):
        tr, te = [], []
        for k in range(n_s):
            idx = np.where(yv == k)[0]; rng.shuffle(idx); h = len(idx) // 2; tr += list(idx[:h]); te += list(idx[h:])
        split_ix.append((torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)))
    X = A_en.reshape(A_en.shape[0], -1); y = torch.tensor(y_en, device=DEV)
    best = torch.full((splits, n_s), -1.0, device=DEV); bj = torch.zeros((splits, n_s), dtype=torch.long, device=DEV); ba = torch.zeros((splits, n_s), device=DEV)
    fbest = torch.full((n_s,), -1.0, device=DEV); fj = torch.zeros(n_s, dtype=torch.long, device=DEV); fa = torch.zeros(n_s, device=DEV)
    def upd(r, labels, B, J, Aa, row, c0):
        for k in range(n_s):
            a = auc_ties(r, labels == k); d = (a - 0.5).abs(); j = int(torch.argmax(d))
            if float(d[j]) > float(B[row][k] if row is not None else B[k]):
                if row is None: B[k], J[k], Aa[k] = d[j], c0 + j, a[j]
                else: B[row, k], J[row, k], Aa[row, k] = d[j], c0 + j, a[j]
    for c0 in range(0, X.shape[1], chunk):
        Xc = X[:, c0:c0 + chunk].to(DEV).float()
        upd(avg_rank(Xc), y, fbest, fj, fa, None, c0)
        for s, (tr, te) in enumerate(split_ix): upd(avg_rank(Xc[tr]), y[tr], best, bj, ba, s, c0)
    col = lambda M, j: M[:, int(j)].to(DEV).float()[:, None]
    held = []
    for s, (tr, te) in enumerate(split_ix):
        aucs = []
        for k in range(n_s):
            sgn = 1.0 if float(ba[s, k]) >= 0.5 else -1.0
            aucs.append(float(auc_ties(avg_rank(sgn * col(X, bj[s, k])[te]), y[te] == k)[0]))
        held.append(np.mean(aucs))
    chosen = [(int(fj[k]), 1.0 if float(fa[k]) >= 0.5 else -1.0) for k in range(n_s)]
    xl = {}
    for lang, (Al, yl) in A_xl.items():
        Xl = Al.reshape(Al.shape[0], -1); yl_t = torch.tensor(yl, device=DEV)
        S = torch.cat([sg * col(Xl, j) for j, sg in chosen], 1)                  # [n, senses]
        per = [float(auc_ties(avg_rank(S[:, k:k + 1]), yl_t == k)[0]) for k in range(n_s)]
        boot = []; brng = draw(1)
        for _ in range(1000):
            ix = torch.tensor(brng.integers(0, len(yl), len(yl)), device=DEV); Sb = S[ix]; rb = avg_rank(Sb)
            vals = []
            for k in range(n_s):
                p = yl_t[ix] == k
                if 0 < int(p.sum()) < len(ix): vals.append(float(auc_ties(rb[:, k:k + 1], p)[0]))
            boot.append(np.mean(vals))
        xl[lang] = {"auc": round(float(np.mean(per)), 3), "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                    "per_sense": [round(float(v), 3) for v in per]}
    I = A_en.shape[2]
    return {"en_heldout_auc": round(float(np.mean(held)), 3), "en_sd": round(float(np.std(held)), 3), "n_splits": splits,
            "xl": xl, "xl_mean": round(float(np.mean([v["auc"] for v in xl.values()])), 3), "ties": "mean rank",
            "neurons": {SENSE_NAMES[k]: [int(j // I), int(j % I), sg] for k, (j, sg) in enumerate(chosen)}}


def zfit(X): mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True) + 1e-4; return mu, sd


@torch.no_grad()
def centroid_acc(Xtr, ytr, Xte, yte, center_te=False):
    mu, sd = zfit(Xtr); Ztr = (Xtr - mu) / sd
    Zte = ((Xte - Xte.mean(0, keepdim=True)) / sd) if center_te else (Xte - mu) / sd
    C = torch.stack([Ztr[ytr == k].mean(0) for k in range(len(SENSE_NAMES))])
    C = torch.nn.functional.normalize(C, dim=-1); Zte = torch.nn.functional.normalize(Zte, dim=-1)
    return (Zte @ C.T).argmax(1)


@torch.no_grad()
def m2_features(R_en, y_en, R_xl, splits=100, seed=0):
    rng = draw(seed); Lp1 = R_en.shape[1]; y = torch.tensor(y_en, device=DEV); n_s = len(SENSE_NAMES)
    sp = []
    for s in range(splits):
        tr, te = [], []
        for k in range(n_s):
            idx = np.where(np.array(y_en) == k)[0]; rng.shuffle(idx); h = len(idx) // 2; tr += list(idx[:h]); te += list(idx[h:])
        sp.append((torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)))
    curve = np.zeros((Lp1, splits))
    for l in range(Lp1):
        X = R_en[:, l].to(DEV).float()
        for s, (tr, te) in enumerate(sp):
            curve[l, s] = float((centroid_acc(X[tr], y[tr], X[te], y[te]) == y[te]).float().mean())
    best = int(np.argmax(curve[:, :splits // 2].mean(1)))                    # layer chosen on the first half of the splits
    en_acc = float(curve[best, splits // 2:].mean())                          # reported on the other half
    # per English sentence: share of the reporting splits in which it was held out and classified right
    hit = np.zeros(len(y_en)); seen = np.zeros(len(y_en)); Xb = R_en[:, best].to(DEV).float()
    for tr, te in sp[splits // 2:]:
        ok = (centroid_acc(Xb[tr], y[tr], Xb[te], y[te]) == y[te]).float().cpu().numpy(); ix = te.cpu().numpy()
        hit[ix] += ok; seen[ix] += 1
    en_item = hit / np.maximum(seen, 1)
    off = {k: sum(len(D.SENSES[x]) for x in SENSE_NAMES[:i]) for i, k in enumerate(SENSE_NAMES)}
    after = np.zeros(len(y_en), dtype=bool)
    for k, i in getattr(D, "CUE_AFTER", []):
        if k in off and len(y_en) == sum(len(D.SENSES[x]) for x in SENSE_NAMES): after[off[k] + i] = True
    items = {"en": en_item}
    xl, Xen = {}, R_en[:, best].to(DEV).float()
    Xen_c = Xen - Xen.mean(0, keepdim=True)
    prng = draw(2)
    for lang, (Rl, yl) in R_xl.items():
        Xl = Rl[:, best].to(DEV).float(); yl_t = torch.tensor(yl, device=DEV)
        pred = centroid_acc(Xen_c, y, Xl, yl_t, center_te=True); acc = float((pred == yl_t).float().mean())
        items[lang] = (pred == yl_t).float().cpu().numpy()
        null = [float((pred == yl_t[torch.tensor(prng.permutation(len(yl)), device=DEV)]).float().mean()) for _ in range(1000)]
        # layer curve for transfer, to see where the language-free sense lives
        xcurve = []
        for l in range(Lp1):
            Xe = R_en[:, l].to(DEV).float(); Xe = Xe - Xe.mean(0, keepdim=True)
            xcurve.append(round(float((centroid_acc(Xe, y, Rl[:, l].to(DEV).float(), yl_t, center_te=True) == yl_t).float().mean()), 3))
        xl[lang] = {"acc": round(acc, 3), "p": float((np.array(null) >= acc).mean()), "curve": xcurve, "best_layer": int(np.argmax(xcurve)), "best": max(xcurve)}
    out = {"best_layer": best, "en_acc": round(en_acc, 3), "chance": round(1 / n_s, 3), "n_splits": splits,
           "en_curve": [round(float(v), 3) for v in curve.mean(1)], "xl": xl,
           "xl_mean": round(float(np.mean([v["acc"] for v in xl.values()])), 3) if xl else None,
           "xl_best_mean": round(float(np.mean([v["best"] for v in xl.values()])), 3) if xl else None, "_items": items}
    if after.any():                                                         # a causal model cannot see a cue that comes after the word
        out |= {"en_acc_cue_before": round(float(en_item[~after].mean()), 3), "en_acc_cue_after": round(float(en_item[after].mean()), 3), "n_cue_after": int(after.sum())}
    return out


def band(L): return (L // 2, (3 * L) // 4)


def lab(hexc):
    r, g, b = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in (r, g, b)]
    X = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047; Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]; Z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return np.array((116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))))


def rdm(X):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-3); Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-6)
    return 1 - Z @ Z.T


def mantel(A, B, perms=1000, seed=0):
    n = A.shape[0]; iu = np.triu_indices(n, 1); rho = spearman(A[iu], B[iu]); rng = draw(seed); null = []
    rb = rank(B[iu])
    for _ in range(perms):
        p = rng.permutation(n); null.append(float(np.corrcoef(rank(A[np.ix_(p, p)][iu]), rb)[0, 1]))
    return rho, float((np.array(null) >= rho).mean())


def m3_geometry(M, colour_R, R_en4, R_xl):
    lo, hi = band(M.L)
    X = colour_R[:, :, lo:hi + 1].float().mean(2).mean(1).numpy()           # [colour, frame, L+1, D] -> [colour, D]
    names = list(D.COLORS); real = np.array([[np.linalg.norm(lab(D.COLORS[a]) - lab(D.COLORS[b])) for b in names] for a in names])
    rho_c, p_c = mantel(rdm(X), real)
    near = np.argsort(rdm(X)[0])[1:6]
    grid = {}
    for nm, (a, b) in {"early": (1, M.L // 4), "middle": (M.L // 4, M.L // 2), "middle-late": (M.L // 2, (3 * M.L) // 4), "late": ((3 * M.L) // 4, M.L)}.items():
        grid[nm] = round(mantel(rdm(colour_R[:, :, a:b + 1].float().mean(2).mean(1).numpy()), real, perms=200)[0], 3)
    out = {"band": [lo, hi], "colour_rho": round(rho_c, 3), "colour_p": p_c, "red_inner_nearest": [names[i] for i in near], "colour_rho_by_band": grid}
    def centroids(R, y):
        Xb = R[:, lo:hi + 1].float().mean(1).numpy(); Xb = Xb - Xb.mean(0)
        return np.stack([Xb[np.array(y) == k].mean(0) for k in range(len(SENSE_NAMES))])
    Ce = centroids(*R_en4); xl = {}
    for lang, (Rl, yl) in R_xl.items():
        rho, p = mantel(rdm(Ce), rdm(centroids(Rl, yl))); xl[lang] = {"rho": round(rho, 3), "p": p}
    out["xl_sense_geometry"] = xl; out["n_perms"] = 1000
    return out, rdm(Ce)


# ───────────────────────────── M4 wiring ─────────────────────────────
@torch.no_grad()
def m4_wiring(M, colour_A, K=48, nulls=1000):
    A = colour_A.float().mean(1)                                              # [colour, L, I]
    names = list(D.COLORS); out = {}
    n_in = len(M.ins[0])
    def sel(c):
        ci = names.index(c); oth = [i for i in range(len(names)) if i != ci]
        z = (A[ci] - A[oth].mean(0)) / (A[oth].std(0) + 1e-3)                 # [L, I]
        top = torch.topk(z.flatten(), K).indices
        return [(int(t // M.I), int(t % M.I)) for t in top]
    def vecs(S):
        go = lambda l: M.gout[l].float() if M.gout[l] is not None else 1.0
        gi = lambda l: M.gin[l].float() if M.gin[l] is not None else 1.0
        O = torch.stack([M.col(M.downs[l], n) * go(l) for l, n in S]).to(DEV)
        Ins = [torch.stack([M.row(M.ins[l][m][0], n + M.ins[l][m][1]) * gi(l) for l, n in S]).to(DEV) for m in range(n_in)]
        lay = torch.tensor([l for l, _ in S], device=DEV)
        return torch.nn.functional.normalize(O, dim=-1), [torch.nn.functional.normalize(x, dim=-1) for x in Ins], lay
    def wiring(Sa, Sb=None):
        Oa, _, la = vecs(Sa); _, Ib, lb = vecs(Sb if Sb is not None else Sa)
        C = sum(Oa @ x.T for x in Ib) / n_in                                  # C[i, j] = output of i aimed at input of j
        mask = la[:, None] < lb[None, :]
        return float(C[mask].mean()) if mask.any() else float("nan"), int(mask.sum())
    rng = draw(0); sets = {}
    for c in ("red", "blue", "green"):
        S = sel(c); sets[c] = S; w, npairs = wiring(S)
        null = np.array([wiring([(l, int(rng.integers(0, M.I))) for l, _ in S])[0] for _ in range(nulls)])   # same layers, random neurons
        z = (w - null.mean()) / (null.std() + 1e-9)
        out[c] = {"wiring": round(w, 5), "null_mean": round(float(null.mean()), 5), "null_sd": round(float(null.std()), 5), "z": round(float(z), 2),
                  "p": float((null >= w).mean()), "pairs": npairs, "layers": sorted(set(l for l, _ in S))}
    if METHODS >= 2:
        # selection-matched null: every other colour's neurons, chosen the same way (top K by its own z), wired among
        # themselves. Co-activated sets may be wired by construction; red has to beat sets selected like it.
        vals = {c: round(wiring(sel(c))[0], 5) for c in names if c != "red"}
        v = np.array([x for x in vals.values() if np.isfinite(x)]); w = out["red"]["wiring"]
        out["matched_null"] = {"kind": "other colours' top-K selections", "n": int(len(v)), "mean": round(float(v.mean()), 5), "sd": round(float(v.std()), 5),
                               "z": round(float((w - v.mean()) / (v.std() + 1e-9)), 2), "p": float((1 + (v >= w).sum()) / (1 + len(v))), "values": vals}
    out["red_to_blue"] = round(wiring(sets["red"], sets["blue"])[0], 5)
    out["red_to_green"] = round(wiring(sets["red"], sets["green"])[0], 5)
    out["n_null"] = nulls; out["K"] = K
    return out


# ───────────────────────────── M4b interactions + M6 redundancy (layer removal on facts) ─────────────────────────────
@torch.no_grad()
def m6_redundancy(M):
    prompts = D.FACTS_RED + [p for p, _ in D.FACTS_OTHER]
    gold = ["red"] * len(D.FACTS_RED) + [c for _, c in D.FACTS_OTHER]
    base = "The car is"
    cid = []
    for c in D.COLOURS8:
        a = M.ids(base); b = M.ids(base + " " + c); cid.append(b[len(a)])
    cid_t = torch.tensor(cid, device=DEV); gi = torch.tensor([D.COLOURS8.index(g) for g in gold], device=DEV)
    nr = len(D.FACTS_RED)
    def metric(lg): return torch.log_softmax(lg[:, cid_t], -1)[torch.arange(len(gi), device=DEV), gi]
    lg, R = M.batch_last(prompts)
    m_clean = metric(lg); acc = float((lg[:, cid_t].argmax(1) == gi).float()[:nr].mean())
    pred = lg[:, cid_t].argmax(1); gold_np = gi.cpu().numpy(); ok_all = (pred == gi).float().cpu().numpy()
    acc_bal = float(np.mean([ok_all[gold_np == g].mean() for g in np.unique(gold_np)]))   # mean recall per answer colour: 'always red' scores 1/n_colours
    # direct effect of each layer: remove its delta from the final residual, norm scale frozen
    rL = R[:, -1].float(); x = rL.to(M.norm.weight.dtype)
    if M.is_ln: xc = rL - rL.mean(-1, keepdim=True); scale = xc.pow(2).mean(-1, keepdim=True).add(M.norm.eps).rsqrt()
    else:
        eps = getattr(M.norm, "variance_epsilon", getattr(M.norm, "eps", 1e-6)); scale = rL.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    g = M.norm.weight.float(); bvec = M.norm.bias.float() if getattr(M.norm, "bias", None) is not None else 0
    WUc = M.WU[cid_t].float()
    def frozen_logits(r):
        if M.is_ln: r = r - r.mean(-1, keepdim=True)
        return ((r * scale * g + bvec) @ WUc.T) * M.logit_scale
    def m_of(cl):
        return torch.log_softmax(cl, -1)[torch.arange(len(gi), device=DEV), gi]
    m_frozen = m_of(frozen_logits(rL))
    direct, total = [], []
    for l in range(M.L):
        d = (R[:, l + 1] - R[:, l]).float()
        direct.append((m_frozen - m_of(frozen_logits(rL - d)))[:nr].cpu().numpy())
        lg2, _ = M.batch_last(prompts, skip=(l,)); total.append((m_clean - metric(lg2))[:nr].cpu().numpy())
    direct, total = np.array(direct), np.array(total)                        # [L, n_red]
    mid = np.arange(2, M.L - 1)                                               # first two and last layer excluded: their removal breaks everything
    top = list(mid[np.argsort(-direct[mid].mean(1))[:5]])
    # self-repair = share of the top layers' direct effect that does not show up as total effect (ratio of sums, bootstrap over facts)
    dsum, tsum = direct[top].sum(0), total[top].sum(0)                      # per fact
    sr = lambda ix: float((dsum[ix].sum() - tsum[ix].sum()) / (dsum[ix].sum() + 1e-9))
    rng = draw(0); allix = np.arange(nr); fr_mean = sr(allix)
    boot = [sr(rng.integers(0, nr, nr)) for _ in range(1000)]
    # the same, with the layers chosen on one half of the facts and measured on the other (no double dipping)
    rs = draw(6); pm = rs.permutation(nr); hA, hB = pm[:nr // 2], pm[nr // 2:]
    topA = list(mid[np.argsort(-direct[mid][:, hA].mean(1))[:5]]); dB, tB = direct[topA][:, hB].sum(0), total[topA][:, hB].sum(0)
    srB = lambda ix: float((dB[ix].sum() - tB[ix].sum()) / (dB[ix].sum() + 1e-9))
    bootB = [srB(rs.integers(0, len(hB), len(hB))) for _ in range(1000)]
    split = {"value": round(srB(np.arange(len(hB))), 3), "ci": [round(float(np.percentile(bootB, 2.5)), 3), round(float(np.percentile(bootB, 97.5)), 3)],
             "layers": [int(t) for t in topA], "chosen_on": [int(i) for i in hA], "n_measured": int(len(hB)), "n_boot": 1000}
    # interactions between the layers whose removal hurts most
    tt = list(mid[np.argsort(-total[mid].mean(1))[:6]]); inter = []
    robust = []
    for l in tt[:3]:
        lg2, _ = M.batch_last(prompts, skip=(int(l),)); robust.append(float((lg2[:, cid_t].argmax(1) == gi).float()[:nr].mean()))
    for a, b in itertools.combinations(tt, 2):
        lg2, _ = M.batch_last(prompts, skip=(int(a), int(b))); tab = (m_clean - metric(lg2))[:nr].cpu().numpy()
        I = tab - total[a] - total[b]; inter.append({"pair": [int(a), int(b)], "joint": round(float(tab.mean()), 3), "sum": round(float((total[a] + total[b]).mean()), 3), "I": round(float(I.mean()), 3)})
    return {"fact_acc": round(acc, 3), "n_facts": nr, "m_clean": round(float(m_clean[:nr].mean()), 3),
            "direct": [round(float(v), 3) for v in direct.mean(1)], "total": [round(float(v), 3) for v in total.mean(1)],
            "top_direct_layers": [int(t) for t in top], "self_repair": round(fr_mean, 3),
            "self_repair_ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)], "n_boot": 1000,
            "interactions": inter, "robust_acc_after_top3": [round(v, 3) for v in robust], "robust_ratio": round(float(np.mean(robust) / max(acc, 1e-9)), 3),
            "backup_pairs": sum(1 for i in inter if i["I"] > 0.1), "serial_pairs": sum(1 for i in inter if i["I"] < -0.1),
            "compensation": round(fr_mean, 3), "compensation_ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "compensation_split": split, "fact_acc_balanced": round(acc_bal, 3), "fact_acc_other": round(float(ok_all[nr:].mean()), 3) if len(ok_all) > nr else None,
            "note_compensation": "compensation = the old self_repair: share of the top layers' direct effect that the total effect does not show",
            "_items": np.stack([direct, total]).astype(np.float32)}


# ───────────────────────────── M5 composition ─────────────────────────────
@torch.no_grad()
def m5_composition(M):
    """For each context k with colours a->ma, b->mb: DiD = [lp(ma|a) - lp(mb|a)] - [lp(ma|b) - lp(mb|b)].
    Positive = the colour moves the meaning the right way in that context. Neither part alone can score."""
    rows = []
    for k, (tpls, cm) in D.CROSSED.items():
        (ca, ma), (cb, mb) = list(cm.items())
        for t in tpls: rows.append((k, t, ca, ma, cb, mb))
    P, C = [], []
    for k, t, ca, ma, cb, mb in rows:
        for c in (ca, cb):
            for m in (ma, mb): P.append(t.format(c=c)); C.append(m)
    lp = M.cont_logprob(P, C).reshape(len(rows), 2, 2)                         # [row, colour a/b, meaning a/b]
    did = (lp[:, 0, 0] - lp[:, 0, 1]) - (lp[:, 1, 0] - lp[:, 1, 1])
    # logit lens: the same score from each layer's residual, first token of each meaning
    prompts = [r[1].format(c=c) for r in rows for c in (r[2], r[4])]
    _, R = M.batch_last(prompts)
    first = lambda p, m: M.ids(p + m)[len(M.ids(p))]
    fid = torch.tensor([[first(r[1].format(c=r[2]), r[3]), first(r[1].format(c=r[2]), r[5])] for r in rows], device=DEV)
    curve = []
    for l in range(1, M.L + 1):
        lpl = torch.log_softmax(M.final_logits(R[:, l]).float(), -1).reshape(len(rows), 2, -1)
        g = torch.stack([lpl[:, 0].gather(1, fid), lpl[:, 1].gather(1, fid)], 1)   # [row, colour, meaning]
        dl = (g[:, 0, 0] - g[:, 0, 1]) - (g[:, 1, 0] - g[:, 1, 1])
        curve.append(round(float((dl > 0).float().mean()), 3))
    frac = float((did > 0).mean()); rng = draw(0)
    null = [float(((did * rng.choice([-1, 1], len(did))) > 0).mean()) for _ in range(1000)]
    keys = list(D.CROSSED); kk = np.array([r[0] for r in rows]); boot = []
    for _ in range(1000):
        pick = rng.integers(0, len(keys), len(keys)); idx = np.concatenate([np.where(kk == keys[i])[0] for i in pick]); boot.append(float((did[idx] > 0).mean()))
    per = {k: [round(float(v), 2) for v in did[kk == k]] for k in keys}
    top = max(curve); emerge = next((i + 1 for i, v in enumerate(curve) if v >= 0.5 + 0.5 * (top - 0.5)), None) if top > 0.5 else None
    out = {"frac_correct": round(frac, 3), "chance": 0.5, "mean_did": round(float(did.mean()), 3), "p": float((np.array(null) >= frac).mean()),
           "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)], "n_tests": len(rows), "n_resamples": 1000,
           "per_context_did": per, "lens_frac": curve, "emerge_layer": emerge, "_items": did.astype(np.float32)}
    out["association"] = {k: out[k] for k in ("frac_correct", "mean_did", "p", "ci")} | {
        "note": "each context's DiD on its own: a colour -> meaning association that ignores the context can score here"}
    if METHODS >= 2 and getattr(D, "CROSSED_MIRRORED", None):
        mrows = []
        for k, (tpls, cm) in D.CROSSED_MIRRORED.items():
            (ca, ma), (cb, mb) = list(cm.items())
            for t in tpls: mrows.append((k, t, ca, ma, cb, mb))
        P2, C2 = [], []
        for k, t, ca, ma, cb, mb in mrows:
            for c in (ca, cb):
                for m in (ma, mb): P2.append(t.format(c=c)); C2.append(m)
        lp2 = M.cont_logprob(P2, C2).reshape(len(mrows), 2, 2); did2 = (lp2[:, 0, 0] - lp2[:, 0, 1]) - (lp2[:, 1, 0] - lp2[:, 1, 1])
        ctx = {k: float(did[kk == k].mean()) for k in keys} | {k: float(did2[[i for i, r in enumerate(mrows) if r[0] == k]].mean()) for k in D.CROSSED_MIRRORED}
        pairs = [(a, b) for a, b in D.MIRROR_PAIRS if a in ctx and b in ctx]
        # each context's DiD is scored with its own mapping, so in a mirrored pair a context-blind association gives
        # +s in one half and -s in the other: (DiD_a + DiD_b) / 2 is the interaction, (DiD_a - DiD_b) / 2 the association
        iv = np.array([(ctx[a] + ctx[b]) / 2 for a, b in pairs]); av = np.array([(ctx[a] - ctx[b]) / 2 for a, b in pairs])
        signs = np.array(list(itertools.product([1, -1], repeat=len(iv)))); null = (signs * iv).mean(1)
        out["interaction"] = {"mean": round(float(iv.mean()), 3), "per_pair": {f"{a} | {b}": round(float(v), 3) for (a, b), v in zip(pairs, iv)},
                              "association_per_pair": {f"{a} | {b}": round(float(v), 3) for (a, b), v in zip(pairs, av)},
                              "p": float((null >= iv.mean() - 1e-12).mean()), "n_clusters": len(iv), "min_p": round(1 / len(signs), 4),
                              "null": "exact sign flips over the mirrored pairs (clusters)", "unit": "nats"}
        out["mirrored_per_context_did"] = {k: [round(float(v), 2) for v in did2[[i for i, r in enumerate(mrows) if r[0] == k]]] for k in D.CROSSED_MIRRORED}
        out["_items"] = np.concatenate([did, did2]).astype(np.float32)
    return out


# ───────────────────────────── M8 self-access ─────────────────────────────
@torch.no_grad()
def m8_self_access(M):
    head = ("How strongly is each word associated with the colour red? 0 = not at all, 9 = extremely.\n"
            "red - cardinal: 8\nred - pillow: 2\nred - crimson: 9\nred - spoon: 1\n")
    P = [head + f"red - {w}:" for w in D.ASSOC for d in range(10)]; C = [f" {d}" for _ in D.ASSOC for d in range(10)]
    lp = M.cont_logprob(P, C).reshape(len(D.ASSOC), 10); pr = np.exp(lp - lp.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    said = (pr * np.arange(10)).sum(1)
    frames = ["I read about {w} today.", "The word {w} came up in the meeting."]
    lo, hi = band(M.L); pool = ["red"] + D.ASSOC + D.COMMON
    reps = []
    for w in pool:
        acc = []
        for f in frames:
            s = f.replace("{w}", "[" + w + "]"); R, _ = M.read_marked([s], neurons=False); acc.append(R[0, lo:hi + 1].float().mean(0))
        reps.append(torch.stack(acc).mean(0))
    X = torch.stack(reps).numpy(); Z = (X - X.mean(0)) / (X.std(0) + 1e-3); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    inside = Z[1:1 + len(D.ASSOC)] @ Z[0]
    # held association: what the residual at 'red' points to (final norm + unembedding, band layers), minus the same for other colours
    wid = [M.ids("The " + w)[len(M.ids("The")):][:1][0] for w in D.ASSOC]
    lens = {}
    for c in ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]:
        acc = []
        for f in ["The colour {w} came up in the meeting.", "I thought about the colour {w} today."]:
            s2 = f.replace("{w}", "[" + c + "]"); R, _ = M.read_marked([s2], neurons=False)
            lp = torch.log_softmax(M.final_logits(R[0, lo:hi + 1].float().to(DEV)), -1)[:, wid].mean(0); acc.append(lp)
        lens[c] = torch.stack(acc).mean(0).cpu().numpy()
    held = lens["red"] - np.mean([lens[c] for c in lens if c != "red"], 0)
    rho = spearman(said, held); rng = draw(0)
    null = [spearman(said, rng.permutation(held)) for _ in range(1000)]
    rho_geom = spearman(said, inside)
    strong = np.arange(len(D.ASSOC)) < 30
    def auc(v): return float(np.mean([a > b for a in v[strong] for b in v[~strong]]))
    out = {"rho_said_held": round(rho, 3), "p": float((np.array(null) >= rho).mean()), "n_perms": 1000, "rho_said_geometry": round(rho_geom, 3),
           "said_auc_strong_vs_weak": round(auc(said), 3), "held_auc_strong_vs_weak": round(auc(held), 3), "geometry_auc_strong_vs_weak": round(auc(inside), 3),
           "held": {w: round(float(v), 3) for w, v in zip(D.ASSOC, held)},
           "said": {w: round(float(s), 2) for w, s in zip(D.ASSOC, said)}, "inside": {w: round(float(s), 3) for w, s in zip(D.ASSOC, inside)}}
    items = [said, held, inside]
    if METHODS >= 2 and getattr(D, "M8_HEADS", None):
        # crossed: the same 40 words rated for blue and green too; said(red, w) - said(other, w) removes what a word
        # scores for any colour (a word that is simply 'strong', or a model that rates everything high)
        S = {"red": said}
        for c in ("blue", "green"):
            Pc = [D.M8_HEADS[c] + f"{c} - {w}:" for w in D.ASSOC for d in range(10)]
            lpc = M.cont_logprob(Pc, C).reshape(len(D.ASSOC), 10); pc = np.exp(lpc - lpc.max(1, keepdims=True)); pc /= pc.sum(1, keepdims=True)
            S[c] = (pc * np.arange(10)).sum(1)
        St = np.stack([S["red"], S["blue"], S["green"]]); diff = St[0] - St[1:].mean(0)
        def aucv(v): return float((v[strong][:, None] > v[~strong][None, :]).mean())
        obs = aucv(diff); prng = draw(8); nl = []
        for _ in range(1000):                                            # each word's three colour labels shuffled
            o = np.argsort(prng.random((3, St.shape[1])), 0); X = np.take_along_axis(St, o, 0); nl.append(aucv(X[0] - X[1:].mean(0)))
        rc = spearman(diff, held); nr_ = [spearman(diff, prng.permutation(held)) for _ in range(1000)]
        out["crossed"] = {"auc_strong_vs_weak": round(obs, 3), "p": float((np.array(nl) >= obs).mean()), "n_perms": 1000,
                          "null_sd": round(float(np.std(nl)), 3), "rho_with_held": round(rc, 3), "p_rho": float((np.array(nr_) >= rc).mean()),
                          "said_blue": {w: round(float(s), 2) for w, s in zip(D.ASSOC, S["blue"])}, "said_green": {w: round(float(s), 2) for w, s in zip(D.ASSOC, S["green"])},
                          "note": "said(red, w) - mean(said(blue, w), said(green, w)); null: each word's colour labels permuted"}
        items += [S["blue"], S["green"]]
    out["_items"] = np.stack(items).astype(np.float32)
    return out


TWIN_KINDS = {"normal": "drawn from init std", "shuffle": "shuffled"}


def token_info(M, sents):
    """Per sentence: index of the token the ladder reads (the last one overlapping the marked word), how many tokens
    the marked word is cut into, how many tokens come before it, and the sentence length."""
    pos, sub, pre, n = [], [], [], []
    for s in sents:
        text, (a, b) = unmark(s); enc = M.tok(text, return_offsets_mapping=True)
        k = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if y > x and x <= b and y > a]
        pos.append(k[-1] if k else -1); sub.append(len(k)); pre.append(k[0] if k else -1); n.append(len(enc["input_ids"]))
    return {"pos": np.array(pos), "subtokens": np.array(sub), "prefix": np.array(pre), "length": np.array(n)}


def provenance(M, path, init):
    """The conditions of the run: versions, code, precision, tokenizer, batching, machine, twin."""
    import hashlib, platform, subprocess, transformers
    def git(*a):
        try: return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception: return None
    dirty = git("status", "--porcelain", "--", "model_ladder.py", "model_ladder_data.py")
    try: tk = __import__("tokenizers").__version__
    except Exception: tk = None
    try: probe = M.tok("red")["input_ids"]; bos_added = bool(M.tok.bos_token_id is not None and probe and probe[0] == M.tok.bos_token_id)
    except Exception: bos_added = None
    return {"methods": METHODS, "torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tk, "numpy": np.__version__,
            "python": platform.python_version(), "code_commit": git("rev-parse", "--short", "HEAD"), "code_dirty": bool(dirty) if dirty is not None else None,
            "data_sha256": hashlib.sha256(open(D.__file__, "rb").read()).hexdigest(),
            "precision": "MXFP4 experts + int8 attention" if M.moe else ("int8 (row absmax)" if M.q8 else M.dtype_name),
            "attn_implementation": getattr(M.m.config, "_attn_implementation", None),
            "tokenizer_fallback": getattr(M.tok, "psi_fallback", None), "bos_token": M.tok.bos_token, "bos_added": bos_added,
            "pad_token": M.tok.pad_token, "pad_is_eos": M.tok.pad_token is not None and M.tok.pad_token == M.tok.eos_token, "padding_side": M.tok.padding_side,
            "batch_sizes": {"batch_last": BS_MOE if M.moe else BS_LAST, "cont_logprob": BS_MOE if M.moe else BS_CONT, "read_marked": 1},
            "device": DEV, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" and torch.cuda.is_available() else None,
            "memory_cap_fraction": round(M.mem_fraction, 3) if M.mem_fraction else None, "minutes_waited_for_gpu": M.minutes_waited,
            "twin_kind": TWIN_KINDS.get(M.init_kind), "init_std": M.init_std, "seed": SEED, "path": path}


def run(path, tag, init=False, q8=False, swap=None):
    t0 = time.time(); init = "normal" if init is True else (init or False)
    log(f"== {tag}  ({path}{'  random init' if init == 'normal' else '  shuffled weights' if init == 'shuffle' else ''}{'  methods ' + str(METHODS) if METHODS != 1 else ''})")
    M = Model(path, init=init, seed=SEED, q8=q8); log(f"   layers={M.L} neurons/layer={M.I} dtype={M.m.dtype}")
    res = {"tag": tag, "path": path, "init": (True if init == "normal" else init), "seed": SEED, "layers": M.L, "neurons": M.I}
    res["twin_kind"] = TWIN_KINDS.get(init) or ("real step-0" if tag.endswith("@0") else None)
    if init == "normal": res["init_std"] = M.init_std
    res["methods"] = METHODS
    try: res["provenance"] = provenance(M, path, init)
    except Exception as e: res["provenance"] = {"error": repr(e)}
    if swap:                                                       # M0 intervention: two words exchange input vectors
        a, b = M.control_pair(SEED) if swap == "control" else swap.split(",")
        ctl = swap == "control" or "red" not in (a, b)                  # blue,green: the matched control for red,blue
        res["intervention"] = {"level": "M0 words", "kind": "swap input vectors", "words": [a, b], "control": ctl, "tokens": M.swap_words(a, b)}
        if ctl: res["intervention"]["control_kind"] = "inert" if swap == "control" else "matched"
        if swap == "control": res["intervention"]["note"] = "inert: neither word occurs in any test text, so on the test set this change is the identity"
        log("   intervention", res["intervention"])
    res["M0_words"] = m0_words(M); log("   M0", {k: res["M0_words"][k] for k in ("related_pct", "p", "foreign_pct", "p_foreign")})
    en = [s for k in SENSE_NAMES for s in D.SENSES[k]]; y_en = [i for i, k in enumerate(SENSE_NAMES) for _ in D.SENSES[k]]
    items, toks = {}, {}
    try:
        toks["en"] = token_info(M, en)
        for lang, T in D.TRANS.items(): toks[lang] = token_info(M, [x for k in SENSE_NAMES for x in T[k]])
        res["tokens"] = {l: {"marked_subtokens_mean": round(float(t["subtokens"].mean()), 3), "marked_one_token": round(float((t["subtokens"] == 1).mean()), 3),
                             "prefix_mean": round(float(t["prefix"].mean()), 2), "n": int(len(t["pos"]))} for l, t in toks.items()}
    except Exception as e: res["tokens"] = {"error": repr(e)}
    R_en, A_en = M.read_marked(en)
    R_xl, A_xl = {}, {}
    for lang, T in D.TRANS.items():
        s = [x for k in SENSE_NAMES for x in T[k]]; y = [i for i, k in enumerate(SENSE_NAMES) for _ in T[k]]
        R, A = M.read_marked(s); R_xl[lang] = (R, y); A_xl[lang] = (A, y)
    cs = [[f.replace("{w}", "[" + w + "]") for f in D.COLOR_FRAMES] for w in D.COLORS]
    cR, cA = M.read_marked([s for row in cs for s in row]); nf = len(D.COLOR_FRAMES)
    colour_R = cR.reshape(len(D.COLORS), nf, *cR.shape[1:]); colour_A = cA.reshape(len(D.COLORS), nf, *cA.shape[1:])
    log(f"   read {len(en)} en + {sum(len(v[1]) for v in R_xl.values())} translated + {len(cs) * nf} colour sentences")
    res["M1_units"] = (m1_units_ties if (M.moe or METHODS >= 2) else m1_units)(A_en, y_en, A_xl)
    items["items_m1"] = np.array([v["per_sense"] for v in res["M1_units"]["xl"].values()], dtype=np.float32); log("   M1", {k: res["M1_units"][k] for k in ("en_heldout_auc", "xl_mean")}, {k: v["auc"] for k, v in res["M1_units"]["xl"].items()})
    del A_en, A_xl; torch.cuda.empty_cache()
    res["M2_features"] = m2_features(R_en, y_en, R_xl); it2 = res["M2_features"].pop("_items", {})
    items["items_m2"] = np.concatenate([it2[k] for k in it2]).astype(np.float32); items["items_m2_lang"] = np.concatenate([[i] * len(it2[k]) for i, k in enumerate(it2)]).astype(np.int8)
    log("   M2", {k: res["M2_features"][k] for k in ("best_layer", "en_acc", "xl_mean", "xl_best_mean")}, {k: (v["acc"], v["p"]) for k, v in res["M2_features"]["xl"].items()})
    idx4 = [i for i in range(len(en)) if i % 10 < 4]
    res["M3_geometry"], C_rdm = m3_geometry(M, colour_R, (R_en[idx4], [y_en[i] for i in idx4]), R_xl)
    log("   M3", {k: res["M3_geometry"][k] for k in ("colour_rho", "colour_p", "red_inner_nearest")}, res["M3_geometry"]["xl_sense_geometry"])
    lo, hi = band(M.L); Xs = R_en[:, lo:hi + 1].float().mean(1).numpy()
    rdms = {"sent_rdm": rdm(Xs).astype(np.float32), "sense_rdm": C_rdm.astype(np.float32)}
    np.savez_compressed(os.path.join(OUT, f"{tag.replace('@', '_')}.npz"), **rdms)
    del R_en, R_xl, colour_R; torch.cuda.empty_cache()
    res["M4_wiring"] = m4_wiring(M, colour_A); log("   M4", {c: (res["M4_wiring"][c]["wiring"], res["M4_wiring"][c]["z"]) for c in ("red", "blue", "green")}, "red->blue", res["M4_wiring"]["red_to_blue"])
    res["M5_composition"] = m5_composition(M); r5 = res["M5_composition"]; items["items_m5"] = r5.pop("_items"); log("   M5", "frac", r5["frac_correct"], r5["ci"], "p", r5["p"], "mean DiD", r5["mean_did"], "emerge@", r5["emerge_layer"], {k: round(float(np.mean(v)), 1) for k, v in r5["per_context_did"].items()})
    res["M6_redundancy"] = m6_redundancy(M); r6 = res["M6_redundancy"]; items["items_m6"] = r6.pop("_items"); log("   M6 facts", r6["fact_acc"], "self-repair", r6["self_repair"], r6["self_repair_ci"], "backup/serial", r6["backup_pairs"], r6["serial_pairs"], "robust", r6["robust_ratio"], "top", r6["top_direct_layers"])
    res["M8_self_access"] = m8_self_access(M); r8 = res["M8_self_access"]; items["items_m8"] = r8.pop("_items"); log("   M8", {k: r8[k] for k in ("rho_said_held", "p", "rho_said_geometry", "said_auc_strong_vs_weak", "held_auc_strong_vs_weak", "geometry_auc_strong_vs_weak")})
    res["seconds"] = round(time.time() - t0, 1)
    # per-item scores beside the layouts (M1 [lang, sense] AUC; M2 per sentence right/wrong, en = share of held-out splits;
    # M5 per test DiD; M6 [direct/total, layer, fact]; M8 [said, held, inside(, said blue, said green)] per word) and tokens
    for l, t in toks.items():
        for k, v in t.items(): items[f"tok_{l}_{k}"] = v.astype(np.int32)
    np.savez_compressed(os.path.join(OUT, f"{tag.replace('@', '_')}.npz"), **rdms, **items)
    res["items_in_npz"] = sorted(k for k in items if k.startswith("items_"))
    with open(os.path.join(OUT, f"{tag.replace('@', '_')}.json"), "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"   done in {res['seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tag")
    ap.add_argument("--init", nargs="?", const="normal", choices=["normal", "shuffle"], default=None,
                    help="untrained twin: '--init' (= normal) draws every weight from the init std; '--init shuffle' shuffles the trained weights within each tensor (tag <name>@shuffle)")
    ap.add_argument("--aggregate", action="store_true"); ap.add_argument("--q8", action="store_true")
    ap.add_argument("--methods", type=int, default=1, choices=[1, 2],
                    help="2: tie-aware M1 for every model, M4 selection-matched null, M5 mirrored contexts (interaction), M8 crossed colours. 1 (default): as the stored runs")
    ap.add_argument("--seed", type=int, default=0, help="replication: another draw of every resample and of the untrained init")
    ap.add_argument("--out", help="results folder (replications go to results/model_ladder/replications/seed<N>)")
    ap.add_argument("--swap", help="M0 intervention: 'red,blue' exchanges the two words' input vectors; 'blue,green' is its matched control (two "
                                   "colours the tests use, red untouched); 'control' swaps two common words absent from every test text, which is "
                                   "the identity on the test set (an inert control) (results/model_ladder/interventions)")
    # arguments stripped: arguments read from a file written on Windows can end in CR, which argparse would reject
    a = ap.parse_args([x.strip() for x in sys.argv[1:] if x.strip()]); SEED = a.seed; METHODS = a.methods
    if a.swap and not a.out: a.out = os.path.join(OUT, "interventions")
    if a.out or a.seed:
        OUT = a.out or os.path.join(OUT, "replications", f"seed{a.seed}"); os.makedirs(OUT, exist_ok=True); LOG = os.path.join(OUT, "ladder.log")
    if a.aggregate:
        import model_ladder_report; model_ladder_report.main()
    else:
        run(a.model, a.tag, a.init or False, a.q8, a.swap)
