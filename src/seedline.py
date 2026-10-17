# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""A seed line: one starting model, frozen tokenizer, that a community grows by branches and recipes.

The first line is stories260K (Andrej Karpathy, karpathy/tinyllamas, MIT): a llama2.c model, 64 wide, 5 layers,
8 query / 4 key-value heads, 512-piece tokenizer, output tied to the input, 260,032 numbers. It is read from the
llama2.c .bin export (raw float32, no pickle), pinned by commit and SHA-256.

Every weight set is a plain dict {name: float32 tensor} ("a vec"), names as in llama2.c / its safetensors.
vec_sha(): SHA-256 over (name, little-endian float32 bytes) in sorted name order, the identity recipes and records use.

Training writes a trace so anyone can replay it (proof-of-learning style, Jia et al. 2021): every `every` steps the
weights and the optimiser state are saved, the batch of step s is a pure function of (seed, s, data), and the data is
named by a salted hash. A verifier loads checkpoint k, runs the next steps on the declared data and checks it lands on
checkpoint k+1 (exactly on the same CPU set-up, within a tolerance elsewhere). Evidence, not proof: published attacks
can spoof such traces, so a verifier reports how many segments it replayed, never "proven".
"""
import hashlib, json, os, re, struct, urllib.request
import numpy as np
import torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PSI_DATA_DIR") or os.path.join(os.path.dirname(HERE), ".userdata")
LINES_DIR = os.path.join(DATA, "commons", "lines")

LINES = {
    "stories260k": {
        "title": "stories260K", "by": "Andrej Karpathy (llama2.c)", "licence": "MIT",
        "about": "A 260,032-number Llama that writes small children's stories; small enough to train in a phone's browser.",
        "source": "https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K",
        "files": {
            "stories260K.bin": {"url": "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/stories260K.bin",
                                "sha256": "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696", "size": 1056540},
            "tok512.bin": {"url": "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/tok512.bin",
                           "sha256": "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312", "size": 6227},
        },
    },
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def line_dir(line): return os.path.join(LINES_DIR, line)


def line_status(line="stories260k"):
    L = LINES[line]; d = line_dir(line); files = {}
    for n, f in L["files"].items():
        p = os.path.join(d, n); files[n] = "ok" if os.path.exists(p) and os.path.getsize(p) == f["size"] else "missing"
    return {"line": line, **{k: L[k] for k in ("title", "by", "licence", "about", "source")}, "files": files, "ready": all(v == "ok" for v in files.values())}


def fetch_line(line="stories260k", progress=None):
    """Download the line's pinned files (each checked against its SHA-256 before it is kept)."""
    L = LINES[line]; d = line_dir(line); os.makedirs(d, exist_ok=True)
    for n, f in L["files"].items():
        p = os.path.join(d, n)
        if os.path.exists(p) and sha_file(p) == f["sha256"]: continue
        tmp = p + ".part"; h = hashlib.sha256()
        req = urllib.request.Request(f["url"], headers={"User-Agent": "psi-commons/1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as o:
            while True:
                b = r.read(1 << 16)
                if not b: break
                o.write(b); h.update(b)
                if o.tell() > f["size"] * 2: raise SystemExit(f"{n}: larger than expected; stopped")
        if h.hexdigest() != f["sha256"]: os.remove(tmp); raise SystemExit(f"{n}: the download does not match its pinned SHA-256; not kept")
        os.replace(tmp, p)
        if progress: progress(n)
    return line_status(line)


# ── the llama2.c export ──
def read_bin(path):
    """llama2.c legacy export: 7 int32 (dim, hidden, layers, heads, kv heads, vocab (negative: separate output), seq len), then float32."""
    raw = open(path, "rb").read(); dim, hid, L, H, KV, V, T = struct.unpack_from("<7i", raw, 0); shared = V > 0; V = abs(V)
    a = np.frombuffer(raw, dtype="<f4", offset=28); o = 0; hd = dim // H; kvd = KV * hd; out = {}
    def take(n, shape):
        nonlocal o; x = a[o:o + n].reshape(shape); o += n; return x
    emb = take(V * dim, (V, dim)); rms_att = take(L * dim, (L, dim)); wq = take(L * dim * dim, (L, dim, dim)); wk = take(L * kvd * dim, (L, kvd, dim))
    wv = take(L * kvd * dim, (L, kvd, dim)); wo = take(L * dim * dim, (L, dim, dim)); rms_ffn = take(L * dim, (L, dim))
    w1 = take(L * hid * dim, (L, hid, dim)); w2 = take(L * dim * hid, (L, dim, hid)); w3 = take(L * hid * dim, (L, hid, dim)); norm = take(dim, (dim,))
    take(T * hd // 2, (T, hd // 2)); take(T * hd // 2, (T, hd // 2))   # rotary tables (recomputed)
    out["tok_embeddings.weight"] = emb; out["norm.weight"] = norm
    for l in range(L):
        p = f"layers.{l}."
        out.update({p + "attention_norm.weight": rms_att[l], p + "attention.wq.weight": wq[l], p + "attention.wk.weight": wk[l], p + "attention.wv.weight": wv[l],
                    p + "attention.wo.weight": wo[l], p + "ffn_norm.weight": rms_ffn[l], p + "feed_forward.w1.weight": w1[l], p + "feed_forward.w2.weight": w2[l],
                    p + "feed_forward.w3.weight": w3[l]})
    if not shared: raise SystemExit("this line expects an output tied to the input embedding")
    args = {"dim": dim, "hidden": hid, "n_layers": L, "n_heads": H, "n_kv_heads": KV, "vocab_size": V, "max_seq_len": T}
    return {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in out.items()}, args


class Tokenizer:
    """llama2.c BPE (tok512.bin: int max_len, then per piece: float score, int length, bytes), merged word by word."""
    def __init__(self, path):
        raw = open(path, "rb").read(); off = 4; self.vocab, self.scores = [], []
        while off < len(raw):
            s, n = struct.unpack_from("<fi", raw, off); off += 8; self.vocab.append(raw[off:off + n].decode("utf-8", "replace")); off += n; self.scores.append(s)
        self.ix = {t: i for i, t in enumerate(self.vocab)}; self._c = {}
    def _word(self, w):
        if w in self._c: return self._c[w]
        ids = []
        for ch in w:
            if ch in self.ix: ids.append(self.ix[ch])
            else: ids += [b + 3 for b in ch.encode("utf-8")]
        while True:
            best, bi, bid = -1e10, -1, -1
            for i in range(len(ids) - 1):
                j = self.ix.get(self.vocab[ids[i]] + self.vocab[ids[i + 1]])
                if j is not None and self.scores[j] > best: best, bi, bid = self.scores[j], i, j
            if bi < 0: break
            ids[bi:bi + 2] = [bid]
        self._c[w] = ids; return ids
    def encode(self, text): return [t for w in re.split(r"(?= )", " " + text) if w for t in self._word(w)]
    def decode(self, ids):
        out = bytearray()
        for i in ids:
            p = self.vocab[i]; m = re.fullmatch(r"<0x([0-9A-F]{2})>", p)
            out += bytes([int(m.group(1), 16)]) if m else p.encode("utf-8")
        return out.decode("utf-8", "replace").lstrip(" ")


class Seed:
    """A line's parent weights, shape and tokenizer, loaded from its pinned files."""
    def __init__(self, line="stories260k"):
        d = line_dir(line); st = line_status(line)
        if not st["ready"]: raise SystemExit(f"the {LINES[line]['title']} files are not here yet: fetch the line first")
        self.line = line; self.vec, self.args = read_bin(os.path.join(d, "stories260K.bin")); self.tok = Tokenizer(os.path.join(d, "tok512.bin"))
        a = self.args; self.d, self.L, self.H, self.KV, self.V = a["dim"], a["n_layers"], a["n_heads"], a["n_kv_heads"], a["vocab_size"]; self.hd = self.d // self.H
        f = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2).float() / self.hd)); ang = torch.outer(torch.arange(a["max_seq_len"]).float(), f)
        self.cos, self.sin = ang.cos(), ang.sin()
        self.sha = vec_sha(self.vec)

    def forward(self, vec, ids):
        B, T = ids.shape; d, H, KV, hd = self.d, self.H, self.KV, self.hd; g = vec.__getitem__
        c, s = self.cos[:T][None, :, None, :], self.sin[:T][None, :, None, :]
        def rope(x): x0, x1 = x[..., 0::2], x[..., 1::2]; return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)
        def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        x = g("tok_embeddings.weight")[ids]
        for l in range(self.L):
            p = f"layers.{l}."; h = rms(x, g(p + "attention_norm.weight"))
            q = rope((h @ g(p + "attention.wq.weight").T).view(B, T, H, hd))
            k = rope((h @ g(p + "attention.wk.weight").T).view(B, T, KV, hd)).repeat_interleave(H // KV, 2)
            v = (h @ g(p + "attention.wv.weight").T).view(B, T, KV, hd).repeat_interleave(H // KV, 2)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, d)
            x = x + o @ g(p + "attention.wo.weight").T
            h2 = rms(x, g(p + "ffn_norm.weight"))
            x = x + (F.silu(h2 @ g(p + "feed_forward.w1.weight").T) * (h2 @ g(p + "feed_forward.w3.weight").T)) @ g(p + "feed_forward.w2.weight").T
        return rms(x, g("norm.weight")) @ g("tok_embeddings.weight").T

    @torch.no_grad()
    def sample(self, vec, n, temp=0.8, seed=0, prompt=""):
        g = torch.Generator().manual_seed(seed); ids = torch.tensor([[1] + self.tok.encode(prompt)]) if prompt else torch.tensor([[1]]); start = ids.shape[1]
        for _ in range(n):
            p = F.softmax(self.forward(vec, ids[:, -256:])[0, -1] / temp, -1); ids = torch.cat([ids, torch.multinomial(p, 1, generator=g)[None]], 1)
        return ids[0, start:]

    @torch.no_grad()
    def loss(self, vec, ids, n=24, seed=99, T=128):
        ids = torch.as_tensor(ids); T = min(T, len(ids) - 2); g = torch.Generator().manual_seed(seed)
        ix = torch.randint(0, len(ids) - T - 1, (n,), generator=g)
        x = torch.stack([torch.cat([torch.tensor([1]), ids[i:i + T - 1]]) for i in ix]); y = torch.stack([ids[i:i + T] for i in ix])
        return float(F.cross_entropy(self.forward(vec, x).reshape(-1, self.V), y.reshape(-1)))


def vec_sha(vec):
    h = hashlib.sha256()
    for k in sorted(vec): h.update(k.encode()); h.update(np.ascontiguousarray(vec[k].detach().cpu().numpy() if torch.is_tensor(vec[k]) else vec[k], dtype="<f4").tobytes())
    return h.hexdigest()


def distance(a, b):
    """Relative L2 distance between two weight sets of one shape: |a - b| / |b|."""
    num = sum(float(((a[k].double() - b[k].double()) ** 2).sum()) for k in b); den = sum(float((b[k].double() ** 2).sum()) for k in b)
    return (num / den) ** 0.5


def data_hash(ids, salt):
    """The training text, named without revealing it: SHA-256(salt || int32 piece ids). The salt goes only to replayers."""
    return hashlib.sha256(bytes.fromhex(salt) + np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


# ── training with a trace ──
DEFAULT_CFG = {"steps": 200, "lr": 3e-4, "batch": 8, "ctx": 128, "betas": [0.9, 0.99], "weight_decay": 0.01, "clip": 1.0,
               "rehearse": 0.25, "rehearsal_stories": 40, "seed": 0, "every": 25, "threads": 1}   # 1 thread: bit-identical replays on CPU (more threads reorder sums)


def rehearsal(seed_obj, parent, n, seed):
    """The parent's own stories, regenerated from the parent (deterministic), mixed in so a branch keeps what it knew.
    Kept in <data>/commons/cache by (parent, seed, n): they take half a minute to write, and every replay needs them."""
    from safetensors.torch import save_file, load_file
    f = os.path.join(DATA, "commons", "cache", f"rehearsal-{vec_sha(parent)[:16]}-{seed}-{n}.safetensors")
    if os.path.exists(f): return load_file(f)["ids"]
    torch.set_num_threads(1)
    ids = torch.cat([seed_obj.sample(parent, 256, 0.8, seed * 1000 + s) for s in range(n)])
    os.makedirs(os.path.dirname(f), exist_ok=True); save_file({"ids": ids.contiguous()}, f); return ids


def _batch(data, reh, cfg, step):
    g = torch.Generator().manual_seed(int(cfg["seed"]) * 1_000_003 + step); T = cfg["ctx"]; xs, ys = [], []
    for _ in range(cfg["batch"]):
        src = reh if (reh is not None and cfg["rehearse"] and torch.rand(1, generator=g).item() < cfg["rehearse"]) else data
        i = int(torch.randint(0, len(src) - T - 1, (1,), generator=g)); xs.append(torch.cat([torch.tensor([1]), src[i:i + T - 1]])); ys.append(src[i:i + T])
    return torch.stack(xs), torch.stack(ys)


def _opt(params, cfg):
    return torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])


def save_ckpt(path, params, opt, step):
    from safetensors.torch import save_file
    t = {f"w.{k}": p.detach().contiguous() for k, p in params.items()}
    for k, p in params.items():
        s = opt.state.get(p, {})
        if s: t[f"m.{k}"] = s["exp_avg"].contiguous(); t[f"v.{k}"] = s["exp_avg_sq"].contiguous()
    save_file(t, path, metadata={"step": str(step)})


def load_ckpt(path, cfg):
    from safetensors.torch import load_file
    t = load_file(path); names = sorted(k[2:] for k in t if k.startswith("w."))
    params = {k: t[f"w.{k}"].clone().requires_grad_(True) for k in names}; opt = _opt(list(params.values()), cfg)
    from safetensors import safe_open
    with safe_open(path, "pt") as f: step = int(f.metadata()["step"])
    if f"m.{names[0]}" in t:
        for k in names:
            opt.state[params[k]] = {"step": torch.tensor(float(step)), "exp_avg": t[f"m.{k}"].clone(), "exp_avg_sq": t[f"v.{k}"].clone()}
    return params, opt, step


def run_steps(seed_obj, params, opt, data, reh, cfg, start, stop, on_step=None):
    torch.set_num_threads(int(cfg.get("threads", 1)))
    plist = list(params.values()); last = None
    for s in range(start, stop):
        x, y = _batch(data, reh, cfg, s)
        l = F.cross_entropy(seed_obj.forward(params, x).reshape(-1, seed_obj.V), y.reshape(-1)); opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(plist, cfg["clip"]); opt.step(); last = float(l)
        if on_step: on_step(s + 1, last)
    return last


def train_branch(seed_obj, parent, ids, cfg, ckpt_dir, on_step=None):
    """Train from `parent` on `ids`; returns (weights, trace). Checkpoints (weights + optimiser) go to ckpt_dir."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}; os.makedirs(ckpt_dir, exist_ok=True); torch.set_num_threads(int(cfg["threads"]))
    data = torch.as_tensor(ids); reh = rehearsal(seed_obj, parent, cfg["rehearsal_stories"], cfg["seed"]) if cfg["rehearse"] else None
    params = {k: parent[k].clone().requires_grad_(True) for k in sorted(parent)}; opt = _opt(list(params.values()), cfg)   # sorted: the same order as a replay
    salt = os.urandom(16).hex(); cks = []
    def ck(step):
        p = os.path.join(ckpt_dir, f"step{step:06d}.safetensors"); save_ckpt(p, params, opt, step)
        cks.append({"step": step, "weights_sha256": vec_sha({k: v.detach() for k, v in params.items()}), "file": os.path.basename(p), "file_sha256": sha_file(p)})
    ck(0)
    for a in range(0, cfg["steps"], cfg["every"]):
        b = min(cfg["steps"], a + cfg["every"]); run_steps(seed_obj, params, opt, data, reh, cfg, a, b, on_step); ck(b)
    trace = {"v": 1, "kind": "trace", "method": "segment replay (proof-of-learning style)", "config": cfg, "torch": torch.__version__, "threads": cfg["threads"],
             "parent_sha256": vec_sha(parent), "data": {"pieces": int(len(data)), "sha256": data_hash(data, salt), "salted": True},
             "rehearsal": {"from": "parent", "stories": cfg["rehearsal_stories"], "seed": cfg["seed"]} if reh is not None else None, "checkpoints": cks}
    return {k: v.detach() for k, v in params.items()}, trace, salt


def replay(seed_obj, parent, ids, salt, trace, ckpt_dir, segments):
    """Replay chosen segments of a trace. Each: load checkpoint k (its file hash checked), run to k+1, compare."""
    cfg = trace["config"]; out = []
    if data_hash(ids, salt) != trace["data"]["sha256"]: return [{"segment": None, "ok": False, "why": "the text given is not the text the trace names"}]
    if vec_sha(parent) != trace["parent_sha256"]: return [{"segment": None, "ok": False, "why": "the parent is not the one the trace names"}]
    torch.set_num_threads(int(cfg.get("threads", 1)))
    data = torch.as_tensor(ids); reh = rehearsal(seed_obj, parent, cfg["rehearsal_stories"], cfg["seed"]) if trace.get("rehearsal") else None
    cks = trace["checkpoints"]
    for i in segments:
        a, b = cks[i], cks[i + 1]; pa = os.path.join(ckpt_dir, a["file"])
        if not os.path.exists(pa): out.append({"segment": i, "ok": False, "why": f"checkpoint {a['step']} is missing"}); continue
        if sha_file(pa) != a["file_sha256"]: out.append({"segment": i, "ok": False, "why": f"checkpoint {a['step']} does not match its hash"}); continue
        params, opt, step = load_ckpt(pa, cfg)
        if vec_sha({k: v.detach() for k, v in params.items()}) != a["weights_sha256"]: out.append({"segment": i, "ok": False, "why": "checkpoint weights differ from the trace"}); continue
        run_steps(seed_obj, params, opt, data, reh, cfg, a["step"], b["step"])
        got = {k: v.detach() for k, v in params.items()}; exact = vec_sha(got) == b["weights_sha256"]; diff = None
        pb = os.path.join(ckpt_dir, b["file"])
        if not exact and os.path.exists(pb):
            from safetensors.torch import load_file
            tb = load_file(pb); want = {k[2:]: v for k, v in tb.items() if k.startswith("w.")}
            moved = distance(want, {k: v for k, v in load_ckpt(pa, cfg)[0].items()}) or 1e-12
            diff = distance(got, want) / moved   # the miss, as a share of how far the segment moved
        out.append({"segment": i, "from": a["step"], "to": b["step"], "ok": exact or (diff is not None and diff < 0.01), "exact": exact, "miss_share": diff})
    return out
