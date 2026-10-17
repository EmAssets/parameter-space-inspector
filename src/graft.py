"""
Graft: take layers out of one model (the host) and put another model's layers (the donor) in their place, then measure
what happens. The question is the tipping point's, turned around: not only "what dies when this part goes", but "does
a part from elsewhere do the job here", and which parts fit.

  python graft.py graft --host H --donor D --host-layers a:b --donor-layers c:d [--align] [--out F]
  python graft.py sweep --host H --donor D [--width 1] [--band 0.15 | --full] [--out F]
  python graft.py rank  --host H --donors D1,D2,... [--out F]

How a donor's layers go in:
  direct     same architecture and the same attention shape (heads, head size, rotary settings): the donor's layer
             modules replace the host's as they are (the feed-forward width may differ: a layer is self-contained)
  wrapped    anything else: the donor's layers run as one block with the donor's own rotary positions; the residual
             stream goes in and out through maps. Same width: the identity (or, with --align, a fitted affine map).
             Different width: a fitted affine map each way (model stitching, Bansal et al. 2021; Lenc & Vedaldi 2015).
             Maps are fitted by ridge regression on activations of the same tokens in both models, so they need the
             same vocabulary; they are fitted on text the measurements do not use.

Each graft is measured against controls on held-out text the maps never saw:
  host       the host as it is            skip    the host with layers a:b removed and nothing put in
  donor      the donor as it is           random  the same graft with the donor's layers re-initialised (same shapes)
  fit (text) = (bits_skip - bits_graft) / (bits_skip - bits_host): 1 = the graft does the removed layers' job,
               0 = no better than leaving the gap, below 0 = worse than the gap. For an insertion (a == b) the baseline
               is the random graft instead of the gap.
  copying    the same, on the copying score (a repeated random sequence: nats saved on the second copy)

A sweep grafts one window (width layers) of the donor at every host depth (all pairs, or a band around the same
relative depth) and maps the fit. rank tries one mid-depth graft per donor. Nothing on disk changes; the result
carries a signed receipt (src/receipts.py). Progress: "PSI-PROGRESS {json}"; last line PSI-DONE or PSI-ERROR.
"""
import argparse, copy, glob, importlib.util, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "research")
sys.path.insert(0, HERE)
import torch
import devices   # noqa: E402
from torch import nn


def say(kind, **o): print(f"PSI-{kind} {json.dumps(o)}", flush=True)


def mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(RES, f"{name}.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ── texts: maps are fitted on the first half, everything is measured on the second ─────────────────────────────────
def texts():
    fs = sorted(glob.glob(os.path.join(RES, "results", "heldout-*.txt")))
    t = [open(f, encoding="utf-8").read() for f in fs]
    if len(t) < 2: raise SystemExit("PSI-ERROR grafting needs the held-out texts in research/results/heldout-*.txt")
    h = len(t) // 2
    return {"fit": t[:h], "eval": t[h:], "fit_files": [os.path.basename(f) for f in fs[:h]], "eval_files": [os.path.basename(f) for f in fs[h:]]}


# ── models ───────────────────────────────────────────────────────────────────────────────────────────────────────
LAYER_PATHS = ("gpt_neox.layers", "model.layers", "transformer.h", "model.decoder.layers")


def layer_slot(model):
    for p in LAYER_PATHS:
        obj, parent = model, None
        for part in p.split("."):
            parent, obj = obj, getattr(obj, part, None)
            if obj is None: break
        if isinstance(obj, nn.ModuleList): return parent, p.split(".")[-1]
    raise SystemExit(f"PSI-ERROR cannot find the layer list of {type(model).__name__}")


def files_bytes(d): return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")) + glob.glob(os.path.join(d, "*.bin")))


class Loaded:
    def __init__(self, path, dev, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.path = os.path.abspath(path); self.name = os.path.relpath(self.path, os.path.dirname(os.path.dirname(self.path))).replace("\\", "/")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(dev).eval()
        self.model.config.use_cache = False
        self.inner, self.attr = layer_slot(self.model)
        self.layers = getattr(self.inner, self.attr); self.n = len(self.layers)
        c = self.model.config; self.cfg = c
        self.hidden = c.hidden_size; self.heads = c.num_attention_heads; self.kv = getattr(c, "num_key_value_heads", None) or self.heads
        self.head_dim = getattr(c, "head_dim", None) or self.hidden // self.heads
        rp = getattr(c, "rope_parameters", None) or {}
        self.rope = json.dumps({"theta": rp.get("rope_theta", getattr(c, "rope_theta", None)), "type": rp.get("rope_type"), "pct": getattr(c, "rotary_pct", None) or rp.get("partial_rotary_factor")}, sort_keys=True)
        v = self.tok.get_vocab(); self.vocab_key = hash(tuple(sorted(v.items())))

    def set_layers(self, mods):
        ml = nn.ModuleList(mods); setattr(self.inner, self.attr, ml)
        if hasattr(self.cfg, "num_hidden_layers"): self.cfg.num_hidden_layers = len(mods)

    def restore(self): self.set_layers(list(self.layers))


def how(host, donor, align=False):
    same_attn = (host.cfg.model_type == donor.cfg.model_type and host.hidden == donor.hidden and host.heads == donor.heads and host.kv == donor.kv
                 and host.head_dim == donor.head_dim and host.rope == donor.rope)
    if same_attn and not align: return "direct"
    return "wrapped"


def compat(host, donor):
    same_vocab = host.vocab_key == donor.vocab_key
    if how(host, donor) == "direct": return {"kind": "drop-in", "why": "same architecture and attention shape: the donor's layers go in as they are", "same_vocab": same_vocab}
    if host.hidden == donor.hidden: return {"kind": "same width", "why": f"both {host.hidden} wide; different {'architecture' if host.cfg.model_type != donor.cfg.model_type else 'attention shape'}: the donor's layers run with their own positions, through the identity (or a fitted map with align)", "same_vocab": same_vocab}
    if same_vocab: return {"kind": "stitched", "why": f"{host.hidden} vs {donor.hidden} wide: fitted maps each way on the same tokens (same vocabulary)", "same_vocab": True}
    return {"kind": "incompatible", "why": f"{host.hidden} vs {donor.hidden} wide and different vocabularies: no shared tokens to fit a map on", "same_vocab": False}


class Wrapped(nn.Module):
    """The donor's layers as one block in the host: maps in and out, the donor's own rotary positions."""
    attention_type = "full_attention"

    def __init__(self, layers, rotary, map_in, map_out):
        super().__init__(); self.block = nn.ModuleList(layers); self.rotary = rotary; self.map_in = map_in; self.map_out = map_out

    def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None, **kw):
        x = self.map_in(hidden_states)
        if position_ids is None: position_ids = torch.arange(x.shape[1], device=x.device)[None]
        pe = self.rotary(x, position_ids) if self.rotary is not None else position_embeddings
        for l in self.block:
            x = l(x, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=pe, use_cache=False)
            if isinstance(x, tuple): x = x[0]
        return self.map_out(x)


def affine(W, b):
    m = nn.Linear(W.shape[0], W.shape[1], bias=True)
    with torch.no_grad(): m.weight.copy_(W.T); m.bias.copy_(b)
    return m


def ridge(X, Y, lam=1e-2):
    """Least squares X -> Y with a bias, ridge-regularised relative to X's scale. Returns (W, b, r2 on the fit data)."""
    X = X.double(); Y = Y.double(); mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    A = Xc.T @ Xc; A += torch.eye(A.shape[0], dtype=A.dtype, device=A.device) * lam * A.diagonal().mean()
    W = torch.linalg.solve(A, Xc.T @ Yc); b = my - mx @ W
    r2 = 1 - float(((Yc - Xc @ W) ** 2).sum() / (Yc ** 2).sum().clamp_min(1e-12))
    return W.float(), b.float(), round(r2, 4)


# ── activations at every layer boundary (for fitting maps) ───────────────────────────────────────────────────────
def windows(L, texts_, dev, cap):
    """512-token windows of the texts, in order, until about `cap` tokens."""
    out, n = [], 0
    for t in texts_:
        ids = L.tok(t, return_tensors="pt").input_ids
        for i in range(0, ids.shape[1] - 32, 512):
            out.append(ids[:, i:i + 512].to(dev)); n += out[-1].shape[1]
            if n >= cap: return out
    return out


def fit_cap(*Ls):
    """Tokens to read for fitting maps: up to 8,192, fewer when every layer boundary of both models would not fit in ~1.5 GB."""
    per = sum((L.n + 1) * L.hidden * 4 for L in Ls)
    return int(max(1024, min(8192, 1.5e9 / per)))


def r2_of(m, X, Y):
    with torch.no_grad():
        P = m(X.to(m.weight.device, m.weight.dtype)).float().cpu(); Yc = Y - Y.mean(0)
        return round(1 - float(((Y - P) ** 2).sum() / (Yc ** 2).sum().clamp_min(1e-12)), 4)


@torch.no_grad()
def boundaries(L, ids_list, cap=8192):
    """resid[i] = the residual stream entering layer i (i = n: the last layer's output, before the final norm),
    over the fit tokens, the first token of each text left out (attention sink)."""
    store = {i: [] for i in range(L.n + 1)}; hooks = []
    for i, layer in enumerate(L.layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, a, kw, i=i: store[i].append((a[0] if a else kw["hidden_states"])[0, 1:].float().cpu()), with_kwargs=True))
    hooks.append(L.layers[-1].register_forward_hook(lambda m, a, o: store[L.n].append((o[0] if isinstance(o, tuple) else o)[0, 1:].float().cpu())))
    try:
        for ids in ids_list: L.model(ids)
    finally:
        for h in hooks: h.remove()
    return {i: torch.cat(v)[:cap] for i, v in store.items()}


# ── measurements ─────────────────────────────────────────────────────────────────────────────────────────────────
class Meter:
    def __init__(self, dev, B, PS, T, quick=False):
        self.dev, self.B, self.PS, self.quick = dev, B, PS, quick; PS.DEV = dev
        self.eval_texts = T["eval"]

    @torch.no_grad()
    def bits(self, L, maxlen):
        tot, n = 0.0, 0
        for t in self.eval_texts:
            ids = L.tok(t, return_tensors="pt").input_ids[:, :maxlen].to(self.dev)
            out = L.model(ids, labels=ids); k = ids.shape[1] - 1; tot += float(out.loss) * k; n += k
        return tot / n / math.log(2)

    @torch.no_grad()
    def margin(self, L, prompt, good, bad):
        lg = torch.log_softmax(L.model(L.tok(prompt, return_tensors="pt").input_ids.to(self.dev)).logits[0, -1].float(), -1)
        f = lambda s: L.tok(s, add_special_tokens=False).input_ids[0]
        return float(max(lg[f(x)] for x in good) - max(lg[f(x)] for x in bad))

    @torch.no_grad()
    def sample(self, L, prompt="The old market opens before dawn, and", n=24):
        ids = L.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        for _ in range(n):
            nxt = L.model(ids).logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], 1)
        return L.tok.decode(ids[0, -n:], skip_special_tokens=True)

    def measure(self, L):
        r = {}
        try:
            r["bits"] = round(self.bits(L, 256 if self.quick else 512), 4)
            r["copying"] = round(self.PS.induction_score(L.model, L.tok, trials=2 if self.quick else 4)["score"], 4)
            if not self.quick:
                B = self.B
                r["facts"] = round(sum(self.margin(L, p, [gd], [bd]) > 0 for p, gd, bd in B.FACTS) / len(B.FACTS), 3)
                r["agreement"] = round(sum(self.margin(L, p, [gd], [bd]) > 0 for n in (0, 3) for p, gd, bd in B.SPAN[n]) / (2 * len(B.SPAN[0])), 3)
                r["sample"] = self.sample(L)
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        for k in ("bits", "copying"):
            if isinstance(r.get(k), float) and not math.isfinite(r[k]): r[k] = None
        return r


FLOOR = {True: 0.05, False: 0.5}   # bits per token / nats of copying: a host this close to its baseline gives no scale


def fit_score(base, graft, host, lower_better):
    if None in (base, graft, host): return None
    d = (base - host) if lower_better else (host - base)
    if d < FLOOR[lower_better]: return None   # the removed layers did not carry this (or the host lacks it): see "gain"
    return round(((base - graft) if lower_better else (graft - base)) / d, 3)


def reinit(mods):
    m = copy.deepcopy(mods)
    with torch.no_grad():
        for mod_ in m:
            for name, p in mod_.named_parameters():
                if p.ndim >= 2: p.normal_(0, 0.02)
                elif "norm" in name.lower() or "ln" in name.lower(): p.fill_(1.0 if name.endswith("weight") else 0.0)
                else: p.zero_()
    return m


# ── building a graft ─────────────────────────────────────────────────────────────────────────────────────────────
def block(host, donor, c, d, kind, maps):
    """The donor's layers c:d, ready to go into the host. maps: (map_in, map_out) or None."""
    mods = [copy.deepcopy(donor.layers[k]) for k in range(c, d)]
    if kind == "direct": return mods
    rot = getattr(donor.inner, "rotary_emb", None)
    mi, mo = maps if maps else (nn.Identity(), nn.Identity())
    dev = next(donor.model.parameters()).device; dt = next(donor.model.parameters()).dtype
    return [Wrapped(mods, rot, mi.to(dev, dt), mo.to(dev, dt))]


def fitted_maps(hR, dR, a, b, c, d, host, donor):
    """map_in: host stream entering layer a -> donor stream entering c; map_out: donor stream after d-1 -> host stream entering b."""
    Wi, bi, r2i = ridge(hR[a], dR[c]); Wo, bo, r2o = ridge(dR[d], hR[b])
    return (affine(Wi, bi), affine(Wo, bo)), {"r2_in": r2i, "r2_out": r2o, "tokens": int(hR[a].shape[0])}


def apply(host, a, b, mods):
    host.set_layers(list(host.layers[:a]) + list(mods) + list(host.layers[b:]))


# ── commands ─────────────────────────────────────────────────────────────────────────────────────────────────────
def setup(a, paths):
    dev = a.device
    need = sum(files_bytes(p) for p in paths)
    if dev == "auto":
        dev = devices.pick((need * 1.6 + 1.5e9) / 2 ** 30)
    if dev == "cpu": torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    dtype = torch.float32 if max(files_bytes(p) for p in paths) < 2.5e9 else (devices.dtype(dev) if dev != "cpu" else torch.bfloat16)
    return dev, dtype


def rng(s, n):
    x, y = s.split(":"); x, y = int(x), int(y)
    if not (0 <= x <= y <= n): raise SystemExit(f"PSI-ERROR layer range {s} is outside 0..{n}")
    return x, y


def do_graft(a, B, PS, T):
    dev, dtype = setup(a, [a.host, a.donor]); say("PROGRESS", stage="load", note=f"loading host and donor on the {dev}")
    H = Loaded(a.host, dev, dtype); D = Loaded(a.donor, dev, dtype)
    hl, hr = rng(a.host_layers, H.n); dl, dr = rng(a.donor_layers, D.n)
    cp = compat(H, D); kind = how(H, D, a.align)
    if cp["kind"] == "incompatible": raise SystemExit(f"PSI-ERROR {cp['why']}")
    M = Meter(dev, B, PS, T); maps, fit = None, None
    if kind == "wrapped" and (H.hidden != D.hidden or a.align):
        if not cp["same_vocab"]: raise SystemExit("PSI-ERROR fitting the maps needs the same vocabulary in both models")
        say("PROGRESS", stage="fit", note="fitting the maps in and out on the fit text")
        cap = fit_cap(H, D); ids = windows(H, T["fit"], dev, cap)
        maps, fit = fitted_maps(boundaries(H, ids, cap), boundaries(D, ids, cap), hl, hr, dl, dr, H, D)
        say("PROGRESS", stage="fit", note="checking the maps on held-out text")
        ev = windows(H, T["eval"], dev, 2048); hE, dE = boundaries(H, ev, 2048), boundaries(D, ev, 2048)
        fit["r2_in_heldout"] = r2_of(maps[0], hE[hl], dE[dl]); fit["r2_out_heldout"] = r2_of(maps[1], dE[dr], hE[hr])
    out = {"host": H.name, "donor": D.name, "host_layers": [hl, hr], "donor_layers": [dl, dr], "compat": cp, "how": kind, "maps": fit, "variants": {}}
    say("PROGRESS", stage="measure", note="the host as it is", done=0, of=5); out["variants"]["host"] = M.measure(H)
    say("PROGRESS", stage="measure", note="the donor as it is", done=1, of=5); out["variants"]["donor"] = M.measure(D)
    say("PROGRESS", stage="measure", note=f"the host without layers {hl}:{hr}", done=2, of=5)
    apply(H, hl, hr, []); out["variants"]["skip"] = M.measure(H) if hr > hl else out["variants"]["host"]; H.restore()
    say("PROGRESS", stage="measure", note="the graft", done=3, of=5)
    apply(H, hl, hr, block(H, D, dl, dr, kind, maps)); out["variants"]["graft"] = M.measure(H); H.restore()
    say("PROGRESS", stage="measure", note="the random control (the donor's layers re-initialised)", done=4, of=5)
    rnd = reinit([D.layers[k] for k in range(dl, dr)]); Dr = copy.copy(D); Dr.layers = nn.ModuleList(rnd)
    apply(H, hl, hr, block(H, Dr, 0, dr - dl, kind, maps)); out["variants"]["random"] = M.measure(H); H.restore()
    v = out["variants"]; base = "skip" if hr > hl else "random"
    out["fit"] = {"baseline": base,
                  "text": fit_score(v[base].get("bits"), v["graft"].get("bits"), v["host"].get("bits"), True),
                  "copying": fit_score(v[base].get("copying"), v["graft"].get("copying"), v["host"].get("copying"), False),
                  "gain_text_bits": None if None in (v["host"].get("bits"), v["graft"].get("bits")) else round(v["host"]["bits"] - v["graft"]["bits"], 4),
                  "gain_copying_nats": None if None in (v["host"].get("copying"), v["graft"].get("copying")) else round(v["graft"]["copying"] - v["host"]["copying"], 4),
                  "rule": f"(baseline - graft) / (baseline - host), baseline = the {'gap (layers removed)' if base == 'skip' else 'random graft'}; 1 = does the job, 0 = no better than the baseline; "
                          "none when the host is within 0.05 bits / 0.5 nats of the baseline (those layers did not carry it, or the host lacks it): then read the gain, graft against host"}
    out["texts"] = {"fit": T["fit_files"], "eval": T["eval_files"]}
    return out, dev, dtype, [a.host, a.donor]


def do_sweep(a, B, PS, T):
    dev, dtype = setup(a, [a.host, a.donor]); say("PROGRESS", stage="load", note=f"loading host and donor on the {dev}")
    H = Loaded(a.host, dev, dtype); D = Loaded(a.donor, dev, dtype)
    cp = compat(H, D); kind = how(H, D, a.align)
    if cp["kind"] == "incompatible": raise SystemExit(f"PSI-ERROR {cp['why']}")
    w = max(1, min(a.width, H.n, D.n)); M = Meter(dev, B, PS, T, quick=True)
    need_maps = kind == "wrapped" and (H.hidden != D.hidden or a.align)
    if need_maps:
        if not cp["same_vocab"]: raise SystemExit("PSI-ERROR fitting the maps needs the same vocabulary in both models")
        say("PROGRESS", stage="fit", note="reading both models' streams on the fit text")
        cap = fit_cap(H, D); ids = windows(H, T["fit"], dev, cap); hR, dR = boundaries(H, ids, cap), boundaries(D, ids, cap)
    hs, ds = list(range(0, H.n - w + 1)), list(range(0, D.n - w + 1))
    rel = lambda i, n: (i + w / 2) / n
    cells = [(i, j) for i in hs for j in ds if a.full or abs(rel(i, H.n) - rel(j, D.n)) <= a.band]
    say("PROGRESS", stage="measure", note="the host as it is"); host = M.measure(H)
    skips = {}
    for k, i in enumerate(hs):
        say("PROGRESS", stage="baseline", done=k, of=len(hs), note=f"the host without layers {i}:{i + w}")
        apply(H, i, i + w, []); skips[i] = M.measure(H); H.restore()
    grid = []
    t0 = time.time()
    for k, (i, j) in enumerate(cells):
        left = (time.time() - t0) / k * (len(cells) - k) if k else None
        say("PROGRESS", stage="sweep", done=k, of=len(cells), note=f"host {i}:{i + w} <- donor {j}:{j + w}" + (f", about {round(left / 60, 1)} min left" if left else ""))
        maps = fitted_maps(hR, dR, i, i + w, j, j + w, H, D)[0] if need_maps else None
        apply(H, i, i + w, block(H, D, j, j + w, kind, maps)); r = M.measure(H); H.restore()
        grid.append({"host": i, "donor": j, "bits": r.get("bits"), "copying": r.get("copying"), "error": r.get("error"),
                     "fit_text": fit_score(skips[i].get("bits"), r.get("bits"), host.get("bits"), True),
                     "fit_copying": fit_score(skips[i].get("copying"), r.get("copying"), host.get("copying"), False),
                     "gain_copying": None if None in (r.get("copying"), host.get("copying")) else round(r["copying"] - host["copying"], 3)})
    best = sorted([g for g in grid if g["fit_text"] is not None], key=lambda g: -g["fit_text"])[:10]
    out = {"host": H.name, "donor": D.name, "host_n": H.n, "donor_n": D.n, "width": w, "band": None if a.full else a.band, "compat": cp, "how": kind,
           "host_measure": host, "skips": {str(i): s for i, s in skips.items()}, "grid": grid, "best": best,
           "rule": "fit = (gap - graft) / (gap - host) per host window, on the eval text (256 tokens a text) and the copying score (2 trials)", "texts": {"fit": T["fit_files"], "eval": T["eval_files"]}}
    return out, dev, dtype, [a.host, a.donor]


def do_rank(a, B, PS, T):
    donors = [d for d in a.donors.split(",") if d]
    dev, dtype = setup(a, [a.host] + donors); say("PROGRESS", stage="load", note=f"loading the host on the {dev}")
    H = Loaded(a.host, dev, dtype); M = Meter(dev, B, PS, T, quick=True); host = M.measure(H)
    i = H.n // 2; apply(H, i, i + 1, []); skip = M.measure(H); H.restore()
    rows = []
    for k, dp in enumerate(donors):
        say("PROGRESS", stage="rank", done=k, of=len(donors), note=f"donor {os.path.basename(dp)}")
        try:
            D = Loaded(dp, dev, dtype); cp = compat(H, D)
            if cp["kind"] == "incompatible": rows.append({"donor": D.name, "compat": cp, "fit_text": None, "fit_copying": None}); del D; continue
            j = min(D.n - 1, round((i + 0.5) / H.n * D.n - 0.5)); kind = how(H, D); maps = None
            if kind == "wrapped" and H.hidden != D.hidden:
                cap = fit_cap(H, D); ids = windows(H, T["fit"], dev, cap)
                maps = fitted_maps(boundaries(H, ids, cap), boundaries(D, ids, cap), i, i + 1, j, j + 1, H, D)[0]
            apply(H, i, i + 1, block(H, D, j, j + 1, kind, maps)); r = M.measure(H); H.restore()
            rows.append({"donor": D.name, "donor_layer": j, "compat": cp, "how": kind, "bits": r.get("bits"), "copying": r.get("copying"),
                         "fit_text": fit_score(skip.get("bits"), r.get("bits"), host.get("bits"), True), "fit_copying": fit_score(skip.get("copying"), r.get("copying"), host.get("copying"), False)})
            del D
        except Exception as e:
            rows.append({"donor": dp, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if dev != "cpu": devices.empty_cache(dev)
    rows.sort(key=lambda r: -(r.get("fit_text") if r.get("fit_text") is not None else -9))
    return {"host": H.name, "host_layer": i, "host_measure": host, "skip": skip, "donors": rows,
            "rule": "one layer at the host's middle depth replaced by the donor's layer at the same relative depth; fit as in a graft"}, dev, dtype, [a.host] + donors


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("graft", "sweep", "rank"):
        p = sub.add_parser(name); p.add_argument("--host", required=True); p.add_argument("--out"); p.add_argument("--device", default="auto"); p.add_argument("--align", action="store_true")
        if name == "rank": p.add_argument("--donors", required=True)
        else: p.add_argument("--donor", required=True)
        if name == "graft": p.add_argument("--host-layers", required=True); p.add_argument("--donor-layers", required=True)
        if name == "sweep": p.add_argument("--width", type=int, default=1); p.add_argument("--band", type=float, default=0.15); p.add_argument("--full", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    torch.manual_seed(0)
    B = mod("emergence_battery"); PS = mod("phase_shift"); T = texts()
    out, dev, dtype, paths = {"graft": do_graft, "sweep": do_sweep, "rank": do_rank}[a.cmd](a, B, PS, T)
    import receipts
    say("PROGRESS", stage="receipt", note="hashing the models' files for the receipt")
    subject = {"models": [{"path": p.replace("\\", "/"), **receipts.model_identity(p)} for p in paths]}
    subject["weights_sha256"] = receipts.sha_bytes("\n".join(m["weights_sha256"] or "" for m in subject["models"]).encode())
    out["kind"] = a.cmd; out["seconds"] = round(time.time() - t0, 1); out["device"] = dev; out["dtype"] = str(dtype).replace("torch.", "")
    out["receipt"] = receipts.make(f"psi-{a.cmd}", subject, {"what": f"graft lab: {a.cmd}", "args": {k: v for k, v in vars(a).items() if k not in ("out",)},
                                                              "code": receipts.code_identity([os.path.abspath(__file__), os.path.join(HERE, "receipts.py"), os.path.join(RES, "emergence_battery.py"), os.path.join(RES, "phase_shift.py")]), "seed": 0},
                                   receipts.env(dev, dtype), {k: v for k, v in out.items() if k not in ("receipt", "seconds")})
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); tmp = a.out + ".part"
        json.dump(out, open(tmp, "w", encoding="utf-8"), indent=1); os.replace(tmp, a.out)
    say("DONE", out=os.path.abspath(a.out) if a.out else None, cmd=a.cmd, seconds=out["seconds"], fit=out.get("fit"), best=(out.get("best") or [None])[0], signed=bool(out["receipt"]["signature"]))


if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        msg = str(e.code) if e.code else ""
        if msg: print(msg if msg.startswith("PSI-ERROR") else f"PSI-ERROR {msg}", flush=True)
        sys.exit(1 if msg else 0)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"PSI-ERROR {type(e).__name__}: {e}", flush=True); sys.exit(1)
