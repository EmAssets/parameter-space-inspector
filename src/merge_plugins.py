# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Merge plugins: every way of making one model out of a base and several branches' changes.

One contract, so a better method can be added without touching the rest:

  @plugin(name, about, linear=..., options={...})
  def method(base, deltas, strengths, options) -> {name: float64 array}     the combined change (not the weights)

  base       {name: float64 array}       the weights every branch started from
  deltas     [{name: float64 array}]     each branch's change, in the recipe's order
  strengths  [float]                     per branch (1 = as trained; a recipe may weaken an old or stale branch)

Every built-in is canonical: float64 arithmetic in a fixed order (the recipe's), randomness only from NumPy's PCG64 with
a pinned seed, rounded once to float32 at the end (recipes.build). The same recipe gives the same bytes on any
computer, in any language that follows the arithmetic written in each `about`.

A plugin is pinned in a recipe by the SHA-256 of its own source, so changing one plugin never invalidates recipes
that use another. Plugins from elsewhere (<data>/commons/plugins/*.py) are never run until the user approves that
exact file (its SHA-256 is added to approved.json by the app's own button), and they get only arrays: no files, no
network, no paths.
"""
import hashlib, importlib.util, inspect, json, os
import numpy as np

REGISTRY = {}


def plugin(name, about, linear=False, options=None):
    def deco(fn):
        src = inspect.getsource(fn)
        REGISTRY[name] = {"fn": fn, "about": about, "linear": linear, "options": options or {}, "code_sha256": hashlib.sha256(src.encode()).hexdigest(), "origin": "built-in"}
        return fn
    return deco


def coefficients(name, n, strengths, options):
    """For a linear method: the c_i with change = sum c_i * delta_i (used to recover one missing branch)."""
    if name == "average": return [s / n for s in strengths]
    if name == "weighted":
        w = [float(x) for x in options.get("weights") or [1.0] * n]; W = sum(w); return [s * wi / W for s, wi in zip(strengths, w)]
    raise ValueError(f"{name} is not linear")


@plugin("average", "Mean of the changes: change = (sum over branches, in recipe order, of strength_i * delta_i) / n, in float64.", linear=True)
def average(base, deltas, strengths, options):
    n = len(deltas); out = {}
    for k in base:
        acc = np.zeros_like(base[k])
        for d, s in zip(deltas, strengths): acc = acc + (d[k] if s == 1 else s * d[k])
        out[k] = acc / n
    return out


@plugin("weighted", "Weighted mean: change = sum of (strength_i * w_i / W) * delta_i, W = sum of w_i (options.weights, e.g. text each branch trained on).",
        linear=True, options={"weights": "list of numbers, one per branch (default all 1)"})
def weighted(base, deltas, strengths, options):
    c = coefficients("weighted", len(deltas), strengths, options); out = {}
    for k in base:
        acc = np.zeros_like(base[k])
        for d, ci in zip(deltas, c): acc = acc + ci * d[k]
        out[k] = acc
    return out


@plugin("trimmed_mean", "Coordinate-wise trimmed mean (Yin et al. 2018): per number, sort the n scaled changes, drop k = floor(n * trim) from each end "
        "(k = 0 below 5 branches), sum the rest in sorted order and divide by n - 2k. Robust to a few outliers; not linear, so a lost branch cannot be recovered.",
        options={"trim": "share dropped at each end (default 0.2)"})
def trimmed_mean(base, deltas, strengths, options):
    n = len(deltas); k = int(n * float(options.get("trim", 0.2))) if n >= 5 else 0; out = {}
    for key in base:
        st = np.sort(np.stack([s * d[key] for d, s in zip(deltas, strengths)]), axis=0)[k:n - k]
        acc = np.zeros_like(base[key])
        for row in st: acc = acc + row
        out[key] = acc / (n - 2 * k)
    return out


def _ties(base, deltas, keep, lam):
    out = {}
    for key in base:
        ts = []
        for d in deltas:
            x = d[key].reshape(-1); a = np.abs(x); kk = max(1, int(len(x) * (1 - keep)))
            thr = np.partition(a, kk - 1)[kk - 1] if len(x) > 1 else a.min()
            ts.append(np.where(a >= thr, x, 0.0))
        S = np.stack(ts); sign = np.sign(S.sum(0)); agree = (np.sign(S) == sign) & (S != 0)
        out[key] = lam * ((S * agree).sum(0) / np.maximum(agree.sum(0), 1)).reshape(base[key].shape)
    return out


@plugin("ties", "TIES-merging (Yadav et al. 2023): per branch keep the largest `keep` share of each tensor's changes (threshold = the k-th smallest |x|), "
        "elect a sign per number from the sum, average the kept changes that agree, times `lambda` (pinned in a recipe).",
        options={"keep": "share of each change kept (default 0.2)", "lambda": "scale of the merged change (default 0.5)"})
def ties(base, deltas, strengths, options):
    return _ties(base, [{k: s * d[k] for k in d} for d, s in zip(deltas, strengths)], float(options.get("keep", 0.2)), float(options.get("lambda", 0.5)))


@plugin("dare_ties", "DARE (Yu et al. 2023) then TIES: drop each number of each change with probability `drop` (NumPy PCG64, seed `seed`, branches and tensors "
        "in recipe and sorted-name order), rescale the rest by 1/(1-drop), then TIES with `keep` and `lambda`.",
        options={"drop": "default 0.7", "seed": "default 0", "keep": "default 0.5", "lambda": "default 0.5"})
def dare_ties(base, deltas, strengths, options):
    p = float(options.get("drop", 0.7)); rng = np.random.Generator(np.random.PCG64(int(options.get("seed", 0)))); ds = []
    for d, s in zip(deltas, strengths):
        ds.append({k: s * d[k] * (rng.random(d[k].shape) >= p) / (1 - p) for k in sorted(d)})
    return _ties(base, ds, float(options.get("keep", 0.5)), float(options.get("lambda", 0.5)))


@plugin("graft_round_robin", "Layer graft: layer l comes whole from branch (l mod n); everything outside the layers (embedding, final norm) is the average.")
def graft_round_robin(base, deltas, strengths, options):
    import re
    n = len(deltas); avg = average(base, deltas, strengths, options); out = {}
    for k in base:
        m = re.match(r"layers\.(\d+)\.", k)
        out[k] = strengths[int(m.group(1)) % n] * deltas[int(m.group(1)) % n][k] if m else avg[k]
    return out


# ── plugins from elsewhere ──
def _data():
    return os.environ.get("PSI_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".userdata")


def plugin_dir(): return os.path.join(_data(), "commons", "plugins")


def approved():
    try: return set(json.load(open(os.path.join(plugin_dir(), "approved.json"), encoding="utf-8")).get("sha256", []))
    except (OSError, ValueError): return set()


def outside():
    """The plugin files waiting in the folder: name, hash, approved or not, and their source (shown before approving)."""
    d = plugin_dir(); out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                b = open(os.path.join(d, f), "rb").read(); h = hashlib.sha256(b).hexdigest()
                out.append({"file": f, "sha256": h, "approved": h in approved(), "source": b.decode("utf-8", "replace")[:20000]})
    return out


def load_outside():
    """Register the approved plugin files (a file's plugins are pinned by the file's own hash)."""
    ok = approved()
    for p in outside():
        if not p["approved"]: continue
        before = set(REGISTRY)
        spec = importlib.util.spec_from_file_location(f"psi_plugin_{p['sha256'][:12]}", os.path.join(plugin_dir(), p["file"]))
        mod = importlib.util.module_from_spec(spec); mod.plugin = plugin; spec.loader.exec_module(mod)
        for name in set(REGISTRY) - before: REGISTRY[name].update(origin=p["file"], code_sha256=p["sha256"])
    return ok


def listing():
    return [{"name": n, "about": r["about"], "linear": r["linear"], "options": r["options"], "code_sha256": r["code_sha256"], "origin": r["origin"]} for n, r in REGISTRY.items()]
