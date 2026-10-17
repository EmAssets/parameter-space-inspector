#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Recipes of public Hugging Face models: merge fine-tunes of one base, and let anyone rebuild the exact bytes.

    pip install numpy torch safetensors
    python hf_recipe.py make  --base Qwen/Qwen2.5-0.5B --branch Qwen/Qwen2.5-0.5B-Instruct --branch Qwen/Qwen2.5-Coder-0.5B-Instruct \
                              --method task_arithmetic --lam 0.5 --name "instruct + coder" --out recipe.json
    python hf_recipe.py build recipe.json --out merged/          # or an hf:// or https:// address of a recipe.json
    python hf_recipe.py check recipe.json                        # every file there and unchanged? (no weights downloaded)

A recipe pins every ingredient by repo, commit and the SHA-256 of each weight file (the hashes Hugging Face itself
publishes for large files), the method and its options, the config and tokenizer files the result carries, and the
hash the result must have. The build:
  - downloads each file at its pinned commit and refuses it unless its SHA-256 matches
  - checks every model has the same tensors and shapes as the base
  - per tensor: change_i = branch_i - base in float64, combined in the recipe's order with the method's written
    arithmetic, added to the base, rounded to float32 and then to the base's dtype (bfloat16: round half to even)
  - hashes the result (tensor names in sorted order, each as its stored little-endian bytes) and compares it
Same recipe, same bytes, on any computer. Nothing in a recipe runs; it is data.

Methods (change_i = branch_i - base, s_i = the branch's strength, n branches):
  average          base + (sum s_i * change_i) / n
  weighted         base + sum (s_i * w_i / sum w) * change_i                (options.weights)
  task_arithmetic  base + lam * sum s_i * change_i                          (options.lam, Ilharco et al. 2023)
  ties             per tensor, per branch keep the largest `keep` share of |s_i * change_i| (threshold = the k-th smallest
                   |x|, k = max(1, int(size * (1 - keep)))), elect a sign per number from the sum, average the kept
                   changes that agree, times lam (Yadav et al. 2023)
"""
import argparse, hashlib, json, os, re, shutil, sys, time, urllib.error, urllib.parse, urllib.request
import numpy as np

H = "https://huggingface.co"
CACHE = os.environ.get("PSI_HF_RECIPE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "psi-hf-recipe")
FAR = 0.2   # a fine-tune moves its base a few hundredths; a branch farther than this is a different model of the same shape, and merging it usually breaks things
DISTANCES = []
CARRY = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]


def say(*a): print(*a, flush=True)


def api(path):
    req = urllib.request.Request(f"{H}/api/{path}", headers={"User-Agent": "psi-hf-recipe/1", **({"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {})})
    for i in range(6):   # Hugging Face rate-limits bursts (429): wait and retry, longer each time
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == 5: raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or 5 * 2 ** i)


def pin(spec):
    """'owner/repo[@commit]' -> {repo, commit, files: {name: {sha256, size}}} for the weight and carried files."""
    repo, _, rev = spec.partition("@")
    commit = rev if re.fullmatch(r"[0-9a-f]{40}", rev or "") else api(f"models/{repo}" + (f"/revision/{rev}" if rev else ""))["sha"]
    tree = api(f"models/{repo}/tree/{commit}"); files = {}
    for f in tree:
        if f["type"] != "file": continue
        if f["path"].endswith(".safetensors") or f["path"] in CARRY:
            files[f["path"]] = {"sha256": (f.get("lfs") or {}).get("oid") or None, "git_oid": f.get("oid"), "size": f.get("size")}
    if not any(p.endswith(".safetensors") for p in files): raise SystemExit(f"{repo}: no safetensors weights at {commit[:10]}")
    meta = api(f"models/{repo}/revision/{commit}"); lic = [t[8:] for t in meta.get("tags", []) if t.startswith("license:")]
    return {"repo": repo, "commit": commit, "licence": lic[0] if lic else None, "files": files}


def fetch(ing, name):
    """A file of an ingredient at its pinned commit, checked: LFS files by SHA-256, small files by git blob hash."""
    f = ing["files"][name]; key = f["sha256"] or f["git_oid"]; p = os.path.join(CACHE, "blobs", key)
    def ok(path):
        b = hashlib.sha256() if f["sha256"] else hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
        with open(path, "rb") as h:
            for c in iter(lambda: h.read(1 << 20), b""): b.update(c)
        return b.hexdigest() == key
    if os.path.exists(p) and ok(p): return p
    os.makedirs(os.path.dirname(p), exist_ok=True); tmp = p + ".part"
    u = f"{H}/{ing['repo']}/resolve/{ing['commit']}/{urllib.parse.quote(name)}"
    req = urllib.request.Request(u, headers={"User-Agent": "psi-hf-recipe/1", **({"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {})})
    t0 = time.time(); n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as o:
        while True:
            b = r.read(1 << 20)
            if not b: break
            o.write(b); n += len(b)
            if f["size"] and n > f["size"]: raise SystemExit(f"{ing['repo']}/{name}: larger than its listed size")
    if not ok(tmp): os.remove(tmp); raise SystemExit(f"{ing['repo']}/{name}: the download does not match its pinned hash")
    os.replace(tmp, p); say(f"  got {ing['repo']}/{name} ({n / 1e6:.0f} MB, {time.time() - t0:.0f} s, hash checked)")
    return p


class Weights:
    """Every tensor of an ingredient, read one at a time (as float64 numpy) from its checked safetensors files."""
    def __init__(self, ing):
        from safetensors import safe_open
        self.where = {}; self.dtype = None; self.files = []
        for name in sorted(n for n in ing["files"] if n.endswith(".safetensors")):
            p = fetch(ing, name); h = safe_open(p, "pt"); self.files.append(h)
            for k in h.keys(): self.where[k] = h
    def keys(self): return sorted(self.where)
    def get(self, k):
        t = self.where[k].get_tensor(k); self.dtype = self.dtype or str(t.dtype); return t.double().numpy(), t.dtype


def combine(method, opts, base, changes, st):
    n = len(changes); acc = np.zeros_like(base)
    if method == "average":
        for c, s in zip(changes, st): acc = acc + (c if s == 1 else s * c)
        return base + acc / n
    if method == "weighted":
        w = [float(x) for x in opts.get("weights") or [1.0] * n]; W = sum(w)
        for c, s, wi in zip(changes, st, w): acc = acc + (s * wi / W) * c
        return base + acc
    if method == "task_arithmetic":
        for c, s in zip(changes, st): acc = acc + (c if s == 1 else s * c)
        return base + float(opts.get("lam", 1.0)) * acc
    if method == "ties":
        keep, lam = float(opts.get("keep", 0.2)), float(opts.get("lam", 1.0)); ts = []
        for c, s in zip(changes, st):
            x = (s * c).reshape(-1); a = np.abs(x); kk = max(1, int(len(x) * (1 - keep)))
            thr = np.partition(a, kk - 1)[kk - 1] if len(x) > 1 else a.min(); ts.append(np.where(a >= thr, x, 0.0))
        S = np.stack(ts); sign = np.sign(S.sum(0)); agree = (np.sign(S) == sign) & (S != 0)
        return base + lam * ((S * agree).sum(0) / np.maximum(agree.sum(0), 1)).reshape(base.shape)
    raise SystemExit(f"unknown method {method}")


def build(r, out_dir=None, progress=True):
    import torch
    from safetensors.torch import save_file
    base = Weights(r["base"]); brs = [Weights(b) for b in r["branches"]]
    keys = base.keys()
    for b, ing in zip(brs, r["branches"]):
        if b.keys() != keys: raise SystemExit(f"{ing['repo']}: not the same tensors as the base ({len(set(b.keys()) ^ set(keys))} differ)")
    st = [float(b.get("strength", 1.0)) for b in r["branches"]]; h = hashlib.sha256(); out = {}; t0 = time.time()
    num = [0.0] * len(brs); den = 0.0   # how far each branch is from the base (relative L2), measured on the way
    for i, k in enumerate(keys):
        bv, dt = base.get(k); ch = []
        for b, ing in zip(brs, r["branches"]):
            v, _ = b.get(k)
            if v.shape != bv.shape: raise SystemExit(f"{ing['repo']}: {k} is {v.shape}, the base has {bv.shape}")
            if not np.isfinite(v).all(): raise SystemExit(f"{ing['repo']}: {k} holds NaN or infinite numbers (a broken upload): refused, it would spoil the whole merge")
            ch.append(v - bv); num[len(ch) - 1] += float((ch[-1] ** 2).sum())
        den += float((bv ** 2).sum())
        m = combine(r["method"]["name"], r["method"].get("options") or {}, bv, ch, st)
        t = torch.from_numpy(m).float().to(dt).contiguous()   # float64 -> float32 -> the base's dtype, each round half to even
        h.update(k.encode()); h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().astype(t.numpy().dtype.newbyteorder("<")).tobytes())
        if out_dir is not None: out[k] = t
        if progress and (i % 50 == 0 or i == len(keys) - 1): say(f"  {i + 1}/{len(keys)} tensors ({time.time() - t0:.0f} s)")
    digest = h.hexdigest()
    global DISTANCES; DISTANCES = [round((n / den) ** 0.5, 4) for n in num]
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True); save_file(out, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt", "psi_recipe_output_sha256": digest})
        carry = r.get("carry_from", "base"); src = r["base"] if carry == "base" else r["branches"][int(carry)]
        for name in CARRY:
            if name in src["files"]: shutil.copyfile(fetch(src, name), os.path.join(out_dir, name))
        json.dump(r, open(os.path.join(out_dir, "recipe.json"), "w"), indent=1)
    return digest


SHAPE = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "tie_word_embeddings", "head_dim"]


def config_of(repo, commit):
    with urllib.request.urlopen(urllib.request.Request(f"{H}/{repo}/resolve/{commit}/config.json", headers={"User-Agent": "psi-hf-recipe/1"}), timeout=30) as r: c = json.loads(r.read(1 << 20))
    return c


def find(base, limit=40):
    """Public models on Hugging Face that can go into a recipe with `base`: fine-tunes of it (and of its fine-tunes
    one step down), with the same shape, full safetensors weights, not quantized, not an adapter."""
    b = api(f"models/{base}"); bc = config_of(base, b["sha"]); want = {k: bc.get(k) for k in SHAPE}
    seen, rows = set(), []
    def cands(of, n):
        q = urllib.parse.urlencode({"filter": f"base_model:finetune:{of}", "sort": "downloads", "direction": "-1", "limit": str(n)})
        return api(f"models?{q}")
    first = cands(base, limit); pool = list(first)
    for m in first[:6]: pool += cands(m["id"], 10)   # fine-tunes of the instruct model are fine-tunes of the base too
    for m in pool:
        rid = m["id"]
        if rid in seen or rid == base: continue
        seen.add(rid); tags = m.get("tags", [])
        row = {"repo": rid, "downloads": m.get("downloads"), "likes": m.get("likes"), "licence": next((t[8:] for t in tags if t.startswith("license:")), None)}
        if "safetensors" not in tags: rows.append({**row, "ok": False, "why": "no safetensors weights"}); continue
        if any(x in rid.lower() for x in ("mlx", "gguf", "4bit", "8bit", "awq", "gptq", "-int4", "-int8")): rows.append({**row, "ok": False, "why": "a quantized or converted copy"}); continue
        try:
            info = api(f"models/{rid}"); c = config_of(rid, info["sha"]); got = {k: c.get(k) for k in SHAPE}
            diff = [k for k in SHAPE if got[k] != want[k] and not (k == "head_dim" and got[k] is None or want[k] is None)]
            q = c.get("quantization_config")
            ok = not diff and not q
            rows.append({**row, "ok": ok, "commit": info["sha"], "why": "same shape (how close it is to the base is measured when the recipe is made)" if ok else ("quantized" if q else "different shape: " + ", ".join(f"{k} {got[k]} vs {want[k]}" for k in diff[:3]))})
        except Exception as e: rows.append({**row, "ok": False, "why": f"unreadable: {str(e)[:80]}"})
    return {"base": base, "commit": b["sha"], "licence": next((t[8:] for t in b.get("tags", []) if t.startswith("license:")), None), "candidates": rows}


def check_json(r):
    rows = []
    for role, ing in [("base", r["base"])] + [("branch", x) for x in r["branches"]]:
        try:
            now = api(f"models/{ing['repo']}/tree/{ing['commit']}"); have = {f["path"]: (f.get("lfs") or {}).get("oid") or None for f in now if f["type"] == "file"}
            miss = [n for n, f in ing["files"].items() if have.get(n, "gone") != f["sha256"]]
        except Exception as e: miss = [f"unreachable: {str(e)[:80]}"]
        cached = all(os.path.exists(os.path.join(CACHE, "blobs", f["sha256"] or f["git_oid"])) for n, f in ing["files"].items() if n.endswith(".safetensors"))
        size = sum(f.get("size") or 0 for n, f in ing["files"].items() if n.endswith(".safetensors"))
        rows.append({"role": role, "repo": ing["repo"], "commit": ing["commit"], "licence": ing.get("licence"), "strength": ing.get("strength"), "ok": not miss, "missing": miss, "here": cached, "bytes": size})
    return {"state": "ready" if all(x["ok"] or x["here"] for x in rows) else "blocked", "rows": rows,
            "download_bytes": sum(x["bytes"] for x in rows if not x["here"])}


def load_recipe(addr):
    if addr.startswith("hf://"):
        o, rp, p = addr[5:].split("/", 2); rev = "main"
        if "@" in rp: rp, rev = rp.split("@", 1)
        addr = f"{H}/{o}/{rp}/resolve/{rev}/{p}"
    if addr.startswith("https://"):
        with urllib.request.urlopen(urllib.request.Request(addr, headers={"User-Agent": "psi-hf-recipe/1"}), timeout=60) as x: return json.loads(x.read(4 << 20))
    return json.load(open(addr, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("cmd", choices=["make", "build", "check", "find"]); ap.add_argument("recipe", nargs="?")
    ap.add_argument("--base"); ap.add_argument("--branch", action="append", default=[]); ap.add_argument("--strength", action="append", type=float, default=[])
    ap.add_argument("--method", default="average"); ap.add_argument("--lam", type=float); ap.add_argument("--keep", type=float); ap.add_argument("--weights")
    ap.add_argument("--carry-from", default="base", help="'base' or the index of the branch whose config and tokenizer the result carries")
    ap.add_argument("--name", default="a recipe"); ap.add_argument("--about", default=""); ap.add_argument("--out")
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError): pass
    if a.cmd == "make":
        opts = {k: v for k, v in (("lam", a.lam), ("keep", a.keep), ("weights", json.loads(a.weights) if a.weights else None)) if v is not None}
        r = {"v": 1, "kind": "psi-hf-recipe", "name": a.name, "about": a.about, "base": pin(a.base), "method": {"name": a.method, "options": opts},
             "branches": [dict(pin(b), strength=(a.strength[i] if i < len(a.strength) else 1.0)) for i, b in enumerate(a.branch)], "carry_from": a.carry_from}
        say(f"pinned {1 + len(r['branches'])} models; building once to fix the expected hash")
        r["expects"] = {"output_sha256": build(r, a.out and os.path.splitext(a.out)[0] + "-model")}
        for b, d in zip(r["branches"], DISTANCES):
            b["distance_from_base"] = d
            if d > FAR: say(f"WARNING {b['repo']} is {d:.2f} of the base's size away from it: not a light fine-tune of this base (more like another model of the same shape). Merging it will likely damage both.")
        json.dump(r, open(a.out or "recipe.json", "w"), indent=1); say(f"recipe written: {a.out or 'recipe.json'}  expects {r['expects']['output_sha256'][:16]}")
        return
    if a.cmd == "find":
        res = find(a.recipe or a.base)
        for c in res["candidates"]: say(f"{'OK ' if c['ok'] else '-- '} {c['repo'][:56]:56} {c.get('licence') or '?':12} {c.get('downloads') or 0:>9} dl  {c['why']}")
        return
    r = load_recipe(a.recipe)
    if a.cmd == "check":
        bad = 0
        for ing in [r["base"]] + r["branches"]:
            now = api(f"models/{ing['repo']}/tree/{ing['commit']}")
            have = {f["path"]: (f.get("lfs") or {}).get("oid") or None for f in now if f["type"] == "file"}
            miss = [n for n, f in ing["files"].items() if have.get(n, "gone") != f["sha256"]]
            say(f"{'ok  ' if not miss else 'GONE'} {ing['repo']}@{ing['commit'][:10]} ({ing.get('licence')})" + (f"  missing or changed: {miss}" if miss else "")); bad += bool(miss)
        sys.exit(1 if bad else 0)
    say(f"recipe {r['name']}: base {r['base']['repo']}, {len(r['branches'])} branches, {r['method']['name']} {json.dumps(r['method'].get('options') or {})}")
    got = build(r, a.out or "merged")
    want = r["expects"]["output_sha256"]
    say(f"built    {got}\nexpected {want}\n" + ("MATCH: the same bytes as the recipe says" if got == want else "DIFFERENT: not the model this recipe describes"))
    sys.exit(0 if got == want else 2)


if __name__ == "__main__":
    main()
