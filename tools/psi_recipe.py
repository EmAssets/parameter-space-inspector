#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Build a PSI recipe without PSI: one file, numpy + safetensors (+ torch to make it write a story).

    pip install numpy safetensors            # cryptography to check signatures, torch for --story
    python psi_recipe.py hf://OWNER/REPO/recipe.json [--out model.safetensors] [--story]

It downloads the seed (stories260K, pinned by commit and SHA-256), every branch the recipe lists (each checked
against its SHA-256), checks that every branch record is the one the recipe pinned, combines them with the method's
written arithmetic (float64, in recipe order, rounded once to float32), and compares the result with the hash the
recipe expects. Same recipe, same bytes, on any computer. Nothing in a recipe runs: it is data.

The merge code here is written from each method's description, not copied from PSI, so a match also shows the
description is enough to rebuild it.
"""
import argparse, base64, hashlib, json, os, re, struct, sys, urllib.parse, urllib.request
import numpy as np

SEED = {"stories260K.bin": ("https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/stories260K.bin",
                            "b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696"),
        "tok512.bin": ("https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/tok512.bin",
                       "037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312")}
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "psi-recipe")


def url(addr):
    if addr.startswith("hf://"):
        rest = addr[5:]; kind = "datasets/" if rest.startswith("datasets/") else ""; rest = rest[len(kind):]
        o, r, p = rest.split("/", 2); rev = "main"
        if "@" in r: r, rev = r.split("@", 1)
        return f"https://huggingface.co/{kind}{o}/{r}/resolve/{rev}/{urllib.parse.quote(p)}"
    m = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+)/(?:blob|raw)/([^/]+)/(.+)", addr)
    if m: return "https://raw.githubusercontent.com/" + "/".join(m.groups())
    if addr.startswith("https://") or os.path.isabs(addr): return addr
    raise SystemExit(f"not an address this reads: {addr}")


def get(addr, sha=None, limit=256 << 20):
    """Bytes from an address; with sha, cached and checked."""
    if sha:
        f = os.path.join(CACHE, sha)
        if os.path.exists(f):
            b = open(f, "rb").read()
            if hashlib.sha256(b).hexdigest() == sha: return b
    u = url(addr)
    if os.path.isabs(u): b = open(u, "rb").read()
    else:
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "psi-recipe/1"}), timeout=60) as r: b = r.read(limit + 1)
        if len(b) > limit: raise SystemExit(f"{addr}: larger than expected")
    if sha:
        h = hashlib.sha256(b).hexdigest()
        if h != sha: raise SystemExit(f"{addr}: wrong bytes (sha256 {h[:12]}, expected {sha[:12]})")
        os.makedirs(CACHE, exist_ok=True); open(os.path.join(CACHE, sha), "wb").write(b)
    return b


def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def vec_sha(v):
    h = hashlib.sha256()
    for k in sorted(v): h.update(k.encode()); h.update(np.ascontiguousarray(v[k], dtype="<f4").tobytes())
    return h.hexdigest()


def read_bin(raw):
    """llama2.c export: 7 int32 header, then float32 weights in a fixed order."""
    dim, hid, L, H, KV, V, T = struct.unpack_from("<7i", raw, 0); V = abs(V); a = np.frombuffer(raw, "<f4", offset=28); o = 0; hd = dim // H; kvd = KV * hd
    def take(n, s):
        nonlocal o; x = a[o:o + n].reshape(s); o += n; return x
    emb = take(V * dim, (V, dim)); ra = take(L * dim, (L, dim)); wq = take(L * dim * dim, (L, dim, dim)); wk = take(L * kvd * dim, (L, kvd, dim)); wv = take(L * kvd * dim, (L, kvd, dim))
    wo = take(L * dim * dim, (L, dim, dim)); rf = take(L * dim, (L, dim)); w1 = take(L * hid * dim, (L, hid, dim)); w2 = take(L * dim * hid, (L, dim, hid)); w3 = take(L * hid * dim, (L, hid, dim)); nm = take(dim, (dim,))
    v = {"tok_embeddings.weight": emb, "norm.weight": nm}
    for l in range(L):
        p = f"layers.{l}."
        v.update({p + "attention_norm.weight": ra[l], p + "attention.wq.weight": wq[l], p + "attention.wk.weight": wk[l], p + "attention.wv.weight": wv[l], p + "attention.wo.weight": wo[l],
                  p + "ffn_norm.weight": rf[l], p + "feed_forward.w1.weight": w1[l], p + "feed_forward.w2.weight": w2[l], p + "feed_forward.w3.weight": w3[l]})
    return {k: np.array(x, dtype=np.float32) for k, x in v.items()}, {"dim": dim, "layers": L, "heads": H, "kv": KV, "vocab": V}


# ── the methods, written from their descriptions ──
def combine(method, opts, base, deltas, st):
    n = len(deltas); out = {}
    for k in base:
        if method == "average":
            acc = np.zeros_like(base[k])
            for d, s in zip(deltas, st): acc = acc + (d[k] if s == 1 else s * d[k])
            ch = acc / n
        elif method == "weighted":
            w = [float(x) for x in opts.get("weights") or [1.0] * n]; W = sum(w); acc = np.zeros_like(base[k])
            for d, s, wi in zip(deltas, st, w): acc = acc + (s * wi / W) * d[k]
            ch = acc
        elif method == "trimmed_mean":
            t = int(n * float(opts.get("trim", 0.2))) if n >= 5 else 0
            rows = np.sort(np.stack([s * d[k] for d, s in zip(deltas, st)]), axis=0)[t:n - t]; acc = np.zeros_like(base[k])
            for r in rows: acc = acc + r
            ch = acc / (n - 2 * t)
        else: raise SystemExit(f"this script builds average, weighted and trimmed_mean; the recipe uses {method} (build it in PSI)")
        out[k] = (base[k] + ch).astype(np.float32)
    return out


def check_sig(rec, sigs):
    try: from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError: return "not checked (pip install cryptography)"
    if not sigs: return "unsigned"
    res = []
    for s in sigs:
        try: Ed25519PublicKey.from_public_bytes(base64.b64decode(s["signer"])).verify(base64.b64decode(s["signature"]), canon(rec)); ok = "valid"
        except Exception: ok = "INVALID"
        res.append(f"{s.get('role', 'signer')} {hashlib.sha256(base64.b64decode(s['signer'])).hexdigest()[:16]} {ok}")
    return ", ".join(res)


def story(v, tok_raw, info, n=120, seed=7):
    import torch, torch.nn.functional as F
    off = 4; vocab, scores = [], []
    while off < len(tok_raw):
        s, ln = struct.unpack_from("<fi", tok_raw, off); off += 8; vocab.append(tok_raw[off:off + ln]); off += ln
    t = {k: torch.from_numpy(x) for k, x in v.items()}; d, L, H, KV = info["dim"], info["layers"], info["heads"], info["kv"]; hd = d // H
    f = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd)); ang = torch.outer(torch.arange(512).float(), f)
    def fwd(ids):
        B, T = ids.shape; c, s_ = ang.cos()[:T][None, :, None, :], ang.sin()[:T][None, :, None, :]
        rope = lambda x: torch.stack([x[..., 0::2] * c - x[..., 1::2] * s_, x[..., 0::2] * s_ + x[..., 1::2] * c], -1).flatten(-2)
        rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        x = t["tok_embeddings.weight"][ids]
        for l in range(L):
            p = f"layers.{l}."; h = rms(x, t[p + "attention_norm.weight"])
            q = rope((h @ t[p + "attention.wq.weight"].T).view(B, T, H, hd)); k = rope((h @ t[p + "attention.wk.weight"].T).view(B, T, KV, hd)).repeat_interleave(H // KV, 2)
            vv = (h @ t[p + "attention.wv.weight"].T).view(B, T, KV, hd).repeat_interleave(H // KV, 2)
            x = x + F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), vv.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, d) @ t[p + "attention.wo.weight"].T
            h2 = rms(x, t[p + "ffn_norm.weight"]); x = x + (F.silu(h2 @ t[p + "feed_forward.w1.weight"].T) * (h2 @ t[p + "feed_forward.w3.weight"].T)) @ t[p + "feed_forward.w2.weight"].T
        return rms(x, t["norm.weight"]) @ t["tok_embeddings.weight"].T
    g = torch.Generator().manual_seed(seed); ids = torch.tensor([[1]])
    with torch.no_grad():
        for _ in range(n): ids = torch.cat([ids, torch.multinomial(F.softmax(fwd(ids[:, -256:])[0, -1] / 0.8, -1), 1, generator=g)[None]], 1)
    out = b""
    for i in ids[0, 1:].tolist():
        m = re.fullmatch(rb"<0x([0-9A-F]{2})>", vocab[i]); out += bytes([int(m.group(1), 16)]) if m else vocab[i]
    return out.decode("utf-8", "replace").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("recipe"); ap.add_argument("--out", default="model.safetensors"); ap.add_argument("--story", action="store_true")
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError): pass
    doc = json.loads(get(a.recipe, limit=4 << 20)); r = doc["recipe"]
    print(f"recipe   {r.get('name')}  ({len(r['branches'])} branches, {r['method']['plugin']})\nsigned   {check_sig(r, doc.get('signatures'))}")
    if r.get("line") != "stories260k": raise SystemExit("this script builds the stories260k line")
    raw = get(SEED["stories260K.bin"][0], SEED["stories260K.bin"][1]); tok = get(SEED["tok512.bin"][0], SEED["tok512.bin"][1])
    base, info = read_bin(raw); print(f"seed     stories260K, {sum(x.size for x in base.values()):,} numbers, sha256 checked")
    if vec_sha(base) != r["base"]: raise SystemExit("this recipe builds on another base than the seed (a built model); use PSI")
    by = {e["record"]["delta_sha256"]: e for e in doc.get("branches", [])}; deltas, st = [], []
    from safetensors.numpy import load as st_load
    for b in r["branches"]:
        e = by.get(b["delta_sha256"]); rec = e and e["record"]
        if not e: raise SystemExit(f"the recipe does not carry the record of {b['delta_sha256'][:12]}")
        if hashlib.sha256(canon(rec)).hexdigest() != b["record_sha256"]: raise SystemExit(f"{rec.get('name')}: its record was changed after the recipe pinned it")
        data = None
        for addr in [x for x in e.get("addrs", []) if not x.startswith("store://")]:
            try: data = get(addr, b["delta_sha256"], limit=(rec.get("delta_size") or 64 << 20) + 1); break
            except (SystemExit, OSError) as x: print(f"         {addr}: {x}")
        if data is None: raise SystemExit(f"{rec.get('name')}: no address gave its file")
        deltas.append({k: v.astype(np.float64) for k, v in st_load(data).items()}); st.append(float(b.get("strength", 1.0)))
        print(f"branch   {rec.get('name')} by {(rec.get('contributor') or {}).get('name')}  ({(rec.get('source') or {}).get('licence')})  sha256 checked, {check_sig(rec, e.get('signatures'))}")
    out = combine(r["method"]["plugin"], r["method"].get("options") or {}, {k: v.astype(np.float64) for k, v in base.items()}, deltas, st)
    got, want = vec_sha(out), r["expects"]["output_sha256"]
    print(f"built    {got}\nexpected {want}\n{'MATCH: the same bytes as the recipe says' if got == want else 'DIFFERENT: this is not the model the recipe describes'}")
    from safetensors.numpy import save_file
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, a.out); print(f"saved    {a.out}")
    if a.story: print("\nit writes:\n" + story(out, tok, info))
    sys.exit(0 if got == want else 2)


if __name__ == "__main__":
    main()
