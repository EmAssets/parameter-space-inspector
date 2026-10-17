# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""The commons: branches, recipes and built models for a seed line (src/seedline.py).

  branch   the change one person made to a base by training on their text, as safetensors named by its SHA-256,
           with a record (line, base, what was trained on and how, the trace that lets anyone replay it, scores
           before and after on the line's evaluation suite, licence) signed by its contributor, or not signed at all
  recipe   a base, a list of branches (each pinned by record hash, with a strength), a merge plugin pinned by its code
           hash, its options, and the output it should give; signed by its curator, or not. The branch entries ride
           inside the recipe file, so one file is enough to check and build it
  model    a recipe's output, rebuilt here, with a record of how (strict, recovered, partial) and what was checked

Nothing here trusts an address or a name: files are kept only when their SHA-256 matches, records only when their
hash matches the recipe, signatures are checked and shown, and a missing, changed or unknown piece is named.

CLI (one JSON line per step "PSI-PROGRESS {...}", the result "PSI-DONE {...}", or "PSI-ERROR why"):
  python recipes.py line-status | line-fetch
  python recipes.py train --text FILE --name NAME [--base MODEL_DIR] [--steps N] [--licence L] [--subject S] [--about A]
  python recipes.py branches | recipes | models | plugins
  python recipes.py make --name N --branches SHA,SHA.. [--plugin P] [--options JSON] [--strengths JSON] [--about A]
  python recipes.py import ADDRESS        (a recipe.json anywhere: hf://, https://, GitHub, a local file)
  python recipes.py check RECIPE_ID
  python recipes.py build RECIPE_ID [--mode strict|recover|partial|download] [--gguf]
  python recipes.py search --branches SHA,.. [--plugin P]       python recipes.py bisect RECIPE_ID
  python recipes.py use RECIPE_ID          (an opt-in, signed record that you used a recipe; it ranks recipes)
"""
import argparse, datetime as dt, hashlib, json, os, re, sys, time
import numpy as np
import torch
from safetensors.torch import save_file, load_file

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hub, keys, merge_plugins as MP, seedline as SL

DATA = SL.DATA
C = os.path.join(DATA, "commons")
DIRS = {k: os.path.join(C, k) for k in ("branches", "recipes", "models", "traces", "uses", "private")}
LINE = "stories260k"
SUITE_FILE = os.path.join(HERE, "lines", "stories260k-eval.json")
MAX_DELTA = 64 << 20   # a stories260K change is ~1 MB; anything far larger is not one


def say(_what, **o): print(f"PSI-{_what} {json.dumps(o, ensure_ascii=False)}", flush=True)
def now(): return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def sha_json(o): return hashlib.sha256(keys.canon(o)).hexdigest()
def mk(d): os.makedirs(d, exist_ok=True); return d
def slug(s): return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:40] or "x"


def me():
    """Who this process acts as: the profile name the app passes (PSI_MAKER), the key it signs with, or nobody."""
    w = keys.who(); name = (os.environ.get("PSI_MAKER") or "").strip()[:60] or "anonymous"
    if w == "em": return {"name": "The Emergence Machine", "key": None, "as": "em"}
    return {"name": name, "key": w or None, "as": "key" if w else "private", "profile_sha256": os.environ.get("PSI_PROFILE_SHA") or None}


def signed(rec, role):
    s, pub = keys.sign(rec)
    return [{"signer": pub, "signature": s, "role": role}] if s else []


def cosign_em(entry, role="emergence-machine"):
    """Add The Emergence Machine's signature to an entry (only when asked, and only where its key is)."""
    s, pub = keys.sign(entry["record"] if "record" in entry else entry["recipe"], as_="em")
    entry.setdefault("signatures", []).append({"signer": pub, "signature": s, "role": role}); return entry


# ── the evaluation suite ──
_SUITE = {}
def suite(seed):
    if seed.sha in _SUITE: return _SUITE[seed.sha]
    s = json.load(open(SUITE_FILE, encoding="utf-8")); parts = {}
    for k, p in s["parts"].items():
        if "text" in p: parts[k] = torch.tensor(seed.tok.encode(p["text"]))
        else:
            f = os.path.join(C, "cache", f"suite-{s['name']}-{k}-{seed.sha[:16]}.safetensors")
            if os.path.exists(f): parts[k] = load_file(f)["ids"]; continue
            torch.set_num_threads(1); a, b = p["generated"]["seeds"]; parts[k] = torch.cat([seed.sample(seed.vec, 256, 0.8, x) for x in range(a, b + 1)])
            mk(os.path.dirname(f)); save_file({"ids": parts[k].contiguous()}, f)
    _SUITE[seed.sha] = (s["name"], parts); return _SUITE[seed.sha]


def evaluate(seed, vec):
    name, parts = suite(seed); per = {k: round(seed.loss(vec, ids), 4) for k, ids in parts.items()}
    return {"suite": name, "mean": round(sum(per.values()) / len(per), 4), "parts": per}


# ── loading weights ──
def load_seed(): return SL.Seed(LINE)


def model_vec(model_dir):
    f = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(f): raise SystemExit(f"no model.safetensors in {model_dir}")
    return {k: v.float() for k, v in load_file(f).items()}


def base_for(seed, model_dir=None):
    return (seed.vec, "the seed") if not model_dir else (model_vec(model_dir), os.path.basename(model_dir.rstrip("/\\")))


# ── branches ──
def train(a):
    seed = load_seed(); base, base_name = base_for(seed, a.base)
    text = open(a.text, encoding="utf-8", errors="replace").read()
    if a.text.endswith(".json"):
        try: text = json.loads(text).get("text", text)
        except ValueError: pass
    ids = seed.tok.encode(text)
    if len(ids) < 2000: raise SystemExit("PSI-ERROR the text is too short to learn from (under 2,000 pieces, about 1,500 words)")
    say("PROGRESS", stage="measuring the start", pieces=len(ids))
    held = max(256, len(ids) // 10); train_ids, own_test = ids[:-held], ids[-held:]
    before = evaluate(seed, base); own_before = round(seed.loss(base, own_test), 4)
    cfg = {"steps": int(a.steps)}; t0 = time.time(); tmp = mk(os.path.join(DIRS["traces"], f"tmp-{os.getpid()}"))
    last = [0]
    def step(s, l):
        if s - last[0] >= 5 or s == cfg["steps"]: last[0] = s; say("PROGRESS", stage="training", step=s, of=cfg["steps"], loss=round(l, 3))
    say("PROGRESS", stage="preparing", note="writing the parent's own stories to rehearse (once per base)")
    vec, trace, salt = SL.train_branch(seed, base, train_ids, cfg, tmp, step)
    secs = round(time.time() - t0, 1)
    say("PROGRESS", stage="measuring the branch")
    after = evaluate(seed, vec); own_after = round(seed.loss(vec, own_test), 4)
    delta = {k: (vec[k] - base[k]).contiguous() for k in sorted(base)}
    f = os.path.join(mk(hub.store_dir()), "tmp.safetensors"); save_file(delta, f); h = hub.sha_file(f); final = hub.store_path(h); move(f, final)
    tdir = os.path.join(DIRS["traces"], h[:16])
    if os.path.exists(tdir): import shutil; shutil.rmtree(tdir)
    move(tmp, tdir); json.dump(trace, open(os.path.join(tdir, "trace.json"), "w"), indent=1)
    # the salt and the text are private: kept here, given only to someone the contributor asks to replay it
    mk(DIRS["private"]); json.dump({"salt": salt, "text_file": os.path.abspath(a.text), "held_out_pieces": held}, open(os.path.join(DIRS["private"], f"{h[:16]}.json"), "w"))
    who = me()
    rec = {"v": 3, "kind": "branch", "line": LINE, "base": SL.vec_sha(base), "base_name": base_name, "delta_sha256": h, "delta_size": os.path.getsize(final),
           "name": str(a.name)[:80], "about": str(a.about or "")[:500], "source": {"licence": str(a.licence)[:120], "subject": str(a.subject or "")[:60], "pieces": len(ids), "url": a.url or None},
           "trained": {"steps": cfg["steps"], "seconds": secs, "device": "cpu", "threads": trace["threads"], "torch": torch.__version__},
           "trace": {"sha256": sha_json(trace), "segments": len(trace["checkpoints"]) - 1, "data_sha256": trace["data"]["sha256"]},
           "eval": {"before": before, "after": after, "own_text": {"before": own_before, "after": own_after, "pieces": held}},
           "contributor": {k: v for k, v in who.items() if k != "as"}, "created": now()}
    entry = {"record": rec, "signatures": signed(rec, "contributor"), "addrs": [f"store://{h}"]}
    json.dump(entry, open(os.path.join(mk(DIRS["branches"]), f"{h[:16]}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("DONE", branch=h[:16], name=rec["name"], seconds=secs, before=before["mean"], after=after["mean"], own_before=own_before, own_after=own_after, signed=bool(entry["signatures"]))


def move(src, dst):
    """Rename, retrying while Windows (a virus scanner or the indexer holding a new file) refuses; copy as a last resort."""
    import shutil
    for i in range(20):
        try: os.replace(src, dst); return
        except PermissionError: time.sleep(0.25 * (i + 1))
    shutil.copytree(src, dst); shutil.rmtree(src, ignore_errors=True)


def listing(kind):
    d = DIRS[kind]; out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                try: out.append(json.load(open(os.path.join(d, f), encoding="utf-8")))
                except (OSError, ValueError): pass
    return out


def branch_entry(ref):
    for e in listing("branches"):
        if e["record"]["delta_sha256"].startswith(ref): return e
    raise SystemExit(f"PSI-ERROR no branch {ref} here")


# ── recipes ──
def make(a):
    seed = load_seed(); refs = [x for x in a.branches.split(",") if x]
    entries = [branch_entry(r) for r in refs]
    bases = {e["record"]["base"] for e in entries}
    if len(bases) != 1: raise SystemExit("PSI-ERROR these branches were trained from different bases: a recipe combines branches of one base")
    plugin = a.plugin or "average"
    if plugin not in MP.REGISTRY: raise SystemExit(f"PSI-ERROR no merge plugin {plugin}")
    opts = json.loads(a.options or "{}"); st = json.loads(a.strengths or "null") or [1.0] * len(entries)
    entries = sorted(zip(entries, st), key=lambda x: x[0]["record"]["delta_sha256"]); st = [s for _, s in entries]; entries = [e for e, _ in entries]
    who = me()
    r = {"v": 3, "kind": "recipe", "line": LINE, "name": str(a.name)[:80], "about": str(a.about or "")[:1000], "base": bases.pop(),
         "branches": [{"delta_sha256": e["record"]["delta_sha256"], "record_sha256": sha_json(e["record"]), "strength": float(s)} for e, s in zip(entries, st)],
         "method": {"plugin": plugin, "code_sha256": MP.REGISTRY[plugin]["code_sha256"], "options": opts}, "curator": {k: v for k, v in who.items() if k != "as"}, "created": now()}
    base, _ = base_by_sha(seed, r["base"])
    out = combine(base, [load_delta(e) for e in entries], st, plugin, opts)
    r["expects"] = {"output_sha256": SL.vec_sha(out)}
    out_file = save_model(out, r, "strict", [])
    r["expects"]["output_file_sha256"] = hub.sha_file(os.path.join(out_file, "model.safetensors"))
    doc = {"recipe": r, "signatures": signed(r, "curator"), "branches": entries, "output_addrs": [f"store://{r['expects']['output_file_sha256']}"]}
    import shutil; shutil.copyfile(os.path.join(out_file, "model.safetensors"), hub.store_path(r["expects"]["output_file_sha256"]))
    rid = sha_json(r)[:16]; json.dump(doc, open(os.path.join(mk(DIRS["recipes"]), f"{rid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rewrite_model_record(out_file, rid)
    say("DONE", recipe=rid, name=r["name"], branches=len(entries), plugin=plugin, output=r["expects"]["output_sha256"][:16], signed=bool(doc["signatures"]))


def base_by_sha(seed, sha):
    if seed.sha == sha: return seed.vec, "the seed"
    for m in listing_models():
        if m["record"]["output_sha256"] == sha: return model_vec(m["dir"]), m["record"]["recipe_name"]
    raise SystemExit(f"PSI-ERROR the base {sha[:12]} is not here: build the recipe that made it first")


def load_delta(entry):
    p = hub.store_path(entry["record"]["delta_sha256"])
    return {k: v.double().numpy() for k, v in load_file(p).items()}


def combine(base, deltas, strengths, plugin, opts):
    b64 = {k: base[k].double().numpy() for k in sorted(base)}
    if isinstance(deltas[0], dict) and torch.is_tensor(next(iter(deltas[0].values()))): deltas = [{k: v.double().numpy() for k, v in d.items()} for d in deltas]
    ch = MP.REGISTRY[plugin]["fn"](b64, deltas, list(strengths), opts)
    return {k: torch.from_numpy((b64[k] + ch[k]).astype(np.float32)) for k in sorted(base)}


def recipe_doc(rid):
    f = os.path.join(DIRS["recipes"], f"{rid}.json")
    if not os.path.exists(f):
        hits = [x for x in os.listdir(DIRS["recipes"]) if x.startswith(rid)] if os.path.isdir(DIRS["recipes"]) else []
        if len(hits) != 1: raise SystemExit(f"PSI-ERROR no recipe {rid} here")
        f = os.path.join(DIRS["recipes"], hits[0])
    return json.load(open(f, encoding="utf-8"))


def counted(addr):
    """Hugging Face counts a download when its query file is fetched (config.json, for a repo of no known library), never
    the weights. An import reads the recipe repo's config.json once, so each import counts as one download; a check or a
    build does not. A repo without one (published before) is simply not counted."""
    m = re.fullmatch(r"hf://([\w.-]+/[\w.-]+)(@[\w.-]+)?/recipe\.json", str(addr))
    if not m: return
    try: hub.get_json(f"hf://{m.group(1)}{m.group(2) or ''}/config.json", max_bytes=64 << 10, timeout=15)
    except Exception: pass


def import_recipe(addr):
    """Read a recipe from anywhere; keep it only if it is well formed (it is data: nothing in it runs)."""
    say("PROGRESS", stage="reading", addr=str(addr)[:200])
    doc = hub.get_json(addr)
    if is_hf(doc):   # the second path: public Hugging Face models; only pinned repos, commits and hashes, nothing runs
        if not isinstance(doc.get("branches"), list) or not doc["branches"] or len(doc["branches"]) > 50: raise SystemExit("PSI-ERROR that recipe lists no models, or more than 50")
        for ing in [doc.get("base") or {}] + doc["branches"]:
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", str(ing.get("repo", ""))) or not re.fullmatch(r"[0-9a-f]{40}", str(ing.get("commit", ""))): raise SystemExit("PSI-ERROR every model must be a repo pinned to a full commit")
        doc.setdefault("imported_from", str(addr)[:500]); doc.setdefault("imported_at", now()); rid = sha_json(hf_recipe_of(doc))[:16]
        json.dump(doc, open(os.path.join(mk(DIRS["recipes"]), f"{rid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        say("DONE", recipe=rid, name=str(doc.get("name"))[:80], branches=len(doc["branches"]), kind="hf"); return
    r = doc.get("recipe") if isinstance(doc, dict) else None
    if not isinstance(r, dict) or r.get("kind") != "recipe" or not isinstance(r.get("branches"), list): raise SystemExit("PSI-ERROR that file is not a PSI recipe")
    if r.get("line") not in SL.LINES: raise SystemExit(f"PSI-ERROR the recipe is for a line PSI does not know: {str(r.get('line'))[:40]}")
    if len(r["branches"]) > 500: raise SystemExit("PSI-ERROR more than 500 branches: refused")
    for b in r["branches"]:
        if not re.fullmatch(r"[0-9a-f]{64}", str(b.get("delta_sha256", ""))): raise SystemExit("PSI-ERROR a branch is not named by a SHA-256")
    doc.setdefault("imported_from", str(addr)[:500]); doc.setdefault("imported_at", now())
    counted(addr)
    if str(addr).startswith(("hf://", "https://")):   # a recipe found at an address: its files are probably beside it
        stem = str(addr).rsplit("/", 1)[0]
        for e in doc.get("branches", []):
            e.setdefault("addrs", []); extra = f"{stem}/branches/{e['record']['delta_sha256']}.safetensors"
            if extra not in e["addrs"]: e["addrs"].append(extra)
        out = f"{stem}/model.safetensors"
        doc.setdefault("output_addrs", []);
        if out not in doc["output_addrs"]: doc["output_addrs"].append(out)
    rid = sha_json(r)[:16]; json.dump(doc, open(os.path.join(mk(DIRS["recipes"]), f"{rid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("DONE", recipe=rid, name=str(r.get("name"))[:80], branches=len(r["branches"]))


def check(rid, quiet=False):
    """Every piece of a recipe, checked before anything large is downloaded."""
    doc = recipe_doc(rid)
    if is_hf(doc): return hf_check(rid, doc, quiet)
    r = doc["recipe"]; rows = []; by = {e["record"]["delta_sha256"]: e for e in doc.get("branches", [])}
    sig = keys.signatures(r, doc.get("signatures"))
    plug = MP.REGISTRY.get(r["method"]["plugin"])
    method = {"plugin": r["method"]["plugin"], "known": bool(plug), "same_code": bool(plug) and plug["code_sha256"] == r["method"]["code_sha256"], "linear": bool(plug and plug["linear"]),
              "options": r["method"].get("options")}
    line = SL.line_status(r["line"])
    seed_ok = line["ready"]; base_ok = None
    if seed_ok:
        seed = load_seed()
        try: base_by_sha(seed, r["base"]); base_ok = True
        except SystemExit: base_ok = False
    total = 0
    for i, b in enumerate(r["branches"]):
        if not quiet: say("PROGRESS", stage="checking", i=i + 1, of=len(r["branches"]))
        e = by.get(b["delta_sha256"]); row = {"delta_sha256": b["delta_sha256"], "strength": b.get("strength", 1.0)}
        if not e: rows.append({**row, "status": "missing", "why": "the recipe does not carry this branch's record"}); continue
        rec = e["record"]; row.update(name=rec.get("name"), contributor=(rec.get("contributor") or {}).get("name"), licence=(rec.get("source") or {}).get("licence"),
                                      subject=(rec.get("source") or {}).get("subject"), size=rec.get("delta_size"), eval_after=((rec.get("eval") or {}).get("after") or {}).get("mean"))
        problems = []
        if sha_json(rec) != b["record_sha256"]: problems.append("its record was changed after the recipe pinned it")
        if rec.get("base") != r["base"]: problems.append("trained from another base")
        if not row["licence"]: problems.append("no licence stated")
        if (rec.get("delta_size") or 0) > MAX_DELTA: problems.append("larger than a branch of this line can be")
        row["signatures"] = keys.signatures(rec, e.get("signatures"))
        if any(not s["valid"] for s in row["signatures"]): problems.append("a signature does not verify")
        sp = hub.store_path(b["delta_sha256"]); have = os.path.exists(sp) and hub.sha_file(sp) == b["delta_sha256"]
        if os.path.exists(sp) and not have:   # a copy here with the wrong bytes: set aside, never used
            os.replace(sp, sp + ".bad"); row["note"] = "the copy on this computer had the wrong bytes: set aside (.bad) and looked for elsewhere"
        found = [] if have else [p for p in (hub.probe(x) for x in e.get("addrs", []) if not str(x).startswith("store://")) if p.get("ok") and (not rec.get("delta_size") or p.get("size") == rec["delta_size"])]
        row["in_store"] = have; row["where"] = "this computer" if have else (found[0]["where"] if found else None); row["hosts"] = len(found) + (1 if have else 0)
        if problems: row.update(status="refused", why="; ".join(problems))
        elif have or found: row.update(status="ok"); total += 0 if have else (rec.get("delta_size") or 0)
        else: row.update(status="missing", why="not found at any address it lists", tried=e.get("addrs", []))
        rows.append(row)
    out_have = os.path.exists(hub.store_path(r["expects"].get("output_file_sha256", "0" * 64)))
    out_found = out_have or any(p.get("ok") for p in (hub.probe(x) for x in doc.get("output_addrs", []) if not str(x).startswith("store://")))
    missing = [x for x in rows if x["status"] == "missing"]; refused = [x for x in rows if x["status"] == "refused"]
    subjects = {}
    for x in rows:
        if x.get("subject"): subjects.setdefault(x["subject"], []).append(x["status"] == "ok")
    only = sorted(s for s, v in subjects.items() if len(v) == 1)
    lost = sorted(s for s, v in subjects.items() if not any(v))
    if not seed_ok: state, why = "blocked", "the line's seed files are not here yet (fetch the line)"
    elif not base_ok: state, why = "blocked", "the base this recipe builds on is not here"
    elif not method["known"]: state, why = "blocked", f"no merge plugin called {method['plugin']} here (a plugin from elsewhere needs your approval first)"
    elif not method["same_code"]: state, why = "blocked", "the merge plugin here is a different version from the one the recipe pins"
    elif refused: state, why = "blocked", f"{len(refused)} branch(es) refused: " + "; ".join(f"{x.get('name') or x['delta_sha256'][:12]}: {x['why']}" for x in refused[:3])
    elif not missing: state, why = "ready", "every piece is here or downloadable"
    elif len(missing) == 1 and method["linear"] and out_found: state, why = "recoverable", "one branch is missing; the method is linear and the built model is published, so it can be rebuilt exactly"
    else: state, why = "partial", f"{len(missing)} of {len(rows)} branches missing: build without them (a different recipe, with a new hash)"
    res = {"recipe": rid, "name": r.get("name"), "about": r.get("about"), "line": r["line"], "state": state, "why": why, "signatures": sig, "curator": r.get("curator"),
           "method": method, "branches": rows, "download_bytes": total, "output_published": out_found, "output_here": out_have,
           "only_source": only, "lost_subjects": lost, "imported_from": doc.get("imported_from"), "seed_ok": seed_ok}
    if not quiet: say("DONE", **res)
    return res


def fetch_branches(doc, rows):
    by = {e["record"]["delta_sha256"]: e for e in doc.get("branches", [])}
    for i, x in enumerate(rows):
        if x["status"] != "ok" or x.get("in_store"): continue
        say("PROGRESS", stage="downloading", i=i + 1, of=len(rows), name=x.get("name"))
        e = by[x["delta_sha256"]]; got = hub.fetch([a for a in e.get("addrs", []) if not a.startswith("store://")], x["delta_sha256"], hub.store_path(x["delta_sha256"]), (e["record"].get("delta_size") or MAX_DELTA) + 1)
        e.setdefault("fetched_from", got["from"])


def save_model(vec, r, mode, notes, rid=None):
    d = os.path.join(mk(DIRS["models"]), f"{slug(r.get('name'))}-{SL.vec_sha(vec)[:12]}"); mk(d)
    save_file({k: v.contiguous() for k, v in vec.items()}, os.path.join(d, "model.safetensors"))
    rec = {"v": 3, "kind": "model", "line": r["line"], "recipe_name": r.get("name"), "recipe": rid, "recipe_sha256": sha_json(r), "base": r["base"], "output_sha256": SL.vec_sha(vec),
           "file_sha256": hub.sha_file(os.path.join(d, "model.safetensors")), "mode": mode, "notes": notes, "matches_recipe": SL.vec_sha(vec) == (r.get("expects") or {}).get("output_sha256"),
           "built_by": {k: v for k, v in me().items() if k != "as"}, "built": now()}
    json.dump({"record": rec, "signatures": signed(rec, "builder")}, open(os.path.join(d, "record.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d


def rewrite_model_record(d, rid):
    f = os.path.join(d, "record.json"); e = json.load(open(f, encoding="utf-8")); e["record"]["recipe"] = rid; e["record"]["matches_recipe"] = True
    e["signatures"] = signed(e["record"], "builder"); json.dump(e, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def build(rid, mode="strict", gguf=False):
    if is_hf(recipe_doc(rid)):
        res = check(rid, quiet=True)
        if res["state"] == "blocked": raise SystemExit(f"PSI-ERROR {res['why']}")
        return hf_build(rid, recipe_doc(rid))
    res = check(rid, quiet=True); doc = recipe_doc(rid); r = doc["recipe"]; seed = load_seed(); t0 = time.time()
    if res["state"] == "blocked": raise SystemExit(f"PSI-ERROR {res['why']}")
    notes = []
    if mode == "download":
        if not res["output_published"]: raise SystemExit("PSI-ERROR this recipe has no published model to download")
        h = r["expects"]["output_file_sha256"]; p = hub.store_path(h)
        if not os.path.exists(p): say("PROGRESS", stage="downloading the model"); hub.fetch([a for a in doc.get("output_addrs", []) if not a.startswith("store://")], h, p, 64 << 20)
        vec = {k: v.float() for k, v in load_file(p).items()}; notes.append("downloaded the published model; its bytes match the recipe's expected file hash")
    else:
        rows = res["branches"]; missing = [x for x in rows if x["status"] != "ok"]
        if missing and mode == "strict": raise SystemExit("PSI-ERROR " + "; ".join(f"the file for {x.get('contributor') or x.get('name') or '?'} is missing (sha256 {x['delta_sha256'][:12]}…)" for x in missing))
        fetch_branches(doc, rows)
        by = {e["record"]["delta_sha256"]: e for e in doc["branches"]}
        base, _ = base_by_sha(seed, r["base"])
        if mode == "recover":
            if res["state"] != "recoverable": raise SystemExit(f"PSI-ERROR not recoverable: {res['why']}")
            m = missing[0]; h = r["expects"]["output_file_sha256"]; p = hub.store_path(h)
            if not os.path.exists(p): say("PROGRESS", stage="downloading the published model"); hub.fetch([a for a in doc.get("output_addrs", []) if not a.startswith("store://")], h, p, 64 << 20)
            out = {k: v.double().numpy() for k, v in load_file(p).items()}; b64 = {k: base[k].double().numpy() for k in base}
            st = [b.get("strength", 1.0) for b in r["branches"]]; c = MP.coefficients(r["method"]["plugin"], len(st), st, r["method"].get("options") or {})
            idx = [b["delta_sha256"] for b in r["branches"]].index(m["delta_sha256"])
            others = [(c[i], load_delta(by[b["delta_sha256"]])) for i, b in enumerate(r["branches"]) if i != idx]
            rec_d = {k: (out[k] - b64[k] - sum(ci * d[k] for ci, d in others)) / c[idx] for k in b64}
            deltas = [load_delta(by[b["delta_sha256"]]) if i != idx else rec_d for i, b in enumerate(r["branches"])]
            vec = combine(base, deltas, st, r["method"]["plugin"], r["method"].get("options") or {})
            notes.append(f"recovered the missing branch {m['delta_sha256'][:12]} from the published model and the other {len(others)}; rebuilt")
        else:
            keep = [b for b in r["branches"] if b["delta_sha256"] not in {x["delta_sha256"] for x in missing}]
            if not keep: raise SystemExit("PSI-ERROR no branch of this recipe is available")
            if missing: notes.append(f"built without {len(missing)} missing branch(es): " + ", ".join((x.get("name") or x["delta_sha256"][:12]) for x in missing) + ". This is a different recipe, with a new hash")
            if res["lost_subjects"]: notes.append("subjects with no source left: " + ", ".join(res["lost_subjects"]))
            say("PROGRESS", stage="combining", branches=len(keep), plugin=r["method"]["plugin"])
            vec = combine(base, [load_delta(by[b["delta_sha256"]]) for b in keep], [b.get("strength", 1.0) for b in keep], r["method"]["plugin"], r["method"].get("options") or {})
    say("PROGRESS", stage="measuring")
    got = SL.vec_sha(vec); ok = got == r["expects"]["output_sha256"]
    d = save_model(vec, r, mode, notes, rid)
    ev = evaluate(seed, vec); e = json.load(open(os.path.join(d, "record.json"), encoding="utf-8")); e["record"]["eval"] = ev; e["signatures"] = signed(e["record"], "builder")
    json.dump(e, open(os.path.join(d, "record.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if gguf:
        say("PROGRESS", stage="writing a GGUF"); import commons_gguf; commons_gguf.export(d, seed)
    gg = [f for f in os.listdir(d) if f.endswith(".gguf")]
    say("DONE", recipe=rid, mode=mode, dir=d.replace("\\", "/"), gguf=os.path.join(d, gg[0]).replace("\\", "/") if gg else None, output=got[:16], matches=ok, expected=r["expects"]["output_sha256"][:16], notes=notes, eval=ev, seconds=round(time.time() - t0, 1),
        story=seed.tok.decode(seed.sample(vec, 80, 0.8, 7).tolist()))


def listing_models():
    out = []
    if os.path.isdir(DIRS["models"]):
        for d in sorted(os.listdir(DIRS["models"])):
            f = os.path.join(DIRS["models"], d, "record.json")
            if os.path.exists(f):
                e = json.load(open(f, encoding="utf-8")); e["dir"] = os.path.join(DIRS["models"], d)
                gg = [x for x in os.listdir(e["dir"]) if x.endswith(".gguf")]
                if gg: e["gguf"] = os.path.join(e["dir"], gg[0]).replace("\\", "/")
                out.append(e)
            elif d.endswith(".link.json"):   # a model of the Hugging Face path, kept in the models folder
                e = json.load(open(os.path.join(DIRS["models"], d), encoding="utf-8"))
                if os.path.isdir(e.get("dir", "")): e["link"] = d; out.append(e)
    return out


# ── recipe tools: search (which branches help) and bisect (which one hurts) ──
def search(a):
    seed = load_seed(); entries = [branch_entry(x) for x in a.branches.split(",") if x]; plugin = a.plugin or "average"
    base, _ = base_by_sha(seed, entries[0]["record"]["base"]); ds = [load_delta(e) for e in entries]
    score = lambda idx: evaluate(seed, combine(base, [ds[i] for i in idx], [1.0] * len(idx), plugin, {}))["mean"] if idx else evaluate(seed, base)["mean"]
    chosen, best, trail = [], score([]), []
    while True:
        cands = [(score(chosen + [i]), i) for i in range(len(ds)) if i not in chosen]
        if not cands: break
        s, i = min(cands); say("PROGRESS", stage="searching", tried=len(cands), best=round(s, 4))
        if s >= best - 1e-4: break
        chosen.append(i); best = s; trail.append({"added": entries[i]["record"]["name"], "branch": entries[i]["record"]["delta_sha256"][:16], "score": round(s, 4)})
    say("DONE", plugin=plugin, chosen=[entries[i]["record"]["delta_sha256"][:16] for i in chosen], score=round(best, 4), everything=round(score(list(range(len(ds)))), 4), trail=trail,
        note="scored on the line's public evaluation suite: anyone can train on it, so treat a search result as a suggestion, not a ranking")


# ── the second lineage path: public Hugging Face models (tools/hf_recipe.py) ──
# A line whose base and branches are existing public models (fine-tunes of one base), pinned by repo, commit and the
# SHA-256 Hugging Face publishes for each weight file. Nothing is uploaded: the recipe is enough. Their training
# happened elsewhere and cannot be replayed, so the verifier shows each model card's account as a claim.
def hf():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hf_recipe", os.path.join(os.path.dirname(HERE), "tools", "hf_recipe.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.CACHE = os.environ.get("PSI_HF_RECIPE_CACHE") or os.path.join(C, "hfcache"); m.say = lambda *x: say("PROGRESS", stage=" ".join(str(v) for v in x).strip()[:160]); return m


def H_FAR(): return 0.2


def is_hf(doc): return isinstance(doc, dict) and doc.get("kind") == "psi-hf-recipe"
def hf_recipe_of(doc): return {k: v for k, v in doc.items() if k not in ("signatures", "imported_from", "imported_at", "published")}


def hf_out_dir(r):
    root = os.environ.get("PSI_MODELS_DIR") or DIRS["models"]   # in the models folder, PSI lists it (Settings, Timeline, inside engine)
    return os.path.join(root, "commons", f"{slug(r.get('name'))}-{r['expects']['output_sha256'][:12]}")


def hf_make(a):
    H = hf(); br = [x.strip() for x in (a.branches or "").split(",") if x.strip()]
    if not br: raise SystemExit("PSI-ERROR choose at least one model to merge into the base")
    say("PROGRESS", stage="pinning", models=1 + len(br))
    opts = json.loads(a.options or "{}"); st = json.loads(a.strengths or "null") or []
    r = {"v": 1, "kind": "psi-hf-recipe", "name": str(a.name)[:80], "about": str(a.about or "")[:1000], "base": H.pin(a.base), "method": {"name": a.plugin or "task_arithmetic", "options": opts},
         "branches": [dict(H.pin(b), strength=float(st[i]) if i < len(st) else 1.0) for i, b in enumerate(br)], "carry_from": a.kind or "base",
         "curator": {k: v for k, v in me().items() if k != "as"}, "created": now()}
    say("PROGRESS", stage="downloading and merging once, to fix the expected hash")
    r["expects"] = {"output_sha256": H.build(r, None)}
    far = []
    for b, d in zip(r["branches"], H.DISTANCES):
        b["distance_from_base"] = d
        if d > H.FAR: far.append(f"{b['repo']} ({d:.2f})")
    doc = dict(r, signatures=signed(r, "curator")); rid = sha_json(r)[:16]
    json.dump(doc, open(os.path.join(mk(DIRS["recipes"]), f"{rid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("DONE", recipe=rid, name=r["name"], branches=len(r["branches"]), plugin=r["method"]["name"], output=r["expects"]["output_sha256"][:16], signed=bool(doc["signatures"]),
        warning=("far from the base, so not a light fine-tune of it; merging will likely damage both: " + ", ".join(far)) if far else None)


def hf_check(rid, doc, quiet=False):
    r = hf_recipe_of(doc); c = hf().check_json(r)
    res = {"recipe": rid, "kind": "hf", "name": r.get("name"), "about": r.get("about"), "line": f"hf:{r['base']['repo']}", "state": c["state"],
           "why": "every model is on Hugging Face at its pinned commit and unchanged, or already here" if c["state"] == "ready" else "a model was removed or changed at its pinned commit, and is not here",
           "signatures": keys.signatures(r, doc.get("signatures")), "curator": r.get("curator"),
           "method": {"plugin": r["method"]["name"], "options": r["method"].get("options"), "known": True, "same_code": True, "linear": r["method"]["name"] in ("average", "weighted", "task_arithmetic")},
           "models": [dict(x, distance=(r["branches"][i - 1].get("distance_from_base") if i else None)) for i, x in enumerate(c["rows"])], "far": H_FAR(), "download_bytes": c["download_bytes"], "output_published": False, "only_source": [], "lost_subjects": [], "branches": [], "imported_from": doc.get("imported_from")}
    if not quiet: say("DONE", **res)
    return res


def hf_build(rid, doc):
    r = hf_recipe_of(doc); H = hf(); t0 = time.time(); out = hf_out_dir(r); got = H.build(r, out); ok = got == r["expects"]["output_sha256"]
    rec = {"v": 3, "kind": "model", "line": f"hf:{r['base']['repo']}", "recipe_name": r.get("name"), "recipe": rid, "recipe_sha256": sha_json(r), "base": f"{r['base']['repo']}@{r['base']['commit']}",
           "output_sha256": got, "mode": "strict", "matches_recipe": ok, "notes": [], "licences": sorted({x.get("licence") or "not stated" for x in [r["base"]] + r["branches"]}),
           "built_by": {k: v for k, v in me().items() if k != "as"}, "built": now()}
    e = {"record": rec, "signatures": signed(rec, "builder")}
    json.dump(e, open(os.path.join(out, "psi-record.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(dict(e, dir=out), open(os.path.join(mk(DIRS["models"]), f"hf-{got[:12]}.link.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    listed = bool(os.environ.get("PSI_MODELS_DIR"))
    say("DONE", recipe=rid, mode="strict", kind="hf", dir=out.replace("\\", "/"), output=got[:16], matches=ok, expected=r["expects"]["output_sha256"][:16], seconds=round(time.time() - t0, 1),
        notes=["A Hugging Face folder (model.safetensors, config, tokenizer). " + ("It is in your models folder: Settings lists it, Convert to GGUF makes it run in the answering engine, and the Timeline can place it." if listed else "Set a models folder in Settings to have it listed there.")])


def bisect(rid):
    doc = recipe_doc(rid); r = doc["recipe"]; seed = load_seed(); base, _ = base_by_sha(seed, r["base"])
    by = {e["record"]["delta_sha256"]: e for e in doc["branches"]}; bs = r["branches"]; ds = {b["delta_sha256"]: load_delta(by[b["delta_sha256"]]) for b in bs}
    sc = lambda grp: evaluate(seed, combine(base, [ds[b["delta_sha256"]] for b in grp], [b.get("strength", 1.0) for b in grp], r["method"]["plugin"], r["method"].get("options") or {}))["mean"]
    builds = 0; cur = list(bs); full = sc(cur); builds += 1
    while len(cur) > 1:
        h = len(cur) // 2; a_, b_ = cur[:h], cur[h:]; sa, sb = sc(a_), sc(b_); builds += 2
        say("PROGRESS", stage="bisecting", left=len(cur), builds=builds)
        cur = a_ if sa > sb else b_   # the half whose merge scores worse holds the suspect
    sus = by[cur[0]["delta_sha256"]]["record"]
    without = sc([b for b in bs if b["delta_sha256"] != cur[0]["delta_sha256"]]); builds += 1
    say("DONE", suspect=sus["name"], branch=sus["delta_sha256"][:16], contributor=(sus.get("contributor") or {}).get("name"), full=round(full, 4), without=round(without, 4), builds=builds,
        note="the branch whose half of the recipe scores worst; leaving it out changes the score by %+.4f (differences under ~0.02 are within this suite's noise)" % (without - full))


def use(rid):
    doc = recipe_doc(rid); r = doc["recipe"]; who = me()
    if who["as"] == "private": raise SystemExit("PSI-ERROR a use record is signed: choose a key to sign with first (private mode records nothing)")
    rec = {"v": 3, "kind": "use", "recipe_sha256": sha_json(r), "output_sha256": r["expects"]["output_sha256"], "by": {k: v for k, v in who.items() if k != "as"}, "created": now()}
    e = {"record": rec, "signatures": signed(rec, "user")}; f = os.path.join(mk(DIRS["uses"]), f"{sha_json(rec)[:16]}.json"); json.dump(e, open(f, "w", encoding="utf-8"), indent=1)
    say("DONE", use=sha_json(rec)[:16], recipe=rid)


# ── publishing: a folder laid out like a Hugging Face repo, pushed only when the user confirmed it (with their token) ──
CARD = """---
license: {licence}
library_name: parameter-space-inspector
{base}tags:
- emergence-machine
- parameter-space-inspector
- psi-{kind}
- psi-line-{line}
---
# {name}

{about}

A {kind} of the **{line}** seed line, made with the Parameter Space Inspector (https://emergencemachine.com).
Everything here is checked by hash; nothing needs to be trusted by address. {use}
To check its claims: `python src/verify_lineage.py`.

{body}
"""
# Seed lines whose seed is a public Hugging Face model: a recipe built on the seed itself names it as its base, so it is
# listed in that model's family tree on Hugging Face.
LINE_HF = {"stories260k": "karpathy/tinyllamas"}
FREE = {"public domain", "cc0", "cc0-1.0", "mit", "apache-2.0", "cc-by-4.0"}
HOWTO = """## How to use it

**In the Parameter Space Inspector** (emergencemachine.com):
1. Beta > Commons > Recipes. Paste `hf://{repo}/recipe.json` in Import, or find it with *Browse Hugging Face*.
2. **Check** it: every branch, the base and the expected output are compared by SHA-256, and any signatures are verified.
3. **Build** it: the app downloads the branches from this repo, combines them on the seed with the pinned method, and keeps the model only if it matches the expected hash. A GGUF can be made at the same time, for Generate and the other views.

**From the command line** (with the PSI source; PSI is in beta):
```
python src/recipes.py import hf://{repo}/recipe.json
python src/recipes.py check <recipe id>
python src/recipes.py build <recipe id> --gguf
```

**Without PSI:** `model.safetensors` is the built output (the seed's weights in the llama2.c layout, plain safetensors: no code runs on load). `recipe.json` names every input and its hash; each `branches/<sha256>.safetensors` is one contributor's change to the seed.

## What is in this repo
| file | what |
|---|---|
| `recipe.json` | the recipe: base, branches (each pinned by hash, with a strength), method, expected output, the branch records |
| `branches/` | each branch's change to the seed, named by its SHA-256 |
| `model.safetensors` | the built model |
| `config.json` | a short summary (kind, line, hashes); the file Hugging Face counts as a download |
"""


def suite_row(rec):
    ev = rec.get("eval") or {}; b, a, o = ev.get("before") or {}, ev.get("after") or {}, ev.get("own_text") or {}
    return f"{b.get('mean', '?')} → {a.get('mean', '?')} | {o.get('before', '?')} → {o.get('after', '?')}"


# How to use what a repo holds, by kind: only a recipe repo has a recipe.json to import.
USE = {"recipe": "To use it: PSI > Commons > Recipes > Import, paste `hf://{repo}/recipe.json`, check, build.",
       "branch": "To use it: a recipe that names this branch fetches it from here by its hash (PSI > Commons > Recipes).",
       "identity": "Verifiers read these public keys to check what this account signed."}


def card_fmt(**kw):
    kw.setdefault("base", "")
    t = CARD.format(use=USE[kw["kind"]].format(repo=kw["repo"]), **kw)
    if kw["kind"] == "identity":   # keys belong to no seed line
        t = t.replace(f"- psi-line-{kw['line']}" + chr(10), "").replace(f"A identity of the **{kw['line']}** seed line, made with", "Published from")
    return t


def stage(what, ref, repo):
    # A fresh folder each time: on Windows a folder just removed can refuse new files for a moment (a dry run then a
    # publish hit "Permission denied"); earlier stagings of the same repo are cleared as far as they can be.
    import shutil, glob
    box = mk(os.path.join(C, "outbox")); stem = slug(repo.replace("/", "--")) + "-" + what
    for old in glob.glob(os.path.join(box, stem + "*")): shutil.rmtree(old, ignore_errors=True)
    out = os.path.join(box, f"{stem}-{int(time.time() * 1000)}"); mk(out); files = []
    def put(rel, src=None, text=None):
        p = os.path.join(out, rel); mk(os.path.dirname(p))
        if text is not None: open(p, "w", encoding="utf-8").write(text)
        else: shutil.copyfile(src, p)
        files.append(rel)
    if what == "recipe":
        doc = recipe_doc(ref); r = doc["recipe"]
        doc = dict(doc, output_addrs=[f"hf://{repo}/model.safetensors"] + [a for a in doc.get("output_addrs", []) if not a.startswith("store://")])
        for e in doc["branches"]:
            h = e["record"]["delta_sha256"]; sp = hub.store_path(h)
            if os.path.exists(sp): put(f"branches/{h}.safetensors", sp); e["addrs"] = [f"hf://{repo}/branches/{h}.safetensors"] + [a for a in e.get("addrs", []) if not a.startswith("store://")]
        op = hub.store_path(r["expects"]["output_file_sha256"])
        if os.path.exists(op): put("model.safetensors", op)
        doc.pop("imported_from", None); doc.pop("imported_at", None); doc.pop("published", None)
        put("recipe.json", text=json.dumps(doc, ensure_ascii=False, indent=1))
        put("config.json", text=json.dumps({"psi": {"kind": "recipe", "line": r["line"], "recipe_id": sha_json(r)[:16], "recipe_sha256": sha_json(r), "base_sha256": r["base"],
                                                    "branches": len(r["branches"]), "method": r["method"]["plugin"], "output_sha256": r["expects"]["output_sha256"],
                                                    "output_file_sha256": r["expects"].get("output_file_sha256"), "import": f"hf://{repo}/recipe.json"}}, indent=1))
        lic = sorted({(e["record"].get("source") or {}).get("licence") or "?" for e in doc["branches"]})
        seed_hf = LINE_HF.get(r["line"]) if r["base"] == load_seed().sha else None
        base_line = f"base_model: {seed_hf}\nbase_model_relation: finetune\npipeline_tag: text-generation\n" if seed_hf else ""
        licence = SL.LINES[r["line"]]["licence"].lower() if all(str(x).lower() in FREE for x in lic) else "other"
        L = SL.LINES[r["line"]]
        rows = "\n".join(f"| {e['record'].get('name')} | {(e['record'].get('contributor') or {}).get('name')} | {(e['record'].get('source') or {}).get('licence')} | {(e['record'].get('source') or {}).get('subject')} | {(e['record'].get('source') or {}).get('url') or ''} | {suite_row(e['record'])} |" for e in doc["branches"])
        body = ("## Branches\n| branch | by | licence | trained on | source | line suite loss (before → after) | own text loss (before → after) |\n|---|---|---|---|---|---|---|\n" + rows +
                "\n\nLower is better. The line suite is the seed line's public evaluation set; own text is each branch's held-out tenth of its own text, never trained on.\n\n"
                f"Method: `{r['method']['plugin']}` {json.dumps(r['method'].get('options') or {})}, pinned by its code hash. Seed: {L['title']} by {L['by']} ({L['licence']}). Branch licences: {', '.join(lic)}.\n\n"
                + HOWTO.format(repo=repo))
        put("README.md", text=card_fmt(licence=licence, kind="recipe", line=r["line"], name=r.get("name"), about=r.get("about") or "", repo=repo, base=base_line, body=body))
    elif what == "branch":
        e = branch_entry(ref); h = e["record"]["delta_sha256"]; put(f"branches/{h}.safetensors", hub.store_path(h))
        e = dict(e, addrs=[f"hf://{repo}/branches/{h}.safetensors"]); e.pop("published", None); put(f"branches/{h}.json", text=json.dumps(e, ensure_ascii=False, indent=1))
        td = os.path.join(DIRS["traces"], h[:16])
        if os.path.isdir(td):
            for f in sorted(os.listdir(td)): put(f"traces/{h[:16]}/{f}", os.path.join(td, f))
        put("README.md", text=card_fmt(licence="other", kind="branch", line=e["record"]["line"], name=e["record"].get("name"), about=e["record"].get("about") or "", repo=repo,
                                          body=f"Trained on: {(e['record'].get('source') or {}).get('subject') or 'not said'} ({(e['record'].get('source') or {}).get('licence')}). The training trace is in traces/: anyone given the text can replay it."))
    elif what == "identity":
        ks = [{"id": k["id"], "name": k["name"], "public": k["public"], "created": k["created"], "replaced_by": k.get("replaced_by")} for k in keys.listing()]
        rot_dir = os.path.join(keys.KDIR, "rotations"); rots = [json.load(open(os.path.join(rot_dir, f))) for f in sorted(os.listdir(rot_dir))] if os.path.isdir(rot_dir) else []
        prof = None
        ps = os.environ.get("PSI_PROFILE_SHA")
        if ps and os.path.exists(os.path.join(C, "profiles", f"{ps}.json")): prof = json.load(open(os.path.join(C, "profiles", f"{ps}.json"), encoding="utf-8"))
        put("psi-identity.json", text=json.dumps({"v": 1, "kind": "psi-identity", "keys": ks, "rotations": rots, "profile": prof}, ensure_ascii=False, indent=1))
        put("README.md", text=card_fmt(licence="cc0-1.0", kind="identity", line="any", name="Signing keys", about="The public keys this person signs PSI branches, recipes and models with.", repo=repo, body=""))
    else: raise SystemExit("PSI-ERROR publish what? recipe, branch or identity")
    return out, files


def publish(a):
    what, ref, repo = a.what, a.ref, a.repo or ""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo): raise SystemExit("PSI-ERROR a repo is OWNER/NAME")
    out, files = stage(what, ref, repo)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in files)
    if a.dry_run: say("DONE", dry_run=True, repo=repo, folder=out.replace("\\", "/"), files=files, bytes=size); return
    say("PROGRESS", stage="pushing", repo=repo, files=len(files), bytes=size)
    # A new repo is private unless the user's setting says otherwise (Settings > Hugging Face); an existing repo keeps its own.
    res = hub.push(repo, {f: os.path.join(out, f) for f in files}, f"PSI: {what} {ref or ''}".strip(), private=bool(a.private))
    pub = {"repo": repo, "commit": res["commit"], "time": res["time"], "url": res["url"], "private": res.get("private", bool(a.private))}
    if what == "recipe":
        f = os.path.join(DIRS["recipes"], f"{sha_json(recipe_doc(ref)['recipe'])[:16]}.json"); d = json.load(open(f, encoding="utf-8")); d["published"] = pub
        d["output_addrs"] = list(dict.fromkeys(d.get("output_addrs", []) + [f"hf://{repo}/model.safetensors"])); json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if what == "branch":
        e = branch_entry(ref); f = os.path.join(DIRS["branches"], f"{e['record']['delta_sha256'][:16]}.json"); e["published"] = pub
        e["addrs"] = list(dict.fromkeys(e.get("addrs", []) + [f"hf://{repo}/branches/{e['record']['delta_sha256']}.safetensors"])); json.dump(e, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("DONE", **pub, files=len(files), bytes=size)


def cosign(kind, ref):
    """Countersign as The Emergence Machine (the explicit button; only where its key is)."""
    if kind == "recipe":
        f = os.path.join(DIRS["recipes"], f"{sha_json(recipe_doc(ref)['recipe'])[:16]}.json")
    else: f = os.path.join(DIRS["branches"], f"{branch_entry(ref)['record']['delta_sha256'][:16]}.json")
    d = json.load(open(f, encoding="utf-8"))
    if any(s.get("role") == "emergence-machine" for s in d.get("signatures", [])): raise SystemExit("PSI-ERROR already countersigned by The Emergence Machine")
    cosign_em(d); json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say("DONE", cosigned=kind, ref=ref, signatures=keys.signatures(d.get("record") or d.get("recipe"), d["signatures"]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("ref", nargs="?")
    for o in ("--text", "--name", "--base", "--licence", "--subject", "--about", "--url", "--branches", "--plugin", "--options", "--strengths", "--mode", "--what", "--repo", "--kind", "--sort"): ap.add_argument(o)
    ap.add_argument("--steps", default="200"); ap.add_argument("--gguf", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", help="publish: create a new repo as private")
    a = ap.parse_args(); MP.load_outside()
    c = a.cmd
    if c == "line-status": print(json.dumps(SL.line_status(LINE))); return
    if c == "line-fetch": say("DONE", **SL.fetch_line(LINE, lambda n: say("PROGRESS", stage="downloaded", file=n))); return
    if c == "train": return train(a)
    if c == "branches": print(json.dumps({"branches": [dict(e, signatures_checked=keys.signatures(e["record"], e.get("signatures"))) for e in listing("branches")]}, ensure_ascii=False)); return
    if c == "recipes":
        def row(d):
            if is_hf(d): return {"id": sha_json(hf_recipe_of(d))[:16], "kind": "hf", "line": f"hf:{d['base']['repo']}", "name": d.get("name"), "about": d.get("about"), "branches": len(d["branches"]),
                                 "plugin": d["method"]["name"], "curator": d.get("curator"), "created": d.get("created"), "imported_from": d.get("imported_from"), "signed": bool(d.get("signatures")),
                                 "models": [d["base"]["repo"]] + [b["repo"] for b in d["branches"]]}
            return {"id": sha_json(d["recipe"])[:16], "kind": "psi", "line": d["recipe"].get("line"), "name": d["recipe"].get("name"), "about": d["recipe"].get("about"), "branches": len(d["recipe"]["branches"]),
                    "plugin": d["recipe"]["method"]["plugin"], "curator": d["recipe"].get("curator"), "created": d["recipe"].get("created"), "imported_from": d.get("imported_from"), "signed": bool(d.get("signatures"))}
        print(json.dumps({"recipes": [row(d) for d in listing("recipes")]}, ensure_ascii=False)); return
    if c == "hf-find": say("DONE", **hf().find(a.ref)); return
    if c == "hf-make": return hf_make(a)
    if c == "models": print(json.dumps({"models": [dict(m, dir=m["dir"].replace("\\", "/")) for m in listing_models()]}, ensure_ascii=False)); return
    if c == "plugins": print(json.dumps({"plugins": MP.listing(), "outside": [{k: v for k, v in p.items()} for p in MP.outside()]}, ensure_ascii=False)); return
    if c == "make": return make(a)
    if c == "import": return import_recipe(a.ref)
    if c == "check": return check(a.ref)
    if c == "build": return build(a.ref, a.mode or "strict", a.gguf)
    if c == "search": return search(a)
    if c == "bisect": return bisect(a.ref)
    if c == "use": return use(a.ref)
    if c == "publish": return publish(a)
    if c == "cosign": return cosign(a.kind or "recipe", a.ref)
    if c == "browse": say("DONE", results=hub.browse(a.ref or "", sort=a.sort or "downloads")); return
    if c == "hf-user": say("DONE", **hub.hf_user(a.ref)); return
    raise SystemExit(f"PSI-ERROR unknown command {c}")


if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if isinstance(e.code, str): print(e.code if e.code.startswith("PSI-ERROR") else f"PSI-ERROR {e.code}", flush=True); sys.exit(1)
        raise
    except hub.Refused as e: print(f"PSI-ERROR {e}", flush=True); sys.exit(1)
