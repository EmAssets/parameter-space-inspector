# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""The lineage verifier: anyone can check a branch, recipe or built model of a seed line, with nothing to trust but
the maths. Each line of the report says how strong it is:

  proof     checked here, exactly: a signature, a hash, a score re-measured on the line's evaluation suite
  evidence  makes a claim likely, not certain: replayed training segments, closeness to the parent's weights
  claim     what the record says and nobody can check from outside: the computer and app it was trained in

What it checks:
  signatures      every signature on the record (contributor, curator, builder, The Emergence Machine), and, online,
                  whether a key is published by the profile it names (psi-identity.json in a psi-identity repo)
  file            the change or model file against the SHA-256 the record pins
  descent         how far the weights moved from the parent they claim (relative L2): a fine-tune of that parent moves a
                  little; a model from elsewhere would be far away. The weights carry the proof, not a phrase: How AI
                  Emerges' birth line is public and any model can be taught to say it, so it is never counted here
  replay          re-runs randomly chosen training segments from the published trace on the declared text; needs the
                  text and its salt (public text, or given privately by the contributor). On this CPU set-up a match is
                  exact; elsewhere a miss under 1% of the segment's movement counts
  scores          the record's scores re-measured on the line's public evaluation suite
  public log      a Hugging Face commit that holds the record: the record existed by that commit's time
  where trained   always a claim: PSI is open source, so nothing on a user's computer can prove which app trained it

Usage:  python verify_lineage.py PATH [--replay N] [--text FILE --salt HEX] [--online] [--json]
        PATH: a branch entry (.json), a recipe (.json) or a built model folder (record.json + model.safetensors)
"""
import argparse, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hub, keys, seedline as SL


def row(claim, ok, strength, detail=""):
    return {"claim": claim, "result": "yes" if ok is True else "no" if ok is False else "n/a", "strength": strength, "detail": detail}


def sig_rows(rec, sigs, online=False):
    out = []
    checked = keys.signatures(rec, sigs)
    if not checked: return [row("signed", None, "proof", "not signed: the record makes its claims without anyone's name on them (allowed: signing is optional)")]
    for s in checked:
        who = "The Emergence Machine" if s["role"] == "emergence-machine" else s["role"]
        out.append(row(f"signed by {who} (key {s['key']})", s["valid"], "proof", "the signature matches the record" if s["valid"] else "the signature does NOT match: the record was changed after signing, or signed by another key"))
    links = ((rec.get("contributor") or rec.get("curator") or rec.get("built_by") or {}).get("links")) or []
    if online:
        for l in links:
            for addr in hub.identity_addrs(l):
                try:
                    ident = hub.get_json(addr, max_bytes=1 << 20); pubs = {k.get("public") for k in ident.get("keys", []) if isinstance(k, dict)}
                    hit = any(s["signer"] in pubs for s in sigs)
                    out.append(row(f"key published by {l.get('host')}:{l.get('user')}", hit, "proof", addr))
                except Exception as e: out.append(row(f"key published by {l.get('host')}:{l.get('user')}", False, "proof", f"could not read {addr}: {str(e)[:100]}"))
    elif links: out.append(row("key belongs to the named profile", None, "proof", "run with --online to read the profile's published keys"))
    return out


def log_rows(entry, online):
    pub = entry.get("published")
    if not pub: return [row("in a public log", None, "proof", "not published yet: nothing outside this computer says when it existed")]
    private = pub.get("private") is True
    if not online: return [row(f"in a public log ({pub.get('repo')} @ {str(pub.get('commit'))[:10]})", None, "proof",
                               f"says {pub.get('time')}{'; published privately' if private else ''}; run with --online to check the commit")]
    try:
        info = hub.get_json(f"https://huggingface.co/api/models/{pub['repo']}/revision/{pub['commit']}", max_bytes=4 << 20)
        if private or info.get("private"):   # read with the owner's token: the commit exists, but nobody else can see it
            return [row("in a public log", None, "proof", f"published privately: Hugging Face commit {pub['commit'][:10]} of {pub['repo']} exists ({info.get('lastModified')}), "
                        "but only its account can see it, so it proves nothing to anyone else until the repo is made public")]
        return [row("in a public log", True, "proof", f"Hugging Face commit {pub['commit'][:10]} of {pub['repo']}, {info.get('lastModified')}: the record existed by then")]
    except Exception as e:
        if private: return [row("in a public log", None, "proof", f"published privately ({pub['repo']}): not readable without its account's token ({str(e)[:80]})")]
        return [row("in a public log", False, "proof", f"the commit could not be read: {str(e)[:120]}")]


def verify_branch(entry, a, base_dir=None):
    rec = entry["record"]; rows = sig_rows(rec, entry.get("signatures"), a.online)
    seed = SL.Seed(rec["line"]); import recipes as R
    try: base, base_name = R.base_by_sha(seed, rec["base"])
    except SystemExit: base = None; rows.append(row("the parent it names is here", False, "proof", f"base {rec['base'][:12]} is not on this computer: build or download it to check descent and scores"))
    p = hub.store_path(rec["delta_sha256"]); have = os.path.exists(p)
    if have: rows.append(row("the change file is the one signed", hub.sha_file(p) == rec["delta_sha256"], "proof", f"SHA-256 {rec['delta_sha256'][:16]}"))
    else: rows.append(row("the change file is the one signed", None, "proof", "the file is not on this computer"))
    if have and base is not None:
        from safetensors.torch import load_file
        d = {k: v.float() for k, v in load_file(p).items()}; vec = {k: base[k] + d[k] for k in base}
        dist = SL.distance(vec, base)
        rows.append(row(f"descends from its parent ({base_name})", dist < 0.5, "evidence", f"the weights moved {dist:.3f} of the parent's size from it (a fine-tune moves a little; an unrelated model of this shape and scale is ~1.3 away)"))
        ev = R.evaluate(seed, vec); claimed = ((rec.get("eval") or {}).get("after") or {}).get("mean")
        if claimed is not None: rows.append(row("its scores are as claimed", abs(ev["mean"] - claimed) < 2e-3, "proof", f"re-measured {ev['mean']} on {ev['suite']}; the record says {claimed}"))
    tdir = os.path.join(R.DIRS["traces"], rec["delta_sha256"][:16]); tf = os.path.join(tdir, "trace.json")
    if a.replay and os.path.exists(tf) and base is not None:
        trace = json.load(open(tf, encoding="utf-8"))
        if R.sha_json(trace) != (rec.get("trace") or {}).get("sha256"): rows.append(row("its training replays", False, "evidence", "the trace here is not the one the record pins"))
        else:
            priv = os.path.join(R.DIRS["private"], f"{rec['delta_sha256'][:16]}.json"); text = a.text; salt = a.salt
            if not (text and salt) and os.path.exists(priv): pv = json.load(open(priv)); text = text or pv.get("text_file"); salt = salt or pv.get("salt"); held = pv.get("held_out_pieces")
            else: held = (rec.get("eval") or {}).get("own_text", {}).get("pieces")
            if not (text and salt and os.path.exists(text)): rows.append(row("its training replays", None, "evidence", "the text it trained on is private: only someone the contributor gives the text and salt to can replay it"))
            else:
                raw = open(text, encoding="utf-8", errors="replace").read()
                if text.endswith(".json"):
                    try: raw = json.loads(raw).get("text", raw)
                    except ValueError: pass
                ids = seed.tok.encode(raw); ids = ids[:-held] if held else ids
                n = len(trace["checkpoints"]) - 1; pick = sorted(random.Random(a.seed).sample(range(n), min(a.replay, n)))
                res = SL.replay(seed, base, ids, salt, trace, tdir, pick); good = sum(1 for x in res if x["ok"])
                exact = sum(1 for x in res if x.get("exact"))
                rows.append(row(f"its training replays ({good} of {len(res)} segments)", good == len(res), "evidence",
                                f"segments {pick}: {exact} exact, {good - exact} within tolerance" + "".join(f"; segment {x['segment']}: {x.get('why') or 'missed by %.1f%%' % (100 * (x.get('miss_share') or 0))}" for x in res if not x["ok"])))
    elif a.replay: rows.append(row("its training replays", None, "evidence", "no trace on this computer for this branch"))
    rows += log_rows(entry, a.online)
    t = rec.get("trained") or {}
    rows.append(row(f"trained in PSI on {t.get('device', '?')}, {t.get('steps')} steps", None, "claim", "said by the contributor; not provable from outside"))
    return {"kind": "branch", "name": rec.get("name"), "id": rec["delta_sha256"][:16], "rows": rows}


def verify_model(d, a):
    e = json.load(open(os.path.join(d, "record.json"), encoding="utf-8")); rec = e["record"]; rows = sig_rows(rec, e.get("signatures"), a.online)
    f = os.path.join(d, "model.safetensors"); rows.append(row("the model file is the one recorded", hub.sha_file(f) == rec["file_sha256"], "proof", f"SHA-256 {rec['file_sha256'][:16]}"))
    import recipes as R
    if rec.get("recipe"):
        try:
            doc = R.recipe_doc(rec["recipe"]); exp = doc["recipe"]["expects"]["output_sha256"]
            from safetensors.torch import load_file
            got = SL.vec_sha({k: v.float() for k, v in load_file(f).items()})
            rows.append(row("it is what its recipe makes", got == exp, "proof", f"weights {got[:16]}, recipe expects {exp[:16]}" + ("" if got == exp else f" ({rec.get('mode')} build: {'; '.join(rec.get('notes') or [])})")))
        except SystemExit: rows.append(row("it is what its recipe makes", None, "proof", "the recipe is not on this computer"))
    rows += log_rows(e, a.online)
    return {"kind": "model", "name": rec.get("recipe_name"), "id": rec["output_sha256"][:16], "rows": rows}


def verify_recipe(doc, a):
    r = doc["recipe"]; rows = sig_rows(r, doc.get("signatures"), a.online)
    import recipes as R
    rid = R.sha_json(r)[:16]
    if not os.path.exists(os.path.join(R.DIRS["recipes"], f"{rid}.json")): json.dump(doc, open(os.path.join(R.mk(R.DIRS["recipes"]), f"{rid}.json"), "w", encoding="utf-8"))
    c = R.check(rid, quiet=True)
    rows.append(row(f"every ingredient checks out ({sum(1 for x in c['branches'] if x['status'] == 'ok')} of {len(c['branches'])})", c["state"] == "ready", "proof", c["why"]))
    for x in c["branches"]:
        rows.append(row(f"branch {x.get('name') or x['delta_sha256'][:12]} by {x.get('contributor') or '?'}", x["status"] == "ok", "proof",
                        x.get("why") or ("signed, " if x.get("signatures") else "unsigned, ") + f"{x.get('licence')}, at {x.get('where')}"))
    rows += log_rows(doc, a.online)
    return {"kind": "recipe", "name": r.get("name"), "id": rid, "rows": rows, "state": c["state"]}


# ── the Hugging Face path: public models pinned by repo, commit and file hash ──
def verify_hf_recipe(doc, a, built=None):
    import recipes as R
    r = R.hf_recipe_of(doc); H = R.hf(); rows = sig_rows(r, doc.get("signatures"), a.online)
    c = H.check_json(r)
    for x in c["rows"]:
        rows.append(row(f"{x['role']} {x['repo']} @ {x['commit'][:10]} is still published, unchanged", x["ok"], "proof",
                        f"licence {x.get('licence') or 'not stated'}; " + ("every pinned file hash matches Hugging Face's" if x["ok"] else f"missing or changed: {x['missing']}") + ("; a checked copy is on this computer" if x["here"] else "")))
    if all(x["here"] for x in c["rows"]):   # descent: how far each branch's weights are from the base's (needs them here)
        import numpy as np
        base = H.Weights(r["base"]); keys_ = base.keys()
        for ing in r["branches"]:
            w = H.Weights(ing); num = den = 0.0
            for k in keys_:
                b, _ = base.get(k); v, _ = w.get(k); num += float(((v - b) ** 2).sum()); den += float((b ** 2).sum())
            d = (num / den) ** 0.5
            rows.append(row(f"{ing['repo']} is a fine-tune of the base", d < 0.5, "evidence", f"its weights are {d:.3f} of the base's size away from it (a fine-tune moves a little; an unrelated model of the same shape is ~1.4 away)"))
    else: rows.append(row("each model is a fine-tune of the base", None, "evidence", "build the recipe first: the weights are needed to measure"))
    if built:
        from safetensors import safe_open
        import hashlib, torch
        h = hashlib.sha256(); f = safe_open(os.path.join(built, "model.safetensors"), "pt")
        for k in sorted(f.keys()):
            t = f.get_tensor(k); h.update(k.encode()); h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().astype(t.numpy().dtype.newbyteorder("<")).tobytes())
        got = h.hexdigest(); want = r["expects"]["output_sha256"]
        rows.append(row("this model is what the recipe makes", got == want, "proof", f"weights {got[:16]}, recipe expects {want[:16]}"))
    for ing in r["branches"]: rows.append(row(f"how {ing['repo']} was trained", None, "claim", f"its model card says (https://huggingface.co/{ing['repo']}); it was trained elsewhere, so there is no trace to replay"))
    rows += log_rows(doc, a.online)
    return {"kind": "hf recipe" if not built else "hf model", "name": r.get("name"), "id": R.sha_json(r)[:16], "rows": rows}


def verify(path, a):
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "psi-record.json")):
        e = json.load(open(os.path.join(path, "psi-record.json"), encoding="utf-8")); import recipes as R
        res = verify_hf_recipe(R.recipe_doc(e["record"]["recipe"]), a, built=path)
        b = sig_rows(e["record"], e.get("signatures"), a.online)[:1]; b[0]["claim"] = "the build record: " + b[0]["claim"]; res["rows"][0]["claim"] = "the recipe: " + res["rows"][0]["claim"]
        res["rows"] = b + res["rows"]; return res
    if path.endswith(".link.json"): return verify(json.load(open(path, encoding="utf-8"))["dir"], a)
    if os.path.isdir(path): return verify_model(path, a)
    doc0 = json.load(open(path, encoding="utf-8"))
    if doc0.get("kind") == "psi-hf-recipe": return verify_hf_recipe(doc0, a)
    doc = json.load(open(path, encoding="utf-8"))
    if "recipe" in doc: return verify_recipe(doc, a)
    if (doc.get("record") or {}).get("kind") == "branch": return verify_branch(doc, a)
    raise SystemExit("not a branch, recipe or model this verifier knows")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("path")
    ap.add_argument("--replay", type=int, default=0, help="training segments to replay (0: none)"); ap.add_argument("--text"); ap.add_argument("--salt")
    ap.add_argument("--online", action="store_true", help="read published keys and public log entries"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(); a.seed = a.seed if a.seed is not None else random.randrange(1 << 30)
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the marks print on any console
    except (AttributeError, ValueError): pass
    r = verify(a.path, a)
    if a.json: print("PSI-DONE " + json.dumps(r, ensure_ascii=False)); return
    mark = {"yes": "✓", "no": "✗", "n/a": "·"}
    print(f"{r['name']}  ({r['kind']} {r['id']})")
    for x in r["rows"]: print(f"  {mark[x['result']]} [{x['strength']}] {x['claim']}" + (f"\n      {x['detail']}" if x["detail"] else ""))


if __name__ == "__main__":
    try: main()
    except SystemExit as e:
        if isinstance(e.code, str): print(f"PSI-ERROR {e.code}"); sys.exit(1)
        raise
