# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Hugging Face for Settings > Hugging Face and "Push to Hugging Face" (src/hf-api.js).

  python hf_push.py whoami                       the token's account, its organisations and its access (read / write / fine-grained)
  python hf_push.py plan <folder> <repo>         what a push would send: the files, their size, the model card (nothing is sent)
  python hf_push.py push <folder> <repo> [--private] [--message M]

The token comes only from PSI_HF_TOKEN, which the app sets for these calls; it is never printed. A push sends only the
files a model needs (weights, config, tokenizer, the cut record, the card) and lists what it left out. A folder without
README.md gets a model card made from its config.json and, for a model saved in Tipping point, its cut.json. Output: one
line PSI-DONE <json> or PSI-ERROR <message>; a push also prints PSI-PROGRESS <json> lines.
"""
import fnmatch, json, os, re, sys

KEEP = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "*.tiktoken",
        "chat_template.jinja", "chat_template.json", "preprocessor_config.json", "cut.json", "README.md", "LICENSE", "LICENSE.txt", "NOTICE"]
REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")


def out(kind, obj): print(f"PSI-{kind} " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)), flush=True)


def api():
    tok = os.environ.get("PSI_HF_TOKEN")
    if not tok: raise ValueError("no Hugging Face token: add one in Settings > Hugging Face")
    from huggingface_hub import HfApi
    return HfApi(token=tok)


def whoami():
    w = api().whoami()
    at = ((w.get("auth") or {}).get("accessToken") or {})
    role = at.get("role")
    scoped = ((at.get("fineGrained") or {}).get("scoped") or [])
    can_write = role == "write" or any("repo.write" in (s.get("permissions") or []) for s in scoped)
    return {"name": w.get("name"), "fullname": w.get("fullname"), "orgs": [o.get("name") for o in (w.get("orgs") or [])],
            "role": role or "unknown", "fine_grained": role == "fineGrained", "can_write": can_write,
            "token_name": at.get("displayName")}


def files_of(folder):
    keep, left = [], []
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if not os.path.isfile(p): left.append(n + "/"); continue
        (keep if any(fnmatch.fnmatch(n, k) for k in KEEP) else left).append(n)
    return keep, left


def card(folder, repo):
    """README.md for a folder that has none: what the model is, where it came from, what was cut."""
    cfg = {}
    try: cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
    except Exception: pass
    cut = None
    try: cut = json.load(open(os.path.join(folder, "cut.json"), encoding="utf-8"))
    except Exception: pass
    src = cfg.get("_name_or_path") or ""
    base_id = src if REPO.fullmatch(src or "") and not os.path.isabs(src) else None
    b = (cut or {}).get("base") or {}
    if b.get("hf_repo") and REPO.fullmatch(b["hf_repo"]): base_id = b["hf_repo"]   # recorded at save from the base's hf-source.json
    lic = _front(folder).get("license")   # the base's licence, from the card it came with
    bc = (b.get("hf_commit") or "")[:12]
    # emergence-machine on everything PSI publishes, so PSI's searches find it by default; then what kind of item it is.
    meta = ["---", "library_name: transformers", "tags:", "- emergence-machine", "- parameter-space-inspector", "- psi-model"]
    if cut: meta += ["- psi-cut", "- ablation", "- pruned"]
    if lic: meta += [f"license: {lic}"]
    if base_id: meta += [f"base_model: {base_id}"]
    meta += ["---", ""]
    name = repo.split("/", 1)[1]
    body = [f"# {name}", ""]
    if cut:
        c = cut.get("cut") or {}; base = base_id or (cut.get("base") or {}).get("name") or "the base model"
        what = {"structural": f"{c.get('neurons', 0):,} MLP neurons of {c.get('of_neurons', 0):,} and {c.get('heads', 0)} attention heads of {c.get('of_heads', 0)} deleted ({c.get('mode', 'random')})",
                "weights": f"{c.get('zeroed', 0):,} of {c.get('of', 0):,} raw weights zeroed ({c.get('criterion')}, {c.get('scope')}; asked {c.get('frac')})",
                "manual": f"chosen parts deleted: layers {c.get('layers') or []}, {c.get('heads', 0)} heads, {c.get('neurons', 0):,} neurons",
                "concept": f"the {c.get('k')} neurons that respond to “{c.get('concept')}” more than to “{c.get('keep')}” deleted"}.get(c.get("kind"), f"a cut of kind {c.get('kind')}")
        body += [f"**{base}** with part of it removed, saved from the Tipping point view of the [Parameter Space Inspector](https://emergencemachine.com/tools/parameter-space-inspector/).",
                 "", f"- What was removed: {what}.",
                 "- How: deleted neurons are zeroed columns of `down_proj`, deleted heads zeroed columns of `o_proj`; every other weight is the base model's own.",
                 f"- The base: {f'[{base_id}](https://huggingface.co/{base_id})' if base_id else base}{f' at commit `{bc}`' if bc else ''}, weights hash `{((cut.get('base') or {}).get('weights_hash') or '')[:16]}` ({(cut.get('base') or {}).get('weights_hash_of') or 'not recorded'}).",
                 "- `cut.json` lists every deleted neuron and head and the SHA-256 of every file.",
                 "", "This is an experiment: a damaged copy for studying what a model depends on, not a better or safer model. The base model's licence and terms apply to it."]
    else:
        body += ["A model folder shared from the [Parameter Space Inspector](https://emergencemachine.com/tools/parameter-space-inspector/).",
                 "", f"Base: {base_id}." if base_id else "", "The licence and terms of the model it was made from apply."]
    return "\n".join(meta + body).rstrip() + "\n"


def _front(folder):
    """The YAML front matter of a README.md in the folder, as flat key: value pairs (only what the card needs)."""
    try: t = open(os.path.join(folder, "README.md"), encoding="utf-8").read()
    except Exception: return {}
    m = re.match(r"^---\s*\n(.*?)\n---", t, flags=re.S)
    out = {}
    for line in (m.group(1).splitlines() if m else []):
        k, _, v = line.partition(":")
        if v.strip() and not line.startswith((" ", "-")): out[k.strip()] = v.strip().strip("'\"")
    return out


def _public_cut(folder):
    """cut.json as published: the same record without paths on this computer."""
    rec = json.load(open(os.path.join(folder, "cut.json"), encoding="utf-8"))
    (rec.get("base") or {}).pop("folder", None)
    return (json.dumps(rec, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def check(folder, repo):
    if not REPO.fullmatch(repo or ""): raise ValueError("a repo is OWNER/NAME, e.g. your-name/my-cut-model")
    if not os.path.isdir(folder): raise ValueError(f"not a folder: {folder}")
    keep, left = files_of(folder)
    if "config.json" not in keep or not any(n.endswith(".safetensors") for n in keep): raise ValueError("not a model folder: it needs config.json and .safetensors weights")
    return keep, left


def plan(folder, repo):
    keep, left = check(folder, repo)
    size = sum(os.path.getsize(os.path.join(folder, n)) for n in keep)
    cut = "cut.json" in keep
    c = None if "README.md" in keep and not cut else card(folder, repo)   # a cut model always gets its own card
    files = [n for n in keep if not (cut and n == "README.md")] + ([] if c is None else ["README.md (made for it)"])
    if cut and "README.md" in keep: left = left + ["README.md (the base model's card; replaced)"]
    return {"folder": folder.replace("\\", "/"), "repo": repo, "files": files, "left_out": left,
            "gb": round(size / 2**30, 3), "card": c, "cut": "cut.json" in keep}


def push(folder, repo, private, message):
    from huggingface_hub import CommitOperationAdd
    keep, _ = check(folder, repo); a = api()
    out("PROGRESS", {"stage": "creating the repo (if new)", "repo": repo})
    a.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    cut = "cut.json" in keep
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=_public_cut(folder) if n == "cut.json" else os.path.join(folder, n))
           for n in keep if not (cut and n == "README.md")]
    if cut or "README.md" not in keep: ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card(folder, repo).encode("utf-8")))
    size = sum(os.path.getsize(os.path.join(folder, n)) for n in keep)
    out("PROGRESS", {"stage": f"uploading {len(ops)} files ({size / 2**30:.2f} GB)", "files": len(ops)})
    c = a.create_commit(repo, ops, commit_message=(message or "Upload from the Parameter Space Inspector")[:200], repo_type="model")
    info = a.repo_info(repo, repo_type="model")
    return {"repo": repo, "url": f"https://huggingface.co/{repo}", "commit": c.oid, "commit_url": c.commit_url, "private": bool(getattr(info, "private", private)), "files": len(ops)}


def main(argv):
    try:
        cmd = argv[0] if argv else ""
        if cmd == "whoami": out("DONE", whoami())
        elif cmd == "plan": out("DONE", plan(os.path.abspath(argv[1]), argv[2]))
        elif cmd == "push":
            msg = argv[argv.index("--message") + 1] if "--message" in argv else ""
            out("DONE", push(os.path.abspath(argv[1]), argv[2], "--private" in argv, msg))
        else: raise ValueError("whoami | plan <folder> <repo> | push <folder> <repo> [--private] [--message M]")
    except Exception as e:
        m = str(e).replace("\n", " ")
        tok = os.environ.get("PSI_HF_TOKEN")
        if tok: m = m.replace(tok, "hf_…")
        if "401" in m or "Invalid user token" in m: m = "the token was refused (401): check it in Settings > Hugging Face"
        elif "403" in m: m = "the token may not write there (403): it needs write access to that account or organisation " + m[:200]
        out("ERROR", f"{type(e).__name__}: {m[:500]}" if not m.startswith("the token") else m)
        sys.exit(1)


if __name__ == "__main__": main(sys.argv[1:])
