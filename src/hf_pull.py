# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Pulling from Hugging Face (src/hf-api.js): models to test, their families, training checkpoints, SAEs, adapters,
dataset text and published PSI records. Run only when Settings > Hugging Face has a token that was checked.

  search <kind> <query> [--all] [--limit N]     kind model | dataset. By default only items tagged emergence-machine
                                                (what PSI publishes: models, recipes, branches, cut models); --all: everything
  info <repo> [--rev R]                         a model's files (size, SHA-256), licence, gating, architecture, whether the
                                                inside engine can read it, GGUF files, adapters, SAE folders, PSI records
  family <repo>                                 its base models and what was made from it (fine-tunes, quantised, adapters, merges)
  get <repo> <dest> --files a,b,.. [--rev R]    download those files at one commit; each file's SHA-256 checked against what
                                                Hugging Face publishes; hf-source.json records repo, commit and hashes
  refs <repo>                                   training-step revisions (step<N>) of a checkpointed model
  ckpt <repo> <step> <root>                     one checkpoint into <root>/step<N>
  dataset <repo> <dest> [--chars N]             text from a dataset (its first split, first text column) into a .txt, with its licence
  record <repo> <file>                          a small published record (cut.json, recipe.json, a PSI report), read as data
  gguf_base <file.gguf>                         which Hugging Face repo a GGUF says it was made from

Never pickles: .bin/.pt/.pth/.pkl/.ckpt are refused, and a model whose config needs its own code (auto_map) is refused.
Output: PSI-PROGRESS <json> lines, then PSI-DONE <json> or PSI-ERROR <message>. The token (PSI_HF_TOKEN) is never printed.
"""
import fnmatch, hashlib, json, os, re, sys, time, urllib.parse, urllib.request

EM_TAG = "emergence-machine"   # every PSI publication carries it, with psi-model / psi-cut / psi-recipe / psi-branch / psi-sae
REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")
UNSAFE = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".h5", ".msgpack", ".onnx", ".py")
MODEL_FILES = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "*.tiktoken",
               "chat_template.jinja", "chat_template.json", "LICENSE", "LICENSE.txt", "NOTICE", "README.md", "cut.json"]
# Architectures the inside engine reads as Llama-like (model.layers, mlp.down_proj, self_attn.o_proj); NeoX partly.
INSIDE = {"llama": "yes", "qwen2": "yes", "qwen3": "yes", "mistral": "yes", "gemma": "yes", "gemma2": "yes", "gemma3_text": "yes", "phi3": "yes",
          "olmo": "yes", "olmo2": "yes", "granite": "yes", "smollm3": "yes", "gpt_neox": "partly (views that need self_attn say so)",
          **{m: "no: a mixture-of-experts model (its MLP is many experts, not the one down_proj the inside views read); the answering engine runs it"
             for m in ("gpt_oss", "mixtral", "qwen2_moe", "qwen3_moe", "deepseek_v2", "deepseek_v3", "olmoe", "granitemoe", "granitemoehybrid", "dbrx", "jamba")},
          "lfm2": "no: hybrid convolution layers the inside views do not read; the answering engine runs it"}
PSI_RECORDS = ("cut.json", "recipe.json", "psi-report.json", "psi-identity.json", "lineage.json")


def out(kind, obj): print(f"PSI-{kind} " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)), flush=True)


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("PSI_HF_TOKEN") or None)


def repo_arg(r):
    if not REPO.fullmatch(r or ""): raise ValueError("a repo is OWNER/NAME")
    return r


def search(kind, query, every=False, limit=30):
    a = api(); flt = None if every else EM_TAG; q = (query or "").strip()[:80] or None
    rows = []
    if kind == "dataset":
        for d in a.list_datasets(search=q, filter=flt, sort="downloads", limit=limit):
            rows.append({"repo": d.id, "downloads": getattr(d, "downloads", None), "likes": getattr(d, "likes", None), "tags": (d.tags or [])[:12], "gated": bool(getattr(d, "gated", False))})
    else:
        for m in a.list_models(search=q, filter=flt, sort="downloads", limit=limit):
            rows.append({"repo": m.id, "downloads": getattr(m, "downloads", None), "likes": getattr(m, "likes", None), "pipeline": getattr(m, "pipeline_tag", None),
                         "library": getattr(m, "library_name", None), "tags": (m.tags or [])[:12], "gated": bool(getattr(m, "gated", False)), "em": EM_TAG in (m.tags or [])})
    return {"kind": kind, "query": q, "scope": "all of Hugging Face" if every else f"tagged {EM_TAG}", "items": rows}


def _sha_of(s):
    lfs = getattr(s, "lfs", None)
    if lfs is None: return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def info(repo, rev=None):
    a = api(); m = a.model_info(repo_arg(repo), revision=rev, files_metadata=True)
    files = [{"name": s.rfilename, "size": s.size, "sha256": _sha_of(s)} for s in (m.siblings or [])]
    names = [f["name"] for f in files]
    card = getattr(m, "card_data", None) or getattr(m, "cardData", None) or {}
    card = card.to_dict() if hasattr(card, "to_dict") else dict(card or {})
    cfg = getattr(m, "config", None) or {}
    mt = (cfg.get("model_type") or "") if isinstance(cfg, dict) else ""
    remote_code = bool(isinstance(cfg, dict) and cfg.get("auto_map"))
    st = [f for f in files if f["name"].endswith(".safetensors") and "/" not in f["name"]]
    gguf = [f for f in files if f["name"].lower().endswith(".gguf")]
    adapters = [f for f in gguf if re.search(r"lora|adapter|cvec|control", f["name"], re.I)]
    sae_dirs = sorted({os.path.dirname(n) for n in names if os.path.basename(n) in ("cfg.json",) and any(x.startswith(os.path.dirname(n) + "/") and x.endswith((".safetensors", ".npz")) for x in names)})
    folder = [f for f in files if "/" not in f["name"] and any(fnmatch.fnmatch(f["name"], p) for p in MODEL_FILES)]
    base = card.get("base_model"); base = [base] if isinstance(base, str) else list(base or [])
    unsafe = [n for n in names if n.lower().endswith(UNSAFE)]
    return {"repo": m.id, "commit": m.sha, "gated": getattr(m, "gated", False) or False, "private": bool(getattr(m, "private", False)),
            "licence": card.get("license") or next((t[8:] for t in (m.tags or []) if t.startswith("license:")), None),
            "tags": m.tags or [], "em": EM_TAG in (m.tags or []), "pipeline": getattr(m, "pipeline_tag", None), "library": getattr(m, "library_name", None),
            "model_type": mt or None, "architectures": (cfg.get("architectures") if isinstance(cfg, dict) else None),
            "inside_engine": ("no: it needs its own code (auto_map), which PSI never runs" if remote_code else INSIDE.get(mt, f"unknown ({mt or 'no model_type'}): it may not load") if st else "no safetensors weights"),
            "remote_code": remote_code,
            "folder": {"files": [f["name"] for f in folder], "bytes": sum(f["size"] or 0 for f in folder)} if st else None,
            "gguf": [{"name": f["name"], "bytes": f["size"]} for f in gguf if f not in adapters],
            "adapters": [{"name": f["name"], "bytes": f["size"]} for f in adapters],
            "peft": "adapter_config.json" in names, "sae_folders": sae_dirs[:200],
            "records": [n for n in names if os.path.basename(n) in PSI_RECORDS],
            "base_models": base, "unsafe_files_skipped": unsafe[:20], "downloads": getattr(m, "downloads", None), "likes": getattr(m, "likes", None)}


def family(repo):
    a = api(); repo_arg(repo)
    try: base = info(repo)["base_models"]
    except Exception: base = []
    made = {}
    for rel in ("finetune", "quantized", "adapter", "merge"):
        try: made[rel] = [{"repo": m.id, "downloads": getattr(m, "downloads", None), "em": EM_TAG in (m.tags or [])}
                          for m in a.list_models(filter=f"base_model:{rel}:{repo}", sort="downloads", limit=15)]
        except Exception as e: made[rel] = {"error": str(e)[:200]}
    return {"repo": repo, "base_models": base, "made_from_it": made}


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def get(repo, dest, files, rev=None, prefix=None):
    from huggingface_hub import hf_hub_download
    a = api(); m = a.model_info(repo_arg(repo), revision=rev, files_metadata=True); commit = m.sha
    meta = {s.rfilename: {"size": s.size, "sha256": _sha_of(s)} for s in (m.siblings or [])}
    files = [f for f in files if f]
    if prefix:   # every file under one folder of the repo (an SAE), pickles and code left out
        files = [n for n in meta if n.startswith(prefix.rstrip("/") + "/") and not n.lower().endswith(UNSAFE)]
        if not files: raise ValueError(f"no files under {prefix} in {repo}")
    for f in files:
        if f not in meta: raise ValueError(f"{f} is not in {repo} at {commit[:12]}")
        if f.lower().endswith(UNSAFE): raise ValueError(f"{f}: PSI does not download pickle or code files")
        if ".." in f.split("/"): raise ValueError(f"{f}: not a plain file name")
    total = sum(meta[f]["size"] or 0 for f in files); done = 0; rec = {}
    # Staged in <dest>.partial and moved into <dest> only when every file is in and checked: a stopped or failed download
    # leaves nothing half-made where models are looked for (the app deletes the .partial folder).
    final, dest = dest, dest + ".partial"
    os.makedirs(dest, exist_ok=True)
    def on_disk():   # bytes written so far under the staging folder, the download's own .incomplete files included
        n = 0
        for root, _, names in os.walk(dest):
            for x in names:
                try: n += os.path.getsize(os.path.join(root, x))
                except OSError: pass
        return n
    for i, f in enumerate(files):
        stage = f"file {i + 1} of {len(files)}: {f}"
        out("PROGRESS", {"stage": stage, "done": done, "of": total})
        # Progress within a file (one GGUF can be 12 GB): what has reached the disk, every second, while it downloads.
        import threading
        stop, base = threading.Event(), on_disk()
        def watch():
            while not stop.wait(1.0): out("PROGRESS", {"stage": stage, "done": done + max(0, min(on_disk() - base, meta[f]["size"] or 0)), "of": total})
        th = threading.Thread(target=watch, daemon=True); th.start()
        try: p = hf_hub_download(repo, f, revision=commit, local_dir=dest, token=os.environ.get("PSI_HF_TOKEN") or None)
        finally: stop.set(); th.join(2)
        want = meta[f]["sha256"]
        if want: out("PROGRESS", {"stage": f"checking {f} (SHA-256)", "done": done + (meta[f]["size"] or 0), "of": total})
        got = _sha256(p) if (want or (meta[f]["size"] or 0) < (64 << 20)) else None
        if want and got != want:
            os.remove(p); raise ValueError(f"{f}: its SHA-256 does not match what Hugging Face publishes; the file was deleted")
        rec[f] = {"bytes": os.path.getsize(p), "sha256": got, "checked_against_hub": bool(want)}
        done += meta[f]["size"] or 0
    import shutil
    shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)
    for f in files:   # into the final folder; a later download into the same folder (another GGUF) adds to it
        to = os.path.join(final, f); os.makedirs(os.path.dirname(to), exist_ok=True); os.replace(os.path.join(dest, f), to)
    shutil.rmtree(dest, ignore_errors=True); dest = final
    old = {}
    try: old = json.load(open(os.path.join(dest, "hf-source.json"), encoding="utf-8")) if os.path.exists(os.path.join(dest, "hf-source.json")) else {}
    except Exception: old = {}
    if old.get("commit") == commit: rec = {**(old.get("files") or {}), **rec}
    src = {"repo": repo, "commit": commit, "url": f"https://huggingface.co/{repo}/tree/{commit}", "files": rec,
           "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "by": "Parameter Space Inspector"}
    with open(os.path.join(dest, "hf-source.json"), "w", encoding="utf-8") as fh: json.dump(src, fh, indent=1)
    return {"repo": repo, "commit": commit, "dest": dest.replace("\\", "/"), "folder": (os.path.join(dest, prefix) if prefix else dest).replace("\\", "/"), "files": len(rec), "bytes": done}


def _step_revs(repo):
    """Training-step revisions: Pythia names them step<N>, OLMo stage1-step<N>-tokens<T>B (the first stage only, so a
    step number means one point in training)."""
    r = api().list_repo_refs(repo_arg(repo)); revs = {}; other = []
    for br in r.branches:
        m = re.fullmatch(r"(?:stage1-)?step(\d+)(?:-tokens\d+B)?", br.name)
        if m: revs[int(m.group(1))] = br.name
        else: other.append(br.name)
    return revs, sorted(other)


def refs(repo):
    revs, other = _step_revs(repo)
    return {"repo": repo, "steps": sorted(revs), "revisions": {str(k): v for k, v in sorted(revs.items())}, "other": other[:40]}


def ckpt(repo, step, root):
    revs, _ = _step_revs(repo); step = int(step)
    if step not in revs: raise ValueError(f"{repo} has no checkpoint at step {step}")
    rev = revs[step]; i = info(repo, rev)
    if not i["folder"]: raise ValueError(f"{repo} at {rev} has no safetensors weights")
    dest = os.path.join(root, f"step{step}")
    if os.path.exists(os.path.join(dest, "config.json")): return {"repo": repo, "step": step, "dest": dest.replace("\\", "/"), "already": True}
    r = get(repo, dest, i["folder"]["files"], rev)
    return {**r, "step": step, "revision": rev, "dest": dest.replace("\\", "/")}


DS = "https://datasets-server.huggingface.co"


def _ds(path, **q):
    url = f"{DS}/{path}?" + urllib.parse.urlencode(q)
    h = {"User-Agent": "parameter-space-inspector"}
    tok = os.environ.get("PSI_HF_TOKEN")
    if tok: h["Authorization"] = f"Bearer {tok}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r: return json.loads(r.read(32 << 20).decode())


def dataset(repo, dest, chars=200000):
    repo_arg(repo); chars = max(1000, min(5_000_000, int(chars)))
    sp = (_ds("splits", dataset=repo).get("splits") or [])
    if not sp: raise ValueError("the dataset viewer has no splits for it (it may need its own loading code, or be gated)")
    s = next((x for x in sp if x["split"] == "train"), sp[0]); cfg, split = s["config"], s["split"]
    first = _ds("first-rows", dataset=repo, config=cfg, split=split)
    cols = [f["name"] for f in first.get("features", []) if (f.get("type") or {}).get("dtype") == "string"]
    if not cols: raise ValueError("no text column in its first split")
    col = next((c for c in cols if c.lower() in ("text", "content", "story", "document", "body")), cols[0])
    parts, n, off = [], 0, 0
    while n < chars:
        rows = _ds("rows", dataset=repo, config=cfg, split=split, offset=off, length=100).get("rows") or []
        if not rows: break
        for r in rows:
            t = str((r.get("row") or {}).get(col) or "").strip()
            if t: parts.append(t); n += len(t) + 2
        off += len(rows); out("PROGRESS", {"stage": f"{n:,} characters", "done": min(n, chars), "of": chars})
    text = "\n\n".join(parts)[:chars]
    try: lic = (api().dataset_info(repo).card_data or {}).get("license")
    except Exception: lic = None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh: fh.write(text)
    return {"repo": repo, "config": cfg, "split": split, "column": col, "chars": len(text), "file": dest.replace("\\", "/"),
            "licence": lic if isinstance(lic, str) else (lic[0] if isinstance(lic, list) and lic else None), "url": f"https://huggingface.co/datasets/{repo}"}


def record(repo, f):
    if os.path.basename(f) not in PSI_RECORDS: raise ValueError("only PSI records are read this way")
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_arg(repo), f, token=os.environ.get("PSI_HF_TOKEN") or None)
    if os.path.getsize(p) > (8 << 20): raise ValueError("too large for a record")
    return {"repo": repo, "file": f, "record": json.load(open(p, encoding="utf-8"))}


def gguf_base(path):
    from gguf import GGUFReader
    r = GGUFReader(path); kv = {}
    for k, fld in r.fields.items():
        if not k.startswith("general."): continue
        try: kv[k] = bytes(fld.parts[fld.data[0]]).decode("utf-8", "replace") if fld.types and fld.types[0].name == "STRING" else None
        except Exception: pass
    cands, how = [], None
    stem = re.sub(r"([-_.](q\d(_[0-9a-z]+)*|iq\d\w*|f16|f32|bf16|fp16|gguf))+$", "", os.path.basename(path)[:-5], flags=re.I)
    # 1. Downloaded here: the GGUF repo it came from (hf-source.json) names the weights it was converted from in its
    #    card (base_model), which is the same model; failing that, the repo's name without "-GGUF".
    try: src = json.load(open(os.path.join(os.path.dirname(path), "hf-source.json"), encoding="utf-8")).get("repo")
    except Exception: src = None
    if src and REPO.fullmatch(src):
        try:
            bm = (api().model_info(src).card_data or {}).get("base_model")
            for b in ([bm] if isinstance(bm, str) else bm or []):
                if REPO.fullmatch(b) and b not in cands: cands.append(b)
        except Exception: pass
        plain = re.sub(r"[-_.]gguf$", "", src, flags=re.I)
        if plain != src and plain not in cands:
            try: api().model_info(plain); cands.append(plain)
            except Exception: pass
        if cands: how = f"named by {src}, the repo it was downloaded from"
    # 2. Named in the file. general.base_model is often the model this one was trained FROM (a base for an instruct
    #    model): other weights, so it comes after the source and says so.
    if not cands:
        for k in ("general.source.repo_url", "general.source.url", "general.source.huggingface.repository", "general.repo_url"):
            v = kv.get(k) or ""
            m = re.search(r"huggingface\.co/([\w.-]+/[\w.-]+)", v) or (re.fullmatch(r"([\w.-]+/[\w.-]+)", v) if v else None)
            if m and m.group(1) not in cands: cands.append(m.group(1))
        if cands: how = "named in the file"
    base = []
    m = re.search(r"huggingface\.co/([\w.-]+/[\w.-]+)", kv.get("general.base_model.0.repo_url") or "")
    if m: base.append(m.group(1))
    org, nm = kv.get("general.base_model.0.organization"), kv.get("general.base_model.0.name")
    if org and nm and REPO.fullmatch(f"{org}/{nm}".replace(" ", "-")): base.append(f"{org}/{nm}".replace(" ", "-"))
    base = [b for b in dict.fromkeys(base) if b not in cands]
    if not cands and base:
        cands, how = base, "the base model named in the file: possibly the model this one was trained from, so other weights; check the layers match"
    elif base: cands += base
    if not cands:   # nothing recorded: search by the file's model name (quantisation suffix removed), safetensors repos only
        try: cands = [m.id for m in api().list_models(search=stem, filter="safetensors", sort="downloads", limit=5)]
        except Exception: cands = []
        how = f"found by its name ({stem}): check it is the same model"
    nm = (kv.get("general.name") or "").strip()
    if not nm or re.fullmatch(r"[0-9a-f]{16,}", nm, flags=re.I): nm = stem   # some publishers put an id there
    return {"file": path.replace("\\", "/"), "name": nm, "candidates": cands, "how": how}


def main(argv):
    try:
        c = argv[0] if argv else ""; opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
        if c == "search": r = search(argv[1], argv[2] if len(argv) > 2 and not argv[2].startswith("--") else "", "--all" in argv, int(opt("--limit", 30)))
        elif c == "info": r = info(argv[1], opt("--rev"))
        elif c == "family": r = family(argv[1])
        elif c == "get": r = get(argv[1], os.path.abspath(argv[2]), (opt("--files", "") or "").split(","), opt("--rev"), opt("--prefix"))
        elif c == "refs": r = refs(argv[1])
        elif c == "ckpt": r = ckpt(argv[1], int(argv[2]), os.path.abspath(argv[3]))
        elif c == "dataset": r = dataset(argv[1], os.path.abspath(argv[2]), int(opt("--chars", 200000)))
        elif c == "record": r = record(argv[1], argv[2])
        elif c == "gguf_base": r = gguf_base(argv[1])
        else: raise ValueError("search | info | family | get | refs | ckpt | dataset | record | gguf_base")
        out("DONE", r)
    except Exception as e:
        m = str(e).replace("\n", " "); tok = os.environ.get("PSI_HF_TOKEN")
        if tok: m = m.replace(tok, "hf_…")
        if "GatedRepo" in type(e).__name__ or "gated" in m.lower() and "403" in m: m = "this repo is gated: accept its terms on its Hugging Face page (signed in as the token's account), then try again"
        elif "401" in m: m = "Hugging Face refused the token (401): check it in Settings > Hugging Face"
        elif "RepositoryNotFound" in type(e).__name__ or "404" in m: m = "not found on Hugging Face (or private and the token cannot read it)"
        out("ERROR", m[:500]); sys.exit(1)


if __name__ == "__main__": main(sys.argv[1:])
