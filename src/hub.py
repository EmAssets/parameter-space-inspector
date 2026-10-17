# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Where ingredients live: any host, checked by hash, never trusted by address.

Addresses a recipe may use (anything else is refused):
  hf://OWNER/REPO[@REV]/PATH            a Hugging Face model repo  (hf://datasets/OWNER/REPO@REV/PATH for a dataset)
  https://huggingface.co/...            the same, written out
  https://github.com/O/R/blob/REF/PATH  a GitHub file (fetched from raw.githubusercontent.com)
  https://ANY.HOST/PATH                 any web host (a personal site, a mirror)
  store://SHA256                        this computer's own store (<data>/commons/store)
  an absolute local path                a file the user has (offline use)
Plain http, and web addresses that resolve to this computer or the local network, are refused: a recipe from a
stranger must not be able to make PSI knock on the router or a local service.

Every file is downloaded to a .part file, hashed as it arrives, kept only when the SHA-256 matches, and stopped at
its declared size. The Hugging Face token (PSI_HF_TOKEN, given by the app only for a private read or a push the user
confirmed) is sent to huggingface.co and nowhere else.
"""
import hashlib, ipaddress, json, os, re, socket, urllib.parse, urllib.request

UA = {"User-Agent": "psi-commons/1"}
HF = "https://huggingface.co"


def data_dir():
    return os.environ.get("PSI_DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".userdata")


def store_dir():
    d = os.path.join(data_dir(), "commons", "store"); os.makedirs(d, exist_ok=True); return d


def store_path(sha): return os.path.join(store_dir(), f"{sha}.safetensors")


class Refused(Exception): pass


def resolve(addr):
    """('local', path) or ('web', https url)."""
    a = str(addr or "").strip()
    if a.startswith("store://"):
        h = a[8:]
        if not re.fullmatch(r"[0-9a-f]{64}", h): raise Refused(f"not a store address: {a[:80]}")
        return "local", store_path(h)
    if a.startswith("hf://"):
        rest = a[5:]; kind = ""
        if rest.startswith("datasets/"): kind, rest = "datasets/", rest[9:]
        m = re.fullmatch(r"([\w.-]+)/([\w.-]+)(?:@([\w.-]+))?/(.+)", rest)
        if not m: raise Refused(f"not a Hugging Face address: {a[:80]}")
        o, r, rev, p = m.groups()
        return "web", f"{HF}/{kind}{o}/{r}/resolve/{rev or 'main'}/{urllib.parse.quote(p)}"
    m = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+)/(?:blob|raw)/([^/]+)/(.+)", a)
    if m: return "web", f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}"
    if a.startswith("https://"): return "web", a
    if a.startswith("http://"): raise Refused("plain http is refused (anyone on the way could change the file): use https")
    if os.path.isabs(a): return "local", a
    raise Refused(f"not an address PSI reads: {a[:80]}")


def _public_host(url):
    host = urllib.parse.urlparse(url).hostname or ""
    if os.environ.get("PSI_ALLOW_LOCAL_HOSTS") == "1": return
    try: infos = socket.getaddrinfo(host, 443)
    except socket.gaierror: raise Refused(f"{host}: no such host")
    for i in infos:
        ip = ipaddress.ip_address(i[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise Refused(f"{host} is on this computer or the local network: refused for addresses that come from recipes")


def _req(url, headers=None):
    h = dict(UA, **(headers or {}))
    tok = os.environ.get("PSI_HF_TOKEN")
    if tok and urllib.parse.urlparse(url).hostname == "huggingface.co": h["Authorization"] = f"Bearer {tok}"
    return urllib.request.Request(url, headers=h)


class _NoForeignRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects (Hugging Face sends files to its CDN), but check every new host, and drop the token off huggingface.co."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"): raise Refused("redirected to a non-https address")
        _public_host(newurl)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).hostname != "huggingface.co": new.headers.pop("Authorization", None); new.unredirected_hdrs.pop("Authorization", None)
        return new


OPENER = urllib.request.build_opener(_NoForeignRedirect)


def probe(addr, timeout=20):
    """Is it there, and how large? Reads one byte (a Range request), never the file."""
    try:
        kind, where = resolve(addr)
        if kind == "local": return {"addr": addr, "ok": os.path.exists(where), "size": os.path.getsize(where) if os.path.exists(where) else None, "where": "this computer"}
        _public_host(where)
        with OPENER.open(_req(where, {"Range": "bytes=0-0"}), timeout=timeout) as r:
            cr = r.headers.get("Content-Range") or ""; size = int(cr.split("/")[-1]) if "/" in cr and cr.split("/")[-1].isdigit() else int(r.headers.get("Content-Length") or 0) or None
            return {"addr": addr, "ok": r.status in (200, 206), "size": size, "where": urllib.parse.urlparse(where).hostname}
    except Refused as e: return {"addr": addr, "ok": False, "why": str(e)}
    except Exception as e: return {"addr": addr, "ok": False, "why": f"{type(e).__name__}: {str(e)[:120]}"}


def fetch(addrs, sha256, dest, max_bytes, progress=None, timeout=60):
    """Get a file from the first address that gives the right bytes. Returns which address worked."""
    tried = []
    for addr in addrs:
        try:
            kind, where = resolve(addr)
            if kind == "local":
                if not os.path.exists(where): tried.append(f"{addr}: not here"); continue
                h = sha_file(where)
                if h != sha256: tried.append(f"{addr}: wrong bytes ({h[:12]})"); continue
                if os.path.abspath(where) != os.path.abspath(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True); import shutil; shutil.copyfile(where, dest)
                return {"from": addr, "tried": tried}
            _public_host(where)
            tmp = dest + ".part"; h = hashlib.sha256(); n = 0; os.makedirs(os.path.dirname(dest), exist_ok=True)
            with OPENER.open(_req(where), timeout=timeout) as r, open(tmp, "wb") as o:
                while True:
                    b = r.read(1 << 16)
                    if not b: break
                    n += len(b)
                    if n > max_bytes: raise Refused(f"larger than the {max_bytes:,} bytes declared")
                    o.write(b); h.update(b)
                    if progress: progress(n)
            if h.hexdigest() != sha256: os.remove(tmp); tried.append(f"{addr}: wrong bytes ({h.hexdigest()[:12]})"); continue
            os.replace(tmp, dest); return {"from": addr, "tried": tried}
        except Exception as e:
            try: os.remove(dest + ".part")
            except OSError: pass
            tried.append(f"{addr}: {str(e)[:160]}")
    raise Refused("no address gave the right file: " + "; ".join(tried))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def get_json(addr, max_bytes=4 << 20, timeout=30):
    kind, where = resolve(addr)
    if kind == "local":
        if os.path.getsize(where) > max_bytes: raise Refused("larger than a recipe or record should be")
        return json.load(open(where, encoding="utf-8"))
    _public_host(where)
    with OPENER.open(_req(where), timeout=timeout) as r:
        b = r.read(max_bytes + 1)
    if len(b) > max_bytes: raise Refused("larger than a recipe or record should be")
    return json.loads(b.decode("utf-8"))


# ── Hugging Face: browsing, profiles, pushing ──
SORTS = ("downloads", "likes", "trendingScore", "lastModified", "createdAt")


def browse(search="", tag="psi-recipe", limit=40, sort="downloads"):
    """Public repos with the tag, in Hugging Face's own order (its downloads are the last 30 days)."""
    q = {"filter": tag, "limit": str(max(1, min(100, int(limit)))), "sort": sort if sort in SORTS else "downloads", "direction": "-1"}
    if search: q["search"] = str(search)[:80]
    with OPENER.open(_req(f"{HF}/api/models?" + urllib.parse.urlencode(q)), timeout=30) as r:
        rows = json.loads(r.read(4 << 20).decode())
    return [{"repo": m.get("id"), "likes": m.get("likes"), "downloads": m.get("downloads"), "trending": m.get("trendingScore"), "tags": m.get("tags", []),
             "created": m.get("createdAt"), "modified": m.get("lastModified") or m.get("createdAt"),
             "recipe": f"hf://{m.get('id')}/recipe.json"} for m in rows if isinstance(m, dict) and m.get("id")]


def hf_user(user):
    if not re.fullmatch(r"[\w.-]{1,96}", str(user or "")): raise Refused("not a Hugging Face user name")
    with OPENER.open(_req(f"{HF}/api/users/{user}/overview"), timeout=20) as r: o = json.loads(r.read(1 << 20).decode())
    return {"username": user, "display_name": o.get("fullname") or user, "avatar": o.get("avatarUrl"), "orgs": [x.get("name") for x in o.get("orgs", []) if isinstance(x, dict)][:20],
            "links": [{"host": "huggingface", "user": user, "url": f"{HF}/{user}"}], "imported_from": "huggingface", "models": o.get("numModels"), "datasets": o.get("numDatasets")}


def identity_addrs(link):
    """Where a person's published keys are read from: a psi-identity.json in a repo named psi-identity they own."""
    u = (link or {}).get("user") or ""
    if (link or {}).get("host") == "huggingface": return [f"hf://{u}/psi-identity/psi-identity.json"]
    if (link or {}).get("host") == "github": return [f"https://raw.githubusercontent.com/{u}/psi-identity/main/psi-identity.json"]
    return []


def push(repo, files, message, create=True, private=False):
    """Upload files {path in repo: local path} to a Hugging Face model repo in one commit. The token comes only from
    PSI_HF_TOKEN (the app gives it for a push the user confirmed). Returns the commit (its time is the public log entry)."""
    tok = os.environ.get("PSI_HF_TOKEN")
    if not tok: raise Refused("no Hugging Face token: add one in Settings > Hugging Face (a fine-grained token, write access to the repos you push to)")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""): raise Refused("a repo is OWNER/NAME")
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=tok)
    if create: api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=l) for p, l in files.items()]
    c = api.create_commit(repo, ops, commit_message=message[:200], repo_type="model")
    info = api.model_info(repo, revision=c.oid)
    return {"repo": repo, "commit": c.oid, "url": c.commit_url, "time": str(info.last_modified) if getattr(info, "last_modified", None) else None,
            "private": bool(getattr(info, "private", private))}
