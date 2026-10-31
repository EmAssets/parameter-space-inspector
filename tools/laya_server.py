"""A loopback HTTP front for Laya (convaiinnovations/laya, Apache 2.0): a non-generative decision model that answers typed
questions about a state with calibrated probabilities, in one pass, without writing text. PSI's hybrid service uses it as
a judge of a special kind: it can choose at a fork (one of the base's k pieces), choose a lettered option, and choose
between two answers, but it cannot write an answer of its own.
POST /choice  {"state": "...", "instructions": "...", "options": {"label": "description", ...}}
  → {"choice": "label", "probabilities": {"label": p, ...}, "confidence": p, "ms": n}
GET /health → {"ok": true, "model": ..., "subfolder": ...}
Usage: python tools/laya_server.py [--port 4322] [--checkpoint convaiinnovations/laya] [--subfolder multilingual] [--device mps]"""
import argparse, json, time, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=4322); ap.add_argument("--checkpoint", default="convaiinnovations/laya"); ap.add_argument("--subfolder", default=None); ap.add_argument("--device", default=None)
A = ap.parse_args()
import laya
t0 = time.time(); kw = {}
if A.subfolder: kw["subfolder"] = A.subfolder
if A.device: kw["device"] = A.device
agent = laya.load(A.checkpoint, **kw); print(f"loaded {A.checkpoint} {A.subfolder or ''} in {time.time() - t0:.1f}s", flush=True)


import threading
LOCK = threading.Lock()


def choose(state, instructions, options):
    q = {"q": {"type": "choice", "instructions": instructions, "criteria": options}}
    t = time.time()
    with LOCK: r = agent.predict(state, q)   # one model, one call at a time; the threads only keep slow clients apart
    a = r["answers"]["q"]
    probs = a.get("probabilities") or a.get("probs") or a.get("distribution") or {}
    if isinstance(probs, list) and len(probs) == len(options): probs = dict(zip(options.keys(), [float(x) for x in probs]))
    return {"choice": a.get("choice"), "probabilities": probs, "confidence": a.get("confidence"), "raw": {k: v for k, v in a.items() if k not in ("probabilities", "probs", "distribution")}, "ms": round((time.time() - t) * 1000)}


class H(BaseHTTPRequestHandler):
    timeout = 30   # a client that stops sending is dropped instead of holding the server
    def log_message(self, *a): pass
    # Programs on this computer only: a Host other than 127.0.0.1 / localhost (DNS rebinding) or any browser Origin is refused.
    def _local(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost") or self.headers.get("Origin"): self._json(403, {"error": "local programs only"}); return False
        return True
    def _json(self, code, o): b = json.dumps(o).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if not self._local(): return
        if self.path.startswith("/health"): return self._json(200, {"ok": True, "model": A.checkpoint, "subfolder": A.subfolder})
        self._json(404, {"error": "GET /health or POST /choice"})
    def do_POST(self):
        if not self._local(): return
        if not self.path.startswith("/choice"): return self._json(404, {"error": "POST /choice"})
        # A request is a short state and a few labelled options: anything past 1 MB, or not JSON, is refused unread.
        try: n = int(self.headers.get("Content-Length") or 0)
        except ValueError: return self._json(400, {"error": "bad Content-Length"})
        if n < 0 or n > 1 << 20: return self._json(413, {"error": "request too large (1 MB at most)"})
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError: return self._json(400, {"error": "bad json"})
        if not isinstance(body, dict): return self._json(400, {"error": "a JSON object"})
        try:
            opts = body.get("options") or {}
            if not isinstance(opts, dict) or len(opts) < 2 or len(opts) > 40: return self._json(400, {"error": "options: 2 to 40 labelled choices"})
            self._json(200, choose(str(body.get("state", ""))[:20000], str(body.get("instructions", "Which is right?"))[:2000], {str(k)[:40]: str(v)[:400] for k, v in opts.items()}))
        except Exception as e: self._json(500, {"error": str(e)[:300]})


srv = ThreadingHTTPServer(("127.0.0.1", A.port), H); print(f"laya on http://127.0.0.1:{A.port} (POST /choice)", flush=True)
try: srv.serve_forever()
except KeyboardInterrupt: sys.exit(0)
