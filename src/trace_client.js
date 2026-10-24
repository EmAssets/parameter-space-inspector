// SPDX-License-Identifier: Apache-2.0
// Starts and talks to the trace engine (src/trace_server.py): the PyTorch side that
// exposes every layer while the model answers. Shared by the app and the dev server.
'use strict';
const { python: PY } = require('./python.js');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

// The trace server's token (trace_server.py refuses POSTs without it). Shared by the app and the dev server, which
// reuse one trace server on the same port: $PSI_TRACE_TOKEN, else a file in the git-ignored .cache/, made once.
function traceToken() {
  if (process.env.PSI_TRACE_TOKEN) return process.env.PSI_TRACE_TOKEN;
  const f = path.join(__dirname, '..', '.cache', 'trace-token');
  try { const t = fs.readFileSync(f, 'utf8').trim(); if (t) return t; } catch (_) { /* first run */ }
  const t = crypto.randomBytes(32).toString('hex'); fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, t, { mode: 0o600 }); return t;
}

// Which weights a folder holds, the way trace_server.py says it: sha256 of model.safetensors.index.json, or of the first
// 1 MB of the single weights file. null when the folder has neither.
function weightsHash(dir) {
  try {
    const idx = path.join(dir, 'model.safetensors.index.json');
    if (fs.existsSync(idx)) return crypto.createHash('sha256').update(fs.readFileSync(idx)).digest('hex');
    const f = fs.readdirSync(dir).sort().find((n) => n.endsWith('.safetensors') || (n.endsWith('.bin') && n.includes('model')));
    if (!f) return null;
    const fd = fs.openSync(path.join(dir, f), 'r'); const b = Buffer.alloc(1 << 20);
    try { const n = fs.readSync(fd, b, 0, b.length, 0); return crypto.createHash('sha256').update(b.subarray(0, n)).digest('hex'); } finally { fs.closeSync(fd); }
  } catch (_) { return null; }
}
const samePath = (a, b) => { const n = (x) => path.resolve(String(x || '')).replace(/\\/g, '/').replace(/\/+$/, ''); return process.platform === 'win32' ? n(a).toLowerCase() === n(b).toLowerCase() : n(a) === n(b); };

class TraceClient {
  constructor({ modelDir = process.env.PSI_TRACE_MODEL || '', port = +(process.env.PSI_TRACE_PORT || 4311), log = () => {}, lensSetting = null } = {}) {
    this.modelDir = modelDir; this.port = port; this.log = log; this.proc = null; this.token = traceToken(); this.lensSetting = lensSetting;
  }
  get headers() { return { 'Content-Type': 'application/json', 'X-PSI-Token': this.token }; }
  get url() { return `http://127.0.0.1:${this.port}`; }
  async alive() { try { const r = await fetch(this.url, { signal: AbortSignal.timeout(1500) }); return r.ok ? await r.json() : null; } catch (_) { return null; } }
  // GET /state of whatever server answers on the port (GET / on an older trace_server.py, which has no conditions).
  // The reply says proven: true only when the server answered the nonce with an HMAC under this app's token, i.e. it is
  // the app's own engine and not some program that took the port first. The token is sent only to a proven server.
  async state() {
    for (const route of ['/state', '/']) {
      try {
        const nonce = crypto.randomBytes(16).toString('hex');
        const r = await fetch(`${this.url}${route}`, { headers: { 'X-PSI-Nonce': nonce }, signal: AbortSignal.timeout(1500) });
        if (!r.ok) continue;
        const st = await r.json(), want = crypto.createHmac('sha256', this.token).update(`psi-trace:${nonce}`).digest('hex');
        const proven = typeof st.proof === 'string' && st.proof.length === want.length && crypto.timingSafeEqual(Buffer.from(st.proof), Buffer.from(want));
        delete st.proof; return Object.assign(st, { proven });
      } catch (_) { return null; }
    }
    return null;
  }
  unproven() { return new Error(`Something answers on port ${this.port} but cannot prove it is this app's inside engine (an older trace_server.py, or another program). The app's token was not sent to it. Stop it, or set PSI_TRACE_PORT to another port.`); }
  // Does the live server hold the model that was asked for? {ok, note, live_model_dir}. ok null: it cannot say
  // (an older trace_server.py that does not report its model).
  verify(st, dir = this.modelDir) {
    const c = st && st.conditions;
    if (!c || !c.model_dir) return { ok: null, note: `The inside engine on port ${this.port} does not report which model it holds (an older trace_server.py); it may not be ${path.basename(dir)}.` };
    if (!samePath(c.model_dir, dir)) return { ok: false, live_model_dir: c.model_dir, note: `A trace server already running on port ${this.port} holds ${c.model_dir}, not ${dir}. This app did not start it, so it was left running; stop it or change PSI_TRACE_PORT.` };
    const want = weightsHash(dir);
    if (want && c.weights_hash && want !== c.weights_hash) return { ok: false, live_model_dir: c.model_dir, note: `The inside engine on port ${this.port} loaded ${c.model_dir}, but the weights in that folder have changed since it loaded them (weights_hash differs). Restart it to read the current weights.` };
    return { ok: true, live_model_dir: c.model_dir, note: null };
  }
  async ensure(onStatus = () => {}) {
    const a = await this.state();
    if (a) {
      if (!a.proven) throw this.unproven();
      const v = this.verify(a); this.note = v.note;
      if (v.ok === false) { const e = new Error(v.note); e.live_model_dir = v.live_model_dir; throw e; }
      return a;
    }
    if (!this.modelDir) throw new Error('The inside engine has no weights folder yet: set it in Settings (the inside engine’s weights folder, a Hugging Face model folder) or PSI_TRACE_MODEL.');
    if (!fs.existsSync(this.modelDir)) throw new Error(`No unpacked weights at ${this.modelDir}. Run research/convert_bf16.py once.`);
    if (!this.proc) {
      onStatus('starting the trace engine (PyTorch)…');
      const script = path.join(__dirname, 'trace_server.py');
      this.proc = spawn(PY(), ['-u', script, '--model', this.modelDir, '--port', String(this.port)], { env: require('./config.js').cacheEnv({ PSI_TRACE_TOKEN: this.token, PSI_SAE_DIR: path.join(require('./config.js').DATA, 'sae'), PSI_OVERRIDE_DIR: path.join(require('./config.js').DATA, 'override-probes') }), windowsHide: true });
      this.proc.stdout.on('data', (d) => this.log(String(d).trim()));
      this.proc.stderr.on('data', (d) => { const s = String(d).trim(); if (s && !/Loading weights|Warning|warn\(/.test(s)) this.log(s); });
      const p = this.proc; p.on('exit', () => { if (this.proc === p) this.proc = null; });   // a late exit of a replaced engine must not clear the new one
    }
    const deadline = Date.now() + 10 * 60 * 1000;
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500));
      const b = await this.state();
      if (b) { if (!b.proven) throw this.unproven(); const v = this.verify(b); this.note = v.note; if (v.ok === false) throw new Error(v.note); return b; }
      if (!this.proc) throw new Error('The trace engine stopped while loading.');
      // Stuck, not slow: stop it so the next request starts a fresh engine instead of waiting on this one.
      if (Date.now() > deadline) { try { this.proc.kill('SIGKILL'); } catch (_) { /* gone */ } this.proc = null; throw new Error('The trace engine did not come up in 10 minutes and was stopped. Its log is in the app’s console; on Apple Silicon a PyTorch call that hangs on MPS looks like this (PSI_DEVICE=cpu runs the inside engine on the CPU instead).'); }
      onStatus('loading the model into the trace engine (a minute or two the first time)…');
    }
  }
  async trace(prompt, opts = {}) {
    const r = await fetch(`${this.url}/trace`, { method: 'POST', headers: this.headers, body: JSON.stringify({ prompt, n_new: opts.n_new ?? 16, lens: this.lensOf(opts.lens ?? true), think: opts.think ?? false }) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `trace: HTTP ${r.status}`);
    return data;
  }
  async embed(text, k = 8) {
    const r = await fetch(`${this.url}/embed`, { method: 'POST', headers: this.headers, body: JSON.stringify({ text, k }) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `embed: HTTP ${r.status}`);
    return data;
  }
  async understand(items) { return this._post('/understand', { items }); }
  async represent(text) { return this._post('/represent', { text }); }
  async vectors(args) { return this._post('/vectors', args || {}); }
  async sae(args) { return this._post('/sae', args || {}); }
  async override(args) { return this._post('/override', args || {}); }
  async context(args) { return this._post('/context', args || {}); }
  async attractor(args) { return this._post('/attractor', args || {}); }
  async scoreText(args) { return this._post('/score_text', args || {}); }
  async ablate(args) { return this._post('/ablate', args); }
  async ckptList() { return this._post('/ckpt_list', {}); }
  async ckptCount(args) { return this._post('/ckpt_count', args || {}); }
  async ckptAsk(args) { return this._post('/ckpt_ask', args); }
  async ckptTest(args) { return this._post('/ckpt_test', args); }
  async heads(args) { return this._post('/heads', args || {}); }
  async ablateHeads(args) { return this._post('/ablate_heads', args || {}); }
  async lesion(args) { return this._post('/lesion', args || {}); }
  async sense(args) { return this._post('/sense', args || {}); }
  async fight(args) { return this._post('/fight', args || {}); }
  async connector(args) { return this._post('/connector', args || {}); }
  async anatomy(args) { return this._post('/anatomy', args || {}); }
  // The lens as sent to the engine: the user's own text when the lens is asked for (Settings > Thinking in E-levels),
  // and never while Emergence Machine content is off (Settings > Content). The engine's default is the E-level lens.
  lensOf(v) { const l = this.lensSetting ? this.lensSetting() : { em: true, text: '' }; if (!v || !l.em) return false; return typeof v === 'string' ? v : (l.text || true); }
  async _post(route, body) {
    if (body && (route === '/ablate' || route === '/anatomy')) body = { ...body, lens: this.lensOf(body.lens ?? true) };
    const r = await fetch(`${this.url}${route}`, { method: 'POST', headers: this.headers, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `${route}: HTTP ${r.status}`);
    return data;
  }
  async stop() {
    if (this.proc) { try { this.proc.kill(); } catch (_) { /* gone */ } this.proc = null; }
    for (let i = 0; i < 25 && (await this.alive()); i++) await new Promise((r) => setTimeout(r, 200));   // let the old server free its port
  }
  // Only a server this app started is stopped. If another one still answers on the port afterwards, the result says
  // what it holds (note, live_model_dir), and the next request fails with the same note unless it holds the model asked for.
  async restart({ modelDir, checkpoints } = {}) {
    await this.stop(); if (modelDir) this.modelDir = modelDir; if (checkpoints) process.env.PSI_CHECKPOINTS = checkpoints;
    const out = { modelDir: this.modelDir, checkpoints: process.env.PSI_CHECKPOINTS || '', note: null };
    const st = await this.state();
    if (st) { const v = this.verify(st); out.note = v.note; out.live_model_dir = v.live_model_dir || null; out.matches = v.ok; this.note = v.note; }
    return out;
  }
  // The raw text's token ids and top-k next tokens on the inside engine (GET /next), for autoconfig.sameModel.
  async nextTop(text, k = 5) {
    const r = await fetch(`${this.url}/next?text=${encodeURIComponent(text)}&k=${k}`, { headers: { 'X-PSI-Token': this.token }, signal: AbortSignal.timeout(60000) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `/next: HTTP ${r.status}`);
    return data;
  }
}
module.exports = { TraceClient, weightsHash };
