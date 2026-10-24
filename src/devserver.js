// SPDX-License-Identifier: Apache-2.0
/*
 * Dev server: the same window UI, served to an ordinary browser on localhost,
 * with the Electron bridge replaced by HTTP calls (src/ui/psi-web.js). For
 * designing and testing the UI in Chrome; the engine runs here in Node.
 *
 *   node src/devserver.js            # http://127.0.0.1:4310
 *   PSI_MODEL=... PSI_LLAMA_DIR=...   # as for the app
 *
 * Guarded against other web pages in the same browser (CSRF, DNS rebinding): only Host 127.0.0.1/localhost:PORT
 * (plus the host names in PSI_ALLOWED_HOSTS or settings.json allowedHosts, comma-separated; none by default) is served, a foreign Origin is refused,
 * and every POST needs Content-Type application/json and a token: /psi/* takes the page's launch token (X-PSI-Token);
 * /mcp takes its own MCP token (Authorization: Bearer), which opens nothing else. The token is new each launch (PSI_DEV_TOKEN fixes it), printed at start and put in
 * the served index.html as <meta name="psi-token"> for psi-web.js.
 */
'use strict';
const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { Engine } = require('./engine.js');
const { TraceClient } = require('./trace_client.js');
const config = require('./config.js');
const autoconfig = require('./autoconfig.js');
const { createLineage } = require('./lineage-api.js');
const { createConverter } = require('./convert-api.js');
const { createMcp } = require('./mcp.js');
const { createLadder } = require('./ladder.js');
const { gpuState } = require('./gpu.js');
const { createIdle } = require('./idle.js');
const EXPERIMENTS = require('./experiments.js').createExperiments(config.DATA, () => config.guard());
require('../tools/write-version.js').writeVersion();   // src/ui/version.js follows package.json

const PORT = +(process.env.PSI_DEV_PORT || 4310);
const UI = path.join(__dirname, 'ui');
// llama-server's folder: the setting, else $PSI_LLAMA_DIR (kept in the settings), else llama-server on the PATH (engine.js).
if (!config.settings.llamaDir && process.env.PSI_LLAMA_DIR) config.update({ llamaDir: process.env.PSI_LLAMA_DIR }, { trusted: true });
// The last 30 lines the engines and the setup said ([trace], [setup], [engine]), for engineState().recent_log.
const RECENT = [];
const remember = (tag, m) => { for (const l of String(m).split(/\r?\n/)) if (l.trim()) RECENT.push(`${new Date().toISOString().slice(11, 19)} [${tag}] ${l}`.slice(0, 500)); RECENT.splice(0, Math.max(0, RECENT.length - 30)); };
const engine = new Engine({ binDir: config.settings.llamaDir, log: (m) => { remember('engine', m); console.log('[engine]', m); } });
process.env.PSI_CHECKPOINTS = process.env.PSI_CHECKPOINTS || config.settings.trace.checkpoints;
const LINEAGE = createLineage({ dataDir: config.DATA, log: (m) => console.log('[lineage]', m) });
const CONVERT = createConverter({ config, dataDir: config.DATA, log: (m) => console.log('[convert]', m) });
const tracer = new TraceClient({ modelDir: config.settings.trace.modelDir, lensSetting: () => ({ em: config.emOn(), text: (config.settings.lens && config.settings.lens.text) || '' }), log: (m) => { remember('trace', m); console.log('[trace]', m); } });
const TIMELINE = require('./timeline-api.js').createTimeline({ config, log: (m) => { remember('timeline', m); console.log('[timeline]', m); } });
// Commons (src/commons-api.js): no safeStorage outside Electron, so its secrets are 0600 files here, and keys protected by
// the desktop app's OS store stay locked (sign from the app).
const SECRETS = require('./secrets.js').createSecrets({ dataDir: config.DATA });
const HF = require('./hf-api.js').createHf({ config, secrets: SECRETS, log: (m) => { remember('hf', m); console.log('[hf]', m); } });
const COMMONS = require('./commons-api.js').createCommons({ config, secrets: SECRETS, log: (m) => { remember('commons', m); console.log('[commons]', m); } });
// Prometheus on the user's ChatGPT plan (src/chatgpt.js): the page opens the sign-in link in a new tab.
const CHATGPT = require('./chatgpt.js').createChatgpt({ dataDir: config.DATA, secrets: SECRETS, log: (m) => { remember('chatgpt', m); console.log('[chatgpt]', m); } });
config.setChatgpt(CHATGPT);
// The tester model on this computer (src/local-llm.js), kept apart from the model under test: never this engine's port.
config.setLocal(require('./local-llm.js').createLocal({ settings: () => config.settings, enginePort: () => (engine.proc ? engine.port : null) }));
config.setSigner(COMMONS.signer);
const SAE = require('./sae-api.js').createSae({ tracer, engine, config, log: (m) => { remember('sae', m); console.log('[sae]', m); } });   // Features: reading and training (src/sae-api.js)
// Calls only the page makes, kept out of `api` (which MCP and job plans reach): updates (src/update-api.js). The dev
// server does not restart itself: after an install the page says to restart it (and to run npm install if it changed).
const UPDATER = require('./update-api.js').createUpdater({ dataDir: config.DATA, log: (m) => { remember('update', m); console.log('[update]', m); },
  stopEngines: async () => { await engine.stop(); await tracer.stop(); } });
let SESSION = null;   // what each view last showed (src/session-api.js); made on first use, once `api` exists
// Every result kept as raw data (src/records.js): view runs through SESSION, MCP tool calls through MCP.
const RECORDS = require('./records.js').createRecords({ dataDir: config.DATA, settings: () => config.settings, log: (m) => { remember('records', m); console.log('[records]', m); } });
// Measuring an Override-map probe on the inside model (src/measure-api.js): an hour or more on the GPU.
const MEASURE = require('./measure-api.js').createMeasure({ tracer, engine, config, log: (m) => { remember('measure', m); console.log('[measure]', m); } });
const PAGE_ONLY = {
  async update(a) { return UPDATER.call(a || {}); },
  async measure(a) { return MEASURE.call(a || {}); },
  async session(a) { SESSION = SESSION || require('./session-api.js').createSession({ dataDir: config.DATA, api, log: (m) => { remember('session', m); console.log('[session]', m); }, records: RECORDS }); return SESSION.call(a || {}); },
};
const TOKEN = process.env.PSI_DEV_TOKEN || crypto.randomBytes(32).toString('hex');
// A separate token for /mcp: an agent holding it reaches the MCP tools only, never /psi/* (settings, keys, the
// spending cap, starting a model with steering). The page's own token opens /psi/* and is never printed for agents.
const MCP_TOKEN = process.env.PSI_MCP_TOKEN || crypto.randomBytes(32).toString('hex');
const EXTRA = String(process.env.PSI_ALLOWED_HOSTS ?? [].concat(config.settings.allowedHosts || []).join(',')).split(',').map((h) => h.trim().toLowerCase()).filter(Boolean);
const HOSTS = new Set([`127.0.0.1:${PORT}`, `localhost:${PORT}`, ...EXTRA]);
const ORIGINS = new Set([`http://127.0.0.1:${PORT}`, `http://localhost:${PORT}`, ...EXTRA.map((h) => `https://${h}`)]);
const same = (a, b) => { const x = Buffer.from(String(a || '')), y = Buffer.from(b); return x.length === y.length && crypto.timingSafeEqual(x, y); };
const tokenOf = (req) => req.headers['x-psi-token'] || (/^Bearer\s+(\S+)$/i.exec(req.headers.authorization || '') || [])[1];
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml' };

const api = {
  async readHead(file, bytes) {
    file = config.checkModelFile(file);
    const fd = await fs.promises.open(file, 'r');
    try { const size = (await fd.stat()).size; const n = Math.min(size, bytes); const buf = Buffer.alloc(n); await fd.read(buf, 0, n, 0); return { size, name: path.basename(file), bytes_b64: buf.toString('base64') }; }
    finally { await fd.close(); }
  },
  async pickModel() { return config.settings.lastModel; },
  async pickControlVector() { return null; },
  async readAll() { throw new Error('readAll is not served in the browser build'); },
  async settings(patch) { const v = config.update(patch); engine.binDir = config.settings.llamaDir; return v; },
  async setKey(k) { return config.setKey(k); },
  async listModels(dir) { return config.listModels(dir); },
  async external(args) { return config.external(args || {}); },
  async neuronNames(args) { return config.names(args || {}); },
  async experiments(args) { return EXPERIMENTS.call(args || {}); },   // the Experiment designer's history (data folder)
  async traceRestart(args) { const t = config.settings.trace; config.checkPatch({ trace: { modelDir: args && args.modelDir, checkpoints: args && args.checkpoints } }); const r = await tracer.restart({ modelDir: (args && args.modelDir) || t.modelDir, checkpoints: (args && args.checkpoints) || t.checkpoints }); config.update({ trace: { modelDir: r.modelDir, checkpoints: r.checkpoints } }); return r; },
  async start(file, options) { config.checkStart(file, options || {}); const p = await engine.start(file, options || {}); config.update({ lastModel: file }); const setup = await autoconfig.syncTrace({ file, config, tracer, log: (m) => { remember('setup', m); console.log('[setup]', m); } }); return { ...p, setup }; },
  async lineage(args) { return LINEAGE.call(args || {}); },
  async convert(args) { return CONVERT.call(args || {}); },
  async convertStatus(q) { return CONVERT.status(q && typeof q === 'object' ? q : q != null ? { id: q } : {}); },   // as in main.js   // Convert to GGUF (Settings > Models), as in main.js
  async setupCheck() { const r = autoconfig.check({ config, engine }); r.inside_engine = (await tracer.state()) || { running: false, note: 'not running now; it starts on the first request (Neural net, Anatomy, experiments)' }; if (tracer.note) r.inside_engine_note = tracer.note; r.same_model = await autoconfig.sameModel({ engine, tracer, config }); r.hardware = await require('./hardware.js').check({ engine, gpuState }).catch((e) => ({ error: e.message })); return r; },
  async stop() { return engine.stop(); },
  async tokenize(text, special) { return engine.tokenize(text, special); },
  async detokenize(ids) { return engine.detokenize(ids); },
  async next(ids, k) { return engine.next(ids, k); },
  async complete(ids, opts) { return engine.complete(ids, opts || {}); },
  async score(ids, k) { return engine.score(ids, k); },
  async template(messages) { return engine.applyTemplate(messages); },
  async embed(text, k) { await tracer.ensure(); return tracer.embed(text, k); },
  async understand(items) { await tracer.ensure(); return tracer.understand(items); },
  async represent(text) { await tracer.ensure(); return tracer.represent(text); },
  async vectors(args) { await tracer.ensure(); return tracer.vectors(args || {}); },
  async override(args) { if (MEASURE.busy()) throw new Error('A probe is being measured on the GPU; the Override map is free again when it ends.'); await tracer.ensure(); return tracer.override(args || {}); },
  async context(args) { if (MEASURE.busy()) throw new Error('A probe is being measured on the GPU; the Context lab is free again when it ends.'); await tracer.ensure(); return tracer.context(args || {}); },
  async attractor(args) { if (MEASURE.busy()) throw new Error('A probe is being measured on the GPU; the Attractor lab is free again when it ends.'); await tracer.ensure(); return tracer.attractor(args || {}); },
  async sae(a) { return SAE.call(a || {}); },
  async timeline(a) { return TIMELINE.call(a || {}); },
  async chatgpt(a) { return CHATGPT.call(a || {}); },   // ChatGPT plan sign-in for Prometheus (src/chatgpt.js)
  async hf(a) { return HF.call(a || {}); },   // Settings > Hugging Face (src/hf-api.js), as in main.js
  async commons(a) { return COMMONS.call(a || {}); },   // Commons: lines, branches, recipes, verifying, profile and keys (src/commons-api.js)   // Timeline: placing models, the graft lab (src/timeline-api.js)   // Features (sparse autoencoders), as in main.js
  async scoreText(args) { await tracer.ensure(); return tracer.scoreText(args || {}); },
  async ablate(args) { await tracer.ensure(); return tracer.ablate(args || {}); },
  async ckptList() { await tracer.ensure(); return tracer.ckptList(); },
  async ckptCount(args) { await tracer.ensure(); return tracer.ckptCount(args); },
  async ckptAsk(args) { await tracer.ensure(); return tracer.ckptAsk(args || {}); },
  async ckptTest(args) { await tracer.ensure(); return tracer.ckptTest(args || {}); },
  async heads(args) { await tracer.ensure(); return tracer.heads(args || {}); },
  async ablateHeads(args) { await tracer.ensure(); return tracer.ablateHeads(args || {}); },
  async lesion(args) { await tracer.ensure(); return tracer.lesion(args || {}); },
  async sense(args) { await tracer.ensure(); return tracer.sense(args || {}); },
  async fight(args) { await tracer.ensure(); return tracer.fight(args || {}); },
  async connector(args) { await tracer.ensure(); return tracer.connector(args || {}); },
  async anatomy(args) { await tracer.ensure(); return tracer.anatomy(args || {}); },
  async traceStatus() { return (await tracer.alive()) || { ok: false }; },
  async trace(prompt, opts) { await tracer.ensure(); return tracer.trace(prompt, opts || {}); },
  async gpu() { return { ...(await gpuState({ pids: [engine.proc && engine.proc.pid, tracer.proc && tracer.proc.pid].filter(Boolean) })), idle: IDLE.state() }; },
  async engineState() { return { model: engine.model, options: engine.options, running: !!engine.proc, props: engine.info(), no_think_supported: engine.proc ? engine.no_think_supported : null, recent_log: RECENT.slice() }; },
  async mcp(rpc) { return MCP.handle(rpc, { caller: 'app' }); },   // the page's own Prometheus; /mcp gets the agent rules
  // Storage ladder: read-only here (the dev server can be reached through a tunnel); runs happen in the desktop app.
  async ladder(a = {}) {
    if (a.action === 'data') return LADDER.data();
    if (a.action === 'status') return LADDER.status();
    if (a.action === 'command') return LADDER.command(a);
    throw new Error('Runs happen in the desktop app (npm start); in the browser the Storage ladder shows the stored results.');
  },
};
// Sleep when idle (src/idle.js): every call below is wrapped, so a model nobody uses gives its GPU memory back.
const IDLE = createIdle({ engine, tracer, config, gpuState, start: (f, o) => engine.start(f, o), log: (m) => { remember('idle', m); console.log('[idle]', m); } });
api.idle = async (a) => IDLE.call(a || {});
api.quiet = async (name, ...a) => IDLE.quiet(name, ...a);
IDLE.wrap(api);
const LADDER = createLadder({ config, picked: new Set(), send: null });
const MCP = createMcp({ call: (n, ...a) => api[n](...a), config, records: RECORDS });
// The job runner (src/jobs.js): a plan sent from a phone runs here and carries on when the phone locks. Through
// POST /psi/jobs (with the token) it is also the API for scripts: {action: 'submit', plan}, then 'get' until done.
const JOBS = require('./jobs.js').createJobs({ dataDir: config.DATA, call: (n, ...a) => api[n](...a), mcp: (rpc) => MCP.handle(rpc, { caller: 'app' }), experiments: EXPERIMENTS, config,
  gpu: () => gpuState({ pids: [engine.proc && engine.proc.pid, tracer.proc && tracer.proc.pid].filter(Boolean) }), log: (m) => { remember('jobs', m); console.log('[jobs]', m); } });
api.jobs = async (a) => { const x = a || {}; if (x.action === 'submit' && x.origin !== 'app') x.origin = 'api'; return JOBS.call(x); };

// A request body, or null past 16 MB (the largest real one, a long text to read, is far smaller).
// Too large: the reply (413) is sent first and the connection closed after it, so the client sees why.
function readBody(req, max = 16 << 20) {
  if (+req.headers['content-length'] > max) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    let n = 0; const parts = [];
    const onData = (c) => { n += c.length; if (n > max) { req.off('data', onData); req.resume(); resolve(null); } else parts.push(c); };   // the rest is discarded
    req.on('data', onData); req.once('error', reject); req.once('end', () => resolve(n > max ? null : Buffer.concat(parts).toString('utf8')));
  });
}

http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const deny = (code, msg) => { res.writeHead(code, { 'Content-Type': 'text/plain', ...(code === 413 ? { Connection: 'close' } : {}) }); res.end(msg, () => { if (code === 413) req.destroy(); }); };
  // On every reply: no other site may frame the app (a hidden frame could trick a click on Run), and nothing is sniffed.
  res.setHeader('X-Frame-Options', 'DENY'); res.setHeader('Content-Security-Policy', "frame-ancestors 'none'");
  res.setHeader('X-Content-Type-Options', 'nosniff'); res.setHeader('Referrer-Policy', 'no-referrer'); res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  const host = String(req.headers.host || '').toLowerCase();
  if (!HOSTS.has(host)) return deny(403, 'forbidden host');
  // An extra host is served only through an authenticating proxy: Cloudflare Access adds this header to every request it
  // lets through, and the server listens on 127.0.0.1 only, so a request without it did not come through Access.
  // Only the header's presence is checked, not the JWT: behind a proxy that does not strip or set it, anyone who reaches the server gets in.
  if (EXTRA.includes(host) && !req.headers['cf-access-jwt-assertion']) return deny(403, 'this host is served only behind Cloudflare Access');
  if (req.headers.origin && !ORIGINS.has(req.headers.origin)) return deny(403, 'forbidden origin');
  if (req.method === 'POST' && (url.pathname === '/mcp' || url.pathname.startsWith('/psi/'))) {
    if (!/^application\/json\s*(;|$)/i.test(req.headers['content-type'] || '')) return deny(415, 'Content-Type must be application/json');
    if (!same(tokenOf(req), url.pathname === '/mcp' ? MCP_TOKEN : TOKEN)) return deny(403, url.pathname === '/mcp' ? 'missing or wrong MCP token (Authorization: Bearer)' : 'missing or wrong token (X-PSI-Token)');
  }
  // MCP endpoint (Streamable HTTP, JSON responses). Same host as the UI, so it is behind Cloudflare Access when tunnelled.
  if (url.pathname === '/mcp') {
    if (req.method !== 'POST') { res.writeHead(405, { Allow: 'POST' }); res.end(); return; }
    const body = await readBody(req); if (body == null) return deny(413, 'request too large');
    let out; try { out = await MCP.handle(JSON.parse(body || '{}')); } catch (e) { out = { jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } }; }
    if (out == null) { res.writeHead(202); res.end(); return; }
    res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(out)); return;
  }
  if (req.method === 'POST' && url.pathname.startsWith('/psi/')) {
    const name = url.pathname.slice(5);
    const body = await readBody(req); if (body == null) return deny(413, 'request too large');
    try {
      const fn = Object.hasOwn(api, name) ? api[name] : Object.hasOwn(PAGE_ONLY, name) ? PAGE_ONLY[name] : null;
      if (!fn) throw new Error(`unknown call ${name}`);
      const value = await fn(...(JSON.parse(body || '[]')));
      res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: true, value }));
    } catch (e) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ ok: false, error: String(e.message || e) })); }
    return;
  }
  let file = url.pathname === '/' ? '/index.html' : url.pathname;
  file = path.normalize(path.join(UI, file));
  if (!file.startsWith(UI) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end('not found'); return; }
  let data = fs.readFileSync(file);
  // In the browser there is no preload: inject the HTTP bridge before the app scripts.
  const tokenMeta = (t) => t.replace('<meta charset="utf-8">', `<meta charset="utf-8">\n<meta name="psi-token" content="${TOKEN}">`);
  if (file.endsWith('index.html')) {
    const t = tokenMeta(data.toString('utf8')).replace('<script src="gguf-header.js"></script>', '<script src="psi-web.js"></script>\n<script src="gguf-header.js"></script>');
    // Ask for every script up front, all at once: the browser otherwise fetches them one after another, and over a
    // tunnel each one costs a round trip (about 50 of them: seconds, or minutes on a slow link). They still run in order.
    const pre = [...t.matchAll(/<script src="([^"]+)"><\/script>/g)].map((m) => `<link rel="preload" as="script" href="${m[1]}">`).join('\n');
    data = Buffer.from(t.replace('</head>', `${pre}\n</head>`));
  }
  // The manual's Prometheus (help-prometheus.js) needs the same bridge, with the page's own token.
  if (file.endsWith('help.html')) data = Buffer.from(tokenMeta(data.toString('utf8')).replace('<script src="help-prometheus.js"></script>', '<script src="psi-web.js"></script>\n<script src="help-prometheus.js"></script>'));
  // The analysis sandbox's worker gets a policy of its own: it may run the code it is given and reach nothing at all.
  if (path.basename(file) === 'sandbox-worker.js') res.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'self' 'unsafe-eval'");
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
  res.end(data);
}).listen(PORT, '127.0.0.1', () => console.log(`Parameter Space Inspector (dev) at http://127.0.0.1:${PORT}  model ${config.settings.lastModel || '(none chosen yet: Settings > Models)'}\nMCP: POST http://127.0.0.1:${PORT}/mcp with header  Authorization: Bearer ${MCP_TOKEN}  (this token opens /mcp only; set PSI_MCP_TOKEN to fix it)`));
process.on('SIGINT', async () => { await engine.stop(); await tracer.stop(); process.exit(0); });
