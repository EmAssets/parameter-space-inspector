// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
/*
 * Parameter Space Inspector (desktop). Electron main process: owns the model
 * engine (llama.cpp's server) and answers the window's requests over IPC.
 * The window never touches the file system or the network directly.
 */
'use strict';
const { app, BrowserWindow, ipcMain, dialog, shell, session, Notification, safeStorage } = require('electron');
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');
const { Engine } = require('./engine.js');
const { TraceClient } = require('./trace_client.js');
const { createLadder } = require('./ladder.js');
const { gpuState } = require('./gpu.js');
const { createIdle } = require('./idle.js');

// Keep settings and Electron's caches beside the app (.userdata/), not in the system profile. PSI_DATA_DIR overrides.
app.setPath('userData', process.env.PSI_DATA_DIR || path.join(__dirname, '..', '.userdata'));

process.env.PSI_DATA_DIR = app.getPath('userData');
const config = require('./config.js');
const autoconfig = require('./autoconfig.js');
const { createLineage } = require('./lineage-api.js');
const { createConverter } = require('./convert-api.js');
try { require('../tools/write-version.js').writeVersion(); } catch (_) { /* packaged: the shipped version.js stays */ }
// llama-server's folder: the setting, else $PSI_LLAMA_DIR (kept in the settings), else llama-server on the PATH (engine.js).
if (!config.settings.llamaDir && process.env.PSI_LLAMA_DIR) config.update({ llamaDir: process.env.PSI_LLAMA_DIR }, { trusted: true });
process.env.PSI_CHECKPOINTS = process.env.PSI_CHECKPOINTS || config.settings.trace.checkpoints;
const settings = config.settings;

let win = null;
const PICKED = new Set();   // model files the user chose in a dialog: readable wherever they are (config.checkModelFile)
const picked = (f) => (f && PICKED.add(config.key4(config.real(f))), f);
const LINEAGE = createLineage({ dataDir: config.DATA, log: (m) => win && win.webContents.send('psi:log', `[lineage] ${m}`) });
// The last 30 lines the engines and the setup said ([trace], [setup], [engine]), for engineState().recent_log.
const RECENT = [];
const remember = (tag, m) => { for (const l of String(m).split(/\r?\n/)) if (l.trim()) RECENT.push(`${new Date().toISOString().slice(11, 19)} [${tag}] ${l}`.slice(0, 500)); RECENT.splice(0, Math.max(0, RECENT.length - 30)); };
const tracer = new TraceClient({ modelDir: config.settings.trace.modelDir, lensSetting: () => ({ em: config.emOn(), text: (config.settings.lens && config.settings.lens.text) || '' }), log: (m) => { remember('trace', m); win && win.webContents.send('psi:log', `[trace] ${m}`); } });
const engine = new Engine({ binDir: settings.llamaDir, log: (m) => { remember('engine', m); win && win.webContents.send('psi:log', m); } });

// The only pages that get the app's API: the app itself and its bundled manual and about page.
const UI_URL = pathToFileURL(path.join(__dirname, 'ui', 'index.html')).href;
const HELP_URLS = new Set(['help.html', 'about.html'].map((f) => pathToFileURL(path.join(__dirname, 'ui', f)).href));
const OWN_PAGE = (u) => { const x = String(u || '').split('#')[0]; return x === UI_URL || HELP_URLS.has(x); };

function createWindow() {
  // Permissions: the microphone for voice input on the app's own pages, nothing else (no camera, location, notifications...).
  const allowPerm = (wc, perm, details) => perm === 'media' && OWN_PAGE((details && (details.requestingUrl || details.securityOrigin)) || (wc && wc.getURL()))
    && !((details && details.mediaTypes) || []).includes('video');
  session.defaultSession.setPermissionRequestHandler((wc, perm, cb, details) => cb(allowPerm(wc, perm, details)));
  session.defaultSession.setPermissionCheckHandler((wc, perm, origin, details) => perm === 'media' && OWN_PAGE((details && details.requestingUrl) || (wc && wc.getURL())));
  win = new BrowserWindow({
    width: 1400, height: 920, backgroundColor: '#000000', title: 'Parameter Space Inspector', icon: path.join(__dirname, 'ui', 'em-logo-300.png'),
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  win.loadFile(path.join(__dirname, 'ui', 'index.html'));
  // The window never leaves the app's own page; the help windows move only between the bundled help and about pages.
  const stay = (ok) => (e, url) => { if (!ok(String(url).split('#')[0])) { e.preventDefault(); if (/^https?:/.test(url)) shell.openExternal(url); } };
  win.webContents.on('will-navigate', stay((u) => u === UI_URL)); win.webContents.on('will-redirect', stay((u) => u === UI_URL));
  win.webContents.on('did-create-window', (child) => {
    const ok = (u) => HELP_URLS.has(u);
    child.webContents.on('will-navigate', stay(ok)); child.webContents.on('will-redirect', stay(ok));
    child.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url)) shell.openExternal(url); return { action: 'deny' }; });
  });
  // Links open in the system browser, never inside the app.
  win.webContents.setWindowOpenHandler(({ url }) => {
    // The bundled manual opens in its own app window; web links open in the browser; nothing else opens.
    // Exactly the bundled pages: any other file: URL (a downloaded help.html, a \\host\share path) would get the app's API.
    if (HELP_URLS.has(String(url).split('#')[0])) return { action: 'allow', overrideBrowserWindowOptions: { width: /about\.html/i.test(url) ? 700 : 1100, height: /about\.html/i.test(url) ? 720 : 860, autoHideMenuBar: true, title: /about\.html/i.test(url) ? 'About · Parameter Space Inspector' : 'Help · Parameter Space Inspector', backgroundColor: '#08080a',
      // The manual's Prometheus (help-prometheus.js) reaches settings and OpenRouter through the same preload as the app.
      webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, sandbox: true } } };
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
}

const API = {};
// Sleep when idle (src/idle.js): every call goes through IDLE.before/after, so a model nobody uses gives its GPU memory back.
const IDLE = createIdle({ engine, tracer, config, gpuState, start: (f, o) => engine.start(f, o), log: (m) => { remember('idle', m); win && win.webContents.send('psi:log', `[idle] ${m}`); } });
const handle = (name, fn) => {
  const run = IDLE.track(name, fn);
  API[name] = run;
  ipcMain.handle(`psi:${name}`, async (e, ...args) => {
    // Only the app's own pages may call it (a frame or window that somehow loaded anything else gets nothing).
    if (!OWN_PAGE(e.senderFrame && e.senderFrame.url)) return { ok: false, error: 'not allowed from this page' };
    try { return { ok: true, value: await run(...args) }; } catch (err) { return { ok: false, error: String(err.message || err) }; }
  });
};
handle('idle', async (a) => IDLE.call(a || {}));
handle('quiet', async (name, ...a) => IDLE.quiet(name, ...a));

// The first bytes of a model file, for the header reader in the window (any size).
handle('readHead', async (file, bytes) => {
  file = config.checkModelFile(file, PICKED);
  const fd = await fs.promises.open(file, 'r');
  try {
    const size = (await fd.stat()).size;
    const n = Math.min(size, bytes);
    const buf = Buffer.alloc(n);
    await fd.read(buf, 0, n, 0);
    return { size, name: path.basename(file), bytes: buf.buffer.slice(buf.byteOffset, buf.byteOffset + n) };
  } finally { await fd.close(); }
});
// Whole file, only for small models the window runs itself (the toy Llama).
handle('readAll', async (file) => {
  file = config.checkModelFile(file, PICKED);
  const st = await fs.promises.stat(file);
  if (st.size > 256 * 2 ** 20) throw new Error('Files over 256 MB are run by the llama.cpp engine, not read whole.');
  const buf = await fs.promises.readFile(file);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
});
handle('pickModel', async () => {
  const r = await dialog.showOpenDialog(win, { title: 'Open a GGUF model', properties: ['openFile'], filters: [{ name: 'GGUF models', extensions: ['gguf'] }] });
  return r.canceled ? null : picked(r.filePaths[0]);
});
handle('pickControlVector', async () => {
  const r = await dialog.showOpenDialog(win, { title: 'Open a control vector', properties: ['openFile'], filters: [{ name: 'GGUF control vectors', extensions: ['gguf'] }] });
  return r.canceled ? null : picked(r.filePaths[0]);
});
handle('settings', async (patch) => { const v = config.update(patch); engine.binDir = config.settings.llamaDir; return v; });
handle('setKey', async (k) => config.setKey(k));
handle('listModels', async (dir) => config.listModels(dir));
handle('external', async (args) => config.external(args || {}));
handle('neuronNames', async (args) => config.names(args || {}));
const EXPERIMENTS = require('./experiments.js').createExperiments(config.DATA, () => config.guard());
handle('experiments', async (args) => EXPERIMENTS.call(args || {}));   // the Experiment designer's history (data folder)
// The Experiment designer's report as a PDF: the page shows only the report under its print styles, the user picks
// where it goes, Chromium prints it. Only the app's own page; the file name is cleaned, the path comes from the dialog.
// The Experiment designer's exports (Word, Excel, PowerPoint, JSON, CSV): the user picks where in a save dialog; the name is
// cleaned, the kind is one of six, the size is capped. Only the app's own page.
const SAVE_KINDS = { docx: 'Word document', xlsx: 'Excel workbook', pptx: 'PowerPoint presentation', json: 'JSON data', csv: 'CSV table', pdf: 'PDF' };
ipcMain.handle('psi:saveFile', async (e, a) => {
  if (!OWN_PAGE(e.senderFrame && e.senderFrame.url)) return { ok: false, error: 'not allowed from this page' };
  try {
    const name = String((a && a.name) || 'experiment').replace(/[^\p{L}\p{N} _.-]/gu, '').trim().slice(0, 90) || 'experiment';
    const ext = (name.match(/\.([a-z]+)$/) || [])[1]; if (!SAVE_KINDS[ext]) throw new Error('saveFile: only .docx, .xlsx, .pptx, .json, .csv or .pdf');
    const data = a.base64 ? Buffer.from(String(a.data || ''), 'base64') : Buffer.from(String(a.data || ''), 'utf8');
    if (data.length > 50 * 1048576) throw new Error('saveFile: larger than 50 MB');
    const { canceled, filePath } = await dialog.showSaveDialog(BrowserWindow.fromWebContents(e.sender), { defaultPath: name, filters: [{ name: SAVE_KINDS[ext], extensions: [ext] }] });
    if (canceled || !filePath) return { ok: true, value: { saved: false } };
    fs.writeFileSync(filePath, data); return { ok: true, value: { saved: true, path: filePath } };
  } catch (err) { return { ok: false, error: String(err.message || err) }; }
});
ipcMain.handle('psi:savePdf', async (e, name) => {
  if (!OWN_PAGE(e.senderFrame && e.senderFrame.url)) return { ok: false, error: 'not allowed from this page' };
  try {
    const safe = String(name || 'experiment').replace(/[^\p{L}\p{N} _.-]/gu, '').trim().slice(0, 80) || 'experiment';
    const { canceled, filePath } = await dialog.showSaveDialog(BrowserWindow.fromWebContents(e.sender), { defaultPath: `${safe}.pdf`, filters: [{ name: 'PDF', extensions: ['pdf'] }] });
    if (canceled || !filePath) return { ok: true, value: { saved: false } };
    const pdf = await e.sender.printToPDF({ printBackground: true, pageSize: 'A4' });
    fs.writeFileSync(filePath, pdf); return { ok: true, value: { saved: true, path: filePath } };
  } catch (err) { return { ok: false, error: String(err.message || err) }; }
});
handle('traceRestart', async (args) => { const t = config.settings.trace; config.checkPatch({ trace: { modelDir: args && args.modelDir, checkpoints: args && args.checkpoints } }); const r = await tracer.restart({ modelDir: (args && args.modelDir) || t.modelDir, checkpoints: (args && args.checkpoints) || t.checkpoints }); config.update({ trace: { modelDir: r.modelDir, checkpoints: r.checkpoints } }); return r; });
handle('start', async (file, options) => { config.checkStart(file, options || {}); const props = await engine.start(file, options || {}); config.update({ lastModel: file }); const setup = await autoconfig.syncTrace({ file, config, tracer, log: (m) => { remember('setup', m); win && win.webContents.send('psi:log', `[setup] ${m}`); } }); return { ...props, setup }; });
handle('lineage', async (args) => LINEAGE.call(args || {}));
// Convert to GGUF (Settings > Local model): a Hugging Face folder from the model list becomes a GGUF file, on the CPU.
const CONVERT = createConverter({ config, dataDir: config.DATA, log: (m) => win && win.webContents.send('psi:log', `[convert] ${m}`) });
handle('convert', async (args) => CONVERT.call(args || {}));
// The conversion jobs (a reloaded page re-attaches to a running one): one by id, or all, most recent first.
handle('convertStatus', async (q) => CONVERT.status(q && typeof q === 'object' ? q : q != null ? { id: q } : {}));
handle('setupCheck', async () => { const r = autoconfig.check({ config, engine }); r.inside_engine = (await tracer.state()) || { running: false, note: 'not running now; it starts on the first request (Neural net, Anatomy, experiments)' }; if (tracer.note) r.inside_engine_note = tracer.note; r.same_model = await autoconfig.sameModel({ engine, tracer, config }); r.hardware = await require('./hardware.js').check({ engine, gpuState }).catch((e) => ({ error: e.message })); return r; });
handle('stop', async () => engine.stop());
handle('tokenize', async (text, special) => engine.tokenize(text, special));
handle('detokenize', async (ids) => engine.detokenize(ids));
handle('next', async (ids, k) => engine.next(ids, k));
handle('complete', async (ids, opts) => engine.complete(ids, opts || {}));
handle('score', async (ids, k) => engine.score(ids, k, (i, n) => win && win.webContents.send('psi:progress', { i, n })));
handle('template', async (messages) => engine.applyTemplate(messages));
handle('embed', async (text, k) => { await tracer.ensure((s) => win && win.webContents.send('psi:log', s)); return tracer.embed(text, k); });
handle('understand', async (items) => { await tracer.ensure(); return tracer.understand(items); });
handle('represent', async (text) => { await tracer.ensure(); return tracer.represent(text); });
handle('vectors', async (args) => { await tracer.ensure(); return tracer.vectors(args || {}); });
// Features (sparse autoencoders): reading through the inside engine, training as its own process (src/sae-api.js).
const SAE = require('./sae-api.js').createSae({ tracer, engine, config, log: (m) => { remember('sae', m); win && win.webContents.send('psi:log', `[sae] ${m}`); } });
handle('sae', async (a) => SAE.call(a || {}));
// Timeline: placing models on the emergence timeline and the graft lab, as their own processes (src/timeline-api.js).
const TIMELINE = require('./timeline-api.js').createTimeline({ config, log: (m) => { remember('timeline', m); win && win.webContents.send('psi:log', `[timeline] ${m}`); } });
handle('timeline', async (a) => TIMELINE.call(a || {}));
// Commons: seed lines grown by many people (src/commons-api.js). Keys and the Hugging Face token are kept by the operating
// system (safeStorage); the signing choice it holds reaches every Python child through config.cacheEnv.
const SECRETS = require('./secrets.js').createSecrets({ dataDir: config.DATA, safeStorage });
const COMMONS = require('./commons-api.js').createCommons({ config, secrets: SECRETS, log: (m) => { remember('commons', m); win && win.webContents.send('psi:log', `[commons] ${m}`); } });
config.setSigner(COMMONS.signer);
handle('commons', async (a) => COMMONS.call(a || {}));
// Settings > Hugging Face (src/hf-api.js): the account the app pushes with; same token as Commons.
const HF = require('./hf-api.js').createHf({ config, secrets: SECRETS, log: (m) => { remember('hf', m); win && win.webContents.send('psi:log', `[hf] ${m}`); } });
handle('hf', async (a) => HF.call(a || {}));
// Prometheus on the user's ChatGPT plan (src/chatgpt.js): sign-in opens in the user's own browser.
const CHATGPT = require('./chatgpt.js').createChatgpt({ dataDir: config.DATA, secrets: SECRETS, open: (u) => shell.openExternal(u), log: (m) => { remember('chatgpt', m); win && win.webContents.send('psi:log', `[chatgpt] ${m}`); } });
config.setChatgpt(CHATGPT);
handle('chatgpt', async (a) => CHATGPT.call(a || {}));
handle('scoreText', async (args) => { await tracer.ensure(); return tracer.scoreText(args || {}); });
handle('ablate', async (args) => { await tracer.ensure(); return tracer.ablate(args || {}); });
handle('ckptList', async () => { await tracer.ensure(); return tracer.ckptList(); });
handle('ckptCount', async (args) => { await tracer.ensure(); return tracer.ckptCount(args); });
handle('ckptAsk', async (args) => { await tracer.ensure(); return tracer.ckptAsk(args || {}); });
handle('ckptTest', async (args) => { await tracer.ensure(); return tracer.ckptTest(args || {}); });
handle('heads', async (args) => { await tracer.ensure(); return tracer.heads(args || {}); });
handle('ablateHeads', async (args) => { await tracer.ensure(); return tracer.ablateHeads(args || {}); });
handle('lesion', async (args) => { await tracer.ensure(); return tracer.lesion(args || {}); });
handle('sense', async (args) => { await tracer.ensure(); return tracer.sense(args || {}); });
handle('fight', async (args) => { await tracer.ensure(); return tracer.fight(args || {}); });
handle('connector', async (args) => { await tracer.ensure(); return tracer.connector(args || {}); });
handle('anatomy', async (args) => { await tracer.ensure(); return tracer.anatomy(args || {}); });
// Storage ladder (M0–M8): results to draw, and runs of research/model_ladder.py on a model folder the user picks.
const PICKED_DIRS = new Set();
const ladder = createLadder({ config, picked: PICKED_DIRS, send: (m) => win && win.webContents.send('psi:ladder', m) });
handle('ladder', async (a = {}) => {
  switch (a.action) {
    case 'data': return ladder.data();
    case 'pick': {
      const r = await dialog.showOpenDialog(win, { title: 'Choose a Hugging Face model folder (config.json + weights)', properties: ['openDirectory'] });
      if (r.canceled || !r.filePaths[0]) return null;
      const d = r.filePaths[0]; PICKED_DIRS.add(config.key4(config.real(d))); return { dir: d, size: ladder.sizeOf(d) };
    }
    case 'run':
      if (a.freeTrace) await tracer.stop();          // the inside engine holds a model on the GPU; it restarts on the next request
      if (a.freeEngine) await engine.stop();
      return ladder.run(a);
    case 'cancel': return ladder.cancel();
    case 'status': return ladder.status();
    case 'command': return ladder.command(a);
    default: throw new Error('unknown ladder action');
  }
});
handle('traceStatus', async () => (await tracer.alive()) || { ok: false });
handle('trace', async (prompt, opts) => { await tracer.ensure((s) => win && win.webContents.send('psi:log', s)); return tracer.trace(prompt, opts || {}); });
handle('gpu', async () => ({ ...(await gpuState({ pids: [engine.proc && engine.proc.pid, tracer.proc && tracer.proc.pid].filter(Boolean) })), idle: IDLE.state() }));
handle('engineState', async () => ({ model: engine.model, options: engine.options, running: !!engine.proc, props: engine.info(), no_think_supported: engine.proc ? engine.no_think_supported : null, recent_log: RECENT.slice() }));

app.whenReady().then(() => { createWindow(); if (process.env.PSI_SELFTEST) selfTest().catch((e) => { console.error('SELFTEST FAILED', e); app.exit(1); }); });

// PSI_SELFTEST=1: drive the real window through every view and save screenshots.
async function selfTest() {
  const out = path.join(__dirname, '..', 'out'); fs.mkdirSync(out, { recursive: true });
  const js = (code) => win.webContents.executeJavaScript(code);
  const wait = async (cond, ms = 120000) => { const t = Date.now(); while (!(await js(cond))) { if (Date.now() - t > ms) throw new Error('timeout: ' + cond); await new Promise((r) => setTimeout(r, 300)); } };
  const shot = async (name) => { const img = await win.webContents.capturePage(); fs.writeFileSync(path.join(out, name + '.png'), img.toPNG()); console.log('shot', name); };
  await new Promise((r) => win.webContents.once('did-finish-load', r));
  await wait("document.getElementById('modelName').textContent.includes('Qwen') || document.getElementById('modelName').textContent.includes('qwen')");
  console.log('model:', await js("document.getElementById('modelName').textContent"), '|', await js("document.getElementById('engineInfo').textContent"));
  await js("document.getElementById('genN').value = 40; document.getElementById('genGo').click()");
  await wait("document.querySelectorAll('#forks .col').length >= 40 || (document.querySelectorAll('#forks .col').length > 5 && document.body.style.cursor === '')");
  await wait("document.body.style.cursor === ''");
  console.log('generated:', JSON.stringify((await js("document.getElementById('ribbon').textContent")).slice(0, 400)));
  await shot('1-generate');
  // Fork at step 6: take the second choice.
  await js("document.querySelector('#forks .col[data-j=\"5\"] .cand:not(.chosen)').click()");
  await new Promise((r) => setTimeout(r, 500)); await wait("document.body.style.cursor === ''");
  console.log('forked:', JSON.stringify((await js("document.getElementById('ribbon').textContent")).slice(0, 300)));
  console.log('ghost:', JSON.stringify((await js("document.getElementById('ghost').textContent")).slice(0, 200)));
  await shot('2-fork');
  // Generate's roads: pick step 6, analyse the fork against its second choice, map the basins.
  await js("document.querySelectorAll('#ribbon span')[5].click(); document.getElementById('genPaths').scrollIntoView()");
  console.log('fork controls:', await js("document.getElementById('forkWhich').textContent"), '|', await js("document.getElementById('forkAlt').options.length"), 'alternatives');
  await js("document.getElementById('forkGo').click()"); await new Promise((r) => setTimeout(r, 500)); await wait("document.body.style.cursor === ''", 300000);
  console.log('road taken  :', JSON.stringify((await js("document.getElementById('forkTaken').innerText")).slice(0, 300)));
  console.log('road not    :', JSON.stringify((await js("document.getElementById('forkAlt2').innerText")).slice(0, 300)));
  console.log('verdict     :', JSON.stringify((await js("document.getElementById('forkVerdict').innerText")).slice(0, 500)));
  await js("document.getElementById('basinK').value = 12; document.getElementById('basinGo').click()"); await new Promise((r) => setTimeout(r, 500)); await wait("document.body.style.cursor === ''", 600000);
  console.log('basins      :', await js("document.getElementById('basinStat').textContent"), '|', JSON.stringify((await js("document.getElementById('basinOut').innerText")).slice(0, 400)));
  console.log('loops       :', await js("document.getElementById('loopOut').textContent"));
  await shot('2b-paths');
  await js("document.querySelector('[data-tab=read]').click(); document.getElementById('readGo').click()");
  await new Promise((r) => setTimeout(r, 500)); await wait("document.body.style.cursor === ''");
  console.log('read:', await js("document.getElementById('readStat').textContent"));
  await shot('3-read');
  await js("document.querySelector('[data-tab=tokens]').click(); document.querySelector('[data-sample]:nth-child(2)').click()");
  await new Promise((r) => setTimeout(r, 500)); await wait("document.body.style.cursor === ''");
  console.log('tokens:', await js("document.getElementById('tokStat').textContent"));
  await shot('4-tokens');
  await js("document.querySelector('[data-tab=compare]').click(); document.getElementById('cmpN').value = 120; document.getElementById('cmpGo').click()");
  await new Promise((r) => setTimeout(r, 500)); await wait("document.body.style.cursor === ''", 300000);
  console.log('lens on :', JSON.stringify((await js("document.getElementById('cmpOn').textContent")).slice(0, 500)));
  console.log('lens off:', JSON.stringify((await js("document.getElementById('cmpOff').textContent")).slice(0, 500)));
  await shot('5-compare');
  await js("document.querySelector('[data-tab=file]').click()"); await new Promise((r) => setTimeout(r, 300));
  await shot('6-file');
  await engine.stop(); app.exit(0);
}
app.on('window-all-closed', async () => { await engine.stop(); await tracer.stop(); app.quit(); });
app.on('before-quit', () => { engine.stop(); });

// The Inspector as an MCP server for the in-app AI helper (and any agent over IPC).
const { createMcp } = require('./mcp.js');
const MCP = createMcp({ call: (n, ...a) => API[n](...a), config });
handle('mcp', async (rpc) => MCP.handle(rpc, { caller: 'app' }));   // the window's own Prometheus

// The job runner (src/jobs.js): an approved plan runs here, in the main process, so it carries on when the window is
// closed or another screen started it; a desktop notice says when it ends.
const JOBS = require('./jobs.js').createJobs({ dataDir: config.DATA, call: (n, ...a) => API[n](...a), mcp: (rpc) => MCP.handle(rpc, { caller: 'app' }), experiments: EXPERIMENTS, config,
  gpu: () => gpuState({ pids: [engine.proc && engine.proc.pid, tracer.proc && tracer.proc.pid].filter(Boolean) }),
  log: (m) => { remember('jobs', m); win && win.webContents.send('psi:log', `[jobs] ${m}`); },
  notify: (j) => { try { if (Notification.isSupported()) new Notification({ title: j.status === 'done' ? 'Experiment finished' : `Experiment ${j.status}`, body: `${j.name} · ${j.done} of ${j.n}${j.error ? ` · ${j.error}` : ''}`.slice(0, 200) }).show(); } catch (_) { /* a notice only */ } } });
handle('jobs', async (a) => JOBS.call(a || {}));
