// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Global configuration shared by the Electron app (main.js) and the browser build (devserver.js):
// persisted settings, the OpenRouter key (kept out of the page), local model discovery, and a
// server-side proxy for external models with a spending cap.
'use strict';
const fs = require('fs');
const path = require('path');

const DATA = process.env.PSI_DATA_DIR || path.join(__dirname, '..', '.userdata');
const FILE = path.join(DATA, 'settings.json');
const KEYFILE = path.join(DATA, 'openrouter.key');   // .userdata/ is git-ignored; never returned to the page

// Guardrails the user can change (Settings > Guardrails): sizes and limits, each with its bounds. The protections that
// keep the app safe whatever is typed (the sandbox has no network or files, only the user runs a plan or trains, a cut
// is restored, a plan is checked) are not settings. Only the page's own Settings view changes these: Prometheus is
// kept off its controls (helper.js BLOCK) and outside agents have no settings tool.
const GUARD = {
  maxPrompts: { def: 60, min: 2, max: 2000, int: true, group: 'Experiments', label: 'prompts in one plan', note: 'More than a few hundred Neural net recordings make a large record; raise the record size too.' },
  maxPromptChars: { def: 400, min: 50, max: 4000, int: true, group: 'Experiments', label: 'characters in one prompt' },
  maxAnswerTokens: { def: 32, min: 4, max: 256, int: true, group: 'Experiments', label: 'answer tokens per recording (Neural net and the other instruments)' },
  maxGenerateTokens: { def: 200, min: 4, max: 2000, int: true, group: 'Experiments', label: 'answer tokens per Generate run' },
  maxRecordMB: { def: 8, min: 1, max: 256, int: true, group: 'Experiments', label: 'size of one saved experiment (MB)' },
  historyKept: { def: 300, min: 10, max: 5000, int: true, group: 'Experiments', label: 'experiments kept in the history (the oldest go first)' },
  jobCache: { def: true, bool: true, group: 'Jobs', label: 'reuse results of identical prompts (cache)' },
  jobCacheMax: { def: 4000, min: 0, max: 200000, int: true, group: 'Jobs', label: 'results kept in the cache' },
  jobCourtesy: { def: true, bool: true, group: 'Jobs', label: 'wait while the GPU lacks room or a training job holds it' },
  sandboxSeconds: { def: 20, min: 2, max: 600, int: true, group: 'Analysis', label: 'seconds an analysis may run' },
  sandboxCodeChars: { def: 20000, min: 1000, max: 200000, int: true, group: 'Analysis', label: 'characters of code in one analysis' },
  steerMax: { def: 4, min: 0.5, max: 50, group: 'Features', label: 'largest steering strength (in units of a feature’s largest activation)' },
  saeMaxPieces: { def: 4000000, min: 20000, max: 200000000, int: true, group: 'Features', label: 'pieces an SAE may learn from' },
  verifyOnline: { def: false, bool: true, group: 'Features', label: 'check each new SAE’s rings online (sends the ring records to emergencemachine.com, never the SAE or the model)' },
};
const GUARD_DEFAULTS = Object.fromEntries(Object.entries(GUARD).map(([k, g]) => [k, g.def]));
// No path defaults to a folder on any particular computer: empty means "not set yet", and the views say what to set
// (Settings > Local model, the inside engine's weights folder, the checkpoints folder). The PSI_* variables fill them in.
const DEFAULTS = {
  lastModel: process.env.PSI_MODEL || '',
  modelsDir: process.env.PSI_MODELS_DIR || '',
  llamaDir: '',   // llama-server's folder: settings, else $PSI_LLAMA_DIR (main.js / devserver.js), else llama-server on the PATH (engine.js)
  cacheDir: '',   // downloads and temp files of the Python engines (cacheEnv below); empty: their own defaults
  loraPath: process.env.PSI_LORA || '',
  em: { on: false },   // Emergence Machine content (ui/em.js), off by default so the tool is neutral until turned on. Off: the E-level lens, the framework's views and links go, Prometheus is "Assistant"
  lens: { on: false, think: false, text: '' },
  steering: { path: '', scale: 0, from: 14, to: 14 },
  trace: { modelDir: process.env.PSI_TRACE_MODEL || '', checkpoints: process.env.PSI_CHECKPOINTS || '' },
  updates: { check: false },
  records: { keep: true, maxGB: 2 },   // every result kept as raw data in <data>/records (src/records.js), for PSI Studio and re-analysis; local only   // Settings > Help and about: ask the app's git remote for a newer version at startup (src/update-api.js); off until the user turns it on, and nothing is downloaded without a press
  idle: { minutes: 60 },   // sleep when idle (src/idle.js): stop the engines after this many minutes unused; 0 = never
  helper: { on: false, model: 'anthropic/claude-haiku-4.5', act: true, ask: true, maxTokens: 1500 },
  guardrails: GUARD_DEFAULTS,
  modelThinking: {},   // per external model, the tokens its last replies spent reasoning (learned; see thinkRoom)
  hf: { owner: '', private: true, checkpoints: '' },   // Settings > Hugging Face (src/hf-api.js): where pushes go; the repo training checkpoints come from
  external: { models: ['google/gemini-3.5-flash', 'openai/gpt-5.5', 'anthropic/claude-opus-5.5'], judge: 'google/gemini-2.5-flash', capUsd: 2, spentUsd: 0, maxTokens: 400 },
  // The second model, on this computer, that tests the first (src/local-llm.js): loopback OpenAI-compatible servers only.
  local: { endpoints: require('./local-llm.js').DEFAULT_ENDPOINTS, timeoutS: 300 },
};
const isObj = (x) => x && typeof x === 'object' && !Array.isArray(x);
const merge = (a, b) => { const o = { ...a }; for (const [k, v] of Object.entries(b || {})) o[k] = isObj(v) && isObj(a[k]) ? merge(a[k], v) : v; return o; };

// A settings file that cannot be read is kept (settings.json.corrupt-<time>) and said so (settings.recovered_from_corrupt),
// never silently replaced by the defaults: that would also reset the OpenRouter spending cap and spend.
let settings = (() => {
  let raw = null;
  try { raw = fs.readFileSync(FILE, 'utf8'); } catch (_) { return merge(DEFAULTS, {}); }   // first run: no file yet
  try { return merge(DEFAULTS, JSON.parse(raw)); } catch (e) {
    const copy = `${FILE}.corrupt-${new Date().toISOString().replace(/[:.]/g, '-')}`;
    try { fs.copyFileSync(FILE, copy); } catch (e2) { console.error(`[config] could not keep a copy of the unreadable settings file: ${e2.message}`); }
    console.error(`[config] ${FILE} is not valid JSON (${e.message}); kept as ${copy}, loaded the defaults`);
    return merge(DEFAULTS, { recovered_from_corrupt: true, corrupt_copy: copy });
  }
})();
// The recovery mark is for this session only: written to disk it would claim a recovery on every later start.
// The data folder and what is in it are the user's only (0700 / 0600; Windows keeps its own ACLs).
const mkData = () => fs.mkdirSync(DATA, { recursive: true, mode: 0o700 });
const writePrivate = (f, text) => { mkData(); fs.writeFileSync(f, text, { mode: 0o600 }); try { fs.chmodSync(f, 0o600); } catch (_) { /* not supported here */ } };
const save = () => { try { const { recovered_from_corrupt, corrupt_copy, ...keep } = settings; writePrivate(FILE, JSON.stringify(keep, null, 1)); } catch (e) { console.error(`[config] could not save ${FILE}: ${e.message}`); } };

// The environment of every Python child (inside engine, Grow, lineage, ladder, conversions). settings.cacheDir, else
// $PSI_CACHE_DIR, puts Hugging Face downloads in <cacheDir>/huggingface (an HF_HOME already set wins), temp files in
// <cacheDir>/tmp and the app's own caches in <cacheDir>/psi. Unset: HF_HOME and TMP/TEMP are left as they are, and the
// app's caches go under the data folder (PSI_DATA_DIR/cache, see src/grow.py and src/anatomy.py).
function cacheDir() { return settings.cacheDir || process.env.PSI_CACHE_DIR || ''; }
// Who the Python children sign as (Settings > Profile & keys, src/commons-api.js): set once by main.js / devserver.js.
// Without it nothing is signed (private), whatever the environment the app was started in says.
let SIGNER = () => ({});
function setSigner(fn) { SIGNER = typeof fn === 'function' ? fn : () => ({}); }
function cacheEnv(extra = {}) {
  // The children never get the OpenRouter key or the app's tokens (the trace server's own comes in extra), nor a
  // signing choice, vault or Hugging Face token from the environment: those come only from the app (SIGNER, extra).
  const { OPENROUTER_API_KEY, PSI_TRACE_TOKEN, PSI_MCP_TOKEN, PSI_HF_TOKEN, PSI_VAULT, PSI_VAULT_MODE, PSI_SIGN_AS, PSI_MAKER, PSI_PROFILE_SHA, ...parent } = process.env;
  const env = { ...parent, PYTHONIOENCODING: 'utf-8', PSI_DATA_DIR: DATA }, c = cacheDir();
  if (c) {
    const tmp = path.join(c, 'tmp'); try { fs.mkdirSync(tmp, { recursive: true }); } catch (_) { /* the child reports it */ }
    Object.assign(env, { PSI_CACHE_DIR: c, HF_HOME: process.env.HF_HOME || path.join(c, 'huggingface'), TMP: tmp, TEMP: tmp, TMPDIR: tmp });
  }
  let who = {}; try { who = SIGNER() || {}; } catch (e) { console.error(`[config] signing choice unreadable, children sign nothing: ${e.message}`); }
  return { ...env, ...who, ...extra };
}

const key = () => { try { return fs.readFileSync(KEYFILE, 'utf8').trim(); } catch (_) { return process.env.OPENROUTER_API_KEY || ''; } };
const keyInfo = () => { const k = key(); return { keySet: !!k, keyHint: k ? `${k.slice(0, 6)}\u2026${k.slice(-4)}` : '' }; };

// What the page may see: everything except the key itself.
function view() { return { ...settings, guardrails: guard(), guardrailBounds: GUARD, external: { ...settings.external, ...keyInfo() }, dataDir: DATA }; }
// The guardrails in force: the defaults, with the user's values where they are valid.
function guard() {
  const g = { ...GUARD_DEFAULTS }; for (const [k, v] of Object.entries(settings.guardrails || {})) { const d = GUARD[k]; if (d && (d.bool ? typeof v === 'boolean' : typeof v === 'number' && v >= d.min && v <= d.max)) g[k] = v; }
  if (!emOn()) g.verifyOnline = false;   // nothing goes to emergencemachine.com while its content is off
  return g;
}
const emOn = () => !(settings.em && settings.em.on === false);
// Python children (names, lineage, grow) read PSI_EM: character names only with Emergence Machine content on.
const syncEnv = () => { process.env.PSI_EM = emOn() ? '1' : '0'; };
syncEnv();

// ── paths the page may hand us: local only (a UNC path leaks the Windows login hash), never the data dir ──
const WIN = process.platform === 'win32';
const isUnc = (p) => /^[\\/]{2}/.test(String(p || ''));
const real = (p) => { const r = path.resolve(p); try { return fs.realpathSync.native(r); } catch (_) { return r; } };
const key4 = (p) => (WIN ? p.toLowerCase() : p);
const inside = (f, dir) => { const rel = path.relative(key4(real(dir)), key4(f)); return rel === '' || (!!rel && !rel.startsWith('..') && !path.isAbsolute(rel)); };
const localPath = (v, what) => { if (typeof v !== 'string' || !v || isUnc(v) || isUnc(real(v))) throw new Error(`${what}: only local paths (no network paths).`); return v; };
const isDir = (d) => { try { return fs.statSync(d).isDirectory(); } catch (_) { return false; } };
const pick = (o, ks) => ks.reduce((x, k) => (x && typeof x === 'object' ? x[k] : undefined), o);
const DIRKEYS = [['llamaDir'], ['modelsDir'], ['cacheDir'], ['trace', 'modelDir'], ['trace', 'checkpoints']], FILEKEYS = [['lastModel'], ['loraPath'], ['steering', 'path']];
// Folders must be existing local folders (checked only when they change, so an old value never blocks a save).
function checkPatch(patch, { trusted = false } = {}) {
  // Folders whose programs and files the app runs or loads: llama.cpp (llama-server, the converter) and the cache (the inside engine's
  // saved tensors). Set in settings.json or PSI_LLAMA_DIR / PSI_CACHE_DIR, never from the page or an agent.
  if (!trusted) for (const k of ['llamaDir', 'cacheDir']) if (patch[k] !== undefined && patch[k] !== settings[k]) throw new Error(`${k}: set it in settings.json or ${k === 'llamaDir' ? 'PSI_LLAMA_DIR' : 'PSI_CACHE_DIR'}, not from the page.`);
  if (patch.external !== undefined) {
    const e = patch.external; if (!e || typeof e !== 'object') throw new Error('external: an object');
    if (e.spentUsd !== undefined) throw new Error('external.spentUsd: only the reset button changes it (Settings > OpenRouter).');
    if (e.capUsd !== undefined && !(typeof e.capUsd === 'number' && Number.isFinite(e.capUsd) && e.capUsd >= 0 && e.capUsd <= 1000)) throw new Error('external.capUsd: dollars from 0 to 1000');
    if (e.maxTokens !== undefined && !(Number.isInteger(e.maxTokens) && e.maxTokens >= 1 && e.maxTokens <= 32768)) throw new Error('external.maxTokens: 1 to 32768');
    if (e.models !== undefined && !(Array.isArray(e.models) && e.models.length <= 20 && e.models.every((m) => typeof m === 'string' && m.length < 200))) throw new Error('external.models: a list of model ids');
  }
  if (patch.local !== undefined) {
    const l = patch.local; if (!isObj(l) || Object.keys(l).some((k) => !['endpoints', 'timeoutS'].includes(k))) throw new Error('local: only { endpoints, timeoutS }');
    if (l.endpoints !== undefined) l.endpoints = require('./local-llm.js').checkEndpoints(l.endpoints);
    if (l.timeoutS !== undefined && !(Number.isInteger(l.timeoutS) && l.timeoutS >= 10 && l.timeoutS <= 3600)) throw new Error('local.timeoutS: 10 to 3600 seconds');
  }
  if (patch.hf !== undefined) {
    const h = patch.hf; if (!h || typeof h !== 'object') throw new Error('hf: an object');
    if (h.owner !== undefined && !(typeof h.owner === 'string' && /^([A-Za-z0-9][\w.-]{0,95})?$/.test(h.owner))) throw new Error('hf.owner: a Hugging Face user or organisation name');
    if (h.private !== undefined && typeof h.private !== 'boolean') throw new Error('hf.private: true or false');
    if (h.checkpoints !== undefined && !(typeof h.checkpoints === 'string' && /^([A-Za-z0-9][\w.-]{0,95}\/[A-Za-z0-9][\w.-]{0,95})?$/.test(h.checkpoints))) throw new Error('hf.checkpoints: OWNER/NAME of a model with training-step revisions');
  }
  if (patch.em !== undefined && !(isObj(patch.em) && Object.keys(patch.em).every((k) => k === 'on') && typeof patch.em.on === 'boolean')) throw new Error('em: only { on: true | false }');
  if (patch.helper && patch.helper.maxTokens !== undefined && !(Number.isInteger(patch.helper.maxTokens) && patch.helper.maxTokens >= 1 && patch.helper.maxTokens <= 32768)) throw new Error('helper.maxTokens: 1 to 32768');
  // The extra host names the dev server answers to (devserver.js) are set in settings.json or $PSI_ALLOWED_HOSTS only, never from the page.
  if (patch.allowedHosts !== undefined) throw new Error('allowedHosts: set it in settings.json or PSI_ALLOWED_HOSTS, not from the page.');
  for (const ks of DIRKEYS) { const v = pick(patch, ks); if (v === undefined || v === '' || v === pick(settings, ks)) continue; localPath(v, ks.join('.')); if (!path.isAbsolute(v) || !isDir(v)) throw new Error(`${ks.join('.')}: not a folder on this computer: ${v}`); }
  for (const ks of FILEKEYS) { const v = pick(patch, ks); if (v) localPath(v, ks.join('.')); }
  // The converted-files folder may not exist yet (it is made on the first conversion), but its parent must, and it is never the data folder.
  if (patch.ggufDir) {
    const v = patch.ggufDir; localPath(v, 'ggufDir');
    if (!path.isAbsolute(v) || !(isDir(v) || isDir(path.dirname(path.resolve(v))))) throw new Error(`ggufDir: not a folder on this computer: ${v}`);
    if (inside(real(v), DATA)) throw new Error('ggufDir: that is the app’s private data folder.');
  }
  if (patch.records !== undefined && !(isObj(patch.records) && Object.keys(patch.records).length && Object.entries(patch.records).every(([k, v]) => (k === 'keep' && typeof v === 'boolean') || (k === 'maxGB' && typeof v === 'number' && Number.isFinite(v) && v >= 0.1 && v <= 1000)))) throw new Error('records: only { keep: true | false, maxGB: 0.1 to 1000 }');
  if (patch.updates !== undefined && !(isObj(patch.updates) && Object.keys(patch.updates).every((k) => k === 'check') && typeof patch.updates.check === 'boolean')) throw new Error('updates: only { check: true | false }');
  if (patch.idle !== undefined) {
    const i = patch.idle; if (!i || typeof i !== 'object' || Object.keys(i).some((k) => k !== 'minutes') || !Number.isInteger(i.minutes) || i.minutes < 0 || i.minutes > 1440) throw new Error('idle: only { minutes: 0 (never) to 1440 }');
  }
  if (patch.guardrails !== undefined) {
    const g = patch.guardrails; if (!isObj(g)) throw new Error('guardrails: an object');
    for (const [k, v] of Object.entries(g)) {
      const d = GUARD[k]; if (!d) throw new Error(`guardrails.${k}: not a guardrail`);
      if (d.bool) { if (typeof v !== 'boolean') throw new Error(`guardrails.${k}: true or false`); continue; }
      if (typeof v !== 'number' || !Number.isFinite(v) || (d.int && !Number.isInteger(v)) || v < d.min || v > d.max) throw new Error(`guardrails.${k}: ${d.int ? 'a whole number' : 'a number'} from ${d.min} to ${d.max}`);
    }
  }
  // Whether outside agents (MCP) may change models: only the user sets it, only to true or false (mcp.js reads === true).
  if (patch.agents !== undefined) {
    const a = patch.agents; if (!a || typeof a !== 'object' || !Object.keys(a).length || Object.entries(a).some(([k, v]) => !['allow_changes', 'allow_spend'].includes(k) || typeof v !== 'boolean')) throw new Error('agents: only { allow_changes, allow_spend: true | false }');
  }
}
// Folders whose model files the page may read: the models folder and the folders of the models in the settings.
const modelRoots = () => [settings.modelsDir, settings.lastModel && path.dirname(settings.lastModel), settings.loraPath && path.dirname(settings.loraPath),
  settings.steering && settings.steering.path && path.dirname(settings.steering.path), settings.trace.modelDir, settings.trace.checkpoints,
  path.join(__dirname, '..', 'assets', 'control-vectors'), ggufDir()].filter((d) => typeof d === 'string' && d && !isUnc(d));
// Where Convert to GGUF writes (src/convert-api.js): settings.ggufDir, or a gguf folder inside the models folder.
function ggufDir() { return settings.ggufDir || (settings.modelsDir ? path.join(settings.modelsDir, 'gguf').replace(/\\/g, '/') : ''); }
// A model file the page may read (readHead / readAll): a local .gguf under modelRoots() or picked in a dialog, never in DATA.
function checkModelFile(file, picked) {
  localPath(file, 'file'); const f = real(file);
  if (path.extname(f).toLowerCase() !== '.gguf') throw new Error('Only .gguf model files can be read.');
  // Models built from Commons recipes are the one part of the data folder a page may open (GGUFs only, checked above).
  if (inside(f, DATA) && !inside(f, path.join(DATA, 'commons', 'models'))) throw new Error('That file is in the app’s private data folder.');
  if (inside(f, path.join(DATA, 'commons', 'models'))) return f;
  if (!(picked && picked.has(key4(f))) && !modelRoots().some((d) => inside(f, d))) throw new Error('That file is not in the models folder (Settings > Local model) or a folder of a model in use.');
  return f;
}
// What start() is given: local paths only, and a context size the server can take (256 to 1,048,576 tokens).
function checkStart(file, o = {}) {
  localPath(file, 'model');
  if (o.lora && o.lora.path) localPath(o.lora.path, 'LoRA'); if (o.controlVector && o.controlVector.path) localPath(o.controlVector.path, 'control vector');
  if (o.ctx !== undefined && o.ctx !== null && !(Number.isInteger(o.ctx) && o.ctx >= 256 && o.ctx <= 1048576)) throw new Error(`ctx: a whole number of tokens from 256 to 1048576, not ${JSON.stringify(o.ctx)}`);
}

function update(patch, opts) {
  if (patch && patch.external) delete patch.external.keySet, delete patch.external.keyHint;
  if (patch) { delete patch.guardrailBounds; delete patch.dataDir; }   // read-only parts of the view, if a page sends it back
  checkPatch(patch || {}, opts);
  // Turning Emergence Machine content off turns the E-level lens off with it (it is the framework's prompt); on again, back on.
  if (patch && patch.em && patch.em.on !== emOn()) patch.lens = { ...(patch.lens || {}), on: patch.em.on };
  settings = merge(settings, patch || {}); save(); syncEnv(); return view();
}

function setKey(k) {
  mkData();
  if (!k) { try { fs.unlinkSync(KEYFILE); } catch (_) { /* none */ } } else writePrivate(KEYFILE, String(k).trim());
  return keyInfo();
}

// ── model discovery: GGUF files (the answering engine runs these) and Hugging Face folders (config.json + weights) ──
// A Hugging Face folder is listed with a sign in Settings: llama.cpp cannot load it, but Convert to GGUF can make it a
// file once (src/convert-api.js). The inside engine (Neural net, Anatomy) reads such folders as they are.
const SKIP_DIRS = new Set(['.git', '.cache', 'node_modules', '__pycache__', 'blobs', 'refs', '.locks', '.no_exist']);
const isGguf = (n) => /\.gguf$/i.test(n) && !/lora|cvec|control/i.test(n) && !/\.partial\.gguf$/i.test(n);
const DTYPE_BYTES = { float32: 4, float16: 2, bfloat16: 2, float64: 8, int8: 1, uint8: 1 };
const HEADS = new Map();   // safetensors header reads, by path + size + time: a rescan does not read them again
function safetensorsParams(file) {
  let st; try { st = fs.statSync(file); } catch (_) { return null; }
  const k = `${file}|${st.size}|${st.mtimeMs}`; if (HEADS.has(k)) return HEADS.get(k);
  let n = null, fd = null;
  try {
    fd = fs.openSync(file, 'r'); const b8 = Buffer.alloc(8); fs.readSync(fd, b8, 0, 8, 0);
    const len = Number(b8.readBigUInt64LE(0));
    if (len > 0 && len < 64 * 2 ** 20) {
      const hb = Buffer.alloc(len); fs.readSync(fd, hb, 0, len, 8);
      const h = JSON.parse(hb.toString('utf8')); n = 0;
      for (const [name, t] of Object.entries(h)) if (name !== '__metadata__' && t && Array.isArray(t.shape)) n += t.shape.reduce((a, x) => a * x, 1);
    }
  } catch (_) { n = null; } finally { if (fd !== null) try { fs.closeSync(fd); } catch (_) { /* closed */ } }
  HEADS.set(k, n); return n;
}
// A segment that says nothing on its own (a training step, "final", a snapshot hash) is joined to its parent's name.
const GENERIC = /^(step\d+|final|main|latest|model|checkpoint-\d+|[0-9a-f]{40})$/i;
function hfFolder(d, ents, rel) {
  const names = new Set(ents.filter((e) => e.isFile() || e.isSymbolicLink()).map((e) => e.name));
  if (!names.has('config.json')) return null;
  const st = [...names].filter((n) => /\.safetensors$/i.test(n));
  const bin = [...names].filter((n) => /^(pytorch_model|model)(-\d+-of-\d+)?\.bin$/i.test(n));
  const weights = st.length ? st : bin;
  if (!weights.length) return { skip: 'config.json but no .safetensors or .bin weights' };
  if (names.has('adapter_config.json')) return { skip: 'a LoRA adapter (adapter_config.json), not a whole model' };
  if (names.has('optimizer.pt') || names.has('optimizer.bin')) return { skip: 'a training checkpoint with optimizer state (optimizer.pt), not a finished model' };
  let cfg = null; try { cfg = JSON.parse(fs.readFileSync(path.join(d, 'config.json'), 'utf8')); } catch (_) { return { skip: 'config.json could not be read' }; }
  if (!cfg || typeof cfg !== 'object' || (!cfg.architectures && !cfg.model_type)) return { skip: 'config.json names no model architecture' };
  let bytes = 0; for (const w of weights) { try { bytes += fs.statSync(path.join(d, w)).size; } catch (_) { /* a broken link */ } }
  let params = null, basis = 'counted from the safetensors headers';
  if (st.length) { params = 0; for (const w of st) { const n = safetensorsParams(path.join(d, w)); if (n == null) { params = null; break; } params += n; } }
  if (params == null) { const b = DTYPE_BYTES[cfg.torch_dtype || cfg.dtype] || 2; params = Math.round(bytes / b); basis = `weight bytes / ${b} (${cfg.torch_dtype || cfg.dtype || 'assumed 16-bit'})`; }
  // The name shown: the path under the folder scanned; a Hugging Face cache entry shows as org/name.
  const hub = /(?:^|\/)models--([^/]+)--([^/]+)\/snapshots\/[^/]+$/.exec(rel);
  const segs = rel.split('/').filter(Boolean);
  const name = hub ? `${hub[1]}/${hub[2]}` : rel || path.basename(d);
  const last = hub ? hub[2] : segs.length ? segs[segs.length - 1] : path.basename(d);
  const file = hub ? hub[2] : GENERIC.test(last) && segs.length > 1 ? `${segs[segs.length - 2]}-${last}` : last;
  return { kind: 'hf', path: d.replace(/\\/g, '/'), name, file_name: file.replace(/[^A-Za-z0-9._-]+/g, '-'), gb: +(bytes / 2 ** 30).toFixed(2),
    arch: (cfg.architectures && cfg.architectures[0]) || cfg.model_type || null, model_type: cfg.model_type || null,
    params_estimate: params, params_basis: basis, weights: weights.length, format: st.length ? 'safetensors' : 'bin', dtype: cfg.torch_dtype || cfg.dtype || null };
}
// A GGUF already made from this folder: in the converted-files folder, inside the folder, or beside it under the same name.
function convertedOf(m) {
  const found = [], add = (p) => { if (!found.includes(p)) found.push(p); };
  const out = ggufDir();
  try { for (const n of fs.readdirSync(out)) if (isGguf(n) && n.toLowerCase().startsWith(`${m.file_name.toLowerCase()}-`)) add(path.join(out, n).replace(/\\/g, '/')); } catch (_) { /* no folder yet */ }
  try { for (const n of fs.readdirSync(m.path)) if (isGguf(n)) add(path.join(m.path, n).replace(/\\/g, '/')); } catch (_) { /* gone */ }
  const base = path.basename(m.path).replace(/[-_](hf|bf16|fp16|f16|fp32)$/i, '').toLowerCase(), up = path.dirname(m.path);
  if (!GENERIC.test(base)) try { for (const n of fs.readdirSync(up)) { const l = n.toLowerCase(); if (isGguf(n) && l.startsWith(base) && /^[-_.]/.test(l.slice(base.length))) add(path.join(up, n).replace(/\\/g, '/')); } } catch (_) { /* gone */ }
  return found;
}
let LISTED = new Map();   // Hugging Face folders the last listing returned: only these may be converted (convert-api.js)
function listModels(dir) {
  const root = dir || settings.modelsDir; const out = [], hf = [], skipped = [];
  if (!root) throw new Error('No models folder is set yet: enter the folder that holds your .gguf files in Settings > Local model and press Scan.');
  localPath(root, 'models folder');
  const seen = new Set();
  const walk = (d, depth, base) => {
    let ents = []; try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch (_) { return; }
    const rel = path.relative(base, d).replace(/\\/g, '/');
    const h = hfFolder(d, ents, rel);
    if (h && h.kind) { const k = key4(real(d)); if (!seen.has(k)) { seen.add(k); hf.push(h); } }
    else if (h && h.skip) skipped.push({ path: d.replace(/\\/g, '/'), why: h.skip });
    for (const e of ents) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) { if (depth < 4 && !SKIP_DIRS.has(e.name) && !e.name.endsWith(".partial") && !(h && h.kind)) walk(p, depth + 1, base); }
      else if (e.isFile() && depth <= 2 && isGguf(e.name)) {
        const k = key4(real(p)); if (seen.has(k)) continue; seen.add(k);
        let size = 0; try { size = fs.statSync(p).size; } catch (_) { /* gone */ }
        out.push({ kind: 'gguf', path: p.replace(/\\/g, '/'), name: e.name, gb: +(size / 2 ** 30).toFixed(2) });
      }
    }
  };
  walk(root, 0, root);
  // The configured roots outside the folder scanned (the inside engine's weights, the converted-files folder, a model in use).
  for (const r of modelRoots()) if (isDir(r) && !inside(real(r), root)) walk(r, 0, path.dirname(r));
  for (const m of hf) { const c = convertedOf(m); m.has_gguf = c[0] || null; m.converted = c; }
  LISTED = new Map(hf.map((m) => [key4(real(m.path)), m]));
  return { dir: root, models: out.sort((a, b) => a.name.localeCompare(b.name)), hf: hf.sort((a, b) => a.name.localeCompare(b.name)), skipped, gguf_dir: ggufDir() };
}
// The listing's entry for a Hugging Face folder, or null (listing the models folder again if the app restarted since).
function listedHf(dir) {
  localPath(dir, 'model folder'); const k = key4(real(dir));
  if (!LISTED.has(k)) try { listModels(); } catch (_) { /* the models folder is gone */ }
  return LISTED.get(k) || null;
}

// ── external models through OpenRouter (server side) ──────────────────────
const OR = 'https://openrouter.ai/api/v1';
const headers = () => ({ Authorization: `Bearer ${key()}`, 'Content-Type': 'application/json', 'HTTP-Referer': 'https://emergencemachine.com', 'X-Title': 'Parameter Space Inspector' });
async function orFetch(route, body) {
  if (!key()) throw new Error('No OpenRouter key: add one in Settings.');
  const r = await fetch(`${OR}${route}`, body ? { method: 'POST', headers: headers(), body: JSON.stringify(body) } : { headers: headers() });
  const d = await r.json().catch(() => ({}));
  if (r.status === 402) throw new Error('OpenRouter: insufficient credits. Add credit at openrouter.ai/settings/credits.');
  if (!r.ok) throw new Error(`OpenRouter ${r.status}: ${(d.error && d.error.message) || r.statusText}`);
  return d;
}

// Prices per token from OpenRouter's model list, kept for an hour: the spending cap is checked against an estimate made
// BEFORE a request is sent, never only after the money is spent.
let PRICES = null, PRICES_AT = 0;
async function prices() {
  if (PRICES && Date.now() - PRICES_AT < 3600e3) return PRICES;
  const d = await orFetch('/models');
  PRICES = new Map((d.data || []).map((m) => [m.id, { in: +((m.pricing || {}).prompt) || 0, out: +((m.pricing || {}).completion) || 0, req: +((m.pricing || {}).request) || 0, known: !!m.pricing,
    effort: (m.supported_parameters || []).includes('reasoning_effort') || (m.supported_parameters || []).includes('reasoning') }]));
  PRICES_AT = Date.now(); return PRICES;
}
// Thinking room, learned per model. "max tokens per answer" is the answer's budget; a model that reasons spends tokens
// thinking first, from the same allowance. After each reply OpenRouter says how many went on reasoning; the last 8 counts
// are kept (settings.modelThinking) and the next request adds room for twice the largest of them (at least 500, at most
// THINK_MAX): a model's thinking varies from question to question (Muse Spark: 254, then 373), and room only reserves
// cap while the request runs; tokens are charged as used. Before anything is learned, models known to always reason
// start with 2,000.
const THINK_MAX = 8000, THINK_KEEP = 8;
const ALWAYS_REASONS = (m) => /claude-(opus|sonnet|fable|mythos)-5/.test(m) || /^meta\/muse-(spark|wonder)/.test(m);
const thinkSeen = (model) => ((settings.modelThinking || {})[model] || {}).recent || [];
function thinkRoom(model) {
  const seen = thinkSeen(model);
  if (seen.length) { const most = Math.max(...seen); return most > 0 ? Math.min(THINK_MAX, Math.max(500, Math.ceil(most * 2 / 100) * 100)) : 0; }
  return ALWAYS_REASONS(model) ? 2000 : 0;
}
function learnThinking(model, tokens) {
  if (!(tokens >= 0)) return;
  const all = settings.modelThinking || (settings.modelThinking = {});
  const recent = [...thinkSeen(model), Math.round(tokens)].slice(-THINK_KEEP);
  all[model] = { recent, at: new Date().toISOString() };
}
// The exact body a chat request sends (max_tokens includes the thinking room), so the estimate and the request agree.
function chatBody(args) {
  const ext = settings.external, model = args.model;
  const body = { model, messages: args.messages || [{ role: 'user', content: args.prompt || '' }], temperature: args.temperature ?? 0, max_tokens: args.maxTokens || ext.maxTokens, usage: { include: true } };
  if (Array.isArray(args.tools) && args.tools.length) body.tools = args.tools;
  const room = thinkRoom(model), p = PRICES && PRICES.get(model);
  if (/^openai\/gpt-5/.test(model)) body.reasoning = { effort: 'minimal' };
  // A model that has shown it reasons by default (or is known to) is asked for little of it, where it takes an effort.
  else if (ALWAYS_REASONS(model) || (room > 0 && p && p.effort)) body.reasoning = { effort: 'low' };
  body.max_tokens += room;
  return body;
}
// Tokens a reply spent reasoning: OpenRouter's count when it gives one; a reply cut off with nothing written spent them all.
// (Cut off with nothing written, every token written was thinking, whatever the count says: Muse Glimmer reported 71
// reasoning tokens of 120 written, with no answer, and room learned from 71 never grew.)
function reasoningTokens(d, msg, finish) {
  const det = d.usage && d.usage.completion_tokens_details;
  if (!msg.content && finish === 'length' && d.usage && typeof d.usage.completion_tokens === 'number') return d.usage.completion_tokens;
  if (det && typeof det.reasoning_tokens === 'number') return det.reasoning_tokens;
  return null;
}
// Upper-leaning estimate: prompt tokens as characters / 3 (English runs about 4 per token), every allowed output token
// charged at the output price, plus any per-request price. Money in flight (reserved) counts against the cap.
let reserved = 0;
async function estimateBody(body) {
  const P = await prices(), p = P.get(body.model);
  if (!p || !p.known) throw new Error(`No price for ${body.model} in OpenRouter's model list, so the spending cap cannot be checked; nothing was sent.`);
  const chars = JSON.stringify(body.messages).length + (body.tools ? JSON.stringify(body.tools).length : 0);
  const promptTok = Math.ceil(chars / 3);
  const usd = +(promptTok * p.in + body.max_tokens * p.out + p.req).toFixed(6);
  const ext = settings.external;
  return { usd, prompt_tokens_est: promptTok, max_tokens: body.max_tokens, price_per_mtok: { in: +(p.in * 1e6).toFixed(3), out: +(p.out * 1e6).toFixed(3) },
    basis: 'prompt tokens estimated as characters / 3; every allowed output token charged at the output price (an upper-leaning estimate)',
    spent: +ext.spentUsd.toFixed(5), reserved: +reserved.toFixed(5), cap: ext.capUsd, remaining: +Math.max(0, ext.capUsd - ext.spentUsd - reserved).toFixed(5),
    fits: ext.spentUsd + reserved + usd <= ext.capUsd };
}

// Models on the user's ChatGPT plan (src/chatgpt.js) are named chatgpt/<slug>: no money, so no estimate or cap.
let CHATGPT = null;
const setChatgpt = (c) => { CHATGPT = c; };
// Models on this computer (src/local-llm.js) are named local/<endpoint>/<model>: no money either, so no estimate or cap.
let LOCAL = null;
const setLocal = (l) => { LOCAL = l; };
async function external(args = {}) {
  const a = args.action;
  if (LOCAL && LOCAL.isLocalModel(args.model) && (a === 'chat' || a === 'estimate')) {
    if (a === 'estimate') return { usd: 0, per_call_usd: 0, n: Math.max(1, Math.floor(+args.n || 1)), fits: true, local: true, basis: 'a model on this computer: no money' };
    return LOCAL.chat(args);
  }
  if (a === 'local_status') { if (!LOCAL) throw new Error('local models are not set up in this build'); return LOCAL.status(); }
  if (CHATGPT && CHATGPT.isPlanModel(args.model) && (a === 'chat' || a === 'estimate')) {
    if (a === 'estimate') return { usd: 0, per_call_usd: 0, n: Math.max(1, Math.floor(+args.n || 1)), fits: true, plan: true, basis: 'ChatGPT plan: no money, counts against the plan’s usage limits' };
    return CHATGPT.chat(args);
  }
  if (a === 'status') {
    const [k, c] = await Promise.all([orFetch('/auth/key').catch((e) => ({ error: e.message })), orFetch('/credits').catch(() => ({}))]);
    const kd = k.data || {}, cd = c.data || {};
    return { ok: !k.error, error: k.error, label: kd.label, keyLimit: kd.limit, keyRemaining: kd.limit_remaining, keyUsage: kd.usage,
      accountCredits: cd.total_credits, accountUsed: cd.total_usage, spentThisApp: settings.external.spentUsd, capUsd: settings.external.capUsd };
  }
  if (a === 'models') {
    // Models on the user's ChatGPT plan come first when signed in: $0, marked plan, usable wherever OpenRouter models are.
    let plan = []; if (CHATGPT) { const st = CHATGPT.status(); if (st.signedIn && st.planUsage) try { plan = (await CHATGPT.call({ action: 'models' })).models.map((m) => ({ id: m.id, name: `${m.name} (ChatGPT plan)`, in: 0, out: 0, ctx: null, plan: true })); } catch (_) { plan = []; } }
    // Then the models on this computer (the tester first): $0, marked local. Without an OpenRouter key these are the list.
    let local = []; if (LOCAL) try { local = await LOCAL.models(); } catch (_) { local = []; }
    plan = local.concat(plan);
    if ((plan.length) && !key()) return { models: plan };
    const d = await orFetch('/models');
    return { models: plan.concat((d.data || []).filter((m) => !/:(free|beta|extended)$/.test(m.id) && (m.architecture ? (m.architecture.output_modalities || ['text']).includes('text') : true))
      .map((m) => ({ id: m.id, name: m.name, in: +(m.pricing.prompt * 1e6).toFixed(3), out: +(m.pricing.completion * 1e6).toFixed(3), ctx: m.context_length }))) };
  }
  // What a chat request would cost at most, without sending it (n: how many such requests).
  if (a === 'estimate') {
    if (!args.model) throw new Error('No model chosen.');
    await prices(); const e = await estimateBody(chatBody(args)), n = Math.max(1, Math.floor(+args.n || 1));
    const usd = +(e.usd * n).toFixed(6), ext = settings.external;
    return { ...e, per_call_usd: e.usd, n, usd, fits: ext.spentUsd + reserved + usd <= ext.capUsd };
  }
  if (a === 'chat') {
    const ext = settings.external;
    if (ext.spentUsd >= ext.capUsd) throw new Error(`Spending cap reached ($${ext.spentUsd.toFixed(2)} of $${ext.capUsd}). Raise it in Settings.`);
    if (!args.model) throw new Error('No model chosen.');
    await prices();   // chatBody reads which models take a reasoning effort
    const t0 = Date.now();
    // One request under the cap. The cap is checked BEFORE it is sent: spent + money in flight + its estimate.
    const send = async (body) => {
      const est = await estimateBody(body);
      if (!est.fits) throw new Error(`Spending cap: this request could cost up to $${est.usd.toFixed(4)} (estimate: ${est.prompt_tokens_est} prompt tokens + ${est.max_tokens} output tokens), and $${(ext.spentUsd + reserved).toFixed(4)} of the $${ext.capUsd} cap is spent or in flight, so it was not sent. Raise the cap in Settings or ask for fewer tokens.`);
      // Re-checked with no await in between, so parallel requests each count the money the others reserved.
      if (ext.spentUsd + reserved + est.usd > ext.capUsd) throw new Error(`Spending cap: $${(ext.spentUsd + reserved).toFixed(4)} of the $${ext.capUsd} cap is spent or in flight, and this request could cost up to $${est.usd.toFixed(4)}, so it was not sent.`);
      reserved += est.usd;
      let d; try { d = await orFetch('/chat/completions', body); } finally { reserved = Math.max(0, reserved - est.usd); }
      const reported = d.usage && d.usage.cost != null;
      const cost = reported ? +d.usage.cost : est.usd;   // no cost reported: the estimate is charged, so the cap errs on the safe side
      settings.external.spentUsd = +(settings.external.spentUsd + cost).toFixed(5);
      const ch = (d.choices || [])[0] || {}, msg = ch.message || {}, think = reasoningTokens(d, msg, ch.finish_reason);
      if (think != null) learnThinking(body.model, think);
      save();
      return { d, est, cost, reported, msg, finish: ch.finish_reason, think, empty: !msg.content && !(msg.tool_calls && msg.tool_calls.length) && ch.finish_reason === 'length' };
    };
    let body = chatBody(args), r = await send(body), first = null;
    const model = body.model;
    // It spent every token thinking and wrote nothing: once, again with the room just learned (its count, plus a
    // quarter), if that is more room and fits the cap. Both requests are charged and both are reported.
    if (r.empty) {
      // The retry's room: what was just learned, at least twice the room that ran out and at least 1,000, up to THINK_MAX.
      const again = chatBody(args), base = args.maxTokens || ext.maxTokens, was = body.max_tokens - base;
      again.max_tokens = base + Math.min(THINK_MAX, Math.max(again.max_tokens - base, 2 * was, 1000));
      if (!again.reasoning && PRICES.get(model) && PRICES.get(model).effort) again.reasoning = { effort: 'low' };
      if (again.max_tokens > body.max_tokens) {
        first = { cost: r.cost, max_tokens: body.max_tokens, thinking_tokens: r.think };
        try { body = again; r = await send(body); } catch (e) { throw new Error(`${model} spent its whole ${first.max_tokens}-token budget thinking ($${first.cost.toFixed(4)}), and the retry with ${again.max_tokens} was not sent: ${e.message}`); }
      }
    }
    // Still nothing: say so, rather than hand back an empty answer that reads as the model's reply.
    if (r.empty) {
      const spent = r.cost + (first ? first.cost : 0);
      throw new Error(`${model} used its whole output budget (${body.max_tokens} tokens${r.d.usage && r.d.usage.completion_tokens ? `, ${r.d.usage.completion_tokens} written` : ''}) on reasoning and gave no answer${first ? ', twice' : ''}; it cost $${spent.toFixed(4)}. ${thinkRoom(model) >= THINK_MAX ? 'It thinks more than the app allows (8,000 tokens): choose a model that reasons less.' : 'Ask again: the next request leaves it more room.'}`);
    }
    const { d, est, cost, reported, msg } = r;
    return { model, text: msg.content || '', toolCalls: msg.tool_calls || null, message: args.tools ? msg : undefined, finish: r.finish, cost: +(cost + (first ? first.cost : 0)).toFixed(6),
      cost_source: reported ? 'reported by OpenRouter' : 'estimate (OpenRouter reported no cost; the estimate was charged to the cap)',
      estimate: { usd: est.usd, prompt_tokens_est: est.prompt_tokens_est, max_tokens: est.max_tokens, basis: est.basis },
      tokens: d.usage ? { in: d.usage.prompt_tokens, out: d.usage.completion_tokens } : null,
      thinking: { tokens: r.think, room: body.max_tokens - (args.maxTokens || ext.maxTokens), learned_from: thinkSeen(model).length },
      retried: first, ms: Date.now() - t0, spentThisApp: settings.external.spentUsd };
  }
  if (a === 'resetSpend') { settings.external.spentUsd = 0; save(); return { spentThisApp: 0 }; }
  throw new Error(`unknown external action ${a}`);
}

// ── neuron names: hypotheses, drafted by Prometheus or written by the user, with their history ──
const NAMESFILE = path.join(DATA, 'neuron-names.json');
function names(args = {}) {
  let all = Object.create(null); try { all = Object.assign(Object.create(null), JSON.parse(fs.readFileSync(NAMESFILE, 'utf8'))); } catch (_) { /* none yet */ }
  const model = String(args.model || 'model'); if (['__proto__', 'constructor', 'prototype'].includes(model)) throw new Error('model: not a model name');
  all[model] = Object.assign(Object.create(null), all[model] || {});
  if (args.action === 'set') {
    for (const e of args.entries || []) {
      const key = `L${e.layer}#${e.neuron}`, prev = all[model][key];
      const rec = { name: String(e.name || '').slice(0, 80), why: String(e.why || '').slice(0, 200), confidence: e.confidence || 'low', source: e.source === 'you' ? 'you' : 'prometheus', concept: e.concept || (prev && prev.concept) || '', at: new Date().toISOString() };
      if (prev && prev.source === 'you' && rec.source !== 'you') continue;   // a name you wrote is never overwritten by a draft
      rec.history = ((prev && prev.history) || []).concat(prev ? [{ name: prev.name, source: prev.source, at: prev.at }] : []).slice(-10);
      all[model][key] = rec;
    }
    writePrivate(NAMESFILE, JSON.stringify(all, null, 1));
  }
  return { model, names: all[model] };
}

module.exports = { get settings() { return settings; }, guard, emOn, GUARD, view, update, setKey, listModels, listedHf, ggufDir, cacheDir, cacheEnv, setSigner, setChatgpt, setLocal, external, names, DATA, checkPatch, checkModelFile, checkStart, real, key4, localPath, inside };
