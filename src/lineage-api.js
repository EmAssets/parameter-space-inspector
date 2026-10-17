// SPDX-License-Identifier: Apache-2.0
// Seeds and their family trees, for the Lineage view, Prometheus and MCP: verify a model's signed lineage, adopt
// a seed from How AI Emerges (give it a signed identity), grow it (a job that trains for minutes and reports as it
// goes), and the scorecard (the reader's line next to Pythia and Qwen). The work is done by src/lineage.py and
// src/grow.py; the signing key stays in .userdata/ and never reaches the page.
'use strict';
const { python: PY } = require('./python.js');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const os = require('os');
let gpuState = null; try { ({ gpuState } = require('./gpu.js')); } catch (_) { /* no GPU reading on this platform: the record says "not read" */ }

const SRC = __dirname;
const REFS = path.join(SRC, '..', 'research', 'results', 'scorecard-refs.json');

// The one place Grow's limits live: the page (lineage.js) and the MCP tool (mcp.js) both read these.
const LIMITS = Object.freeze({
  width: Object.freeze({ choices: [1, 2], default: 2 }),
  depth: Object.freeze({ min: 0, max: 8, default: 2, choices: [0, 1, 2, 4] }),
  steps: Object.freeze({ min: 100, max: 20000, default: 1500, step: 100 }),
  data: Object.freeze({ choices: ['street', 'all', 'file'], default: 'all' }),
  device: Object.freeze({ choices: ['cpu', 'cuda'], default: 'cpu' }),
  // The understanding tests' noise floor (src/scorecard.py FLOOR / APPEARED): below `appeared` nothing is signed as appeared.
  understanding: Object.freeze({ of: 9, floor: 6, appeared: 8 }),
  spillMB: 256,   // shared (system) memory a GPU process may use before it counts as spilled (src/gpu.js, src/grow.py SPILL_MB)
});
const SPILL_MB = LIMITS.spillMB;
const MB = 2 ** 20;

function createLineage({ dataDir, log = () => {} }) {
  const REG = path.join(dataDir, 'lineage.json');   // every model of every tree this install has made or adopted
  const jobs = new Map(); let nextJob = 1;

  const readReg = () => { try { return JSON.parse(fs.readFileSync(REG, 'utf8')); } catch (_) { return { models: {} }; } };
  const writeReg = (r) => { fs.mkdirSync(dataDir, { recursive: true }); fs.writeFileSync(REG, JSON.stringify(r, null, 1)); };
  function remember(info, file) {
    if (!info || !info.record) return;
    const r = readReg(), rec = info.record;
    const prev = r.models[rec.id] || {};
    // A scorecard measured later (the "Measure it" button) lives only here, unsigned; a signed record's own score replaces nothing of it.
    const score = rec.score && rec.score.after ? rec.score : prev.score;
    r.models[rec.id] = { id: rec.id, file: file.replace(/\\/g, '/'), name: info.name, v: rec.v || 1, gen: rec.gen, parent: rec.parent, root: rec.root, op: rec.op,
      street: rec.street, data: rec.data, shape: rec.shape, train: rec.train, conditions: rec.conditions || null, appeared: rec.appeared, appeared_floor: rec.appeared_floor || null, score, created: rec.created, status: info.status, by_em: info.by_em };
    writeReg(r);
  }

  // Run a Python helper; every printed line that is JSON goes to onLine; resolves with the last one.
  function py(script, args, onLine = () => {}, onProc = () => {}) {
    return new Promise((resolve, reject) => {
      const p = spawn(PY(), [path.join(SRC, script), ...args], { cwd: SRC, windowsHide: true,
        env: require('./config.js').cacheEnv() });
      onProc(p);
      let buf = '', last = null, err = '';
      p.stdout.on('data', (d) => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); try { last = JSON.parse(line); onLine(last); } catch (_) { /* a library's own output */ } } });
      p.stderr.on('data', (d) => { err = (err + d).slice(-3000); });
      p.on('exit', (code) => {
        if (buf.trim()) { try { last = JSON.parse(buf.trim()); onLine(last); } catch (_) { /* not JSON */ } }
        if (code === 0 && last) return resolve(last);
        let msg = err.split('\n').filter((l) => l.trim() && !/Warning|warn\(|it\/s\]/.test(l)).slice(-3).join(' ');
        try { const j = JSON.parse(msg.match(/\{.*\}/)?.[0] || ''); if (j.error) msg = j.error; } catch (_) { /* plain text */ }
        reject(new Error(msg || `${script} stopped (code ${code})`));
      });
    });
  }

  // Paths from the page or an agent become absolute local files of the right kind, outside the data dir (never an
  // option for the Python side's argv, never a network path, never the signing key).
  // Real paths on both sides, so a junction, symlink or 8.3 short name into the data folder is still the data folder.
  const inData = (f) => { const c = require('./config.js'); return c.inside(c.real(f), dataDir); };
  function arg(file, exts, what) {
    if (typeof file !== 'string' || !file.trim() || /^[\\/]{2}/.test(file)) throw new Error(`${what}: a local file path is needed`);
    const f = path.resolve(file.trim());
    if (!exts.includes(path.extname(f).toLowerCase())) throw new Error(`${what}: only ${exts.join(' ')} files`);
    if (inData(f)) throw new Error(`${what}: that file is in the app's private data folder`);
    return f;
  }
  const gguf = (f) => arg(f, ['.gguf'], 'model');
  const int = (v, d, lo, hi) => { const n = Math.round(Number(v ?? d)); if (!Number.isFinite(n) || n < lo || n > hi) throw new Error(`grow: ${v} is out of range`); return String(n); };

  async function info(file) { file = gguf(file); const r = await py('lineage.py', ['info', file]); if (r.record) remember(r, file); return r; }
  async function adopt(file) { file = gguf(file); const r = await py('lineage.py', ['adopt', file]); const out = path.join(path.dirname(file), r.file); remember(r, out); return { ...r, path: out.replace(/\\/g, '/') }; }

  // One reading of the GPU for the record: the card, what is free, and this grow's own process (its dedicated and shared memory).
  async function snapshot(pid) {
    if (!gpuState) return { at: new Date().toISOString(), ok: false, reason: 'no_reader', spill_detect: 'unavailable' };
    let g = null; try { g = await gpuState({ maxAgeMs: 0 }); } catch (e) { return { at: new Date().toISOString(), ok: false, reason: 'sensor_failed', detail: e.message, spill_detect: 'unavailable' }; }
    if (!g || !g.ok) return { at: (g && g.at) || new Date().toISOString(), ok: false, reason: (g && g.reason) || 'no_gpu', detail: g && (g.detail || g.note), spill_detect: 'unavailable' };
    const me = pid && (g.holders || []).find((h) => h.pid === pid);
    return { at: g.at || new Date().toISOString(), ok: true, gpu: g.gpu, usedMB: g.usedMB, freeMB: g.freeMB, totalMB: g.totalMB,
      spill_detect: g.spill_detect === 'unavailable' ? 'unavailable' : 'read', this: me ? { mb: me.mb, sharedMB: me.sharedMB } : null,
      busy: (g.busy || []).filter((h) => h.pid !== pid).map((h) => h.label) };
  }
  // Would a grow on the GPU crowd out a long job, or not fit? The same question app.js asks before it starts an engine
  // (window.PSIMachine.holdBack). Returns null when it is fine, or a sentence saying what holds the card.
  async function holdBack(needMB) {
    if (!gpuState) return null;
    let g = null; try { g = await gpuState({ maxAgeMs: 0 }); } catch (_) { return null; }
    if (!g || !g.ok) return g && g.reason === 'no_gpu' ? 'no GPU was found' : null;   // unreadable: not a reason to refuse
    if (g.busy && g.busy.length) return `${g.busy[0].label} is using the GPU (${(g.busy[0].mb / 1024).toFixed(1)} GB)`;
    if (needMB && g.freeMB != null && needMB > g.freeMB) return `it needs about ${(needMB / 1024).toFixed(1)} GB of GPU memory and ${(g.freeMB / 1024).toFixed(1)} GB is free`;
    return null;
  }
  // Rough GPU memory a grow needs: fp32 weights, gradients and two Adam moments (16 bytes a number), plus activations for a
  // batch of 32 short sequences and PyTorch's own, from the parent's size in the registry (0 when the parent is not known).
  function needMB(file, o) {
    const m = Object.values(readReg().models).find((x) => x.file.toLowerCase() === file.replace(/\\/g, '/').toLowerCase());
    const p0 = m && m.shape && m.shape.params, L = m && m.shape && m.shape.layers;
    if (!p0) return 0;
    const w = +(o.width || LIMITS.width.default), d = +(o.depth ?? LIMITS.depth.default);
    const p1 = p0 * w * w * (L ? (L + d) / L : 1);
    return Math.round((p1 * 16 * 1.5) / MB + 900);
  }

  function grow(file, o = {}) {
    file = gguf(file);
    const data = o.data || LIMITS.data.default, device = o.device || LIMITS.device.default;
    if (!LIMITS.data.choices.includes(data)) throw new Error(`grow: unknown data ${o.data}`);
    if (!LIMITS.device.choices.includes(device)) throw new Error(`grow: unknown device ${o.device}`);
    const width = Math.round(Number(o.width ?? LIMITS.width.default));
    if (!LIMITS.width.choices.includes(width)) throw new Error(`grow: width ${o.width} is not one of ${LIMITS.width.choices.join(', ')}`);
    const text = data === 'file' && o.textFile ? arg(o.textFile, ['.txt', '.md', '.json'], 'text file') : null;
    const id = String(nextJob++), job = { id, file, started: Date.now(), lines: [], done: false, error: null, result: null, gpu: {}, device };
    jobs.set(id, job);
    const args = ['--width', String(width), '--depth', int(o.depth, LIMITS.depth.default, LIMITS.depth.min, LIMITS.depth.max), '--data', data,
      '--steps', int(o.steps || LIMITS.steps.default, LIMITS.steps.default, LIMITS.steps.min, LIMITS.steps.max), '--device', device];
    if (text) args.push('--file', text);
    const gpuFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'psi-grow-')), 'gpu.json');   // a new private folder: no planted file or link
    args.push('--gpu-file', gpuFile);
    const writeSnaps = () => { try { fs.writeFileSync(gpuFile, JSON.stringify(job.gpu)); } catch (_) { /* the record then says "not read" */ } };
    const snap = async (when) => { job.gpu[when] = await snapshot(job.proc && job.proc.pid); writeSnaps(); };
    args.push('--', file);   // the parent last, after '--': never read as an option
    log(`grow ${path.basename(file)}: ${args.slice(0, -2).join(' ')}`);
    const start = async () => {
      // A grow on the GPU runs the same hold-back check as an engine start, unless the caller said "anyway" after seeing why.
      if (device === 'cuda' && !o.anyway) {
        const why = await holdBack(needMB(file, o));
        if (why) { job.heldBack = why; throw new Error(`not started: ${why}. Choose CPU, wait for the other job, or grow anyway.`); }
      }
      let mid = false;
      return py('grow.py', args, (l) => {
        if (l.stage !== 'done') job.lines.push(l);
        if (l.stage === 'train' && !mid && l.of && l.step >= l.of / 2) { mid = true; snap('middle').catch(() => {}); }
        if (l.stage === 'scored') snap('end').catch(() => {});   // grow.py waits up to 10 s for this before it signs
      }, (p) => { job.proc = p; snap('start').catch(() => {}); });
    };
    start()
      .then((r) => { job.result = r; job.done = true; if (r.file) info(r.file).catch(() => {}); })
      .catch((e) => { job.error = job.cancelled ? 'cancelled' : e.message; job.done = true; })
      .finally(() => { job.proc = null; try { fs.unlinkSync(gpuFile); } catch (_) { /* never written */ } });
    return { job: id, gpuFile: path.basename(gpuFile) };
  }
  // Stop a grow job: the Python process is ended; grow.py writes the grown model only when training finishes, so nothing half-made is kept.
  function cancel(id) {
    const j = jobs.get(String(id)); if (!j) throw new Error(`no grow job ${id}`);
    if (!j.done && j.proc) { j.cancelled = true; try { j.proc.kill(); } catch (_) { /* already gone */ } }
    return { id: j.id, cancelled: !!j.cancelled };
  }
  function job(id) {
    const j = jobs.get(String(id)); if (!j) throw new Error(`no grow job ${id}`);
    const spilled = Object.values(j.gpu || {}).some((g) => g && g.this && g.this.sharedMB >= SPILL_MB);
    return { id: j.id, file: j.file, device: j.device, seconds: Math.round((Date.now() - j.started) / 1000), done: j.done, error: j.error, heldBack: j.heldBack || null,
      progress: j.lines.slice(-40), result: j.result, gpu: j.gpu, spilled, spill_detect: Object.values(j.gpu || {}).some((g) => g && g.spill_detect === 'unavailable') ? 'unavailable' : 'read' };
  }

  function scorecard() {
    let refs = null; try { refs = JSON.parse(fs.readFileSync(REFS, 'utf8')); } catch (_) { /* not measured yet */ }
    const models = Object.values(readReg().models).sort((a, b) => (a.root || '').localeCompare(b.root || '') || a.gen - b.gen);
    const u = LIMITS.understanding;
    return { refs: refs ? refs.rows : [], measured: refs && refs.measured, lineage: models, limits: LIMITS,
      note: `Rows are measured the same way (src/scorecard.py). Bits per character: lower is better, comparable across alphabets for trained models (an untrained word-piece model starts near 3.3, an untrained letter model near 8.2). Understanding: ${u.of} two-choice tests; an untrained Pythia passes ${u.floor} of them, so below ${u.appeared} is within the noise floor and is not signed as understanding. Copying: nats a repeated sequence saves the second time (the phase shift). Look-back heads: heads that attend to what followed last time. A row's score is signed when it is in the model's record; one taken later with Measure it is measured later, unsigned. Frontier models give no access to their insides: their tests are in the Fight class view.` };
  }

  async function call(a = {}) {
    switch (a.action) {
      case 'info': return info(a.file);
      case 'adopt': return adopt(a.file);
      case 'grow': return grow(a.file, a);
      case 'job': return job(a.job);
      case 'cancel': return cancel(a.job);
      case 'scorecard': return scorecard();
      case 'score': {   // measure a model already in the tree that has no scorecard yet: kept in the registry, not in its signed record, and said so
        const r = readReg(), m = r.models[a.id]; if (!m) throw new Error(`no model ${a.id} in the tree`);
        const after = await py('lineage.py', ['score', m.file]);
        m.score = { ...(m.score || {}), after: { ...after, measured_later: true, unsigned: true, measured_at: new Date().toISOString() } }; writeReg(r); return m;
      }
      case 'limits': return LIMITS;
      case 'holdback': return { why: await holdBack(needMB(gguf(a.file), a)), needMB: needMB(gguf(a.file), a) };
      case 'marks': return py('fingerprint.py', ['check', gguf(a.file)]);   // ask the model for its signatures (fingerprint.py)
      case 'tree': return readReg();
      default: throw new Error(`unknown lineage action ${a.action}`);
    }
  }
  return { call, info, adopt, grow, job, scorecard, limits: LIMITS };
}

module.exports = { createLineage, LIMITS };
