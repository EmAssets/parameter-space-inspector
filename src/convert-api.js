// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Convert to GGUF, for Settings > Models: a Hugging Face model folder (config.json + safetensors) becomes one
// GGUF file the answering engine (llama.cpp) can load. llama.cpp's own converter does the work (src/to_gguf.py wraps
// it for progress and plain-words errors); 4-bit goes through a 16-bit file and llama-quantize. A background job with
// progress, Stop (the process is ended and the half-written file deleted) and a job registry in the data folder
// (convert-jobs.json, the last 20; see convertStatus). It runs on the CPU, writes one file into the converted-files folder, and changes nothing else.
'use strict';
const { python: PY } = require('./python.js');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { findLlama } = require('./engine.js');

const SRC = __dirname;
// Bytes per parameter of the output, for the size estimate (Q8_0: 34 bytes per block of 32; Q4_K_M: about 4.85 bits).
const PRECISIONS = Object.freeze({
  q8_0: Object.freeze({ label: '8-bit (Q8_0)', outtype: 'q8_0', bpp: 34 / 32 }),
  f16: Object.freeze({ label: '16-bit (F16)', outtype: 'f16', bpp: 2 }),
  q4_k_m: Object.freeze({ label: '4-bit (Q4_K_M)', outtype: 'f16', bpp: 0.61, quantize: 'Q4_K_M' }),
});
const DEFAULT = 'q8_0';
const GB = 2 ** 30;

function createConverter({ config, dataDir, log = () => {} }) {
  // The job registry: every job of this session, and the last 20 records on disk (convert-jobs.json), so a reloaded page
  // finds a running job again and a restarted server still shows the last results. A job that was running when the
  // server stopped is marked interrupted (its half-written file is deleted at the next start).
  const REG = path.join(dataDir, 'convert-jobs.json');
  const jobs = new Map(); let nextJob = 1, running = null;
  let past = []; try { past = JSON.parse(fs.readFileSync(REG, 'utf8')); if (!Array.isArray(past)) past = []; } catch (_) { past = []; }
  let stale = false;
  for (const r of past) if (r.state === 'running') {
    Object.assign(r, { state: 'interrupted', error: 'interrupted: the server restarted', message: 'interrupted: the server restarted; the half-written file was deleted', updated_at: new Date().toISOString() });
    for (const f of [`${r.output}.partial`, `${r.output}.f16.partial`]) try { fs.unlinkSync(f); } catch (_) { /* none */ }
    stale = true;
  }
  const writeReg = () => { try { fs.mkdirSync(dataDir, { recursive: true }); fs.writeFileSync(REG, JSON.stringify(past.slice(0, 20), null, 1)); } catch (e) { log(`could not save the conversion record: ${e.message}`); } };
  if (stale) writeReg();
  const STAGE = { convert: 'converting', quantize: 'llama-quantize: 16-bit to 4-bit' };
  // One job as a record: what the page and the registry file show.
  function record(j) {
    const state = !j.finished ? 'running' : j.cancelled ? 'cancelled' : j.error ? 'failed' : 'done';
    const left = j.done > 0 && j.of > j.done ? Math.round(((Date.now() - (j.stageStarted || j.started)) / j.done) * (j.of - j.done) / 1000) : null;
    const msg = state === 'running' ? `${j.stages > 1 ? `step ${j.step || 1} of 2, ` : ''}${j.of ? `${STAGE[j.stage] || j.stage} ${j.name} to ${j.label}: ${j.done} of ${j.of} tensors${left != null ? `, about ${left < 60 ? `${left} s` : `${Math.round(left / 60)} min`} left` : ''}` : `reading ${j.name}`}`
      : state === 'done' ? `converted: ${j.result.file}, ${j.result.gb} GB` : state === 'cancelled' ? 'stopped: the half-written file was deleted; the folder is unchanged' : `failed: ${j.error}`;
    return { id: j.id, name: j.name, source: j.folder, output: j.out, precision: j.precision, label: j.label, arch: j.arch, state, stage: j.stage, step: j.step || 1, stages: j.stages,
      done: j.done, of: j.of, seconds_left: left, started_at: new Date(j.started).toISOString(), updated_at: new Date(j.updated || j.started).toISOString(),
      seconds: Math.round(((j.finished ? j.updated : Date.now()) - j.started) / 1000), message: msg, error: state === 'failed' ? j.error : null, result: j.result, log: j.lines.slice(-8) };
  }
  let savedAt = 0;
  function persist(j, force) {
    j.updated = Date.now();
    if (!force && Date.now() - savedAt < 5000) return;
    savedAt = Date.now(); const r = record(j); delete r.log;
    past = [r, ...past.filter((x) => x.id !== j.id)].slice(0, 20); writeReg();
  }

  // llama.cpp's converter and llama-quantize: beside the engine's folder (settings.llamaDir) or in $PSI_LLAMA_SRC (a llama.cpp
  // checkout); llama-quantize also on the PATH when no engine folder is set.
  const exe = process.platform === 'win32' ? '.exe' : '';
  const isFile = (f) => { try { return fs.statSync(f).isFile(); } catch (_) { return false; } };
  function tools() {
    const ll = config.settings.llamaDir || '', roots = [ll, ll && path.dirname(ll), process.env.PSI_LLAMA_SRC].filter(Boolean);
    // A converter needs its `conversion` package beside it (llama.cpp's scripts/ copy can lack it).
    const conv = roots.flatMap((r) => [path.join(r, 'src', 'convert_hf_to_gguf.py'), path.join(r, 'convert_hf_to_gguf.py'), path.join(r, 'scripts', 'convert_hf_to_gguf.py')])
      .find((f) => isFile(f) && isFile(path.join(path.dirname(f), 'conversion', '__init__.py'))) || null;
    const quant = findLlama('llama-quantize', ll);
    return { converter: conv && conv.replace(/\\/g, '/'), quantize: quant && quant.replace(/\\/g, '/') };
  }
  // Free space on the drive of `dir` (or of its nearest existing parent: the folder is made on the first conversion).
  function freeBytes(dir) {
    let d = path.resolve(dir);
    while (!fs.existsSync(d) && path.dirname(d) !== d) d = path.dirname(d);
    try { const s = fs.statfsSync(d); return s.bavail * s.bsize; } catch (_) { return null; }
  }

  function plan(a = {}) {
    if (typeof a.folder !== 'string' || !a.folder) throw new Error('convert: no model folder given');
    const m = config.listedHf(a.folder);
    if (!m) throw new Error('convert: only a Hugging Face folder from the model list (Settings > Models) can be converted. Scan the folder first.');
    const precision = a.precision || DEFAULT, P = PRECISIONS[precision];
    if (!P) throw new Error(`convert: precision ${a.precision} is not one of ${Object.keys(PRECISIONS).join(', ')}`);
    const dir = config.ggufDir(); if (!dir) throw new Error('convert: no folder for converted files (set the models folder in Settings)');
    config.localPath(dir, 'converted-files folder');
    const out = path.join(dir, `${m.file_name}-${precision}.gguf`).replace(/\\/g, '/');
    const params = m.params_estimate || 0;
    const est = Math.round(params * P.bpp + 8 * 2 ** 20);   // + the vocabulary and settings in the header
    const inter = P.quantize ? Math.round(params * 2 + 8 * 2 ** 20) : 0;
    const need = Math.round(est * 1.1 + inter), free = freeBytes(dir);
    const t = tools(), problems = [];
    if (!t.converter) problems.push("llama.cpp's converter (convert_hf_to_gguf.py with its conversion folder) was not found beside the engine's folder (Settings) or in PSI_LLAMA_SRC (a llama.cpp source checkout)");
    if (P.quantize && !t.quantize) problems.push(`llama-quantize${exe} was not found in the engine's folder (needed for 4-bit)`);
    if (free != null && need > free) problems.push(`not enough disk space: it needs about ${(need / GB).toFixed(1)} GB${inter ? ` (the ${(inter / GB).toFixed(1)} GB 16-bit step file included, deleted after)` : ''} and ${(free / GB).toFixed(1)} GB is free on the drive of ${dir}`);
    return { folder: m.path, name: m.name, arch: m.arch, params_estimate: params, params_basis: m.params_basis, source_gb: m.gb, precision, label: P.label,
      out, out_exists: fs.existsSync(out), out_dir: dir, est_bytes: est, intermediate_bytes: inter, need_bytes: need, free_bytes: free,
      enough: free == null ? null : need <= free, converter: t.converter, quantize: P.quantize ? t.quantize : null, problems, ok: !problems.length,
      runs_on: 'CPU (CUDA_VISIBLE_DEVICES=-1)', has_gguf: m.has_gguf, converted: m.converted };
  }

  const unlinkSoon = (f) => { let n = 0; const go = () => { try { fs.unlinkSync(f); } catch (e) { if (e.code !== 'ENOENT' && ++n < 10) setTimeout(go, 300); } }; go(); };

  // One child process; JSON lines on stdout go to onLine, other lines (llama-quantize) to onText. Resolves with the last JSON.
  function run(cmd, args, job, onLine, onText = () => {}) {
    return new Promise((resolve, reject) => {
      const p = spawn(cmd, args, { cwd: SRC, windowsHide: true, env: config.cacheEnv({ CUDA_VISIBLE_DEVICES: '-1' }) });
      job.proc = p;
      let buf = '', last = null, err = '';
      const line = (l) => { l = l.trim(); if (!l) return; if (l[0] === '{') { try { last = JSON.parse(l); onLine(last); return; } catch (_) { /* not JSON */ } } onText(l); };
      p.stdout.on('data', (d) => { buf += d; let i; while ((i = buf.search(/[\r\n]/)) >= 0) { line(buf.slice(0, i)); buf = buf.slice(i + 1); } });
      p.stderr.on('data', (d) => { const s = String(d); err = (err + s).slice(-4000); s.split(/[\r\n]+/).forEach((l) => l.trim() && !l.includes('%|') && onText(l.trim())); });
      p.on('error', (e) => reject(new Error(`${path.basename(cmd)} could not start: ${e.message}`)));
      p.on('exit', (code) => {
        job.proc = null; line(buf);
        if (job.cancelled) return reject(new Error('cancelled'));
        if (last && last.error) return reject(new Error(last.error));
        if (code === 0) return resolve(last);
        const msg = err.split(/[\r\n]+/).filter((l) => l.trim() && !/%\||Warning|warn\(/.test(l)).slice(-3).join(' ');
        reject(new Error(msg || `${path.basename(cmd)} stopped (code ${code})`));
      });
    });
  }

  function start(a = {}) {
    if (running && !running.finished) throw new Error(`convert: ${running.name} is being converted now; wait for it or stop it first`);
    const pl = plan(a);
    if (!pl.ok) throw new Error(`convert: not started: ${pl.problems.join('; ')}`);
    if (pl.out_exists && a.overwrite !== true) throw new Error(`convert: ${pl.out} already exists; nothing was written. Load it, or confirm that it should be replaced.`);
    fs.mkdirSync(pl.out_dir, { recursive: true });
    const P = PRECISIONS[pl.precision], id = `${Date.now().toString(36)}-${nextJob++}`;
    const partial = `${pl.out}.partial`, inter = P.quantize ? `${pl.out}.f16.partial` : null;
    const job = { id, ...pl, started: Date.now(), stage: 'convert', stages: P.quantize ? 2 : 1, done: 0, of: 0, lines: [], finished: false, error: null, cancelled: false, result: null };
    jobs.set(id, job); running = job; persist(job, true);
    const note = (l) => { job.lines.push(l); if (job.lines.length > 40) job.lines.shift(); };
    log(`convert ${pl.name} → ${pl.out} (${P.label})`);
    (async () => {
      const conv = await run(PY(), [path.join(SRC, 'to_gguf.py'), '--converter', pl.converter, '--outfile', inter || partial, '--outtype', P.outtype, '--', pl.folder], job,
        (l) => { if (l.stage === 'write') { if (!job.of) job.stageStarted = Date.now(); job.done = l.done; job.of = l.of; } if (l.stage === 'prepare') job.arch = l.arch || job.arch; persist(job); }, note);
      let q = null;
      if (P.quantize) {
        job.stage = 'quantize'; job.step = 2; job.stageStarted = Date.now(); job.done = 0; job.of = conv.tensors || 0; persist(job, true);
        await run(pl.quantize, [inter, partial, P.quantize], job, () => {}, (l) => {
          const m = /^\[\s*(\d+)\s*\/\s*(\d+)\s*\]/.exec(l); if (m) { job.done = +m[1]; job.of = +m[2]; persist(job); } else note(l);
        });
        unlinkSoon(inter); q = P.quantize;
      }
      if (pl.out_exists) fs.unlinkSync(pl.out);   // only here when the user confirmed the replacement
      fs.renameSync(partial, pl.out);
      return { conv, q };
    })().then(({ conv, q }) => {
      const bytes = fs.statSync(pl.out).size, seconds = Math.round((Date.now() - job.started) / 100) / 10;
      job.result = { file: pl.out, bytes, gb: +(bytes / GB).toFixed(3), tensors: conv.tensors, arch: conv.arch || pl.arch, precision: pl.precision, label: P.label,
        outtype: P.outtype, quantized_to: q, source: pl.folder, name: pl.name, seconds, restored_keys: conv.restored_keys || [], replaced: !!pl.out_exists, runs_on: 'CPU' };
      job.result.converter = pl.converter; if (q) job.result.quantize = pl.quantize;
      job.stage = 'done';
    }).catch((e) => {
      job.error = job.cancelled ? 'cancelled' : e.message; job.stage = job.cancelled ? 'cancelled' : 'failed';
      unlinkSoon(partial); if (inter) unlinkSoon(inter);
    }).finally(() => { job.finished = true; job.proc = null; persist(job, true); });
    return { job: id, out: pl.out, precision: pl.precision, stages: job.stages };
  }

  // Stop: the process is ended; the half-written file (and a 4-bit run's 16-bit step file) is deleted, the source is untouched.
  function cancel(id) {
    const j = jobs.get(String(id)); if (!j) throw new Error(`no conversion job ${id}`);
    if (!j.finished) { j.cancelled = true; try { if (j.proc) j.proc.kill(); } catch (_) { /* already gone */ } }
    return { id: j.id, cancelled: j.cancelled };
  }
  // One job (live, or from the registry file) with the fields the progress bar needs; finished is true for any ended state.
  function job(id) {
    const j = jobs.get(String(id));
    if (j) return { ...record(j), finished: j.finished, cancelled: j.cancelled, folder: j.folder, out: j.out, seconds: Math.round((Date.now() - j.started) / 1000) };
    const r = past.find((x) => x.id === String(id)); if (!r) throw new Error(`no conversion job ${id}`);
    return { ...r, finished: true, cancelled: r.state === 'cancelled', folder: r.source, out: r.output, log: [] };
  }
  // convertStatus: one job by id, or every job, most recent first (this session's live ones and the last 20 on disk).
  function status(a = {}) {
    if (a && a.id != null) return job(a.id);
    const live = [...jobs.values()].map((j) => job(j.id)), ids = new Set(live.map((x) => x.id));
    return [...live, ...past.filter((r) => !ids.has(r.id)).map((r) => job(r.id))].sort((x, y) => String(y.started_at).localeCompare(String(x.started_at)));
  }
  async function call(a = {}) {
    switch (a.action) {
      case 'precisions': return { precisions: PRECISIONS, default: DEFAULT, out_dir: config.ggufDir() };
      case 'plan': return plan(a);
      case 'start': return start(a);
      case 'job': return job(a.job);
      case 'cancel': return cancel(a.job);
      case 'running': return running && !running.finished ? job(running.id) : null;
      case 'history': case 'status': return status(a);
      default: throw new Error(`unknown convert action ${a.action}`);
    }
  }
  return { call, plan, start, job, cancel, status, PRECISIONS };
}

module.exports = { createConverter, PRECISIONS };
