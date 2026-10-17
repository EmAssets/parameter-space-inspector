// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
/*
 * Storage ladder runner (main process): research/model_ladder.py on a model folder, then its untrained control
 * (same architecture, random weights), then research/make_ladder_data.py so the view shows the new rows.
 * One run at a time. A new model goes into results/model_ladder; a model already there is replicated with
 * --seed N into results/model_ladder/replications/seed<N> (every resample and the random init re-drawn),
 * so a stored run is never overwritten. Folders: picked in a dialog, or the folder a stored run used.
 */
'use strict';
const { python: PY } = require('./python.js');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.join(__dirname, '..');
const RESEARCH = path.join(ROOT, 'research');
const OUT = path.join(RESEARCH, 'results', 'model_ladder');
const ENV = () => require('./config.js').cacheEnv();   // HF_HOME / TMP from settings.cacheDir or PSI_CACHE_DIR, when set
const TAG = /^[A-Za-z0-9._-]{1,60}(@[A-Za-z0-9._-]{1,20})?$/;   // name, or name@checkpoint
const twinOf = (tag) => `${tag.split('@')[0]}@init`;              // the untrained twin of a run and of all its checkpoints
const fileOf = (tag, seed) => path.join(seed ? path.join(OUT, 'replications', `seed${seed}`) : OUT, `${tag.replace(/@/g, '_')}.json`);

function createLadder({ config, picked, send, log }) {
  const S = { running: false, proc: null, queue: [], step: null, lines: [], started: 0, cancelled: false, last: null };
  const emit = (m) => { send && send(m); };
  const line = (t) => { S.lines.push(t); if (S.lines.length > 400) S.lines.shift(); emit({ type: 'line', text: t }); };

  function py(args, { collect = false } = {}) {
    return new Promise((resolve, reject) => {
      const p = spawn(PY(), ['-u', ...args], { cwd: ROOT, env: ENV(), windowsHide: true }); S.proc = p;
      let out = '', err = '', buf = '';
      p.stdout.on('data', (d) => {
        if (collect) { out += d; return; }
        buf += d; const parts = buf.split(/\r?\n/); buf = parts.pop(); parts.forEach((x) => x.trim() && line(x));
      });
      p.stderr.on('data', (d) => {
        err = (err + d).slice(-4000);
        if (!collect) String(d).split(/\r|\n/).map((x) => x.trim()).filter((x) => x && !/^Loading weights|it\/s\]$/.test(x) && !/Warning|warn\(/.test(x)).slice(-2).forEach(line);
      });
      p.on('error', reject);
      p.on('close', (code) => { S.proc = null; if (buf.trim() && !collect) line(buf.trim()); code === 0 ? resolve(out) : reject(new Error(S.cancelled ? 'cancelled' : (err.trim().split('\n').pop() || `exit ${code}`))); });
    });
  }

  // Everything the view draws, rebuilt from the result files (the same rules as the written report). When the results on
  // disk are newer than src/ui/ladder-data.js, the same run also rewrites that file, so the bundled view catches up;
  // the returned object says so (written: true) and the view tells the reader. With no result files on disk (a fresh
  // install: the stored runs ship only as src/ui/ladder-data.js) the bundled view is the data: nothing is rebuilt, and the
  // answer has no rows, so the page keeps what it loaded.
  async function data() {
    const st = stale();
    if (st.bundled_only) return { bundled: true, n_results: st.n_results_in_view };
    return JSON.parse(await py([path.join(RESEARCH, 'make_ladder_data.py'), '--json', ...(st.stale ? ['--write'] : [])], { collect: true }));
  }

  // Is the view older than the results? Compares every result file's mtime with the data file the view is built from.
  const DATA_JS = path.join(ROOT, 'src', 'ui', 'ladder-data.js');
  function resultFiles() {
    const keep = (f) => f.endsWith('.json') && !path.basename(f).startsWith('summary');
    const ls = (d) => { try { return fs.readdirSync(d).filter(keep).map((f) => path.join(d, f)); } catch (_) { return []; } };
    let fs_ = ls(OUT).concat(ls(path.join(OUT, 'interventions')), ls(path.join(RESEARCH, 'results', 'model_deep')));
    try { fs.readdirSync(path.join(OUT, 'replications')).filter((d) => d.startsWith('seed')).forEach((d) => { fs_ = fs_.concat(ls(path.join(OUT, 'replications', d))); }); } catch (_) { /* none */ }
    return fs_;
  }
  let viewCache = { mtime: -1, generated: null, n: null };
  function viewInfo() {
    let st; try { st = fs.statSync(DATA_JS); } catch (_) { return { mtime: 0, generated: null, n: null }; }
    if (st.mtimeMs !== viewCache.mtime) {
      const head = fs.readFileSync(DATA_JS, 'utf8');
      const g = /"generated":"([^"]+)"/.exec(head), n = /"n_results":(\d+)/.exec(head);
      viewCache = { mtime: st.mtimeMs, generated: g ? g[1] : null, n: n ? +n[1] : null };   // null: an older data file that did not count its sources
    }
    return { mtime: viewCache.mtime, generated: viewCache.generated, n: viewCache.n };
  }
  // No result files at all (a fresh install) is not stale: the bundled view is all there is. Once a run writes results,
  // the view is compared with them as usual (and rebuilt from them).
  function stale() {
    const files = resultFiles(), v = viewInfo();
    let newest = 0, newer = 0;
    files.forEach((f) => { try { const m = fs.statSync(f).mtimeMs; if (m > newest) newest = m; if (m > v.mtime + 1000) newer++; } catch (_) { /* gone */ } });
    const bundledOnly = files.length === 0;
    return { data_generated: v.generated, view_mtime: Math.round(v.mtime / 1000), newest_result_mtime: Math.round(newest / 1000), stale: !bundledOnly && (newest > v.mtime + 1000 || (v.n !== null && files.length !== v.n)), bundled_only: bundledOnly, n_results_on_disk: files.length, n_results_in_view: v.n, n_newer: newer };
  }

  function storedPaths() {
    const s = new Set();
    try { fs.readdirSync(OUT).filter((f) => f.endsWith('.json') && !f.startsWith('summary')).forEach((f) => { try { const p = JSON.parse(fs.readFileSync(path.join(OUT, f), 'utf8')).path; if (p) s.add(config.key4(config.real(p))); } catch (_) { /* skip */ } }); } catch (_) { /* no results yet */ }
    return s;
  }

  // An 8-bit load when the bf16 weights would not fit beside the app on a 16 GB card (~5B parameters and up).
  function sizeOf(dir) {
    try {
      const c = JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf8')); const h = c.hidden_size || c.d_model || 1024, L = c.num_hidden_layers || c.n_layer || 24;
      const ff = c.intermediate_size || 4 * h, V = c.vocab_size || 50000, X = c.num_local_experts || c.num_experts || 1; const n = L * (4 * h * h + 3 * h * ff * X) + 2 * V * h;   // experts: every expert's weights count
      return { params: n, q8: n > 5e9 || c.model_type === 'gpt_oss', arch: (c.architectures || [c.model_type || '?'])[0], layers: L };
    } catch (_) { return null; }
  }

  function checkFolder(dir) {
    config.localPath(dir, 'model folder');
    if (!path.isAbsolute(dir)) throw new Error('The model folder must be a full path.');
    const f = config.real(dir);
    let st; try { st = fs.statSync(f); } catch (_) { st = null; }
    if (!st || !st.isDirectory()) throw new Error(`Not a folder on this computer: ${dir}`);
    if (!fs.existsSync(path.join(f, 'config.json'))) throw new Error('That folder has no config.json: the ladder needs a Hugging Face model folder (safetensors or .bin), not a GGUF file.');
    if (config.key4(f).startsWith(config.key4(config.real(config.DATA)))) throw new Error('That folder is the app’s private data folder.');
    if (!picked.has(config.key4(f)) && !storedPaths().has(config.key4(f))) throw new Error('Choose the folder with “Choose folder” first.');
    return f;
  }

  async function runQueue() {
    S.running = true; S.cancelled = false; S.started = Date.now(); S.lines = [];
    let ok = true, error = null;
    try {
      while (S.queue.length && !S.cancelled) {
        const job = S.queue.shift(); S.step = job.label; emit({ type: 'step', label: job.label, left: S.queue.length });
        line(`── ${job.label}`);
        if (job.kind === 'data') {
          const js = path.join(ROOT, 'src', 'ui', 'ladder-data.js');
          await py([path.join(RESEARCH, 'make_ladder_data.py')]); line(`results rebuilt (${path.relative(ROOT, js)})`);
        } else await py(job.args);
      }
    } catch (e) { ok = false; error = String(e.message || e); line(`stopped: ${error}`); S.queue = []; }
    S.running = false; S.step = null; S.last = { ok: ok && !S.cancelled, error, seconds: Math.round((Date.now() - S.started) / 1000) };
    emit({ type: 'done', ...S.last });
  }

  async function run(a = {}) {
    if (S.running) throw new Error('A ladder run is already going. Cancel it first.');
    const dir = checkFolder(a.modelDir);
    const tag = String(a.tag || '').trim(); if (!TAG.test(tag)) throw new Error('Name: letters, digits, dot, dash or underscore, up to 60 characters (optionally name@checkpoint).');
    const size = sizeOf(dir) || {}; const q8 = a.q8 === undefined ? !!size.q8 : !!a.q8;
    // a name already in the results is replicated with a new seed, never overwritten
    let seed = Math.max(0, Math.floor(+a.seed || 0));
    if (fs.existsSync(fileOf(tag, seed))) { if (a.seed !== undefined && +a.seed > 0) throw new Error(`Seed ${seed} of ${tag} exists already; pick another seed.`); seed = 1; while (fs.existsSync(fileOf(tag, seed))) seed++; }
    const base = [path.join(RESEARCH, 'model_ladder.py'), '--model', dir, '--tag', tag, ...(q8 ? ['--q8'] : []), ...(seed ? ['--seed', String(seed)] : [])];
    S.queue = [{ label: `${tag}${seed ? ` · replication, seed ${seed}` : ''}: M0–M8`, args: base }];
    const ctl = twinOf(tag);
    if (a.control !== false && !fs.existsSync(fileOf(ctl, seed))) S.queue.push({ label: `${ctl}${seed ? ` · seed ${seed}` : ''}: the same architecture untrained`, args: [...base.slice(0, 4), ctl, '--init', ...base.slice(5)] });
    S.queue.push({ kind: 'data', label: 'verdicts against the untrained control, rebuilt' });
    runQueue();
    return { tag, seed, q8, size, steps: S.queue.length + 1 };
  }

  function cancel() { S.cancelled = true; S.queue = []; if (S.proc) { try { S.proc.kill(); } catch (_) { /* gone */ } } return { cancelled: true }; }
  function status() { return { running: S.running, step: S.step, lines: S.lines.slice(-80), seconds: S.running ? Math.round((Date.now() - S.started) / 1000) : 0, last: S.last, ...stale() }; }

  return { data, run, cancel, status, stale, sizeOf, checkFolder, command: (a) => commandLine(a) };
}

// The same run as a command, for replicating outside the app.
function commandLine({ modelDir, tag, q8, seed }) {
  const q = (s) => (/\s/.test(s) ? `"${s}"` : s);
  const run = (t, extra) => `python research/model_ladder.py --model ${q(modelDir)} --tag ${t}${extra}${q8 ? ' --q8' : ''}${seed ? ` --seed ${seed}` : ''}`;
  return [run(tag, ''), run(twinOf(tag), ' --init'), 'python research/make_ladder_data.py'].join('\n');
}

module.exports = { createLadder, commandLine };
