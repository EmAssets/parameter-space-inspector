// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Measuring an Override-map probe on the inside model (tools/measure_override.py), for a model no probe was measured
// on. It asks a few thousand questions and holds the GPU for an hour or more, so it runs as its own process, one at a
// time, after both engines are stopped, and starts only from the Override map's own button (confirm: true). It is a
// page-only call (main.js, devserver.js): MCP agents and job plans cannot reach it. The result lands in
// <data>/override-probes, where the inside engine finds it the next time it loads that model.
'use strict';
const { python: PY } = require('./python.js');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const TOOL = path.join(__dirname, '..', 'tools', 'measure_override.py');
const MAX_FILE = 5 * 1048576;

function createMeasure({ tracer, engine, config, log }) {
  const OUT = () => path.join(config.DATA, 'override-probes');
  let job = null;   // {proc, started, model, lines, progress, done, error, finished, cancelled}

  // A question file from the page, kept in the data folder under its hash (so a re-run with the same file resumes).
  function questionsFile(a) {
    if (!a.questions_text) return null;
    const text = String(a.questions_text); if (text.length > MAX_FILE) throw new Error('the question file is larger than 5 MB');
    const ext = /\.csv$/i.test(String(a.questions_name || '')) ? '.csv' : '.jsonl';
    const dir = path.join(config.DATA, 'override-questions'); fs.mkdirSync(dir, { recursive: true });
    const f = path.join(dir, crypto.createHash('sha256').update(text).digest('hex').slice(0, 16) + ext);
    if (!fs.existsSync(f)) fs.writeFileSync(f, text, 'utf8');
    return f;
  }

  function run(args, j) {
    const env = config.cacheEnv({ PYTHONIOENCODING: 'utf-8', PSI_OVERRIDE_DIR: OUT() });
    j.proc = spawn(PY(), args, { env, windowsHide: true });
    const reader = () => { let buf = ''; return (d) => {
      buf += d; let i;
      while ((i = buf.search(/[\r\n]/)) >= 0) {
        const l = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!l) continue;
        const m = /^PSI-(PROGRESS|DONE|ERROR) ?(.*)$/.exec(l);
        if (m && m[1] === 'PROGRESS') { try { j.progress = JSON.parse(m[2]); } catch (_) { /* a line only */ } }
        else if (m && m[1] === 'DONE') { try { j.done = JSON.parse(m[2]); } catch (_) { j.done = {}; } }
        else if (m && m[1] === 'ERROR') j.error = m[2];
        else if (!/^Loading weights:|it\/s\]$/.test(l)) { j.lines.push(l.slice(0, 300)); if (j.lines.length > 40) j.lines.shift(); }
      }
    }; };
    j.proc.stdout.on('data', reader()); j.proc.stderr.on('data', reader());
    return new Promise((resolve) => j.proc.on('close', (code) => {
      j.finished = true; j.proc = null;
      if (code && !j.error) j.error = j.cancelled ? 'cancelled (it continues where it stopped if you start it again with the same questions)' : `stopped (exit ${code}): ${j.lines.slice(-2).join(' ')}`;
      resolve(code);
    }));
  }

  // Check a question set without asking anything: how many questions per part, or what is wrong with the file.
  async function check(a) {
    const t = config.settings.trace || {};
    const f = questionsFile(a);
    const j = { lines: [], finished: false };
    await run(['-u', TOOL, '--check', ...(f ? ['--questions', f] : [])], j);
    if (j.error) return { ok: false, error: j.error };
    return { ok: true, ...(j.done || {}), model: t.modelDir || null };
  }

  function start(a) {
    if (a.confirm !== true) throw new Error('Measuring starts only from the Override map’s own button.');
    if (job && !job.finished) throw new Error('A measurement is already running.');
    // The model the Override map reads: the inside engine's (Settings' weights folder until a view switched it).
    const dir = tracer.modelDir || (config.settings.trace || {}).modelDir; if (!dir) throw new Error('No inside model is set (Settings > Models).');
    const t = { modelDir: dir };
    const f = questionsFile(a);
    const args = ['-u', TOOL, '--model', t.modelDir, '--out', OUT(), ...(f ? ['--questions', f] : [])];
    if (Number(a.limit) > 0) args.push('--limit', String(Math.round(Number(a.limit))));
    job = { started: new Date().toISOString(), model: t.modelDir, questions: f ? path.basename(String(a.questions_name || f)) : 'the default set', lines: [], progress: null, done: null, error: null, finished: false };
    const j = job;
    (async () => {
      try { await tracer.stop(); } catch (_) { /* not running */ }
      if (engine) { try { await engine.stop(); } catch (_) { /* not running */ } }
      j.lines.push('Both engines were stopped for the measurement (it needs the GPU to itself); they start again when a view needs them.');
      await run(args, j);
      log && log(j.error ? `override measurement failed: ${j.error}` : `override probe measured for ${t.modelDir}: ${j.done && j.done.trust}`);
    })().catch((e) => { j.finished = true; j.error = String(e.message || e); });
    return status();
  }

  function status() { if (!job) return { running: false }; const { proc, ...rest } = job; return { running: !job.finished, ...rest }; }

  return {
    busy: () => !!(job && !job.finished),
    async call(a = {}) {
      if (a.action === 'start') return start(a);
      if (a.action === 'status') return status();
      if (a.action === 'check') return check(a);
      if (a.action === 'cancel') { if (job && job.proc) { job.cancelled = true; job.proc.kill(); } return status(); }
      throw new Error('measure: action is check, start, status or cancel');
    },
  };
}

module.exports = { createMeasure };
