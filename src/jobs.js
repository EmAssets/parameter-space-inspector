// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The job runner: an approved experiment plan runs here, in the app's own process (the desktop app's main process or
// the dev server), not in the screen that started it. Close the window or lock the phone and it carries on; open the
// Experiment designer anywhere and the history shows it running, then the report.
//
//   - One job at a time (one GPU), in the order they were sent. Each job is a file in <data folder>/jobs/, rewritten
//     after every prompt, so a job cut short (the app closed, the machine slept) can resume where it stopped.
//   - GPU courtesy: before an engine is loaded, the job waits while a training or research job holds the GPU, or while
//     the card lacks room for the model (another program, e.g. a video editor or Ollama, is using it). “Run anyway”
//     lifts the wait for that job. A model that could never fit is not waited for.
//   - Result cache: recordings are greedy and were measured identical run after run, so the same prompt, instrument,
//     settings and model returns the stored result instead of running again. Never for a Tipping-point plan (it cuts
//     the model) or while a change (steering, a cut) is in force on the engine the plan uses.
//   - The plan is checked here with the same check the page uses (design-core.js validate): a job sent through the API
//     gets no more than the page would.
//   - Measuring and drawing happen in the page that opens the record (the same code as every report); a job saves the
//     raw results, its activity, and the record's live status (“running · 3 of 12”) as it goes.
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const C = require('./ui/design-core.js');

const ID = /^j[0-9]{13}-[0-9a-f]{6}$/, KEEP = 200, CACHE_MAX = 4000, HEARTBEAT_MS = 30000, COURTESY_POLL_MS = 20000;
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; } };
const sha = (x) => crypto.createHash('sha256').update(typeof x === 'string' ? x : JSON.stringify(x)).digest('hex');

// opts: { dataDir, call(name, ...args) → the app's API (engine calls), mcp(rpc) → the app's MCP, experiments (the
// history store), gpu() → gpuState, config, notify(job) when a job ends, log(text) }
function createJobs(opts) {
  const { dataDir, call, experiments, config } = opts;
  const DIR = path.join(dataDir, 'jobs'), CACHE = path.join(DIR, 'cache');
  const mk = (d) => fs.mkdirSync(d, { recursive: true, mode: 0o700 });
  const fileOf = (id) => { if (!ID.test(String(id))) throw new Error('jobs: not a job id'); return path.join(DIR, `${id}.json`); };
  // Written beside the file, then renamed over it. On Windows the rename fails (EPERM/EBUSY) while a reader has the file
  // open (the page polls the jobs every few seconds), so it is tried again for a moment, then written in place.
  const pause = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
  const write = (f, o) => {
    mk(path.dirname(f)); const text = JSON.stringify(o), tmp = `${f}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, text, { mode: 0o600 });
    for (let i = 0; i < 12; i++) {
      try { fs.renameSync(tmp, f); return; } catch (e) { if (!['EPERM', 'EBUSY', 'EACCES'].includes(e.code)) throw e; pause(15 + i * 10); }
    }
    fs.writeFileSync(f, text, { mode: 0o600 }); try { fs.unlinkSync(tmp); } catch (_) { /* gone */ }
  };
  const read = (id) => JSON.parse(fs.readFileSync(fileOf(id), 'utf8'));
  const log = (m) => opts.log && opts.log(m);
  const OWNER = { pid: process.pid, label: process.versions.electron ? 'desktop app' : 'dev server' };
  const STOP = new Set(), ANYWAY = new Set();
  let current = null, wake = null;

  // ── the files ──
  const all = () => { mk(DIR); return fs.readdirSync(DIR).filter((f) => /^j[0-9].*\.json$/.test(f)).map((f) => { try { return JSON.parse(fs.readFileSync(path.join(DIR, f), 'utf8')); } catch (_) { return null; } }).filter(Boolean).sort((a, b) => (a.at < b.at ? 1 : -1)); };
  const brief = (j) => ({ id: j.id, name: j.name, record: j.record, instrument: j.plan.instrument, status: j.status, done: j.runs.length, n: j.plan.items.length, at: j.at, started: j.started || null, finished: j.finished || null,
    wait: j.wait || null, error: j.error || null, cached: j.cached || 0, owner: j.owner, mine: j.owner && j.owner.pid === process.pid, tip: j.tip || null });
  function save(j) { j.updated = new Date().toISOString(); write(fileOf(j.id), j); }
  // Jobs this process owned when it last ran, or whose process is gone: interrupted, resumable.
  function recover() {
    for (const j of all()) {
      if (!['queued', 'waiting', 'running'].includes(j.status)) continue;
      if (j.owner && j.owner.pid !== process.pid && alive(j.owner.pid)) continue;   // another live process (the app or the dev server) runs it
      j.status = 'interrupted'; j.error = `The ${j.owner ? j.owner.label : 'app'} closed before it finished (${j.runs.length} of ${j.plan.items.length} done). Resume carries on from there.`;
      save(j); liveRecord(j, `interrupted · ${j.runs.length} of ${j.plan.items.length}`).catch(() => {});
    }
    const old = all().filter((j) => !['queued', 'waiting', 'running'].includes(j.status)).slice(KEEP);
    for (const j of old) { try { fs.unlinkSync(fileOf(j.id)); } catch (_) { /* gone */ } }
  }

  // ── the activity, in the shape the page shows (design.js actData) ──
  function step(j, kind, text, detail) {
    const a = { at: new Date().toISOString(), kind, text: String(text).slice(0, 300), detail: detail ? String(detail).slice(0, 1500) : '', state: 'run', ms: null, result: '' };
    j.activity.push(a); if (j.activity.length > 800) j.activity.shift();
    const t0 = Date.now();
    return { end: (ok, result) => { if (a.state === 'run') { a.state = ok ? 'ok' : 'bad'; a.ms = Date.now() - t0; a.result = String(result == null ? '' : result).slice(0, 300); } return a; } };
  }
  const short = (v, n = 60) => { const t = typeof v === 'string' ? v : JSON.stringify(v); return t == null ? '' : t.length > n ? `${t.slice(0, n - 1)}…` : t; };
  function ctxOf(j) {
    return {
      eng: async (name, ...args) => {
        const a = step(j, 'call', `${name}${args[0] && typeof args[0] === 'string' ? ` · “${short(args[0], 50)}”` : args[0] && args[0].action ? ` · ${args[0].action}` : ''}`, `psi.${name}(${args.map((x) => short(x, 500)).join(', ')})`);
        try { const r = await call(name, ...args); a.end(true, 'done'); return r; } catch (e) { a.end(false, e.message); throw e; }
      },
      mcp: async (name, args) => {
        const a = step(j, 'tool', `tool call · ${name}${args && (args.text || args.prompt) ? ` · “${short(args.text || args.prompt, 50)}”` : ''}`, `${name}(${short(args, 900)})`);
        try { const r = C.toolReply(await opts.mcp({ jsonrpc: '2.0', id: Date.now(), method: 'tools/call', params: { name, arguments: args } })); a.end(true, 'answered'); return r; } catch (e) { a.end(false, e.message); throw e; }
      },
      explain: (e) => String((e && e.message) || e),
      insideModel: () => path.basename(String((config.settings.trace || {}).modelDir || '')) || null,
      localModel: () => path.basename(String(config.settings.lastModel || '')) || null,
    };
  }

  // ── the history record the page reads (experiments.js) ──
  async function liveRecord(j, status, extra = {}) {
    const record = { name: j.name, at: j.at, updated: new Date().toISOString(), status, live: /^running|^waiting|^queued/.test(status), job: j.id, plan: j.plan, edited: j.edited || [],
      partial: j.runs.length < j.plan.items.length, tip: j.tip || null, runs: C.slim(j.runs), activity: j.activity, reading: null, summary: null, ...extra };
    const r = experiments.call({ action: 'save', id: j.record, record }); if (r.id !== j.record) { j.record = r.id; save(j); }
  }

  // ── the cache ──
  const sizeOf = (f) => { try { return fs.statSync(f).size; } catch (_) { return 0; } };
  const weightsOf = (dir) => { try { return fs.readdirSync(dir).filter((f) => /\.safetensors$/.test(f)).reduce((s, f) => s + sizeOf(path.join(dir, f)), 0); } catch (_) { return 0; } };
  // What the plan's results depend on besides the prompt: the model file and settings of the engine it uses. null: do
  // not cache (a change is in force, or the model could not be named).
  async function modelKey(plan) {
    const need = C.needOf(plan);
    if (plan.instrument === 'tipping') return null;
    if (config.guard && !config.guard().jobCache) return null;   // the cache is off (Settings > Guardrails)
    if (need === 'model') {
      const st = await call('engineState').catch(() => null); if (!st || !st.running || !st.model) return null;
      const o = st.options || {}; if (o.controlVector || o.lora || o.cvec || o.loras) return null;
      let m; try { const x = fs.statSync(st.model); m = [st.model, x.size, x.mtimeMs]; } catch (_) { return null; }
      return sha(['model', m, o]);
    }
    if (need === 'checkpoints') return sha(['checkpoints', (config.settings.trace || {}).checkpoints]);
    const t = await call('traceStatus').catch(() => null);
    const ch = t && t.conditions && t.conditions.changed; if (ch && Object.values(ch).some((v) => v && (!v.kind || v.kind !== 'none'))) return null;
    const dir = (config.settings.trace || {}).modelDir; if (!dir) return null;
    return sha(['inside', dir, weightsOf(dir), t && t.conditions && t.conditions.weights_hash]);
  }
  const itemKey = (plan, it, mk) => sha([mk, plan.instrument, plan.answer_tokens, plan.framing, it.text, ...Object.keys(C.EXTRA[plan.instrument] || {}).map((k) => it[k])]);
  const cacheGet = (k) => { try { const f = path.join(CACHE, `${k}.json`); const v = JSON.parse(fs.readFileSync(f, 'utf8')); const now = new Date(); fs.utimesSync(f, now, now); return v; } catch (_) { return null; } };
  function cachePut(k, t) {
    try {
      write(path.join(CACHE, `${k}.json`), t);
      const max = config.guard ? config.guard().jobCacheMax : CACHE_MAX, fl = fs.readdirSync(CACHE); if (fl.length <= max) return;
      fl.map((f) => ({ f, t: fs.statSync(path.join(CACHE, f)).mtimeMs })).sort((a, b) => a.t - b.t).slice(0, fl.length - max).forEach((x) => fs.unlinkSync(path.join(CACHE, x.f)));
    } catch (e) { log(`cache: ${e.message}`); }
  }

  // ── GPU courtesy ──
  async function loaded(need) {
    if (need === 'model') { const st = await call('engineState').catch(() => null); return !!(st && st.running); }
    const t = await call('traceStatus').catch(() => null); return !!(t && t.ok);
  }
  function needMB(need) {
    if (need === 'model') { const f = config.settings.lastModel; return f ? sizeOf(f) / 1048576 + 1024 : null; }
    const dir = need === 'checkpoints' ? null : (config.settings.trace || {}).modelDir; const b = dir ? weightsOf(dir) : 0;
    return b ? b * 1.15 / 1048576 + 1024 : null;
  }
  // null: go ahead; else why it waits.
  async function courtesy(j) {
    if (ANYWAY.has(j.id) || j.anyway || !opts.gpu || (config.guard && !config.guard().jobCourtesy)) return null;
    const need = C.needOf(j.plan); if (await loaded(need)) return null;
    const g = await opts.gpu().catch(() => null); if (!g || !g.ok) return null;
    if ((g.busy || []).length) return `${g.busy.map((h) => h.label).join(', ')} is using the GPU`;
    const mb = needMB(need); if (mb == null || g.freeMB == null || (g.totalMB != null && mb > g.totalMB)) return null;   // unknown, or would never fit: do not wait
    if (g.freeMB < mb) {
      const big = (g.holders || []).filter((h) => h.role === 'other' || !['app', 'engine', 'inside'].includes(h.role)).sort((a, b) => (b.mb || 0) - (a.mb || 0))[0];
      return `the GPU has ${(g.freeMB / 1024).toFixed(1)} GB free and the model needs about ${(mb / 1024).toFixed(1)} GB${g.otherMB ? ` (other programs hold ${(g.otherMB / 1024).toFixed(1)} GB${big ? `, e.g. ${big.label}` : ''})` : ''}`;
    }
    return null;
  }
  const sleep = (ms) => new Promise((r) => { const t = setTimeout(r, ms); wake = () => { clearTimeout(t); r(); }; });

  // ── running ──
  async function runJob(j) {
    const plan = j.plan, n = plan.items.length, ctx = ctxOf(j);
    current = j; j.status = 'running'; j.started = j.started || new Date().toISOString(); j.error = null; save(j);
    const resumed = j.runs.length > 0;
    step(j, 'run', `${resumed ? 'Resumed' : 'Started'} by the job runner (${OWNER.label}) · ${n} prompts · ${C.INSTRUMENTS[plan.instrument]}${resumed ? ` · from prompt ${j.runs.length + 1}` : ''}`, `plan: ${plan.title}`).end(true);
    await liveRecord(j, `running · ${j.runs.length} of ${n}`);
    const beat = setInterval(() => { save(j); liveRecord(j, `running · ${j.runs.length} of ${n}`).catch(() => {}); }, HEARTBEAT_MS);
    let cutting = false;
    try {
      // Wait for room on the GPU (before the engine loads); the record says why.
      for (let why = await courtesy(j); why && !STOP.has(j.id); why = await courtesy(j)) {
        if (j.wait !== why) { j.wait = why; j.status = 'waiting'; step(j, 'safety', `Waiting for the GPU: ${why}`).end(true, 'checks again every 20 s; “Run anyway” starts now'); save(j); await liveRecord(j, 'waiting for the GPU', { wait: why }); }
        await sleep(COURTESY_POLL_MS);
      }
      if (j.wait) { j.wait = null; j.status = 'running'; step(j, 'safety', ANYWAY.has(j.id) || j.anyway ? 'Run anyway: starting without waiting for the GPU' : 'The GPU has room: running').end(true); save(j); }
      if (plan.instrument === 'tipping' && !STOP.has(j.id)) {
        const sa = step(j, 'safety', 'Tipping point: check no cut of yours is in force, record the intact answers');
        try { await C.tipBefore(plan, ctx); sa.end(true, 'ready to cut'); } catch (e) { sa.end(false, e.message); throw e; }
        cutting = true; j.cutting = true; save(j);
      }
      let mk = null;
      for (let i = j.runs.length; i < n && !STOP.has(j.id); i++) {
        const it = plan.items[i];
        const st = step(j, 'prompt', `Prompt ${i + 1} of ${n}`, `text: ${it.text}${Object.keys(C.EXTRA[plan.instrument] || {}).map((k) => `\n${k}: ${it[k]}`).join('')}`);
        if (mk == null) mk = await modelKey(plan);   // known once the engine has answered once; asked again until then
        const key = mk ? itemKey(plan, it, mk) : null, hit = key && cacheGet(key);
        let t;
        if (hit) { t = hit; t.cached = true; j.cached = (j.cached || 0) + 1; st.end(true, 'from the cache: the same prompt, settings and model were run before'); }
        else {
          try { t = await C.runItem(plan, it, ctx); } catch (e) { st.end(false, e.message); throw e; }
          st.end(!(t && t.error), t && t.error ? `not measured: ${t.error}` : 'done');
          if (!mk) mk = await modelKey(plan);
          if (mk && t && !t.error) cachePut(itemKey(plan, it, mk), C.slim(t));
        }
        j.runs.push({ item: it, t: C.slim(t) }); save(j);
        await liveRecord(j, `running · ${j.runs.length} of ${n}`);
      }
      const stopped = j.runs.length < n;
      if (stopped) step(j, 'run', `Stopped after ${j.runs.length} of ${n}`).end(true);
      if (cutting) { const ra = step(j, 'safety', 'Tipping point: restore the model and ask the engine to confirm'); j.tip = await C.tipAfter(ctx); cutting = false; j.cutting = false; ra.end(j.tip.ok, j.tip.ok ? 'restored' : j.tip.msg); }
      j.status = stopped ? 'stopped' : 'done'; j.finished = new Date().toISOString();
      step(j, 'report', j.runs.length >= 2 ? 'Results saved: the report is measured and drawn when the record is opened' : 'Fewer than two results: nothing to compare').end(j.runs.length >= 2);
      await liveRecord(j, stopped ? (j.runs.length >= 2 ? 'stopped early' : 'stopped: nothing to compare') : 'ran', { measured: false });
    } catch (e) {
      if (cutting) { const ra = step(j, 'safety', 'Tipping point: restore the model and ask the engine to confirm'); j.tip = await C.tipAfter(ctx); j.cutting = false; ra.end(j.tip.ok, j.tip.ok ? 'restored' : j.tip.msg); }
      step(j, 'error', `Stopped by an error after ${j.runs.length} of ${n}`).end(false, e.message);
      j.status = 'failed'; j.error = String(e.message || e).slice(0, 500); j.finished = new Date().toISOString();
      await liveRecord(j, j.runs.length >= 2 ? 'stopped by an error' : `failed: ${j.error.slice(0, 40)}`, { measured: false }).catch(() => {});
    } finally {
      clearInterval(beat); STOP.delete(j.id); ANYWAY.delete(j.id); j.wait = null; save(j); current = null;
      log(`${j.status}: ${j.name} (${j.runs.length} of ${n}${j.cached ? `, ${j.cached} from the cache` : ''})`);
      try { opts.notify && opts.notify(brief(j)); } catch (_) { /* a notice only */ }
    }
  }
  let pumping = false;
  async function pump() {
    if (pumping) return; pumping = true;
    try {
      for (;;) {
        const next = all().filter((j) => j.status === 'queued' && j.owner && j.owner.pid === process.pid).sort((a, b) => (a.at < b.at ? -1 : 1))[0];
        if (!next) break;
        await runJob(next);
      }
    } finally { pumping = false; }
  }

  // ── the calls ──
  function submit(a) {
    const v = C.validate(a.plan, config.guard ? config.guard() : {}); if (!v.ok) throw new Error(`The plan was refused: ${v.errors.join('; ')}`);
    const plan = v.plan, n = plan.items.length;
    // Results already made (the rest of an on-screen run sent here): kept when they are for this plan's prompts, in order.
    const runs = Array.isArray(a.runs) ? a.runs.slice(0, n).filter((r, i) => r && r.item && r.item.text === plan.items[i].text && r.t && typeof r.t === 'object') : [];
    if (runs.length !== (Array.isArray(a.runs) ? Math.min(a.runs.length, n) : 0)) throw new Error('The results sent with the plan do not match its prompts.');
    const id = `j${Date.now()}-${crypto.randomBytes(3).toString('hex')}`;
    const j = { id, name: C.nameFor(plan), plan, edited: Array.isArray(a.edited) ? a.edited.map((x) => String(x).slice(0, 140)).slice(0, 60) : [], at: new Date().toISOString(),
      status: 'queued', runs: C.slim(runs), activity: Array.isArray(a.activity) ? a.activity.slice(-400) : [], owner: OWNER, origin: a.origin === 'api' ? 'api' : 'app', record: a.record || null, cached: 0 };
    step(j, 'run', `Sent to the job runner (${OWNER.label})${runs.length ? ` with ${runs.length} of ${n} already done` : ''}: it runs here even if this screen closes`).end(true, current ? 'queued behind the running job' : 'starting');
    save(j);
    liveRecord(j, `queued · ${runs.length} of ${n}`).then(() => save(j)).catch((e) => log(`record: ${e.message}`)).finally(() => pump());
    return { ...brief(j), record: j.record };
  }
  function call_(a = {}) {
    switch (a.action) {
      case 'submit': return submit(a);
      case 'list': return all().map(brief);
      case 'get': { const j = read(a.id); return { ...brief(j), plan: j.plan, activity: j.activity, runs: a.runs ? j.runs : undefined }; }
      case 'active': { const j = all().find((x) => ['queued', 'waiting', 'running'].includes(x.status)); return j ? brief(j) : null; }
      case 'stop': { const j = read(a.id); if (j.status === 'queued') { j.status = 'stopped'; j.finished = new Date().toISOString(); save(j); liveRecord(j, 'stopped before it started').catch(() => {}); } else STOP.add(j.id); if (wake) wake(); return brief(j); }
      case 'run_anyway': { const j = read(a.id); j.anyway = true; save(j); ANYWAY.add(j.id); if (wake) wake(); return brief(j); }
      case 'resume': {
        const j = read(a.id); if (!['interrupted', 'failed', 'stopped'].includes(j.status)) throw new Error(`This job is ${j.status}: only an interrupted, failed or stopped job resumes.`);
        if (j.runs.length >= j.plan.items.length) throw new Error('Every prompt of this job is already done.');
        j.status = 'queued'; j.owner = OWNER; j.error = null; j.finished = null; save(j); pump(); return brief(j);
      }
      case 'delete': { const j = read(a.id); if (['running', 'waiting'].includes(j.status)) throw new Error('Stop the job first.'); fs.unlinkSync(fileOf(j.id)); return { deleted: j.id }; }
      case 'clear_cache': { let k = 0; try { for (const f of fs.readdirSync(CACHE)) { fs.unlinkSync(path.join(CACHE, f)); k++; } } catch (_) { /* none */ } return { cleared: k }; }
      default: throw new Error('jobs: action is one of submit, list, get, active, stop, run_anyway, resume, delete, clear_cache');
    }
  }
  recover(); setTimeout(pump, 1000);
  return { call: call_, current: () => (current ? brief(current) : null) };
}

module.exports = { createJobs };
