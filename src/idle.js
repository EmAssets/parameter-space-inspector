// SPDX-License-Identifier: Apache-2.0
// Sleep when idle: a loaded model holds GPU memory (and RAM) whether or not anyone is using it. After `idleMin`
// minutes with no use (Settings > This computer; default 60, 0 = never), both engines are stopped and the memory
// goes back to the computer. The next call that needs a model wakes it with the same file and options (steering,
// LoRA, context), so a view just takes a few seconds longer the first time. Unloading by hand never wakes it.
//
// What counts as use: any call that asks the model something or changes it. The app's own polling (the machine
// strip, status checks, settings reads, conversion progress) does not, or the model would never sleep. A call still
// running (a long Anatomy pass, a generation) holds the model awake until it ends.
//
// The state is reported (state(), shown in the header and in Settings): when it last saw use, when it will sleep,
// and, once asleep, when, which model, and how much GPU memory that gave back. A sleep is never silent.
'use strict';

// Calls that are reading, not using: they neither count as use nor wake a sleeping model.
const PASSIVE = new Set(['gpu', 'engineState', 'setupCheck', 'traceStatus', 'convertStatus', 'settings', 'listModels', 'readHead',
  'pickModel', 'pickControlVector', 'setKey', 'external', 'lineage', 'convert', 'ladder', 'idle', 'neuronNames', 'mcp', 'quiet', 'experiments', 'jobs', 'timeline', 'commons', 'chatgpt', 'hf']);
// Calls served by the answering engine (llama-server): these wake it if it was put to sleep. The inside-the-model
// engine wakes itself (tracer.ensure() starts it on the first request).
const ENGINE = new Set(['tokenize', 'detokenize', 'next', 'complete', 'score', 'template']);

function createIdle({ engine, tracer, config, start, gpuState, log = () => {} }) {
  let lastUse = Date.now(), lastWhat = 'launch', inflight = 0, asleep = null, waking = null;
  const raw = {};   // the calls unwrapped, for quiet()
  const minutes = () => { const m = Number((config.settings.idle || {}).minutes); return Number.isFinite(m) && m >= 0 ? m : 60; };
  const running = () => !!(engine.proc || tracer.proc);

  // Before a call: note the use, and wake the answering engine if this call needs it and it was put to sleep.
  async function before(name) {
    if (PASSIVE.has(name)) return;
    lastUse = Date.now(); lastWhat = name; inflight++;
    if (name === 'stop') { asleep = null; return; }                 // unloaded by hand: stays unloaded
    if (name === 'start') { asleep = null; return; }
    if (ENGINE.has(name) && asleep && !engine.proc && asleep.model) {
      if (!waking) {
        const a = asleep; log(`waking ${a.model.split(/[\\/]/).pop()} (asleep since ${new Date(a.at).toLocaleTimeString()})`);
        waking = start(a.model, a.options || {}).then(() => { asleep = null; lastUse = Date.now(); }).finally(() => { waking = null; });
      }
      await waking;
    }
  }
  function after(name) { if (!PASSIVE.has(name)) { inflight = Math.max(0, inflight - 1); lastUse = Date.now(); } }
  // One call, wrapped so it goes through before/after (the unwrapped one is kept for quiet()).
  function track(name, fn) { raw[name] = fn; return async (...args) => { await before(name); try { return await fn(...args); } finally { after(name); } }; }
  // Wrap an API table: every call goes through before/after.
  function wrap(api) {
    for (const [name, fn] of Object.entries(api)) if (typeof fn === 'function' && name !== 'idle' && name !== 'quiet') api[name] = track(name, fn);
    return api;
  }
  // A measurement the page makes by itself (the context meter on prompt boxes, measured as a page opens): runs on the
  // answering engine only if it is running now, and neither counts as use nor wakes it, or an open tab would keep a
  // model awake (or wake it on every reload).
  async function quiet(name, ...args) {
    if (!ENGINE.has(name) || !raw[name]) throw new Error(`quiet: not an engine call: ${name}`);
    if (!engine.proc) throw new Error(asleep ? 'the model is asleep; measured when it wakes' : 'no model is running');
    return raw[name](...args);
  }

  async function sleep(why) {
    if (!running()) return null;
    const pids = [engine.proc && engine.proc.pid, tracer.proc && tracer.proc.pid].filter(Boolean);
    let heldMB = null;
    try { const g = await gpuState({ pids }); if (g && g.ok) heldMB = (g.holders || []).filter((h) => h.role === 'engine' || h.role === 'inside').reduce((s, h) => s + (h.mb || 0) + (h.sharedMB || 0), 0); } catch { /* not read: said as null */ }
    const a = { at: Date.now(), why, idle_min: minutes(), model: engine.model || null, options: engine.options || {}, inside: !!tracer.proc, freedMB: heldMB };
    await engine.stop(); await tracer.stop();
    asleep = a; log(`asleep: ${why}; ${a.model ? a.model.split(/[\\/]/).pop() : 'no answering model'}${a.inside ? ' + inside engine' : ''} stopped${heldMB != null ? `, ${(heldMB / 1024).toFixed(1)} GB GPU given back` : ''}`);
    return a;
  }

  const timer = setInterval(() => {
    const m = minutes();
    if (!m || inflight || waking || !running()) return;
    if (Date.now() - lastUse >= m * 60000) sleep(`no use for ${m} min`).catch((e) => log(`sleep failed: ${e.message || e}`));
  }, 30000);
  if (timer.unref) timer.unref();

  function state() {
    const m = minutes();
    return { minutes: m, running: running(), last_use_at: lastUse, last_use: lastWhat, in_flight: inflight,
      sleeps_at: m && running() ? lastUse + m * 60000 : null, waking: !!waking,
      asleep: asleep && { at: asleep.at, why: asleep.why, model: asleep.model, inside: asleep.inside, freedMB: asleep.freedMB } };
  }
  // The idle() call: read the state, set the minutes, or sleep now.
  async function call(a = {}) {
    if (a.action === 'set') {
      const m = Number(a.minutes); if (!Number.isInteger(m) || m < 0 || m > 24 * 60) throw new Error('idle minutes: a whole number from 0 (never) to 1440');
      config.update({ idle: { minutes: m } }); lastUse = Date.now();
    } else if (a.action === 'sleep') await sleep('put to sleep by hand');
    else if (a.action === 'forget') asleep = null;            // stay unloaded: no wake on the next call
    return state();
  }
  return { wrap, track, quiet, before, after, state, call, sleep };
}
module.exports = { createIdle, PASSIVE, ENGINE };
