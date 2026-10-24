// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// What each view last showed, kept on the server so it survives a reload, a phone dropping the tab, back/forward, and
// carries over to another device: the view's inputs, its last result, and a call still running. A view's long call goes
// through run(): the result is written into the view's state when it finishes, whether or not the page is still there
// to receive it, so coming back shows it. Files: <data>/session/<view>.json. Only the app's own page calls this (main.js
// and devserver.js keep it out of the API that MCP agents and job plans reach); run() calls only what the page itself
// may call.
//   get    { view }                          -> { inputs, result, error, pending, at, done_at } or {}
//   put    { view, inputs }                  -> saves the inputs (a few KB), keeps the result
//   run    { view, id, name, args, inputs }  -> the call's value (or its error), also kept in the view's state
//   store  { view, inputs, result }          -> a result the page built itself (Generate's text, step by step)
//   call   { id }                            -> { state: running | done | failed | unknown, value?, error? }
//   clear  { view }
'use strict';
const fs = require('fs');
const path = require('path');

const VIEW = /^[a-z][a-z0-9-]{0,40}$/;
const ID = /^[A-Za-z0-9_-]{6,64}$/;
const MAX_RESULT = 6 * 1048576;    // characters of JSON kept per view; a larger result keeps the inputs only, and says so
const MAX_INPUTS = 64 * 1024;
const CALL_HOURS = 24;

function createSession({ dataDir, api, log = () => {}, records = null }) {
  const DIR = path.join(dataDir, 'session');
  const file = (v) => path.join(DIR, `${v}.json`);
  const check = (v) => { if (!VIEW.test(String(v || ''))) throw new Error('session: a view name'); return v; };
  const read = (v) => { try { return JSON.parse(fs.readFileSync(file(v), 'utf8')); } catch (_) { return {}; } };
  const write = (v, s) => {
    try { fs.mkdirSync(DIR, { recursive: true, mode: 0o700 }); const tmp = `${file(v)}.tmp`; fs.writeFileSync(tmp, JSON.stringify(s), { mode: 0o600 }); fs.renameSync(tmp, file(v)); }
    catch (e) { log(`could not keep ${v}: ${e.message}`); }
  };
  const inputsOf = (x) => {
    if (x == null) return undefined;
    if (typeof x !== 'object' || Array.isArray(x)) throw new Error('session: inputs is an object');
    if (JSON.stringify(x).length > MAX_INPUTS) throw new Error('session: inputs larger than 64 KB');
    return x;
  };

  // Calls by id, for a page that comes back while one is still running (memory only: a restart ends the call anyway).
  const calls = new Map();
  const prune = () => { const old = Date.now() - CALL_HOURS * 3600e3; for (const [k, c] of calls) if ((c.ended || c.started) < old) calls.delete(k); };

  async function run({ view, id, name, args, inputs }) {
    check(view); if (!ID.test(String(id || ''))) throw new Error('session: a call id');
    if (typeof name !== 'string' || !Object.hasOwn(api, name) || typeof api[name] !== 'function') throw new Error(`session: unknown call ${name}`);
    if (!Array.isArray(args)) throw new Error('session: args is a list');
    prune();
    const at = new Date().toISOString();
    const s = read(view); const kept = inputsOf(inputs);
    write(view, { ...s, ...(kept ? { inputs: kept } : {}), pending: { id, name, at } });
    const c = { id, view, state: 'running', started: Date.now() }; calls.set(id, c);
    const finish = (patch) => {
      const now = read(view);
      // A newer call on this view replaced the pending mark: its result wins, this one is only kept by id.
      if (now.pending && now.pending.id !== id) return;
      write(view, { ...now, ...patch, pending: null, done_at: new Date().toISOString() });
    };
    try {
      const value = await api[name](...args);
      c.state = 'done'; c.value = value; c.ended = Date.now();
      if (records) records.save({ source: 'view', view, call: name, args, value });   // the raw data, for PSI Studio (src/records.js)
      const size = JSON.stringify(value === undefined ? null : value).length;
      finish(size <= MAX_RESULT ? { result: value === undefined ? null : value, error: null, too_large: false } : { result: null, error: null, too_large: true });
      return value;
    } catch (e) {
      c.state = 'failed'; c.error = String(e.message || e); c.ended = Date.now();
      finish({ result: null, error: c.error, too_large: false });
      throw e;
    }
  }

  return {
    async call(a = {}) {
      const action = a.action || 'get';
      if (action === 'get') { const s = read(check(a.view)); if (s.pending && !calls.has(s.pending.id)) s.pending = { ...s.pending, lost: true }; return s; }
      if (action === 'put') { const v = check(a.view); const s = read(v); write(v, { ...s, inputs: inputsOf(a.inputs) || {}, at: new Date().toISOString() }); return { ok: true }; }
      if (action === 'run') return run(a);
      if (action === 'store') {
        const v = check(a.view); const s = read(v); const kept = inputsOf(a.inputs);
        const big = JSON.stringify(a.result === undefined ? null : a.result).length > MAX_RESULT;
        write(v, { ...s, ...(kept ? { inputs: kept } : {}), result: big ? null : (a.result === undefined ? null : a.result), too_large: big, error: null, pending: null, done_at: new Date().toISOString() });
        if (records) records.save({ source: 'view', view: v, call: 'view_result', args: kept || null, value: a.result });
        return { ok: true, too_large: big };
      }
      if (action === 'call') { const c = calls.get(String(a.id || '')); return c ? { state: c.state, value: c.value, error: c.error } : { state: 'unknown' }; }
      if (action === 'clear') { try { fs.unlinkSync(file(check(a.view))); } catch (_) { /* nothing kept */ } return { ok: true }; }
      throw new Error(`session: unknown action ${action}`);
    },
  };
}

module.exports = { createSession };
