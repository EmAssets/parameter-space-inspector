// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Every result, kept as raw data: what a view's run returned (src/session-api.js run and store) and what each MCP tool
// call returned (src/mcp.js: Prometheus and outside agents), with the call, its arguments, the time and the source.
// For PSI Studio and anyone else who wants to visualise or re-analyse PSI's measurements without running them again.
// Local only (the data folder), never sent anywhere. Settings > records: { keep: true | false, maxGB } (on by default;
// the oldest days are removed past the cap).
//
// The format (psi-record 1), one gzipped JSON file per result:
//   <data>/records/<YYYY-MM-DD>/<HHMMSS-mmm>-<call>-<4 hex>.json.gz
//   { "psi_record": 1, "id", "at" (ISO time), "source": "view" | "prometheus" | "agent", "view" (a view, or null),
//     "call" (the API call or MCP tool), "args" (its arguments), "value" (what it returned, unchanged), "app" (version) }
// and an index, one JSON line per record: <data>/records/index.jsonl  { id, at, source, view, call, file, bytes, title }.
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const crypto = require('crypto');

const ID = /^\d{4}-\d{2}-\d{2}\/\d{6}-\d{3}-[a-z0-9_-]{1,48}-[0-9a-f]{4}$/i;
// Calls that hold no measurement (settings, keys, status, the records themselves) are not kept.
const SKIP = new Set(['records_list', 'record_read', 'psi_status', 'setup_check', 'glossary', 'guide_search', 'checkpoints_list', 'seed_grow_status']);
let VERSION = '0'; try { VERSION = require('../package.json').version || '0'; } catch (_) { /* version 0 */ }

function createRecords({ dataDir, settings = () => ({}), log = () => {} }) {
  const DIR = path.join(dataDir, 'records'), INDEX = path.join(DIR, 'index.jsonl');
  const on = () => { const r = settings().records || {}; return r.keep !== false; };
  const capBytes = () => { const g = Number((settings().records || {}).maxGB); return (Number.isFinite(g) && g > 0 ? g : 2) * 2 ** 30; };
  // A short title for lists: the prompt or question if the call had one.
  const titleOf = (args) => {
    const a = Array.isArray(args) ? args[0] : args;
    const t = typeof a === 'string' ? a : a && typeof a === 'object' ? (a.prompt || a.q || a.text || a.question || a.claim || a.action || '') : '';
    return String(t).replace(/\s+/g, ' ').slice(0, 140);
  };
  let saved = 0;

  function save({ source, view = null, call, args, value }) {
    if (!on() || !call || SKIP.has(call)) return null;
    try {
      const now = new Date(), day = now.toISOString().slice(0, 10), hms = now.toISOString().slice(11, 23).replace(/[:.]/g, '').replace(/^(\d{6})(\d{3})$/, '$1-$2');
      const id = `${day}/${hms}-${String(call).replace(/[^a-z0-9_-]/gi, '').slice(0, 48)}-${crypto.randomBytes(2).toString('hex')}`;
      const rec = { psi_record: 1, id, at: now.toISOString(), source, view, call, args: args === undefined ? null : args, value: value === undefined ? null : value, app: VERSION };
      const gz = zlib.gzipSync(Buffer.from(JSON.stringify(rec), 'utf8'), { level: 6 });
      fs.mkdirSync(path.join(DIR, day), { recursive: true, mode: 0o700 });
      const file = path.join(DIR, `${id}.json.gz`); fs.writeFileSync(file, gz, { mode: 0o600 });
      fs.appendFileSync(INDEX, JSON.stringify({ id, at: rec.at, source, view, call, file: `${id}.json.gz`, bytes: gz.length, title: titleOf(args) }) + '\n', { mode: 0o600 });
      if (++saved % 50 === 1) prune();
      return id;
    } catch (e) { log(`could not keep a record of ${call}: ${e.message}`); return null; }
  }

  // Past the cap, the oldest days go (whole days, so a day's records stay together), and the index with them.
  function prune() {
    try {
      const days = fs.readdirSync(DIR).filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d)).sort();
      const size = (d) => fs.readdirSync(path.join(DIR, d)).reduce((s, f) => s + fs.statSync(path.join(DIR, d, f)).size, 0);
      const sizes = days.map(size); let total = sizes.reduce((a, b) => a + b, 0); const gone = new Set();
      for (let i = 0; i < days.length - 1 && total > capBytes(); i++) { fs.rmSync(path.join(DIR, days[i]), { recursive: true, force: true }); total -= sizes[i]; gone.add(days[i]); }
      if (gone.size) {
        const keep = read_index().filter((r) => !gone.has(r.id.slice(0, 10)));
        fs.writeFileSync(INDEX, keep.map((r) => JSON.stringify(r)).join('\n') + (keep.length ? '\n' : ''), { mode: 0o600 });
        log(`records: removed ${gone.size} oldest day(s) to stay under ${(capBytes() / 2 ** 30).toFixed(1)} GB`);
      }
    } catch (_) { /* nothing kept yet */ }
  }
  function read_index() {
    let txt = ''; try { txt = fs.readFileSync(INDEX, 'utf8'); } catch (_) { return []; }
    return txt.split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (_) { return null; } }).filter(Boolean);
  }

  function list({ call, source, view, since, text, offset = 0, limit = 50 } = {}) {
    const t = String(text || '').toLowerCase();
    const rows = read_index().reverse().filter((r) => (!call || r.call === call) && (!source || r.source === source) && (!view || r.view === view)
      && (!since || r.at >= since) && (!t || String(r.title || '').toLowerCase().includes(t)));
    const o = Math.max(0, +offset || 0), n = Math.max(1, Math.min(500, +limit || 50));
    return { folder: DIR, keeping: on(), total: rows.length, offset: o, rows: rows.slice(o, o + n),
      calls: [...new Set(rows.map((r) => r.call))].sort(), format: 'psi-record 1: gzipped JSON per result; see src/records.js' };
  }
  function read(id) {
    if (!ID.test(String(id || ''))) throw new Error('records: an id from records_list');
    return JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(DIR, `${id}.json.gz`))).toString('utf8'));
  }
  return { save, list, read, prune, dir: DIR };
}

module.exports = { createRecords, SKIP };
