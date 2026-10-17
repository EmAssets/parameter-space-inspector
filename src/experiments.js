// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The Experiment designer's history: one JSON file per experiment in <data folder>/experiments/, plus a small index.
// The page sends a record; this file names it (the id is made here, never taken from the page), caps its size and
// the number kept, and writes it private (0600 in a 0700 folder). Only the app's own pages reach it (main.js / the
// dev server's token). Nothing here is ever run or rendered as markup: records are data the page draws as text.
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const MAX_BYTES = 8 * 1024 * 1024, MAX_KEPT = 300, ID = /^[0-9]{13}-[0-9a-f]{8}$/;

// limits(): {maxRecordMB, historyKept} from the guardrails (config.guard), read on every save.
function createExperiments(DATA, limits = () => ({})) {
  const DIR = path.join(DATA, 'experiments'), INDEX = path.join(DIR, 'index.json');
  const mk = () => fs.mkdirSync(DIR, { recursive: true, mode: 0o700 });
  const write = (f, text) => { mk(); fs.writeFileSync(f, text, { mode: 0o600 }); try { fs.chmodSync(f, 0o600); } catch (_) { /* not supported here */ } };
  const readIndex = () => { try { const a = JSON.parse(fs.readFileSync(INDEX, 'utf8')); return Array.isArray(a) ? a : []; } catch (_) { return []; } };
  const fileOf = (id) => { if (!ID.test(String(id))) throw new Error('experiments: not an experiment id'); return path.join(DIR, `${id}.json`); };
  const str = (v, n) => String(v == null ? '' : v).slice(0, n);
  // The index entry: what the history list shows (no data).
  const entryOf = (id, r) => ({ id, name: str(r.name, 140), at: str(r.at, 40), instrument: str(r.plan && r.plan.instrument, 30), items: r.plan && Array.isArray(r.plan.items) ? r.plan.items.length : 0, status: str(r.status, 60), updated: str(r.updated, 40), job: r.job ? str(r.job, 30) : undefined });

  function call(a = {}) {
    const act = a.action;
    if (act === 'list') return readIndex();
    if (act === 'get') return JSON.parse(fs.readFileSync(fileOf(a.id), 'utf8'));
    if (act === 'delete') { const f = fileOf(a.id); try { fs.unlinkSync(f); } catch (_) { /* already gone */ } write(INDEX, JSON.stringify(readIndex().filter((x) => x.id !== a.id))); return { deleted: a.id }; }
    if (act === 'save') {
      const r = a.record; if (!r || typeof r !== 'object' || Array.isArray(r)) throw new Error('experiments: a record is an object');
      const L = limits() || {}, maxBytes = (L.maxRecordMB || MAX_BYTES / 1048576) * 1048576, kept = L.historyKept || MAX_KEPT;
      const text = JSON.stringify(r); if (Buffer.byteLength(text) > maxBytes) throw new Error(`experiments: a record is limited to ${maxBytes / 1048576} MB (Settings > Guardrails) (this one is ${(Buffer.byteLength(text) / 1048576).toFixed(1)} MB)`);
      // An existing id updates that record (e.g. Prometheus's reading arrives after the run); otherwise a new id is made here.
      const id = a.id && ID.test(String(a.id)) && fs.existsSync(fileOf(a.id)) ? a.id : `${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
      write(fileOf(id), text);
      let idx = readIndex().filter((x) => x.id !== id); idx.unshift(entryOf(id, r));
      for (const old of idx.slice(kept)) { try { fs.unlinkSync(fileOf(old.id)); } catch (_) { /* gone */ } }
      idx = idx.slice(0, kept); write(INDEX, JSON.stringify(idx));
      return { id };
    }
    throw new Error('experiments: action is one of list, get, save, delete');
  }
  return { call };
}

module.exports = { createExperiments };
