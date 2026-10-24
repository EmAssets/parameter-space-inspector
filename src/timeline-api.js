// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// The Timeline view's server side: placing a model on the emergence timeline (src/place_model.py) and the graft lab
// (src/graft.py), one Python process at a time, each result kept with its signed receipt in <data>/timeline/.
// Runs start only from the view's own buttons (confirm: true), never from an agent. The timeline itself is a data file
// the page loads (src/ui/timeline-data.js, from research/emergence_timeline.py).
const { python: PY } = require('./python.js');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

function createTimeline({ config, log }) {
  const DIR = path.join(config.DATA, 'timeline'), PLACED = path.join(DIR, 'placements'), GRAFTS = path.join(DIR, 'grafts'), SERIES = path.join(DIR, 'series'), FITS = path.join(DIR, 'fits');
  const mk = (d) => fs.mkdirSync(d, { recursive: true });
  const readJson = (f) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (_) { return null; } };
  const safe = (s) => String(s).replace(/[^a-z0-9._-]+/gi, '_').replace(/^_+|_+$/g, '').slice(0, 120) || 'model';

  // ── the models a run may use: the Hugging Face folders of the model listing, with their shape ──
  const stepOf = (p) => { const m = /(?:^|\/)step(\d+)(?:\/|$)/.exec(`${p.replace(/\\/g, '/')}/`); return m ? Number(m[1]) : null; };
  // The vocabulary, not the file: a tokenizer re-saved by another library version has other bytes but the same pieces.
  const tokHash = (dir) => {
    const h = (s) => crypto.createHash('sha256').update(s).digest('hex').slice(0, 16);
    const sorted = (o) => (Array.isArray(o) ? o : Object.keys(o).sort().map((k) => [k, o[k]]));
    try {
      const j = JSON.parse(fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf8'));
      const added = (j.added_tokens || []).map((a) => [a.id, a.content]).sort((a, b) => a[0] - b[0]);
      const merges = (j.model && j.model.merges || []).map((x) => (Array.isArray(x) ? x.join(' ') : x));   // pairs or "a b" strings, by version
      if (j.model && j.model.vocab) return h(JSON.stringify([sorted(j.model.vocab), merges, added]));
    } catch (_) { /* no tokenizer.json, or another shape */ }
    try { return h(JSON.stringify(sorted(JSON.parse(fs.readFileSync(path.join(dir, 'vocab.json'), 'utf8'))))); } catch (_) { /* next */ }
    try { return h(fs.readFileSync(path.join(dir, 'tokenizer.model'))); } catch (_) { /* none */ }
    return null;
  };
  function shape(m) {
    let c = {}; try { c = JSON.parse(fs.readFileSync(path.join(m.path, 'config.json'), 'utf8')); } catch (_) { /* unreadable */ }
    const t = c.text_config || c, heads = t.num_attention_heads || null, hidden = t.hidden_size || null;
    const rp = t.rope_parameters || {};
    return { name: m.name, path: m.path, arch: m.arch, model_type: t.model_type || c.model_type || null, params: m.params_estimate || null, gb: m.gb,
      layers: t.num_hidden_layers || null, hidden, heads, kv: t.num_key_value_heads || heads, head_dim: t.head_dim || (hidden && heads ? hidden / heads : null),
      ffn: t.intermediate_size || null, vocab: t.vocab_size || null, rope: JSON.stringify([rp.rope_theta ?? t.rope_theta ?? null, rp.rope_type ?? t.rope_scaling?.rope_type ?? 'default', t.rotary_pct ?? rp.partial_rotary_factor ?? null]),
      tokenizer: tokHash(m.path), step: stepOf(m.path), family: stepOf(m.path) != null ? path.basename(path.dirname(m.path)) : null };
  }
  let MODELS = null;
  function models(a = {}) {
    if (!MODELS || a.refresh) MODELS = config.listModels().hf.map(shape);
    return { models: MODELS };
  }
  const byPath = (p) => { const k = path.resolve(String(p || '')).toLowerCase(); return models().models.find((m) => path.resolve(m.path).toLowerCase() === k) || null; };
  // The same classes graft.py decides by (it decides again at run time, from the loaded models and their vocabularies).
  function compatOf(h, d) {
    const sameVocab = !!h.tokenizer && h.tokenizer === d.tokenizer;
    if (h.model_type === d.model_type && h.hidden === d.hidden && h.heads === d.heads && h.kv === d.kv && h.head_dim === d.head_dim && h.rope === d.rope)
      return { kind: 'drop-in', same_vocab: sameVocab, why: 'same architecture and attention shape: the donor’s layers go in as they are' };
    if (h.hidden && h.hidden === d.hidden) return { kind: 'same width', same_vocab: sameVocab, why: `both ${h.hidden} wide: the donor’s layers run with their own positions${sameVocab ? '' : ' (different vocabularies: no fitted map)'}` };
    if (sameVocab) return { kind: 'stitched', same_vocab: true, why: `${h.hidden} vs ${d.hidden} wide, same vocabulary: maps fitted each way on the same tokens` };
    return { kind: 'incompatible', same_vocab: false, why: `${h.hidden} vs ${d.hidden} wide and different vocabularies: no shared tokens to fit a map on` };
  }
  function compat(a) {
    const h = byPath(a.host); if (!h) throw new Error('the host is not a model folder from the listing (Settings > Local model > Scan)');
    const ds = a.donor ? [byPath(a.donor)].filter(Boolean) : models().models.filter((m) => m.path !== h.path);
    return { host: h, donors: ds.map((d) => ({ ...d, compat: compatOf(h, d) })) };
  }

  // ── one Python process at a time ──
  let job = null;
  const SRC = __dirname;
  function run(kind, args, out, meta) {
    if (job && !job.finished) throw new Error(`Something is already running here (${job.label}). Wait for it or stop it.`);
    mk(path.dirname(out));
    job = { id: crypto.randomBytes(6).toString('hex'), kind, label: meta.label, started: new Date().toISOString(), out, progress: null, done: null, error: null, lines: [], finished: false, meta };
    const j = job;
    j.proc = spawn(PY(), ['-u', ...args], { cwd: SRC, windowsHide: true, env: config.cacheEnv({ PYTHONIOENCODING: 'utf-8', PSI_DATA_DIR: config.DATA }) });
    const reader = () => { let buf = ''; return (d) => {
      buf += d; let i;
      while ((i = buf.search(/[\r\n]/)) >= 0) {
        const l = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!l) continue;
        const m = /^PSI-(PROGRESS|DONE|ERROR) ?(.*)$/.exec(l);
        if (m && m[1] === 'PROGRESS') { try { j.progress = JSON.parse(m[2]); } catch (_) { /* a line only */ } }
        else if (m && m[1] === 'DONE') { try { j.done = JSON.parse(m[2]); } catch (_) { j.done = {}; } }
        else if (m && m[1] === 'ERROR') j.error = m[2];
        else if (!/^Loading weights|it\/s\]|%\|/.test(l)) { j.lines.push(l.slice(0, 300)); if (j.lines.length > 40) j.lines.shift(); }
      }
    }; };
    j.proc.stdout.on('data', reader()); j.proc.stderr.on('data', reader());
    j.proc.on('close', (code) => {
      j.finished = true; j.proc = null; j.ended = new Date().toISOString();
      if (code && !j.error) j.error = j.cancelled ? 'stopped' : `it stopped (exit ${code}): ${j.lines.slice(-2).join(' ')}`;
      log && log(j.error ? `${j.label}: ${j.error}` : `${j.label}: done`);
    });
    log && log(`${j.label}: started`);
    return status();
  }
  function status() {
    if (!job) return { running: false, job: null };
    const { proc, ...j } = job; return { running: !job.finished, job: j };
  }
  function cancel() { if (job && job.proc) { job.cancelled = true; try { job.proc.kill(); } catch (_) { /* gone */ } } return status(); }
  const device = (d) => (['cpu', 'cuda', 'auto'].includes(d) ? d : 'auto');

  // ── placing a model ──
  function place(a) {
    if (a.confirm !== true) throw new Error('Placing a model starts only from the Timeline view’s own button.');
    const m = byPath(a.model); if (!m) throw new Error('not a model folder from the listing (Settings > Local model > Scan)');
    const id = safe(m.name), out = path.join(PLACED, `${id}.json`);
    const args = [path.join(SRC, 'place_model.py'), '--model', m.path, '--out', out, '--device', device(a.device), '--name', m.name];
    if (a.abstraction === false) args.push('--no-abstraction');
    return run('place', args, out, { label: `placing ${m.name}`, model: m.name, id });
  }
  function summary(f) {
    const d = readJson(f); if (!d) return null;
    const ev = (d.result || {}).events || {};
    return { id: path.basename(f, '.json'), model: d.model, measured_at: d.measured_at, device: d.device, seconds: d.seconds, events: ev,
      nearest: ((d.derived || {}).nearest || [])[0] || null, signed: !!(d.receipt || {}).signature, receipt_sha256: (d.receipt || {}).receipt_sha256 || null };
  }
  function placements() {
    let fl = []; try { fl = fs.readdirSync(PLACED).filter((f) => f.endsWith('.json')); } catch (_) { /* none yet */ }
    return { placements: fl.map((f) => summary(path.join(PLACED, f))).filter(Boolean).sort((x, y) => String(y.measured_at).localeCompare(String(x.measured_at))) };
  }
  const fileOf = (kind, id) => {
    const dir = kind === 'graft' ? GRAFTS : kind === 'series' ? SERIES : PLACED, f = path.join(dir, `${safe(id)}.json`);
    if (!fs.existsSync(f)) throw new Error(`no ${kind} called ${id}`); return f;
  };
  const get = (a) => readJson(fileOf(a.kind || 'placement', a.id));

  // ── the graft lab ──
  function graft(a) {
    if (a.confirm !== true) throw new Error('A graft starts only from the Timeline view’s own button.');
    const cmd = ['graft', 'sweep', 'rank'].includes(a.cmd) ? a.cmd : 'graft';
    const h = byPath(a.host); if (!h) throw new Error('the host is not a model folder from the listing');
    const args = [path.join(SRC, 'graft.py'), cmd, '--host', h.path, '--device', device(a.device)];
    let label;
    if (cmd === 'rank') {
      const ds = (Array.isArray(a.donors) ? a.donors : []).map(byPath).filter(Boolean).slice(0, 12);
      if (!ds.length) throw new Error('choose at least one donor'); args.push('--donors', ds.map((d) => d.path).join(','));
      label = `ranking ${ds.length} donor${ds.length > 1 ? 's' : ''} for ${h.name}`;
    } else {
      const d = byPath(a.donor); if (!d) throw new Error('the donor is not a model folder from the listing');
      if (compatOf(h, d).kind === 'incompatible') throw new Error(compatOf(h, d).why);
      args.push('--donor', d.path);
      if (cmd === 'graft') {
        const r = (x, n, what) => { const m = /^(\d+):(\d+)$/.exec(String(x || '')); if (!m || +m[1] > +m[2] || +m[2] > n) throw new Error(`${what} must be from:to within 0..${n}, e.g. 3:6`); return `${m[1]}:${m[2]}`; };
        args.push('--host-layers', r(a.hostLayers, h.layers, 'the host’s layers'), '--donor-layers', r(a.donorLayers, d.layers, 'the donor’s layers'));
        label = `graft ${d.name} ${a.donorLayers} into ${h.name} ${a.hostLayers}`;
      } else {
        args.push('--width', String(Math.max(1, Math.min(8, Number(a.width) || 1))));
        if (a.full) args.push('--full'); else args.push('--band', String(Math.max(0.02, Math.min(1, Number(a.band) || 0.15))));
        label = `sweep ${d.name} into ${h.name}`;
      }
    }
    if (a.align) args.push('--align');
    const id = `${new Date().toISOString().replace(/[:.]/g, '-')}-${cmd}`, out = path.join(GRAFTS, `${id}.json`);
    args.push('--out', out);
    return run('graft', args, out, { label, id, cmd });
  }
  function grafts() {
    let fl = []; try { fl = fs.readdirSync(GRAFTS).filter((f) => f.endsWith('.json')); } catch (_) { /* none yet */ }
    const out = fl.map((f) => { const d = readJson(path.join(GRAFTS, f)); if (!d) return null;
      return { id: path.basename(f, '.json'), kind: d.kind, host: d.host, donor: d.donor || null, donors: d.donors ? d.donors.length : null, host_layers: d.host_layers || null, donor_layers: d.donor_layers || null,
        width: d.width || null, how: d.how || null, compat: d.compat || null, fit: d.fit || null, best: (d.best || [])[0] || (d.donors || [])[0] || null, seconds: d.seconds, signed: !!(d.receipt || {}).signature,
        at: (d.receipt || {}).record ? d.receipt.record.created : null }; }).filter(Boolean);
    return { grafts: out.sort((x, y) => String(y.at).localeCompare(String(x.at))) };
  }

  // ── receipts: check one, sign a series ──
  function py(code, args) {
    return new Promise((resolve, reject) => {
      const p = spawn(PY(), ['-c', code, ...args], { cwd: SRC, windowsHide: true, env: config.cacheEnv({ PYTHONIOENCODING: 'utf-8', PSI_DATA_DIR: config.DATA }) });
      let o = '', e = ''; p.stdout.on('data', (d) => { o += d; }); p.stderr.on('data', (d) => { e += d; });
      p.on('close', (c) => { if (c) return reject(new Error(e.trim().split(/\r?\n/).slice(-1)[0] || `exit ${c}`)); try { resolve(JSON.parse(o)); } catch (_) { reject(new Error(`unreadable answer: ${o.slice(0, 200)}`)); } });
    });
  }
  async function verify(a) {
    const f = fileOf(a.kind || 'placement', a.id);
    return py('import json,sys,receipts as r\nd=json.load(open(sys.argv[1],encoding="utf-8"))\nres=d["result"] if "result" in d else {k:v for k,v in d.items() if k not in ("receipt","seconds")}\n'
      + 'print(json.dumps(r.check(d["receipt"],res,(d.get("model") or {}).get("path") if sys.argv[2]=="1" else None)))', [f, a.files ? '1' : '0']);
  }
  // A series: every placement of one family's checkpoints, in step order, under one signed Merkle root.
  async function signSeries(a) {
    const fam = String(a.family || ''); if (!fam) throw new Error('which family? (the folder that holds the step folders)');
    const ps = placements().placements.filter((p) => p.model && p.model.family === fam && p.model.step != null).sort((x, y) => x.model.step - y.model.step);
    if (ps.length < 2) throw new Error(`a series needs at least two placed checkpoints of ${fam}; ${ps.length} placed`);
    mk(SERIES); const out = path.join(SERIES, `${safe(fam)}.json`);
    const files = ps.map((p) => path.join(PLACED, `${p.id}.json`));
    const s = await py('import json,sys,receipts as r\nfs=json.loads(sys.argv[2])\nrs=[json.load(open(f,encoding="utf-8"))["receipt"] for f in fs]\nprint(json.dumps(r.sign_series(sys.argv[1],rs)))', [fam, JSON.stringify(files)]);
    s.members = ps.map((p) => ({ id: p.id, step: p.model.step, receipt_sha256: p.receipt_sha256 }));
    fs.writeFileSync(out, JSON.stringify(s, null, 1));
    return s;
  }
  function seriesList() {
    let fl = []; try { fl = fs.readdirSync(SERIES).filter((f) => f.endsWith('.json')); } catch (_) { /* none */ }
    return { series: fl.map((f) => { const d = readJson(path.join(SERIES, f)); return d && { id: path.basename(f, '.json'), name: d.record.name, root: d.record.merkle_root, leaves: d.record.leaves, created: d.record.created, signed: !!d.signature, members: d.members }; }).filter(Boolean) };
  }
  // ── what fits here: training this model on this GPU (src/train_fit.py), estimated, then measured on request ──
  const METHODS = ['full', 'full8', 'lora', 'qlora'];
  const fitFile = (m, method) => path.join(FITS, `${safe(m.name)}__${method}.json`);
  async function gpuNow() {
    try { const g = await require('./gpu.js').gpuState({ maxAgeMs: 0 }); const x = (g.gpus || [])[0]; if (x && x.freeMB != null) return { name: x.name, free_gb: +(x.freeMB / 1024).toFixed(2), total_gb: +(x.totalMB / 1024).toFixed(2) }; } catch (_) { /* no GPU reader */ }
    return null;
  }
  async function fit(a) {
    const m = byPath(a.model); if (!m) throw new Error('not a model folder from the listing (Settings > Local model > Scan)');
    const g = await gpuNow();
    const args = [path.join(SRC, 'train_fit.py'), 'estimate', m.path, '--seq', String(a.seq || 1024)];
    if (g) args.push('--free-gb', String(g.free_gb), '--gpu', g.name);
    const r = require('child_process').spawnSync(PY(), args, { cwd: SRC, windowsHide: true, encoding: 'utf8', env: config.cacheEnv({ PYTHONIOENCODING: 'utf-8' }) });
    const line = String(r.stdout || '').split(/\r?\n/).filter((l) => l.startsWith('{')).pop();
    if (!line) throw new Error(`the estimate failed: ${String(r.stderr || '').trim().split(/\r?\n/).pop() || 'no output'}`);
    const est = JSON.parse(line);
    for (const k of METHODS) {   // idle: the whole card less about 1 GB the desktop keeps
      est.methods[k].measured = readJson(fitFile(m, k));
      est.methods[k].fits_idle = g ? est.methods[k].gb + 0.6 <= g.total_gb - 1.0 : null;
    }
    const partners = compat({ host: m.path }).donors.filter((d) => d.compat.kind !== 'incompatible').map((d) => ({ name: d.name, path: d.path, kind: d.compat.kind, why: d.compat.why }));
    return { ...est, gpu_now: g, partners };
  }
  async function fitMeasure(a) {
    if (a.confirm !== true) throw new Error('A measured run starts only from the Timeline view’s own button.');
    const m = byPath(a.model); if (!m) throw new Error('not a model folder from the listing');
    const method = METHODS.includes(a.method) ? a.method : 'lora';
    const g = await gpuNow(); if (!g) throw new Error('no GPU reading: a measured run needs a GPU');
    mk(FITS); const out = fitFile(m, method);
    const args = [path.join(SRC, 'train_fit.py'), 'measure', m.path, '--method', method, '--seq', String(a.seq || 1024), '--cap-gb', String(Math.max(0.5, g.free_gb - 0.5)), '--out', out];
    return run('fit', args, out, { label: `measuring ${method} training of ${m.name}`, model: m.name, method });
  }

  function remove(a) {
    const f = fileOf(a.kind || 'placement', a.id); fs.unlinkSync(f); return { removed: a.id };
  }

  async function call(a = {}) {
    switch (a.action) {
      case 'models': return models(a);
      case 'compat': return compat(a);
      case 'place': return place(a);
      case 'placements': return placements();
      case 'get': return get(a);
      case 'graft': return graft(a);
      case 'grafts': return grafts();
      case 'status': return status();
      case 'cancel': return cancel();
      case 'verify': return verify(a);
      case 'sign_series': return signSeries(a);
      case 'series': return seriesList();
      case 'delete': return remove(a);
      case 'fit': return fit(a);
      case 'fit_measure': return fitMeasure(a);
      default: throw new Error(`timeline: unknown action ${a.action}`);
    }
  }
  return { call, status };
}

module.exports = { createTimeline };
