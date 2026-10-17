// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Hugging Face for the whole app. Configured in Settings > Hugging Face: the token (the same secret Commons publishes with,
// 'hf-token' in src/secrets.js, handed to Python as PSI_HF_TOKEN only for the call that needs it, never returned to the
// page) and settings.hf (owner and private for pushes, the checkpoint repo for Timeline and "When was it learned?").
// Nothing here works until the token has been checked against Hugging Face (status().ready): the page hides every
// Hugging Face control until then. The check is kept in hf-state.json with a fingerprint of the token, so a new token
// is checked again.
//   pull  search, info, family, refs, record, gguf_base (quick reads); get_model, get_adapter, get_sae, get_ckpt,
//         get_text (downloads into the models folder, the checkpoints folder or the data folder; one at a time, Stop)
//   push  plan, push (confirm: true, sent after the user saw the plan). Not offered to outside agents.
'use strict';
const { python: PY } = require('./python.js');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

const SRC = __dirname;
const REPO = /^[A-Za-z0-9][\w.-]{0,95}\/[A-Za-z0-9][\w.-]{0,95}$/;
const FILE = /^[\w.@+-]+(\/[\w.@+-]+)*$/;

function createHf({ config, secrets, log = () => {} }) {
  const STATE = path.join(config.DATA, 'hf-state.json');
  const token = () => secrets.get('hf-token');
  const fp = (t) => (t ? crypto.createHash('sha256').update(t).digest('hex').slice(0, 16) : '');
  const readState = () => { try { return JSON.parse(fs.readFileSync(STATE, 'utf8')); } catch (_) { return {}; } };
  const writeState = (s) => { try { fs.mkdirSync(config.DATA, { recursive: true }); fs.writeFileSync(STATE, JSON.stringify(s, null, 1)); } catch (e) { log(`could not keep the token check: ${e.message}`); } };
  const ready = () => { const t = token(), s = readState(); return !!(t && s.ok && s.fp === fp(t)); };
  const needReady = () => { if (!ready()) throw new Error('Hugging Face is not set up: add a token in Settings > Hugging Face and check it'); };
  const env = (withToken) => config.cacheEnv(withToken && token() ? { PSI_HF_TOKEN: token() } : {});
  const prefs = () => ({ owner: '', private: true, checkpoints: '', ...((config.settings && config.settings.hf) || {}) });

  function py(script, args, timeoutMs = 90000) {
    return new Promise((resolve, reject) => {
      const p = spawn(PY(), ['-u', path.join(SRC, script), ...args], { cwd: SRC, windowsHide: true, env: env(true) });
      let o = '', e = ''; const t = setTimeout(() => { try { p.kill(); } catch (_) { /* gone */ } }, timeoutMs);
      p.stdout.on('data', (d) => { o += d; }); p.stderr.on('data', (d) => { e += d; });
      p.on('close', (c) => {
        clearTimeout(t);
        const last = o.split(/\r?\n/).filter((l) => /^PSI-(DONE|ERROR)/.test(l)).pop() || '';
        const m = /^PSI-(DONE|ERROR) ?(.*)$/.exec(last);
        if (m && m[1] === 'ERROR') return reject(new Error(m[2]));
        if (m) { try { return resolve(JSON.parse(m[2])); } catch (_) { /* below */ } }
        reject(new Error((e.trim().split(/\r?\n/).pop() || `exit ${c}`).slice(0, 400)));
      });
    });
  }

  // One long job at a time (a download or a push), polled by the page; Stop ends it and deletes what was half-written.
  let job = null;
  const jobView = () => (job ? { id: job.id, kind: job.kind, label: job.label, state: job.state, stage: job.stage, done: job.done, of: job.of, result: job.result, error: job.error,
    started_at: job.started_at, seconds: Math.round(((job.ended || Date.now()) - job.t0) / 1000) } : null);
  function run(kind, script, args, label, cleanup) {
    if (job && job.state === 'running') throw new Error(`Hugging Face is busy (${job.label}); wait for it or stop it`);
    const j = job = { id: Date.now().toString(36), kind, label, state: 'running', stage: 'starting', done: 0, of: 0, result: null, error: null, started_at: new Date().toISOString(), t0: Date.now() };
    j.proc = spawn(PY(), ['-u', path.join(SRC, script), ...args], { cwd: SRC, windowsHide: true, env: env(true) });
    let buf = '', tail = [];
    const on = (d) => {
      buf += d; let i;
      while ((i = buf.search(/\r?\n/)) >= 0) {
        const l = buf.slice(0, i).trim(); buf = buf.slice(i + 1).replace(/^\n/, ''); if (!l) continue;
        const m = /^PSI-(PROGRESS|DONE|ERROR) ?(.*)$/.exec(l);
        if (m && m[1] === 'PROGRESS') { try { const p = JSON.parse(m[2]); j.stage = p.stage; j.done = p.done || 0; j.of = p.of || 0; } catch (_) { /* a line */ } }
        else if (m && m[1] === 'DONE') { try { j.result = JSON.parse(m[2]); } catch (_) { j.result = {}; } }
        else if (m && m[1] === 'ERROR') j.error = m[2];
        else if (!/it\/s|%\|/.test(l)) { tail.push(l.slice(0, 200)); tail = tail.slice(-3); }
      }
    };
    j.proc.stdout.on('data', on); j.proc.stderr.on('data', on);
    j.proc.on('close', (code) => {
      j.proc = null; j.ended = Date.now();
      j.state = j.cancelled ? 'cancelled' : j.result && !code ? 'done' : 'failed';
      if (j.state === 'failed' && !j.error) j.error = `stopped (exit ${code}): ${tail.join(' ')}`;
      if (j.state !== 'done' && cleanup) try { cleanup(); } catch (_) { /* best effort */ }
      log(`${j.label}: ${j.state}${j.error ? `: ${j.error}` : ''}`);
    });
    log(`${label}: started`);
    return jobView();
  }

  const repoArg = (v) => { const s = String(v || '').trim(); if (!REPO.test(s)) throw new Error('a repo is OWNER/NAME on Hugging Face'); return s; };
  const fileArg = (v) => { const s = String(v || ''); if (!FILE.test(s) || s.split('/').includes('..')) throw new Error(`not a file name in a repo: ${s}`); return s; };
  const slug = (repo) => repo.replace('/', '--');
  const modelsDir = () => { const d = config.settings.modelsDir; if (!d) throw new Error('set the models folder first (Settings > Local model): downloads go into it'); return d; };
  // A folder the user may push: on this computer, a model folder, never the app's private data folder.
  function folderArg(v) {
    const f = String(v || '');
    if (!f || !path.isAbsolute(f)) throw new Error('a model folder on this computer (a full path)');
    const r = path.resolve(f), data = path.resolve(config.DATA);
    if (r === data || r.startsWith(data + path.sep)) throw new Error('that folder is in the app’s private data folder');
    if (!fs.existsSync(path.join(r, 'config.json'))) throw new Error('not a model folder: no config.json in it');
    return r;
  }
  // Downloads are staged in <dest>.partial (src/hf_pull.py); a stopped or failed one leaves it, and it is deleted here.
  const rmPartial = (d) => () => fs.rmSync(`${d}.partial`, { recursive: true, force: true });

  async function call(a = {}) {
    switch (a.action) {
      case 'status': { const s = readState(); return { token: !!token(), ready: ready(), account: ready() ? s.name : null, orgs: ready() ? s.orgs || [] : [], can_write: ready() ? !!s.can_write : false, checked_at: s.checked_at || null, storage: secrets.mode(), ...prefs(), job: jobView() }; }
      case 'whoami': {
        if (!token()) throw new Error('no Hugging Face token saved');
        try { const w = await py('hf_push.py', ['whoami']); writeState({ ok: true, fp: fp(token()), name: w.name, orgs: w.orgs, can_write: w.can_write, checked_at: new Date().toISOString() }); return { ...w, ready: true }; }
        catch (e) { writeState({ ok: false, fp: fp(token()), error: e.message, checked_at: new Date().toISOString() }); throw e; }
      }
      case 'job': return jobView();
      case 'cancel': if (job && job.proc) { job.cancelled = true; try { job.proc.kill(); } catch (_) { /* gone */ } } return jobView();
    }
    needReady();
    switch (a.action) {
      // reads
      case 'search': return py('hf_pull.py', ['search', a.kind === 'dataset' ? 'dataset' : 'model', String(a.query || '').slice(0, 80), ...(a.all ? ['--all'] : []), '--limit', String(Math.max(1, Math.min(50, +a.limit || 20)))]);
      case 'info': return py('hf_pull.py', ['info', repoArg(a.repo), ...(a.rev ? ['--rev', String(a.rev).slice(0, 80)] : [])]);
      case 'family': return py('hf_pull.py', ['family', repoArg(a.repo)], 120000);
      case 'refs': return py('hf_pull.py', ['refs', repoArg(a.repo)]);
      case 'record': return py('hf_pull.py', ['record', repoArg(a.repo), fileArg(a.file)]);
      case 'gguf_base': { const f = String(a.file || ''); if (!/\.gguf$/i.test(f) || !path.isAbsolute(f) || !fs.existsSync(f)) throw new Error('a .gguf file on this computer'); return py('hf_pull.py', ['gguf_base', f], 120000); }
      // downloads
      case 'get_model': {
        const repo = repoArg(a.repo), dest = path.join(modelsDir(), 'hf', slug(repo));
        let files;
        if (a.gguf) files = [fileArg(a.gguf)];
        else { const i = await py('hf_pull.py', ['info', repo]); if (!i.folder) throw new Error(`${repo} has no safetensors weights folder`); if (i.remote_code) throw new Error(i.inside_engine); files = i.folder.files; }
        return run('model', 'hf_pull.py', ['get', repo, dest, '--files', files.join(',')], `downloading ${repo}${a.gguf ? ` (${a.gguf})` : ''}`, rmPartial(dest));
      }
      case 'get_adapter': { const repo = repoArg(a.repo), f = fileArg(a.file); if (!/\.gguf$/i.test(f)) throw new Error('adapters and control vectors are GGUF files'); const dest = path.join(modelsDir(), 'hf-adapters', slug(repo)); return run('adapter', 'hf_pull.py', ['get', repo, dest, '--files', f], `downloading ${f} from ${repo}`, rmPartial(dest)); }
      case 'get_sae': {   // one SAE folder of a repo (every file under it), to import in Features
        const repo = repoArg(a.repo), sub = fileArg(a.folder), dest = path.join(config.DATA, 'hf', 'sae', slug(repo));
        return run('sae', 'hf_pull.py', ['get', repo, dest, '--prefix', sub], `downloading the SAE ${sub} from ${repo}`, rmPartial(dest));
      }
      case 'get_ckpt': {
        const repo = repoArg(a.repo || prefs().checkpoints), step = Math.max(0, Math.round(+a.step));
        const root = (config.settings.trace && config.settings.trace.checkpoints) || path.join(modelsDir(), 'checkpoints', slug(repo));
        if (!(config.settings.trace && config.settings.trace.checkpoints)) { fs.mkdirSync(root, { recursive: true }); config.update({ trace: { checkpoints: root.replace(/\\/g, '/') } }, { trusted: true }); }
        return run('ckpt', 'hf_pull.py', ['ckpt', repo, String(step), root], `downloading step ${step} of ${repo}`, rmPartial(path.join(root, `step${step}`)));
      }
      case 'get_text': { const repo = repoArg(a.repo); return run('text', 'hf_pull.py', ['dataset', repo, path.join(config.DATA, 'hf', 'text', `${slug(repo)}.txt`), '--chars', String(Math.max(1000, Math.min(5e6, +a.chars || 200000)))], `reading text from ${repo}`); }
      case 'read_text': {   // the text get_text wrote, for Commons' training box
        const f = path.join(config.DATA, 'hf', 'text', `${slug(repoArg(a.repo))}.txt`); if (!fs.existsSync(f)) throw new Error('get the text first');
        return { text: fs.readFileSync(f, 'utf8').slice(0, 5e6) };
      }
      // push
      case 'plan': return py('hf_push.py', ['plan', folderArg(a.folder), repoArg(a.repo)]);
      case 'push': {
        if (a.confirm !== true) throw new Error('push: confirm first (the page shows the repo, the files and their size before it sends anything)');
        const folder = folderArg(a.folder), repo = repoArg(a.repo), priv = a.private !== false;
        const args = ['push', folder, repo, '--message', String(a.message || 'Upload from the Parameter Space Inspector').slice(0, 200)]; if (priv) args.push('--private');
        return run('push', 'hf_push.py', args, `pushing ${path.basename(folder)} to ${repo} (${priv ? 'private' : 'public'})`);
      }
      default: throw new Error(`hf: unknown action ${a.action}`);
    }
  }
  return { call, ready };
}

module.exports = { createHf };
