// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Commons: seed lines grown by many people (src/recipes.py, src/seedline.py, src/verify_lineage.py, src/keys.py).
// The view's server side: the profile and signing choice, personal keys, the Hugging Face token, and the runs
// (fetch a line, train a branch, check / build a recipe, verify, publish), one Python process at a time.
//
// Rules kept here, whatever the page sends:
//   - signing is never automatic: Python children sign only as identity.json says (PSI_SIGN_AS), private by default;
//     The Emergence Machine's key signs only through 'cosign_em', and only where its key file is
//   - anything that trains, builds, downloads large files, pushes or signs needs confirm: true (the view's own buttons;
//     the in-app helper cannot press anything in this view, helper.js)
//   - the Hugging Face token goes to Python only for a push or a private read; the page only learns whether it is set
//   - recipe text is data: it is parsed by Python, its fields are shown escaped, nothing in it runs
const { python: PY } = require('./python.js');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');

function createCommons({ config, secrets, log }) {
  const DIR = path.join(config.DATA, 'commons'), ID = path.join(DIR, 'identity.json'), TEXTS = path.join(DIR, 'texts'), PROFILES = path.join(DIR, 'profiles');
  const SRC = __dirname;
  const mk = (d) => fs.mkdirSync(d, { recursive: true });
  const readJson = (f, d = null) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (_) { return d; } };
  const writeJson = (f, o) => { mk(path.dirname(f)); fs.writeFileSync(f, JSON.stringify(o, null, 1), { mode: 0o600 }); };
  const canon = (o) => JSON.stringify(sortKeys(o));
  const sortKeys = (o) => (Array.isArray(o) ? o.map(sortKeys) : o && typeof o === 'object' ? Object.fromEntries(Object.keys(o).sort().map((k) => [k, sortKeys(o[k])])) : o);
  const str = (v, n = 200) => String(v == null ? '' : v).replace(/[\u0000-\u001f]/g, ' ').trim().slice(0, n);
  const hex = (v, n) => { const s = String(v || ''); if (!new RegExp(`^[0-9a-f]{${n}}$`).test(s)) throw new Error('not an id'); return s; };
  const hexPrefix = (v) => { const s = String(v || ''); if (!/^[0-9a-f]{8,64}$/.test(s)) throw new Error('not an id'); return s; };

  // ── identity: profile and signing choice ──
  const FIELDS = ['username', 'display_name', 'bio', 'avatar', 'links', 'orgs'];
  function identity() {
    const d = readJson(ID, {}) || {};
    const p = d.profile || {}; const s = d.signing || {};
    return { profile: { username: '', display_name: '', bio: '', avatar: '', links: [], orgs: [], anonymous: true, visibility: {}, ...p }, signing: { level: 'private', key: '', ...s } };
  }
  // The public snapshot: the fields marked public (an anonymous profile shows only its username). Its hash goes into records.
  function snapshot(p) {
    const vis = p.visibility || {}; const out = { username: p.username || 'anonymous' };
    if (!p.anonymous) for (const f of FIELDS.slice(1)) if (vis[f] !== 'private' && p[f] && (!Array.isArray(p[f]) || p[f].length)) out[f] = p[f];
    return out;
  }
  function snapshotSha(p) { const s = snapshot(p); const h = crypto.createHash('sha256').update(canon(s)).digest('hex'); writeJson(path.join(PROFILES, `${h}.json`), s); return h; }
  function cleanProfile(p) {
    const o = {};
    o.username = str(p.username, 40).replace(/[^\p{L}\p{N}_.-]/gu, '');
    o.display_name = str(p.display_name, 60); o.bio = str(p.bio, 500); o.avatar = /^https:\/\//.test(p.avatar || '') ? str(p.avatar, 300) : '';
    o.links = (Array.isArray(p.links) ? p.links : []).slice(0, 8).map((l) => ({ host: ['huggingface', 'github', 'web'].includes(l && l.host) ? l.host : 'web', user: str(l && l.user, 60), url: /^https:\/\//.test((l && l.url) || '') ? str(l.url, 300) : '' })).filter((l) => l.user || l.url);
    o.orgs = (Array.isArray(p.orgs) ? p.orgs : []).slice(0, 20).map((x) => str(x, 60)).filter(Boolean);
    o.anonymous = p.anonymous !== false;
    o.visibility = Object.fromEntries(FIELDS.map((f) => [f, (p.visibility || {})[f] === 'private' ? 'private' : 'public']));
    if (p.imported_from === 'huggingface') o.imported_from = 'huggingface';
    return o;
  }

  // Who every Python child acts as (config.setSigner, so placements, grafts, SAE rings and Grow follow the same choice),
  // and the key vault. Private: PSI_SIGN_AS is empty and nothing is signed.
  function signer() {
    const id = identity(); const e = {};
    const p = id.profile; e.PSI_MAKER = (p.anonymous ? p.username : p.display_name || p.username) || 'anonymous';
    e.PSI_PROFILE_SHA = snapshotSha(p);
    e.PSI_SIGN_AS = id.signing.level === 'private' ? '' : id.signing.key || '';
    const v = secrets.mode() !== 'os' && fs.existsSync(path.join(config.DATA, 'secrets', 'vault.bin')) ? 'locked' : secrets.vault();
    e.PSI_VAULT = v; e.PSI_VAULT_MODE = secrets.mode();
    return e;
  }
  const env = (extra = {}) => config.cacheEnv({ PSI_MODELS_DIR: config.settings.modelsDir || '', ...extra });

  // ── Python: quick answers, and one long run at a time ──
  function py(args, extra = {}, timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
      const p = spawn(PY(), ['-u', ...args], { cwd: SRC, windowsHide: true, env: env(extra) });
      let o = '', e = ''; const t = setTimeout(() => { try { p.kill(); } catch (_) { /* gone */ } }, timeoutMs);
      p.stdout.on('data', (d) => { o += d; }); p.stderr.on('data', (d) => { e += d; });
      p.on('close', (c) => {
        clearTimeout(t);
        const lines = o.split(/\r?\n/).filter(Boolean); const last = lines[lines.length - 1] || '';
        const m = /^PSI-(DONE|ERROR) ?(.*)$/.exec(last);
        if (m && m[1] === 'ERROR') return reject(new Error(m[2]));
        try { return resolve(JSON.parse(m ? m[2] : last)); } catch (_) { /* below */ }
        const err = /"error":\s*"([^"]+)"/.exec(o); reject(new Error(err ? err[1] : (e.trim().split(/\r?\n/).slice(-1)[0] || `exit ${c}`)));
      });
    });
  }
  let job = null;
  function run(kind, args, label, extra = {}) {
    if (job && !job.finished) throw new Error(`Something is already running here (${job.label}). Wait for it or stop it.`);
    job = { id: crypto.randomBytes(6).toString('hex'), kind, label, started: new Date().toISOString(), progress: null, done: null, error: null, lines: [], finished: false };
    const j = job;
    j.proc = spawn(PY(), ['-u', ...args], { cwd: SRC, windowsHide: true, env: env(extra) });
    let buf = '';
    const on = (d) => {
      buf += d; let i;
      while ((i = buf.search(/[\r\n]/)) >= 0) {
        const l = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!l) continue;
        const m = /^PSI-(PROGRESS|DONE|ERROR) ?(.*)$/.exec(l);
        if (m && m[1] === 'PROGRESS') { try { j.progress = JSON.parse(m[2]); } catch (_) { /* a line only */ } }
        else if (m && m[1] === 'DONE') { try { j.done = JSON.parse(m[2]); } catch (_) { j.done = {}; } }
        else if (m && m[1] === 'ERROR') j.error = m[2];
        else if (!/it\/s\]|%\||Warning|warn/i.test(l)) { j.lines.push(l.slice(0, 300)); if (j.lines.length > 30) j.lines.shift(); }
      }
    };
    j.proc.stdout.on('data', on); j.proc.stderr.on('data', on);
    j.proc.on('close', (code) => {
      j.finished = true; j.proc = null; j.ended = new Date().toISOString();
      if (code && !j.error) j.error = j.cancelled ? 'stopped' : `it stopped (exit ${code}): ${j.lines.slice(-2).join(' ')}`;
      log && log(j.error ? `${j.label}: ${j.error}` : `${j.label}: done`);
    });
    log && log(`${label}: started`);
    return status();
  }
  function status() { if (!job) return { running: false, job: null }; const { proc, ...j } = job; return { running: !job.finished, job: j }; }
  function cancel() { if (job && job.proc) { job.cancelled = true; try { job.proc.kill(); } catch (_) { /* gone */ } } return status(); }
  const need = (a, what) => { if (a.confirm !== true) throw new Error(`${what} starts only from the Commons view’s own button.`); };

  // ── actions ──
  async function overview() {
    const [line, ks] = await Promise.all([py(['recipes.py', 'line-status']).catch((e) => ({ error: e.message })), py(['keys.py', 'status']).catch((e) => ({ error: e.message, keys: [] }))]);
    const id = identity();
    return { line, keys: ks, identity: id, snapshot: snapshot(id.profile), hf_token: secrets.has('hf-token'), secrets_mode: secrets.mode(), secrets_where: secrets.where(), em_key_here: !!(ks && ks.em_key_here), job: status() };
  }
  function setProfile(a) { const id = identity(); id.profile = cleanProfile({ ...id.profile, ...(a.profile || {}) }); writeJson(ID, id); return identity(); }
  async function setSigning(a) {
    const id = identity(); const level = ['private', 'local', 'shared'].includes(a.level) ? a.level : 'private';
    let key = level === 'private' ? '' : String(a.key || id.signing.key || '');
    if (level !== 'private' && key === 'em') { if (!fs.existsSync(path.join(config.DATA, 'em-signing.key'))) throw new Error('The Emergence Machine’s key is not on this computer'); }
    else if (level !== 'private') { const ks = await py(['keys.py', 'list']); if (!(ks.keys || []).some((k) => k.id === key)) throw new Error('choose one of your keys first (or make one)'); }
    id.signing = { level, key }; writeJson(ID, id); return identity();
  }
  function textFile(a) {
    if (a.practice) return path.join(SRC, 'lines', 'stories260k-practice.json');
    const t = String(a.text || ''); if (t.length < 3000) throw new Error('paste at least a few pages of text (3,000 characters or more)');
    if (t.length > 20e6) throw new Error('larger than 20 MB of text: split it into several branches');
    mk(TEXTS); const f = path.join(TEXTS, `${crypto.createHash('sha256').update(t).digest('hex').slice(0, 16)}.txt`); fs.writeFileSync(f, t, { mode: 0o600 }); return f;
  }
  const idArg = (v) => hexPrefix(v);
  // Reading from Hugging Face (import, check, build, verify): the token, when one is saved, so the user's own private
  // repos can be read. hub.py sends it to huggingface.co only, and drops it on any redirect elsewhere.
  const readEnv = () => { const t = secrets.get('hf-token'); return t ? { PSI_HF_TOKEN: t } : {}; };
  const repoArg = (v) => { const s = String(v || ''); if (!/^[\w.-]+\/[\w.-]+$/.test(s)) throw new Error('a repo is OWNER/NAME'); return s; };
  const addrArg = (v) => { const s = str(v, 500); if (!/^(hf:\/\/|https:\/\/|store:\/\/)/.test(s) && !path.isAbsolute(s)) throw new Error('an address: hf://OWNER/REPO/recipe.json, an https:// link, or a file on this computer'); return s; };

  async function call(a = {}) {
    switch (a.action) {
      case 'overview': return overview();
      case 'status': return status();
      case 'cancel': return cancel();
      // identity
      case 'profile_set': return setProfile(a);
      case 'profile_import_hf': { const u = str(a.user, 96); const r = await py(['recipes.py', 'hf-user', u]); return { imported: r }; }
      case 'signing_set': return setSigning(a);
      case 'key_create': return py(['keys.py', 'create', str(a.name || 'my key', 60)]);
      case 'key_restore': return py(['keys.py', 'restore', str(a.name || 'restored key', 60), str(a.code, 200)]);
      case 'key_rotate': return py(['keys.py', 'rotate', hex(a.old, 16), hex(a.new, 16)]);
      case 'hf_token_set': { const t = String(a.token || '').trim(); if (!/^hf_[A-Za-z0-9]{20,}$/.test(t)) throw new Error('that does not look like a Hugging Face token (hf_…)'); secrets.set('hf-token', t); return { hf_token: true }; }
      case 'hf_token_clear': secrets.set('hf-token', ''); return { hf_token: false };
      // the line
      case 'line_fetch': need(a, 'Downloading the seed'); return run('line', ['recipes.py', 'line-fetch'], 'fetching stories260K');
      // contributing
      case 'train': {
        need(a, 'Training'); const f = textFile(a);
        const args = ['recipes.py', 'train', '--text', f, '--name', str(a.name || 'my branch', 80), '--steps', String(Math.max(20, Math.min(5000, Math.round(+a.steps || 200)))),
          '--licence', str(a.licence || (a.practice ? 'public domain' : ''), 120) || 'not stated', '--subject', str(a.subject, 60), '--about', str(a.about, 500)];
        if (a.url) args.push('--url', addrArg(a.url));
        if (a.base) args.push('--base', path.join(DIR, 'models', path.basename(String(a.base))));
        return run('train', args, `training ${str(a.name || 'a branch', 40)}`);
      }
      case 'branches': return py(['recipes.py', 'branches']);
      case 'recipes': return py(['recipes.py', 'recipes']);
      case 'models': return py(['recipes.py', 'models']);
      case 'plugins': return py(['recipes.py', 'plugins']);
      case 'plugin_approve': {
        need(a, 'Approving a plugin'); const h = hex(a.sha256, 64); const f = path.join(DIR, 'plugins', 'approved.json'); const d = readJson(f, { sha256: [] });
        if (!d.sha256.includes(h)) d.sha256.push(h); writeJson(f, d); return d;
      }
      // recipes
      case 'recipe_make': {
        need(a, 'Making a recipe'); const bs = (Array.isArray(a.branches) ? a.branches : []).map(idArg); if (bs.length < 1) throw new Error('choose at least one branch');
        const args = ['recipes.py', 'make', '--name', str(a.name || 'my recipe', 80), '--branches', bs.join(','), '--plugin', str(a.plugin || 'average', 40), '--about', str(a.about, 1000)];
        if (a.options) args.push('--options', JSON.stringify(a.options)); if (a.strengths) args.push('--strengths', JSON.stringify((a.strengths || []).map((x) => Math.max(0, Math.min(2, +x || 0)))));
        return run('make', args, `making ${str(a.name || 'a recipe', 40)}`);
      }
      case 'browse': return py(['recipes.py', 'browse', str(a.search, 80), '--sort', ['downloads', 'likes', 'trendingScore', 'lastModified', 'createdAt'].includes(a.sort) ? a.sort : 'downloads'], {}, 45000);
      case 'import': return run('import', ['recipes.py', 'import', addrArg(a.addr)], 'importing a recipe', readEnv());
      case 'check': return run('check', ['recipes.py', 'check', idArg(a.id)], 'checking a recipe', readEnv());
      case 'build': {
        need(a, 'Building'); const mode = ['strict', 'recover', 'partial', 'download'].includes(a.mode) ? a.mode : 'strict';
        const args = ['recipes.py', 'build', idArg(a.id), '--mode', mode]; if (a.gguf) args.push('--gguf');
        return run('build', args, `building a recipe (${mode})`, readEnv());
      }
      case 'search': need(a, 'A recipe search'); return run('search', ['recipes.py', 'search', '--branches', (a.branches || []).map(idArg).join(','), '--plugin', str(a.plugin || 'average', 40)], 'searching for the best branches');
      // the second lineage path: public Hugging Face models (tools/hf_recipe.py through recipes.py)
      case 'hf_find': { const b = String(a.base || ''); if (!/^[\w.-]+\/[\w.-]+$/.test(b)) throw new Error('a base is OWNER/NAME on Hugging Face'); return run('hf-find', ['recipes.py', 'hf-find', b], `finding models that merge with ${b}`); }
      case 'hf_make': {
        need(a, 'Making a recipe of Hugging Face models (it downloads them)');
        const b = String(a.base || ''), bs = (Array.isArray(a.branches) ? a.branches : []).map(String);
        if (![b, ...bs].every((x) => /^[\w.-]+\/[\w.-]+(@[0-9a-f]{40})?$/.test(x))) throw new Error('models are OWNER/NAME on Hugging Face');
        if (!bs.length || bs.length > 12) throw new Error('choose 1 to 12 models to merge into the base');
        const method = ['task_arithmetic', 'average', 'ties', 'weighted'].includes(a.method) ? a.method : 'task_arithmetic';
        const opts = {}; if (a.lam != null) opts.lam = Math.max(0, Math.min(2, +a.lam || 0)); if (method === 'ties') opts.keep = Math.max(0.01, Math.min(1, +a.keep || 0.2));
        const args = ['recipes.py', 'hf-make', '--base', b, '--branches', bs.join(','), '--plugin', method, '--options', JSON.stringify(opts), '--name', str(a.name || 'my merge', 80), '--about', str(a.about, 1000), '--kind', String(Math.max(0, Math.min(bs.length - 1, +a.carry || 0)))];
        if (a.strengths) args.push('--strengths', JSON.stringify(a.strengths.map((x) => Math.max(0, Math.min(2, +x || 0)))));
        return run('hf-make', args, `making ${str(a.name || 'a merge', 40)} from Hugging Face models`);
      }
      case 'bisect': need(a, 'Bisecting'); return run('bisect', ['recipes.py', 'bisect', idArg(a.id)], 'bisecting a recipe');
      case 'use': need(a, 'A use record'); return py(['recipes.py', 'use', idArg(a.id)]);
      // checking anyone's claims
      case 'verify': {
        const kind = ['branch', 'recipe', 'model'].includes(a.kind) ? a.kind : 'branch'; let target;
        if (kind === 'model' && !String(a.id).startsWith('hf-')) target = path.join(DIR, 'models', path.basename(String(a.id)));
        else if (kind === 'model' && String(a.id).startsWith('hf-')) target = path.join(DIR, 'models', path.basename(String(a.id)));
        else { const d = path.join(DIR, kind === 'branch' ? 'branches' : 'recipes'); const f = (fs.existsSync(d) ? fs.readdirSync(d) : []).find((x) => x.startsWith(idArg(a.id).slice(0, 16))); if (!f) throw new Error(`no ${kind} ${a.id} here`); target = path.join(d, f); }
        const args = ['verify_lineage.py', target, '--json', '--replay', String(Math.max(0, Math.min(20, +a.replay || 0)))]; if (a.online) args.push('--online');
        return run('verify', args, `verifying a ${kind}`, a.online ? readEnv() : {});
      }
      // publishing (outward: the user's token, the user's confirmation)
      case 'publish': {
        const what = ['recipe', 'branch', 'identity'].includes(a.what) ? a.what : null; if (!what) throw new Error('publish what? recipe, branch or identity');
        const args = ['recipes.py', 'publish', what === 'identity' ? '-' : idArg(a.id), '--what', what, '--repo', repoArg(a.repo)];
        // New repos follow Settings > Hugging Face > "new repos start private" (on by default), unless the call says.
        const priv = a.private != null ? a.private !== false : (((config.settings || {}).hf || {}).private !== false);
        if (priv) args.push('--private');
        if (a.dry_run) { args.push('--dry-run'); return run('publish', args, `preparing ${what} for ${a.repo}`); }
        need(a, 'Publishing'); const tok = secrets.get('hf-token'); if (!tok) throw new Error('add a Hugging Face token first (Settings > Hugging Face)');
        return run('publish', args, `publishing ${what} to ${a.repo}`, { PSI_HF_TOKEN: tok });
      }
      case 'cosign_em': need(a, 'Signing as The Emergence Machine'); return run('cosign', ['recipes.py', 'cosign', idArg(a.id), '--kind', a.kind === 'branch' ? 'branch' : 'recipe'], 'countersigning as The Emergence Machine');
      default: throw new Error(`commons: unknown action ${a.action}`);
    }
  }
  return { call, status, signer };
}

module.exports = { createCommons };
