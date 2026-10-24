// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Updates for an app folder installed with git clone (INSTALL_PROMPT.md). Settings > Help and about:
//   check     asks the folder's upstream (the repository it was cloned from) for its newest commit: git ls-remote, a few
//             hundred bytes, nothing downloaded and no file changed. At startup only if the user turned it on
//             (settings.updates.check, off by default); the page runs it, never an agent.
//   download  git fetch: the new commits go into .git only; the app's files are unchanged. Says the new version, what
//             changed and whether it can be installed (a fast-forward, no local edits).
//   install   git merge --ff-only, then a restart. If package.json or package-lock.json changed, npm install runs after
//             the app has closed (Electron's own files may be among those replaced), then the app starts again.
// Only the page's own Settings calls this (main.js registers it outside the API that MCP and job plans reach;
// devserver.js likewise). Nothing is installed without the user's press.
'use strict';
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.join(__dirname, '..');

function createUpdater({ dataDir, log = () => {}, stopEngines = async () => {}, restart = null } = {}) {
  const fwd = ROOT.replace(/\\/g, '/');
  // safe.directory on the command line counts as protected config, so a folder owned by another account still works.
  const git = (args, { timeout = 30000 } = {}) => new Promise((resolve, reject) => {
    let p; try { p = spawn('git', ['-c', `safe.directory=${fwd}`, ...args], { cwd: ROOT, windowsHide: true, env: { ...process.env, GIT_TERMINAL_PROMPT: '0' } }); } catch (e) { return reject(e); }
    let o = '', e = ''; const t = setTimeout(() => { try { p.kill(); } catch (_) { /* gone */ } reject(new Error(`git ${args[0]} took longer than ${timeout / 1000} s`)); }, timeout);
    p.stdout.on('data', (d) => { o += d; }); p.stderr.on('data', (d) => { e += d; });
    p.on('error', (err) => { clearTimeout(t); reject(err.code === 'ENOENT' ? new Error('git is not installed or not on the PATH, so the app cannot update itself') : err); });
    p.on('close', (c) => { clearTimeout(t); if (c === 0) resolve(o.trim()); else { const err = new Error((e.trim().split(/\r?\n/).pop() || `git ${args[0]} failed (${c})`).slice(0, 300)); err.code = c; reject(err); } });
  });
  const ok = (args) => git(args).then(() => true, () => false);

  // Where updates come from: the branch this folder tracks (git clone sets it, e.g. origin/main).
  async function source() {
    if (!fs.existsSync(path.join(ROOT, '.git'))) throw new Error('This app folder is not a git clone, so it cannot update itself. Install a new copy to update.');
    let up; try { up = await git(['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}']); } catch (_) { throw new Error('This folder tracks no remote branch (git branch --set-upstream-to), so there is nowhere to check.'); }
    const i = up.indexOf('/'); const remote = up.slice(0, i), branch = up.slice(i + 1);
    const url = await git(['remote', 'get-url', remote]).catch(() => '');
    return { remote, branch, ref: up, url: url.replace(/\/\/[^@/]*@/, '//') };   // a user:token in the URL is never shown
  }
  const version = () => { try { return JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version; } catch (_) { return null; } };

  let last = null;   // the latest result, for the page to show again (status)
  const keep = (r) => { last = { ...r, at: new Date().toISOString() }; return last; };

  async function check() {
    const src = await source();
    const head = await git(['rev-parse', 'HEAD']);
    const line = await git(['ls-remote', src.remote, `refs/heads/${src.branch}`], { timeout: 20000 });
    const remote = (line.split(/\s+/)[0] || '').trim();
    if (!remote) throw new Error(`${src.ref} was not found on ${src.url || src.remote}`);
    // Up to date: the same commit, or one this folder already has in its history (a copy ahead of the published one).
    const have = remote === head || (await ok(['merge-base', '--is-ancestor', remote, 'HEAD']));
    return keep({ state: have ? 'current' : 'available', version: version(), head: head.slice(0, 7), remote: remote.slice(0, 7), source: src.url, branch: src.branch });
  }

  async function download() {
    const src = await source();
    log(`downloading ${src.ref} from ${src.url}`);
    await git(['fetch', '--no-tags', src.remote, src.branch], { timeout: 600000 });
    const target = src.ref;
    if (await ok(['merge-base', '--is-ancestor', target, 'HEAD'])) return keep({ state: 'current', version: version(), source: src.url });
    const files = (await git(['diff', '--name-only', 'HEAD', target])).split(/\r?\n/).filter(Boolean);
    let next = null; try { next = JSON.parse(await git(['show', `${target}:package.json`])).version; } catch (_) { /* unknown */ }
    const changes = (await git(['log', '--format=%s', '--max-count=30', `HEAD..${target}`])).split(/\r?\n/).filter(Boolean);
    const fastForward = await ok(['merge-base', '--is-ancestor', 'HEAD', target]);
    const edited = (await git(['status', '--porcelain', '--untracked-files=no'])).split(/\r?\n/).filter(Boolean).map((l) => l.replace(/^\s*\S{1,2}\s+/, ''));   // "XY path"; git() trimmed the first line's leading space
    return keep({ state: 'downloaded', version: version(), next, changes, files: files.length, node: files.some((f) => /^package(-lock)?\.json$/.test(f)),
      python: files.some((f) => /^requirements[\w-]*\.txt$/.test(f)), fastForward, edited, source: src.url });
  }

  // After the app closes: npm install (it may replace Electron itself), then start the app again. A detached shell that
  // waits a few seconds for this process to end; its output goes to <data>/update-npm.log.
  function npmAfterExit() {
    const logFile = path.join(dataDir || ROOT, 'update-npm.log');
    const out = fs.openSync(logFile, 'w');
    const win = process.platform === 'win32';
    const cmd = win ? 'ping -n 5 127.0.0.1 >nul & npm install & npm start' : 'sleep 4; npm install && npm start';
    const p = spawn(win ? 'cmd.exe' : 'sh', win ? ['/d', '/c', cmd] : ['-c', cmd], { cwd: ROOT, detached: true, windowsHide: true, stdio: ['ignore', out, out] });
    p.unref(); return logFile;
  }

  async function install() {
    const d = last && last.state === 'downloaded' ? last : await download();
    if (d.state === 'current') return d;
    if (d.edited && d.edited.length) throw new Error(`Files in the app folder have been edited (${d.edited.slice(0, 5).join(', ')}${d.edited.length > 5 ? ', …' : ''}). Undo or commit those edits, then install.`);
    if (!d.fastForward) throw new Error('This copy has its own commits, or the published history was replaced, so the update cannot be applied on top of it. Update by hand (git pull), or install a new copy.');
    const src = await source();
    await stopEngines();
    const from = await git(['rev-parse', '--short', 'HEAD']);
    await git(['merge', '--ff-only', src.ref], { timeout: 120000 });
    const to = await git(['rev-parse', '--short', 'HEAD']);
    log(`updated ${from} -> ${to} (version ${version()})`);
    const r = keep({ state: 'installed', from, to, version: version(), node: d.node, python: d.python, restart: restart ? 'app' : 'server' });
    if (restart) setTimeout(() => restart({ npm: d.node ? npmAfterExit : null }), 400);   // after the reply reaches the page
    return r;
  }

  return {
    async call({ action = 'status' } = {}) {
      if (action === 'status') return last || { state: 'unknown', version: version() };
      if (action === 'check') return check();
      if (action === 'download') return download();
      if (action === 'install') return install();
      throw new Error(`update: unknown action ${action}`);
    },
  };
}

module.exports = { createUpdater };
