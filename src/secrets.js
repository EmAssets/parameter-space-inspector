// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Secrets kept by the operating system wherever it can, so a copy of the data folder alone does not give them away.
//   os         the desktop app: Electron's safeStorage (Windows DPAPI, macOS Keychain, Linux libsecret/kwallet)
//   dpapi      the dev server on Windows: DPAPI for the current Windows user (through PowerShell), stored as <name>.dpapi
//   keychain   the dev server on macOS: the login Keychain (the `security` tool); <name>.keychain only marks it
//   libsecret  the dev server on Linux with a Secret Service (GNOME Keyring, KWallet; `secret-tool`); <name>.libsecret marks it
//   file       none of these works (or PSI_SECRETS=file): a 0600 file, <name>.txt, and mode() says so
// Each write is read back before it counts; a store that fails is dropped for the next, down to the file. A secret
// found in a plain file is moved into the store when one works, and the file is overwritten before it is deleted.
//   vault     the secret that encrypts the personal signing keys (src/keys.py), made once, handed to Python as PSI_VAULT
//   hf-token  the Hugging Face token, handed to Python as PSI_HF_TOKEN for reads from Hugging Face (imports, checks, downloads,
//             online verification: so private repos can be read) and for a push the user confirmed; hub.py sends it to
//             huggingface.co only
// Never returned to the page: the page learns only whether a secret is set, and where() it is kept.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const SERVICE = 'Parameter Space Inspector';
const WHERE = {
  os: 'encrypted by the operating system',
  dpapi: 'encrypted by Windows for your user account (DPAPI)',
  keychain: 'in your macOS Keychain',
  libsecret: 'in your system keyring (Secret Service)',
  file: 'in a private file in the data folder, not encrypted (no system keyring was available)',
};

function createSecrets({ dataDir, safeStorage = null }) {
  const DIR = path.join(dataDir, 'secrets');
  const p = (name, ext) => path.join(DIR, `${name}.${ext}`);
  // Keyring entries are per data folder, so two installs on one account do not share secrets.
  const account = (name) => `${name}@${crypto.createHash('sha256').update(path.resolve(DIR)).digest('hex').slice(0, 10)}`;
  const run = (cmd, args, input) => execFileSync(cmd, args, { input, encoding: 'utf8', timeout: 20000, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  const b64 = (s) => Buffer.from(String(s), 'utf8').toString('base64');
  const unb64 = (s) => Buffer.from(String(s).trim(), 'base64').toString('utf8');
  const write = (f, data) => { fs.mkdirSync(DIR, { recursive: true, mode: 0o700 }); fs.writeFileSync(f, data, { mode: 0o600 }); try { fs.chmodSync(f, 0o600); } catch (_) { /* Windows */ } };
  // Overwritten before it is deleted, so a plain copy does not linger in the free space (best effort: flash and
  // journalling file systems may still keep old blocks).
  const rm = (f) => { try { const n = fs.statSync(f).size; fs.writeFileSync(f, crypto.randomBytes(Math.max(n, 1))); fs.unlinkSync(f); } catch (_) { /* none */ } };

  const electron = () => {
    if (!(safeStorage && safeStorage.isEncryptionAvailable && safeStorage.isEncryptionAvailable())) return false;
    try { return !safeStorage.getSelectedStorageBackend || safeStorage.getSelectedStorageBackend() !== 'basic_text'; } catch (_) { return true; }
  };
  // The dev server's stores. Values pass through stdin and stdout only, base64, never on a command line.
  const PS = (body) => ['-NoProfile', '-NonInteractive', '-Command', `$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.Security; $e=[Text.Encoding]::UTF8.GetBytes('psi-secrets'); $i=[Convert]::FromBase64String([Console]::In.ReadToEnd().Trim()); ${body}`];
  const has = (cmd) => { try { run('sh', ['-c', `command -v ${cmd}`]); return true; } catch (_) { return false; } };
  const BACKENDS = {
    dpapi: {
      ok: () => process.platform === 'win32',
      put: (name, v) => write(p(name, 'dpapi'), run('powershell.exe', PS("[Console]::Out.Write([Convert]::ToBase64String([Security.Cryptography.ProtectedData]::Protect($i,$e,'CurrentUser')))"), b64(v)).trim()),
      get: (name) => fs.existsSync(p(name, 'dpapi')) ? unb64(run('powershell.exe', PS("[Console]::Out.Write([Convert]::ToBase64String([Security.Cryptography.ProtectedData]::Unprotect($i,$e,'CurrentUser')))"), fs.readFileSync(p(name, 'dpapi'), 'utf8'))) : null,
      del: (name) => rm(p(name, 'dpapi')),
    },
    keychain: {
      ok: () => process.platform === 'darwin' && fs.existsSync('/usr/bin/security'),
      // `security -i` reads the command from stdin; the value is base64, so it needs no quoting.
      put: (name, v) => { run('/usr/bin/security', ['-i'], `add-generic-password -U -s "${SERVICE}" -a "${account(name)}" -w "${b64(v)}"\n`); write(p(name, 'keychain'), `${SERVICE} / ${account(name)}\n`); },
      get: (name) => fs.existsSync(p(name, 'keychain')) ? unb64(run('/usr/bin/security', ['find-generic-password', '-s', SERVICE, '-a', account(name), '-w'])) : null,
      del: (name) => { if (fs.existsSync(p(name, 'keychain'))) { try { run('/usr/bin/security', ['delete-generic-password', '-s', SERVICE, '-a', account(name)]); } catch (_) { /* gone */ } } rm(p(name, 'keychain')); },
    },
    libsecret: {
      ok: () => process.platform === 'linux' && has('secret-tool'),
      put: (name, v) => { run('secret-tool', ['store', `--label=${SERVICE}: ${name}`, 'service', SERVICE, 'account', account(name)], b64(v)); write(p(name, 'libsecret'), `${SERVICE} / ${account(name)}\n`); },
      get: (name) => fs.existsSync(p(name, 'libsecret')) ? unb64(run('secret-tool', ['lookup', 'service', SERVICE, 'account', account(name)])) : null,
      del: (name) => { if (fs.existsSync(p(name, 'libsecret'))) { try { run('secret-tool', ['clear', 'service', SERVICE, 'account', account(name)]); } catch (_) { /* gone */ } } rm(p(name, 'libsecret')); },
    },
  };
  const okMemo = {};
  const ok = (k) => (okMemo[k] === undefined ? (okMemo[k] = BACKENDS[k].ok()) : okMemo[k]);

  const broken = new Set();
  function store() {   // the dev server's store: the first that works on this system and has not failed
    if (process.env.PSI_SECRETS === 'file') return null;
    return Object.keys(BACKENDS).find((k) => !broken.has(k) && ok(k)) || null;
  }
  function mode() { return electron() ? 'os' : store() || 'file'; }

  const cache = new Map();
  function get(name) {
    if (cache.has(name)) return cache.get(name);
    let v = '', from = null;
    try {
      if (electron() && fs.existsSync(p(name, 'bin'))) { v = safeStorage.decryptString(fs.readFileSync(p(name, 'bin'))); from = 'os'; }
      for (const k of Object.keys(BACKENDS)) if (!v && ok(k)) { try { v = BACKENDS[k].get(name) || ''; if (v) from = k; } catch (_) { /* unreadable here */ } }
      if (!v && fs.existsSync(p(name, 'txt'))) { v = fs.readFileSync(p(name, 'txt'), 'utf8').trim(); if (v) from = 'file'; }
    } catch (_) { v = ''; /* unreadable: treated as not set */ }
    if (v && from !== mode()) set(name, v);   // moved into the best store that works now
    if (v) cache.set(name, v);
    return v;
  }
  function clearAll(name, keep) {
    for (const k of Object.keys(BACKENDS)) if (k !== keep && (fs.existsSync(p(name, k)))) { try { BACKENDS[k].del(name); } catch (_) { /* none */ } }
    if (keep !== 'os') rm(p(name, 'bin'));
    if (keep !== 'file') rm(p(name, 'txt'));
  }
  function set(name, value) {
    cache.delete(name);
    if (!value) { clearAll(name, null); return; }
    value = String(value);
    if (electron()) { write(p(name, 'bin'), safeStorage.encryptString(value)); clearAll(name, 'os'); cache.set(name, value); return; }
    for (let k = store(); k; k = store()) {
      try { BACKENDS[k].put(name, value); if (BACKENDS[k].get(name) === value) { clearAll(name, k); cache.set(name, value); return; } } catch (_) { /* this store failed */ }
      try { BACKENDS[k].del(name); } catch (_) { /* none */ }
      broken.add(k);
    }
    write(p(name, 'txt'), value); clearAll(name, 'file'); cache.set(name, value);
  }
  // The vault secret: made on first use, never shown.
  function vault() { let v = get('vault'); if (!/^[0-9a-f]{64}$/.test(v)) { v = crypto.randomBytes(32).toString('hex'); set('vault', v); } return v; }
  return { get, set, vault, has: (n) => !!get(n), mode, where: () => WHERE[mode()] };
}

module.exports = { createSecrets };
