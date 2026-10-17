// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// The Python interpreter every engine and job runs: $PSI_PYTHON when set (a path or a name), else the first of
// python3 / python found on the PATH. macOS and many Linux systems have only "python3"; Windows installs "python"
// (its "python3" may be the Microsoft Store stub, so "python" is tried first there). Found once, then kept.
const fs = require('fs');
const path = require('path');

let found = null;
function python() {
  if (found) return found;
  if (process.env.PSI_PYTHON) return (found = process.env.PSI_PYTHON);
  const names = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
  const exts = process.platform === 'win32' ? ['.exe', ''] : [''];
  const dirs = String(process.env.PATH || '').split(path.delimiter).filter(Boolean).map((d) => d.replace(/^"|"$/g, ''));
  for (const n of names) for (const d of dirs) for (const e of exts) {
    const f = path.join(d, n + e);
    // The Windows Store stub (WindowsApps\python.exe) opens the Store instead of running: skipped.
    if (/WindowsApps/i.test(d)) continue;
    try { if (fs.statSync(f).isFile()) return (found = f); } catch (_) { /* next */ }
  }
  return (found = names[0]);   // not found: spawn reports ENOENT with this name, and the setup check says so
}

module.exports = { python };
