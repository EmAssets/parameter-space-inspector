// SPDX-License-Identifier: Apache-2.0
// Writes src/ui/version.js from package.json, so the About page and Settings always show the real version.
// Runs automatically: on `npm version patch|minor|major` (the "version" script in package.json) and at app
// start (main.js, devserver.js). Never fails the caller: a read-only install keeps the file it shipped with.
'use strict';
const fs = require('fs');
const path = require('path');

function writeVersion() {
  const root = path.join(__dirname, '..');
  try {
    const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
    const out = path.join(root, 'src', 'ui', 'version.js');
    let prev = ''; try { prev = fs.readFileSync(out, 'utf8'); } catch (_) { /* first time */ }
    const m = prev.match(/"version":\s*"([^"]+)"[^}]*"date":\s*"([^"]+)"/);
    if (m && m[1] === pkg.version) return pkg.version;               // unchanged: keep the date it was released
    const info = { version: pkg.version, date: new Date().toISOString().slice(0, 10) };
    fs.writeFileSync(out, `// Generated from package.json by tools/write-version.js. Do not edit.\nwindow.PSI_VERSION = ${JSON.stringify(info)};\n`);
    return pkg.version;
  } catch (_) { return null; }
}

module.exports = { writeVersion };
if (require.main === module) { const v = writeVersion(); console.log(v ? `version.js: ${v}` : 'version.js: not written'); }
