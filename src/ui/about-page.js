// SPDX-License-Identifier: Apache-2.0
// The About page's own script, kept in a file so the page can forbid inline script (its Content-Security-Policy).
(() => { const v = window.PSI_VERSION; const n = document.getElementById('ver');
  if (!v) { n.textContent = ''; return; }
  const d = new Date(v.date + 'T00:00:00'); n.textContent = `version ${v.version} · ${isNaN(d) ? v.date : d.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })}`; })();

// Back to the app: the manual opens beside the app (its own window in the desktop app, a new tab in the browser), so
// going back brings that app forward and closes this page, with the view asked for; opened on its own, it navigates.
(() => {
  document.querySelectorAll('[data-to-app]').forEach((a) => a.addEventListener('click', (e) => {
    const view = a.dataset.toApp; let app = null;
    try { app = window.opener && !window.opener.closed ? window.opener : null; if (app) void app.location.href; } catch { app = null; }
    if (!app) return;                                   // no app behind this page: follow the link (index.html)
    e.preventDefault();
    try { if (view) app.location.hash = view; app.focus(); } catch { /* the app window refused: closing still leaves it */ }
    window.close();
  }));
})();
