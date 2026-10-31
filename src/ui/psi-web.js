// The Electron bridge, re-created over HTTP for the dev server (src/devserver.js).
// Only present in a browser; the app itself gets window.psi from preload.js.
(() => {
  'use strict';
  if (window.psi) return;
  // The dev server's launch token, put in index.html as it is served; every call carries it (devserver.js).
  const token = (document.querySelector('meta[name="psi-token"]') || {}).content || '';
  // Once the server has refused the page's token, every later call would be refused too (the page's pollers would log a
  // 403 each second): say so once, in a bar with a Reload button, and stop calling.
  const STALE = 'This page is stale: the server was restarted since it loaded. Reload the page and try again.';
  let stale = false;
  const markStale = () => {
    if (stale) return; stale = true;
    const bar = document.createElement('div'); bar.className = 'psi-stale'; bar.setAttribute('role', 'alert');
    bar.textContent = 'The server was restarted since this page loaded. ';
    const b = document.createElement('button'); b.type = 'button'; b.textContent = 'Reload'; b.addEventListener('click', () => location.reload());
    bar.appendChild(b); document.body.appendChild(bar);
  };
  const call = async (name, ...args) => {
    if (stale) throw new Error(STALE);
    const x = await fetch(`/psi/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-PSI-Token': token }, body: JSON.stringify(args) });
    // A refusal comes as plain text, not JSON: above all 403 when the page's token is stale (the dev server was restarted
    // and minted a new one), which only a reload fixes. Say that instead of a JSON parse error.
    const txt = await x.text(); let r;
    try { r = JSON.parse(txt); } catch (_) { if (x.status === 403 && /wrong token/i.test(txt)) { markStale(); throw new Error(STALE); } throw new Error(`${x.status} ${x.statusText}: ${txt.slice(0, 200)}`); }
    if (!r.ok) throw new Error(r.error);
    if (name === 'readHead' && r.value && r.value.bytes_b64) {
      const bin = atob(r.value.bytes_b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return { size: r.value.size, name: r.value.name, bytes: buf.buffer };
    }
    return r.value;
  };
  const NAMES = ['readHead', 'readAll', 'pickModel', 'pickControlVector', 'settings', 'start', 'stop', 'tokenize', 'detokenize', 'next', 'complete', 'score', 'template', 'engineState', 'trace', 'traceStatus', 'embed', 'represent', 'idle', 'quiet', 'scoreText', 'ablate', 'understand', 'ckptList', 'ckptAsk', 'ckptTest', 'heads', 'ablateHeads', 'lesion', 'sense', 'fight', 'connector', 'setKey', 'listModels', 'external', 'traceRestart', 'anatomy', 'mcp', 'neuronNames', 'setupCheck', 'ckptCount', 'lineage', 'ladder', 'gpu', 'convert', 'convertStatus', 'experiments', 'jobs', 'sae', 'vectors', 'override', 'context', 'attractor', 'timeline', 'commons', 'hybrid', 'chatgpt', 'hf', 'update', 'session', 'measure'];
  const api = Object.fromEntries(NAMES.map((n) => [n, (...a) => call(n, ...a)]));
  const listeners = { log: [], progress: [] };
  api.on = (channel, fn) => { (listeners[channel] || []).push(fn); return () => {}; };
  // The dev server has no push channel; the Read view's progress line stays quiet.
  window.psi = api;
  document.documentElement.classList.add('dev');
})();
