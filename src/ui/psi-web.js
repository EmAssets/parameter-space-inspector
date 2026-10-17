// The Electron bridge, re-created over HTTP for the dev server (src/devserver.js).
// Only present in a browser; the app itself gets window.psi from preload.js.
(() => {
  'use strict';
  if (window.psi) return;
  // The dev server's launch token, put in index.html as it is served; every call carries it (devserver.js).
  const token = (document.querySelector('meta[name="psi-token"]') || {}).content || '';
  const call = async (name, ...args) => {
    const r = await fetch(`/psi/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-PSI-Token': token }, body: JSON.stringify(args) }).then((x) => x.json());
    if (!r.ok) throw new Error(r.error);
    if (name === 'readHead' && r.value && r.value.bytes_b64) {
      const bin = atob(r.value.bytes_b64), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return { size: r.value.size, name: r.value.name, bytes: buf.buffer };
    }
    return r.value;
  };
  const NAMES = ['readHead', 'readAll', 'pickModel', 'pickControlVector', 'settings', 'start', 'stop', 'tokenize', 'detokenize', 'next', 'complete', 'score', 'template', 'engineState', 'trace', 'traceStatus', 'embed', 'represent', 'idle', 'quiet', 'scoreText', 'ablate', 'understand', 'ckptList', 'ckptAsk', 'ckptTest', 'heads', 'ablateHeads', 'lesion', 'sense', 'fight', 'connector', 'setKey', 'listModels', 'external', 'traceRestart', 'anatomy', 'mcp', 'neuronNames', 'setupCheck', 'ckptCount', 'lineage', 'ladder', 'gpu', 'convert', 'convertStatus', 'experiments', 'jobs', 'sae', 'vectors', 'timeline', 'commons', 'chatgpt', 'hf'];
  const api = Object.fromEntries(NAMES.map((n) => [n, (...a) => call(n, ...a)]));
  const listeners = { log: [], progress: [] };
  api.on = (channel, fn) => { (listeners[channel] || []).push(fn); return () => {}; };
  // The dev server has no push channel; the Read view's progress line stays quiet.
  window.psi = api;
  document.documentElement.classList.add('dev');
})();
