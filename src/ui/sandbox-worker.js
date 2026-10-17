// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The analysis sandbox: code written by the user or by Prometheus runs here, over an experiment's data, and can only
// compute and show. It runs in a Web Worker (no page, no DOM, no app bridge: window.psi does not exist here), started
// fresh for each run and ended by the page after a time limit. Before any code runs, this file removes every way out:
//   - no network or storage: fetch, XMLHttpRequest, WebSocket, EventSource, importScripts, workers, channels,
//     IndexedDB, caches, WebAssembly and the rest are deleted from the global object and its prototypes;
//   - no new code: eval, the Function constructors (reached through any function's .constructor) and string timers
//     are removed once the code is compiled; the word “import” is refused (import() would load a module);
//   - one way back: the result goes to the page by the reply this file sends, as data the page shows as text.
// The dev server also serves this file with a Content-Security-Policy of its own (no connections at all).
// Randomness is seeded (rand(seed), and Math.random is a seeded generator) so an analysis gives the same answer twice.
'use strict';
(() => {
  const post = self.postMessage.bind(self);
  const compile = Function;   // kept for the one compile below, then removed with the rest
  const BAN = ['fetch', 'XMLHttpRequest', 'WebSocket', 'WebSocketStream', 'EventSource', 'importScripts', 'Worker', 'SharedWorker', 'BroadcastChannel', 'MessageChannel',
    'indexedDB', 'caches', 'WebTransport', 'RTCPeerConnection', 'WebAssembly', 'navigator', 'FileReader', 'FileReaderSync', 'Request', 'Response', 'Headers',
    'webkitRequestFileSystem', 'webkitRequestFileSystemSync', 'webkitResolveLocalFileSystemURL', 'webkitResolveLocalFileSystemSyncURL', 'CacheStorage', 'IDBFactory',
    'location', 'origin', 'Notification', 'StorageManager', 'LockManager', 'FileSystemHandle', 'FileSystemFileHandle', 'FileSystemDirectoryHandle'];
  function lockDown() {
    for (let o = self; o && o !== Object.prototype; o = Object.getPrototypeOf(o)) {
      for (const k of BAN) { try { if (Object.prototype.hasOwnProperty.call(o, k)) delete o[k]; } catch (_) { /* not configurable */ } try { if (k in o) Object.defineProperty(o, k, { value: undefined, configurable: false, writable: false }); } catch (_) { /* fixed */ } }
    }
    // No code from strings: eval, the Function constructors, string timers.
    const kill = (proto) => { try { Object.defineProperty(proto, 'constructor', { value: undefined, configurable: false, writable: false }); } catch (_) { /* fixed */ } };
    kill(Function.prototype); kill(Object.getPrototypeOf(async () => {})); kill(Object.getPrototypeOf(function* () {})); kill(Object.getPrototypeOf(async function* () {}));
    for (const k of ['eval', 'Function']) { try { Object.defineProperty(self, k, { value: undefined, configurable: false, writable: false }); } catch (_) { /* fixed */ } }
    const st = self.setTimeout, si = self.setInterval;
    self.setTimeout = (f, ...a) => (typeof f === 'function' ? st(f, ...a) : 0); self.setInterval = (f, ...a) => (typeof f === 'function' ? si(f, ...a) : 0);
    self.postMessage = undefined; self.onmessage = null;
  }

  // ── seeded randomness ──
  const rng = (seed) => { let a = (seed >>> 0) || 0x9e3779b9; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };

  // ── statistics ──
  const num = (a) => a.filter((x) => typeof x === 'number' && Number.isFinite(x));
  const sum = (a) => num(a).reduce((s, x) => s + x, 0);
  const mean = (a) => { const b = num(a); return b.length ? sum(b) / b.length : null; };
  const variance = (a) => { const b = num(a), m = mean(b); return b.length > 1 ? b.reduce((s, x) => s + (x - m) ** 2, 0) / (b.length - 1) : null; };
  const sd = (a) => { const v = variance(a); return v == null ? null : Math.sqrt(v); };
  const quantile = (a, q) => { const b = num(a).sort((x, y) => x - y); if (!b.length) return null; const i = (b.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };
  const median = (a) => quantile(a, 0.5);
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const k of c) s += k / ++y; return -t + Math.log(2.5066282746310005 * s / x);
  }
  function betacf(a, b, x) {
    const FP = 1e-300; let c = 1, d = 1 - (a + b) * x / (a + 1); if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
      aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break;
    }
    return h;
  }
  const ibeta = (a, b, x) => { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; };
  const tP = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));   // two-sided p of Student's t
  const tCrit = (df, conf = 0.95) => { let lo = 0, hi = 100; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (tP(m, df) > 1 - conf) lo = m; else hi = m; } return (lo + hi) / 2; };
  function ci(a, conf = 0.95) { const b = num(a), n = b.length; if (n < 2) return { mean: mean(b), lo: null, hi: null, n }; const m = mean(b), h = tCrit(n - 1, conf) * sd(b) / Math.sqrt(n); return { mean: m, lo: m - h, hi: m + h, n, conf }; }
  function ttest(a, b) {   // Welch's t-test (unequal variances)
    const x = num(a), y = num(b); if (x.length < 2 || y.length < 2) return { t: null, df: null, p: null, note: 'each group needs at least 2 values' };
    const vx = variance(x) / x.length, vy = variance(y) / y.length, t = (mean(x) - mean(y)) / Math.sqrt(vx + vy);
    const df = (vx + vy) ** 2 / (vx ** 2 / (x.length - 1) + vy ** 2 / (y.length - 1)); return { t, df, p: tP(t, df), diff: mean(x) - mean(y) };
  }
  function pairedT(a, b) { const d = a.map((x, i) => x - b[i]).filter((v) => Number.isFinite(v)); if (d.length < 2) return { t: null, p: null, n: d.length }; const t = mean(d) / (sd(d) / Math.sqrt(d.length)); return { t, df: d.length - 1, p: tP(t, d.length - 1), mean_diff: mean(d), n: d.length }; }
  function permutationTest(a, b, { n = 5000, seed = 1, stat = (x, y) => mean(x) - mean(y) } = {}) {
    const x = num(a), y = num(b), all = x.concat(y), r = rng(seed), obs = stat(x, y); let k = 0;
    for (let i = 0; i < n; i++) { for (let j = all.length - 1; j > 0; j--) { const q = Math.floor(r() * (j + 1)); [all[j], all[q]] = [all[q], all[j]]; } if (Math.abs(stat(all.slice(0, x.length), all.slice(x.length))) >= Math.abs(obs) - 1e-12) k++; }
    return { observed: obs, p: (k + 1) / (n + 1), permutations: n };
  }
  function bootstrapCI(a, fn = mean, { n = 2000, conf = 0.95, seed = 1 } = {}) {
    const x = num(a), r = rng(seed), out = []; if (!x.length) return { estimate: null, lo: null, hi: null };
    for (let i = 0; i < n; i++) { const s = x.map(() => x[Math.floor(r() * x.length)]); out.push(fn(s)); }
    return { estimate: fn(x), lo: quantile(out, (1 - conf) / 2), hi: quantile(out, 1 - (1 - conf) / 2), resamples: n };
  }
  const cohenD = (a, b) => { const x = num(a), y = num(b); const sp = Math.sqrt(((x.length - 1) * variance(x) + (y.length - 1) * variance(y)) / (x.length + y.length - 2)); return sp ? (mean(x) - mean(y)) / sp : null; };
  function corr(a, b) { const p = a.map((x, i) => [x, b[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)); if (p.length < 3) return null; const mx = mean(p.map((q) => q[0])), my = mean(p.map((q) => q[1])); let sxy = 0, sxx = 0, syy = 0; for (const [x, y] of p) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2; } return sxy / Math.sqrt(sxx * syy); }
  const ranks = (a) => { const o = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]), r = new Array(a.length); for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
  const spearman = (a, b) => corr(ranks(a), ranks(b));
  function linreg(x, y) { const p = x.map((v, i) => [v, y[i]]).filter(([u, w]) => Number.isFinite(u) && Number.isFinite(w)); const mx = mean(p.map((q) => q[0])), my = mean(p.map((q) => q[1])); let sxy = 0, sxx = 0; for (const [u, w] of p) { sxy += (u - mx) * (w - my); sxx += (u - mx) ** 2; } const slope = sxy / sxx, r = corr(p.map((q) => q[0]), p.map((q) => q[1])); return { slope, intercept: my - slope * mx, r2: r == null ? null : r * r, n: p.length }; }
  // Rows: groups and pivots over the data table (an array of objects).
  function groupBy(rows, keys) { const ks = [].concat(keys), m = new Map(); for (const r of rows) { const k = ks.map((x) => r[x]).join(' · '); if (!m.has(k)) m.set(k, []); m.get(k).push(r); } return Object.fromEntries(m); }
  function summarize(rows, keys, field = 'value') { return Object.entries(groupBy(rows, keys)).map(([k, g]) => { const v = g.map((r) => r[field]); const c = ci(v); return { group: k, n: num(v).length, mean: c.mean, sd: sd(v), ci_lo: c.lo, ci_hi: c.hi }; }); }
  function pivot(rows, rowKey, colKey, field = 'value', agg = mean) { const out = {}; for (const [k, g] of Object.entries(groupBy(rows, rowKey))) { out[k] = {}; for (const [c, h] of Object.entries(groupBy(g, colKey))) out[k][c] = agg(h.map((r) => r[field])); } return out; }
  // Vectors and matrices (arrays of numbers / arrays of rows).
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a));
  const cosine = (a, b) => { const d = norm(a) * norm(b); return d ? dot(a, b) / d : null; };
  const center = (X) => { const d = X[0].length, m = new Array(d).fill(0); for (const r of X) for (let j = 0; j < d; j++) m[j] += r[j] / X.length; return X.map((r) => Array.from(r, (v, j) => v - m[j])); };
  const gram = (X) => X.map((a) => X.map((b) => dot(a, b)));
  function cka(X, Y) {   // linear CKA (Kornblith et al. 2019) between two sets of representations of the same n inputs
    if (X.length !== Y.length || X.length < 3) return null; const K = gram(center(X)), L = gram(center(Y));
    let kl = 0, kk = 0, ll = 0; for (let i = 0; i < K.length; i++) for (let j = 0; j < K.length; j++) { kl += K[i][j] * L[i][j]; kk += K[i][j] ** 2; ll += L[i][j] ** 2; } return kl / Math.sqrt(kk * ll);
  }
  function pca(X, k = 2) {   // through the n×n Gram matrix (n prompts is small): the first k components and the share of variance each holds
    const C = center(X), G = gram(C), n = G.length, comps = [], vals = []; let tr = 0; for (let i = 0; i < n; i++) tr += G[i][i];
    const M = G.map((r) => r.slice());
    for (let c = 0; c < Math.min(k, n); c++) {
      let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1 + c)), lam = 0;
      for (let it = 0; it < 300; it++) { const w = M.map((r) => dot(r, v)); const nw = norm(w); if (!nw) break; v = w.map((x) => x / nw); lam = nw; }
      vals.push(lam); comps.push(v.map((x) => x * Math.sqrt(lam))); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) M[i][j] -= lam * v[i] * v[j];
    }
    return { coords: Array.from({ length: n }, (_, i) => comps.map((c) => c[i])), explained: vals.map((l) => (tr ? l / tr : null)) };
  }
  // A linear probe: logistic regression (L2), scored by stratified k-fold cross-validation. labels: any two or more values.
  function probe(X, labels, { folds = 5, l2 = 1e-2, epochs = 300, lr = 0.1, permutations = 0, seed = 1 } = {}) {
    const n = X.length, cls = [...new Set(labels)]; if (n < 4 || cls.length < 2) return { accuracy: null, note: 'a probe needs at least 4 examples of at least 2 kinds' };
    const d = X[0].length, y = labels.map((l) => cls.indexOf(l));
    const fit = (idx) => {   // one-vs-rest weights, features standardized on the training part
      const mu = new Array(d).fill(0), sg = new Array(d).fill(0); for (const i of idx) for (let j = 0; j < d; j++) mu[j] += X[i][j] / idx.length;
      for (const i of idx) for (let j = 0; j < d; j++) sg[j] += (X[i][j] - mu[j]) ** 2 / idx.length; for (let j = 0; j < d; j++) sg[j] = Math.sqrt(sg[j]) || 1;
      const Z = (x) => Array.from(x, (v, j) => (v - mu[j]) / sg[j]), Zi = idx.map((i) => Z(X[i]));
      const W = cls.map((_, c) => { const w = new Array(d).fill(0); let b = 0; for (let e = 0; e < epochs; e++) { const gw = new Array(d).fill(0); let gb = 0; Zi.forEach((z, t) => { const p = 1 / (1 + Math.exp(-(dot(w, z) + b))), g = p - (yy[idx[t]] === c ? 1 : 0); for (let j = 0; j < d; j++) gw[j] += g * z[j] / Zi.length; gb += g / Zi.length; }); for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] + l2 * w[j]); b -= lr * gb; } return { w, b }; });
      return (x) => { const z = Z(x); let best = 0, bs = -Infinity; W.forEach((m, c) => { const s = dot(m.w, z) + m.b; if (s > bs) { bs = s; best = c; } }); return best; };
    };
    let yy = y;
    const cv = () => { const order = [...Array(n).keys()].sort((a, b) => yy[a] - yy[b] || a - b), fold = new Array(n); order.forEach((i, r) => { fold[i] = r % folds; }); let ok = 0; for (let f = 0; f < folds; f++) { const tr = [...Array(n).keys()].filter((i) => fold[i] !== f), te = [...Array(n).keys()].filter((i) => fold[i] === f); if (!te.length) continue; const pr = fit(tr); te.forEach((i) => { if (pr(X[i]) === yy[i]) ok++; }); } return ok / n; };
    const acc = cv(), counts = cls.map((_, c) => y.filter((v) => v === c).length), chance = Math.max(...counts) / n;
    let p = null; if (permutations > 0) { const r = rng(seed); let k = 0; for (let t = 0; t < permutations; t++) { yy = y.slice(); for (let j = n - 1; j > 0; j--) { const q = Math.floor(r() * (j + 1)); [yy[j], yy[q]] = [yy[q], yy[j]]; } if (cv() >= acc - 1e-12) k++; } yy = y; p = (k + 1) / (permutations + 1); }
    return { accuracy: acc, chance, classes: cls, n, folds, permutation_p: p };
  }
  const stats = { sum, mean, variance, sd, median, quantile, ci, ttest, pairedT, permutationTest, bootstrapCI, cohenD, corr, spearman, linreg, groupBy, summarize, pivot, dot, norm, cosine, cka, pca, probe };

  // ── running ──
  self.onmessage = (ev) => {
    const { code, input, seed, codeChars } = ev.data || {}; const out = [], t0 = Date.now(), maxCode = Math.min(Math.max(+codeChars || 20000, 1000), 200000);
    const cap = (s, n) => String(s).slice(0, n);
    const show = {
      text: (t) => { out.push({ kind: 'text', text: cap(t, 4000) }); },
      number: (label, value, note) => { out.push({ kind: 'number', label: cap(label, 120), value: typeof value === 'number' ? value : value == null ? null : cap(value, 200), note: note == null ? '' : cap(note, 300) }); },
      table: (title, rows) => { const r = Array.isArray(rows) ? rows : Object.entries(rows || {}).map(([k, v]) => (v && typeof v === 'object' ? { key: k, ...v } : { key: k, value: v })); out.push({ kind: 'table', title: cap(title, 160), rows: JSON.parse(JSON.stringify(r.slice(0, 500))), total: r.length }); },
      chart: (spec) => { out.push({ kind: 'chart', spec: JSON.parse(JSON.stringify(spec || {})) }); },
    };
    try {
      if (typeof code !== 'string' || !code.trim()) throw new Error('No code to run.');
      if (code.length > maxCode) throw new Error(`The code is longer than ${maxCode.toLocaleString()} characters (Settings > Guardrails).`);
      if (/\bimport\b/.test(code)) throw new Error('The word “import” is not allowed here (nothing can be loaded into the sandbox; everything it needs is given).');
      const fn = compile('data', 'experiments', 'vectors', 'stats', 'show', 'rand', `'use strict';\n${code}`);
      lockDown();
      Math.random = rng(seed || 1);
      // The vectors come as base64 float32 per prompt; here each prompt's layers are arrays of numbers.
      const vecs = (input.vectors || []).map((v) => { const bin = atob(v.b64), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); const f = new Float32Array(u.buffer); const layers = []; for (let l = 0; l < v.layers; l++) layers.push(Array.from(f.subarray(l * v.dims, (l + 1) * v.dims))); const { b64, ...rest } = v; return { ...rest, layers }; });
      const value = fn(input.data || [], input.experiments || [], vecs, stats, show, (s) => rng(s));
      let ret = null; if (value !== undefined) { try { ret = JSON.parse(JSON.stringify(value, (k, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v))); } catch (_) { ret = String(value); } }
      const txt = JSON.stringify(ret); if (txt && txt.length > 200000) ret = `${txt.slice(0, 200000)}… (cut)`;
      post({ ok: true, outputs: out.slice(0, 60), value: ret, ms: Date.now() - t0 });
    } catch (e) { post({ ok: false, outputs: out.slice(0, 60), error: `${e && e.name ? `${e.name}: ` : ''}${e && e.message ? e.message : e}`.slice(0, 1000), ms: Date.now() - t0 }); }
  };
})();
