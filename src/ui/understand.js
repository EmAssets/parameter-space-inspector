// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Understanding tests (the live battery, understanding-tests.js): questions next-token training never posed, scored against a
// surface-plausible alternative, with the layer at which the model "gets it".

// Helpers shared by the experiment views (Understanding, Emergence, Tipping point, Sensing surface); this file loads first of them.
//   PSIExp.empty(canvas, height, text)   a dark canvas with one centred message (wrapped when narrow), and its aria-label
//   PSIExp.inside()                      {name, layers} of the inside engine's model, as far as known
//   PSIExp.setLayers(n)                  a run reported the layer count: remember it and refill the copy
//   [data-inside] spans in index.html    filled with " (name, n layers)" when known, left empty otherwise
//   PSIExp.put(node, ...parts)           replace a node's content with strings and nodes (for PSI.term inside sentences)
//   PSIExp.setChanged(label|null)        the inside engine's model is cut (Tipping point) or intact again; fires 'psi-changed'
//   PSIExp.changed()                     that label, or null while the model is as loaded
//   PSIExp.mark(node)                    append "on the changed model" to a result line while the model is changed
//   .chg-banner at the top of every view that reads the inside engine's model, shown while it is changed
window.PSIExp = (() => {
  'use strict';
  const info = { name: '', layers: 0 };
  const pretty = (d) => (String(d || '').split(/[\\/]/).filter(Boolean).pop() || '').replace(/[-_](bf16|fp16|fp32|f16|f32|hf)$/i, '');
  const label = () => (info.name ? `${info.name}${info.layers ? `, ${info.layers} layers` : ''}` : '');
  function paint() { const l = label(); document.querySelectorAll('[data-inside]').forEach((n) => { n.textContent = l ? ` (${l})` : ''; }); }
  // Reads the settings and pings the engine if it is already up: never starts anything.
  let timer = 0;
  async function refresh() {
    let name = '';
    try { if (PSI.state.engine) { const s = await psi.settings(); name = pretty(s.trace && s.trace.modelDir); } } catch (_) { /* keep */ }
    if (name !== info.name) {
      const from = info.name; info.name = name; info.layers = 0;
      // The inside engine's model was switched in Settings: views archive results from other weights (never mixed).
      if (name) document.dispatchEvent(new CustomEvent('psi-inside-model', { detail: { from, to: name } }));
    }
    try { const t = await psi.traceStatus(); if (name && t && t.layers) info.layers = t.layers; } catch (_) { /* not running */ }
    paint();
  }
  const soon = () => { clearTimeout(timer); timer = setTimeout(() => refresh().catch(() => {}), 300); };
  if (window.PSI && PSI.onState) PSI.onState(soon);
  document.addEventListener('psi-model', soon);

  function empty(cv, h, text) {
    if (!cv) return;
    const r = Math.min(2, devicePixelRatio || 1), w = cv.clientWidth; cv.style.height = `${h}px`;
    if (window.PSI) PSI.canvasLabel(cv, text);
    if (!w) return;
    cv.width = w * r; cv.height = h * r;
    const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height);
    g.font = `${11 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = '#a9a69f'; g.textAlign = 'center'; g.textBaseline = 'middle';
    const max = cv.width - 24 * r, lines = []; let cur = '';
    String(text).split(' ').forEach((wd) => { const t = cur ? `${cur} ${wd}` : wd; if (cur && g.measureText(t).width > max) { lines.push(cur); cur = wd; } else cur = t; });
    if (cur) lines.push(cur);
    const lh = 16 * r, y0 = cv.height / 2 - ((lines.length - 1) * lh) / 2;
    lines.forEach((l, i) => g.fillText(l, cv.width / 2, y0 + i * lh));
  }
  const put = (n, ...parts) => { if (!n) return; n.textContent = ''; parts.forEach((p) => { if (p != null && p !== '') n.append(p); }); };

  // A cut made in Tipping point stays in the inside engine's model, which these views also read: say so on them.
  // Known only from this window's own requests (the engine has no "what is cut" query); a restart of the engine undoes it.
  // Kept in step with PSI.setChange('lesion', …) (psi-ui.js: the header chip), which also sees cuts this file does not make.
  let own = null;
  const VIEWS = ['understand', 'emergence', 'sensing', 'fight', 'connectors', 'anatomy'];
  const changed = () => {
    if (window.PSI && typeof PSI.changesFor === 'function') { const c = PSI.changesFor('inside'); return c.length ? c.map((x) => x.label).join('; ') : null; }
    return own;
  };
  const restoreLink = () => (window.PSI && PSI.link ? PSI.link('Restore it in Tipping point', 'lesion', 'lsRestore') : '');
  function banners() {
    const now = changed();
    VIEWS.forEach((v) => {
      const sec = document.getElementById(`tab-${v}`); if (!sec) return;
      let b = sec.querySelector(':scope > .chg-banner');
      if (!now) { if (b) b.remove(); return; }
      if (!b) { b = document.createElement('p'); b.className = 'chg-banner'; b.setAttribute('role', 'status'); sec.prepend(b); }
      put(b, `The inside engine’s model is changed (${now}). Live results in this view come from the changed model, not the model as loaded. `, restoreLink());
    });
  }
  function setChanged(label) {
    own = label || null;
    if (window.PSI && typeof PSI.setChange === 'function') PSI.setChange('lesion', own ? { label: `Tipping-point cut: ${own}`, short: 'cut', engine: 'inside', view: 'lesion', focus: 'lsRestore' } : null);
    banners(); document.dispatchEvent(new CustomEvent('psi-changed', { detail: { label: changed() } }));
  }
  function mark(n) {
    const now = changed(); if (!n || !now) return;
    const s = document.createElement('span'); s.className = 'chg-mark'; s.textContent = ` · on the changed model (${now})`; n.append(s);
  }
  if (window.PSI && PSI.onState) PSI.onState(() => banners());
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', banners);

  // ── round 2: shared by the experiment views ─────────────────────────────
  //   PSIExp.tests()                   the live battery (understanding-tests.js, shared with src/lesion.py)
  //   PSIExp.custom() / addCustom(t)   tests the user added (kept in this browser; also scored in Tipping point)
  //   PSIExp.key(cond)                 which weights a reply came from (model folder + weights hash): results never mix across keys
  //   PSIExp.cutOf(cond)               the cut state a reply was measured under ('intact' or the cut)
  //   PSIExp.stamp(cond, {protocol, seed, time})   one line per result row: model, dtype, protocol, seed, cut state, time
  //   PSIExp.condLine(anchor, cond, extra)         the conditions line under a result (PSI.cond)
  //   PSIExp.proto(P)                  the copying protocol as a label beside a copying number
  //   PSIExp.keep(view, key, data, model) / saved(view) / archive(view) / archived(view)
  //                                    results kept in localStorage keyed by model; a different model archives, never merges
  //   PSIExp.wilson(p, n)              95% interval of a share of n two-way items
  //   PSIExp.other(cond)               true when a reply's model is not the inside engine's model now set (after a switch)
  //   'psi-inside-model' event          fired when that model becomes known or is switched; views holding results from
  //                                    other weights (PSIExp.other) archive them and start fresh
  //   PSIExp.NOISE                     0.5 nats: one test's margin within this of zero (or of the untrained twin) is noise
  const NOISE = 0.5;
  (function loadTests() {
    if (window.PSI_UNDERSTANDING_TESTS) return;
    const s = document.createElement('script'); s.src = 'understanding-tests.js'; s.async = false;
    s.onerror = () => console.warn('[psi] understanding-tests.js is missing: the live battery cannot run');
    (document.head || document.documentElement).appendChild(s);
  })();
  // The framework's own item (levels-market) is left out while Emergence Machine content is off (Settings > Content).
  const tests = () => (Array.isArray(window.PSI_UNDERSTANDING_TESTS) ? window.PSI_UNDERSTANDING_TESTS : []).filter((t) => !window.PSIEM || PSIEM.on || !/^levels-/.test(t.id));
  const CUSTOM = 'psi-und-custom';
  const custom = () => { try { const a = JSON.parse(localStorage.getItem(CUSTOM) || '[]'); return Array.isArray(a) ? a : []; } catch (_) { return []; } };
  const addCustom = (t) => { const a = custom().filter((x) => x.prompt !== t.prompt); a.push({ what: 'your test', prompt: t.prompt, good: t.good, bad: t.bad, custom: true }); try { localStorage.setItem(CUSTOM, JSON.stringify(a.slice(-20))); } catch (_) { /* private mode: kept for this window only */ } return a; };
  const key = (c) => (c && (c.model_dir || c.model) ? `${c.model_dir || c.model}|${c.weights_hash || 'no hash'}` : '');
  const cutOf = (c) => {
    if (!c || c.error || !c.changed) return changed() ? `changed (${changed()})` : 'cut state not reported';
    const l = c.changed.lesion; if (!l) return 'intact';
    const pc = (a, b) => `${((100 * a) / (b || 1)).toFixed(1)}%`;
    if (l.kind === 'structural') return `cut: ${pc(l.neurons, l.of_neurons)} neurons, ${pc(l.heads, l.of_heads)} heads (${l.mode})`;
    if (l.kind === 'weights') return `cut: ${pc(l.zeroed, l.of)} of ${l.scope} weights zeroed (${l.criterion})`;
    return `cut: ${l.kind}`;
  };
  const proto = (P) => (P && P.label ? `copying protocol: ${P.label}` : 'copying protocol not reported (an older inside engine)');
  function stamp(c, extra = {}) {
    const parts = [];
    if (c && !c.error && c.model) parts.push(c.model, c.dtype || 'precision not reported'); else parts.push('model not reported (an older inside engine)');
    if (extra.protocol) parts.push(extra.protocol);
    if (extra.seed != null) parts.push(`seed ${extra.seed}`);
    parts.push(cutOf(c));
    parts.push(new Date(extra.time || Date.now()).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }));
    return parts.join(' · ');
  }
  function condLine(anchor, c, extra = {}) {
    if (!window.PSI || !PSI.cond || !anchor) return null;
    const parts = [];
    if (c && !c.error && c.model) {
      parts.push({ t: c.model, title: c.model_dir || '' }, `${c.dtype || 'precision not reported'} on ${c.device || 'a device not reported'}`);
      if (c.weights_hash) parts.push({ t: `weights ${String(c.weights_hash).slice(0, 10)}`, title: `sha256 of ${c.weights_hash_of || 'the weights'}: ${c.weights_hash}` });
    } else parts.push({ t: 'conditions not reported (an older inside engine)', warn: true });
    const cut = cutOf(c); parts.push(cut.startsWith('cut') || cut.startsWith('changed') ? { t: cut, chg: true, term: 'changed model' } : cut);
    (extra.parts || []).forEach((x) => parts.push(x));
    return PSI.cond(anchor, parts, extra.lead || 'measured on');
  }
  const LS = 'psi-exp-results';
  const readLS = () => { try { const o = JSON.parse(localStorage.getItem(LS) || '{}'); return o && typeof o === 'object' ? o : {}; } catch (_) { return {}; } };
  const writeLS = (o) => { try { localStorage.setItem(LS, JSON.stringify(o)); return true; } catch (_) { return false; } };
  const archiveEntry = (o, view, e) => { o.archive = (o.archive || []).concat([{ view, ...e, archived: Date.now() }]).slice(-12); };
  function keep(view, k, data, model) {
    const o = readLS(); o.cur = o.cur || {}; const prev = o.cur[view];
    if (prev && prev.key !== k) archiveEntry(o, view, prev);
    o.cur[view] = { key: k, model: model || '', saved: Date.now(), data }; return writeLS(o);
  }
  function archive(view) { const o = readLS(); if (o.cur && o.cur[view]) { archiveEntry(o, view, o.cur[view]); delete o.cur[view]; writeLS(o); } }
  const saved = (view) => { const o = readLS(); return (o.cur && o.cur[view]) || null; };
  const archived = (view) => (readLS().archive || []).filter((a) => a.view === view);
  function wilson(p, n, z = 1.96) {
    if (!n || p == null) return [0, 1];
    const d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d, h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  }
  // c: a conditions block, or a results key (model_dir|weights hash)
  const other = (c) => { const d = typeof c === 'string' ? c.split('|')[0] : c && c.model_dir; return !!(info.name && d && pretty(d) !== info.name); };
  return { empty, put, refresh, other, setChanged, changed, mark, inside: () => ({ ...info }), setLayers: (n) => { if (n && info.name && n !== info.layers) { info.layers = n; paint(); } },
    NOISE, tests, custom, addCustom, key, cutOf, stamp, condLine, proto, keep, saved, archive, archived, wilson };
})();

(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => String(s == null ? '' : s).replace(/�/g, '▫').replace(/\n/g, '⏎');
  const err = (e, o) => PSI.err(e, o);
  const EX = window.PSIExp, N = EX.NOISE;
  const sgn = (v) => `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}`;
  // Three test sets, three names: the live battery (here and in Tipping point), the training replay (recorded, 40/24/40
  // items per measure) and the stored battery (recorded per Pythia size, 7 items; Sensing surface).
  // R: one run on one model. key: the weights it came from; a run on other weights starts fresh (the old one is archived).
  let R = null;
  ['undGo', 'undSave', 'undTwin'].forEach((id) => PSI.requires($(id), 'engine'));
  $('undAdd').addEventListener('click', () => $('undForm').classList.toggle('hidden'));
  $('undSave').addEventListener('click', () => {
    const p = $('undPrompt').value.trim(), g = $('undGood').value.split(',').map((s) => ' ' + s.trim()).filter((s) => s.trim()), b = $('undBad').value.split(',').map((s) => ' ' + s.trim()).filter((s) => s.trim());
    if (!p || !g.length || !b.length) return PSI.status($('undStat'), 'Fill in the prompt and both answer lists first.', 'warn');
    EX.addCustom({ prompt: p, good: g, bad: b }); $('undForm').classList.add('hidden');
    run([{ what: 'your test', prompt: p, good: g, bad: b, custom: true }], true);
  });
  $('undGo').addEventListener('click', () => {
    const T = EX.tests(); if (!T.length) return PSI.status($('undStat'), 'The live battery (understanding-tests.js) is missing from this copy of the app.', 'error');
    run([...T, ...EX.custom()], false);
  });
  $('undTwin').addEventListener('click', twin);

  // ── the training replay: recorded, Pythia-410m (research/phase_shift.py) ──
  const MEASURES = [
    { key: 'agreement', name: 'grammar across a distractor', dash: [] },
    { key: 'consequence', name: 'physical consequence', dash: [6, 4] },
    { key: 'state', name: 'where the object is now', dash: [2, 3] },
  ];
  const META = () => window.UNDERSTANDING_TRAINING_META || null;
  const nOf = (k) => { const m = META(); return m && m.n && m.n[k]; };
  // Each measure against its own untrained baseline (step 0), with a 95% interval per point: "switches on" only where
  // the interval clears the step-0 interval and stays clear; where they overlap the difference is within noise.
  function onset(D, k) {
    const n = nOf(k); if (!n) return null;
    const ci = D.map((d) => EX.wilson(d[k], n)), base = ci[0];
    const above = ci.map((c) => c[0] > base[1]), below = ci.map((c) => c[1] < base[0]);
    const first = above.findIndex((_, i) => i > 0 && above.slice(i).every(Boolean));
    return { n, ci, base, first, below: D.filter((_, i) => below[i]).map((d) => d.step), aboveSome: D.filter((_, i) => above[i]).map((d) => d.step) };
  }
  const pct = (v) => `${Math.round(v * 100)}%`;
  function trainingCaption(D) {
    const parts = MEASURES.map((m) => {
      const o = onset(D, m.key); if (!o) return `${m.name}: n not recorded, so no interval can be drawn`;
      const b = `step 0: ${pct(D[0][m.key])}, 95% interval ${pct(o.base[0])}–${pct(o.base[1])}, n = ${o.n}`;
      if (o.first > 0) return `${m.name} switches on between step ${D[o.first - 1].step.toLocaleString()} and ${D[o.first].step.toLocaleString()} and stays above its untrained baseline beyond noise (${b})`;
      return `${m.name} never stays above its untrained baseline beyond noise: within noise of step 0 ${o.aboveSome.length ? `at most steps (clear of it only at ${o.aboveSome.map((s) => s.toLocaleString()).join(', ')})` : 'throughout'} (${b})`;
    });
    const dips = MEASURES.map((m) => { const o = onset(D, m.key); return o && o.below.length ? `${m.name} at ${o.below.map((s) => s.toLocaleString()).join(', ')}` : null; }).filter(Boolean);
    const txt = parts.join('. ');
    return `${txt.charAt(0).toUpperCase()}${txt.slice(1)}. ${dips.length ? `Below the untrained baseline beyond noise: ${dips.join('; ')}.` : 'No point falls below its untrained baseline beyond noise.'}`;
  }
  function drawTraining() {
    const D = window.UNDERSTANDING_TRAINING; if (!D) return EX.empty($('undTrain'), 220, 'The recorded training results are missing from this copy of the app.');
    const cv = $('undTrain'), r = Math.min(2, devicePixelRatio || 1); cv.style.height = '220px'; cv.width = cv.clientWidth * r; cv.height = 220 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, left = 44 * r, right = W - 44 * r, top = 22 * r, bot = H - 26 * r;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const lx = (s) => Math.log10(Math.max(1, s)), X = (s) => left + (lx(s) / lx(200000)) * (right - left), Y = (v) => bot - v * (bot - top);
    const lo = 5, hi = 16, Y2 = (v) => bot - ((v - lo) / (hi - lo)) * (bot - top);
    g.font = `${9 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = '#6f6c66'; g.textAlign = 'right';
    [0, 0.5, 1].forEach((v) => g.fillText(`${v * 100}%`, left - 4 * r, Y(v) + 3 * r));
    // the 95% interval of each measure, as a band; its step-0 interval, as a faint band across the chart
    MEASURES.forEach((m) => {
      const o = onset(D, m.key); if (!o) return;
      g.fillStyle = 'rgba(242,241,238,0.035)'; g.fillRect(left, Y(o.base[1]), right - left, Y(o.base[0]) - Y(o.base[1]));
      g.fillStyle = 'rgba(242,241,238,0.08)'; g.beginPath();
      D.forEach((d, i) => { const x = X(d.step), y = Y(o.ci[i][1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      for (let i = D.length - 1; i >= 0; i--) g.lineTo(X(D[i].step), Y(o.ci[i][0]));
      g.closePath(); g.fill();
    });
    const line = (key, Yf, color, dash, width) => { g.strokeStyle = color; g.lineWidth = width * r; g.setLineDash(dash.map((d) => d * r)); g.beginPath(); D.forEach((d, i) => { const x = X(d.step), y = Yf(d[key]); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.stroke(); g.setLineDash([]); };
    line('bits_per_token', Y2, 'rgba(169,166,159,0.9)', [], 1);
    MEASURES.forEach((m) => line(m.key, Y, '#f2f1ee', m.dash, m.key === 'agreement' ? 1.6 : 1.4));
    MEASURES.forEach((m) => { const o = onset(D, m.key); if (o && o.first > 0) { g.fillStyle = '#ffc36b'; g.beginPath(); g.arc(X(D[o.first].step), Y(D[o.first][m.key]), 3 * r, 0, Math.PI * 2); g.fill(); } });
    g.fillStyle = '#6f6c66'; g.textAlign = 'center';
    [1, 10, 100, 1000, 10000, 100000].forEach((s) => g.fillText(s >= 1000 ? `${s / 1000}k` : String(s), X(s), H - 10 * r));
    g.textAlign = 'left'; g.fillStyle = '#a9a69f';
    g.fillText('solid: grammar   dashed: consequence   dotted: object tracking   bands: 95% interval   faint bands: step-0 baseline   amber: switched on   grey: loss', left + 4 * r, 12 * r);
    g.textAlign = 'right'; g.fillStyle = '#6f6c66'; g.fillText('training step', right, H - 10 * r);
    const cap = trainingCaption(D); $('undTrainWhy').textContent = cap;
    PSI.canvasLabel(cv, `Recorded Pythia-410m training (training replay, ${MEASURES.map((m) => nOf(m.key) || '?').join('/')} items): the loss falls smoothly from the first step. ${cap}`);
  }
  // The phase-shift panel: four measures of using the context, each scaled to its own maximum.
  function drawShift() {
    const D = window.UNDERSTANDING_TRAINING; if (!D || D[0].induction == null) return EX.empty($('undShift'), 200, 'The recorded phase-shift measurements are missing from this copy of the app.');
    const cv = $('undShift'), r = Math.min(2, devicePixelRatio || 1); cv.style.height = '200px'; cv.width = cv.clientWidth * r; cv.height = 200 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, left = 44 * r, right = W - 44 * r, top = 22 * r, bot = H - 26 * r;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const lx = (s) => Math.log10(Math.max(1, s)), X = (s) => left + (lx(s) / lx(200000)) * (right - left), Y = (v) => bot - Math.max(0, Math.min(1, v)) * (bot - top);
    const series = [
      { f: (d) => d.induction, dash: [], w: 1.6 },
      { f: (d) => d.ctx, dash: [6, 4], w: 1.4 },
      { f: (d) => 1 - (d.absorption == null ? 1 : d.absorption), dash: [6, 2, 2, 2], w: 1.4 },
      { f: (d) => d.abstraction_gain, dash: [2, 3], w: 1.6 },
    ];
    let below = 0;
    series.forEach((s) => {
      const vals = D.map(s.f), max = Math.max(1e-9, ...vals);
      g.strokeStyle = '#f2f1ee'; g.lineWidth = s.w * r; g.setLineDash(s.dash.map((d) => d * r)); g.beginPath();
      D.forEach((d, i) => { const x = X(d.step), y = Y(vals[i] / max); if (i) g.lineTo(x, y); else g.moveTo(x, y); }); g.stroke(); g.setLineDash([]);
      // a value below 0 is off this scale: marked, not drawn as 0
      D.forEach((d, i) => { if (vals[i] < 0) { below++; const x = X(d.step); g.fillStyle = '#ffc36b'; g.beginPath(); g.moveTo(x, bot + 6 * r); g.lineTo(x - 3 * r, bot); g.lineTo(x + 3 * r, bot); g.closePath(); g.fill(); } });
    });
    g.fillStyle = 'rgba(242,241,238,0.08)'; g.fillRect(X(512), top, X(1000) - X(512), bot - top);
    g.font = `${9 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = '#6f6c66'; g.textAlign = 'center';
    [1, 10, 100, 1000, 10000, 100000].forEach((s) => g.fillText(s >= 1000 ? `${s / 1000}k` : String(s), X(s), H - 10 * r));
    g.fillStyle = '#a9a69f'; g.fillText('512 → 1,000', (X(512) + X(1000)) / 2, top - 6 * r);
    g.textAlign = 'left'; g.fillText(`solid: induction (copying)   dashed: context sensitivity   dash-dot: absorption   dotted: E-level probe gain   (each scaled to its maximum)${below ? '   amber ▾: below 0, off this scale' : ''}`, left + 4 * r, 12 * r);
    g.textAlign = 'right'; g.fillStyle = '#6f6c66'; g.fillText('training step', right, H - 10 * r);
    PSI.canvasLabel(cv, `Recorded Pythia-410m training: four measures of using the context are flat until step 512 and rise together between steps 512 and 1,000.${below ? ` ${below} recorded values are below 0 and are marked under the axis, not drawn as 0.` : ''}`);
  }
  function drawMeta() {
    const m = META(), n = $('undTrainMeta'); if (!n) return;
    if (!m) { n.textContent = 'Recording conditions of the training replay: not recorded in this copy of the app.'; return; }
    const v = (x) => (x == null || x === '' ? 'not recorded' : x);
    n.textContent = `Training replay, as recorded: ${v(m.model)} (${v(m.variant)}), ${v(m.checkpoints)} checkpoints, ${v(m.dtype)} on ${v(m.device)}, ${v(m.date)} · n per measure: grammar ${v(m.n && m.n.agreement)}, consequence ${v(m.n && m.n.consequence)}, tracking ${v(m.n && m.n.state)} two-way items · ${v(m.scoring)} · seed: ${v(m.seed)} · copying in the phase-shift panel: ${v(m.copy_protocol)} · source ${v(m.source)}.`;
  }
  function drawAll() { drawTraining(); drawShift(); drawMeta(); if (R) draw(); else EX.empty($('undCanvas'), 200, 'Nothing measured yet: press Run the tests.'); }
  drawAll(); addEventListener('resize', drawAll);
  // Hidden canvases have no width: redraw whenever the tab gives them one.
  if (window.ResizeObserver) new ResizeObserver(() => { if ($('undTrain').clientWidth && $('undTrain').width !== Math.round($('undTrain').clientWidth * Math.min(2, devicePixelRatio || 1))) drawAll(); }).observe($('undTrain'));
  // Registered on the tab bar so it runs after app.js has unhidden the panel; a timer (not rAF) so it also fires in a background tab.
  document.getElementById('tabs').addEventListener('click', (ev) => { if (ev.target.closest('button[data-tab=understand]')) setTimeout(drawAll, 0); });

  // ── the live battery ─────────────────────────────────────────────────────
  const kindOf = (t) => (t && t.custom ? 'yours' : t && t.self_referential ? 'self' : 'core');
  const testOf = (it) => it._t || EX.tests().find((x) => x.prompt === it.prompt) || EX.custom().find((x) => x.prompt === it.prompt) || { what: '' };
  // One meaning of "pass": ahead of the untrained twin by more than the noise; until the twin is measured, the fixed
  // threshold (log-odds > 0), said as such.
  function verdictOf(it) {
    const tw = R && R.twin && R.twin[it.prompt];
    if (tw) { const d = it.logodds - tw.logodds; return { pass: d > N, cls: d > N ? 'ok' : d < -N ? 'no' : 'noise', text: `${sgn(it.logodds)}, twin ${sgn(tw.logodds)}: ` + (d > N ? `beats the untrained twin by ${sgn(d)}` : d < -N ? `below the untrained twin (${sgn(d)})` : `within noise of the untrained twin (${sgn(d)}; noise ±${N})`) }; }
    return { pass: it.logodds > 0, cls: it.logodds > 0 ? 'ok' : 'no', text: `${it.logodds > 0 ? 'understanding answer ahead' : 'surface answer ahead'} (${sgn(it.logodds)}) · fixed threshold` };
  }
  // One request per test, so Stop takes effect between tests (the test in flight finishes first).
  async function run(items, append) {
    const btns = [$('undGo'), $('undSave'), $('undTwin')]; btns.forEach((b) => { b.disabled = true; });
    PSI.status($('undStat'), '');
    const p = PSI.progress($('undProg'), { total: items.length, label: 'test' });
    const got = []; let layers = R ? R.layers : 0, secs = 0, cond = null, fresh = '';
    try {
      for (let i = 0; i < items.length; i++) {
        if (p.cancelled) break;
        p.step(i, i ? items[i].what : 'the first one also starts the inside engine if it is not running');
        const r = await psi.understand([items[i]]);
        cond = r.conditions || cond;
        r.items.forEach((it) => { it._t = items[i]; it._stamp = EX.stamp(r.conditions, { time: Date.now() }); it._cut = EX.cutOf(r.conditions); got.push(it); }); layers = r.layers || layers; secs += r.seconds || 0;
        p.step(i + 1);
      }
      if (!got.length) { p.finish('stopped before any test finished', { markDone: false }); PSI.status($('undStat'), 'Stopped: no results.', 'warn'); return; }
      const k = EX.key(cond);
      // Results from different weights never mix: a test added after the model changed starts a fresh run.
      if (append && R && R.key !== k) { fresh = 'The model changed since the last run, so this is a fresh run (the earlier one is archived, not merged). '; EX.archive('understand'); R = null; }
      if (append && R) R.items.push(...got.filter((g) => !R.items.some((x) => x.prompt === g.prompt))); else R = { key: k, cond, layers, items: got, twin: null, twinInfo: null };
      R.cond = cond; R.layers = layers; EX.setLayers(layers);
      const msg = fresh + render() + ` · ${secs.toFixed(1)}s`;
      if (p.cancelled) { p.finish(`stopped after ${got.length} of ${items.length} tests`, { markDone: false }); PSI.status($('undStat'), `${msg} (stopped early)`, 'warn'); }
      else { p.finish(`${got.length} tests scored`); PSI.status($('undStat'), msg, 'ok'); }
      EX.mark($('undStat')); save(); draw();
    } catch (e) {
      p.fail(`stopped by an error after ${got.length} of ${items.length} tests`); err(e);
      PSI.status($('undStat'), `The tests could not run: ${PSI.explain(e).text}`, 'error');
    } finally { btns.forEach((b) => { b.disabled = false; }); }
  }
  // The untrained twin: the same items on a randomly initialised copy of the same config (built once per model).
  async function twin() {
    if (!R || !R.items.length) return PSI.status($('undTwinStat'), 'Run the tests first: the twin is measured on the same items.', 'warn');
    const btns = [$('undGo'), $('undSave'), $('undTwin')]; btns.forEach((b) => { b.disabled = true; });
    const items = R.items.map((it) => { const t = testOf(it); return { prompt: it.prompt, good: t.good, bad: t.bad }; }).filter((t) => t.good && t.bad);
    const p = PSI.progress($('undProg'), { total: items.length, label: 'twin test' }); const tw = {}; let info = null, k = null, secs = 0;
    PSI.status($('undTwinStat'), 'building the untrained twin (as much memory again as the model; GPU if it has room, else system memory)…', 'busy');
    try {
      for (let i = 0; i < items.length; i++) {
        if (p.cancelled) break;
        p.step(i, i ? 'scoring the twin' : 'the first request builds the twin');
        const r = await psi.understand({ twin: true, items: [items[i]] });
        if (!r.twin) throw new Error('This inside engine does not build an untrained twin (an older trace_server.py): restart it from this copy of the app.');
        info = r.twin; k = EX.key(r.conditions); secs += r.seconds || 0;
        r.items.forEach((it) => { tw[it.prompt] = { logodds: it.logodds, by_layer: it.by_layer }; });
        p.step(i + 1);
      }
      if (k !== R.key) { p.fail('the model changed while the twin was measured'); PSI.status($('undTwinStat'), 'The inside engine’s model is not the one the tests ran on: the twin was discarded. Run the tests again, then the twin.', 'warn'); return; }
      R.twin = { ...(R.twin || {}), ...tw }; R.twinInfo = info;
      const msg = render();
      p.finish(`${Object.keys(tw).length} twin tests scored`);
      PSI.status($('undTwinStat'), `twin: random weights (${info.init.split(':')[0]}, seed ${info.seed}), ${info.dtype} on ${info.device}, built in ${info.build_s}s · ${secs.toFixed(1)}s`, 'ok');
      PSI.status($('undStat'), msg, 'ok'); save(); draw();
    } catch (e) {
      p.fail('stopped by an error'); err(e); PSI.status($('undTwinStat'), `The untrained twin could not be measured: ${PSI.explain(e).text}`, 'error');
    } finally { btns.forEach((b) => { b.disabled = false; }); }
  }
  function save() {
    if (!R) return;
    const ok = EX.keep('understand', R.key, R, R.cond && R.cond.model);
    if (!ok) PSI.status($('undSaved'), 'Results are not kept between sessions here (this browser blocks storage).', 'warn');
  }
  // The headline: the core tests (not the self-referential one, not yours), against the twin when it is measured.
  function headline() {
    const by = { core: [], self: [], yours: [] }; R.items.forEach((it) => by[kindOf(testOf(it))].push(it));
    const pass = (a) => a.filter((it) => verdictOf(it).pass).length, twinAll = R.twin && R.items.every((it) => R.twin[it.prompt]);
    const rule = R.twin ? (twinAll ? `beat the untrained twin by more than ${N} nats` : `beat the untrained twin by more than ${N} nats (twin measured for some tests only)`) : 'above the fixed threshold (log-odds > 0); no untrained baseline measured';
    let s = `live battery: ${pass(by.core)} of ${by.core.length} core tests ${rule}`;
    if (by.self.length) s += ` · self-referential test (rewards this framework’s own words, not counted): ${by.self.map((it) => (verdictOf(it).pass ? 'passed' : 'not passed')).join(', ')}`;
    if (by.yours.length) s += ` · your tests: ${pass(by.yours)} of ${by.yours.length}`;
    return s;
  }
  function render() {
    const tb = $('undTable').querySelector('tbody'); tb.textContent = '';
    const order = { core: 0, self: 1, yours: 2 };
    R.items.slice().sort((a, b) => (order[kindOf(testOf(a))] - order[kindOf(testOf(b))]) || ((verdictOf(a).pass ? 1 : 0) - (verdictOf(b).pass ? 1 : 0))).forEach((it) => {   // the core tests it failed lead
      const t = testOf(it), v = verdictOf(it), kd = kindOf(t), tw = R.twin && R.twin[it.prompt];
      const tr = el('tr'); if (kd !== 'core') tr.className = `und-${kd}`;
      const w = el('td', null, t.what); if (kd === 'self') w.appendChild(el('span', 'und-tag', t.note || 'self-referential: rewards this framework’s own words')); if (kd === 'yours') w.appendChild(el('span', 'und-tag', 'yours · not in the headline'));
      tr.appendChild(w); tr.appendChild(el('td', null, it.prompt.length > 70 ? '…' + it.prompt.slice(-70) : it.prompt));
      tr.appendChild(el('td', null, `${t.good ? t.good.join('/') : ''} vs ${t.bad ? t.bad.join('/') : ''}`));
      tr.appendChild(el('td', v.cls, v.text));
      tr.appendChild(el('td', tw ? null : 'muted', tw ? sgn(tw.logodds) : 'not measured'));
      tr.appendChild(el('td', it.readable_from ? null : 'no', it.readable_from ? `L${it.readable_from} of ${R.layers}` : 'never settles: no layer from which it stays ahead'));
      tr.appendChild(el('td', null, it.top.map(([pc, val]) => `${show(pc)} ${(val * 100).toFixed(0)}%`).join(' · ')));
      tr.appendChild(el('td', 'und-stamp', it._stamp || 'conditions not recorded'));
      tb.appendChild(tr);
    });
    EX.condLine($('undProg'), R.cond, { parts: [R.twinInfo ? `untrained twin: ${R.twinInfo.dtype} on ${R.twinInfo.device}, seed ${R.twinInfo.seed}` : { t: 'no untrained twin measured', warn: true }] });
    return headline();
  }

  function draw() {
    const cv = $('undCanvas'), r = Math.min(2, devicePixelRatio || 1); cv.style.height = '200px'; if (!cv.clientWidth) return; cv.width = cv.clientWidth * r; cv.height = 200 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, L = R.layers, left = 40 * r, right = W - 14 * r, top = 14 * r, bot = H - 22 * r;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const twins = R.twin ? R.items.map((it) => R.twin[it.prompt]).filter(Boolean) : [];
    const all = [...R.items.flatMap((it) => it.by_layer), ...twins.flatMap((t) => t.by_layer || [])], lo = Math.min(-2, ...all), hi = Math.max(2, ...all);
    const X = (l) => left + (l / Math.max(1, L - 1)) * (right - left), Y = (v) => bot - ((v - lo) / ((hi - lo) || 1)) * (bot - top);
    g.strokeStyle = 'rgba(242,241,238,0.35)'; g.lineWidth = r; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(left, Y(0)); g.lineTo(right, Y(0)); g.stroke(); g.setLineDash([]);
    twins.forEach((t) => { if (!t.by_layer) return; g.strokeStyle = 'rgba(169,166,159,0.45)'; g.lineWidth = 0.8 * r; g.setLineDash([2 * r, 3 * r]); g.beginPath(); t.by_layer.forEach((v, l) => { l ? g.lineTo(X(l), Y(v)) : g.moveTo(X(l), Y(v)); }); g.stroke(); g.setLineDash([]); });
    R.items.forEach((it, i) => {
      const pass = verdictOf(it).pass;
      g.strokeStyle = pass ? `rgba(242,241,238,${0.35 + 0.5 * (i / R.items.length)})` : 'rgba(169,166,159,0.35)'; g.lineWidth = r * (pass ? 1.2 : 0.8);
      g.beginPath(); it.by_layer.forEach((v, l) => { l ? g.lineTo(X(l), Y(v)) : g.moveTo(X(l), Y(v)); }); g.stroke();
      if (it.readable_from) { g.fillStyle = '#f2f1ee'; g.beginPath(); g.arc(X(it.readable_from - 1), Y(it.by_layer[it.readable_from - 1]), 2.5 * r, 0, Math.PI * 2); g.fill(); }
    });
    g.font = `${9 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = '#6f6c66'; g.textAlign = 'center';
    for (let l = 0; l < L; l += 5) g.fillText(`L${l + 1}`, X(l), H - 8 * r);
    g.textAlign = 'left'; g.fillText(`understanding answer ahead ↑${twins.length ? '   grey dotted: the untrained twin' : ''}`, left + 4 * r, top + 8 * r); g.fillText('surface answer ahead ↓', left + 4 * r, bot - 4 * r);
    const from = R.items.filter((it) => verdictOf(it).pass).map((it) => it.readable_from).filter(Boolean);
    PSI.canvasLabel(cv, `Layer curves for ${R.items.length} tests across ${L} layers${twins.length ? ', with the untrained twin dotted' : ''}. ${headline()}${from.length ? `; readable from layer ${Math.min(...from)} at the earliest and ${Math.max(...from)} at the latest` : ''}.`);
  }
  // Results kept from an earlier session (this model or another): shown as such, never merged with a new run.
  const prev = EX.saved('understand');
  if (prev && prev.data && Array.isArray(prev.data.items) && prev.data.items.length) {
    R = prev.data; render(); draw();
    PSI.status($('undSaved'), `Shown: the last run, kept from ${new Date(prev.saved).toLocaleString()} (${prev.model || 'model not recorded'}). A run on other weights starts fresh and archives it.`, '');
  }
  // The inside engine's model was switched: the run on screen belongs to the old weights, so it is archived and cleared.
  function switched() {
    if (!R || !EX.other(R.cond)) return;
    EX.archive('understand'); const was = (R.cond && R.cond.model) || 'the previous model'; R = null;
    $('undTable').querySelector('tbody').textContent = ''; const c = $('undProg').nextElementSibling; if (c && c.classList.contains('cond')) c.remove();
    EX.empty($('undCanvas'), 200, 'Nothing measured on this model yet: press Run the tests.');
    PSI.status($('undStat'), ''); PSI.status($('undTwinStat'), '');
    PSI.status($('undSaved'), `The inside engine’s model changed: the run on ${was} was archived (kept in this browser, not mixed with new results).`, 'warn');
  }
  document.addEventListener('psi-inside-model', switched);
  const arch = EX.archived('understand');
  if (arch.length) $('undSaved').title = `Archived runs on other models: ${arch.map((a) => `${a.model || '?'} (${new Date(a.saved).toLocaleDateString()})`).join(', ')}`;
})();
