// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Anatomy: the loaded model's real structure and weights, and a recorded forward pass you can play,
// pause and scrub: which neurons fire, which heads look where, how context is stored and read.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  const mark = (id) => { if (window.PSIExp) PSIExp.mark($(id)); };   // while Tipping point has the model cut, results say so
  const INK = '237,235,228', AMBER = '#ffc36b', show = (s) => String(s == null ? '' : s).replace(/\n/g, '⏎');
  const fmtN = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} k` : String(n);
  const canvas = (cv, h) => { const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || 800) * r; cv.height = h * r; const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); return { g, r, W: cv.width, H: cv.height }; };
  const A = { st: null, links: null, rec: null, step: 0, playing: false, timer: null, sel: null, maps: {}, geo: null, bins: 0, zoom: null, zsort: false };
  // Automatic binning: the neuron column gets one bin per ~6 drawn pixels (16 to 160, never more than the
  // layer has neurons); bigger models get a taller drawing, so they get more bins as well as more neurons per bin.
  const netH = (mlp) => (mlp > 20000 ? 640 : mlp > 8000 ? 540 : 470);
  // A small model (the toy Llama from How AI Emerges has 176 per layer) fits at 2 pixels a neuron: draw every one, no bins.
  const autoBins = (mlp) => (mlp * 2 <= netH(mlp) - 118 ? mlp : Math.max(16, Math.min(160, Math.floor((netH(mlp) - 118) / 6), mlp)));

  async function call(args) { return psi.anatomy(args); }
  // A reply's conditions (trace_server.py), on a conditions line under the row of `id`: the same parts as Neural net.
  const cond = (id, r, extra) => { if (!PSI.insideCondParts || !$(id)) return; const row = $(id).closest('.row') || $(id); PSI.cond(row, PSI.insideCondParts(r && r.conditions, extra), 'run on'); };
  const stat = (id, text, kind) => PSI.status($(id), text, kind);
  const failed = (id, what, e) => stat(id, `${what} failed: ${PSI.explain(e).text}`, 'error');

  // What needs what: everything here reads the weights through the inside-the-model engine; drafting
  // neuron names also asks Prometheus through OpenRouter (the other Prometheus parts fall back without a key).
  ['anRecord', 'paRun', 'ccRun', 'ccKO', 'cnRun'].forEach((id) => PSI.requires($(id), 'engine'));
  PSI.requires($('ccName'), ['engine', 'key']);

  // An empty drawing says what will appear there and which button makes it.
  function hint(cv, h, text) {
    const { g, r, W, H } = canvas(cv, h);
    g.fillStyle = '#a7abb6'; g.font = `${12 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center'; g.textBaseline = 'middle'; const lines = []; let ln = ''; String(text).split(' ').forEach((wd) => { const t2 = ln ? ln + ' ' + wd : wd; if (ln && g.measureText(t2).width > W * 0.9) { lines.push(ln); ln = wd; } else ln = t2; }); lines.push(ln); const lh = 18 * r; lines.forEach((t2, i) => g.fillText(t2, W / 2, H / 2 + (i - (lines.length - 1) / 2) * lh)); g.textBaseline = 'alphabetic';
    PSI.canvasLabel(cv, text);
  }
  function netHint() {
    if (A.st) return;
    hint($('anNet'), 470, A.loading ? 'reading the model’s structure…' : PSI.state.engine ? 'Nothing read yet: press “Record a forward pass” below' : 'Nothing to draw yet: the inside-the-model engine is not set up (see Settings)');
  }
  const patchHint = () => { if (!PA.d) hint($('paCv'), 360, 'Nothing patched yet: press “Find where it is decided”'); };

  // A long engine job with a progress line and a Stop button under its button row. The engine cannot be interrupted
  // mid-job, so Stop lets go of it: the view is free at once, the result is dropped, and the engine finishes in the
  // background before it takes the next job. Resolves to null when stopped.
  const progHost = (id) => { const row = $(id).closest('.row'); let h = row.nextElementSibling; if (!h || !h.classList.contains('prog-host')) { h = el('div', 'prog-host'); row.after(h); } return h; };
  function longCall(btnId, { text, total = 0, est = 0, label = 'step', doneText }, fn) {
    let stopped = false, onStop = () => {};
    const p = PSI.progress(progHost(btnId), { total, label, onCancel: () => { stopped = true; onStop(); } });
    const t0 = performance.now(), pt = p.el.querySelector('.psi-ptext');
    const tick = () => {
      const s = (performance.now() - t0) / 1000;
      if (total && est) p.step(Math.min(total - 1, Math.floor((s / est) * total)), `${text} · estimated from time`);
      else if (pt) pt.textContent = `${text} · ${Math.round(s)} s`;
    };
    const iv = setInterval(tick, 1000); tick();
    const later = () => setTimeout(() => { if (p.el.isConnected && p.el.classList.contains('ended')) p.remove(); }, 6000);
    return new Promise((resolve, reject) => {
      onStop = () => { clearInterval(iv); p.finish('stopped · the engine finishes this job in the background before the next one', { markDone: false }); later(); resolve(null); };
      fn().then((r) => { if (stopped) return; clearInterval(iv); p.finish(typeof doneText === 'function' ? doneText(r) : doneText || 'done'); later(); resolve(r); },
        (e) => { if (stopped) return; clearInterval(iv); p.fail(`stopped by an error: ${PSI.explain(e).text}`); reject(e); });
    });
  }

  // ── 1. structure ────────────────────────────────────────────────────────
  async function loadStructure() {
    stat('anStat', 'reading the model’s structure (starts the inside-the-model engine)…', 'busy'); A.loading = true; netHint();
    try {
      A.st = await call({ action: 'structure' }); A.bins = autoBins(A.st.mlp); A.st = await call({ action: 'structure', bins: A.bins }); const s = A.st;
      $('anSizeWord').textContent = A.bins < s.mlp ? `this ${fmtN(s.params)}-parameter model` : 'a large model';
      const cards = [['parameters', fmtN(s.params)], ['layers', s.layers], ['width (hidden)', s.hidden], ['attention heads', `${s.heads} (KV ${s.kv_heads}, ${s.heads / s.kv_heads} share each)`],
        ['head size', s.head_dim], ['MLP neurons per layer', s.mlp.toLocaleString()], ['vocabulary', s.vocab.toLocaleString()], ['context window', s.context ? s.context.toLocaleString() : 'not stated in its config'],
        ['context memory per token', `${(s.kv_bytes_per_token / 1024).toFixed(0)} KB (keys + values, every layer)`], ['precision', s.dtype], ['embeddings', `${fmtN(s.embedding_params)}${s.tied_embeddings ? ', shared with the output' : ''}`]];
      const box = $('anCards'); box.textContent = '';
      cards.forEach(([k, v]) => { const c = el('div', 'an-card'); c.appendChild(el('p', 'k', k)); c.appendChild(el('p', 'an-v', String(v))); box.appendChild(c); });
      const lay = s.per_layer[0]; $('anLayerParams').textContent = `Each of the ${s.layers} layers holds ${fmtN(lay.params)} parameters: ${Object.entries(lay.tensors).map(([n, t]) => `${n.replace(/\.weight$/, '')} ${t.shape.join('×')}`).join(', ')}.`;
      $('anModelNote').textContent = `This is ${s.name}, the copy the inside-the-model engine holds in memory: an unpacked folder (${s.dtype}). The engine that answers in Generate runs a GGUF file, ${(window.PSISettingsModel || '').split(/[\\/]/).pop() || 'see Settings'}, which the File view reads.`;
      const mism = PSI.insideVsFile ? PSI.insideVsFile({ layers: s.layers, width: s.hidden, vocab: s.vocab, name: s.name, dtype: s.dtype }) : [];
      if (mism.length) $('anModelNote').appendChild(el('span', 'cond-warn', ` They are not the same shape (${mism.join('; ')}): what this view shows is not the model Generate runs.`));
      if (s.conditions && s.conditions.weights_hash) A.nameKey = `${s.name}@${s.conditions.weights_hash.slice(0, 12)}`;
      else if (s.conditions && s.conditions.model_dir) A.nameKey = `folder:${s.conditions.model_dir}`;
      stat('anStat', 'computing the links from the weights…', 'busy');
      A.links = await call({ action: 'links', bins: A.bins });
      stat('anStat', A.links.skipped ? `links skipped: ${A.links.skipped}` : `links computed from the weights in ${A.links.seconds}s`, A.links.skipped ? 'warn' : 'ok');
      cond('anStat', A.links && A.links.conditions ? A.links : s); drawNet();
    } catch (e) { A.stFailed = true; err(e); failed('anStat', 'Reading the model’s structure', e); } finally { A.loading = false; netHint(); }
  }

  // ── 2. the network ──────────────────────────────────────────────────────
  // Deep models: when a column per layer would be narrower than ~9 px, consecutive layers share a column
  // (the column shows the strongest of its layers, bin by bin), so the drawing never becomes unreadable.
  function drawNet() {
    const s = A.st; if (!s) return; const L = s.layers, Hh = s.heads, rec = A.rec;
    const G = rec ? rec.bins : s.bins, I = s.mlp, edges = (rec && rec.edges) || (A.links && A.links.edges) || null;
    const { g, r, W, H } = canvas($('anNet'), netH(I));
    const left = 54 * r, right = W - 54 * r, top = 26 * r, headH = 44 * r, gap = 14 * r, binTop = top + headH + gap, binBot = H - 34 * r;
    const ncol = Math.max(1, Math.min(L, Math.floor((right - left) / (9 * r)))), per = Math.ceil(L / ncol), C = Math.ceil(L / per);
    const colW = (right - left) / C, X = (c) => left + (c + 0.5) * colW, cellH = (binBot - binTop) / G;
    const yN = (n) => binTop + (n / I) * (binBot - binTop);             // a neuron index -> its height (bins can be uneven)
    const yB = (b) => edges ? yN(edges[b]) : binTop + b * cellH, hB = (b) => edges ? yN(edges[b + 1]) - yN(edges[b]) : cellH;
    A.geo = { left, right, top, binTop, binBot, colW, cellH, r, L, G, per, C, edges, I };
    const st = rec ? rec.steps[A.step] : null;
    const colOf = (l) => Math.floor(l / per);
    // links: only between the last layer of a column and the first layer of the next (that pair is real)
    if (A.links && A.links.pairs) {
      const nbG = A.links.bins;
      A.links.pairs.forEach((p) => {
        if ((p.from_layer + 1) % per !== 0) return;
        const c = colOf(p.from_layer), flat = []; p.bins.forEach((row, i) => row.forEach((v, j) => flat.push([v, i, j]))); flat.sort((a, b) => b[0] - a[0]);
        const y = (b) => binTop + ((b + 0.5) / nbG) * (binBot - binTop);
        flat.slice(0, 10).forEach(([v, i, j]) => { g.strokeStyle = `rgba(${INK},${(0.05 + 0.18 * v).toFixed(3)})`; g.lineWidth = r; g.beginPath(); g.moveTo(X(c) + colW * 0.3, y(j)); g.lineTo(X(c + 1) - colW * 0.3, y(i)); g.stroke(); });
        p.top.slice(0, 6).forEach((t) => { g.strokeStyle = t.w > 0 ? 'rgba(255,195,107,.55)' : 'rgba(150,170,255,.45)'; g.lineWidth = 1.2 * r; g.beginPath(); g.moveTo(X(c) + colW * 0.3, yN(t.from + 0.5)); g.lineTo(X(c + 1) - colW * 0.3, yN(t.to + 0.5)); g.stroke(); });
      });
    }
    for (let c = 0; c < C; c++) {
      const ls = []; for (let l = c * per; l < Math.min(L, (c + 1) * per); l++) ls.push(l);
      const x0 = X(c) - colW * 0.3, w = colW * 0.6, lays = st ? ls.map((l) => st.layers[l]) : null;
      const bins = new Array(G).fill(0); if (lays) lays.forEach((lay) => lay.bins.forEach((v, b) => { if (v > bins[b]) bins[b] = v; }));
      let mx = 1e-9; bins.forEach((v) => { if (v > mx) mx = v; });
      for (let b = 0; b < G; b++) {
        const v = lays ? bins[b] / mx : 0.12; g.fillStyle = lays ? `rgba(${INK},${(0.06 + 0.9 * v * v).toFixed(3)})` : 'rgba(237,235,228,.10)';
        g.fillRect(x0, yB(b), w, Math.max(1, hB(b) - (cellH > 3 * r ? 0.6 * r : 0)));
      }
      if (A.concept && $('ccShow') && $('ccShow').checked && A.concept.bins === G) {   // the concept's neurons, lit
        for (let b = 0; b < G; b++) { let v = 0; ls.forEach((l) => { v = Math.max(v, A.concept.light[l][b]); }); if (v <= 0.02) continue;
          g.fillStyle = `rgba(255,195,107,${(0.25 + 0.75 * Math.sqrt(v)).toFixed(3)})`; g.fillRect(x0 - 1.5 * r, yB(b) - 0.5 * r, w + 3 * r, Math.max(1.5 * r, hB(b))); }
      }
      const kvper = Hh / s.kv_heads, hw = w / kvper, hh = headH / s.kv_heads;
      for (let h = 0; h < Hh; h++) {
        const kv = Math.floor(h / kvper), j = h % kvper; const focus = lays ? 1 - Math.min(...lays.map((lay) => lay.ent[h])) : 0.15;
        g.fillStyle = lays ? `rgba(255,195,107,${(0.08 + 0.85 * Math.max(0, focus)).toFixed(3)})` : 'rgba(237,235,228,.12)';
        g.fillRect(x0 + j * hw, top + kv * hh, Math.max(1, hw - 0.6 * r), Math.max(1, hh - 0.6 * r));
      }
      if (A.sel != null && colOf(A.sel) === c) { g.strokeStyle = AMBER; g.lineWidth = 1.5 * r; g.strokeRect(x0 - 2 * r, top - 4 * r, w + 4 * r, binBot - top + 8 * r); }
      if (A.zoom && colOf(A.zoom.layer) === c) { g.strokeStyle = '#fff'; g.lineWidth = 1.5 * r; g.strokeRect(x0 - 3 * r, yB(A.zoom.bin) - r, w + 6 * r, hB(A.zoom.bin) + 2 * r); }
      if (A.kcur && colOf(A.kcur.layer) === c && document.activeElement === $('anNet')) {   // the keyboard cursor: dashed, on the column or one bin
        g.save(); g.strokeStyle = AMBER; g.lineWidth = 2 * r; g.setLineDash([4 * r, 3 * r]);
        if (A.kcur.bin == null) g.strokeRect(x0 - 5 * r, top - 6 * r, w + 10 * r, binBot - top + 12 * r); else g.strokeRect(x0 - 5 * r, yB(A.kcur.bin) - 2 * r, w + 10 * r, Math.max(4 * r, hB(A.kcur.bin) + 4 * r));
        g.restore();
      }
      const every = Math.max(1, Math.round(5 / per));
      if (c % every === 0 || c === C - 1) { g.fillStyle = '#7d818c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center'; g.fillText(per > 1 ? `L${ls[0]}` : `L${ls[0]}`, X(c), H - 18 * r); }
      if (lays) { const rn = Math.min(1, Math.max(...lays.map((lay) => lay.resid)) / 400); g.fillStyle = 'rgba(167,171,182,.6)'; g.fillRect(x0, H - 12 * r, w * rn, 3 * r); }
    }
    g.fillStyle = '#a7abb6'; g.font = `${10 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'left';
    g.fillText('heads', 6 * r, top + 12 * r); g.fillText('neurons', 6 * r, binTop + 12 * r); g.fillText(`${G} bins`, 6 * r, binTop + 26 * r); g.fillText(`×${Math.round(I / G)}`, 6 * r, binTop + 40 * r);
    if (per > 1) g.fillText(`${per} layers`, 6 * r, binTop + 58 * r), g.fillText('per column', 6 * r, binTop + 72 * r);
    if (A.concept && $('ccShow') && $('ccShow').checked) { g.fillStyle = '#ffc36b'; g.textAlign = 'left'; g.fillText(`lit: “${A.concept.concept}”`, 6 * r, H - 4 * r); g.fillStyle = '#a7abb6'; }
    g.textAlign = 'right'; g.fillText(st ? `${st.phase === 'read' ? 'reading' : 'writing'} · ${show(rec.pieces[st.pos])}${rec ? ' · click a neuron bin to zoom in' : ''}` : 'the weights (no input yet) · click a neuron bin to zoom in', right, 14 * r);
    $('anBinNote').textContent = G >= I ? `Every one of the ${I.toLocaleString()} neurons in each layer is drawn, one cell each: this model is small enough to need no bins.` : `Automatic binning: ${I.toLocaleString()} neurons per layer drawn as ${G} bins of about ${Math.round(I / G)}${per > 1 ? `; ${L} layers drawn as ${C} columns of ${per}, each showing the strongest of its layers` : ''}. Bins are a down-sampling: neuron order inside a layer is arbitrary.`;
    PSI.canvasLabel($('anNet'), `The network: ${L} layers${per > 1 ? ` in ${C} columns` : ''}, ${Hh} attention heads and ${I.toLocaleString()} neurons (${G} bins) each${st ? `; at step ${A.step + 1}, ${st.phase === 'read' ? 'reading' : 'writing'} “${show(rec.pieces[st.pos])}”` : ', the weights with no input yet'}${A.sel != null ? `; layer ${A.sel} selected` : ''}. Keyboard: arrow keys move, Enter opens.`);
  }
  // where the pointer is on the network: which layer column, which bin, which layers it holds
  function hit(e) {
    const G_ = A.geo; if (!G_) return null; const rc = $('anNet').getBoundingClientRect(), x = (e.clientX - rc.left) * G_.r, y = (e.clientY - rc.top) * G_.r;
    const c = Math.floor((x - G_.left) / G_.colW); if (c < 0 || c >= G_.C) return null;
    const layers = []; for (let l = c * G_.per; l < Math.min(G_.L, (c + 1) * G_.per); l++) layers.push(l);
    let bin = null;
    if (y >= G_.binTop && y <= G_.binBot) { const n = ((y - G_.binTop) / (G_.binBot - G_.binTop)) * G_.I; if (G_.edges) { bin = G_.edges.findIndex((v, i) => i < G_.edges.length - 1 && n >= v && n < G_.edges[i + 1]); if (bin < 0) bin = G_.G - 1; } else bin = Math.min(G_.G - 1, Math.floor((y - G_.binTop) / G_.cellH)); }
    return { c, layers, bin };
  }
  // in a grouped column, zoom into the layer that is firing hardest in that bin right now
  function pickLayer(h) {
    if (h.layers.length === 1 || !A.rec || h.bin == null) return h.layers[0];
    const st = A.rec.steps[A.step]; return h.layers.reduce((best, l) => (st.layers[l].bins[h.bin] > st.layers[best].bins[h.bin] ? l : best), h.layers[0]);
  }
  $('anNet').addEventListener('click', (e) => {
    const h = hit(e); if (!h) return; const l = pickLayer(h);
    A.sel = l; if (h.bin != null) openZoom(l, h.bin); else drawNet(); layerDetail(l);
  });
  $('anNet').addEventListener('mousemove', (e) => {
    const h = hit(e); if (!h) { $('anNet').title = ''; return; }
    const lay = h.layers.length > 1 ? `layers ${h.layers[0]}–${h.layers[h.layers.length - 1]}` : `layer ${h.layers[0]}`;
    if (h.bin == null) { $('anNet').title = `${lay} · click for its weights`; return; }
    const E = A.geo.edges, lo = E ? E[h.bin] : Math.round(h.bin * A.geo.I / A.geo.G), hi = E ? E[h.bin + 1] : Math.round((h.bin + 1) * A.geo.I / A.geo.G);
    $('anNet').title = `${lay} · bin ${h.bin}: neurons ${lo}–${hi - 1} · click to zoom in`;
  });
  // The keyboard way in: left and right move between layer columns, down steps into the neuron bins and up back to
  // the whole column, Enter opens what is under the dashed cursor (a layer's weights, or a bin's zoom), Escape closes the zoom.
  function cursorText() {
    const G_ = A.geo, k = A.kcur; if (!G_ || !k) return '';
    const c = Math.floor(k.layer / G_.per), l0 = c * G_.per, l1 = Math.min(G_.L, l0 + G_.per) - 1, lay = l1 > l0 ? `layers ${l0}–${l1}` : `layer ${l0}`;
    if (k.bin == null) return `${lay}: press Enter for its weights, Down for its neuron bins`;
    const E = G_.edges, lo = E ? E[k.bin] : Math.round(k.bin * G_.I / G_.G), hi = E ? E[k.bin + 1] : Math.round((k.bin + 1) * G_.I / G_.G);
    return `${lay}, bin ${k.bin + 1} of ${G_.G}: neurons ${lo}–${hi - 1}; press Enter to zoom in`;
  }
  $('anNet').addEventListener('keydown', (e) => {
    const G_ = A.geo; if (!G_ || !A.st) return;
    const k = A.kcur || (A.kcur = { layer: A.sel != null ? A.sel : 0, bin: null }), c = Math.floor(k.layer / G_.per);
    switch (e.key) {
      case 'ArrowRight': k.layer = Math.min(G_.L - 1, (c + 1) * G_.per); break;
      case 'ArrowLeft': k.layer = Math.max(0, (c - 1) * G_.per); break;
      case 'Home': k.layer = 0; break;
      case 'End': k.layer = (G_.C - 1) * G_.per; break;
      case 'ArrowDown': k.bin = k.bin == null ? 0 : Math.min(G_.G - 1, k.bin + 1); break;
      case 'ArrowUp': k.bin = k.bin == null || k.bin === 0 ? null : k.bin - 1; break;
      case 'PageDown': k.bin = Math.min(G_.G - 1, (k.bin == null ? 0 : k.bin) + 10); break;
      case 'PageUp': k.bin = k.bin == null ? null : Math.max(0, k.bin - 10); break;
      case 'Enter': case ' ': {
        const layers = []; for (let l = c * G_.per; l < Math.min(G_.L, (c + 1) * G_.per); l++) layers.push(l);
        const l = pickLayer({ layers, bin: k.bin }); A.sel = l; if (k.bin != null) openZoom(l, k.bin); else drawNet(); layerDetail(l); break;
      }
      case 'Escape': if (!A.zoom) return; A.zoom = null; $('anZoom').classList.add('hidden'); break;
      default: return;
    }
    e.preventDefault(); drawNet(); const t = cursorText(); $('anNet').title = t; PSI.canvasLabel($('anNet'), `The network. ${t}.`);
  });
  $('anNet').addEventListener('focus', () => { if (A.st) { if (!A.kcur) A.kcur = { layer: A.sel != null ? A.sel : 0, bin: null }; drawNet(); PSI.canvasLabel($('anNet'), `The network. ${cursorText()}. Arrow keys move.`); } });
  $('anNet').addEventListener('blur', () => { if (A.st) drawNet(); });

  // ── 2b. zoom: every neuron in one bin, over the whole recording ─────────
  async function openZoom(layer, bin) {
    A.zoom = { layer, bin, data: null }; drawNet();
    const box = $('anZoom'); box.classList.remove('hidden'); $('anZoomTitle').textContent = `Layer ${layer} · bin ${bin}`; stat('anZoomStat', 'reading the neurons…', 'busy');
    try {
      const bins = A.rec ? A.rec.bins : A.bins;
      const d = await call({ action: 'zoom', layer, bin, bins }); if (!A.zoom || A.zoom.layer !== layer || A.zoom.bin !== bin) return;
      A.zoom.data = d; A.zoom.stale = !A.rec || (d.steps != null && d.steps !== A.rec.steps.length);
      stat('anZoomStat', `${d.to - d.from} real neurons, #${d.from} to #${d.to - 1}` + (d.acts ? ` · activations at all ${d.steps} steps of the recording` : ' · record a forward pass to see them fire'));
      drawZoom();
    } catch (e) { err(e); failed('anZoomStat', 'Reading the neurons', e); }
  }
  function zoomOrder() {
    const d = A.zoom.data, n = d.to - d.from, idx = [...Array(n).keys()];
    if (A.zsort && d.acts) { const row = d.acts[Math.min(A.step, d.acts.length - 1)]; idx.sort((a, b) => Math.abs(row[b]) - Math.abs(row[a])); }
    return idx;
  }
  function drawZoom() {
    const z = A.zoom; if (!z || !z.data) return; const d = z.data, n = d.to - d.from, order = zoomOrder();
    const rowH = n > 120 ? 2 : n > 60 ? 3 : 5, h = Math.max(120, n * rowH + 40);
    const { g, r, W, H } = canvas($('anZoomCv'), h), left = 60 * r, barW = 120 * r, right = W - barW - 20 * r, top = 22 * r, bot = H - 16 * r, rh = (bot - top) / n;
    const S = d.acts ? d.acts.length : 1, cw = (right - left) / S;
    let mx = 1e-9; if (d.acts) d.acts.forEach((row) => row.forEach((v) => { const a = Math.abs(v); if (a > mx) mx = a; }));
    if (d.acts) {
      for (let t = 0; t < S; t++) for (let k = 0; k < n; k++) {
        const v = d.acts[t][order[k]] / mx, a = Math.min(1, Math.abs(v) * 1.6);
        g.fillStyle = v >= 0 ? `rgba(255,195,107,${(0.04 + 0.95 * a).toFixed(3)})` : `rgba(150,170,255,${(0.04 + 0.95 * a).toFixed(3)})`;
        g.fillRect(left + t * cw, top + k * rh, Math.max(1, cw - (cw > 3 * r ? 0.5 * r : 0)), Math.max(1, rh - (rh > 3 * r ? 0.5 * r : 0)));
      }
      const pl = A.rec ? A.rec.prompt_len : 0; g.strokeStyle = 'rgba(237,235,228,.35)'; g.lineWidth = r; g.beginPath(); g.moveTo(left + pl * cw, top - 4 * r); g.lineTo(left + pl * cw, bot + 2 * r); g.stroke();
      const t = Math.min(A.step, S - 1); g.strokeStyle = '#fff'; g.lineWidth = 1.5 * r; g.strokeRect(left + t * cw, top - 2 * r, Math.max(2 * r, cw), bot - top + 4 * r);
      // this step's activations as bars
      const row = d.acts[t]; for (let k = 0; k < n; k++) { const v = row[order[k]] / mx, bw = Math.abs(v) * (barW / 2); g.fillStyle = v >= 0 ? AMBER : '#96aaff'; g.fillRect(right + 20 * r + barW / 2 + (v < 0 ? -bw : 0), top + k * rh, Math.max(0.5, bw), Math.max(1, rh - 0.3 * r)); }
      g.strokeStyle = 'rgba(237,235,228,.3)'; g.beginPath(); g.moveTo(right + 20 * r + barW / 2, top); g.lineTo(right + 20 * r + barW / 2, bot); g.stroke();
    } else {
      let wm = 1e-9; d.write_norm.forEach((v) => { if (v > wm) wm = v; });
      for (let k = 0; k < n; k++) { g.fillStyle = `rgba(${INK},${(0.1 + 0.85 * d.write_norm[order[k]] / wm).toFixed(3)})`; g.fillRect(left, top + k * rh, (right - left) * d.write_norm[order[k]] / wm, Math.max(1, rh - 0.3 * r)); }
    }
    g.fillStyle = '#7d818c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'right';
    const lab = Math.max(1, Math.ceil(12 * r / rh)); for (let k = 0; k < n; k += lab) g.fillText(`#${d.from + order[k]}`, left - 6 * r, top + k * rh + Math.min(rh, 9 * r));
    g.textAlign = 'left'; g.fillText(d.acts ? `steps → (prompt | written)${A.zsort ? ' · sorted by this step' : ''}` : 'how strongly each neuron can write (no recording yet)', left, 12 * r);
    if (d.acts) { g.textAlign = 'center'; g.fillText('this step', right + 20 * r + barW / 2, 12 * r); }
    z.geo = { left, right, top, rh, n, order, r };
    if (z.krow != null && document.activeElement === $('anZoomCv')) { g.save(); g.strokeStyle = AMBER; g.lineWidth = 1.5 * r; g.setLineDash([4 * r, 3 * r]); g.strokeRect(left - 58 * r, top + Math.min(z.krow, n - 1) * rh - r, W - left + 56 * r - 4 * r, Math.max(3 * r, rh + 2 * r)); g.restore(); }
    PSI.canvasLabel($('anZoomCv'), `Layer ${z.layer}, bin ${z.bin}: ${n} neurons, #${d.from} to #${d.to - 1}${d.acts ? `, across ${S} steps of the recording` : ', how strongly each can write (no recording yet)'}. Keyboard: Up and Down choose a neuron, Enter explains it${d.acts ? ', Left and Right step through the recording' : ''}.`);
  }
  $('anZoomCv').addEventListener('click', (e) => {
    const z = A.zoom; if (!z || !z.geo) return; const rc = $('anZoomCv').getBoundingClientRect(), y = (e.clientY - rc.top) * z.geo.r, x = (e.clientX - rc.left) * z.geo.r;
    const k = Math.floor((y - z.geo.top) / z.geo.rh); if (k < 0 || k >= z.geo.n) return;
    if (z.data.acts && x >= z.geo.left && x < z.geo.right && A.rec) { const t = Math.floor((x - z.geo.left) / ((z.geo.right - z.geo.left) / z.data.acts.length)); if (t >= 0 && t < A.rec.steps.length) { stop(); A.step = t; render(); } }
    neuronCard(z.layer, z.data.from + z.geo.order[k]);
  });
  $('anZoomCv').addEventListener('mousemove', (e) => {
    const z = A.zoom; if (!z || !z.geo) return; const rc = $('anZoomCv').getBoundingClientRect(), y = (e.clientY - rc.top) * z.geo.r;
    const k = Math.floor((y - z.geo.top) / z.geo.rh); if (k < 0 || k >= z.geo.n) { $('anZoomCv').title = ''; return; }
    const j = z.geo.order[k], v = z.data.acts ? z.data.acts[Math.min(A.step, z.data.acts.length - 1)][j] : null;
    $('anZoomCv').title = `neuron #${z.data.from + j}${v != null ? ` · ${v} at this step` : ''} · click for what it does`;
  });
  $('anZoomCv').tabIndex = 0;
  $('anZoomCv').addEventListener('keydown', (e) => {
    const z = A.zoom; if (!z || !z.geo) return; const n = z.geo.n;
    if (e.key === 'ArrowDown') z.krow = z.krow == null ? 0 : Math.min(n - 1, z.krow + 1);
    else if (e.key === 'ArrowUp') z.krow = z.krow == null ? 0 : Math.max(0, z.krow - 1);
    else if (e.key === 'Home') z.krow = 0; else if (e.key === 'End') z.krow = n - 1;
    else if (e.key === 'Enter' || e.key === ' ') { if (z.krow == null) z.krow = 0; neuronCard(z.layer, z.data.from + z.geo.order[z.krow]); }
    else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && A.rec) { stop(); A.step = Math.max(0, Math.min(A.rec.steps.length - 1, A.step + (e.key === 'ArrowRight' ? 1 : -1))); render(); }
    else if (e.key === 'Escape') { $('anZoomClose').click(); $('anNet').focus(); e.preventDefault(); return; }
    else return;
    e.preventDefault(); drawZoom();
    const j = z.geo.order[z.krow || 0]; $('anZoomCv').title = `neuron #${z.data.from + j}: press Enter for what it does`;
  });
  $('anZoomCv').addEventListener('focus', () => { if (A.zoom && A.zoom.krow == null) A.zoom.krow = 0; drawZoom(); });
  $('anZoomCv').addEventListener('blur', () => drawZoom());
  $('anZoomSort').addEventListener('change', () => { A.zsort = $('anZoomSort').checked; drawZoom(); });
  $('anZoomClose').addEventListener('click', () => { A.zoom = null; $('anZoom').classList.add('hidden'); drawNet(); });
  async function neuronCard(layer, j) {
    const box = $('anNeuron'); box.textContent = ''; box.appendChild(el('p', 'small muted', `reading neuron #${j}…`));
    try {
      const d = await call({ action: 'neuron', layer, neuron: j }); box.textContent = '';
      box.appendChild(el('p', 'k', `Layer ${layer} · neuron #${j}`));
      if (!Object.keys(NAMES).length) await loadNames();
      const nm = NAMES[nkey(layer, j)]; if (nm) { const p0 = el('p', 'small'); p0.append(`Name: “${nm.name}” `, nm.source === 'you' ? el('span', 'nm-tag nm-you', 'yours') : PSI.term('drafted', 'drafted name'), ` ${nm.source === 'you' ? '' : `by Prometheus, a hypothesis (confidence ${nm.confidence}). `}${nm.why}${nm.legacy ? ` (${LEGACY})` : ''}`); box.appendChild(p0); }
      const wr = el('p', 'small'); wr.append(`When it fires it writes a vector of size ${d.write_norm} into the `, PSI.term('residual stream'), '. Read straight out as words, that vector pushes the next token toward:'); box.appendChild(wr);
      box.appendChild(el('p', 'an-chips', d.promotes.map(([t, v]) => `“${show(t)}” ${v}`).join('  ·  ')));
      box.appendChild(el('p', 'small', 'and away from:'));
      box.appendChild(el('p', 'an-chips', d.suppresses.map(([t, v]) => `“${show(t)}” ${v}`).join('  ·  ')));
      if (d.fed_by) box.appendChild(el('p', 'small', `Fed most by layer ${layer - 1} neurons: ${d.fed_by.map(([i, w]) => `#${i} (${w > 0 ? '+' : ''}${w})`).join(', ')}`));
      if (d.feeds) box.appendChild(el('p', 'small', `Feeds most into layer ${layer + 1} neurons: ${d.feeds.map(([i, w]) => `#${i} (${w > 0 ? '+' : ''}${w})`).join(', ')}`));
      if (d.acts && A.rec) {
        const top = d.acts.map((v, t) => [Math.abs(v), v, t]).sort((a, b) => b[0] - a[0]).slice(0, 5);
        box.appendChild(el('p', 'small', 'It fired hardest while taking in: ' + top.map(([, v, t]) => `“${show(A.rec.pieces[A.rec.steps[t].pos])}” (${v})`).join(', ')));
      }
      box.appendChild(el('p', 'small muted', 'Direct readouts skip the layers after this one, so treat the words as a hint, not a label.'));
      box.appendChild(el('p', 'k', 'where it fires hardest in ordinary text'));
      if (d.examples && d.examples.length) {
        box.appendChild(el('p', 'small muted', 'The strongest moments from a sample of 2,400 Wikipedia passages. The highlighted piece is where it fired; the text before it is what the model had read. Usually the most reliable clue to what a neuron responds to.'));
        const ul = el('div', 'an-ex'); d.examples.forEach((x) => { const r = el('p', 'an-exrow'); r.appendChild(el('span', 'an-exv', x.act.toFixed(1))); r.appendChild(el('span', 'muted', show(x.before))); r.appendChild(el('mark', null, show(x.token))); r.appendChild(el('span', 'muted', show(x.after))); ul.appendChild(r); }); box.appendChild(ul);
      } else {
        box.appendChild(el('p', 'small muted', 'Not built yet for this model. One pass over 2,400 Wikipedia passages finds the strongest examples for every neuron at once. It takes under a minute on a GPU, the engine is busy meanwhile, and the examples are saved on disk for this model, so every later neuron card reads them.'));
        if (window.PSIExp && PSIExp.changed()) box.appendChild(el('p', 'small chg-mark', `The model is changed right now (${PSIExp.changed()}): examples built now would describe the changed model and be kept as if they were the model’s own. Restore it in Tipping point first.`));
        const b = el('button', 'ghost', 'Find examples for every neuron'); b.addEventListener('click', async () => { b.disabled = true; b.textContent = 'reading 2,400 passages…'; try { const r = await call({ action: 'build_examples' }); b.textContent = `done: ${r.tokens.toLocaleString()} tokens in ${r.seconds}s`; neuronCard(layer, j); } catch (e) { err(e); b.disabled = false; b.textContent = 'Find examples for every neuron'; } }); box.appendChild(b);
      }
    } catch (e) { err(e); box.textContent = ''; }
  }

  // ── 3. one layer, in detail ─────────────────────────────────────────────
  async function layerDetail(l) {
    $('anLayerTitle').textContent = `Layer ${l}`;
    if (!A.maps[l]) { stat('anLayerStat', 'reading the weights…', 'busy'); try { A.maps[l] = await call({ action: 'layer', layer: l }); } catch (e) { failed('anLayerStat', `Reading layer ${l}’s weights`, e); return err(e); } }
    stat('anLayerStat', '');
    const box = $('anMaps'); box.textContent = '';
    Object.entries(A.maps[l].maps).filter(([k]) => k !== 'head_out_norm').forEach(([k, m]) => {
      const card = el('div', 'an-map'); card.appendChild(el('p', 'k', `${k} · ${m.shape.join('×')}`)); const cv = el('canvas'); card.appendChild(cv); card.appendChild(el('p', 'small muted', `norm ${m.norm} · max |w| ${m.max_abs}`)); box.appendChild(card);
      const n = m.map.length, mm = m.map[0].length, r = Math.min(2, devicePixelRatio || 1); cv.width = 128 * r; cv.height = Math.round(128 * r * n / mm); cv.style.width = '128px'; cv.style.height = `${Math.round(128 * n / mm)}px`;
      const g = cv.getContext('2d'), cw = cv.width / mm, ch = cv.height / n;
      m.map.forEach((row, i) => row.forEach((v, j) => { g.fillStyle = `rgba(${INK},${(v * 0.95 + 0.03).toFixed(3)})`; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }));
    });
    stepDetail();
  }
  function stepDetail() {
    const l = A.sel, box = $('anStepLayer'); box.textContent = ''; if (l == null || !A.rec) return;
    const st = A.rec.steps[A.step], lay = st.layers[l], P = A.rec.pieces;
    const p1 = el('p', 'small'); p1.append(`At this step, layer ${l} would predict “${show(lay.lens[0])}” (${Math.round(lay.lens[1] * 100)}%, read with the `, PSI.term('logit lens'), `). Residual stream size ${lay.resid}.`); box.appendChild(p1);
    box.appendChild(el('p', 'small', 'Strongest neurons firing: ' + lay.top.map(([j, v]) => `#${j} (${v})`).join(', ')));
    const hl = lay.look.map((p, h) => [h, p, lay.ent[h]]).sort((a, b) => a[2] - b[2]).slice(0, 6);
    const p2 = el('p', 'small'); p2.append('Most focused ', PSI.term('attention heads', 'attention head'), ' look at: ' + hl.map(([h, p]) => `H${h} → “${show(P[p])}”`).join(' · ') + (lay.sink != null ? ` (ignoring the first token, where ${Math.round(lay.sink * 100)}% of this layer’s attention parks unused)` : '')); box.appendChild(p2);
  }

  // ── 3b. activation patching ─────────────────────────────────────────────
  const PA = { d: null };
  // Built-in examples: the two prompts of a pair must split into the same number of pieces, which depends on the model's
  // vocabulary (the Eiffel Tower pair does in Qwen3, not in Granite). Still on an example: try the next one that splits evenly.
  const PA_PAIRS = [['The Eiffel Tower is located in the city of', 'The Colosseum is located in the city of'],
    ['The capital of France is the city of', 'The capital of Italy is the city of'],
    ['Paris is the capital of the country of', 'Rome is the capital of the country of'],
    ['The doctor said that she would see the patient in', 'The lawyer said that she would see the client in']];
  const isExample = (c, x) => PA_PAIRS.some(([a, b]) => a === c && b === x);
  $('paRun').addEventListener('click', async () => {
    if (!$('paClean').value.trim() || !$('paCorrupt').value.trim()) return stat('paStat', 'Write both prompts first.', 'warn');
    const b = $('paRun'); b.disabled = true; stat('paStat', 'patching every layer at every word…', 'busy'); $('paSummary').textContent = '';
    let switched = null;
    try {
      const c0 = $('paClean').value.trim(), x0 = $('paCorrupt').value.trim();
      const tries = isExample(c0, x0) ? [[c0, x0], ...PA_PAIRS.filter(([a, z]) => a !== c0 || z !== x0)] : [[c0, x0]];
      for (let i = 0; i < tries.length; i++) {
        try { PA.d = await call({ action: 'patch', clean: tries[i][0], corrupt: tries[i][1] }); if (i) { switched = tries[i]; $('paClean').value = tries[i][0]; $('paCorrupt').value = tries[i][1]; } break; }
        catch (e) { if (i === tries.length - 1 || !/same number of pieces/.test(String(e.message || e))) throw e; }
      }
      stat('paStat', switched ? 'done · the first example splits unevenly in this model’s vocabulary, so this one was used' : 'done', 'ok'); mark('paStat'); cond('paStat', PA.d); drawPatch();
    }
    catch (e) { err(e); failed('paStat', 'Patching', e); } finally { b.disabled = false; }
  });
  $('paKind').addEventListener('change', drawPatch);
  function drawPatch() {
    const d = PA.d; if (!d) return; const kind = $('paKind').value, G = d[kind], L = G.length, T = G[0].length;
    const { g, r, W, H } = canvas($('paCv'), Math.max(260, L * 9 + 70)), left = 40 * r, right = W - 12 * r, top = 12 * r, bot = H - 52 * r, cw = (right - left) / T, ch = (bot - top) / L;
    let best = [0, 0, -1];
    for (let l = 0; l < L; l++) for (let t = 0; t < T; t++) {
      const v = G[l][t]; if (v > best[2]) best = [l, t, v];
      g.fillStyle = v >= 0 ? `rgba(255,195,107,${(0.04 + 0.96 * Math.min(1, v)).toFixed(3)})` : `rgba(150,170,255,${(0.04 + 0.9 * Math.min(1, -v)).toFixed(3)})`;
      g.fillRect(left + t * cw, top + l * ch, Math.max(1, cw - 0.8 * r), Math.max(1, ch - 0.6 * r));
    }
    g.fillStyle = '#7d818c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'right';
    for (let l = 0; l < L; l += 5) g.fillText(`L${l}`, left - 6 * r, top + (l + 0.8) * ch);
    g.save(); g.textAlign = 'right';
    d.pieces_clean.forEach((p, t) => { g.save(); g.translate(left + (t + 0.6) * cw, bot + 8 * r); g.rotate(-Math.PI / 5); g.fillStyle = d.pieces_clean[t] !== d.pieces_corrupt[t] ? '#ffc36b' : '#a7abb6'; g.fillText(show(p), 0, 0); g.restore(); });
    g.restore();
    // where it peaks: on the changed words, and on the last word
    const changed = d.pieces_clean.map((p, t) => p !== d.pieces_corrupt[t] ? t : -1).filter((t) => t >= 0), last = T - 1;
    const peak = (ts) => { let m = [0, -1]; for (let l = 0; l < L; l++) ts.forEach((t) => { if (G[l][t] > m[1]) m = [l, G[l][t]]; }); return m; };
    const [lc, vc] = peak(changed), [ll, vl] = peak([last]);
    const what = kind === 'resid' ? 'the stream' : kind === 'mlp' ? 'the MLP output' : 'the attention output';
    let arrive = -1; for (let l = 0; l < L; l++) if (G[l][last] >= 0.5) { arrive = l; break; }
    $('paSummary').textContent = `First prompt’s answer “${show(d.clean_answer)}” (${Math.round(d.p_clean * 100)}%), second prompt’s “${show(d.corrupt_answer)}” (${Math.round(d.p_corrupt * 100)}%). On the changed name, copying ${what} restores the most at layer ${lc}, bringing back ${Math.round(vc * 100)}% of the gap between the two answers: that is where the fact sits. ` +
      (arrive >= 0 ? `On the last word, where the answer is written, copying restores half or more from layer ${arrive} on: that is when the fact has been moved there.` : `On the last word copying never restores half (best ${Math.round(vl * 100)}% at layer ${ll}), so in ${what} alone the fact is not carried there.`);
    PSI.canvasLabel($('paCv'), `Heat map, ${L} layers by ${T} words: how much copying ${what} from the first prompt restores its answer. Strongest on the changed name at layer ${lc} (${Math.round(vc * 100)}%), on the last word at layer ${ll} (${Math.round(vl * 100)}%).`);
  }

  // ── 3c. light up a concept ──────────────────────────────────────────────
  let NAMES = {};
  // Names are kept per weights (name@hash), so two checkpoints in folders of the same name never share them. Names saved
  // before that, under the folder name alone, are still shown, marked: they may belong to another model of that name.
  const nameKey = () => A.nameKey || (A.st ? A.st.name : 'model');
  async function loadNames() {
    try {
      NAMES = (await psi.neuronNames({ model: nameKey() })).names || {};
      if (!Object.keys(NAMES).length && A.st && nameKey() !== A.st.name) {
        const old = (await psi.neuronNames({ model: A.st.name })).names || {};
        Object.values(old).forEach((n) => { n.legacy = true; }); NAMES = old;
      }
    } catch (e) { NAMES = {}; stat('ccNameStat', `Saved neuron names could not be read, so none are shown: ${PSI.explain(e).text}`, 'warn'); }
  }
  const LEGACY = 'saved under the folder name only, before names were tied to the weights: it may belong to another model in a folder of this name';
  const nameTag = (nm) => (!nm ? '' : nm.source === 'you' ? 'yours' : 'drafted') + (nm && nm.legacy ? ', unverified' : '');
  const nkey = (l, j) => `L${l}#${j}`;
  async function tailored(word) {
    const hcfg = (await psi.settings()).helper || {}; const model = hcfg.model || 'anthropic/claude-haiku-4.5';
    const prompt = `Write probe sentences about the concept "${word}" for an experiment that reads a language model's neurons.
Return only JSON: {"facets": {"<facet name>": ["sentence", ...], ...}}
- exactly 6 facets covering different sides of ${word}: what it is, what it is used for, how it works, where it and its name came from, its problems or limits, and what it sits next to. Rename a facet if a more specific one fits ${word} better.
- 4 sentences per facet, 8 to 16 words, plain and factual in tone.
- every sentence contains the exact text "${word}" once, spelled exactly like that.`;
    const r = await psi.external({ action: 'chat', model, messages: [{ role: 'user', content: prompt }], maxTokens: 1400 });
    const m = String(r.text || '').match(/\{[\s\S]*\}/); if (!m) throw new Error('Prometheus did not return sentences.');
    const f = JSON.parse(m[0]).facets || {}; const out = {};
    Object.entries(f).slice(0, 8).forEach(([k, v]) => { const ok = (v || []).filter((x) => String(x).toLowerCase().includes(word.toLowerCase())).slice(0, 5); if (ok.length >= 2) out[String(k).replace(/_/g, ' ').trim()] = ok; });
    if (Object.keys(out).length < 3) throw new Error('Too few usable sentences came back.'); return { facets: out, cost: r.cost };
  }
  $('ccRun').addEventListener('click', async () => {
    const word = $('ccWord').value.trim(); if (!word) return stat('ccStat', 'Type a concept first.', 'warn'); const b = $('ccRun'); b.disabled = true; $('ccSummary').textContent = '';
    try {
      if (!A.st) await loadStructure();
      let facets = null, note = 'generic sentences';
      if ($('ccTailor').checked && !PSI.state.key) note = 'generic sentences (Prometheus needs an OpenRouter key, set in Settings)';
      else if ($('ccTailor').checked) { stat('ccStat', 'Prometheus is writing sentences about it…', 'busy'); try { const t = await tailored(word); facets = t.facets; note = `sentences written by Prometheus ($${(+t.cost || 0).toFixed(4)})`; } catch (e) { note = `generic sentences (Prometheus: ${PSI.explain(e).text})`; } }
      stat('ccStat', 'reading every neuron at the concept’s words…', 'busy');
      const d = await call({ action: 'concept', concept: word, facets, bins: A.rec ? A.rec.bins : A.bins });
      A.conceptFacets = facets;
      A.concept = d; await loadNames(); stat('ccStat', `${d.sentences} sentences, ${note}, ${d.seconds}s`, 'ok'); mark('ccStat'); cond('ccStat', d, [{ t: 'no untrained baseline here', title: 'the concept map is not compared with an untrained twin' }]);
      $('ccOut').classList.remove('hidden'); drawNet(); drawConcept();
    } catch (e) { err(e); failed('ccStat', 'Lighting up the concept', e); } finally { b.disabled = false; }
  });
  $('ccShow').addEventListener('change', drawNet);
  function drawConcept() {
    const d = A.concept; if (!d) return; const L = d.depth.length;
    const formSum = d.depth.slice(0, 3).reduce((a, b) => a + b, 0), all = d.depth.reduce((a, b) => a + b, 0) || 1;
    const peak = d.depth.indexOf(Math.max(...d.depth.slice(3)));
    const sl = [...d.slices].sort((a, b) => b.strength - a.strength);
    $('ccSummary').textContent = `“${d.concept}”: ${d.nodes.length} neurons light up, ${d.core_count} of them core. ${L >= 8 ? `${Math.round(formSum / all * 100)}% of the signal is in layers 0–2, the word’s form; its meaning peaks at layer ${peak}.` : `Across its ${L} layers: ${d.depth.map((v, i) => `layer ${i} ${Math.round(v / all * 100)}%`).join(', ')} (too shallow to tell the word’s form from its meaning).`} Strongest side: ${sl[0].facet}. Thinnest: ${sl[sl.length - 1].facet}.`;
    // depth profile
    const { g, r, W, H } = canvas($('ccDepth'), 150), left = 30 * r, right = W - 10 * r, top = 10 * r, bot = H - 22 * r, cw = (right - left) / L, mx = Math.max(...d.depth) || 1;
    g.fillStyle = 'rgba(237,235,228,.05)'; g.fillRect(left, top, 3 * cw, bot - top);
    d.depth.forEach((v, l) => { const hgt = (bot - top) * v / mx; g.fillStyle = l <= 2 ? 'rgba(150,170,255,.7)' : 'rgba(255,195,107,.85)'; g.fillRect(left + l * cw + 1 * r, bot - hgt, Math.max(1, cw - 2 * r), hgt); });
    g.fillStyle = '#7d818c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center';
    for (let l = 0; l < L; l += 5) g.fillText(`L${l}`, left + (l + 0.5) * cw, H - 8 * r);
    g.textAlign = 'left'; g.fillStyle = '#96aaff'; g.fillText('word form', left + 2 * r, top + 10 * r); g.fillStyle = '#ffc36b'; g.fillText('meaning', left + 4 * cw, top + 10 * r);
    PSI.canvasLabel($('ccDepth'), `Bar chart: “${d.concept}”’s strength in each of ${L} layers; layers 0–2 (the word’s form) in blue, the rest in amber; it peaks at layer ${peak}.`);
    // slices
    const box = $('ccSlices'); box.textContent = ''; const smax = Math.max(...d.slices.map((x) => x.strength)) || 1;
    sl.forEach((x, i) => {
      const c = el('div', 'cc-slice' + (i === sl.length - 1 ? ' thin' : '')); c.appendChild(el('p', 'k', x.facet));
      const bar = el('div', 'cc-bar'); const fill = el('i'); fill.style.width = `${Math.round(100 * x.strength / smax)}%`; bar.appendChild(fill); c.appendChild(bar);
      c.appendChild(el('p', 'small muted', `${x.count} neurons of its own${x.neurons.length ? ': ' + x.neurons.slice(0, 5).map(([l, j]) => { const nm = NAMES[nkey(l, j)]; return nm ? `${nm.name} (${nameTag(nm)})` : `L${l}#${j}`; }).join(', ') : ''}`));
      if (x.sentences[0]) c.appendChild(el('p', 'small cc-sent', `“${x.sentences[0]}”`));
      c.title = x.sentences.join('\n'); box.appendChild(c);
    });
    // neurons
    const tb = $('ccTable').querySelector('tbody'); tb.textContent = '';
    const facetNames = d.slices.map((x) => x.facet);
    d.nodes.slice(0, 60).forEach((n) => {
      const tr = el('tr'); const k = nkey(n.layer, n.neuron), nm = NAMES[k];
      const a = el('button', 'hlp-link', `L${n.layer} #${n.neuron}`); a.type = 'button'; a.addEventListener('click', () => { A.sel = n.layer; neuronCard(n.layer, n.neuron); $('anNeuron').scrollIntoView({ behavior: 'smooth', block: 'center' }); });
      const td0 = el('td'); td0.appendChild(a); tr.appendChild(td0);
      tr.appendChild(el('td', null, n.form ? 'word form' : n.kind)); tr.appendChild(el('td', null, n.kind === 'core' ? 'all sides' : n.facet));
      const td = el('td'); const inp = el('input'); inp.type = 'text'; inp.className = 'cc-name'; inp.value = nm ? nm.name : ''; inp.placeholder = 'unnamed'; inp.id = `ccn-${n.layer}-${n.neuron}`;
      if (nm) inp.title = `${nm.source === 'you' ? 'your name' : 'drafted by Prometheus: a hypothesis'} · confidence ${nm.confidence}${nm.legacy ? ` · ${LEGACY}` : ''}`; if (nm && nm.source === 'you') inp.classList.add('mine'); if (nm && nm.source !== 'you') inp.classList.add('drafted');
      inp.addEventListener('change', async () => {
        try { await psi.neuronNames({ action: 'set', model: nameKey(), entries: [{ layer: n.layer, neuron: n.neuron, name: inp.value.trim(), source: 'you', concept: d.concept }] }); await loadNames(); drawConcept(); }
        catch (e) { err(e); stat('ccNameStat', `Your name for L${n.layer} #${n.neuron} was not saved: ${PSI.explain(e).text}`, 'error'); }
      });
      td.appendChild(inp); if (nm) td.appendChild(el('span', `nm-tag ${nm.source === 'you' ? 'nm-you' : 'nm-drafted'}`, nameTag(nm))); tr.appendChild(td);
      tr.appendChild(el('td', 'small muted', nm ? `${nm.why}${nm.confidence ? ` (${nm.confidence})` : ''}` : ''));
      tr.title = facetNames.map((f, i) => `${f}: ${n.facets[i]}`).join('\n'); tb.appendChild(tr);
    });
  }
  $('ccName').addEventListener('click', async () => {
    const d = A.concept; if (!d) return stat('ccNameStat', 'Light up a concept first.', 'warn'); const b = $('ccName'); b.disabled = true;
    let p = null;
    try {
      const pick = [...d.nodes.filter((n) => n.kind === 'core' && !n.form).slice(0, 8), ...d.nodes.filter((n) => n.kind === 'core' && n.form).slice(0, 2)];
      d.slices.forEach((x) => x.neurons.slice(0, 3).forEach(([l, j]) => { const n = d.nodes.find((m) => m.layer === l && m.neuron === j); if (n && !pick.includes(n)) pick.push(n); }));
      const facetNames = d.slices.map((x) => x.facet), ev = [];
      p = PSI.progress(progHost('ccName'), { total: pick.length + 1, label: 'neuron', track: false });
      for (let i = 0; i < pick.length; i++) {
        if (p.cancelled) { p.finish('stopped: no names drafted', { markDone: false }); stat('ccNameStat', ''); return; }
        const n = pick[i]; p.step(i + 1, 'gathering evidence'); stat('ccNameStat', `gathering evidence ${i + 1} of ${pick.length}…`, 'busy');
        const r = await call({ action: 'neuron', layer: n.layer, neuron: n.neuron });
        ev.push({ id: nkey(n.layer, n.neuron), layer: n.layer, kind: n.form ? 'word form (early layer)' : n.kind, sides: Object.fromEntries(facetNames.map((f, k) => [f, n.facets[k]])),
          fires_on: (r.examples || []).slice(0, 5).map((x) => `…${x.before.slice(-60)}[[${x.token}]]${x.after}`), pushes_toward: (r.promotes || []).slice(0, 6).map((x) => x[0]), pushes_away: (r.suppresses || []).slice(0, 4).map((x) => x[0]) });
      }
      if (p.cancelled) { p.finish('stopped: no names drafted', { markDone: false }); stat('ccNameStat', ''); return; }
      p.step(pick.length + 1, 'Prometheus is drafting names'); stat('ccNameStat', 'Prometheus is drafting names…', 'busy');
      const hcfg = (await psi.settings()).helper || {}; const model = hcfg.model || 'anthropic/claude-haiku-4.5';
      const size = A.st && A.st.params ? `${fmtN(A.st.params).replace(' B', '-billion').replace(' M', '-million').replace(' k', '-thousand')}-parameter ` : '';
      const prompt = `These are MLP neurons from a ${size}language model that respond to the concept "${d.concept}". For each, draft a short name (2 to 5 words) for what it appears to respond to, using only the evidence: "sides" is how strongly it answers "${d.concept}" in sentences about each side of it (higher = stronger), "fires_on" is where it fires hardest in ordinary Wikipedia text (the firing token is in [[ ]]), and the push lists are words its output favours or disfavours. Many neurons respond to several unrelated things; say so when they do. If the evidence does not agree, name it "unclear" with low confidence. Do not invent meaning the evidence does not show.
Return only JSON: [{"id": "...", "name": "...", "why": "at most 14 words citing the evidence", "confidence": "low|medium|high"}]
Evidence:
${JSON.stringify(ev)}`;
      const r = await psi.external({ action: 'chat', model, messages: [{ role: 'user', content: prompt }], maxTokens: 3000 });
      const m = String(r.text || '').match(/\[[\s\S]*\]/); if (!m) throw new Error('No names came back.');
      const out = JSON.parse(m[0]).map((x) => { const [, l, j] = String(x.id).match(/L(\d+)#(\d+)/) || []; return l ? { layer: +l, neuron: +j, name: x.name, why: x.why, confidence: x.confidence, source: 'prometheus', concept: d.concept } : null; }).filter(Boolean);
      if (p.cancelled) { p.finish('stopped: the drafted names were not saved', { markDone: false }); stat('ccNameStat', ''); return; }
      await psi.neuronNames({ action: 'set', model: nameKey(), entries: out }); await loadNames(); drawConcept();
      p.remove(); stat('ccNameStat', `${out.length} names drafted ($${(+r.cost || 0).toFixed(4)}). Hover a row for its evidence by side.`, 'ok');
    } catch (e) { err(e); if (p) p.fail('stopped by an error'); failed('ccNameStat', 'Drafting names', e); } finally { b.disabled = false; }
  });

  $('ccKO').addEventListener('click', async () => {
    const d = A.concept; if (!d) return stat('ccKOStat', 'Light up a concept first.', 'warn'); const out = $('ccKOOut'); out.textContent = '';
    // Its restores between rounds would remove a Tipping-point cut, so the engine refuses while one is in force; say so first.
    if (window.PSIExp && PSIExp.changed()) return stat('ccKOStat', `A Tipping-point cut is in force (${PSIExp.changed()}). Delete what is lit restores the model between rounds, which would remove it. Restore the model in Tipping point first, then run this again.`, 'warn');
    const b = $('ccKO'); b.disabled = true; const rounds = +$('ccRounds').value; stat('ccKOStat', '');
    try {
      const r = await longCall('ccKO', { text: 'deleting and looking again, then the random control and the floor', total: rounds + 2, est: 20 + 14 * rounds, label: 'stage', doneText: `done · ${rounds} round(s) + controls` },
        () => call({ action: 'concept_knockout', concept: d.concept, facets: A.conceptFacets || null, rounds, bins: d.bins, controls: 3 }));
      if (!r) return;
      cond('ccKOStat', r, [{ t: 'zero-ablation, undone after', term: 'zero-ablation' }, { t: 'no untrained baseline here' }]);
      const P = (v) => `${Math.round(v * 100)}%`, PR = ([a, b]) => (P(a) === P(b) ? P(a) : `${P(a)} to ${P(b)}`), RC = r.random_control, FL = r.floor;
      const tbl = el('table', 'cc-ko'); const hd = el('tr'); ['', 'neurons deleted', 'concept signal left', 'new carriers', 'its answer'].forEach((x) => hd.appendChild(el('th', null, x))); tbl.appendChild(hd);
      const row = (a, b2, c, dd, e, cls) => { const tr = el('tr', cls); [a, b2, c, dd].forEach((x) => tr.appendChild(el('td', null, x))); tr.appendChild(el('td', 'cc-ans', e)); tbl.appendChild(tr); };
      row('intact', '0', '100%', '', r.answer_before);
      const span = (ls) => ls.length ? `layers ${ls[0]}–${ls[ls.length - 1]}` : '';
      r.rounds.forEach((x) => row(`round ${x.round}`, x.deleted_total.toLocaleString(), P(x.remaining_signal), `${x.after.nodes} lit, ${span(x.new_carriers_by_layer)}`, x.answer));
      if (RC && RC.range) row(`control: ${RC.seeds.length} random deletions`, `${RC.deleted.toLocaleString()} each, same layers`, PR(RC.range), '', RC.seeds[0] && RC.seeds[0].answer ? `seed 0: ${RC.seeds[0].answer}` : '', 'ctrl');
      if (FL && FL.range) row(`floor: unrelated words (${FL.words.filter((w) => w.ratio != null).map((w) => `“${w.word}”`).join(', ')})`, '0 (intact model)', PR(FL.range), '', 'what a re-picked set of the same size finds for words that are not the concept', 'ctrl');
      const de = r.direction_erased; row('its direction removed', `0 (layers ${de.layers[0]}–${de.layers[1]})`, '', '', de.answer, 'dir');
      row('control, direction removed', '', '', '', `${de.control_question} → ${de.control_after}`, 'dir');
      const wrap = el('div', 'scroll'); wrap.appendChild(tbl); out.appendChild(wrap);
      const last = r.rounds[r.rounds.length - 1], R = last.remaining_signal, same = (x) => x.toLowerCase().includes(d.concept.toLowerCase());
      const share = `${(last.deleted_total / (A.st.mlp * A.st.layers) * 100).toFixed(2)}% of all MLP neurons`;
      // The conclusion follows the two baselines, not a fixed story. tol: differences smaller than this are not read.
      const tol = 0.05, p = el('p', 'small');
      p.append(`After deleting ${last.deleted_total.toLocaleString()} neurons (${share}), ${P(R)} of the concept’s signal is left. `);
      const verdict = el('b');
      if (!RC || !RC.range || !FL || !FL.range) verdict.textContent = 'The engine returned no random control or floor (an older trace_server.py), so this cannot say whether the deletion removed more than any deletion of that size would, nor what level counts as gone.';
      else {
        const [lo] = RC.range, fhi = FL.range[1];
        if (R >= lo - tol) verdict.textContent = `Random deletions of the same size left ${P(lo)} or more: deleting what was lit did no more than deleting any neurons. This test cannot locate the concept.`;
        else if (R <= fhi + tol) verdict.textContent = `That is down to the floor (${P(fhi)} at most for unrelated words), while random deletions left ${P(lo)} or more: by this measure the concept was held in the deleted neurons.`;
        else verdict.textContent = `Less than random deletions left (${P(lo)} or more), so the lit neurons did carry it, but still above the floor for unrelated words (${P(fhi)} at most), carried by neurons that were not lit before: the concept is not held in a set of nodes.`;
      }
      p.appendChild(verdict);
      p.append(` ${same(last.answer) ? 'Its answer still names it; read the answers in the table to judge whether the meaning survived.' : 'Its answer no longer names it.'} Removing the concept’s main direction ${same(de.answer) ? 'leaves it named' : 'changes the answer'}, while the control question ${de.control_after.toLowerCase().includes(d.concept.toLowerCase()) ? 'is also affected, so that removal was not specific' : 'is unaffected'}.`);
      out.appendChild(p);
      out.appendChild(el('p', 'small muted', `Signal: the summed score of a re-picked top-${FL && FL.k ? FL.k : 240} set, as a share of the intact concept’s. A re-picked set always finds something, which is why the floor, not 0%, is the level of “gone”. ${RC && RC.seeds ? `${RC.seeds.length} random seeds: a narrow sample of what deletion of that size does.` : ''}`));
      A.concept = Object.assign({}, A.concept, { light: last.after.light }); $('ccShow').checked = true; drawNet();
      out.appendChild(el('p', 'small muted', 'The network above now shows where the concept is carried after the last round: the backups.'));
    } catch (e) { err(e); failed('ccKOStat', 'The knock-out rounds', e); } finally { b.disabled = false; }   // the engine restores the model even on an error
  });

  // ── 3d. map what it understood ──────────────────────────────────────────
  async function drafts(word) {
    const hcfg = (await psi.settings()).helper || {}; const model = hcfg.model || 'anthropic/claude-haiku-4.5';
    const prompt = `Write 8 short fact checks about "${word}" for testing whether a small language model knows it. Each prompt is a plain sentence that stops right before a single-word answer (no blank, no question), covering different sides: what it is, how it works, where it or its name came from, what it is used for, numbers or physical facts. The answer is one common English word.
Return only JSON: [{"facet": "...", "prompt": "...", "answer": "..."}]`;
    const r = await psi.external({ action: 'chat', model, messages: [{ role: 'user', content: prompt }], maxTokens: 900 });
    const m = String(r.text || '').match(/\[[\s\S]*\]/); if (!m) return { probes: [], cost: r.cost };
    return { probes: JSON.parse(m[0]).filter((x) => x.prompt && x.answer && !/\s/.test(String(x.answer).trim())).slice(0, 10), cost: r.cost };
  }
  $('cnRun').addEventListener('click', async () => {
    const word = $('ccWord').value.trim(); if (!word) return stat('cuStat', 'Type a concept in the box above (“concept”) first.', 'warn'); const b = $('cnRun'); b.disabled = true;
    try {
      let probes = [], note = '';
      if ($('cnChecks').checked && !PSI.state.key) note = ' · no fact checks (Prometheus needs an OpenRouter key, set in Settings)';
      else if ($('cnChecks').checked) { stat('cuStat', 'Prometheus is writing fact checks…', 'busy'); try { const d = await drafts(word); probes = d.probes; note = ` · fact checks by Prometheus ($${(+d.cost || 0).toFixed(4)})`; } catch (e) { note = ` · no fact checks (${PSI.explain(e).text})`; } }
      stat('cuStat', 'asking it, and reading where the concept sits among ~200 words…', 'busy');
      const isColor = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'grey', 'cyan', 'magenta', 'violet', 'crimson', 'scarlet', 'teal', 'navy', 'gold', 'maroon', 'lime', 'turquoise', 'beige', 'olive'].includes(word.toLowerCase());
      A.net = await call({ action: 'concept_net', concept: word, facets: (A.concept && A.concept.concept === word && A.conceptFacets) || null, probes });
      A.color = isColor ? await call({ action: 'color_geometry', focus: word.toLowerCase() }) : null;
      stat('cuStat', `${A.net.seconds}s${note}`, 'ok'); mark('cuStat'); cond('cuStat', A.net, [{ t: 'no untrained baseline here' }]); $('cnOut').classList.remove('hidden'); drawNetMap();
    } catch (e) { err(e); failed('cuStat', 'Mapping what it understood', e); } finally { b.disabled = false; }
  });
  function drawNetMap() {
    const d = A.net; if (!d) return;
    const pool = d.pool, saidR = d.said_rank || {}, far = Object.entries(saidR).filter(([, r]) => r > pool / 2).map(([w]) => w);
    const top = d.nodes.filter((n) => n.source.includes('inside')).slice(0, 3).map((n) => n.word);
    $('cnSummary').textContent = !d.said.length ? `Asked about “${d.concept}”, it used none of the comparison words to explain it, so there is nothing to compare. Inside, unprompted, it sits closest to ${top.join(', ')}.` : `Asked about “${d.concept}”, it explains it with ${d.said.slice(0, 5).join(', ')}. Inside, unprompted, it sits closest to ${top.join(', ')}. ${far.length ? `${far.length} of the words it says sit in the far half of ${pool} words inside (${far.slice(0, 4).join(', ')}): it can say them, but they are not where the concept lives.` : 'What it says and where it sits mostly agree.'}`;
    // the net: radial, distance = 1 - similarity
    const { g, r, W, H } = canvas($('cnNet'), 440), cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 46 * r;
    const ns = d.nodes, sims = ns.slice(1).map((n) => n.sim), lo = Math.min(...sims), hi = Math.max(...sims);
    const pos = ns.map((n, i) => { if (i === 0) return [cx, cy]; const t = hi > lo ? (n.sim - lo) / (hi - lo) : 0.5; const rad = R * (0.5 + 0.5 * (1 - t)); const ang = (i - 1) / (ns.length - 1) * Math.PI * 2 - Math.PI / 2; return [cx + rad * Math.cos(ang), cy + rad * Math.sin(ang) * 0.86]; });
    d.edges.forEach(([a2, b2, w]) => { const A2 = pos[a2], B2 = pos[b2]; if (!A2 || !B2) return; g.strokeStyle = `rgba(237,235,228,${(0.05 + 0.4 * Math.min(1, (w - 0.3) / 0.4)).toFixed(3)})`; g.lineWidth = r; g.beginPath(); g.moveTo(A2[0], A2[1]); g.lineTo(B2[0], B2[1]); g.stroke(); });
    ns.slice(1).forEach((n, k) => { const [x, y] = pos[k + 1]; g.strokeStyle = 'rgba(255,195,107,.18)'; g.lineWidth = r; g.beginPath(); g.moveTo(cx, cy); g.lineTo(x, y); g.stroke(); });
    const LABELS = [];
    ns.forEach((n, i) => {
      const [x, y] = pos[i], said = n.source.includes('said'), ins = n.source.includes('inside'), rad = (i === 0 ? 9 : 5.5) * r;
      g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2);
      g.fillStyle = i === 0 ? '#ffc36b' : ins ? 'rgba(157,180,255,.9)' : 'rgba(18,18,22,1)'; g.fill();
      if (said) { g.strokeStyle = '#ffc36b'; g.lineWidth = 2 * r; g.stroke(); }
      g.fillStyle = i === 0 ? '#ffc36b' : '#edebe4'; g.font = `${(i === 0 ? 13 : 11) * r}px "Public Sans", sans-serif`; g.textAlign = x < cx - 4 ? 'right' : x > cx + 4 ? 'left' : 'center';
      // labels never overlap: nudge each one down until it clears every label already placed
      let lx = x + (g.textAlign === 'left' ? 9 * r : g.textAlign === 'right' ? -9 * r : 0), ly = y + (i === 0 ? -14 * r : 4 * r);
      const tw = g.measureText(n.word).width, th = 12 * r, box = () => { const L0 = g.textAlign === 'left' ? lx : g.textAlign === 'right' ? lx - tw : lx - tw / 2; return [L0, ly - th, L0 + tw, ly]; };
      for (let tries = 0; tries < 12 && LABELS.some((b2) => { const a3 = box(); return a3[0] < b2[2] && a3[2] > b2[0] && a3[1] < b2[3] && a3[3] > b2[1]; }); tries++) ly += (tries % 2 ? -1 : 1) * (tries + 1) * th * 0.55;
      LABELS.push(box()); g.fillText(n.word, lx, ly);
    });
    PSI.canvasLabel($('cnNet'), `Network around “${d.concept}”: ${ns.length - 1} words placed by closeness inside the model; nearest unprompted: ${top.join(', ') || 'none'}; ringed: words it used when asked (${d.said.slice(0, 5).join(', ') || 'none'}).`);
    // lists
    const bar = (label, v, mx, extra) => { const row = el('div', 'cn-row'); row.appendChild(el('span', 'cn-w', label)); const b2 = el('span', 'cn-bar'); const f = el('i'); f.style.width = `${Math.max(2, Math.round(100 * v / mx))}%`; b2.appendChild(f); row.appendChild(b2); row.appendChild(el('span', 'cn-v', extra)); return row; };
    const said = $('cnSaid'); said.textContent = '';
    d.said.forEach((w) => { const rk = saidR[w]; said.appendChild(bar(w, rk ? pool - rk + 1 : 0, pool, rk ? `#${rk} of ${pool} inside` : 'not measured')); });
    const ins = $('cnInside'); ins.textContent = ''; const mx = Math.max(...d.nodes.filter((n) => n.source.includes('inside')).map((n) => n.sim), 1e-6);
    d.nodes.filter((n) => n.source.includes('inside')).forEach((n) => ins.appendChild(bar(n.word, n.sim, mx, n.sim.toFixed(2))));
    $('cnCore').textContent = d.side_vocab && d.side_vocab.core ? `Its core neurons push toward the words: ${d.side_vocab.core.slice(0, 6).join(', ')}.` : '';
    // fact checks
    const co = $('cnChecksOut'); co.textContent = '';
    if (d.checks && d.checks.length) {
      const k = d.checks.filter((c) => c.verdict === 'knows').length;
      co.appendChild(el('p', 'k', `knowing: ${k} of ${d.checks.length} facts completed first try`)); co.appendChild(el('p', 'small muted', 'A strict test: only the exact word counts, so a synonym (“festivals” for “celebrations”) is scored as a miss. “Close” means the right word was among its top five.'));
      const t = el('table', 'cc-ko'); d.checks.forEach((c) => { const tr = el('tr'); tr.appendChild(el('td', null, c.facet)); tr.appendChild(el('td', 'cc-ans', `${c.prompt} … ${c.answer}`)); tr.appendChild(el('td', null, c.verdict === 'knows' ? `knows (${Math.round(c.p * 100)}%)` : c.verdict === 'close' ? `close (rank ${c.rank})` : `does not (expected “${c.it_expected[0]}”)`)); t.appendChild(tr); });
      const w2 = el('div', 'scroll'); w2.appendChild(t); co.appendChild(w2);
    }
    // colour grounding
    const cg = A.color; $('cnColor').classList.toggle('hidden', !cg);
    if (cg) {
      const { g: g2, r: r2, W: W2, H: H2 } = canvas($('cnColorCv'), 320), xs = cg.colors.map((c) => c.x), ys = cg.colors.map((c) => c.y);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 40 * r2;
      cg.colors.forEach((c) => { const x = pad + (c.x - x0) / (x1 - x0 || 1) * (W2 - 2 * pad), y = pad + (c.y - y0) / (y1 - y0 || 1) * (H2 - 2 * pad);
        g2.beginPath(); g2.arc(x, y, (c.word === cg.focus ? 11 : 8) * r2, 0, Math.PI * 2); g2.fillStyle = c.hex; g2.fill(); g2.strokeStyle = c.word === cg.focus ? '#ffc36b' : 'rgba(237,235,228,.35)'; g2.lineWidth = (c.word === cg.focus ? 2.5 : 1) * r2; g2.stroke();
        g2.fillStyle = '#a7abb6'; g2.font = `${10 * r2}px "IBM Plex Mono", monospace`; g2.textAlign = 'center'; g2.fillText(c.word, x, y + 20 * r2); });
      PSI.canvasLabel($('cnColorCv'), `Map of ${cg.colors.length} colour words placed by meaning inside the model and painted with their real colours; rank correlation with real colour ${cg.rho}; “${cg.focus}” sits nearest ${cg.inner_nearest.slice(0, 3).map((x) => x[0]).join(', ')}.`);
      const conds = Object.entries(cg.grid).map(([k, v]) => `${k.replace('/', ' sentences, ')} layers ${v}`).join(' · ');
      $('cnColorTxt').textContent = `The colour words placed by their meaning inside the model (its two main axes), painted with their real colours. How well inner distances match real colour differences: rank correlation ${cg.rho} at its best (${cg.best.replace('/', ' sentences, ')} layers; ${cg.p === 0 ? 'none' : Math.round(cg.p * 500)} of 500 shuffles did as well; this is the best of eight settings). Inside, “${cg.focus}” sits nearest ${cg.inner_nearest.slice(0, 4).map((x) => x[0]).join(', ')}; in real colour it is nearest ${cg.real_nearest.slice(0, 4).map((x) => x[0]).join(', ')}. All conditions: ${conds}. Weak but real: it has rebuilt some of colour’s shape from text alone, more when the words are used as colours.`;
    }
  }

  // ── 4. record and play ──────────────────────────────────────────────────
  PSI.followLens($('anLens'));   // starts as Settings > Think in E-levels says, until changed here
  $('anRecord').addEventListener('click', async () => {
    if (!$('anPrompt').value.trim()) return stat('anRecStat', 'Write a prompt first.', 'warn');
    const b = $('anRecord'); b.disabled = true; stop(); stat('anRecStat', '');
    try {
      const lens = $('anLens').checked && (!window.PSIEM || PSIEM.on), think = $('thinkOn') ? $('thinkOn').checked : false, n = +$('anN').value || 24;
      const rec = await longCall('anRecord', { text: `recording a forward pass (${n} tokens to write)`, doneText: (x) => `done · ${x.seconds}s` }, async () => {
        if (!A.st) await loadStructure();
        if (!A.st) throw new Error('The model’s structure could not be read, so there is nothing to record into.');
        return call({ action: 'record', prompt: $('anPrompt').value, n_new: n, lens, think, bins: A.bins });
      });
      if (!rec) return;
      A.rec = rec; A.step = 0; $('anScrub').max = String(A.rec.steps.length - 1); $('anScrub').value = '0';
      const kv = ((A.rec.pieces.length * A.rec.kv_bytes_per_token) / 1024 / 1024).toFixed(2), rs = $('anRecStat');
      stat('anRecStat', `${A.rec.prompt_len} tokens read, ${A.rec.steps.length - A.rec.prompt_len + 1} written · ${A.rec.seconds}s · `, 'ok');
      rs.append('its ', PSI.term('KV cache'), ` holds ${kv} MB by the end`); mark('anRecStat'); cond('anRecStat', rec, [{ t: lens ? 'E-level framing on' : 'no E-level framing', frame: lens }, { t: 'no untrained baseline here' }]);
      $('anPlayer').classList.remove('hidden'); render(); if (A.zoom) openZoom(A.zoom.layer, A.zoom.bin); play();
    } catch (e) { err(e); failed('anRecStat', 'Recording', e); } finally { b.disabled = false; }
  });
  function render() {
    if (!A.rec) return; const st = A.rec.steps[A.step], P = A.rec.pieces;
    $('anScrub').value = String(A.step);
    $('anStepLabel').textContent = st.phase === 'read' ? `step ${A.step + 1} of ${A.rec.steps.length} · reading “${show(P[st.pos])}” (token ${st.pos + 1} of the prompt)` : `step ${A.step + 1} of ${A.rec.steps.length} · writing: took in “${show(P[st.pos])}”`;
    $('anWrites').textContent = st.writes != null ? `→ next token: “${show(st.writes)}”${st.p != null ? ` (${Math.round(st.p * 100)}%)` : ''}` : '';
    drawNet(); drawTokens(); drawKV(); drawLens(); stepDetail(); drawZoom();
  }
  function drawTokens() {
    const box = $('anTokens'); box.textContent = ''; const st = A.rec.steps[A.step], P = A.rec.pieces;
    const L = st.layers.length, att = new Array(st.pos + 1).fill(0); st.layers.forEach((lay) => lay.row.forEach((v, i) => { if (i <= st.pos) att[i] += v / L; }));
    const mx = Math.max(1e-9, ...att.slice(0, st.pos));
    P.forEach((p, i) => {
      const sp = el('span', 'an-tok', show(p)); if (i > st.pos) sp.classList.add('future');
      else if (i === st.pos) sp.classList.add('cur'); else sp.style.background = `rgba(255,195,107,${(Math.min(1, att[i] / mx) * 0.55).toFixed(3)})`;
      if (i >= A.rec.prompt_len) sp.classList.add('gen'); sp.title = i <= st.pos ? `position ${i} · ${(att[i] * 100).toFixed(1)}% of this step’s attention, averaged over all heads and layers` : 'not yet'; box.appendChild(sp);
    });
  }
  function drawKV() {
    const st = A.rec.steps[A.step], L = st.layers.length, T = st.pos + 1, total = A.rec.pieces.length;
    const { g, r, W, H } = canvas($('anKV'), 230), left = 40 * r, right = W - 10 * r, top = 18 * r, bot = H - 24 * r, cw = (right - left) / total, ch = (bot - top) / L;
    for (let l = 0; l < L; l++) {
      const row = st.layers[l].row; let mx = 1e-9; for (let i = 0; i < T - 1; i++) mx = Math.max(mx, row[i]);
      for (let i = 0; i < T; i++) { const v = i === T - 1 ? 0 : row[i] / mx; g.fillStyle = i === T - 1 ? 'rgba(237,235,228,.55)' : `rgba(255,195,107,${(0.06 + 0.9 * v).toFixed(3)})`; g.fillRect(left + i * cw, top + l * ch, Math.max(1, cw - 0.5 * r), Math.max(1, ch - 0.5 * r)); }
    }
    g.strokeStyle = 'rgba(237,235,228,.35)'; g.lineWidth = r; g.beginPath(); g.moveTo(left + A.rec.prompt_len * cw, top - 4 * r); g.lineTo(left + A.rec.prompt_len * cw, bot + 4 * r); g.stroke();
    g.fillStyle = '#7d818c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'right'; [0, Math.floor(L / 2), L - 1].forEach((l) => g.fillText(`L${l}`, left - 6 * r, top + (l + 0.7) * ch));
    g.textAlign = 'left'; g.fillText(`stored: ${T} tokens × ${L} layers = ${((T * A.rec.kv_bytes_per_token) / 1024 / 1024).toFixed(2)} MB of keys and values`, left, H - 8 * r);
    g.textAlign = 'right'; g.fillText('prompt | written →', right, 12 * r);
    PSI.canvasLabel($('anKV'), `Working memory (KV cache) at step ${A.step + 1}: ${T} stored tokens × ${L} layers; brightness is how much the current token reads from each stored token.`);
  }
  // Each layer's guess is a button for the keyboard too: one Tab stop, arrow keys move, Enter or Space opens that layer.
  function drawLens() {
    const box = $('anLens2'), had = box.contains(document.activeElement), fi = had ? [...box.children].indexOf(document.activeElement) : -1; box.textContent = ''; const st = A.rec.steps[A.step];
    const cur = fi >= 0 ? fi : A.sel != null ? Math.min(A.sel, st.layers.length - 1) : 0;
    st.layers.forEach((lay, l) => {
      const c = el('span', 'an-lens'); c.title = `layer ${l}: ${Math.round(lay.lens[1] * 100)}%`; c.appendChild(el('b', null, `L${l}`)); c.appendChild(document.createTextNode(' ' + show(lay.lens[0]))); c.style.opacity = String(0.35 + 0.65 * lay.lens[1]);
      c.setAttribute('role', 'button'); c.tabIndex = l === cur ? 0 : -1; c.setAttribute('aria-label', `layer ${l} would say “${show(lay.lens[0])}”, ${Math.round(lay.lens[1] * 100)}%; open this layer`);
      c.addEventListener('click', () => { A.sel = l; drawNet(); layerDetail(l); }); box.appendChild(c);
    });
    if (!box._keys) {
      box._keys = true;
      box.addEventListener('keydown', (e) => {
        const all = [...box.children], i = all.indexOf(e.target); if (i < 0) return;
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.target.click(); return; }
        const n = /Right|Down/.test(e.key) ? all[i + 1] : /Left|Up/.test(e.key) ? all[i - 1] : e.key === 'Home' ? all[0] : e.key === 'End' ? all[all.length - 1] : null;
        if (n) { e.preventDefault(); all.forEach((x) => { x.tabIndex = x === n ? 0 : -1; }); n.focus(); }
      });
    }
    if (had && box.children[cur]) box.children[cur].focus({ preventScroll: true });
  }
  function play() { stop(); A.playing = true; $('anPlay').textContent = 'Pause'; A.timer = setInterval(() => { if (A.step >= A.rec.steps.length - 1) return stop(); A.step++; render(); }, Math.max(60, 1000 / (+$('anSpeed').value || 4))); }
  function stop() { A.playing = false; if (A.timer) clearInterval(A.timer); A.timer = null; if ($('anPlay')) $('anPlay').textContent = 'Play'; }
  $('anPlay').addEventListener('click', () => { if (!A.rec) return; if (A.playing) stop(); else { if (A.step >= A.rec.steps.length - 1) A.step = 0; play(); } });
  $('anPrev').addEventListener('click', () => { if (!A.rec) return; stop(); A.step = Math.max(0, A.step - 1); render(); });
  $('anNext').addEventListener('click', () => { if (!A.rec) return; stop(); A.step = Math.min(A.rec.steps.length - 1, A.step + 1); render(); });
  $('anToGen').addEventListener('click', () => { if (!A.rec) return; stop(); A.step = A.rec.prompt_len - 1; render(); });
  $('anScrub').addEventListener('input', () => { stop(); A.step = +$('anScrub').value; render(); });
  $('anSpeed').addEventListener('input', () => { if (A.playing) play(); });
  // Space plays and pauses, arrow keys step: only on this view, and never instead of what the focused control does
  // (a button or link activates with Space, a field or slider takes its own keys, a drawing handles its own arrows).
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || $('tab-anatomy').classList.contains('hidden') || !A.rec) return;
    const t = e.target; if (t && t.closest && t.closest('button, a[href], input, select, textarea, summary, [role=button], [contenteditable=""], [contenteditable=true], canvas[tabindex]')) return;
    if (e.key === ' ') { e.preventDefault(); $('anPlay').click(); } if (e.key === 'ArrowRight') $('anNext').click(); if (e.key === 'ArrowLeft') $('anPrev').click();
  });
  addEventListener('resize', () => { if (!$('tab-anatomy').classList.contains('hidden')) { if (A.st) drawNet(); else netHint(); drawZoom(); if (A.rec) drawKV(); if (PA.d) drawPatch(); else patchHint(); } });
  // The numbers behind this view's drawings, for Prometheus (read through current_view).
  window.PSIViewData = window.PSIViewData || {};
  window.PSIViewData.anatomy = () => {
    const out = {};
    if (A.rec) {
      const R = A.rec, L = R.layers, pick = [0, Math.floor(L / 4), Math.floor(L / 2), Math.floor(3 * L / 4), L - 1];
      out.recording = { prompt_pieces: R.prompt_len, written: R.steps.length - R.prompt_len + 1, current_step: A.step, bins: R.bins,
        steps: R.steps.map((st, i) => { const fin = st.layers[L - 1].lens[0]; const first = st.layers.findIndex((x) => x.lens[0] === fin);
          return { step: i, phase: st.phase, taking_in: R.pieces[st.pos], writes_next: st.writes, p: st.p, final_guess: fin, first_layer_with_final_guess: first, lens: pick.map((l) => [l, st.layers[l].lens[0], st.layers[l].lens[1]]) }; }),
        note: 'Each step: the piece taken in, what the model writes next (with probability), and what layers would say (logit lens). first_layer_with_final_guess = where the final answer first appears.' };
    }
    if (typeof PA !== 'undefined' && PA.d) { const d = PA.d; out.patching = { clean: d.pieces_clean.join(''), corrupt: d.pieces_corrupt.join(''), clean_answer: d.clean_answer, corrupt_answer: d.corrupt_answer, shown: $('paKind').value, summary: $('paSummary').textContent }; }
    if (A.zoom && A.zoom.data) { const z = A.zoom.data; out.zoom = { layer: z.layer, bin: z.bin, neurons: `${z.from}-${z.to - 1}`, strongest_now: z.acts ? z.acts[Math.min(A.step, z.acts.length - 1)].map((v, k) => [z.from + k, v]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8) : null }; }
    if (A.concept) { const c = A.concept; out.concept = { concept: c.concept, nodes: c.nodes.length, core: c.core_count, depth_by_layer: c.depth, slices: c.slices.map((x) => ({ side: x.facet, strength: x.strength, own_neurons: x.count })), top_neurons: c.nodes.slice(0, 15).map((n) => ({ neuron: nkey(n.layer, n.neuron), kind: n.form ? 'word form' : n.kind, side: n.facet, name: NAMES[nkey(n.layer, n.neuron)] ? NAMES[nkey(n.layer, n.neuron)].name : null })) }; }
    if (A.net) { const d = A.net; out.understood = { concept: d.concept, says: d.said, inner_rank_of_what_it_says: d.said_rank, of_words: d.pool, nearest_inside_unprompted: d.nodes.filter((n) => n.source.includes('inside')).map((n) => [n.word, n.sim]), core_neurons_push_toward: d.side_vocab && d.side_vocab.core, fact_checks: (d.checks || []).map((c) => [c.prompt + ' … ' + c.answer, c.verdict]) }; if (A.color) out.understood.colour_check = { rho: A.color.rho, best: A.color.best, by_condition: A.color.grid, nearest_inside: A.color.inner_nearest, nearest_real: A.color.real_nearest }; }
    if (A.st) out.model = { name: A.st.name, layers: A.st.layers, heads: A.st.heads, mlp_neurons: A.st.mlp, bins: A.bins };
    return out;
  };
  // Opening the view reads the structure, which starts the inside-the-model engine: only when that engine is set up.
  // Until then the drawings say what is missing, and the view loads as soon as the engine is ready.
  const visible = () => !$('tab-anatomy').classList.contains('hidden');
  const open = () => { if (A.st) drawNet(); else if (PSI.state.engine && !A.loading) loadStructure(); else netHint(); if (PA.d) drawPatch(); else patchHint(); if (A.rec) render(); };
  document.querySelector('#tabs button[data-tab=anatomy]').addEventListener('click', () => setTimeout(open, 0));
  PSI.onState((s) => { if (visible()) { if (s.engine && !A.st && !A.loading && !A.stFailed) loadStructure(); else if (!A.st) netHint(); } });
  netHint(); patchHint();
})();
