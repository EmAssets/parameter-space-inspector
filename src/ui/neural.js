// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The Neural net tab: the loaded model seen from inside while it answers (data from src/trace_server.py).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => String(s == null ? '' : s).replace(/�/g, '▫').replace(/\n/g, '⏎');
  const pct = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
  // How much the residual stream's size changes through the network: 'grew 39×', or 'shrank to 0.6×' in a small model.
  const growth = (x) => { const f = x.resid[x.resid.length - 1] / (x.resid[0] || 1); return f >= 1.5 ? `grew ${f.toFixed(0)}×` : f >= 0.95 ? `stayed about the same size (${f.toFixed(2)}×)` : `shrank to ${f.toFixed(2)}×`; };
  const N = { trace: null, pos: 0, layer: null, dropped: [], recNo: 0, zmax: 1, amax: 1 };
  const err = (e, o) => PSI.err(e, o);
  const stat = (text, kind) => PSI.status($('nnStat'), text, kind);

  // Terms these views use, defined once (hover and focus); the glossary in help.html says the same.
  Object.assign(PSI.TERMS || (PSI.TERMS = {}), {
    'zero-ablation': 'Switching a neuron off by setting its output to zero. Zero can itself be an unusual value for that neuron, so a change may come from the odd value as much as from the missing signal. Compare mean-ablation.',
    'mean-ablation': 'Switching a neuron off by holding it at its average value over ordinary text instead of at zero: what it says about this input is removed, the network’s usual level is kept.',
    'z-score': 'How unusual a value is for that neuron: how many standard deviations it lies from the neuron’s average on ordinary text. 0 is typical; 5 or more is rare.',
    'e-level framing': 'A system prompt that asks the model to think in the fifteen E-levels. It changes the input, not the weights; its tokens are part of what the model reads.',
    'random control': 'The same change made to randomly chosen parts of the same size and in the same layers, with several seeds. A result inside the range of the random sets is not specific to what was chosen.',
    'drafted name': 'A neuron name proposed by Prometheus from the neuron’s evidence: a hypothesis, not a measurement. Names you write yourself are marked as yours and never overwritten by a draft.',
  });

  // The inside engine reads an unpacked folder (bf16 safetensors), the File view a GGUF: say when the two disagree on shape.
  // inside: {layers, width, vocab, name, dtype}; returns the mismatches (strings) and updates the File view's note.
  PSI.insideVsFile = (inside) => {
    if (inside) PSI.insideMeta = { ...(PSI.insideMeta || {}), ...inside };
    const I = PSI.insideMeta, F = window.PSIModelMeta, out = [];
    if (I && F) [['layers', 'layers'], ['width', 'width'], ['vocabulary', 'vocab']].forEach(([w, k]) => { if (I[k] != null && F[k] != null && +I[k] !== +F[k]) out.push(`${w}: ${I[k]} inside vs ${F[k]} in the GGUF`); });
    const note = $('fileEngines');
    if (note && I) {
      note.textContent = `This view reads the GGUF file the local engine runs${F && F.name ? ` (${F.name})` : ''}. Neural net and Anatomy read another copy: the unpacked folder the inside engine loaded (${I.name || 'its folder'}${I.dtype ? `, ${I.dtype}` : ''}). `;
      if (out.length) note.appendChild(el('span', 'cond-warn', `They do not match: ${out.join('; ')}. Results in those views describe another model than this file.`));
      else if (F) note.append(`Compared so far (${['layers', 'width', 'vocab'].filter((k) => I[k] != null && F[k] != null).join(', ') || 'nothing yet'}): they agree.`);
    }
    return out;
  };
  document.addEventListener('psi-model-meta', () => PSI.insideVsFile(null));

  // Everything here runs in the inside-the-model engine (PyTorch), not the GGUF engine; knock-outs also need a recording.
  ['nnGo', 'koLayer', 'koAll'].forEach((id) => PSI.requires($(id), 'engine'));
  const KO = ['koLayer', 'koAll'];
  const needTrace = () => KO.forEach((id) => PSI.block($(id), 'trace', N.trace ? null : 'Record an answer first.'));
  needTrace();

  async function status() {
    try { const s = await psi.traceStatus(); if (!N.trace) stat(s && s.ok ? `trace engine: ready on ${s.device}` : 'trace engine: not started (starts on first use)'); } catch (_) { /* no bridge */ }
  }
  // Before a recording the drawing says what will appear and which button makes it.
  function hint() {
    const cv = $('nnCanvas'), r = Math.min(2, devicePixelRatio || 1); cv.style.height = '520px'; cv.width = (cv.clientWidth || 800) * r; cv.height = 520 * r;
    const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height);
    const text = 'Nothing recorded yet: press “Answer and record”';
    g.fillStyle = '#a9a69f'; g.font = `${12 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle'; const W = cv.width, H = cv.height; const lines = []; let ln = ''; String(text).split(' ').forEach((wd) => { const t2 = ln ? ln + ' ' + wd : wd; if (ln && g.measureText(t2).width > W * 0.9) { lines.push(ln); ln = wd; } else ln = t2; }); lines.push(ln); const lh = 18 * r; lines.forEach((t2, i) => g.fillText(t2, W / 2, H / 2 + (i - (lines.length - 1) / 2) * lh));
    PSI.canvasLabel(cv, text);
  }
  hint();

  // The conditions a reply was measured under (trace_server.py's `conditions`), as conditions-line parts.
  function condParts(c, extra) {
    if (!c) return [{ t: 'conditions not reported by the inside engine (an older trace_server.py)', warn: true }].concat(extra || []);
    const parts = [{ t: c.model || 'the inside engine’s model', title: `${c.model_dir || ''}${c.weights_hash ? `\nweights ${c.weights_hash} (sha256 of ${c.weights_hash_of})` : ''}` }];
    if (c.weights_hash) parts.push({ t: `weights ${c.weights_hash.slice(0, 8)}`, title: `sha256 of ${c.weights_hash_of}` });
    parts.push({ t: `${c.dtype || '?'} on ${c.device || '?'}`, term: 'inside engine' });
    const lz = c.changed && c.changed.lesion;
    if (lz) parts.push({ t: `Tipping-point cut in force (${lz.kind}${lz.neurons != null ? `, ${lz.neurons} neurons` : ''}${lz.heads != null ? `, ${lz.heads} heads` : ''})`, chg: true, term: 'changed model' });
    else parts.push({ t: 'model as loaded (no cut)', title: 'the inside engine reported no lesion hooks and no changed tensors at the time of this reply' });
    if (c.no_think_appended != null) parts.push(c.no_think_appended ? '“/no_think” appended' : 'question sent as written');
    return parts.concat(extra || []);
  }
  PSI.insideCondParts = condParts;   // Anatomy renders the same conditions
  PSI.followLens($('nnLens'));   // starts as Settings > Think in E-levels says, until changed here

  // The last recording is kept (PSI.keep): a reload or another device shows it again, with the conditions it was
  // recorded under. The trace server echoes the prompt and framing it recorded, so a kept recording is self-contained.
  const when = (iso) => { try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); } catch (_) { return ''; } };
  const KEEP = PSI.keep('neural', { inputs: ['nnPrompt', 'nnN', 'nnLens'], status: 'nnStat',
    render: (t, info) => { if (t && t.tokens && t.tokens.length) load(t, t.prompt != null ? t.prompt : $('nnPrompt').value, !!t.lens, `kept recording from ${when(info.at)}`, true); } });
  $('nnGo').addEventListener('click', async () => {
    const b = $('nnGo'); b.disabled = true; stat('recording… (the first run loads the model into the trace engine)', 'busy');
    try {
      const lens = $('nnLens').checked && (!window.PSIEM || PSIEM.on);
      load(await KEEP.call('trace', $('nnPrompt').value, { n_new: +$('nnN').value || 20, lens }), $('nnPrompt').value, lens);
    } catch (e) { err(e); stat(`Recording failed: ${PSI.explain(e).text}`, 'error'); } finally { b.disabled = false; }
  });
  // A recording into the view: this view's own, or one an experiment made (`from` names it in the conditions line).
  // restored: kept from before (PSI.keep), so the engines' state now is not stamped onto it.
  function load(t, prompt, lens, from, restored) {
    N.raw = Object.assign({}, t, { tokens: t.tokens.slice() }); N.rawPrompt = prompt;   // as recorded, before this view trims it
    // Qwen sometimes re-emits the closing think tag first: it is kept (greyed) in the text, the attention row and the
    // knock-outs' context, so positions stay those the model saw, but the token slider starts after it.
    const dropped = [];
    while (t.tokens.length > 1 && /^\s*(<\/think>|\n)+\s*$/.test(t.tokens[0].piece)) dropped.push(t.tokens.shift());
    if (t.prompt == null) { t.prompt = prompt; t.lens = lens; t.think = false; t.recordedLocally = true; }
    N.trace = t; N.dropped = dropped; N.recNo += 1; t.recNo = N.recNo; N.pos = Math.min(t.tokens.length - 1, 2);
    // One scale for every token of the recording, so a dot's size means the same as you scrub.
    N.zmax = Math.max(1, ...t.tokens.flatMap((x) => (x.mlp_z || []).flat().map(Math.abs)));
    N.amax = Math.max(1e-6, ...t.tokens.flatMap((x) => x.mlp.flat().map((m) => Math.abs(m[1]))));
    $('nnPos').max = t.tokens.length - 1; $('nnPos').value = N.pos; $('nnPos').disabled = false;
    // The layer slider follows this model's depth: start a little past the middle, where attention is usually most telling.
    if (N.layer == null || N.layer > t.layers) N.layer = Math.max(1, Math.round(t.layers * 0.55));
    $('nnLayer').max = t.layers; $('nnLayer').value = N.layer; $('nnLayer').disabled = false; $('nnLayerLabel').textContent = `${N.layer} of ${t.layers}`;
    stat(`recording #${t.recNo}: ${t.tokens.length} tokens in ${t.seconds}s on ${t.device}${dropped.length ? ` (${dropped.length} leading “</think>” token${dropped.length > 1 ? 's' : ''} kept in context, greyed)` : ''}`, 'ok');
    const c = t.conditions, mism = PSI.insideVsFile({ layers: t.layers, name: c && c.model, dtype: c && c.dtype });
    // Another engine than Generate's: steering and LoRA do not reach it; a cut does.
    PSI.cond($('nnGo').closest('.row'), condParts(c, [
      { t: lens ? 'E-level framing on' : 'no E-level framing', frame: lens, term: 'E-level framing' }, { t: 'greedy: the likeliest piece each step', term: 'greedy' },
      { t: `recording #${t.recNo}` }, ...(t.perf && t.perf.tps != null ? [{ t: `${t.perf.tps.toFixed(t.perf.tps < 10 ? 1 : 0)} tokens/s recorded`, title: 'Speed while recording every layer (details below the buttons)' }] : []), ...(from ? [{ t: from }] : []),
      { t: 'no untrained baseline here', title: 'Nothing in this view is compared with an untrained twin: a pattern drawn here may come from the architecture as much as from training. The Emergence ladder compares with the twin.' },
      ...(!restored && PSI.changesFor('local').length ? [{ t: 'steering/LoRA: on the local engine only, not here' }] : []),
      ...mism.map((m) => ({ t: `not the GGUF’s shape: ${m}`, warn: true })),
      ...(t.recordedLocally ? [{ t: 'the engine did not echo the prompt: knock-outs re-send the text as it was when recorded', warn: true }] : []),
      ...(t.extras_error ? [{ t: `z-scores and prompt parts unavailable: ${t.extras_error}`, warn: true }] : [])]), 'recorded on');
    statsRow(); perfRow(t);
    needTrace(); render();
    // Every recorded token was thinking: the answer had not begun. Offer a longer recording (this view's own only).
    const th = PSI.thinking((t.prompt_tokens || []).join(''), [...dropped, ...t.tokens].map((x) => x.piece).join(''));
    const asked = +$('nnN').value || 20;
    if (th.cut && !from && t.tokens.length + dropped.length >= asked && window.PSIThinkingCut)
      PSIThinkingCut($('nnN'), asked, `all ${asked} recorded tokens were the model’s thinking`, () => $('nnGo').click());
  }
  // Speed of the recording (trace_server.py perf): the model's own forward pass, the time spent recording every layer,
  // the prompt read, and the GPU memory it peaked at. A recording from an older engine has none: the row stays hidden.
  function perfRow(t) {
    const box = $('nnPerf'), p = t && t.perf; if (!box) return;
    box.textContent = ''; box.classList.toggle('hidden', !p);
    if (!p) return;
    const n = (v, d = 0) => (v == null ? '–' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
    const ms = (v) => (v == null ? '–' : v >= 1000 ? `${n(v / 1000, 2)} s` : `${n(v, v < 10 ? 1 : 0)} ms`);
    const cells = [
      ['tokens/s', n(p.tps, 1), 'Answer tokens per second while recording: the forward pass plus reading out every layer (the first token, which reads the prompt, is left out)'],
      ['model alone', p.model_tps == null ? '–' : `${n(p.model_tps, 1)} tok/s`, 'The forward pass alone, keeping every layer’s states and attention: what recording costs on top is the next figure'],
      ['per token', `${ms(p.forward_ms_per_token)} + ${ms(p.record_ms_per_token)}`, 'Forward pass + recording (logit lens at every layer, top neurons, attention rows), averaged over the answer tokens'],
      ['first token', ms(p.first_token_ms), 'Reading the whole prompt and recording the first answer token'],
      ['prompt read', p.prefill_tps == null ? '–' : `${n(p.prompt_tokens)} tok at ${n(p.prefill_tps)} tok/s`, 'The prompt (with its framing and chat template) read in one forward pass'],
      ['total', `${n(p.new_tokens)} tok in ${n(p.wall_s, 2)} s`, 'Wall time of the recording, start to end'],
    ];
    if (p.peak_gpu_gb != null) cells.push(['peak GPU', `${n(p.peak_gpu_gb, 1)} of ${n(p.gpu_total_gb, 1)} GB`, 'The most GPU memory the inside engine held during this recording (weights, cache, and the states it kept)']);
    const w = p.power;   // the whole card's power while it recorded (NVML); absent without an NVIDIA GPU
    if (w) {
      const whole = `The whole card, as its driver reports it (${w.source}): it includes everything else on this GPU during the recording, such as the answering engine and the desktop.`;
      cells.push(['peak power', w.peak_w == null ? '–' : `${n(w.peak_w)} W${w.limit_w ? ` of ${n(w.limit_w)}` : ''}`, `The highest reading during the recording, of the card’s power limit. ${whole}`]);
      cells.push(['average', w.avg_w == null ? '–' : `${n(w.avg_w)} W${w.idle_w != null ? ` (${n(w.idle_w)} before)` : ''}`, `Energy over the recording’s time; in brackets the reading just before it started. ${whole}`]);
      cells.push(['energy', w.energy_j == null ? '–' : `${n(w.energy_j)} J · ${n(w.energy_wh * 1000, 1)} mWh`, `What the card used during the recording. ${whole}`]);
      cells.push(['energy / token', w.j_per_token == null ? '–' : `${n(w.j_per_token, 1)} J`, 'Energy divided by the tokens recorded']);
      if (w.above_idle_j != null) cells.push(['above idle', `${n(w.above_idle_j)} J`, 'Energy beyond what the card drew just before the recording started, over the same time: an estimate of what the recording itself added']);
    }
    cells.push(['precision', [p.dtype, p.attention].filter(Boolean).join(' · ') || '–', 'Weights precision, and the attention code the engine runs (eager: needed to read the attention weights)']);
    for (const [k, v, title] of cells) { const c = el('div', 'nn-perf-c'); c.title = title; c.append(el('span', 'k', k), el('b', null, v)); box.appendChild(c); }
  }
  // The experiment designer opens its recordings here (a copy: this view trims and numbers what it is given).
  window.PSINeural = {
    recNo: () => N.recNo,
    // The last recording as the engine returned it (a copy), with the prompt it was made from: the experiment tour reads it.
    last: () => (N.raw ? { recNo: N.recNo, prompt: N.rawPrompt, trace: JSON.parse(JSON.stringify(N.raw)) } : null),
    load(t, prompt, lens, from) {
      PSI.goto('neural'); $('nnPrompt').value = prompt; $('nnLens').checked = !!lens; $('nnLens').dataset.own = '1';   // after opening: an empty view copies Generate's prompt on open
      load(JSON.parse(JSON.stringify(t)), prompt, !!lens, from);
    },
  };
  $('nnPos').addEventListener('input', () => { N.pos = +$('nnPos').value; render(); });
  $('nnLayer').addEventListener('input', () => { N.layer = +$('nnLayer').value; $('nnLayerLabel').textContent = `${N.layer} of ${N.trace ? N.trace.layers : '?'}`; renderAttn(); lensRow(); summary(); });
  addEventListener('resize', () => { if (N.trace) draw(); else if (!$('tab-neural').classList.contains('hidden')) hint(); });

  function render() {
    const t = N.trace; if (!t) return;
    const T = $('nnText'), had = T.contains(document.activeElement); T.textContent = '';
    N.dropped.forEach((x) => { const s = el('span', 'nn-seg-dropped', show(x.piece)); s.title = 'written by the model, left out of the token slider; still part of its context'; T.appendChild(s); });
    // The tokens are buttons for the keyboard too: one Tab stop, arrow keys move, Enter or Space picks (the token slider does the same).
    t.tokens.forEach((x, i) => {
      const s = el('span', i === N.pos ? 'cur' : '', show(x.piece)); s.style.background = `rgba(242,241,238,${(0.04 + (1 - x.p) * 0.5).toFixed(3)})`; s.title = `${(x.p * 100).toFixed(1)}%`;
      s.setAttribute('role', 'button'); s.tabIndex = i === N.pos ? 0 : -1; s.setAttribute('aria-pressed', String(i === N.pos)); s.setAttribute('aria-label', `token ${i + 1}: “${show(x.piece)}”, ${(x.p * 100).toFixed(0)}%`);
      s.addEventListener('click', () => { N.pos = i; $('nnPos').value = i; render(); }); T.appendChild(s);
    });
    if (!T._keys) {
      T._keys = true;
      T.addEventListener('keydown', (e) => {
        const all = [...T.querySelectorAll('[role=button]')], i = all.indexOf(e.target); if (i < 0) return;
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.target.click(); return; }
        const n = /Right|Down/.test(e.key) ? all[i + 1] : /Left|Up/.test(e.key) ? all[i - 1] : e.key === 'Home' ? all[0] : e.key === 'End' ? all[all.length - 1] : null;
        if (n) { e.preventDefault(); n.click(); }
      });
    }
    if (had) { const f = T.querySelector('[tabindex="0"]'); if (f) f.focus(); }
    const x = t.tokens[N.pos];
    $('nnPosLabel').textContent = `“${show(x.piece)}”  ${N.pos + 1}/${t.tokens.length}`;
    draw(); renderAttn(); lensRow(); summary();
  }

  // Stable vertical slot per neuron id, so a neuron that keeps firing stays in one place.
  const slot = (id) => { let h = id * 2654435761 % 4294967296; return (h >>> 8) / 16777216; };
  // The logit lens: the first layer whose reading is the token written, and the layer from which the reading STAYS it.
  function lensRead(x) {
    const L = x.lens.length, same = x.lens.map(([p]) => p === x.piece);
    const first = same.indexOf(true);
    let stays = null; for (let l = L - 1; l >= 0 && same[l]; l--) stays = l;
    return { first: first < 0 ? null : first + 1, stays: stays == null ? null : stays + 1, L };
  }
  const lensWords = (R) => R.stays == null ? 'at the last layer is not the token written (the lens reads it differently from the output)'
    : R.stays === R.L ? `is decided only at the last layer${R.first != null && R.first < R.L ? ` (it showed at layer ${R.first}, then gave way)` : ''}`
      : `stays from layer ${R.stays} of ${R.L}${R.first != null && R.first < R.stays ? ` (it first showed at layer ${R.first}, gave way, and came back)` : ''}`;

  function draw() {
    const t = N.trace, x = t.tokens[N.pos], cv = $('nnCanvas');
    const r = Math.min(2, devicePixelRatio || 1); cv.style.height = '520px'; cv.width = cv.clientWidth * r; cv.height = 520 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, L = t.layers;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const left = 54 * r, right = W - 16 * r, colW = (right - left) / L, colX = (l) => left + (l + 0.5) * colW;
    const mono = (px, w = 400) => { g.font = `${w} ${px * r}px "IBM Plex Mono", ui-monospace, monospace`; };
    const final = x.piece, R = lensRead(x), top3 = x.lens_top3 || x.lens.map((v) => [v]);
    // Band 1 (top): the logit lens, what each layer would say. Brightness = confidence; bright where it equals the final token.
    // Under each word, its top three readings as stacked ticks (1st bright, 2nd and 3rd dimmer).
    const y1 = 26 * r, h1 = 150 * r;
    mono(9); g.fillStyle = '#a9a69f'; g.textAlign = 'left'; g.fillText('what it would say if it stopped here (word: top reading · ticks: top 3)', left, 14 * r);
    for (let l = 0; l < L; l++) {
      const [piece, p] = x.lens[l]; const same = piece === final;
      g.save(); g.translate(colX(l), y1 + h1); g.rotate(-Math.PI / 2);
      mono(10, same ? 500 : 400); g.textAlign = 'left';
      g.fillStyle = same ? `rgba(242,241,238,${0.55 + 0.45 * Math.min(1, p * 2)})` : `rgba(169,166,159,${0.15 + 0.6 * Math.min(1, p * 3)})`;
      g.fillText(show(piece).slice(0, 14) || '∅', 0, 3 * r); g.restore();
      let yy = y1 + h1 + 6 * r;
      top3[l].forEach((e, k) => { const hh = Math.max(k ? r : 2 * r, (e[1] || 0) * 26 * r); g.fillStyle = k === 0 ? (same ? '#f2f1ee' : 'rgba(169,166,159,0.6)') : `rgba(169,166,159,${k === 1 ? 0.35 : 0.2})`; g.fillRect(colX(l) - 1.5 * r, yy, 3 * r, hh); yy += hh; });
    }
    if (R.stays != null) {
      const xs = colX(R.stays - 1) - colW / 2, late = R.stays > L * 0.7;
      g.strokeStyle = 'rgba(242,241,238,0.5)'; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(xs, y1 - 6 * r); g.lineTo(xs, y1 + h1 + 36 * r); g.stroke(); g.setLineDash([]);
      mono(9); g.fillStyle = '#f2f1ee'; g.textAlign = late ? 'right' : 'left'; g.fillText(R.stays === L ? 'decided only at the last layer' : `stays from layer ${R.stays}`, xs + (late ? -4 : 4) * r, y1 - 2 * r);
    }
    if (R.first != null && R.stays != null && R.first < R.stays) { g.fillStyle = 'rgba(242,241,238,0.45)'; g.beginPath(); g.arc(colX(R.first - 1), y1 - 8 * r, 2.5 * r, 0, Math.PI * 2); g.fill(); }
    // Band 2: the 8 neurons with the largest |activation| per layer, in stable slots. Size: z-score against the neuron's
    // normal range when built (comparable across layers), else |activation|; one scale for the whole recording.
    const y2 = 230 * r, h2 = 200 * r, Z = x.mlp_z, of = t.mlp ? ` of ${t.mlp.toLocaleString()}` : '';
    mono(9); g.fillStyle = '#a9a69f'; g.textAlign = 'left';
    g.fillText(Z ? `neurons that fired hardest (8${of} per layer), sized by z-score against their normal range` : `neurons that fired hardest (8${of} per layer), sized by raw |activation|: no normal range built`, left, y2 - 8 * r);
    g.strokeStyle = 'rgba(42,41,39,1)'; g.lineWidth = r;
    for (let l = 0; l <= L; l += 6) { g.beginPath(); g.moveTo(left + l * colW, y2); g.lineTo(left + l * colW, y2 + h2); g.stroke(); }
    for (let l = 0; l < L; l++) x.mlp[l].forEach(([id, a], k) => {
      const v = Z ? Math.min(1, Math.abs(Z[l][k]) / N.zmax) : Math.abs(a) / N.amax;
      const yy = y2 + slot(id) * h2, rr = r * (1.2 + 4 * v);
      g.beginPath(); g.arc(colX(l), yy, rr, 0, Math.PI * 2);
      if (a >= 0) { g.fillStyle = `rgba(242,241,238,${0.35 + 0.65 * v})`; g.fill(); } else { g.strokeStyle = `rgba(242,241,238,${0.4 + 0.6 * v})`; g.lineWidth = r; g.stroke(); }
    });
    mono(8); g.fillStyle = '#6f6c66'; g.textAlign = 'right';
    g.fillText(`largest dot = ${Z ? `z ${N.zmax.toFixed(0)}` : N.amax.toFixed(2)}, the same scale for every token of this recording · filled +, ring −`, right, y2 + h2 + 10 * r);
    // Band 3: how much the signal grew (residual norm, log scale).
    const y3 = 452 * r, h3 = 40 * r, lo = Math.log(Math.min(...x.resid)), hi = Math.log(Math.max(...x.resid));
    mono(9); g.fillStyle = '#a9a69f'; g.textAlign = 'left'; g.fillText(`the signal ${growth(x)} through the layers`, left, y3 - 6 * r);
    for (let l = 0; l < L; l++) { const v = (Math.log(x.resid[l]) - lo) / ((hi - lo) || 1); g.fillStyle = 'rgba(242,241,238,0.55)'; g.fillRect(colX(l) - colW * 0.3, y3 + h3 - v * h3, colW * 0.6, v * h3); }
    // Axis
    mono(9); g.fillStyle = '#6f6c66'; g.textAlign = 'center';
    for (let l = 0; l < L; l += 5) g.fillText(`L${l + 1}`, colX(l), H - 6 * r);
    g.textAlign = 'left'; g.fillText('layer →', left - 50 * r, H - 6 * r);
    g.save(); g.translate(14 * r, y1 + h1 / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillStyle = '#6f6c66'; g.fillText('← reads as', 0, 0); g.restore();
    N.read = R; N.geo = { left, colW, r, L };
    PSI.canvasLabel(cv, `Token ${N.pos + 1} “${show(final)}” through ${L} layers: the logit lens reading ${lensWords(R)}; the 8 neurons that fired hardest per layer${Z ? ', sized by z-score' : ''}; the signal ${growth(x)}.`);
  }
  // Hovering a column names its top three readings.
  $('nnCanvas').addEventListener('mousemove', (e) => {
    const t = N.trace, G = N.geo; if (!t || !G) return;
    const rc = $('nnCanvas').getBoundingClientRect(), l = Math.floor(((e.clientX - rc.left) * G.r - G.left) / G.colW); if (l < 0 || l >= G.L) { $('nnCanvas').title = ''; return; }
    const x = t.tokens[N.pos], row = x.lens_top3 ? x.lens_top3[l] : [x.lens[l]];
    $('nnCanvas').title = `layer ${l + 1} would say: ${row.map(([p, v]) => `“${show(p)}” ${pct(v)}`).join(' · ')}`;
  });
  // The chosen layer's top three readings, as text (the layer slider picks it).
  function lensRow() {
    const t = N.trace, box = $('nnLens3'); if (!t || !box) return;
    const x = t.tokens[N.pos], row = x.lens_top3 ? x.lens_top3[N.layer - 1] : [x.lens[N.layer - 1]];
    box.textContent = `logit lens at layer ${N.layer}: ${row.map(([p, v]) => `“${show(p)}” ${pct(v)}`).join(' · ')}${x.lens_top3 ? '' : ' (top reading only: an older trace_server.py)'}`;
  }

  const SEG = { template: 'template (role markers, special tokens)', system: 'E-level framing (system prompt)', user: 'your question', prompt: 'prompt', dropped: 'written by the model, left out of the token slider', own: 'its own earlier words' };
  function renderAttn() {
    const t = N.trace; if (!t) return;
    const x = t.tokens[N.pos], A = x.attn[N.layer - 1] || [], box = $('nnAttn'); box.textContent = '';
    // Missing is not zero: without a recording for this layer, say so instead of drawing every piece unlit.
    if (!A.length) { box.appendChild(el('span', 'muted', `No attention was recorded for layer ${N.layer}: not measured, not zero.`)); return; }
    // Positions as the model saw them: the prompt, any leading tokens left out of the slider (greyed), the tokens written before this one.
    const all = [...t.prompt_tokens.map((p, i) => [p, (t.prompt_segments || [])[i] || 'prompt']), ...N.dropped.map((y) => [y.piece, 'dropped']), ...t.tokens.slice(0, N.pos).map((y) => [y.piece, 'own'])];
    if (A.length !== all.length) box.appendChild(el('p', 'small cond-warn', `The attention row has ${A.length} positions and the text ${all.length}: the highlights may be shifted. Reported, not hidden.`));
    const mx = Math.max(...A.slice(0, all.length), 1e-6);
    all.forEach(([piece, seg], i) => {
      const w = (A[i] || 0) / mx; const s = el('span', `nn-seg-${seg}`, show(piece)); s.style.background = `rgba(242,241,238,${(0.03 + 0.7 * w).toFixed(3)})`;
      s.title = `${((A[i] || 0) * 100).toFixed(1)}% of this token’s attention at layer ${N.layer}, averaged over all heads · ${SEG[seg]}`;
      if (seg === 'own') s.style.textDecoration = 'underline'; box.appendChild(s);
    });
    const cur = el('span', 'cur', show(x.piece)); cur.style.outline = '1px solid #f2f1ee'; box.appendChild(cur);
  }

  function summary() {
    const t = N.trace, x = t.tokens[N.pos], L = t.layers, R = N.read || lensRead(x);
    const strongest = x.mlp_z ? x.mlp.flatMap((m, l) => m.map(([id, a], k) => [l + 1, id, x.mlp_z[l][k], a])).sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]))[0]
      : x.mlp.map((m, l) => [l + 1, m[0][0], null, m[0][1]]).sort((a, b) => Math.abs(b[3]) - Math.abs(a[3]))[0];
    const A = x.attn[N.layer - 1] || [], seg = t.prompt_segments, P = t.prompt_len;
    const sum = (f) => A.reduce((s, v, i) => s + (f(i) ? v : 0), 0);
    const onPrompt = sum((i) => i < P), parts = seg ? [['user', 'your question'], ['system', 'the E-level framing'], ['template', 'template tokens']].map(([k, w]) => [w, sum((i) => i < P && seg[i] === k)]).filter(([w, v]) => v > 0 || w !== 'the E-level framing' || t.lens) : null;
    const sink = A.length && A[0] === Math.max(...A) ? ' Most of it sits on the very first token, a "sink" where transformers park attention they do not need; the real signal is in the dimmer highlights.' : '';
    // Technical terms carry their definition on hover and focus (PSI.term).
    const box = $('nnSummary'); box.textContent = '';
    box.append(`Token ${N.pos + 1} “${show(x.piece)}” (${(x.p * 100).toFixed(0)}% sure). Read with the `, PSI.term('logit lens'), `, the reading ${lensWords(R)}. The `, PSI.term('residual stream'), ` ${growth(x)} from layer 1 to ${L}.`);
    if (strongest[2] != null) box.append(' The most unusual of the hardest-firing neurons was ', `L${strongest[0]} #${strongest[1]}: activation ${strongest[3]}, `, PSI.term('z-score'), ` ${strongest[2]} against its normal range on ordinary text.`);
    else box.append(` The neuron with the largest activation was L${strongest[0]} #${strongest[1]} (${strongest[3]}); without a normal range, activations in different layers are not comparable.`);
    if (A.length) box.append(` At layer ${N.layer}, averaged over all heads, ${pct(onPrompt)} of this token’s attention went to the prompt${parts ? ` (${parts.map(([w, v]) => `${pct(v)} to ${w}`).join(', ')})` : ' (which parts are template, framing and question was not reported)'}, ${pct(Math.max(0, 1 - onPrompt))} to the words written since.${sink}`);
    else box.append(` Attention at layer ${N.layer} was not recorded.`);
    box.append(' (Where attention goes permits an answer but does not predict it: a measured finding, not an assumption.)');
  }

  // ── the normal range: each neuron's mean and SD on ordinary text, for z-scores and mean-ablation ──
  function statsRow() {
    const t = N.trace, box = $('nnStatsNote'), b = $('nnStats'); if (!box || !b) return;
    const ns = t && t.neuron_stats;
    b.hidden = !!ns; box.textContent = '';
    if (ns) box.textContent = `Normal range: each neuron’s mean and SD over ${ns.tokens.toLocaleString()} tokens (${ns.windows} passages of ${ns.corpus || 'the text sample'}), built ${ns.built_at || 'at an unknown time'} for these weights. Used for the z-scores above and for mean-ablation.`;
    else if (t) box.textContent = 'No normal range is built for these weights, so dots are sized by raw activation (not comparable between layers) and mean-ablation is unavailable. Building it reads 800 passages of the Wikipedia sample through the inside engine (busy meanwhile: under a minute on a GPU, several on a CPU) and writes a file of a few MB to the examples folder (psi/ in your cache folder when one is set with PSI_CACHE_DIR or cacheDir in settings.json, else cache/ in the app’s data folder), keyed by these weights. It does not change the model, and it is refused while a Tipping-point cut is active.';
    const mean = $('koAbl').querySelector('option[value=mean]'); mean.disabled = !ns; mean.textContent = ns ? 'held at their average (mean-ablation)' : 'held at their average (mean-ablation): build the normal range first';
    if (!ns && $('koAbl').value === 'mean') $('koAbl').value = 'zero';
  }
  $('nnStats').addEventListener('click', async () => {
    const b = $('nnStats'); b.disabled = true; PSI.status($('nnStatsStat'), 'reading 800 passages for every neuron’s normal range…', 'busy');
    try {
      const r = await psi.anatomy({ action: 'build_stats' });
      PSI.status($('nnStatsStat'), `built: ${r.tokens.toLocaleString()} tokens in ${r.seconds}s, saved to ${r.path}. Record again to see z-scores.`, 'ok');
    } catch (e) { err(e); PSI.status($('nnStatsStat'), `Building the normal range failed: ${PSI.explain(e).text}`, 'error'); } finally { b.disabled = false; }
  });

  // Knock-outs: how much of the chosen token lived in the neurons that fired? Re-asks the RECORDED prompt with the
  // recorded framing (not what is in the box now), with the tokens before the chosen one as written, against random
  // sets of the same size in the same layers (5 seeds).
  async function knockout(layers) {
    const t = N.trace; if (!t) return PSI.status($('koStat'), 'Record an answer first: press “Answer and record” above.', 'warn');
    const x = t.tokens[N.pos], prefix = [...N.dropped, ...t.tokens.slice(0, N.pos)].map((y) => y.id), abl = $('koAbl').value;
    const stamp = `token ${N.pos + 1}/${t.tokens.length} “${show(x.piece)}” of recording #${t.recNo}`;
    PSI.status($('koStat'), `asking again with neurons switched off (${stamp}), then with 5 random sets…`, 'busy');
    KO.forEach((id) => PSI.block($(id), 'busy', 'waiting for the current knock-out…'));
    try {
      const r = await psi.ablate({ prompt: t.prompt, lens: t.lens, think: !!t.think, target: x.id, plan: 'top', k: 8, layers, prefix_ids: prefix, ablation: abl, seeds: 5 });
      const where = layers ? `layer ${layers[0] + 1}` : 'every layer', how = abl === 'mean' ? 'held at their average on ordinary text' : 'set to zero';
      const card = el('div', 'ko-card'); card.appendChild(el('p', 'k', stamp));
      const line = el('p');
      line.append(`The 8 strongest neurons at ${where} switched off (${how}: `, PSI.term(abl === 'mean' ? 'mean-ablation' : 'zero-ablation'), `; ${r.knocked_out}${r.of_neurons ? ` of ${r.of_neurons.toLocaleString()}` : ''}): “${show(r.target)}” went from ${pct(r.p_before)} to ${pct(r.p_after)}${r.now_says !== r.target ? `, and the model now says “${show(r.now_says)}” instead` : ''}.`);
      card.appendChild(line);
      const cr = r.control_p_after, ctrl = r.control || [], verdict = el('p');
      if (!cr) verdict.appendChild(el('span', 'cond-warn', 'No random control came back (an older trace_server.py): this change cannot be told apart from switching off any 8 neurons.'));
      else {
        const [lo, hi] = cr, flips = ctrl.filter((c) => c.now_id !== r.target_id).length, eps = 0.005;
        verdict.append(PSI.term('Random control', 'random control'), `: ${ctrl.length} random sets of 8 per layer, in the same layers, left it at ${pct(lo)} to ${pct(hi)}${flips ? ` (${flips} of ${ctrl.length} changed the answer)` : ''}. `);
        const b = el('b');
        if (r.p_after < lo - eps) b.textContent = `Lower than every random set: this token leaned on these neurons more than on any 8 here.${r.now_says !== r.target ? '' : ' Weakened, but still chosen.'}`;
        else if (r.p_after > hi + eps) b.textContent = 'Higher than every random set: switching these off made it more sure.';
        else if (Math.abs(r.p_before - r.p_after) < 0.02) b.textContent = 'Nothing changed: within the range of the random sets.';
        else b.textContent = 'Within the range of the random sets: no different from switching off any 8 neurons here.';
        verdict.appendChild(b);
      }
      card.appendChild(verdict);
      const c = r.conditions, parts = condParts(c, [{ t: `${t.lens ? 'E-level framing on' : 'no E-level framing'}, as recorded`, frame: t.lens }, { t: 'for this one answer, undone after', title: 'hooks on the MLP inputs at every position of the input, removed before the reply' }]);
      if (c && t.conditions && c.weights_hash !== t.conditions.weights_hash) parts.push({ t: 'the inside engine holds other weights than when this was recorded', warn: true });
      const cut = PSI.changesFor('inside'); if (cut.length && !(c && c.changed && c.changed.lesion)) parts.push({ t: `on the changed model: ${cut.map((z) => z.label).join(', ')}`, chg: true });
      const holder = el('span'); card.appendChild(holder); PSI.cond(holder, parts, 'run on'); holder.remove();
      $('koOut').prepend(card); PSI.status($('koStat'), '');
    } catch (e) { err(e); PSI.status($('koStat'), `The knock-out failed: ${PSI.explain(e).text}`, 'error'); }
    finally { KO.forEach((id) => PSI.block($(id), 'busy', null)); }
  }
  $('koLayer').addEventListener('click', () => knockout([N.layer - 1]));
  $('koAll').addEventListener('click', () => knockout(null));

  // Share the prompt with the Generate tab the first time.
  document.querySelector('[data-tab=neural]').addEventListener('click', () => { if (!N.trace) { $('nnPrompt').value = $('prompt').value; setTimeout(hint, 0); } else setTimeout(draw, 0); status(); });   // draw: a kept recording restored while hidden is sized now
})();
