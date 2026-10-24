// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Connectors tab: the model's shared idea of "a connection" across twelve levels, and where language sits.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  const D = window.E8_CONNECTORS;
  const AMBER = '#ffc36b', INK = '#f2f1ee', INK2 = '#a9a69f', INK3 = '#75726c';
  const font = (r, px) => `${px * r}px "IBM Plex Mono", ui-monospace, monospace`;
  const canvas = (cv, h) => { const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || 640) * r; cv.height = h * r; const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); return { g, r, W: cv.width, H: cv.height }; };
  const short = (l) => l.replace(/^E\d+(?:-E\d+)? /, '').replace(' and nuclei', ', nuclei');
  const code = (l) => (l.match(/^E\d+(?:-E\d+)?/) || ['lang'])[0];
  const f2 = (x) => x.toFixed(2);
  // per-level hue: low levels cool, high levels warm (neutral greys, so amber stays for language)
  const shade = (i, n) => `hsl(${210 - (i / Math.max(1, n - 1)) * 170}, 18%, ${58 + (i % 2) * 8}%)`;
  // A centred line on an empty canvas (wrapped to its width), and the same line as its label.
  const hint = (cv, h, text) => {
    const { g, r, W, H } = canvas(cv, h); g.fillStyle = INK3; g.font = font(r, 12); g.textAlign = 'center'; g.textBaseline = 'middle';
    const lines = []; let cur = ''; String(text).split(' ').forEach((w) => { const t = cur ? `${cur} ${w}` : w; if (cur && g.measureText(t).width > W - 24 * r) { lines.push(cur); cur = w; } else cur = t; }); if (cur) lines.push(cur);
    lines.forEach((l, i) => g.fillText(l, W / 2, H / 2 + (i - (lines.length - 1) / 2) * 16 * r)); PSI.canvasLabel(cv, text);
  };
  // The server's generic `conditions` block (when present), as conditions-line parts.
  const condBlock = (c) => (!c || typeof c !== 'object') ? [] : Object.entries(c).filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${typeof v === 'object' ? Object.entries(v).map(([a, b]) => `${a} ${b && typeof b === 'object' ? JSON.stringify(b) : b}`).join(', ') : v}`);
  const inside = () => (PSI.changeParts ? PSI.changeParts('inside') : []);
  const metaParts = (m, extra = []) => {
    if (!m) return ['conditions not recorded in this copy of the data'];
    const v = (x) => (x == null || x === '' ? 'not recorded' : x);
    return [m.model, `revision: ${v(m.revision)}`, `dtype: ${v(m.dtype)}`, `seed: ${v(m.seed)}`, `run ${v(m.date)} (git ${v(m.git)})`, ...extra, { t: `prompt set ${v(m.prompt_set_hash)}`, title: m.hash_note || '' }];
  };

  // ── spread from the stored sentence positions (coords[i][0] is each sentence on the connection axis) ──
  const groupOf = (s) => (s.kind === 'connector' ? s.level : s.kind === 'language' ? 'language' : s.kind === 'meta' ? 'meta' : null);
  const quant = (a, q) => { const s = [...a].sort((x, y) => x - y), i = (s.length - 1) * q, lo = Math.floor(i); return s[lo] + (s[Math.min(s.length - 1, lo + 1)] - s[lo]) * (i - lo); };
  // 95% bootstrap interval of a mean (2,000 draws, fixed seed so it reads the same every time)
  const bootCI = (a, draws = 2000) => { let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const m = []; for (let d = 0; d < draws; d++) { let s = 0; for (let k = 0; k < a.length; k++) s += a[Math.floor(rnd() * a.length)]; m.push(s / a.length); } return [quant(m, 0.025), quant(m, 0.975)]; };
  const spreadCache = new Map();
  function spread(v) {
    if (!v || !v.coords || !D || !D.sentences) return null; if (spreadCache.has(v)) return spreadCache.get(v);
    const by = {}; D.sentences.forEach((s, i) => { const k = groupOf(s); if (k) (by[k] = by[k] || []).push(v.coords[i][0]); });
    const out = {}; Object.entries(by).forEach(([k, a]) => { out[k] = { n: a.length, mean: a.reduce((p, c) => p + c, 0) / a.length, iqr: [quant(a, 0.25), quant(a, 0.75)], ci: bootCI(a) }; });
    spreadCache.set(v, out); return out;
  }

  // ── 1. the live meter (needs the inside engine on Qwen3-4B; not the recorded study) ──
  const LIVE = [$('cnGo'), $('cnEx')];
  LIVE.forEach((b) => PSI.requires(b, 'engine'));
  const qwenGate = async () => {
    let dir = ''; try { const s = await psi.settings(); dir = (s.trace && s.trace.modelDir) || ''; } catch (_) { return; }
    const why = dir && !/qwen3[-_ ]?4b/i.test(dir) ? { text: 'Its connection axis was learned inside Qwen3-4B; the inside engine reads another model:', link: () => PSI.link('Point it at Qwen3-4B in Settings', 'settings', 'stTraceDir') } : null;
    LIVE.forEach((b) => PSI.block(b, 'qwen', why));
  };
  let qwenT = 0; PSI.onState(() => { clearTimeout(qwenT); qwenT = setTimeout(qwenGate, 50); });
  let LAST = [], LASTresp = null;
  const liveSpread = () => (D && D.qwen && D.qwen.deep ? spread(D.qwen.deep) : null);
  function drawMeter() {
    if (!LAST.length) { hint($('cnMeter'), 190, 'Nothing yet: press Measure, or Try examples.'); return; }
    const { g, r, W, H } = canvas($('cnMeter'), 190), left = 30 * r, right = W - 30 * r, y = 58 * r, SP = liveSpread();
    const ref = LAST[0].reference, lvKeys = Object.keys(ref.levels);
    const vals = [...Object.values(ref.levels), ref.language, ref.meta, ...LAST.map((x) => x.position), 0, 1, ...(SP ? Object.values(SP).flatMap((s) => s.iqr) : [])], lo = Math.min(-0.3, ...vals), hi = Math.max(1.4, ...vals);
    const X = (x) => left + ((x - lo) / (hi - lo)) * (right - left);
    g.strokeStyle = '#3a3936'; g.lineWidth = 2 * r; g.beginPath(); g.moveTo(left, y); g.lineTo(right, y); g.stroke();
    [[0, 'thing on its own'], [1, 'connection']].forEach(([x, t]) => { g.fillStyle = INK3; g.font = font(r, 9); g.textAlign = 'center'; g.fillText(t, X(x), H - 8 * r); g.fillRect(X(x) - r, y - 6 * r, 2 * r, 12 * r); g.fillStyle = 'rgba(242,241,238,.08)'; g.fillRect(X(x) - 0.5 * r, y, r, H - 22 * r - y); });
    // each level's middle half (IQR of its connection sentences, recorded study), one thin row per level under the axis
    if (SP) { lvKeys.forEach((l, i) => { const s = SP[l]; if (!s) return; const yy = y + 12 * r + i * 6 * r; g.strokeStyle = shade(i, lvKeys.length); g.lineWidth = 2 * r; g.beginPath(); g.moveTo(X(s.iqr[0]), yy); g.lineTo(X(s.iqr[1]), yy); g.stroke(); });
      const s = SP.language; if (s) { const yy = y + 12 * r + lvKeys.length * 6 * r; g.strokeStyle = AMBER; g.lineWidth = 2 * r; g.beginPath(); g.moveTo(X(s.iqr[0]), yy); g.lineTo(X(s.iqr[1]), yy); g.stroke(); } }
    lvKeys.forEach((l, i) => { g.fillStyle = shade(i, lvKeys.length); g.beginPath(); g.arc(X(ref.levels[l]), y, 3 * r, 0, Math.PI * 2); g.fill(); });
    g.strokeStyle = AMBER; g.lineWidth = 1.2 * r; g.beginPath(); g.arc(X(ref.language), y, 5 * r, 0, Math.PI * 2); g.stroke();
    LAST.forEach((it, k) => { const x = X(it.position); g.fillStyle = INK; g.beginPath(); g.moveTo(x, y - 9 * r); g.lineTo(x - 6 * r, y - 20 * r); g.lineTo(x + 6 * r, y - 20 * r); g.closePath(); g.fill(); g.font = font(r, 9); g.textAlign = 'center'; g.fillText(LAST.length > 1 ? String(k + 1) : f2(it.position), x, y - 25 * r - (k % 2) * 11 * r); });
    g.fillStyle = INK2; g.font = font(r, 9); g.textAlign = 'left'; g.fillText(`dots: the twelve levels’ means · rows under the axis: each level’s middle half${SP ? '' : ' (not in this copy)'} · amber: language · triangle: your sentence`, left, 12 * r);
    PSI.canvasLabel($('cnMeter'), `Meter: ${LAST.map((x) => `“${x.text || 'your sentence'}” at ${f2(x.position)}`).join('; ')} on the connection axis (0 = a thing on its own, 1 = an average connection; language sentences average ${f2(ref.language)}). ${$('cnWhy').textContent}`);
  }
  function tokens(box, list, cap) {
    box.textContent = ''; if (!list || !list.length) { box.classList.add('hidden'); return; } box.classList.remove('hidden');
    const mx = Math.max(1e-6, ...list.map((t) => Math.abs(t.pull)));
    list.forEach((t) => { const s = el('span', null, t.piece); const a = Math.min(1, Math.abs(t.pull) / mx); s.style.background = t.pull >= 0 ? `rgba(255,195,107,${(a * 0.8).toFixed(2)})` : `rgba(122,162,214,${(a * 0.7).toFixed(2)})`; if (a > 0.6 && t.pull >= 0) s.style.color = '#000'; s.title = `pull ${t.pull >= 0 ? '+' : '−'}${Math.abs(t.pull).toFixed(3)} axis units`; box.appendChild(s); });
    box.setAttribute('aria-label', cap); box.title = cap;
  }
  function explainOne(it) {
    const ref = it.reference, SP = liveSpread(), pos = it.position;
    const lv = Object.entries(ref.levels).map(([l, x]) => [l, x, Math.abs(x - pos)]).sort((a, b) => a[2] - b[2]), [a, b] = lv;
    const inIqr = (l) => SP && SP[l] && pos >= SP[l].iqr[0] && pos <= SP[l].iqr[1];
    const close = `Closest level: ${short(a[0])} at ${f2(a[1])} (${f2(a[2])} away); runner-up ${short(b[0])} at ${f2(b[1])} (${f2(b[2])} away, ${f2(b[2] - a[2])} further).`
      + (SP ? (inIqr(a[0]) && inIqr(b[0]) ? ' Your sentence sits inside the middle half of both, so which one it is closest to is within noise.' : inIqr(a[0]) ? ` It sits inside ${short(a[0])}’s middle half (${f2(SP[a[0]].iqr[0])} to ${f2(SP[a[0]].iqr[1])}).` : ' It sits outside the closest level’s middle half.') : '');
    const deep = D && D.qwen ? D.qwen.deep_layer : it.layer, best = D && D.qwen ? D.qwen.best_layer : null;
    const layer = `Read at layer ${it.layer}${it.layer === deep && best === 0 ? `, the best layer past the input embeddings. Qwen3-4B’s best layer overall is the input embeddings themselves (layer 0): mostly word choice, so the meter reads the next best` : ''}.`;
    const bl = it.baseline || { state: 'not returned' };
    const words = bl.state === 'read' ? ` Words alone (layer 0) put this sentence at ${f2(bl.position)}.`
      : D && D.qwen && D.qwen.best ? ` Words alone for this sentence: not stored (the axis file holds only layer ${it.layer}). In the recorded study, read from the embeddings alone, language sat at ${f2(D.qwen.best.language)} and linking language at ${f2(D.qwen.best.meta)}, against ${f2(D.qwen.deep.language)} and ${f2(D.qwen.deep.meta)} at layer ${deep}.` : '';
    const L = SP && SP.language ? ` Language sentences average ${f2(ref.language)} (95% interval ${f2(SP.language.ci[0])} to ${f2(SP.language.ci[1])}, n ${SP.language.n}).` : ` Language sentences average ${f2(ref.language)}.`;
    return `Position ${f2(pos)} on the connection axis. ${layer} ${close}${L}${words}`;
  }
  // A measurement on screen: from a run just now, or kept from before (PSI.keep); a kept one shows the conditions it was read under.
  function render(resp, info = {}) {
    if (!resp || !resp.items || !resp.items.length) return;
    LASTresp = resp; LAST = resp.items;
    $('cnWhy').textContent = LAST.length === 1 ? explainOne(LAST[0]) : LAST.map((x, i) => `${i + 1}. ${f2(x.position)} — ${x.text}`).join('   ');
    tokens($('cnTokens'), LAST.length === 1 ? LAST[0].tokens : null, `Each piece’s pull along the connection axis at layer ${LAST[0].layer}: amber towards connection, blue towards a thing on its own; the pulls add up to the position.`);
    drawMeter();
    const it = LAST[0];
    PSI.cond($('cnWhy'), [it.model || 'model not reported', `layer ${it.layer}`, it.unit || 'connection-axis units (0 = thing on its own, 1 = average connection)', `reference: recorded study (n ${D && D.meta && D.meta.n ? D.meta.n.sentences : 'not recorded'} sentences)`, `words alone: ${(it.baseline && it.baseline.state) || 'not returned'}`, ...(info.restored ? [] : inside()), ...condBlock(resp.conditions)]);
  }
  const KEEP = PSI.keep('connectors', { inputs: ['cnText'], render, status: 'cnStat' });
  async function measure(texts) {
    texts = texts.filter(Boolean); if (!texts.length) { PSI.status($('cnStat'), 'Write a sentence first.', 'warn'); $('cnText').focus(); return; }
    LIVE.forEach((b) => { b.disabled = true; }); PSI.status($('cnStat'), 'reading…', 'busy'); PSI.setRunning('connectors', true);
    try {
      render(await KEEP.call('connector', { texts }));
      PSI.status($('cnStat'), ''); if (window.PSIExp) PSIExp.mark($('cnStat')); PSI.setDone('connectors');
    } catch (e) { err(e); PSI.status($('cnStat'), `failed: ${PSI.explain(e).text}`, 'error'); } finally { LIVE.forEach((b) => { b.disabled = false; }); PSI.setRunning('connectors', false); }
  }
  $('cnGo').addEventListener('click', () => measure([$('cnText').value.trim()]));
  $('cnEx').addEventListener('click', () => measure(['A synapse passes a signal from one neuron to the next.', 'The neuron has a long, thin axon wrapped in myelin.', 'A treaty binds two states to keep their border open.', 'The mountain is covered in snow for most of the year.', 'She translated her grandmother’s letters so her children could read them.']));
  const onOpen = () => { if (!LAST.length) drawMeter(); };

  if (!D) {
    document.querySelector('[data-tab=connectors]').addEventListener('click', () => setTimeout(() => {
      onOpen(); ['cnLadder', 'cnMap', 'cnGrid'].forEach((id, i) => hint($(id), [340, 340, 380][i], 'Nothing recorded yet: the connector study has not been run.'));
    }, 0));
    $('cnLadderWhy').textContent = 'The recorded connector study is not in this copy yet. The meter above still works live.';
    ['cnModel', 'cnStepWrap'].forEach((id) => $(id).closest('label').classList.add('hidden'));
    return;
  }

  // ── model picker: Qwen, Pythia sizes, Pythia-410m across training ──────
  const SOURCES = [['qwen', 'Qwen3-4B (finished)', () => D.qwen]];
  Object.keys(D.sizes || {}).forEach((s) => SOURCES.push([`size-${s}`, `Pythia-${s} (finished)`, () => D.sizes[s]]));
  SOURCES.push(['p410', 'Pythia-410m across training', null]);
  SOURCES.forEach(([k, n]) => { const o = document.createElement('option'); o.value = k; o.textContent = n; $('cnModel').appendChild(o); });
  const steps = Object.keys(D.pythia410m || {}).map(Number).sort((a, b) => a - b);
  $('cnStep').max = String(Math.max(0, steps.length - 1)); $('cnStep').value = String(Math.max(0, steps.length - 1));
  const current = () => { const k = $('cnModel').value; if (k === 'p410') { const s = steps[+$('cnStep').value]; return { run: D.pythia410m[String(s)], label: `Pythia-410m at step ${s.toLocaleString()}` }; } const src = SOURCES.find((x) => x[0] === k); return { run: src[2](), label: src[1] }; };
  const view = (run) => run.deep || run.best;   // the layer past the word embeddings when available
  const layerNote = (run) => {
    const shown = run.deep ? run.deep_layer : run.best_layer;
    if (run.best_layer === shown) return `layer ${shown} of ${run.layers}`;
    return `layer ${shown} of ${run.layers} (best overall: ${run.best_layer === 0 ? 'layer 0, the input embeddings: mostly word choice' : `layer ${run.best_layer}, too close to the embeddings`}; shown: the best past them)`;
  };
  // Is a group "past every level"? Read with its 95% interval, not its mean alone.
  function placeOf(name, s, levels, v) {
    if (!s) return '';
    const means = levels.map((l) => [l, v.per_level[l]]), inside_ = means.filter(([, m]) => m >= s.ci[0] && m <= s.ci[1]), top = Math.max(...means.map((x) => x[1]));
    const iv = `${f2(s.ci[0])} to ${f2(s.ci[1])}, n ${s.n}`;
    if (s.ci[0] > top) return `${name} sits past every level, and its 95% interval (${iv}) clears the top level’s mean: here it is the connector of connectors.`;
    if (inside_.length) return `${name}: its 95% interval (${iv}) spans the means of ${inside_.length} level${inside_.length > 1 ? 's' : ''} (${inside_.map(([l]) => short(l)).join(', ')}), so its place among them is within noise; it is not past every level.`;
    const above = means.filter(([, m]) => m > s.ci[1]).length;
    return `${name}: its 95% interval (${iv}) sits below ${above} of the ${levels.length} levels, so here it is not the connector of connectors.`;
  }

  // ── 2. the ladder ───────────────────────────────────────────────────────
  function drawLadder() {
    const { run, label } = current(); const v = view(run); const levels = v.levels, SP = spread(v);
    $('cnStepWrap').style.display = $('cnModel').value === 'p410' ? '' : 'none'; if ($('cnModel').value === 'p410') $('cnStepLabel').textContent = steps[+$('cnStep').value].toLocaleString();
    const { g, r, W, H } = canvas($('cnLadder'), 340), left = 190 * r, right = W - 60 * r, top = 26 * r, rowH = (H - top - 40 * r) / (levels.length + 2);
    const vals = [...levels.map((l) => v.per_level[l]), v.language, v.meta, 0, ...(SP ? Object.values(SP).flatMap((s) => s.ci) : [])], lo = Math.min(-0.2, ...vals), hi = Math.max(1.4, ...vals);
    const X = (x) => left + ((x - lo) / (hi - lo)) * (right - left);
    [[0, 'a thing on its own'], [1, 'an average connection']].forEach(([x, t]) => { g.strokeStyle = 'rgba(242,241,238,.25)'; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(X(x), top - 8 * r); g.lineTo(X(x), H - 30 * r); g.stroke(); g.setLineDash([]); g.fillStyle = INK3; g.font = font(r, 9); g.textAlign = 'center'; g.fillText(t, X(x), H - 14 * r); });
    const row = (i, name, x, color, bold, s) => { const y = top + (i + 0.5) * rowH; g.fillStyle = bold ? AMBER : INK2; g.font = font(r, 10); g.textAlign = 'right'; g.fillText(name, left - 10 * r, y + 3 * r);
      g.strokeStyle = 'rgba(242,241,238,.12)'; g.lineWidth = r; g.beginPath(); g.moveTo(X(0), y); g.lineTo(X(x), y); g.stroke();
      if (s) { g.strokeStyle = color; g.lineWidth = 1.2 * r; g.beginPath(); g.moveTo(X(s.ci[0]), y); g.lineTo(X(s.ci[1]), y); [s.ci[0], s.ci[1]].forEach((c) => { g.moveTo(X(c), y - 3 * r); g.lineTo(X(c), y + 3 * r); }); g.stroke(); }
      g.fillStyle = color; g.beginPath(); g.arc(X(x), y, (bold ? 6 : 4.5) * r, 0, Math.PI * 2); g.fill(); g.fillStyle = bold ? AMBER : INK3; g.textAlign = 'left'; g.fillText(`${f2(x)}${s ? ` n${s.n}` : ''}`, X(s ? Math.max(x, s.ci[1]) : x) + 9 * r, y + 3 * r); };
    levels.forEach((l, i) => row(i, `${code(l)} ${short(l)}`, v.per_level[l], shade(i, levels.length), false, SP && SP[l]));
    row(levels.length, 'language between minds', v.language, AMBER, true, SP && SP.language);
    row(levels.length + 1, 'language linking two kinds', v.meta, AMBER, true, SP && SP.meta);
    g.fillStyle = INK2; g.font = font(r, 10); g.textAlign = 'left'; g.fillText(`${label} · ${layerNote(run)} · whiskers: 95% bootstrap interval of each mean`, 12 * r, 14 * r);
    const top3 = levels.map((l) => [l, v.per_level[l]]).sort((a, b) => b[1] - a[1]);
    const verdict = SP ? `${placeOf('Language linking two kinds', SP.meta, levels, v)} ${placeOf('Language between minds', SP.language, levels, v)}`
      : `If language were the connector of connectors it would sit past every level; ${v.meta > top3[0][1] ? 'here its mean does' : 'here its mean does not'} (no sentence positions stored, so no interval).`;
    $('cnLadderWhy').textContent = `Each level’s connection sentences, placed on the model’s shared connection axis. Language sentences sit at ${f2(v.language)}, ${v.language >= 1 ? 'as connection-like as the average level' : v.language >= 0.6 ? 'clearly on the connection side, but below the average level' : 'only partly on the connection side'}; language linking two kinds of connection sits at ${f2(v.meta)}. Most connection-like here: ${short(top3[0][0])} (${f2(top3[0][1])}); least: ${short(top3[top3.length - 1][0])} (${f2(top3[top3.length - 1][1])}). ${verdict}${$('cnModel').value === 'qwen' && run.best_layer === 0 ? ' At the input embeddings (layer 0, Qwen’s best layer by the probe) language sits at ' + f2(run.best.language) + ': much of the shared axis is word choice.' : ''}`;
    PSI.cond($('cnLadderWhy'), metaParts(D.meta, [`n ${D.meta && D.meta.n ? `${D.meta.n.connector} connection, ${D.meta.n.entity} thing, ${D.meta.n.language} language, ${D.meta.n.meta} linking sentences` : 'not recorded'}`]), 'recorded with');
    PSI.canvasLabel($('cnLadder'), `Chart: ${label}, twelve levels and language placed on the connection axis with 95% intervals. Language ${f2(v.language)}, language linking two kinds ${f2(v.meta)}; most connection-like ${short(top3[0][0])} (${f2(top3[0][1])}), least ${short(top3[top3.length - 1][0])} (${f2(top3[top3.length - 1][1])}). ${verdict}`);
    drawMap(); drawGrid();
  }

  // ── 3. the map of sentences ─────────────────────────────────────────────
  let PTS = [];
  function drawMap() {
    const { run, label } = current(); const v = view(run); const S = D.sentences; if (!v.coords || !S) { PTS = []; hint($('cnMap'), 340, 'Nothing recorded for this model: no sentence positions were stored.'); return; }
    const { g, r, W, H } = canvas($('cnMap'), 340), pad = 30 * r;
    const xs = v.coords.map((c) => c[0]), ys = v.coords.map((c) => c[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const X = (x) => pad + ((x - x0) / ((x1 - x0) || 1)) * (W - 2 * pad), Y = (y) => H - pad - ((y - y0) / ((y1 - y0) || 1)) * (H - 2 * pad);
    [0, 1].forEach((x) => { g.strokeStyle = 'rgba(242,241,238,.22)'; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(X(x), pad / 2); g.lineTo(X(x), H - pad / 2); g.stroke(); g.setLineDash([]); g.fillStyle = INK3; g.font = font(r, 9); g.textAlign = 'center'; g.fillText(x === 0 ? 'things' : 'connections', X(x), H - 8 * r); });
    const levels = v.levels; PTS = [];
    S.forEach((s, i) => {
      const x = X(v.coords[i][0]), y = Y(v.coords[i][1]); PTS.push([x / r, y / r, i]);
      const li = levels.indexOf(s.level); const col = s.kind === 'language' || s.kind === 'meta' ? AMBER : shade(li, levels.length);
      g.beginPath(); g.arc(x, y, (s.kind === 'meta' || s.kind === 'language' ? 4 : 3) * r, 0, Math.PI * 2);
      if (s.kind === 'entity') { g.strokeStyle = col; g.lineWidth = 1.2 * r; g.stroke(); }
      else { g.fillStyle = col; g.fill(); if (s.kind === 'meta') { g.strokeStyle = AMBER; g.lineWidth = 1.2 * r; g.beginPath(); g.arc(x, y, 7 * r, 0, Math.PI * 2); g.stroke(); } }
    });
    PSI.canvasLabel($('cnMap'), `Scatter plot: ${S.length} sentences for ${label}, placed on the connection axis; hover a dot, or focus it and use the arrow keys, to read its sentence.`);
  }
  const say = (best) => { const s = D.sentences[best], c = view(current().run).coords[best];
    $('cnHover').textContent = `${s.kind === 'meta' ? 'language linking two kinds of connection' : s.kind === 'language' ? 'language' : `${short(s.level)} · ${s.kind === 'connector' ? 'connection' : 'thing on its own'}`} · position ${f2(c[0])} — “${s.text}”`; };
  $('cnMap').addEventListener('mousemove', (e) => {
    const rc = $('cnMap').getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top; let best = null, bd = 1e9;
    PTS.forEach(([x, y, i]) => { const d = (x - mx) ** 2 + (y - my) ** 2; if (d < bd) { bd = d; best = i; } });
    if (best == null || bd > 200) return; say(best);
  });
  // keyboard: arrow keys step through the sentences from left to right on the axis
  let kIdx = -1; $('cnMap').tabIndex = 0; $('cnHover').setAttribute('aria-live', 'polite');
  $('cnMap').addEventListener('keydown', (e) => {
    if (!PTS.length || !['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return; e.preventDefault();
    const order = PTS.slice().sort((a, b) => a[0] - b[0]).map((p) => p[2]); let k = order.indexOf(kIdx);
    k = e.key === 'Home' ? 0 : e.key === 'End' ? order.length - 1 : Math.max(0, Math.min(order.length - 1, k + (/Right|Down/.test(e.key) ? 1 : -1)));
    kIdx = order[k]; say(kIdx);
  });

  // ── 4. similarity grid ──────────────────────────────────────────────────
  function drawGrid() {
    const { run } = current(); const v = view(run); const MX = v.cos_matrix; if (!MX) { hint($('cnGrid'), 380, 'Nothing recorded for this model: no similarity grid was stored.'); $('cnGridWhy').textContent = ''; return; } const L = v.levels, n = L.length;
    const { g, r, W, H } = canvas($('cnGrid'), 380), left = 150 * r, top = 20 * r, size = Math.min(W - left - 20 * r, H - top - 110 * r), cell = size / n;
    let sum = 0, cnt = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const c = MX[i][j]; if (i !== j) { sum += c; cnt++; }
      const a = i === j ? 0.9 : Math.max(0, c); g.fillStyle = i === j ? '#2a2927' : `rgba(242,241,238,${(a * 1.6).toFixed(2)})`; g.fillRect(left + j * cell, top + i * cell, cell - 1, cell - 1);
      if (i !== j && cell > 26 * r) { g.fillStyle = a > 0.35 ? '#000' : INK3; g.font = font(r, 8); g.textAlign = 'center'; g.fillText(f2(c), left + (j + 0.5) * cell, top + (i + 0.5) * cell + 3 * r); }
    }
    g.font = font(r, 9); g.fillStyle = INK2; g.textAlign = 'right'; L.forEach((l, i) => g.fillText(`${code(l)} ${short(l)}`.slice(0, 24), left - 6 * r, top + (i + 0.5) * cell + 3 * r));
    g.save(); g.textAlign = 'right'; L.forEach((l, j) => { g.save(); g.translate(left + (j + 0.5) * cell, top + size + 6 * r); g.rotate(-Math.PI / 3); g.fillText(code(l), 0, 0); g.restore(); }); g.restore();
    $('cnGridWhy').textContent = `How much each pair of levels’ connection directions point the same way inside the model (1 = identical, 0 = unrelated; brighter = more alike). Average across pairs: ${f2(sum / cnt)}, where shuffling the labels gives ${v.cos_null == null ? 'no number (the shuffle was not stored for this model)' : f2(v.cos_null)}. ${v.loo_mean == null ? 'The leave-one-level-out probe was not stored for this model.' : `A probe that learns “connection” on eleven levels recognises it on the twelfth ${Math.round(v.loo_mean * 100)}% of the time (chance 50%).`} Part of this is the vocabulary of connection (“links”, “between”, “carries”); it is present early in training and partly in the raw word embeddings${D.qwen && D.qwen.best_layer === 0 ? ' (for Qwen3-4B the probe is best at the embeddings themselves)' : ''}.`;
    PSI.cond($('cnGridWhy'), metaParts(D.meta), 'recorded with');
    PSI.canvasLabel($('cnGrid'), `Grid: how alike the ${n} levels’ connection directions are; average across pairs ${f2(sum / cnt)}, shuffled labels ${v.cos_null == null ? 'not stored' : f2(v.cos_null)}.`);
  }
  $('cnModel').addEventListener('change', drawLadder); $('cnStep').addEventListener('input', drawLadder);
  addEventListener('resize', () => { if (!$('tab-connectors').classList.contains('hidden')) { drawLadder(); drawMeter(); } });
  document.querySelector('[data-tab=connectors]').addEventListener('click', () => setTimeout(() => { drawLadder(); onOpen(); }, 0));
})();
