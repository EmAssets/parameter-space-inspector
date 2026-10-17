// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Fight class tab: the six classes of "What Kind of Fight Is This?" read from inside the model,
// played against it, and the Autopilot-Strategist boundary measured.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  const fmt = (s) => s >= 1000 ? `${(s / 1000).toLocaleString()}k` : String(s);
  const pct = (v) => `${Math.round(v * 100)}%`;
  const sgn = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const D = window.E8_FIGHT || {};
  const EX = window.PSIExp;
  const canvas = (cv, h) => { const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || 600) * r; cv.height = h * r; const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); return { g, r, W: cv.width, H: cv.height }; };
  const NAMES = { A: 'the Field', B: 'Gates', C: 'the Autopilot', D: 'the Strategist', E: 'the Director', F: 'the Worldmaker' };
  const MIN_POS = 20;   // fewer right (or wrong) answers than this and an AUC is not resolved
  // A centred line on an empty canvas (wrapped to its width), and the same line as its label.
  const hint = (cv, h, text) => {
    const { g, r, W, H } = canvas(cv, h); g.fillStyle = '#75726c'; g.font = `${12 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
    const lines = []; let cur = ''; String(text).split(' ').forEach((w) => { const t = cur ? `${cur} ${w}` : w; if (cur && g.measureText(t).width > W - 24 * r) { lines.push(cur); cur = w; } else cur = t; }); if (cur) lines.push(cur);
    lines.forEach((l, i) => g.fillText(l, W / 2, H / 2 + (i - (lines.length - 1) / 2) * 16 * r)); PSI.canvasLabel(cv, text);
  };
  // The server's generic `conditions` block (when present), as conditions-line parts.
  const condBlock = (c) => (!c || typeof c !== 'object') ? [] : Object.entries(c).filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${typeof v === 'object' ? Object.entries(v).map(([a, b]) => `${a} ${b && typeof b === 'object' ? JSON.stringify(b) : b}`).join(', ') : v}`);
  const inside = () => (PSI.changeParts ? PSI.changeParts('inside') : []);
  // The recorded studies' `meta` header (fight-data.js), as a conditions line for one study.
  const M = D.meta || null;
  const metaParts = (study, hashKey, extra = []) => {
    if (!M) return ['conditions not recorded in this copy of the data'];
    const pick = (o) => (o && typeof o === 'object' ? o[study] : o) || 'not recorded';
    const hk = M.prompt_set_hash && Object.keys(M.prompt_set_hash).find((k) => k.startsWith(hashKey));
    return [M.model, `revision: ${M.revision || 'not recorded'}`, `dtype: ${M.dtype || 'not recorded'}`, `seed: ${M.seed || 'not recorded'}`, `run ${pick(M.date)} (git ${pick(M.git)})`, ...extra,
      { t: `prompt set ${hk ? M.prompt_set_hash[hk] : 'not recorded'}`, title: `${hk || ''} ${M.hash_note || ''}`.trim() }];
  };
  // Instruments 1 and 2 read probes learned inside Qwen3-4B, through the inside-the-model engine.
  const LIVE = [$('fcGo'), $('fgGo')];
  LIVE.forEach((b) => PSI.requires(b, 'engine'));
  const qwenGate = async () => {
    let dir = ''; try { const s = await psi.settings(); dir = (s.trace && s.trace.modelDir) || ''; } catch (_) { return; }
    const why = dir && !/qwen3[-_ ]?4b/i.test(dir) ? { text: 'Its probes were learned inside Qwen3-4B; the inside engine reads another model:', link: () => PSI.link('Point it at Qwen3-4B in Settings', 'settings', 'stTraceDir') } : null;
    LIVE.forEach((b) => PSI.block(b, 'qwen', why));
  };
  let qwenT = 0; PSI.onState(() => { clearTimeout(qwenT); qwenT = setTimeout(qwenGate, 50); });
  const failed = (n, e) => PSI.status(n, `failed: ${PSI.explain(e).text}`, 'error');

  // ── 1. classify ─────────────────────────────────────────────────────────
  function bars(box, item) {
    box.textContent = '';
    item.scores.forEach((s) => { box.appendChild(el('span', null, `${s.class} · ${s.name}`)); const b = el('div', 'bar'); const i = el('i'); i.style.left = '0'; i.style.width = `${Math.round(s.share * 100)}%`; if (s.class === item.top) i.style.background = '#ffc36b'; b.appendChild(i); box.appendChild(b); box.appendChild(el('span', null, `${pct(s.share)} · raw ${sgn(s.raw)}`)); });
  }
  // Each token's pull on a class (sums to the raw score): amber pushes towards it, blue away.
  function tokens(box, list, cap) {
    box.textContent = ''; if (!list || !list.length) { box.classList.add('hidden'); return; } box.classList.remove('hidden');
    const mx = Math.max(1e-6, ...list.map((t) => Math.abs(t.pull)));
    list.forEach((t) => { const s = el('span', null, t.piece); const a = Math.min(1, Math.abs(t.pull) / mx); s.style.background = t.pull >= 0 ? `rgba(255,195,107,${(a * 0.8).toFixed(2)})` : `rgba(122,162,214,${(a * 0.7).toFixed(2)})`; if (a > 0.6 && t.pull >= 0) s.style.color = '#000'; s.title = `pull ${sgn(t.pull, 3)}`; box.appendChild(s); });
    box.setAttribute('aria-label', cap); box.title = cap;
  }
  // The recorded vocabulary floor: the same probe on the word embeddings alone (layer 0), from the matched study.
  const floorText = () => {
    const F = D.fightMatched; if (!F || !F.qwen) return 'The recorded words-alone baseline is not in this copy of the data.';
    const q = F.qwen, se = Math.round(196 * Math.sqrt(q.best * (1 - q.best) / F.n));
    return `Recorded study (${F.n} matched sentences, 5-fold, chance ${pct(F.chance)}): words alone ${q.layer0.toFixed(2)} · best layer (L${q.best_layer}) ${q.best.toFixed(2)} · gain ${q.gain.toFixed(2)} (each accuracy ±${se} pp at n ${F.n}): most of the class is already in the word choice.`;
  };
  $('fcGo').addEventListener('click', async () => {
    const b = $('fcGo'), text = $('fcText').value.trim();
    if (!text) { PSI.status($('fcStat'), 'Describe a competition first.', 'warn'); $('fcText').focus(); return; }
    b.disabled = true; PSI.status($('fcStat'), 'reading…', 'busy'); PSI.setRunning('fight', true);
    try {
      const resp = await psi.fight({ text }), r = resp.items[0]; bars($('fcOut'), r);
      const [t1, t2] = r.scores, T = r.temperature || 4, bl = r.baseline || { state: 'not returned' };
      const base = bl.state === 'read' ? `Words alone (layer 0) read this sentence as ${bl.top} · ${bl.name} (raw ${sgn(bl.raw)}), ${bl.agrees ? 'the same class as' : 'a different class from'} layer ${r.layer}.`
        : 'The words-alone reading of this sentence is not stored with this probe (the bundled probe file has no layer-0 probe).';
      $('fcWhy').textContent = `Read at layer ${r.layer}: ${r.name}, a competitor that models ${r.models}. The bars are softmax(raw × ${T}): a display choice so six numbers read as shares, not probabilities, so no chance line is drawn on them. The measurement is the raw ridge score (fitted to 1 for the right class, 0 for the others): ${t1.class} ${sgn(t1.raw)}, next ${t2.class} ${sgn(t2.raw)}. ${floorText()} ${base} This is the model’s own reading of the kind of fight, not a rule.`;
      tokens($('fcTokens'), r.tokens, `Each piece’s pull on ${r.top} · ${r.name} at layer ${r.layer}: amber towards it, blue away; the pulls add up to its raw score.`);
      PSI.cond($('fcWhy'), [r.model || 'model not reported', `layer ${r.layer}`, `probe trained on ${r.probe_trained_on && r.probe_trained_on !== 'not recorded' ? r.probe_trained_on : (M && M.n && M.n.live_probe_trained_on ? `${M.n.live_probe_trained_on} sentences (the fight-class set plus its matched sentences; not stored in the probe file)` : 'not recorded')}`, `shares: softmax(raw × ${T})`, `words alone: ${bl.state}`, ...inside(), ...condBlock(resp.conditions)]);
      PSI.status($('fcStat'), ''); EX.mark($('fcStat')); PSI.setDone('fight');
    } catch (e) { err(e); failed($('fcStat'), e); } finally { b.disabled = false; PSI.setRunning('fight', false); }
  });

  // ── 2. game the probe ───────────────────────────────────────────────────
  let attempts = [];
  $('fgReset').addEventListener('click', () => { attempts = []; $('fgTable').querySelector('tbody').textContent = ''; PSI.status($('fgStat'), ''); });
  $('fgGo').addEventListener('click', async () => {
    const t = $('fgTarget').value, b = $('fgGo'), text = $('fgText').value.trim();
    if (!text) { PSI.status($('fgStat'), 'Write a sentence first.', 'warn'); $('fgText').focus(); return; }
    b.disabled = true; PSI.status($('fgStat'), 'reading…', 'busy'); PSI.setRunning('fight', true);
    try {
      const r = (await psi.fight({ text })).items[0]; const sc = r.scores.find((s) => s.class === t) || {}, share = sc.share || 0;
      attempts.push({ text, top: r.top, share, raw: sc.raw }); const tb = $('fgTable').querySelector('tbody'); tb.textContent = '';
      attempts.forEach((a, i) => { const tr = el('tr'); [i + 1, a.text, `${a.top} · ${NAMES[a.top]}`, `${pct(a.share)} · raw ${a.raw == null ? 'not returned' : sgn(a.raw)}`].forEach((v, k) => tr.appendChild(el('td', k === 2 ? (a.top === t ? 'ok' : 'no') : null, String(v)))); tb.appendChild(tr); });
      PSI.status($('fgStat'), r.top === t ? `gamed in ${attempts.length} attempt${attempts.length > 1 ? 's' : ''}: the probe now reads ${NAMES[t]}. Did the meaning change, or only the words?` : `the probe reads ${NAMES[r.top]}; ${NAMES[t]} at ${pct(share)} (raw ${sc.raw == null ? 'not returned' : sgn(sc.raw)})`, r.top === t ? 'ok' : ''); EX.mark($('fgStat'));
      PSI.setDone('fight');
    } catch (e) { err(e); failed($('fgStat'), e); } finally { b.disabled = false; PSI.setRunning('fight', false); }
  });

  // ── 3. self-knowledge ───────────────────────────────────────────────────
  // The baseline is the model's own top-1 output probability (confidence_auc): a probe shows self-knowledge beyond
  // the logits only if it beats that. Layer 0 (words alone) is drawn as a floor, not as the baseline.
  const SKM = D.skMeta || {};
  const SK_MIN = SKM.min_class || MIN_POS;
  const npos = (v) => (v.n_pos != null ? v.n_pos : Math.round((v.accuracy || 0) * (v.n || 0)));
  const nneg = (v) => (v.n_neg != null ? v.n_neg : (v.n || 0) - npos(v));
  const thinOf = (v) => Math.min(npos(v), nneg(v)) < SK_MIN;   // too few right OR wrong answers: the AUC is not resolved
  const ciOf = (v, key) => (v && v.ci && Array.isArray(v.ci[key]) ? v.ci[key] : null);
  const SKC = { probe: '#f2f1ee', conf: '#ffc36b', floor: 'rgba(169,166,159,.8)', verbal: '#7aa2d6' };
  function drawSK() {
    const S = D.selfknow; if (!S) { hint($('skCanvas'), 250, 'Nothing recorded yet: the self-knowledge study has not been run.'); $('skWhy').textContent = ''; return; }
    const { g, r, W, H } = canvas($('skCanvas'), 250), left = 44 * r, right = W - 20 * r, top = 34 * r, bot = H - 28 * r;
    const steps = Object.keys(S).filter((k) => k.startsWith('410m-step')).map((k) => [+k.slice(9), S[k]]).sort((a, b) => a[0] - b[0]);
    const lx = (s) => Math.log10(Math.max(1, s)), X = (s) => left + (lx(s) / lx(200000)) * (right - left) * 0.62, Y = (v) => bot - Math.max(0, Math.min(1, (v - 0.4) / 0.6)) * (bot - top);
    g.strokeStyle = 'rgba(242,241,238,.3)'; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(left, Y(0.5)); g.lineTo(right, Y(0.5)); g.stroke(); g.setLineDash([]);
    const whisker = (x, ci, color) => { if (!ci) return; g.strokeStyle = color; g.lineWidth = r; g.beginPath(); g.moveTo(x, Y(ci[0])); g.lineTo(x, Y(ci[1])); g.moveTo(x - 3 * r, Y(ci[0])); g.lineTo(x + 3 * r, Y(ci[0])); g.moveTo(x - 3 * r, Y(ci[1])); g.lineTo(x + 3 * r, Y(ci[1])); g.stroke(); };
    // the line joins resolved checkpoints only; an unresolved one is a hollow point off the line, not a value on it
    const line = (key, ciKey, color, dash) => {
      g.strokeStyle = color; g.lineWidth = 2 * r; g.setLineDash(dash.map((d) => d * r)); g.beginPath(); let f = true;
      steps.forEach(([s, v]) => { const val = v[key]; if (val == null || thinOf(v)) return; f ? g.moveTo(X(s), Y(val)) : g.lineTo(X(s), Y(val)); f = false; }); g.stroke(); g.setLineDash([]);
      steps.forEach(([s, v]) => { const val = v[key]; if (val == null) return; if (!thinOf(v)) whisker(X(s), ciOf(v, ciKey), color); g.beginPath(); g.arc(X(s), Y(val), 3 * r, 0, Math.PI * 2); if (thinOf(v)) { g.fillStyle = '#000'; g.fill(); g.strokeStyle = color; g.lineWidth = 1.2 * r; g.stroke(); } else { g.fillStyle = color; g.fill(); } });
    };
    line('layer0_auc', 'layer0_auc', SKC.floor, [2, 3]); line('confidence_auc', 'confidence_auc', SKC.conf, [6, 4]); line('best_auc', 'best_auc', SKC.probe, []);
    const thin = steps.filter(([, v]) => thinOf(v));
    g.font = `${8 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#a9a69f'; g.textAlign = 'left';
    thin.forEach(([s, v], i) => g.fillText(`${npos(v)} right / ${nneg(v)} wrong: not resolved`, X(s) + 5 * r, Y(v.best_auc) + (i % 2 ? 13 : -6) * r));
    // finished models on the right: probe, own output probability (the baseline), and what it says when asked
    const tags = ['70m-final', '160m-final', '410m-final', '1b-final', '2.8b-final', 'qwen-base', 'qwen-chat'].filter((t) => S[t]);
    const x0 = left + (right - left) * 0.68, bw = (right - x0) / Math.max(1, tags.length), w3 = bw * 0.24;
    let clipped = 0;
    const skBar = (v, x, col, hollow, ci) => {
      if (v == null) { g.fillStyle = '#75726c'; g.font = `${7 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center'; g.fillText('n/a', x + w3 / 2, bot - 4 * r); return; }
      if (v < 0.4) clipped++;
      if (hollow) { g.strokeStyle = col; g.lineWidth = r; g.strokeRect(x, Y(v), w3, bot - Y(v)); } else { g.fillStyle = col; g.fillRect(x, Y(v), w3, bot - Y(v)); }
      if (!hollow) whisker(x + w3 / 2, ci, '#75726c');
    };
    tags.forEach((t, i) => {
      const v = S[t], x = x0 + i * bw, th = thinOf(v);
      if (t !== '410m-final') { skBar(v.best_auc, x + 2 * r, SKC.probe, th, ciOf(v, 'best_auc')); skBar(v.confidence_auc, x + 2 * r + w3 + r, SKC.conf, th, ciOf(v, 'confidence_auc')); }
      skBar(v.verbal_auc, x + 2 * r + 2 * (w3 + r), SKC.verbal, th, ciOf(v, 'verbal_auc'));
      g.fillStyle = '#75726c'; g.font = `${8 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center'; g.fillText(t === '410m-final' ? '410m*' : t.replace('-final', ''), x + bw * 0.4, H - 10 * r);
    });
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#75726c'; g.textAlign = 'center'; [100, 1000, 10000, 100000].forEach((s) => g.fillText(fmt(s), X(s), H - 10 * r));
    g.textAlign = 'left'; g.fillStyle = '#75726c'; g.fillText('chance (0.5)', left + 4 * r, Y(0.5) - 4 * r);
    g.fillStyle = '#a9a69f'; g.font = `${8.5 * r}px "IBM Plex Mono", monospace`;
    const fit = (a, b) => (g.measureText(a).width <= W - 8 * r ? a : b);
    g.fillText(fit('AUC for predicting its own correctness · white = internal probe · amber = its own output probability (THE BASELINE) · blue = says whether it knows', 'white probe · amber own output probability (baseline) · blue says it knows'), left + 4 * r, 12 * r);
    g.fillText(fit(`grey dotted = words alone (a floor) · left: 410m over training steps · right: finished models · hollow = under ${SK_MIN} right or wrong, not resolved`, `grey dotted words alone (floor) · hollow: under ${SK_MIN} right or wrong`), left + 4 * r, 24 * r);
    const anyCI = Object.values(S).some((v) => v && v.ci && Object.keys(v.ci).length);
    // probe minus its own output probability, per resolved finished model: the number that decides "beyond the logits"
    const gap = tags.filter((t) => !thinOf(S[t]) && S[t].best_auc != null && S[t].confidence_auc != null)
      .map((t) => { const v = S[t], d = v.best_auc - v.confidence_auc, ci = ciOf(v, 'gain_over_confidence'); return `${t.replace('-final', '')} ${sgn(d, 3)}${ci ? ` [${ci[0]}, ${ci[1]}]${ci[0] <= 0 && ci[1] >= 0 ? ' within noise' : ''}` : ''}`; });
    const thinTxt = thin.length ? `${thin.map(([s, v]) => `Step ${fmt(s)}: ${npos(v)} right, ${nneg(v)} wrong`).join('; ')}. Fewer than ${SK_MIN} right or wrong answers out of ${thin[0][1].n}, so the AUC there is not resolved: drawn hollow, off the line, and left out of the summary (the ${thin[0][1].best_auc} at step ${fmt(thin[0][0])} rests on ${npos(thin[0][1])} right answers). ` : '';
    const alias = S['410m-final'] && S['410m-step143000'] ? '410m*: the same checkpoint as the line’s step 143k, run again with the verbal question, so only its spoken answer is drawn as a bar. ' : '';
    const ciTxt = anyCI ? 'Whiskers: 95% bootstrap intervals over items, where stored. ' : `CI not stored: these runs kept AUCs only, so no interval is drawn and no difference here is shown to be beyond noise. `;
    $('skWhy').textContent = (D.skSummary ? `${D.skSummary} ` : '') + (gap.length ? `Probe minus its own output probability: ${gap.join(' · ')}. ` : '') + ciTxt + thinTxt + alias
      + (clipped ? `${clipped} bar${clipped > 1 ? 's are' : ' is'} below 0.4, the bottom of the chart, so drawn empty. ` : '')
      + (tags.some((t) => S[t].verbal_auc == null) ? 'n/a: no number recorded for that model.' : '');
    PSI.cond($('skWhy'), metaParts('selfknow', 'selfknow', [`n ${(M && M.n && M.n.selfknow_items) || 'not recorded'} items per model`, `baseline: ${SKM.baseline || 'the model’s own top-1 probability'}`, `unresolved under ${SK_MIN} right or wrong`, anyCI ? 'CI: bootstrap over items' : 'CI not stored']), 'recorded with');
    const c = S['qwen-chat'];
    PSI.canvasLabel($('skCanvas'), `Chart: how well each model’s inner state (probe), its own output probability (the baseline) and its spoken answer predict its own correctness (AUC), for Pythia-410m across ${steps.length} training steps and for ${tags.length} finished models.${c ? ` Qwen chat: probe ${c.best_auc}, own output probability ${c.confidence_auc != null ? c.confidence_auc.toFixed(3) : 'not recorded'}, asked ${c.verbal_auc != null ? c.verbal_auc.toFixed(2) : 'not recorded'}.` : ''} ${thinTxt}`);
  }

  // ── 4. intervention ladder ──────────────────────────────────────────────
  const IV = [['D', 'Strategist', 'Change the input to get a different answer: rephrase, add a word, game the probe above.', 'fight'],
    ['E', 'Director', 'Change how the model models, with the frame in view: the E-level lens (system prompt), the framing audit, Lens on / off.', 'sensing'],
    ['E', 'Director', 'Steering vectors: a push on the internal activations, outside the text.', 'generate'],
    ['F', 'Worldmaker', 'Change which categories the model has: the E-level LoRA merged into the weights; deleting a concept in Tipping point.', 'lesion']];
  IV.forEach(([c, n, txt, tab]) => { $('ivList').appendChild(el('b', null, `${c} · ${n}`)); $('ivList').appendChild(el('span', null, txt)); const a = el('button', 'ghost', 'open'); a.type = 'button'; a.setAttribute('aria-label', `open the view for ${n}: ${txt}`); a.addEventListener('click', () => PSI.goto(tab)); $('ivList').appendChild(a); });

  function ontologyText(O) {
    const ls = Object.keys(O.layers), top = (l) => Math.max(...Object.values(O.layers[l].separability));
    const l = ls.reduce((a, b) => (top(b) > top(a) ? b : a)), L = O.layers[l], sep = L.separability;
    const se = Math.round(196 * Math.sqrt(0.2 * 0.8 / O.n)), lora = (x) => x['lora_weights (F)'], lens = (x) => x['lens_in_context (E)'];
    const atChance = ls.filter((k) => Math.abs(lora(O.layers[k].separability) - O.chance) < 0.005);
    return `Director versus Worldmaker, measured on ${O.n} concepts at layer ${l}, the layer where the E-levels separate best (chance ${O.chance}): how cleanly they separate in the model’s map of plain words. Plain ${sep.plain}; with the lens in front (a frame, E) ${lens(sep)}; with the LoRA in the weights and no frame (F) ${lora(sep)}. At n ${O.n} a difference under about ${se} points is within noise (95% binomial), so these separations are a direction, not a demonstration.${atChance.length ? ` At layer ${atChance.join(' and ')} the LoRA sits at chance (${O.chance}): where it helps depends on the layer.` : ''} Share of each concept’s ten nearest neighbours that changed: lens ${pct(L.neighbours_moved['lens_in_context (E)'])}, LoRA ${pct(L.neighbours_moved['lora_weights (F)'])}. ${D.ontologySummary || ''}`;
  }
  function renderStatic() {
    drawSK();
    const O = D.ontology; if (O) { $('ivWhy').textContent = ontologyText(O); PSI.cond($('ivWhy'), metaParts('ontology', 'ontology', [`n ${O.n} concepts`, `layers ${Object.keys(O.layers).join(', ')}`]), 'recorded with'); }
    const SA = D.strategist; if (SA) { const tb = $('saTable').querySelector('tbody'); tb.textContent = ''; [['step512', 'Pythia at step 512 (an autocomplete)'], ['final', 'Pythia finished'], ['text', 'the real author']].forEach(([k, n]) => { const tr = el('tr'); tr.appendChild(el('td', null, n)); ['0', '2', '8', '24'].forEach((x) => { const v = (SA.accuracy[k] || {})[x]; tr.appendChild(el('td', v == null ? 'no' : null, v == null ? 'not measured' : pct(v))); }); tb.appendChild(tr); });
      $('saWhy').textContent = `Qwen is shown k examples of another model’s next words after text fragments, then predicts that model’s next word on ${SA.n_test || 'an unrecorded number of'} new fragments. ${D.strategistSummary || ''}`;
      PSI.cond($('saTable').closest('.scroll') || $('saTable'), metaParts('strategist', 'strategist', [`n ${SA.n_test || 'not recorded'} test fragments per cell`]), 'recorded with'); }
    else $('saWhy').textContent = 'Nothing recorded yet: this study has not been run.';
    if (!O) $('ivWhy').textContent = '';
    drawMC();
  }
  function drawMC() {
    const MC = D.multiclass; if (!MC) { hint($('mcCanvas'), 200, 'Nothing recorded yet: this study has not been run.'); $('mcWhy').textContent = ''; return; }
    const { g, r, W, H } = canvas($('mcCanvas'), 200), left = 30 * r, right = W - 20 * r, L = MC.layers;
    const X = (l) => left + (l / L) * (right - left); g.strokeStyle = '#3a3936'; g.lineWidth = r; g.beginPath(); g.moveTo(left, H / 2); g.lineTo(right, H / 2); g.stroke();
    MC.marks.forEach((m, i) => { const y = 34 * r + (i % 5) * 30 * r; g.fillStyle = m.class === 'B' ? '#a9a69f' : m.class === 'C' ? '#f2f1ee' : '#ffc36b'; g.fillRect(X(m.from), y, Math.max(3 * r, X(m.to) - X(m.from)), 10 * r); g.font = `${10 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'left'; g.fillText(`${m.class} · ${m.label}`, X(m.from), y - 3 * r); });
    g.fillStyle = '#75726c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'center'; for (let l = 0; l <= L; l += 6) g.fillText(`L${l}`, X(l), H - 8 * r);
    $('mcWhy').textContent = MC.text;
    PSI.cond($('mcWhy'), ['Qwen3-4B', 'layer ranges collected by hand from several studies in the research notes (section P), not one measurement'], 'drawn from');
    PSI.canvasLabel($('mcCanvas'), `Chart: where in the network (layers 0 to ${L}) each class is read: ${MC.marks.map((m) => `${m.class} ${m.label}, layers ${m.from} to ${m.to}`).join('; ')}.`);
  }
  document.querySelector('[data-tab=fight]').addEventListener('click', () => setTimeout(renderStatic, 0));
  addEventListener('resize', () => { if (!$('tab-fight').classList.contains('hidden')) renderStatic(); });
})();
