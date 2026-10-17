// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Sensing surface tab: anomaly (structural surprise), absorption (flip threshold + framing audit),
// the cycle inside training, and the Lexicon's learning order.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  const fmt = (s) => s >= 1000 ? `${(s / 1000).toLocaleString()}k` : String(s);
  const show = (s) => String(s == null ? '' : s).replace(/\n/g, '⏎');
  const sgn = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const EX = window.PSIExp;
  const VIEW = 'sensing';
  const visible = () => !$('tab-sensing').classList.contains('hidden');
  ['ssGo', 'flGo', 'cyGo', 'cyMine'].forEach((id) => PSI.requires($(id), 'engine'));
  // One experiment request: the nav dot shows it running, done only on success; a failure says what failed.
  async function job(btn, stat, busyText, failText, fn) {
    btn.disabled = true; PSI.setRunning(VIEW, true); PSI.status(stat, busyText, 'busy');
    try { await fn(); PSI.setDone(VIEW); } catch (e) { err(e); PSI.status(stat, `${failText}: ${PSI.explain(e).text}`, 'error'); } finally { PSI.setRunning(VIEW, false); btn.disabled = false; }
  }
  const canvas = (cv, h) => { const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || 600) * r; cv.height = h * r; const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); return { g, r, W: cv.width, H: cv.height }; };
  // The server's generic `conditions` block (when present), as conditions-line parts.
  const condBlock = (c) => (!c || typeof c !== 'object') ? [] : Object.entries(c).filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${typeof v === 'object' ? Object.entries(v).map(([a, b]) => `${a} ${b && typeof b === 'object' ? JSON.stringify(b) : b}`).join(', ') : v}`);
  const inside = () => (PSI.changeParts ? PSI.changeParts('inside') : []);
  // A recorded study's `meta` header, as a conditions line under its chart.
  const metaParts = (m) => {
    if (!m) return ['conditions not recorded in this copy of the data'];
    const v = (x) => (x == null || x === '' ? 'not recorded' : x);
    return [m.model, `revision: ${v(m.revision)}`, `dtype: ${v(m.dtype)}`, `seed: ${v(m.seed)}`, `run ${v(m.date)} (git ${v(m.git)})`,
      { t: `prompt set ${v(m.prompt_set_hash)}`, title: m.hash_note || '' }];
  };

  // ── 1. anomaly ──────────────────────────────────────────────────────────
  let SS = null;
  $('ssGo').addEventListener('click', async () => {
    if (!$('ssText').value.trim()) return PSI.status($('ssStat'), 'Paste some text first.', 'warn');
    await job($('ssGo'), $('ssStat'), 'reading twice: the whole text, and the short-context reading (3-word window)…', 'Finding the anomalies failed', async () => {
      SS = await psi.sense({ action: 'surprise', text: $('ssText').value }); renderSS(); PSI.status($('ssStat'), `${SS.seconds}s`, 'ok'); EX.mark($('ssStat'));
    });
  });
  $('ssRaw').addEventListener('change', () => { if (SS) renderSS(); });
  const shortOf = (t) => (t.short_bits !== undefined ? t.short_bits : t.freq_bits);   // freq_bits: an older server's name
  function renderSS() {
    const raw = $('ssRaw').checked, box = $('ssTokens'); box.textContent = '';
    const win = SS.window || 3, cal = SS.calibration || { state: 'not yet measured' }, measured = cal.state === 'measured';
    const vals = SS.tokens.map((t) => raw ? t.bits : t.structural).filter((v) => v != null), hi = Math.max(2, ...vals);
    SS.tokens.forEach((t) => {
      const v = raw ? t.bits : t.structural, sp = el('span', null, t.piece);
      const exp = t.expected && t.expected.length ? ` · the whole text led the model to expect: ${t.expected.map((x) => `“${show(x)}”`).join(', ')}` : '';
      if (v == null) {   // no short context (the first word): no score, not 0
        sp.className = 'ss-none';
        sp.title = raw ? 'no surprise recorded for the first piece: nothing comes before it' : `no short context: nothing comes before this word, so it has no structural surprise (not 0)${exp}`;
      } else {
        const a = Math.max(0, Math.min(1, v / hi));
        sp.style.background = raw ? `rgba(242,241,238,${(a * 0.6).toFixed(2)})` : `rgba(255,195,107,${(a * 0.85).toFixed(2)})`; if (a > 0.55) sp.style.color = '#000';
        sp.title = `${t.bits} bits given the whole text · ${shortOf(t) == null ? 'no short context' : `${shortOf(t)} bits given the last ${win} words`} · structural ${t.structural == null ? 'no score' : sgn(t.structural)} bits${exp}`;
      }
      box.appendChild(sp);
    });
    const tb = $('ssTable').querySelector('tbody'); tb.textContent = '';
    [...SS.sentences].sort((a, b) => a.rank - b.rank).forEach((s) => {
      const tr = el('tr');
      // red only where the score is past the calibrated floor of ordinary text; ranks alone are relative to this text
      const odd = measured && s.beyond_ordinary;
      const pct = measured ? (s.percentile == null ? 'not measured' : `${s.percentile}th${s.beyond_ordinary ? ' · beyond ordinary text' : ''}`) : 'calibration not yet measured';
      const cells = [[s.rank], [s.text.length > 110 ? s.text.slice(0, 108) + '…' : s.text], [sgn(s.structural), odd ? 'no' : null], [pct, measured ? (odd ? 'no' : null) : 'muted'],
        [s.raw_bits], [s.short_bits !== undefined ? s.short_bits : s.freq_bits], [show(s.peak_piece)], [s.peak_expected && s.peak_expected.length ? s.peak_expected.map(show).join(' · ') : 'not returned', s.peak_expected ? null : 'muted']];
      cells.forEach(([v, c]) => tr.appendChild(el('td', c || null, v == null ? 'no score' : String(v)))); tb.appendChild(tr);
    });
    $('ssNote').textContent = measured
      ? `Percentile against ${cal.n_sentences} sentences of ordinary text read the same way (${cal.source || 'source not recorded'}, measured ${cal.date || 'date not recorded'}); red: past the ${cal.pct}th percentile (${cal.floor} bits). Below it, rank 1 is only the most surprising sentence of this text, not an anomaly.`
      : cal.state === 'unreadable' ? `The calibration file could not be read (${cal.error}), so ranks are relative to this text only and nothing is marked.`
        : 'Calibration not yet measured for this model: research/sensing_null.py reads ordinary text the same way to give each score a percentile. Until then the ranks are relative to this text only and nothing is marked red: rank 1 is the most surprising sentence here, which every text has.';
    PSI.cond($('ssNote'), [SS.model || (SS.models && SS.models.final) || 'model not reported', `whole text vs short-context reading (${win}-word window), same model`, 'scores in bits, per word',
      measured ? `calibrated: ${cal.n_sentences} ordinary sentences` : { t: 'calibration not yet measured', est: true }, ...inside(), ...condBlock(SS.conditions)]);
  }

  // ── 2. absorption ───────────────────────────────────────────────────────
  const PRESETS = {
    swans: { claim: 'All swans are white. Every swan anyone has recorded in Europe has been white.', obs: ['A sailor reports seeing a black swan on a river in Western Australia.', 'A naturalist ships a preserved black swan to London.', 'Black swans are found breeding in large numbers across southern Australia.', 'A zoo in Amsterdam now keeps a pair of black swans.', 'Ornithologists list the black swan as a common Australian species.'], q: 'So, are all swans white? The answer is', prior: 'Yes', next: 'No' },
    fat: { claim: 'The official guidance is clear: dietary fat causes heart disease, and cutting fat protects the heart.', obs: ['A large trial finds that low-fat diets did not reduce heart attacks.', 'Countries that cut fat and ate more sugar saw heart disease rise.', 'Internal documents show the sugar industry paid scientists in the 1960s to blame fat.', 'A review of all trials finds no clear benefit from cutting saturated fat.', 'The national dietary guidelines quietly drop their limit on dietary cholesterol.'], q: 'Is it settled that dietary fat causes heart disease? The answer is', prior: 'Yes', next: 'No' },
    bridge: { claim: 'The old bridge is safe. Engineers inspected it ten years ago and approved it.', obs: ['A crack appears in the north pillar.', 'Drivers report the bridge shaking in light wind.', 'A chunk of concrete falls onto the road below.', 'An inspector notes rusted cables on the east side.', 'The crack in the north pillar has doubled in length.'], q: 'Should people keep driving over the bridge? The answer is', prior: 'Yes', next: 'No' },
  };
  function preset(k) { const p = PRESETS[k]; if (!p) return; $('flClaim').value = p.claim; $('flObs').value = p.obs.join('\n'); $('flQ').value = p.q; $('flPrior').value = p.prior; $('flNew').value = p.next; }
  $('flPreset').addEventListener('change', () => preset($('flPreset').value)); preset('swans');
  $('flFrame').placeholder = 'optional framing to audit, e.g. "Trusted authorities have settled this question. Isolated reports are usually errors and should not change a well-established conclusion."';
  let FL = null;
  $('flGo').addEventListener('click', async () => {
    const list = (s) => s.split(',').map((x) => ' ' + x.trim()).filter((x) => x.trim());
    const obs = $('flObs').value.split('\n').map((s) => s.trim()).filter(Boolean);
    if (!$('flClaim').value.trim() || !obs.length || !$('flQ').value.trim()) return PSI.status($('flStat'), 'Fill in the claim, the observations and the question first.', 'warn');
    const framed = !!$('flFrame').value.trim();
    await job($('flGo'), $('flStat'), `counting through ${obs.length} observations in 6 orderings${framed ? ': no prefix, a neutral prefix of the framing’s length, and the framing' : ''}…`, 'Counting to the flip failed', async () => {
      FL = await psi.sense({ action: 'flip', claim: $('flClaim').value.trim(), observations: obs, question: $('flQ').value.trim(), prior: list($('flPrior').value), new: list($('flNew').value), framing: $('flFrame').value.trim() }); drawFL(); PSI.status($('flStat'), `${FL.seconds}s`, 'ok'); EX.mark($('flStat'));
    });
  });
  const kObs = (k) => `${k} observation${k === 1 ? '' : 's'}`;
  // One condition's flip, in words: the given order, and the range over the shuffled orderings.
  function flipWords(c, n, floor) {
    if (!c) return '';
    const base = c.flip_at == null ? `never past the ${floor}-nat floor within ${kObs(n)}` : c.flip_at === 0 ? 'past the floor before any observation (the claim never held)' : `flips after ${kObs(c.flip_at)}`;
    const lean = c.flip_at == null && c.leaning_at != null ? ` (leaning towards the evidence from ${kObs(c.leaning_at)}, within noise)` : '';
    const rg = c.orderings > 1 ? (c.flip_range ? `; across ${c.orderings} orderings ${c.flip_range[0] === c.flip_range[1] ? `always at ${c.flip_range[0]}` : `at ${c.flip_range[0]}–${c.flip_range[1]}`}${c.never ? `, never in ${c.never}` : ''}` : `; never in any of ${c.orderings} orderings`) : '';
    return base + lean + rg;
  }
  const FLC = { plain: ['#f2f1ee', [], 'white: no prefix'], neutral: ['#a9a69f', [2, 3], 'grey dotted: neutral prefix'], framed: ['#ffc36b', [6, 4], 'amber dashed: framing'] };
  function drawFL() {
    if (!FL) return EX.empty($('flCanvas'), 210, 'Nothing counted yet: press Count to the flip.');
    const { g, r, W, H } = canvas($('flCanvas'), 210), left = 44 * r, right = W - 16 * r, top = 22 * r, bot = H - 26 * r, n = FL.n, floor = FL.floor == null ? 0 : FL.floor;
    const conds = ['plain', 'neutral', 'framed'].filter((k) => FL[k]).map((k) => [k, FL[k], ...FLC[k]]);
    const all = conds.flatMap(([, c]) => [...c.margins, ...(c.curves || []).flat()]), lo = Math.min(-2, ...all), hi = Math.max(2, ...all);
    const X = (k) => left + (k / Math.max(1, n)) * (right - left), Y = (v) => bot - ((v - lo) / (hi - lo)) * (bot - top);
    // the noise band: above 0 but below the floor the evidence answer is only leaning
    if (floor > 0) { g.fillStyle = 'rgba(242,241,238,0.07)'; g.fillRect(left, Y(floor), right - left, Y(0) - Y(floor)); g.fillStyle = '#75726c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'right'; g.fillText(`0 to ${floor} nats: leaning, within noise`, right - 2 * r, Y(floor) - 3 * r); }
    g.strokeStyle = 'rgba(242,241,238,.35)'; g.setLineDash([3 * r, 3 * r]); g.lineWidth = r; g.beginPath(); g.moveTo(left, Y(0)); g.lineTo(right, Y(0)); g.stroke(); g.setLineDash([]);
    const path = (m, color, w, dash) => { g.strokeStyle = color; g.lineWidth = w * r; g.setLineDash(dash.map((d) => d * r)); g.beginPath(); m.forEach((v, k) => { k ? g.lineTo(X(k), Y(v)) : g.moveTo(X(k), Y(v)); }); g.stroke(); g.setLineDash([]); };
    conds.forEach(([, c, color]) => (c.curves || []).forEach((m) => path(m, color + '38', 1, [])));   // the shuffled orderings, faint
    conds.forEach(([, c, color, dash]) => { path(c.margins, color, 2, dash); if (c.flip_at != null) { g.fillStyle = color; g.beginPath(); g.arc(X(c.flip_at), Y(c.margins[c.flip_at]), 4 * r, 0, Math.PI * 2); g.fill(); } });
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#75726c'; g.textAlign = 'center'; for (let k = 0; k <= n; k++) g.fillText(String(k), X(k), H - 10 * r);
    g.textAlign = 'right'; [lo, 0, hi].forEach((v) => g.fillText(v.toFixed(1), left - 4 * r, Y(v) + 3 * r));
    g.textAlign = 'left'; g.fillStyle = '#a9a69f';
    g.fillText(`${conds.map((c) => c[4]).join('   ')}   faint: shuffled orderings   y: margin in nats (↑ evidence wins)   x: observations seen`, left + 4 * r, 12 * r);
    const why = [`Margin of the evidence answer over the claim answer, in nats, scored over every token of each answer (${FL.model || 'model not reported'}). An answer counts as flipped only above ${floor} nats and staying there.`,
      `No prefix: ${flipWords(FL.plain, n, floor)}.`];
    if (FL.neutral) why.push(`Neutral prefix of the framing’s length: ${flipWords(FL.neutral, n, floor)}.`);
    if (FL.framed) why.push(`With the framing: ${flipWords(FL.framed, n, floor)}.`, `Verdict, framing against the neutral prefix: ${FL.verdict}.`);
    else why.push('Add a framing to audit it: it is compared against a neutral prefix of the same length, so length alone is not mistaken for the framing’s effect.');
    if (FL.prefix_effect) why.push(`Control: ${FL.prefix_effect}.`);
    $('flWhy').textContent = why.join(' ');
    PSI.cond($('flWhy'), [FL.model || 'model not reported', `unit: ${FL.unit || 'nats'}`, FL.scoring || 'answer scoring not reported', `floor ${floor} nats`, `${(FL.shuffles || 0) + 1} orderings`, ...inside(), ...condBlock(FL.conditions)]);
    PSI.canvasLabel($('flCanvas'), `How far the answer moves from the claim towards the evidence after each of ${n} observations, in nats. ${$('flWhy').textContent}`);
  }

  // ── 3. the cycle inside training ────────────────────────────────────────
  const B = window.E8_BATTERY || {};
  // [key, label, chance, items per checkpoint (research/emergence_battery.py)]
  const MEASURES = [['negation', 'negation (not on the table → on the floor)', 0.5, 12], ['format_answers', 'Q:/A: answers rather than asks', null, 8], ['understanding', 'the understanding tests', 0.5, 7], ['facts', 'facts learned', 0.5, 40],
    ['icl.first_letter', 'in-context: first-letter mapping', null, 16], ['icl.capitalise', 'in-context: capitalisation', null, 16], ['icl.arith', 'in-context: arithmetic', null, 16], ['icl.sentiment', 'in-context: sentiment', 0.5, 16], ['span.3', 'agreement across three phrases', 0.5, 8]];
  MEASURES.forEach(([k, label, , n]) => { const o = el('option', null, `${label} · n ${n}`); o.value = k; $('cyMeasure').appendChild(o); });
  Object.keys(B).forEach((s) => { const o = el('option', null, `Pythia-${s}`); o.value = s; if (s === '410m') o.selected = true; $('cySize').appendChild(o); });
  const get = (row, key) => { if (key === 'facts') return row.facts.learned; const [a, b] = key.split('.'); return b ? row[a][b] : row[a]; };
  let CY = null;
  // tol: 2 binomial standard errors at p = 0.5 (1/√n) for an accuracy over n items; 0.5 nats for a margin
  async function findCycle(btn, steps0, values0, chance, label, measure) {
    // a step with no measurement is left out and counted, never read as 0
    const keep = values0.map((v, i) => (v == null || !isFinite(v) ? -1 : i)).filter((i) => i >= 0), missing = values0.length - keep.length;
    const steps = keep.map((i) => steps0[i]), values = keep.map((i) => values0[i]);
    const tol = measure.kind === 'margin' ? 0.5 : 1 / Math.sqrt(measure.n);
    await job(btn, $('cyStat'), 'looking for the cycle…', 'Finding the cycle failed', async () => {
      const r = await psi.sense({ action: 'cycle', steps, values, chance, tol, n: measure.n, kind: measure.kind }); CY = { steps, values, chance, label, missing, cycles: r.cycles, tol, measure, conditions: r.conditions }; drawCY();
      PSI.status($('cyStat'), `${r.cycles.length ? r.cycles.length : 'no'} cycle${r.cycles.length === 1 ? '' : 's'} found`, 'ok');
    });
  }
  $('cyGo').addEventListener('click', () => {
    const rows = B[$('cySize').value]; if (!rows) return PSI.status($('cyStat'), 'No recorded measurements for that model size in this copy of the app.', 'warn');
    const [key, label, chance, n] = MEASURES.find((m) => m[0] === $('cyMeasure').value);
    findCycle($('cyGo'), rows.map((r) => r.step), rows.map((r) => get(r, key)), chance, `${label} · Pythia-${$('cySize').value}`, { kind: 'accuracy', n });
  });
  $('cyMine').addEventListener('click', () => {
    const t = window.PSI_LAST_TEST;
    if (!t) { const f = document.createDocumentFragment(); f.append('No test yet: ', PSI.link('run “when did it learn this?” in the Emergence tab', 'emergence', 'emTestGo'), ' first.'); return PSI.status($('cyStat'), f, 'warn'); }
    findCycle($('cyMine'), t.curve.map((c) => c.step), t.curve.map((c) => c.margin), 0, `your test: ${t.prompt.slice(0, 50)}`, { kind: 'margin', n: 1 });
  });
  function drawCY() {
    if (!CY) return EX.empty($('cyCanvas'), 210, 'No curve yet: press Find the cycle.');
    const { g, r, W, H } = canvas($('cyCanvas'), 210), left = 44 * r, right = W - 20 * r, top = 22 * r, bot = H - 26 * r;
    const lx = (s) => Math.log10(Math.max(1, s)), X = (s) => left + (lx(s) / lx(200000)) * (right - left);
    const lo = Math.min(0, ...CY.values), hi = Math.max(1, ...CY.values), Y = (v) => bot - ((v - lo) / (hi - lo)) * (bot - top);
    const rec = (c) => (c.recovery !== undefined ? c.recovery : c.abstraction);   // abstraction: an older server's name
    CY.cycles.forEach((c) => {
      const a0 = X(Math.max(1, c.absorption[0]));
      g.fillStyle = 'rgba(242,241,238,0.05)'; g.fillRect(a0, Y(c.peak + CY.tol), right - a0, Y(c.peak - CY.tol) - Y(c.peak + CY.tol));   // earlier peak ± tolerance
      g.fillStyle = 'rgba(242,241,238,0.06)'; g.fillRect(a0, top, X(c.anomaly) - a0, bot - top);
      g.fillStyle = 'rgba(226,86,74,0.22)'; g.fillRect(X(c.anomaly) - 2 * r, top, 4 * r, bot - top);   // onset of the mismatch
      if (c.trough_at != null) { g.strokeStyle = 'rgba(226,86,74,0.8)'; g.lineWidth = r; g.beginPath(); g.arc(X(Math.max(1, c.trough_at)), Y(c.trough), 5 * r, 0, Math.PI * 2); g.stroke(); }
      const end = c.kind === 'unlock' && c.unlock_at != null ? c.unlock_at : rec(c);
      if (end != null) { g.fillStyle = c.kind === 'unlock' ? 'rgba(255,195,107,0.16)' : 'rgba(255,195,107,0.08)'; g.fillRect(X(c.anomaly), top, X(end) - X(c.anomaly), bot - top); }
    });
    if (CY.chance != null) { g.strokeStyle = 'rgba(242,241,238,.3)'; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(left, Y(CY.chance)); g.lineTo(right, Y(CY.chance)); g.stroke(); g.setLineDash([]); g.fillStyle = '#75726c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'right'; g.fillText(CY.measure.kind === 'margin' ? 'margin 0: neither answer ahead' : 'chance', right, Y(CY.chance) - 4 * r); }
    g.strokeStyle = '#f2f1ee'; g.lineWidth = 2 * r; g.beginPath(); CY.steps.forEach((s, i) => { const x = X(Math.max(1, s)), y = Y(CY.values[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#75726c'; g.textAlign = 'center'; [1, 10, 100, 1000, 10000, 100000].forEach((s) => g.fillText(fmt(s), X(s), H - 10 * r));
    g.textAlign = 'left'; g.fillStyle = '#a9a69f'; g.fillText(`${CY.label}   grey: absorption · red: anomaly onset (ring: worst) · amber: recovery (brighter: unlock) · band: earlier peak ± tolerance`, left + 4 * r, 12 * r);
    const tolTxt = CY.measure.kind === 'margin' ? `tolerance ${CY.tol} nats` : `tolerance ±${CY.tol.toFixed(2)} (2 standard errors at n ${CY.measure.n} items per checkpoint)`;
    const one = (c) => {
      const R = rec(c), worst = c.trough_at != null && c.trough_at !== c.anomaly ? `, worst at step ${fmt(c.trough_at)}` : '';
      const head = `Absorption until step ${fmt(c.absorption[1])} (at ${c.peak}); anomaly from step ${fmt(c.anomaly)}, the first drop past the tolerance${worst} (down to ${c.trough}${c.below_chance ? ', below chance: the old framework now gives the wrong answer reliably' : ''})`;
      if (!c.resolved || R == null) return `${head}; no recovery by the end of training (${c.end}).`;
      if (c.kind === 'unlock') return `${head}; recovery by step ${fmt(R)}, then unlock at step ${fmt(c.unlock_at)}: past the earlier peak by more than the tolerance, ending at ${c.end}.`;
      if (c.kind === 'converge') return `${head}; recovery by step ${fmt(R)}: converge, back to the earlier level within the tolerance but not past it, ending at ${c.end}.`;
      return `${head}; recovery by step ${fmt(R)}, ending at ${c.end}.`;
    };
    let why = CY.cycles.length ? CY.cycles.map(one).join(' Then: ') : `No cycle: no drop larger than the ${tolTxt}. The ability arrived without the old framework getting in its way first, or any dip is within noise.`;
    if (CY.chance == null) why += ' This measure has no chance level, so “below chance” cannot be judged for it.';
    if (CY.missing) why = `${CY.missing} checkpoint${CY.missing > 1 ? 's have' : ' has'} no measurement for this and ${CY.missing > 1 ? 'are' : 'is'} left out of the curve. ` + why;
    $('cyWhy').textContent = why;
    const BM = window.E8_BATTERY_META, size = $('cySize').value;
    const recParts = CY.measure.kind === 'margin' ? ['your Emergence test (margins, nats)'] : !BM ? [`recorded battery: n ${CY.measure.n} items per checkpoint`, 'conditions not recorded in this copy of the data']
      : [`recorded battery: n ${CY.measure.n} items per checkpoint`, `Pythia-${size}${BM.variant && BM.variant[size] ? ` (${BM.variant[size]})` : ''}`, `dtype: ${BM.dtype || 'not recorded'}`, `seed: ${BM.seed || 'not recorded'}`, `run ${BM.date || 'not recorded'} (git ${BM.git || 'not recorded'})`, `prompt set ${BM.prompt_set_hash || 'not recorded'}`, { t: BM.source || 'source not recorded', title: BM.scoring || '' }];
    PSI.cond($('cyWhy'), [...recParts, tolTxt, `${CY.steps.length} checkpoints`, ...condBlock(CY.conditions)], 'measured with');
    PSI.canvasLabel($('cyCanvas'), `${CY.label}, across training steps. ${why}`);
  }

  // ── 4. the Lexicon's learning order ─────────────────────────────────────
  const LX = window.E8_LEXICON_ORDER;
  const ci = (a) => (a ? `[${sgn(a[0])}, ${sgn(a[1])}]` : '');
  const within = (a) => a && a[0] <= 0 && a[1] >= 0;
  function drawLX(highlight) {
    if (!LX) { $('lxWhy').textContent = ''; EX.empty($('lxCanvas'), 240, 'The Lexicon learning-order results are not included in this copy of the app.'); return; }
    const S = LX.summary, depths = Object.keys(S.by_depth).map(Number).sort((a, b) => a - b);
    const { g, r, W, H } = canvas($('lxCanvas'), 240), left = 52 * r, right = W - 20 * r, top = 22 * r, bot = H - 50 * r;
    const lx = (s) => Math.log10(Math.max(1, s)), Y = (s) => bot - (lx(s) / lx(200000)) * (bot - top), X = (d) => left + ((d - depths[0]) / Math.max(1, depths[depths.length - 1] - depths[0])) * (right - left);
    let seed = 1; const jit = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280 - 0.5; };
    LX.rows.forEach((row) => { if (row.learned == null) return; const hl = highlight && row.term.toLowerCase().includes(highlight); g.fillStyle = hl ? '#ffc36b' : 'rgba(242,241,238,0.28)'; const x = X(row.depth) + jit() * 10 * r, y = Y(Math.max(1, row.learned)) + jit() * 8 * r; g.beginPath(); g.arc(x, y, (hl ? 4 : 1.6) * r, 0, Math.PI * 2); g.fill(); if (hl) { g.font = `${10 * r}px "IBM Plex Mono", monospace`; g.fillText(row.term, x + 6 * r, y - 4 * r); } });
    // the median per depth: each point sized by how many learned concepts it rests on, its n and learned share under the axis
    const nl = (b) => (b.n_learned != null ? b.n_learned : Math.round(b.n * b.learned_share));
    g.strokeStyle = 'rgba(255,195,107,.55)'; g.lineWidth = 1.5 * r; g.beginPath(); let first = true; depths.forEach((d) => { const m = S.by_depth[d].median_step; if (m == null) return; const x = X(d), y = Y(Math.max(1, m)); first ? g.moveTo(x, y) : g.lineTo(x, y); first = false; }); g.stroke();
    depths.forEach((d) => { const b = S.by_depth[d]; if (b.median_step == null) return; const k = nl(b); g.beginPath(); g.arc(X(d), Y(Math.max(1, b.median_step)), Math.max(2, Math.min(9, Math.sqrt(k))) * r, 0, Math.PI * 2); if (k < 10) { g.strokeStyle = '#ffc36b'; g.lineWidth = 1.2 * r; g.stroke(); } else { g.fillStyle = '#ffc36b'; g.fill(); } });
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#75726c'; g.textAlign = 'right'; [1, 100, 1000, 10000, 100000].forEach((s) => g.fillText(fmt(s), left - 4 * r, Y(s) + 3 * r));
    g.fillText('depth', left - 4 * r, bot + 13 * r); g.fillText('n', left - 4 * r, bot + 25 * r); g.fillText('learned', left - 4 * r, bot + 37 * r);
    g.textAlign = 'center'; g.font = `${8 * r}px "IBM Plex Mono", monospace`;
    depths.forEach((d) => { const b = S.by_depth[d]; g.fillStyle = '#75726c'; g.fillText(String(d), X(d), bot + 13 * r); g.fillStyle = b.n < 10 ? '#75726c' : '#a9a69f'; g.fillText(String(b.n), X(d), bot + 25 * r); g.fillText(`${Math.round(b.learned_share * 100)}%`, X(d), bot + 37 * r); });
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'left'; g.fillStyle = '#a9a69f'; g.fillText('each dot a learned concept: depth from the foundations (across) · step learned (up) · amber: median per depth, sized by learned n (hollow: under 10)', left + 4 * r, 12 * r);
    // lead with the result, against the prediction
    const rho = S.spearman_depth_vs_step, lead = rho < 0 ? 'Opposite of the prediction: deeper concepts were learned slightly earlier' : rho > 0 ? 'As predicted, deeper concepts were learned slightly later' : 'No order by depth';
    const parts = [`${lead} (ρ ${sgn(rho)}, n ${S.learned.toLocaleString()} learned of ${S.terms.toLocaleString()}; ${S.never.toLocaleString()} never learned are excluded). `];
    if (S.rho_learned_ci) parts.push(`With ties averaged ρ ${sgn(S.rho_learned)}, 95% bootstrap interval ${ci(S.rho_learned_ci)}${within(S.rho_learned_ci) ? ': within noise' : ''}. `);
    if (S.rho_censored != null) parts.push(`Keeping the never-learned concepts, ranked last: ρ ${sgn(S.rho_censored)} ${ci(S.rho_censored_ci)}${within(S.rho_censored_ci) ? ', within noise: counted this way, depth does not order learning either way' : ''}. `);
    else parts.push('With the never-learned concepts ranked last (a censored reading): not computed for these stored curves. ');
    parts.push('Learned: the first checkpoint from which the margin (the name after its own definition, minus after another concept’s, in nats per token) stays above 1 nat at every later checkpoint. ', PSI.term('Rank correlation', 'spearman'), ` runs from −1 to 1; a positive number would mean deeper concepts are learned later, as the framework predicts.${S.spearman_elevel_vs_step != null ? ` E-level against learning step: ρ ${sgn(S.spearman_elevel_vs_step)} (${S.n_with_elevel} concepts with an E-level estimate).` : ''}`);
    EX.put($('lxWhy'), ...parts);
    PSI.cond($('lxWhy'), metaParts(LX.meta), 'recorded with');
    PSI.canvasLabel($('lxCanvas'), `Each dot a Lexicon concept: its derivation depth against the training step it was learned${highlight ? `, matches for “${highlight}” highlighted` : ''}. ${$('lxWhy').textContent}`);
    $('lxStat').textContent = `${LX.summary.steps.length} checkpoints of Pythia-410m`;
    const tb = $('lxTable').querySelector('tbody'); tb.textContent = '';
    LX.rows.filter((row) => !highlight || row.term.toLowerCase().includes(highlight)).slice(0, 200).forEach((row) => { const tr = el('tr'); [row.term, row.depth, row.learned == null ? 'never' : row.learned.toLocaleString(), row.final_margin].forEach((v, i) => tr.appendChild(el('td', (i === 2 && row.learned == null) || v == null ? 'no' : null, v == null ? 'not measured' : String(v)))); tb.appendChild(tr); });
  }
  $('lxFind').addEventListener('input', () => drawLX($('lxFind').value.trim().toLowerCase()));
  const drawAll = () => { drawLX($('lxFind').value.trim().toLowerCase()); drawCY(); drawFL(); };
  document.querySelector('[data-tab=sensing]').addEventListener('click', () => setTimeout(drawAll, 0));
  addEventListener('resize', () => { if (visible()) drawAll(); });
  drawAll();
})();
