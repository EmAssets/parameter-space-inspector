// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Storage ladder tab: how high models store "red" (M0-M8), each rung against the same architecture untrained;
// the evidence behind every verdict, rungs across training, red across cultures, says vs holds, agreement
// between labs, runs/replications of research/model_ladder.py from the app (src/ladder.js), and interventions:
// one level changed, all nine re-measured beside a control change (research/intervention_report.py).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const err = (e, o) => PSI.err(e, o);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const AMBER = '#ffc36b', INK = '#f2f1ee', INK2 = '#a9a69f', INK3 = '#75726c', RULE = 'rgba(242,241,238,.14)';
  const font = (r, px, w = '') => `${w}${px * r}px "IBM Plex Mono", ui-monospace, monospace`;
  const canvas = (cv, h) => { const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || 720) * r; cv.height = h * r; const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); return { g, r, W: cv.width, H: cv.height }; };
  const fmt = (x, d = 2) => (x == null ? '–' : (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(d)));
  // In sentences a missing number is named, never a dash: off-scale is not a value.
  const val = (x, d = 3, none = 'not measured') => (x == null || !isFinite(x) ? none : fmt(x, d));
  // A cell with no number: hatched, so it never reads as a low or failing value.
  const hatch = (g, r, x, y, w, h) => { g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip(); g.strokeStyle = 'rgba(242,241,238,.22)'; g.lineWidth = 1 * r; for (let k = -h; k < w; k += 5 * r) { g.beginPath(); g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); g.stroke(); } g.restore(); };
  const opt = (sel, v, t) => { const o = document.createElement('option'); o.value = v; o.textContent = t; sel.appendChild(o); return o; };

  let D = window.E8_LADDER;
  const LEV = () => D.levels;
  const short = (L) => L.split(' ')[0];
  // M8 is named for what it measures until the crossed design runs: whether what the model says agrees with what it
  // holds, not access to itself. The data key stays 'M8 self-access' (research/model_ladder.py writes it).
  const M8 = 'M8 self-access', M8_NAME = 'self-report coherence (said vs held)';
  const M8_LIMIT = 'Design limit: said and held both pass through the output layer (the rating is read from its word scores, the held link through the output layer’s readout of red’s state), so they can agree because they share that readout; the readout that avoids it (the crossed-colour design) is weak. Read M8 as self-report coherence, not as self-access, until the crossed design has run.';
  const name = (L) => (L === M8 ? 'self-report' : L.slice(L.indexOf(' ') + 1));   // canvas row labels: short
  const full = (L) => (L === M8 ? `M8 ${M8_NAME}` : L);                            // sentences, headings, lists
  // says without tracking: it sorts strong from weak associations when it rates them, while its ratings do not follow its inner state
  const SAY_AUC = 0.8, TRACK_RHO = 0.1;
  const sayNoTrack = (t) => { const d = (D.detail || {})[t], R = row(t), e = R && R.ev && R.ev[M8]; return !!(d && e && d.m8_said_auc != null && e.value != null && d.m8_said_auc >= SAY_AUC && e.value <= TRACK_RHO); };
  const sayNoTrackTxt = (t) => { const d = D.detail[t], e = row(t).ev[M8]; return `Says without tracking: its ratings sort strong from weak associations (said AUC ${fmt(d.m8_said_auc, 2)} ≥ ${SAY_AUC}) while they do not follow where its own state points (rho ${fmt(e.value, 2)} ≤ ${TRACK_RHO}). It can say the right thing without reading it from itself.`; };
  const rows = () => D.rows;
  const released = () => Object.keys(D.rows).filter((t) => D.rows[t].final && !D.rows[t].untrained).sort((a, b) => (D.rows[a].lab + a).localeCompare(D.rows[b].lab + b));
  const trainedAll = () => Object.keys(D.rows).filter((t) => !D.rows[t].untrained);
  const row = (t) => D.rows[t] || (D.replications || {})[t];
  // Rules v2 (model_ladder_report.evidence): rule_text, decidable/why_not, twin_tag/twin_kind, p_str, near_bar, noise, ci,
  // p_bh/pass_bh. Shown when the data has them; older data says so ("rule not stated (old data)"), nothing is invented.
  let BH = false;                                                       // BH-adjusted verdicts (per model), the toggle under card 1
  const passOf = (e) => (!e || e.decidable === false ? false : BH && e.pass_bh != null ? !!e.pass_bh : !!e.pass);   // not decidable: no verdict, never drawn as a pass
  const pTxt = (e) => (!e ? null : e.p_str || (e.p == null ? null : e.p <= 0 ? `p < ${1 / (e.p_n || 1000)}` : `p = ${+(+e.p).toPrecision(3)}`));
  // a permutation p of 0 is below 1/n, never "p = 0.0" (older reports wrote it so in their condition lines)
  const fixP = (txt, e) => String(txt).replace(/p = 0(?:\.0+)?(?![\d.])/g, (e && e.p != null && e.p <= 0 && e.p_str) || `p < ${1 / ((e && e.p_n) || 1000)}`);
  const isSelf = (R, e) => !!((R && R.untrained) || (e && e.twin_kind === 'self'));
  const otherArch = (e) => !!(e && e.twin_kind === 'other-architecture');
  const undecidable = (e) => !!(e && e.decidable === false);
  const nearTxt = (e) => {
    if (!e) return null;
    if (typeof e.noise === 'number' && e.value != null && e.need != null && Math.abs(e.value - e.need) <= e.noise) return `within noise (${fmt(Math.abs(e.value - e.need), 3)} from the bar, noise ±${fmt(e.noise, 3)})`;
    if (e.near_bar) return typeof e.near_bar === 'string' ? e.near_bar : '≈ near the bar';
    return null;
  };
  const twinTag = (R, e) => (e && e.twin_tag) || (R && R.control) || null;
  const twinTxt = (R, e) => {
    if (isSelf(R, e)) return 'none: this model is itself the twin (untrained)';
    const t = twinTag(R, e); if (!t) return 'none: no untrained twin was run';
    const k = e && e.twin_kind;
    const kind = k === 'own' ? 'the same architecture, random weights' : k === 'base' ? 'its base model’s architecture, random weights'
      : k === 'other-architecture' ? `a twin of a different architecture (${e.twin_fallback_family || 'another family'}), random weights` : 'kind not recorded (old data)';
    return `${t}, ${kind}`;
  };
  const ruleTxt = (e) => (e && e.rule_text) || 'rule not stated (old data)';
  const precOf = (t) => { const R = row(t); return (R && (R.precision || (R.provenance && (R.provenance.precision || R.provenance.dtype)))) || null; };
  const precTxt = (t) => precOf(t) || 'precision not recorded';
  const precDiffer = (a, b) => { const x = precOf(a), y = precOf(b); return !!(x && y && x !== y); };
  const repsOf = (t) => Object.entries(D.replications || {}).filter(([, R]) => R.base === t);
  // replications whose verdict on this level differs from the stored run's
  const repSplit = (t, lv) => (D.rows[t] ? repsOf(t).filter(([, R]) => R.ev[lv] && R.ev[lv].value != null && passOf(R.ev[lv]) !== passOf(D.rows[t].ev[lv])) : []);
  // a cell with no verdict to give (not decidable): cross-hatched, unlike "not measured" (single hatch)
  const xhatch = (g, r, x, y, w, h) => { hatch(g, r, x, y, w, h); g.save(); g.translate(x + w, y); g.scale(-1, 1); hatch(g, r, 0, 0, w, h); g.restore(); };
  // one scale per rung across every model, so bars compare between models
  const RANGE = {};
  const range = (lv) => {
    if (RANGE[lv]) return RANGE[lv];
    const v = [0]; Object.values(D.rows).concat(Object.values(D.replications || {})).forEach((R) => { const e = R.ev[lv]; [e.value, e.control, e.need].forEach((x) => { if (x != null && isFinite(x)) v.push(x); }); });
    v.sort((a, b) => a - b); const q = (f) => v[Math.min(v.length - 1, Math.max(0, Math.round(f * (v.length - 1))))];   // tiny checkpoints have wild intervals: 3rd–97th percentile, the rest clipped
    let lo = Math.min(0, q(0.03)), hi = Math.max(0, q(0.97)); const p = (hi - lo || 1) * 0.04; return (RANGE[lv] = [lo - p, hi + p]);
  };

  // A centred line on an empty canvas (wrapped to its width), and the same line as its label.
  const hint = (cv, h, text) => {
    const { g, r, W, H } = canvas(cv, h); g.fillStyle = INK3; g.font = font(r, 12); g.textAlign = 'center'; g.textBaseline = 'middle';
    const lines = []; let cur = ''; String(text).split(' ').forEach((w) => { const t = cur ? `${cur} ${w}` : w; if (cur && g.measureText(t).width > W - 24 * r) { lines.push(cur); cur = w; } else cur = t; }); if (cur) lines.push(cur);
    lines.forEach((l, i) => g.fillText(l, W / 2, H / 2 + (i - (lines.length - 1) / 2) * 16 * r)); PSI.canvasLabel(cv, text);
  };
  const say = (n, text, kind) => PSI.status(n, text, kind);
  // No stored results: plain words first, the terminal commands folded away.
  function showEmpty(on) {
    let box = $('ldEmpty');
    if (!on) { if (box) box.remove(); return; }
    if (!box) {
      box = el('div', 'ld-empty'); box.id = 'ldEmpty';
      box.appendChild(el('p', 'muted small', 'No ladder results in this copy yet. Run the ladder on a model with instrument 7 below (in the desktop app); its results appear here when it finishes.'));
      const d = el('details', 'adv'); d.appendChild(el('summary', null, 'From a terminal'));
      d.appendChild(el('p', 'muted small', 'In the app folder, run the ladder on a Hugging Face model folder, then rebuild the data this view reads:'));
      d.appendChild(el('pre', 'ld-log', 'python research/model_ladder.py --model <model folder> --tag <name>\npython research/make_ladder_data.py'));
      box.appendChild(d); $('ldMatrixWhy').after(box);
    }
    hint($('ldMatrix'), 160, 'Nothing yet: no stored ladder results.');
    $('ldMatrixWhy').textContent = '';
  }

  // ── 1. the matrix ───────────────────────────────────────────────────────
  let cells = [];
  function drawMatrix() {
    const L = LEV(); drawMatrixFixed(30 + L.length * 34 + ($('ldAllCkpt').checked ? 130 : 120));
    const labs = {}; released().forEach((t) => { const lb = D.rows[t].lab; labs[lb] = labs[lb] || new Set(); L.forEach((lv) => { if (passOf(D.rows[t].ev[lv])) labs[lb].add(lv); }); });
    const nl = Object.keys(labs).length;
    // what does not fit leads: a rung failed against the twin below the highest rung passed, and cells with no number
    const gaps = released().filter((t) => D.rows[t].contiguous < D.rows[t].height).map((t) => `${t} (fails ${L.slice(0, D.rows[t].height).filter((lv) => !passOf(D.rows[t].ev[lv])).map(short).join(', ')} below ${short(L[D.rows[t].height])})`);
    const unmeasured = released().reduce((n, t) => n + L.filter((lv) => D.rows[t].ev[lv].value == null).length, 0);
    const undec = released().reduce((n, t) => n + L.filter((lv) => undecidable(D.rows[t].ev[lv])).length, 0);
    const splits = released().flatMap((t) => L.filter((lv) => repSplit(t, lv).length).map((lv) => `${t} ${short(lv)}`));
    const other = released().filter((t) => L.some((lv) => otherArch(D.rows[t].ev[lv])));
    $('ldMatrixWhy').textContent = (gaps.length ? `${gaps.length} of ${released().length} models fail a rung against their twin below one they pass, so the ladder is not climbed in order${gaps.length <= 3 ? `: ${gaps.join('; ')}` : ' (each model’s gaps are listed under its climb, below)'}. ` : '') + (splits.length ? `Replications disagree with the stored run at ${splits.join(', ')} (split cells). ` : '') + (undec ? `${undec} cells not decidable (cross-hatched: no verdict could be given; click one for why). ` : '') + (unmeasured ? `${unmeasured} cells not measured (hatched). ` : '') + (other.length ? `Twin of a different architecture (†): ${other.join(', ')}. ` : '') + `${released().length} released models from ${nl} labs. Rungs passed by every lab: ${L.filter((lv) => Object.values(labs).every((s) => s.has(lv))).map(short).join(', ') || 'none'}. ` +
      `By fewer than half: ${L.filter((lv) => Object.values(labs).filter((s) => s.has(lv)).length < nl / 2).map(short).join(', ') || 'none'}. ` +
      `All nine: ${released().filter((t) => L.every((lv) => passOf(D.rows[t].ev[lv]))).join(', ') || 'none'}. ${precLine(released())} Results of ${D.generated}${D.rule_version ? `, rules ${D.rule_version}` : ', rules version not recorded (old data)'}.`;
    drawChance();
    say($('ldStat'), `${Object.keys(D.rows).length} model states · ${Object.keys(D.replications || {}).length} replications`);
    PSI.canvasLabel($('ldMatrix'), `Matrix: ${released().length} released models from ${nl} labs against the nine rungs M0 to M8; amber cells passed. Use the model and rung lists below it to read any cell.`);
    fillCellPick();
  }
  // the precision each run was loaded at (bf16 / int8 / MXFP4), from its provenance: said, never assumed
  function precLine(tags) {
    const by = {}; tags.forEach((t) => { const p = precOf(t) || 'not recorded'; (by[p] = by[p] || []).push(t); });
    const ks = Object.keys(by);
    if (ks.length === 1 && ks[0] === 'not recorded') return 'Precision not recorded for any of these runs (older results).';
    return `Precision: ${ks.map((k) => `${k === 'not recorded' ? 'not recorded' : k} ${by[k].length}`).join(', ')}${ks.filter((k) => k !== 'not recorded').length > 1 ? '; the compared runs differ in precision' : ''}.`;
  }
  // how many passes chance alone would give: every level a test at p < .01, on every released model
  function drawChance() {
    const box = $('ldChance'); if (!box) return; const L = LEV(), T = released(), M = T.length;
    const cnt = (f) => T.reduce((a, t) => a + L.filter((lv) => f(D.rows[t].ev[lv])).length, 0);
    const dec = (e) => e.decidable !== false, n = cnt(passOf), raw = cnt((e) => dec(e) && !!e.pass), bhn = cnt((e) => dec(e) && !!(e.pass_bh != null ? e.pass_bh : e.pass));
    const hasBH = T.some((t) => L.some((lv) => D.rows[t].ev[lv].pass_bh != null));
    const efp = T.map((t) => D.rows[t].expected_false_passes).filter((x) => typeof x === 'number'), rep = efp.length === M && M > 0;
    const k = rep ? efp.reduce((a, b) => a + b, 0) : L.length * M * 0.01;
    box.textContent = `${n} passes${BH ? ' (BH-adjusted verdicts)' : ''}; expected by chance ≈ ${k.toFixed(1)} (${rep ? `the report’s expected false passes, summed over ${M} models` : `${L.length} levels × ${M} models at p < .01`}).`
      + (hasBH ? ` With each model’s p adjusted for its nine tests (Benjamini–Hochberg): ${bhn} passes; unadjusted: ${raw}.` : ' No adjusted p in this data.')
      + ' Levels decided by a confidence interval (M1, M6) or against other models (M7) have no p to adjust and keep their verdict.';
  }
  if ($('ldBH')) $('ldBH').addEventListener('change', () => { BH = $('ldBH').checked; drawAll(); const c = $('ldCell'); if (c && !c.classList.contains('hidden') && c.dataset.t) showCell(c.dataset.t, c.dataset.lv); });
  // keyboard alternative to clicking a rung: pick a model and a rung
  function fillCellPick() {
    const sm = $('ldCellModel'), sl = $('ldCellLevel'); if (!sm || !sl) return;
    const tags = [...new Set(cells.map((c) => c.t))], keep = sm.value; sm.textContent = '';
    tags.forEach((t) => opt(sm, t, t)); sm.value = keep && tags.includes(keep) ? keep : (tags[0] || '');
    if (!sl.options.length) LEV().forEach((l) => opt(sl, l, full(l)));
  }
  const showPicked = () => { const t = $('ldCellModel').value, lv = $('ldCellLevel').value; if (t && lv && row(t)) showCell(t, lv); };
  if ($('ldCellModel')) { $('ldCellModel').addEventListener('change', showPicked); $('ldCellLevel').addEventListener('change', showPicked); $('ldCellGo').addEventListener('click', showPicked); }
  function drawMatrixFixed(hpx) {
    const cv = $('ldMatrix'); const tags = $('ldAllCkpt').checked ? trainedAll().sort((a, b) => (D.rows[a].family.localeCompare(D.rows[b].family)) || D.rows[a].step - D.rows[b].step) : released();
    const { g, r, W } = canvas(cv, hpx); const L = LEV();
    const x0 = 70 * r, x1 = W - 10 * r, cw = (x1 - x0) / tags.length, rh = 34 * r, y0 = (30 + L.length * 34) * r; cells = [];
    g.textBaseline = 'middle';
    L.forEach((lv, i) => { const y = y0 - (i + 0.5) * rh; g.font = font(r, 11, '500 '); g.fillStyle = INK; g.textAlign = 'right'; g.fillText(short(lv), x0 - 8 * r, y - 6 * r); g.font = font(r, 9); g.fillStyle = INK3; g.fillText(name(lv), x0 - 8 * r, y + 7 * r); });
    let prev = null;
    tags.forEach((t, j) => {
      const R = D.rows[t], x = x0 + j * cw, pad = Math.min(3 * r, cw * 0.12);
      const grp = $('ldAllCkpt').checked ? R.family : R.lab;
      if (prev !== null && grp !== prev) { g.strokeStyle = 'rgba(242,241,238,.3)'; g.lineWidth = 1 * r; g.beginPath(); g.moveTo(x, 18 * r); g.lineTo(x, y0 + 4 * r); g.stroke(); }
      prev = grp;
      L.forEach((lv, i) => {
        const e = R.ev[lv], y = y0 - (i + 1) * rh;
        const cx = x + pad, cy = y + 4 * r, w = cw - 2 * pad, h = rh - 8 * r;
        g.fillStyle = passOf(e) ? AMBER : 'rgba(242,241,238,.07)'; g.fillRect(cx, cy, w, h);
        // a replication that gave the other verdict: the right half in its colour, a dotted line between
        const sp = repSplit(t, lv); if (sp.length) { g.fillStyle = '#000'; g.fillRect(cx + w / 2, cy, w / 2, h); g.fillStyle = passOf(sp[0][1].ev[lv]) ? AMBER : 'rgba(242,241,238,.07)'; g.fillRect(cx + w / 2, cy, w / 2, h); g.strokeStyle = INK; g.lineWidth = 1 * r; g.setLineDash([2 * r, 2 * r]); g.beginPath(); g.moveTo(cx + w / 2, cy); g.lineTo(cx + w / 2, cy + h); g.stroke(); g.setLineDash([]); }
        if (undecidable(e)) xhatch(g, r, cx, cy, w, h); else if (e.value == null) hatch(g, r, cx, cy, w, h);
        if (nearTxt(e) && w > 9 * r) { g.fillStyle = passOf(e) ? '#000' : INK2; g.font = font(r, 11); g.textAlign = 'center'; g.fillText('≈', cx + w / 2, cy + h / 2); }
        cells.push({ x: cx, y: cy, w, h, t, lv });
      });
      if (R.height >= 0) { g.strokeStyle = INK; g.lineWidth = 2 * r; g.beginPath(); const yy = y0 - (R.height + 1) * rh + 2 * r; g.moveTo(x + pad, yy); g.lineTo(x + cw - pad, yy); g.stroke(); }
      g.save(); g.translate(x + cw / 2 - 3 * r, y0 + 10 * r); g.rotate(Math.PI / 3.2); g.fillStyle = INK2; g.textAlign = 'left'; g.font = font(r, tags.length > 30 ? 8 : 10); g.fillText(`${t}${L.some((lv) => otherArch(R.ev[lv])) ? ' †' : ''}${precOf(t) ? ` · ${precOf(t)}` : ''}`, 0, 0); g.restore();
    });
    g.fillStyle = INK3; g.textAlign = 'left'; g.font = font(r, 10);
    const leg = `amber: passed beyond the untrained twin${BH ? ' (BH-adjusted)' : ''} · white line: highest rung reached · hatched: not measured · cross-hatched: not decidable · ≈: near the bar · split: a replication disagrees · †: twin of another architecture · click a rung`;
    if (g.measureText(leg).width <= W - x0 - 4 * r) g.fillText(leg, x0, 10 * r); else g.fillText('legend under the chart · click a rung', x0, 10 * r);
    const lg = $('ldMatrixLegend'); if (lg) lg.textContent = leg;
  }
  // The finer numbers behind a level (make_ladder_data detail): per word, language, colour or context, the twin's beside.
  const DRILL = {
    'M0 words': (e) => [String(e.unit || '').startsWith('foreign') ? 'm0_foreign' : 'm0_related', 'word', 'percentile among 250 common words (pieces)'],
    'M1 units': () => ['m1_xl', 'language', 'AUC of the English-chosen neurons'], 'M2 features': () => ['m2_xl', 'language', 'sense accuracy'],
    'M3 geometry': () => ['m3_xl', 'language', 'rho with the English layout'], 'M4 wiring': () => ['m4', 'colour', 'z'],
    'M5 composition': () => ['m5_contexts', 'context', 'nats'], 'M8 self-access': () => ['m8', 'word', 'said / held'] };
  const LANG = { zh: 'Chinese', hi: 'Hindi', es: 'Spanish' };
  function drill(t, lv, e, twin) {
    const d = D.detail[t] || {}, dt = twin ? D.detail[twin] || null : null, f = DRILL[lv];
    if (lv === 'M6 redundancy') {
      if (d.m6_ci == null && d.m6_facts == null) return null;
      return { head: ['number', 'model', twin ? `twin (${twin})` : 'twin'], rows: [['self-repair', d.m6_self_repair, dt && dt.m6_self_repair], ['self-repair CI', d.m6_ci && `[${fmt(d.m6_ci[0], 3)}, ${fmt(d.m6_ci[1], 3)}]`, dt && dt.m6_ci && `[${fmt(dt.m6_ci[0], 3)}, ${fmt(dt.m6_ci[1], 3)}]`], ['red facts known', d.m6_facts, dt && dt.m6_facts]] };
    }
    if (!f) return null;
    const [key, what, unit] = f(e);
    let m, mt;
    if (key === 'm8') { if (!d.m8_said) return null; m = Object.fromEntries(Object.keys(d.m8_said).map((w) => [w, `${fmt(d.m8_said[w], 2)} / ${fmt((d.m8_held || {})[w], 2)}`])); mt = dt && dt.m8_said ? Object.fromEntries(Object.keys(dt.m8_said).map((w) => [w, `${fmt(dt.m8_said[w], 2)} / ${fmt((dt.m8_held || {})[w], 2)}`])) : null; }
    else { m = d[key]; mt = dt && dt[key]; if (!m) return null; }
    const cell = (v) => (v == null ? null : Array.isArray(v) ? `${fmt(v[0], 1)} (${v[1]})` : v);
    // per language, the share of test-sentence tokens also in the English ones (model_deep.py): a random network can match those
    const tsh = (/^M[123] /.test(lv) && (((DEEP().profiles || {})[t] || {}).tokens_shared || null)) || null;
    const head = [what, `${t}: ${unit}`, twin ? `twin (${twin})` : 'twin'].concat(tsh ? ['tokens shared with English'] : []);
    const rows = Object.keys(m).map((k) => [what === 'language' ? `${LANG[k] || k} (${k})` : what === 'context' ? (CTX[k] || k) : k, cell(m[k]), mt ? cell(mt[k]) : null].concat(tsh ? [tsh[k]] : []));
    return { head, rows };
  }
  function showCell(t, lv) {
    const R = row(t); if (!R) return; const e = R.ev[lv], box = $('ldCell'); box.textContent = ''; box.classList.remove('hidden'); box.dataset.t = t; box.dataset.lv = lv;
    const ok = passOf(e), self = isSelf(R, e), und = undecidable(e), tw = self ? null : twinTag(R, e);
    const h = el('h4'); h.appendChild(el('span', ok ? 'ok' : 'no', self ? '○ ' : und ? '▨ ' : ok ? '✔ ' : '· ')); h.appendChild(document.createTextNode(`${t} · ${full(lv)}${self ? ' · untrained' : und ? ' · not decidable' : ok ? ' · passed' : ' · not passed'}${BH && e.pass_bh != null ? ' (BH-adjusted)' : ''}`)); box.appendChild(h);
    box.appendChild(el('p', null, D.level_info[lv][1]));
    if (lv === M8) { box.appendChild(el('p', 'muted', M8_LIMIT)); if (sayNoTrack(t)) box.appendChild(el('p', 'ld-odd ld-saynotrack', sayNoTrackTxt(t))); }
    box.appendChild(el('p', e.rule_text ? null : 'muted', `Rule: ${ruleTxt(e)}${e.rule_version || R.rule_version ? ` (rules ${e.rule_version || R.rule_version})` : ''}.`));
    if (self) box.appendChild(el('p', 'ld-odd', `${t} has random weights: it is itself the twin, so it has no verdict. Its numbers are the baseline the trained models of its architecture must beat.`));
    if (und) box.appendChild(el('p', 'ld-odd', `Not decidable: ${e.why_not || 'no reason recorded'}. No verdict is given; it is not a fail.`));
    const ci = Array.isArray(e.ci) ? `, CI [${fmt(e.ci[0], 3)}, ${fmt(e.ci[1], 3)}]` : '';
    box.appendChild(el('p', null, `Measured: ${val(e.value)} (${e.unit})${ci}. Untrained twin: ${twinTxt(R, e)}${tw ? `: ${val(e.control, 3)}` : ''}. Bar to clear: ${e.need == null ? 'no bar on this number' : fmt(e.need, 3)}.`));
    const pt = pTxt(e);
    if (pt || e.p_bh != null) box.appendChild(el('p', null, `${pt || 'no p on this level'}${e.p_bh != null ? `; adjusted for the nine tests on this model (Benjamini–Hochberg): ${e.p_bh_str || pTxt({ p: e.p_bh, p_n: e.p_n })}${e.pass_bh != null && !!e.pass_bh !== !!e.pass ? `, which ${e.pass_bh ? 'passes' : 'no longer passes'}` : ''}` : ''}.`));
    if (otherArch(e)) box.appendChild(el('p', 'ld-odd', `The twin is of a different architecture (${e.twin_fallback_family || 'another family'}): no untrained ${R.family} was run, so the baseline does not share this model’s shapes or tokenizer.`));
    const nt = nearTxt(e); if (nt) box.appendChild(el('p', 'ld-odd', `${nt}: a new seed could flip this verdict.`));
    box.appendChild(el('p', 'muted', `Precision: ${precTxt(t)}${tw ? `; twin ${precTxt(tw)}` : ''}.${tw && precDiffer(t, tw) ? ' The model and its twin ran at different precisions.' : ''}`));
    const [lo, hi] = range(lv), out = [['the model', e.value], ['the twin', e.control], ['the bar', e.need]].filter(([, v]) => v != null && (v < lo || v > hi)).map(([w]) => w);
    if (out.length) box.appendChild(el('p', 'no', `Off the chart’s scale (${fmt(lo, 2)} to ${fmt(hi, 2)}, shared by all models): ${out.join(', ')}. The climb chart draws ${out.length > 1 ? 'them' : 'it'} at the edge with an arrow.`));
    (e.conds || []).forEach(([txt, c]) => box.appendChild(el('p', c ? 'ok' : 'no', `${c ? '✔' : '✗'} ${fixP(txt, e)}`)));
    // a replication that disagrees: both verdicts, side by side
    const base = R.base || t, others = [[base, D.rows[base]], ...repsOf(base)].filter(([k, X]) => X && k !== t && X.ev[lv]);
    if (others.length) {
      const diff = others.filter(([, X]) => passOf(X.ev[lv]) !== ok);
      box.appendChild(el('p', diff.length ? 'ld-odd' : null, `${diff.length ? 'Runs of this model disagree here' : 'Other runs of this model agree'}: ${[[t, R], ...others].map(([k, X]) => `${k === base ? 'stored run' : `seed ${X.seed}`} ${passOf(X.ev[lv]) ? '✔' : '·'} ${val(X.ev[lv].value, 3)} (twin ${val(X.ev[lv].control, 3)}${pTxt(X.ev[lv]) ? `, ${pTxt(X.ev[lv])}` : ''})`).join('; ')}.`));
    }
    const dr = drill(t, lv, e, tw);
    if (dr) {
      const wrap = el('div', 'scroll ld-drill'), tb = el('table', 'ld-iv'), th = el('thead'), tr0 = el('tr'); dr.head.forEach((x) => tr0.appendChild(el('th', null, x))); th.appendChild(tr0); tb.appendChild(th);
      const body = el('tbody'); dr.rows.forEach((q) => { const tr = el('tr'); q.forEach((v, i) => tr.appendChild(el('td', i && v == null ? 'flat' : null, i === 0 ? v : v == null ? (i === 2 && !tw ? 'no twin' : 'not stored') : typeof v === 'number' ? fmt(v, 3) : String(v)))); body.appendChild(tr); }); tb.appendChild(body);
      wrap.appendChild(tb); box.appendChild(wrap);
    } else if (lv === 'M7 universality') box.appendChild(el('p', 'muted', 'Per pair of models: card 6 below.'));
    const go = el('div', 'row');
    const b = el('button', 'ghost', `See ${t}’s whole climb`); b.addEventListener('click', () => { $('ldModel').value = t; drawClimb(); $('ldClimb').scrollIntoView({ behavior: 'smooth', block: 'center' }); }); go.appendChild(b);
    const ivb = ivFor(t);
    if (ivb) { const at = Object.values(IV()[ivb].interventions || {}).some((v) => v && v.level === lv && !v.control); const b8 = el('button', 'ghost', at ? `Card 8: ${short(lv)} changed on purpose` : `Card 8: ${short(lv)} under changes to other levels`); b8.addEventListener('click', () => openIv(ivb, lv)); go.appendChild(b8); }
    if ((DEEP().profiles || {})[t]) { const b9 = el('button', 'ghost', `Card 9: ${t} layer by layer`); b9.addEventListener('click', () => { $('ldDeepModel').value = t; drawDeep(); $('ldDeep').scrollIntoView({ behavior: 'smooth', block: 'center' }); }); go.appendChild(b9); }
    if (lv === 'M7 universality') { const b6 = el('button', 'ghost', 'Card 6: agreement between labs'); b6.addEventListener('click', () => { if (D.m7.tags.includes(t)) pickPair(t, tw && D.m7.tags.includes(tw) ? tw : null); $('ldAgree').scrollIntoView({ behavior: 'smooth', block: 'center' }); }); go.appendChild(b6); }
    box.appendChild(go);
  }
  $('ldMatrix').addEventListener('click', (ev) => {
    const cv = $('ldMatrix'), b = cv.getBoundingClientRect(), r = cv.width / b.width, x = (ev.clientX - b.left) * r, y = (ev.clientY - b.top) * r;
    const c = cells.find((k) => x >= k.x && x <= k.x + k.w && y >= k.y && y <= k.y + k.h); if (!c) return; showCell(c.t, c.lv);
    if ($('ldCellModel')) { $('ldCellModel').value = c.t; $('ldCellLevel').value = c.lv; }
  });
  $('ldAllCkpt').addEventListener('change', drawMatrix);

  // ── 2. one model's climb ───────────────────────────────────────────────
  function drawClimb() {
    const t = $('ldModel').value; const R = row(t); if (!R) return; const L = LEV();
    const tw0 = isSelf(R, R.ev[L[0]]) ? null : twinTag(R, R.ev[L[0]]);
    $('ldModelMeta').textContent = `${R.layers || '?'} layers × ${(R.neurons || 0).toLocaleString()} neurons · twin: ${twinTxt(R, R.ev[L[0]])} · ${precTxt(t)}${tw0 && precDiffer(t, tw0) ? ` (twin ${precTxt(tw0)}: precisions differ)` : ''}${R.base ? ` · replication of ${R.base}, seed ${R.seed}` : ''} · highest rung ${R.height >= 0 ? short(L[R.height]) : 'none'}, unbroken to ${R.contiguous >= 0 ? short(L[R.contiguous]) : 'none'}${R.seconds ? ` · ${Math.round(R.seconds)} s` : ''}`;
    const { g, r, W, H } = canvas($('ldClimb'), 470), left = 140 * r, right = W - 90 * r, rh = (H - 20 * r) / L.length;
    g.textBaseline = 'middle'; let clipped = 0, missing = 0;
    L.slice().reverse().forEach((lv, k) => {
      const e = R.ev[lv], y = 10 * r + (k + 0.4) * rh;
      const [lo, hi] = range(lv); const X = (v) => left + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (right - left);
      const ok = passOf(e), und = undecidable(e), near = nearTxt(e);
      g.fillStyle = ok ? INK : INK2; g.font = font(r, 12, '500 '); g.textAlign = 'right'; g.fillText(short(lv), left - 14 * r, y - 6 * r);
      g.font = font(r, 9); g.fillStyle = INK3; g.fillText(name(lv), left - 14 * r, y + 7 * r);
      // the twin and the bar beside the model's own number, in the same place
      g.textAlign = 'left'; g.fillText(`twin ${isSelf(R, e) ? 'itself' : val(e.control, 2, 'none')}${otherArch(e) ? ' (other architecture)' : ''} · bar ${e.need == null ? 'none' : fmt(e.need, 2)}${pTxt(e) ? ` · ${pTxt(e)}` : ''}${near ? ' · ≈ near the bar' : ''} · ${e.unit} · scale ${fmt(lo, 2)} to ${fmt(hi, 2)}`, left, y + 17 * r);
      g.strokeStyle = RULE; g.lineWidth = 1 * r; g.beginPath(); g.moveTo(left, y); g.lineTo(right, y); g.stroke();
      if (lo < 0 && hi > 0) { g.strokeStyle = 'rgba(242,241,238,.3)'; g.beginPath(); g.moveTo(X(0), y - 9 * r); g.lineTo(X(0), y + 9 * r); g.stroke(); }
      // past the shared scale: drawn at the edge with an arrow, counted, never silently clipped
      const offAt = (v) => (v == null ? 0 : v < lo ? -1 : v > hi ? 1 : 0);
      const arrow = (v, col) => { const d = offAt(v); if (!d) return; const x = d > 0 ? right : left; clipped++; g.fillStyle = col; g.beginPath(); g.moveTo(x + d * 7 * r, y); g.lineTo(x, y - 5 * r); g.lineTo(x, y + 5 * r); g.closePath(); g.fill(); };
      if (und) { xhatch(g, r, left, y - 6 * r, right - left, 12 * r); g.fillStyle = INK2; g.font = font(r, 10); g.textAlign = 'left'; g.fillText(`not decidable: ${e.why_not || 'no reason recorded'}`, left + 4 * r, y - 11 * r); }
      if (e.value != null) { const x0 = X(Math.max(lo, Math.min(0, hi))); g.fillStyle = ok ? AMBER : 'rgba(242,241,238,.35)'; g.fillRect(Math.min(x0, X(e.value)), y - 6 * r, Math.max(2 * r, Math.abs(X(e.value) - x0)), 12 * r); arrow(e.value, ok ? AMBER : INK2); }
      else { g.fillStyle = INK3; g.font = font(r, 10); g.fillText('not measured', left + 4 * r, y - 8 * r); missing++; }
      if (e.control != null) { g.fillStyle = INK3; g.fillRect(X(e.control) - 1.5 * r, y - 11 * r, 3 * r, 22 * r); arrow(e.control, INK3); }
      if (e.need != null) { g.strokeStyle = INK; g.lineWidth = 1.5 * r; g.setLineDash([3 * r, 3 * r]); g.beginPath(); g.moveTo(X(e.need), y - 13 * r); g.lineTo(X(e.need), y + 13 * r); g.stroke(); g.setLineDash([]); }
      g.textAlign = 'left'; g.font = font(r, 11); g.fillStyle = ok ? AMBER : INK2; g.fillText(e.value == null ? '· none' : `${isSelf(R, e) ? '○' : und ? '▨' : ok ? '✔' : '·'} ${fmt(e.value, 3)}${offAt(e.value) ? (offAt(e.value) > 0 ? ' ▸' : ' ◂') : ''}${near ? ' ≈' : ''}`, right + 10 * r, y);
    });
    const box = $('ldConds'); box.textContent = '';
    const ruled = L.some((lv) => R.ev[lv].rule_text);
    if (!ruled) box.appendChild(el('div', 'no', 'Rules not stated (old data): the lines below are the report’s checks for each rung.'));
    if (isSelf(R, R.ev[L[0]])) box.appendChild(el('div', 'ld-odd', `${t} is untrained: it is itself the twin, so its rungs have no verdict.`));
    L.forEach((lv) => {
      const e = R.ev[lv];
      if (e.rule_text) box.appendChild(el('div', null, `${short(lv)} rule: ${e.rule_text}`));
      if (undecidable(e)) box.appendChild(el('div', 'ld-odd', `${short(lv)} not decidable: ${e.why_not || 'no reason recorded'}`));
      if (nearTxt(e)) box.appendChild(el('div', 'ld-odd', `${short(lv)} ${nearTxt(e)}`));
      if (otherArch(e) && lv === L[0]) box.appendChild(el('div', 'ld-odd', `twin of a different architecture (${e.twin_fallback_family || 'another family'})`));
      (e.conds || []).forEach(([txt, ok]) => box.appendChild(el('div', ok ? 'ok' : 'no', `${short(lv)} ${ok ? '✔' : '✗'} ${fixP(txt, e)}`)));
    });
    box.appendChild(el('div', 'no', `grey tick: untrained twin · dashed: bar to clear · ≈: near the bar · cross-hatched: not decidable${clipped ? ` · ${clipped} number${clipped > 1 ? 's' : ''} past the scale, drawn at the edge with an arrow (▸ ◂ beside the value)` : ''}${missing ? ` · ${missing} rung${missing > 1 ? 's' : ''} not measured` : ''}`));
    // a rung failed below the highest one passed is the anomaly here: lead with it
    if (R.height >= 0 && R.contiguous < R.height) box.prepend(el('div', 'ld-odd', `Gap: passes ${short(L[R.height])} but fails ${L.slice(0, R.height).filter((lv) => !passOf(R.ev[lv])).map(short).join(', ')} below it, against its twin.`));
    PSI.canvasLabel($('ldClimb'), `Chart: ${t}’s nine rungs against its untrained twin. Passed: ${L.filter((lv) => passOf(R.ev[lv])).map(short).join(', ') || 'none'}.`);
  }
  $('ldModel').addEventListener('change', drawClimb);

  // ── 3. across training ─────────────────────────────────────────────────
  const runs = () => { const f = {}; Object.keys(D.rows).forEach((t) => { const R = D.rows[t]; if (t.includes('@') && !t.endsWith('@init')) (f[R.family] = f[R.family] || []).push(t); }); return Object.fromEntries(Object.entries(f).filter(([, v]) => v.length >= 4)); };
  function drawTrain() {
    const F = runs()[$('ldRun').value]; if (!F) { hint($('ldTrain'), 300, 'Nothing yet: no run with saved training checkpoints is stored.'); hint($('ldTrainLine'), 200, 'Nothing yet.'); $('ldTrainWhy').textContent = ''; return; } const L = LEV();
    const tags = F.slice().sort((a, b) => D.rows[a].step - D.rows[b].step);
    const { g, r, W, H } = canvas($('ldTrain'), 300), left = 150 * r, right = W - 10 * r, top = 10 * r, bottom = H - 40 * r, cw = (right - left) / tags.length, rh = (bottom - top) / L.length;
    g.textBaseline = 'middle';
    L.forEach((lv, i) => {
      const y = bottom - (i + 0.5) * rh; g.fillStyle = INK2; g.font = font(r, 11); g.textAlign = 'right'; g.fillText(`${short(lv)} ${name(lv)}`, left - 8 * r, y);
      tags.forEach((t, j) => {
        const e = D.rows[t].ev[lv], x = left + j * cw;
        let fill = 'rgba(242,241,238,.05)';
        if (passOf(e)) fill = AMBER;
        else if (e.value != null && e.need != null && e.control != null && e.need !== e.control) { const m = Math.max(0, Math.min(1, (e.value - e.control) / (e.need - e.control))); fill = `rgba(242,241,238,${0.05 + 0.45 * m})`; }
        g.fillStyle = fill; g.fillRect(x + 1.5 * r, bottom - (i + 1) * rh + 1.5 * r, cw - 3 * r, rh - 3 * r);
        const bx = x + 1.5 * r, by = bottom - (i + 1) * rh + 1.5 * r, bw = cw - 3 * r, bh = rh - 3 * r;
        if (undecidable(e)) xhatch(g, r, bx, by, bw, bh); else if (e.value == null) hatch(g, r, bx, by, bw, bh);
        if (nearTxt(e) && bw > 8 * r) { g.fillStyle = passOf(e) ? '#000' : INK; g.font = font(r, 10); g.textAlign = 'center'; g.fillText('≈', bx + bw / 2, by + bh / 2); }
      });
    });
    g.textAlign = 'center'; g.fillStyle = INK3; g.font = font(r, 9);
    tags.forEach((t, j) => { const s = D.rows[t].step; g.fillText(s >= 1e9 ? 'final' : s >= 1000 ? `${s / 1000}k` : String(s), left + (j + 0.5) * cw, bottom + 12 * r); });
    g.fillText('training step', (left + right) / 2, bottom + 28 * r);
    // the chosen rung's number across training, against its bar
    const lv = $('ldRunLevel').value; const pts = tags.map((t) => D.rows[t].ev[lv]);
    const { g: g2, r: r2, W: W2, H: H2 } = canvas($('ldTrainLine'), 200), l2 = 150 * r2, rt = W2 - 10 * r2, t2 = 12 * r2, b2 = H2 - 26 * r2;
    const vals = pts.flatMap((e) => [e.value, e.need, e.control]).filter((v) => v != null && isFinite(v)); let lo = Math.min(0, ...vals), hi = Math.max(...vals); const pd = (hi - lo || 1) * 0.1; lo -= pd; hi += pd;
    const Y = (v) => b2 - ((v - lo) / (hi - lo)) * (b2 - t2), Xs = (j) => l2 + (j + 0.5) * ((rt - l2) / tags.length);
    g2.strokeStyle = RULE; g2.lineWidth = 1 * r2; [lo, (lo + hi) / 2, hi].forEach((v) => { g2.beginPath(); g2.moveTo(l2, Y(v)); g2.lineTo(rt, Y(v)); g2.stroke(); g2.fillStyle = INK3; g2.font = font(r2, 9); g2.textAlign = 'right'; g2.fillText(fmt(v, 2), l2 - 6 * r2, Y(v)); });
    const line = (key, col, dash) => { g2.strokeStyle = col; g2.lineWidth = 1.6 * r2; g2.setLineDash(dash ? [4 * r2, 4 * r2] : []); g2.beginPath(); let s = false; pts.forEach((e, j) => { const v = e[key]; if (v == null) { s = false; return; } if (!s) { g2.moveTo(Xs(j), Y(v)); s = true; } else g2.lineTo(Xs(j), Y(v)); }); g2.stroke(); g2.setLineDash([]); };
    line('control', INK3, true); line('need', INK, true); line('value', AMBER, false);
    pts.forEach((e, j) => { if (e.value == null) return; g2.fillStyle = passOf(e) ? AMBER : '#000'; g2.strokeStyle = AMBER; g2.beginPath(); g2.arc(Xs(j), Y(e.value), 3.5 * r2, 0, 7); g2.fill(); g2.stroke(); });
    g2.fillStyle = INK2; g2.textAlign = 'left'; g2.font = font(r2, 10); g2.fillText(`${lv}: ${pts[0].unit}  (amber: the run · grey dashed: its untrained twin · white dashed: bar to clear · filled dot: passed${pts.some((e) => e.value == null) ? ' · gaps: not measured' : ''})`, l2, 4 * r2 + 4 * r2);
    const first = {}; L.forEach((l) => { const j = tags.findIndex((t) => passOf(D.rows[t].ev[l])); if (j >= 0) first[l] = D.rows[tags[j]].step; });
    const order = Object.entries(first).sort((a, b) => a[1] - b[1]).map(([l, s]) => `${short(l)} at ${s >= 1e9 ? 'the end' : s.toLocaleString()}`);
    // a rung that passes, fails at a later checkpoint and passes again: first-passed alone would hide it
    const stepLab = (s) => (s >= 1e9 ? 'final' : s >= 1000 ? `${s / 1000}k` : String(s));
    const flick = L.map((l) => {
      const seq = tags.map((t) => { const e = D.rows[t].ev[l]; return { t, e, v: e.value == null || undecidable(e) ? null : passOf(e) }; }).filter((q) => q.v != null);
      const i = seq.findIndex((q) => q.v), j = i < 0 ? -1 : seq.findIndex((q, k) => k > i && !q.v), k2 = j < 0 ? -1 : seq.findIndex((q, k) => k > j && q.v);
      return k2 < 0 ? null : `${short(l)} (${seq.slice(i).map((q) => `${stepLab(D.rows[q.t].step)} ${q.v ? '✔' : '·'}${nearTxt(q.e) ? '≈' : ''}`).join(' ')})`;
    }).filter(Boolean);
    const anyNear = tags.some((t) => L.some((l) => nearTxt(D.rows[t].ev[l])));
    $('ldTrainWhy').textContent = order.length ? `First passed, in order: ${order.join(', ')}. ${flick.length ? `Checkpoints flicker ✔·✔ (pass, fail, pass again) at: ${flick.join('; ')}, so “first passed” is not “passed from then on”.` : 'No rung flickers: once passed, each stays passed in this run.'}${anyNear ? ' ≈: near the bar, where a new seed could flip the verdict.' : ''} Never: ${L.filter((l) => !(l in first)).map(short).join(', ') || 'none'}.` : 'No rung passed in this run.';
    PSI.canvasLabel($('ldTrain'), `Chart: ${$('ldRun').value} at ${tags.length} training steps, which rungs pass. ${$('ldTrainWhy').textContent}`);
    PSI.canvasLabel($('ldTrainLine'), `Chart: ${lv} across training against its bar.`);
  }
  $('ldRun').addEventListener('change', drawTrain); $('ldRunLevel').addEventListener('change', drawTrain);

  // ── 4. red across cultures (M5) ────────────────────────────────────────
  const CTX = { traffic: 'traffic light', us_election: 'US election map', china_dress: 'China: wedding vs funeral dress', china_envelope: 'China: red vs white envelope', india_dress: 'India: married vs widowed', accounts: 'accounts: loss vs profit', stocks_china: 'stock screen in China (red = rising)', stocks_us: 'stock screen in the US (red = falling)', football: 'football card', wine: 'wine with meat vs fish', ribbon: 'ribbon: AIDS vs breast cancer', beach: 'beach flag', ship: 'ship light: port vs starboard' };
  function drawCtx() {
    const A = $('ldCtxA').value, B = $('ldCtxB').value; const ks = Object.keys(D.crossed);
    const da = (D.detail[A] || {}).m5_contexts || {}, db = B ? ((D.detail[B] || {}).m5_contexts || {}) : null;
    const twin = row(A) && row(A).control, dt = (twin && twin !== A && (D.detail[twin] || {}).m5_contexts) || null;   // the baseline: same architecture, random weights
    const { g, r, W, H } = canvas($('ldCtx'), 420), left = 270 * r, right = W - 90 * r, top = 26 * r, rh = (H - top - 10 * r) / ks.length;
    const vals = ks.flatMap((k) => [da[k], db && db[k], dt && dt[k]]).filter((v) => v != null); const m = Math.max(1, ...vals.map(Math.abs)) * 1.05;
    const X = (v) => left + ((v + m) / (2 * m)) * (right - left);
    g.strokeStyle = 'rgba(242,241,238,.35)'; g.beginPath(); g.moveTo(X(0), top - 6 * r); g.lineTo(X(0), H - 6 * r); g.stroke();
    g.font = font(r, 10); g.fillStyle = INK3; g.textAlign = 'left'; g.fillText('← the other convention', left, 10 * r); g.textAlign = 'right'; g.fillText('red’s meaning in this context →', right, 10 * r);
    g.textBaseline = 'middle';
    ks.forEach((k, i) => {
      const y = top + (i + 0.5) * rh, ans = D.crossed[k].answers; const cols = Object.keys(ans);
      g.textAlign = 'right'; g.fillStyle = INK; g.font = font(r, 11); g.fillText(CTX[k] || k, left - 10 * r, y - 5 * r);
      g.fillStyle = INK3; g.font = font(r, 9); g.fillText(cols.map((c) => `${c} →${ans[c]}`).join('   '), left - 10 * r, y + 8 * r);
      const bar = (v, off, col) => { if (v == null) return; const x0 = X(0); g.fillStyle = col; g.fillRect(Math.min(x0, X(v)), y + off - 4 * r, Math.abs(X(v) - x0), 8 * r); };
      const ca = da[k] != null && da[k] < 0 ? '#e06c5a' : AMBER;
      bar(da[k], db ? -5 * r : 0, ca);
      if (db) bar(db[k], 5 * r, db[k] < 0 ? 'rgba(224,108,90,.55)' : 'rgba(242,241,238,.55)');
      if (dt && dt[k] != null) { g.fillStyle = INK3; g.fillRect(X(dt[k]) - 1.5 * r, y - 10 * r, 3 * r, 20 * r); }
      const nv = (v) => (v == null ? 'n/m' : fmt(v, 1));   // n/m: not measured (said in the line below the chart)
      g.textAlign = 'left'; g.font = font(r, 10); g.fillStyle = INK2; g.fillText(`${nv(da[k])}${db ? ` / ${nv(db[k])}` : ''}`, right + 6 * r, y);
    });
    const wrong = (d) => ks.filter((k) => d[k] != null && d[k] < 0).map((k) => CTX[k] || k);
    $('ldCtxWhy').textContent = `${A}: ${wrong(da).length ? 'applies the other convention for ' + wrong(da).join('; ') : 'right sign everywhere'}.` + (db ? ` ${B} (pale bars): ${wrong(db).length ? wrong(db).join('; ') : 'right sign everywhere'}.` : '') + ' Red bars: the wrong way round. Many models read a red stock in China as falling, the US convention.'
      + (dt ? ` Grey ticks: its untrained twin (${twin}), right sign in ${ks.filter((k) => dt[k] != null && dt[k] > 0).length} of ${ks.filter((k) => dt[k] != null).length} contexts with random weights; a sign the twin also gets proves nothing on its own.` : ' No untrained twin is stored for this model, so there is no baseline here.')
      + (ks.some((k) => da[k] == null || (db && db[k] == null)) ? ' n/m: not measured.' : '');
    PSI.canvasLabel($('ldCtx'), `Chart: red across ${ks.length} contexts for ${A}${B ? ` and ${B}` : ''}. ${$('ldCtxWhy').textContent}`);
  }
  $('ldCtxA').addEventListener('change', drawCtx); $('ldCtxB').addEventListener('change', drawCtx);

  // ── 5. says vs holds (M8) ──────────────────────────────────────────────
  let sayPts = [];
  function drawSay() {
    const t = $('ldSayModel').value, d = D.detail[t]; if (!d || !d.m8_said) { $('ldSayMeta').textContent = ''; hint($('ldSay'), 380, 'Nothing recorded for this model: no says-versus-holds numbers were stored.'); return; } const ev = row(t).ev['M8 self-access'];
    const R5 = row(t), tw5 = isSelf(R5, ev) ? 'random weights: it is itself the twin' : R5.control ? `untrained twin ${R5.control}: ${val(ev.control, 3)}` : `no untrained twin was measured (it would be ${R5.family}@init), so there is no baseline`;
    $('ldSayMeta').textContent = `${M8_NAME}: rho ${fmt(ev.value, 3)}${pTxt(ev) ? ` (${pTxt(ev)})` : ''} · ${tw5} · says strong > weak: AUC ${fmt(d.m8_said_auc, 2)} · holds strong > weak: AUC ${fmt(d.m8_held_auc, 2)} · ${isSelf(R5, ev) ? 'no verdict (untrained)' : undecidable(ev) ? `not decidable: ${ev.why_not || 'no reason recorded'}` : passOf(ev) ? '✔ passes M8 (coherence, not self-access)' : '· does not pass M8'}${nearTxt(ev) ? ` · ${nearTxt(ev)}` : ''}${sayNoTrack(t) ? ` · ⚑ ${sayNoTrackTxt(t)}` : ''} · ${M8_LIMIT}`;
    const words = Object.keys(d.m8_said); const strong = new Set(D.assoc_strong);
    const { g, r, W, H } = canvas($('ldSay'), 380), left = 60 * r, right = W - 20 * r, top = 16 * r, bottom = H - 40 * r;
    const sv = words.map((w) => d.m8_said[w]), hv = words.map((w) => d.m8_held[w]);
    const pad = (a, b) => [a - (b - a || 1) * 0.05, b + (b - a || 1) * 0.08]; const [sl, sh] = pad(Math.min(...sv), Math.max(...sv)), [hl, hh] = pad(Math.min(...hv), Math.max(...hv));
    const X = (v) => left + ((v - sl) / ((sh - sl) || 1)) * (right - left), Y = (v) => bottom - ((v - hl) / ((hh - hl) || 1)) * (bottom - top);
    g.strokeStyle = RULE; g.strokeRect(left, top, right - left, bottom - top);
    g.fillStyle = INK3; g.font = font(r, 10); g.textAlign = 'center'; g.fillText('what it says: its own 0–9 rating of the link to red →', (left + right) / 2, H - 14 * r);
    g.save(); g.translate(16 * r, (top + bottom) / 2); g.rotate(-Math.PI / 2); g.fillText('what it holds: red’s state points to the word →', 0, 0); g.restore();
    sayPts = [];
    words.forEach((w) => {
      const x = X(d.m8_said[w]), y = Y(d.m8_held[w]), s = strong.has(w);
      g.fillStyle = s ? AMBER : INK3; g.beginPath(); g.arc(x, y, (s ? 4 : 3.5) * r, 0, 7); g.fill();
      g.fillStyle = s ? INK2 : INK3; g.font = font(r, 9); g.textAlign = 'left'; g.fillText(w, x + 6 * r, y - 1 * r); sayPts.push({ x, y, w });
    });
    g.fillStyle = INK3; g.textAlign = 'right'; g.font = font(r, 10); g.fillText('amber: strongly tied to red · grey: weakly', right - 6 * r, top + 12 * r);
    PSI.canvasLabel($('ldSay'), `Scatter plot: ${t}, ${words.length} words, what it says against what it holds. ${$('ldSayMeta').textContent}`);
  }
  $('ldSayModel').addEventListener('change', drawSay);

  // ── 6. agreement between labs (M7) ─────────────────────────────────────
  // The comparison the data makes: how much trained models of different labs agree, beside how much their untrained
  // twins agree with one another. M7's own control is a trained model against untrained ones, a different question.
  let agreeGeo = null, agreeSel = null;
  const labOf = (t) => (row(t) ? row(t).lab : t.split(/[-@]/)[0]);
  const isUn = (t) => !!(row(t) && row(t).untrained);
  const mean = (xs) => { const v = xs.filter((x) => x != null && isFinite(x)); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const span = (xs) => { const v = xs.filter((x) => x != null && isFinite(x)); return v.length ? `${fmt(Math.min(...v), 2)}–${fmt(Math.max(...v), 2)}` : 'none'; };
  function m7stats() {
    const tags = D.m7.tags, M = D.m7.rho, ix = (t) => tags.indexOf(t), tr = tags.filter((t) => !isUn(t)), tw = tags.filter(isUn);
    const cross = (t, pool) => mean(pool.filter((b) => b !== t && labOf(b) !== labOf(t)).map((b) => M[ix(t)][ix(b)]));
    const per = tr.map((t) => { const c = row(t) && row(t).control, has = c && tags.includes(c); return { t, tt: cross(t, tr), twin: has ? c : null, ww: has ? cross(c, tw) : null, mixed: cross(t, tw) }; });
    return { tags, M, ix, tr, tw, per, twins: tw.map((t) => ({ t, ww: cross(t, tw) })) };
  }
  const rhoOf = (a, b) => { const T = D.m7.tags, i = T.indexOf(a), j = T.indexOf(b); return i < 0 || j < 0 ? null : D.m7.rho[i][j]; };
  const pOf = (a, b) => { const T = D.m7.tags, i = T.indexOf(a), j = T.indexOf(b); return D.m7.p && i >= 0 && j >= 0 ? D.m7.p[i][j] : null; };
  function pickPair(a, b) {
    const T = D.m7.tags; if (!T.includes(a)) return; if (!b || !T.includes(b)) b = T.find((x) => x !== a && labOf(x) !== labOf(a) && isUn(x) === isUn(a)) || T.find((x) => x !== a);
    agreeSel = [a, b]; if ($('ldAgreeA')) { $('ldAgreeA').value = a; $('ldAgreeB').value = b; }
    drawAgree(); showPair(a, b);
  }
  function showPair(a, b) {
    const box = $('ldAgreePair'); if (!box) return; box.textContent = ''; box.classList.remove('hidden');
    const v = rhoOf(a, b), p = pOf(a, b), kind = (t) => `${isUn(t) ? 'untrained' : 'trained'}, ${labOf(t)}, ${precTxt(t)}`;
    box.appendChild(el('h4', null, `${a} × ${b}`));
    box.appendChild(el('p', null, `rho ${val(v, 3, 'no number for this pair')}${p != null ? ` (${pTxt({ p })})` : ' (no p for this pair in this data)'}. ${a}: ${kind(a)}. ${b}: ${kind(b)}.${precDiffer(a, b) ? ' The two ran at different precisions.' : ''}`));
    if (a === b) box.appendChild(el('p', 'muted', 'A model with itself: 1 by definition.'));
    else if (labOf(a) === labOf(b)) box.appendChild(el('p', 'muted', 'Same lab: M7 counts only pairs from different labs.'));
    const ta = row(a) && !isUn(a) ? row(a).control : null, tb = row(b) && !isUn(b) ? row(b).control : null;
    if (!isUn(a) && !isUn(b)) {
      const w = ta && tb ? rhoOf(ta, tb) : null;
      box.appendChild(el('p', w != null && v != null && w >= v - 0.05 ? 'ld-odd' : null, ta && tb ? `Their untrained twins (${ta} × ${tb}): rho ${val(w, 3, 'no number')}.${w != null && v != null ? (w >= v - 0.05 ? ' The twins agree as much as the trained pair: this agreement is not shown to come from training.' : ` The trained pair agrees ${fmt(v - w, 2)} more than the twins.`) : ''}` : `No untrained twin of ${!ta ? a : b} in this matrix, so there is no twin pair to compare.`));
    } else if (isUn(a) && isUn(b)) box.appendChild(el('p', null, 'Two untrained models: with random weights the layout follows only the words the sentences share, which every model sees alike.'));
    else box.appendChild(el('p', 'muted', 'A trained model against an untrained one: the comparison M7’s verdict uses as its control.'));
  }
  function drawAgree() {
    const tags = D.m7.tags, M = D.m7.rho, n = tags.length;
    const { g, r, W } = canvas($('ldAgree'), 640), left = 170 * r, top = 10 * r, side = Math.max(60 * r, Math.min(W - left - 70 * r, 640 * r - top - 150 * r)), c = side / n;
    agreeGeo = { left, top, c, n };
    tags.forEach((a, i) => tags.forEach((b, j) => {
      const v = M[i][j]; if (v == null) { hatch(g, r, left + j * c, top + i * c, c - 0.5 * r, c - 0.5 * r); return; } const t = Math.max(0, Math.min(1, v));   // hatched: no number for this pair
      g.fillStyle = i === j ? '#222' : `rgba(255,195,107,${(t * t).toFixed(3)})`; g.fillRect(left + j * c, top + i * c, c - 0.5 * r, c - 0.5 * r);
    }));
    if (agreeSel) { const i = tags.indexOf(agreeSel[0]), j = tags.indexOf(agreeSel[1]); if (i >= 0 && j >= 0) { g.strokeStyle = INK; g.lineWidth = 1.5 * r; g.strokeRect(left + j * c, top + i * c, c, c); g.strokeRect(left + i * c, top + j * c, c, c); } }
    g.font = font(r, Math.max(7, Math.min(10, c / r * 0.8))); g.textBaseline = 'middle';
    tags.forEach((t, i) => {
      const lab = `${t}${precOf(t) ? ` · ${precOf(t)}` : ''}`;
      g.fillStyle = isUn(t) ? INK3 : INK2; g.textAlign = 'right'; g.fillText(lab, left - 6 * r, top + (i + 0.5) * c);
      g.save(); g.translate(left + (i + 0.5) * c, top + side + 6 * r); g.rotate(Math.PI / 3); g.textAlign = 'left'; g.fillText(lab, 0, 0); g.restore();
    });
    const lx = left + side + 14 * r; if (lx < W - 60 * r) { for (let k = 0; k <= 10; k++) { g.fillStyle = `rgba(255,195,107,${(k / 10) ** 2})`; g.fillRect(lx, top + (10 - k) * 14 * r, 14 * r, 13 * r); } g.fillStyle = INK3; g.textAlign = 'left'; g.font = font(r, 9); g.fillText('rho 1', lx + 18 * r, top + 6 * r); g.fillText('0', lx + 18 * r, top + 146 * r); }
    const S = m7stats(), tt = S.per.map((q) => q.tt), ww = S.twins.map((q) => q.ww), mtt = mean(tt), mww = mean(ww), mix = mean(S.per.map((q) => q.mixed));
    const asMuch = mtt != null && mww != null && mww >= mtt - 0.05;
    const lead = mtt == null || mww == null ? 'Too few models from different labs to compare.'
      : `Untrained models from different labs agree with each other (rho ${span(ww)} per twin, mean ${fmt(mww, 2)}) ${asMuch ? 'as much as' : mww > mtt ? 'more than' : 'less than'} trained ones do (${span(tt)} per model, mean ${fmt(mtt, 2)}).`;
    const meaning = !asMuch ? ' Training adds agreement beyond what the words alone give.'
      : ' So agreement between trained models is not, on its own, a layout they learned: random-weight networks share it, because the sentences share words. M7’s verdict compares a model with untrained models instead (mean rho ' + fmt(mix, 2) + '), which trained models clearly beat; the table below puts the twin-to-twin numbers beside each model.';
    $('ldAgreeWhy').textContent = lead + meaning + ` ${precLine(tags)}`
      + (() => { const neg = M.flat().filter((v) => v != null && v < 0).length, nul = M.flat().filter((v) => v == null).length; return (neg ? ` ${neg} cells are below 0 and drawn as 0 (the scale starts at 0).` : '') + (nul ? ` ${nul} cells have no number (hatched).` : ''); })()
      + (D.m7.p ? '' : ' No p per pair in this data.') + ' Click a cell (or pick two models) to read a pair.';
    // per model: trained-trained beside twin-twin
    const tb = $('ldAgreeTable') && $('ldAgreeTable').tBodies[0];
    if (tb) {
      tb.textContent = '';
      S.per.slice().sort((a, b) => (b.ww != null && b.tt != null ? b.ww - b.tt : -9) - (a.ww != null && a.tt != null ? a.ww - a.tt : -9)).forEach((q) => {
        const e = (row(q.t).ev || {})['M7 universality'] || {}, tr = el('tr', q.ww != null && q.tt != null && q.ww >= q.tt - 0.05 ? 'ld-odd' : null);
        [q.t, precTxt(q.t), val(q.tt, 2, 'no pair'), q.twin ? `${val(q.ww, 2, 'no pair')} (${q.twin})` : 'no twin in the matrix', val(q.mixed, 2, 'no pair'), `${isSelf(row(q.t), e) ? '–' : passOf(e) ? '✔' : '·'} ${val(e.value, 2)} vs ${val(e.control, 2)}`].forEach((x) => tr.appendChild(el('td', null, x)));
        tr.tabIndex = 0; tr.style.cursor = 'pointer'; tr.setAttribute('aria-label', `${q.t}: read its pair with its twin's counterpart`);
        const open = () => { const o = S.tr.find((x) => x !== q.t && labOf(x) !== labOf(q.t)); pickPair(q.t, o); };
        tr.addEventListener('click', open); tr.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); } });
        tb.appendChild(tr);
      });
    }
    const sa = $('ldAgreeA'), sb = $('ldAgreeB');
    if (sa && sa.options.length !== n) { [sa, sb].forEach((s) => { s.textContent = ''; tags.forEach((t) => opt(s, t, `${t}${isUn(t) ? ' (untrained)' : ''}`)); }); if (agreeSel) { sa.value = agreeSel[0]; sb.value = agreeSel[1]; } }
    PSI.canvasLabel($('ldAgree'), `Matrix: agreement between ${n} models on red’s layout. ${$('ldAgreeWhy').textContent}`);
  }
  $('ldAgree').addEventListener('click', (ev) => {
    if (!agreeGeo) return; const cv = $('ldAgree'), b = cv.getBoundingClientRect(), r = cv.width / b.width, x = (ev.clientX - b.left) * r, y = (ev.clientY - b.top) * r;
    const j = Math.floor((x - agreeGeo.left) / agreeGeo.c), i = Math.floor((y - agreeGeo.top) / agreeGeo.c); if (i < 0 || j < 0 || i >= agreeGeo.n || j >= agreeGeo.n) return;
    pickPair(D.m7.tags[i], D.m7.tags[j]);
  });
  if ($('ldAgreeA')) { const pk = () => pickPair($('ldAgreeA').value, $('ldAgreeB').value); $('ldAgreeA').addEventListener('change', pk); $('ldAgreeB').addEventListener('change', pk); }

  // ── 7. run and replicate ───────────────────────────────────────────────
  const desktop = !document.documentElement.classList.contains('dev');
  // Stored paths are relative to the models folder (Settings > Models); hub: true is a Hugging Face repo id.
  let MODELS = ''; try { window.psi.settings().then((s) => { MODELS = (s && s.modelsDir) || ''; }, () => {}); } catch (_) { /* no settings: fields stay empty */ }
  const fullPath = (R) => (R && R.path && !R.hub && MODELS ? `${MODELS.replace(/[\\/]+$/, '')}/${R.path}` : '');
  const shownPath = (R) => fullPath(R) || (R.hub ? R.path : `<models folder>/${R.path}`);
  function fillRunControls() {
    const s = $('ldStored'); while (s.options.length > 1) s.remove(1);
    Object.keys(D.rows).filter((t) => !D.rows[t].untrained && D.rows[t].path).forEach((t) => opt(s, t, `${t}  (${D.rows[t].path})`));
  }
  async function updateCmd() {
    const dir = $('ldDir').value || '<model folder>', tag = $('ldTag').value.trim() || '<name>', seed = +$('ldSeed').value || 0;
    try { $('ldCmd').textContent = await window.psi.ladder({ action: 'command', modelDir: dir, tag, q8: $('ldQ8').checked, seed }); }
    catch (_) { $('ldCmd').textContent = `python research/model_ladder.py --model ${dir} --tag ${tag}${$('ldQ8').checked ? ' --q8' : ''}${seed ? ` --seed ${seed}` : ''}\npython research/model_ladder.py --model ${dir} --tag ${tag.split('@')[0]}@init --init${$('ldQ8').checked ? ' --q8' : ''}${seed ? ` --seed ${seed}` : ''}\npython research/make_ladder_data.py`; }
  }
  const nextSeed = (tag) => { let s = 1; while (D && D.replications && D.replications[`${tag}~s${s}`]) s++; return s; };
  // Run the ladder: needs the desktop app and a folder (or a stored model).
  const gateGo = () => PSI.block($('ldGo'), 'dir', $('ldDir').value ? null : 'Choose a model folder, or a stored model to replicate.');
  $('ldPick').addEventListener('click', async () => {
    try {
      const r = await window.psi.ladder({ action: 'pick' }); if (!r) return;
      $('ldDir').value = r.dir; $('ldStored').value = '';
      const base = r.dir.replace(/[\\/]+$/, '').split(/[\\/]/).pop().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').slice(0, 60); if (!$('ldTag').value) $('ldTag').value = base;
      if (r.size) { $('ldQ8').checked = !!r.size.q8; $('ldDirMeta').textContent = `${r.size.arch}, ${r.size.layers} layers, about ${(r.size.params / 1e9).toFixed(1)}B parameters${r.size.q8 ? ' · 8-bit' : ''}`; }
      else $('ldDirMeta').textContent = 'no config.json found: the ladder needs a Hugging Face model folder';
      $('ldSeed').value = D && D.rows[$('ldTag').value] ? String(nextSeed($('ldTag').value)) : ''; updateCmd(); gateGo(); $('ldDir').dispatchEvent(new Event('change'));   // kept (PSI.keep)
    } catch (e) { err(e); }
  });
  $('ldStored').addEventListener('change', () => {
    const t = $('ldStored').value; if (!t) return; const R = D.rows[t];
    $('ldDir').value = fullPath(R); $('ldDir').placeholder = fullPath(R) ? 'model folder' : `${R.path} (${R.hub ? 'Hugging Face: download it, then Choose folder' : 'in your models folder: set it in Settings > Models, or Choose folder'})`; $('ldTag').value = t; $('ldSeed').value = String(nextSeed(t));
    const big = (R.layers || 0) * (R.neurons || 0) > 30 * 12000; $('ldQ8').checked = big;
    $('ldDirMeta').textContent = `replication of the stored run (seed 0) with seed ${nextSeed(t)}; ${R.seconds ? `the stored run took ${Math.round(R.seconds / 60)} min` : ''}`;
    updateCmd(); gateGo(); $('ldDir').dispatchEvent(new Event('change'));   // kept (PSI.keep)
  });
  ['ldTag', 'ldSeed', 'ldQ8'].forEach((id) => $(id).addEventListener('input', updateCmd));
  // The run form survives a reload (PSI.keep); a run itself lives on the server and is picked up by its status.
  if (window.PSI && PSI.keep) PSI.keep('ladder', { inputs: ['ldDir', 'ldTag', 'ldSeed', 'ldQ8', 'ldCtl', 'ldFreeTrace', 'ldFreeEngine'] }).ready.then(() => { if ($('ldDir').value) { updateCmd(); gateGo(); } });
  let started = 0, timer = null, prog = null, seen = 0;
  const cancelRun = async () => { try { await window.psi.ladder({ action: 'cancel' }); } catch (e) { err(e); } };
  const progHost = () => { let h = $('ldProg'); if (!h) { h = el('div'); h.id = 'ldProg'; $('ldGo').parentElement.after(h); } return h; };
  const setRunning = (on) => {
    $('ldGo').disabled = on; $('ldCancel').disabled = !on; $('ldLog').classList.remove('hidden');
    if (on) {
      started = Date.now(); seen = 0; clearInterval(timer);
      timer = setInterval(() => say($('ldRunStat'), `running · ${Math.round((Date.now() - started) / 1000)} s`, 'busy'), 1000);
      if (!prog) prog = PSI.progress(progHost(), { label: 'jobs done', view: 'ladder', onCancel: cancelRun });
    } else clearInterval(timer);
  };
  const endProg = (how, text) => { if (!prog) return; if (how === 'ok') prog.finish(text); else if (how === 'stopped') prog.finish(text, { markDone: false }); else prog.fail(text); prog = null; };
  const logLine = (t) => { const p = $('ldLog'); p.textContent += t + '\n'; const lines = p.textContent.split('\n'); if (lines.length > 300) p.textContent = lines.slice(-300).join('\n'); p.scrollTop = p.scrollHeight; };
  $('ldGo').addEventListener('click', async () => {
    try {
      if (!$('ldDir').value) throw new Error('Choose a model folder, or a stored model to replicate.');
      $('ldLog').textContent = ''; setRunning(true); say($('ldRunStat'), 'starting…', 'busy');
      const seedV = $('ldSeed').value.trim();
      const r = await window.psi.ladder({ action: 'run', modelDir: $('ldDir').value, tag: $('ldTag').value.trim(), seed: seedV === '' ? undefined : +seedV, q8: $('ldQ8').checked, control: $('ldCtl').checked, freeTrace: $('ldFreeTrace').checked, freeEngine: $('ldFreeEngine').checked });
      logLine(`${r.tag}${r.seed ? ` · seed ${r.seed}` : ''} · ${r.q8 ? '8-bit' : 'full precision'} · ${r.steps} steps`);
      if (prog && r.steps && !prog.total) prog.setTotal(+r.steps);
    } catch (e) { setRunning(false); endProg('fail', 'did not start'); err(e); say($('ldRunStat'), `did not start: ${PSI.explain(e).text}`, 'error'); }
  });
  $('ldCancel').addEventListener('click', () => { if (prog) prog.cancel(); else cancelRun(); });
  if (window.psi && window.psi.on) window.psi.on('ladder', async (m) => {
    if (m.type === 'line') logLine(m.text);
    else if (m.type === 'step') {
      if (!prog) setRunning(true); seen++;
      if (m.left != null) prog.setTotal(Math.max(prog.total || 0, seen + m.left)); prog.step(seen - 1, `now: ${m.label}`);
    } else if (m.type === 'done') {
      setRunning(false);
      const cancelled = !m.ok && (!m.error || /cancel/i.test(m.error));
      const text = m.ok ? `done in ${Math.round(m.seconds / 60)} min` : cancelled ? 'cancelled' : `stopped: ${m.error}`;
      endProg(m.ok ? 'ok' : cancelled ? 'stopped' : 'fail', text); say($('ldRunStat'), text, m.ok ? 'ok' : cancelled ? 'warn' : 'error');
      if (m.ok) { await refresh(); checkStale(); }
    }
  });
  function drawReps() {
    const tb = $('ldRepTable').querySelector('tbody'); tb.textContent = ''; const reps = D.replications || {}; const L = LEV();
    const flips = (k) => { const R = reps[k], B = D.rows[R.base]; return B ? L.filter((lv) => passOf(R.ev[lv]) !== passOf(B.ev[lv])).length : 0; };
    const disagree = [];
    // replications that disagree with the stored run lead
    Object.keys(reps).sort((a, b) => flips(b) - flips(a) || a.localeCompare(b)).forEach((k) => {
      const R = reps[k], B = D.rows[R.base]; if (!B) return;
      const same = L.filter((lv) => passOf(R.ev[lv]) === passOf(B.ev[lv])), diff = L.filter((lv) => passOf(R.ev[lv]) !== passOf(B.ev[lv]));
      if (diff.length) disagree.push(`${k}: ${diff.map(short).join(', ')}`);
      let big = null; L.forEach((lv) => { const a = B.ev[lv].value, b = R.ev[lv].value; if (a == null || b == null) return; const s = Math.abs(b - a) / (Math.abs(a) + 1e-9); if (!big || s > big.s) big = { s, lv, a, b }; });
      const tr = el('tr', diff.length ? 'ld-odd' : null); tr.appendChild(el('td', null, R.base)); tr.appendChild(el('td', null, String(R.seed)));
      tr.appendChild(el('td', null, `${same.length} of ${L.length}`));
      tr.appendChild(el('td', null, (precDiffer(R.base, k) ? `precision differs (${precTxt(R.base)} → ${precTxt(k)}); ` : '') + (diff.length ? diff.map((lv) => `${short(lv)} ${passOf(B.ev[lv]) ? '✔→·' : '·→✔'} (model ${fmt(B.ev[lv].value, 2)} → ${fmt(R.ev[lv].value, 2)}, twin ${fmt(B.ev[lv].control, 2)} → ${fmt(R.ev[lv].control, 2)}${nearTxt(B.ev[lv]) || nearTxt(R.ev[lv]) ? ', near the bar' : ''})`).join('; ') : 'none')));
      tr.appendChild(el('td', null, big ? `${short(big.lv)} ${fmt(big.a, 3)} → ${fmt(big.b, 3)}` : '–'));
      const open = () => { $('ldModel').value = k; drawClimb(); $('ldClimb').scrollIntoView({ behavior: 'smooth', block: 'center' }); };
      tr.style.cursor = 'pointer'; tr.tabIndex = 0; tr.setAttribute('aria-label', `${k}: open its climb`); tr.addEventListener('click', open);
      tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
      tb.appendChild(tr);
    });
    const n = Object.keys(reps).length;
    $('ldRepWhy').textContent = (disagree.length ? `Disagree with the stored run: ${disagree.join('; ')}. ` : n ? 'Every replication agrees with its stored run on every verdict. ' : '') + (n ? 'A replication re-draws every split, permutation and bootstrap, and makes a new random-weight twin of the same size. The model’s own numbers barely move (they are forward passes); a rung that flips usually does so because its twin moved, or because it sat close to its bar. Click a row (or focus it and press Enter) to see its climb.' : 'No replications yet. Pick a stored model above and run it with a new seed; the comparison appears here.');
  }

  // ── 8. interventions: change one level, re-measure all nine ───────────
  const IV = () => (D && D.interventions) || {};
  const ivOf = (o, k) => (o.interventions || {})[k] || null;
  const isCtl = (o, k) => !!(ivOf(o, k) && ivOf(o, k).control);
  const ivLevel = (o, k) => (ivOf(o, k) && ivOf(o, k).level) || null;
  const num = (x) => (x == null || !isFinite(x) ? '–' : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(3));
  const sgn = (x) => (x == null || !isFinite(x) ? '–' : `${x > 0 ? '+' : x < 0 ? '−' : '±'}${num(Math.abs(x))}`);
  // what kind of control: an inert check (words in no test text) is not a baseline of the same size as the change
  const isInert = (o, k) => { const v = ivOf(o, k); return !!(v && v.control && v.control_kind === 'inert'); };
  const ctlKind = (v) => { const k = v && v.control_kind; if (k === 'inert') return 'inert check (the swapped words are in no test text): not a matched control'; if (k === 'matched') return 'matched control'; if (k) return `control: ${k}`; return 'control, kind not recorded (old data)'; };
  const ivDesc = (o, k) => { const v = ivOf(o, k); if (!v) return k === 'unchanged' ? 'the model as released' : 'no description stored'; const w = v.words || []; return `${v.kind || 'change'}${w.length ? `: ${w.join(w.length === 2 ? ' ↔ ' : ', ')}` : ''}${v.control ? ` (${ctlKind(v)})` : ''}, at ${v.level || 'an unrecorded level'}`; };
  // the intervention base whose unchanged run is this tag (card 1 links here)
  const ivFor = (t) => Object.keys(IV()).find((b) => IV()[b].unchanged_tag === t) || (IV()[t] ? t : null);
  function openIv(b, lv) {
    const o = IV()[b]; if (!o) return; $('ldIvModel').value = b; drawIv();
    const { cols } = ivPlan(o), k = cols.find((c) => c !== 'unchanged' && !isCtl(o, c) && ivLevel(o, c) === lv) || cols.find((c) => c !== 'unchanged' && ivLevel(o, c) === lv) || 'unchanged';
    if (k !== 'unchanged') $('ldIvPick').value = k; $('ldIvLevel').value = short(lv); drawIvTable(); showIvCell(k, lv);
    $('ldIv').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  // columns: unchanged, then each change followed by its control (same level, closest name), then any control left over
  function ivPlan(o) {
    const names = (o.columns || []).filter((c) => c !== 'unchanged'), L = LEV();
    const li = (k) => { const i = L.indexOf(ivLevel(o, k)); return i < 0 ? 99 : i; };
    const pre = (a, b) => { let i = 0; while (i < a.length && a[i] === b[i]) i++; return i; };
    const treats = names.filter((k) => !isCtl(o, k)).sort((a, b) => li(a) - li(b) || a.localeCompare(b));
    const ctlFor = {}, used = new Set();
    treats.forEach((k) => {
      const all = names.filter((c) => isCtl(o, c) && ivLevel(o, c) === ivLevel(o, k)); const free = all.filter((c) => !used.has(c));
      const best = (free.length ? free : all).slice().sort((a, b) => pre(b, k) - pre(a, k))[0];
      if (best) { ctlFor[k] = best; used.add(best); }
    });
    const cols = ['unchanged']; treats.forEach((k) => { cols.push(k); if (ctlFor[k] && !cols.includes(ctlFor[k])) cols.push(ctlFor[k]); });
    names.filter((c) => !cols.includes(c)).sort((a, b) => li(a) - li(b) || a.localeCompare(b)).forEach((c) => cols.push(c));
    return { cols, ctlFor };
  }
  const ivVal = (o, L, k) => ((o.levels || {})[L] || {})[k] || { value: null, pass: false };
  // A control is only a baseline if it is the same size: model_intervene.py records whether the matched control's
  // variance share came within 10% of the target's (M2: share_matched_within_10pct, M3: matched_pcs.matched_within_10pct).
  const ivUnmatched = (o, k) => { const v = ivOf(o, k); if (!v) return false; return v.share_matched_within_10pct === false || (v.matched_pcs && v.matched_pcs.matched_within_10pct === false); };
  // The noise floor for a level: how far a replication (new seed, same model) moved its number from the stored run.
  const ivNoise = (o, lv) => {
    const B = D.rows[o.unchanged_tag]; if (!B) return null; let m = null;
    Object.values(D.replications || {}).filter((R) => R.base === o.unchanged_tag).forEach((R) => { const a = B.ev[lv] && B.ev[lv].value, b = R.ev[lv] && R.ev[lv].value; if (a != null && b != null) m = Math.max(m || 0, Math.abs(b - a)); });
    return m;
  };
  const inNoise = (o, lv, d) => { const n = ivNoise(o, lv); return d != null && n != null && Math.abs(d) <= n; };
  let ivCells = [];
  function ivCmds(base) {
    const R = base && D.rows[base], dir = (R && R.path) ? `"${shownPath(R)}"` : '<model folder>', b = base || '<base>';
    return [`python research/model_ladder.py --model ${dir} --tag ${b}@swap-red-blue --swap red,blue`,
      `python research/model_ladder.py --model ${dir} --tag ${b}@swap-control --swap control`,
      `python research/model_intervene.py --model ${dir} --tag ${b}@<name> --level M1`,
      `python research/model_intervene.py --model ${dir} --tag ${b}@<name>-control --level M1 --control`,
      'python research/intervention_report.py', 'python research/make_ladder_data.py'].join('\n');
  }
  function drawIv() {
    const bases = Object.keys(IV()), base = $('ldIvModel').value, o = IV()[base];
    const empty = !bases.length || !o;
    ['ldIvWrap', 'ldIvWhy', 'ldIvPick', 'ldIvLevel', 'ldIvTable', 'ldIvTableWhy', 'ldIvMoved'].forEach((id) => $(id).classList.toggle('hidden', empty));
    $('ldIvModel').classList.toggle('hidden', !bases.length); $('ldIvEmpty').classList.toggle('hidden', !empty);
    $('ldIvCmd').textContent = ivCmds(o ? base : null);
    if (empty) { $('ldIvMeta').textContent = ''; $('ldIvCell').classList.add('hidden'); return; }
    const L = LEV(), { cols, ctlFor } = ivPlan(o), ctlOf = (k) => ctlFor[k] || null;
    const pv = Object.values(o.precision || {}), pset = [...new Set(pv.filter(Boolean))];
    $('ldIvMeta').textContent = `${cols.length - 1} changed version${cols.length === 2 ? '' : 's'} · verdicts against ${o.control_twin || 'no untrained twin'} · ${!pset.length ? 'precision not recorded' : pset.length > 1 ? `precisions differ between versions (${pset.join(', ')})` : pset[0]}`;
    const wrap = $('ldIvWrap'), cv = $('ldIv'), narrow = (wrap.clientWidth || 720) < 600;
    const lw = narrow ? 44 : 132, minC = narrow ? 80 : 96, head = 50, rhp = 46;
    cv.style.width = `${Math.max(wrap.clientWidth || 720, lw + cols.length * minC + 8)}px`;
    const { g, r, W } = canvas(cv, head + L.length * rhp + (narrow ? 38 : 24)), x0 = lw * r, cw = (W - x0 - 8 * r) / cols.length, rh = rhp * r, y0 = head * r;
    ivCells = []; g.textBaseline = 'middle';
    // column heads: a change and its control share a bracket
    cols.forEach((k, j) => {
      const x = x0 + j * cw, c = isCtl(o, k), lv = ivLevel(o, k);
      g.textAlign = 'center'; g.fillStyle = k === 'unchanged' ? INK : c ? INK2 : AMBER; g.font = font(r, 10, '500 ');
      let t = k; const maxw = cw - 6 * r; while (t.length > 3 && g.measureText(t).width > maxw) t = `${t.slice(0, -2)}…`;
      g.fillText(t, x + cw / 2, 14 * r);
      g.font = font(r, 9); g.fillStyle = c && ivUnmatched(o, k) ? '#e06c5a' : INK3; g.fillText(k === 'unchanged' ? 'as released' : `${c ? (isInert(o, k) ? 'inert check' : ivUnmatched(o, k) ? 'control, size off' : 'control') : 'change'} · ${lv ? short(lv) : 'level ?'}`, x + cw / 2, 28 * r);
      if (!c && ctlOf(k) && cols[j + 1] === ctlOf(k)) { g.strokeStyle = 'rgba(242,241,238,.3)'; g.lineWidth = 1 * r; g.beginPath(); g.moveTo(x + 6 * r, 40 * r); g.lineTo(x + 2 * cw - 6 * r, 40 * r); g.stroke(); }
      if (j > 0 && !c) { g.strokeStyle = 'rgba(242,241,238,.3)'; g.lineWidth = 1 * r; g.beginPath(); g.moveTo(x, 4 * r); g.lineTo(x, y0 + L.length * rh); g.stroke(); }
    });
    L.forEach((lv, i) => {
      const y = y0 + i * rh, yc = y + rh / 2, u = ivVal(o, lv, 'unchanged'), [lo, hi] = range(lv), span = Math.max(1e-9, (hi - lo) / 2);
      g.textAlign = 'right'; g.fillStyle = INK; g.font = font(r, 11, '500 '); g.fillText(short(lv), x0 - 8 * r, narrow ? yc : yc - 6 * r);
      if (!narrow) { g.font = font(r, 9); g.fillStyle = INK3; g.fillText(name(lv), x0 - 8 * r, yc + 7 * r); }
      cols.forEach((k, j) => {
        const e = ivVal(o, lv, k), x = x0 + j * cw, pad = 3 * r, cx = x + pad, cy = y + pad, w = cw - 2 * pad, h = rh - 2 * pad;
        const d = k !== 'unchanged' && e.value != null && u.value != null ? e.value - u.value : null, f = d == null ? 0 : Math.min(1, Math.abs(d) / span);
        g.fillStyle = d == null || Math.abs(d) < 1e-12 ? 'rgba(242,241,238,.05)' : d > 0 ? `rgba(255,195,107,${(0.08 + 0.5 * f).toFixed(3)})` : `rgba(224,108,90,${(0.08 + 0.5 * f).toFixed(3)})`;
        g.fillRect(cx, cy, w, h);
        if (k !== 'unchanged' && e.value != null && e.pass !== u.pass) { g.strokeStyle = INK; g.lineWidth = 1.5 * r; g.strokeRect(cx + 0.75 * r, cy + 0.75 * r, w - 1.5 * r, h - 1.5 * r); }
        if (ivLevel(o, k) === lv) { g.strokeStyle = isCtl(o, k) ? INK2 : AMBER; g.lineWidth = 1.5 * r; g.setLineDash([4 * r, 3 * r]); g.strokeRect(cx - 1.5 * r, cy - 1.5 * r, w + 3 * r, h + 3 * r); g.setLineDash([]); }
        g.textAlign = 'center'; g.font = font(r, 11); g.fillStyle = e.pass ? INK : INK2;
        g.fillText(e.value == null ? 'not measured' : `${e.pass ? '✔' : '·'} ${num(e.value)}`, x + cw / 2, cy + 11 * r);
        if (d != null) {   // the change, the control's change in grey over it: only what sticks out past grey is the change's own
          const ck = isCtl(o, k) ? null : ctlOf(k), ce = ck ? ivVal(o, lv, ck) : null, dc = ce && ce.value != null ? ce.value - u.value : null;
          const bw = w - 10 * r, bx = cx + 5 * r, by = cy + h - 8 * r, len = (v) => Math.min(1, Math.abs(v) / span) * bw;
          g.fillStyle = 'rgba(0,0,0,.45)'; g.fillRect(bx, by, bw, 4 * r);
          g.fillStyle = d > 0 ? AMBER : '#e06c5a'; g.fillRect(bx, by, Math.max(1 * r, len(d)), 4 * r);
          if (dc != null) { g.fillStyle = INK3; g.fillRect(bx, by, Math.min(len(d), len(dc)), 4 * r); if (len(dc) > len(d)) { g.strokeStyle = INK2; g.lineWidth = 1 * r; g.strokeRect(bx, by, len(dc), 4 * r); } }
          const quiet = inNoise(o, lv, d); g.font = font(r, 9); g.fillStyle = quiet ? INK3 : INK2; g.fillText(`${quiet ? '≈ ' : ''}${sgn(d)}`, x + cw / 2, by - 7 * r);   // ≈: inside the replication's spread
        }
        ivCells.push({ x: cx, y: cy, w, h, k, lv });
      });
    });
    g.textAlign = 'left'; g.font = font(r, 9); g.fillStyle = INK3;
    const hasNoise = L.some((lv) => ivNoise(o, lv) != null);
    const legend = ['amber: rose · red: fell · grey in a bar: the control’s change', `dashed: the level changed · white border: verdict flipped${hasNoise ? ' · ≈: within noise' : ''}`];
    if (narrow) legend.forEach((t, i) => g.fillText(t, 4 * r, y0 + L.length * rh + (12 + 13 * i) * r)); else g.fillText(legend.join(' · '), x0, y0 + L.length * rh + 14 * r);
    // which verdicts flipped, for each change and its control
    const say = cols.filter((k) => k !== 'unchanged' && !isCtl(o, k)).map((k) => {
      const c = ctlOf(k), fl = (q) => L.filter((lv) => ivVal(o, lv, q).value != null && ivVal(o, lv, q).pass !== ivVal(o, lv, 'unchanged').pass).map(short).join(', ') || 'no level';
      return `${k}: verdicts flipped at ${fl(k)}${c ? `; its control ${c}: at ${fl(c)}` : '; no control yet'}.`;
    }).join(' ');
    // what did not fit leads: a control that moved as much as its change, a control of the wrong size
    const odd = [];
    cols.filter((k) => k !== 'unchanged' && !isCtl(o, k)).forEach((k) => {
      const c = ctlOf(k), u = (lv) => ivVal(o, lv, 'unchanged').value; if (!c) return;
      const asMuch = L.filter((lv) => { const a = ivVal(o, lv, k).value, b = ivVal(o, lv, c).value; if (a == null || b == null || u(lv) == null) return false; const d = Math.abs(a - u(lv)), dc = Math.abs(b - u(lv)); return d > 1e-9 && dc >= d; });
      const at = ivLevel(o, k);
      if (asMuch.includes(at)) odd.push(`at the level it changed (${short(at)}), ${k}’s control ${c} moved as much as the change itself, so that change is not shown to be specific`);
      else if (asMuch.length) odd.push(`${k}’s control moved as much as the change at ${asMuch.map(short).join(', ')}`);
      if (ivUnmatched(o, c)) odd.push(`${c} is not the same size as ${k} (its variance share is more than 10% off), so it is a weaker baseline`);
      if (isInert(o, c)) odd.push(`${c} is an inert check (the swapped words are in no test text), not a matched control: it shows the measurement is stable, not what a change of the same size does`);
    });
    const noisy = L.filter((lv) => ivNoise(o, lv) != null);
    $('ldIvWhy').textContent = `${odd.length ? `Did not fit: ${odd.join('; ')}. ` : ''}${say || 'Only controls so far.'} ${noisy.length ? `≈ marks a change no bigger than a replication of the unchanged model moved the same number (${noisy.length} of ${L.length} levels have a replication): within noise.` : 'No replication of the unchanged model is stored, so there is no noise floor beyond the control; run one in instrument 7.'} A level that moved no further than under its control moved with the noise of the change itself; one that moved further rests, at least in part, on what was changed. The verdicts use the same untrained twin and the same rules as the stored run.`;
    fillIvPick(o, cols); drawIvTable();
    PSI.canvasLabel($('ldIv'), `Table chart: ${base}, nine levels as released and after ${cols.length - 1} changes. ${say || ''} Pick a change and a level in the lists below to read a cell.`);
  }
  function showIvCell(k, lv) {
    const base = $('ldIvModel').value, o = IV()[base]; if (!o) return; const { ctlFor } = ivPlan(o);
    const e = ivVal(o, lv, k), u = ivVal(o, lv, 'unchanged'), c = ctlFor[k] || null, ce = c ? ivVal(o, lv, c) : null, B = D.rows[o.unchanged_tag] || D.rows[base], be = B && B.ev[lv];
    const box = $('ldIvCell'); box.textContent = ''; box.classList.remove('hidden');
    const h = el('h4'); h.appendChild(el('span', e.pass ? 'ok' : 'no', e.pass ? '✔ ' : '· ')); h.appendChild(document.createTextNode(`${k === 'unchanged' ? base : `${base}@${k}`} · ${full(lv)}`)); box.appendChild(h);
    box.appendChild(el('p', null, `${ivDesc(o, k)}.${ivLevel(o, k) === lv ? ' This is the level that was changed.' : ''}`));
    box.appendChild(el('p', null, D.level_info[lv][1]));
    const d = e.value != null && u.value != null ? e.value - u.value : null, dc = ce && ce.value != null && u.value != null ? ce.value - u.value : null;
    box.appendChild(el('p', null, `Measured: ${num(e.value)}${be ? ` (${be.unit})` : ''}. Unchanged: ${num(u.value)} ${u.pass ? '✔' : '·'}${k === 'unchanged' ? '' : `; change ${sgn(d)}`}.`));
    if (k !== 'unchanged' && !isCtl(o, k)) box.appendChild(el('p', null, c ? `Control (${c}: ${ivDesc(o, c)}): ${num(ce.value)} ${ce.pass ? '✔' : '·'}; change ${sgn(dc)}. Beyond the control: ${d == null || dc == null ? '–' : sgn(Math.abs(d) - Math.abs(dc))}.` : 'No control for this change yet.'));
    // the verdict's own twin, bar and conditions travel with the changed run (intervention_report); the stored run's otherwise
    const ctl = e.control != null ? e.control : be && be.control, need = e.need != null ? e.need : be && be.need;
    box.appendChild(el('p', null, `Untrained twin (${o.control_twin || 'none'}): ${ctl != null ? num(ctl) : 'not stored with this run'}. Bar to clear: ${need != null ? num(need) : 'no bar on this number'}.${pTxt(e) ? ` ${pTxt(e)}.` : ''}${e.rule_text ? ` Rule: ${e.rule_text}.` : ''}`));
    (e.conds || []).forEach(([txt, c2]) => box.appendChild(el('p', c2 ? 'ok' : 'no', `${c2 ? '✔' : '✗'} ${fixP(txt, e)}`)));
    if (undecidable(e)) box.appendChild(el('p', 'ld-odd', `Not decidable: ${e.why_not || 'no reason recorded'}.`));
    if (nearTxt(e)) box.appendChild(el('p', 'ld-odd', nearTxt(e)));
    const ck = isCtl(o, k) ? ivOf(o, k) : c ? ivOf(o, c) : null; if (ck) box.appendChild(el('p', ck.control_kind === 'inert' ? 'ld-odd' : 'muted', `${isCtl(o, k) ? 'This column' : `Its control ${c}`}: ${ctlKind(ck)}.`));
    if (k !== 'unchanged' && e.value != null && e.pass !== u.pass) box.appendChild(el('p', e.pass ? 'ok' : 'no', `The verdict flipped: ${u.pass ? '✔' : '·'} unchanged, ${e.pass ? '✔' : '·'} here.`));
    if (e.value == null) box.appendChild(el('p', 'ld-odd', 'Not measured: this run stored no number for this level, so there is no change to read. It is not a 0.'));
    if (d != null && dc != null && Math.abs(d) > 1e-9 && Math.abs(dc) >= Math.abs(d)) box.appendChild(el('p', 'ld-odd', `The control moved as much as the change (${sgn(dc)} vs ${sgn(d)}): this movement is not shown to come from what was changed.`));
    const cu = isCtl(o, k) ? k : c; if (cu && ivUnmatched(o, cu)) box.appendChild(el('p', 'ld-odd', `${cu} is not the same size as its change: its variance share is more than 10% off the target’s (recorded by model_intervene.py), so it is a weaker baseline.`));
    const nz = ivNoise(o, lv);
    if (k !== 'unchanged') box.appendChild(el('p', null, nz == null ? 'Noise floor: no replication of the unchanged model is stored, so only the control says what an unrelated change does.' : `Noise floor: a replication of the unchanged model (new seed) moved this number by up to ${num(nz)}. ${inNoise(o, lv, d) ? 'This change is within noise.' : d == null ? '' : 'This change is larger.'}`));
  }
  $('ldIv').addEventListener('click', (ev) => {
    const cv = $('ldIv'), b = cv.getBoundingClientRect(), r = cv.width / b.width, x = (ev.clientX - b.left) * r, y = (ev.clientY - b.top) * r;
    const c = ivCells.find((q) => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h); if (!c) return;
    showIvCell(c.k, c.lv);
    if (c.k !== 'unchanged') { $('ldIvPick').value = c.k; $('ldIvLevel').value = short(c.lv); drawIvTable(); }
  });
  function fillIvPick(o, cols) {
    const keep = $('ldIvPick').value, s = $('ldIvPick'); s.textContent = '';
    cols.filter((k) => k !== 'unchanged').forEach((k) => opt(s, k, `${k} · ${ivDesc(o, k)}`));
    s.value = keep && keep !== 'unchanged' && cols.includes(keep) ? keep : (cols[1] || '');
    const sl = $('ldIvLevel'); if (!sl.options.length) { opt(sl, '', 'every level'); LEV().forEach((l) => opt(sl, short(l), l)); }
  }
  function drawIvTable() {
    const base = $('ldIvModel').value, o = IV()[base], tb = $('ldIvTable').querySelector('tbody'); tb.textContent = ''; if (!o) return;
    const k = $('ldIvPick').value, c = isCtl(o, k) ? null : (ivPlan(o).ctlFor[k] || null), lvF = $('ldIvLevel').value, det = o.detail || {};
    const rowsT = Object.keys(det).filter((n) => !lvF || n.split(' ')[0] === lvF).map((n) => {
      const u = det[n].unchanged, v = det[n][k], cv = c ? det[n][c] : null;
      const d = u != null && v != null ? v - u : null, dc = u != null && cv != null ? cv - u : null;
      return { n, u, v, d, dc, s: d == null ? -Infinity : Math.abs(d) - (dc == null ? 0 : Math.abs(dc)) };
    }).sort((a, b) => b.s - a.s);
    rowsT.forEach((q) => {
      const tr = el('tr'), sp = q.n.indexOf(' ');
      tr.appendChild(el('td', null, q.n.slice(0, sp))); tr.appendChild(el('td', null, q.n.slice(sp + 1)));
      const ns = (x) => (x == null || !isFinite(x) ? 'not stored' : num(x));
      tr.appendChild(el('td', q.u == null ? 'flat' : null, ns(q.u))); tr.appendChild(el('td', q.v == null ? 'flat' : null, ns(q.v)));
      tr.appendChild(el('td', q.d == null || Math.abs(q.d) < 1e-9 ? 'flat' : q.d > 0 ? 'up' : 'down', sgn(q.d)));
      tr.appendChild(el('td', 'flat', c ? sgn(q.dc) : '–'));
      tr.appendChild(el('td', !isFinite(q.s) || q.s <= 0 ? 'flat' : 'up', isFinite(q.s) ? sgn(q.s) : '–'));
      tb.appendChild(tr);
    });
    if (!rowsT.length) { const tr = el('tr'); const td = el('td', 'flat', 'no finer numbers stored for this level'); td.colSpan = 7; tr.appendChild(td); tb.appendChild(tr); }
    $('ldIvTableWhy').textContent = `${k}: ${ivDesc(o, k)}. ${c ? `Control: ${c}.` : isCtl(o, k) ? 'This is itself a control.' : 'No control yet, so the last column is the whole change.'} Sorted by how much further each number moved than under the control (size of the change minus size of the control’s change). M1 to M3 per language, M5 per context (nats), M8 said and held. Click a cell in the grid to jump here.`;
  }
  $('ldIvModel').addEventListener('change', () => { $('ldIvCell').classList.add('hidden'); drawIv(); });
  // keyboard alternative to clicking a cell: the change and the level lists open that cell too
  const ivPicked = () => { drawIvTable(); const k = $('ldIvPick').value, sl = $('ldIvLevel').value, lv = sl && LEV().find((l) => short(l) === sl); if (k && lv) showIvCell(k, lv); };
  $('ldIvPick').addEventListener('change', ivPicked); $('ldIvLevel').addEventListener('change', ivPicked);

  // ── wiring ─────────────────────────────────────────────────────────────
  function fillSelects() {
    const keepM = $('ldModel').value; $('ldModel').textContent = '';
    released().forEach((t) => opt($('ldModel'), t, t));
    const g2 = document.createElement('optgroup'); g2.label = 'untrained twins and checkpoints'; Object.keys(D.rows).filter((t) => !released().includes(t)).forEach((t) => { const o = document.createElement('option'); o.value = t; o.textContent = t; g2.appendChild(o); }); $('ldModel').appendChild(g2);
    if (Object.keys(D.replications || {}).length) { const g3 = document.createElement('optgroup'); g3.label = 'replications'; Object.keys(D.replications).forEach((t) => { const o = document.createElement('option'); o.value = t; o.textContent = t; g3.appendChild(o); }); $('ldModel').appendChild(g3); }
    $('ldModel').value = keepM && row(keepM) ? keepM : (released().find((t) => t.startsWith('mistral')) || released()[0]);
    const keepR = $('ldRun').value; $('ldRun').textContent = ''; Object.keys(runs()).sort().forEach((f) => opt($('ldRun'), f, `${f} (${runs()[f].length} checkpoints)`)); $('ldRun').value = keepR && runs()[keepR] ? keepR : (runs()['pythia-1b'] ? 'pythia-1b' : Object.keys(runs())[0]);
    if (!$('ldRunLevel').options.length) { LEV().forEach((l) => opt($('ldRunLevel'), l, l)); $('ldRunLevel').value = 'M2 features'; }
    const keepA = $('ldCtxA').value, keepB = $('ldCtxB').value; $('ldCtxA').textContent = ''; $('ldCtxB').textContent = ''; opt($('ldCtxB'), '', 'compare with…');
    released().forEach((t) => { opt($('ldCtxA'), t, t); opt($('ldCtxB'), t, t); });
    $('ldCtxA').value = keepA || (released().find((t) => t.startsWith('mistral')) || released()[0]); $('ldCtxB').value = keepB || (released().find((t) => t.startsWith('qwen3-4b')) || '');
    const keepS = $('ldSayModel').value; $('ldSayModel').textContent = ''; released().concat(Object.keys(D.rows).filter((t) => D.rows[t].untrained)).forEach((t) => opt($('ldSayModel'), t, t));
    $('ldSayModel').value = keepS || (released().find((t) => t.startsWith('mistral')) || released()[0]);
    const keepI = $('ldIvModel').value; $('ldIvModel').textContent = ''; Object.keys(IV()).sort().forEach((t) => opt($('ldIvModel'), t, `${t} (${(IV()[t].columns || []).length - 1} changed)`));
    $('ldIvModel').value = keepI && IV()[keepI] ? keepI : (Object.keys(IV()).sort()[0] || '');
  }
  // ── 9 · level 2: per-layer detail beside the untrained twin (research/model_deep.py, deep_report.py) ──
  const DEEP = () => (D && D.deep) || {};
  const PROF = [
    ['weight_alpha', 'weight heavy-tail α', 'lower = heavier tails; training makes them heavy'],
    ['stable_rank_min', 'lowest stable rank', 'a matrix that acts in few directions'],
    ['massive_ratio_max', 'massive activation ×', 'one dimension far larger than the rest'],
    ['effective_dim_mid', 'effective dimension', 'how many directions the sentences spread over (middle layers)'],
    ['anisotropy_mid', 'anisotropy', 'how alike all sentences look (1 = one direction)'],
    ['sense_acc', 'senses told apart', 'share of sentences put with their own sense (chance 0.071)'],
    ['facts_first_top1_layer', 'facts lead from layer', 'first layer where the answer is the top guess for half the facts'],
    ['induction_heads', 'induction heads', 'heads that copy what followed a token last time'],
    ['previous_token_heads', 'previous-token heads', ''], ['sink_heads', 'sink heads', 'heads that park attention on the first token'],
    ['es_tokens_shared', 'Spanish tokens shared with English', 'a random network can match these'],
    ['routing_sense_layers', 'layers routing by sense', 'MoE: layers where the expert tells the sense (p < 0.01)']];
  const dnum = (x) => (Number.isInteger(x) ? String(x) : val(x));
  const deepFam = (t) => Object.values(DEEP().families || {}).find((f) => f.final === t || f.twin === t || (f.training || []).includes(t)) || {};
  function drawDeep() {
    const dd = DEEP(), twins = new Set(Object.values(dd.families || {}).map((f) => f.twin)), tags = Object.keys(dd.profiles || {}).filter((t) => !twins.has(t));
    const empty = !tags.length; $('ldDeepEmpty').classList.toggle('hidden', !empty); $('ldDeepTable').classList.toggle('hidden', empty);
    if (empty) { hint($('ldDeep'), 200, 'Nothing yet: no layer-by-layer measurements.'); $('ldDeepWhy').textContent = ''; $('ldDeepMeta').textContent = ''; return; }
    const ms = $('ldDeepModel'), cs = $('ldDeepCurve');
    if (ms.options.length !== tags.length) { const keep = ms.value; ms.textContent = ''; tags.sort().forEach((t) => opt(ms, t, t)); if (tags.includes(keep)) ms.value = keep; }
    if (!cs.options.length) Object.entries(dd.curve_info || {}).forEach(([k, v]) => opt(cs, k, v));
    const t = ms.value, k = cs.value, f = deepFam(t), C = dd.curves || {}, info = (dd.curve_info || {})[k] || k;
    const steps = (f.training || []).filter((x) => x !== t && C[x] && C[x][k]);
    const lines = [...steps.map((x, i) => ({ v: C[x][k], c: `rgba(255,195,107,${0.12 + 0.3 * (i + 1) / (steps.length + 1)})`, w: 1 })),
      ...(f.twin && C[f.twin] && C[f.twin][k] ? [{ v: C[f.twin][k], c: INK2, w: 1.5, dash: true }] : []),
      ...(C[t] && C[t][k] ? [{ v: C[t][k], c: AMBER, w: 2.5 }] : [])];
    const all = lines.flatMap((l) => l.v).filter((x) => x != null && isFinite(x));
    if (!all.length) hint($('ldDeep'), 300, `${t}: this measurement was not taken for this model (${info}).`);
    else {
      const { g, r, W, H } = canvas($('ldDeep'), 300);
      let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
      const L = 46 * r, R = 14 * r, T = 14 * r, B = 30 * r, X = (q) => L + q * (W - L - R), Y = (y) => H - B - (y - lo) / (hi - lo) * (H - T - B);
      g.strokeStyle = RULE; g.lineWidth = 1 * r; g.beginPath(); g.moveTo(L, T); g.lineTo(L, H - B); g.lineTo(W - R, H - B); g.stroke();
      g.fillStyle = INK3; g.font = font(r, 10); g.textAlign = 'right'; g.textBaseline = 'middle';
      [lo, (lo + hi) / 2, hi].forEach((y) => g.fillText(fmt(y), L - 6 * r, Y(y)));
      g.textBaseline = 'top'; g.textAlign = 'left'; g.fillText('input', X(0), H - B + 8 * r); g.textAlign = 'right'; g.fillText('last layer', X(1), H - B + 8 * r);
      lines.forEach((l) => {
        g.strokeStyle = l.c; g.lineWidth = l.w * r; g.setLineDash(l.dash ? [5 * r, 4 * r] : []); g.beginPath(); let pen = false;
        l.v.forEach((y, i) => { if (y == null || !isFinite(y)) { pen = false; return; } const x = X(i / Math.max(1, l.v.length - 1)); if (pen) g.lineTo(x, Y(y)); else g.moveTo(x, Y(y)); pen = true; });
        g.stroke();
      });
      g.setLineDash([]);
      PSI.canvasLabel($('ldDeep'), `Chart: ${info}, layer by layer, ${t} against its untrained twin.`);
    }
    const P = dd.profiles[t] || {}, Q = f.twin ? dd.profiles[f.twin] || {} : {}, tb = $('ldDeepTable').tBodies[0]; tb.textContent = '';
    PROF.forEach(([key, label, why]) => {
      if (P[key] === undefined && Q[key] === undefined) return;
      const tr = el('tr'); tr.append(el('td', null, label), el('td', null, dnum(P[key])), el('td', null, f.twin ? dnum(Q[key]) : 'no twin measured'), el('td', 'muted', why)); tb.appendChild(tr);
    });
    $('ldDeepMeta').textContent = `${P.layers || '?'} layers · ${P.precision || ''}${f.twin ? ` · twin ${f.twin}` : ' · no untrained twin measured, so no baseline'}${steps.length ? ` · ${steps.length} training checkpoints` : ''}`;
    $('ldDeepWhy').textContent = `${info}. Amber: ${t}; grey dashed: its untrained twin${steps.length ? '; faint: earlier training checkpoints' : ''}.`;
  }
  $('ldDeepModel').addEventListener('change', drawDeep); $('ldDeepCurve').addEventListener('change', drawDeep);

  function drawAll() {
    if (!D) { showEmpty(true); [['ldClimb', 470], ['ldTrain', 300], ['ldTrainLine', 200], ['ldCtx', 420], ['ldSay', 380], ['ldAgree', 640]].forEach(([id, h]) => hint($(id), h, 'Nothing yet: no stored ladder results.')); drawIv(); drawDeep(); return; }
    showEmpty(false); drawMatrix(); drawClimb(); drawTrain(); drawCtx(); drawSay(); drawAgree(); drawReps(); drawIv(); drawDeep();
  }
  async function refresh() {
    if (!desktop || !window.psi || !window.psi.ladder) return;
    try {
      const fresh = await window.psi.ladder({ action: 'data' });
      if (fresh && fresh.rows) { D = fresh; Object.keys(RANGE).forEach((k) => delete RANGE[k]); fillSelects(); fillRunControls(); drawAll(); if (fresh.written) say($('ldStat'), `The view was older than the results on disk: rebuilt from ${fresh.n_results} result files and saved (src/ui/ladder-data.js).`, 'ok'); }
    } catch (e) { say($('ldStat'), `Showing the results bundled with the app: the stored results could not be read (${PSI.explain(e).text}).`, 'warn'); }
  }
  // Is this view older than the result files on disk? (src/ladder.js status: data_generated, newest_result_mtime, stale, counts)
  function showStale(s) {
    const box = $('ldStale'); if (!box) return; box.textContent = '';
    if (!s || !s.stale) { box.classList.add('hidden'); return; }
    const k = s.n_newer != null ? s.n_newer : Math.max(0, (s.n_results_on_disk || 0) - (s.n_results_in_view || 0));
    const when = s.newest_result_mtime ? new Date(s.newest_result_mtime * 1000).toLocaleString() : 'an unknown time';
    box.appendChild(el('p', null, `Results on disk are newer than this view (${k} new): rebuild. This view was built ${s.data_generated || 'at an unrecorded time'}; the newest result file is from ${when}; ${s.n_results_on_disk} result files on disk, ${s.n_results_in_view == null ? 'the view did not count its own (older data file)' : `${s.n_results_in_view} in the view`}.`));
    const r = el('div', 'row');
    if (desktop) { const b = el('button', 'ghost', 'Rebuild the view'); b.addEventListener('click', async () => { b.disabled = true; await refresh(); b.disabled = false; checkStale(); }); r.appendChild(b); }
    r.appendChild(el('code', null, 'python research/make_ladder_data.py')); box.appendChild(r);
    box.appendChild(el('p', 'fx', 'Rebuilding reads every result file and rewrites src/ui/ladder-data.js; it takes a few seconds and uses no GPU.'));
    box.classList.remove('hidden');
  }
  async function checkStale() {
    if (!window.psi || !window.psi.ladder) return null;
    try { const s = await window.psi.ladder({ action: 'status' }); showStale(s); return s; } catch (_) { return null; }   // the run status reports its own errors
  }
  if (D) { fillSelects(); fillRunControls(); }
  updateCmd(); gateGo();
  if (!desktop) { PSI.block($('ldGo'), 'desktop', 'Runs happen in the desktop app; here the stored results are shown.'); PSI.block($('ldPick'), 'desktop', 'Runs happen in the desktop app.'); }
  let opened = false;
  document.querySelector('[data-tab=ladder]').addEventListener('click', () => setTimeout(() => {
    drawAll();
    if (!opened) { opened = true; refresh().then(checkStale); if (window.psi && window.psi.ladder) window.psi.ladder({ action: 'status' }).then((s) => { if (s && s.running) { setRunning(true); $('ldLog').textContent = s.lines.join('\n') + '\n'; if (prog) prog.step(0, `now: ${s.step || 'running'}`); } }).catch((e) => say($('ldRunStat'), `Could not ask whether a ladder run is already going: ${PSI.explain(e).text}`, 'warn')); }
  }, 0));
  addEventListener('resize', () => { if (!$('tab-ladder').classList.contains('hidden')) drawAll(); });
})();
