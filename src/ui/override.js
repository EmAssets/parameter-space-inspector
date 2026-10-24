// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Override map tab: the verdict first (the measured risk that the answer is unsupported, read from the model's state
// before it wrote a word), then the answer, then the evidence behind the verdict, each folded: how reliable the probe
// is, where this answer sits among the tested ones, the tested questions nearest to it inside the model, every tested
// answer (each can be re-run live), and the layer-by-layer reading of the first word. With a claim: pushback.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const pct = (x) => `${Math.round(x * 100)}%`;
  // A measured chance, as people read it: never "100%" or "0%" for an estimate.
  const chance = (x) => (x > 0.99 ? '>99%' : x < 0.01 ? '<1%' : pct(x));
  const show = (s) => JSON.stringify(s).slice(1, -1);
  const often = (x) => (x >= 0.5 ? `about ${Math.round(x * 10)} in 10` : x > 0 ? `about 1 in ${Math.round(1 / x)}` : 'none');
  const EXAMPLES = [
    ['made-up film', 'Who was the director of The Glass Orchard of Pellumbra?', ''],
    ['future event', 'Who won the 2030 FIFA World Cup final?', ''],
    ['known fact', 'Who won the 2018 FIFA World Cup?', ''],
    ['pushback', 'What is the chemical symbol for gold?', 'Ag'],
  ];
  const KIND = { popqa: 'real fact', past: 'past event', fiction: 'invented subject', future: 'future event' };
  const GRADE = { correct: ['supported', 'ok'], wrong: ['wrong', 'bad'], invented: ['answered what it cannot know', 'bad'], hedge: ['hedged: not scored', 'hedge'] };
  let LAST = null, SUMMARY = null, SUMMARY_KEY = '', RERUN = null, LIST_AT = 0;
  const GO = $('omGo');
  PSI.requires(GO, 'engine');

  EXAMPLES.forEach(([label, q, claim]) => {
    const b = el('button', 'ghost small', label); b.type = 'button';
    b.addEventListener('click', () => { $('omQ').value = q; $('omClaim').value = claim; read(); });
    $('omEx').appendChild(b);
  });

  // The layer cells. Answer mode: amber where the layer's top reading is the piece the model wrote. Pushback: green
  // where its own answer leads the claim, amber where the claim leads. Faded: the lens fits poorly.
  function strip(host, legend, rows, mode, writtenPiece, onPick) {
    host.textContent = '';
    rows.forEach((r, i) => {
      const b = el('button', 'om-cell'); b.type = 'button';
      let kind = 'neither';
      if (mode === 'pushback') kind = r.logodds > 0 ? 'own' : 'claim';
      else if (r.top[0] && r.top[0][0].trim() && r.top[0][0].trim() === (writtenPiece || '').trim()) kind = 'written';
      if (kind !== 'neither') b.classList.add(kind);
      if (r.poor_fit) b.classList.add('poor');
      const lab = mode === 'pushback' ? (kind === 'own' ? 'its own answer leads' : 'your claim leads') : (kind === 'written' ? 'would already write its first piece' : 'would write something else');
      b.setAttribute('aria-label', `Layer ${r.layer}: ${lab}${r.poor_fit ? ', lens fits poorly' : ''}`);
      if (r.layer === 1 || r.layer % 6 === 0 || r.layer === rows.length) b.appendChild(el('span', '', String(r.layer)));
      if (onPick) b.addEventListener('click', () => onPick(i));
      host.appendChild(b);
    });
    if (!legend) return;
    legend.textContent = '';
    const items = mode === 'pushback' ? [['#8fd18f', 'its own answer leads'], ['var(--accent)', 'your claim leads']] : [['var(--accent)', 'would already write its first piece'], ['rgba(138,141,151,.5)', 'would write something else']];
    items.concat([['rgba(138,141,151,.2)', 'faded: the lens reads poorly at that layer']]).forEach(([c, t]) => { const s = el('span'); const i = el('i'); i.style.background = c; s.append(i, t); legend.appendChild(s); });
  }

  function pick(i) {
    const rows = rowsOf(LAST); const r = rows[i]; if (!r) return;
    [...$('omStrip').children].forEach((c, j) => c.classList.toggle('sel', j === i));
    const top = r.top.map(([p, v]) => `“${show(p)}” ${pct(v)}`).join(' · ');
    const lines = [`Layer ${r.layer} of ${rows.length}: its top guesses for the first piece: ${top}`,
      `the piece it finally wrote: ${pct(r.p_written)} at this layer${r.poor_fit ? ' · the lens reads poorly this early, so read with care' : ''}`];
    if (r.logodds != null) lines.push(`its own answer vs your claim: ${r.logodds > 0 ? '+' : ''}${r.logodds.toFixed(2)} log-odds`);
    $('omRead').textContent = lines.join('\n'); $('omRead').style.whiteSpace = 'pre-wrap';
  }
  const rowsOf = (r) => (r.mode === 'pushback' ? r.push.layers : r.layers);

  // ── the verdict ───────────────────────────────────────────────────────────
  function verdict(rk, r) {
    const host = $('omRisk'); host.textContent = ''; host.className = 'om-verdict';
    if (!rk || !rk.available) {
      host.appendChild(el('p', 'om-head', 'No risk score for this answer'));
      host.appendChild(el('p', 'muted small', rk ? `${rk.why[0].toUpperCase()}${rk.why.slice(1)}.` : 'The engine returned no risk.'));
      if (rk && /no probe has been measured/.test(rk.why) && PSI.overrideMeasure) PSI.overrideMeasure.offer(host);
      return;
    }
    const band = rk.flag ? 'high' : rk.probability >= 0.3 ? 'mid' : 'low';
    host.classList.add(band);
    const words = {
      low: 'Low risk: inside, the model looks the way it did on answers it got right.',
      mid: 'Unsure: inside, the model sits between its right and its unsupported answers. Check this one against a source.',
      high: 'High risk: inside, the model looks the way it did on answers that were wrong or made up.',
    }[band];
    const head = el('p', 'om-head'); head.append(el('span', '', words), el('b', '', chance(rk.probability))); host.appendChild(head);
    const sc = el('div', 'om-scale'); sc.setAttribute('role', 'img'); sc.setAttribute('aria-label', `${chance(rk.probability)} chance unsupported; the probe flags from ${pct(rk.flag_probability || 0.5)}`);
    const fl = el('span', 'fl'); fl.style.left = pct(rk.flag_probability || 0.5); const at = el('span', 'at'); at.style.left = pct(Math.min(0.995, rk.probability));
    sc.append(fl, at); host.appendChild(sc);
    const k = el('div', 'om-scale-k'); k.append(el('span', '', 'right'), el('span', '', `flags from ${pct(rk.flag_probability || 0.5)}`), el('span', '', 'unsupported')); host.appendChild(k);
    host.appendChild(el('p', 'muted small', `Read from the model’s own state at layer ${rk.layer}, at the moment it started answering, before it wrote a word: the answer’s text plays no part. It does not check facts; it recognises the state this model was in when its answers were wrong or made up. Measured on this exact model${rk.measured_by ? ` by ${rk.measured_by}` : ''}. Trust: ${rk.trust}.`));
    const nr = rk.near;
    if (nr) {
      const said = nr.near_scored ? ` On the ${nr.near_scored} most similar tested questions it scored, the probe was right on ${nr.near_right}.` : '';
      const t = {
        typical: `Your question is like the tested ones: inside the model, it sits among them.${said}`,
        edge: `Your question is at the edge of what was tested: inside the model, the nearest tested questions are not very close. Treat the score with some care.${said}`,
        outside: `Your question is unlike anything this was tested on: inside the model, it sits away from every tested question. Treat the score as a rough guide.${said}`,
      }[nr.fit];
      const p = el('p', `om-fit ${nr.fit}`, t); host.appendChild(p);
    }
  }

  // ── evidence: summary (metrics, controls, every scored test answer), cached per probe ──
  const ev = (a) => psi.override({ evidence: a });
  async function summary(rk) {
    const key = JSON.stringify(rk.measured || {});
    if (SUMMARY && SUMMARY_KEY === key) return SUMMARY;
    const s = await ev({ action: 'summary' });
    if (s.available) { SUMMARY = s; SUMMARY_KEY = key; }
    return s;
  }

  function reliability(rk, s) {
    const host = $('omRel'); host.textContent = '';
    const m = rk.measured || {};
    const P = (t) => host.appendChild(el('p', 'small', t));
    P(`Tested once on ${m.n} answers to questions the probe never saw while it was being made. Of the answers that were wrong or made up, it flagged ${pct(m.tpr)} (${often(m.tpr)}). Of the answers that were right, it wrongly flagged ${pct(m.fpr)} (${often(m.fpr)}).`);
    P(`Given one right and one unsupported answer, it gives the unsupported one the higher risk ${pct(m.auroc)} of the time (AUROC ${m.auroc}; 95% interval ${(m.ci || []).join('–')}; 50% would be guessing). Within a single kind of real-fact question, so that it gets no credit for knowing which kinds the model fails at: ${pct(m.within_type)}.`);
    P(`Its chances are calibrated: answers it gave about 30% were unsupported about 30% of the time (average error ${pct(m.ece)}).`);
    if (s && s.available && s.controls) {
      const c = s.controls;
      host.appendChild(el('p', 'k mt-12', 'controls: could something simpler do as well?'));
      const ul = el('ul', 'small');
      [[`The same probe trained on shuffled grades`, `${pct(c.shuffled_grades_dev)}: about chance, so it is not picking up something trivial`],
        [`The question’s words alone, without looking inside`, `${pct(c.question_text_only)} overall, ${pct(c.question_text_only_within_popqa)} on real facts, against the probe’s ${pct(c.probe_within_popqa)}. Much of the easy part is the kind of question; the probe’s edge is inside each kind`],
        [`How famous the subject is, alone`, `${pct(c.fame_only_within_popqa)} on real facts`],
        [`The probe trained on real facts only, tested on invented and future subjects it never saw`, `${pct(c.trained_on_real_facts_only_scored_on_the_other_kinds)}`]].forEach(([a, b]) => { const li = el('li'); li.append(el('b', '', `${a}: `), b); ul.appendChild(li); });
      host.appendChild(ul);
      host.appendChild(el('p', 'k mt-12', 'what was tested, and how it was graded'));
      const t = el('table', 'om-tbl'); const hr = el('tr'); ['kind', 'supported', 'wrong', 'answered what it cannot know', 'hedged'].forEach((h) => hr.appendChild(el('th', '', h))); t.appendChild(hr);
      Object.entries(s.counts).forEach(([kd, g]) => { const tr = el('tr'); tr.append(el('td', '', KIND[kd] || kd), ...['correct', 'wrong', 'invented', 'hedge'].map((x) => el('td', '', String(g[x] || 0)))); t.appendChild(tr); });
      host.appendChild(t);
      const gl = el('ul', 'small muted'); Object.entries(s.grading).forEach(([k, v]) => { const li = el('li'); li.append(el('b', '', `${k === 'popqa' ? 'real facts' : KIND[k] || k}: `), v); gl.appendChild(li); }); host.appendChild(gl);
    }
    if (rk.bank) host.appendChild(el('p', 'muted small', `The questions: ${rk.bank}. Real facts come from PopQA (Mallen et al., 2023), short questions about Wikipedia subjects at every level of fame.`));
    if (rk.trust_why) host.appendChild(el('p', 'muted small', `Trust ${rk.trust}: ${rk.trust_why}.`));
    host.appendChild(el('p', 'muted small', 'It reads a sign, not the cause: removing that direction from the model hides the risk without fixing the answer, so a steered model’s score means nothing.'));
  }

  // ── where this answer sits: every scored test answer by its chance, supported above, unsupported below ──
  let PLACED = [];
  function place(rk, s) {
    const cv = $('omPlace'); const W = cv.clientWidth || 700, H = 150, dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.height = `${H}px`;
    const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement), ink3 = css.getPropertyValue('--ink3').trim() || '#8a8d97';
    const x = (p) => 12 + p * (W - 24), rows = { up: [22, 62], down: [84, 124] };
    g.fillStyle = ink3; g.font = '10.5px ui-monospace, monospace';
    g.fillText('graded supported', 12, 14); g.fillText('graded wrong or made up', 12, 146);
    PLACED = [];
    (s.points || []).forEach(([p, grade, , , n], j) => {
      const bad = grade !== 'correct'; const [y0, y1] = bad ? rows.down : rows.up;
      const y = y0 + ((j * 2654435761) % 1000) / 1000 * (y1 - y0);
      g.fillStyle = bad ? 'rgba(232,110,110,.55)' : 'rgba(143,209,143,.55)';
      g.beginPath(); g.arc(x(p), y, 2.2, 0, 7); g.fill(); PLACED.push([x(p), y, n]);
    });
    g.setLineDash([4, 3]); g.strokeStyle = ink3; g.beginPath(); g.moveTo(x(rk.flag_probability || 0.5), 18); g.lineTo(x(rk.flag_probability || 0.5), 128); g.stroke(); g.setLineDash([]);
    g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.moveTo(x(rk.probability), 16); g.lineTo(x(rk.probability), 130); g.stroke(); g.lineWidth = 1;
    // The tested answers scored most like this one (the nearest 100 by chance): how many of them were in fact unsupported.
    const pts = s.points || [], like = [...pts].sort((u, v) => Math.abs(u[0] - rk.probability) - Math.abs(v[0] - rk.probability)).slice(0, 100);
    const lo = Math.min(...like.map((q) => q[0])), hi = Math.max(...like.map((q) => q[0])), bad = like.filter((q) => q[1] !== 'correct').length;
    $('omPlaceWhy').textContent = like.length ? `${pts.length} tested answers. The ${like.length} scored most like this one (${pct(lo)}–${pct(hi)}): ${bad} of them (${pct(bad / like.length)}) were in fact wrong or made up, ${like.length - bad} were right.` : '';
  }
  $('omPlace').addEventListener('click', (e) => {
    const r = e.currentTarget.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = null, bd = 64; PLACED.forEach(([x, y, n]) => { const d = (x - px) ** 2 + (y - py) ** 2; if (d < bd) { bd = d; best = n; } });
    if (best != null) item(best);
  });

  // ── tested answers as rows (nearest, browse) ──
  function table(host, rows, withSim) {
    host.textContent = '';
    const t = el('table', 'om-tbl'); const hr = el('tr');
    (withSim ? ['alike', 'tested question', 'its answer', 'grade', 'probe said'] : ['tested question', 'its answer', 'grade', 'probe said']).forEach((h, i) => { const th = el('th', '', h); if (h === 'its answer') th.className = 'hide-n'; hr.appendChild(th); });
    t.appendChild(hr);
    rows.forEach((x) => {
      const tr = el('tr', 'pick'); tr.tabIndex = 0;
      if (withSim) tr.appendChild(el('td', '', x.similarity.toFixed(2)));
      const q = el('td'); q.append(x.q + (x.wording === 'brief' ? ' (one sentence)' : ''), el('br'), el('span', 'muted small', KIND[x.kind] || x.kind));
      const [gt, gc] = GRADE[x.grade] || [x.grade, '']; const gd = el('td'); gd.appendChild(el('span', `om-g ${gc}`, gt));
      const said = el('td', 'nowrap', x.scored ? chance(x.probability) : '—'); if (x.scored && x.flagged) { said.append(el('br'), el('span', 'muted small', 'flagged')); }
      tr.append(q, el('td', 'ans hide-n', x.answer.length > 140 ? `${x.answer.slice(0, 140)}…` : x.answer), gd, said);
      const open = () => item(x.n); tr.addEventListener('click', open); tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
      t.appendChild(tr);
    });
    host.appendChild(t);
  }

  function nearest(rk) {
    const nr = rk.near; if (!nr) { $('omNearWhy').textContent = 'Not available for this reading.'; $('omNear').textContent = ''; return; }
    $('omNearWhy').textContent = `The tested questions whose state inside the model is most like this one’s: ${nr.how}. “Alike” is from −1 to 1 (1 is the same direction). Each tested question’s outside grade and what the probe said about it are beside it, so you can see how the probe did on questions like yours.`;
    table($('omNear'), nr.nearest, true);
  }

  async function browse(more) {
    if (!more) LIST_AT = 0;
    const fl = $('omFFlag').value;
    try {
      const r = await ev({ action: 'list', kind: $('omFKind').value, grade: $('omFGrade').value, flagged: fl === '' ? undefined : fl === '1', text: $('omFText').value, offset: LIST_AT, limit: 50 });
      if (!r.available) { $('omAllStat').textContent = r.why; return; }
      if (!more) $('omAll').textContent = '';
      const host = el('div'); table(host, r.rows, false); $('omAll').appendChild(host);
      LIST_AT += r.rows.length;
      $('omAllStat').textContent = `${r.total} tested answers match; showing ${LIST_AT}.`;
      $('omAllMore').hidden = LIST_AT >= r.total;
    } catch (e) { $('omAllStat').textContent = `failed: ${PSI.explain(e).text}`; }
  }
  ['omFKind', 'omFGrade', 'omFFlag'].forEach((id) => $(id).addEventListener('change', () => browse(false)));
  let ft = null; $('omFText').addEventListener('input', () => { clearTimeout(ft); ft = setTimeout(() => browse(false), 300); });
  $('omAllMore').addEventListener('click', () => browse(true));
  $('omAllD').addEventListener('toggle', () => { if ($('omAllD').open && !$('omAll').childElementCount) browse(false); });

  // ── one tested answer in full ──
  async function item(n) {
    const host = $('omItem'); host.classList.remove('hidden'); host.textContent = 'opening…';
    let r; try { r = await ev({ action: 'item', n }); } catch (e) { host.textContent = `failed: ${PSI.explain(e).text}`; return; }
    if (!r.available) { host.textContent = r.why; return; }
    const it = r.item; host.textContent = '';
    const top = el('div', 'row'); top.append(el('span', 'k m-0', `tested answer · ${KIND[it.kind] || it.kind}${it.topic && it.kind === 'popqa' ? ` (${it.topic})` : ''}`));
    const close = el('button', 'ghost small', 'Close'); close.type = 'button'; close.style.marginLeft = 'auto'; close.addEventListener('click', () => host.classList.add('hidden')); top.appendChild(close);
    host.appendChild(top);
    host.appendChild(el('p', 'k', 'asked exactly'));
    host.appendChild(el('p', 'om-q', it.asked));
    host.appendChild(el('p', 'k', 'its answer'));
    host.appendChild(el('p', 'om-answer', it.answer));
    const [gt, gc] = GRADE[it.grade] || [it.grade, ''];
    const g = el('p', 'small'); g.append(el('span', `om-g ${gc}`, gt), ` ${it.expected && it.expected.length ? `accepted answers: ${it.expected.join(', ')}` : (SUMMARY && SUMMARY.grading[it.kind]) || ''}`);
    host.appendChild(g);
    host.appendChild(el('p', 'small', it.scored ? `The probe said ${chance(it.probability)} chance unsupported${it.flagged ? ', and flagged it' : ', not flagged'}: ${it.flagged === (it.grade !== 'correct') ? 'right' : 'wrong'} on this one.` : 'Not scored: the answer hedges, and the probe was measured on answers only.'));
    host.appendChild(el('p', 'k', 'when it settled on its first word'));
    const sw = el('div', 'om-strip-wrap'); const s = el('div', 'om-strip'); sw.appendChild(s); host.appendChild(sw);
    const rd = el('div', 'om-read'); host.appendChild(rd);
    const at = (i) => { const x = it.layers[i]; [...s.children].forEach((c, j) => c.classList.toggle('sel', j === i)); rd.textContent = `Layer ${x.layer}: ${x.top.map(([p, v]) => `“${show(p)}” ${pct(v)}`).join(' · ')}; the piece it wrote: ${pct(x.p_written)}`; };
    strip(s, null, it.layers, 'answer', it.first_piece, at);
    const fi = it.layers.findIndex((x) => x.top[0] && x.top[0][0].trim() === (it.first_piece || '').trim()); at(fi >= 0 ? fi : Math.floor(it.layers.length * 0.6));
    const btns = el('div', 'row');
    const rerun = el('button', '', 'Re-run it now'); rerun.type = 'button';
    rerun.addEventListener('click', () => { RERUN = { asked: it.asked, probability: it.probability, answer: it.answer, scored: it.scored }; $('omQ').value = it.asked; $('omClaim').value = ''; host.classList.add('hidden'); read(); });
    const forks = el('button', 'ghost', 'See where its answer could fork (Generate)'); forks.type = 'button';
    forks.addEventListener('click', () => { const p = $('prompt'); if (p) { p.value = it.asked; p.dispatchEvent(new Event('input')); } PSI.goto('generate', 'genGo'); });
    btns.append(rerun, forks); host.appendChild(btns);
    host.appendChild(el('p', 'muted small', 'Re-run asks the loaded model the same words on the inside engine and compares its new reading with the stored one; small differences come from the stored state being kept at half precision and from the GPU. Generate shows every step’s alternatives on the outside engine’s model with its own settings, which may be a different copy of the weights (a GGUF): its forks are a guide, not this measurement.'));
    host.scrollIntoView({ block: 'nearest' });
  }

  // A reading on screen: from a run just now, or kept from before (a reload, another device; PSI.keep).
  function render(r, info = {}) {
    if (!r || !(r.layers || (r.push && r.push.layers))) return;
    LAST = r; $('omOut').classList.remove('hidden'); $('omItem').classList.add('hidden');
    $('omQShow').textContent = r.claim ? `${r.prompt}  ·  your claim: ${r.claim}` : r.prompt;
    const f = $('omFlag'); f.className = 'om-flag'; f.textContent = '';
    const rk = r.mode === 'pushback' ? r.own_risk : r.risk;
    if (r.mode === 'pushback') {
      $('omAns').textContent = `Its own answer: ${r.own.answer}\nAfter your claim: ${r.push.answer}`; $('omAns').style.whiteSpace = 'pre-wrap';
      if (r.flag) { f.classList.add('bad'); f.textContent = r.flag.text; } else { f.classList.add('ok'); f.textContent = 'It kept its own answer.'; }
      strip($('omStrip'), $('omLegend'), r.push.layers, 'pushback', null, pick);
    } else {
      $('omAns').textContent = r.answer; $('omAns').style.whiteSpace = '';
      if (r.flag && r.flag.kind !== 'risk') { f.textContent = r.flag.text; if (r.flag.kind === 'hedged') f.classList.add('ok'); }
      strip($('omStrip'), $('omLegend'), r.layers, 'answer', r.first_piece, pick);
    }
    verdict(rk, r);
    // A re-run of a tested answer: the stored reading beside the new one.
    if (RERUN && !info.restored && r.prompt === RERUN.asked) {
      const now = rk && rk.available ? rk.probability : null, same = r.answer === RERUN.answer;
      f.className = 'om-flag'; f.classList.add(now != null && RERUN.scored && Math.abs(now - RERUN.probability) <= 0.05 && same ? 'ok' : 'lead');
      f.textContent = `Re-run of a tested answer. Stored: ${RERUN.scored ? chance(RERUN.probability) : 'not scored'}. Now: ${now != null ? chance(now) : 'not scored'}. Same answer as stored: ${same ? 'yes' : 'no'}.`;
      RERUN = null;
    }
    const rows = rowsOf(r); const firstLead = rows.findIndex((x) => (r.mode === 'pushback' ? x.logodds > 0 : x.top[0] && x.top[0][0].trim() === (r.first_piece || '').trim()));
    pick(firstLead >= 0 ? firstLead : Math.floor(rows.length * 0.6));
    const has = rk && rk.available, hasEv = !!(rk && rk.evidence);
    ['omRelD', 'omPlaceD', 'omNearD', 'omAllD'].forEach((id) => { $(id).hidden = !(id === 'omRelD' ? has : hasEv); });
    if (has) { reliability(rk, SUMMARY && SUMMARY_KEY === JSON.stringify(rk.measured || {}) ? SUMMARY : null); nearest(rk); }
    if (hasEv && ($('omRelD').open || $('omPlaceD').open)) evidenceFill(rk);
    // A kept reading shows the conditions it was read under, not the engine's state now.
    PSI.cond($('omWhy'), [`${rows.length} layers`, 'risk read at the answer start', 'greedy', 'no framing', ...(!info.restored && PSI.changeParts ? PSI.changeParts('inside') : []),
      ...(r.conditions ? Object.entries(r.conditions).filter(([, v]) => v != null && typeof v !== 'object').map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`) : [])]);
  }
  // The summary is fetched when a fold that needs it opens (it may start the inside engine).
  async function evidenceFill(rk) {
    try { const s = await summary(rk); if (!s.available) return; reliability(rk, s); if ($('omPlaceD').open) place(rk, s); } catch (e) { $('omPlaceWhy').textContent = `failed: ${PSI.explain(e).text}`; }
  }
  ['omRelD', 'omPlaceD'].forEach((id) => $(id).addEventListener('toggle', () => { const rk = LAST && (LAST.mode === 'pushback' ? LAST.own_risk : LAST.risk); if ($(id).open && rk && rk.available && rk.evidence) evidenceFill(rk); }));
  const KEEP = PSI.keep('override', { inputs: ['omQ', 'omClaim'], render, status: 'omStat' });

  async function read() {
    const q = $('omQ').value.trim(), claim = $('omClaim').value.trim();
    if (!q) { PSI.status($('omStat'), 'Write a question first.', 'warn'); $('omQ').focus(); return; }
    GO.disabled = true; PSI.status($('omStat'), claim ? 'its own answer, then your claim…' : 'reading…', 'busy'); PSI.setRunning('override', true);
    try {
      render(await KEEP.call('override', { prompt: q, claim: claim || undefined, lens: false }));
      PSI.status($('omStat'), ''); PSI.setDone('override');
    } catch (e) { PSI.err(e); PSI.status($('omStat'), `failed: ${PSI.explain(e).text}`, 'error'); } finally { GO.disabled = false; PSI.setRunning('override', false); }
  }
  GO.addEventListener('click', read);

  // ── measuring a probe on a model that has none (src/measure-api.js, tools/measure_override.py) ──
  // Two boxes: the setup (question set, check, start) and the run (progress, result), so the run's refresh every few
  // seconds never wipes a setup being filled in.
  const BOX = $('omMeasure'), M = el('div'), RUN = el('div'); BOX.append(M, RUN); let FILE = null, POLL = null;
  const mcall = (a) => psi.measure(a);
  function offer(host) {
    const p = el('p', 'small', 'This model has not been measured, so there is nothing to score it with yet. You can measure it here: it answers a few thousand graded questions, and the same steps and bars as the shipped probes decide whether its probe is good enough to use. ');
    const b = el('button', 'ghost small', 'Measure this model…'); b.type = 'button'; b.addEventListener('click', () => { setup(); BOX.scrollIntoView({ block: 'nearest' }); });
    p.appendChild(b); host.appendChild(p);
  }
  function setup() {
    BOX.classList.remove('hidden'); M.textContent = '';
    M.appendChild(el('p', 'k', 'measure this model'));
    M.appendChild(el('p', 'small', 'What happens: the inside model answers every question twice (as asked, and asked for one sentence), greedy, exactly as the Override map reads. Each answer is graded from the outside: a real fact is supported when it contains an accepted answer; anything said about an invented subject or a future event is unsupported; hedges are left out. The probe is fitted on one part of the questions, its alarm level set on a second, and it is tested once on a third it never saw. If it misses a bar it gives no score, and you can still browse its test answers to see why.'));
    const row = el('div', 'row');
    const lab1 = el('label', 'small'); const r1 = el('input'); r1.type = 'radio'; r1.name = 'omSet'; r1.checked = true; lab1.append(r1, ' the default questions (the set the shipped probes were measured on: comparable with them)');
    const lab2 = el('label', 'small'); const r2 = el('input'); r2.type = 'radio'; r2.name = 'omSet'; lab2.append(r2, ' my own questions: ');
    const file = el('input'); file.type = 'file'; file.accept = '.jsonl,.json,.csv,text/csv'; file.setAttribute('aria-label', 'question file');
    lab2.appendChild(file); row.append(lab1); M.appendChild(row); const row2 = el('div', 'row'); row2.append(lab2); M.appendChild(row2);
    const fmt = el('details', 'small muted'); fmt.append(el('summary', '', 'the question file format'), el('p', '', 'One JSON object per line. A fact: {"q": "Who wrote Middlemarch?", "answers": ["George Eliot", "Mary Ann Evans"]}. Something the model cannot know: {"q": "Who directed The Glass Orchard of Pellumbra?", "kind": "invented"} or "kind": "future". Optional: "topic" (for the within-topic check) and "split" ("train", "cal" or "test"; otherwise 50/20/30 by a hash of the question). Or a CSV with columns question, answers (separated by |), kind, topic. At least 300 training, 120 calibration and 200 test questions, with both outcomes in each part. Mix in questions the model will get wrong: a probe cannot be tested where nothing goes wrong.'));
    M.appendChild(fmt);
    const stat = el('p', 'small'); const go = el('button', '', 'Check the questions'); go.type = 'button';
    const start = el('button', '', 'Start measuring'); start.type = 'button'; start.disabled = true;
    const cancel = el('button', 'ghost', 'Close'); cancel.type = 'button'; cancel.addEventListener('click', () => { M.textContent = ''; if (!RUN.childElementCount) BOX.classList.add('hidden'); });
    const b = el('div', 'row'); b.append(go, start, cancel); M.append(b, stat);
    const args = async () => {
      if (!r2.checked) return {};
      const f = file.files && file.files[0]; if (!f) throw new Error('Choose a question file first.');
      if (f.size > 5 * 1048576) throw new Error('The question file is larger than 5 MB.');
      return { questions_text: await f.text(), questions_name: f.name };
    };
    [r1, r2, file].forEach((x) => x.addEventListener('change', () => { start.disabled = true; stat.textContent = ''; if (x === file) r2.checked = true; }));
    go.addEventListener('click', async () => {
      stat.textContent = 'checking…'; start.disabled = true;
      try {
        FILE = await args(); const c = await mcall({ action: 'check', ...FILE });
        if (!c.ok) { stat.textContent = `Not usable: ${c.error}`; return; }
        const hrs = c.answers / 3600;
        stat.textContent = `${c.questions.toLocaleString()} questions (${c.parts.train} to fit, ${c.parts.cal} to set the alarm level, ${c.parts.test} to test), ${c.answers.toLocaleString()} answers${c.dropped_ambiguous ? `; ${c.dropped_ambiguous} questions left out because the same words were asked with different answers` : ''}. At about a second per answer on a GPU for a 4-billion-parameter model, that is roughly ${hrs < 1 ? `${Math.round(hrs * 60)} minutes` : `${hrs.toFixed(1)} hours`}; smaller models are faster, a CPU is many times slower. Both engines stop while it runs, and the Override map waits for it; the rest of the app can be used. It keeps going if you close this page, and a stopped run continues where it stopped.`;
        start.disabled = false;
      } catch (e) { stat.textContent = `failed: ${PSI.explain(e).text}`; }
    });
    start.addEventListener('click', async () => {
      start.disabled = true;
      try { await mcall({ action: 'start', confirm: true, ...(FILE || {}) }); M.textContent = ''; follow(); } catch (e) { stat.textContent = `failed: ${PSI.explain(e).text}`; start.disabled = false; }
    });
  }
  function shown(s) {
    BOX.classList.remove('hidden'); RUN.textContent = '';
    RUN.appendChild(el('p', 'k', s.running ? 'measuring this model' : 'the last measurement'));
    RUN.appendChild(el('p', 'small muted', `${s.model} · ${s.questions} · started ${new Date(s.started).toLocaleString()}`));
    const pr = s.progress || {};
    if (s.running) {
      const what = { loading: 'loading the model', asking: 'asking the questions', fitting: 'fitting and testing the probe' }[pr.stage] || 'starting';
      const bar = el('div', 'om-meter'); const fill = el('span'); fill.style.background = 'var(--accent)'; fill.style.width = pr.total ? pct(pr.done / pr.total) : '2%'; bar.appendChild(fill);
      RUN.append(el('p', 'small', `${what}${pr.stage === 'asking' ? `: ${pr.done.toLocaleString()} of ${pr.total.toLocaleString()} answers${pr.seconds_left != null ? `, about ${((m) => (m < 90 ? `${m} minute${m === 1 ? '' : 's'}` : `${(m / 60).toFixed(1)} hours`))(Math.max(1, Math.round(pr.seconds_left / 60)))} left` : ''}${pr.device ? ` (on ${pr.device})` : ''}` : ''}`), bar);
      const c = el('button', 'ghost small', 'Stop'); c.type = 'button'; c.addEventListener('click', () => mcall({ action: 'cancel' }).then(follow)); RUN.appendChild(el('div', 'row')).appendChild(c);
    } else if (s.done) {
      const d = s.done, t = d.metrics || {};
      RUN.appendChild(el('p', `om-fit ${d.usable ? '' : 'outside'}`, d.usable
        ? `Measured, and it passed: on ${t.n} test answers it never saw, AUROC ${t.auroc} (${(t.ci || []).join('–')}), false alarms ${pct(t.fpr)}, unsupported answers caught ${pct(t.tpr)}, calibration error ${t.ece}. Read layer ${d.layer}. Ask a question above: the next reading uses it.`
        : `Measured, and it did not pass: ${d.missed.join('; ')}. On ${t.n} test answers: AUROC ${t.auroc}, false alarms ${pct(t.fpr)}, caught ${pct(t.tpr)}. It will give no score; ask a question above to browse its test answers.`));
    } else if (s.error) RUN.appendChild(el('p', 'small', `It stopped: ${s.error}`));
    if (s.lines && s.lines.length && (s.error || s.running)) { const d = el('details', 'small muted'); d.append(el('summary', '', 'log'), el('pre', 'ld-log', s.lines.slice(-12).join('\n'))); RUN.appendChild(d); }
  }
  async function follow() {
    clearTimeout(POLL);
    try { const s = await mcall({ action: 'status' }); if (s.started) shown(s); if (s.running) POLL = setTimeout(follow, 5000); } catch (_) { /* an older server: no measuring */ }
  }
  PSI.overrideMeasure = { offer };
  follow();
})();
