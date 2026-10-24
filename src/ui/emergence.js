// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Emergence tab: perform the "Where Understanding Begins" measurements yourself.
//   1. Ask a recorded model at any saved training step (Pythia-410m, its checkpoints on disk).
//   2. "When did it learn this?": your own test, scored at every checkpoint.
//   3. The heads that look back in the loaded model, and what dies when you knock them out.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => String(s == null ? '' : s).replace(/\n/g, '⏎');
  const err = (e, o) => PSI.err(e, o);
  const EX = window.PSIExp;
  const fmt = (s) => s >= 1000 ? `${(s / 1000).toLocaleString()}k` : String(s);
  // R.model: the weights the head scan and knock-outs came from; results on other weights are archived, never mixed.
  const R = { steps: [], key: [], ask: {}, test: null, scan: null, ko: [], model: '' };
  const N = EX.NOISE, HT = 0.3;   // noise floor for one test's margin (nats); a head "looks back" above this induction attention
  const COPY = (P) => (P && P.label ? `${P.n} tokens × ${P.trials} trials, ids ${P.id_lo.toLocaleString()}–${P.id_hi.toLocaleString()}, seed ${P.seed}` : 'protocol not reported');
  const heatMode = () => ($('emHeadsShow') ? $('emHeadsShow').value : 'induction');
  const VIEW = 'emergence';
  const visible = () => !$('tab-emergence').classList.contains('hidden');
  const frag = (...parts) => { const f = document.createDocumentFragment(); parts.forEach((p) => { if (p != null && p !== '') f.append(p); }); return f; };
  const ckptLink = () => PSI.link('Set the checkpoints folder in Settings', 'settings', 'stCkpt');

  // ── what each button needs ──────────────────────────────────────────────
  const CK_BTNS = ['emAsk', 'emCmp', 'emTestGo'];
  CK_BTNS.forEach((id) => PSI.requires($(id), ['engine', 'checkpoints']));
  ['emScan', 'emKo', 'emKoRnd'].forEach((id) => PSI.requires($(id), 'engine'));
  // Until the checkpoints are listed, the step buttons wait too (with their own reason).
  const stepsReason = (r) => CK_BTNS.forEach((id) => PSI.block($(id), 'steps', r));

  // ── 1. ask the model at a step ───────────────────────────────────────────
  const slider = $('emStep');
  function noCheckpoints(why) {
    slider.disabled = true; $('emStepLabel').textContent = 'no checkpoints';
    if (!why && PSI.state.known && !PSI.state.engine) PSI.status($('emStat'), '');   // the buttons' note already says the engine is missing
    else PSI.status($('emStat'), frag(why ? `No training checkpoints: ${why} ` : 'No training checkpoints found. ', ckptLink()), 'warn');
    stepsReason(PSI.state.checkpoints ? { text: 'No checkpoints were found:', link: ckptLink } : null);
  }
  slider.disabled = true;
  let listing = false;
  // Lists the checkpoints only when the tab is open and a checkpoints folder is set (listing starts the inside engine).
  async function init() {
    if (listing || R.steps.length) return;
    if (!PSI.state.checkpoints) { if (PSI.state.known) noCheckpoints(); return; }
    listing = true; stepsReason({ text: 'Listing the checkpoints…' });
    PSI.status($('emStat'), 'looking for checkpoints (this starts the inside engine if it is not running)…', 'busy');
    try {
      const l = await psi.ckptList();
      if (!l || !l.steps || !l.steps.length) { noCheckpoints('the folder has none.'); return; }
      R.steps = l.steps; R.key = l.key_steps || l.steps; R.device = l.device;
      slider.disabled = false; slider.min = '0'; slider.max = String(R.steps.length - 1); slider.value = String(Math.max(0, R.steps.indexOf(512)));
      stepLabel(); stepsReason(null);
      PSI.status($('emStat'), frag(`${R.steps.length} `, PSI.term('checkpoints', 'checkpoint'), ` on disk · ${l.device}`), 'ok');
      $('emTestAllLbl').textContent = `all ${R.steps.length} checkpoints (slower)`;
    } catch (e) { noCheckpoints(`${PSI.explain(e).text}`); } finally { listing = false; }
  }
  const curStep = () => R.steps[Number(slider.value)] || 0;
  function stepLabel() { if (!R.steps.length) return; const s = curStep(); $('emStepLabel').textContent = `step ${s.toLocaleString()} · ${(s * 2097152 / 1e9).toFixed(2)} B tokens`; }
  slider.addEventListener('input', stepLabel);

  // Run state for the nav dot; a failure goes to the status line (what failed) and the message area (how to fix it).
  async function job(stat, busyText, failText, fn) {
    PSI.setRunning(VIEW, true); PSI.status(stat, busyText, 'busy');
    try { await fn(); PSI.setDone(VIEW); return true; }
    catch (e) { err(e); PSI.status(stat, `${typeof failText === 'function' ? failText() : failText}: ${PSI.explain(e).text}`, 'error'); return false; }
    finally { PSI.setRunning(VIEW, false); }
  }
  async function ask(steps) {
    if (!steps.length) return PSI.status($('emStat'), 'Steps 512 and 1,000 are not among the checkpoints on disk.', 'warn');
    const btns = [$('emAsk'), $('emCmp')]; btns.forEach((b) => { b.disabled = true; });
    const prompt = $('emPrompt').value.trim(), text = $('emText').value.trim(); let cur = steps[0];
    await job($('emStat'), `loading step ${cur.toLocaleString()} and asking…`, () => `Asking at step ${cur.toLocaleString()} failed`, async () => {
      for (const s of steps) {
        cur = s; PSI.status($('emStat'), `loading step ${s.toLocaleString()} and asking…`, 'busy');
        R.ask[s] = await psi.ckptAsk({ step: s, prompt, text: text || undefined, n_new: 40 });
      }
      renderAsk(steps); KEEP_ASK.save({ steps, ask: Object.fromEntries(steps.map((s) => [s, R.ask[s]])) });
      PSI.status($('emStat'), `done · ${steps.map((s) => `${R.ask[s].seconds}s`).join(', ')}`, 'ok');
    });
    btns.forEach((b) => { b.disabled = false; });
  }
  // Kept views (PSI.keep): these panels build their results here from one call per checkpoint, so the finished result is
  // stored with save() and redrawn by render() after a reload (info.restored: from before). The head scan and
  // knock-outs below are already kept in this browser (EX.keep); only their inputs are kept here.
  const KEEP_ASK = PSI.keep('emergence-ask', { inputs: ['emPrompt', 'emText'], status: 'emStat', render: (a) => {
    if (!a || !Array.isArray(a.steps) || !a.ask) return;
    a.steps.forEach((s) => { if (a.ask[s]) R.ask[s] = a.ask[s]; });
    const steps = a.steps.filter((s) => R.ask[s]); if (steps.length) renderAsk(steps);
  } });
  const KEEP_TEST = PSI.keep('emergence-test', { inputs: ['emTestPrompt', 'emTestGood', 'emTestBad', 'emTestAll'], status: 'emTestStat', render: (t) => {
    if (!t || !Array.isArray(t.curve) || !t.curve.length) return;
    R.test = t; if (!t.partial) window.PSI_LAST_TEST = t;
    if (!$('emTestCanvas').clientWidth) return;   // drawn when the tab opens (drawAll)
    renderTest();
  } });
  PSI.keep('emergence-heads', { inputs: ['emK', 'emHeadsShow'], render: () => {} });
  $('emAsk').addEventListener('click', () => ask([curStep()]));
  $('emCmp').addEventListener('click', () => ask([512, 1000].filter((s) => R.steps.includes(s))));
  function renderAsk(steps) {
    const box = $('emAskOut'); box.textContent = ''; R.lastAsk = steps;
    steps.forEach((s) => {
      const r = R.ask[s]; const col = el('div', 'em-col');
      col.appendChild(el('p', 'k', `step ${s.toLocaleString()} · ${r.tokens_seen_b} B tokens · loaded in ${r.load_s}s`));
      if (r.continuation) {
        const p = el('p', 'em-cont'); p.appendChild(el('span', 'muted', show($('emPrompt').value.trim())));
        r.continuation.pieces.forEach((pc, i) => { const sp = el('span', null, pc); sp.title = `p = ${r.continuation.p[i]}`; sp.style.background = `rgba(242,241,238,${(r.continuation.p[i] * 0.35).toFixed(2)})`; p.appendChild(sp); });
        col.appendChild(p);
      }
      const ind = r.induction;
      const p1 = el('p', 'small'); EX.put(p1, `copying a repeated sequence: ${ind.score >= 0 ? '+' : ''}${ind.score} `, PSI.term('nats'), ` saved (${COPY(ind.protocol)}${ind.protocol ? '' : '; an older engine used 50 tokens × 4 trials'}) · ${ind.n_heads} heads look back (induction attention > ${ind.head_threshold || HT}; strongest ${ind.max_head}) · strongest previous-token head ${ind.prev_token_max}`); col.appendChild(p1);
      if (r.position) { const p2 = el('p', 'small'); EX.put(p2, `loss by position in the text: ${r.position.early} `, PSI.term('bits', 'bits per token'), ` early, ${r.position.late} bits late, over ${r.position.tokens} tokens (compare the curve’s shape across steps; its level depends on the text)`); col.appendChild(p2); }
      const cv = el('canvas'); cv.className = 'em-heads'; col.appendChild(cv);
      const cv2 = el('canvas'); cv2.className = 'em-pos'; col.appendChild(cv2);
      box.appendChild(col);
      const prevMode = heatMode() === 'prev' && ind.prev_token;
      drawHeads(cv, prevMode ? ind.prev_token : ind.heads, `${prevMode ? 'previous-token' : 'induction'} score, heads at step ${fmt(s)}`);
      PSI.canvasLabel(cv, prevMode ? `Heat map of every attention head's previous-token score at step ${s.toLocaleString()}: the strongest ${ind.prev_token_max}.` : `Heat map of every attention head's induction score at step ${s.toLocaleString()}: ${ind.n_heads} heads above ${ind.head_threshold || HT}, the strongest ${ind.max_head}.`);
      if (r.position) { drawPos(cv2, r.position.bins); PSI.canvasLabel(cv2, `Loss by token position at step ${s.toLocaleString()}: ${r.position.early} bits early, ${r.position.late} bits late.`); } else cv2.remove();
    });
    const w = steps.length > 1 ? 'Same prompt, two checkpoints. At 512 the model continues from word statistics; at 1,000 it can look back, and the second copy of a repeated sequence costs it almost nothing.' : '';
    $('emAskWhy').textContent = w;
  }
  function drawHeads(cv, M, title) {
    if (!M) return; const r = Math.min(2, devicePixelRatio || 1), W = cv.clientWidth || 300; const L = M.length, Hn = M[0].length, Hpx = Math.min(W * (L / Hn), 260); cv.width = W * r; cv.height = (Hpx + 16) * r; cv.style.height = `${Hpx + 16}px`;
    const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); const cw = cv.width / Hn, ch = (cv.height - 16 * r) / L;
    M.forEach((row, l) => row.forEach((v, h) => { g.fillStyle = `rgba(242,241,238,${Math.min(1, v) * 0.95 + 0.03})`; g.fillRect(h * cw, 16 * r + l * ch, cw - 1, ch - 1); }));
    g.fillStyle = '#a9a69f'; g.font = `${10 * r}px "IBM Plex Mono", monospace`; g.fillText(title, 0, 11 * r);
  }
  function drawPos(cv, bins) {
    const r = Math.min(2, devicePixelRatio || 1), W = cv.clientWidth || 300; cv.width = W * r; cv.height = 90 * r; cv.style.height = '90px';
    const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height);
    const lo = Math.min(...bins) - 0.5, hi = Math.max(...bins) + 0.5; const X = (i) => 28 * r + (i / Math.max(1, bins.length - 1)) * (cv.width - 34 * r), Y = (v) => cv.height - 14 * r - ((v - lo) / (hi - lo)) * (cv.height - 26 * r);
    g.strokeStyle = '#f2f1ee'; g.lineWidth = 1.6 * r; g.beginPath(); bins.forEach((v, i) => { i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); }); g.stroke();
    g.fillStyle = '#75726c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillText(`${hi.toFixed(0)}`, 2 * r, 18 * r); g.fillText(`${lo.toFixed(0)}`, 2 * r, cv.height - 14 * r); g.fillText('loss (bits) by token position →', 28 * r, cv.height - 3 * r);
  }

  // ── 2. when did it learn this? ──────────────────────────────────────────
  // One checkpoint per request, so Stop works between checkpoints; the verdict is the same rule as checkpoints.py verdict:
  // "learned" needs a margin above the noise (0.5 nats) that stays above it; identical answers are indistinguishable.
  const status = (c) => (c.indistinguishable ? 'indistinguishable' : c.margin > N ? 'learned' : c.margin >= -N ? 'within noise' : 'surface');
  function verdict(curve) {
    curve = curve.slice().sort((x, y) => x.step - y.step); const st = curve.map(status);
    const first = st.findIndex((_, i) => st.slice(i).every((x) => x === 'learned'));
    const learned = first < 0 ? null : { after_step: first > 0 ? curve[first - 1].step : null, by_step: curve[first].step };
    if (learned && learned.after_step != null) { const done = new Set(curve.map((c) => c.step)); learned.between = R.steps.filter((x) => x > learned.after_step && x < learned.by_step && !done.has(x)); }
    return { curve, learned, never: first < 0, end: st[st.length - 1] };
  }
  $('emTestGo').addEventListener('click', () => scoreTest(null));
  // For ui/hf-pull.js: the bracket of the last test, a fresh listing after a step was fetched, and scoring added steps.
  window.PSIEm = { learned: () => (R.test && R.test.learned) || null, steps: () => R.steps.slice(),
    relist: async () => { R.steps = []; await init(); }, score: (steps) => (R.test ? scoreTest(steps, R.test) : null) };
  // "score every checkpoint between": the same test at the unscored checkpoints inside the bracket, added to the curve.
  $('emTestFill').addEventListener('click', () => { if (R.test && R.test.learned && R.test.learned.between && R.test.learned.between.length) scoreTest(R.test.learned.between, R.test); });
  async function scoreTest(only, base) {
    const prompt = base ? base.prompt : $('emTestPrompt').value.trim(), good = base ? base.good : $('emTestGood').value.split(',').map((s) => ' ' + s.trim()).filter((s) => s.trim()), bad = base ? base.bad : $('emTestBad').value.split(',').map((s) => ' ' + s.trim()).filter((s) => s.trim());
    if (!prompt || !good.length || !bad.length) return PSI.status($('emTestStat'), 'Fill in the prompt and both answers first.', 'warn');
    const steps = only || ($('emTestAll').checked ? R.steps : R.key);
    if (!steps.length) return noCheckpoints();
    const b = $('emTestGo'), bf = $('emTestFill'); b.disabled = true; bf.disabled = true; PSI.status($('emTestStat'), '');
    const p = PSI.progress($('emTestProg'), { total: steps.length, label: 'checkpoint' });
    const curve = []; let secs = 0, lastCond = null;
    try {
      for (let i = 0; i < steps.length; i++) {
        if (p.cancelled) break;
        p.step(i, `loading step ${steps[i].toLocaleString()}`);
        const r = await psi.ckptTest({ prompt, good, bad, steps: [steps[i]] });
        curve.push(...r.curve); secs += r.seconds || 0; lastCond = r.conditions || lastCond; p.step(i + 1);
      }
      if (!curve.length) { p.finish('stopped before any checkpoint was scored', { markDone: false }); PSI.status($('emTestStat'), 'Stopped: nothing scored.', 'warn'); return; }
      const all = base ? [...base.curve.filter((c) => !curve.some((x) => x.step === c.step)), ...curve] : curve;
      R.test = { prompt, good, bad, ...verdict(all), seconds: Math.round(((base ? base.seconds : 0) + secs) * 10) / 10, partial: p.cancelled ? `${curve.length} of ${steps.length}` : '', cond: lastCond, time: Date.now() };
      renderTest(); KEEP_TEST.save(R.test);
      if (p.cancelled) { p.finish(`stopped after ${curve.length} of ${steps.length} checkpoints`, { markDone: false }); PSI.status($('emTestStat'), `${R.test.seconds}s · stopped early, so the verdict covers only the checkpoints scored`, 'warn'); }
      else { window.PSI_LAST_TEST = R.test; p.finish(`${steps.length} checkpoints scored`); PSI.status($('emTestStat'), `${R.test.seconds}s · ${R.test.curve.length} of ${R.steps.length} checkpoints on disk scored`, 'ok'); }
    } catch (e) {
      p.fail(`stopped by an error after ${curve.length} of ${steps.length} checkpoints`); err(e);
      PSI.status($('emTestStat'), `Scoring at step ${(steps[curve.length] || 0).toLocaleString()} failed: ${PSI.explain(e).text}`, 'error');
    } finally { b.disabled = false; bf.disabled = false; }
  }
  const says = (c) => c.top.map(([p, v]) => `${show(p)} ${(v * 100).toFixed(0)}%`).join(' · ');
  function renderTest() {
    const t = R.test, cv = $('emTestCanvas'), r = Math.min(2, devicePixelRatio || 1); cv.style.height = '200px'; if (!cv.clientWidth) return; cv.width = cv.clientWidth * r; cv.height = 200 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, left = 44 * r, right = W - 20 * r, top = 20 * r, bot = H - 26 * r;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const lx = (s) => Math.log10(Math.max(1, s)), X = (s) => left + (lx(s) / lx(200000)) * (right - left);
    const vals = t.curve.map((c) => c.margin), lo = Math.min(-1, ...vals), hi = Math.max(1, ...vals), Y = (v) => bot - ((v - lo) / (hi - lo)) * (bot - top);
    g.strokeStyle = 'rgba(242,241,238,0.35)'; g.setLineDash([3 * r, 3 * r]); g.lineWidth = r; g.beginPath(); g.moveTo(left, Y(0)); g.lineTo(right, Y(0)); g.stroke(); g.setLineDash([]);
    g.fillStyle = 'rgba(242,241,238,0.06)'; g.fillRect(left, Y(N), right - left, Y(-N) - Y(N));   // the noise band: |margin| <= 0.5 nats
    if (t.learned && t.learned.after_step != null) { g.fillStyle = 'rgba(255,195,107,0.12)'; g.fillRect(X(t.learned.after_step), top, X(t.learned.by_step) - X(t.learned.after_step), bot - top); }
    g.strokeStyle = '#ffc36b'; g.lineWidth = 2 * r; g.beginPath(); t.curve.forEach((c, i) => { const x = X(c.step), y = Y(c.margin); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
    t.curve.forEach((c) => { const st = status(c), x = X(c.step), y = Y(c.margin); g.fillStyle = st === 'learned' ? '#ffc36b' : '#75726c';
      if (st === 'indistinguishable') { g.strokeStyle = '#a9a69f'; g.lineWidth = r; g.beginPath(); g.arc(x, y, 3.5 * r, 0, Math.PI * 2); g.stroke(); } else { g.beginPath(); g.arc(x, y, 3 * r, 0, Math.PI * 2); g.fill(); } });
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#75726c'; g.textAlign = 'center'; [1, 10, 100, 1000, 10000, 100000].forEach((s) => g.fillText(fmt(s), X(s), H - 10 * r));
    g.textAlign = 'left'; g.fillStyle = '#a9a69f'; g.fillText(`understanding answer ahead ↑  (log-odds margin; grey band: within noise, ±${N} nats; hollow: answers indistinguishable)`, left + 4 * r, 12 * r); g.fillText('surface answer ahead ↓', left + 4 * r, bot - 4 * r);
    const last = t.curve[t.curve.length - 1], ind = t.curve.filter((c) => c.indistinguishable).length, shared = t.curve.some((c) => c.same_first_piece);
    const scored = `${t.curve.length} of ${R.steps.length || '?'} checkpoints on disk scored. `;
    const B = t.learned && t.learned.between && t.learned.between.length ? t.learned.between : null;
    const why = scored + (t.partial ? `Stopped early (${t.partial} checkpoints scored), so this covers only those. ` : '')
      + (ind === t.curve.length ? `Answers indistinguishable at the first piece at every checkpoint: the two answers are the same pieces to this tokenizer, so the test cannot tell them apart (not "not learned"). Rewrite one answer.`
      : t.never ? (t.end === 'within noise' ? `Within noise at the last checkpoint scored (step ${last.step.toLocaleString()}, margin ${last.margin}; noise ±${N} nats): neither learned nor refused.` : t.end === 'indistinguishable' ? `At the last checkpoint scored the answers are indistinguishable at the first piece.` : `${t.partial ? 'Not learned by the last checkpoint scored' : 'Not learned by the end of this run'}: at step ${last.step.toLocaleString()} the surface answer is ahead by more than the noise (margin ${last.margin}).`) + ` It says: ${says(last)}.`
      : t.learned.after_step == null ? `Already above the noise at the first checkpoint scored (step ${t.learned.by_step.toLocaleString()}); test earlier steps to find the moment.`
      : `Learned between step ${t.learned.after_step.toLocaleString()} and step ${t.learned.by_step.toLocaleString()} (${(t.learned.after_step * 2097152 / 1e9).toFixed(1)} to ${(t.learned.by_step * 2097152 / 1e9).toFixed(1)} billion tokens): above the noise (${N} nats) from then on. ${B ? `${B.length} checkpoint${B.length > 1 ? 's' : ''} on disk inside that bracket ${B.length > 1 ? 'were' : 'was'} not scored. ` : 'No unscored checkpoint lies inside that bracket. '}Final margin ${last.margin}; at the end it says ${says(last)}.`)
      + (shared && ind < t.curve.length ? ' The two answers share their first piece, so they were compared whole (every piece summed).' : '');
    $('emTestWhy').textContent = why;
    $('emTestFill').classList.toggle('hidden', !B); if (B) $('emTestFill').textContent = `Score the ${B.length} checkpoint${B.length > 1 ? 's' : ''} between ${t.learned.after_step.toLocaleString()} and ${t.learned.by_step.toLocaleString()}`;
    // the checkpoints, not the inside engine's loaded model, are what was scored here
    PSI.cond($('emTestWhy'), [`training checkpoints from the checkpoints folder (fp32 on ${R.device || 'a device not reported'})`, `${t.curve.length} of ${R.steps.length || '?'} steps`, `noise ±${N} nats`, { t: 'not the loaded model', title: 'the inside engine’s own model and any cut on it play no part here' }, new Date(t.time || Date.now()).toLocaleString()], 'scored on');
    PSI.canvasLabel(cv, `Your test’s margin at ${t.curve.length} training checkpoints. ${why}`);
    const tb = $('emTestTable').querySelector('tbody'); tb.textContent = '';
    t.curve.forEach((c) => { const st = status(c), tr = el('tr'); tr.appendChild(el('td', null, c.step.toLocaleString()));
      tr.appendChild(el('td', st === 'learned' ? 'ok' : st === 'surface' ? 'no' : 'noise', st === 'indistinguishable' ? 'answers indistinguishable at first piece' : `${c.margin >= 0 ? '+' : ''}${c.margin} · ${st}`));
      tr.appendChild(el('td', null, says(c))); tb.appendChild(tr); });
  }

  // ── 3. the heads in the loaded model, and knocking them out ─────────────
  // A reply from other weights than the results on screen: archive those (kept in this browser), start fresh.
  function sameModel(cond) {
    const k = EX.key(cond); if (!k || k === R.model) return '';
    const had = R.scan || R.ko.length; if (had) { EX.archive('emergence-heads'); R.scan = null; R.ko = []; }
    R.model = k; return had ? 'The inside engine’s model changed: earlier scans and knock-outs were archived, not mixed with these. ' : '';
  }
  const saveHeads = () => { if (!EX.keep('emergence-heads', R.model, { scan: R.scan, ko: R.ko }, R.scan ? R.scan.model : '')) PSI.status($('emHeadSaved'), 'Results are not kept between sessions here (this browser blocks storage).', 'warn'); };
  $('emScan').addEventListener('click', async () => {
    const b = $('emScan'); b.disabled = true;
    await job($('emHeadStat'), 'scanning all heads of the loaded model…', 'The head scan failed', async () => {
      const r = await psi.heads({}); const note = sameModel(r.conditions);
      r.model = r.conditions && r.conditions.model; r.stamp = EX.stamp(r.conditions, { protocol: `copying ${COPY(r.protocol)}` }); r.cut = EX.cutOf(r.conditions);
      R.scan = r; EX.setLayers(R.scan.layers); renderScan(); saveHeads();
      PSI.status($('emHeadStat'), frag(`${note}${R.scan.seconds}s · ${R.scan.n_heads} of ${R.scan.layers * R.scan.heads_per_layer} heads look back (induction attention > ${R.scan.head_threshold || HT}) · copying saves ${R.scan.induction_score} `, PSI.term('nats'), ` (${COPY(R.scan.protocol)})`), 'ok'); EX.mark($('emHeadStat'));
    });
    b.disabled = false;
  });
  $('emHeadsShow').addEventListener('change', () => { renderScan(); if (R.lastAsk) renderAsk(R.lastAsk); });
  function renderScan() {
    if (!R.scan) return EX.empty($('emScanCanvas'), 140, 'Nothing scanned yet: press Scan all heads.');
    const cv = $('emScanCanvas'); if (!cv.clientWidth) return;
    const prev = heatMode() === 'prev';
    if (prev && !R.scan.prev_token) return EX.empty(cv, 140, 'This scan has no previous-token scores (an older inside engine): scan again.');
    drawHeads(cv, prev ? R.scan.prev_token : R.scan.heads, `${prev ? 'previous-token' : 'induction'} score, ${R.scan.layers} layers × ${R.scan.heads_per_layer} heads`);
    const top = R.scan.top.slice(0, 8).map((t) => `L${t.layer}H${t.head} ${t.score}`).join(' · ');
    let ptop = '';
    if (R.scan.prev_token) { const f = []; R.scan.prev_token.forEach((row, l) => row.forEach((v, h) => f.push([v, l, h]))); ptop = f.sort((a, b) => b[0] - a[0]).slice(0, 6).map(([v, l, h]) => `L${l}H${h} ${v}`).join(' · '); }
    $('emScanTop').textContent = `strongest induction heads: ${top}${ptop ? ` · strongest previous-token heads: ${ptop}` : ''} · ${R.scan.stamp || ''}`;
    PSI.canvasLabel(cv, prev ? `Heat map of the previous-token score of all ${R.scan.layers * R.scan.heads_per_layer} heads. Strongest: ${ptop}.` : `Heat map of the induction score of all ${R.scan.layers * R.scan.heads_per_layer} heads (${R.scan.layers} layers × ${R.scan.heads_per_layer}): ${R.scan.n_heads} above ${R.scan.head_threshold || HT}. Strongest: ${top}.`);
  }
  async function knock(plan) {
    const k = Number($('emK').value) || 12; const btns = [$('emKo'), $('emKoRnd')]; btns.forEach((b) => { b.disabled = true; });
    await job($('emHeadStat'), `knocking out ${k} ${plan === 'random' ? 'random' : 'induction'} heads and re-measuring…`, `Knocking out ${k} heads failed`, async () => {
      // the control's seed: one more than the random controls already run on these weights, shown in its row
      const seed = plan === 'random' ? R.ko.filter((x) => x.plan === 'random').length : null;
      const r = await psi.ablateHeads({ top_k: k, plan, seed: seed || 0 }); const note = sameModel(r.conditions);
      r.seed = seed; r.cut = EX.cutOf(r.conditions); r.stamp = EX.stamp(r.conditions, { protocol: 'copying 40 tokens × 3 trials', seed }); r.time = Date.now();
      R.ko.push(r); renderKo(); saveHeads(); PSI.status($('emHeadStat'), `${note}${r.seconds}s · the heads were removed only while measuring; the model is as it was`, 'ok'); EX.mark($('emHeadStat'));
    });
    btns.forEach((b) => { b.disabled = false; });
  }
  $('emKo').addEventListener('click', () => knock('top')); $('emKoRnd').addEventListener('click', () => knock('random'));
  function renderKo() {
    const tb = $('emKoTable').querySelector('tbody'); tb.textContent = '';
    R.ko.forEach((r) => {
      const tr = el('tr'); const who = el('td', null, `${r.n} ${r.plan === 'random' ? `random heads (control, seed ${r.seed != null ? r.seed : 'not recorded'})` : 'induction heads'}`); who.appendChild(el('div', 'und-stamp', r.stamp || 'conditions not recorded')); tr.appendChild(who);
      // a number the engine did not return says so, never 0%
      const cell = (b, a, fmtv, worseIfLower) => {
        if (a == null || b == null) return el('td', 'no', `${b == null ? 'not measured' : fmtv(b)} → ${a == null ? 'not measured' : fmtv(a)}`);
        const t = el('td', null, `${fmtv(b)} → ${fmtv(a)}`); const worse = worseIfLower ? a < b * 0.5 : a > b * 1.5; t.className = worse ? 'no' : Math.abs(a - b) < Math.abs(b) * 0.1 ? 'ok' : ''; return t;
      };
      const cc = cell(r.before.copy_nats_saved, r.after.copy_nats_saved, (v) => `${v} nats`, true);
      if (r.after.copy_nats_saved != null && r.after.copy_nats_saved < 0) cc.appendChild(el('span', 'ls-off', ' ▾ below 0: the second copy got harder (off the scale, not “no copying”)'));
      tr.appendChild(cc);
      tr.appendChild(cell(r.before.invented_second_over_first, r.after.invented_second_over_first, (v) => `${Math.round(v * 100)}%`, false));
      tr.appendChild(cell(r.before.text_bits_per_token, r.after.text_bits_per_token, (v) => `${v} bits`, false));
      tb.appendChild(tr);
    });
    const last = R.ko[R.ko.length - 1];
    const B = last.before, A = last.after, textUp = (A.text_bits_per_token - B.text_bits_per_token) / (B.text_bits_per_token || 1), random = last.plan === 'random';
    // Say only what these numbers support: the explanations were written for Qwen3-4B, and a small model (the toy Llama) differs.
    $('emKoWhy').textContent = last.total && last.n >= last.total
      ? `That is every attention head this model has (${last.total}), so this shows the model with no attention at all, and the random control cannot be told apart from it. Use fewer than ${last.total}.`
      : B.copy_nats_saved < 0.5
        ? `This model barely copies to begin with (repeating a sequence saves only ${B.copy_nats_saved} nats): it has no induction circuit to remove, like Pythia before its phase shift at step 512.`
        : random
          ? (B.copy_nats_saved - A.copy_nats_saved < B.copy_nats_saved * 0.3 && textUp < 0.1 ? 'Random heads: little changes. The circuit is specific.' : `Random heads hurt here too (copying ${B.copy_nats_saved} → ${A.copy_nats_saved} nats, text ${B.text_bits_per_token} → ${A.text_bits_per_token} bits): in this model the work is not concentrated in a few heads.`)
          : A.copy_nats_saved < B.copy_nats_saved * 0.5
            ? (textUp < 0.1 ? 'With the induction heads gone the model can no longer copy what it has seen, and a new word is no cheaper the second time, while ordinary text costs about the same: word statistics survive, context use does not. This is the step-512 model, re-created in a finished one by removing a handful of heads.' : `Copying is gone, but ordinary text also got worse (${B.text_bits_per_token} → ${A.text_bits_per_token} bits): in this model these heads do more than copy.`)
            : `Copying survived: this model carries the circuit with backups. Raise the count and knock out again${last.total >= 1000 ? ' (in Qwen3-4B about 100 ends copying and 160 kills it)' : ''}.`;
    // the baseline beside the claim: the induction knock-out against the random control of the same size, and what did not fit leads
    // pairs only from the same weights (R.ko holds one model's results) and the same cut state
    const same = (x) => x.n === last.n && x.cut === last.cut;
    const top = R.ko.filter((x) => x.plan !== 'random' && same(x)).pop(), rnd = R.ko.filter((x) => x.plan === 'random' && same(x)).pop();
    const lost = (x) => x.before.copy_nats_saved - x.after.copy_nats_saved;
    if (top && rnd) $('emKoWhy').textContent = (lost(rnd) >= lost(top) * 0.8
      ? `Not specific: ${last.n} random heads (seed ${rnd.seed}) cost as much copying as the induction heads (${lost(rnd).toFixed(2)} vs ${lost(top).toFixed(2)} nats lost). `
      : `Against the control: the induction heads cost ${lost(top).toFixed(2)} nats of copying, ${last.n} random heads (seed ${rnd.seed}) ${lost(rnd).toFixed(2)}. `) + $('emKoWhy').textContent;
    else if (top) $('emKoWhy').textContent += ` No control yet at ${last.n} heads on this model in this state (${last.cut}): press random heads (control) to see how much of this any ${last.n} heads would do.`;
    $('emKoWhy').textContent += ' Copying: 40 random tokens (ids 100–30,000) shown twice, 3 trials, seed 0, the same protocol as the scan and the Tipping-point scorecard.';
  }

  // ── empty charts, redraws, and when to list the checkpoints ─────────────
  function drawAll() {
    if (R.test) renderTest(); else EX.empty($('emTestCanvas'), 200, 'Nothing scored yet: press Find the step.');
    renderScan();
  }
  addEventListener('resize', () => { if (visible()) drawAll(); });
  document.querySelector('[data-tab=emergence]').addEventListener('click', () => setTimeout(() => { drawAll(); init(); }, 0));
  PSI.onState((s) => {
    if (R.steps.length) return;
    if (s.checkpoints) { if (visible()) init(); else { stepsReason(null); if ($('emStepLabel').textContent === 'no checkpoints') { $('emStepLabel').textContent = ''; PSI.status($('emStat'), ''); } } }
    else if (s.known) noCheckpoints();
  });
  // The inside engine's model was switched in Settings: the scan and knock-outs belong to the old weights.
  document.addEventListener('psi-inside-model', (e) => {
    if ((!R.scan && !R.ko.length) || !EX.other(R.model)) return;
    EX.archive('emergence-heads'); R.scan = null; R.ko = []; R.model = '';
    $('emKoTable').querySelector('tbody').textContent = ''; $('emKoWhy').textContent = ''; $('emScanTop').textContent = ''; renderScan();
    PSI.status($('emHeadSaved'), `The inside engine’s model changed (now ${e.detail.to}): the earlier scan and knock-outs were archived, not mixed with new ones.`, 'warn');
  });
  const kept = EX.saved('emergence-heads');
  if (kept && kept.data && (kept.data.scan || (kept.data.ko || []).length)) {
    R.model = kept.key; R.scan = kept.data.scan || null; R.ko = kept.data.ko || [];
    if (R.ko.length) renderKo();
    PSI.status($('emHeadSaved'), `Shown: the last scan and knock-outs, kept from ${new Date(kept.saved).toLocaleString()} (${kept.model || 'model not recorded'}). Results on other weights start fresh and archive these.`, '');
  }
  drawAll();
})();
