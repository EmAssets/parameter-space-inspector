// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Tipping point tab: remove structure from the finished model and score what remains.
// Every row is read against the intact row (a 0% cut) of the same model; rows from other weights are archived, never mixed.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  const EX = window.PSIExp;
  const VIEW = 'lesion';
  // R.model: the weights these rows came from (PSIExp.key); R.intact: that model's intact row.
  const R = { cards: [], sweep: null, ref: {}, intact: null, model: '' };
  const BTNS = ['lsBase', 'lsRestore', 'lsSave', 'lsApply', 'lsSweep', 'lsManual', 'lsFind', 'lsConceptGo'];
  BTNS.forEach((id) => PSI.requires($(id), 'engine'));
  const stat = (text, kind) => PSI.status($('lsStat'), text, kind);
  const pc = (a, b, d = 1) => `${((100 * a) / (b || 1)).toFixed(d)}%`;
  const big = (n) => Number(n || 0).toLocaleString();
  const COPY_LABEL = (P) => (P && P.label ? `${P.n} tokens × ${P.trials} trials, ids ${P.id_lo.toLocaleString()}–${P.id_hi.toLocaleString()}, seed ${P.seed}` : 'copying protocol not reported (an older inside engine: 40 tokens × 2 trials)');
  // What the inside engine's model is now: as loaded, or cut. Every view that reads it carries the mark (PSIExp).
  const setState = (label) => {
    EX.setChanged(label);
    PSI.status($('lsState'), label ? `Model now: cut (${label}). It stays cut, in every view that uses the inside engine, until you press Restore the model.` : 'Model now: as loaded, nothing cut.', label ? 'warn' : '');
  };
  // While one request runs, the others wait (the Stop button of a sweep stays usable).
  const busy = (on, msg) => { BTNS.forEach((id) => { $(id).disabled = on; }); if (msg != null) stat(msg, on ? 'busy' : ''); };
  const qs = () => { const q = $('lsQ').value.trim(); const c = $('lsConcept').value.trim(), k = $('lsKeep').value.trim(); return [q, ...(c ? [`What is ${c}? Answer in two sentences.`] : []), ...(k ? [`What is ${k}? Answer in two sentences.`] : [])].filter(Boolean); };
  // The tests the user added in Understanding are scored too, apart from the headline count.
  const yours = () => EX.custom().map((t) => ({ what: t.what || 'your test', prompt: t.prompt, good: t.good, bad: t.bad }));
  $('lsFrac').addEventListener('input', () => { $('lsFracLabel').textContent = `${$('lsFrac').value}%`; });

  // A reply from other weights than the rows on screen: archive the rows (kept in this browser), forget the intact row
  // and the recorded answers, and start fresh.
  function track(cond) {
    const k = EX.key(cond); if (!k || k === R.model) return '';
    const had = R.cards.length || R.sweep;
    if (had) EX.archive(VIEW);
    R.cards = []; R.sweep = null; R.intact = null; R.ref = {}; R.model = k; showRef();
    if (had) { $('lsTable').querySelector('tbody').textContent = ''; drawSweep(); }
    return had ? 'The inside engine’s model changed: the earlier rows were archived, and the intact row and recorded answers belong to the old model, so they were cleared. ' : '';
  }
  const save = () => { if (!EX.keep(VIEW, R.model, { cards: R.cards, sweep: R.sweep, ref: R.ref }, R.cards[0] && R.cards[0]._model)) PSI.status($('lsSaved'), 'Results are not kept between sessions here (this browser blocks storage).', 'warn'); };
  // One request to the inside engine. On failure: the message area says how to fix it, the status line says what failed.
  async function call(args, label, failText) {
    busy(true, label || 'working…');
    try { const r = await psi.lesion({ ...args, questions: qs(), tests: yours() }); stat(''); return r; }
    catch (e) {
      if (['structural', 'weights', 'manual', 'concept'].includes(args.action)) setState('perhaps cut: the last cut failed partway, so restore to be sure');
      err(e); stat(`${failText || 'That step failed'}: ${PSI.explain(e).text}`, 'error'); e.shown = true; throw e;
    }
    finally { busy(false); }
  }
  // An experiment run: the nav dot shows it running; done only when it succeeds.
  async function experiment(fn) {
    PSI.setRunning(VIEW, true);
    try { await fn(); PSI.setDone(VIEW); } catch (e) { if (!e || !e.shown) { err(e); stat(`Stopped: ${PSI.explain(e).text}`, 'error'); } } finally { PSI.setRunning(VIEW, false); }
  }

  // The intact model's answers, one question per request (each request restores the model first).
  function showRef() { const q = qs().filter((x) => R.ref[x] != null); $('lsRef').textContent = q.map((x) => `${x} → ${R.ref[x]}`).join('\n'); $('lsRef').style.whiteSpace = 'pre-wrap'; }
  async function record(force) {
    const todo = qs().filter((q) => force || R.ref[q] == null); if (!todo.length) return true;
    const p = todo.length > 1 ? PSI.progress($('lsRefProg'), { total: todo.length, label: 'answer', track: false }) : null;
    try {
      for (let i = 0; i < todo.length; i++) {
        if (p && p.cancelled) break;
        if (p) p.step(i, 'recording the intact model’s answers');
        const r = await psi.lesion({ action: 'baseline', questions: [todo[i]] });
        track(r.conditions); Object.assign(R.ref, r.reference || {}); showRef(); setState(null); if (p) p.step(i + 1);   // a baseline restores the model first
      }
      if (p) { if (p.cancelled) { p.finish('stopped: some answers were not recorded', { markDone: false }); return false; } p.remove(); }
      return true;
    } catch (e) { if (p) p.fail('stopped by an error'); throw e; }
  }
  $('lsBase').addEventListener('click', async () => {
    busy(true, 'recording the intact model’s answers…');
    try { if (await record(true)) stat('reference recorded', 'ok'); else stat('Stopped before every answer was recorded.', 'warn'); }
    catch (e) { err(e); stat(`Recording the intact answers failed: ${PSI.explain(e).text}`, 'error'); }
    finally { busy(false); }
  });
  async function ensureBaseline() {
    busy(true, 'recording the intact answer first…');
    try { if (!(await record(false))) throw Object.assign(new Error('Stopped before the intact answers were recorded.'), { shown: true }); }
    catch (e) { if (!e.shown) { err(e); stat(`Recording the intact answer failed: ${PSI.explain(e).text}`, 'error'); e.shown = true; } else stat(e.message, 'warn'); throw e; }
    finally { busy(false); }
  }
  // Save the cut as a new model folder (the original is never changed); then, if asked, a GGUF of it through the converter.
  let savedDir = null;
  const saveStat = (text, kind) => PSI.status($('lsSaveStat'), text, kind);
  $('lsSave').addEventListener('click', async () => {
    busy(true, 'saving the cut model: reading the weight files and writing the copy…'); saveStat('');
    try {
      const r = await psi.lesion({ action: 'save', name: $('lsSaveName').value.trim(), questions: [] }); const s = r.saved; savedDir = s.folder;
      saveStat(`Saved: ${s.folder} (${s.gb} GB, ${s.changed_tensors} tensors changed, ${big(s.zeroed)} values zeroed, ${s.seconds}s). The model in memory stays cut.`, 'ok');
      $('lsGguf').classList.remove('hidden'); stat('');
      // Push it to Hugging Face with the account set in Settings > Hugging Face (prepare first, then Push now).
      if (window.PSIHf) PSIHf.mount($('lsHfBox'), { folder: s.folder, editable: true });
    } catch (e) { err(e); saveStat(`Not saved: ${PSI.explain(e).text}. The cut is still in force.`, 'error'); stat(''); }
    finally { busy(false); }
  });
  $('lsGguf').addEventListener('click', async () => {
    if (!savedDir) return;
    let j; try { j = await psi.convert({ action: 'start', folder: savedDir, precision: 'q8_0' }); } catch (e) { err(e); return saveStat(`No GGUF: ${PSI.explain(e).text}`, 'error'); }
    $('lsGguf').disabled = true; saveStat(`converting to ${j.out}…`, 'busy');
    const tick = async () => {
      let r; try { r = await psi.convert({ action: 'job', job: j.job }); } catch (e) { $('lsGguf').disabled = false; return saveStat(`Lost track of the conversion: ${PSI.explain(e).text}`, 'error'); }
      if (r.state === 'running') { saveStat(r.message, 'busy'); return setTimeout(tick, 3000); }
      $('lsGguf').disabled = false; saveStat(r.state === 'done' ? `GGUF ${r.message}. Pick it in Settings > Models.` : `GGUF ${r.message}`, r.state === 'done' ? 'ok' : 'error');
    };
    tick();
  });
  $('lsRestore').addEventListener('click', async () => { try { const r = await call({ action: 'restore' }, 'restoring…', 'Restoring the model failed'); stat(r.state.restored_tensors ? `restored ${r.state.restored_tensors} tensors from disk in ${r.state.restore_s}s` : 'restored', 'ok'); setState(null); } catch (_) { /* shown */ } });

  // What was cut, as achieved (counts), with what ranked it: "least/most active" means two different rankings.
  function stateLabel(s) {
    if (!s || s.kind === 'none') return 'intact';
    if (s.kind === 'structural') {
      const how = s.mode === 'random' ? `at random, seed ${s.seed != null ? s.seed : 'not recorded'}`
        : `${s.mode === 'most' ? 'most' : 'least'} first: neurons by ${s.neurons_by || 'activation on question 1'}, heads by ${s.heads_by || 'induction score'}${s.seed != null ? `, seed ${s.seed}` : ''}`;
      return `${big(s.neurons)} of ${big(s.of_neurons)} neurons (${pc(s.neurons, s.of_neurons)}), ${s.heads} of ${s.of_heads} heads (${pc(s.heads, s.of_heads)}) · ${how}`;
    }
    if (s.kind === 'weights') return `${pc(s.zeroed, s.of, 2)} of ${s.scope} weights zeroed (${big(s.zeroed)} of ${big(s.of)}; asked ${(s.frac * 100).toFixed(0)}%) · ${s.criterion === 'random' ? `at random, seed ${s.seed != null ? s.seed : 'not recorded'}` : s.criterion === 'small' ? 'smallest by magnitude first' : 'largest by magnitude first'}`;
    if (s.kind === 'manual') return `by hand: layers ${s.layers.join(',') || '-'}, ${s.heads} heads, ${big(s.neurons)} neurons`;
    if (s.kind === 'concept') return `concept "${s.concept}" (${s.k} neurons; kept "${s.keep}")`;
    return s.kind;
  }
  // A 0% cut (the sweep's first level) is the intact model: kept as the baseline every later row is read against.
  const isIntact = (s) => s && ((s.kind === 'structural' && !s.neurons && !s.heads) || (s.kind === 'weights' && !s.zeroed));
  // Colours relative to the intact row, never fixed cut-offs: green = about as intact, red = most of it gone.
  function judge(card) {
    const I = R.intact; if (!I || card === I) return { und: '', copy: '', text: '' };
    const und = I.understanding_passed ? (card.understanding_passed >= 0.9 * I.understanding_passed ? 'ok' : card.understanding_passed <= I.understanding_passed / 2 ? 'no' : '') : '';
    const copy = I.copy_nats > 0.5 ? (card.copy_nats >= 0.75 * I.copy_nats ? 'ok' : card.copy_nats < 0.25 * I.copy_nats ? 'no' : '') : '';
    const text = card.text_bits <= I.text_bits + 0.5 ? 'ok' : card.text_bits > I.text_bits + 2 ? 'no' : '';
    return { und, copy, text };
  }
  function addCard(c, cond) {
    c._stamp = EX.stamp(cond, { protocol: c.copy_protocol ? `copying ${c.copy_protocol.n}×${c.copy_protocol.trials}` : null, seed: c.state && c.state.seed, time: Date.now() });
    c._model = cond && cond.model;
    if (isIntact(c.state)) R.intact = c;
    R.cards.unshift(c); renderTable(); save();
  }
  function testRows(card) {
    const T = EX.tests();
    const list = card.tests || (card.margins || []).map((m, i) => ({ what: T[i] ? T[i].what : `test ${i + 1}`, margin: m, pass: m > 0, self_referential: !!(T[i] && T[i].self_referential) }));
    const I = R.intact && R.intact !== card ? R.intact : null;
    const iv = (x) => { if (!I) return null; const t = (I.tests || []).find((y) => y.id === x.id && y.what === x.what); return t ? t.margin : null; };
    const box = el('div', 'ls-drill');
    list.forEach((x) => {
      const d = el('div', x.pass ? 'ok' : 'no'), v = iv(x);
      d.textContent = `${x.pass ? 'passed' : 'not passed'} ${x.margin >= 0 ? '+' : ''}${x.margin} · ${x.what}${x.self_referential ? ' (self-referential: not in the headline)' : ''}${x.custom ? ' (yours: not in the headline)' : ''}${v != null ? ` · intact ${v >= 0 ? '+' : ''}${v}` : ''}`;
      box.appendChild(d);
    });
    box.appendChild(el('div', 'muted', `passed = log-odds of the understanding answer over the surface answer > 0 (fixed threshold); read the row against the intact row${I ? '' : ' (none scored yet on this model: sweep from 0% or cut 0%)'}.`));
    return box;
  }
  function renderTable() {
    const tb = $('lsTable').querySelector('tbody'); tb.textContent = '';
    const I = R.intact, vs = (card, v) => (I && card !== I ? ` (intact ${v(I)})` : '');
    R.cards.forEach((card) => {
      const j = judge(card), tr = el('tr'); const st = el('td', null, stateLabel(card.state)); if (card === I) st.appendChild(el('span', 'und-tag', 'the intact row'));
      tr.appendChild(st);
      // the headline count leaves out the self-referential test and your own tests: both shown apart; click for every margin
      const u = el('td', `${j.und} ls-und`); const btn = el('button', 'ghost ls-drill-btn', `${card.understanding_passed}/${card.understanding_of}`); btn.type = 'button'; btn.title = 'Show every test’s margin'; btn.setAttribute('aria-expanded', 'false');
      u.appendChild(btn); u.append(` · ${card.understanding_margin >= 0 ? '+' : ''}${card.understanding_margin}${vs(card, (x) => `${x.understanding_passed}/${x.understanding_of}`)}`);
      const sr = (card.self_referential || [])[0]; if (sr) u.appendChild(el('div', 'muted small', `self-referential: ${sr.pass ? 'passed' : 'not passed'} (${sr.margin >= 0 ? '+' : ''}${sr.margin})`));
      if (card.yours) u.appendChild(el('div', 'muted small', `yours: ${card.yours.passed}/${card.yours.of}`));
      tr.appendChild(u);
      const cp = el('td', j.copy, `${card.copy_nats} nats${vs(card, (x) => x.copy_nats)}`); cp.title = COPY_LABEL(card.copy_protocol);
      if (card.copy_nats < 0) cp.appendChild(el('span', 'ls-off', ' ▾ below 0: the second copy got harder (off the scale, not “no copying”)'));
      tr.appendChild(cp);
      tr.appendChild(el('td', j.text, `${card.text_bits} bits${vs(card, (x) => x.text_bits)}`));
      // retention needs the intact answer to that question: without it the cell says so, never a blank or 0
      const ret = card.answers.map((a) => (a.retention ? `${Math.round(a.retention.token_agreement * 100)}%` : 'not scored (no intact answer recorded)')).join(' · ');
      tr.appendChild(el('td', card.answers.some((a) => !a.retention) ? 'no' : null, ret || 'no question asked'));
      const ans = el('td'); card.answers.forEach((a) => { const d = el('div'); d.appendChild(el('span', 'muted small', a.question.slice(0, 40) + ' → ')); d.appendChild(document.createTextNode(a.answer.slice(0, 220) || '(nothing)')); ans.appendChild(d); }); tr.appendChild(ans);
      tr.appendChild(el('td', 'und-stamp', card._stamp || 'conditions not recorded'));
      tb.appendChild(tr);
      btn.addEventListener('click', () => {
        const open = tr.nextElementSibling && tr.nextElementSibling.classList.contains('ls-drill-row');
        if (open) { tr.nextElementSibling.remove(); btn.setAttribute('aria-expanded', 'false'); return; }
        const dr = el('tr', 'ls-drill-row'), td = el('td'); td.colSpan = 7; td.appendChild(testRows(card)); dr.appendChild(td); tr.after(dr); btn.setAttribute('aria-expanded', 'true');
      });
    });
    $('lsColours').textContent = I ? `Colours are read against the intact row (${I.understanding_passed}/${I.understanding_of} tests, ${I.copy_nats} nats copying, ${I.text_bits} bits): green = about as intact (≥ 90% of its passes, ≥ 75% of its copying, ≤ +0.5 bits), red = most of it gone (≤ half its passes, < 25% of its copying, > +2 bits).${I.copy_nats <= 0.5 ? ' The intact model barely copies, so copying is not coloured.' : ''}`
      : 'No colours yet: they are read against the intact row of this model, which a sweep scores first (its 0% level), or cut 0% to score it.';
  }
  const cutArgs = (frac, seed = 0) => {
    const what = $('lsWhat').value, how = $('lsHow').value;
    return what === 'structural' ? { action: 'structural', frac_neurons: frac, frac_heads: frac, mode: how, seed }
      : { action: 'weights', frac, scope: what.split('-')[1], criterion: how === 'random' ? 'random' : how === 'least' ? 'small' : 'large', seed };
  };

  $('lsApply').addEventListener('click', () => experiment(async () => {
    await ensureBaseline(); const frac = Number($('lsFrac').value) / 100;
    const r = await call(cutArgs(frac), `deleting ${Math.round(frac * 100)}% and scoring…`, `Deleting ${Math.round(frac * 100)}% failed`); const note = track(r.conditions); addCard(r, r.conditions);
    setState(frac ? stateLabel(r.state) : null); stat(`${note}${r.seconds}s · asked ${Math.round(frac * 100)}%, cut ${stateLabel(r.state).split(' · ')[0]} · the model stays cut until you restore it or cut again`, 'ok');
  }));
  // The sweep, one request per level and seed so Stop works between them; three seeds per level when the cut is random
  // (one when it is ranked, which is deterministic). The model is restored at the end, on Stop, and after an error.
  const LEVELS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7];
  $('lsSweep').addEventListener('click', () => experiment(async () => {
    await ensureBaseline();
    const kind = $('lsWhat').value === 'structural' ? 'structural' : 'weights', how = $('lsHow').value, seeds = how === 'random' ? [0, 1, 2] : [0];
    const plan = []; LEVELS.forEach((f) => (f ? seeds : [0]).forEach((sd) => plan.push([f, sd])));
    const runs = []; let secs = 0, note = '';
    const p = PSI.progress($('lsProg'), { total: plan.length, label: 'cut', track: false });
    busy(true, `sweeping 0 to 70% (${seeds.length} seed${seeds.length > 1 ? 's' : ''} per level)…`);
    let failed = null;
    try {
      for (let i = 0; i < plan.length; i++) {
        if (p.cancelled) break;
        const [f, sd] = plan[i];
        p.step(i, `deleting ${Math.round(f * 100)}%${seeds.length > 1 && f ? `, seed ${sd}` : ''} and scoring`);
        const c = await psi.lesion({ ...cutArgs(f, sd), questions: qs(), tests: yours() }); c.level = f; c.seed = sd; c._cond = c.conditions;
        if (!i) note = track(c.conditions);
        runs.push(c); secs += c.seconds || 0;
        p.step(i + 1);
      }
    } catch (e) { failed = e; }
    // Put the model back whatever happened.
    let restored = true;
    try { await psi.lesion({ action: 'restore', questions: [] }); setState(null); } catch (e) { restored = false; if (!failed) failed = e; setState('a sweep level, and restoring it failed'); }
    busy(false);
    if (runs.length) {
      runs.forEach((c) => { const cond = c._cond; delete c._cond; addCard(c, cond); });
      R.sweep = { kind, how, seeds, runs, partial: runs.length < plan.length, model: R.model }; save(); drawSweep();
    }
    const back = restored ? 'model restored' : 'the model could not be restored: press Restore the model';
    if (failed) {
      p.fail(`stopped by an error after ${runs.length} of ${plan.length} cuts`); err(failed);
      const at = plan[runs.length] || [0, 0];
      stat(`The sweep failed at ${Math.round(at[0] * 100)}%${seeds.length > 1 ? ` (seed ${at[1]})` : ''}: ${PSI.explain(failed).text} · ${back}`, 'error'); failed.shown = true; throw failed;
    }
    if (p.cancelled) { p.finish(`stopped after ${runs.length} of ${plan.length} cuts · ${back}`, { markDone: false }); stat(`Stopped early · ${back}`, 'warn'); throw Object.assign(new Error('stopped'), { shown: true }); }
    p.finish(`${plan.length} cuts scored · ${back}`, { markDone: false }); stat(`${note}${secs.toFixed(1)}s · ${back}`, 'ok');
  }));
  $('lsManual').addEventListener('click', () => experiment(async () => {
    await ensureBaseline();
    const layers = []; $('lsLayers').value.split(',').map((s) => s.trim()).filter(Boolean).forEach((s) => { const m = s.match(/^(\d+)\s*-\s*(\d+)$/); if (m) { for (let i = +m[1]; i <= +m[2]; i++) layers.push(i); } else if (/^\d+$/.test(s)) layers.push(+s); });
    const heads = $('lsHeads').value.split(',').map((s) => s.trim()).filter(Boolean).map((s) => s.split(':').map(Number)).filter((a) => a.length === 2 && a.every(Number.isFinite));
    const r = await call({ action: 'manual', layers, heads, neurons_per_layer: Number($('lsNpl').value) || 0 }, 'deleting by hand and scoring…', 'Deleting by hand failed'); const note = track(r.conditions); addCard(r, r.conditions); setState(stateLabel(r.state)); stat(`${note}${r.seconds}s · the model stays cut until you restore it or cut again`, 'ok');
  }));
  $('lsFind').addEventListener('click', async () => {
    try {
      const r = await call({ action: 'concept_find', concept: $('lsConcept').value.trim(), keep: $('lsKeep').value.trim(), k: Number($('lsK').value) || 300 }, 'contrasting activations…', 'Finding the concept’s neurons failed');
      EX.put($('lsFound'), `${r.k} `, PSI.term('neurons', 'MLP neuron'), ` that fire for "${r.concept}" more than for "${r.keep}" · by layer: ${Object.entries(r.by_layer).map(([l, n]) => `L${l}:${n}`).join(' ')} · strongest: ${r.neurons.slice(0, 6).map((n) => `L${n.layer}N${n.neuron} (${n.score})`).join(', ')}`);
    } catch (_) { /* shown */ }
  });
  $('lsConceptGo').addEventListener('click', () => experiment(async () => {
    await ensureBaseline(); const r = await call({ action: 'concept', concept: $('lsConcept').value.trim(), keep: $('lsKeep').value.trim(), k: Number($('lsK').value) || 300 }, 'deleting the concept and scoring…', 'Deleting the concept failed'); const note = track(r.conditions); addCard(r, r.conditions); setState(stateLabel(r.state));
    const [, c, k] = r.answers; stat(note + (c && k && c.retention && k.retention ? `${r.seconds}s · "${$('lsConcept').value}" retention ${Math.round(c.retention.token_agreement * 100)}% vs "${$('lsKeep').value}" ${Math.round(k.retention.token_agreement * 100)}%` : `${r.seconds}s`), 'ok');
  }));

  // ── the sweep chart: every series relative to the intact run (1 = as intact), a min–max band across seeds ─────
  function drawSweep() {
    const S = R.sweep, cv = $('lsSweepCanvas');
    if (!S || !S.runs || !S.runs.length) { $('lsSweepWhy').textContent = ''; return EX.empty(cv, 200, 'Nothing swept yet: press Sweep 0–70%.'); }
    const r = Math.min(2, devicePixelRatio || 1); cv.style.height = '220px'; if (!cv.clientWidth) return; cv.width = cv.clientWidth * r; cv.height = 220 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, left = 40 * r, right = W - 20 * r, top = 22 * r, bot = H - 30 * r;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const X = (f) => left + (f / 0.7) * (right - left), lo = -0.2, hi = 1.2, Y = (v) => bot - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (bot - top);
    const base = S.runs.find((p) => p.level === 0);
    const levels = [...new Set(S.runs.map((p) => p.level))].sort((a, b) => a - b), at = (f) => S.runs.filter((p) => p.level === f);
    // null = not measured (a gap, never a 0); outside -0.2…1.2 = off this scale: an amber triangle at the edge, counted
    const series = [
      { f: (p) => (base && base.understanding_passed ? p.understanding_passed / base.understanding_passed : null), dash: [], w: 2, name: 'tests passed' },
      { f: (p) => (base && base.copy_nats > 0.5 ? p.copy_nats / base.copy_nats : null), dash: [6, 4], w: 1.4, name: 'copying' },
      { f: (p) => (p.answers[0] && p.answers[0].retention ? p.answers[0].retention.token_agreement : null), dash: [2, 3], w: 1.6, name: 'retention of its answer' },
      { f: (p) => (base ? 1 - (p.text_bits - base.text_bits) / 6 : null), dash: [6, 2, 2, 2], w: 1.2, name: 'ordinary text (0 = +6 bits)' },
    ];
    [0, 1].forEach((v) => { g.strokeStyle = 'rgba(242,241,238,0.22)'; g.setLineDash([3 * r, 3 * r]); g.lineWidth = r; g.beginPath(); g.moveTo(left, Y(v)); g.lineTo(right, Y(v)); g.stroke(); g.setLineDash([]); });
    g.fillStyle = '#75726c'; g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'right'; g.fillText('as intact', left - 3 * r, Y(1) + 3 * r); g.fillText('0', left - 3 * r, Y(0) + 3 * r);
    const off = { clip: 0, missing: 0 }, notDrawn = [];
    series.forEach((s) => {
      if (base && s.f(base) == null) { notDrawn.push(s.name); return; }
      const pts = levels.map((f) => { const v = at(f).map(s.f).filter((x) => x != null && isFinite(x)); if (at(f).length > v.length) off.missing += at(f).length - v.length; return v.length ? { f, mean: v.reduce((a, b) => a + b, 0) / v.length, min: Math.min(...v), max: Math.max(...v), vals: v } : { f, mean: null }; });
      // the min–max band across seeds
      const band = pts.filter((q) => q.mean != null);
      if (band.some((q) => q.max > q.min)) { g.fillStyle = 'rgba(242,241,238,0.08)'; g.beginPath(); band.forEach((q, i) => (i ? g.lineTo(X(q.f), Y(q.max)) : g.moveTo(X(q.f), Y(q.max)))); for (let i = band.length - 1; i >= 0; i--) g.lineTo(X(band[i].f), Y(band[i].min)); g.closePath(); g.fill(); }
      g.strokeStyle = '#f2f1ee'; g.lineWidth = s.w * r; g.setLineDash(s.dash.map((d) => d * r)); g.beginPath(); let on = false;
      pts.forEach((q) => { if (q.mean == null) { on = false; return; } const x = X(q.f), y = Y(q.mean); on ? g.lineTo(x, y) : g.moveTo(x, y); on = true; });
      g.stroke(); g.setLineDash([]);
      pts.forEach((q) => (q.vals || []).forEach((v) => { if (v >= lo && v <= hi) return; off.clip++; const x = X(q.f), y = Y(v), d = v > hi ? -1 : 1; g.fillStyle = '#ffc36b'; g.beginPath(); g.moveTo(x, y + d * 7 * r); g.lineTo(x - 4 * r, y); g.lineTo(x + 4 * r, y); g.closePath(); g.fill(); }));
    });
    // The tipping point, per seed: the first level at which half the intact passes are lost or the answer is gone
    // (retention < 30%); across seeds, an interval. A model that passes few tests even intact is judged by its answer.
    const passed0 = base ? base.understanding_passed : 0, of0 = base ? base.understanding_of : 0, useTests = base && passed0 >= 0.75 * of0;
    const tipped = (p) => (useTests && p.understanding_passed < passed0 / 2) || (p.answers[0] && p.answers[0].retention && p.answers[0].retention.token_agreement < 0.3);
    const seedTips = S.seeds.map((sd) => { const mine = S.runs.filter((p) => p.level > 0 && p.seed === sd).sort((a, b) => a.level - b.level); const i = mine.findIndex(tipped); return { seed: sd, at: i < 0 ? null : mine[i].level, before: i < 0 ? null : (i ? mine[i - 1].level : 0), scored: mine.length }; });
    const hit = seedTips.filter((t) => t.at != null), lo2 = hit.length ? Math.min(...hit.map((t) => t.before)) : null, hi2 = hit.length ? Math.max(...hit.map((t) => t.at)) : null;
    if (hit.length) {
      g.fillStyle = 'rgba(255,195,107,0.15)'; g.fillRect(X(lo2), top, X(hi2) - X(lo2), bot - top); g.fillStyle = '#ffc36b'; g.font = `${10 * r}px "IBM Plex Mono", monospace`; g.textAlign = 'left';
      g.fillText(`tipping point between ${Math.round(lo2 * 100)}% and ${Math.round(hi2 * 100)}%`, Math.min(X(lo2) + 4 * r, right - 210 * r), top + 10 * r);
    }
    g.font = `${9 * r}px "IBM Plex Mono", monospace`; g.fillStyle = '#75726c'; g.textAlign = 'center'; [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].forEach((f) => g.fillText(`${Math.round(f * 100)}%`, X(f), H - 14 * r));
    g.fillText('share asked (the achieved shares are in the scorecard)', (left + right) / 2, H - 3 * r);
    g.textAlign = 'left'; g.fillStyle = '#a9a69f'; g.fillText(series.filter((s) => !notDrawn.includes(s.name)).map((s) => `${['solid', 'dashed', 'dotted', 'dash-dot'][series.indexOf(s)]}: ${s.name}`).join('   ') + (S.seeds.length > 1 ? '   band: min–max over seeds' : ''), left + 4 * r, 12 * r);
    const lastLevel = levels[levels.length - 1];
    const seedsTxt = S.seeds.length > 1 ? `${S.seeds.length} seeds per level (${S.seeds.join(', ')}), mean line, min–max band. ` : 'One run per level: a ranked cut is deterministic. ';
    const tipTxt = !base ? 'No intact (0%) run in this sweep, so nothing is read against it. '
      : hit.length ? `Tipping point between ${Math.round(lo2 * 100)}% and ${Math.round(hi2 * 100)}% removed (${seedTips.map((t) => `seed ${t.seed}: ${t.at == null ? 'held to the last level scored' : `held at ${Math.round(t.before * 100)}%, tipped at ${Math.round(t.at * 100)}%`}`).join('; ')}).${hit.length < seedTips.length ? ' Not every seed tipped: the interval is a lower bound for those.' : ''} `
      : S.partial ? 'No tipping point in the levels scored. ' : 'No tipping point within 70%: the model still shows the signs after the largest cut. ';
    const why = seedsTxt + tipTxt
      + (useTests ? 'Tipped = half the intact passes lost, or under 30% of its own answer kept. ' : base ? `Intact, this model passes ${passed0} of ${of0} tests, too few to show a fall, so the tipping point is read from its answer alone. ` : '')
      + (notDrawn.length ? `Not drawn: ${notDrawn.join(', ')} (the intact run gives nothing to read it against). ` : '')
      + (off.missing ? `${off.missing} value${off.missing > 1 ? 's were' : ' was'} not measured (retention needs the intact answer): gaps, not zeros. ` : '')
      + (off.clip ? `${off.clip} value${off.clip > 1 ? 's lie' : ' lies'} off this scale (amber triangles at the edge; negative copying means the second copy got harder, not zero). ` : '')
      + (S.partial ? `Stopped at ${Math.round(lastLevel * 100)}%, so the sweep is incomplete. ` : '')
      + 'Least / most active ranks neurons by their activation on question 1 and heads by induction score: two rankings, both shown in each row.';
    $('lsSweepWhy').textContent = why;
    PSI.canvasLabel(cv, `The scorecard, relative to the intact model, as 0 to ${Math.round(lastLevel * 100)}% of the model is removed. ${why}`);
  }
  // The push panel (ui/hf-push.js): any saved cut model folder; a save fills in its folder.
  if (window.PSIHf) PSIHf.mount($('lsHfBox'), { editable: true });
  addEventListener('resize', () => { if (!$('tab-lesion').classList.contains('hidden')) drawSweep(); });
  document.querySelector('[data-tab=lesion]').addEventListener('click', () => setTimeout(drawSweep, 0));
  // Anatomy's "delete what is lit" restores the model when it ends: the line follows it.
  document.addEventListener('psi-changed', (e) => { if (!e.detail.label) PSI.status($('lsState'), 'Model now: as loaded, nothing cut.', ''); });
  PSI.status($('lsState'), 'Model now: as loaded as far as this window knows (a cut from an earlier window stays in a running engine: Restore the model to be sure).', '');
  // The inside engine's model was switched in Settings: the rows, the intact row and the recorded answers belong to the
  // old weights. Archive them and clear R.intact, so no later row is coloured against another model's intact row.
  document.addEventListener('psi-inside-model', (e) => {
    if (!R.model || !EX.other(R.model)) return;
    if (R.cards.length || R.sweep) EX.archive(VIEW);
    R.cards = []; R.sweep = null; R.intact = null; R.ref = {}; R.model = ''; showRef(); renderTable(); drawSweep();
    PSI.status($('lsSaved'), `The inside engine’s model changed (now ${e.detail.to}): the earlier rows were archived and the intact row cleared; record the intact answer again.`, 'warn');
  });
  // Rows kept from an earlier session, on the model they came from (a reply from other weights archives them).
  const kept = EX.saved(VIEW);
  if (kept && kept.data && (kept.data.cards || []).length) {
    R.model = kept.key; R.cards = kept.data.cards; R.sweep = kept.data.sweep && kept.data.sweep.runs ? kept.data.sweep : null; R.ref = kept.data.ref || {};
    R.intact = R.cards.find((c) => isIntact(c.state)) || null; renderTable(); showRef();
    PSI.status($('lsSaved'), `Shown: rows kept from ${new Date(kept.saved).toLocaleString()} (${kept.model || 'model not recorded'}). A reply from other weights archives them and starts fresh.`, '');
  } else renderTable();
  drawSweep();
})();
