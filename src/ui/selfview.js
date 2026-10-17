// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Self-portrait: one page per open model, built only from stored results (window.PSI_SELF, from
// src/ui/self-data.js, bundled with the app). Written so an AI agent, or a person, can read what PSI knows about that model's
// insides: every claim beside its baseline and noise, "not measured" said in words, the subject always named.
(() => {
  'use strict';
  const D = window.PSI_SELF;
  const root = document.getElementById('selfRoot');
  if (!root) return;
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const add = (p, ...kids) => { kids.forEach((k) => { if (k != null) p.append(k); }); return p; };
  const NM = 'not measured';
  const num = (v) => {
    if (v == null || (typeof v === 'number' && !Number.isFinite(v))) return null;
    if (typeof v !== 'number') return String(v);
    const a = Math.abs(v); return a >= 100 ? v.toFixed(1) : a >= 10 ? v.toFixed(2) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  };
  // A value cell: a number, or why there is none (never blank, never 0).
  const val = (v, why) => { const s = num(v); if (s != null) return el('span', 'sp-num', s); const n = el('span', 'sp-nm', NM); if (why) n.title = why; return n; };
  const ciTxt = (ci) => (ci && ci[0] != null && ci[1] != null ? `[${num(ci[0])}, ${num(ci[1])}]` : null);
  const VCLS = { learned: 'sp-v-learned', architecture: 'sp-v-arch', 'within noise': 'sp-v-noise', 'below its twin': 'sp-v-below', 'not decidable': 'sp-v-nd', 'not beyond its twin': 'sp-v-noise' };
  const verdict = (v, why) => { const s = el('span', `sp-v ${VCLS[v] || ''}`, v || NM); if (why) s.title = why; return s; };
  const k = (txt) => el('p', 'k mt-22', txt);
  const note = (txt, cls = 'muted small') => el('p', cls, txt);

  if (!D || !D.models || !D.order || !D.order.length) {
    add(root, el('div', 'view-head'), note('No self-portrait data: src/ui/self-data.js is missing or empty. It ships with the app: reinstall or restore that file.', 'sp-banner'));
    return;
  }

  // ── header ────────────────────────────────────────────────────────────────
  const head = el('div', 'view-head');
  add(head, el('p', 'kicker', 'Look inside · Self-portrait'), el('h1', null, 'What PSI knows about one open model'),
    el('p', 'view-sub', 'Built only from stored results. Each number sits beside its untrained twin (same architecture, random weights) or another stated baseline, with its noise and n. Written to be read by an AI agent as well as a person: it describes the named model, not the reader.'));
  root.appendChild(head);

  const bar = el('div', 'row sp-bar');
  const lab = el('label', 'small muted', 'model '); const sel = el('select'); sel.id = 'spModel'; lab.appendChild(sel);
  const copy = el('button', 'ghost', 'Copy as JSON for an agent'); copy.type = 'button'; copy.id = 'spCopy';
  const copied = el('span', 'small muted'); copied.setAttribute('role', 'status');
  const help = el('a', 'small', 'how to read this'); help.href = 'help.html#self-portrait'; help.target = '_blank';
  add(bar, lab, copy, copied, help); root.appendChild(bar);
  const groups = {};
  D.order.forEach((t) => { const m = D.models[t]; (groups[m.lab] = groups[m.lab] || []).push(t); });
  Object.entries(groups).forEach(([g, ts]) => {
    const og = el('optgroup'); og.label = g;
    ts.forEach((t) => { const o = el('option', null, D.models[t].name + (t.includes('@') ? ` (${t.split('@')[1]})` : '')); o.value = t; og.appendChild(o); });
    sel.appendChild(og);
  });
  let start = 'mistral-7b';
  try { const s = localStorage.getItem('psi.self.model'); if (s && D.models[s]) start = s; } catch (_) { /* per-viewer convenience only */ }
  if (!D.models[start]) start = D.order[0];
  sel.value = start;

  const body = el('div', 'sp-body'); root.appendChild(body);
  const gen = note(`Built ${D.generated} by ${D.builder} from ${D.sources.ladder_results} stored ladder results (newest file ${D.sources.ladder_newest}); rules v${D.rule_version}.`, 'muted small sp-gen');
  root.appendChild(gen);

  // ── tables ────────────────────────────────────────────────────────────────
  function table(cols, rows, cls) {
    const wrap = el('div', 'scroll sp-scroll'); const tb = el('table', `sp-table ${cls || ''}`);
    const thead = el('thead'); const tr = el('tr'); cols.forEach((c) => tr.appendChild(el('th', null, c))); thead.appendChild(tr); tb.appendChild(thead);
    const tbody = el('tbody');
    rows.forEach((r) => { const row = el('tr'); r.forEach((c) => { const td = el('td'); if (c instanceof Node) td.appendChild(c); else td.textContent = c == null ? '' : c; row.appendChild(td); }); tbody.appendChild(row); });
    tb.appendChild(tbody); wrap.appendChild(tb); return wrap;
  }
  const measureCell = (r, extra) => {
    const d = el('div'); d.appendChild(el('span', 'sp-m', r.measure));
    const bits = [];
    if (r.p) bits.push(r.p); if (r.p_bh) bits.push(r.p_bh); if (r.n) bits.push(`n ${r.n}`);
    if (r.noise_note) bits.push(r.noise_note); else if (r.noise === 'none') bits.push('no noise estimate stored');
    if (r.near_bar) bits.push('near the bar');
    if (extra) bits.push(extra);
    if (bits.length) d.appendChild(el('small', 'sp-sub', bits.join(' · ')));
    if (r.question) d.title = r.question + (r.rule ? `\nRule: ${r.rule}` : '');
    return d;
  };
  const diffCell = (r) => { const d = el('span'); d.appendChild(val(r.diff, r.twin == null ? 'no twin value' : null)); const c = ciTxt(r.ci); if (c) d.appendChild(el('small', 'sp-sub', `95% CI ${c}`)); return d; };
  const verdictCell = (r) => { const d = el('div'); d.appendChild(verdict(r.verdict, r.why)); if (r.why) d.appendChild(el('small', 'sp-sub', r.why)); return d; };

  // ── sections ──────────────────────────────────────────────────────────────
  function sTraining(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('1 · what training gave it'));
    const T = m.training;
    s.appendChild(note(`Twin: ${m.twin || 'none'} (${m.twin_kind}${m.twin && !m.twin_same_architecture ? ', another architecture: not a baseline' : ''}). ${T.conditions}.`));
    const c = T.counts || {}; const efp = T.expected_false_passes || {};
    s.appendChild(note(`Storage ladder (concept "red"), rules v${T.rule_version}: ${efp.n_pass ?? NM} of 9 levels beyond the twin; chance alone would give about ${num(efp.expected) ?? NM} of those passes at their own p; ${efp.n_not_decidable ?? 0} not decidable.`, 'small'));
    s.appendChild(table(['measure', 'twin', 'trained', 'difference', 'verdict'],
      T.ladder.map((r) => [measureCell(r, r.unit), val(r.twin, r.why), val(r.trained), diffCell(r), verdictCell(r)]), 'sp-train'));
    if (T.atlas.length) {
      const st = T.atlas[0].stale; if (st) s.appendChild(note(st, 'sp-stale'));
      s.appendChild(note(`Concept atlas, middle layers: share of each language's words placed at their own concept (p@1) among ${m.falls_out.n_concepts} concepts (chance ${num(m.falls_out.chance)}), trained vs twin. The twin scores what shared spelling gives with no learning. Atlas file ${m.falls_out.date || 'undated'}.`, 'small'));
      s.appendChild(table(['language', 'twin', 'trained', 'learned (difference)', 'verdict'],
        T.atlas.map((r) => [measureCell({ ...r, measure: r.lang, noise_note: null }, `${num(r.tokens_per_word)} tokens/word · surface overlap ${num(r.surface_overlap)}`), val(r.twin), val(r.trained), diffCell(r), verdictCell({ ...r, why: null })]), 'sp-atlas'));
      s.appendChild(note('CIs on the atlas are approximate (binomial, computed by the builder; the run stored none). Hover a verdict for its reason.'));
    } else {
      s.appendChild(note(`Concept atlas: ${NM}. ${m.falls_out.why || ''}`, 'small sp-nmline'));
    }
    const l2 = T.level2_status || {};
    if (T.level2.length) {
      s.appendChild(note(`Level 2 (per-layer profile), twin ${l2.twin || 'none'}:`, 'small'));
      s.appendChild(table(['measure', 'twin', 'trained', 'difference', 'verdict'], T.level2.map((r) => [measureCell(r), val(r.twin), val(r.trained), diffCell(r), verdictCell(r)])));
    } else s.appendChild(note(`Level 2 (per-layer profile): ${l2.why || 'level 2 not yet measured'}.`, 'small sp-nmline'));
    return s;
  }

  function sFallsOut(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('2 · where it falls out'));
    const F = m.falls_out;
    if (!F.measured) { s.appendChild(note(`${NM}: ${F.why}.`, 'small sp-nmline')); return s; }
    if (F.stale) s.appendChild(note(F.stale, 'sp-stale'));
    s.appendChild(note(`Weak languages: ${F.weak_rule}.`, 'small'));
    if (F.weak_languages.length) {
      s.appendChild(table(['language', 'learned', 'trained / twin', 'tokens per word', 'surface overlap', 'no-overlap words: trained / twin (n)'],
        F.weak_languages.map((w) => [w.lang, add(el('span'), val(w.learned), ciTxt(w.ci) ? el('small', 'sp-sub', `≈95% CI ${ciTxt(w.ci)}`) : null), `${num(w.trained) ?? NM} / ${num(w.twin) ?? NM}`,
          val(w.tokens_per_word), val(w.surface_overlap), `${num(w.trained_no_overlap) ?? NM} / ${num(w.twin_no_overlap) ?? NM} (${w.n_no_overlap ?? NM})`])));
    } else s.appendChild(note(`None: every measured language gains at least ${0.15} over the twin.`, 'small'));
    s.appendChild(note(`Anomalies: ${F.anomaly_rule}. ${F.n_anomalies} remain; ${F.also_in_twin ?? NM} more also appear in the twin and are left out. ${F.labels_note}`, 'small'));
    Object.entries(F.anomalies).forEach(([g, list]) => {
      const d = el('details', 'sp-det'); const sm = el('summary'); sm.append(el('b', null, g), ` · ${list.length}`); d.appendChild(sm);
      const shown = list.slice(0, 60);
      d.appendChild(table(['concept / language', 'word', 'rank', 'sits with', 'tokens', 'reason'],
        shown.map((a) => [`${a.concept} / ${a.lang}`, a.word, String(a.rank), a.nearest_concept || NM, a.n_tokens == null ? NM : String(a.n_tokens), add(el('span'), a.reason, a.label === 're-derived' ? el('small', 'sp-sub', 'label re-derived from the stored word') : null)])));
      if (list.length > shown.length) d.appendChild(note(`${list.length - shown.length} more in the JSON export.`));
      s.appendChild(d);
    });
    s.appendChild(note(F.instrument_noise.length ? `Instrument noise, not the model's: ${F.instrument_noise.length} words identical to the English (${F.instrument_noise.map((a) => `${a.concept}/${a.lang}`).join(', ')}). ${F.instrument_noise_note}.` : `Instrument noise: no dictionary-error words among this model's anomalies (${F.instrument_noise_note}).`, 'small'));
    return s;
  }

  function sSays(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('3 · what it says vs what it holds'));
    const M = m.says_vs_holds.m8, K = m.says_vs_holds.selfknow;
    s.appendChild(note(`M8, read here as ${M.name} (not "${M.not_called}"): asked to rate ${M.concept}, does what the model says agree with where its own state for red points? AUC null SD ${num(M.auc_null_sd)}; chance AUC ${M.chance_auc}.`, 'small'));
    s.appendChild(table(['', 'said (AUC strong vs weak)', 'held (AUC from its state)', 'said ↔ held (rho)'], [
      [`${m.name}`, val(M.said_auc), val(M.held_auc), add(el('span'), val(M.rho_said_held), M.p ? el('small', 'sp-sub', M.p) : null)],
      [`untrained twin${M.twin_tag ? ` (${M.twin_tag})` : ''}`, val(M.twin_said_auc, 'no same-architecture twin'), val(M.twin_held_auc, 'no same-architecture twin'), val(M.twin_rho, 'no same-architecture twin')],
    ]));
    s.appendChild(add(el('p', 'small'), verdict(M.verdict, M.why), ' ', M.why || ''));
    if (M.says_without_tracking) s.appendChild(note(`Flag: ${M.says_without_tracking_note}.`, 'sp-flag'));
    if (K.measured) {
      s.appendChild(note(`Self-knowledge probe (run "${K.run}", ${K.n_items} trivia items, accuracy ${num(K.accuracy)}, measured ${K.date || 'undated'}): does its internal state predict whether its own answer is right? Baseline: the model's own top-1 probability. AUC null SD ${num(K.auc_null_sd)}.`, 'small'));
      s.appendChild(table(['signal', 'AUC', 'baseline'], [
        [`probe on its state (layer ${K.probe_layer})`, val(K.probe_auc), `own top-1 confidence ${num(K.confidence_auc)}`],
        ['probe at layer 0 (the words alone)', val(K.layer0_auc), 'chance 0.5'],
        ['verbal: "do you know the answer?"', val(K.verbal_auc, 'the verbal question was not asked in this run'), 'chance 0.5'],
      ]));
      s.appendChild(note(K.probe_vs_confidence + '.', 'sp-flag'));
      if (K.verbal) s.appendChild(note(K.verbal + (K.verbal_near_chance ? ': near chance.' : '.'), 'small'));
    } else s.appendChild(note(`Self-knowledge probe: ${NM}. ${K.why}.`, 'small sp-nmline'));
    return s;
  }

  function sRests(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('4 · what rests on what'));
    const X = m.rests_on;
    if (!X.measured) s.appendChild(note(`For this model: ${NM}. ${X.why}.`, 'small sp-nmline'));
    else {
      s.appendChild(note(`${X.note} Twin: ${X.twin}.`, 'small'));
      X.cells.forEach((c) => {
        const d = el('details', 'sp-det'); const sm = el('summary');
        const lost = c.lost_only_here.length ? `loses ${c.lost_only_here.join(', ')} (control keeps them)` : 'no level lost beyond what the control loses';
        sm.append(el('b', null, c.name), ` · ${c.change} · ${lost}`); d.appendChild(sm);
        d.appendChild(note(`Control: ${c.control_change || NM}.${c.lost_in_control_too.length ? ` Lost in the control too (not specific): ${c.lost_in_control_too.join(', ')}.` : ''}`));
        const pv = (v, p) => add(el('span', p ? 'sp-pass' : 'sp-fail'), val(v), p == null ? '' : p ? ' ✔' : ' ·');
        d.appendChild(table(['level', 'unchanged', 'changed', 'control change'], c.levels.map((l) => [l.level, pv(l.unchanged, l.unchanged_pass), pv(l.changed, l.changed_pass), l.control == null ? val(null, 'no control run') : pv(l.control, l.control_pass)])));
        s.appendChild(d);
      });
    }
    const C = D.cross_family;
    if (C) {
      s.appendChild(add(el('p', 'small'), el('span', 'sp-tag', C.label), ' ', C.finding, ' ', C.status + '.'));
      s.appendChild(table(['model', 'M3: unchanged → removed (control)', 'M3', 'M7: unchanged → removed (control)', 'M7'],
        C.families.map((r) => [r.model, `${num(r.M3[0])} → ${num(r.M3[1])} (${num(r.M3[2])}); twin ${num(r.M3_twin) ?? NM}`, r.M3_status, `${num(r.M7[0])} → ${num(r.M7[1])} (${num(r.M7[2])}); twin ${num(r.M7_twin) ?? NM}`, r.M7_status])));
      s.appendChild(note(`${C.rule}. A pattern across families is a hypothesis for other models, not a measurement of them.`));
    }
    return s;
  }

  function sNot(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('5 · not measured here'));
    const ul = el('ul', 'sp-list'); m.not_measured.forEach(([a, b]) => { const li = el('li'); li.append(el('b', null, a), `: ${b}.`); ul.appendChild(li); });
    s.appendChild(ul); return s;
  }

  function sTransfers(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('6 · what transfers'));
    const ul = el('ul', 'sp-list');
    [['may transfer', m.transfers.may_transfer], ['does not transfer', m.transfers.does_not_transfer], ['the reader', m.transfers.reader]].forEach(([a, b]) => { const li = el('li'); li.append(el('b', null, a), `: ${b}`); ul.appendChild(li); });
    s.appendChild(ul); return s;
  }

  function strip(passes, dec, steps) {
    const d = el('span', 'sp-strip'); d.setAttribute('role', 'img');
    d.setAttribute('aria-label', passes.map((p, i) => `${steps[i]}: ${!dec[i] ? 'not decidable' : p ? 'passes' : 'fails'}`).join(', '));
    passes.forEach((p, i) => { const c = el('i', !dec[i] ? 'nd' : p ? 'on' : ''); c.title = `step ${steps[i]}: ${!dec[i] ? 'not decidable' : p ? 'passes' : 'fails'}`; d.appendChild(c); });
    return d;
  }
  function timelineTable(T) {
    return table(['level', `checkpoints (${T.steps[0]} → ${T.steps[T.steps.length - 1]})`, 'first passed', 'holds from', 'flips'],
      T.levels.map((l) => [l.level, strip(l.passes, l.decidable, T.steps),
        l.never_decidable ? val(null, 'not decidable at any checkpoint') : l.first_passed == null ? 'never' : add(el('span'), `by ${l.bracket.by}`, el('small', 'sp-sub', `after ${l.bracket.after}${l.bracket.bracket_only ? ' · bracket only' : ''}`)),
        l.never_decidable ? '' : l.holds_from || 'does not hold', String(l.flips)]), 'sp-tl');
  }
  function sFormation(m) {
    const s = el('section', 'sp-sec'); s.appendChild(k('formation timeline'));
    const T = m.formation && D.families[m.formation];
    if (!T) { s.appendChild(note(`${NM}: ${m.formation_note}.`, 'small sp-nmline')); return s; }
    s.appendChild(note(`${T.name}, ${T.steps.length} stored checkpoints, each against the untrained twin ${T.twin}. Filled: passes; empty: fails; hatched: not decidable. ${T.bracket_rule}. ${T.note}.`, 'small'));
    s.appendChild(timelineTable(T));
    const sibs = Object.keys(D.families).filter((f) => f !== m.formation && f.split('-')[0] === m.formation.split('-')[0]);
    sibs.forEach((f) => { const d = el('details', 'sp-det'); const sm = el('summary'); sm.append(el('b', null, D.families[f].name), ` · ${D.families[f].steps.length} checkpoints (a separate training run)`); d.appendChild(sm); d.appendChild(timelineTable(D.families[f])); s.appendChild(d); });
    return s;
  }

  // ── render ────────────────────────────────────────────────────────────────
  function render(t) {
    const m = D.models[t]; body.replaceChildren();
    const subj = el('div', 'sp-subject');
    add(subj, el('p', 'sp-subject-line', `Subject: ${m.subject}.`),
      note(`Precision: ${m.precision.value || NM} (${m.precision.source}). Date: ${m.date.value || NM} (${m.date.source}). ${m.layers ?? NM} layers, ${m.neurons_per_layer ?? NM} neurons per layer. Weights: ${m.path ? `${m.path} (${m.hub ? 'Hugging Face' : 'in the models folder'})` : NM}.`));
    body.appendChild(subj);
    if (D.stale && D.stale.length) {
      const st = el('details', 'sp-stale-all'); const sm = el('summary', null, `What is stale or missing in the stored results (${D.stale.length})`); st.appendChild(sm);
      const ul = el('ul', 'sp-list'); D.stale.forEach((x) => ul.appendChild(el('li', null, x))); st.appendChild(ul); body.appendChild(st);
    }
    [sTraining, sFallsOut, sSays, sRests, sNot, sTransfers, sFormation].forEach((f) => {
      try { body.appendChild(f(m)); } catch (e) { body.appendChild(note(`This section could not be drawn: ${e.message}`, 'sp-stale')); console.error(e); }
    });
  }

  // ── export: the same fields, plus the analogue block ─────────────────────
  function exportJSON(t) {
    const m = D.models[t];
    return {
      about: 'PSI self-portrait of one open model, from stored results only (Parameter Space Inspector).',
      subject: m.subject, model: { tag: m.tag, name: m.name, family: m.family, path: m.path, layers: m.layers, neurons_per_layer: m.neurons_per_layer },
      precision: m.precision, date: m.date, twin: { tag: m.twin, kind: m.twin_kind, same_architecture: m.twin_same_architecture },
      what_training_gave_it: m.training, where_it_falls_out: m.falls_out, what_it_says_vs_what_it_holds: m.says_vs_holds,
      what_rests_on_what: { this_model: m.rests_on, cross_family: D.cross_family },
      formation_timeline: m.formation ? D.families[m.formation] : { measured: false, why: m.formation_note },
      not_measured_here: m.not_measured.map(([a, b]) => ({ item: a, why: b })), what_transfers: m.transfers,
      stale_or_missing: D.stale, analogue: m.analogue, generated: D.generated, rule_version: D.rule_version,
    };
  }
  async function doCopy() {
    const txt = JSON.stringify(exportJSON(sel.value), null, 1);
    try { await navigator.clipboard.writeText(txt); copied.textContent = `copied ${Math.round(txt.length / 1024)} KB of JSON`; }
    catch (_) {
      const ta = el('textarea'); ta.value = txt; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select();
      let ok = false; try { ok = document.execCommand('copy'); } catch (__) { ok = false; }
      ta.remove(); copied.textContent = ok ? `copied ${Math.round(txt.length / 1024)} KB of JSON` : 'copy failed: the clipboard is not available here';
    }
  }
  copy.addEventListener('click', doCopy);
  sel.addEventListener('change', () => { picked = true; copied.textContent = ''; try { localStorage.setItem('psi.self.model', sel.value); } catch (_) { /* ignore */ } render(sel.value); });
  window.PSISelf = { exportJSON, render };
  render(sel.value);
  // Opens on the loaded model when the list has it: matched on its file name or the header's name, the longest match
  // winning (Qwen3-4B-Base before Qwen3-4B). A model picked here in this session stays picked.
  const squash = (x) => String(x || '').toLowerCase().replace(/[^a-z0-9.]+/g, '');
  function loadedTag() {
    const st = window.PSI && PSI.state, hay = [st && st.model && String(st.model).split(/[\\/]/).pop(), st && st.model && window.PSIModelMeta && PSIModelMeta.name].filter(Boolean).map(squash);
    let best = null, bl = 0;
    if (hay.length) D.order.forEach((t) => { const m = D.models[t]; [m.name, m.family, t.split('@')[0]].map(squash).forEach((k) => {
      if (k.length > bl && hay.some((h) => { const i = h.indexOf(k); return i >= 0 && !/[0-9]/.test(h[i + k.length] || ''); })) { best = t; bl = k.length; }
    }); });
    return best;
  }
  let picked = false, lastModel;
  const follow = () => { if (picked) return; const t = loadedTag(); if (t && t !== sel.value) { sel.value = t; render(t); } };
  if (window.PSI && PSI.onState) PSI.onState((st) => { if (st.model !== lastModel) { lastModel = st.model; follow(); } });
  document.addEventListener('psi-model-meta', follow);
})();
