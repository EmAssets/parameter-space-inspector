// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Hybrid models view (beta, src/hybrid-api.js): define hybrids (the loaded base + a judge + a rule + gates), switch on the
// OpenAI-compatible endpoint for other applications, and read every request as a fork graph: which steps the judge was
// asked at, where it departed from the base, when it answered by itself, and when the failure gate ran it again.
// Everything from a request is shown as text (textContent), never as markup.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const VIEW = 'hybrid';
  const sec = $('tab-hybrid'); if (!sec || !window.psi) return;
  const api = (a) => psi.hybrid(a);
  let OV = null, EDIT = null, MODELS = [], POLL = null, LAST = null;   // LAST: the record open in a detail panel, for Prometheus
  const pct = (p) => (p == null ? 'not reported' : `${(p * 100).toFixed(p >= 0.1 ? 0 : 1)}%`);   // null: a piece of a character, whose odds llama.cpp does not report
  const show = (s) => String(s == null ? '' : s).replace(/\n/g, '⏎');

  // A click handler whose failure is shown in the status line, never an unhandled rejection.
  const safely = (fn) => async (...a) => { try { await fn(...a); } catch (e) { PSI.status($('hyStat'), e.message, 'error'); } };

  // ── sub-views ──
  function tab(which) {
    for (const b of sec.querySelectorAll('[data-hy]')) { b.classList.toggle('on', b.dataset.hy === which); b.setAttribute('aria-selected', String(b.dataset.hy === which)); }
    for (const p of sec.querySelectorAll('[data-hy-panel]')) p.hidden = p.dataset.hyPanel !== which;
    clearInterval(POLL);
    if (which === 'dash') { refresh(); live(); let n = 0; POLL = setInterval(() => { if (sec.classList.contains('hidden')) return; live(); if (++n % 5 === 0) refresh(true); }, 1000); }
    try { localStorage.setItem('psi-hy-tab', which); } catch (_) { /* private mode */ }
  }
  for (const b of sec.querySelectorAll('[data-hy]')) b.addEventListener('click', () => tab(b.dataset.hy));

  // ── judges: the models Prometheus can use (local first), plus the random control ──
  async function judges() {
    try { MODELS = (await psi.external({ action: 'models' })).models || []; } catch (_) { MODELS = []; }
    const sel = $('hyJudge'); const cur = sel.value; sel.textContent = '';
    sel.appendChild(el('option', null, '— none (base alone) —')).value = '';
    sel.appendChild(el('option', null, 'random (the control)')).value = 'random';
    sel.appendChild(el('option', null, 'laya/4322 · a decision model on this computer (tools/laya_server.py; chooses, never writes)')).value = 'laya/4322';
    for (const m of MODELS) { const o = el('option', null, `${m.name || m.id}${m.in === 0 && m.out === 0 ? '' : ` · $${m.in}/$${m.out} per M`}`); o.value = m.id; sel.appendChild(o); }
    if (cur) setJudge(cur);
  }
  // The judge choice holds the judge even when the list does not (models not loaded, no key, another laya port), so saving
  // a hybrid never silently turns it into the base alone.
  function setJudge(v) {
    const sel = $('hyJudge'); v = v || '';
    if (v && ![...sel.options].some((o) => o.value === v)) sel.appendChild(el('option', null, `${v} · not in the list now`)).value = v;
    sel.value = v;
  }

  // ── hybrids ──
  function fill(h) {
    EDIT = h ? h.id : null;
    $('hyId').value = h ? h.id : ''; setJudge(h ? h.judge : ''); $('hyK').value = h ? h.k : 8; $('hyMax').value = h ? h.maxTokens : 200; $('hyRule').value = h ? h.rule : '';
    const g = (h && h.gates) || {}; const m = (h && h.modes) || {};
    $('hyMBase').checked = !h || m.base !== false; $('hyMHybrid').checked = !h || m.hybrid !== false; $('hyMJudge').checked = !h || m.judge !== false;
    $('hyGComp').checked = !h || !!(g.competence && g.competence.on); $('hyWeak').value = h && g.competence ? (g.competence.weak || []).join('\n') : '';
    $('hyPace').value = h ? (h.pace || 0) : 0; $('hyIntervene').checked = !!(h && h.intervene); const pa = (h && h.price_as) || {}; $('hyPriceBase').value = pa.base || ''; $('hyPriceJudge').value = pa.judge || '';
    $('hyGShape').checked = !h || !!(g.shape && g.shape.on); $('hyShapeWho').value = h && g.shape && g.shape.who === 'judge' ? 'judge' : 'hybrid'; $('hyFields').value = h && g.competence ? (g.competence.fields || []).join(', ') : ''; $('hyGDoubt').checked = !h || !!(g.doubt && g.doubt.on); $('hyBits').value = h && g.doubt ? g.doubt.bits : 1; $('hyRoute').checked = !!(h && g.doubt && g.doubt.route); $('hyGFail').checked = !h || !!(g.failure && g.failure.on); $('hyGAgree').checked = !!(h && g.agreement && g.agreement.on); $('hyAgBase').checked = !!(h && g.agreement && g.agreement.base_thinks); $('hyAgJudge').checked = !!(h && g.agreement && g.agreement.judge_thinks); $('hyAgVerify').checked = !!(h && g.agreement && g.agreement.verify); const cx = (h && h.context) || {}; $('hyCxQ').checked = cx.question !== false; $('hyCxW').value = cx.window || 0; $('hyCxO').checked = cx.odds !== false; $('hyCxR').checked = cx.rule !== false;
    $('hyRemove').hidden = !h;
  }
  function read() {
    return { id: $('hyId').value.trim(), judge: $('hyJudge').value, k: +$('hyK').value, maxTokens: +$('hyMax').value, rule: $('hyRule').value.trim(), pace: +$('hyPace').value, intervene: $('hyIntervene').checked, price_as: { base: $('hyPriceBase').value.trim(), judge: $('hyPriceJudge').value.trim() },
      modes: { base: $('hyMBase').checked, hybrid: $('hyMHybrid').checked, judge: $('hyMJudge').checked },
      gates: { competence: { on: $('hyGComp').checked, weak: $('hyWeak').value.split(/[\n,]/).map((w) => w.trim()).filter(Boolean), fields: $('hyFields').value.split(',').map((w) => w.trim().toLowerCase()).filter(Boolean) }, shape: { on: $('hyGShape').checked, who: $('hyShapeWho').value }, doubt: { on: $('hyGDoubt').checked, bits: +$('hyBits').value, route: $('hyRoute').checked }, failure: { on: $('hyGFail').checked }, agreement: { on: $('hyGAgree').checked, base_thinks: $('hyAgBase').checked, judge_thinks: $('hyAgJudge').checked, verify: $('hyAgVerify').checked } }, context: { question: $('hyCxQ').checked, window: +$('hyCxW').value, odds: $('hyCxO').checked, rule: $('hyCxR').checked } };
  }
  function list() {
    const L = $('hyList'); L.textContent = ''; const T = $('hyTryModel'); const tryWas = T.value; T.textContent = '';
    if (!OV.hybrids.length) L.appendChild(el('p', 'muted small', 'No hybrids yet: define one below.'));
    for (const h of OV.hybrids) {
      const row = el('div', 'cm-item'); const g = h.gates || {};
      row.appendChild(el('b', null, h.id)); row.appendChild(el('span', 'small muted', ` · base ${OV.base || '(no model loaded)'} · judge ${h.judge || 'none'} · k ${h.k} · gates: ${['competence', 'shape', 'doubt', 'failure', 'agreement'].filter((k) => g[k] && g[k].on).join(', ') || 'none'}${g.doubt && g.doubt.on ? ` (doubt ≥ ${g.doubt.bits} bits)` : ''}`));
      const b = el('button', 'ghost small', 'Edit'); b.setAttribute('aria-label', `edit the hybrid ${h.id}`); b.addEventListener('click', () => fill(h)); row.appendChild(b); L.appendChild(row);
      T.appendChild(el('option', null, h.id)).value = h.id;
    }
    if (tryWas && OV.hybrids.some((h) => h.id === tryWas)) T.value = tryWas;
  }
  $('hySave').addEventListener('click', async () => {
    try { const r = await api({ action: 'save', hybrid: read() }); OV.hybrids = r.hybrids; list(); fill(OV.hybrids.find((x) => x.id === read().id)); PSI.status($('hyStat'), 'saved', 'ok'); }
    catch (e) { PSI.status($('hyStat'), e.message, 'error'); }
  });
  $('hyNew').addEventListener('click', () => fill(null));
  $('hyRemove').addEventListener('click', async () => { if (!EDIT) return; try { const r = await api({ action: 'remove', id: EDIT }); OV.hybrids = r.hybrids; list(); fill(null); } catch (e) { PSI.status($('hyStat'), e.message, 'error'); } });
  $('hyTry').addEventListener('click', () => PSI.run(VIEW, async () => {
    const out = $('hyTryOut'); out.textContent = ''; PSI.status($('hyStat'), 'asking…', 'busy');
    try { const rec = await api({ action: 'try', model: $('hyTryModel').value, prompt: $('hyTryQ').value }); detail(rec, out); PSI.status($('hyStat'), `${rec.mode} · ${rec.pieces} pieces · judge ${rec.judge_calls} calls · ${(rec.ms / 1000).toFixed(1)} s`, 'ok'); refresh(true); }
    catch (e) { PSI.status($('hyStat'), e.message, 'error'); }
  }));

  // ── the external API ──
  function serveOut(s) {
    $('hyServe').checked = !!s.enabled; $('hyPort').value = s.port; PSI.status($('hyServeStat'), s.running ? `serving at ${s.url}` : s.enabled ? 'enabled but not running (port in use?)' : 'off', s.running ? 'ok' : s.enabled ? 'warn' : '');
    const o = $('hyApiOut'); o.textContent = '';
    o.appendChild(el('div', null, `Base URL   ${s.url}`)); o.appendChild(el('div', null, `Token      ${s.token}`)); o.appendChild(el('div', null, `Models     ${(OV.hybrids.map((h) => h.id).join(', ')) || '(define a hybrid first)'}`));
    const ex = OV.hybrids[0] ? OV.hybrids[0].id : 'hybrid/cautious';
    o.appendChild(el('div', 'mt-12', `curl ${s.url}/chat/completions -H "Authorization: Bearer ${s.token}" -H "Content-Type: application/json" -d '{"model":"${ex}","messages":[{"role":"user","content":"What is the capital of Australia?"}]}'`));
    o.appendChild(el('div', 'mt-12', `from openai import OpenAI\nclient = OpenAI(base_url="${s.url}", api_key="${s.token}")\nr = client.chat.completions.create(model="${ex}", messages=[{"role": "user", "content": "What is the capital of Australia?"}])\nprint(r.choices[0].message.content)   # r.model_extra["psi_hybrid"]: mode, gates, every step`));
  }
  $('hyServe').addEventListener('change', async () => { try { serveOut(await api({ action: 'serve', enabled: $('hyServe').checked, port: +$('hyPort').value })); } catch (e) { PSI.status($('hyServeStat'), e.message, 'error'); } });
  $('hyPort').addEventListener('change', async () => { try { serveOut(await api({ action: 'serve', port: +$('hyPort').value })); } catch (e) { PSI.status($('hyServeStat'), e.message, 'error'); } });
  $('hyRotate').addEventListener('click', safely(async () => { serveOut(await api({ action: 'serve', rotate: true })); }));

  // ── the dashboard ──
  // Live requests: the answer so far, Pause, and when paused the current fork's menu to choose from.
  let LIVE_SIG = '';
  async function live() {
    let L; try { L = (await api({ action: 'live' })).live; } catch (_) { return; }
    const box = $('hyLive'); box.hidden = !L.length; if (!L.length) { box.textContent = ''; LIVE_SIG = ''; return; }
    const sig = JSON.stringify(L.map((l) => [l.id, l.step, l.paused, !!l.menu, (l.answer || '').length])); if (sig === LIVE_SIG) return; LIVE_SIG = sig;
    box.textContent = '';
    for (const l of L) {
      const w = el('div'); w.appendChild(el('p', 'k m-0', `live · ${l.model} · ${l.mode} · step ${l.step + 1}${l.paused ? ' · paused' : ''} · ${l.client || ''}`));
      w.appendChild(el('div', 'small', `Q: ${l.question}`)); const R = el('div', 'hy-ribbon'); R.textContent = l.answer || '…'; w.appendChild(R);
      const row = el('div', 'row');
      if (!l.paused) { const b = el('button', 'small', 'Pause at the next fork'); b.addEventListener('click', safely(async () => { await api({ action: 'pause', id: l.id }); live(); })); row.appendChild(b); }
      else {
        const r = el('button', 'ghost small', 'Resume (let the hybrid choose)'); r.addEventListener('click', safely(async () => { await api({ action: 'resume', id: l.id }); live(); })); row.appendChild(r);
      }
      const st = el('button', 'ghost small', 'Stop'); st.addEventListener('click', safely(async () => { await api({ action: 'stop', id: l.id }); live(); })); row.appendChild(st); w.appendChild(row);
      if (l.paused && l.menu) {
        w.appendChild(el('p', 'small muted', `the fork at step ${l.step + 1} · doubt ${l.entropy == null ? '?' : l.entropy.toFixed(2)} bits${l.judge ? ` · the judge suggests #${l.judge.pick + 1}: ${l.judge.reason}` : ' · the judge was not asked here'} — choose a piece, and the request goes on paused at the next fork`));
        const M = el('div', 'hy-menu');
        l.menu.forEach((c, i) => { const b = el('button', 'ghost' + (i === l.suggested ? ' sug' : ''), `${i + 1}. ${show(c.piece) || '∅'} ${pct(c.p)}`); b.addEventListener('click', safely(async () => { await api({ action: 'choose', id: l.id, pick: i, stayPaused: true }); live(); })); M.appendChild(b); });
        w.appendChild(M);
      } else if (l.paused) w.appendChild(el('p', 'small muted', 'pausing: waits for the next fork…'));
      box.appendChild(w);
    }
  }
  function stats(s) {
    // Four headline numbers; every counter (the guide reads them too) in a list one click away.
    const S = $('hyStats'); S.textContent = '';
    const n = s.requests || 0, judged = (s.by_mode || {}).judge || 0, pct = (x) => `${Math.round(x * 100)}%`;
    const head = el('div', 'hy-head');
    const cell = (num, lab) => { const c = el('div', 'hy-stat'); c.append(el('div', 'hy-num', String(num)), el('div', 'hy-lab', lab)); head.appendChild(c); };
    cell(n, 'requests, from outside and here');
    cell(judged, `answered by the judge${n ? ` (${pct(judged / n)})` : ''}`);
    cell(pct(s.judge_share_of_steps || 0), `of steps judged · ${s.judge_calls} judge calls`);
    cell(`${((s.mean_ms || 0) / 1000).toFixed(1)} s`, 'mean time per request');
    S.appendChild(head);
    const d = el('details', 'hy-all'); d.appendChild(el('summary', null, 'all counters'));
    const T = el('table', 'hy-steps hy-counters');
    const add = (k, v) => { const tr = el('tr'); tr.append(el('td', null, k), el('td', null, String(v))); T.appendChild(tr); };
    add('requests', n); for (const [m, c] of Object.entries(s.by_mode || {})) add(m === 'judge' ? 'judge answered' : m === 'base' ? 'base alone' : 'hybrid', c);
    add('pieces', s.pieces); add('judge calls', s.judge_calls); add('chosen by a person', s.human_picks || 0); add('of steps judged', pct(s.judge_share_of_steps || 0)); add('departures', s.judge_departures); add('agreed with the base', s.judge_agreed); add('reruns (failure gate)', s.reruns); add('mean ms', s.mean_ms);
    if (s.tokens) { add('tokens the base read / wrote', `${s.tokens.base_in} / ${s.tokens.base_out}`); add(`tokens the judge read / wrote${s.tokens.estimated ? ' (some estimated)' : ''}`, `${s.tokens.judge_in} / ${s.tokens.judge_out}`); }
    d.appendChild(T); S.appendChild(d);
    costs(s.by_model || {});
  }
  // What the recent requests would cost per thousand at OpenRouter's list prices: each hybrid's traffic priced as the
  // models it names (price_as), or, for a judge that is itself an OpenRouter model, at that model's own price.
  const fmt$ = (x) => (x >= 10 ? `$${x.toFixed(2)}` : x >= 1 ? `$${x.toFixed(3)}` : `$${x.toFixed(4)}`);
  let PRICES = {}, PRICES_FOR = '', COSTS = [], COST_RUN = 0;   // COSTS: the table's rows as numbers, for the guide (PSIViewData)
  async function costs(byModel) {
    const C = $('hyCosts'); const run = ++COST_RUN;   // only the latest of overlapping refreshes draws the table
    const rows = Object.entries(byModel).filter(([, m]) => m.counted > 0); if (!rows.length) { C.textContent = ''; COSTS = []; C.hidden = true; return; }
    const asOf = (id) => (id === 'base' ? { base: OV.price_base_as || '' } : ((OV.hybrids || []).find((x) => x.id === id) || {}).price_as || {});
    const want = [...new Set(rows.flatMap(([id, m]) => [asOf(id).base, asOf(id).judge, m.judge]).filter((x) => x && x !== 'random' && !/^laya\//.test(x)))].sort();
    if (want.length && want.join(',') !== PRICES_FOR) { try { PRICES = (await api({ action: 'prices', ids: want })).per_m || {}; PRICES_FOR = want.join(','); } catch (_) { PRICES = {}; } }
    if (run !== COST_RUN) return; C.textContent = ''; COSTS = []; C.hidden = false;
    const price = (id) => (id && PRICES[id] && (PRICES[id].in || PRICES[id].out) ? PRICES[id] : null);
    C.appendChild(el('p', 'k mt-12', 'what the traffic would cost'));
    C.appendChild(el('p', 'small muted', 'Per 1,000 requests at OpenRouter’s list prices. Models on this computer and the tester cost nothing to run; this is what the same traffic would cost bought.'));
    const head = el('div', 'row'); head.appendChild(el('span', 'small muted', 'price the plain base row as'));
    const pb0 = el('input'); pb0.type = 'text'; pb0.className = 'w-14'; pb0.placeholder = 'price the plain base row as, e.g. qwen/qwen3-14b'; pb0.setAttribute('aria-label', 'price the plain base row as (each hybrid uses its own price it as)'); pb0.value = OV.price_base_as || '';
    pb0.addEventListener('change', async () => { try { OV.price_base_as = (await api({ action: 'price_base', id: pb0.value })).price_base_as; costs(byModel); } catch (e) { PSI.status($('hyStat'), e.message, 'error'); } }); head.appendChild(pb0); C.appendChild(head);
    const T = el('table', 'hy-steps'); const h = el('tr'); for (const c of ['model', 'requests', 'tokens per request (base in / out · judge in / out)', 'priced as', 'per 1,000 requests']) h.appendChild(el('th', null, c)); T.appendChild(h);
    for (const [id, m] of rows) {
      const pa = asOf(id); const pb = price(pa.base), pj = price(pa.judge) || price(m.judge);
      const per = (k) => m[k] / m.counted; const cost = (pb ? (per('base_in') * pb.in + per('base_out') * pb.out) / 1e6 : 0) + (pj ? (per('judge_in') * pj.in + per('judge_out') * pj.out) / 1e6 : 0);
      const tr = el('tr'); tr.appendChild(el('td', null, id)); tr.appendChild(el('td', null, String(m.counted)));
      tr.appendChild(el('td', null, `${Math.round(per('base_in'))} / ${Math.round(per('base_out'))} · ${Math.round(per('judge_in'))} / ${Math.round(per('judge_out'))}${m.estimated ? ' (some estimated)' : ''}`));
      tr.appendChild(el('td', null, `${pb ? `${pa.base} ($${pb.in}/$${pb.out} per M)` : per('base_out') ? 'the base: not priced' : '—'}${m.judge ? ` · ${pj ? `${pa.judge || m.judge} ($${pj.in}/$${pj.out} per M)` : 'the judge: not priced'}` : ''}`));
      tr.appendChild(el('td', null, pb || pj ? fmt$(cost * 1000) : '—')); T.appendChild(tr);
      COSTS.push({ hybrid: id, requests_counted: m.counted, tokens_per_request: { base_in: Math.round(per('base_in')), base_out: Math.round(per('base_out')), judge_in: Math.round(per('judge_in')), judge_out: Math.round(per('judge_out')) }, base_priced_as: pb ? pa.base : null, judge_priced_as: pj ? (pa.judge || m.judge) : null, usd_per_1000_requests: pb || pj ? +(cost * 1000).toFixed(4) : null });
    }
    C.appendChild(T);
  }
  let SIG = '';   // the list is rebuilt only when its rows change, so its buttons are not pulled from under a click
  function reqs(force) {
    const L = $('hyReqs'); const fm = $('hyFMode').value, fj = $('hyFJudge').checked, fd = $('hyFDep').checked;
    const rows = OV.recent.filter((r) => (!fm || r.mode === fm) && (!fj || r.judge_calls > 0) && (!fd || r.departures > 0));
    const sig = `${fm}|${fj}|${fd}|` + rows.map((r) => r.id).join(','); if (!force && sig === SIG) return; SIG = sig; L.textContent = '';
    if (!rows.length) L.appendChild(el('p', 'muted small', 'No requests yet. Switch on the external API and call a hybrid, or try one on the Hybrids tab.'));
    for (const r of rows) {
      const row = el('div', 'cm-item'); const t = new Date(r.ts);
      row.appendChild(el('span', 'small muted', `${t.toLocaleDateString()} ${t.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · `));
      row.appendChild(el('b', null, r.mode === 'judge' ? 'judge answered' : r.mode === 'base' ? 'base alone' : 'hybrid')); row.appendChild(el('span', 'small muted', ` · ${r.model} · ${r.pieces} pieces · judge ${r.judge_calls} calls, ${r.departures} departures${r.rerun ? ' · rerun' : ''}${r.replay_of ? ` · replay from step ${r.at_step + 1}` : ''} · ${(r.ms / 1000).toFixed(1)} s · ${r.client || ''}`));
      row.appendChild(el('div', 'small', `Q: ${r.prompt}`)); row.appendChild(el('div', 'small muted', `A: ${r.answer}`));
      const b = el('button', 'ghost small', 'Fork graph'); b.setAttribute('aria-label', `fork graph of the ${r.model} request “${String(r.prompt || '').slice(0, 60)}”`); b.addEventListener('click', safely(async () => { const rec = await api({ action: 'request', id: r.id }); const D = $('hyDetail'); D.hidden = false; detail(rec, D); D.scrollIntoView({ block: 'nearest' }); })); row.appendChild(b); L.appendChild(row);
    }
  }
  for (const id of ['hyFMode', 'hyFJudge', 'hyFDep']) $(id).addEventListener('change', () => reqs(true));
  $('hyRefresh').addEventListener('click', () => refresh());
  // The most recent request's record, without hunting through the list (the list is newest first).
  $('hyLatest').addEventListener('click', async () => {
    try { await refresh(true); const r = (OV && OV.recent || []).slice().sort((a, b) => new Date(b.ts) - new Date(a.ts))[0]; if (!r) { PSI.status($('hyStat'), 'no requests yet', 'warn'); return; }
      const rec = await api({ action: 'request', id: r.id }); const D = $('hyDetail'); D.hidden = false; detail(rec, D); D.scrollIntoView({ block: 'nearest' }); }
    catch (e) { PSI.status($('hyStat'), e.message, 'error'); }
  });
  $('hyClear').addEventListener('click', safely(async () => { await api({ action: 'clear' }); refresh(); }));

  // The fork graph of one request: the answer as a ribbon, each piece shaded by the base's doubt at that step, dotted
  // where the judge was asked, outlined where it departed from the base's top choice; then every step's menu.
  function ribbon(steps, title) {
    const wrap = el('div'); if (title) wrap.appendChild(el('p', 'k mt-12', title));
    const R = el('div', 'hy-ribbon');
    // The chat template's empty thinking block (<think>, line breaks, </think>, line breaks) at the start is the
    // template's, not the answer's: kept (replay counts every step) but drawn faint and small.
    let tmplEnd = -1; { const k = steps.findIndex((s) => /<\/think>/.test(s.piece || '')); if (k >= 0 && k < 8 && steps.slice(0, k + 1).every((s) => /^(<\/?think>|\s*)$/.test(s.piece || ''))) { tmplEnd = k; while (tmplEnd + 1 < steps.length && /^\s+$/.test(steps[tmplEnd + 1].piece || '')) tmplEnd++; } }
    steps.forEach((s, j) => {
      const sp = el('span', (s.judge ? 'judged ' : '') + (s.judge && s.judge.pick !== 0 ? 'departed ' : '') + (s.human ? 'human ' : '') + (s.before ? 'before ' : '') + (s.judge && s.judge.failed ? 'failed' : ''), show(s.piece) || '∅');
      if (j <= tmplEnd) { sp.classList.add('tmpl'); sp.title = 'the chat template’s empty thinking block'; }
      const a = Math.min(1, (s.entropy || 0) / 5); sp.style.background = s.unreported ? 'none' : `rgba(242,241,238,${(0.04 + a * 0.5).toFixed(3)})`; sp.dataset.step = j;
      if (s.unreported) sp.classList.add('unreported');   // no odds reported: no doubt to shade
      sp.title = stepTitle(s, j);
      sp.addEventListener('mouseenter', () => hl(wrap.parentNode || wrap, j, true)); sp.addEventListener('mouseleave', () => hl(wrap.parentNode || wrap, j, false));
      R.appendChild(sp);
    });
    wrap.appendChild(R); return wrap;
  }
  // Each step's menu; with a record that has piece ids, every other piece is a button that replays the request from
  // that step with it, and "ask the judge here" replays with the judge forced at that step.
  function table(steps, rec) {
    const canReplay = !!(rec && rec.id && steps.length && steps[0].id != null);
    const T = el('table', 'hy-steps'); const h = el('tr'); for (const c of ['step', canReplay ? 'base’s menu (odds) — click another piece to replay from here' : 'base’s menu (odds)', 'doubt', 'judge']) h.appendChild(el('th', null, c)); T.appendChild(h);
    const replay = async (step, pick, judge) => {
      const D = $('hyDetail'); PSI.status($('hyStat'), `replaying from step ${step + 1}…`, 'busy');
      try { const out = await api({ action: 'replay', id: rec.id, step, pick, judge }); detail(out, D); D.scrollIntoView({ block: 'nearest' }); PSI.status($('hyStat'), `replayed: ${out.pieces} pieces · judge ${out.judge_calls} calls · ${(out.ms / 1000).toFixed(1)} s`, 'ok'); refresh(true); }
      catch (e) { PSI.status($('hyStat'), e.message, 'error'); }
    };
    steps.forEach((s, j) => {
      const tr = el('tr', s.before ? 'before' : ''); tr.dataset.step = j; tr.appendChild(el('td', null, String(j + 1)));
      const taken = s.human ? s.human.pick : s.judge ? s.judge.pick : 0;
      const menu = el('td', 'menu'); (s.top || []).forEach((c, i) => {
        const label = `${i + 1}. ${show(c.piece)} ${pct(c.p)}`;
        if (canReplay && !s.before && i !== taken) { const b = el('button', 'ghost small', label); b.title = `replay this request from step ${j + 1} with “${show(c.piece)}” chosen`; b.addEventListener('click', () => replay(j, i, false)); menu.appendChild(b); }
        else menu.appendChild(el('span', taken === i ? 'pick' : '', label));
      });
      if (canReplay && !s.before && !s.judge && !s.unreported && rec.judge) { const b = el('button', 'ghost small', 'ask the judge here'); b.title = `replay from step ${j + 1} with the judge choosing at this step`; b.addEventListener('click', () => replay(j, null, true)); menu.appendChild(b); }
      tr.appendChild(menu);
      tr.appendChild(el('td', null, s.entropy == null ? '—' : s.entropy.toFixed(2)));
      tr.appendChild(el('td', null, `${s.human ? `${s.human.forced ? 'replayed with' : 'a person took'} #${s.human.pick + 1}` : ''}${s.human && s.judge ? ' · ' : ''}${s.judge ? `${s.judge.pick === 0 ? 'judge agreed' : `judge took #${s.judge.pick + 1}`} · ${s.judge.reason} · ${s.judge.ms} ms${s.judge.failed ? ' · failed' : ''}` : s.human ? '' : '—'}`)); T.appendChild(tr);
    });
    return T;
  }
  // ── drawings (SVG built element by element; every string from a record is a text node or an attribute, never markup) ──
  const NS = 'http://www.w3.org/2000/svg';
  const sv = (t, attrs, txt) => { const e = document.createElementNS(NS, t); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, String(v)); if (txt != null) e.textContent = txt; return e; };
  const stepTitle = (s, j) => `step ${j + 1} · “${show(s.piece) || '∅'}” · odds ${pct(s.p)} · doubt ${s.entropy == null ? '?' : s.entropy.toFixed(2)} bits · ${s.human ? `chosen by a person (#${s.human.pick + 1}) · ` : ''}${s.judge ? (s.judge.pick === 0 ? `judge asked, agreed with the base: ${s.judge.reason}` : `judge chose #${s.judge.pick + 1} over the base's top choice “${show(s.top[0].piece)}”: ${s.judge.reason}`) : 'judge not asked'}`;
  // The fork graph proper: one column per step. Above, the base's doubt at that step as an area. Below, the step's menu as a
  // stack of bars by odds, the likeliest on top; the bar taken is lit, and the road through the lit bars shows where the
  // answer left the base's first choice (a judge's departure, in amber; a person's pick, ringed). Hovering a column lights
  // the same piece in the ribbon; clicking it opens that step in the table.
  function graph(steps, wrap) {
    const n = steps.length; if (!n) return null;
    const W = Math.max(320, (wrap && wrap.clientWidth) || 900); const col = Math.max(4, Math.min(n <= 12 ? 40 : 26, Math.floor(W / n))); const gap = col >= 12 ? 2 : col >= 7 ? 1 : 0;
    const top = 44, H = 112, Y0 = top + 12, width = Math.max(n * col, Math.min(W, 380));   // sized to the answer, not the window
    const svg = sv('svg', { class: 'hy-graph', width, height: Y0 + H + 18, viewBox: `0 0 ${width} ${Y0 + H + 18}`, role: 'img', 'aria-label': `fork graph of ${n} steps` });
    const maxE = Math.max(1, ...steps.map((s) => s.entropy || 0));
    // doubt: an area over the steps, the scale from 0 to the highest doubt in this answer (at least one bit)
    const ex = (j) => j * col + col / 2, ey = (e) => top - 4 - ((e || 0) / maxE) * (top - 10);
    const area = steps.map((s, j) => `${j ? 'L' : 'M'}${ex(j)},${ey(s.entropy)}`).join(' ');
    svg.appendChild(sv('path', { class: 'doubt-area', d: `${area} L${ex(n - 1)},${top - 4} L${ex(0)},${top - 4} Z` }));
    svg.appendChild(sv('path', { class: 'doubt-line', d: area }));
    svg.appendChild(sv('text', { class: 'ax', x: 2, y: 9 }, `doubt, up to ${maxE.toFixed(1)} bits`));
    svg.appendChild(sv('text', { class: 'ax', x: 2, y: Y0 - 2 }, width < 700 ? 'the menu, by odds; lit: the piece taken' : 'the menu at each step, by odds (likeliest on top); the lit bar is the piece taken, the road runs through them'));
    const road = []; const marks = sv('g');
    steps.forEach((s, j) => {
      const g = sv('g', { class: `col${s.before ? ' before' : ''}`, 'data-step': j }); g.appendChild(sv('title', null, stepTitle(s, j)));
      const taken = s.human ? s.human.pick : s.judge ? s.judge.pick : 0; const menu = (s.top && s.top.length ? s.top : [{ piece: s.piece, p: s.p }]).slice(0, 8);
      const sum = menu.reduce((a, c) => a + (c.p || 0), 0) || 1; let y = Y0; const x = j * col;
      g.appendChild(sv('rect', { class: 'hit', x, y: 0, width: col, height: Y0 + H + 18 }));
      menu.forEach((c, i) => {
        const h = Math.max(1, ((c.p || 0) / sum) * H - gap); const lit = i === taken;
        g.appendChild(sv('rect', { class: lit ? `bar lit${s.judge && s.judge.pick !== 0 ? ' departed' : ''}${s.judge && s.judge.failed ? ' failed' : ''}` : 'bar', x: x + (gap > 1 ? 1 : 0), y, width: Math.max(1, col - (gap > 1 ? 2 : gap)), height: h, rx: col >= 12 ? 1.5 : 0 }));
        if (lit) { road.push([x + col / 2, y + h / 2]); if (s.human) marks.appendChild(sv('circle', { class: 'human', cx: x + col / 2, cy: y + h / 2, r: Math.min(5, col / 2) })); }
        y += h + gap;
      });
      if (s.judge) marks.appendChild(sv('rect', { class: `ask${s.judge.pick !== 0 ? ' departed' : ''}`, x: x + col / 2 - 1.5, y: Y0 + H + 4, width: 3, height: 8 }));
      g.addEventListener('mouseenter', () => hl(wrap, j, true)); g.addEventListener('mouseleave', () => hl(wrap, j, false));
      g.addEventListener('click', () => { const det = wrap.querySelector('details'); if (det) det.open = true; const tr = wrap.querySelector(`tr[data-step="${j}"]`); if (tr) { tr.scrollIntoView({ block: 'center' }); tr.classList.add('hl'); setTimeout(() => tr.classList.remove('hl'), 1600); } });
      svg.appendChild(g);
    });
    svg.appendChild(sv('path', { class: 'road', d: road.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') }));
    svg.appendChild(marks);
    svg.appendChild(sv('text', { class: 'ax', x: 2, y: Y0 + H + 16 }, width < 700 ? 'ticks: judge asked (amber: departed)' : 'ticks: the judge was asked at this step (amber: it departed from the base’s first choice) · hover a column, click it for the step'));
    const box = el('div', 'hy-graph-wrap'); box.appendChild(svg); return box;
  }
  function hl(wrap, j, on) { for (const e of wrap.querySelectorAll(`[data-step="${j}"]`)) e.classList.toggle('hl', on); }
  // The agreement gate as a picture: the two answers' commitments, and which one the answer followed.
  function agreementViz(ag, rec) {
    // What each answer committed to, by the server's own rule (hybrid-api.js agree()): an option letter if either is one,
    // else the last number if both have one, else the first word; the answer returned is shown the same way.
    const parse = (a) => { const t0 = String(a || '').replace(/\*\*/g, ''); const m = /final answer\s*[:：]\s*([\s\S]*)$/i.exec(t0); const t = (m ? m[1] : t0).trim();
      const w = ((/^\(?([A-Za-z0-9][\w.'-]*)/.exec(t) || [])[1] || '').replace(/[.)]$/, ''); const nums = t.replace(/,/g, '').match(/-?\d+(?:\.\d+)?/g) || [];
      return { word: w, num: nums.length ? nums[nums.length - 1] : null, text: t }; };
    const P = parse(ag.base_answer), Q = parse(ag.judge_answer), R = parse(rec.answer);
    const byLetter = /^[a-e]$/i.test(P.word) || /^[a-e]$/i.test(Q.word), byNum = !byLetter && P.num != null && Q.num != null;
    const shown = (x) => (byLetter ? x.word.toUpperCase() : byNum ? (x.num != null ? x.num : x.word) : x.word || x.text).slice(0, 24) || '∅';
    const b = shown(P), j = shown(Q), out = shown(R);
    const W = 560, H = 112; const svg = sv('svg', { class: 'hy-agree', viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, role: 'img', 'aria-label': `the base committed to ${b}, the judge to ${j}; the answer is ${out}` });
    const node = (x, y, label, text, cls) => { const g = sv('g', { class: `node ${cls || ''}` }); g.appendChild(sv('rect', { x: x - 70, y: y - 16, width: 140, height: 32, rx: 16 })); g.appendChild(sv('text', { class: 'lab', x, y: y - 22, 'text-anchor': 'middle' }, label)); g.appendChild(sv('text', { class: 'val', x, y: y + 5, 'text-anchor': 'middle' }, text)); svg.appendChild(g); };
    const won = ag.agreed ? 'both' : out === b && out !== j ? 'base' : out === j ? 'judge' : 'neither';
    const link = (y, cls) => svg.appendChild(sv('path', { class: `link ${cls}`, d: `M150,${y} C 260,${y} 300,62 410,62` }));
    link(36, won === 'both' || won === 'base' ? 'won' : ''); link(90, won === 'both' || won === 'judge' ? 'won' : '');
    node(80, 36, `the base${ag.base_thinks ? ', reasoning first' : ''}`, b, won === 'both' || won === 'base' ? 'won' : ''); node(80, 90, `the judge${ag.judge_thinks ? ', reasoning first' : ''}`, j, won === 'both' || won === 'judge' ? 'won' : '');
    node(480, 62, ag.agreed ? (ag.verify_changed ? 'agreed; the judge checked and changed it' : ag.verified != null ? 'agreed; the judge checked and kept it' : 'agreed') : won === 'neither' ? 'adjudicated: a third answer' : `adjudicated: the ${won}'s`, out, 'out');
    const box = el('div', 'hy-agree-wrap'); box.appendChild(svg); return box;
  }
  // The recent requests as a strip, newest on the right: one cell each, by mode; a mark under it for the agreement gate's
  // verdict. Hover for the question, click for the fork graph.
  function strip(recent) {
    const S = $('hyStrip'); S.textContent = ''; const rows = [...recent].reverse(); if (!rows.length) { S.hidden = true; return; } S.hidden = false;
    const cell = 11, W = Math.max(300, rows.length * cell), H = 22;
    const svg = sv('svg', { class: 'hy-strip', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${rows.length} recent requests` });
    rows.forEach((r, i) => {
      const t = (r.triggers || {}); const g = sv('g', { class: `cell ${r.mode}` }); g.appendChild(sv('title', null, `${r.model} · ${r.mode === 'judge' ? 'judge answered' : r.mode === 'base' ? 'base alone' : 'hybrid'}${t.disagreed ? ' · adjudicated' : t.disagreed === false ? ' · the judge agreed' : ''}${t.verify_changed ? ' · the judge changed it on checking' : ''}${r.departures ? ` · ${r.departures} departures` : ''}\nQ: ${r.prompt}`));
      g.appendChild(sv('rect', { x: i * cell + 1, y: 2, width: cell - 2, height: 12, rx: 2 }));
      if (t.disagreed != null) g.appendChild(sv('rect', { class: `mark ${t.disagreed ? 'adjudicated' : 'agreed'}${t.verify_changed ? ' changed' : ''}`, x: i * cell + 1, y: 17, width: cell - 2, height: 3, rx: 1 }));
      if (r.departures) g.appendChild(sv('rect', { class: 'mark departed', x: i * cell + 1, y: 17, width: cell - 2, height: 3, rx: 1 }));
      g.addEventListener('click', safely(async () => { const rec = await api({ action: 'request', id: r.id }); const D = $('hyDetail'); D.hidden = false; detail(rec, D); D.scrollIntoView({ block: 'nearest' }); }));
      svg.appendChild(g);
    });
    const wrap = el('div', 'hy-strip-wrap'); wrap.appendChild(svg); S.appendChild(wrap); wrap.scrollLeft = W;
    const leg = el('div', 'hy-legend'); for (const [c, t] of [['base', 'base alone'], ['hybrid', 'hybrid'], ['judge', 'judge answered'], ['agreed', 'the judge agreed'], ['adjudicated', 'adjudicated'], ['changed', 'changed on checking'], ['departed', 'departures']]) { leg.appendChild(el('i', `st ${c}`)); leg.appendChild(document.createTextNode(t)); } S.appendChild(leg);
  }

  function detail(rec, D) {
    D.textContent = ''; LAST = rec;
    D.appendChild(el('p', 'cond', `recorded on base ${rec.base_model} with judge ${rec.judge || 'none'} · hybrid ${rec.model} · ${rec.mode === 'judge' ? 'the judge answered' : rec.mode === 'base' ? 'the base alone' : 'hybrid, the judge at ' + rec.judgeAt}${rec.agreement ? ' · agreement gate' : ''}${rec.replay_of ? ' · a replay' : ''}`));
    const head = el('div', 'small muted', `${rec.model} · mode ${rec.mode === 'judge' ? 'judge answered' : rec.mode === 'base' ? 'base alone' : 'hybrid'} · judge at: ${rec.judgeAt} · triggers: ${[rec.triggers.routed ? `routed by first-step doubt (${rec.triggers.first_entropy} bits)` : '', rec.triggers.assertion ? 'a confident claim' : '', rec.triggers.future ? 'future / undisclosed' : '', rec.triggers.weak && rec.triggers.weak.length ? `weak words: ${rec.triggers.weak.join(', ')}` : '', rec.triggers.field ? `field: ${rec.triggers.field}` : '', rec.triggers.disagreed ? 'the base and the judge disagreed' : ''].filter(Boolean).join('; ') || 'none'} · base ${rec.base_model} · judge ${rec.judge || 'none'}${rec.context ? ` · judge saw: ${[rec.context.question ? 'the question' : 'no question', rec.context.window ? `last ${rec.context.window} pieces` : 'the whole answer so far', rec.context.odds ? 'odds' : 'no odds', rec.context.rule ? 'the rule' : 'no rule'].join(', ')}` : ''} · ${(rec.ms / 1000).toFixed(1)} s (judge ${(rec.judge_ms / 1000).toFixed(1)} s)`);
    D.appendChild(head); D.appendChild(el('div', 'small mt-12', `Q: ${rec.question}`));
    if (rec.replay_of) {
      const p = el('div', 'small mt-12'); p.appendChild(document.createTextNode(`A replay of request ${rec.replay_of} from step ${rec.at_step + 1}: ${rec.forced.pick != null ? `piece #${rec.forced.pick + 1} chosen` : ''}${rec.forced.pick != null && rec.forced.judge ? ', ' : ''}${rec.forced.judge ? 'the judge asked there' : ''}. The base's menu at that step ${rec.menu_matched ? 'was the same as recorded' : 'differed from the recorded one'}${rec.was ? `; the original took “${show(rec.was.piece)}” (${pct(rec.was.p)})` : ''}. Dimmed steps are the shared prefix. `));
      const b = el('button', 'ghost small', 'the original'); b.addEventListener('click', async () => { const o = await api({ action: 'request', id: rec.replay_of }); detail(o, D); }); p.appendChild(b); D.appendChild(p);
    }
    if (rec.agreement) {
      const ag = rec.agreement; const who = `${ag.base_thinks ? 'the base reasoned first; ' : ''}${ag.judge_thinks ? 'the judge reasoned first; ' : ''}`;
      D.appendChild(el('p', 'k mt-12', who + (ag.agreed ? (ag.verify_changed ? 'the agreement gate: the two committed to the same thing, the judge checked it and changed the answer' : ag.verified != null ? 'the agreement gate: the two committed to the same thing, and the judge checked it and kept it' : 'the agreement gate: the judge answered on its own too and committed to the same thing') : 'the agreement gate: the two answers disagreed, so the judge adjudicated (the answer returned)')));
      D.appendChild(agreementViz(ag, rec));
      // the answers rendered (markdown, maths), each with a raw toggle (PSI.answer)
      const ans = (t) => { const o = el('div', 'out minh-0'); o.appendChild(PSI.answer(t)); return o; };
      D.appendChild(el('div', 'small muted', 'the base’s answer:')); D.appendChild(ans(ag.base_answer)); D.appendChild(el('div', 'small muted mt-12', 'the judge’s own answer:')); D.appendChild(ans(ag.judge_answer));
      if (!ag.agreed) { D.appendChild(el('div', 'small muted mt-12', 'the adjudication:')); D.appendChild(ans(ag.reasoning || ag.adjudicated)); }
      if (ag.verified != null) { D.appendChild(el('div', 'small muted mt-12', 'the judge’s check:')); D.appendChild(ans(ag.verify_reasoning || ag.verified)); }
    }
    if (rec.mode === 'judge' && !rec.agreement) { D.appendChild(el('p', 'k mt-12', 'the judge’s own answer (the base was not asked)')); const o = el('div', 'out minh-0'); o.appendChild(PSI.answer(rec.answer)); D.appendChild(o); return; }
    const leg = el('div', 'hy-legend'); leg.appendChild(document.createTextNode('shade: the base’s doubt at that step')); leg.appendChild(el('i', 'judged')); leg.appendChild(document.createTextNode('judge asked')); leg.appendChild(el('i', 'departed')); leg.appendChild(document.createTextNode('judge departed from the base’s top choice')); leg.appendChild(el('i', 'human')); leg.appendChild(document.createTextNode('chosen by a person')); D.appendChild(leg);
    if (rec.rerun) {
      D.appendChild(ribbon(rec.rerun.first_steps, `first answer (${rec.mode === 'hybrid' ? 'gates as set' : 'base alone'}) — the failure gate: ${rec.rerun.why}`));
      D.appendChild(ribbon(rec.steps, 'run again with the judge at every step (the answer returned)'));
    } else D.appendChild(ribbon(rec.steps, `${rec.mode === 'judge' ? 'the base’s pass (the judge’s answer was returned)' : 'the answer'} · ${rec.judge_calls} judge calls on ${rec.pieces} pieces, ${rec.departures} departures`));   // when the judge answered, these steps are the base's run, not the reply
    const G = graph(rec.steps, D); if (G) D.appendChild(G);
    const det = el('details'); det.appendChild(el('summary', null, 'every step: the base’s menu, its doubt, the judge’s pick and reason')); det.appendChild(table(rec.steps, rec)); det.open = !!rec.replay_of; D.appendChild(det);
    // The same replay as a small form (keyboard users, and the guide, which has no control for a cell of the table).
    if (rec.id && rec.steps && rec.steps.length && rec.steps[0].id != null) {
      const f = el('div', 'row'); f.appendChild(el('span', 'k m-0', 'replay from'));
      const st = el('input'); st.type = 'number'; if (D.id === 'hyDetail') st.id = 'hyRpStep'; st.min = 1; st.max = rec.steps.length; st.value = 1; st.className = 'w-5'; st.setAttribute('aria-label', 'replay from step (1 = the first piece)');
      const pk = el('input'); pk.type = 'number'; if (D.id === 'hyDetail') pk.id = 'hyRpPick'; pk.min = 1; pk.max = 20; pk.value = 2; pk.className = 'w-5'; pk.setAttribute('aria-label', 'taking which piece of the menu at that step (1 = the likeliest; 0 = let the judge choose there)');
      const go = el('button', 'ghost small', 'Replay'); if (D.id === 'hyDetail') go.id = 'hyReplay'; go.setAttribute('aria-label', 'replay this request from that step with that piece');
      go.addEventListener('click', async () => {
        const step = Math.max(1, Math.min(rec.steps.length, +st.value || 1)) - 1, pick = +pk.value || 0; PSI.status($('hyStat'), `replaying from step ${step + 1}…`, 'busy');
        try { const out = await api({ action: 'replay', id: rec.id, step, pick: pick ? pick - 1 : null, judge: !pick }); detail(out, D); D.scrollIntoView({ block: 'nearest' }); PSI.status($('hyStat'), `replayed: ${out.pieces} pieces · judge ${out.judge_calls} calls · ${(out.ms / 1000).toFixed(1)} s`, 'ok'); refresh(true); }
        catch (e) { PSI.status($('hyStat'), e.message, 'error'); }
      });
      f.append(el('label', 'small muted', 'step '), st, el('label', 'small muted', ' taking menu piece '), pk, go); D.appendChild(f);
    }
  }

  async function refresh(quiet) {
    try { OV = await api({ action: 'overview' }); } catch (e) { if (!quiet) PSI.status($('hyStat'), e.message, 'error'); return; }
    list(); serveOut(OV.serve); stats(OV.stats); strip(OV.recent); reqs();
  }
  // The numbers behind the drawings, for Prometheus (current_view's view_data): the open record step by step, and the
  // dashboard's totals. Nothing here is markup; the guide quotes these instead of guessing from the picture.
  window.PSIViewData = window.PSIViewData || {};
  window.PSIViewData.hybrid = () => {
    const out = { note: 'doubt_bits: the base’s entropy over its top-k at that step (0 certain, 1 a coin toss, more torn). odds: the probability of the piece taken. judge.pick 0 = agreed with the base’s first choice, n = took the n+1th piece of the menu (a departure). The agreement gate’s verdict is not proof: an agreed answer can be wrong.' };
    if (OV && OV.stats) out.totals = OV.stats;
    if (OV && OV.serve) out.endpoint = { serving: !!OV.serve.running, enabled: !!OV.serve.enabled, base_url: OV.serve.url || null, port: OV.serve.port, models: (OV.hybrids || []).map((h) => h.id), token: 'shown on the External API tab, with a curl and a Python example' };
    if (COSTS.length) out.costs = { note: 'per 1,000 requests at OpenRouter’s list prices, each hybrid’s recent traffic priced as the models named (the tester and this computer cost nothing; this is what the same traffic would cost bought). Multiply by the user’s volume in thousands. An estimate on the recent requests, not a bill.', rows: COSTS };
    const r = LAST; if (!r) return out;
    out.record = { id: r.id, hybrid: r.model, base: r.base_model, judge: r.judge || null, mode: r.mode, judge_at: r.judgeAt, triggers: r.triggers, question: r.question, answer: r.answer, pieces: r.pieces, judge_calls: r.judge_calls, departures: r.departures, usage: r.usage || null, replay_of: r.replay_of || null,
      agreement: r.agreement ? { base_answer: r.agreement.base_answer, judge_answer: r.agreement.judge_answer, agreed: r.agreement.agreed, base_reasoned_first: !!r.agreement.base_thinks, judge_reasoned_first: !!r.agreement.judge_thinks, adjudicated: r.agreement.agreed ? null : (r.agreement.adjudicated || null), adjudication_reasoning: r.agreement.agreed ? null : (r.agreement.reasoning || null), verified: r.agreement.verified == null ? null : r.agreement.verified, verify_changed: !!r.agreement.verify_changed } : null,
      steps: (r.steps || []).slice(0, 400).map((s, j) => ({ step: j + 1, piece: s.piece, ...(s.unreported ? { odds_not_reported: 'a piece of a character: llama.cpp reports no odds for it' } : {}), odds: s.p == null ? null : +(+s.p).toFixed(4), doubt_bits: s.entropy == null ? null : +s.entropy.toFixed(3), menu: (s.top || []).slice(0, 5).map((t) => [t.piece, +(+t.p).toFixed(4)]),
        judge: s.judge ? { pick: s.judge.pick, departed: s.judge.pick !== 0, reason: s.judge.reason, failed: !!s.judge.failed } : null, person: s.human ? { pick: s.human.pick, forced: !!s.human.forced } : null, shared_prefix: !!s.before })) };
    return out;
  };
  document.querySelector('#tabs [data-tab="hybrid"]').addEventListener('click', async () => { await judges(); await refresh(); if (EDIT == null && OV && OV.hybrids[0]) fill(OV.hybrids[0]); });
  try { tab(localStorage.getItem('psi-hy-tab') || 'hybrids'); } catch (_) { tab('hybrids'); }
})();
