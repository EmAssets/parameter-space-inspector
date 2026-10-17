// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Prometheus in the manual (help.html): the same guide as the app's panel (helper.js), in docs mode. It finds and
// explains things in this manual, links to its sections, and can switch the app window to a view. Same model and
// settings (settings.helper), same OpenRouter key and spending cap (checked by the app before every call, in
// config.external). Its tools read and move this page only: nothing runs a model, nothing touches the GPU.
// Needs window.psi: the preload in the desktop app, psi-web.js (with the page's token) from the dev server.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const VIEWS = { generate: 'Generate', neural: 'Neural net', read: 'Read', tokens: 'Tokens', compare: 'Lens on / off', understand: 'Understanding', emergence: 'Emergence', lesion: 'Tipping point', sensing: 'Sensing surface', fight: 'Fight class', connectors: 'Connectors', ladder: 'Storage ladder', anatomy: 'Anatomy', file: 'File', lineage: 'Seed & lineage', settings: 'Settings' };
  const HAS_PSI = () => !!(window.psi && typeof window.psi.external === 'function' && typeof window.psi.settings === 'function');
  let CFG = { on: false, model: 'anthropic/claude-haiku-4.5', maxTokens: 1500 }, keySet = false, spent = null, cap = null;
  let MSGS = [], busy = false, stopReq = false, cost = 0, unpriced = 0;

  // ── the manual's map: every h2/h3 (ids added where the page has none, so each one can be linked) ────────
  const main = document.querySelector('main');
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 48);
  let SECS = null;
  function sections() {
    if (SECS) return SECS;
    const hs = [...main.querySelectorAll('h2, h3')]; let parent = null;
    SECS = hs.map((h) => {
      if (h.tagName === 'H2') parent = h;
      if (!h.id) { let id = `${parent && parent !== h ? `${parent.id}--` : ''}${slug(h.textContent)}`, n = 2; while (document.getElementById(id)) id = `${id}-${n++}`; h.id = id; }
      return { id: h.id, title: h.textContent.replace(/\s+/g, ' ').trim(), level: h.tagName === 'H2' ? 2 : 3, parent: h.tagName === 'H3' && parent ? parent.id : null, h };
    });
    return SECS;
  }
  // The text of a block, with lists, tables, glossary entries and closed <details> written out.
  function txt(n) {
    const t = (x) => x.textContent.replace(/\s+/g, ' ').trim();
    if (n.matches('ul, ol')) return [...n.children].map((li) => `- ${t(li)}`).join('\n');
    if (n.matches('table')) return [...n.querySelectorAll('tr')].map((r) => [...r.children].map(t).join(' | ')).join('\n');
    if (n.matches('dl')) return [...n.children].map((c) => (c.tagName === 'DT' ? `${t(c)}:` : `  ${t(c)}`)).join('\n');
    if (n.matches('.tbl, .qm')) return [...n.children].map(txt).filter(Boolean).join('\n');
    if (n.matches('details')) return [...n.children].map((c) => (c.tagName === 'SUMMARY' ? `Q: ${t(c)}` : t(c))).join('\n');
    if (n.matches('script, style, noscript')) return '';
    return t(n);
  }
  // The blocks under a heading: up to the next heading (leaf), or for an h2 with whole=true up to the next h2.
  function blocks(h, whole) {
    const out = []; let n = h.nextElementSibling;
    while (n && !(n.tagName === 'H2' || (!whole && n.tagName === 'H3'))) { out.push(n); n = n.nextElementSibling; }
    return out;
  }
  const secText = (s, whole) => blocks(s.h, whole).map((n) => (n.matches('h3') ? `\n## ${n.textContent.trim()}` : txt(n))).filter(Boolean).join('\n');
  const cut = (t, limit) => (t.length > limit ? `${t.slice(0, limit)} [cut: ${t.length - limit} chars]` : t);
  const find = (id) => sections().find((s) => s.id === String(id).replace(/^#/, ''));

  // The section in view: the last heading above the reading line (under the sticky header).
  function inView() {
    const line = Math.max(150, innerHeight * 0.25); let cur = null; for (const s of sections()) { if (s.h.getBoundingClientRect().top <= line) cur = s; else break; }
    return cur || sections()[0];
  }
  // The user's selection in the manual. Clicking into the panel's box moves the browser's selection there, so the
  // last selection made on the page is kept until the user selects (or clicks) somewhere else on the page.
  let lastSel = '';
  function readSel() { const sel = window.getSelection && window.getSelection(); if (!sel || (sel.anchorNode && P.contains(sel.anchorNode)) || document.activeElement === $('hlpIn')) return; lastSel = String(sel).trim().slice(0, 600); }
  document.addEventListener('selectionchange', readSel);
  const selection = () => lastSel;
  function appWin() { try { const a = window.opener && !window.opener.closed ? window.opener : null; if (a) void a.location.href; return a; } catch (_) { return null; } }

  // ── tools (this page only) ─────────────────────────────────────────────
  function flash(n) { n.classList.add('hlp-flash'); setTimeout(() => n.classList.remove('hlp-flash'), 2600); }
  function search(q) {
    const words = String(q).toLowerCase().split(/[^a-z0-9À-ɏ]+/).filter((w) => w.length > 1 && !STOP.has(w));
    if (!words.length) throw new Error('give a few words to search for');
    const hits = sections().map((s) => {
      const title = s.title.toLowerCase(), body = secText(s, false), low = body.toLowerCase(); let score = 0, first = -1;
      for (const w of words) {
        if (title.includes(w)) score += 6;
        let i = low.indexOf(w), c = 0; if (i >= 0 && (first < 0 || i < first)) first = i;
        while (i >= 0 && c < 20) { c++; i = low.indexOf(w, i + w.length); } score += Math.min(c, 8);
      }
      if (words.length > 1 && low.includes(words.join(' '))) score += 8;
      const at = Math.max(0, first - 80);
      return { id: s.id, title: s.title, in: s.parent ? find(s.parent).title : undefined, score, excerpt: first < 0 ? body.slice(0, 200) : `${at ? '…' : ''}${body.slice(at, at + 240).trim()}…` };
    }).filter((x) => x.score > 0).sort((a, b) => b.score - a.score).slice(0, 8);
    const terms = [...document.querySelectorAll('#gl dt')].filter((dt) => words.some((w) => dt.textContent.toLowerCase().includes(w))).slice(0, 5)
      .map((dt) => ({ term: dt.firstChild ? dt.firstChild.textContent : dt.textContent, id: dt.id || (dt.querySelector('[id]') || {}).id }));
    return { query: q, sections: hits.map(({ score, ...x }) => x), glossary_terms: terms.length ? terms : undefined, note: hits.length ? 'Link a section as [title](#id); read it with help_read before quoting details.' : 'Nothing in the manual matches. Say so plainly and suggest the closest section from the contents.' };
  }
  function openSec(id) {
    const raw = String(id).replace(/^#/, ''), s = find(raw), n = s ? s.h : document.getElementById(raw);
    if (!n || !main.contains(n)) throw new Error(`no section ${raw} in this manual; use an id from the contents or from help_search`);
    const target = n.closest('dt') || n;
    target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    flash(target); try { history.replaceState(null, '', `#${raw}`); } catch (_) { /* file:// may refuse; the scroll still happened */ }
    return { opened: s ? s.title : target.textContent.trim().slice(0, 80), id: raw };
  }
  function read(id) {
    const s = find(id); if (!s) throw new Error(`no section ${String(id)}; use an id from the contents or from help_search`);
    return { id: s.id, title: s.title, in: s.parent ? find(s.parent).title : undefined, text: cut(secText(s, true), 6000) };
  }
  function glossary(term) {
    const q = String(term).toLowerCase().trim(), dts = [...document.querySelectorAll('#gl dt')], name = (dt) => (dt.firstChild ? dt.firstChild.textContent : dt.textContent);
    if (!dts.length) throw new Error('the glossary is not built on this page (it needs guide-data.js)');
    const byId = document.getElementById(q.startsWith('term-') ? q : `term-${q.replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')}`);
    const dt = (byId && byId.closest('dt')) || dts.find((d) => name(d).toLowerCase() === q) || dts.find((d) => name(d).toLowerCase().startsWith(q)) || dts.find((d) => name(d).toLowerCase().includes(q));
    if (!dt) { const near = dts.filter((d) => q.split(/\s+/).some((w) => w.length > 2 && name(d).toLowerCase().includes(w))).slice(0, 6).map(name); return { term, found: false, near: near.length ? near : undefined, note: 'Not in the glossary. Try help_search.' }; }
    const dd = dt.nextElementSibling; return { term: name(dt), id: dt.id || (dt.querySelector('[id]') || {}).id, definition: dd ? dd.textContent.trim() : '' };
  }
  function appView(view) {
    const v = String(view).replace(/^#/, ''); if (!VIEWS[v]) throw new Error(`no view ${v}; views: ${Object.keys(VIEWS).join(', ')}`);
    const a = appWin();
    if (a) { try { a.location.hash = v; a.focus(); return { switched: true, view: VIEWS[v], note: 'The app window now shows this view; this manual stays open.' }; } catch (_) { /* fall through to the link */ } }
    return { switched: false, view: VIEWS[v], link: `index.html#${v}`, note: `The app window cannot be reached from this page. Give the user the link written as [${VIEWS[v]}](index.html#${v}), or tell them to open ${VIEWS[v]} from the app's left menu.` };
  }
  const STOP = new Set('the a an and or of to in on for is it how do does what where which when can i my me this that with be are was by at as from'.split(' '));
  const S = { type: 'string' };
  const TOOLS = [
    { name: 'help_search', d: 'Search this manual. Returns the matching sections (id, title, the h2 it sits in, a short excerpt) and matching glossary terms.', p: { query: S }, req: ['query'], run: (a) => search(a.query), done: (a) => `Searched the manual for “${String(a.query).slice(0, 40)}”` },
    { name: 'help_open', d: 'Scroll the user to a section (id from the contents, help_search, or a glossary term-* id), highlight it briefly and put it in the address.', p: { id: S }, req: ['id'], run: (a) => openSec(a.id), done: (a) => `Showed you ${(find(a.id) || {}).title || a.id}` },
    { name: 'help_read', d: 'Read one section of the manual in full (an h2 includes its h3 subsections). Long sections are cut at 6000 characters, marked “[cut: N chars]”.', p: { id: S }, req: ['id'], run: (a) => read(a.id), done: (a) => `Read ${(find(a.id) || {}).title || a.id}` },
    { name: 'glossary', d: 'Look up a term in the manual’s glossary: its definition and its #term-* id.', p: { term: S }, req: ['term'], run: (a) => glossary(a.term), done: (a) => `Looked up “${String(a.term).slice(0, 40)}”` },
    { name: 'app_view', d: `Switch the app window to a view (when the app window is reachable from this page; otherwise returns a link). view is one of: ${Object.entries(VIEWS).map(([k, v]) => `${k} (${v})`).join(', ')}`, p: { view: S }, req: ['view'], run: (a) => appView(a.view), done: (a) => `Opened ${VIEWS[a.view] || a.view} in the app` },
  ];
  // Strict arguments: an unknown, missing or non-string argument is refused, and nothing runs.
  function checkArgs(t, a) {
    const extra = Object.keys(a).filter((k) => !(k in t.p)); if (extra.length) throw new Error(`unknown argument${extra.length > 1 ? 's' : ''} ${extra.join(', ')}; ${t.name} takes only: ${Object.keys(t.p).join(', ')}`);
    for (const k of t.req) if (a[k] == null || String(a[k]).trim() === '') throw new Error(`missing argument ${k}`);
    for (const k of Object.keys(a)) if (t.p[k].type === 'string' && typeof a[k] !== 'string') throw new Error(`argument ${k} must be a string`);
  }
  function runTool(name, a) {
    const t = TOOLS.find((x) => x.name === name); if (!t) throw new Error(`no tool ${name}; tools: ${TOOLS.map((x) => x.name).join(', ')}`);
    checkArgs(t, a || {}); return t.run(a || {});
  }
  const toolDefs = () => TOOLS.map((t) => ({ type: 'function', function: { name: t.name, description: t.d, parameters: { type: 'object', properties: t.p, required: t.req, additionalProperties: false } } }));

  // ── the prompt, rebuilt every turn ─────────────────────────────────────
  function contents() { return sections().map((s) => `${s.level === 3 ? '  ' : ''}- #${s.id}: ${s.title}`).join('\n'); }
  function context() { const v = inView(), sel = selection(), a = !!appWin(); return { section: v ? { id: v.id, title: v.title } : null, selection: sel, app: a }; }
  function system(c = context()) {
    return `You are Prometheus, the AI guide of the Parameter Space Inspector (PSI), the same guide as the app's own panel and the chat on emergencemachine.com. Here you are the guide to the Inspector's manual (help.html): you help the user find and understand things in it. PSI is an app for looking inside a language model and at how models learn; many users are new to AI terms. You run on a frontier model (${CFG.model}), ${onPlan() ? 'on the user’s ChatGPT plan' : 'through OpenRouter'}; you are not the local model the app inspects.

This page now:
- section in view: ${c.section ? `[${c.section.title}](#${c.section.id})` : 'not known'}
- text the user selected: ${c.selection ? `“${c.selection}”` : 'none'}
- app window: ${c.app ? 'reachable from this page (app_view switches it to a view)' : 'not reachable from this page (app_view returns a link; the user switches windows themselves)'}

The manual's contents (id: title; indented = subsection):
${contents()}

Your jobs:
1. Answer from the manual first. Find the place with help_search, read it with help_read before quoting details, and cite it as a link written exactly [title](#id) with an id from the contents. Use help_open to take the user there when they want to see it.
2. Explain terms simply, with the glossary tool for the manual's own definitions (cite them as [term](#term-…)).
3. When the manual does not cover something, say so plainly ("The manual does not cover this"), then suggest the closest section or the app view where it could be tried. Never invent a control, button, view, setting or result that the manual does not describe. Do not answer how-to questions from memory.
4. To point at the app, use app_view (or a link written [View name](index.html#viewid)). You cannot run anything in the app from this page: no models, no experiments.

Reading results the manual describes (the app panel's rules, kept here):
- Bounds stay bounds: "at least", "at most", off-scale values and lower bounds are reported as bounds, never as readings.
- A missing or unread value is "not measured", never 0 or a guess. A difference inside the stated noise is "within noise".
- Never supply a cause the manual or an instrument did not establish: say "cause not tested" and name the view that could test it.
- Findings written in the manual are stored results, not a run on the user's model: label them "stored result (as the manual reports it)" with the date and model when the manual gives them, "not recorded" when it does not.
- Say what was measured, not a verdict ("passed 9 of 9 tests", not "it understands").

Rules:
- Short answers, plain words, one idea per sentence. A short list for steps. No headers.
- Never mention tool names or ids to the user except inside links.
- Never ask for, show or change the OpenRouter key or the spending cap; for those, point to [Settings](index.html#settings).
- If a tool fails, say so plainly and suggest the next step.`;
  }

  // ── panel ──────────────────────────────────────────────────────────────
  const fab = el('button', 'hlp-fab', 'Ask Prometheus'); fab.id = 'hlpFab'; fab.type = 'button'; fab.title = 'Prometheus, the AI guide: find and explain things in this manual (Ctrl+/)'; document.body.appendChild(fab);
  const P = el('aside', 'hlp'); P.id = 'hlp'; P.hidden = true; P.setAttribute('aria-label', 'Prometheus');
  P.innerHTML = `<div class="hlp-head"><div><p class="hlp-t">Prometheus</p><p class="hlp-m" id="hlpMode"></p></div><div class="hlp-hb"><button class="hlp-btn ghost" id="hlpNew" type="button" title="start a new conversation">New</button><button class="hlp-btn ghost" id="hlpClose" type="button" aria-label="close">×</button></div></div>
    <div class="hlp-log" id="hlpLog" aria-live="polite"></div>
    <div class="hlp-foot"><div class="hlp-ctx" id="hlpCtx"></div><textarea id="hlpIn" rows="2" placeholder="Ask where something is in the manual, or what a term means…"></textarea>
    <div class="hlp-row"><span class="hlp-cost" id="hlpCost"></span><button class="hlp-btn ghost" id="hlpStop" type="button" hidden>Stop</button><button class="hlp-btn" id="hlpSend" type="button">Send</button></div></div>`;
  document.body.appendChild(P);
  const log = $('hlpLog');
  const narrow = () => matchMedia('(max-width: 760px)').matches;

  function mode() { $('hlpMode').textContent = `${CFG.model.split('/').pop()} · guide to this manual`; }
  // On the user's ChatGPT plan (model chatgpt/<slug>, src/chatgpt.js) nothing is spent: the line says so, as OpenAI asks.
  const onPlan = () => /^chatgpt\//.test(CFG.model || '');
  function planLine(c) { c.textContent = `Using ChatGPT plan (${CFG.model.split('/').pop()}) · no money, counts against your plan’s limits · `; const a = document.createElement('a'); a.href = 'https://chatgpt.com/settings/usage?tab=overview'; a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Manage usage'; c.appendChild(a); }
  function costLine() {
    if (onPlan()) return planLine($('hlpCost'));
    const c = $('hlpCost'), capTxt = spent != null && cap != null ? ` · app total $${spent.toFixed(2)} of the $${cap} cap` : '';
    c.textContent = cost || unpriced ? `this chat $${cost.toFixed(4)}${unpriced ? ` + ${unpriced} repl${unpriced > 1 ? 'ies' : 'y'} with no reported cost` : ''}${capTxt}`
      : `each question spends OpenRouter credit (${CFG.model.split('/').pop()})${capTxt}`;
  }
  // Settings are read on every open and every question: the user may have just changed them in the app.
  async function readCfg() {
    const v = await window.psi.settings(); if (v.helper) CFG = Object.assign(CFG, v.helper);
    const e = v.external || {}; keySet = !!e.keySet;
    if (onPlan()) { let st = {}; try { st = await window.psi.chatgpt({ action: 'status' }); } catch (_) { /* not signed in */ } keySet = !!(st.signedIn && st.planUsage); }
    spent = +e.spentUsd || 0; cap = e.capUsd; mode(); costLine(); return v;
  }
  const ready = () => HAS_PSI() && CFG.on && keySet;
  async function open(v) {
    P.hidden = !v; document.body.classList.toggle('hlp-open', !!v); fab.setAttribute('aria-expanded', String(!!v));
    if (!v) { fab.focus(); return; }
    let failed = null; if (HAS_PSI()) { try { await readCfg(); } catch (e) { failed = e; } }
    const ok = ready() && !failed; $('hlpIn').disabled = !ok; $('hlpSend').disabled = !ok || busy;
    if (!ok) { setupCard(failed); setTimeout(() => $('hlpClose').focus(), 0); return; }
    if (log.querySelector('.hlp-setup')) log.textContent = '';
    ctxChips(); if (!log.childNodes.length) intro(); setTimeout(() => $('hlpIn').focus(), 0);
  }
  // Why it cannot answer yet, and the one step that gets there (the gear opens Settings in the app).
  function setupCard(failed) {
    log.textContent = ''; $('hlpCtx').textContent = ''; $('hlpCost').textContent = '';
    const w = el('div', 'hlp-intro hlp-setup'), row = el('div', 'hlp-row');
    w.appendChild(el('p', null, 'Prometheus can find things in this manual, explain them, and take you to the right section or view.'));
    if (!HAS_PSI()) w.appendChild(el('p', null, 'Prometheus needs the app running. Open this manual from the app (Help in its side menu, or Settings > Help) to ask it here; the manual itself works without it.'));
    else if (failed) w.appendChild(el('p', null, `Prometheus could not read the app’s settings (${failed.message || failed}). Is the app still running?`));
    else if (!keySet && onPlan()) w.appendChild(el('p', null, 'It is set to run on your ChatGPT plan, but you are not signed in (or plan usage was not allowed). Sign in in the app’s Settings, under Prometheus.'));
    else if (!keySet) w.appendChild(el('p', null, 'It answers through OpenRouter, a paid service for frontier models, so it needs an OpenRouter key first. Add one in the app’s Settings, under External models; the spending cap there limits what it can cost. A typical question costs under a cent.'));
    else w.appendChild(el('p', null, 'It is off. Turn on “Prometheus, the AI guide” in the app’s Settings. It uses your saved OpenRouter key and the spending cap there; a typical question costs under a cent.'));
    if (HAS_PSI() && !failed) { const a = el('a', 'hlp-btn', keySet ? 'Turn it on in Settings' : 'Add a key in Settings'); a.href = 'index.html#settings'; a.dataset.toApp = 'settings'; row.appendChild(a); }
    const h = el('a', null, 'About Prometheus'); h.href = '#prometheus'; h.addEventListener('click', (e) => { e.preventDefault(); goSection('prometheus'); }); row.appendChild(h);
    w.appendChild(row); log.appendChild(w);
  }
  function intro() {
    const w = el('div', 'hlp-intro'); w.appendChild(el('p', null, 'I’m Prometheus. Ask me where something is in this manual or what a term means; I answer from the manual and link the section.'));
    const chips = el('div', 'hlp-sugg');
    ['Where should I start?', 'Explain the section I’m reading', 'Why does a view say “not measured”?', 'What do I need for Neural net to work?'].forEach((q) => { const b = el('button', 'hlp-q', q); b.type = 'button'; b.addEventListener('click', () => ask(q)); chips.appendChild(b); });
    w.appendChild(chips); log.appendChild(w);
  }
  function ctxChips() {
    const c = $('hlpCtx'); c.textContent = ''; const v = inView(); if (v) c.appendChild(el('span', 'hlp-chip', `reading: ${v.title.slice(0, 48)}`));
    const s = selection(); if (s) { const b = el('button', 'hlp-chip sel', `explain the selection: “${s.slice(0, 40)}${s.length > 40 ? '…' : ''}”`); b.type = 'button'; b.addEventListener('click', () => ask('Explain the text I selected, in plain words, and where the manual says more about it.')); c.appendChild(b); }
  }
  fab.addEventListener('click', () => open(P.hidden)); $('hlpClose').addEventListener('click', () => open(false));
  $('hlpNew').addEventListener('click', () => { MSGS = []; cost = 0; unpriced = 0; log.textContent = ''; costLine(); if (ready()) intro(); else setupCard(); });
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === '/') { e.preventDefault(); open(P.hidden); }
    else if (e.key === 'Escape' && !P.hidden) { e.preventDefault(); open(false); }
  });
  let selT = 0; document.addEventListener('selectionchange', () => { if (P.hidden) return; clearTimeout(selT); selT = setTimeout(ctxChips, 150); });
  let scrT = 0; addEventListener('scroll', () => { if (P.hidden) return; clearTimeout(scrT); scrT = setTimeout(ctxChips, 200); }, { passive: true });

  // ── rendering ──────────────────────────────────────────────────────────
  // The model's text is drawn as nodes in one pass, never as HTML: it can only become text, http(s) links and links to
  // this manual's sections or the app's views.
  const MD = /\[([^\]]+)\]\(#([A-Za-z0-9_-]+)\)|\[([^\]]+)\]\(index\.html(?:#([a-z]+))?\)|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|(?<=^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:])|\*\*([^*]+)\*\*|`([^`]+)`/g;
  function md(text) {
    const link = (u, t) => { let ok = false; try { ok = /^https?:$/.test(new URL(u).protocol); } catch (_) { /* not a URL */ } if (!ok) return document.createTextNode(t); const a = el('a', '', t); a.href = u; a.target = '_blank'; a.rel = 'noopener'; return a; };
    const inline = (s, into) => {
      let at = 0;
      for (const m of s.matchAll(MD)) {
        if (m.index > at) into.append(s.slice(at, m.index)); at = m.index + m[0].length;
        const [, st, sid, at2, av, lt, lu, bare, bold, code] = m;
        if (sid) { if (find(sid) || document.getElementById(sid)) { const a = el('a', '', st); a.href = `#${sid}`; a.dataset.sec = sid; into.appendChild(a); } else into.append(st); }
        else if (at2 !== undefined) { if (!av || VIEWS[av]) { const a = el('a', '', at2); a.href = `index.html${av ? `#${av}` : ''}`; a.dataset.app = av || ''; into.appendChild(a); } else into.append(at2); }
        else if (lu) into.append(link(lu, lt));
        else if (bare) into.append(link(bare, bare.replace(/^https?:\/\/(www\.)?/, '')));
        else if (bold) into.appendChild(el('b', '', bold));
        else into.appendChild(el('code', '', code));
      }
      if (at < s.length) into.append(s.slice(at));
    };
    const frag = document.createDocumentFragment(); let list = null;
    for (const l of String(text).split(/\n/)) {
      const li = l.match(/^\s*[-*•]\s+(.*)/) || l.match(/^\s*\d+[.)]\s+(.*)/);
      if (li) { if (!list) list = frag.appendChild(el('ul')); inline(li[1], list.appendChild(el('li'))); continue; }
      list = null; if (l.trim()) inline(l.replace(/^#+\s*/, ''), frag.appendChild(el('p')));
    }
    return frag;
  }
  function goSection(id) { openSec(id); if (narrow()) open(false); }   // at phone width the panel covers the page: get out of the way
  log.addEventListener('click', (e) => {
    const a = e.target.closest('a'); if (!a) return;
    if (a.dataset.sec) { e.preventDefault(); try { goSection(a.dataset.sec); } catch (_) { /* the link was checked when drawn */ } }
    else if ('app' in a.dataset) {   // no app window reachable: the link is followed as written
      const v = a.dataset.app, w = appWin();
      if (v) { if (appView(v).switched) e.preventDefault(); } else if (w) { e.preventDefault(); w.focus(); }
    }
  });
  function say(role, text) { const d = el('div', `hlp-msg ${role}`); if (role === 'ai') d.appendChild(md(text)); else d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
  function step(text, cls = '') { const d = el('p', `hlp-step ${cls}`, text); log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }

  // ── the conversation ───────────────────────────────────────────────────
  const LIM = 12000;
  async function ask(q) {
    if (busy || !String(q).trim()) return;
    try { await readCfg(); } catch (_) { /* reported below */ }
    if (!ready()) { open(true); return; }
    const c = context(); say('me', q); $('hlpIn').value = '';
    MSGS.push({ role: 'user', content: `${q}\n\n(Context for Prometheus, not part of the question: the user is reading ${c.section ? `“${c.section.title}” (#${c.section.id})` : 'the manual'}${c.selection ? ` and has selected “${c.selection}”` : ''}.)` });
    busy = true; stopReq = false; $('hlpSend').disabled = true; $('hlpStop').hidden = false;
    const thinking = step('thinking…', 'wait');
    try {
      const tools = toolDefs();
      for (let round = 0; round < 10 && !stopReq; round++) {
        // The spending cap is checked by the app before the request goes out (config.external); a refusal comes back as an error.
        const r = await window.psi.external({ action: 'chat', model: CFG.model, messages: [{ role: 'system', content: system(context()) }, ...MSGS], tools, maxTokens: CFG.maxTokens || 1500 });
        if (r.tokens || r.cost) cost += +r.cost || 0; else unpriced++;
        if (r.spentThisApp != null) spent = +r.spentThisApp; costLine();
        const msg = Object.assign({ role: 'assistant' }, r.message || { content: r.text, tool_calls: r.toolCalls }); if (!msg.tool_calls || !msg.tool_calls.length) delete msg.tool_calls;
        MSGS.push(msg); if (msg.content && String(msg.content).trim()) say('ai', msg.content);
        if (!msg.tool_calls) break;
        for (const tc of msg.tool_calls) {
          const nm = tc.function && tc.function.name; let args = {}, out;
          try { args = JSON.parse(tc.function.arguments || '{}'); if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('not a JSON object'); }
          catch (e) { out = `Error: the arguments for ${nm} were not valid JSON (${e.message}); nothing was run. Send them again as JSON.`; step(`${nm}: its arguments did not parse, so it was not run`, 'bad'); }
          if (out == null) {
            if (stopReq) out = 'Stopped by the user.';
            else {
              const t = TOOLS.find((x) => x.name === nm), s = step(`→ ${t ? t.done(args) : nm}`);
              try { out = JSON.stringify(runTool(nm, args)); s.classList.add('done'); } catch (e) { out = `Error: ${e.message || e}`; s.textContent += ` · failed: ${e.message || e}`; s.classList.add('bad'); }
            }
          }
          MSGS.push({ role: 'tool', tool_call_id: tc.id, content: cut(out, LIM) });
        }
      }
    } catch (e) { say('ai', `I could not reach the model: ${e.message || e}. ${onPlan() ? 'Check the ChatGPT sign-in under Prometheus' : 'Check the OpenRouter key and spending cap'} in [Settings](index.html#settings).`); }
    finally { thinking.remove(); busy = false; $('hlpSend').disabled = false; $('hlpStop').hidden = true; ctxChips(); }
  }
  $('hlpSend').addEventListener('click', () => ask($('hlpIn').value));
  $('hlpStop').addEventListener('click', () => { stopReq = true; });
  $('hlpIn').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask($('hlpIn').value); } });

  mode(); costLine();
  if (HAS_PSI()) readCfg().catch(() => { /* benign: the panel says so when opened */ });
  // For tests: the prompt and the tool handlers, no network.
  window.PSIHelpPrometheus = { open: () => open(true), close: () => open(false), ask, _test: { system, context, contents, sections, runTool, toolDefs, md, msgs: () => MSGS, cfg: (o) => Object.assign(CFG, o || {}) } };
})();
