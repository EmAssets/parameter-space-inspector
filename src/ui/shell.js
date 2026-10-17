// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The redesigned shell: view headings (with the nav label as a kicker), experiment headers with instrument
// chips, the lens and steering controls moved into Settings, the engine log into Settings, the narrow-screen
// "View ▾" drawer, and each view's own address. Run-state dots live in psi-ui.js (PSI.setRunning / setDone).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };

  const VIEWS = {
    generate: { h: 'Watch it answer, one fork at a time', sub: 'Every piece is a choice among others. Brighter means it was more torn; click any piece to see the roads it did not take.' },
    neural: { h: 'What each layer would say', sub: 'The same model seen from inside while it answers: the guess after every layer, the neurons that fired hardest, what each piece looked at.' },
    paths: { h: 'What one choice closed', sub: 'Early choices constrain what can follow, and answers settle into basins. Generate first, then pick a fork.' },
    read: { h: 'Where the model was surprised', sub: 'Paste any text. The model reads it one piece at a time; brighter means more surprised.' },
    tokens: { h: 'The pieces the model actually sees', sub: 'Text as the model receives it, and where each piece lives in its map of meaning.' },
    compare: { h: 'Same weights, two framings', sub: 'The same question asked with and without a framing, on the same weights.' },
    file: { h: 'What the file says about itself', sub: 'Read from the file header only; the weights stay on disk for the engine.' },
    lineage: { h: 'Seed & lineage', sub: 'Sign a seed, grow it, and compare it with real models.' },
    commons: { h: 'Commons', sub: 'Seed lines grown by many people: find and check recipes, build the model, contribute a branch, verify anyone’s claims.' },
    design: { h: 'Design an experiment', sub: 'Prometheus drafts a plan; you read it, edit it, and only you can run it. The app records, measures and draws every number itself.' },
    timeline: { h: 'What emerged when', sub: 'The order in which capabilities appear during training, in five model sizes; your models placed on the same axis; and a graft lab that puts one model’s layers into another.' },
    features: { h: 'Features', sub: 'What the model’s layers represent, as named features from a sparse autoencoder: see them fire on any text, search them, steer an answer with one, and ask which ones chose the next word.' },
    anatomy: { h: 'Inside the loaded model', sub: 'Its real layers, heads, neurons and the links between them, drawn from the weights; then a recorded forward pass you can play, pause and scrub, from reading the prompt to writing each token.' },
  };
  const EXPS = {
    understand: { t: 'Understanding', essay: 'Where Understanding Begins', q: 'Does the model understand, or only sound right?' },
    emergence: { t: 'Emergence', essay: 'Where Understanding Begins', q: 'When was your question learned, and what dies when the looking-back circuit is removed?' },
    lesion: { t: 'Tipping point', essay: 'The reverse of training', q: 'How much structure can be removed before the signs of intelligence go out?' },
    sensing: { t: 'Sensing surface', essay: 'The Sensing Surface', q: 'How does a framework absorb, meet an anomaly, and abstract?' },
    fight: { t: 'Fight class', essay: 'What Kind of Fight Is This?', q: 'Is today’s AI an Autopilot, or is it crossing into Strategist?' },
    connectors: { t: 'Connectors', essay: 'Language', q: 'Does the model hold one idea of “a connection”?' },
    ladder: { t: 'Storage ladder', essay: 'Abstraction', q: 'How high does a model store “red”: in words, neurons, directions, layout, wiring, combinations, backups, across labs, in its view of itself?' },
  };

  // A subtitle with its key terms explained on hover and focus (PSI.term).
  const TERMED = { neural: [['the guess after every layer', 'logit lens']], anatomy: [['heads', 'attention head']] };
  function sub(id, text) {
    const p = el('p', 'view-sub'); let rest = text;
    (TERMED[id] || []).forEach(([phrase, key]) => { const i = rest.indexOf(phrase); if (i < 0) return; p.append(rest.slice(0, i)); p.appendChild(PSI.term(phrase, key)); rest = rest.slice(i + phrase.length); });
    p.append(rest); return p;
  }
  // The nav label and its group, shown as a small kicker above each view's h1.
  const navOf = (id) => document.querySelector(`#tabs button[data-tab="${id}"]`);
  const labelOf = (b) => ((b && (b.querySelector('.label') || b)).textContent || '').replace(/\s+/g, ' ').trim();
  function groupOf(b) { let n = b && b.previousElementSibling; while (n && !n.classList.contains('nav-group')) n = n.previousElementSibling; return n ? n.firstChild.textContent.trim() : ''; }
  const kicker = (id) => { const b = navOf(id), g = groupOf(b); return el('p', 'kicker', g ? `${g} · ${labelOf(b)}` : labelOf(b)); };

  // ── headings ──────────────────────────────────────────────────────────────
  Object.entries(VIEWS).forEach(([id, v]) => {
    const sec = $(`tab-${id}`); if (!sec) return;
    const head = el('div', 'view-head'); head.appendChild(kicker(id)); head.appendChild(el('h1', null, v.h)); head.appendChild(sub(id, v.sub));
    if (id === 'neural') {   // the real depth once a model's file has been read
      const h1 = head.querySelector('h1'), set = (m) => { h1.textContent = m && m.layers ? `What each of the ${m.layers} layers would say` : v.h; };
      set(window.PSIModelMeta); document.addEventListener('psi-model-meta', (e) => set(e.detail));
    }
    sec.insertBefore(head, sec.firstChild);
    const first = head.nextElementSibling; if (first && first.matches('p.muted') && first.textContent.length < 400) first.classList.add('hidden');   // the old one-line intro, now the subtitle
  });
  Object.entries(EXPS).forEach(([id, x]) => {
    const sec = $(`tab-${id}`); if (!sec) return;
    const head = el('div', 'view-head exp-head');
    const k = el('p', 'kicker'); k.appendChild(document.createTextNode('Experiment')); const e = el('span', null, ` · from “${x.essay}”`); e.setAttribute('data-em', ''); k.appendChild(e); head.appendChild(k);
    head.appendChild(el('h1', null, x.t)); head.appendChild(el('p', 'exp-q', x.q));
    const chips = el('div', 'chips-row');
    [...sec.querySelectorAll('p.k')].filter((k) => /^\s*\d+\s*·/.test(k.textContent)).forEach((k) => {
      const m = k.textContent.match(/^\s*(\d+)\s*·\s*(.+)$/); if (!m) return;
      const c = el('button', 'chip-btn'); c.type = 'button'; c.appendChild(el('b', null, m[1])); c.appendChild(document.createTextNode(' ' + m[2].replace(/\s*\(.*\)\s*$/, '')));
      c.addEventListener('click', () => k.scrollIntoView({ behavior: 'smooth', block: 'start' })); chips.appendChild(c);
      k.classList.add('instrument-k');
    });
    if (chips.children.length) head.appendChild(chips);
    sec.insertBefore(head, sec.firstChild);
  });

  // ── the old sidebar: lens, weights and steering into Settings, engine log into Settings ──
  const stash = $('sideStash');
  if (stash) {
    const secs = [...stash.querySelectorAll(':scope > section')];
    const lensBox = $('setLens'), engBox = $('setEngine');
    secs.slice(0, 2).forEach((sct) => (lensBox || $('tab-generate')).appendChild(sct));
    if (secs[2]) { secs[2].classList.add('engine-box'); (engBox || $('tab-file')).appendChild(secs[2]); }
    stash.remove();
    const line = el('p', 'gen-status small muted'); line.id = 'genStatus';
    const gen = $('tab-generate'); gen.insertBefore(line, gen.querySelector('.view-head').nextSibling);
  }
  if ($('tab-settings')) { const h = el('div', 'view-head'); h.appendChild(el('p', 'kicker', 'Settings')); h.appendChild(el('h1', null, 'Settings')); h.appendChild(el('p', 'view-sub', 'Everything that applies across the app: which model answers and how much it can hold, how it is framed, the engine that looks inside it, this computer, and external models to compare with.')); $('tab-settings').insertBefore(h, $('tab-settings').firstChild); }

  // ── run-state dots and the "n of 7 run" counter: psi-ui.js, set explicitly by each view (PSI.setRunning, PSI.setDone) ──

  // ── the header's height (it wraps on narrow screens): the sticky nav and the message area sit below it ──
  const top = document.querySelector('header.top');
  if (top && window.ResizeObserver) new ResizeObserver(() => document.documentElement.style.setProperty('--top-h', `${Math.ceil(top.getBoundingClientRect().height)}px`)).observe(top);

  // ── narrow screens: the nav is a "View ▾" drawer ──
  const nav = $('tabs'), tog = $('navToggle');
  const drawer = (open) => { if (!tog) return; nav.classList.toggle('open', open); tog.setAttribute('aria-expanded', String(open)); };
  if (tog) {
    tog.addEventListener('click', () => drawer(!nav.classList.contains('open')));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && nav.classList.contains('open')) { drawer(false); tog.focus(); } });
    document.addEventListener('click', (e) => { if (nav.classList.contains('open') && !nav.contains(e.target) && !tog.contains(e.target)) drawer(false); });
  }
  nav.addEventListener('click', (e) => { const b = e.target.closest('button[data-tab]'); if (!b) return; drawer(false); if ($('navToggleName')) $('navToggleName').textContent = labelOf(b); });
  // Ask Prometheus, always in the nav: explains itself and how to set it up when it is not on yet (helper.js).
  if ($('navAsk')) $('navAsk').addEventListener('click', () => { drawer(false); if (window.PSIHelper) window.PSIHelper.open(); });

  // each view opens at its top
  $('tabs').addEventListener('click', (e) => { if (e.target.closest('button[data-tab]')) window.scrollTo({ top: 0 }); });

  // ── every view has its own address (#anatomy, #fight…): refresh, share and back/forward keep the view ──
  const BASE_TITLE = document.title || 'Parameter Space Inspector';
  let routing = false;
  const nameOf = labelOf;
  $('tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-tab]'); if (!b) return;
    document.title = `${nameOf(b)} · ${BASE_TITLE}`;
    if (routing) return;
    const h = `#${b.dataset.tab}`; if (location.hash !== h) history.pushState({ view: b.dataset.tab }, '', h);
  });
  function route() {
    let v = decodeURIComponent(location.hash.slice(1));
    if (v === 'paths') { v = 'generate'; history.replaceState({ view: v }, '', '#generate'); setTimeout(() => { const s = $('genPaths'); if (s) s.scrollIntoView({ block: 'start' }); }, 30); }   // Paths & attractors is now part of Generate
    let b = v && document.querySelector(`#tabs button[data-tab="${CSS.escape(v)}"]`);
    // A view hidden with Emergence Machine content (Settings > Content) opens Generate instead.
    if (b && b.hasAttribute('data-em') && document.documentElement.classList.contains('em-off')) { history.replaceState({ view: 'generate' }, '', '#generate'); b = document.querySelector('#tabs button[data-tab="generate"]'); }
    if (!b || b.getAttribute('aria-current') === 'page') return;
    routing = true; try { b.click(); } finally { routing = false; }
  }
  addEventListener('popstate', route); addEventListener('hashchange', route);
  // As soon as every view's script has run and wired its own tab listener (they all run before DOMContentLoaded), not
  // at 'load', which also waits for images and anything else still arriving.
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', () => setTimeout(route, 0)); else setTimeout(route, 0);

  // ⌘/Ctrl+Enter generates from anywhere in Generate
  document.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !$('tab-generate').classList.contains('hidden')) { const g = $('genGo'); if (g && !g.disabled) g.click(); } });
})();
