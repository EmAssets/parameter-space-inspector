// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Emergence Machine content on or off (Settings > General, settings.em.on; off by default). Off, PSI is a plain
// AI-inspection tool:
//   - html.em-off hides everything marked data-em (the E-level lens and its settings, Fight class, Connectors, the
//     Lexicon instrument, essay tags, links to the framework's essays)
//   - an element with data-plain="text" shows that text instead of its own; a textarea with data-plain-value gets that
//     value while it still holds its Emergence Machine example
//   - the assistant is "Assistant", not "Prometheus": the name is changed in the page's text as it appears (never inside
//     text boxes, code or model output), and its instructions and tools leave the framework out (helper.js)
// window.PSIEM.on says which; the 'psi-em' event fires when it changes. The brand in the top bar, the credits and the
// signing trust root stay either way.
(() => {
  'use strict';
  const KEY = 'psi-em-on';
  let on = false;
  try { on = localStorage.getItem(KEY) === '1'; } catch (_) { /* no storage: off, until the settings say */ }
  const SKIP = 'textarea, input, select, script, style, code, pre, [contenteditable], #ribbon, .ribbon, [data-no-rename], .cm-story';
  const ORIG = new WeakMap();   // text node → its text before the rename, to put back when content is turned on again
  const RX = /\bPrometheus\b/g;

  function rename(root) {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
      const p = n.parentElement; if (!p || p.closest(SKIP)) continue;
      if (!on) { if (RX.test(n.nodeValue)) { RX.lastIndex = 0; if (!ORIG.has(n)) ORIG.set(n, n.nodeValue); n.nodeValue = n.nodeValue.replace(RX, 'Assistant'); } RX.lastIndex = 0; }
      else if (ORIG.has(n)) { n.nodeValue = ORIG.get(n); ORIG.delete(n); }
    }
    for (const a of ['title', 'aria-label', 'placeholder']) for (const el of root.querySelectorAll ? root.querySelectorAll(`[${a}*="Prometheus"], [data-em-${a}]`) : []) {
      if (!el.hasAttribute(`data-em-${a}`)) el.setAttribute(`data-em-${a}`, el.getAttribute(a));
      const o = el.getAttribute(`data-em-${a}`); el.setAttribute(a, on ? o : o.replace(RX, 'Assistant'));
    }
  }
  function swap() {
    for (const el of document.querySelectorAll('[data-plain]')) {
      if (!el.hasAttribute('data-em-text')) el.setAttribute('data-em-text', el.textContent);
      el.textContent = on ? el.getAttribute('data-em-text') : el.getAttribute('data-plain');
    }
    for (const el of document.querySelectorAll('[data-plain-value]')) {
      if (!el.hasAttribute('data-em-value')) el.setAttribute('data-em-value', el.defaultValue || el.value);
      const em = el.getAttribute('data-em-value'), plain = el.getAttribute('data-plain-value');
      if (!on && el.value === em) el.value = plain; else if (on && el.value === plain) el.value = em;   // never over the user's own text
    }
  }
  let obs = null;
  function apply() {
    document.documentElement.classList.toggle('em-off', !on);
    if (!document.body) return;
    swap(); rename(document.body);
    // A view that is hidden now (an old link, a reload on it) gives way to Generate.
    const v = location.hash.slice(1).split(/[?&]/)[0];
    if (!on && v && document.querySelector(`#tabs button[data-tab="${CSS.escape(v)}"][data-em]`)) { if (window.PSI && PSI.goto) PSI.goto('generate'); else location.hash = 'generate'; }
    if (!on && !obs) {
      obs = new MutationObserver((ms) => { for (const m of ms) for (const n of m.addedNodes) { if (n.nodeType === 1) rename(n); else if (n.nodeType === 3 && n.parentElement) rename(n.parentElement); } });
      obs.observe(document.body, { childList: true, subtree: true });
    } else if (on && obs) { obs.disconnect(); obs = null; }
  }
  function set(v) {
    const was = on; on = v !== false;
    try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (_) { /* per-browser memory only */ }
    apply(); if (was !== on) window.dispatchEvent(new CustomEvent('psi-em', { detail: { on } }));
  }
  window.PSIEM = { get on() { return on; }, name: () => (on ? 'Prometheus' : 'Assistant'), set, apply };
  document.documentElement.classList.toggle('em-off', !on);   // before the first paint, from the last known value
  const start = () => { apply(); if (window.psi && psi.settings) psi.settings().then((s) => set(!(s.em && s.em.on === false))).catch(() => { /* keep the last known value */ }); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
