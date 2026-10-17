// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Settings > Guardrails: the sizes and limits the user may change (config.js GUARD gives each one's bounds, and checks
// every value again when it is saved), beside the protections that are not settings. Prometheus cannot change these
// (helper.js keeps it off every Settings control) and outside agents have no settings tool.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const FIXED = [
    'An analysis runs in a sandbox with no network, no files and nothing loaded, and its code is shown beside its result.',
    'Only you run a plan: Prometheus proposes it, you read it, edit it and press Run. It cannot press anything in the designer.',
    'A plan is data in one checked shape (known keys, bounded sizes, a closed list of measures); nothing in it runs as code.',
    'An SAE trains only from your press in Features (it holds the GPU for minutes), never from Prometheus or an agent.',
    'A Tipping-point cut is undone at the end of its run and the engine is asked to confirm; a cut of yours is never undone for you.',
    'Rings are signed with the app’s own key; the online check sends ring records only, never an SAE or a model.',
  ];
  const block = el('div', 'set-block'); block.id = 'setGuardrails'; block.tabIndex = -1;
  block.append(el('h2', null, 'Guardrails'), el('p', 'muted small', 'Sizes and limits for experiments, jobs, analyses and features. Each has bounds the app checks again when it saves. Raising them makes larger runs possible, and larger records, longer waits and more GPU time with them.'));
  const form = el('div', 'gr-form'), row = el('div', 'row'), save = el('button', null, 'Save the guardrails'), reset = el('button', 'ghost', 'Back to the defaults'), st = el('span', 'k m-0');
  save.type = reset.type = 'button'; save.id = 'grSave'; reset.id = 'grReset'; row.append(save, reset, st);
  const fx = el('details', 'small gr-fixed'); fx.appendChild(el('summary', null, 'not settings: the protections that always hold')); const ul = el('ul'); FIXED.forEach((t) => ul.appendChild(el('li', null, t))); fx.appendChild(ul);
  block.append(form, row, fx);
  const after = $('setAgents'); if (after) after.after(block);

  let B = {}, V = {};
  function draw() {
    form.textContent = ''; let group = '';
    for (const [k, d] of Object.entries(B)) {
      if (d.group !== group) { group = d.group; form.appendChild(el('p', 'k gr-group', group)); }
      const r = el('div', 'row gr-row'), lab = el('label', 'small', d.label); lab.htmlFor = `gr_${k}`;
      let inp;
      if (d.bool) { inp = el('input'); inp.type = 'checkbox'; inp.checked = !!V[k]; const sw = el('label', 'switch small'); sw.append(inp, ` ${d.label}`); inp.id = `gr_${k}`; r.appendChild(sw); }
      else { inp = el('input'); inp.type = 'number'; inp.id = `gr_${k}`; inp.min = d.min; inp.max = d.max; inp.step = d.int ? 1 : 0.5; inp.value = V[k]; inp.className = 'gr-num'; r.append(inp, lab, el('span', 'small muted', `${d.min.toLocaleString()} to ${d.max.toLocaleString()}${V[k] !== d.def ? ` · default ${d.def.toLocaleString()}` : ''}`)); }
      if (d.note) r.appendChild(el('span', 'small muted gr-note', d.note));
      form.appendChild(r);
    }
  }
  async function load() { try { const v = await psi.settings(); B = v.guardrailBounds || {}; V = v.guardrails || {}; draw(); } catch (e) { form.textContent = `The guardrails could not be read: ${PSI.explain(e).text}`; } }
  function read() {
    const out = {};
    for (const [k, d] of Object.entries(B)) {
      const n = $(`gr_${k}`); if (!n) continue;
      if (d.bool) out[k] = n.checked;
      else { const x = Number(n.value); if (!Number.isFinite(x) || x < d.min || x > d.max || (d.int && !Number.isInteger(x))) throw new Error(`“${d.label}” must be ${d.int ? 'a whole number' : 'a number'} from ${d.min.toLocaleString()} to ${d.max.toLocaleString()}.`); out[k] = x; }
    }
    return out;
  }
  async function store(g, msg) {
    try { const v = await psi.settings({ guardrails: g }); V = v.guardrails; draw(); PSI.status(st, msg, 'ok'); window.dispatchEvent(new Event('psi-guardrails')); }
    catch (e) { PSI.status(st, PSI.explain(e).text, 'error'); }
  }
  save.addEventListener('click', () => { let g; try { g = read(); } catch (e) { PSI.status(st, e.message, 'error'); return; } store(g, 'Saved: they apply from the next plan, job, analysis or training.'); });
  reset.addEventListener('click', () => store(Object.fromEntries(Object.entries(B).map(([k, d]) => [k, d.def])), 'Back to the defaults.'));
  load();
})();
