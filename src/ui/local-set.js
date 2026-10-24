// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Settings > Prometheus > Run it on a tester model on this computer (src/local-llm.js). Lists the local servers that
// answer (the isolated tester first) and their models. Choosing one writes local/<endpoint>/<model> into Prometheus's
// model field, or into the judge, which routes their questions to that server (config.external).
(function () {
  const $ = (id) => document.getElementById(id);
  if (!$('lcCheck') || !window.psi || typeof psi.external !== 'function') return;
  const opt = (value, text) => { const o = document.createElement('option'); o.value = value; o.textContent = text; return o; };

  async function check() {
    $('lcCheck').disabled = true; $('lcStat').textContent = 'asking…';
    try {
      const st = await psi.external({ action: 'local_status' }), sel = $('lcModel'), cur = ($('hlpModel') || {}).value || '';
      sel.textContent = '';
      for (const e of st.endpoints) for (const m of e.models || []) {
        if (/embed/i.test(m)) continue;
        const id = `local/${e.name}/${m}`;
        sel.appendChild(opt(id, `${m} · ${e.name}${e.isolated ? ' (isolated tester)' : ''}`));
        if (id === cur) sel.value = id;
      }
      $('lcStat').textContent = st.endpoints.map((e) => `${e.name}: ${e.up ? `${(e.models || []).length} model${(e.models || []).length === 1 ? '' : 's'}` : e.error}`).join(' · ');
      $('lcPick').hidden = !sel.options.length;
      if (!sel.options.length) $('lcStat').textContent += ' — start the tester (tools/tester-model.sh start) and pull a model.';
      // Offered in the model field too, beside the OpenRouter suggestions.
      const dl = $('hlpModels'); if (dl) { dl.querySelectorAll('option[data-local]').forEach((o) => o.remove()); [...sel.options].forEach((o) => { const d = opt(o.value, o.textContent); d.dataset.local = '1'; dl.prepend(d); }); }
    } catch (e) { $('lcStat').textContent = `Could not ask: ${e.message || e}`; }
    finally { $('lcCheck').disabled = false; }
  }

  $('lcCheck').addEventListener('click', check);
  $('lcUse').addEventListener('click', () => { const m = $('hlpModel'), id = $('lcModel').value; if (!m || !id) return; m.value = id; m.dispatchEvent(new Event('change')); $('lcStat').textContent = `Prometheus now runs on ${id.slice(6)}.`; });
  $('lcJudge').addEventListener('click', async () => {
    const id = $('lcModel').value; if (!id) return;
    try { await psi.settings({ external: { judge: id } }); $('lcStat').textContent = `The judge is now ${id.slice(6)}.`; const j = $('stJudge'); if (j) { if (![...j.options].some((o) => o.value === id)) j.appendChild(opt(id, id)); j.value = id; } }
    catch (e) { $('lcStat').textContent = `Could not set the judge: ${e.message || e}`; }
  });
  check();
})();
