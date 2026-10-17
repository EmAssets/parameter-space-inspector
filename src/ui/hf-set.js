// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Settings > Hugging Face (src/hf-api.js): configuration only. The token (entered in this block, handled by commons.js,
// the same secret Commons publishes with) and the defaults (settings.hf) other views read. Every Hugging Face control in
// the app carries data-hf and stays hidden until the token is saved and Hugging Face has confirmed it: this script
// checks it (once on load when it has not been, and after every save) and then sets html.hf-on and sends 'psi-hf'.
(function () {
  const $ = (id) => document.getElementById(id);
  if (!$('setHf') || !window.psi || typeof psi.hf !== 'function') return;
  const stat = (node, text, kind) => PSI.status(node, text, kind);
  let ST = null, tried = false;

  function apply(st) {
    ST = st; const on = !!(st && st.ready);
    const was = document.documentElement.classList.contains('hf-on');
    document.documentElement.classList.toggle('hf-on', on);
    window.PSIHfState = st;
    if (was !== on) document.dispatchEvent(new CustomEvent('psi-hf', { detail: st }));
  }
  async function show() {
    let st; try { st = await psi.hf({ action: 'status' }); } catch (e) { stat($('hfWhoStat'), `Could not read the Hugging Face settings: ${e.message || e}`, 'error'); return; }
    apply(st);
    $('hfOwner').value = st.owner || ''; $('hfPrivate').checked = st.private !== false; $('hfWho').disabled = !st.token;
    if ($('hfCkptRepo')) $('hfCkptRepo').value = st.checkpoints || '';
    if (!st.token) return stat($('hfWhoStat'), 'No token yet. With one, the Hugging Face parts of the app appear: getting models, checkpoints, SAEs, adapters and text, and pushing.', '');
    if (st.ready) return stat($('hfWhoStat'), `Working: account ${st.account}${st.orgs && st.orgs.length ? ` (organisations: ${st.orgs.join(', ')})` : ''} · ${st.can_write ? 'can push' : 'read only: pushes will be refused'} · checked ${new Date(st.checked_at).toLocaleString()}`, 'ok');
    if (!tried) { tried = true; return check(); }
    stat($('hfWhoStat'), 'Hugging Face has not confirmed this token, so its parts of the app stay hidden. Check it again, or save a new one.', 'warn');
  }
  async function check() {
    stat($('hfWhoStat'), 'asking Hugging Face whether the token works…', 'busy');
    try { await psi.hf({ action: 'whoami' }); } catch (e) { stat($('hfWhoStat'), `Not working: ${e.message || e}. The Hugging Face parts of the app stay hidden.`, 'error'); apply({ ...(ST || {}), ready: false }); return; }
    show();
  }
  $('hfWho').addEventListener('click', () => { tried = true; check(); });
  $('hfSave').addEventListener('click', async () => {
    try { await psi.settings({ hf: { owner: $('hfOwner').value.trim(), private: $('hfPrivate').checked } }); stat($('hfStat'), 'saved', 'ok'); show(); }
    catch (e) { stat($('hfStat'), `Not saved: ${e.message || e}`, 'error'); }
  });
  if ($('hfCkptSave')) $('hfCkptSave').addEventListener('click', async () => {
    try { await psi.settings({ hf: { checkpoints: $('hfCkptRepo').value.trim() } }); stat($('hfCkptStat'), 'saved: Emergence can fetch its training steps', 'ok'); document.dispatchEvent(new CustomEvent('psi-hf-ckpt')); }
    catch (e) { stat($('hfCkptStat'), `Not saved: ${e.message || e}`, 'error'); }
  });
  // The token row is commons.js's (Save / Remove): check a new token at once.
  if ($('cmTokenSave')) $('cmTokenSave').addEventListener('click', () => setTimeout(() => { tried = false; show(); }, 700));
  if ($('cmTokenClear')) $('cmTokenClear').addEventListener('click', () => setTimeout(() => { tried = true; show(); }, 700));
  // Settings > Models > Import from Hugging Face, before a token works: take the reader to the token row.
  if ($('hfGoSet')) $('hfGoSet').addEventListener('click', () => {
    const b = $('setHf'); b.scrollIntoView({ block: 'start', behavior: 'smooth' });
    b.classList.add('hlp-flash'); setTimeout(() => b.classList.remove('hlp-flash'), 1800);
    setTimeout(() => { const t = $('cmToken'); if (t) t.focus({ preventScroll: true }); }, 500);
  });
  window.PSIHfRefresh = show;
  const tab = $('tab-settings');
  if (tab) new MutationObserver(() => { if (!tab.classList.contains('hidden')) show(); }).observe(tab, { attributes: true, attributeFilter: ['class'] });
  show();
})();
