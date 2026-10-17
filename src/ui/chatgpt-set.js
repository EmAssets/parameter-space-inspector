// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Settings > Prometheus > Run it on your ChatGPT plan (src/chatgpt.js). Sign-in happens on OpenAI's page in the user's
// browser; this page only learns whether it worked, the account's email and the models the plan offers. Choosing a model
// writes chatgpt/<slug> into Prometheus's model field, which routes its questions to the plan (config.external).
(function () {
  const $ = (id) => document.getElementById(id);
  if (!$('cgSignIn') || !window.psi || typeof psi.chatgpt !== 'function') return;
  const ELECTRON = /Electron/.test(navigator.userAgent);   // the desktop app opens the sign-in in the system browser itself
  const DEFAULT = 'anthropic/claude-haiku-4.5';
  let poll = null;
  const setModel = (id) => { const m = $('hlpModel'); if (!m) return; m.value = id; m.dispatchEvent(new Event('change')); show(); };
  function fill(models) {
    const sel = $('cgModel'), cur = ($('hlpModel') || {}).value || ''; sel.textContent = '';
    for (const m of models || []) { const o = document.createElement('option'); o.value = m.id; o.textContent = m.name; if (m.id === cur) o.selected = true; sel.appendChild(o); }
    $('cgUse').disabled = !sel.options.length;
  }
  async function show() {
    let st; try { st = await psi.chatgpt({ action: 'status' }); } catch (e) { $('cgStat').textContent = `Could not read the ChatGPT sign-in: ${e.message || e}`; return st; }
    const on = st.signedIn, cur = ($('hlpModel') || {}).value || '', using = /^chatgpt\//.test(cur);
    $('cgSignIn').hidden = on; $('cgSignOut').hidden = !on; $('cgManage').hidden = !on; $('cgPick').hidden = !(on && st.planUsage);
    $('cgSignIn').textContent = st.pending ? 'Start the sign-in again' : 'Continue with ChatGPT';
    $('cgBack').hidden = !using;
    const who = st.email || st.name || 'your ChatGPT account';
    $('cgStat').textContent = on
      ? `Signed in as ${who}. ${st.planUsage ? (using ? `Prometheus is using your ChatGPT plan (${cur.split('/').pop()}).` : 'Plan usage allowed: choose a model below to use it for Prometheus.') : 'Plan usage was not allowed at sign-in, so Prometheus cannot run on it: sign out and sign in again to allow it.'} Kept ${st.storage}.`
      : st.error ? `Not signed in: ${st.error}` : st.pending ? 'Waiting for you to finish signing in on OpenAI’s page. If no tab opened:' : 'Not signed in.';
    if (st.pending && !ELECTRON) { const a = document.createElement('a'); a.href = st.pending.url; a.target = '_blank'; a.rel = 'noopener'; a.textContent = ' Open the sign-in page'; $('cgStat').appendChild(a); }
    if (on && st.planUsage) fill(st.models || []);
    return st;
  }
  $('cgSignIn').addEventListener('click', async () => {
    // In a browser the new tab must open inside the click, before any await, or it is blocked as a pop-up.
    const tab = ELECTRON ? null : window.open('about:blank', '_blank');
    try {
      const r = await psi.chatgpt({ action: 'signin' });
      if (tab) { try { tab.opener = null; tab.location.href = r.url; } catch (_) { /* closed: the link in the status line works */ } }
    } catch (e) { if (tab) tab.close(); $('cgStat').textContent = `Sign-in could not start: ${e.message || e}`; return; }
    clearInterval(poll); const t0 = Date.now();
    poll = setInterval(async () => {
      const st = await show();
      if (!st || !st.pending || Date.now() - t0 > 610e3) {
        clearInterval(poll);
        if (st && st.signedIn && st.planUsage && !(st.models || []).length) { try { fill((await psi.chatgpt({ action: 'models' })).models); await show(); } catch (e) { $('cgStat').textContent += ` The model list could not be read: ${e.message || e}`; } }
      }
    }, 2000);
    show();
  });
  $('cgSignOut').addEventListener('click', async () => {
    try { await psi.chatgpt({ action: 'signout' }); if (/^chatgpt\//.test(($('hlpModel') || {}).value || '')) setModel(DEFAULT); } catch (e) { $('cgStat').textContent = `Sign-out failed: ${e.message || e}`; }
    show();
  });
  $('cgModels').addEventListener('click', async () => { $('cgModels').disabled = true; try { fill((await psi.chatgpt({ action: 'models' })).models); } catch (e) { $('cgStat').textContent = `The model list could not be read: ${e.message || e}`; } finally { $('cgModels').disabled = false; } });
  $('cgUse').addEventListener('click', () => { if ($('cgModel').value) setModel($('cgModel').value); });
  $('cgBack').addEventListener('click', () => setModel(DEFAULT));
  if ($('hlpModel')) $('hlpModel').addEventListener('change', () => setTimeout(show, 300));
  show();
})();
