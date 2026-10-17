// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// The push panel, mounted where the model is made (Tipping point): PSIHf.mount(box, {folder, editable}) shows the folder,
// a repo field (owner from Settings > Hugging Face), Prepare (the plan: files, size, card; nothing is sent) and Push now,
// which is the user's confirmation. The account and defaults are configured in Settings; pushing happens here.
(function () {
  if (!window.psi || typeof psi.hf !== 'function') return;
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const stat = (node, text, kind) => PSI.status(node, text, kind);
  let ST = null;

  const slug = (f) => String(f || '').replace(/[\\/]+$/, '').split(/[\\/]/).pop().replace(/[^A-Za-z0-9._-]+/g, '-').slice(0, 96);
  function mount(box, opt = {}) {
    box.textContent = '';
    const row = el('div', 'row');
    const folder = el('input', 'grow'); folder.placeholder = 'model folder (config.json + .safetensors), full path'; folder.setAttribute('aria-label', 'model folder to push');
    folder.value = opt.folder || ''; folder.hidden = !!opt.folder && !opt.editable;
    const repo = el('input', 'grow'); repo.placeholder = 'owner/name on Hugging Face'; repo.setAttribute('aria-label', 'Hugging Face repo');
    const priv = el('input'); priv.type = 'checkbox';
    const lab = el('label', 'small muted'); lab.append(priv, ' private');
    const prep = el('button', 'ghost', 'Prepare the push'), go = el('button', 'hf-send', 'Push now'), stop = el('button', 'ghost', 'Stop');
    go.hidden = true; stop.hidden = true;
    const say = el('span', 'small muted'), out = el('div', 'out minh-0');
    row.append(folder, repo, lab, prep, go, stop); box.append(row, say, out);
    const fillRepo = () => { if (!repo.value && folder.value) repo.value = `${(ST && ST.owner) || 'your-name'}/${slug(folder.value)}`; };
    psi.hf({ action: 'status' }).then((s) => { ST = s; priv.checked = s.private !== false; fillRepo(); if (!s.token) stat(say, 'Add a Hugging Face token in Settings > Hugging Face first.', 'warn'); }).catch(() => {});
    folder.addEventListener('change', () => { go.hidden = true; fillRepo(); });
    repo.addEventListener('input', () => { go.hidden = true; }); priv.addEventListener('change', () => { go.hidden = true; });
    let plan = null;
    prep.addEventListener('click', async () => {
      go.hidden = true; out.textContent = ''; stat(say, 'reading the folder…', 'busy');
      try { plan = await psi.hf({ action: 'plan', folder: folder.value.trim(), repo: repo.value.trim() }); }
      catch (e) { return stat(say, `Cannot push: ${e.message || e}`, 'error'); }
      out.append(el('p', 'small', `To https://huggingface.co/${plan.repo} as a ${priv.checked ? 'PRIVATE' : 'PUBLIC'} repo (if it already exists, its visibility is kept and these files are added as a new commit): ${plan.files.length} files, ${plan.gb} GB.`));
      out.append(el('p', 'small muted', `Sent: ${plan.files.join(', ')}`));
      if (plan.left_out.length) out.append(el('p', 'small muted', `Not sent: ${plan.left_out.join(', ')}`));
      if (plan.card) { const d = el('details'); d.append(el('summary', 'small', 'the model card it will get (README.md)'), el('pre', 'small', plan.card)); out.append(d); }
      if (!priv.checked) out.append(el('p', 'warn small', 'Public: anyone can download it, and copies cannot be called back. The base model’s licence applies to it.'));
      stat(say, 'Nothing has been sent. Press Push now to send exactly this.', 'warn'); go.hidden = false;
    });
    go.addEventListener('click', async () => {
      if (!plan) return; go.hidden = true; prep.disabled = true;
      let j; try { j = await psi.hf({ action: 'push', folder: plan.folder, repo: plan.repo, private: priv.checked, confirm: true }); }
      catch (e) { prep.disabled = false; return stat(say, `Not pushed: ${e.message || e}`, 'error'); }
      stop.hidden = false;
      const tick = async () => {
        try { j = await psi.hf({ action: 'job' }); } catch (_) { return setTimeout(tick, 3000); }
        if (j && j.state === 'running') { stat(say, `${j.stage} · ${j.seconds}s`, 'busy'); return setTimeout(tick, 2000); }
        stop.hidden = true; prep.disabled = false;
        if (j && j.state === 'done') {
          stat(say, '', ''); out.textContent = '';
          const p = el('p', 'ok small', `Pushed (${j.result.private ? 'private' : 'public'}, ${j.result.files} files): `); const a = el('a', null, j.result.url); a.href = j.result.url; a.target = '_blank'; a.rel = 'noopener'; p.append(a); out.append(p);
        } else stat(say, j && j.state === 'cancelled' ? 'Stopped. Files already uploaded may sit unreferenced on Hugging Face; nothing was committed.' : `Push failed: ${(j && j.error) || 'unknown'}`, 'error');
      };
      tick();
    });
    stop.addEventListener('click', () => psi.hf({ action: 'cancel' }).catch(() => {}));
  }
  window.PSIHf = { mount };
})();
