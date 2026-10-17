// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Background work in the header: while the job runner (src/jobs.js) runs or queues an experiment, or an SAE trains
// (Features), the machine strip shows a looping snake and how far it has got. The strip already opens Settings >
// This computer; there, under the meters, every job is listed with its details and its controls.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const strip = $('gpuChip'); if (!strip || !psi.jobs) return;
  const ACTIVE = new Set(['queued', 'waiting', 'running']);

  // ── beside the strip: a snake running round a square while work is going on; tapping it opens the job details ──
  const m = el('button', 'ghost job-snake'); m.type = 'button'; m.id = 'jobSnake'; m.hidden = true;
  m.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect class="js-track" x="3" y="3" width="18" height="18" rx="5" pathLength="100"/><rect class="js-snake" x="3" y="3" width="18" height="18" rx="5" pathLength="100"/></svg>';
  const v = el('span', 'js-v'); m.appendChild(v); strip.before(m);
  m.addEventListener('click', () => { PSI.goto('settings', 'setJobs'); setTimeout(() => { const b = $('setJobs'); if (b) b.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 80); });

  // ── the list under the meters (Settings > This computer) ──
  const box = el('div', 'js-box'); box.id = 'setJobs';
  box.append(el('h3', 'k', 'jobs · experiments running without a screen, and SAE training'));
  const list = el('div', 'js-list'); box.appendChild(list);
  const place = () => { const mb = $('setMachine'); if (mb && !box.isConnected) { const l = mb.querySelector('.gpu-list'); if (l) l.after(box); else mb.appendChild(box); } };

  const ago = (t) => { if (!t) return ''; const s = Math.max(0, (Date.now() - Date.parse(t)) / 1000); return s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`; };
  const STATE = { queued: 'queued', waiting: 'waiting for the GPU', running: 'running', done: 'done', stopped: 'stopped', failed: 'failed', interrupted: 'interrupted' };
  function row(j) {
    const r = el('div', `js-row ${ACTIVE.has(j.status) ? 'on' : ''}`);
    const head = el('div', 'js-head'); head.appendChild(el('b', null, j.name)); head.appendChild(el('span', `small js-state js-${j.status}`, STATE[j.status] || j.status)); r.appendChild(head);
    const bar = el('div', 'js-bar'); const fill = el('i'); fill.style.width = `${j.n ? Math.round((j.done / j.n) * 100) : 0}%`; bar.appendChild(fill); r.appendChild(bar);
    const bits = [`${j.done} of ${j.n} prompts`, j.instrument, j.cached ? `${j.cached} from the cache` : null,
      j.started ? `started ${ago(j.started)} ago` : `sent ${ago(j.at)} ago`, j.finished ? `ended ${ago(j.finished)} ago` : null,
      j.owner ? `in the ${j.owner.label}` : null].filter(Boolean);
    r.appendChild(el('p', 'small muted', bits.join(' · ')));
    if (j.wait) r.appendChild(el('p', 'small js-wait', `Waiting: ${j.wait}`));
    if (j.error) r.appendChild(el('p', 'small cond-warn', j.error));
    if (j.tip) r.appendChild(el('p', `small ${j.tip.ok ? 'muted' : 'cond-warn'}`, j.tip.msg));
    const ctl = el('div', 'row');
    const b = (label, fn, cls = 'ghost') => { const x = el('button', `${cls} js-btn`, label); x.type = 'button'; x.addEventListener('click', async () => { x.disabled = true; try { await fn(); } catch (e) { PSI.err(e); } finally { x.disabled = false; poll(true); } }); ctl.appendChild(x); };
    if (j.record) b(ACTIVE.has(j.status) ? 'Follow it' : 'Open the report', async () => { PSI.goto('design'); if (window.PSIDesign && PSIDesign.open) await PSIDesign.open(j.record); });
    if (j.status === 'waiting') b('Run anyway', () => psi.jobs({ action: 'run_anyway', id: j.id }));
    if (ACTIVE.has(j.status)) b('Stop', () => psi.jobs({ action: 'stop', id: j.id }));
    if (['interrupted', 'failed', 'stopped'].includes(j.status) && j.done < j.n) b('Resume', () => psi.jobs({ action: 'resume', id: j.id }));
    r.appendChild(ctl); return r;
  }
  function saeRow(s) {
    const r = el('div', 'js-row on'), p = s.progress || {};
    const head = el('div', 'js-head'); head.appendChild(el('b', null, `Training an SAE · layer ${s.layer}`)); head.appendChild(el('span', 'small js-state js-running', p.stage || 'starting')); r.appendChild(head);
    const bar = el('div', 'js-bar'); const fill = el('i'); fill.style.width = `${p.of ? Math.round((p.done / p.of) * 100) : 0}%`; bar.appendChild(fill); r.appendChild(bar);
    r.appendChild(el('p', 'small muted', [p.note, p.fvu != null ? `unexplained ${(p.fvu * 100).toFixed(0)}%` : null, `started ${ago(s.started)} ago`, 'both engines are stopped until it ends'].filter(Boolean).join(' · ')));
    const ctl = el('div', 'row'), go = el('button', 'ghost js-btn', 'Open Features'), stop = el('button', 'ghost js-btn', 'Cancel'); go.type = stop.type = 'button';
    go.addEventListener('click', () => PSI.goto('features')); stop.addEventListener('click', () => psi.sae({ action: 'cancel' }).catch(PSI.err));
    ctl.append(go, stop); r.appendChild(ctl); return r;
  }

  let timer = null, seen = '';
  async function poll(now) {
    clearTimeout(timer);
    let jobs = [], sae = null;
    try { jobs = await psi.jobs({ action: 'list' }); } catch (_) { /* not available: the meter stays hidden */ }
    try { if (psi.sae) { const s = await psi.sae({ action: 'train_status' }); if (s.training) sae = s; } } catch (_) { /* no Features in this build */ }
    const act = jobs.filter((j) => ACTIVE.has(j.status)), run = act.find((j) => j.status === 'running') || act[0];
    const busy = !!(run || sae); m.hidden = !busy; m.classList.toggle('waiting', !!run && run.status === 'waiting' && !sae);
    if (sae && !run) { const p = sae.progress || {}; v.textContent = `SAE ${p.of ? `${Math.round((p.done / p.of) * 100)}%` : '…'}`; m.title = `Training an SAE for layer ${sae.layer}: ${p.note || 'starting'}. Tap for details.`; m.setAttribute('aria-label', m.title); }
    else if (run) { v.textContent = `${run.done}/${run.n}${act.length > 1 ? ` +${act.length - 1}` : ''}`; m.title = `${STATE[run.status]}: ${run.name} (${run.done} of ${run.n})${run.wait ? `. ${run.wait}` : ''}${act.length > 1 ? `; ${act.length - 1} more queued` : ''}${sae ? '; an SAE is training too' : ''}. Tap for details.`; m.setAttribute('aria-label', m.title); }
    place();
    const sig = JSON.stringify([jobs.slice(0, 8), sae && [sae.layer, sae.progress]]);
    if (sig !== seen) {
      seen = sig; list.textContent = '';
      if (sae) list.appendChild(saeRow(sae));
      jobs.slice(0, 8).forEach((j) => list.appendChild(row(j)));
      if (!jobs.length && !sae) list.appendChild(el('p', 'small muted', 'No jobs yet. An experiment sent with “Run in the background” runs here, in the app, and carries on when its screen closes.'));
    }
    timer = setTimeout(poll, busy ? (document.hidden ? 20000 : 3000) : (document.hidden ? 60000 : 15000));
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(true); });
  poll(true);
  window.PSIJobs = { refresh: () => poll(true) };
})();
