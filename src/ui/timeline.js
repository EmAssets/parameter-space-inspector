// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Timeline view: what emerged at which point of training (window.E8_TIMELINE, from research/emergence_timeline.py),
// any model placed on it with the same tests (src/place_model.py through src/timeline-api.js), what emerged nearby,
// and the graft lab (src/graft.py): another model's layers put in place of the host's, measured against the gap and
// a random control. Every result carries a signed receipt; the view can check it.
(() => {
  'use strict';
  const T = window.E8_TIMELINE; if (!T) return;
  // The framework's name for a row is dropped while Emergence Machine content is off (Settings > Content).
  for (const e of T.events) if (/^E-level /.test(e.label)) { const em = e.label, plain = em.replace(/^E-level a/, 'A'); Object.defineProperty(e, 'label', { get: () => (!window.PSIEM || PSIEM.on ? em : plain) }); }
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const VIEW = 'timeline', TPS = T.meta.tokens_per_step;
  const COL = { '70m': '#7aa2ff', '160m': '#6fd3c1', '410m': '#ffc36b', '1b': '#ff8f8f', '2.8b': '#c9a0ff' };
  const EV = Object.fromEntries(T.events.map((e) => [e.id, e]));
  const tok = (step) => (step * TPS) / 1e9;   // billions of tokens
  const fmtTok = (b) => (b >= 100 ? `${b.toFixed(0)} B` : b >= 10 ? `${b.toFixed(0)} B` : b >= 1 ? `${b.toFixed(1)} B` : `${(b * 1000).toFixed(0)} M`);
  const fmtStep = (s) => Number(s).toLocaleString();
  const SHOW = new Set(T.sizes.map((s) => s.id)), SHOWN = new Set();
  let PLACED = [], SEL_EV = null, SEL_P = null, MODELS = [], POLL = null;

  // ── 1. the chart ──────────────────────────────────────────────────────────
  const X0 = Math.log10(0.1), X1 = Math.log10(400);
  function draw() {
    const cv = $('tlChart'); if (!cv || !cv.clientWidth) return;
    const r = Math.min(2, devicePixelRatio || 1), W = cv.clientWidth, LH = 14, LANE = 20, RH = LH + LANE, NP = PLACED.filter((p) => SHOWN.has(p.id) && p.nearest).length, TOP = 58 + NP * 12, BOT = 8, H = TOP + T.events.length * RH + BOT;
    cv.width = W * r; cv.height = H * r; cv.style.height = `${H}px`;
    const g = cv.getContext('2d'); g.scale(r, r); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const L = 6, R = W - 14, x = (b) => L + ((Math.log10(Math.max(b, 0.1)) - X0) / (X1 - X0)) * (R - L);
    // the phase shift: where the most densely measured size had three or more emergences at once
    for (const [ci, c] of (T.clusters['410m'] || []).filter((c) => c.events.length >= 3).entries()) {
      const a = x(tok(c.from)), b = x(tok(c.to)); g.fillStyle = 'rgba(255,195,107,0.09)'; g.fillRect(a, 26, b - a, H - 26);
      g.fillStyle = '#ffc36b'; g.font = '10px "IBM Plex Mono", monospace'; g.fillText(`${c.events.length} at once (410m)`, Math.min(a + 3, W - 110), 36 + (ci % 2) * 11);
    }
    g.font = '10px "IBM Plex Mono", monospace';
    for (const b of [0.1, 1, 10, 100]) {
      g.strokeStyle = 'rgba(255,255,255,0.12)'; g.beginPath(); g.moveTo(x(b), TOP - 4); g.lineTo(x(b), H - BOT); g.stroke();
      g.fillStyle = '#8a877f'; g.fillText(`${fmtTok(b)} tokens`, x(b) + 2, 11); g.fillStyle = '#5f5c56'; g.fillText(`step ${fmtStep(Math.round(b * 1e9 / TPS))}`, x(b) + 2, 22);
    }
    const sizes = T.sizes.map((s) => s.id).filter((s) => SHOW.has(s)), off = (k) => (k - (sizes.length - 1) / 2) * 3.2;
    T.events.forEach((e, i) => {
      const y0 = TOP + i * RH;
      if (SEL_EV === e.id) { g.fillStyle = 'rgba(255,255,255,0.06)'; g.fillRect(0, y0 - 2, W, RH); }
      g.fillStyle = '#d8d6d0'; g.font = '12px "Public Sans", system-ui, sans-serif'; g.fillText(e.label, L, y0 + 10);
      const ly = y0 + LH + LANE / 2;
      sizes.forEach((s, k) => {
        const o = e.onsets[s] || {}, c = COL[s], yy = ly + off(k); g.strokeStyle = c; g.fillStyle = c; g.lineWidth = 2;
        if (o.status === 'emerged') {
          const a = x(tok(o.prev_step || 0)), b = x(tok(o.step)); g.globalAlpha = 0.55; g.beginPath(); g.moveTo(a, yy); g.lineTo(b, yy); g.stroke(); g.globalAlpha = 1;
          g.beginPath(); g.arc(b, yy, 3.4, 0, 7); g.fill();
          if (o.jump && o.jump.label === 'sudden') { g.strokeStyle = '#fff'; g.lineWidth = 1; g.beginPath(); g.arc(b, yy, 5, 0, 7); g.stroke(); }
        } else if (o.status === 'by the first checkpoint' || o.status === 'from the start') {
          const b = x(tok(o.step || 0)); g.beginPath(); g.arc(b, yy, 3.4, 0, 7); g.fill(); g.beginPath(); g.moveTo(b - 3, yy); g.lineTo(L, yy); g.globalAlpha = 0.35; g.stroke(); g.globalAlpha = 1;
        } else if (o.status === 'not yet') {
          const b = x(tok(o.last_step)); g.lineWidth = 1.4; g.beginPath(); g.arc(b, yy, 3.2, 0, 7); g.stroke(); g.beginPath(); g.moveTo(b + 5, yy - 3); g.lineTo(b + 9, yy); g.lineTo(b + 5, yy + 3); g.stroke();
        }
      });
      for (const p of PLACED.filter((p) => SHOWN.has(p.id) && p.nearest)) {
        const s = (p.events || {})[e.id] || {}, b = x(tok(p.nearest.step)), yy = ly; g.lineWidth = 1.4;
        g.beginPath(); g.moveTo(b, yy - 5); g.lineTo(b + 5, yy); g.lineTo(b, yy + 5); g.lineTo(b - 5, yy); g.closePath();
        if (s.status === 'has it') { g.fillStyle = '#fff'; g.fill(); } else if (s.status === 'not yet') { g.strokeStyle = '#fff'; g.stroke(); } else { g.strokeStyle = '#5f5c56'; g.stroke(); }
      }
    });
    PLACED.filter((p) => SHOWN.has(p.id) && p.nearest).forEach((p, k) => {
      const b = x(tok(p.nearest.step)); g.setLineDash([3, 4]); g.strokeStyle = 'rgba(255,255,255,0.45)'; g.lineWidth = 1; g.beginPath(); g.moveTo(b, 66 + k * 12); g.lineTo(b, H - BOT); g.stroke(); g.setLineDash([]);
      g.fillStyle = '#fff'; g.font = '10px "IBM Plex Mono", monospace'; const t = `◇ ${shortName(p.model.name)}`; g.fillText(t, Math.max(2, Math.min(b - 4, W - g.measureText(t).width - 2)), 62 + k * 12);
    });
    cv.onclick = (ev) => { const rc = cv.getBoundingClientRect(), i = Math.floor((ev.clientY - rc.top - TOP + 2) / RH); if (i >= 0 && i < T.events.length) pick(T.events[i].id); };
    PSI.canvasLabel(cv, `Chart: ${T.events.length} capabilities, when each emerged in Pythia ${sizes.join(', ')}, on a training-tokens axis` + (SHOWN.size ? `, with ${SHOWN.size} placed model${SHOWN.size > 1 ? 's' : ''}` : ''));
  }
  const shortName = (n) => String(n || '').replace(/^independent\//, '').replace(/^pythia-/, 'pythia-');

  function legend() {
    const box = $('tlSizes'); box.textContent = '';
    T.sizes.forEach((s) => {
      const lab = el('label', 'small tl-size'), cb = el('input'); cb.type = 'checkbox'; cb.checked = SHOW.has(s.id);
      cb.addEventListener('change', () => { if (cb.checked) SHOW.add(s.id); else SHOW.delete(s.id); draw(); });
      const sw = el('span', 'tl-swatch'); sw.style.background = COL[s.id];
      lab.append(cb, sw, document.createTextNode(` Pythia-${s.id}`)); box.appendChild(lab);
    });
  }

  // the event's card: what it measures, its onset in each size, and its curve
  function pick(id) {
    SEL_EV = id; draw(); const e = EV[id], box = $('tlDetail'); box.textContent = '';
    box.appendChild(el('p', 'k m-0', e.label));
    box.appendChild(el('p', 'small', e.means));
    box.appendChild(el('p', 'fx', `Pass: ${e.unit} ${e.direction} ${e.threshold} · ${e.items} · ${e.group}. Onset: the first checkpoint that passes and keeps passing at every later one; it lies between the checkpoint before and that one.`));
    const tb = el('table'), th = el('thead'), hr = el('tr'); ['size', 'onset', 'between', 'rise behind it', 'flickered before'].forEach((h) => hr.appendChild(el('th', null, h))); th.appendChild(hr); tb.appendChild(th);
    const bd = el('tbody');
    T.sizes.forEach((s) => {
      const o = e.onsets[s.id] || {}, tr = el('tr');
      const on = o.status === 'emerged' ? `step ${fmtStep(o.step)} (${fmtTok(tok(o.step))} tokens)` : o.status === 'not yet' ? `not by step ${fmtStep(o.last_step)}` : o.status;
      const btw = o.status === 'emerged' ? `${fmtStep(o.prev_step)} – ${fmtStep(o.step)}` : '—';
      const j = o.jump ? `${o.jump.label} (${Math.round(o.jump.share * 100)}% of the rise in ${Math.round(o.jump.interval_share * 100)}% of log-tokens)` : (e.cont_path ? '—' : 'pass/fail only');
      const sw = el('td'); const d = el('span', 'tl-swatch'); d.style.background = COL[s.id]; sw.append(d, document.createTextNode(` ${s.id}`));
      tr.append(sw, el('td', null, on), el('td', null, btw), el('td', null, j), el('td', null, o.flickers ? `${o.flickers}×` : '')); bd.appendChild(tr);
    });
    tb.appendChild(bd); const sc = el('div', 'scroll'); sc.appendChild(tb); box.appendChild(sc);
    const cv = el('canvas', 'tl-curve'); cv.height = 170; box.appendChild(cv); requestAnimationFrame(() => curve(cv, e)); setTimeout(() => curve(cv, e), 60);
    box.appendChild(el('p', 'fx', '“Rise behind it” is the mirage check (Schaeffer et al. 2023): where a continuous measure stands behind the pass/fail one, how much of its whole rise falls inside the onset interval. Sudden: at least half of it, at twice the even rate or more. A steady rise means the cut-off made it look sudden. A ring around a dot on the chart marks a sudden onset.'));
  }
  function curve(cv, e) {
    if (!cv.clientWidth) return; const r = Math.min(2, devicePixelRatio || 1), W = cv.clientWidth, H = 170; cv.width = W * r; cv.height = H * r; cv.style.height = `${H}px`;
    const g = cv.getContext('2d'); g.scale(r, r); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const pts = []; T.sizes.forEach((s) => (e.curves[s.id] || []).forEach(([st, v]) => { if (v != null) pts.push(v); }));
    PLACED.filter((p) => SHOWN.has(p.id)).forEach((p) => { const v = ((p.events || {})[e.id] || {}).value; if (v != null) pts.push(v); });
    if (!pts.length) return; let lo = Math.min(...pts, e.threshold), hi = Math.max(...pts, e.threshold); if (hi - lo < 1e-9) hi = lo + 1; const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const L = 36, R = W - 8, x = (b) => L + ((Math.log10(Math.max(b, 0.1)) - X0) / (X1 - X0)) * (R - L), y = (v) => H - 16 - ((v - lo) / (hi - lo)) * (H - 26);
    g.setLineDash([4, 4]); g.strokeStyle = 'rgba(255,255,255,0.4)'; g.beginPath(); g.moveTo(L, y(e.threshold)); g.lineTo(R, y(e.threshold)); g.stroke(); g.setLineDash([]);
    g.fillStyle = '#75726c'; g.font = '10px "IBM Plex Mono", monospace'; g.fillText(String(+hi.toFixed(2)), 2, 12); g.fillText(String(+lo.toFixed(2)), 2, H - 16); g.fillText('pass line', L + 2, y(e.threshold) - 3); g.fillText('training tokens (log) →', L, H - 3);
    T.sizes.filter((s) => SHOW.has(s.id)).forEach((s) => {
      const c = (e.curves[s.id] || []).filter(([, v]) => v != null); if (!c.length) return; g.strokeStyle = COL[s.id]; g.fillStyle = COL[s.id]; g.lineWidth = 1.6; g.beginPath();
      c.forEach(([st, v], i) => { const px = x(tok(st)), py = y(v); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke(); c.forEach(([st, v]) => { g.beginPath(); g.arc(x(tok(st)), y(v), 2, 0, 7); g.fill(); });
    });
    PLACED.filter((p) => SHOWN.has(p.id) && p.nearest).forEach((p) => { const v = ((p.events || {})[e.id] || {}).value; if (v == null) return; const px = x(tok(p.nearest.step)), py = y(v); g.fillStyle = '#fff'; g.beginPath(); g.moveTo(px, py - 5); g.lineTo(px + 5, py); g.lineTo(px, py + 5); g.lineTo(px - 5, py); g.fill(); });
    PSI.canvasLabel(cv, `Chart: ${e.label}, value at each checkpoint per size, with the pass line`);
  }

  // ── 2. place a model ─────────────────────────────────────────────────────
  async function loadModels(refresh) {
    try { MODELS = (await psi.timeline({ action: 'models', refresh: !!refresh })).models; } catch (e) { $('tlPlaceStat').textContent = `The model list could not be read: ${PSI.explain(e).text}`; return; }
    const first = !$('tgHost').value;
    fillModels($('tlModel'), MODELS); fillModels($('tgHost'), MODELS);
    if (first) { const d = MODELS.find((m) => /pythia-160m\/step143000/.test(m.name)); if (d) { $('tgHost').value = d.path; $('tlModel').value = d.path; } }
    await hostChanged();
  }
  function fillModels(sel, list, label) {
    const was = sel.value; sel.textContent = ''; const groups = new Map();
    list.forEach((m) => { const fam = m.family || (m.name.includes('/') ? m.name.split('/')[0] : 'models'); if (!groups.has(fam)) groups.set(fam, []); groups.get(fam).push(m); });
    for (const [fam, ms] of groups) {
      const og = document.createElement('optgroup'); og.label = fam;
      ms.sort((a, b) => (a.step ?? -1) - (b.step ?? -1) || a.name.localeCompare(b.name)).forEach((m) => {
        const o = new Option(label ? label(m) : `${m.name}${m.params ? ` · ${(m.params / 1e9).toFixed(m.params < 1e9 ? 2 : 1)} B` : ''}`, m.path); if (m.disabled) o.disabled = true; og.appendChild(o);
      });
      sel.appendChild(og);
    }
    if (was && [...sel.options].some((o) => o.value === was)) sel.value = was;
  }
  async function refreshPlaced() {
    try { PLACED = (await psi.timeline({ action: 'placements' })).placements; } catch (e) { PLACED = []; }
    if (!SHOWN.size && PLACED.length) SHOWN.add(PLACED[0].id);
    const box = $('tlPlaced'); box.textContent = '';
    if (!PLACED.length) box.appendChild(el('p', 'muted small', 'No model placed yet.'));
    PLACED.forEach((p) => {
      const row = el('div', 'tl-placed'), cb = el('input'); cb.type = 'checkbox'; cb.checked = SHOWN.has(p.id); cb.title = 'show on the chart';
      cb.addEventListener('change', () => { if (cb.checked) SHOWN.add(p.id); else SHOWN.delete(p.id); draw(); if (SEL_EV) pick(SEL_EV); });
      const ev = Object.values(p.events || {}), has = ev.filter((x) => x.status === 'has it').length, dec = ev.filter((x) => x.status === 'not decidable').length;
      const txt = el('span', 'small');
      txt.append(el('b', null, p.model.name), document.createTextNode(` · has ${has} of ${ev.length}${dec ? ` (${dec} not decidable)` : ''} · ` + (p.nearest ? `most like Pythia-${p.nearest.size} at step ${fmtStep(p.nearest.step)} (distance ${p.nearest.distance.toFixed(2)})` : 'no near checkpoint') + ` · ${p.signed ? 'signed' : 'not signed'}`));
      const nb = el('button', 'ghost small', 'Nearby'); nb.addEventListener('click', () => nearby(p.id));
      const vb = el('button', 'ghost small', 'Check receipt'); vb.addEventListener('click', async () => { vb.disabled = true; try { const v = await psi.timeline({ action: 'verify', kind: 'placement', id: p.id, files: true }); vb.textContent = v.ok ? 'receipt checks ✓' : `does not check: ${v.checks.filter((c) => !c.ok).map((c) => c.what).join(', ')}`; vb.title = v.checks.map((c) => `${c.ok ? '✓' : '✗'} ${c.what}: ${c.why}`).join('\n'); } catch (e) { vb.textContent = PSI.explain(e).text; } });
      const rb = el('button', 'ghost small', 'Remove'); rb.addEventListener('click', async () => { if (rb.dataset.arm !== '1') { rb.dataset.arm = '1'; rb.textContent = 'Remove: press again'; setTimeout(() => { rb.dataset.arm = ''; rb.textContent = 'Remove'; }, 4000); return; } await psi.timeline({ action: 'delete', kind: 'placement', id: p.id }); SHOWN.delete(p.id); refreshPlaced(); });
      row.append(cb, txt, nb, vb, rb); box.appendChild(row);
    });
    const fams = [...new Set(PLACED.filter((p) => p.model.family && p.model.step != null).map((p) => p.model.family))].filter((f) => PLACED.filter((p) => p.model.family === f).length >= 2);
    const sb = $('tlSeries'); sb.textContent = '';
    fams.forEach((f) => { const b = el('button', 'ghost small', `Sign the ${f} series (${PLACED.filter((p) => p.model.family === f).length} checkpoints)`); b.addEventListener('click', async () => { b.disabled = true; try { const s = await psi.timeline({ action: 'sign_series', family: f }); b.textContent = `${f}: Merkle root ${s.record.merkle_root.slice(0, 16)}… over ${s.record.leaves} receipts, ${s.signature ? 'signed' : 'not signed (no key)'}`; } catch (e) { b.textContent = PSI.explain(e).text; } }); sb.appendChild(b); });
    draw(); if (SEL_P) nearby(SEL_P);
  }
  async function nearby(id) {
    SEL_P = id; const box = $('tlNearby'); box.textContent = '';
    let d; try { d = await psi.timeline({ action: 'get', kind: 'placement', id }); } catch (e) { box.textContent = PSI.explain(e).text; return; }
    const n = d.derived && d.derived.nearby, ev = d.result.events;
    box.appendChild(el('p', 'k m-0', `${d.model.name}${d.model.step != null ? ` (its own step ${fmtStep(d.model.step)})` : ''}: measured ${new Date(d.measured_at).toLocaleString()} on the ${d.device}, ${Math.round(d.seconds)} s`));
    if (!n) { box.appendChild(el('p', 'muted small', 'Too few events could be decided to find a near checkpoint.')); return; }
    const nm = (i) => (EV[i] ? EV[i].label : i);
    box.appendChild(el('p', 'small', `Most like Pythia-${n.neighbour.size} at step ${fmtStep(n.neighbour.step)} (${fmtTok(tok(n.neighbour.step))} training tokens). Next closest: ${d.derived.nearest.slice(1, 4).map((x) => `${x.size} @ ${fmtStep(x.step)} (${x.distance.toFixed(2)})`).join(', ')}.`));
    const nbB = d.derived.nearest_battery || [];
    if (nbB.length) box.appendChild(el('p', 'small', `On the ${nbB[0].shared_events} battery tests alone, which every size was measured on: closest to ${nbB.slice(0, 3).map((x) => `Pythia-${x.size} @ ${fmtStep(x.step)} (${x.distance.toFixed(2)})`).join(', ')}.`));
    const list = (title, items, fmt) => { if (!items.length) return; box.appendChild(el('p', 'k mt-12 m-0', title)); const ul = el('ul', 'small tl-list'); items.forEach((i) => ul.appendChild(el('li', null, fmt(i)))); box.appendChild(ul); };
    list('Emerged there by then, and this model has it', n.shared, nm);
    list('Emerged there by then, and this model lacks it', n.behind, nm);
    list(`What emerged next in Pythia-${n.neighbour.size}`, n.next, (x) => `${nm(x.id)}: ${x.step != null ? `by step ${fmtStep(x.step)} (${fmtTok(tok(x.step))} tokens, between ${fmtStep(x.prev_step)} and ${fmtStep(x.step)})` : x.note}${x.has_it ? ' · this model already has it' : ''}`);
    const nd = Object.entries(ev).filter(([, v]) => v.status === 'not decidable');
    if (nd.length) list('Not decidable in this model’s vocabulary', nd.map(([k, v]) => `${nm(k)}: ${v.why}`), (s) => s);
    box.appendChild(el('p', 'fx', 'Nearest = the recorded Pythia checkpoint whose test results are closest (mean over shared tests of the gap between squashed, threshold-centred values). It says which stage of training this model resembles on these tests, not that it was trained the same way. The numbers are signed; this reading of them is not.'));
  }
  function place() {
    const model = $('tlModel').value; if (!model) return;
    run($('tlPlace'), $('tlPlaceStat'), () => psi.timeline({ action: 'place', model, device: $('tlDevice').value, abstraction: $('tlAbs').checked, confirm: true }), async (j) => { SHOWN.add(j.meta.id); await refreshPlaced(); nearby(j.meta.id); });
  }

  // ── one run at a time, polled ──
  function run(btn, stat, start, done) {
    btn.disabled = true; PSI.setRunning(VIEW, true);
    start().then(() => poll(btn, stat, done)).catch((e) => { PSI.err(e); PSI.status(stat, PSI.explain(e).text, 'error'); btn.disabled = false; PSI.setRunning(VIEW, false); });
  }
  function poll(btn, stat, done) {
    clearTimeout(POLL);
    const tick = async () => {
      let s; try { s = await psi.timeline({ action: 'status' }); } catch (_) { POLL = setTimeout(tick, 3000); return; }
      const j = s.job; if (!j) { btn.disabled = false; PSI.setRunning(VIEW, false); return; }
      const p = j.progress || {};
      if (!j.finished) { PSI.status(stat, `${j.label}: ${p.note || p.stage || 'starting'}${p.of ? ` (${(p.done || 0) + 1} of ${p.of})` : ''}`, 'busy'); $('tlStop').hidden = false; POLL = setTimeout(tick, 2000); return; }
      $('tlStop').hidden = true; btn.disabled = false; PSI.setRunning(VIEW, false);
      if (j.error) { PSI.status(stat, `${j.label}: ${j.error}`, 'error'); return; }
      PSI.status(stat, `${j.label}: done in ${Math.round((new Date(j.ended) - new Date(j.started)) / 1000)} s${j.done && j.done.signed ? ', signed' : ''}`, 'ok'); PSI.setDone(VIEW);
      if (done) await done(j);
    };
    tick();
  }

  // ── 3. the graft lab ─────────────────────────────────────────────────────
  let COMPAT = [], LAST_HOST = null;
  async function hostChanged() {
    const host = $('tgHost').value; if (!host) return;
    try { COMPAT = (await psi.timeline({ action: 'compat', host })).donors; } catch (e) { COMPAT = []; }
    const kinds = { 'drop-in': 'drop-in', 'same width': 'same width', stitched: 'stitched', incompatible: 'does not fit' };
    fillModels($('tgDonor'), COMPAT.map((d) => ({ ...d, disabled: d.compat.kind === 'incompatible' })), (m) => `${m.name} · ${kinds[m.compat.kind]}`);
    const h = MODELS.find((m) => m.path === host); $('tgHostInfo').textContent = h ? `${h.layers} layers, ${h.hidden} wide` : '';
    if (host !== LAST_HOST || (COMPAT.find((d) => d.path === $('tgDonor').value) || {}).compat?.kind === 'incompatible') {
      LAST_HOST = host;
      // The host's own nearest earlier checkpoint when it has one, else the first that fits.
      const fam = h && h.family ? COMPAT.filter((x) => x.family === h.family && x.step != null && x.step < h.step && x.step > 0).sort((a, b) => b.step - a.step) : [];
      const d = fam[0] || COMPAT.find((x) => x.compat.kind === 'drop-in') || COMPAT.find((x) => x.compat.kind !== 'incompatible'); if (d) $('tgDonor').value = d.path;
    }
    donorChanged(); rankList();
  }
  function donorChanged() {
    const d = COMPAT.find((x) => x.path === $('tgDonor').value), box = $('tgCompat');
    box.textContent = d ? `${d.compat.kind}: ${d.compat.why}. ${d.layers} layers, ${d.hidden} wide.` : '';
  }
  function rankList() {
    const box = $('tgDonors'); box.textContent = ''; const host = MODELS.find((m) => m.path === $('tgHost').value);
    const ok = COMPAT.filter((d) => d.compat.kind !== 'incompatible').sort((a, b) => (a.params || 0) - (b.params || 0));
    ok.forEach((d, i) => { const lab = el('label', 'small tl-donor'), cb = el('input'); cb.type = 'checkbox'; cb.value = d.path; cb.checked = i < 6 && (d.params || 0) <= Math.max(1.6e9, (host && host.params) || 0) && d.family === (host && host.family);
      lab.append(cb, document.createTextNode(` ${d.name} · ${d.compat.kind}`)); box.appendChild(lab); });
    if (!ok.length) box.appendChild(el('p', 'muted small', 'No model in the listing fits this host.'));
  }
  function mode() {
    const m = $('tgMode').value; $('tgGraftRow').hidden = m !== 'graft'; $('tgSweepRow').hidden = m !== 'sweep'; $('tgRankRow').hidden = m !== 'rank'; $('tgDonorRow').hidden = m === 'rank';
  }
  function graftGo() {
    const m = $('tgMode').value, a = { action: 'graft', cmd: m, host: $('tgHost').value, align: $('tgAlign').checked, device: $('tgDevice').value, confirm: true };
    if (m === 'rank') a.donors = [...$('tgDonors').querySelectorAll('input:checked')].map((c) => c.value).slice(0, 12); else a.donor = $('tgDonor').value;
    if (m === 'graft') { a.hostLayers = $('tgHL').value.trim(); a.donorLayers = $('tgDL').value.trim(); }
    if (m === 'sweep') { a.width = +$('tgWidth').value; a.band = +$('tgBand').value; a.full = $('tgFull').checked; }
    run($('tgRun'), $('tgStat'), () => psi.timeline(a), async (j) => { await history(); openGraft(j.meta.id); });
  }
  const f2 = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d));
  async function openGraft(id) {
    const box = $('tgOut'); box.textContent = ''; let d;
    try { d = await psi.timeline({ action: 'get', kind: 'graft', id }); } catch (e) { box.textContent = PSI.explain(e).text; return; }
    const head = el('p', 'k m-0'); box.appendChild(head);
    if (d.kind === 'graft') {
      head.textContent = `${d.donor} layers ${d.donor_layers[0]}:${d.donor_layers[1]} → ${d.host} layers ${d.host_layers[0]}:${d.host_layers[1]} · ${d.compat.kind}${d.how === 'wrapped' ? (d.maps ? `, maps fitted (R² in ${d.maps.r2_in}, out ${d.maps.r2_out})` : ', identity maps') : ''}`;
      const tb = el('table'), hr = el('tr'); ['', 'text (bits/token, lower is better)', 'copying (nats saved)', 'facts', 'agreement', 'it says'].forEach((h) => hr.appendChild(el('th', null, h))); const th = el('thead'); th.appendChild(hr); tb.appendChild(th);
      const names = { host: 'host as it is', donor: 'donor as it is', skip: `host without ${d.host_layers[0]}:${d.host_layers[1]} (the gap)`, graft: 'the graft', random: 'random control (donor layers re-initialised)' };
      const bd = el('tbody'); ['host', 'skip', 'graft', 'random', 'donor'].forEach((k) => { const v = d.variants[k] || {}, tr = el('tr'); if (k === 'graft') tr.className = 'tl-hl'; tr.append(el('td', null, names[k]), el('td', null, f2(v.bits)), el('td', null, f2(v.copying)), el('td', null, f2(v.facts)), el('td', null, f2(v.agreement)), el('td', 'mono small', v.error || v.sample || '')); bd.appendChild(tr); });
      tb.appendChild(bd); const sc = el('div', 'scroll'); sc.appendChild(tb); box.appendChild(sc);
      const f = d.fit, verdict = (x) => (x == null ? 'no scale (see the gain)' : x >= 0.8 ? 'does the job' : x >= 0.3 ? 'partly fits' : x > 0 ? 'barely better than the gap' : 'worse than the gap');
      box.appendChild(el('p', 'small', `Fit on text: ${f2(f.text)} (${verdict(f.text)}); on copying: ${f2(f.copying)} (${verdict(f.copying)}). Gain against the host: ${f2(f.gain_text_bits, 3)} bits per token, ${f2(f.gain_copying_nats)} nats of copying.`));
      box.appendChild(el('p', 'fx', f.rule));
    } else if (d.kind === 'sweep') {
      head.textContent = `${d.donor} → ${d.host}: ${d.width} layer${d.width > 1 ? 's' : ''} at a time, ${d.grid.length} grafts · ${d.compat.kind}`;
      const sel = el('select'); sel.append(new Option('fit on text', 'fit_text'), new Option('fit on copying', 'fit_copying'), new Option('copying gained (nats)', 'gain_copying'));
      const cv = el('canvas', 'tl-heat'); box.append(el('label', 'small muted', 'colour shows '), sel, cv);
      const drawIt = () => heat(cv, d, sel.value); sel.addEventListener('change', drawIt); requestAnimationFrame(drawIt); setTimeout(drawIt, 60);
      const best = d.best.slice(0, 6); box.appendChild(el('p', 'k mt-12 m-0', 'Best fits on text'));
      const ul = el('ul', 'small tl-list'); best.forEach((b) => ul.appendChild(el('li', null, `host ${b.host}:${b.host + d.width} ← donor ${b.donor}:${b.donor + d.width}: fit ${f2(b.fit_text)} on text, ${f2(b.fit_copying)} on copying`))); box.appendChild(ul);
      if (!best.some((b) => b.fit_text >= 0.8)) box.appendChild(el('p', 'small', 'No window does the removed layers’ job (fit 0.8 or more): the host’s layers are fitted to their neighbours, and the donor’s are not. Try align (a fitted map in and out), or a closer donor.'));
      box.appendChild(el('p', 'fx', d.rule));
    } else {
      head.textContent = `Donors for ${d.host}, one layer at depth ${d.host_layer}`;
      const tb = el('table'), hr = el('tr'); ['donor', 'fits as', 'its layer', 'fit on text', 'fit on copying'].forEach((h) => hr.appendChild(el('th', null, h))); const th = el('thead'); th.appendChild(hr); tb.appendChild(th);
      const bd = el('tbody'); d.donors.forEach((r) => { const tr = el('tr'); tr.append(el('td', null, r.donor), el('td', null, r.error ? 'error' : (r.compat || {}).kind || ''), el('td', null, r.donor_layer ?? '—'), el('td', null, r.error ? r.error : f2(r.fit_text)), el('td', null, f2(r.fit_copying))); bd.appendChild(tr); });
      tb.appendChild(bd); const sc = el('div', 'scroll'); sc.appendChild(tb); box.appendChild(sc); box.appendChild(el('p', 'fx', d.rule));
    }
    const vb = el('button', 'ghost small', 'Check receipt'); vb.addEventListener('click', async () => { vb.disabled = true; try { const v = await psi.timeline({ action: 'verify', kind: 'graft', id }); vb.textContent = v.ok ? 'receipt checks ✓' : `does not check: ${v.checks.filter((c) => !c.ok).map((c) => c.what).join(', ')}`; vb.title = v.checks.map((c) => `${c.ok ? '✓' : '✗'} ${c.what}: ${c.why}`).join('\n'); } catch (e) { vb.textContent = PSI.explain(e).text; } });
    box.appendChild(el('p', 'fx', `${Math.round(d.seconds)} s on the ${d.device} (${d.dtype}).`)); box.appendChild(vb);
  }
  function heat(cv, d, key) {
    if (!cv.clientWidth) return; const r = Math.min(2, devicePixelRatio || 1), W = cv.clientWidth, nh = d.host_n - d.width + 1, nd = d.donor_n - d.width + 1;
    const L = 34, TOPm = 18, cell = Math.max(6, Math.min(26, Math.floor((W - L - 4) / nd))), H = TOPm + nh * cell + 18; cv.width = W * r; cv.height = H * r; cv.style.height = `${H}px`;
    const g = cv.getContext('2d'); g.scale(r, r); g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const col = (v) => { if (v == null) return '#1b1a18'; if (key === 'gain_copying') { const t = Math.max(-1, Math.min(1, v / 10)); return t >= 0 ? `rgba(111,211,193,${0.15 + 0.85 * t})` : `rgba(255,143,143,${0.15 - 0.85 * t})`; }
      const t = Math.max(-1, Math.min(1, v)); return t >= 0 ? `rgba(111,211,193,${0.12 + 0.88 * t})` : `rgba(255,143,143,${0.12 + 0.88 * -t})`; };
    d.grid.forEach((c) => { g.fillStyle = col(c[key]); g.fillRect(L + c.donor * cell, TOPm + c.host * cell, cell - 1, cell - 1); });
    g.fillStyle = '#8a877f'; g.font = '10px "IBM Plex Mono", monospace'; g.fillText('donor layer →', L, 12);
    for (let i = 0; i < nh; i += Math.max(1, Math.round(nh / 8))) g.fillText(String(i), 4, TOPm + i * cell + cell * 0.75);
    g.fillText('host layer ↓ · green fits, red is worse than the gap, grey not tried', L, H - 4);
    PSI.canvasLabel(cv, `Heat map: ${key} for each host layer (rows) and donor layer (columns)`);
  }
  async function history() {
    let gs = []; try { gs = (await psi.timeline({ action: 'grafts' })).grafts; } catch (_) { /* none */ }
    const box = $('tgHist'); box.textContent = '';
    gs.slice(0, 20).forEach((x) => { const b = el('button', 'ghost small tl-hist');
      b.textContent = `${x.kind}: ${x.kind === 'rank' ? `${x.donors} donors for ${x.host}` : `${x.donor} → ${x.host}`}${x.kind === 'graft' ? ` ${x.donor_layers.join(':')}→${x.host_layers.join(':')} · fit ${f2(x.fit && x.fit.text)}` : ''} · ${x.at ? new Date(x.at).toLocaleString() : ''}`;
      b.addEventListener('click', () => openGraft(x.id)); box.appendChild(b); });
  }
  function preset(which) {
    const find = (re) => (MODELS.find((m) => re.test(m.name)) || {}).path;
    if (which === 'tip') { $('tgHost').value = find(/pythia-160m\/step512$/) || $('tgHost').value; hostChanged().then(() => { $('tgDonor').value = find(/pythia-160m\/step143000$/) || $('tgDonor').value; donorChanged(); $('tgMode').value = 'sweep'; $('tgWidth').value = '2'; $('tgFull').checked = false; $('tgBand').value = '0.15'; mode(); }); }
    if (which === 'late') { $('tgHost').value = find(/pythia-160m\/step143000$/) || $('tgHost').value; hostChanged().then(() => { $('tgDonor').value = find(/pythia-160m\/step16000$/) || $('tgDonor').value; donorChanged(); $('tgMode').value = 'sweep'; $('tgWidth').value = '1'; mode(); }); }
    if (which === 'rank') { $('tgMode').value = 'rank'; mode(); }
  }

  // ── wiring ──
  function init() {
    legend(); draw(); pick(T.events[0].id);
    $('tlPlace').addEventListener('click', place);
    $('tlStop').addEventListener('click', () => psi.timeline({ action: 'cancel' }));
    $('tlRescan').addEventListener('click', () => loadModels(true));
    $('tgHost').addEventListener('change', hostChanged); $('tgDonor').addEventListener('change', donorChanged); $('tgMode').addEventListener('change', mode);
    $('tgRun').addEventListener('click', graftGo);
    document.querySelectorAll('[data-tg-preset]').forEach((b) => b.addEventListener('click', () => preset(b.dataset.tgPreset)));
    mode();
    let loaded = false;
    const onShow = () => { if (!loaded) { loaded = true; loadModels(false); refreshPlaced(); history(); psi.timeline({ action: 'status' }).then((s) => { if (s.running) poll(s.job.kind === 'place' ? $('tlPlace') : $('tgRun'), s.job.kind === 'place' ? $('tlPlaceStat') : $('tgStat'), s.job.kind === 'place' ? refreshPlaced : history); }).catch(() => {}); } draw(); if (SEL_EV) pick(SEL_EV); };
    const sec = $('tab-timeline');
    new MutationObserver(() => { if (!sec.classList.contains('hidden')) onShow(); }).observe(sec, { attributes: true, attributeFilter: ['class'] });
    if (!sec.classList.contains('hidden')) onShow();
    window.addEventListener('resize', () => { if (!$('tab-timeline').classList.contains('hidden')) { draw(); if (SEL_EV) pick(SEL_EV); } });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
