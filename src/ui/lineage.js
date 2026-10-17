// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Seed & lineage: the loaded model's signed identity and family tree, adopting a seed from How AI Emerges, growing
// it (wider and deeper, same answers, then trained on the next stage of text), and the scorecard that puts the
// reader's line next to Pythia and Qwen.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  const fmtN = (n) => n == null ? '–' : n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)} k` : String(n);
  const call = (a) => psi.lineage(a);
  const L = { info: null, job: null, timer: 0, limits: null, anyway: false };
  const NR = 'not recorded';   // a v1 record has no conditions block: say so, never guess
  const model = () => (window.PSIContext && window.PSIContext.model()) || null;
  const stat = (id, text, kind) => PSI.status($(id), text, kind);

  // ── what each button needs ───────────────────────────────────────────────
  ['lnVerify', 'lnAdopt', 'lnMarks', 'lnGrow'].forEach((id) => PSI.requires($(id), 'model'));
  const NOT_CHECKED = 'Grows a checked, signed seed: press Check it first.';
  PSI.block($('lnGrow'), 'verified', NOT_CHECKED);
  // Before the first check the button says "Check it"; after one, "Check it again".
  const checked = (on) => { $('lnVerify').textContent = on ? 'Check it again' : 'Check it'; };
  checked(false);

  // ── identity ─────────────────────────────────────────────────────────────
  async function check() {
    const f = model(); const box = $('lnId'); box.textContent = '';
    if (!f) {
      L.info = null; checked(false); PSI.block($('lnGrow'), 'verified', NOT_CHECKED);
      const p = el('p', 'muted', 'No model is loaded, so there is nothing to check. '); p.appendChild(PSI.loadModelLink()); p.append(': a seed from How AI Emerges, or a model grown from one.'); box.appendChild(p);
      rings(); estimate(); effects(); return;
    }
    stat('lnStat', 'checking…', 'busy');
    try { L.info = await call({ action: 'info', file: f }); } catch (e) { stat('lnStat', `could not check it: ${PSI.explain(e).text}`, 'error'); return err(e); }
    stat('lnStat', ''); checked(true); render();
  }
  const cut = (s, a = 10, b = 6) => s && s.length > a + b + 1 ? `${s.slice(0, a)}…${s.slice(-b)}` : s;
  // An ability already there at step 0 of a grow was inherited (growth keeps the parent's answers): never "appeared at step 0".
  // Rings signed before this rule stored 0; the record is unchanged, only its reading (src/lineage.py appeared_text).
  const INHERITED = 'inherited from parent';
  const apTxt = (k, v, op) => (v === INHERITED || (v === 0 && op === 'grow') ? `${k}: ${INHERITED}` : `${k} at step ${v}`);
  // What a signature does and does not cover (src/lineage.py VOUCHES / NOT_MEASURED).
  const VOUCH = 'vouches for the origin and the numbers as measured, not for abilities';
  const NOT_MEASURED = ['honesty', 'harms', 'multilingual ability', 'reasoning faithfulness', 'calibration', 'behaviour outside these prompts'];
  const notMeasuredTxt = (list, fallback) => (Array.isArray(list) && list.length ? list.join(' · ') : `not listed in this record (signed before the list existed); none of these was measured either: ${(fallback || NOT_MEASURED).join(' · ')}`);
  function render() {
    const d = L.info, box = $('lnId'); box.textContent = ''; $('lnMarksOut').textContent = ''; L.marks = null;
    const row = (k, v, cls) => { const p = el('div'); p.appendChild(el('span', 'muted small', `${k}  `)); const s = el('span', cls, v); p.appendChild(s); box.appendChild(p); return s; };
    const st = d.status === 'verified' ? ['ok', 'verified: The Emergence Machine'] : d.status === 'signed' ? ['ok', `signed by key ${d.signer}`] : d.status === 'unsigned' ? ['none', d.record ? 'not signed (private)' : 'no lineage yet'] : ['bad', 'does not verify'];
    const b = row('status', ''); const badge = el('span', `ln-badge ${st[0]}`, st[1]); b.appendChild(badge);
    if (d.name) { const n = el('div', 'ln-name', d.name); box.insertBefore(n, box.firstChild); }
    row('file', d.file);
    row('fingerprint', d.short).title = d.id;
    if (d.status === 'verified' || d.status === 'signed') row('signature', `${d.signature_scope || VOUCH}`, 'small muted');
    if (d.certificate) row('certificate', `${cut(d.certificate)} · ${d.why.replace(/^issued by /, 'by ')}`).title = d.certificate;
    if (d.root_fingerprint) row('root CA', cut(d.root_fingerprint, 11, 8), 'small').title = d.root_fingerprint + '\nCompare with the fingerprint published at emergencemachine.com/.well-known/em-lineage/. Anyone can check this file with tools/em-verify.py.';
    if (d.street && d.street.place) row('street', `${d.street.place}${d.street.week ? `, ${d.street.week}` : ''}`);
    if (d.record) {
      const r = d.record;
      row('generation', `${r.gen}${r.gen === 0 ? ' (the seed)' : ` · ×${r.op.width} wide, +${r.op.depth} layers, ${r.train ? r.train.steps : '?'} steps, ${r.data ? (r.data.stage === 'all' ? `${r.data.streets} streets` : r.data.stage) : '?'}`}`);
      if (r.shape) row('shape', `${fmtN(r.shape.params || null)} · ${r.shape.layers} × ${r.shape.width}`);
      const appeared = Object.entries(r.appeared || {}).filter(([, s]) => s != null);
      if (r.appeared) row(r.appeared_rule ? 'abilities' : 'first seen', (appeared.length ? appeared.map(([k, s]) => apTxt(k, s, r.op && r.op.kind)).join(' · ') : `none appeared within ${r.train && r.train.steps ? r.train.steps.toLocaleString() : 'its'} steps; cause not tested`) + (r.appeared_floor && r.appeared_floor.verdict !== 'appeared' ? ` · understanding: ${r.appeared_floor.verdict}` : ''));
      if (r.v !== 2 && r.gen > 0) row('record', 'v1: the conditions of the measurement were not recorded', 'small muted');
      row('not measured', notMeasuredTxt(r.not_measured, d.not_measured_default), 'small muted');
      const chain = el('div', 'ln-chain'); chain.appendChild(el('span', 'muted small', 'family  '));
      (d.ancestry || []).forEach((a) => { chain.appendChild(el('span', null, `g${a.gen} ${a.id} ${a.signed && a.linked ? '✓' : '✗'}`)); chain.appendChild(document.createTextNode('→')); });
      chain.appendChild(el('span', 'me', `g${r.gen} ${d.short} ${d.status === 'verified' ? '✓' : '✗'}`)); box.appendChild(chain);
    }
    if (d.status !== 'verified' && d.note) box.appendChild(el('p', d.status === 'broken' ? 'small bad' : 'small muted', d.note));
    rings();
    $('lnAdopt').classList.toggle('hidden', !!d.record || d.status !== 'unsigned');
    // Signing is optional (Settings > Profile & keys): a ring signed by anyone, or by nobody, can grow; a broken chain cannot.
    PSI.block($('lnGrow'), 'verified', d.record && d.status !== 'broken' ? null : !d.record ? 'Adopt the seed first, so its children can be traced to it.' : 'This model does not verify, so it cannot grow.');
    if (!L.growing) stat('lnGrowStat', '');
    estimate(); effects();
  }

  // ── growth rings ─────────────────────────────────────────────────────────
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (t, a) => { const e = document.createElementNS(NS, t); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); return e; };
  const markName = (n) => n === 'seed' ? 'tree' : n.startsWith('ring') ? `ring g${n.slice(4)}` : n;
  function rings() {
    const svg = $('lnRings'), rs = (L.info && L.info.rings) || []; svg.textContent = '';
    const empty = $('lnRingsEmpty');
    if (!rs.length) {
      $('lnRing').textContent = ''; svg.closest('.ln-rings').classList.add('hidden'); empty.classList.remove('hidden');
      empty.textContent = !model() ? 'No rings to show: no model is loaded.' : !L.info ? 'No rings yet: check the model first.'
        : 'No rings yet: this model has no signed record. A seed from How AI Emerges gets its first ring when you adopt it here, and every Grow adds one.';
      return;
    }
    svg.closest('.ln-rings').classList.remove('hidden'); empty.classList.add('hidden');
    if (L.ring == null || L.ring >= rs.length) L.ring = rs.length - 1;
    const p0 = rs[0].params || 1;   // each ring: one step outward for the stage, more for the numbers it added
    const t = rs.map((r, k) => 1.6 + k + 2 * Math.max(0, Math.log10((r.params || p0) / p0))), R = 112 / t[t.length - 1];
    const said = (L.marks && L.marks.marks) || {};
    for (let k = rs.length - 1; k >= 0; k--) {
      const r = rs[k], ok = r.signed && r.by_em && r.linked;
      const c = svgEl('circle', { cx: 120, cy: 120, r: t[k] * R, class: 'ring', fill: `rgba(255,195,107,${k % 2 ? 0.2 : 0.1})`,
        stroke: k === L.ring ? '#ffc36b' : ok ? 'rgba(143,209,143,.75)' : '#ff7a7a', 'stroke-width': k === L.ring ? 2.5 : 1.2 });
      c.appendChild(svgEl('title', {})).textContent = `g${r.gen} · ${r.params ? fmtN(r.params) : '?'}`;
      c.addEventListener('click', () => { L.ring = k; rings(); });
      c.setAttribute('tabindex', '0'); c.setAttribute('role', 'button'); c.setAttribute('aria-label', `ring g${r.gen}: show what happened in it`);
      c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); L.ring = k; rings(); const n = $('lnRings').querySelectorAll('circle')[rs.length - 1 - k]; if (n) n.focus(); } });
      svg.appendChild(c);
    }
    rs.forEach((r, k) => {   // labels, and a diamond where a ring's own signature lives in the weights
      const inner = k ? t[k - 1] * R : 0, mid = (inner + t[k] * R) / 2;
      if (t[k] * R - inner >= 11 || !k) { const tx = svgEl('text', { x: 120, y: k ? 120 - mid + 3.5 : 123.5, 'text-anchor': 'middle', 'font-size': 10, fill: '#d8d4cc', 'pointer-events': 'none', 'font-family': 'IBM Plex Mono, monospace' }); tx.textContent = `g${r.gen}`; svg.appendChild(tx); }
      const n = k ? `ring${r.gen}` : 'birth'; if (!(n in r.marks) && !(n in said)) return;
      const m = said[n], col = !m ? '#d8d4cc' : m.exact && m.signature_valid ? '#8fd18f' : m.recognised ? '#ffc36b' : '#ff7a7a';
      const a = Math.PI / 4 + k * 0.9, x = 120 + Math.cos(a) * (k ? mid : 0) , y = 120 + Math.sin(a) * (k ? mid : 0) + (k ? 0 : 12);
      svg.appendChild(svgEl('path', { d: `M${x} ${y - 4}L${x + 4} ${y}L${x} ${y + 4}L${x - 4} ${y}Z`, fill: col, 'pointer-events': 'none' }));
    });
    const r = rs[L.ring], box = $('lnRing'); box.textContent = '';
    const row = (k, v, cls) => { const p = el('div'); p.appendChild(el('span', 'muted small', `${k}  `)); p.appendChild(el('span', cls || null, v)); box.appendChild(p); return p; };
    const h = el('div'); h.appendChild(el('span', 'ln-name small-name', `${r.name}  `)); h.appendChild(el('span', 'muted small', `g${r.gen} · ${r.id}  `)); h.appendChild(el('span', `ln-badge ${r.signed && r.by_em && r.linked ? 'ok' : 'bad'}`, r.signed && r.by_em && r.linked ? 'signed' : 'does not verify')); box.appendChild(h);
    const mins = (x) => x == null ? null : x < 60 ? `${Math.round(x)} s` : `${Math.round(x / 6) / 10} min`;
    const gbm = (mb) => mb == null ? null : `${(mb / 1024).toFixed(1)} GB`;
    const c = r.conditions || null, sch = (c && c.schedule) || null, g = (c && c.gpu) || null, tm = (c && c.time) || null, uof = r.understanding_of || 9;
    row('stage', r.op === 'adopt' ? 'the seed, trained in How AI Emerges' : `grown ×${r.width} wide, +${r.depth} layers`);
    row('size', `${r.params ? fmtN(r.params) : 'size ' + NR} · ${r.layers} × ${r.dim}`);
    if (r.op === 'adopt') row('trained', `${r.steps ? `${r.steps.toLocaleString()} steps` : `steps ${NR}`} on ${r.street || 'its street'} · device ${NR} · time ${NR}`);
    else row('trained', `${r.steps ? r.steps.toLocaleString() : NR} steps on ${r.data || r.street || `text ${NR}`} · ${r.device || `device ${NR}`} · ${r.seconds ? `${mins(r.seconds)}${r.seconds_rule ? ', training only' : ' (v1: fetch and scoring included)'}` : `time ${NR}`}`);
    if (r.data_sha256) row('text', `sha256 ${r.data_sha256}…`);
    if (r.op !== 'adopt') {
      row('record', c ? 'v2: the conditions of the measurement are in the signed record' : `v${r.v || 1}: conditions ${NR}; the fields below say so`, 'small muted');
      row('time split', tm ? `fetch ${mins(tm.fetch) || NR} · train ${mins(tm.train) || NR} · score ${mins(tm.score) || NR} · total ${mins(tm.total) || NR}` : NR);
      row('seed · schedule', sch ? `random seed ${sch.seed} · lr ${sch.lr} · batch ${sch.batch} × ${sch.sequence} · warm-up ${Math.round((sch.warmup_frac || 0) * 100)}% (${sch.warmup_steps} steps) · cosine floor ${sch.cosine_floor} · betas ${(sch.betas || []).join(', ')} · weight decay ${sch.weight_decay} · clip ${sch.grad_clip}` : NR);
      row('validation', sch && sch.validation ? `the last ${Math.round(sch.validation.split * 100)}% of the text (${(sch.validation.pieces || 0).toLocaleString()} pieces), at most ${sch.validation.max_windows} windows, every ${sch.validation.every} steps` : NR);
      row('marks', sch ? (sch.marks_rehearsed ? 'rehearsed at every step; the loss shown includes them' : 'not rehearsed') : NR);
      row('precision · versions', c ? `${c.precision || NR} · python ${c.versions && c.versions.python || NR} · torch ${c.versions && c.versions.torch || NR} · transformers ${c.versions && c.versions.transformers || NR}` : NR);
      if (!c) row('GPU', NR);
      else if (r.device === 'cpu' && !g.name) row('GPU', 'not used (trained on the CPU)');
      else {
        const spill = g.spill_detect === 'unavailable' ? 'spill detection unavailable' : g.spilled === true ? `spilled ${gbm(g.shared_mb_max)} into system memory (slow)` : g.spilled === false ? `nothing spilled (shared ${gbm(g.shared_mb_max) || '0.0 GB'})` : 'spill not read';
        row('GPU', `${g.name || `name ${NR}`} · peak ${gbm(g.peak_mb) || NR} · ${spill}`, g.spilled === true ? 'bad' : null);
      }
      row('growth check', r.check ? `largest probability difference ${(+r.check.max_prob_difference).toExponential(1)} over ${r.check.sentences} sentences (limit ${r.check.limit})` : r.max_diff != null ? `largest probability difference ${(+r.max_diff).toExponential(1)}, checked on one sentence (v1)` : NR);
      row('scored', r.score_device ? `before and after on ${r.score_device}` : r.device === 'cpu' ? 'before and after on cpu (v1)' : `before on cpu, after on ${r.device || NR}: not the same device (v1)`, r.score_device || r.device === 'cpu' ? null : 'small muted');
    }
    if (r.bits_street != null) row('bits / char', `street ${r.bits_street} · other ${r.bits_general}`);
    const fl = r.appeared_floor, lim = (L.limits && L.limits.understanding) || { floor: 6, appeared: 8 };
    if (r.understanding != null) row('understanding', `${r.understanding} of ${uof}${r.understanding < lim.appeared ? ` · within the untrained floor (${lim.floor} of ${uof} untrained)` : ''}`);
    const ap = Object.entries(r.appeared || {});
    row('appeared', (ap.length ? ap.map(([k, st]) => apTxt(k, st, r.op)).join(' · ') : r.op === 'adopt' ? 'not tracked for the seed' : `none within ${r.steps ? r.steps.toLocaleString() : 'its'} steps; cause not tested`)
      + (fl && fl.verdict !== 'appeared' ? ` · understanding: ${fl.verdict}` : !fl && r.op !== 'adopt' ? ` · v1 rule: 7 of ${uof} counted; the floor is ${lim.floor}` : ''));
    const mk = Object.entries(r.marks || {}); if (mk.length) row('in its weights', mk.map(([n, st]) => `${markName(n)} ${st}`).join(' · '));
    row('made', `${(r.created || '').replace('T', ' ').replace('Z', ' UTC')}${r.maker ? ` · ${r.maker}` : ''}`);
    if (r.signature) row('signature', `${r.signature}… · ${r.signature_scope || VOUCH}`);
    row('not measured', notMeasuredTxt(r.not_measured, L.info && L.info.not_measured_default), 'small muted');
    if (r.op !== 'adopt') {
      const p = row('loss curve', '');
      if (!r.curve) p.lastChild.textContent = `not found beside the file (${r.op === 'grow' ? 'NAME.lineage.json is missing' : 'not kept'})`;
      else {
        const b = el('button', 'ghost', 'Show its loss curve'); b.type = 'button';
        const v = r.curve.verified === true ? 'matches the signed record' : r.curve.verified === false ? 'does NOT match the signed record' : 'unverified: a v1 record has no hash of it';
        b.addEventListener('click', () => { curve(r.curve.rows, `ring g${r.gen}, reopened from ${r.curve.file}: ${v}`); $('lnCurve').scrollIntoView({ block: 'nearest' }); });
        p.lastChild.append(b, el('span', `small ${r.curve.verified === false ? 'bad' : 'muted'}`, `  ${r.curve.rows.length} points · ${v}`));
      }
    }
  }
  $('lnVerify').addEventListener('click', check);
  $('lnAdopt').addEventListener('click', async () => {
    const b = $('lnAdopt'); b.disabled = true; stat('lnStat', 'signing it and measuring it… (a few seconds)', 'busy');
    try {
      const r = await call({ action: 'adopt', file: model() });
      stat('lnStat', `Adopted as ${r.file}: your download is unchanged, the copy with its lineage record sits next to it. Opening it…`, 'ok');
      await psi.settings({ lastModel: r.path }); await window.PSIOpenModel(r.path); await check(); table();
    } catch (e) { err(e); stat('lnStat', `adopting failed: ${PSI.explain(e).text}`, 'error'); } finally { b.disabled = false; }
  });

  // ── the signature in its weights ─────────────────────────────────────────
  $('lnMarks').addEventListener('click', async () => {
    const f = model(), box = $('lnMarksOut'); if (!f) { err('No model is loaded.', { text: 'Load a model first: there is nothing to ask.', action: { label: PSI.NEED_MODEL, view: 'settings', focus: 'stModelPick' } }); return; } box.textContent = '';
    $('lnMarks').disabled = true; stat('lnMarksStat', 'asking…', 'busy');
    let d; try { d = await call({ action: 'marks', file: f }); } catch (e) { err(e); stat('lnMarksStat', `asking failed: ${PSI.explain(e).text}`, 'error'); return; } finally { $('lnMarks').disabled = false; }
    stat('lnMarksStat', ''); L.marks = d; rings();
    const names = { birth: 'birth' };
    const recorded = d.recorded || {};
    Object.entries(d.marks).forEach(([n, m]) => {
      const p = el('div'); p.appendChild(el('span', 'muted small', `${names[n] || markName(n)}  `));
      const said = m.exact && m.signature_valid ? ['ok', 'says it · verified'] : m.recognised ? ['ok', 'recognises it'] : ['none', 'not there'];
      p.appendChild(el('span', `ln-badge ${said[0]}`, said[1]));
      p.appendChild(el('span', 'small', `  ${Math.round(m.chars_right * 100)}% said · z ${m.z} vs ${m.decoys} decoys`));
      if ((n === 'seed' || n.startsWith('ring')) && 'commit_matches' in m) p.appendChild(el('span', 'small muted', m.commit_matches ? ' · commitment matches' : ' · commitment differs'));
      if (recorded[n]) p.appendChild(el('span', 'small muted', ` · at signing: ${recorded[n].exact && recorded[n].signature_valid ? 'said it' : recorded[n].recognised ? 'recognised it' : 'not there'}`));
      if (m.exact && m.said) { const s = el('div', 'small muted', m.said); s.style.font = '11px var(--mono)'; s.style.wordBreak = 'break-all'; p.appendChild(s); }
      box.appendChild(p);
    });
    const note = d.seed_note ? `Tree mark: ${d.seed_note}. ` : '';
    box.appendChild(el('p', 'small muted', `${note}Saying it is proof: nobody can make a valid signature without the key. Not saying it proves nothing, because whoever trains a model can wash a mark out. Recognition is statistical: z 4 or more, with no decoy as familiar, counts as recognised.`));
  });

  // ── growing ──────────────────────────────────────────────────────────────
  // How long a grow takes: measured on an earlier ring of this line on the same device when there is one,
  // otherwise the manual's rule of thumb (1,500 steps of a 1-million-number model: about three minutes on the CPU).
  const CPU_RATE = 180 / 1500;   // seconds per step per million numbers
  function estimate() {
    const out = $('lnEst'); if (!out) return;
    const rs = (L.info && L.info.rings) || [], rec = L.info && L.info.record, top = rs[rs.length - 1] || {};
    const p0 = (rec && rec.shape && rec.shape.params) || top.params || 0, layers = (rec && rec.shape && rec.shape.layers) || top.layers || 0;
    const w = +$('lnWidth').value || 1, d = +$('lnDepth').value || 0, steps = +$('lnSteps').value || 1500, dev = $('lnDevice').value;
    const parts = [];
    if (p0) {
      const p1 = p0 * w * w * (layers ? (layers + d) / layers : 1), M = Math.max(0.05, p1 / 1e6);
      // a v2 ring's seconds are training only; a v1 ring's include fetching and scoring, so a v2 ring is preferred
      const rate = (device) => { const ok = rs.filter((r) => r.device === device && r.seconds && r.steps && r.params), m = ok.filter((r) => r.seconds_rule).pop() || ok.pop(); return m ? { v: m.seconds / m.steps / (m.params / 1e6), from: `ring g${m.gen}${m.seconds_rule ? '' : ', fetch and scoring included'}` } : null; };
      const mins = (v) => { const m = (v * steps * M) / 60; return m < 1 ? 'under a minute' : `about ${Math.round(m)} min`; };
      const cpu = rate('cpu') || { v: CPU_RATE, from: 'the rule of thumb in the manual' }, gpu = dev === 'cuda' ? rate('cuda') : null;
      const fetch = $('lnData').value === 'all' ? ', plus fetching the streets' : '';
      if (dev === 'cuda') parts.push(gpu ? `${mins(gpu.v)} on the GPU (timed on ${gpu.from}), grown to about ${fmtN(p1)} numbers${fetch}.` : `On the GPU usually faster than the CPU’s ${mins(cpu.v)} (grown to about ${fmtN(p1)} numbers, ${steps.toLocaleString()} steps${fetch}).`);
      else parts.push(`${mins(cpu.v)[0].toUpperCase()}${mins(cpu.v).slice(1)} on the CPU for ${steps.toLocaleString()} steps, grown to about ${fmtN(p1)} numbers (timed on ${cpu.from})${fetch}.`);
    }
    if (dev === 'cuda') parts.push('The GPU is shared: while another job uses it (the inside engine, a Storage ladder run, the local model), both slow down and the grow can run out of memory. Pause the others, or choose CPU.');
    out.textContent = parts.join(' ');
  }
  // What Grow causes, said before it runs (everything: files, their size, the network, the cache, the marks, the device).
  function effects() {
    const fx = $('lnFx'); if (!fx) return;
    const rs = (L.info && L.info.rings) || [], rec = L.info && L.info.record, top = rs[rs.length - 1] || {};
    const p0 = (rec && rec.shape && rec.shape.params) || top.params || 0, layers = (rec && rec.shape && rec.shape.layers) || top.layers || 0, gen = (rec && rec.gen) || 0;
    const w = +$('lnWidth').value || 1, d = +$('lnDepth').value || 0, data = $('lnData').value, dev = $('lnDevice').value;
    const p1 = p0 ? p0 * w * w * (layers ? (layers + d) / layers : 1) : 0, mb = p1 ? `${Math.max(1, Math.round(p1 * 4 / 2 ** 20))} MB` : 'its size in float32';
    const nm = `…-g${gen + 1}`;
    fx.textContent = `Writes, next to the parent: ${nm}.gguf (signed, ${mb}), ${nm}.lineage.json (the record, the ancestry and the loss curve) and ${nm}-hf/ (weights, tokenizer and optimizer state, about 3× the model file). `
      + (data === 'all' ? 'Fetches this week’s streets from storage.googleapis.com (emergence-e8-lens) and caches them in the app’s cache folder (<cache folder>/psi, or the data folder’s cache/; PSI_CACHE overrides). ' : data === 'street' ? 'Trains on the street text inside the file; nothing is fetched. ' : 'Reads your text file; nothing is fetched. ')
      + 'Rehearses the signature marks at every step; their loss is part of the curve shown. '
      + (dev === 'cuda' ? 'Takes GPU memory for the time above, after the same check as an engine start (a long GPU job or too little free memory holds it back); whether it spilled is written into the ring. ' : 'Uses the CPU for the time above. ')
      + 'The parent is unchanged; Stop keeps nothing half-made.';
  }
  call({ action: 'limits' }).then((l) => {   // the limits live in lineage-api.js; the inputs and the floor wording follow them
    L.limits = l; const st = $('lnSteps'); st.min = l.steps.min; st.max = l.steps.max; st.step = l.steps.step; if (!st.value) st.value = l.steps.default;
    const dp = $('lnDepth'); if (dp && l.depth.choices && [...dp.options].map((o) => +o.value).join() !== l.depth.choices.join()) { dp.textContent = ''; l.depth.choices.forEach((v) => { const o = el('option', null, String(v)); o.value = v; if (v === l.depth.default) o.selected = true; dp.appendChild(o); }); }
  }).catch(() => { /* an old server: the inputs keep their own limits */ });
  ['lnWidth', 'lnDepth', 'lnSteps', 'lnDevice', 'lnData'].forEach((id) => $(id).addEventListener('input', () => { estimate(); effects(); }));
  ['lnWidth', 'lnDepth', 'lnDevice', 'lnData'].forEach((id) => $(id).addEventListener('change', effects));
  document.addEventListener('psi-model', () => setTimeout(effects, 0)); effects();
  ['lnWidth', 'lnDepth', 'lnDevice', 'lnData'].forEach((id) => $(id).addEventListener('change', estimate));

  $('lnData').addEventListener('change', () => $('lnFileRow').classList.toggle('hidden', $('lnData').value !== 'file'));
  let prog = null, fails = 0;
  const progHost = () => { let h = $('lnProg'); if (!h) { h = el('div'); h.id = 'lnProg'; $('lnGrow').parentElement.after(h); } return h; };
  const growEnd = () => { L.growing = false; $('lnGrow').disabled = false; };
  // A grow on the GPU runs the same hold-back check as an engine start (window.PSIMachine.holdBack; the server checks too):
  // a long GPU job or too little free memory stops it, and says why; "Grow anyway" goes past it once.
  async function heldBack(o) {
    if (o.device !== 'cuda' || L.anyway) return false;
    let hb = null; try { hb = await call({ action: 'holdback', file: o.file, width: o.width, depth: o.depth }); } catch (_) { return false; }   // an old server: it refuses by itself, or runs
    if (!hb || !hb.why) return false;
    stat('lnGrowStat', `not started: ${hb.why}.`, 'warn');
    const b = $('lnGrowAnyway'); if (b) { b.classList.remove('hidden'); b.onclick = () => { L.anyway = true; b.classList.add('hidden'); $('lnGrow').click(); }; }
    return true;
  }
  $('lnGrow').addEventListener('click', async () => {
    const f = model();
    if (!f) { err('No model is loaded.', { text: 'Load a seed model first: there is nothing to grow.', action: { label: PSI.NEED_MODEL, view: 'settings', focus: 'stModelPick' } }); return; }
    const o = { action: 'grow', file: f, width: +$('lnWidth').value, depth: +$('lnDepth').value, data: $('lnData').value, steps: +$('lnSteps').value || (L.limits ? L.limits.steps.default : 1500), device: $('lnDevice').value, textFile: $('lnFile').value.trim(), anyway: L.anyway };
    if (o.data === 'file' && !o.textFile) { stat('lnGrowStat', 'Give the full path to a text file.', 'warn'); $('lnFile').focus(); return; }
    if (await heldBack(o)) return;
    L.anyway = false; if ($('lnGrowAnyway')) $('lnGrowAnyway').classList.add('hidden');
    L.growing = true; fails = 0; $('lnGrow').disabled = true; $('lnLoadRow').classList.add('hidden'); $('lnGrowOut').textContent = '';
    stat('lnGrowStat', 'growing…', 'busy');
    prog = growProg(o.steps);
    try { L.job = (await call(o)).job; poll(); } catch (e) { err(e); growEnd(); prog.fail('did not start'); prog = null; stat('lnGrowStat', `did not start: ${PSI.explain(e).text}`, 'error'); }
  });
  // Stop ends the Python process; grow.py writes the grown model only when training finishes, so nothing half-made is kept
  function growProg(total) {
    return PSI.progress(progHost(), { total, label: 'step', view: 'lineage', onCancel: () => { if (L.job != null) call({ action: 'cancel', job: L.job }).catch((e) => err(e)); stat('lnGrowStat', 'stopping…', 'busy'); } });
  }
  async function poll() {
    clearTimeout(L.timer);
    let j;
    try { j = await call({ action: 'job', job: L.job }); fails = 0; } catch (e) {
      // one lost answer is not the end of the job: try again, waiting longer each time, then say so
      if (++fails <= 5) { const wait = 1500 * 2 ** fails; stat('lnGrowStat', `lost contact with the grow job; trying again in ${Math.round(wait / 1000)} s (${fails} of 5)`, 'warn'); L.timer = setTimeout(poll, wait); return; }
      growEnd(); if (prog) { prog.fail('lost contact with the grow job'); prog = null; }
      stat('lnGrowStat', 'lost contact with the grow job; it may still be running', 'error');
      err(e, { text: 'Lost contact with the grow job after five tries. It may still be running in the background.', action: { label: 'Try again', onClick: () => { fails = 0; L.growing = true; $('lnGrow').disabled = true; prog = growProg(+$('lnSteps').value || 1500); poll(); } } });
      return;
    }
    const p = j.progress, last = p[p.length - 1] || {}, g = p.find((x) => x.stage === 'grow');
    const lines = [];
    if (g) lines.push(`Grown from ${fmtN(g.params_before)} to ${fmtN(g.params_after)} numbers; before any training its answers differ from its parent's by at most ${g.max_prob_difference.toExponential(1)} in probability (checked on ${g.checked_sentences ? `${g.checked_sentences} sentences` : 'one sentence'}): the same model, bigger.`);
    if (j.spilled) lines.push('The GPU is overfull: this grow spilled into system memory, so it runs many times slower; the ring records it.');
    else if (j.device === 'cuda' && j.spill_detect === 'unavailable') lines.push('Spill detection unavailable on this machine: the ring will say so, not "nothing spilled".');
    if (last.stage === 'data') lines.push(`fetching the streets: ${last.fetched} of ${last.of}`);
    if (last.stage === 'train') lines.push(`step ${last.step} of ${last.of} · ${last.val_bits_per_char} bits per character on text it has not seen · understanding ${last.understanding} of 9 · copying saves ${last.copying_nats} nats · ${last.lookback_heads} look-back heads`);
    if (!j.done) stat('lnGrowStat', `growing · ${j.seconds} s`, 'busy');
    if (prog && !j.done) { if (last.stage === 'train') prog.step(last.step, `${last.val_bits_per_char} bits/char`); else if (last.stage === 'data') prog.step(0, `fetching the streets: ${last.fetched} of ${last.of}`); else prog.step(0, g ? 'grown, starting to train' : 'growing'); }
    curve(p.filter((x) => x.stage === 'train'));
    if (j.error === 'cancelled') { stat('lnGrowStat', 'stopped: nothing was saved, the parent is unchanged', 'warn'); growEnd(); if (prog) { prog.fail('stopped'); prog = null; } return; }
    if (j.error) { $('lnGrowOut').textContent = j.error; stat('lnGrowStat', `growing failed: ${PSI.explain(j.error).text}`, 'error'); growEnd(); if (prog) { prog.fail('stopped by an error'); prog = null; } return; }
    if (j.done && j.result) {
      const r = j.result.record, s0 = r.score.before_training, s1 = r.score.after;
      const app = Object.entries(r.appeared).filter(([, v]) => v != null);
      const mk = r.marks ? ` Signatures in its weights: ${Object.entries(r.marks).map(([n, m]) => `${markName(n)} ${m.exact && m.signature_valid ? 'said and verified' : m.recognised ? 'recognised only' : 'missing'} (z ${m.z})`).join(', ')}.` : '';
      const fl = r.appeared_floor, gc = r.conditions && r.conditions.gpu, tm = r.conditions && r.conditions.time;
      const floor = fl && fl.verdict !== 'appeared' ? ` Understanding: ${fl.verdict}.` : '';
      const gpuNote = gc && gc.spilled === true ? ` It spilled ${(gc.shared_mb_max / 1024).toFixed(1)} GB into system memory while training (recorded in the ring).` : gc && gc.spill_detect === 'unavailable' && r.train.device === 'cuda' ? ' Spill detection was unavailable; the ring says so.' : '';
      lines.push(`Generation ${r.gen}, ${j.result.short}: ${j.result.status}${j.result.by_em ? ', signed by The Emergence Machine' : ''} (record v${r.v || 1}). Bits per character on street text ${s0.bits_per_char_street} → ${s1.bits_per_char_street}, on other text ${s0.bits_per_char_general} → ${s1.bits_per_char_general}; understanding ${s0.understanding} → ${s1.understanding} of ${s1.understanding_of || 9}; copying ${s0.copying_nats} → ${s1.copying_nats} nats; before and after scored on ${r.score.device || r.train.device}. ${app.length ? `Appeared and stayed: ${app.map(([k, v]) => apTxt(k, v, 'grow')).join(', ')}.` : `None of the tracked abilities appeared within ${r.train.steps.toLocaleString()} steps; cause not tested.`}${floor}${tm ? ` Time: fetch ${Math.round(tm.fetch || 0)} s, train ${Math.round(tm.train)} s, score ${Math.round(tm.score)} s.` : ''}${gpuNote}${mk} The signature ${r.signature_scope || VOUCH}. Not measured: ${notMeasuredTxt(r.not_measured)}.`);
      L.grown = j.result.file; $('lnLoadRow').classList.remove('hidden'); growEnd(); table(); if (j.result.wrote) lines.push(`Written: ${j.result.wrote.join(', ')}${j.result.cache ? ` · streets cached in ${j.result.cache}` : ''}.`);
      stat('lnGrowStat', `generation ${r.gen} grown and signed`, 'ok'); if (prog) { prog.finish(`done in ${Math.round(j.seconds / 6) / 10} min`); prog = null; }
    }
    $('lnGrowOut').textContent = lines.join('\n');
    if (!j.done) L.timer = setTimeout(poll, 1500);
  }
  function curve(rows, who) {
    const cv = $('lnCurve'); if (!rows.length) { cv.classList.add('hidden'); return; } cv.classList.remove('hidden');
    const r = Math.min(2, devicePixelRatio || 1), H = 120; cv.style.height = `${H}px`; cv.width = (cv.clientWidth || 700) * r; cv.height = H * r;
    const g = cv.getContext('2d'), W = cv.width, h = cv.height, pad = 28 * r, total = rows[0].of || rows[rows.length - 1].step || 1;
    const ys = rows.map((x) => x.val_bits_per_char), hi = Math.max(...ys), lo = Math.min(...ys, hi - 0.5);
    const X = (s) => pad + (s / total) * (W - 2 * pad), Yb = (v) => pad + ((hi - v) / ((hi - lo) || 1)) * (h - 2 * pad);   // higher bits at the top
    g.clearRect(0, 0, W, h); g.strokeStyle = '#ffc36b'; g.lineWidth = 1.5 * r; g.beginPath();
    rows.forEach((x, i) => { const px = X(x.step), py = Yb(x.val_bits_per_char); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
    g.fillStyle = '#a9a69f'; g.font = `${10 * r}px "IBM Plex Mono", monospace`;
    g.fillText(`${who ? `${who} · ` : ''}bits per character on unseen text: ${ys[0]} → ${ys[ys.length - 1]} (lower is better)`, pad, 12 * r);
    g.fillText(`step ${rows[rows.length - 1].step} of ${total}`, W - pad - 90 * r, h - 6 * r);
    PSI.canvasLabel(cv, `Chart: ${who ? `${who}: ` : ''}bits per character on unseen text while it trains, ${ys[0]} to ${ys[ys.length - 1]} (lower is better), step ${rows[rows.length - 1].step} of ${total}.`);
  }
  $('lnLoad').addEventListener('click', async () => {
    if (!L.grown) return;
    try { await psi.settings({ lastModel: L.grown }); await window.PSIOpenModel(L.grown); await check(); } catch (e) { err(e); }
  });

  // ── the scorecard ────────────────────────────────────────────────────────
  async function table() {
    let d; try { d = await call({ action: 'scorecard' }); } catch (e) { return err(e); }
    const tb = $('lnTable').querySelector('tbody'); tb.textContent = '';
    // a number that was not measured says so: never a dash, a blank or the word null
    const nm = (v, f = (x) => x) => (v == null ? 'not measured' : f(v));
    const lim = (d.limits && d.limits.understanding) || (L.limits && L.limits.understanding) || { floor: 6, appeared: 8 };
    const cells = (s) => [nm(s.params, fmtN), s.layers == null ? 'not measured' : `${s.layers} × ${s.width}`, nm(s.bits_per_char_street), nm(s.bits_per_char_general), s.understanding == null ? 'not measured' : `${s.understanding} of ${s.understanding_of}${s.understanding < lim.appeared ? ' · within floor' : ''}`, nm(s.copying_nats, (x) => `${x} nats`), nm(s.lookback_heads)];
    const add = (name, s, cls, title) => { const tr = el('tr', cls); const n = el('td', null, name); if (title) n.title = title; tr.appendChild(n); cells(s).forEach((v) => tr.appendChild(el('td', v === 'not measured' ? 'muted' : null, String(v)))); tb.appendChild(tr); };
    const head = (t) => { const tr = el('tr'); const td = el('td', 'rung', t); td.colSpan = 8; tr.appendChild(td); tb.appendChild(tr); };
    if (d.lineage.length) {
      head('your line');
      d.lineage.forEach((m) => {
        const s = m.score && m.score.after;
        // "loaded" is the model the engine runs now (the one Check it read); a model only remembered in the registry is not called that
        const name = `g${m.gen} ${m.id.slice(0, 8)}${m.street && m.street.place ? ` · ${m.street.place}` : ''}${m.id === (L.info && L.info.id) ? ' (loaded)' : ''}${s && s.measured_later ? ' · measured later, unsigned' : ''}`;
        if (s) add(name, s, s.measured_later ? 'yours later' : 'yours', `${m.file}\n${s.measured_later ? `measured ${s.measured_at || 'later'} with Measure it: not in the signed record` : m.by_em ? 'score from the record signed by The Emergence Machine' : ''}`);
        else { const tr = el('tr', 'yours'); tr.appendChild(el('td', null, name)); const td = el('td'); td.colSpan = 7; const b = el('button', 'ghost', 'Measure it'); b.addEventListener('click', async () => { b.disabled = true; b.textContent = 'measuring…'; try { await call({ action: 'score', id: m.id }); table(); } catch (e) { err(e); } }); td.appendChild(b); tr.appendChild(td); tb.appendChild(tr); }
      });
    }
    let rung = null;
    d.refs.forEach((x) => { if (x.rung !== rung) { rung = x.rung; head({ seed: 'the toy, trained in the browser', 'training time': 'one model across its training (Pythia-410m)', size: 'size, after full training (Pythia)', aligned: 'raw and aligned (Qwen3-4B)' }[rung] || rung); } add(x.name, x, '', x.note); });
    if (!d.refs.length) head('Reference rows not available (research/results/scorecard-refs.json is missing).');
    $('lnNote').textContent = d.note + (d.measured ? ` Reference rows measured ${d.measured}.` : '');
  }

  document.addEventListener('psi-model', () => { if (!$('tab-lineage').classList.contains('hidden')) check(); else { L.info = null; checked(false); PSI.block($('lnGrow'), 'verified', NOT_CHECKED); } });
  document.querySelector('#tabs button[data-tab="lineage"]').addEventListener('click', () => { check(); table(); });
  if (location.hash === '#lineage') setTimeout(() => { check(); table(); }, 1200);
  window.PSIViewData = window.PSIViewData || {};
  window.PSIViewData.lineage = () => ({ identity: L.info && { id: L.info.id, status: L.info.status, signed_by_em: L.info.by_em, record: L.info.record, note: L.info.note }, weight_signatures: L.marks || null });
})();
