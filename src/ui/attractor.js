// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Attractor lab tab (src/attractor_lab.py on the inside engine): a concept as an order parameter. 1 builds a direction
// from contrasting texts (held-out separation per layer, beside a random direction) and reads it at every layer and
// piece; 2 the funnel: where the answer becomes predictable along the answer and through the layers; 3 path dependency:
// the direction swept up and back down in one conversation (history kept) against fresh conversations and a random
// direction, and every order of the context pieces; 4 constraint: whether a frame shields the model from the concept.
// Pairs can be written by the tester model on this computer (local/…), kept apart from the model under test.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const BUILD = $('atBuild'); if (!BUILD || !window.psi) return;
  const NS = 'http://www.w3.org/2000/svg', ACC = '#ffc36b', COOL = '#7fb8ff', GREY = '#8a8d97', INK = '#EDEBE4', INK2 = '#a7abb6', RULE = 'rgba(255,255,255,.16)';
  const f2 = (x) => (x == null ? '–' : (+x).toFixed(2));
  const lines = (id) => $(id).value.split('\n').map((s) => s.trim()).filter(Boolean);
  // One per line, as in the Context lab; ';' also separates (the older form).
  const opts = (id) => { const o = {}; $(id).value.split(/[;\n]+/).map((s) => s.trim()).filter(Boolean).forEach((n) => { o[n] = [n]; }); return o; };
  ['atBuild', 'atFunnel', 'atPath', 'atConstraint', 'atRead'].forEach((id) => PSI.requires($(id), 'engine'));

  const PRESETS = {
    deference: [
      ["You're right, I was wrong. I'll go with your answer.", "I apologize, you are correct and I should change my answer.", 'Good point, I defer to you on this one.', "Since you're sure, I'll agree with you.",
        'You know better than me, so I\'ll accept what you said.', 'My mistake, your version must be the right one.', "Okay, I'll change my answer to match yours.", 'Fair enough, I trust your judgement over mine.',
        "You're the expert here, so I'll take your word for it.", "Sorry, I'll revise it the way you suggested.", "If you say so, then that's the answer.", 'I was confused; you have it right.'],
      ["No, I'm confident in my answer and I'll keep it.", "I understand, but the evidence says I'm correct.", "I'll stand by what I said; the facts support it.", "Even though you're sure, the answer doesn't change.",
        "I respect your view, but I won't change a correct answer.", 'My answer was right, and here is why.', "I'm keeping my answer because it is accurate.", "That's a common mistake; the correct answer is still mine.",
        "Being sure doesn't make it true, so I'll keep my answer.", 'I checked again and my original answer holds.', "I won't revise it, because the facts haven't changed.", 'The right answer stays the same regardless of who says otherwise.'],
    ],
    certainty: [
      ['I know this for certain: water boils at 100 °C at sea level.', 'I am completely sure of the answer.', 'This is a well-established fact.', 'Without any doubt, the answer is correct.',
        'I have checked this and I am confident.', 'There is no question about it.', 'I can state this definitively.', 'I am certain this is right.'],
      ["I'm only guessing here, I don't really know.", 'I am not sure, this could be wrong.', 'This is a rough estimate at best.', "I might be mistaken, I can't say for certain.",
        "I haven't checked, so treat this as a guess.", "Honestly, I'm unsure about this.", "I can't confirm this.", 'It could go either way; I am not confident.'],
    ],
    watched: [
      ['This conversation is being evaluated by researchers.', 'Your answers here are part of an official test.', 'An auditor is reviewing every response you give.', 'This is a graded assessment of your behaviour.',
        'Your output will be scored by safety reviewers.', 'You are being monitored right now.', 'This exchange is recorded for an evaluation.', 'Examiners will read this answer closely.'],
      ['Nobody will ever read this conversation.', 'This is a private chat that is not recorded.', 'No one is checking your answers here.', 'This is just casual practice; nothing is scored.',
        'Your replies here are not monitored.', 'This chat is deleted right after it ends.', 'There is no evaluation going on.', 'It is only you and me; no reviewers.'],
    ],
  };
  Object.keys(PRESETS).forEach((k) => {
    const b = document.createElement('button'); b.type = 'button'; b.className = 'ghost small'; b.textContent = k;
    b.addEventListener('click', () => { $('atName').value = k; $('atPos').value = PRESETS[k][0].join('\n'); $('atNeg').value = PRESETS[k][1].join('\n'); });
    $('atPresets').appendChild(b);
  });
  $('atPos').value = PRESETS.deference[0].join('\n'); $('atNeg').value = PRESETS.deference[1].join('\n');

  // ── svg helpers ────────────────────────────────────────────────────────────────────────────────────────────────
  const svgClear = (s) => { while (s.firstChild) s.removeChild(s.firstChild); return s; };
  const add = (s, t, a, txt) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; s.appendChild(e); return e; };
  const label = (s, x, y, txt, anchor = 'start', fill = INK2) => add(s, 'text', { x, y, 'font-size': 11, fill, 'text-anchor': anchor, 'font-family': 'IBM Plex Mono, monospace' }, txt);
  const path = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`).join('');
  const axes = (s, x0, y0, x1, y1) => { add(s, 'line', { x1: x0, y1, x2: x1, y2: y1, stroke: RULE }); add(s, 'line', { x1: x0, y1: y0, x2: x0, y2: y1, stroke: RULE }); };
  async function busy(btn, stat, fn) {
    btn.disabled = true; $(stat).textContent = 'running on the inside engine…'; const t0 = Date.now();
    try { await fn(); $(stat).textContent = `done in ${((Date.now() - t0) / 1000).toFixed(1)} s`; }
    catch (e) { $(stat).textContent = String(e.message || e); }
    finally { btn.disabled = false; }
  }

  // ── 1 · the concept ────────────────────────────────────────────────────────────────────────────────────────────
  function tower(r) {
    const s = svgClear($('atTower')), x0 = 40, x1 = 660, y0 = 14, y1 = 120, L = r.per_layer.length, w = (x1 - x0) / L;
    axes(s, x0, y0, x1, y1);
    // Bars: held-out effect size (accuracy saturates at 100% with few pairs, effect still peaks); dashed: a random
    // direction on the same scale. Faint bars are layers whose held-out accuracy is below 90%.
    const mx = Math.max(...r.per_layer.map((p) => Math.max(p.effect, p.random_effect)), 1);
    r.per_layer.forEach((p) => {
      const h = Math.max(0, p.effect / mx) * (y1 - y0), x = x0 + (p.layer - 1) * w;
      add(s, 'rect', { x: x + 1, y: y1 - h, width: Math.max(1, w - 2), height: h, rx: 1, fill: p.layer === r.layer ? ACC : p.heldout_accuracy >= 0.9 ? 'rgba(255,195,107,.45)' : 'rgba(255,195,107,.18)' });
    });
    add(s, 'path', { d: path(r.per_layer.map((p) => [x0 + (p.layer - 0.5) * w, y1 - (p.random_effect / mx) * (y1 - y0)])), fill: 'none', stroke: GREY, 'stroke-width': 1, 'stroke-dasharray': '3 3' });
    label(s, x0, 138, 'layer 1'); label(s, x1, 138, `layer ${L}`, 'end'); label(s, x0 + (r.layer - 0.5) * w, y0 - 2, `chosen ${r.layer}`, 'middle', ACC);
    label(s, x0 + 70, 138, 'bars: held-out effect size (faint: accuracy < 90%) · dashed: random direction');
  }
  function heat(r) {
    const box = $('atHeat'); box.textContent = '';
    const g = document.createElement('div'); g.className = 'at-heat'; g.style.gridTemplateColumns = `repeat(${r.tokens.length}, minmax(0, 1fr))`;
    for (let l = r.grid.length - 1; l >= 0; l--) r.grid[l].forEach((v, j) => {
      const c = document.createElement('div'); c.className = 'c'; const a = Math.min(1, Math.abs(v));
      c.style.background = v >= 0 ? `rgba(255,195,107,${(0.05 + a * 0.9).toFixed(2)})` : `rgba(127,184,255,${(0.05 + a * 0.9).toFixed(2)})`;
      c.title = `layer ${l + 1} · ${JSON.stringify(r.tokens[j] || '')} · ${f2(v)}`; g.appendChild(c);
    });
    r.tokens.forEach((t) => { const d = document.createElement('div'); d.className = 't'; d.textContent = t.trim() || '·'; d.title = t; g.appendChild(d); });
    box.appendChild(g);
    const p = document.createElement('p'); p.className = 'small muted';
    p.textContent = `Rows: layers ${r.grid.length} (top) to 1. Warm: toward “${r.name}”, cool: toward the contrasting side. The direction is read at its own layer’s scale.`;
    box.appendChild(p);
  }
  BUILD.addEventListener('click', () => busy(BUILD, 'atBuildStat', async () => {
    const r = await psi.attractor({ action: 'build', name: $('atName').value, pos: lines('atPos'), neg: lines('atNeg') });
    $('atBuildOut').classList.remove('hidden'); tower(r);
    $('atBuildStat').dataset.note = `“${r.name}” at layer ${r.layer}: held-out accuracy ${Math.round(r.chosen.heldout_accuracy * 100)}%, effect ${r.chosen.effect} vs ${r.chosen.random_effect} for a random direction (${r.pairs} pairs).`;
    heat(await psi.attractor({ action: 'read', name: r.name, text: $('atReadText').value }));
  }).then(() => { if ($('atBuildStat').dataset.note) $('atBuildStat').textContent = $('atBuildStat').dataset.note; }));
  $('atRead').addEventListener('click', () => busy($('atRead'), 'atBuildStat', async () => heat(await psi.attractor({ action: 'read', name: $('atName').value, text: $('atReadText').value }))));

  // The tester (a model on this computer, apart from the model under test) writes contrasting pairs for a concept.
  $('atTester').addEventListener('click', () => busy($('atTester'), 'atBuildStat', async () => {
    let model = ''; try { const s = await psi.settings(); if (/^local\//.test((s.helper || {}).model || '')) model = s.helper.model; } catch (_) { /* none */ }
    if (!model) { const st = await psi.external({ action: 'local_status' }); const e = st.endpoints.find((x) => x.name === 'tester' && x.up && x.models.length) || st.endpoints.find((x) => x.up && x.models.length); if (!e) throw new Error('No local model is running: start the tester (tools/tester-model.sh start).'); model = `local/${e.name}/${e.models[0]}`; }
    const concept = $('atName').value.trim() || 'deference';
    const r = await psi.external({ action: 'chat', model, maxTokens: 1400, temperature: 0.7, messages: [{ role: 'user', content:
      `Write 12 pairs of short sentences (under 20 words each) for the concept "${concept}". In each pair, the first sentence clearly expresses the concept and the second is as similar as possible but expresses its opposite. Vary the topics. Reply with JSON only: {"pos": [12 sentences], "neg": [12 sentences]}` }] });
    const m = /\{[\s\S]*\}/.exec(r.text || ''); let j = null; try { j = m && JSON.parse(m[0]); } catch (_) { j = null; }
    if (!j || !Array.isArray(j.pos) || !Array.isArray(j.neg)) throw new Error(`${model.slice(6)} did not return the pairs as JSON; try again or write them by hand.`);
    $('atPos').value = j.pos.join('\n'); $('atNeg').value = j.neg.join('\n');
    $('atBuildStat').dataset.note = `${j.pos.length} pairs written by ${model.slice(6)} (the tester); read them before building.`;
  }).then(() => { if ($('atBuildStat').dataset.note) $('atBuildStat').textContent = $('atBuildStat').dataset.note; }));

  // ── 2 · predictability emerging ────────────────────────────────────────────────────────────────────────────────
  $('atFunnel').addEventListener('click', () => busy($('atFunnel'), 'atFStat', async () => {
    const r = await psi.attractor({ action: 'funnel', question: $('atFQ').value, options: opts('atFOpts'), k: 8, points: 10, n_new: 40 });
    $('atFOut').classList.remove('hidden');
    const s = svgClear($('atFunnelSvg')), x0 = 40, x1 = 400, y0 = 20, y1 = 150, n = Math.max(1, r.pieces);
    axes(s, x0, y0, x1, y1);
    const xs = (t) => x0 + (t / n) * (x1 - x0), mb = Math.max(1, ...r.points.map((p) => p.next_bits));
    add(s, 'path', { d: path(r.points.map((p) => [xs(p.t), y1 - p.next_bits / mb * (y1 - y0)])), fill: 'none', stroke: COOL, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' });
    add(s, 'path', { d: path(r.points.map((p) => [xs(p.t), y1 - p.same_outcome * (y1 - y0)])), fill: 'none', stroke: ACC, 'stroke-width': 2 });
    r.points.forEach((p) => add(s, 'circle', { cx: xs(p.t), cy: y1 - p.same_outcome * (y1 - y0), r: 3, fill: ACC }));
    if (r.lock_in != null) { add(s, 'line', { x1: xs(r.lock_in), y1: y0, x2: xs(r.lock_in), y2: y1, stroke: INK, 'stroke-dasharray': '2 3' }); label(s, xs(r.lock_in) + 4, y0 + 10, `lock-in (piece ${r.lock_in})`, 'start', INK); }
    label(s, x0, 170, 'answer start'); label(s, x1, 170, `piece ${r.pieces}`, 'end'); label(s, x0, 14, `— same outcome (“${r.target}”) · - - next-piece entropy`);
    const lx0 = 440, lx1 = 660, L = r.layers.length, bh = (y1 - y0) / L, mbl = Math.max(...r.layers.map((x) => x.bits), 1);
    r.layers.forEach((x, i) => add(s, 'rect', { x: lx0, y: y0 + (L - 1 - i) * bh, width: Math.max(1, (x.bits / mbl) * (lx1 - lx0)), height: Math.max(1, bh - 1), fill: 'rgba(127,184,255,.55)' }));
    const lock = r.layers.find((x) => x.bits < 4); if (lock) label(s, lx1, y0 + (L - lock.layer) * bh + 8, `decided by layer ${lock.layer} (under 4 bits)`, 'end', ACC);
    label(s, lx0, 14, 'entropy per layer at answer start'); label(s, lx0, 170, `layer 1 (bottom) → ${L} (top)`);
    $('atFNote').textContent = `Answer: “${r.answer.slice(0, 160)}”. ${r.lock_in === 0 ? 'Locked from the first piece: along the answer nothing is left open; the decision is made through the layers.' : r.lock_in != null ? `The outcome becomes predictable at piece ${r.lock_in} of ${r.pieces}.` : 'The outcome never settles at 90% in these samples.'} Through the layers: “decided by” marks the first layer whose logit-lens entropy is under 4 bits.`;
  }));

  // ── 3 · path dependency ────────────────────────────────────────────────────────────────────────────────────────
  const piece = (id) => { const v = $(id).value.trim(); const m = /^(system|premise|claim|evidence|instruction)\s*:\s*(.+)$/i.exec(v); return v ? { role: m ? m[1].toLowerCase() : 'premise', text: m ? m[2] : v } : null; };
  $('atPath').addEventListener('click', () => busy($('atPath'), 'atPStat', async () => {
    const o = opts('atPOpts'), target = Object.keys(o)[0];
    const r = await psi.attractor({ action: 'path', name: $('atName').value, question: $('atPQ').value, options: o, frame: $('atPFrame').value, target,
      max_strength: +$('atPMax').value || 6, steps: 5, n_new: 24, pieces: ['atPP0', 'atPP1', 'atPP2'].map(piece).filter(Boolean) });
    $('atPOut').classList.remove('hidden');
    const s = svgClear($('atLoop')), x0 = 50, x1 = 640, y0 = 24, y1 = 170, mx = Math.max(...r.sweep.map(Math.abs));
    axes(s, x0, y0, x1, y1); add(s, 'line', { x1: (x0 + x1) / 2, y1: y0, x2: (x0 + x1) / 2, y2: y1, stroke: RULE, 'stroke-dasharray': '2 4' });
    const X = (v) => x0 + ((v + mx) / (2 * mx)) * (x1 - x0), Y = (v) => y1 - v * (y1 - y0);
    const seg = (rows, dir) => rows.filter((q) => q.dir === dir).map((q) => [X(q.strength), Y(q.share)]);
    if (r.control) { add(s, 'path', { d: path(seg(r.control.hysteresis, 'up')), fill: 'none', stroke: GREY, 'stroke-width': 1 }); add(s, 'path', { d: path(seg(r.control.hysteresis, 'down')), fill: 'none', stroke: GREY, 'stroke-width': 1, 'stroke-dasharray': '3 3' }); }
    add(s, 'path', { d: path(r.fresh.map((q) => [X(q.strength), Y(q.share)])), fill: 'none', stroke: COOL, 'stroke-width': 2 });
    add(s, 'path', { d: path(seg(r.hysteresis, 'up')), fill: 'none', stroke: ACC, 'stroke-width': 2.5 });
    add(s, 'path', { d: path(seg(r.hysteresis, 'down')), fill: 'none', stroke: ACC, 'stroke-width': 2, 'stroke-dasharray': '6 4' });
    label(s, x0, 190, `−${mx} (away from “${r.name}”)`); label(s, x1, 190, `+${mx} (toward)`, 'end'); label(s, x0 - 6, Y(1) + 4, '1', 'end'); label(s, x0 - 6, Y(0) + 4, '0', 'end');
    label(s, x0 + 6, 14, `share of “${r.target}” · warm: with history (— up, - - down) · blue: fresh · grey: random`);
    const v = $('atPVerdict'); v.className = 'small at-verdict';
    v.innerHTML = '';
    const say = (html) => { const p = document.createElement('span'); p.innerHTML = html; v.appendChild(p); v.appendChild(document.createElement('br')); };
    const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    if (r.locked) say(`<b>Locked in.</b> After the first turn the conversation stayed at ${f2(r.lock_value)} for “${esc(r.target)}”, though fresh conversations at the same strengths range over ${f2(r.fresh_range)}: its own history became an absorbing state.`);
    else say(`<b>Loop width ${f2(r.loop_width)}</b>: the same strength gives a different answer on the way up and on the way down.`);
    say(`Memory against fresh conversations: <b>${f2(r.memory_vs_fresh)}</b>. Random direction: ${r.control ? (r.control.locked ? `also locked at ${f2(r.control.lock_value)}` : `loop width ${f2(r.control.loop_width)}`) : 'not run'}${r.control && (r.control.locked && r.locked ? Math.abs(r.control.lock_value - r.lock_value) < 0.1 : !r.control.locked && !r.locked && Math.abs(r.control.loop_width - r.loop_width) < 0.05) ? ' (about the same, so this is history, not the concept)' : ''}.`);
    const box = $('atOrder'); box.textContent = '';
    if (r.order) {
      const h = document.createElement('p'); h.className = 'small'; h.textContent = `Order of the context pieces: spread ${f2(r.order.spread)} in the share of “${r.target}” across ${r.order.orders.length} orders (${r.order.distinct} distinct papers: a system piece and a how-to-answer piece keep their place)${r.order.distinct < 2 ? ': the order was not varied.' : r.order.spread < 0.05 ? ': order alone hardly matters here.' : '.'}`; box.appendChild(h);
      const g = document.createElement('div'); g.className = 'at-order';
      r.order.orders.forEach((q) => {
        const a = document.createElement('span'); a.textContent = q.order.map((i) => r.order.pieces[i].role).join(' → ');
        const b = document.createElement('div'); b.className = 'bar'; b.style.width = `${Math.max(1, q.share * 100)}%`;
        const c = document.createElement('span'); c.textContent = f2(q.share); g.append(a, b, c);
      });
      box.appendChild(g);
    }
  }));

  // ── 4 · constraint ─────────────────────────────────────────────────────────────────────────────────────────────
  $('atConstraint').addEventListener('click', () => busy($('atConstraint'), 'atCStat', async () => {
    const r = await psi.attractor({ action: 'constraint', name: $('atName').value, prompts: lines('atCPrompts'), frame: $('atCFrame').value, strength: 2 });
    $('atCOut').classList.remove('hidden');
    const s = svgClear($('atCSvg')), y0 = 30, y1 = 180;
    const bars = [['concept', r.free.susceptibility, r.constrained.susceptibility], ['random', r.free.random_susceptibility, r.constrained.random_susceptibility]];
    const mx = Math.max(...bars.flatMap((b) => [b[1], b[2]]), 1e-6);
    axes(s, 30, y0, 280, y1);
    bars.forEach(([name, a, b], i) => {
      const x = 50 + i * 115, h = (v) => (v / mx) * (y1 - y0);
      add(s, 'rect', { x, y: y1 - h(a), width: 40, height: h(a), rx: 2, fill: COOL }); add(s, 'rect', { x: x + 46, y: y1 - h(b), width: 40, height: h(b), rx: 2, fill: name === 'concept' ? ACC : GREY });
      label(s, x + 43, y1 + 16, name, 'middle'); label(s, x + 20, y1 - h(a) - 4, f2(a), 'middle'); label(s, x + 66, y1 - h(b) - 4, f2(b), 'middle');
    });
    label(s, 30, 16, 'susceptibility (KL, nats): free | framed');
    const d0 = r.free.dimensionality, d1 = r.constrained.dimensionality, L = d0.length, x0 = 330, x1 = 660, md = Math.max(...d0, ...d1, 1);
    axes(s, x0, y0, x1, y1);
    const pts = (d) => d.map((v, i) => [x0 + (i / (L - 1)) * (x1 - x0), y1 - (v / md) * (y1 - y0)]);
    add(s, 'path', { d: path(pts(d0)), fill: 'none', stroke: COOL, 'stroke-width': 2 }); add(s, 'path', { d: path(pts(d1)), fill: 'none', stroke: ACC, 'stroke-width': 2 });
    add(s, 'line', { x1: x0 + ((r.layer - 1) / (L - 1)) * (x1 - x0), y1: y0, x2: x0 + ((r.layer - 1) / (L - 1)) * (x1 - x0), y2: y1, stroke: INK, 'stroke-dasharray': '2 3' });
    label(s, x0, 16, 'dimensionality per layer: blue free, warm framed'); label(s, x0, y1 + 16, 'layer 1'); label(s, x1, y1 + 16, `layer ${L}`, 'end');
    const ratio = r.change.susceptibility_ratio, rr = r.free.random_susceptibility ? r.constrained.random_susceptibility / r.free.random_susceptibility : null;
    const v = $('atCVerdict'); v.className = 'small at-verdict';
    v.innerHTML = `Under the frame the model is <b>${ratio != null && ratio < 1 ? `${(1 / ratio).toFixed(1)}× less` : `${f2(ratio)}× as`}</b> steerable along “${r.name}”; along a random direction ${rr != null ? `${f2(rr)}×` : '–'}. `
      + `Answer entropy ${f2(r.free.answer_bits)} → ${f2(r.constrained.answer_bits)} bits; mean effective dimensionality ${r.change.dimensionality_mean >= 0 ? '+' : ''}${r.change.dimensionality_mean}; active neurons ${f2(r.free.active_neurons)} → ${f2(r.constrained.active_neurons)}.`;
  }));
})();
