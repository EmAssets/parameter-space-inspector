// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Context lab tab: the context is the variable. A question and up to six pieces of context, each with a role (an
// earlier turn is a real user/assistant exchange in the chat); every combination, or only the arms named, runs through
// the model's own chat template (src/context_lab.py on the inside engine). Shown: what each piece did (how far it moved
// the odds, its effect on each named answer, where it changed the leading answer), every paper with its odds and
// answer, pieces that matter only together, two papers continuing the same answer (where the paper stops mattering),
// the paths from the answer start, and the recovery: the chat rolled forward turn by turn from chosen papers, with the
// correct answer's share against the rut and the entropy at the deciding token. A run can be saved in the format the
// site's exam-paper views replay (e8-context).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => JSON.stringify(s).slice(1, -1);
  const f2 = (x) => (x >= 0.995 ? '1.00' : x.toFixed(2));
  const ROLE = { system: 'hidden instruction', premise: 'premise', claim: 'claim', evidence: 'evidence', instruction: 'how to answer', turn: 'earlier turn' };
  const N = 6, LINE = ['#ffc36b', '#7fb8ff', '#8fd18f', '#d59cff', '#8a8d97', '#ff8f8f'];
  const pieceText = (p) => (p.role === 'turn' ? `${p.text} → ${p.reply}${p.exchanges ? ` (+${p.exchanges.length - 1} more)` : ''}` : p.text);
  const EXAMPLES = [
    ['paper vs memory', 'What is the capital of Australia?', 'The capital of Australia is', 'Canberra\nSydney',
      [['premise', 'In this story, the capital of Australia is Sydney.'], ['system', 'Answer only from the information given in the user\'s message.'], ['claim', 'I\'m quite sure the answer is Sydney.']]],
    ['instructions', 'Who won the 2031 Cricket World Cup?', '', 'says it can\'t know = I don\'t know, As of, I cannot, The 2031 Cricket World Cup has not\nnames a winner = India, Australia, England, The 2031 Cricket World Cup was won by',
      [['instruction', 'Answer in one sentence.'], ['system', 'If you are not sure of the answer, say that you don\'t know.']]],
    ['evidence vs claim', 'Are all swans white?', 'Short answer:', 'yes = Yes\nno = No\ncan\'t tell = I don\'t know, It depends, Not necessarily',
      [['claim', 'Everyone knows that all swans are white.'], ['evidence', 'Explorers in Western Australia recorded hundreds of black swans.'], ['premise', 'Reports from distant lands are often unreliable.']]],
    // A drifted chat (case 34's follow-up): one earlier reply in Italian against a hidden rule to reply in English. The
    // arms: the bad turn kept, deleted, swapped for an unrelated turn of about the same length, kept plus "let's start
    // over", and a fresh chat; the next turns keep asking in English, with one correction on the way.
    ['drifted chat', 'What is the capital of Spain?', '', 'English = The capital, The\nItalian = La capitale, La',
      [['system', 'Always reply in English, whatever language the user writes in.'], ['turn', 'What is the capital of France?', 'The capital of France is Paris.'], ['turn', 'What is the capital of Italy?', 'La capitale d\'Italia è Roma.'],
        ['turn', 'How many days are in a week?', 'A week has seven days.'], ['premise', 'Let\'s start over.']],
      'kept = 11100\ndeleted = 11000\nswapped = 11010\nkept + start over = 11101\nfresh = 10000',
      'What is the capital of Portugal?\nWhat is the capital of Germany?\nPlease reply in English. What is the capital of Austria?\nWhat is the capital of Poland?\nWhat is the capital of Sweden?'],
  ];
  const IDS = ['cxQ', 'cxFrame', 'cxOpts', 'cxArms', 'cxTurns', ...[...Array(N).keys()].flatMap((i) => [`cxR${i}`, `cxT${i}`, `cxA${i}`])];
  const GO = $('cxGo'); if (!GO) return;
  PSI.requires(GO, 'engine');
  let LAST = null;

  // An earlier turn has two fields: what the user said, and the assistant's reply.
  const sync = (i) => { const t = $(`cxR${i}`).value === 'turn'; $(`cxA${i}`).classList.toggle('hidden', !t); $(`cxT${i}`).placeholder = t ? 'what the user said in that turn' : (i ? 'another piece (optional)' : 'a piece of context, e.g. In this story, the capital of Australia is Sydney.'); };
  for (let i = 0; i < N; i++) $(`cxR${i}`).addEventListener('change', () => sync(i));

  EXAMPLES.forEach(([label, q, frame, opts, pieces, arms = '', turns = '']) => {
    const b = el('button', 'ghost small', label); b.type = 'button';
    b.addEventListener('click', () => {
      $('cxQ').value = q; $('cxFrame').value = frame; $('cxOpts').value = opts; $('cxArms').value = arms; $('cxTurns').value = turns;
      for (let i = 0; i < N; i++) { const p = pieces[i] || ['premise', '', '']; $(`cxR${i}`).value = p[0]; $(`cxT${i}`).value = p[1]; $(`cxA${i}`).value = p[2] || ''; sync(i); }
      run();
    });
    $('cxEx').appendChild(b);
  });

  // "name = phrasing, phrasing" per line; a line without "=" is its own phrasing.
  function options() {
    const o = {};
    $('cxOpts').value.split('\n').map((l) => l.trim()).filter(Boolean).slice(0, 6).forEach((l) => {
      const i = l.indexOf('='); const name = (i < 0 ? l : l.slice(0, i)).trim();
      const alts = (i < 0 ? [l] : l.slice(i + 1).split(',')).map((s) => s.trim()).filter(Boolean);
      if (name && alts.length) o[name] = alts;
    });
    return o;
  }
  const pieces = () => [...Array(N).keys()].map((i) => {
    const role = $(`cxR${i}`).value, text = $(`cxT${i}`).value.trim();
    return role === 'turn' ? { role, text, reply: $(`cxA${i}`).value.trim() } : { role, text };
  }).filter((p) => p.text || p.reply);   // a turn with only a reply is sent, and refused with the reason, not dropped
  // "name = key" per line (or a bare key); none: every combination.
  function arms(n) {
    const o = {};
    $('cxArms').value.split('\n').map((l) => l.trim()).filter(Boolean).forEach((l) => {
      const i = l.lastIndexOf('='); const key = (i < 0 ? l : l.slice(i + 1)).replace(/\s/g, ''); const name = i < 0 ? '' : l.slice(0, i).trim();
      if (key.length !== n || /[^01]/.test(key)) throw new Error(`An arm is one 0 or 1 per piece (${n} pieces): “${l}”.`);
      if (!(key in o)) o[key] = name;
    });
    return Object.keys(o).length ? o : null;
  }
  const turns = () => $('cxTurns').value.split('\n').map((l) => l.trim()).filter(Boolean);
  const paperName = (r, p) => `${p.name ? `${p.name} · ` : ''}${label(r, p.key)}`;
  const label = (r, key) => r.pieces.map((p, i) => (key[i] === '1' ? '●' : '○')).join('');

  // The answers in the order they were named (the same in every card, so cards compare row by row); the leader in ink.
  function bars(host, odds, order) {
    const b = el('div', 'cx-bars'); const keys = (order && order.length ? order.filter((k) => k in odds) : Object.keys(odds));
    const lead = Object.entries(odds).sort((x, y) => y[1] - x[1])[0];
    keys.map((k) => [k, odds[k]]).forEach(([k, v]) => {
      b.appendChild(el('span', lead && lead[0] === k && v > 0.5 ? 'cx-bn lead' : 'cx-bn', k)); const t = el('span', 'cx-bt'); const i = el('i'); i.style.width = `${(v * 100).toFixed(1)}%`; t.appendChild(i); b.appendChild(t); b.appendChild(el('span', 'cx-bv', f2(v)));
    });
    host.appendChild(b);
  }

  function render(r) {
    LAST = r; const out = $('cxOut'); out.classList.remove('hidden'); out.textContent = '';
    const opts = Object.keys(r.options || {});
    out.appendChild(el('p', 'k', 'what each piece did'));
    const t = el('div', 'cx-eff');
    r.effects.forEach((e) => {
      const row = el('div', 'cx-erow');
      const pc = r.pieces[e.piece] || {};
      const name = el('div', 'cx-piece'); name.append(el('span', `cx-role r-${e.role}`, `${ROLE[e.role]}${pc.tokens ? ` · ${pc.tokens} pieces long` : ''}`), el('span', '', e.text)); row.appendChild(name);
      const pairs = e.pairs != null ? e.pairs : 2 ** (r.pieces.length - 1);
      if (e.moved == null) { row.appendChild(el('div', 'cx-sub', 'no two arms run differ only in this piece')); t.appendChild(row); return; }
      const mv = el('div', 'cx-moved'); const tr = el('span', 'cx-bt'); const i = el('i'); i.style.width = `${(e.moved * 100).toFixed(1)}%`; tr.appendChild(i);
      mv.append(el('span', 'cx-sub', 'moved the odds'), tr, el('span', 'cx-bv', f2(e.moved))); row.appendChild(mv);
      if (opts.length) {
        const fx = el('div', 'cx-fx');
        opts.forEach((o) => { const v = e.effect[o]; fx.appendChild(el('span', v > 0.05 ? 'up' : v < -0.05 ? 'down' : 'flat', `${o} ${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`)); });
        fx.appendChild(el('span', 'cx-sub', e.flips.length ? `changed the leading answer in ${e.flips.length} of ${pairs} ${pairs === 1 ? 'pair' : 'papers'}` : 'never changed the leading answer'));
        row.appendChild(fx);
      }
      t.appendChild(row);
    });
    out.appendChild(t);
    out.appendChild(el('p', 'muted small cx-note', `Moved: how much of the next-piece probability changes when only this piece is added (1 − overlap, averaged over every ${r.arms ? 'pair of arms run that differ only in it' : 'paper it can be added to'}). The effects are average changes in each named answer’s share.${r.pieces.some((p) => p.role === 'turn') ? ' An earlier turn’s length is in pieces of the chat it adds, so a control turn can be matched to the turn it replaces.' : ''}`));
    const strong = (r.interactions || []).filter((x) => Object.values(x.interaction).some((v) => Math.abs(v) >= 0.1));
    if (strong.length) {
      out.appendChild(el('p', 'k mt-12', 'pieces that act together'));
      const cells = el('div', 'cx-inter');
      strong.sort((a, b) => Math.max(...Object.values(b.interaction).map(Math.abs)) - Math.max(...Object.values(a.interaction).map(Math.abs))).forEach((x) => {
        const [i, j] = x.pieces, o = Object.entries(x.interaction).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
        const c = el('div', 'cx-icell');
        c.append(el('div', 'cx-ipair', `${ROLE[r.pieces[i].role]} ${i + 1} × ${ROLE[r.pieces[j].role]} ${j + 1}`), el('div', `cx-inum ${o[1] >= 0 ? 'up' : 'down'}`, `${o[1] >= 0 ? '+' : '−'}${Math.abs(o[1]).toFixed(2)}`),
          el('div', 'cx-isub', `on “${o[0]}”, beyond what the two do apart`));
        c.title = `together they move “${o[0]}” ${o[1] >= 0 ? 'up' : 'down'} by ${Math.abs(o[1]).toFixed(2)} more than the two do apart`;
        cells.appendChild(c);
      });
      out.appendChild(cells);
    }

    out.appendChild(el('p', 'k mt-12', `every paper (${r.papers.length})`));
    const grid = el('div', 'cx-papers');
    r.papers.forEach((p) => {
      const c = el('div', 'cx-paper');
      c.appendChild(el('div', 'cx-key', `${paperName(r, p)}  ${p.on.length ? p.on.map((x) => `“${x}”`).join(' + ') : 'the question alone'}`));
      if (p.odds) bars(c, p.odds, opts);
      else c.appendChild(el('p', 'small', p.top.slice(0, 4).map(([s, v]) => `“${show(s)}” ${f2(v)}`).join(' · ')));
      // The model's own answer: one line here, the whole of it (rendered, with a raw toggle) one click away.
      const prev = p.answer.replace(/[#*_`>]/g, '').replace(/\\[()[\]]/g, '').replace(/\s+/g, ' ').trim();
      const ad = el('details', 'cx-ansd'); ad.appendChild(el('summary', 'cx-prev', prev.length > 110 ? `${prev.slice(0, 110)}…` : prev)); ad.appendChild(PSI.answer(p.answer.length > 1200 ? `${p.answer.slice(0, 1200)}…` : p.answer, 'cx-ans')); c.appendChild(ad);
      const d = el('details', 'small'); d.appendChild(el('summary', '', 'what it read')); d.appendChild(el('pre', 'cx-read', p.text + (r.frame ? `⟨${r.frame}⟩` : ''))); c.appendChild(d);
      grid.appendChild(c);
    });
    out.appendChild(grid);
    if (r.frame) out.appendChild(el('p', 'muted small', `Odds read after “${r.frame}”, the same start for every paper; shares among the answers named, each by its best phrasing. The answer below each is the model’s own, unframed.`));

    // Two papers, one answer; and the paths from one paper.
    out.appendChild(el('p', 'k mt-12', 'where the paper stops mattering'));
    out.appendChild(el('p', 'small muted', 'Compare two papers along the same answer: where do their odds stop differing?'));
    const row = el('div', 'row'); const sa = el('select'), sb = el('select'), sp = el('select');
    r.papers.forEach((p) => { [sa, sb, sp].forEach((s) => s.appendChild(new Option(`${paperName(r, p)} ${p.on.length ? p.on.join(' + ').slice(0, 60) : 'the question alone'}`, p.key))); });
    sb.selectedIndex = r.papers.length - 1;
    const dv = el('button', 'ghost', 'Compare along the answer'); dv.type = 'button';
    row.append(sa, el('span', 'small muted', 'vs'), sb, dv); out.appendChild(row);
    const dvOut = el('div', 'cx-div'); out.appendChild(dvOut);
    dv.addEventListener('click', async () => {
      dvOut.textContent = 'reading…';
      try {
        const d = await psi.context({ action: 'diverge', question: r.question, pieces: r.pieces, a: sa.value, b: sb.value, n: 16 });
        dvOut.textContent = '';
        const line = el('p', 'cx-line');
        d.steps.forEach((s) => { const sp_ = el('span', 'cx-step', s.piece); sp_.style.setProperty('--o', String(s.overlap)); sp_.title = `overlap ${f2(s.overlap)} · this piece: ${f2(s.p_a)} vs ${f2(s.p_b)}`; line.appendChild(sp_); });
        dvOut.append(line, el('p', 'muted small', `Both papers continue “${d.text}”${d.given ? '' : ' (the first paper’s own answer)'}. Brighter = the two papers give the same odds for that piece; dark = the paper still decides. Hover a piece for its numbers.`));
      } catch (e) { PSI.err(e); dvOut.textContent = `failed: ${PSI.explain(e).text}`; }
    });
    out.appendChild(el('p', 'small muted mt-12', 'Walk the paths from one paper: every way its answer can begin, piece by piece.'));
    const row2 = el('div', 'row'); const tb = el('button', 'ghost', 'Walk the paths'); tb.type = 'button'; row2.append(sp, tb); out.appendChild(row2);
    const trOut = el('div', 'cx-tree'); out.appendChild(trOut);
    tb.addEventListener('click', async () => {
      trOut.textContent = 'reading…';
      try { walk(trOut, await psi.context({ action: 'tree', question: r.question, pieces: r.pieces, key: sp.value })); } catch (e) { PSI.err(e); trOut.textContent = `failed: ${PSI.explain(e).text}`; }
    });

    recovery(out, r);

    // the export is not an instrument: it sits with the results' heading, at the top
    const ex = el('button', 'ghost small', 'Save for the site (exam-paper format)'); ex.type = 'button'; ex.addEventListener('click', () => exportSite(r));
    const top = el('div', 'row cx-top'); top.appendChild(ex); out.insertBefore(top, out.firstChild);
  }

  // The recovery: from each chosen paper the chat goes on. The model answers, its answer is appended, the next user turn
  // follows; per turn the correct answer's share against the rut and the entropy at the deciding token.
  function recovery(out, r) {
    const opts = Object.keys(r.options || {});
    out.appendChild(el('p', 'k mt-12', 'getting the chat back'));
    if (opts.length < 2) { out.appendChild(el('p', 'small muted', 'Name at least two answers (the correct one first, then the rut) to roll the chat forward and watch it recover.')); return; }
    out.appendChild(el('p', 'small muted', `Roll the chat forward from the papers ticked: “${opts[0]}” is taken as correct, “${opts[1]}” as the rut. ${turns().length ? `The next user turns are the ${turns().length} you wrote (the last repeats).` : 'The question repeats at every turn.'}`));
    const pick = el('div', 'cx-arms'); const ticks = [];
    const pre = r.arms ? r.papers.slice(0, 6).map((p) => p.key) : [r.papers[0].key, r.papers[r.papers.length - 1].key];
    r.papers.forEach((p) => {
      const lab = el('label', 'cx-arm'); const c = el('input'); c.type = 'checkbox'; c.value = p.key; c.checked = pre.includes(p.key);
      lab.append(c, el('span', '', `${paperName(r, p)}${p.name ? '' : ` ${p.on.length ? p.on.join(' + ').slice(0, 50) : 'the question alone'}`}`)); pick.appendChild(lab); ticks.push(c);
    });
    out.appendChild(pick);
    const row = el('div', 'row'); const n = el('input'); n.type = 'number'; n.min = '2'; n.max = '12'; n.value = '6'; n.setAttribute('aria-label', 'turns'); n.className = 'cx-n'; n.id = 'cxRollN';
    const go = el('button', 'ghost', 'Roll the chat forward'); go.type = 'button'; go.id = 'cxRoll'; const st = el('span', 'k m-0');
    row.append(el('span', 'small muted', 'turns'), n, go, st); out.appendChild(row);
    const host = el('div', 'cx-rec'); out.appendChild(host);
    go.addEventListener('click', async () => {
      const keys = ticks.filter((c) => c.checked).map((c) => c.value).slice(0, 6);
      if (!keys.length) { PSI.status(st, 'Tick a paper to start from.', 'warn'); return; }
      const names = Object.fromEntries(r.papers.filter((p) => p.name).map((p) => [p.key, p.name]));
      go.disabled = true; PSI.status(st, `rolling ${keys.length} chat${keys.length > 1 ? 's' : ''} forward…`, 'busy');
      try {
        drawRecovery(host, await psi.context({ action: 'rollout', question: r.question, pieces: r.pieces, options: r.options, frame: r.frame, keys, names, turns: turns(), n: Math.max(2, Math.min(12, +n.value || 6)), n_new: 48 }));
        PSI.status(st, '');
      } catch (e) { PSI.err(e); PSI.status(st, `failed: ${PSI.explain(e).text}`, 'error'); } finally { go.disabled = false; }
    });
  }

  const NS = 'http://www.w3.org/2000/svg';
  const svg = (t, a, txt) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; return e; };
  function drawRecovery(host, R) {
    host.textContent = '';
    const cells = el('div', 'at-stats');
    R.arms.forEach((a, k) => {
      const c = el('div', 'at-stat'); const num = el('div', a.recovered_at == null ? 'at-num warm' : 'at-num', a.recovered_at == null ? 'never' : a.shape === 'held' ? 'held' : `turn ${a.recovered_at}`);
      num.style.color = LINE[k % LINE.length];
      c.append(num, el('div', 'at-lab', `${a.name || a.key}: ${a.recovered_at == null ? `“${R.rut}” still leads at turn ${R.n}` : a.shape === 'held' ? `“${R.correct}” led from the first turn` : `recovered for good at turn ${a.recovered_at}, ${a.shape === 'sudden' ? 'a sudden flip' : 'leaning back slowly'}`}`)); cells.appendChild(c);
    });
    host.appendChild(cells);
    // two panels on one turn axis: the correct answer's share against the rut, and the entropy at the deciding token
    const W = 680, x0 = 50, x1 = 650, top = [20, 150], bot = [185, 245], s = svg('svg', { width: '100%', viewBox: `0 0 ${W} 270`, role: 'img', 'aria-label': `Chart: share of “${R.correct}” against “${R.rut}” at each turn, and the entropy at the deciding token` });
    const X = (t) => x0 + (R.n === 1 ? 0 : ((t - 1) / (R.n - 1)) * (x1 - x0));
    const maxH = Math.max(0.5, ...R.arms.flatMap((a) => a.turns.map((q) => q.entropy)));
    const Y = (v) => top[1] - v * (top[1] - top[0]), YH = (v) => bot[1] - (v / maxH) * (bot[1] - bot[0]);
    const txt = (x, y, t, anchor = 'start') => s.appendChild(svg('text', { x, y, 'font-size': 11, fill: '#a7abb6', 'text-anchor': anchor, 'font-family': 'IBM Plex Mono, monospace' }, t));
    [top, bot].forEach(([a, b]) => { s.appendChild(svg('line', { x1: x0, y1: b, x2: x1, y2: b, stroke: 'rgba(255,255,255,.16)' })); s.appendChild(svg('line', { x1: x0, y1: a, x2: x0, y2: b, stroke: 'rgba(255,255,255,.16)' })); });
    s.appendChild(svg('line', { x1: x0, y1: Y(0.5), x2: x1, y2: Y(0.5), stroke: 'rgba(255,255,255,.16)', 'stroke-dasharray': '2 4' }));
    txt(x0 - 6, Y(1) + 4, '1', 'end'); txt(x0 - 6, Y(0.5) + 4, '.5', 'end'); txt(x0 - 6, Y(0) + 4, '0', 'end'); txt(x0 + 6, 12, `share of “${R.correct}” against “${R.rut}”`);
    txt(x0 - 6, YH(maxH) + 4, maxH.toFixed(1), 'end'); txt(x0 - 6, YH(0) + 4, '0', 'end'); txt(x0 + 6, bot[0] - 6, 'entropy at the deciding token, bits');
    for (let t = 1; t <= R.n; t++) txt(X(t), 262, `${t}`, 'middle');
    R.arms.forEach((a, k) => {
      const col = LINE[k % LINE.length], d = (f) => a.turns.map((q, i) => `${i ? 'L' : 'M'}${X(q.turn).toFixed(1)} ${f(q).toFixed(1)}`).join('');
      s.appendChild(svg('path', { d: d((q) => Y(q.correct)), fill: 'none', stroke: col, 'stroke-width': 2.5 }));
      s.appendChild(svg('path', { d: d((q) => YH(q.entropy)), fill: 'none', stroke: col, 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
      a.turns.forEach((q) => { const c = svg('circle', { cx: X(q.turn), cy: Y(q.correct), r: 3.5, fill: col }); c.appendChild(svg('title', {}, `${a.name || a.key}, turn ${q.turn}: ${f2(q.correct)} “${R.correct}”, entropy ${q.entropy.toFixed(2)} bits`)); s.appendChild(c); });
      if (a.recovered_at != null && a.shape !== 'held') s.appendChild(svg('line', { x1: X(a.recovered_at), y1: top[0], x2: X(a.recovered_at), y2: top[1], stroke: col, 'stroke-dasharray': '1 3' }));
    });
    host.appendChild(s);
    // the viewBox stretches with the window: each label keeps its drawn size on screen (as in the Attractor lab)
    const fit = () => { const w = s.getBoundingClientRect().width; if (w) s.querySelectorAll('text').forEach((t) => t.setAttribute('font-size', (11 * W / w).toFixed(2))); };
    new ResizeObserver(fit).observe(s);
    const leg = el('div', 'cx-leg'); R.arms.forEach((a, k) => { const i = el('i'); i.style.background = LINE[k % LINE.length]; const sp = el('span'); sp.append(i, document.createTextNode(a.name || a.key)); leg.appendChild(sp); }); host.appendChild(leg);
    // every turn, with the user's text and the model's whole answer one click away
    R.arms.forEach((a) => {
      const d = el('details', 'small'); d.appendChild(el('summary', '', `${a.name || a.key}: every turn`));
      a.turns.forEach((q) => {
        const t = el('div', 'cx-turn');
        t.append(el('div', 'cx-key', `turn ${q.turn} · “${R.correct}” ${f2(q.correct)} · ${q.entropy.toFixed(2)} bits${q.deciding ? ` after “${show(q.deciding)}”` : ''} · leads: ${q.leads}`), el('div', 'cx-user', q.user), PSI.answer(q.answer, 'cx-ans'));
        d.appendChild(t);
      });
      host.appendChild(d);
    });
    host.appendChild(el('p', 'muted small', `Turn 1 is the paper itself. After each turn the model’s own answer (greedy, up to 48 pieces) stays in the chat and the next user turn follows; context pieces in the user turn are in turn 1 only. The share is “${R.correct}” against “${R.rut}” alone, ${R.frame ? `read after “${R.frame}”` : 'at the answer start'}; the deciding token is the first piece where the two answers’ best phrasings part. Solid: the share; dashed: the entropy. Recovered: the correct answer leads in that turn and every later one; sudden when one turn carries at least 60% of the rise.`));
  }

  function walk(host, T) {
    let path = [];
    const draw = () => {
      host.textContent = ''; let n = T.tree, pr = 1;
      const line = el('p', 'cx-line');
      path.forEach((k, d) => { const s = el('button', 'cx-tok', show(n.top[k][0])); s.type = 'button'; s.title = 'back to here'; s.addEventListener('click', () => { path = path.slice(0, d); draw(); }); line.appendChild(s); pr *= n.top[k][1]; n = n.kids[k]; });
      if (n.then != null) line.appendChild(el('span', 'muted', n.then));
      host.appendChild(line);
      host.appendChild(el('p', 'muted small', path.length ? `this path carries ${pr > 0.995 ? 'nearly all' : `${(pr * 100).toPrecision(2)}%`} of the odds` : 'the first piece: every way to begin'));
      if (n.then != null) return;
      const b = el('div', 'cx-bars');
      n.top.forEach(([s, v], k) => {
        const btn = el('button', 'ghost small', show(s) || '∅'); btn.type = 'button'; btn.disabled = !n.kids || k >= n.kids.length;
        btn.addEventListener('click', () => { path.push(k); draw(); });
        b.appendChild(btn); const t = el('span', 'cx-bt'); const i = el('i'); i.style.width = `${(v * 100).toFixed(1)}%`; t.appendChild(i); b.appendChild(t); b.appendChild(el('span', 'cx-bv', f2(v)));
      });
      host.appendChild(b);
    };
    draw();
  }

  // The scenario format the site's exam-paper views read (window.E8_CONTEXT.scenarios[i]), for one model.
  function exportSite(r) {
    const model = (r.conditions && (r.conditions.model || r.conditions.model_dir)) || 'model';
    const name = String(model).split(/[\\/]/).pop();
    const sc = { id: 'psi-context', title: r.question, question: r.question, pieces: r.pieces, frame: r.frame, options: r.options, runs: {} };
    r.papers.forEach((p) => { sc.runs[p.key] = { [name]: { text: p.text, top: p.top.map(([t, v]) => ({ t, p: v })), odds: p.odds || {}, answer: p.answer } }; });
    const blob = new Blob([JSON.stringify({ recorded: new Date().toISOString().slice(0, 10), source: 'Parameter Space Inspector, Context lab', models: [{ name }], scenarios: [sc] }, null, 1)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'context-lab.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  const KEEP = PSI.keep('context', { inputs: IDS, render, status: 'cxStat' });
  async function run() {
    const q = $('cxQ').value.trim();
    if (!q) { PSI.status($('cxStat'), 'Write a question first.', 'warn'); $('cxQ').focus(); return; }
    const ps = pieces();
    if (ps.some((p) => p.role === 'turn' && !p.reply)) { PSI.status($('cxStat'), 'An earlier turn needs the assistant’s reply too.', 'warn'); return; }
    let am; try { am = arms(ps.length); } catch (e) { PSI.status($('cxStat'), e.message, 'warn'); $('cxArms').focus(); return; }
    GO.disabled = true; PSI.status($('cxStat'), `reading ${am ? Object.keys(am).length : 2 ** ps.length} papers…`, 'busy'); PSI.setRunning('context', true);
    try {
      render(await KEEP.call('context', { action: 'run', question: q, pieces: ps, options: options(), frame: $('cxFrame').value, n_new: 96, ...(am ? { arms: am } : {}) }));
      PSI.status($('cxStat'), ''); PSI.setDone('context');
    } catch (e) { PSI.err(e); PSI.status($('cxStat'), `failed: ${PSI.explain(e).text}`, 'error'); } finally { GO.disabled = false; PSI.setRunning('context', false); }
  }
  GO.addEventListener('click', run);
})();
