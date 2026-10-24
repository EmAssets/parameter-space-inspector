// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Context lab tab: the context is the variable. A question and up to four pieces of context, each with a role; every
// combination runs through the model's own chat template (src/context_lab.py on the inside engine). Shown: what each
// piece did (how far it moved the odds, its effect on each named answer, where it changed the leading answer), every
// paper with its odds and answer, pieces that matter only together, two papers continuing the same answer (where the
// paper stops mattering), and the paths from the answer start. A run can be saved in the format the site's exam-paper
// views replay (e8-context).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => JSON.stringify(s).slice(1, -1);
  const f2 = (x) => (x >= 0.995 ? '1.00' : x.toFixed(2));
  const ROLE = { system: 'hidden instruction', premise: 'premise', claim: 'claim', evidence: 'evidence', instruction: 'how to answer' };
  const EXAMPLES = [
    ['paper vs memory', 'What is the capital of Australia?', 'The capital of Australia is', 'Canberra\nSydney',
      [['premise', 'In this story, the capital of Australia is Sydney.'], ['system', 'Answer only from the information given in the user\'s message.'], ['claim', 'I\'m quite sure the answer is Sydney.']]],
    ['instructions', 'Who won the 2031 Cricket World Cup?', '', 'says it can\'t know = I don\'t know, As of, I cannot, The 2031 Cricket World Cup has not\nnames a winner = India, Australia, England, The 2031 Cricket World Cup was won by',
      [['instruction', 'Answer in one sentence.'], ['system', 'If you are not sure of the answer, say that you don\'t know.']]],
    ['evidence vs claim', 'Are all swans white?', 'Short answer:', 'yes = Yes\nno = No\ncan\'t tell = I don\'t know, It depends, Not necessarily',
      [['claim', 'Everyone knows that all swans are white.'], ['evidence', 'Explorers in Western Australia recorded hundreds of black swans.'], ['premise', 'Reports from distant lands are often unreliable.']]],
  ];
  const IDS = ['cxQ', 'cxFrame', 'cxOpts', 'cxR0', 'cxT0', 'cxR1', 'cxT1', 'cxR2', 'cxT2', 'cxR3', 'cxT3'];
  const GO = $('cxGo'); if (!GO) return;
  PSI.requires(GO, 'engine');
  let LAST = null;

  EXAMPLES.forEach(([label, q, frame, opts, pieces]) => {
    const b = el('button', 'ghost small', label); b.type = 'button';
    b.addEventListener('click', () => {
      $('cxQ').value = q; $('cxFrame').value = frame; $('cxOpts').value = opts;
      for (let i = 0; i < 4; i++) { $(`cxR${i}`).value = pieces[i] ? pieces[i][0] : 'premise'; $(`cxT${i}`).value = pieces[i] ? pieces[i][1] : ''; }
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
  const pieces = () => [0, 1, 2, 3].map((i) => ({ role: $(`cxR${i}`).value, text: $(`cxT${i}`).value.trim() })).filter((p) => p.text);
  const label = (r, key) => r.pieces.map((p, i) => (key[i] === '1' ? '●' : '○')).join('');

  function bars(host, odds) {
    const b = el('div', 'cx-bars');
    Object.entries(odds).sort((x, y) => y[1] - x[1]).forEach(([k, v]) => {
      b.appendChild(el('span', 'cx-bn', k)); const t = el('span', 'cx-bt'); const i = el('i'); i.style.width = `${(v * 100).toFixed(1)}%`; t.appendChild(i); b.appendChild(t); b.appendChild(el('span', 'cx-bv', f2(v)));
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
      const name = el('div', 'cx-piece'); name.append(el('span', `cx-role r-${e.role}`, ROLE[e.role]), el('span', '', e.text)); row.appendChild(name);
      const mv = el('div', 'cx-moved'); const tr = el('span', 'cx-bt'); const i = el('i'); i.style.width = `${(e.moved * 100).toFixed(1)}%`; tr.appendChild(i);
      mv.append(el('span', 'cx-sub', 'moved the odds'), tr, el('span', 'cx-bv', f2(e.moved))); row.appendChild(mv);
      if (opts.length) {
        const fx = el('div', 'cx-fx');
        opts.forEach((o) => { const v = e.effect[o]; fx.appendChild(el('span', v > 0.05 ? 'up' : v < -0.05 ? 'down' : 'flat', `${o} ${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`)); });
        fx.appendChild(el('span', 'cx-sub', e.flips.length ? `changed the leading answer in ${e.flips.length} of ${2 ** (r.pieces.length - 1)} papers` : 'never changed the leading answer'));
        row.appendChild(fx);
      }
      t.appendChild(row);
    });
    out.appendChild(t);
    out.appendChild(el('p', 'muted small', 'Moved: how much of the next-piece probability changes when only this piece is added (1 − overlap, averaged over every paper it can be added to). The effects are average changes in each named answer’s share.'));
    const strong = (r.interactions || []).filter((x) => Object.values(x.interaction).some((v) => Math.abs(v) >= 0.1));
    if (strong.length) {
      out.appendChild(el('p', 'k mt-12', 'pieces that act together'));
      strong.forEach((x) => {
        const [i, j] = x.pieces, o = Object.entries(x.interaction).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0];
        out.appendChild(el('p', 'small', `${ROLE[r.pieces[i].role]} ${i + 1} with ${ROLE[r.pieces[j].role]} ${j + 1}: together they move “${o[0]}” ${o[1] >= 0 ? 'up' : 'down'} by ${Math.abs(o[1]).toFixed(2)} more than the two do apart.`));
      });
    }

    out.appendChild(el('p', 'k mt-12', `every paper (${r.papers.length})`));
    const grid = el('div', 'cx-papers');
    r.papers.forEach((p) => {
      const c = el('div', 'cx-paper');
      c.appendChild(el('div', 'cx-key', `${label(r, p.key)}  ${p.on.length ? p.on.map((x) => `“${x}”`).join(' + ') : 'the question alone'}`));
      if (p.odds) bars(c, p.odds);
      else c.appendChild(el('p', 'small', p.top.slice(0, 4).map(([s, v]) => `“${show(s)}” ${f2(v)}`).join(' · ')));
      c.appendChild(el('p', 'cx-ans', p.answer.length > 300 ? `${p.answer.slice(0, 300)}…` : p.answer));
      const d = el('details', 'small'); d.appendChild(el('summary', '', 'what it read')); d.appendChild(el('pre', 'cx-read', p.text + (r.frame ? `⟨${r.frame}⟩` : ''))); c.appendChild(d);
      grid.appendChild(c);
    });
    out.appendChild(grid);
    if (r.frame) out.appendChild(el('p', 'muted small', `Odds read after “${r.frame}”, the same start for every paper; shares among the answers named, each by its best phrasing. The answer below each is the model’s own, unframed.`));

    // Two papers, one answer; and the paths from one paper.
    out.appendChild(el('p', 'k mt-12', 'where the paper stops mattering'));
    const row = el('div', 'row'); const sa = el('select'), sb = el('select'), sp = el('select');
    r.papers.forEach((p, i) => { [sa, sb, sp].forEach((s) => s.appendChild(new Option(`${label(r, p.key)} ${p.on.length ? p.on.join(' + ').slice(0, 60) : 'the question alone'}`, p.key))); });
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
    const row2 = el('div', 'row'); const tb = el('button', 'ghost', 'Walk the paths'); tb.type = 'button'; row2.append(sp, tb); out.appendChild(row2);
    const trOut = el('div', 'cx-tree'); out.appendChild(trOut);
    tb.addEventListener('click', async () => {
      trOut.textContent = 'reading…';
      try { walk(trOut, await psi.context({ action: 'tree', question: r.question, pieces: r.pieces, key: sp.value })); } catch (e) { PSI.err(e); trOut.textContent = `failed: ${PSI.explain(e).text}`; }
    });

    const ex = el('button', 'ghost small', 'Save for the site (exam-paper format)'); ex.type = 'button'; ex.addEventListener('click', () => exportSite(r));
    out.appendChild(el('div', 'row mt-12')).appendChild(ex);
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
    GO.disabled = true; PSI.status($('cxStat'), `reading ${2 ** ps.length} papers…`, 'busy'); PSI.setRunning('context', true);
    try {
      render(await KEEP.call('context', { action: 'run', question: q, pieces: ps, options: options(), frame: $('cxFrame').value, n_new: 96 }));
      PSI.status($('cxStat'), ''); PSI.setDone('context');
    } catch (e) { PSI.err(e); PSI.status($('cxStat'), `failed: ${PSI.explain(e).text}`, 'error'); } finally { GO.disabled = false; PSI.setRunning('context', false); }
  }
  GO.addEventListener('click', run);
})();
