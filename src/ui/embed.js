// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// "Where your words live": each piece of the reader's text as the model stores it, a
// row of the embedding matrix, placed on a map of the model's whole vocabulary, with
// its nearest neighbours. Data from src/trace_server.py (/embed).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => String(s == null ? '' : s).replace(/�/g, '▫').replace(/\n/g, '⏎').replace(/ /g, '·');
  const M = { data: null, sel: 0 };
  const err = (e, o) => PSI.err(e, o);

  const status = (id, text, kind) => PSI.status($(id), text, kind);
  // What a reading came from: the inside engine's own copy of the model, as changed by a Tipping-point cut or not.
  const cond = (id, extra) => PSI.cond($(id).closest('.row'), [{ t: PSI.state.insideModel || 'the inside engine’s model', title: 'the unpacked weights folder the inside engine reads' },
    { t: 'inside engine', term: 'inside engine' }, ...PSI.changeParts('inside'), ...(extra || [])], 'read from');
  // Both parts read the weights through the inside-the-model engine, not the GGUF engine.
  PSI.requires($('embGo'), 'engine'); PSI.requires($('repGo'), 'engine');

  // Before anything is drawn the canvases say what will appear there and how to get it.
  function hint(cv, h, text) {
    const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || 600) * r; cv.height = h * r;
    const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = '#a9a69f'; g.font = `${12 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle'; const W = cv.width, H = cv.height; const lines = []; let ln = ''; String(text).split(' ').forEach((wd) => { const t2 = ln ? ln + ' ' + wd : wd; if (ln && g.measureText(t2).width > W * 0.9) { lines.push(ln); ln = wd; } else ln = t2; }); lines.push(ln); const lh = 18 * r; lines.forEach((t2, i) => g.fillText(t2, W / 2, H / 2 + (i - (lines.length - 1) / 2) * lh));
    PSI.canvasLabel(cv, text);
  }
  function empty() {
    if (M.data) return;
    hint($('embMap'), 420, 'Nothing placed yet: press “Show where they live”'); hint($('embRow'), 54, 'the selected piece’s row of numbers appears here');
  }
  empty();

  // Four depths from the layers a model has: about a sixth, a third, two thirds and the last.
  const depths = (Ls) => { const top = Ls[Ls.length - 1]; return [...new Set([top / 6, top / 3, (2 * top) / 3, top].map((x) => Ls.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a))))].map(String); };
  const heads = (cols) => { const hr = $('repTable').querySelector('thead tr'); hr.textContent = ''; hr.appendChild(el('th', null, 'piece')); cols.forEach((l) => hr.appendChild(el('th', null, `after layer ${l}`))); };
  // Until a reading comes back, the headers follow the loaded model's layer count (neutral wording before one is loaded).
  document.addEventListener('psi-model-meta', (e) => { const n = e.detail && e.detail.layers; if (n && !$('repTable').querySelector('tbody tr')) heads(depths([...Array(n).keys()].map((i) => i + 1))); });

  $('embGo').addEventListener('click', async () => {
    if (!$('tokText').value.trim()) return status('embStat', 'Type or paste some text in the box above first.', 'warn');
    const b = $('embGo'); b.disabled = true; status('embStat', 'reading the embedding matrix… (starts the inside-the-model engine if needed)', 'busy');
    try {
      const text = $('tokText').value;
      M.data = await psi.embed(text); M.sel = 0;
      const agree = await tokenizersAgree(text, M.data.tokens);
      status('embStat', `${M.data.tokens.length} pieces · each stored as ${M.data.dims.toLocaleString()} numbers · vocabulary of ${M.data.vocab.toLocaleString()} pieces · ${M.data.seconds}s${agree && !agree.same ? ` · ${agree.text}` : ''}`, agree && !agree.same ? 'warn' : 'ok');
      cond('embGo', [{ t: 'the map is a flattening: distances on it are approximate', est: true, title: 'Thousands of dimensions drawn in two; the nearest pieces below are measured in the full space (cosine)' },
        { t: 'special tokens not added', title: 'The inside engine cuts the text without adding the model’s own start token' },
        agree ? { t: agree.same ? 'pieces match the GGUF tokenizer above' : 'pieces differ from the GGUF tokenizer above', warn: !agree.same, title: agree.text } : { t: 'not compared with the GGUF tokenizer (no model loaded there)' }]);
      render();
    } catch (e) { err(e); status('embStat', `Reading the embedding matrix failed: ${PSI.explain(e).text}`, 'error'); } finally { b.disabled = false; }
  });
  addEventListener('resize', () => { if (M.data) { drawMap(); drawRow(); } else if (!$('tab-tokens').classList.contains('hidden')) empty(); });

  // Two tokenizers read this text: llama.cpp's from the GGUF (the pieces above) and the inside engine's (its own copy of
  // the model). If they cut it differently, the rows shown here are not the pieces above: say so, never hide it.
  async function tokenizersAgree(text, inside) {
    const C = window.PSIContext; if (!C || !C.model()) return null;
    let g; try { g = await psi.tokenize(text, false); } catch (_) { return null; }
    const a = g.map((t) => t.id), b = inside.map((t) => t.id);
    const same = a.length === b.length && a.every((x, i) => x === b[i]);
    if (same) return { same, text: `Both cut it into the same ${a.length} pieces with the same numbers.` };
    let k = 0; while (k < a.length && k < b.length && a[k] === b[k]) k++;
    return { same, text: `the inside engine’s lookup disagrees with the GGUF tokenizer: ${a.length} pieces there, ${b.length} here${k < Math.max(a.length, b.length) ? `, first difference at piece ${k + 1} (“${show((g[k] || {}).piece)}” #${a[k] ?? '–'} vs “${show((inside[k] || {}).piece)}” #${b[k] ?? '–'})` : ''}. The two engines may hold different models or tokenizers.` };
  }

  // A word it has never seen: what each piece becomes after reading the sentence.
  $('repGo').addEventListener('click', async () => {
    const b = $('repGo'); b.disabled = true; status('repStat', 'reading…', 'busy');
    try {
      const r = await psi.represent($('repText').value);
      const tb = $('repTable').querySelector('tbody'); tb.textContent = '';
      // The layers the engine read back (e.g. both layers of the 2-layer toy), else the ones in the reply.
      const Ls = r.layers || Object.keys((r.tokens[0] || {}).by_layer || {}).map(Number).sort((a, b) => a - b), top = Ls[Ls.length - 1];
      const cols = depths(Ls);
      const mid = cols.length > 2 ? cols[cols.length - 2] : null;
      heads(cols);
      // Which pieces belong to a word the model cannot know: any word of 6+ letters that the
      // tokenizer had to break into 3+ pieces (real words are usually one or two).
      const pieces = r.tokens.map((t) => t.piece), words = [];
      let cur = null;
      pieces.forEach((p, i) => { if (/^\s/.test(p) || i === 0) { cur = { idx: [i], text: p.trim() }; words.push(cur); } else if (cur) { cur.idx.push(i); cur.text += p.trim(); } });
      const invented = new Set(words.filter((w) => w.idx.length >= 3 && /^[a-z]{6,}$/i.test(w.text)).flatMap((w) => w.idx));
      const lastOf = new Set(words.filter((w) => w.idx.length >= 3 && /^[a-z]{6,}$/i.test(w.text)).map((w) => w.idx[w.idx.length - 1]));
      const keep = (i) => $('repAll').checked || invented.has(i) || /^\s?(stall|bread|market|shop|opens|sells)$/i.test(pieces[i]);
      let mention = 0; const readings = [];
      r.tokens.forEach((t, i) => {
        if (!keep(i)) return;
        const tr = el('tr'); if (invented.has(i)) tr.className = 'hi';
        tr.appendChild(el('td', null, show(t.piece) + (lastOf.has(i) ? ` ← mention ${++mention}` : '')));
        for (const l of cols) {
          const td = el('td'); const row = t.by_layer[l] || [];
          if (!row.length) { td.className = 'muted small'; td.textContent = 'not read'; td.title = 'The engine sent nothing for this layer: not measured, not empty'; tr.appendChild(td); continue; }
          if (row[0][1] < 0.04) td.style.opacity = '0.45';
          row.slice(0, 4).forEach(([p, v], k) => { const w = el('span', 'w'); if (k === 0) w.appendChild(el('b', null, show(p))); else w.textContent = show(p); w.title = `${(v * 100).toFixed(1)}%`; td.appendChild(w); });
          tr.appendChild(td);
        }
        tb.appendChild(tr);
        const last = (t.by_layer[String(top)] || [])[0];
        if (lastOf.has(i)) readings.push(`mention ${mention}: ${last ? `after the whole network it expects “${show(last[0])}” next` : 'the last layer was not read'}` + (mid && (t.by_layer[mid] || [])[0] ? `, at layer ${mid} “${show(t.by_layer[mid][0][0])}”` : ''));
      });
      const name = words.find((w) => w.idx.length >= 3 && /^[a-z]{6,}$/i.test(w.text));
      const letters = r.tokens.length >= 0.8 * [...$('repText').value].length;   // a letter-by-letter model (the toy Llama) has rows only for letters
      $('repStat').textContent = letters ? `This model reads letter by letter (${r.tokens.length} pieces for ${[...$('repText').value].length} characters): it has a row for no word at all, so every word is built from letters as it reads. ${readings.slice(0, 2).join('; ')}${readings.length ? '. ' : ''}Dim cells: the network has not decided there yet.` : name ? `“${name.text}” is ${name.idx.length} pieces the model has no row for as a word. ${readings.join('; ')}. Dim cells: the network has not decided there yet.` : `${r.tokens.length} pieces read.`;
      $('repStat').classList.remove('is-busy');
      cond('repGo', [{ t: 'each cell: the logit lens after that layer, the top 4 guesses', term: 'logit lens' }]);
    } catch (e) { err(e); status('repStat', `Reading the sentence failed: ${PSI.explain(e).text}`, 'error'); } finally { b.disabled = false; }
  });
  document.querySelector('#tabs button[data-tab=tokens]').addEventListener('click', () => setTimeout(() => { if (M.data) { drawMap(); drawRow(); } else empty(); }, 0));

  // The pieces are buttons for the keyboard too: one Tab stop, arrow keys move, Enter or Space selects.
  function render() {
    const d = M.data, box = $('embChips'), had = box.contains(document.activeElement); box.textContent = '';
    d.tokens.forEach((t, i) => {
      const c = el('span', 'chip' + (i === M.sel ? ' on' : ''), show(t.piece)); c.appendChild(el('small', null, t.id));
      c.setAttribute('role', 'button'); c.tabIndex = i === M.sel ? 0 : -1; c.setAttribute('aria-pressed', String(i === M.sel)); c.setAttribute('aria-label', `piece “${show(t.piece)}”, number ${t.id}`);
      c.addEventListener('click', () => { M.sel = i; render(); }); box.appendChild(c);
    });
    if (!box._keys) {
      box._keys = true;
      box.addEventListener('keydown', (e) => {
        const all = [...box.querySelectorAll('[role=button]')], i = all.indexOf(e.target); if (i < 0) return;
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.target.click(); return; }
        const n = /Right|Down/.test(e.key) ? all[i + 1] : /Left|Up/.test(e.key) ? all[i - 1] : e.key === 'Home' ? all[0] : e.key === 'End' ? all[all.length - 1] : null;
        if (n) { e.preventDefault(); all.forEach((x) => { x.tabIndex = x === n ? 0 : -1; }); n.focus(); }
      });
    }
    if (had) { const f = box.querySelector('[tabindex="0"]'); if (f) f.focus(); }
    drawMap(); drawRow(); neighbours();
  }

  function drawMap() {
    const d = M.data, cv = $('embMap'), r = Math.min(2, devicePixelRatio || 1);
    cv.style.height = '420px'; cv.width = cv.clientWidth * r; cv.height = 420 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, pts = d.map.pts;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    const xs = pts.map((p) => p[0]).concat(d.tokens.map((t) => t.xy[0])), ys = pts.map((p) => p[1]).concat(d.tokens.map((t) => t.xy[1]));
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 24 * r;
    const X = (x) => pad + (x - x0) / ((x1 - x0) || 1) * (W - 2 * pad), Y = (y) => H - pad - (y - y0) / ((y1 - y0) || 1) * (H - 2 * pad);
    g.fillStyle = 'rgba(242,241,238,0.22)';
    for (const p of pts) { g.fillRect(X(p[0]), Y(p[1]), r, r); }
    // A few backdrop labels for orientation: every 400th sampled piece that is plain text.
    g.font = `${9 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = 'rgba(169,166,159,0.45)'; g.textAlign = 'left';
    d.map.labels.forEach((lab, i) => { if (i % 400 === 0 && /^[ a-zA-Z]{3,}$/.test(lab)) g.fillText(lab.trim(), X(pts[i][0]) + 3 * r, Y(pts[i][1]) - 2 * r); });
    // The reader's pieces.
    d.tokens.forEach((t, i) => {
      const x = X(t.xy[0]), y = Y(t.xy[1]), on = i === M.sel;
      g.beginPath(); g.arc(x, y, (on ? 5 : 3.2) * r, 0, Math.PI * 2); g.fillStyle = on ? '#f2f1ee' : 'rgba(242,241,238,0.85)'; g.fill();
      if (on) { g.beginPath(); g.arc(x, y, 11 * r, 0, Math.PI * 2); g.strokeStyle = 'rgba(242,241,238,0.5)'; g.lineWidth = r; g.stroke(); }
      g.font = `${on ? 500 : 400} ${(on ? 12 : 10.5) * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = on ? '#f2f1ee' : 'rgba(242,241,238,0.8)';
      g.fillText(show(t.piece), x + 7 * r, y + 4 * r);
    });
    g.font = `${9 * r}px "IBM Plex Mono", ui-monospace, monospace`; g.fillStyle = '#6f6c66'; g.textAlign = 'left';
    g.fillText(`the model's dictionary, flattened from ${d.dims.toLocaleString()} dimensions to two (${pts.length >= d.vocab ? `all ${d.vocab.toLocaleString()} pieces` : `${pts.length.toLocaleString()} of ${d.vocab.toLocaleString()} pieces shown`}) · your pieces in white`, pad, H - 8 * r);
    const sel = d.tokens[M.sel];
    PSI.canvasLabel(cv, `Map of ${pts.length.toLocaleString()} of the model’s ${d.vocab.toLocaleString()} pieces flattened to two dimensions, with your ${d.tokens.length} pieces placed on it; selected: “${sel ? show(sel.piece) : ''}”.`);
  }

  function drawRow() {
    const t = M.data.tokens[M.sel], cv = $('embRow'), r = Math.min(2, devicePixelRatio || 1);
    cv.style.height = '54px'; cv.width = cv.clientWidth * r; cv.height = 54 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, s = t.sketch, mx = Math.max(...s.map(Math.abs)) || 1, w = W / s.length;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.fillStyle = 'rgba(42,41,39,1)'; g.fillRect(0, H / 2, W, r);
    s.forEach((v, i) => { const h = Math.abs(v) / mx * (H / 2 - 6 * r); g.fillStyle = v >= 0 ? 'rgba(242,241,238,0.9)' : 'rgba(169,166,159,0.9)'; g.fillRect(i * w + 1, v >= 0 ? H / 2 - h : H / 2, Math.max(1, w - 2), h); });
    $('embRowLabel').textContent = `“${show(t.piece)}” (piece #${t.id}): its row in the embedding matrix, ${M.data.dims.toLocaleString()} numbers${M.data.dims > 128 ? ` (shown as ${s.length} averaged blocks)` : ', every one shown'}, length ${t.norm}. This is all the model “knows” about the piece before context: a position in a ${M.data.dims.toLocaleString()}-dimensional space.`;
    PSI.canvasLabel(cv, `Bar chart: the ${s.length} ${M.data.dims > 128 ? 'averaged blocks' : 'numbers'} of “${show(t.piece)}”’s row in the embedding matrix, length ${t.norm}.`);
  }

  function neighbours() {
    const t = M.data.tokens[M.sel], box = $('embNear'); box.textContent = '';
    box.appendChild(el('span', 'k', 'its nearest pieces in that space (cosine):'));
    t.neighbors.forEach(([p, c]) => { const ch = el('span', 'chip', show(p)); ch.appendChild(el('small', null, c.toFixed(2))); box.appendChild(ch); });
    // A cosine means little without the cosine of a typical pair of pieces in the same model (often far from 0).
    const ref = M.data.reference || M.data.ref || {}, tc = ref.typical_cos ?? ref.typical_cosine ?? M.data.typical_cos, mn = ref.median_norm ?? M.data.median_norm;
    const r = el('p', 'small' + (tc == null ? ' muted' : ''));
    r.textContent = tc == null && mn == null ? `Reference: no reference value. The engine sends no typical-pair cosine or median length, so a ${t.neighbors[0] ? t.neighbors[0][1].toFixed(2) : 'cosine'} cannot be told from a typical pair of pieces here; read the numbers only against each other.`
      : `Reference: a typical pair of pieces has cosine ${tc == null ? 'not reported' : (+tc).toFixed(2)}; the median row length is ${mn == null ? 'not reported' : (+mn).toFixed(2)} (this piece: ${t.norm}).`;
    box.appendChild(r);
    const note = el('p', 'muted small', 'Near means the model treats them alike before reading any context: often the same word with different casing or spacing, its plural, or a translation. Everything else the model knows about your word is not stored here; it is computed as the word passes through the layers (the Neural net tab).');
    box.appendChild(note);
  }
})();
