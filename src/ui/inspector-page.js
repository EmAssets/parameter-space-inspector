// SPDX-License-Identifier: Apache-2.0
// The inspector page's own script, kept in a file so the page can forbid inline script (its Content-Security-Policy).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const SP = '▁';

  // ── GGUF reader ───────────────────────────────────────────────────────────
  // The same reader as the app's File view (gguf-header.js, next to this page). Guardrails: the header is read from a
  // slice of the file (never the whole file), and every count, length and offset is checked against the real file
  // size before anything is allocated. Only small F32 files are read in full.
  if (!window.PSIHeader) { document.body.prepend(Object.assign(document.createElement('p'), { textContent: 'gguf-header.js, the file reader this page needs, did not load; nothing can be read.' })); return; }
  const { LIMITS, typeName, parseHeader, NeedMore } = window.PSIHeader;
  const damaged = (what) => new Error(`${what}: the file looks damaged or is not a model this inspector can read.`);

  // Full mode only: point each F32 tensor at its numbers in the whole file.
  function attachF32(g, buf) {
    const T = {};
    for (const t of g.tensors) {
      const at = g.start + t.offset;
      if (t.type !== 0) throw new Error(`Tensor ${t.name} is not F32 (${typeName(t.type)}).`);
      if (at % 4 || at + t.count * 4 > buf.byteLength) throw damaged(`Tensor ${t.name} is out of place`);
      t.data = new Float32Array(buf, at, t.count);
      T[t.name] = t;
    }
    g.T = T;
    return g;
  }

  // Kept for tests and small in-memory files: header + F32 tensors from one buffer.
  function parseGGUF(buf) { return attachF32(parseHeader(buf, buf.byteLength), buf); }

  // Can the live views (net, generation, attention, letter map) run on this file?
  function fullModeReason(g) {
    const kv = g.kv, a = kv['general.architecture'];
    const odd = g.tensors.filter((t) => t.type !== 0);
    if (odd.length) return `Its weights are stored as ${[...new Set(odd.map((t) => typeName(t.type)))].join(', ')}, and the live views read F32 only.`;
    if (g.bytes > LIMITS.fullBytes) return `It is ${(g.bytes / 2 ** 20).toFixed(0)} MB; the live views read files up to ${LIMITS.fullBytes / 2 ** 20} MB.`;
    if (g.params > LIMITS.fullParams) return `It has ${g.params.toLocaleString()} parameters; the live views handle up to ${LIMITS.fullParams.toLocaleString()}.`;
    if (!a || !kv[`${a}.embedding_length`] || !kv[`${a}.block_count`] || !kv[`${a}.attention.head_count`] || !Array.isArray(kv['tokenizer.ggml.tokens'])) return 'It does not describe a Llama-style model the live views can run.';
    if (!g.tensors.some((t) => t.name === 'token_embd.weight')) return 'It has no token embedding tensor.';
    return '';
  }

  // ── the model, with everything captured ───────────────────────────────────
  function makeModel(g) {
    const kv = g.kv, a = kv['general.architecture'];
    const c = { d: kv[`${a}.embedding_length`], L: kv[`${a}.block_count`], H: kv[`${a}.attention.head_count`],
      hid: kv[`${a}.feed_forward_length`], ctx: kv[`${a}.context_length`], eps: kv[`${a}.attention.layer_norm_rms_epsilon`] || 1e-5,
      base: kv[`${a}.rope.freq_base`] || 10000 };
    c.hd = c.d / c.H;
    const W = (n) => (g.T[n] || {}).data;
    const out = W('output.weight') || W('token_embd.weight');
    const tokens = kv['tokenizer.ggml.tokens'], V = tokens.length;
    const ix = Object.fromEntries(tokens.map((t, i) => [t, i]));
    const byteId = (b) => ix[`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`];
    function encode(text) {
      const ids = []; const s = (' ' + text).replace(/ /g, SP);
      for (const ch of s) { if (ch in ix) ids.push(ix[ch]); else for (const b of new TextEncoder().encode(ch)) ids.push(byteId(b) ?? 0); }
      return ids;
    }
    function show(id) { const t = tokens[id]; if (/^<0x[0-9A-F]{2}>$/.test(t)) return t; return t === SP ? '␣' : t; }
    function decode(ids) {
      let s = '', by = [];
      const flush = () => { if (by.length) { s += new TextDecoder().decode(new Uint8Array(by)); by = []; } };
      for (const i of ids) { const t = tokens[i]; const m = /^<0x([0-9A-F]{2})>$/.exec(t); if (m) { by.push(parseInt(m[1], 16)); continue; } flush(); if (/^<.*>$/.test(t)) continue; s += t.split(SP).join(' '); }
      flush(); return s;
    }
    const lin = (x, w, o, n) => { const y = new Float32Array(o); for (let r = 0; r < o; r++) { let s = 0, b = r * n; for (let i = 0; i < n; i++) s += w[b + i] * x[i]; y[r] = s; } return y; };
    const rms = (x, w) => { let m = 0; for (const v of x) m += v * v; const k = 1 / Math.sqrt(m / x.length + c.eps); return x.map((v, i) => v * k * w[i]); };
    const inv = Array.from({ length: c.hd / 2 }, (_, i) => Math.pow(c.base, -(2 * i) / c.hd));
    const rope = (v, pos) => { for (let h = 0; h < c.H; h++) for (let i = 0; i < c.hd / 2; i++) { const an = pos * inv[i], co = Math.cos(an), si = Math.sin(an), j = h * c.hd + 2 * i, x0 = v[j], x1 = v[j + 1]; v[j] = x0 * co - x1 * si; v[j + 1] = x0 * si + x1 * co; } };
    // Run a token sequence; return per position: embedding, per layer (attention per head,
    // feed-forward neurons, residual stream after the layer) and the next-token probabilities.
    function run(ids) {
      const K = Array.from({ length: c.L }, () => []), Vc = Array.from({ length: c.L }, () => []);
      const trace = [];
      ids.forEach((tok, pos) => {
        const tr = { tok, emb: null, layers: [], probs: null };
        let x = W('token_embd.weight').slice(tok * c.d, tok * c.d + c.d);
        tr.emb = x.slice();
        for (let l = 0; l < c.L; l++) {
          const p = `blk.${l}.`;
          const h = rms(x, W(p + 'attn_norm.weight'));
          const q = lin(h, W(p + 'attn_q.weight'), c.d, c.d), k = lin(h, W(p + 'attn_k.weight'), c.d, c.d), v = lin(h, W(p + 'attn_v.weight'), c.d, c.d);
          rope(q, pos); rope(k, pos); K[l].push(k); Vc[l].push(v);
          const att = new Float32Array(c.d), heads = [];
          for (let hh = 0; hh < c.H; hh++) {
            const sc = K[l].map((kk) => { let s = 0; for (let i = 0; i < c.hd; i++) s += q[hh * c.hd + i] * kk[hh * c.hd + i]; return s / Math.sqrt(c.hd); });
            const mx = Math.max(...sc); let z = 0; const e = sc.map((s) => { const t = Math.exp(s - mx); z += t; return t; });
            const wts = e.map((t) => t / z); heads.push(wts);
            wts.forEach((w, t) => { for (let i = 0; i < c.hd; i++) att[hh * c.hd + i] += w * Vc[l][t][hh * c.hd + i]; });
          }
          const o = lin(att, W(p + 'attn_output.weight'), c.d, c.d);
          const mid = x.map((vv, i) => vv + o[i]);
          const h2 = rms(mid, W(p + 'ffn_norm.weight'));
          const gt = lin(h2, W(p + 'ffn_gate.weight'), c.hid, c.d), up = lin(h2, W(p + 'ffn_up.weight'), c.hid, c.d);
          const ff = gt.map((gv, i) => (gv / (1 + Math.exp(-gv))) * up[i]);
          const f = lin(ff, W(p + 'ffn_down.weight'), c.d, c.hid);
          x = mid.map((vv, i) => vv + f[i]);
          tr.layers.push({ heads, h2, ff, out: x.slice() });
        }
        const xn = rms(x, W('output_norm.weight'));
        const lg = lin(xn, out, V, c.d);
        const mx = Math.max(...lg); let z = 0; const pr = Array.from(lg, (v) => { const e = Math.exp(v - mx); z += e; return e; });
        tr.xn = xn; tr.probs = pr.map((v) => v / z);
        trace.push(tr);
      });
      return trace;
    }
    function generate(prompt, n, temp) {
      const ids = [kv['tokenizer.ggml.bos_token_id'] ?? 1, ...encode(prompt)];
      const out2 = [];
      for (let i = 0; i < n && ids.length < c.ctx; i++) {
        const tr = run(ids); const pr = tr[tr.length - 1].probs;
        let next;
        if (temp <= 0) next = pr.indexOf(Math.max(...pr));
        else { const q = pr.map((p) => Math.pow(p, 1 / temp)); let z = q.reduce((a2, b) => a2 + b, 0), r = Math.random() * z; next = 0; for (; next < q.length - 1; next++) { r -= q[next]; if (r <= 0) break; } }
        if (next === (kv['tokenizer.ggml.eos_token_id'] ?? 2)) break;
        ids.push(next); out2.push(next);
      }
      return decode(out2);
    }
    return { c, W, tokens, V, encode, decode, show, run, generate, bos: kv['tokenizer.ggml.bos_token_id'] ?? 1 };
  }

  // ── rendering ─────────────────────────────────────────────────────────────
  let G = null, Mdl = null;
  const fmt = (n) => (n == null || Number.isNaN(Number(n)) ? '?' : Number(n).toLocaleString());  // files from elsewhere may lack some notes
  // The CSS height comes from the original height attribute, remembered once:
  // setting cv.height rewrites that attribute, so reading it back would double it.
  function sizeCanvas(cv) {
    if (!cv.dataset.h) cv.dataset.h = cv.getAttribute('height');
    const r = Math.min(2, devicePixelRatio || 1), h = +cv.dataset.h;
    cv.style.height = h + 'px';
    cv.width = cv.clientWidth * r; cv.height = h * r;
    return r;
  }

  function showFacts() {
    const kv = G.kv, a = kv['general.architecture'], n = (k) => kv[`${a}.${k}`];
    // A value the header does not carry is said to be absent, never shown as "?" or 0.
    const ABS = 'absent from header', or = (v, f = (x) => x) => (v == null ? ABS : f(v));
    const types = PSIHeader.byType(G);
    const vocab = Array.isArray(kv['tokenizer.ggml.tokens']) ? kv['tokenizer.ggml.tokens'].length : null;
    const facts = [['architecture', or(a)], ['parameters', fmt(G.params)], ['layers', or(n('block_count'))], ['width', or(n('embedding_length'))],
      ['attention heads', or(n('attention.head_count'))], ['feed-forward neurons', or(n('feed_forward_length'), (v) => `${v} per layer`)],
      ['context', or(n('context_length'), (v) => `${fmt(v)} tokens`)], ['vocabulary', or(vocab, (v) => `${fmt(v)} tokens`)],
      ['file', `${fmt(G.bytes)} bytes, GGUF v${G.version}`], ['quantisation recipe', or(PSIHeader.fileTypeName(kv['general.file_type']))],
      ['tensors', `${G.tensors.length}: ${types.map((t) => `${t.tensors} ${t.name} (${fmt(t.params)} parameters, ${t.bytes == null ? 'size unknown for this type' : `${fmt(t.bytes)} bytes, ${t.bitsPerParam.toFixed(2)} bits each`})`).join('; ')}`]];
    if (kv['general.license']) facts.push(['licence', kv['general.license.name'] || kv['general.license']]);
    if (kv['e8.attribution']) facts.push(['credit', kv['e8.attribution']]);
    $('title').textContent = PSIHeader.displayName(kv, G.name || '');
    $('facts').textContent = '';
    facts.forEach(([k, v]) => { const d = el('div'); d.appendChild(el('b', null, k)); d.appendChild(document.createTextNode(String(v))); $('facts').appendChild(d); });
  }

  // Every header note, long values shortened so a big tokenizer cannot flood the page.
  function showMeta() {
    const tb = $('meta').querySelector('tbody'); tb.textContent = '';
    const short = (v) => {
      if (Array.isArray(v)) return `[${v.length.toLocaleString()} items] ${v.slice(0, 12).map((x) => JSON.stringify(x)).join(', ')}${v.length > 12 ? ', …' : ''}`;
      const t = String(v); return t.length > 400 ? `${t.slice(0, 400)}… (${t.length.toLocaleString()} characters)` : t;
    };
    Object.keys(G.kv).forEach((k) => { const tr = el('tr'); tr.appendChild(el('td', null, k)); tr.appendChild(el('td', null, short(G.kv[k]))); tb.appendChild(tr); });
  }

  function showHeaderOnly(why) {
    $('main').classList.remove('hidden');
    document.querySelectorAll('section.live').forEach((x) => x.classList.add('hidden'));
    $('mode').textContent = `Header-only view. ${why} So this page read only the file's header, never its weights, and shows what the file says about itself. Run the model itself with Ollama or llama.cpp.`;
    $('mode').classList.remove('hidden');
    showFacts(); showLearned(); showMeta(); tensorTable(false);
  }

  function showAll() {
    const m = Mdl, c = m.c;
    $('main').classList.remove('hidden');
    document.querySelectorAll('section.live').forEach((x) => x.classList.remove('hidden'));
    $('mode').classList.add('hidden');
    showFacts(); showLearned(); showMeta();
    afterLearned(m, c);
  }

  function showLearned() {
    const kv = G.kv;
    const L = $('learned'); L.textContent = '';
    if (kv['e8.street.country'] != null) {
      L.appendChild(el('h2', null, `A street in ${kv['e8.street.place']} (${kv['e8.street.week'] || 'one week'})`));
      L.appendChild(el('p', null, `Trained in a web browser for ${fmt(kv['e8.training.steps'])} steps on ${fmt(kv['e8.training.text_chars'])} characters written about that street${kv['e8.training.reader_steps'] ? `, then for ${fmt(kv['e8.training.reader_steps'])} more steps on your own inputs` : ''}. ${kv['e8.trained_at'] ? `Saved ${new Date(kv['e8.trained_at']).toLocaleString()}.` : ''}`));
      const mine = kv['e8.training.reader_inputs'] || [];
      if (mine.length) {
        L.appendChild(el('p', 'k', 'your inputs it was fine-tuned on'));
        const ul = el('ul', 'inputs'); mine.forEach((t) => ul.appendChild(el('li', null, t))); L.appendChild(ul);
      } else L.appendChild(el('p', 'muted', 'It was not fine-tuned on any of your inputs.'));
      const d = el('details'); d.appendChild(el('summary', null, `the whole street text it read (${fmt(kv['e8.training.text_chars'])} characters)`));
      d.appendChild(el('pre', 'text', kv['e8.training.street_text'] || '')); L.appendChild(d);
      if (/^https?:\/\//i.test(kv['general.source.url'] || '')) { /* a link from the file: http(s) only, never javascript: */ const p = el('p', 'muted'); p.append('From '); const a = el('a', null, kv['general.source.url']); a.href = kv['general.source.url']; a.style.color = 'inherit'; p.appendChild(a); L.appendChild(p); }
    } else L.appendChild(el('p', 'muted', 'This file has no training notes, so where it learned from is unknown. Most real models do not say either.'));
  }

  function afterLearned(m, c) {
    // Alphabet
    const chars = m.tokens.map((t, i) => [t, i]).filter(([t]) => !/^<.*>$/.test(t));
    $('alphaTitle').textContent = `${chars.length} letters and signs, plus ${m.V - chars.length} special and byte tokens`;
    $('alpha').textContent = '';
    chars.forEach(([t]) => $('alpha').appendChild(el('span', 'tok', t === SP ? '␣' : t)));
    drawMap(chars);

    // Layer/head pickers
    $('attLayer').textContent = ''; $('attHead').textContent = '';
    for (let l = 0; l < c.L; l++) { const o = el('option', null, `layer ${l + 1}`); o.value = l; $('attLayer').appendChild(o); }
    for (let h = 0; h < c.H; h++) { const o = el('option', null, `head ${h + 1}`); o.value = h; $('attHead').appendChild(o); }

    tensorTable(true); netRun(); gen(); attention();
  }

  function drawMap(chars) {
    chars = chars.slice(0, 1500);
    const cv = $('map'), r = sizeCanvas(cv), g = cv.getContext('2d'), W = cv.width, H = cv.height;
    const E = Mdl.W('token_embd.weight'), d = Mdl.c.d;
    const rows = chars.map(([, i]) => Array.from(E.subarray(i * d, i * d + d)));
    const mean = new Array(d).fill(0); rows.forEach((r2) => r2.forEach((v, j) => { mean[j] += v / rows.length; }));
    const X = rows.map((r2) => r2.map((v, j) => v - mean[j]));
    const pc = (deflate) => { let v = Array.from({ length: d }, (_, j) => Math.sin(j * 7.3 + deflate.length)); for (let it = 0; it < 60; it++) { const s = X.map((x) => x.reduce((a, b, j) => a + b * v[j], 0)); let nv = new Array(d).fill(0); X.forEach((x, i) => x.forEach((b, j) => { nv[j] += b * s[i]; })); deflate.forEach((u) => { const dot = nv.reduce((a, b, j) => a + b * u[j], 0); nv = nv.map((b, j) => b - dot * u[j]); }); const n = Math.hypot(...nv) || 1; v = nv.map((b) => b / n); } return v; };
    const p1 = pc([]), p2 = pc([p1]);
    const P = X.map((x) => [x.reduce((a, b, j) => a + b * p1[j], 0), x.reduce((a, b, j) => a + b * p2[j], 0)]);
    const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.font = `${14 * r}px ui-monospace, monospace`; g.textAlign = 'center';
    P.forEach(([x, y], i) => {
      const px = 30 * r + (x - x0) / (x1 - x0 || 1) * (W - 60 * r), py = 30 * r + (y - y0) / (y1 - y0 || 1) * (H - 60 * r);
      const t = chars[i][0];
      g.fillStyle = /[aeiou]/.test(t) ? '#fff' : '#a9a69f';
      g.fillText(t === SP ? '␣' : t, px, py);
    });
    g.fillStyle = '#8a8780'; g.font = `${11 * r}px ui-monospace, monospace`; g.textAlign = 'left';
    g.fillText('vowels in white — the model was never told which they are', 12 * r, H - 10 * r);
  }

  // The neural net: columns of neurons for one position, lines for the live signal.
  let netTrace = null, netIds = [];
  function netRun() {
    netIds = [Mdl.bos, ...Mdl.encode($('netPrompt').value.toLowerCase())].slice(0, Mdl.c.ctx);
    netTrace = Mdl.run(netIds);
    $('netPos').max = netIds.length - 1; $('netPos').value = netIds.length - 1;
    drawNet();
  }
  function drawNet() {
    const cv = $('net'), r = sizeCanvas(cv), g = cv.getContext('2d'), W = cv.width, H = cv.height, m = Mdl, c = m.c;
    const pos = +$('netPos').value, tr = netTrace[pos];
    $('netPosLabel').textContent = `“${m.show(netIds[pos])}”  ${pos + 1}/${netIds.length}`;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    // Columns: embedding → (layer l: feed-forward neurons → stream) ... → top next tokens
    const cols = [{ label: 'letter in', short: 'in', vals: tr.emb, kind: 'stream' }];
    tr.layers.forEach((ly, l) => { cols.push({ label: `layer ${l + 1}: ${c.hid} neurons`, short: `L${l + 1}`, vals: ly.ff, kind: 'ff', l }); cols.push({ label: `stream after layer ${l + 1}`, short: `S${l + 1}`, vals: ly.out, kind: 'stream', l }); });
    const top = tr.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 10);
    const n = cols.length + 1, colX = (i) => 70 * r + i * (W - 160 * r) / (n - 1);
    const pts = cols.map((col, i) => {
      const N = col.vals.length, mx = Math.max(...Array.from(col.vals, Math.abs)) || 1;
      const sub = Math.ceil(N / 88), rowsN = Math.ceil(N / sub);
      return Array.from({ length: rowsN }, (_, k) => {
        const j = k * sub, y = 40 * r + (k + 0.5) * (H - 80 * r) / rowsN;
        return { x: colX(i), y, v: col.vals[j] / mx, j };
      });
    });
    // Connections that carry the most signal between consecutive columns.
    const link = (from, to, weightOf) => {
      const cand = [];
      from.forEach((a) => to.forEach((b) => { const s = weightOf(b.j, a.j) * a.v; if (s) cand.push([Math.abs(s), a, b]); }));
      cand.sort((p, q) => q[0] - p[0]);
      const best = cand.slice(0, 70), mx = best.length ? best[0][0] : 1;
      best.forEach(([s, a, b]) => { g.strokeStyle = `rgba(255,255,255,${0.08 + 0.55 * s / mx})`; g.lineWidth = r * (0.5 + 1.2 * s / mx); g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); });
    };
    for (let i = 0; i < cols.length - 1; i++) {
      const to = cols[i + 1];
      if (to.kind === 'ff') { const wu = m.W(`blk.${to.l}.ffn_up.weight`); link(pts[i], pts[i + 1], (o, j) => wu[o * c.d + j]); }
      else { const wd = m.W(`blk.${to.l}.ffn_down.weight`); link(pts[i], pts[i + 1], (o, j) => wd[o * c.hid + j]); }
    }
    // Last stream → the top next-token guesses
    const outW = m.W('output.weight') || m.W('token_embd.weight'), last = pts[pts.length - 1];
    const outPts = top.map(([p, id], k) => ({ x: colX(n - 1), y: 60 * r + k * (H - 120 * r) / 9, p, id }));
    const cand = []; last.forEach((a) => outPts.forEach((b) => { const s = outW[b.id * c.d + a.j] * a.v; cand.push([Math.abs(s), a, b]); }));
    cand.sort((p, q) => q[0] - p[0]); const mx2 = cand[0] ? cand[0][0] : 1;
    cand.slice(0, 50).forEach(([s, a, b]) => { g.strokeStyle = `rgba(255,255,255,${0.06 + 0.5 * s / mx2})`; g.lineWidth = r; g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); });
    // Neurons
    pts.forEach((col) => col.forEach((p) => {
      const a = Math.min(1, Math.abs(p.v)), rad = r * (1.6 + 2.6 * a);
      g.beginPath(); g.arc(p.x, p.y, rad, 0, Math.PI * 2);
      if (p.v >= 0) { g.fillStyle = `rgba(255,255,255,${0.15 + 0.85 * a})`; g.fill(); } else { g.strokeStyle = `rgba(255,255,255,${0.2 + 0.8 * a})`; g.lineWidth = r; g.stroke(); }
    }));
    g.font = `${11 * r}px ui-monospace, monospace`; g.textAlign = 'center'; g.fillStyle = '#a9a69f';
    const tight = (W - 160 * r) / (n - 1) < 130 * r;   // narrow screens: short labels so neighbours do not overlap
    cols.forEach((col, i) => g.fillText(tight ? col.short : col.label, colX(i), 22 * r));
    g.fillText(tight ? 'next' : 'next letter', colX(n - 1), 22 * r);
    if (tight) { g.fillStyle = '#8a8780'; g.textAlign = 'left'; g.fillText('L = a layer’s neurons · S = the stream after it', 12 * r, H - 26 * r); }
    g.textAlign = 'left';
    outPts.forEach((o) => { g.fillStyle = `rgba(255,255,255,${0.35 + 0.65 * o.p / outPts[0].p})`; g.font = `${13 * r}px ui-monospace, monospace`; g.fillText(`${m.show(o.id)}  ${(o.p * 100).toFixed(1)}%`, o.x - 8 * r, o.y + 4 * r); });
    g.fillStyle = '#8a8780'; g.font = `${10 * r}px ui-monospace, monospace`; g.textAlign = 'left';
    g.fillText('large columns are sampled: one dot per few neurons, so they fit', 12 * r, H - 10 * r);
  }

  function gen() {
    const p = $('prompt').value.toLowerCase(), t = +$('temp').value;
    $('tempV').textContent = t.toFixed(2);
    const cont = Mdl.generate(p, 100, t);
    $('gen').textContent = ''; $('gen').appendChild(el('b', null, p)); $('gen').appendChild(document.createTextNode(cont));
    const tr = Mdl.run([Mdl.bos, ...Mdl.encode(p)]); const pr = tr[tr.length - 1].probs;
    const top = pr.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 10);
    $('probs').textContent = '';
    top.forEach(([v, i]) => { const d = el('div'); d.appendChild(el('span', null, Mdl.show(i))); const bar = el('span'); const b = el('i'); b.style.width = `${(v / top[0][0]) * 100}%`; bar.appendChild(b); d.appendChild(bar); d.appendChild(el('span', null, `${(v * 100).toFixed(1)}%`)); $('probs').appendChild(d); });
  }

  function attention() {
    const ids = [Mdl.bos, ...Mdl.encode($('attPrompt').value.toLowerCase())].slice(0, 41);
    const tr = Mdl.run(ids), l = +$('attLayer').value || 0, h = +$('attHead').value || 0;
    const cv = $('att'), r = sizeCanvas(cv), g = cv.getContext('2d'), W = cv.width, H = cv.height, n = ids.length;
    const pad = 28 * r, cell = Math.min((W - pad * 2) / n, (H - pad * 2) / n);
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.font = `${11 * r}px ui-monospace, monospace`; g.textAlign = 'center';
    for (let i = 0; i < n; i++) {
      const w = tr[i].layers[l].heads[h];
      for (let j = 0; j <= i; j++) { g.fillStyle = `rgba(255,255,255,${w[j]})`; g.fillRect(pad + j * cell, pad + i * cell, cell - 1, cell - 1); }
      g.fillStyle = '#a9a69f';
      const lab = i === 0 ? '·' : Mdl.show(ids[i]);
      g.fillText(lab, pad - 10 * r, pad + i * cell + cell * 0.7);
      g.fillText(lab, pad + i * cell + cell / 2, pad - 8 * r);
    }
  }

  function tensorTable(withStats) {
    const tb = $('tensors').querySelector('tbody'); tb.textContent = '';
    const MAX = 2000, list = G.tensors.slice(0, MAX);
    $('tensorsMore').textContent = G.tensors.length > MAX ? `Showing the first ${MAX.toLocaleString()} of ${G.tensors.length.toLocaleString()} tensors.` : '';
    $('tensorsMore').classList.toggle('hidden', G.tensors.length <= MAX);
    list.forEach((t) => {
      const tr = el('tr');
      [t.name, `[${t.shape.join(' × ')}]`, fmt(t.count), typeName(t.type)].forEach((v) => tr.appendChild(el('td', null, v)));
      if (!withStats || !t.data) { for (let k = 0; k < 5; k++) tr.appendChild(el('td', 'muted', '—')); tb.appendChild(tr); return; }
      const a = t.data, n = a.length; let s = 0, s2 = 0, mn = Infinity, mx = -Infinity;
      for (const v of a) { s += v; s2 += v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
      const mean = s / n, std = Math.sqrt(Math.max(0, s2 / n - mean * mean));
      [mean.toFixed(3), std.toFixed(3), mn.toFixed(2), mx.toFixed(2)].forEach((v) => tr.appendChild(el('td', null, v)));
      const td = el('td'), cv = el('canvas'); cv.width = 120; cv.height = 22; cv.style.width = '120px'; cv.style.height = '22px';
      const bins = new Array(24).fill(0); for (const v of a) bins[Math.min(23, Math.floor((v - mn) / ((mx - mn) || 1) * 24))]++;
      const g = cv.getContext('2d'), bm = Math.max(...bins); g.fillStyle = '#f2f1ee';
      bins.forEach((b, i) => { const h = (b / bm) * 20; g.fillRect(i * 5, 22 - h, 4, h); });
      td.appendChild(cv); tr.appendChild(td); tb.appendChild(tr);
    });
  }

  // ── loading ───────────────────────────────────────────────────────────────
  async function readHeader(file) {
    for (let n = 4 << 20; ; n *= 4) {
      const len = Math.min(file.size, n, LIMITS.headMax);
      const buf = await file.slice(0, len).arrayBuffer();
      try { return parseHeader(buf, file.size); }
      catch (e) {
        if (!(e instanceof NeedMore)) throw e;
        if (len >= LIMITS.headMax) throw new Error(`Its header is larger than ${LIMITS.headMax / 2 ** 20} MB, so this inspector stops there.`);
      }
    }
  }
  async function load(file) {
    $('err').classList.add('hidden');
    const busy = (t) => { $('busy').textContent = t || ''; $('busy').classList.toggle('hidden', !t); };
    try {
      if (!file || file.size < 24) throw new Error('This file is too small to be a GGUF model.');
      busy(`Reading the header of ${file.name || 'the file'} (${(file.size / 2 ** 20).toFixed(1)} MB)…`);
      const g = await readHeader(file); g.name = g.name || file.name;
      const why = fullModeReason(g);
      if (why) { G = g; Mdl = null; showHeaderOnly(why); busy(''); return; }
      busy('Reading the weights…');
      G = attachF32(g, await file.arrayBuffer()); Mdl = makeModel(G);
      showAll();
      busy('');
    } catch (e) {
      // Never leave the previous model's details on screen under an error.
      busy(''); G = null; Mdl = null; $('main').classList.add('hidden');
      $('err').textContent = String(e.message || e); $('err').classList.remove('hidden');
    }
  }
  $('file').addEventListener('change', (e) => { if (e.target.files[0]) load(e.target.files[0]); });
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) load(f); });
  $('netGo').addEventListener('click', netRun); $('netPrompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') netRun(); });
  $('netPos').addEventListener('input', drawNet);
  $('go').addEventListener('click', gen); $('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') gen(); });
  $('temp').addEventListener('input', () => { $('tempV').textContent = (+$('temp').value).toFixed(2); });
  $('attGo').addEventListener('click', attention); $('attLayer').addEventListener('change', attention); $('attHead').addEventListener('change', attention);
  window.addEventListener('resize', () => { if (Mdl && $('mode').classList.contains('hidden')) { drawNet(); attention(); drawMap(Mdl.tokens.map((t, i) => [t, i]).filter(([t]) => !/^<.*>$/.test(t))); } });
  window.E8Inspector = { load, parseGGUF, parseHeader, LIMITS };   // for testing

  // Opened from the training page ("Look inside this model"): the page hands the
  // model over directly, browser to browser, same site; nothing is uploaded.
  const host = window.opener || (window.parent !== window ? window.parent : null);
  if (host) {
    window.addEventListener('message', (e) => {
      if (e.origin !== location.origin || !e.data || e.data.type !== 'e8-model') return;
      $('drop').querySelector('p').textContent = 'Model received from the training page.';
      load(new File([e.data.bytes], e.data.name || 'model.gguf'));
    });
    try { host.postMessage({ type: 'e8-inspector-ready' }, location.origin); } catch (_) { /* host gone */ }
  }
})();
