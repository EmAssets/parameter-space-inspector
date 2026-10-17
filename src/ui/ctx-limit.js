// SPDX-License-Identifier: Apache-2.0
// The context meter on every prompt box: text past what the model can read turns red, with a note saying how
// much does not fit. A textarea cannot colour part of its own text, so a backdrop behind it (same font, same
// wrapping, scrolled with it) carries the red; the text stays in the textarea, so typing and selecting are unchanged.
// Which model reads a box decides the limit: the loaded model (Generate, Neural net, Anatomy, the experiments) or a
// Pythia checkpoint (Emergence). Counting needs the model's own tokenizer, so a short text is never sent: a piece
// is at least one character, so fewer characters than the limit always fit.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  // id: [who reads it, chat format?, checkbox that adds the E-level lens, fixed limit, note]
  const BOXES = {
    prompt: ['local', true, 'lensOn'], cmpQ: ['local', true, 'lensOn'], nnPrompt: ['local', true, 'nnLens'], anPrompt: ['local', true, 'anLens'],
    lsQ: ['local', true], paClean: ['local', true], paCorrupt: ['local', true],
    readText: ['local'], tokText: ['local'], repText: ['local'], ssText: ['local'], flObs: ['local'], flFrame: ['local'],
    fcText: ['local'], fgText: ['local'], cnText: ['local'],
    emPrompt: ['pythia', false, null, 2048 - 40, 'a Pythia checkpoint reads 2,048 pieces, 40 of them kept for its answer'],
    emText: ['pythia', false, null, 512, 'only the first 512 pieces are measured'],
  };
  const css = document.createElement('style');
  css.textContent = `.ctxw{position:relative;display:block;flex:1;width:100%;min-width:0}
.ctxw>textarea{position:relative;z-index:1;background:transparent!important}
.ctxb{position:absolute;top:0;left:0;z-index:0;overflow:hidden;pointer-events:none;color:transparent;white-space:pre-wrap;overflow-wrap:break-word;word-break:normal;box-sizing:border-box}
.ctxb mark{color:transparent;background:rgba(255,92,92,.30);border-radius:2px;text-decoration:underline wavy #ff5c5c;text-decoration-thickness:1px;text-underline-offset:3px}
.ctxn{margin:4px 0 0;font:12px/1.4 var(--mono,monospace);color:#ff7a7a}`;
  document.head.appendChild(css);

  const state = new Map();   // id -> { wrap, back, note, timer, seq }
  function attach(id) {
    const ta = $(id); if (!ta || state.has(id)) return;
    const cs = getComputedStyle(ta);
    const wrap = document.createElement('span'); wrap.className = 'ctxw';
    wrap.style.background = cs.backgroundColor; wrap.style.borderRadius = cs.borderRadius;
    ta.parentNode.insertBefore(wrap, ta); wrap.appendChild(ta);
    const back = document.createElement('div'); back.className = 'ctxb'; back.setAttribute('aria-hidden', 'true'); wrap.insertBefore(back, ta);
    const note = document.createElement('p'); note.className = 'ctxn'; note.hidden = true; note.setAttribute('role', 'status');
    wrap.after(note);
    const st = { wrap, back, note, timer: 0, seq: 0, cut: null }; state.set(id, st);
    ta.addEventListener('input', () => schedule(id));
    ta.addEventListener('scroll', () => { back.scrollTop = ta.scrollTop; });
    new ResizeObserver(() => paint(id)).observe(ta);
    const lens = BOXES[id][2]; if (lens && $(lens)) $(lens).addEventListener('change', () => schedule(id));
    schedule(id);
  }
  function schedule(id) { const st = state.get(id); clearTimeout(st.timer); st.timer = setTimeout(() => measure(id), 300); }

  // Where the piece past `limit` starts, as a character index, for the loaded model (llama.cpp's own tokenizer).
  async function localCut(text, limit) {
    const toks = await (window.PSIQuiet || psi).tokenize(text, false); if (toks.length <= limit) return { n: toks.length, cut: null };
    const pieces = toks.map((t) => t.piece || ''); const joined = pieces.join('');
    let pos = -Math.max(0, joined.length - text.length);   // a tokenizer that adds a leading space (the toy Llama) is one ahead
    for (let i = 0; i < limit; i++) pos += pieces[i].length;
    return { n: toks.length, cut: Math.max(0, Math.min(text.length, pos)) };
  }

  async function measure(id) {
    const ta = $(id), st = state.get(id), [who, chat, lensId, fixed, why] = BOXES[id], seq = ++st.seq, text = ta.value;
    let limit, r = { n: 0, cut: null }, about = '';
    try {
      if (who === 'pythia') {
        limit = fixed; about = why;
        if ([...text].length > limit) r = await psi.ckptCount({ text, limit });
      } else {
        const C = window.PSIContext; if (!C || !C.model()) return show(id, null);
        const ctx = C.ctx(); if (!isFinite(ctx)) return show(id, null);
        const over = chat ? await C.overhead(!!(lensId && $(lensId) && $(lensId).checked && (!window.PSIEM || PSIEM.on))) : 0;
        limit = ctx - over - 1;
        // The engine's context (n_ctx) is the limit a run meets; the trained length is said beside it when it differs.
        const t = C.trained ? C.trained() : null, e = C.engine ? C.engine() : null;
        const whose = e ? `this engine runs ${ctx.toLocaleString()} pieces${t && t !== e ? ` (the model was trained for ${t.toLocaleString()})` : ''}` : `this model reads ${ctx.toLocaleString()} pieces (as trained; the engine did not report its own limit)`;
        about = `${whose}${over > 1 ? `, ${over} of them taken by its chat format${lensId && $(lensId) && $(lensId).checked && (!window.PSIEM || PSIEM.on) ? ' and the lens' : ''}` : ''}`;
        if (text.length >= limit) r = await localCut(text, limit);
      }
    } catch (_) { return; }   // no engine yet: measure again on the next edit or model load
    if (seq !== st.seq) return;   // a newer edit is being measured
    show(id, r.cut == null ? null : { cut: r.cut, n: r.n, limit, about });
  }

  function show(id, over) {
    const st = state.get(id); st.cut = over ? over.cut : null;
    if (!over) { st.note.hidden = true; st.back.textContent = ''; return; }
    const chars = $(id).value.length - over.cut;
    st.note.textContent = `The red part (${chars.toLocaleString()} character${chars === 1 ? '' : 's'}) is past what the engine can read: ${over.about}, and this is ${over.n.toLocaleString()}. It will not see it.`;
    st.note.hidden = false; paint(id);
  }
  function paint(id) {
    const ta = $(id), st = state.get(id); if (!st || st.cut == null) return;
    const cs = getComputedStyle(ta), b = st.back;
    for (const k of ['fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'letterSpacing', 'tabSize', 'textAlign', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft', 'borderTopWidth', 'borderLeftWidth', 'borderRightWidth', 'borderBottomWidth']) b.style[k] = cs[k];
    b.style.borderStyle = 'solid'; b.style.borderColor = 'transparent';
    b.style.width = `${ta.clientWidth + parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth)}px`;   // without a scrollbar's width, so lines wrap alike
    b.style.height = `${ta.offsetHeight}px`;
    const v = ta.value; b.textContent = '';
    b.appendChild(document.createTextNode(v.slice(0, st.cut)));
    const m = document.createElement('mark'); m.textContent = v.slice(st.cut); b.appendChild(m);
    b.appendChild(document.createTextNode('\n'));   // a trailing newline keeps the last line's height
    b.scrollTop = ta.scrollTop;
  }

  function attachAll() { Object.keys(BOXES).forEach(attach); }
  // Text set by the app (presets, a new model's starting prompt) does not fire 'input': re-measure on model load and on clicks.
  document.addEventListener('psi-model', () => state.forEach((_, id) => schedule(id)));
  document.addEventListener('click', () => setTimeout(() => state.forEach((st, id) => { if ($(id).value !== st.last) { st.last = $(id).value; schedule(id); } }), 50), true);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attachAll); else attachAll();
  window.PSIContextMeter = { measure: (id) => schedule(id) };
})();
