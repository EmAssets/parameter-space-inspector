// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Parameter Space Inspector (desktop): the window. Talks to the engine only through window.psi.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const LENS = `Think through the Emergence Machine's E-levels, the fifteen floors of reality: E0 fields, E1 particles, E2 nuclei, E3 chemistry, E4 information-carrying molecules, E5 cells, E6 organisms, E7 ecosystems, E8 minds, E9 cultures, E10 institutions, E11 organizations, E12 states, E13 civilizations, E14 planetary-scale coordination.
For any question: name the level or levels the subject lives on; say what had to exist on the floors below for it to appear; say what it makes possible on the floors above; and point out what emerges at that level that none of the parts does on its own. Use the E-numbers. Be concise and concrete.`;
  const S = { model: null, header: null, eos: new Set(), steps: [], promptIds: [], busy: false, abort: false, cvPath: null, loraPath: null, sel: -1,
    cv: null, gen: null, props: {}, noThink: false, frozen: null };   // cv: the control vector the loaded model runs with; gen: how the current text was made (its runs' conditions);
  // props: what the engine reports about itself (engineState().props: n_ctx, build…); noThink: its chat format honours "/no_think";
  // frozen: the text on screen was made under conditions that have since changed (a model, steering or LoRA change) and is kept as it was
  // Why a run ended (meta.stop from the engine, or seen here), in the reader's words.
  const STOP = { eos: 'end token', limit: 'step limit', context: 'context full', stopped: 'stopped by you', error: 'an error', unknown: 'not reported' };
  const STOP_WHY = { eos: 'The model produced its end-of-answer token: it chose to stop.', limit: 'The number of steps asked for ran out; the model had more to say.', context: 'The engine’s context filled up: nothing more could be read or written.', stopped: 'You pressed Stop.', error: 'An error ended the run.', unknown: 'The engine did not say why it stopped.' };

  const err = (e, o) => PSI.err(e, o);
  const log = (m) => { const L = $('log'); L.textContent = (L.textContent + m + '\n').slice(-3000); L.scrollTop = L.scrollHeight; };
  psi.on('log', log);
  const show = (s) => (s == null ? '' : s.replace(/�/g, '▫'));   // a piece that is part of a character
  // One run at a time. While one runs, the buttons that would start another are disabled with a reason, and a
  // click that still arrives (a fork in the map) says so instead of vanishing. `view` marks that view's nav dot.
  const DEPENDENT = ['genGo', 'genMore', 'readGo', 'tokGo', 'cmpGo', 'forkGo', 'basinGo', 'cvApply', 'loraPick'];
  const lock = (on) => DEPENDENT.forEach((id) => PSI.block($(id), 'busy', on ? 'waiting for the current run…' : null));
  const busy = async (fn, view) => {
    if (S.busy) { PSI.notice('Waiting for the current run to finish.'); return undefined; }
    S.busy = true; S.abort = false; lock(true); document.body.style.cursor = 'progress'; if (view) PSI.setRunning(view, true);
    try { return await fn(); } catch (e) { err(e); return undefined; }
    finally { S.busy = false; lock(false); document.body.style.cursor = ''; if (view) PSI.setRunning(view, false); }
  };
  // A Stop-able progress line under a view's button row (PSI.progress), cleared a little after it ends.
  const progHost = (id) => { const row = $(id).closest('.row'); let h = row.nextElementSibling; if (!h || !h.classList.contains('prog-host')) { h = el('div', 'prog-host'); row.after(h); } return h; };
  const progress = (id, total) => PSI.progress(progHost(id), { total, track: false, onCancel: () => { S.abort = true; } });
  const settle = (p, text) => { p.finish(S.abort ? 'stopped' : text, { markDone: false }); setTimeout(() => { if (p.el.isConnected && p.el.classList.contains('ended')) p.remove(); }, 4000); };

  // ── the conditions of a local-model result (PSI.cond): which model, as changed or not, framed how, picked how ──
  const fileName = (f) => String(f || '').split(/[\\/]/).pop();
  const modelLabel = () => (S.header && PSIHeader.displayName(S.header.kv, S.header.name || S.model)) || fileName(S.model) || 'no model';
  // The thinking switch exists only when the engine says the model honours "/no_think"; otherwise the prompt is sent as written.
  const thinkWord = () => (!S.noThink ? 'thinking as the model does (no “/no_think” switch here)' : $('thinkOn').checked ? 'thinking shown' : 'thinking off: “/no_think” appended');
  const framing = (lensOn) => (plain() ? { t: 'no chat format: plain continuation', title: 'This model has no chat template, so there is nowhere to put the lens' }
    : { t: `${lensOn ? 'framed by the E-level lens' : 'no lens'} · ${thinkWord()}`, frame: lensOn, title: lensOn ? 'A system message asks the model to think in E-levels (Settings > Thinking in E-levels)' : '' });
  const localParts = () => [{ t: modelLabel(), title: `${S.model || ''}\nthe local engine (llama.cpp)` }, ...PSI.changeParts('local')];
  // The engine's own speed for a run: under 5 tokens/s on a GPU is how a spill into system memory looks from inside.
  const speedPart = (tps) => {
    const v = Array.isArray(tps) ? (tps.length ? tps.reduce((a, b) => a + b, 0) / tps.length : null) : tps;
    return v == null ? { t: 'tokens/s not reported', title: 'This engine sent no timings with its answer' }
      : { t: `${v.toFixed(v < 10 ? 1 : 0)} tokens/s${v < 5 ? ' · slow for this GPU?' : ''}`, warn: v < 5, title: v < 5 ? 'Under 5 tokens/s can mean the model spilled into system memory: see what holds the GPU (Settings)' : 'The engine’s own timing of its answer (predicted tokens per second)' };
  };
  // "show exact prompt": the text sent to the model, verbatim, under a conditions line. sent: {label, text} or a list of them.
  function showExact(condEl, sent) {
    if (!condEl) return;
    let d = condEl.nextElementSibling;
    if (!d || !d.classList.contains('exact')) { d = el('details', 'exact'); d.appendChild(el('summary', null, 'show exact prompt')); d.appendChild(el('pre')); condEl.after(d); }
    const items = [].concat(sent || []).filter(Boolean); d.hidden = !items.length;
    d.querySelector('pre').textContent = items.map((s) => (items.length > 1 ? `── ${s.label} ──\n` : '') + s.text).join('\n\n');
  }
  const rowOf = (id) => $(id).closest('.row');
  const genK = $('ribbon').previousElementSibling;   // Generate's "the text…" label: its conditions line goes under it

  // ── tabs ────────────────────────────────────────────────────────────────
  $('tabs').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-tab]'); if (!b) return;
    document.querySelectorAll('#tabs button[data-tab]').forEach((x) => { if (x === b) x.setAttribute('aria-current', 'page'); else x.removeAttribute('aria-current'); });
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('hidden', t.id !== `tab-${b.dataset.tab}`));
  });

  // ── the lens: system message + Qwen-style thinking switch ───────────────
  $('lensText').value = LENS;
  // Models without a chat template (the toy Llama from How AI Emerges, base models) continue plain text,
  // and small ones read only a short context, so the prompt and the run must fit inside it.
  const kvOf = (k) => (S.header && S.header.kv ? S.header.kv[k] : undefined);
  const plain = () => !kvOf('tokenizer.chat_template');
  // Two context lengths, kept apart: what the GGUF says the model was trained for, and what THIS engine runs (its
  // n_ctx, from engineState().props). A run meets the engine's limit; the trained length only says what a bigger
  // engine could read. The meter, fits() and the context-full stop all use the engine's.
  const trainedCtx = () => +kvOf(`${kvOf('general.architecture')}.context_length`) || null;
  const engineCtx = () => (S.props && Number.isInteger(S.props.n_ctx) && S.props.n_ctx > 0 ? S.props.n_ctx : null);
  const ctxLen = () => engineCtx() || trainedCtx() || Infinity;
  const ctxWords = () => {
    const t = trainedCtx(), e = engineCtx(), f = (n) => n.toLocaleString();
    return e && t && e !== t ? `trained for ${f(t)} · this engine runs ${f(e)}` : e ? `this engine runs ${f(e)}${t ? ' (as trained)' : ' (trained length not in the file)'}`
      : t ? `trained for ${f(t)} · this engine’s own limit not reported` : 'context not reported';
  };
  function fits(ids) {
    const c = ctxLen();
    if (ids.length >= c) throw new Error(`This engine runs ${c.toLocaleString()} pieces of context${engineCtx() && trainedCtx() && trainedCtx() !== c ? ` (the model was trained for ${trainedCtx().toLocaleString()})` : ''} and the prompt alone is ${ids.length.toLocaleString()}. Shorten the prompt.`);
    return ids;
  }
  // The page's own measurements (the context meter) go through psi.quiet: they never count as use or wake a sleeping model (src/idle.js).
  const QUIET = psi.quiet ? { tokenize: (...a) => psi.quiet('tokenize', ...a), template: (...a) => psi.quiet('template', ...a) } : psi;
  window.PSIQuiet = QUIET;
  // For the context meter on prompt boxes (ctx-limit.js): the engine's limit (and the trained one, in words) and what the chat format adds.
  window.PSIContext = { ctx: () => ctxLen(), trained: () => trainedCtx(), engine: () => engineCtx(), words: () => ctxWords(), plain: () => plain(), model: () => S.model,
    overhead: async (lensOn) => { if (plain()) return kvOf('tokenizer.ggml.add_bos_token') ? 1 : 0; try { return (await chatIds('', lensOn, QUIET)).length; } catch (_) { return 0; } } };
  // The exact text sent rides on the ids as `ids.text` (shown under "show exact prompt"); `ids.noThink` says "/no_think" was appended.
  async function chatIds(question, lensOn, P = psi) {
    if (plain()) {
      const ids = (await P.tokenize(question, false)).map((t) => t.id);
      const bos = kvOf('tokenizer.ggml.bos_token_id'), withBos = kvOf('tokenizer.ggml.add_bos_token') && bos != null;
      const out = fits(withBos ? [bos, ...ids] : ids); out.text = (withBos ? '[beginning-of-text token]' : '') + question; out.noThink = false;
      return out;
    }
    const messages = [];
    if (lensOn) messages.push({ role: 'system', content: typeof lensOn === 'string' ? lensOn : $('lensText').value });   // a string: Compare's own framing
    // "/no_think" only where the chat format knows it (engineState().no_think_supported); elsewhere the question goes as written.
    const noThink = S.noThink && !$('thinkOn').checked;
    messages.push({ role: 'user', content: noThink ? `${question} /no_think` : question });
    let prompt = await P.template(messages);
    // Qwen-style models: with thinking off, pre-fill the empty thinking block so the answer starts at once.
    if (noThink && /<\|im_start\|>assistant\n$/.test(prompt)) prompt += '<think>\n\n</think>\n\n';
    const out = fits((await P.tokenize(prompt, true)).map((t) => t.id)); out.text = prompt; out.noThink = noThink;
    return out;
  }
  // The thinking switch exists only when the engine says the model's chat format honours "/no_think".
  function thinkSwitch() {
    const cb = $('thinkOn'), lab = cb && cb.closest('label'); if (!lab) return;
    lab.classList.toggle('hidden', !S.noThink);
    let n = lab.nextElementSibling; if (!n || !n.classList.contains('think-note')) { n = el('p', 'muted small think-note'); lab.after(n); }
    n.textContent = !S.noThink && S.model ? 'No thinking switch for this model: its chat format has no “/no_think”, so questions are sent as written and it thinks as it does.' : '';
    n.hidden = !n.textContent;
  }
  // What the loaded text was made with, for freezing it when that changes: the model and its changes, and its tokenizer.
  const runSig = () => `${S.model}|${PSI.changesFor('local').map((c) => c.label).join('|')}`;
  const tokSig = () => { const t = kvOf('tokenizer.ggml.tokens'); return `${kvOf('tokenizer.ggml.model')}|${kvOf('tokenizer.ggml.pre')}|${Array.isArray(t) ? `${t.length}:${t.slice(1000, 1008).join(',')}` : '?'}`; };

  // ── opening a model ─────────────────────────────────────────────────────
  async function readHeader(file) {
    for (const n of [16 << 20, 128 << 20]) {
      const h = await psi.readHead(file, n);
      try { return { ...PSIHeader.parseHeader(h.bytes, h.size), name: h.name }; } catch (e) { if (!(e instanceof PSIHeader.NeedMore)) throw e; }
    }
    throw new Error('The header is larger than 128 MB.');
  }
  window.PSIOpenModel = (...a) => openModel(...a);
  async function openModel(file, cv, attach = false) {
    // A model asked for while another is still loading (say, from Settings during start-up) waits its turn instead of being dropped.
    for (let i = 0; S.busy && i < 1200; i++) await new Promise((r) => setTimeout(r, 250));
    await busy(async () => {
      const lora = $('loraOn').checked && S.loraPath ? { path: S.loraPath } : null;
      const mn = $('modelName'); mn.classList.remove('failed'); mn.textContent = 'loading…'; mn.title = file;
      PSI.setState({ loading: true });
      let started;
      const t0 = performance.now();
      try {
        S.header = await readHeader(file);
        showFile();
        // The context size chosen in Settings (window.PSIStartCtx, settings.js); omitted when unset, so the engine's default stands.
        const ctx = window.PSIStartCtx ? window.PSIStartCtx() : undefined;
        // Already running this model (the page was reloaded while the engine kept it): attach, never restart it.
        started = attach ? null : await psi.start(file, { ...(cv ? { controlVector: cv } : {}), ...(lora ? { lora } : {}), ...(ctx != null ? { ctx } : {}) });
      } catch (e) {
        // The header must not say "loading…" forever: say it failed, and where to fix it.
        S.model = null; S.cv = null; S.props = {}; S.noThink = false; thinkSwitch(); PSI.setChange('steer', null); PSI.setChange('lora', null); PSI.setState({ model: null, loading: false });
        mn.textContent = 'no model loaded · load failed · '; mn.classList.add('failed'); mn.appendChild(PSI.loadModelLink());
        err(e, { text: `The model could not be loaded (${String(file).split(/[\\/]/).pop()}). ${PSI.explain(e).text}` });
        return;
      }
      const setup = (started && started.setup) || {};
      if (window.PSISetup) window.PSISetup.refresh().catch((e) => console.warn('[psi] setup refresh', e));   // benign: the next state check repeats it
      S.model = file; S.cv = cv || null; PSI.setState({ model: file, loading: false });
      // What the engine says about itself (engineState().props: n_ctx, build, flash attention, KV type) and whether the
      // chat format honours "/no_think"; the start reply carries the same when the state call does not. Unsaid stays unsaid.
      const es = await psi.engineState().catch(() => null);
      S.props = (es && es.props) || (started && started.info) || {};
      PSI.state.moe = S.props.moe || null;
      const nt = es && es.no_think_supported != null ? es.no_think_supported : started && started.no_think_supported;
      S.noThink = !!nt; thinkSwitch();
      // What the loaded model runs with now (not what the sliders say): every local result and the header chip read this.
      PSI.setChange('steer', cv ? { label: `steered ${cv.scale > 0 ? '+' : ''}${cv.scale} at layer${cv.layers[0] === cv.layers[1] ? ` ${cv.layers[0]}` : `s ${cv.layers[0]}–${cv.layers[1]}`} (${fileName(cv.path)})`, short: `steered ${cv.scale > 0 ? '+' : ''}${cv.scale}`, scale: cv.scale, engine: 'local', view: 'settings', focus: 'cvScale' } : null);
      PSI.setChange('lora', lora ? { label: `LoRA: E-levels in the weights (${fileName(lora.path)})`, short: 'LoRA', engine: 'local', view: 'settings', focus: 'loraOn' } : null);
      // Text on screen made under other conditions is frozen (marked, not continued); restoring those exact conditions thaws it.
      if (S.gen && S.steps.length) { if (runSig() !== S.gen.sig) freeze(tokSig() !== S.gen.tokSig); else unfreeze(); }
      if (plain() && ctxLen() <= 512) {
        for (const id of ['prompt', 'cmpQ', 'anPrompt']) if ($(id).defaultValue === $(id).value) $(id).value = 'The street';
      }
      document.dispatchEvent(new CustomEvent('psi-model'));   // prompt boxes re-measure against the new model's context
      // Loaded beside something else on the card? Say so if anything spilled into system memory.
      if (window.PSIMachine) setTimeout(() => window.PSIMachine.spilled().then((sp) => { if (sp) PSI.err(`The GPU is overfull: ${sp.map((h) => h.label).join(', ')} spilled into system memory, so it runs many times slower. Unload a model or wait for the other job.`, { action: { label: 'See what holds the GPU', view: 'settings', focus: 'setMachine' } }); }).catch(() => {}), 4000);
      S.eos = new Set();
      for (const t of ['<|im_end|>', '<|endoftext|>', '</s>', '<|eot_id|>']) {
        const ids = await psi.tokenize(t, true).catch(() => []);   // benign: this model has no such end token
        if (ids.length === 1) S.eos.add(ids[0].id);
      }
      $('modelName').textContent = PSIHeader.displayName(S.header.kv, S.header.name || file);   // steering and LoRA: the "changed" chip beside it (PSI.setChange)
      $('modelName').title = `${file}\ncontext: ${ctxWords()}`;
      // The model line: the engine's own conditions beside the file's. Anything the engine did not report says so.
      const pr = S.props || {}, fa = pr.flash_attn;
      const eng = [`context: ${ctxWords()}`, pr.build ? `build ${pr.build}` : 'build not reported', fa == null ? 'flash attention not reported' : `flash attention ${fa === true ? 'on' : fa === false ? 'off' : fa}`,
        pr.kv_type ? `KV cache ${pr.kv_type}` : 'KV cache type not reported', pr.ngl != null ? `${pr.ngl} layers offered to the GPU` : null].filter(Boolean).join(' · ');
      $('engineInfo').textContent = `llama.cpp on this computer · loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${S.header.params.toLocaleString()} parameters · ${eng}${setup.modelDir ? ` · inside engine: ${setup.modelDir.split(/[\/]/).pop()}${setup.changed ? ` (${setup.how})` : ''}` : setup.note ? ` · ${setup.note}` : ''}`;
    });
  }
  // ── frozen text: made under conditions that have changed since (decision: never relabel, never continue) ─────
  function freeze(tokDiff) {
    const prev = S.gen.parts.map((p) => (typeof p === 'string' ? p : p.t)).join(' · '), now = PSI.changeParts('local').map((c) => c.t).join(', ');
    S.frozen = tokDiff
      ? { what: 'a previous model (another tokenizer)', why: `This text was made with a previous model (${prev}) whose pieces are not this model’s, so it cannot be continued or forked here. Generate afresh.` }
      : { what: 'a previous model/steering', why: `This text was made with ${prev}; the model has changed since (${modelLabel()}, ${now}). It is kept as it was: Generate afresh to continue with the model as it is now.` };
    freezeUI();
  }
  function unfreeze() { if (!S.frozen) return; S.frozen = null; freezeUI(); }
  function freezeUI() {
    ['genMore', 'forkGo', 'basinGo'].forEach((id) => PSI.block($(id), 'frozen', S.frozen ? `text made with ${S.frozen.what}: generate afresh` : null));
    $('ribbon').classList.toggle('frozen', !!S.frozen);
    render(); genCond();
  }
  $('open').addEventListener('click', async () => { try { const f = await psi.pickModel(); if (f) await openModel(f); } catch (e) { err(e); } });

  // ── generation, the fork map, the ribbon and the dice ───────────────────
  const K = () => Math.max(2, Math.min(20, +$('topK').value || 8));
  const temp = () => +$('temp').value;
  $('temp').addEventListener('input', () => { $('tempV').textContent = temp().toFixed(2); });
  function adjust(top, t) {   // probabilities after the creativity setting, over the returned choices
    if (t <= 0.001) return top.map((x, i) => (i === 0 ? 1 : 0));
    const w = top.map((x) => Math.pow(Math.max(x.p, 1e-12), 1 / t)); const s = w.reduce((a, b) => a + b, 0);
    return w.map((x) => x / s);
  }
  function sample(top, t) {
    const q = adjust(top, t); let r = Math.random(), i = 0;
    for (; i < q.length - 1; i++) { r -= q[i]; if (r <= 0) break; }
    return { i, q, r: Math.random() };
  }
  // One step: the model's odds for the next piece (its top 20, the die), a draw in the app, and the step's doubt.
  // Returns null to go on, or why it ended: 'context' (the engine's context is full) or 'eos' (an end token).
  // Doubt is entropy over the choices read with the rest as one bucket. It is exact when those choices held 98% of
  // the odds; otherwise the engine is asked once more for the top 200, for this step only, and the mark says which.
  async function step() {
    const ids = [...S.promptIds, ...S.steps.map((s) => s.id)];
    if (ids.length >= ctxLen()) return 'context';
    const d = await psi.next(ids, 20);
    if (!d.top.length) return 'eos';   // no odds at all: the model had already ended
    let entropy = d.entropy, hcov = d.covered, hk = 20;
    if (!((d.covered || 0) >= 0.98)) { const w = await psi.next(ids, 200).catch(() => null); if (w && w.top && w.top.length && w.entropy != null) { entropy = w.entropy; hcov = w.covered; hk = 200; } }
    const pick = sample(d.top, temp()), tm = (d.meta && d.meta.timings) || {};
    // The engine's own "eos" is about ITS greedy pick (the top choice); the app's draw ended only if it drew that one.
    const end = S.eos.has(d.top[pick.i].id) || (pick.i === 0 && d.meta && d.meta.stop === 'eos');
    S.steps.push({ top: d.top, entropy, covered: d.covered, hcov, hk, exact: (hcov || 0) >= 0.98, id: d.top[pick.i].id, piece: d.top[pick.i].piece, q: pick.q, chosen: pick.i, t: temp(), tps: tm.predicted_tps == null ? null : tm.predicted_tps, end });
    return end ? 'eos' : null;
  }
  // Colour scales clip: Generate's doubt at 5 bits, Read's surprise at 12. Past that a piece is marked, not just bright.
  const GEN_CLIP = 5, READ_CLIP = 12;
  // An end token has no text: name it in the fork map, the die and the labels instead of an empty piece.
  const END = 'end of answer', isEnd = (s, c, i) => S.eos.has(c.id) || (i === s.chosen && !!s.end);
  const lbl = (s, c, i) => (isEnd(s, c, i) ? `⟨${END}⟩` : show(c.piece).replace(/\n/g, '⏎'));
  const loose = (s) => !(s.exact != null ? s.exact : 1 - (s.covered == null ? 1 : s.covered) <= 0.02);
  const doubt = (s, dp = 1) => `${loose(s) ? 'at least ' : ''}${s.entropy.toFixed(dp)}`;
  const doubtTitle = (s) => { const k = s.hk || 20, c = s.hcov == null ? s.covered : s.hcov; return loose(s) ? `at least: ${((1 - (c == null ? 0 : c)) * 100).toFixed(0)}% of the odds lay outside the ${k} choices read` : `exact: its top ${k} choices held ${((c == null ? 1 : c) * 100).toFixed(1)}% of the odds`; };
  // The conditions of the text on screen, recorded per run (Generate, +20, a fork) when it was made, so a later
  // change of a setting or model never relabels it; a frozen text says what it was made with.
  function genCond() {
    if (!S.gen) return;
    const ts = [...new Set(S.steps.map((s) => s.t))].map((t) => t.toFixed(2)), runs = S.gen.runs || [], last = runs[runs.length - 1];
    const tps = S.steps.map((s) => s.tps).filter((v) => v != null), lb = S.steps.some(loose), ex = S.steps.some((s) => !loose(s)), forced = S.steps.filter((s) => s.forced).length;
    // At creativity 0 the top choice always wins, but two choices within 10 points can swap places between identical
    // runs (GPU arithmetic is not bit-exact): say how many steps were that close.
    const ties = S.steps.filter((s) => s.t <= 0.001 && !s.forced && s.top.length > 1 && s.top[0].p - s.top[1].p < 0.1).length;
    const p = PSI.cond(S.gen.anchor, [...S.gen.parts,
      S.frozen ? { t: `made with ${S.frozen.what}: frozen`, chg: true, title: S.frozen.why } : null,
      { t: `creativity ${ts.join(' then ') || temp().toFixed(2)}, drawn in the app from the model’s top 20`, term: 'creativity', title: 'The app reshapes the model’s odds for its 20 likeliest choices by creativity (the die odds) and draws one itself. The server draws nothing here: it only reports the odds.' },
      ties ? { t: `${ties} near-tie${ties === 1 ? '' : 's'}: a rerun may differ`, warn: true, title: 'Steps at creativity 0 where its top two choices were within 10 points (e.g. 52% and 48%). The likeliest one is taken, but such a close order can flip between identical runs, and the text after it with it.' } : null,
      { t: 'app draw unseeded: a rerun differs', term: 'seed' },
      forced ? { t: `${forced} step${forced === 1 ? '' : 's'} forced by you`, title: 'At a fork you picked the piece yourself: the die was not thrown there, so that step is your choice, not the model’s draw' } : null,
      last ? { t: `ended: ${STOP[last.stop]}${runs.length > 1 ? ` (${runs.length} runs)` : ''}`, title: runs.map((r, i) => `${i + 1}. ${r.how}: ${r.n} piece${r.n === 1 ? '' : 's'}, ended: ${STOP[r.stop]}. ${STOP_WHY[r.stop]}`).join('\n') } : null,
      last ? speedPart(tps) : null,
      S.gen.thinkingCut ? { t: 'no tokens left for the answer: still thinking', warn: true, title: 'The model opened a thinking block and every step went on it; the text is its thinking, not its answer. Continue to reach the answer.' } : null,
      { t: lb ? (ex ? 'doubt exact where its top 200 held 98% of the odds, else “at least”' : 'doubt is a lower bound') : 'doubt exact (the choices read held 98% of the odds)', est: lb, term: 'estimated', title: 'Entropy over the choices read, the rest counted as one bucket. Where the 20 on the die held under 98% of the odds the engine was asked for its top 200 for that step; “at least” remains where even those did not.' }]);
    showExact(p, S.gen.sent);
  }
  async function generate(n, how) {   // with a Stop-able progress line; S.abort ends the loop between steps; `how` names the run
    const p = progress('genGo', n), before = S.steps.length; let i = 0, stop = 'limit';
    try {
      for (; i < n && !S.abort; i++) { const end = await step(); p.step(i + 1); if (i % 4 === 3 || end) render(); if (end) { stop = end; break; } }
      if (S.abort && stop === 'limit') stop = 'stopped';
    } catch (e) { stop = 'error'; p.fail('stopped by an error'); throw e; } finally { if (S.gen) (S.gen.runs = S.gen.runs || []).push({ how, n: S.steps.length - before, stop }); render(); genCond(); }
    settle(p, `done · ${S.steps.length - before} pieces · ended: ${STOP[stop]}`);
    // Every step so far went on thinking: no tokens were left for the answer. Offer to go on from here.
    const th = PSI.thinking(S.promptIds.text, S.steps.map((s) => s.piece).join(''));
    if (S.gen) S.gen.thinkingCut = stop === 'limit' && th.cut;
    if (S.gen && S.gen.thinkingCut && !S.abort) {
      const more = Math.max(100, S.steps.length);
      PSI.warn(`No tokens left for the answer: all ${S.steps.length.toLocaleString()} pieces went on thinking, and the thinking block is still open. Add more steps to reach the answer.`,
        { action: { label: `Continue for ${more.toLocaleString()} more`, onClick: () => busy(() => generate(more, `+${more} (still thinking)`), 'generate') } });
      genCond();
    }
  }
  $('genGo').addEventListener('click', () => busy(async () => {
    if (!S.model) throw new Error('No model is loaded. Load a model in Settings first.');
    const lensOn = $('lensOn').checked;
    S.promptIds = await chatIds($('prompt').value, lensOn);
    S.steps = []; S.sel = -1; $('ghost').classList.add('hidden'); unfreeze();
    S.gen = { anchor: genK, parts: [...localParts(), framing(lensOn)], framed: framing(lensOn), sig: runSig(), tokSig: tokSig(), runs: [], sent: { label: 'sent to the model', text: S.promptIds.text } };
    await generate(+$('genN').value || 60, 'Generate');
  }, 'generate'));
  $('genMore').addEventListener('click', () => busy(async () => {
    if (!S.promptIds.length) throw new Error('Generate something first; this adds 20 more pieces to it.');
    if (S.frozen) throw new Error(S.frozen.why);
    await generate(20, '+20');
  }, 'generate'));

  // Path dependence: force another road at step j (your choice, not a draw), keep the old one as a ghost.
  async function fork(j, c) {
    if (S.frozen) { PSI.err(S.frozen.why, { action: { label: 'Generate afresh', view: 'generate', focus: 'genGo' } }); return; }
    await busy(async () => {
      const old = S.steps.slice(j).map((s) => s.piece).join('');
      const s = S.steps[j];
      S.steps = S.steps.slice(0, j);
      S.steps.push({ ...s, id: s.top[c].id, piece: s.top[c].piece, chosen: c, forced: true, end: S.eos.has(s.top[c].id) });
      $('ghost').textContent = `the road not taken from here: ${show(old)}`;
      $('ghost').classList.remove('hidden');
      S.sel = j;
      if (!S.eos.has(s.top[c].id)) await generate(Math.max(20, (+$('genN').value || 60) - j), `fork at step ${j + 1}`);
      else if (S.gen) { (S.gen.runs = S.gen.runs || []).push({ how: `fork at step ${j + 1}`, n: 0, stop: 'eos' }); genCond(); }
      render();
    }, 'generate');
  }

  function copiedMask() {   // generated runs of 4+ tokens that repeat the prompt's own tokens
    const P = S.promptIds, G = S.steps.map((s) => s.id), mask = new Array(G.length).fill(false), L = 4;
    const grams = new Set(); for (let i = 0; i + L <= P.length; i++) grams.add(P.slice(i, i + L).join(','));
    for (let i = 0; i + L <= G.length; i++) if (grams.has(G.slice(i, i + L).join(','))) for (let k = i; k < i + L; k++) mask[k] = true;
    return mask;
  }
  // Keyboard for the clickable pieces: one Tab stop per ribbon (or fork map), arrow keys move, Enter or Space picks.
  // After a re-render the focus comes back to the selected piece if it was inside before.
  const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
  function keysFor(box, move) {
    if (box._psiKeys) return; box._psiKeys = true;
    box.addEventListener('keydown', (e) => {
      const t = e.target.closest('[role=button]'); if (!t || !box.contains(t)) return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); t.click(); return; }
      const n = move(t, e.key); if (!n) return;
      e.preventDefault(); box.querySelectorAll('[role=button]').forEach((x) => { x.tabIndex = x === n ? 0 : -1; }); n.focus();
    });
  }
  function ribbonInto(R, mask, loop) {
    const had = R.contains(document.activeElement); R.textContent = '';
    const cur = S.sel >= 0 ? S.sel : 0;
    S.steps.forEach((s, j) => {
      const a = Math.min(1, s.entropy / GEN_CLIP), clip = s.entropy > GEN_CLIP;
      const sp = el('span', (j === S.sel ? 'sel ' : '') + (mask[j] ? 'copied ' : '') + (loop && loop.mark.has(j) ? 'loop ' : '') + (s.forced ? 'forced ' : '') + (clip ? 'clip' : ''), show(s.piece));
      sp.style.background = `rgba(242,241,238,${(0.04 + a * 0.5).toFixed(3)})`;
      const how = s.forced ? 'forced by you (not drawn)' : 'drawn by the app';
      sp.title = `${how} · model’s odds ${(s.top[s.chosen].p * 100).toFixed(1)}% · ${doubt(s, 2)} bits of doubt (${doubtTitle(s)})${clip ? ` · past the colour scale (${GEN_CLIP} bits): drawn at full brightness, marked ▴` : ''}`;
      sp.setAttribute('role', 'button'); sp.tabIndex = j === cur ? 0 : -1;
      if (s.end) sp.title = `${END} (the end token) · ${sp.title}`;
      sp.setAttribute('aria-label', `step ${j + 1}: “${lbl(s, s, s.chosen)}”, ${how}, model’s odds ${pct(s.top[s.chosen].p)}, ${doubt(s)} bits of doubt${clip ? ', past the colour scale' : ''}`);
      if (j === S.sel) sp.setAttribute('aria-pressed', 'true');
      sp.addEventListener('click', () => { S.sel = j; render(); document.querySelector(`.col[data-j="${j}"]`)?.scrollIntoView({ inline: 'center', block: 'nearest' }); });
      R.appendChild(sp);
    });
    keysFor(R, (t, key) => {
      const all = [...R.querySelectorAll('[role=button]')], i = all.indexOf(t);
      return key === 'ArrowRight' || key === 'ArrowDown' ? all[i + 1] : key === 'ArrowLeft' || key === 'ArrowUp' ? all[i - 1] : key === 'Home' ? all[0] : key === 'End' ? all[all.length - 1] : null;
    });
    if (had) { const f = R.querySelector('[tabindex="0"]'); if (f) f.focus({ preventScroll: true }); }
  }
  function render() {
    const mask = copiedMask();
    const loop = loopReport(); forkControls();
    ribbonInto($('ribbon'), mask, loop);
    const F = $('forks'), hadF = F.contains(document.activeElement); F.textContent = '';
    const k = K(), fj = S.sel >= 0 ? S.sel : 0;
    S.steps.forEach((s, j) => {
      const col = el('div', 'col' + (j === S.sel ? ' sel' : '')); col.dataset.j = j;
      const h = el('div', 'h', `${j + 1} · ${loose(s) ? '≥' : ''}${s.entropy.toFixed(1)} bits${s.forced ? ' · forced by you' : ''}`); h.title = doubtTitle(s) + (s.forced ? '\nForced by you: you picked this road; the die was not thrown here.' : '');
      if (s.forced) col.classList.add('forced');
      col.appendChild(h);
      s.top.slice(0, Math.max(k, s.chosen + 1)).forEach((c, i) => {
        const b = el('div', 'cand' + (i === s.chosen ? ' chosen' : ''));
        const bar = el('i'); bar.style.width = `${Math.max(2, c.p * 100)}%`; b.appendChild(bar);
        b.appendChild(el('b', null, pct(c.p)));
        const ce = isEnd(s, c, i); if (ce) b.classList.add('end');
        b.appendChild(document.createTextNode(lbl(s, c, i) || '∅'));
        const taken = i === s.chosen ? (s.forced ? 'forced by you' : 'the road taken (drawn)') : 'force this road instead';
        b.title = `${ce ? 'the end token: the answer ends here · ' : ''}${taken} · model’s odds ${pct(c.p)}`;
        b.dataset.i = i; b.setAttribute('role', 'button'); b.tabIndex = j === fj && i === s.chosen ? 0 : -1;
        b.setAttribute('aria-label', `step ${j + 1}: “${lbl(s, c, i) || 'nothing'}” model’s odds ${pct(c.p)}, ${taken}`);
        if (i === s.chosen) b.setAttribute('aria-disabled', 'true'); else b.addEventListener('click', () => fork(j, i));
        col.appendChild(b);
      });
      F.appendChild(col);
    });
    // Fork map keys: left and right change step (same rank), up and down change choice.
    keysFor(F, (t, key) => {
      const col = t.closest('.col'), cands = [...col.querySelectorAll('.cand')], i = cands.indexOf(t);
      if (key === 'ArrowDown') return cands[i + 1]; if (key === 'ArrowUp') return cands[i - 1];
      const other = key === 'ArrowRight' ? col.nextElementSibling : key === 'ArrowLeft' ? col.previousElementSibling : key === 'Home' ? F.firstElementChild : key === 'End' ? F.lastElementChild : null;
      if (!other) return null; const oc = other.querySelectorAll('.cand'); return oc[Math.min(i, oc.length - 1)];
    });
    if (hadF) { const f = F.querySelector('.cand[tabindex="0"]'); if (f) f.focus({ preventScroll: true }); }
    if (S.sel < 0 && F.lastChild) F.scrollLeft = F.scrollWidth;
    drawDice(S.steps[S.sel >= 0 ? S.sel : S.steps.length - 1]);
  }

  function drawDice(s) {
    const cv = $('dice'), r = Math.min(2, devicePixelRatio || 1);
    cv.width = cv.clientWidth * r; cv.height = 70 * r;
    const g = cv.getContext('2d'), W = cv.width, H = cv.height;
    g.fillStyle = '#0b0b0b'; g.fillRect(0, 0, W, H);
    if (!s) return;
    const q = s.q || adjust(s.top, temp());
    let x = 0; g.font = `${11 * r}px Consolas, monospace`; g.textBaseline = 'middle';
    q.forEach((p, i) => {
      const w = p * W; if (w < 0.5) return;
      g.fillStyle = i === s.chosen ? '#f2f1ee' : `rgba(242,241,238,${0.12 + 0.25 * (i % 2)})`;
      g.fillRect(x, 10 * r, w - 1, 30 * r);
      if (w > 34 * r) { g.fillStyle = i === s.chosen ? '#000' : '#f2f1ee'; g.fillText(isEnd(s, s.top[i], i) ? '⟨end⟩' : show(s.top[i].piece).replace(/\n/g, '⏎').slice(0, 10), x + 4 * r, 25 * r); }
      if (i === s.chosen) { g.fillStyle = '#f2f1ee'; g.beginPath(); g.moveTo(x + w / 2, 44 * r); g.lineTo(x + w / 2 - 6 * r, 54 * r); g.lineTo(x + w / 2 + 6 * r, 54 * r); g.fill(); }
      x += w;
    });
    const t = s.t == null ? temp() : s.t, pc = s.end ? `the end token (${END})` : `“${show(s.piece).replace(/\n/g, '⏎')}”`, sj = S.steps.indexOf(s) + 1;
    g.fillStyle = '#a9a69f'; g.fillText(s.forced ? `die odds (model’s odds at creativity ${t.toFixed(2)}); not thrown: ${pc} was forced by you` : `die odds (model’s odds at creativity ${t.toFixed(2)}); it landed on ${pc}`, 4 * r, 62 * r);
    cv.setAttribute('aria-label', s.forced ? `The die odds for step ${sj}; the die was not thrown: ${pc} was forced by you` : `The die odds for step ${sj}: the model’s odds reshaped by creativity ${t.toFixed(2)}; it landed on ${pc}`);
    // Off the die: the odds beyond the 20 choices read can never be drawn here. Say how much that is.
    let note = $('diceNote'); if (!note) { note = el('p', 'small muted'); note.id = 'diceNote'; cv.after(note); }
    const rest = 1 - (s.covered == null ? 1 : s.covered);
    note.textContent = rest > 0.005 ? `Only its 20 likeliest choices are on the die: the other ${(rest * 100).toFixed(rest < 0.1 ? 1 : 0)}% of its odds cannot be drawn here.` : '';
  }
  addEventListener('resize', () => render());

  // ── where the roads go (Generate, below the die; once its own view, Paths & attractors) ────────────────────────────────────────────────
  const P = window.PSIPaths;
  const tagE = (e) => el('span', 'tag', `E${e} ${P.NAMES[e]}`);
  // E-level readings and Lexicon concepts are the framework's: left out while its content is off (Settings > Content).
  const EMON = () => !window.PSIEM || PSIEM.on;
  // A level reading with the evidence behind it: how many votes (labelled words, and E-numbers at 3 votes each).
  const lensWas = () => !!(S.gen && S.gen.framed && S.gen.framed.frame);
  function levelTag(r) {
    const t = el('span', 'tag'); if (!r || r.e == null) { t.textContent = 'no clear level'; t.classList.add('muted'); t.title = r && r.votes ? `${r.votes} votes, no level ahead` : 'no labelled words in these roads: nothing to read a level from'; return t; }
    t.textContent = `E${r.e} ${P.NAMES[r.e]} · ${r.votes} vote${r.votes === 1 ? '' : 's'}`;
    t.title = `${r.votes} pieces of evidence${r.eMentions ? `, ${r.eMentions} of them literal E-number mentions (3 votes each)` : ''}; strength ${r.score}`;
    return t;
  }
  // With the lens on, the model is told to write E-numbers, so a reading also counts its own echo of the prompt:
  // say so, and give the reading from words alone beside it.
  function levelLine(box, label, r, rNoE) {
    if (!EMON()) return;
    const p = el('p'); p.append(label); p.appendChild(levelTag(r));
    if (lensWas()) {
      p.append(' '); p.appendChild(el('span', 'muted small', `lens on: E-number mentions counted (${(r && r.eMentions) || 0}) · without them: `)); p.appendChild(levelTag(rNoE));
      if (rNoE && r && r.e != null && rNoE.e !== r.e) p.appendChild(el('span', 'odd small', ' · the reading changes without the mentions'));
    } else p.appendChild(el('span', 'muted small', ' read by this app from the words; the model was not asked about E-levels'));
    box.appendChild(p);
  }
  // The server's sampler, as it reported using it (meta.settings), beside what was asked; its stops and its speed.
  const STOPS = { eos: 'end token', limit: 'piece limit', context: 'context full', unknown: 'not reported' };
  function serverParts(m, asked) {
    if (!m || !m.reported) return [{ t: `server sampler (llama-server): asked creativity ${asked.temperature}, top-p ${asked.topP}; settings used not reported`, term: 'creativity' }, { t: 'stops not reported' }, speedPart(null)];
    const s = m.sampler || {}, v = (x, d = 2) => (x == null ? 'not reported' : +x.toFixed(d));
    const stops = Object.entries(m.stops || {}).map(([k, n]) => `${n} ${STOPS[k] || k}`).join(', ');
    return [{ t: `server sampler (llama-server): creativity ${m.temps && m.temps.length > 1 ? m.temps.map((x) => +x.toFixed(2)).join('/') : v(s.temperature)}, top-p ${v(s.top_p)}, top-k ${v(s.top_k, 0)}, min-p ${v(s.min_p)}${m.settingsVary ? ' (varied between roads)' : ''}`, term: 'creativity', title: 'What the server reported it used for these roads; the server draws them itself, unlike Generate’s die' },
      { t: `roads ended: ${stops}${m.reported < m.of ? ` (${m.of - m.reported} not reported)` : ''}` }, speedPart(m.tps)];
  }
  const tagC = (c) => { const t = el('span', 'tag'); const a = el('a', null, c.term); a.href = c.url; a.target = '_blank'; a.rel = 'noopener'; a.title = c.definition || ''; t.appendChild(a); return t; };
  function forkControls() {
    const sel = $('forkAlt'); sel.textContent = '';
    if (S.sel < 0 || !S.steps[S.sel]) { $('forkWhich').textContent = S.steps.length ? 'no step selected: pick a word in the text above' : 'nothing generated yet'; return; }
    const s = S.steps[S.sel], certain = s.top[s.chosen].p >= 0.9;
    $('forkWhich').textContent = `step ${S.sel + 1}: ${s.forced ? 'forced by you' : 'drew'} “${lbl(s, s, s.chosen)}” (model’s odds ${(s.top[s.chosen].p * 100).toFixed(0)}%)${certain ? ' · not a real fork: the model was nearly certain here; pick a dimmer word' : ''}`;
    s.top.slice(0, K()).forEach((c, i) => { if (i === s.chosen) return; const o = el('option', null, `“${lbl(s, c, i)}” ${(c.p * 100).toFixed(c.p < 0.1 ? 1 : 0)}%`); o.value = i; sel.appendChild(o); });
  }
  function describeRoad(box, r, forced) {
    box.textContent = '';
    levelLine(box, `“${show(r.piece).replace(/\n/g, '⏎')}”${forced ? ' (forced by you)' : ''} → reads as `, r.level, r.levelNoE);
    box.appendChild(el('p', 'muted small', `${r.samples.length} sampled continuations, ${r.clusters.length} distinct openings; the main one ${(r.clusters[0].share * 100).toFixed(0)}%:`));
    box.appendChild(el('p', 'small', show(r.clusters[0].example).replace(/\s+/g, ' ').slice(0, 220)));
    if (EMON()) { const lv = el('p'); lv.append('levels its words point to: '); [...r.levels].sort((a, b) => a - b).forEach((e) => lv.appendChild(tagE(e))); box.appendChild(lv);
    if (r.concepts.length) { const cp = el('p'); cp.append('lexicon concepts: '); r.concepts.slice(0, 12).forEach((c) => cp.appendChild(tagC(c))); box.appendChild(cp); } }
  }
  // The roads work on the last generated text: with none, or no step picked, say what to do instead of failing.
  const toPrompt = { label: 'Write a prompt', view: 'generate', focus: 'prompt' };
  // The first step where the model was really torn (its top choice under 60%): where a fork is worth analysing.
  function pickFork() {
    const f = P.firstFork(S.steps); if (f < 0) { PSI.notice('No step here was a real fork: the model was nearly certain at every step. Raise creativity, or ask something more open.'); return; }
    S.sel = f; render(); const t = $('ribbon').querySelector('[tabindex="0"]'); if (t) { t.scrollIntoView({ block: 'center' }); t.focus({ preventScroll: true }); }
  }
  function pathsReady(needStep) {
    if (!S.steps.length || !S.promptIds.length) { PSI.err('nothing generated', { text: 'Nothing has been generated yet: press Generate first; the roads start from that text.', action: toPrompt }); return false; }
    if (needStep && S.sel < 0) { PSI.err('no step', { text: 'Pick a step first: click a word in the text above (or Tab to it and press Enter).', action: { label: 'Pick the first real fork', onClick: pickFork } }); return false; }
    return true;
  }
  const failStat = (id, what) => (e) => { PSI.status($(id), `${what} failed: ${PSI.explain(e).text}`, 'error'); throw e; };
  // The roads are sampled on the server with fixed seeds: the same fork usually gives the same roads again (GPU arithmetic can differ).
  const seeds = (k) => ({ t: `seeded ${P.SEED0}–${P.SEED0 + k - 1}: usually repeatable`, term: 'seed', title: 'Fixed seeds make the server draw the same roads again on the same model and settings; GPU arithmetic can still differ between runs' });
  // The roads continue Generate's last prompt as it was sent: a lens switched since then is not applied to them, so say so.
  const pathParts = () => {
    const was = lensWas(), now = $('lensOn').checked;
    const stale = S.gen && S.gen.framed && was !== now && { t: `generated with the lens ${was ? 'on' : 'off'}; it is ${now ? 'on' : 'off'} now, but these roads continue that prompt: Generate again to change it`, warn: true };
    return [...localParts(), (S.gen && S.gen.framed) || framing(now), stale].filter(Boolean);
  };
  const pathSent = () => (S.gen && S.gen.sent ? S.gen.sent : null);
  $('forkGo').addEventListener('click', () => pathsReady(true) && busy(async () => {
    const alt = +$('forkAlt').value; if (Number.isNaN(alt)) throw new Error('No alternative at this step.');
    const r = await P.fork(S.promptIds, S.steps, S.sel, alt, { onProgress: (t) => { PSI.status($('forkStat'), t, 'busy'); }, promptText: $('prompt').value }).catch(failStat('forkStat', 'The fork analysis'));
    PSI.status($('forkStat'), 'done', 'ok');
    const F = P.FORK, nz = r.noise || { total: 0, noisy: 0 };
    const pc = PSI.cond(rowOf('forkGo'), [...pathParts(), { t: `${F.K} roads per choice × ${F.n} pieces, asked at creativity ${F.temperature}, top-p ${F.topP}`, term: 'creativity' }, ...serverParts(r.meta, F), seeds(F.K),
      { t: `same-road control: each road split ${nz.halves ? nz.halves.join('/') : 'in half'}`, est: true, title: 'Each road’s continuations are split in two; anything that differs between two halves of the SAME road is sampling noise, and is marked so below' }]);
    showExact(pc, pathSent());
    describeRoad($('forkTaken'), r.roads[0], r.forced); describeRoad($('forkAlt2'), r.roads[1]);
    const v = $('forkVerdict'); v.textContent = '';
    if (r.forced) v.appendChild(el('p', 'small odd', `The road taken at step ${S.sel + 1} was forced by you, not drawn: “closed” and “kept open” compare your pick with the alternative.`));
    const noisyTag = (node, noisy) => { if (noisy) { node.classList.add('noisy'); node.title = `${node.title ? node.title + '\n' : ''}also differs between two halves of one road: within noise`; } return node; };
    const line = (label, x0) => {
      const x = EMON() ? x0 : { ...x0, levels: [], concepts: [] };
      const p = el('p'); p.appendChild(el('b', null, label + ' '));
      if (!x.levels.length && !x.words.length && !x.concepts.length) p.append('nothing distinct in this sample');
      x.levels.forEach((o) => p.appendChild(noisyTag(tagE(o.e), o.noisy)));
      x.concepts.forEach((c) => p.appendChild(noisyTag(tagC(c), c.noisy)));
      if (x.words.length) { p.append((x.levels.length || x.concepts.length ? ' · ' : '') + 'words only on this road: '); x.words.forEach((o, k) => { if (k) p.append(', '); p.appendChild(noisyTag(el('span', null, o.w), o.noisy)); }); }
      v.appendChild(p);
      x.concepts.slice(0, 3).forEach((c) => { if (c.definition) v.appendChild(el('p', 'muted small', `${c.term}: ${c.definition}${c.parents && c.parents.length ? ' · emerges from ' + c.parents.slice(0, 4).join(', ') : ''}`)); });
    };
    line('This choice closed:', r.closed); line('And kept open:', r.kept);
    // The control leads when it ate the finding: most named differences also differ inside one road.
    const most = nz.total && nz.noisy / nz.total >= 0.5, ctl = el('p', 'small' + (most ? ' odd' : ' muted'));
    ctl.textContent = !nz.total ? 'Same-road control: nothing distinct to check.' : `Same-road control: ${nz.noisy} of ${nz.total} differences named above also differ between two halves of one road (struck through): within noise${most ? '. Most of this verdict is noise at this sample size' : ''}.`;
    v.appendChild(ctl);
    v.appendChild(el('p', 'muted small', 'Levels are a lexical reading from the site’s labelled street sentences; concepts come from the Emergence Lexicon with their derivation chains; “closed” means present in the other road’s continuations and absent from this road’s. Small samples: a sketch, not a proof.'));
  }, 'generate'));
  $('basinGo').addEventListener('click', () => pathsReady(false) && busy(async () => {
    const ids = $('basinFrom').value === 'start' ? S.promptIds : [...S.promptIds, ...S.steps.map((s) => s.id)];
    const K = +$('basinK').value || 16;
    const r = await P.basins(ids, { K, onProgress: (t) => { PSI.status($('basinStat'), t, 'busy'); } }).catch(failStat('basinStat', 'Mapping the basins'));
    // The escape point is only known between two tried settings; outside the range tried it is off scale, not a value.
    const T = r.temps, i = T.indexOf(r.escape), f = (x) => x.toFixed(1), se = r.se != null ? r.se : Math.sqrt(0.25 / K), sePts = Math.floor(100 * se);   // K=16: ±12 points
    // A share whose ±1 SE spans 50% cannot say which side of a majority it is on: an escape there is within noise.
    const spans = (x) => x != null && Math.abs(x - 0.5) <= se, share = (t) => (r.byT[t] ? r.byT[t].mainShare : null);
    const noisy = r.escape != null && (spans(share(r.escape)) || (i > 0 && spans(share(T[i - 1]))));
    PSI.status($('basinStat'), r.escape == null ? `no escape within the range tried (creativity ${f(T[0])}–${f(T[T.length - 1])}): one basin keeps the majority throughout; any escape lies above ${f(T[T.length - 1])}`
      : i === 0 ? `no basin holds a majority even at creativity ${f(T[0])}, the lowest tried: any escape lies at or below it${noisy ? ` · within noise: the main basin’s ${(share(T[0]) * 100).toFixed(0)}% ±${sePts} spans 50%` : ''}`
        : noisy ? `within noise: the main basin goes from ${(share(T[i - 1]) * 100).toFixed(0)}% to ${(share(r.escape) * 100).toFixed(0)}% between creativity ${f(T[i - 1])} and ${f(r.escape)}, but ±${sePts} points (${K} roads) spans 50%: no escape can be placed`
          : `escape between creativity ${f(T[i - 1])} and ${f(r.escape)}: from there on no basin holds the majority`, r.escape == null || i === 0 || noisy ? 'warn' : 'ok');
    const pc = PSI.cond(rowOf('basinGo'), [...pathParts(), { t: `${K} roads × ${P.BASIN.n} pieces, asked at each creativity ${T.map(f).join(', ')}, top-p ${P.BASIN.topP}`, term: 'creativity' }, ...serverParts(r.meta, { temperature: T.map(f).join('/'), topP: P.BASIN.topP }), seeds(K),
      { t: `shares ±${sePts} points (1 SE, ${K} roads)`, est: true, title: 'The sampling noise of a share near 50% with this many roads: a difference smaller than this is within noise' }]);
    showExact(pc, pathSent());
    const out = $('basinOut'); out.textContent = '';
    for (const T of r.temps) {
      const b = r.byT[T]; out.appendChild(el('p', 'k', `creativity ${T.toFixed(1)}: ${b.distinct} distinct openings, main basin ${(b.mainShare * 100).toFixed(0)}% ±${sePts}${spans(b.mainShare) ? ' (spans 50%: majority within noise)' : ''}`));
      b.clusters.slice(0, 4).forEach((c) => {
        const d = el('div', 'basin'); const l = el('div'); l.appendChild(el('b', null, `${(c.share * 100).toFixed(0)}%`)); const bar = el('div', 'bar'); bar.style.width = `${Math.max(4, c.share * 70)}px`; l.appendChild(bar);
        if (EMON()) l.appendChild(levelTag(c.level)); if (EMON() && lensWas() && c.levelNoE) { l.appendChild(el('span', 'muted small', ' without E-mentions: ')); l.appendChild(levelTag(c.levelNoE)); }
        d.appendChild(l); d.appendChild(el('div', 'ex', show(c.example).replace(/\s+/g, ' ').slice(0, 200))); out.appendChild(d);
      });
    }
    if (lensWas()) out.appendChild(el('p', 'muted small', 'lens on: E-number mentions counted (the lens asks the model to write them); each basin also shows its reading from words alone.'));
  }, 'generate'));
  function loopReport() {
    const L = P.loops(S.steps); const box = $('loopOut');
    if (!S.steps.length) { box.textContent = 'Nothing generated yet.'; return null; }
    if (!L) { box.textContent = 'No repeated phrase yet: the text has not fallen into a rut. (Push creativity down or steering up to find one.)'; return null; }
    const al = L.lowerBound ? 'at least ' : '';
    box.textContent = `Attractor: “${show(L.text)}” repeats ${L.repeats} times from step ${L.from + 1}; average doubt after entering it ${al}${L.entropyAfter} bits, before it ${L.entropyBefore == null ? 'not measured (the loop starts at step 1)' : `${al}${L.entropyBefore} bits over ${L.nBefore} step${L.nBefore === 1 ? '' : 's'}`}${L.lowerBound ? ' (“at least”: some steps’ odds were not fully read)' : ''}. A loop is a basin the model cannot climb out of at this creativity: each repeat makes the next more likely.`;
    return L;
  }

  // ── read: surprise, piece by piece ──────────────────────────────────────
  // The last reading as data (the experiment designer's tour collects it after pressing "Read it").
  let READ_LAST = null, READ_NO = 0;
  window.PSIRead = { no: () => READ_NO, last: () => (READ_LAST ? JSON.parse(JSON.stringify(READ_LAST)) : null) };
  psi.on('progress', ({ i, n }) => { $('readStat').textContent = `reading ${i}/${n}`; });
  // Surprise is −log2 of the odds the model gave the piece that came; doubt (entropy) is the surprise it expected on
  // average. Surprise − doubt says how much more (or less) surprised it was than it expected to be.
  function readHist(rows) {
    const out = $('readOut'); let h = $('readHist');
    if (!h) { h = el('div', 'surp-hist'); h.id = 'readHist'; h.setAttribute('role', 'img'); out.after(h); }
    h.textContent = '';
    const bins = Array.from({ length: READ_CLIP }, (_, b) => ({ label: `${b}–${b + 1}`, n: 0 }));
    const clip = { label: `${READ_CLIP}+`, n: 0, cls: 'clip', title: `past the colour scale (${READ_CLIP} bits)` }, off = { label: 'off scale', n: 0, cls: 'offscale', title: 'not in its top 20: only a floor is known, so not placed in a bin' };
    rows.forEach((r) => { if (r.surprise == null) off.n++; else if (r.surprise >= READ_CLIP) clip.n++; else bins[Math.floor(r.surprise)].n++; });
    const all = [...bins, clip, off], mx = Math.max(1, ...all.map((b) => b.n));
    h.appendChild(el('p', 'k', 'surprise per piece, in bits (how many pieces fell in each)'));
    const row = el('div', 'surp-bars');
    all.forEach((b) => {
      const c = el('div', 'surp-bin' + (b.cls ? ` ${b.cls}` : '')); c.title = `${b.label} bits: ${b.n} piece${b.n === 1 ? '' : 's'}${b.title ? ` · ${b.title}` : ''}`;
      const bar = el('i'); bar.style.height = `${Math.round((b.n / mx) * 48)}px`; if (!b.n) bar.classList.add('zero');
      c.appendChild(el('b', null, b.n ? String(b.n) : '')); c.appendChild(bar); c.appendChild(el('small', null, b.cls === 'offscale' ? 'off' : b.label.split('–')[0]));
      row.appendChild(c);
    });
    h.appendChild(row);
    h.setAttribute('aria-label', `Histogram of surprise: ${all.filter((b) => b.n).map((b) => `${b.n} at ${b.label} bits`).join(', ')}`);
  }
  $('readGo').addEventListener('click', () => busy(async () => {
    if (!S.model) throw new Error('No model is loaded. Load a model in Settings first.');
    const text = $('readText').value;
    if (!text.trim()) { PSI.status($('readStat'), 'Type something first: there is no text to read.', 'warn'); return; }
    const toks = await psi.tokenize(text, false);
    if (toks.length > 400) throw new Error('Keep it under 400 pieces for now.');
    if (toks.length < 2) { PSI.status($('readStat'), 'Only one piece: nothing comes before it, so there is no guess to measure. Type a few more words.', 'warn'); return; }
    const sc = await psi.score(toks.map((t) => t.id), 20);   // top 20: fewer pieces fall outside the list and saturate
    // The pieces are replaced: their hover tip never gets a mouseleave, so hide it. A preset's note shows only under its own text.
    const out = $('readOut'); out.textContent = ''; $('tip').classList.add('hidden');
    document.querySelectorAll('[data-read][data-note]').forEach((b) => $(b.dataset.note).classList.toggle('hidden', b.dataset.read !== text.trim()));
    const first =el('span', 'prompt', show(toks[0].piece)); first.title = 'the first piece: nothing comes before it (no beginning-of-text token was given), so there is no guess to measure'; out.appendChild(first);
    let total = 0, n = 0, off = 0, floors = 0, nFloor = 0, clipped = 0;
    const rows = [], tps = [], kept = [];   // kept: each scored piece, for the experiment designer (PSIRead.last)
    sc.forEach((s, i) => {
      // A piece outside its top 20 is off scale: only a floor is known (the 20th guess's surprise). Drawn hatched, never in the plain mean.
      const floor = s.top.length ? -Math.log2(s.top[s.top.length - 1].p) : null, surprise = s.p ? -Math.log2(s.p) : null;
      const cov = s.top.reduce((a, x) => a + x.p, 0), ent = s.entropy == null ? null : s.entropy;
      // Doubt is a lower bound when the top 20 left odds unread (the engine flags it; 98% covered counts as exact here).
      const entLow = ent != null && s.entropy_is_lower_bound !== false && cov < 0.98;
      if (surprise == null) { off++; if (floor != null) { floors += floor; nFloor++; } } else { total += surprise; n++; if (surprise > READ_CLIP) clipped++; }
      if (s.meta && s.meta.timings && s.meta.timings.predicted_tps != null) tps.push(s.meta.timings.predicted_tps);
      rows.push({ surprise }); kept.push({ piece: toks[i + 1].piece, surprise, floor, rank: s.rank == null ? null : s.rank });
      const clip = surprise != null && surprise > READ_CLIP;
      const sp = el('span', surprise == null ? 'offscale' : clip ? 'clip' : null, show(toks[i + 1].piece));
      if (surprise != null) sp.style.background = `rgba(242,241,238,${(0.03 + Math.min(1, surprise / READ_CLIP) * 0.6).toFixed(3)})`;
      const over = floor == null ? 'off scale' : `over ${floor.toFixed(1)}`;
      const entTxt = ent == null ? 'doubt not measured (the model had ended here)' : `doubt ${entLow ? '≥ ' : ''}${ent.toFixed(1)} bits${entLow ? ` (lower bound: ${((1 - cov) * 100).toFixed(0)}% of its odds lay outside the 20 read)` : ''}`;
      const ex = surprise != null && ent != null ? surprise - ent : null;
      const exTxt = ex == null ? (surprise == null ? 'surprise − doubt: off scale' : 'surprise − doubt: not measured') : `surprise − doubt ${entLow ? '≤ ' : ''}${ex >= 0 ? '+' : ''}${ex.toFixed(1)} bits (${ex > 1 ? 'more surprised than it expected to be' : ex < -1 ? 'less surprised than it expected to be' : 'about as surprised as it expected'})`;
      const rank = s.rank != null ? `rank ${s.rank} of its guesses` : 'rank: not in its top 20';
      sp.dataset.bits = surprise != null ? surprise.toFixed(2) : floor == null ? 'off scale' : `>${floor.toFixed(1)}`; sp.dataset.expected = s.p ? (s.p * 100).toFixed(1) + '%' : 'outside its top 20 (off scale)';   // exact values, for readers and Prometheus
      if (ex != null) sp.dataset.excess = ex.toFixed(2); if (s.rank != null) sp.dataset.rank = s.rank; if (ent != null) sp.dataset.entropy = `${entLow ? '>=' : ''}${ent.toFixed(2)}`;
      sp.setAttribute('aria-label', `“${show(toks[i + 1].piece)}”: ${s.p ? `surprise ${surprise.toFixed(1)} bits` : `off scale, surprise ${over} bits`}, ${rank}`);
      sp.addEventListener('mousemove', (ev) => {
        const t = $('tip'); t.classList.remove('hidden'); t.style.left = `${Math.min(ev.clientX + 14, innerWidth - 280)}px`; t.style.top = `${ev.clientY + 14}px`;
        t.textContent = `${s.p ? `${(s.p * 100).toFixed(1)}% expected · surprise ${surprise.toFixed(1)} bits${clip ? ` (past the colour scale at ${READ_CLIP})` : ''}` : `off scale: not in its top 20 · surprise ${over} bits`}\n${rank} · ${entTxt}\n${exTxt}\nit expected: ${s.top.slice(0, 5).map((x) => `"${show(x.piece)}" ${(x.p * 100).toFixed(0)}%`).join(', ') || 'nothing (it had ended)'}`;
        t.style.whiteSpace = 'pre-wrap';
      });
      sp.addEventListener('mouseleave', () => $('tip').classList.add('hidden'));
      out.appendChild(sp);
    });
    readHist(rows);
    READ_LAST = { no: ++READ_NO, text, first: toks[0].piece, pieces: kept, model: S.model };
    // Two means: the plain one over the pieces in its top 20, and a bounded one that counts each off-scale piece at
    // its floor, so it can only be higher in truth ("≥"). The bounded one leads: the plain one leaves out the most
    // surprising pieces, so new text would look more familiar than it is.
    const bounded = n + nFloor ? (total + floors) / (n + nFloor) : null, plainMean = n ? `${(total / n).toFixed(2)} bits` : 'not measured';
    const avg = off ? `average surprise ≥ ${bounded == null ? '?' : bounded.toFixed(2)} bits, counting each off-scale piece at its floor (${plainMean} over the ${n} piece${n === 1 ? '' : 's'} it ranked in its top 20) · ${off} off scale (hatched)` : `average surprise ${plainMean} (all ${n} in its top 20)`;
    PSI.status($('readStat'), `${sc.length} pieces scored · ${avg}${clipped ? ` · ${clipped} past the colour scale (${READ_CLIP} bits, marked ▴)` : ''}`, off ? 'warn' : 'ok');
    const bos = kvOf('tokenizer.ggml.add_bos_token');
    PSI.cond(rowOf('readGo'), [...localParts(), 'no lens: the text is read as it is',
      { t: bos ? 'no BOS: read without the beginning-of-text token this model expects first' : 'no BOS (this model adds none)', title: 'The text is read as written, with no beginning-of-text token in front: the first piece is not scored, and the next few can read more surprised than in a model-formatted prompt' },
      { t: 'special tokens not parsed', title: 'Text like <|im_end|> is read as ordinary characters, not as the model’s control token' },
      { t: 'surprise measured from its top 20 guesses', est: !!off, title: 'Exact for pieces in its top 20; for the rest only a floor is known' }, speedPart(tps.length ? tps : null)]);
  }, 'read'));

  // ── tokens: the sensing surface ─────────────────────────────────────────
  document.querySelectorAll('[data-read]').forEach((b) => b.addEventListener('click', () => { $('readText').value = b.dataset.read; $('readGo').click(); }));
  document.querySelectorAll('[data-sample]').forEach((b) => b.addEventListener('click', () => { $('tokText').value = b.dataset.sample; $('tokGo').click(); }));
  $('tokGo').addEventListener('click', () => busy(async () => {
    if (!S.model) throw new Error('No model is loaded. Load a model in Settings first.');
    const text = $('tokText').value; if (!text.trim()) { PSI.status($('tokStat'), 'Type or paste some text first.', 'warn'); return; }
    PSI.status($('tokStat'), 'cutting into pieces…', 'busy');
    const toks = await psi.tokenize(text, false).catch(failStat('tokStat', 'Cutting the text into pieces'));
    const out = $('tokOut'); out.textContent = '';
    toks.forEach((t) => { const c = el('span', 'chip', show(t.piece).replace(/ /g, '·').replace(/\n/g, '⏎')); c.appendChild(el('small', null, t.id)); out.appendChild(c); });
    const frags = toks.filter((t) => (t.piece || '').includes('�')).length;
    PSI.status($('tokStat'), `${[...text].length} characters → ${toks.length} pieces (${([...text].length / toks.length).toFixed(2)} characters per piece)${frags ? ` · ${frags} pieces are fragments of a character (▫)` : ''}`, 'ok');
    PSI.cond(rowOf('tokGo'), [{ t: `${modelLabel()}'s tokenizer (llama.cpp, from the GGUF)`, title: S.model || '' }, { t: 'special tokens not parsed', title: 'Text like <|im_end|> is cut as ordinary characters here, not read as the model’s control token' },
      PSI.changesFor('local').length ? 'steering and LoRA do not change how text is cut' : null], 'cut by');
  }, 'tokens'));

  // A thinking model that used every token allowed on its thinking: say so, and offer to run again with twice as many
  // (up to the input's own maximum), so the answer can be reached.
  function thinkingCut(input, n, what, rerun) {
    const max = +input.max || n, more = Math.min(max, Math.max(n * 2, n + 200));
    PSI.warn(`No tokens left for the answer: ${what} and never finished (the thinking block was not closed). ${more > n ? `Add more tokens: ${more.toLocaleString()} gives it room to answer.` : `${n.toLocaleString()} is the most allowed here; turn thinking off for this model, or ask a shorter question.`}`,
      more > n ? { action: { label: `Run again with ${more.toLocaleString()} tokens`, onClick: () => { input.value = more; input.dispatchEvent(new Event('change')); rerun(); } } } : {});
  }
  window.PSIThinkingCut = thinkingCut;

  // ── lens on / off ───────────────────────────────────────────────────────
  // Greedy: the top choice each step, picked in the app from the engine's odds (no draw). Returns why it ended, its
  // speed, and whether a thinking model ever left its think block (if not, the answer was never reached).
  async function greedy(ids, n, outEl, p, done0 = 0) {
    const gen = [], tps = []; outEl.textContent = ''; let stop = 'limit';
    for (let i = 0; i < n; i++) {
      if (S.abort) { stop = 'stopped'; break; }
      if (ids.length + gen.length >= ctxLen()) { stop = 'context'; break; }
      if (p) p.step(done0 + i + 1);
      const d = await psi.next([...ids, ...gen], 1);
      if (d.meta && d.meta.timings && d.meta.timings.predicted_tps != null) tps.push(d.meta.timings.predicted_tps);
      if (d.ended || !d.top.length || S.eos.has(d.top[0].id)) { stop = 'eos'; break; }
      gen.push(d.top[0].id);
      if (i % 6 === 5) outEl.textContent = await psi.detokenize(gen);
    }
    const raw = await psi.detokenize(gen);
    const th = PSI.thinking(ids.text, raw);
    outEl.textContent = raw.replace(/^<think>\s*<\/think>\s*/, '');
    const notReached = th.cut && stop === 'limit';   // every piece went on thinking
    const foot = el('p', 'cmp-stop small' + (notReached || stop === 'context' ? ' odd' : ' muted'),
      `${gen.length} piece${gen.length === 1 ? '' : 's'} · ended: ${STOP[stop]}${notReached ? ' · no tokens left for the answer: all of them went on thinking' : ''}`);
    foot.title = STOP_WHY[stop] + (notReached ? '\nThe model opened a thinking block and never closed it within the pieces allowed: what shows is its thinking, not its answer. Raise “tokens” to reach the answer.' : '');
    outEl.appendChild(foot);
    return { stop, tps, notReached, n: gen.length };
  }
  $('cmpGo').addEventListener('click', () => busy(async () => {
    if (!S.model) throw new Error('No model is loaded. Load a model in Settings first.');
    const n = +$('cmpN').value || 160, q = $('cmpQ').value;
    // With Emergence Machine content off (Settings > Content) the framing compared is the user's own, not the E-level lens.
    const em = !window.PSIEM || PSIEM.on, frame = em ? true : $('cmpFrame').value.trim(), ON = em ? 'lens on' : 'with the framing', OFF = em ? 'lens off' : 'without it';
    if (!frame) throw new Error('Write the framing to compare (the box under the question).');
    const p = progress('cmpGo', plain() ? n : 2 * n);
    const runs = [], sent = [];
    // Both columns come from the same loaded model: if it is changed, "lens off" is not the unchanged model either.
    const cond = () => {
      const tps = runs.flatMap((r) => r.tps), nr = runs.filter((r) => r.notReached).length;
      const c = PSI.cond(rowOf('cmpGo'), [...localParts(), PSI.changesFor('local').length ? { t: '“lens off” is the changed model too', chg: true } : null,
        plain() ? 'no chat format: plain continuation' : thinkWord(), { t: `greedy, drawn in the app (the top choice each step), up to ${n} pieces: repeatable`, term: 'greedy' },
        runs.length ? { t: `ended: ${runs.map((r) => `${r.label} ${STOP[r.stop]}`).join(', ')}` } : null,
        nr ? { t: `answer not reached in ${nr === runs.length && nr > 1 ? 'both' : nr} column${nr === 1 ? '' : 's'}`, warn: true, title: 'A thinking model never left its think block within the pieces allowed' } : null,
        runs.length ? speedPart(tps.length ? tps : null) : null]);
      showExact(c, sent);
    };
    cond();
    try {
      if (plain()) {   // no chat format: there is nowhere to put the lens, so one answer is all there is
        $('cmpOn').textContent = 'This model has no chat format, so the lens (a system message) cannot be given to it: it only continues text. Its one answer is on the right.';
        const ids = await chatIds(q, false); sent.push({ label: 'sent to the model', text: ids.text });
        runs.push({ label: 'the answer', ...(await greedy(ids, n, $('cmpOff'), p)) });
      } else {
        const on = await chatIds(q, frame), offIds = await chatIds(q, false);
        sent.push({ label: `${ON}: sent to the model`, text: on.text }, { label: `${OFF}: sent to the model`, text: offIds.text });
        runs.push({ label: ON, ...(await greedy(on, n, $('cmpOn'), p)) });
        runs.push({ label: OFF, ...(await greedy(offIds, n, $('cmpOff'), p, n)) });
      }
      settle(p, 'done');
    } catch (e) { p.fail('stopped by an error'); throw e; } finally { cond(); }
    const cut = runs.filter((r) => r.notReached);
    if (cut.length && !S.abort) thinkingCut($('cmpN'), n, `${cut.length === runs.length && cut.length > 1 ? 'both columns' : cut.map((r) => `“${r.label}”`).join(' and ')} used all ${n} tokens thinking`, () => $('cmpGo').click());
    if (!S.abort && $('cmpExt') && $('cmpExt').checked && window.PSISettings) await window.PSISettings.compareExternal(q, frame === true ? $('lensText').value : frame, $('cmpExtOut'));
    if (!S.abort) PSI.setDone('compare');
  }, 'compare'));

  // ── steering: a control vector inside the model ─────────────────────────
  // The button names what it will do: steer, change the strength, or take steering off (each reloads the model).
  function cvLabel() {
    const v = +$('cvScale').value, on = S.cv ? S.cv.scale : 0;
    $('cvApply').textContent = !v && on ? 'Remove steering (reloads the model)' : 'Apply (reloads the model)';
  }
  $('cvScale').addEventListener('input', () => { $('cvScaleV').textContent = (+$('cvScale').value).toFixed(1); cvLabel(); });
  PSI.onState(() => cvLabel());
  $('cvPick').addEventListener('click', async () => {
    try { const f = await psi.pickControlVector(); if (f) { S.cvPath = f; $('cvName').textContent = f.split(/[\\/]/).pop(); $('cvApply').disabled = !S.model; } } catch (e) { err(e); }
  });
  $('cvApply').addEventListener('click', async () => {
    if (!S.model || !S.cvPath) return;
    const scale = +$('cvScale').value;
    const layers = [+$('cvFrom').value || 14, +$('cvTo').value || 14];
    await openModel(S.model, scale ? { path: S.cvPath, scale, layers } : null);
  });

  // ── the LoRA switch ─────────────────────────────────────────────────────
  $('loraPick').addEventListener('click', async () => {
    try { const f = await psi.pickControlVector(); if (f) { S.loraPath = f; $('loraName').textContent = fileName(f); await psi.settings({ loraPath: f }).catch((e) => err(e, { text: `The adapter is chosen for now but was not saved for next time: ${PSI.explain(e).text}` })); } } catch (e) { err(e); }
  });
  // An adapter fetched from Hugging Face (ui/hf-pull.js) is chosen as if picked here.
  document.addEventListener('psi-adapter', async (e) => { const f = e.detail && e.detail.path; if (!f) return; S.loraPath = f; $('loraName').textContent = fileName(f); await psi.settings({ loraPath: f }).catch((x) => err(x)); });
  // Reloading for the LoRA keeps the steering that is applied now (it used to drop it silently).
  $('loraOn').addEventListener('change', async () => { if (!S.model) return; if ($('loraOn').checked && !S.loraPath) { $('loraOn').checked = false; return err(new Error('Choose the adapter file first.')); } await openModel(S.model, S.cv); });

  // ── the file itself (any size: header only) ─────────────────────────────
  function showFile() {
    const g = S.header, kv = g.kv, a = kv['general.architecture'], n = (k) => kv[`${a}.${k}`];
    // A value the header does not carry is "absent from header", never "?" or blank (gguf-header.js reads the types).
    const ABS = 'absent from header', or = (v, f = (x) => x) => (v == null ? ABS : f(v));
    const types = PSIHeader.byType(g), vocab = Array.isArray(kv['tokenizer.ggml.tokens']) ? kv['tokenizer.ggml.tokens'].length : null;
    const facts = [['name', PSIHeader.displayName(kv, g.name)], ['architecture', or(a)], ['parameters', g.params.toLocaleString()], ['layers', or(n('block_count'))],
      ['width', or(n('embedding_length'))], ['attention heads', or(n('attention.head_count'))], ['context', or(n('context_length'), (v) => v.toLocaleString())],
      ['vocabulary', or(vocab, (v) => v.toLocaleString())],
      ['file', `${(g.bytes / 2 ** 30).toFixed(2)} GB, GGUF v${g.version}`], ['quantisation recipe', or(PSIHeader.fileTypeName(kv['general.file_type']))],
      ['weights', types.map((t) => `${t.tensors} ${t.name}`).join(', ')], ['licence', kv['general.license'] || ABS]];
    const F = $('facts'); F.textContent = '';
    facts.forEach(([k, v]) => { const d = el('div'); d.appendChild(el('b', null, k)); d.appendChild(document.createTextNode(String(v ?? ABS))); F.appendChild(d); });
    const tt = $('fileTypes') && $('fileTypes').querySelector('tbody');
    if (tt) {
      tt.textContent = '';
      types.forEach((t) => { const tr = el('tr'); [t.name, t.tensors.toLocaleString(), t.params.toLocaleString(), t.bytes == null ? 'unknown: this reader has no block size for this type' : `${(t.bytes / 2 ** 20).toFixed(1)} MB`, t.bitsPerParam == null ? 'unknown' : t.bitsPerParam.toFixed(2)].forEach((v) => tr.appendChild(el('td', null, v))); tt.appendChild(tr); });
      $('fileTypesWrap').classList.remove('hidden');
    }
    const tb = $('meta').querySelector('tbody'); tb.textContent = '';
    Object.keys(kv).forEach((k) => {
      const v = kv[k]; let s = Array.isArray(v) ? `[${v.length.toLocaleString()} items] ${v.slice(0, 10).map((x) => JSON.stringify(x)).join(', ')}${v.length > 10 ? ', …' : ''}` : String(v).slice(0, 400);
      if (k === 'general.file_type') s += ` = ${PSIHeader.fileTypeName(v)}`;
      const tr = el('tr'); tr.appendChild(el('td', null, k)); tr.appendChild(el('td', null, s)); tb.appendChild(tr);
    });
    $('fileNote').textContent = 'Read from the file header only; the weights stay on disk for the engine. Bytes per type are computed from each type’s block size, not measured on disk.';
    $('metaWrap').classList.remove('hidden');
    // What the views say about "the model" comes from here (layer counts, size), not from one model's numbers.
    window.PSIModelMeta = { name: PSIHeader.displayName(kv, g.name), arch: a, layers: +n('block_count') || null, params: g.params || null, heads: +n('attention.head_count') || null, width: +n('embedding_length') || null, vocab };
    document.dispatchEvent(new CustomEvent('psi-model-meta', { detail: window.PSIModelMeta }));
  }
  $('fileChoose').addEventListener('click', () => PSI.goto('settings', 'stModelPick'));

  // The buttons of these views wait for a loaded model, and say so (PSI.requires).
  ['genGo', 'genMore', 'readGo', 'tokGo', 'cmpGo', 'forkGo', 'basinGo'].forEach((id) => PSI.requires($(id), 'model'));

  // Reopen the last model on launch, unless that would crowd a long GPU job (a ladder run, training) or not fit:
  // a model loaded beside them spills into system memory and slows both, silently. Then say why, and offer it.
  async function reopen(file) {
    // Put to sleep for being idle (src/idle.js)? Opening the page is not use: say so and offer to wake it, so a tab
    // that reloads by itself (a phone switching back to it) never takes the GPU again on its own.
    const nap = psi.idle ? await psi.idle({}).catch(() => null) : null;
    if (nap && nap.asleep && !nap.running) {
      const name = fileName(file), mn = $('modelName'), a = nap.asleep;
      PSI.setState({ model: null, known: true });
      mn.classList.remove('failed'); mn.textContent = 'asleep · ';
      mn.title = `${name} was put to sleep at ${new Date(a.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} (${a.why})${a.freedMB != null ? `, giving back ${(a.freedMB / 1024).toFixed(1)} GB of GPU memory` : ''}. Wake loads it again.`;
      const go = document.createElement('a'); go.href = '#'; go.className = 'psi-link'; go.textContent = 'Wake';
      go.title = `Load ${name} again`;
      go.addEventListener('click', (e) => { e.preventDefault(); openModel(file); });
      mn.appendChild(go); return;
    }
    // The engine may still hold this very model (the page was reloaded): its GPU memory is not "in the way" of itself.
    const es = await psi.engineState().catch(() => null);
    const same = (a, b) => String(a || '').replace(/\\/g, '/').toLowerCase() === String(b || '').replace(/\\/g, '/').toLowerCase();
    if (es && es.running && same(es.model, file)) {
      const o = es.options || {};
      return openModel(file, o.controlVector || null, true);
    }
    const why = window.PSIMachine ? await window.PSIMachine.holdBack(file).catch(() => null) : null;
    if (!why) return openModel(file);
    const name = String(file).split(/[\\/]/).pop(), mn = $('modelName');
    PSI.setState({ model: null, known: true });
    // Short in the header (it shares one row with the machine meters); the full reason is the tooltip.
    mn.classList.remove('failed'); mn.textContent = `not loaded · ${/fit|free/.test(why) ? 'too big for free GPU' : 'GPU busy'} · `;
    mn.title = `${name} was not reopened: ${why}. Loading it beside that job would spill into system memory and slow both.`;
    const go = document.createElement('a'); go.href = '#'; go.className = 'psi-link'; go.textContent = 'Load anyway';
    go.title = `Load ${name} anyway`;
    go.addEventListener('click', (e) => { e.preventDefault(); openModel(file); });
    mn.appendChild(go);
  }
  psi.settings().then((st) => { if (st.loraPath) { S.loraPath = st.loraPath; $('loraName').textContent = st.loraPath.split(/[\/]/).pop(); } if (st.lastModel) reopen(st.lastModel); else PSI.setState({ known: true }); })
    .catch((e) => { PSI.setState({ known: true }); err(e, { text: `The app's settings could not be read, so the last used model was not reopened: ${PSI.explain(e).text}` }); });
  drawDice(null);
  forkControls();
})();
