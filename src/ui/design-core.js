// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The Experiment designer's core: a plan's shape and its check, and the instruments that run one prompt. Shared by the
// page (design.js) and the app's job runner (src/jobs.js), so a plan runs the same whether a screen watches it or not.
// Nothing here touches the page or the disk: every call to an engine or a tool goes through the run's context
//   ctx = { eng(name, ...args), mcp(name, args), explain(e) → text, insideModel() → name, localModel() → name }.
(function (root, make) { const m = make(); if (typeof module === 'object' && module.exports) module.exports = m; else root.PSIDesignCore = m; })(typeof self !== 'undefined' ? self : this, () => {
  'use strict';
  const show = (s) => String(s == null ? '' : s).replace(/\n/g, '⏎');
  const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);

  // ── the plan's shape ──────────────────────────────────────────────────────
  const MAX_ITEMS = 60, MAX_TEXT = 400, OPENING = 4, NEURONS = 8;
  // Two instruments the plan can use: the Neural net recording (inside engine) and Read's surprise (local engine).
  const INSTRUMENTS = { neural: 'Neural net', read: 'Read', generate: 'Generate', odds: 'Next-token odds', tokens: 'Tokens', words: 'Word map', connector: 'Connectors',
    fight: 'Fight class', sense: 'Sensing surface', learned: 'When was it learned? (Emergence)', understand: 'Understanding tests', knockout: 'Knock-outs (Neural net)', patch: 'Activation patching (Anatomy)',
    tipping: 'Tipping point (cuts the model, restored at the end)', vectors: 'Layer vectors (for probes and similarity in the analysis)', features: 'Features (sparse autoencoder)' };
  const MEASURES = {
    surprise: 'how surprised the model is by each prompt, in bits, piece by piece (Read)',
    values: 'the instrument’s numbers for every prompt, grouped by the factors',
    similarity: 'how alike the outputs are (shared words or neighbours), for each kind of pair (the second factor is the baseline)',
    paired: 'for the two values of the first factor: the difference in surprise, pair by pair (Read)',
    neuron_overlap: 'neuron overlap by layer, for each kind of pair (the second factor is the baseline)',
    shared_neurons: 'neurons every item of a group shares, beside a mixed group of the same size',
    lens_script: 'which script each layer’s reading is in (does the middle pass through another language, such as English or Chinese?)',
    answers: 'the answers, shaded by how sure the model was of each piece',
  };
  const KEYS = ['title', 'hypothesis', 'controls_why', 'instrument', 'factors', 'labels', 'items', 'answer_tokens', 'framing', 'measures'];
  const OF = { neural: ['neuron_overlap', 'shared_neurons', 'lens_script', 'answers'], read: ['surprise', 'paired'] };
  ['generate', 'words', 'features'].forEach((k) => { OF[k] = ['values', 'paired', 'similarity']; });
  ['odds', 'tokens', 'connector', 'fight', 'sense', 'learned', 'understand', 'knockout', 'patch', 'tipping', 'vectors'].forEach((k) => { OF[k] = ['values', 'paired']; });
  // Fields an item needs besides its text, for the instruments that compare two continuations or two prompts.
  const EXTRA = { learned: { good: 80, bad: 80 }, understand: { good: 80, bad: 80 }, patch: { corrupt: 400 }, tipping: { cut: 60 } };
  // A Tipping-point cut, written in an item: none | neurons:0.2 | heads:0.1 | both:0.3 (a share up to 0.9, optionally :random, :most or :least,
  // most/least by activity on the item's prompt) | concept:cricket (delete the neurons that carry a concept).
  const CUT = /^(none|(neurons|heads|both):(0|0?\.\d{1,3})(?::(random|most|least))?(?::s(\d{1,3}))?|concept:([\p{L}\p{N} -]{1,30}))$/u;
  function parseCut(t) {
    const m = String(t || '').trim().match(CUT); if (!m) return null;
    if (m[1] === 'none') return { kind: 'none' };
    if (m[6]) return { kind: 'concept', concept: m[6].trim() };
    const frac = +m[3]; return frac > 0.9 ? null : { kind: m[2], frac, mode: m[4] || 'random', seed: m[5] ? +m[5] : 0 };
  }
  // Text from the plan: control characters and direction overrides out (they could make shown text differ from what runs).
  const clean = (s) => String(s).replace(/[\u0000-\u0008\u000b-\u001f\u007f‪-‮⁦-⁩]/g, '').trim();
  // L: the guardrails in force (config.js GUARD): maxPrompts, maxPromptChars, maxAnswerTokens, maxGenerateTokens.
  function validate(p, L = {}) {
    const MAX_ITEMS = L.maxPrompts || 60, MAX_TEXT = L.maxPromptChars || 400;
    const errs = [], out = {};
    if (!p || typeof p !== 'object' || Array.isArray(p)) return { ok: false, errors: ['the plan must be an object'] };
    const extra = Object.keys(p).filter((k) => !KEYS.includes(k)); if (extra.length) errs.push(`unknown keys: ${extra.join(', ')}`);
    const str = (k, max, req = true) => {
      if (p[k] == null || p[k] === '') { if (req) errs.push(`${k} is required`); return ''; }
      if (typeof p[k] !== 'string') { errs.push(`${k} must be text`); return ''; }
      const s = clean(p[k]); if (s.length > max) errs.push(`${k} is longer than ${max} characters`); return s;
    };
    out.title = str('title', 100); out.hypothesis = str('hypothesis', 400); out.controls_why = str('controls_why', 400);
    const F = p.factors;
    if (!Array.isArray(F) || F.length < 1 || F.length > 2 || F.some((f) => typeof f !== 'string' || !/^[a-z][a-z_]{0,19}$/.test(f)) || new Set(F).size !== F.length) errs.push('factors must be 1 or 2 distinct names in lower case (letters and _), e.g. ["question", "language"]');
    else out.factors = F.slice();
    const items = p.items, inst = INSTRUMENTS[p.instrument] ? p.instrument : 'neural', ex = EXTRA[inst] || {};
    if (!Array.isArray(items) || items.length < 2 || items.length > MAX_ITEMS) errs.push(`items must be a list of 2 to ${MAX_ITEMS}`);
    else if (out.factors) {
      out.items = []; const seen = new Set();
      items.forEach((it, i) => {
        if (!it || typeof it !== 'object' || Array.isArray(it)) { errs.push(`item ${i + 1} is not an object`); return; }
        const bad = Object.keys(it).filter((k) => k !== 'text' && !out.factors.includes(k) && !ex[k]); if (bad.length) errs.push(`item ${i + 1}: unknown keys ${bad.join(', ')}`);
        const t = typeof it.text === 'string' ? clean(it.text) : '';
        if (!t) errs.push(`item ${i + 1}: text is required`); else if (t.length > MAX_TEXT) errs.push(`item ${i + 1}: text is longer than ${MAX_TEXT} characters`);
        const key = [t, ...(out.factors || []).map((f) => it[f])].join('\u0000');
        if (seen.has(key)) errs.push(`item ${i + 1}: the same text with the same factor values twice (a repeat needs its own value, e.g. run: 1, 2, 3)`); seen.add(key);
        const o = { text: t };
        Object.entries(ex).forEach(([k, max]) => { const v = typeof it[k] === 'string' ? clean(it[k]) : ''; if (!v) errs.push(`item ${i + 1}: ${k} is required for the ${inst} instrument`); else if (v.length > max) errs.push(`item ${i + 1}: ${k} is longer than ${max} characters`); else o[k] = v; });
        out.factors.forEach((f) => { const v = it[f]; if (typeof v !== 'string' || !/^[\p{L}\p{N} _.:-]{1,24}$/u.test(v)) errs.push(`item ${i + 1}: ${f} must be a short name (letters, digits, space, _ . : -; up to 24)`); else o[f] = v; });   // ':' too, so a factor can be the cut itself (cut: neurons:0.2)
        out.items.push(o);
      });
    }
    out.labels = {};
    if (p.labels != null) {
      if (typeof p.labels !== 'object' || Array.isArray(p.labels)) errs.push('labels must map a factor value to a display name');
      else Object.entries(p.labels).forEach(([k, v]) => { if (typeof v !== 'string' || clean(v).length > 60 || !clean(v)) errs.push(`label for ${k}: up to 60 characters of text`); else out.labels[k] = clean(v); });
    }
    const n = p.answer_tokens == null ? (inst === 'generate' ? 60 : 12) : p.answer_tokens, nMax = inst === 'generate' ? (L.maxGenerateTokens || 200) : (L.maxAnswerTokens || 32);
    if (!Number.isInteger(n) || n < 4 || n > nMax) errs.push(`answer_tokens must be a whole number from 4 to ${nMax}`); else out.answer_tokens = n;
    if (p.framing != null && typeof p.framing !== 'boolean') errs.push('framing must be true or false'); out.framing = p.framing === true;
    out.instrument = p.instrument == null ? 'neural' : p.instrument;
    if (!INSTRUMENTS[out.instrument]) { errs.push(`instrument must be one of: ${Object.keys(INSTRUMENTS).join(', ')}`); out.instrument = 'neural'; }
    const M = p.measures, ok = OF[out.instrument];
    if (!Array.isArray(M) || !M.length || M.some((m) => !ok.includes(m)) || new Set(M).size !== M.length) errs.push(`measures for the ${out.instrument} instrument must be a list from: ${ok.join(', ')}`);
    else out.measures = M.slice();
    if (inst === 'tipping' && out.items) out.items.forEach((it, i) => { if (it.cut && !parseCut(it.cut)) errs.push(`item ${i + 1}: cut “${it.cut}” is not one of none, neurons:0.2, heads:0.1, both:0.3 (a share up to 0.9, optionally :most or :least, and :s2 for another random draw), concept:<word>`); });
    if (out.measures && out.measures.includes('similarity') && out.factors && out.items) {
      if (out.factors.length < 2) errs.push('similarity needs two factors so it has a baseline (e.g. question and language)');
      else { const [a, b] = out.factors, va = new Set(out.items.map((x) => x[a])), vb = new Set(out.items.map((x) => x[b])); if (va.size < 2 || vb.size < 2 || new Set(out.items.map((x) => `${x[a]}\u0000${x[b]}`)).size !== out.items.length || out.items.length !== va.size * vb.size) errs.push(`similarity needs a crossed design: every ${a} once with every ${b}`); }
    }
    // A paired comparison needs exactly two values of the first factor, each once with every value of the second.
    if (out.measures && out.measures.includes('paired') && out.factors && out.items) {
      const [a, b] = out.factors, va = [...new Set(out.items.map((x) => x[a]))], vb = b ? [...new Set(out.items.map((x) => x[b]))] : [];
      if (out.factors.length < 2) errs.push('paired needs two factors: the first with the two things compared (e.g. truth: true, false), the second naming the pairs (e.g. domain)');
      else if (va.length !== 2) errs.push(`paired compares exactly two values of ${a}; it has ${va.length}`);
      else if (vb.length < 2 || new Set(out.items.map((x) => `${x[a]}\u0000${x[b]}`)).size !== out.items.length || out.items.length !== 2 * vb.length) errs.push(`paired needs every ${b} once with each ${a} (${2 * vb.length} items)`);
    }
    // A baseline is part of the design: overlap means nothing without pairs that differ in the other factor.
    if (out.measures && out.factors && out.items && out.measures.some((m) => m === 'neuron_overlap' || m === 'shared_neurons')) {
      if (out.factors.length < 2) errs.push('neuron_overlap and shared_neurons need two factors (e.g. question and language, with an unrelated control question) so the overlap has a baseline');
      else {
        const [a, b] = out.factors, va = [...new Set(out.items.map((x) => x[a]))], vb = [...new Set(out.items.map((x) => x[b]))];
        const combos = new Set(out.items.map((x) => `${x[a]}\u0000${x[b]}`));
        if (va.length < 2 || vb.length < 2) errs.push(`each factor needs at least 2 values (${a}: ${va.length}, ${b}: ${vb.length})`);
        else if (combos.size !== out.items.length || combos.size !== va.length * vb.length) errs.push(`the design must be crossed: every ${a} once with every ${b} (${va.length} × ${vb.length} = ${va.length * vb.length} items, each combination once)`);
      }
    }
    return errs.length ? { ok: false, errors: errs } : { ok: true, plan: out };
  }


  // ── the instruments: one prompt each ─────────────────────────────────────
  // A tool's reply over MCP (the app's own tool calls, as Prometheus and agents make them), as {result, conditions}.
  function toolReply(r) {
    if (r.error) throw new Error(r.error.message);
    const t = (r.result.content || []).map((x) => x.text).join('\n'); if (r.result.isError) throw new Error(t.slice(0, 300));
    const j = JSON.parse(t); if (typeof j.result === 'string') { try { j.result = JSON.parse(j.result); } catch (_) { /* text result */ } }
    return j;
  }
  const wordSet = (txt) => new Set(String(txt || '').toLowerCase().match(/[\p{L}\p{N}]+/gu) || []);
  const first = (r) => (r && Array.isArray(r.items) ? r.items[0] : r);
  const TOOLS = {
    generate: { tool: 'generate', need: 'model', what: 'the local model’s answer (greedy, seed 1)', view: 'generate', fields: [['prompt', (it) => it.text]], press: null,
      noPress: 'Generate’s button samples with the view’s own settings; the plan’s greedy answer comes from the same engine through its tool call',
      args: (it, p) => ({ prompt: it.text, max_tokens: p.answer_tokens, lens: p.framing, temperature: 0 }),
      read: (r) => ({ text: r.answer, set: wordSet(r.answer), metrics: { words: (String(r.answer).match(/\S+/g) || []).length } }), primary: 'words', unit: 'words' },
    odds: { tool: 'next_token_odds', need: 'model', what: 'how torn the local model is about the next piece after the text (entropy) and its top guesses', view: null,
      args: (it) => ({ text: it.text, k: 20 }),
      read: (r) => ({ text: (r.top || []).slice(0, 3).map((x) => `“${show(x.piece)}” ${(x.p * 100).toFixed(0)}%`).join(' · '), metrics: { entropy_bits: r.entropy, top_p: r.top && r.top[0] ? r.top[0].p : null }, atLeast: r.entropy_is_lower_bound ? ['entropy_bits'] : [] }),
      primary: 'entropy_bits', unit: 'bits of entropy' },
    tokens: { tool: 'tokenize', need: 'model', what: 'how many pieces the model splits the text into', view: 'tokens', fields: [['tokText', (it) => it.text]], press: 'tokGo',
      args: (it) => ({ text: it.text }),
      read: (r, it) => ({ text: (r || []).map((x) => x[1]).join('|'), metrics: { pieces: (r || []).length, chars_per_piece: (r || []).length ? it.text.length / r.length : null } }), primary: 'pieces', unit: 'pieces' },
    words: { tool: 'word_map', need: 'engine', what: 'the nearest words in the model’s map of meaning, for each piece', view: 'tokens', fields: [['tokText', (it) => it.text]], press: 'embGo',
      run: async (it, p, ctx) => ({ result: await ctx.eng('embed', it.text, 8) }),   // the engine call itself: the tool's reply also carries the whole word map and is cut at 60,000 characters
      read: (r) => { const ts = (r && r.tokens) || []; const nb = ts.flatMap((x) => (x.neighbors || []).map((n) => String(n[0]).trim().toLowerCase())); return { text: ts.map((x) => `${show(x.piece)} → ${(x.neighbors || []).slice(0, 3).map((n) => show(n[0])).join(', ')}`).join(' · '), set: new Set(nb), metrics: { pieces: ts.length } }; },
      primary: 'pieces', unit: 'pieces' },
    connector: { tool: 'connector_meter', need: 'engine', what: 'where the text falls on the model’s connection axis (0 a thing on its own, 1 an average connection)', view: 'connectors', fields: [['cnText', (it) => it.text]], press: 'cnGo',
      args: (it) => ({ text: it.text }), read: (r) => ({ metrics: { position: first(r).position } }), primary: 'position', unit: 'connection-axis units' },
    fight: { tool: 'fight_class', need: 'engine', what: 'the fight-class probe’s reading of a described competition (A Field … F Worldmaker)', view: 'fight', fields: [['fcText', (it) => it.text]], press: 'fcGo',
      args: (it) => ({ text: it.text }),
      read: (r) => { const x = first(r), sc = x.scores || []; return { text: `${x.top} ${x.name || ''}`, cat: x.top, metrics: { top_share: sc[0] ? sc[0].share : null, ...Object.fromEntries(sc.map((c) => [`share_${c.class}`, c.share])) } }; },
      primary: 'top_share', unit: 'share of the top class (a display share, not a confidence)' },
    sense: { tool: 'sensing_surface', need: 'engine', what: 'structural surprise: how much the text breaks the model’s picture of the world, beyond plain surprise', view: 'sensing', fields: [['ssText', (it) => it.text]], press: 'ssGo',
      args: (it) => ({ action: 'surprise', text: it.text }),
      read: (r) => { const x = (r.sentences || [])[0] || {}; return { text: x.peak_piece != null ? `peak at “${show(x.peak_piece)}” (expected ${(x.peak_expected || []).slice(0, 3).map(show).join(', ')})` : '', metrics: { structural: x.structural, raw_bits: x.raw_bits } }; },
      primary: 'structural', unit: 'structural surprise' },
    learned: { tool: 'checkpoint_when_learned', need: 'checkpoints', what: 'at which training step Pythia came to prefer the good continuation over the bad one', slow: 'about 5 minutes per prompt (12 training steps, each loaded from disk)',
      view: 'emergence', fields: [['emTestPrompt', (it) => it.text], ['emTestGood', (it) => it.good], ['emTestBad', (it) => it.bad]], press: null,
      noPress: 'the view’s “Find the step” runs the same test; pressing it as well would double minutes of work, so the tool call runs it once',
      args: (it) => ({ prompt: it.text, good: [it.good], bad: [it.bad] }),
      read: (r) => ({ text: r.never ? 'not learned by the last step scored' : r.learned ? `learned after step ${r.learned.after_step == null ? 0 : r.learned.after_step}, by step ${r.learned.by_step}` : '', metrics: { by_step: r.never || !r.learned ? null : r.learned.by_step }, atMost: ['by_step'] }),
      primary: 'by_step', unit: 'training step (learned by)' },
    understand: { tool: 'understanding_tests', need: 'engine', what: 'how strongly the model prefers the good continuation over the surface-plausible bad one (log-odds), and from which layer', view: 'understand',
      fields: [['undPrompt', (it) => it.text], ['undGood', (it) => it.good], ['undBad', (it) => it.bad]], press: null,
      noPress: 'the view’s “Test it” also saves the test in your list; the tool call scores it without saving',
      args: (it) => ({ items: [{ what: 'experiment', prompt: it.text, good: [it.good], bad: [it.bad] }] }),
      read: (r) => { const x = first(r) || {}; return { text: x.pass ? 'prefers the good continuation' : 'prefers the bad continuation', metrics: { logodds: x.logodds, readable_from: x.readable_from } }; },
      primary: 'logodds', unit: 'nats (log-odds, good over bad)' },
    knockout: { tool: 'knock_out_neurons', need: 'engine', what: 'how much of the next token’s probability lives in the 8 hardest-firing neurons per layer, beside 3 random sets of the same size', view: 'neural', fields: [['nnPrompt', (it) => it.text]], press: null,
      noPress: 'in the view a knock-out needs a recording first; the run records and knocks out in one step, with its random controls',
      // As Neural net does: record the answer, skip the template tokens Qwen writes first (</think>, blank lines), and
      // knock out on the first answer word with those tokens as the prefix; otherwise the target is a template token.
      run: async (it, p, ctx) => {
        const t = await ctx.eng('trace', it.text, { n_new: 4, lens: p.framing }), toks = t.tokens.slice(), dropped = [];
        while (toks.length > 1 && /^\s*(<\/think>|\n)+\s*$/.test(toks[0].piece)) dropped.push(toks.shift());
        const x = toks[0]; if (!x) throw new Error('the model wrote nothing to knock out');
        const r = await ctx.eng('ablate', { prompt: it.text, lens: p.framing, think: false, target: x.id, plan: 'top', k: 8, prefix_ids: dropped.map((y) => y.id), ablation: 'zero', seeds: 3 });
        return { result: r, conditions: r.conditions || t.conditions || null };
      },
      read: (r) => { const c = mean(r.control_p_after || []); return { text: `tracks “${show(r.target)}” · after: “${show(r.now_says)}”`, warn: /^<\/?think>|^<\|/.test(String(r.target)) ? `it tracked a template token (“${r.target}”), not an answer word` : null,
        metrics: { drop: r.p_before - r.p_after, control_drop: c == null ? null : r.p_before - c, p_before: r.p_before } }; },
      primary: 'drop', unit: 'probability lost when the top neurons are off' },
    tipping: { tool: 'tipping_point', need: 'engine', changes: true, view: 'lesion', fields: [], press: null,
      what: 'what survives each cut: understanding tests passed and their mean margin, copying (nats), bits per character on ordinary text (the model’s fixed test set: the same for every prompt at the same cut), and how much of the intact answer to the prompt is kept (the one number that depends on the prompt)',
      noPress: 'its controls set one cut at a time; the run applies each prompt’s cut, scores what survives, and restores the model at the end',
      run: async (it, p, ctx) => {
        const c = parseCut(it.cut); if (!c) throw new Error(`not a cut: ${it.cut}`); let r;
        if (c.kind === 'none') { await ctx.eng('lesion', { action: 'restore' }); r = await ctx.eng('lesion', { action: 'scorecard', questions: [it.text] }); }
        else if (c.kind === 'concept') r = await ctx.eng('lesion', { action: 'concept', concept: c.concept, keep: '', k: 300, questions: [it.text] });
        else r = await ctx.eng('lesion', { action: 'structural', frac_neurons: c.kind === 'heads' ? 0 : c.frac, frac_heads: c.kind === 'neurons' ? 0 : c.frac, mode: c.mode, seed: c.seed, questions: [it.text] });
        return { result: r, conditions: { engine: 'inside engine (PyTorch)', model: ctx.insideModel ? ctx.insideModel() : null } };   // the lesion reply names no model: the app's own record of it
      },
      read: (r, it) => { const a = (r.answers || [])[0] || {}, ret = a.retention || {};
        return { text: `cut ${it.cut} · answer: “${String(a.answer || '').slice(0, 100)}”`, metrics: { tests_passed: r.understanding_passed, mean_margin: r.understanding_margin, copy_nats: r.copy_nats, text_bits: r.text_bits, answer_kept: ret.token_agreement == null ? null : ret.token_agreement } }; },
      primary: 'tests_passed', unit: 'understanding tests passed' },
    features: { tool: 'sae', need: 'engine', view: 'features', fields: [['feText', (it) => it.text]], press: 'feGo',
      what: 'which sparse-autoencoder features fire on the prompt at one layer (the middle one of those with an SAE): the strongest, how many are active, and how much of the layer’s state they leave unexplained; similarity compares the sets of the 20 strongest',
      run: async (it, p, ctx) => {
        const ls = ((await ctx.eng('sae', { action: 'list' })).saes || []).map((x) => x.layer); if (!ls.length) throw new Error('no SAE for this model: train or import one in Features');
        return { result: await ctx.eng('sae', { action: 'encode', text: it.text, layer: ls[Math.floor((ls.length - 1) / 2)], k: 8 }) };
      },
      read: (r) => { const top = r.overall || [], toks = (r.tokens || []).filter((t) => !t.first);
        return { text: `layer ${r.layer}: ${top.slice(0, 3).map(([f]) => `#${f} ${(r.names || {})[f] || ''}`).join(' · ')}`, set: new Set(top.slice(0, 20).map(([f]) => String(f))),
          metrics: { top_act: top[0] ? top[0][1] : null, active_per_piece: toks.length ? mean(toks.map((t) => t.active)) : null, unexplained: r.unexplained } }; },
      primary: 'top_act', unit: 'activation of the strongest feature (summed over the pieces)' },
    vectors: { tool: 'vectors', need: 'engine', view: null, what: 'the inside model’s state after every layer at the prompt’s last piece, projected to 256 numbers per layer by a fixed random matrix (distances and angles kept); the vectors are kept for the analysis: probes, similarity, CKA',
      noPress: 'no view shows these vectors; they go to the analysis under the report',
      run: async (it, p, ctx) => ({ result: await ctx.eng('vectors', { text: it.text, pool: 'last', dims: 256, seed: 0 }) }),
      read: (r) => ({ text: `${r.layers} layers × ${r.dims} numbers (from ${r.hidden}), ${r.pieces} pieces`, metrics: { last_norm: r.norms[r.norms.length - 1], mid_norm: r.norms[Math.floor(r.norms.length / 2)], pieces: r.pieces } }),
      primary: 'last_norm', unit: 'size of the last layer’s state' },
    patch: { tool: 'anatomy', need: 'engine', what: 'activation patching: copy each layer’s state from the clean prompt into the corrupt one, and find where the clean answer comes back (the two prompts must split into the same number of pieces)', view: 'anatomy',
      fields: [['paClean', (it) => it.text], ['paCorrupt', (it) => it.corrupt]], press: 'paRun',
      args: (it) => ({ action: 'patch', clean: it.text, corrupt: it.corrupt }),
      read: (r) => { const g = r.resid || []; let best = { v: -Infinity, l: null }; g.forEach((row, l) => (row || []).forEach((v) => { if (v > best.v) best = { v, l }; }));
        return { text: `clean answer “${show(r.clean_answer)}” (${((r.p_clean || 0) * 100).toFixed(0)}%) · corrupt “${show(r.corrupt_answer)}”`, metrics: { peak_layer: best.l == null ? null : best.l + 1, peak_restored: Number.isFinite(best.v) ? best.v : null } }; },
      primary: 'peak_layer', unit: 'layer where patching restores the most' },
  };
  const explain = (ctx, e) => (ctx.explain ? ctx.explain(e) : String((e && e.message) || e));
  async function runTool(plan, it, ctx) {
    const T = TOOLS[plan.instrument];
    try {
      const j = T.run ? await T.run(it, plan, ctx) : await ctx.mcp(T.tool, T.args(it, plan));
      return { item: it, raw: j.result, conditions: j.conditions || null, ...T.read(j.result, it) };
    } catch (e) {   // one prompt the tool refuses (or that fails) is not measured, with the tool's own reason; the others go on
      const m = explain(ctx, e); return { item: it, error: m, metrics: {}, text: `not measured: ${m}`, warn: `not measured: ${m.slice(0, 160)}` };
    }
  }
  // The background run scores as the Read view does: the text as written (no BOS), the first piece unscored, odds from the
  // model's top 20 guesses; a piece outside them is off scale and only a floor of its surprise is known.
  async function readScore(text, ctx) {
    const toks = await ctx.eng('tokenize', text, false);
    if (toks.length < 2) throw new Error(`“${text.slice(0, 40)}” is one piece: nothing to measure`);
    const sc = await ctx.eng('score', toks.map((t) => t.id), 20);
    return { text, first: toks[0].piece, model: ctx.localModel ? ctx.localModel() : null, pieces: sc.map((x, i) => ({ piece: toks[i + 1].piece, surprise: x.p ? -Math.log2(x.p) : null, floor: x.top.length ? -Math.log2(x.top[x.top.length - 1].p) : null, rank: x.rank == null ? null : x.rank })) };
  }
  // One prompt of a plan, with whichever instrument the plan names.
  const runItem = (plan, it, ctx) => (TOOLS[plan.instrument] ? runTool(plan, it, ctx) : plan.instrument === 'read' ? readScore(it.text, ctx) : ctx.eng('trace', it.text, { n_new: plan.answer_tokens, lens: plan.framing }));
  // What a plan needs: the local engine (model), the inside engine (engine), or Pythia's checkpoints.
  const needOf = (plan) => (TOOLS[plan.instrument] ? TOOLS[plan.instrument].need : plan.instrument === 'read' ? 'model' : 'engine');
  // Before and after a Tipping-point plan: refuse if a cut of the user's is in force (never undo it silently), record
  // the intact answers (for “answer kept”); afterwards restore and ask the engine to confirm. The page and the job
  // runner each mark the model as changed in their own way.
  async function tipBefore(plan, ctx) {
    const st = await ctx.eng('traceStatus').catch(() => null), lz = st && st.conditions && st.conditions.changed && st.conditions.changed.lesion;
    if (lz && lz.kind && lz.kind !== 'none') throw new Error('A Tipping-point cut of yours is already in force on the inside model. Restore it in Tipping point first, so this experiment does not undo it.');
    await ctx.eng('lesion', { action: 'baseline', questions: [...new Set(plan.items.map((x) => x.text))] });
  }
  async function tipAfter(ctx) {
    let ok = false, why = '';
    try { const r = await ctx.eng('lesion', { action: 'restore' }), st = r && r.state; ok = !st || !st.kind || st.kind === 'none'; if (!ok) why = `the engine still reports a ${st.kind} cut`; } catch (e) { why = explain(ctx, e); }
    const msg = ok ? 'Tipping-point experiment done: the inside model was cut for each prompt and is now restored (the engine reports no cut in force).'
      : `Warning: the inside model could not be confirmed restored after the experiment (${why}). Open Tipping point and press “Restore the model” before trusting any other inside result.`;
    return { ok, msg };
  }

  // The name a record is shown under in the history: the plan's title when it is a question, else one made from it.
  function nameFor(plan) {
    const t = String(plan.title || '').trim(); if (/\?$/.test(t)) return t.slice(0, 140);
    const what = TOOLS[plan.instrument] ? TOOLS[plan.instrument].unit : plan.instrument === 'read' ? 'its surprise' : 'the path through the network';
    const [A, B] = plan.factors; return (B ? `Does the ${A} change ${what}, ${B} by ${B}?` : `Does the ${A} change ${what}?`).slice(0, 140);
  }
  // Heavy parts the report never reads (attention rows, the whole word map) are left out of the saved copy.
  const HEAVY = new Set(['attn', 'lens_top3', 'map', 'sketch', 'pts']);
  const slim = (v) => (Array.isArray(v) ? v.map(slim) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).filter(([k]) => !HEAVY.has(k)).map(([k, x]) => [k, slim(x)])) : v);

  return { nameFor, slim, HEAVY, show, mean, clean, MAX_ITEMS, MAX_TEXT, OPENING, NEURONS, INSTRUMENTS, MEASURES, KEYS, OF, EXTRA, CUT, parseCut, validate,
    toolReply, wordSet, first, TOOLS, runTool, readScore, runItem, needOf, tipBefore, tipAfter };
});
