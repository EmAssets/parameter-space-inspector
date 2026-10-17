// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
/*
 * Where the roads go (Generate, below the die; once its own view, Paths & attractors): the Emergence Parameters
 * pillar, made measurable.
 *
 *  - level(text): a lexical E-level reading of a short text, from the lexicon built
 *    out of the site's labelled street sentences (two thirds of a sentence's level
 *    is in its words). Returns {e, score, votes}.
 *  - fork analysis: at one step, sample continuations from the road taken and from
 *    an alternative; report each road's E-level and what the choice closed.
 *  - loops(steps): attractors already visible in the generated text (repeated
 *    n-grams with entropy collapsing).
 *  - basins(ids): sample many continuations from the same point, cluster them,
 *    and find the escape temperature (where the main basin loses its majority).
 */
window.PSIPaths = (() => {
  'use strict';
  const LEX = window.ELEVEL_LEXICON || { levels: [], base: {}, words: {} };
  const NAMES = ['fields', 'particles', 'nuclei', 'chemistry', 'complex molecules', 'cells', 'organisms', 'ecosystems', 'minds',
    'cultures', 'institutions', 'organizations', 'states', 'civilizations', 'planetary'];
  const words = (t) => (String(t).toLowerCase().match(/[a-z][a-z'-]{3,}/g) || []);
  // The Emergence Lexicon (emergencemachine.com/tools/lexicon): term -> concept with its derivation chain.
  const LC = window.LEXICON_COMPACT || { terms: {} };
  const GENERIC = new Set(['below', 'above', 'single', 'live', 'subject', 'other', 'within', 'without', 'thing', 'things', 'place', 'part', 'level', 'levels', 'exist', 'exists', 'floor', 'floors', 'make', 'makes', 'possible', 'emerge', 'emerges', 'between', 'through', 'where', 'there', 'their', 'these', 'those', 'because', 'something', 'anything', 'everything',
    'good', 'instead', 'multiple', 'several', 'open', 'often', 'way', 'ways', 'set', 'sets', 'state', 'states', 'many', 'much', 'more', 'most', 'less', 'least', 'same', 'different', 'certain', 'various', 'particular', 'general', 'simple', 'complex', 'specific', 'common', 'whole', 'individual', 'given', 'based']);
  // Senses that describe form rather than a thing in the world.
  const STRUCTURAL = new Set(['temporal', 'spatial', 'threshold', 'state', 'quantity', 'condition', 'size', 'visual', 'color', 'drift', 'margin', 'excess', 'stacked', 'path', 'movement', 'method', 'fix', 'general', 'location', 'action', 'upkeep', 'capacity', 'quality', 'degree', 'manner', 'logical', 'evaluative', 'relation', 'order', 'sequence', 'direction', 'frequency', 'intensity']);
  function concepts(text, exclude = new Set()) {
    const seen = new Set(), out = [];
    for (const w of words(text)) {
      const term = (x) => (Object.hasOwn(LC.terms, x) ? LC.terms[x] : null);   // a word like "constructor" is not a term
      const c = term(w) || term(w.replace(/s$/, '')) || term(w.replace(/(ing|ed|es)$/, ''));
      if (!c || seen.has(c.id) || GENERIC.has(w) || GENERIC.has(c.term) || STRUCTURAL.has(c.sense) || (c.level != null && c.level < 6) || exclude.has(w) || exclude.has(c.term)) continue;
      seen.add(c.id); out.push(c);
    }
    return out;
  }
  // The first real fork: the earliest step whose top choice was under `p`.
  const firstFork = (steps, p = 0.6) => steps.findIndex((s) => s.top[0].p < p);

  // A literal "E8" in the text counts as three votes; with the lens on the model is told to write them, so a reading
  // with noE: true (words only) is reported beside it. `votes` is the evidence behind the reading, `eMentions` how many
  // of them were literal E-numbers.
  function level(text, { noE = false } = {}) {
    const tally = {}; let votes = 0, eMentions = 0;
    if (!noE) for (const m of String(text).matchAll(/\bE(1[0-4]|[0-9])\b/g)) { tally[m[1]] = (tally[m[1]] || 0) + 3; votes++; eMentions++; }
    for (const w of new Set(words(text))) {
      const d = Object.hasOwn(LEX.words, w) ? LEX.words[w] : null; if (!d) continue;
      votes++;
      for (const [e, lo] of Object.entries(d)) tally[e] = (tally[e] || 0) + lo;
    }
    if (!votes) return { e: null, score: 0, votes: 0, eMentions };
    const [e, score] = Object.entries(tally).sort((a, b) => b[1] - a[1])[0];
    return { e: +e, name: NAMES[+e], score: +(score / votes).toFixed(2), votes, eMentions, tally };
  }
  const levelsOf = (text) => {   // all levels with positive evidence, strongest first
    const r = level(text); if (!r.tally) return [];
    return Object.entries(r.tally).filter(([, v]) => v > 0.5).sort((a, b) => b[1] - a[1]).map(([e]) => +e);
  };

  // Repeated n-grams in the generated tokens: a rut the model has fallen into.
  function loops(steps, n = 3, times = 3) {
    const ids = steps.map((s) => s.id), seen = {}, hits = [];
    for (let i = 0; i + n <= ids.length; i++) { const k = ids.slice(i, i + n).join(','); (seen[k] = seen[k] || []).push(i); }
    for (const [k, at] of Object.entries(seen)) if (at.length >= times) hits.push({ gram: k, at, text: steps.slice(at[0], at[0] + n).map((s) => s.piece).join('') });
    if (!hits.length) return null;
    const best = hits.sort((a, b) => b.at.length - a.at.length)[0];
    const from = best.at[1];                                             // the rut starts at the first repeat
    const mean = (xs) => (xs.length ? xs.reduce((a, s) => a + s.entropy, 0) / xs.length : null);
    // Doubt per step is a lower bound where the choices read did not cover the odds, so both averages are "at least".
    const after = mean(steps.slice(from)), before = mean(steps.slice(0, from));
    return { text: best.text, repeats: best.at.length, from, entropyAfter: +after.toFixed(2), entropyBefore: before == null ? null : +before.toFixed(2), nBefore: from,
      lowerBound: steps.some((s) => s.exact === false || (s.exact == null && 1 - (s.covered == null ? 1 : s.covered) > 0.02)),
      mark: new Set(best.at.flatMap((i) => Array.from({ length: n }, (_, k) => i + k))) };
  }

  // Cluster sampled continuations by their opening words (the first ~8 content words).
  function cluster(samples, span = 8) {
    const key = (t) => words(t).slice(0, span);
    const groups = [];
    for (const s of samples) {
      const k = key(s.text);
      let g = groups.find((x) => { const a = new Set(x.key), b = new Set(k); const inter = [...a].filter((w) => b.has(w)).length; return inter / Math.max(1, Math.min(a.size, b.size)) >= 0.6; });
      if (!g) { g = { key: k, members: [] }; groups.push(g); }
      g.members.push(s);
    }
    groups.sort((a, b) => b.members.length - a.members.length);
    return groups.map((g) => { const txt = g.members.map((m) => m.text).join(' '); return { share: g.members.length / samples.length, n: g.members.length, example: g.members[0].text, level: level(txt), levelNoE: level(txt, { noE: true }) }; });
  }

  // The sampling each analysis uses, exported so a result can name its own conditions. The server (llama-server)
  // draws these roads itself with fixed seeds: seeded, usually repeatable (GPU arithmetic can differ between runs).
  // topP is the engine's default for complete() (src/engine.js); top-k and min-p are whatever the server reports back.
  const SEED0 = 1000;
  const FORK = { K: 8, n: 30, temperature: 0.8, topP: 0.95 }, BASIN = { K: 16, n: 36, temps: [0.3, 0.6, 0.9, 1.2], topP: 0.95 };
  // Sample K continuations from `ids` at `temperature`. Each sample keeps the server's `meta` (why it stopped, the
  // sampler it used, its speed) when the engine sends one.
  async function sampleMany(ids, K, temperature, n) {
    const out = [];
    for (let i = 0; i < K; i++) { const r = await psi.complete(ids, { n, temperature, seed: SEED0 + i }); out.push(r); }
    return out;
  }
  // The conditions a set of samples was made under, from their meta: {sampler, stops, tps, seeds}; null fields where
  // the engine did not say. `sampler` is the first sample's settings, `settingsVary` true if any later one differs.
  function metaOf(samples) {
    const metas = samples.map((s) => s && s.meta).filter(Boolean);
    if (!metas.length) return { sampler: null, stops: null, tps: [], seeds: [], settingsVary: false, reported: 0, of: samples.length };
    const stops = {}; metas.forEach((m) => { const k = m.stop || 'unknown'; stops[k] = (stops[k] || 0) + 1; });
    // Temperature is what an analysis varies on purpose (basins), so it is listed, not counted as a variation.
    const key = (s) => JSON.stringify([s.top_p, s.top_k, s.min_p]);
    const first = metas[0].settings || null;
    return { sampler: first, stops, tps: metas.map((m) => m.timings && m.timings.predicted_tps).filter((v) => v != null),
      seeds: metas.map((m) => m.settings && m.settings.seed).filter((v) => v != null), temps: [...new Set(metas.map((m) => m.settings && m.settings.temperature).filter((v) => v != null))], settingsVary: first ? metas.some((m) => m.settings && key(m.settings) !== key(first)) : false, reported: metas.length, of: samples.length };
  }

  // Basins from a point, and how hot it must get before the main basin loses its majority.
  async function basins(ids, { K = BASIN.K, n = BASIN.n, temps = BASIN.temps, onProgress = () => {} } = {}) {
    const byT = {}, all = [];
    let escape = null;
    for (const T of temps) {
      onProgress(`sampling ${K} roads at creativity ${T.toFixed(1)}…`);
      const s = await sampleMany(ids, K, T, n); all.push(...s);
      const c = cluster(s);
      byT[T] = { clusters: c, distinct: c.length, mainShare: c[0] ? c[0].share : 0 };
      if (escape == null && c[0] && c[0].share < 0.5) escape = T;
    }
    // A share's sampling noise: ±1 SE of a share near 50% with K roads (K=16 → ±12 points). An escape whose shares
    // sit within that of 50% is within noise, not a finding.
    const se = Math.sqrt(0.25 / K);
    return { byT, escape, temps, se, meta: metaOf(all) };
  }

  // Split-half control for a road: the features (levels, common words, concepts) present in one half of its roads and
  // absent from the other half. Anything a fork verdict names that also differs between halves is within noise.
  function halfDiff(samples, inPrompt) {
    const h = Math.floor(samples.length / 2), A = samples.slice(0, h), B = samples.slice(h);
    const feat = (ss) => {
      const wc = {}; ss.forEach((s) => new Set(words(s.text)).forEach((w) => { wc[w] = (wc[w] || 0) + 1; }));
      return { levels: new Set(ss.flatMap((s) => levelsOf(s.text))), words: new Set(ss.flatMap((s) => words(s.text))),
        common: new Set(Object.entries(wc).filter(([, c]) => c >= Math.ceil(ss.length / 2)).map(([w]) => w)), concepts: new Set(concepts(ss.map((s) => s.text).join(' '), inPrompt).map((c) => c.id)) };
    };
    const a = feat(A), b = feat(B), sym = (x, y) => new Set([...x].filter((v) => !y.has(v)).concat([...y].filter((v) => !x.has(v))));
    return { levels: sym(a.levels, b.levels), words: new Set([...a.common].filter((w) => !b.words.has(w)).concat([...b.common].filter((w) => !a.words.has(w)))), concepts: sym(a.concepts, b.concepts), halves: [A.length, B.length] };
  }

  // What a choice at step j closed: continuations from the road taken vs an alternative.
  async function fork(promptIds, steps, j, altIndex, { K = FORK.K, n = FORK.n, temperature = FORK.temperature, onProgress = () => {}, promptText = '' } = {}) {
    const before = [...promptIds, ...steps.slice(0, j).map((s) => s.id)];
    const inPrompt = new Set(words(promptText));
    const roads = [{ name: 'taken', id: steps[j].id, piece: steps[j].piece }, { name: 'alternative', id: steps[j].top[altIndex].id, piece: steps[j].top[altIndex].piece }];
    for (const r of roads) {
      onProgress(`sampling ${K} continuations after “${r.piece}”…`);
      r.samples = await sampleMany([...before, r.id], K, temperature, n);
      const all = r.samples.map((s) => s.text).join(' ');
      r.level = level(all); r.levelNoE = level(all, { noE: true });
      r.levels = new Set(r.samples.flatMap((s) => levelsOf(s.text)));
      r.words = new Set(r.samples.flatMap((s) => words(s.text)));
      const wc = {}; r.samples.forEach((s) => new Set(words(s.text)).forEach((w) => { wc[w] = (wc[w] || 0) + 1; }));
      r.common = Object.entries(wc).filter(([w, c]) => c >= Math.ceil(K / 2) && !/^(that|this|with|from|which|their|there|these|those|about|into|where|when|what|while|being|have|were|also|more|than|then|them|they|such|other|through|between)$/.test(w)).map(([w]) => w);
      r.clusters = cluster(r.samples);
      r.concepts = concepts(r.samples.map((s) => s.text).join(' '), inPrompt);
      r.conceptIds = new Set(r.concepts.map((c) => c.id));
      r.half = halfDiff(r.samples, inPrompt);
    }
    const [a, b] = roads;
    // The control: a feature that also differs between the two halves of ONE road is within sampling noise.
    const noise = { levels: new Set([...a.half.levels, ...b.half.levels]), words: new Set([...a.half.words, ...b.half.words]), concepts: new Set([...a.half.concepts, ...b.half.concepts]) };
    const mark = (x) => ({ levels: x.levels.map((e) => ({ e, noisy: noise.levels.has(e) })), words: x.words.map((w) => ({ w, noisy: noise.words.has(w) })), concepts: x.concepts.map((c) => ({ ...c, noisy: noise.concepts.has(c.id) })) });
    const closed = mark({ levels: [...b.levels].filter((e) => !a.levels.has(e)), words: b.common.filter((w) => !a.words.has(w)).slice(0, 10), concepts: b.concepts.filter((c) => !a.conceptIds.has(c.id)).slice(0, 8) });
    const kept = mark({ levels: [...a.levels].filter((e) => !b.levels.has(e)), words: a.common.filter((w) => !b.words.has(w)).slice(0, 10), concepts: a.concepts.filter((c) => !b.conceptIds.has(c.id)).slice(0, 8) });
    const count = (x) => [...x.levels, ...x.words, ...x.concepts], all = [...count(closed), ...count(kept)];
    return { roads, closed, kept, names: NAMES, certainty: steps[j].top[steps[j].chosen].p, forced: !!steps[j].forced,
      noise: { total: all.length, noisy: all.filter((i) => i.noisy).length, halves: a.half.halves }, meta: metaOf([...a.samples, ...b.samples]) };
  }

  return { level, levelsOf, loops, cluster, basins, fork, concepts, firstFork, metaOf, NAMES, SEED0, FORK, BASIN };
})();
