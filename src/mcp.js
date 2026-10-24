// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The Inspector as an MCP server (JSON-RPC 2.0, Streamable HTTP with JSON responses).
// Every capability of the app is a tool, so Prometheus (the in-app AI guide), Claude Code or any other
// agent can drive the same instruments a person uses in the window.
//
//   devserver:  POST http://127.0.0.1:4310/mcp       (behind Cloudflare Access when tunnelled)
//   Electron:   psi.mcp(rpc) over the preload bridge
//
// `call(name, ...args)` is the app's own API table (devserver.js `api`, main.js handlers).
//
// Every tool result is one envelope, so an agent reads a result together with what it was measured under:
//   { result, conditions, changed_state, not_measured: [...], truncated, cut_note? }
// conditions: which engine and model, its settings and the time it was read (never cut);
// changed_state: steering / LoRA on the local model, a Tipping-point cut on the inside model, and anything this call
// itself changed (a model it started, a cut it removed); not_measured: what could not be read, said rather than left out.
'use strict';
const fs = require('fs');
const path = require('path');
let LIMITS = null; try { ({ LIMITS } = require('./lineage-api.js')); } catch (_) { /* the schema falls back to fixed bounds */ }

const VERSION = '1.2.0';
const SITE_MCP = process.env.EM_SITE_MCP || 'https://emergence-chat-r45njbvcdq-uc.a.run.app/mcp';   // the site's own Prometheus MCP server
// Calls to the site say they come from PSI and which version (no ID, nothing about the computer): the site counts them.
let APP_VERSION = '0'; try { APP_VERSION = require('../package.json').version || '0'; } catch (_) { /* counted as version 0 */ }
const SITE_UA = `ParameterSpaceInspector/${APP_VERSION}`;
const MAX_CHARS = 60000;   // a result longer than this is cut and marked (truncated: true); the conditions are never cut

// ── for agents: PSI measures open models, never the caller ────────────────
// Every result carries `analogue` (what was measured, and what may and may not carry over to the agent reading it)
// and `cost` (GPU, money, model changes, files, time). psi_status.for_agents is the short version.
const FOR_AGENTS = 'PSI cannot read your weights or activations. Every inside measurement is of an open model named in conditions: an analogue, not you. Three routes: (1) study the analogue (mechanisms, GPU time); (2) score your own text under an open model (score_my_text: the open model’s surprise, not where you chose); (3) sample yourself through OpenRouter and compare (behaviour only; spends the user’s money under their cap). Saying the right thing about yourself is not access to yourself (one model rated red’s associates correctly, AUC 0.91, while its internal state disagreed, rho −0.13). Your top-1 probability already predicts your correctness as well as any probe found here. An untrained network scored 0.84 on a self-report task: part of any self-report score is the prompt. Start with psi_status; nothing that changes a model or spends money runs without the user’s yes.';
const NO_CHANGES = 'the user has not allowed agents to change models (Settings)';
// Human-written control passages for score_my_text: three neutral paragraphs from WikiText-103 (Wikipedia, CC BY-SA),
// the sample the app's neuron corpus uses, with its tokenizer spacing undone. Wikipedia is in most training sets, so
// an open model may have seen them: a control, not a guaranteed-unseen text.
const CONTROLS = [
  { source: 'Wikipedia via WikiText-103 (CC BY-SA): Fort Scott', text: 'Settlers also had some continuing conflicts with Indians and insurgents using the recent Confederate cause as excuse for robbery. By the spring of 1873, the US Army withdrew the troops from Fort Scott for good. From 1873 to 1965, the buildings of the fort were left unattended, and slowly deteriorated. Many military buildings were razed and replaced with structures built for civilian use.' },
  { source: 'Wikipedia via WikiText-103 (CC BY-SA): Upsilon Andromedae b', text: 'Upsilon Andromedae b appears to be responsible for increased chromospheric activity on its parent star. Observations suggest that there is a "hot spot" on the star around 169° away from the sub-planetary point. This may be the result of interactions between the magnetic fields of the planet and the star. The mechanism may be similar to that responsible for the activity of RS Canum Venaticorum variable stars, or the interaction between Jupiter and its moon Io.' },
  { source: 'Wikipedia via WikiText-103 (CC BY-SA): Business US 2 (Ironwood)', text: 'Business U.S. Highway 2 (Bus. US 2) is a 1.270-mile (2.044 km) business route that runs from the Wisconsin state line at the Montreal River. The route extends through downtown Ironwood on Silver and Aurora streets before turning northward along Suffolk Street. Bus. US 2 stays on Suffolk Street for a short while until it turns onto Frederick Street. On Frederick Street, Bus. US 2 bears north through a residential area along Douglas Street. The eastern terminus of the route is at its junction with US 2 at the corner of Cloverland Drive and Douglas Street north of downtown.' },
];
// Answers compared as text: lower case, no punctuation, single spaces; and by first sentence.
const stripThink = (s) => String(s || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim();
const normText = (s) => stripThink(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
const firstSentence = (s) => normText((stripThink(s).match(/^[\s\S]*?[.!?](\s|$)/) || [stripThink(s)])[0]);
function clusters(texts, key) {
  const m = new Map(); texts.forEach((t, i) => { const k = key(t); if (!m.has(k)) m.set(k, []); m.get(k).push(i); });
  return [...m.entries()].map(([k, idx]) => ({ key: k.slice(0, 160), n: idx.length, members: idx })).sort((a, b) => b.n - a.n);
}
// Wilson score interval for k of n (95%).
function wilson(k, n, z = 1.96) {
  if (!n) return null; const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d, h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return { rate: +p.toFixed(3), ci95: [+Math.max(0, c - h).toFixed(3), +Math.min(1, c + h).toFixed(3)], k, n };
}
const bitsOf = (p) => (p > 0 ? -Math.log2(p) : null);
const r3 = (x) => (x == null ? null : +(+x).toFixed(3));

// What each tool costs, in the four ways a user cares about, plus a rough time. A function gets the arguments, so a
// tool whose actions differ (tipping_point, anatomy, external_models) says what THIS call costs.
const C = (gpu, usd, changes_model, writes_files, seconds, note) => ({ gpu, usd, changes_model, writes_files, seconds, ...(note ? { note } : {}) });
const SPEND_NOTE = 'spends the user’s OpenRouter money under their cap; the spend counter is written to settings.json';
const FREE_NOTE = 'no money: only models on this computer (local/…) or on the user’s ChatGPT plan (chatgpt/…)';
// Only OpenRouter ids cost money: local/<endpoint>/<model> (this computer) and chatgpt/<slug> (the user's plan) do not.
const paid = (models) => (Array.isArray(models) ? models : [models]).some((x) => typeof x === 'string' && x && !/^(local|chatgpt)\//.test(x));
// The tool list describes a call with no arguments yet: it says what the tool may cost, so it counts as paid there.
const listing = (a) => !a || !Object.keys(a).length;
const TIP_RO = ['baseline', 'restore', 'status', 'scorecard', 'concept_find'];
const ANAT_CHANGE = ['concept_knockout', 'concept_erase', 'concept_inject', 'build_examples', 'build_stats'];
// Anatomy actions that only read; any other action (a new one included) counts as a change for an outside agent.
const ANAT_RO = ['structure', 'layer', 'links', 'record', 'zoom', 'neuron', 'examples_status', 'stats_status', 'patch', 'concept', 'color_geometry', 'concept_net'];
const act = (a) => (a && a.action) || '';
const COST = {
  psi_status: C(false, false, false, false, '~2'), seed_lineage: C(false, false, false, false, '~1'),
  seed_signature: C(false, false, false, false, '~10-120', 'loads the .gguf on the CPU (fingerprint.py)'), scorecard: C(false, false, false, false, '~1'),
  seed_grow: (a) => C(!!(a && a.device === 'cuda'), false, false, true, 'minutes', 'trains a new model and writes a new signed .gguf; the parent file is not changed'),
  seed_grow_status: C(false, false, false, false, '~1'), setup_check: C(false, false, false, false, '~2'), guide_search: C(false, false, false, false, '~0'), glossary: C(false, false, false, false, '~0'),
  search_site: C(false, false, false, true, '~3', 'free site key; the first call writes it to .userdata/em-site.key'), site_concept: C(false, false, false, true, '~3', 'free site key; the first call writes it to .userdata/em-site.key'),
  generate: C(true, false, false, false, '~2-60', 'may start the local model (GPU memory) if it is not running; changed_state says so'),
  next_token_odds: C(true, false, false, false, '~1'), read_surprise: C(true, false, false, false, '~0.2 per piece'), tokenize: C(true, false, false, false, '~1'),
  layer_by_layer: C(true, false, false, false, '~5-30'), word_map: C(true, false, false, false, '~2'), understanding_tests: C(true, false, false, false, '~5-60'),
  knock_out_neurons: C(true, false, false, false, '~5-60', 'silences neurons for this request only'), checkpoints_list: C(false, false, false, false, '~1'),
  checkpoint_ask: C(true, false, false, false, '~10-60', 'loads a training checkpoint'), checkpoint_when_learned: C(true, false, false, false, '~30-300', 'loads every saved checkpoint in turn'),
  heads_scan: C(true, false, false, false, '~10-60'), ablate_heads: C(true, false, false, false, '~10-120', 'knocks heads out for this request only'),
  tipping_point: (a) => C(true, false, !TIP_RO.includes(act(a)) && act(a) !== 'save', act(a) === 'save', '~10-300', act(a) === 'save' ? 'writes the cut model as a new folder beside the original (which is not changed); the cut stays in memory' : TIP_RO.includes(act(a)) ? 'reads or restores; changes nothing' : 'the cut STAYS in the in-memory model until lesion_restore'),
  lesion_restore: C(true, false, true, false, '~5-60', 'puts the inside model back as loaded'),
  sensing_surface: C(true, false, false, false, '~5-60'), fight_class: C(true, false, false, false, '~2-10'), connector_meter: C(true, false, false, false, '~2-10'),
  anatomy: (a) => C(true, false, ['concept_knockout', 'concept_erase'].includes(act(a)), act(a) === 'build_examples', '~5-600',
    act(a) === 'build_examples' ? 'writes an examples file' : ANAT_CHANGE.includes(act(a)) ? 'changes the inside model during the request and restores it at the end' : 'read-only action'),
  neuron_names: C(false, false, false, false, '~0'),
  external_models: (a) => C(false, act(a) === 'chat' && paid([a.model]), false, act(a) === 'chat', '~2-60', act(a) === 'chat' ? (paid([a.model]) ? SPEND_NOTE : FREE_NOTE) : 'free: account status, the model list, or the local models’ status'),
  score_my_text: C(true, false, false, false, '~0.3 per token, 3 texts', 'needs a running engine; never starts one'),
  framing_side_by_side: (a) => { const p = listing(a) || paid(a.external || []); return C(true, p, false, true, '~5-120', `${p ? SPEND_NOTE : FREE_NOTE}; the open model column uses the local engine only if it is already running`); },
  self_consistency: (a) => { const p = listing(a) || paid([a.external_model]); return C(true, p, false, true, '~5-200', `${p ? SPEND_NOTE : FREE_NOTE}; the open model column uses the local engine only if it is already running`); },
};
// Calls that change the user's model (or train a new one) need settings.agents.allow_changes when an outside agent asks.
// lesion_restore is an undo and stays open; knock_out_neurons / ablate_heads last for the request only.
const CHANGES = { tipping_point: (a) => !TIP_RO.includes(act(a)), anatomy: (a) => !ANAT_RO.includes(act(a)), seed_grow: () => true };
const NO_SPEND = 'the user has not allowed agents to spend OpenRouter credit (Settings)';

// What an analogue result may and may not tell the agent reading it. you_measured is always false: nothing here reads
// the caller. Per tool, over a default per kind of measurement.
const AN_MODEL = { transfers: 'Mechanisms and failure patterns that recur across transformer language models (how an answer forms through the layers, copying by look-back heads, sensitivity to framing) may apply to you: hypotheses about yourself, to test by your behaviour.',
  does_not_transfer: 'The numbers, layer and neuron indices, probabilities and tokenizer splits belong to the open model named in measured; your architecture, size, training and tokenizer differ. Nothing here read your weights or activations.' };
const AN_STORED = { transfers: 'The pattern of a finding, as a hypothesis.', does_not_transfer: 'The numbers were measured earlier on the models named in the finding: not on you, and not now.' };
const AN_NONE = { transfers: 'Nothing about you: this reads the app or the machine.', does_not_transfer: 'No model was measured.' };
const AN_SELF = { transfers: 'Your behaviour under these prompts and settings: whether framing moves your answer and how often your samples agree (a rate, with its interval). One flip is an anecdote; framing sensitivity is a rate.',
  does_not_transfer: 'Why you answered so: no internal state is read. The provider’s settings (system prompt, reasoning effort, quantisation) may differ from how you run elsewhere; whether a model sampled IS you depends on the exact id and version. The open model’s column is an analogue.' };
const AN = {
  psi_status: { measured: 'no model measured (the state of this machine and its engines)', ...AN_NONE }, setup_check: { measured: 'no model measured (the setup)', ...AN_NONE },
  glossary: { measured: 'stored result', transfers: 'Definitions, which apply to any model.', does_not_transfer: 'No measurement.' },
  guide_search: { measured: 'stored result', ...AN_STORED }, scorecard: { measured: 'stored result', ...AN_STORED },
  neuron_names: { measured: 'stored result', transfers: 'Nothing about you: names are hypotheses about one model’s units.', does_not_transfer: 'The units and their names are that model’s; a direction is not a concept.' },
  search_site: { measured: 'stored result', transfers: 'The framework’s claims, as claims to test.', does_not_transfer: 'No model was measured.' },
  site_concept: { measured: 'stored result', transfers: 'The framework’s claims, as claims to test.', does_not_transfer: 'No model was measured.' },
  seed_lineage: { measured: 'stored result', transfers: 'Nothing about you.', does_not_transfer: 'A signature vouches for origin, not ability.' },
  seed_signature: { transfers: 'Nothing about you.', does_not_transfer: 'Whether this seed recalls its signature says nothing about you. A signature vouches for origin, not ability.' },
  seed_grow: { measured: 'no model measured (a training job)', ...AN_NONE },
  seed_grow_status: { measured: 'stored result', transfers: 'How a small model learns, as a pattern.', does_not_transfer: 'Its numbers are a small seed’s, not yours.' },
  checkpoints_list: { measured: 'stored result', ...AN_NONE },
  generate: { transfers: 'What an open model says to the same question: a comparison for your own answer.', does_not_transfer: 'Its answer is not yours, and its fluency says nothing about your internals.' },
  next_token_odds: { transfers: 'Where text is predictable to language models generally.', does_not_transfer: 'The probabilities are this model’s under its tokenizer: not your confidence.' },
  read_surprise: { transfers: 'Where text is predictable to language models generally.', does_not_transfer: 'The probabilities are this model’s under its tokenizer: not your confidence.' },
  tokenize: { transfers: 'That models see pieces, not letters or words.', does_not_transfer: 'Your tokenizer splits text differently.' },
  layer_by_layer: { transfers: 'That an answer becomes readable in the late layers of transformer models.', does_not_transfer: 'The layers and neurons are this model’s. The logit lens shows when an answer becomes readable at one layer through one readout, not when it was decided.' },
  word_map: { transfers: 'That related words sit near each other in a learned embedding.', does_not_transfer: 'The neighbours and distances are this model’s map. A direction is not a concept.' },
  understanding_tests: { transfers: 'Which kinds of item trip a model up (surface-plausible distractors).', does_not_transfer: 'The margins and the layer where it gets an item are this model’s.' },
  knock_out_neurons: { transfers: 'That an ability can be concentrated or spread out, and that only a same-size random control can say which.', does_not_transfer: 'Which units matter: your units are not these. A direction is not a concept.' },
  ablate_heads: { transfers: 'That copying relies on look-back heads in transformers, judged against a random control.', does_not_transfer: 'Which heads, and how much: this model’s.' },
  tipping_point: { transfers: 'How abilities degrade as parts are removed, as a shape, beside a random control.', does_not_transfer: 'The fractions and the units are this model’s. A direction is not a concept.' },
  lesion_restore: { transfers: 'Nothing about you.', does_not_transfer: 'This restores the open model.' },
  anatomy: { transfers: 'That knowledge has an address that can be found and moved (activation patching), as a method.', does_not_transfer: 'The neurons, bins and layers are this model’s. A direction is not a concept.' },
  checkpoint_ask: { transfers: 'The order in which abilities appear in training, as a pattern seen in Pythia.', does_not_transfer: 'Steps and timing are Pythia’s; you were not trained like it.' },
  checkpoint_when_learned: { transfers: 'The order in which abilities appear in training, as a pattern seen in Pythia.', does_not_transfer: 'Steps and timing are Pythia’s (between two saved checkpoints: a bound); you were not trained like it.' },
  heads_scan: { transfers: 'Look-back (induction) heads are common to transformers.', does_not_transfer: 'Which heads and their scores are this model’s.' },
  sensing_surface: { transfers: 'How much evidence moves an answer, as a rate across items.', does_not_transfer: 'One flip is an anecdote; framing sensitivity is a rate. The numbers are this model’s.' },
  fight_class: { transfers: 'That such a reading can be made from a model’s internal state.', does_not_transfer: 'A probe share is not a confidence, and a direction is not a concept; your internal state was not read.' },
  connector_meter: { transfers: 'That such a reading can be made from a model’s internal state.', does_not_transfer: 'A direction is not a concept; your internal state was not read.' },
  external_models: { transfers: 'The sampled model’s behaviour (chat), or nothing (status, models).', does_not_transfer: 'No internal state is read; whether the model sampled IS you depends on the exact id and version.' },
  score_my_text: { relation: 'your text scored by another model', transfers: 'How predictable your wording is to a similar open model: conventional against unusual phrasing, where a reader-model is surprised.',
    does_not_transfer: 'Where you chose or how sure you were: the surprise is the open model’s, under its tokenizer, not your probabilities.' },
  framing_side_by_side: { measured: 'your sampled output via OpenRouter (behaviour only)', relation: 'your behaviour, if a model sampled is you; the open model column is an analogue', ...AN_SELF },
  self_consistency: { measured: 'your sampled output via OpenRouter (behaviour only)', relation: 'your behaviour, if the model sampled is you; the open model column is an analogue', ...AN_SELF },
};
const precisionOf = (f) => { const m = /(MXFP4|BF16|F16|F32|IQ\d[_A-Z0-9]*|Q\d[_A-Z0-9]*)/i.exec(String(f || '').split(/[\\/]/).pop()); return m ? m[1].toUpperCase() : 'precision not read from the file name'; };
// Where the sampled models ran: every id local/… (this computer), every id chatgpt/… (the user's plan), or OpenRouter.
function modelIds(args) { const a = args || {}; return [a.model, a.external_model, ...(Array.isArray(a.external) ? a.external : [])].filter((x) => typeof x === 'string' && x); }
function sampledVia(ctx) {
  const ids = modelIds(ctx.args);
  if (ids.length && ids.every((x) => /^local\//.test(x))) return 'from a model on this computer (local/…)';
  if (ids.length && ids.every((x) => /^chatgpt\//.test(x))) return 'via your ChatGPT plan';
  return 'via OpenRouter';
}
// The analogue block of one result (see AN).
function analogueOf(t, conditions, ctx) {
  const a = AN[t.name] || {}, eng = ctx.eng || t.eng, base = { ...(['local', 'inside'].includes(eng) ? AN_MODEL : AN_STORED), ...a };
  let measured = (ctx.an && ctx.an.measured) || a.measured;
  if (!measured) {
    if (eng === 'local') measured = `${conditions.model_file || 'the local model (file not reported)'}, ${precisionOf(conditions.model_file)}, llama.cpp`;
    else if (eng === 'inside') measured = conditions.model ? `${conditions.model}, ${conditions.dtype || 'precision not reported'}, PyTorch${conditions.device ? ` on ${conditions.device}` : ''}` : 'the inside engine’s model (not reported)';
    else if (eng === 'file') measured = conditions.file && conditions.file.path ? `${base0(conditions.file.path)}, ${precisionOf(conditions.file.path)}, fingerprint.py on the CPU` : 'a model file (not named)';
    else if (eng === 'external') measured = act(ctx.args) === 'chat' ? `your sampled output ${sampledVia(ctx)} (behaviour only)` : 'no model measured (OpenRouter account or model list)';
    else measured = 'stored result';
  }
  if (eng === 'self') measured = measured.replace('via OpenRouter', sampledVia(ctx));   // local/… and chatgpt/… models never go through OpenRouter
  return { measured, you_measured: false, ...(a.relation ? { relation: a.relation } : {}), transfers: base.transfers, does_not_transfer: base.does_not_transfer, ...(ctx.an && ctx.an.extra ? ctx.an.extra : {}) };
}
const base0 = (f) => (f ? String(f).split(/[\\/]/).filter(Boolean).pop() : f);
const costOf = (t, args) => { const c = COST[t.name]; return typeof c === 'function' ? c(args || {}) : c || C(null, null, null, null, 'not estimated'); };

function loadGuide() {
  try {
    const src = fs.readFileSync(path.join(__dirname, 'ui', 'guide-data.js'), 'utf8');
    const sandbox = { window: {} }; new Function('window', src)(sandbox.window);   // our own file, data only
    return { guide: sandbox.window.PSI_GUIDE || [], glossary: sandbox.window.PSI_GLOSSARY || {} };
  } catch (_) { return { guide: [], glossary: {} }; }
}
function defaultLens() {
  try { const m = fs.readFileSync(path.join(__dirname, 'ui', 'app.js'), 'utf8').match(/const LENS = `([\s\S]*?)`;/); return m ? m[1] : ''; } catch (_) { return ''; }
}
// The live understanding battery, shared with the Understanding view and lesion.py: the value after
// "PSI_UNDERSTANDING_TESTS =" in ui/understanding-tests.js is strict JSON.
function builtinTests() {
  const src = fs.readFileSync(path.join(__dirname, 'ui', 'understanding-tests.js'), 'utf8'), at = src.slice(src.indexOf('PSI_UNDERSTANDING_TESTS ='));
  return JSON.parse(at.slice(at.indexOf('['), at.lastIndexOf(']') + 1));
}

const obj = (props, req = []) => ({ type: 'object', properties: props, required: req, additionalProperties: true });
const S = (d) => ({ type: 'string', description: d }), N = (d) => ({ type: 'number', description: d }), B = (d) => ({ type: 'boolean', description: d });
const ARR = (d, items = { type: 'string' }) => ({ type: 'array', items, description: d });
const base = (f) => (f ? String(f).split(/[\\/]/).filter(Boolean).pop() : f);

// The Emergence Machine site, through its MCP server (free key registered once, kept in .userdata/em-site.key).
async function site(config, name, args) {
  if (config.emOn && !config.emOn()) throw new Error('Emergence Machine content is off (Settings > Content), so the Emergence Machine site is not asked.');
  const kf = path.join(config.DATA, 'em-site.key'); let key = '';
  try { key = fs.readFileSync(kf, 'utf8').trim(); } catch (_) { /* register below */ }
  const post = async (body, k) => { const r = await fetch(SITE_MCP, { method: 'POST', headers: Object.assign({ 'Content-Type': 'application/json', 'User-Agent': SITE_UA }, k ? { Authorization: `Bearer ${k}` } : {}), body: JSON.stringify(body) }); return r.json(); };
  if (!key) { const r = await post({ jsonrpc: '2.0', id: 1, method: 'register', params: { label: 'parameter-space-inspector' } }); key = (r.result && r.result.api_key) || ''; if (!key) throw new Error('Could not get a key from the site.'); fs.mkdirSync(config.DATA, { recursive: true }); fs.writeFileSync(kf, key); }
  const r = await post({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name, arguments: args } }, key);
  if (r.error) throw new Error(r.error.message || 'site error');
  return ((r.result && r.result.content) || []).map((c) => c.text).join(String.fromCharCode(10));
}

function tools(call, config, records = null) {
  const allowChanges = () => !!(config.settings.agents && config.settings.agents.allow_changes === true);
  const allowSpend = () => !!(config.settings.agents && config.settings.agents.allow_spend === true);
  const G = loadGuide();
  const L = LIMITS || {};
  // The local model starts on demand for agents. It starts as the plain file: steering and a LoRA live in the window
  // (its sliders and adapter switch), so a start here says exactly which of the user's saved settings it left out.
  const ready = async (ctx) => {
    const st = await call('engineState'); if (st && st.running) return st;
    const file = config.settings.lastModel, s = config.settings, dropped = [];
    if (!file) throw new Error('The local model is not running and no model file was used before, so nothing was started. Load a model in the window (Generate) first.');
    if (!fs.existsSync(file)) throw new Error(`The local model is not running and the last used file is missing (${file}), so nothing was started. Load a model in the window first.`);
    if (s.steering && s.steering.path && +s.steering.scale) dropped.push(`steering ${s.steering.scale} at layers ${s.steering.from}–${s.steering.to} (${base(s.steering.path)})`);
    if (s.loraPath) dropped.push(`the LoRA adapter ${base(s.loraPath)} (if it was switched on in the window)`);
    await call('start', file, {});
    ctx.changed.started_local_model = { file, with: 'no steering, no LoRA', left_out: dropped.length ? dropped : 'nothing: no steering or LoRA was saved',
      note: 'The local model was not running, so this call started it; it now holds GPU memory. The window may have had other settings.' };
    return call('engineState');
  };
  // A file argument left out: the model the local engine holds, else the last used file, said as such.
  const fileOf = async (file, ctx) => {
    if (file) { ctx.cond.file = { path: file, source: 'given' }; return file; }
    const st = await call('engineState').catch(() => null);
    if (st && st.running && st.model) { ctx.cond.file = { path: st.model, source: 'loaded (the local engine holds it now)' }; return st.model; }
    ctx.cond.file = { path: config.settings.lastModel, source: 'last used file, not loaded (the local engine is not running)' }; return config.settings.lastModel;
  };
  const inside = (api) => async (a) => call(api, a || {});

  // ── helpers for the routes to the agent itself ──
  const engines = async () => {
    const [st, ins] = await Promise.all([call('engineState').catch(() => null), call('traceStatus').catch(() => null)]);
    return { st, ins, localUp: !!(st && st.running), insideUp: !!(ins && ins.ok !== false && ins.conditions) };
  };
  const mb = (n) => `${(n / 1024).toFixed(1)} GB`;
  const sizeMB = (p) => { try { const s = fs.statSync(p); if (s.isFile()) return s.size / 2 ** 20; return fs.readdirSync(p).filter((f) => /\.(safetensors|bin)$/i.test(f)).reduce((x, f) => x + fs.statSync(path.join(p, f)).size / 2 ** 20, 0); } catch (_) { return null; } };
  // Refuse to start an engine, saying what it would start and the GPU memory it needs (the same rule as the window: file size × 1.05 + 1.2 GB).
  async function refuseStart(which, tool) {
    const file = which === 'local' ? config.settings.lastModel : config.settings.trace && config.settings.trace.modelDir;
    const size = file ? sizeMB(file) : null, need = size == null ? null : size * 1.05 + 1200;
    const g = await call('gpu').catch(() => null), free = g && g.ok !== false && g.freeMB != null ? g.freeMB : null;
    const what = which === 'local' ? `the local engine (llama.cpp) with ${base(file) || 'no model file set'}` : `the inside engine (PyTorch) with ${base(file) || 'no weights folder set'}`;
    throw new Error(`${tool} did not start anything: no engine that can do this is running. It would start ${what}, which needs about ${need == null ? 'an unknown amount of (file not readable)' : mb(need)} GPU memory; ${free == null ? 'free GPU memory was not read' : `${mb(free)} is free now`}${need != null && free != null && need > free ? ' (it would not fit: it would spill into system memory and run many times slower)' : ''}. Ask the user to start it in the window (${which === 'local' ? 'Generate loads the local model' : 'Neural net starts the inside engine'}).`);
  }
  // The local engine's answer to messages, as generate writes it (thinking off when the format honours "/no_think").
  async function localPrefix(msgs, st) {
    let prefix = await call('template', msgs), think = 'none added';
    if (st && st.no_think_supported === true && /<\|im_start\|>assistant\n$/.test(prefix)) { prefix += '<think>\n\n</think>\n\n'; think = 'an empty think block (thinking off, as the model answers with "/no_think")'; }
    return { prefix, think };
  }
  async function localAnswer(msgs, st, { n, temperature = 0, seed = 1 }) {
    const { prefix } = await localPrefix(msgs, st);
    const ids = (await call('tokenize', prefix, true)).map((t) => t.id);
    const out = await call('complete', ids, { n, temperature, seed });
    return { text: stripThink(out.text), meta: out.meta };
  }
  // Score continuation ids after prefix ids on the local engine: per piece bits / rank / top 3 / entropy, top-20 bounded.
  async function scoreIds(pre, ids, pieces, ctx) {
    const out = [];
    for (let i = 0; i < ids.length; i++) {
      const d = await call('next', pre.concat(ids.slice(0, i)), 20); if (d && d.meta) ctx.meta = d.meta;
      const top = (d && d.top) || [], hit = top.find((t) => t.id === ids[i]);
      const piece = pieces ? pieces[i] : hit ? hit.piece : null;
      if (!top.length) { out.push({ piece, bits: null, off_scale: 'not measured (the model gave no probabilities: it had ended)' }); continue; }
      const floor = top[top.length - 1].p;
      out.push(hit ? { piece, bits: r3(bitsOf(hit.p)), rank: top.indexOf(hit) + 1, top3: top.slice(0, 3).map((t) => [t.piece, r3(t.p)]), entropy_bits: r3(d.entropy), entropy_is_lower_bound: !!d.entropy_is_lower_bound }
        : { piece, bits: null, bits_at_least: r3(bitsOf(floor)), rank: `>${top.length}`, off_scale: `outside the top ${top.length}: surprisal is at least this (a bound)`, top3: top.slice(0, 3).map((t) => [t.piece, r3(t.p)]), entropy_bits: r3(d.entropy), entropy_is_lower_bound: !!d.entropy_is_lower_bound });
    }
    return out;
  }
  function totals(pieces) {
    const on = pieces.filter((p) => p.bits != null), off = pieces.filter((p) => p.bits_at_least != null), nm = pieces.length - on.length - off.length;
    const sum = on.reduce((x, p) => x + p.bits, 0) + off.reduce((x, p) => x + p.bits_at_least, 0), n = on.length + off.length;
    return { pieces: pieces.length, total_bits: off.length ? null : r3(sum), total_bits_at_least: off.length ? r3(sum) : undefined, mean_bits_per_piece: off.length ? null : n ? r3(sum / n) : null,
      mean_bits_per_piece_at_least: off.length && n ? r3(sum / n) : undefined, off_scale: off.length, not_measured: nm, bound_note: off.length ? `${off.length} piece(s) were off scale (outside the top 20): totals are lower bounds` : undefined };
  }
  const pickControl = (chars) => { const c = CONTROLS.slice().sort((x, y) => Math.abs(x.text.length - chars) - Math.abs(y.text.length - chars))[0]; return c; };

  async function scoreMyText(a, ctx) {
    const context = String(a.context == null ? '' : a.context), text = String(a.text == null ? '' : a.text);
    if (!text.trim()) throw new Error('score_my_text: text is empty; nothing was run.');
    if (context.length > 8000 || text.length > 4000) throw new Error('score_my_text: context is limited to 8000 characters and text to 4000; nothing was run.');
    const E = await engines();
    let choice = a.engine || (E.insideUp ? 'inside' : 'local');
    ctx.cond.engine_choice = a.engine ? 'given' : E.insideUp ? 'default: the inside engine is loaded' : 'default: the inside engine is not loaded, so the local engine';
    if (choice === 'inside') {
      if (!E.insideUp) await refuseStart('inside', 'score_my_text');
      try { return await scoreInside(context, text, ctx); } catch (e) {
        const missing = /not a function|no POST route|404|unknown call/i.test(String(e.message || e));
        if (!missing) throw e;
        if (a.engine === 'inside' || !E.localUp) throw new Error(`score_my_text: the inside engine has no per-token scoring route in this build (POST /score_text), so nothing was scored${a.engine === 'inside' ? '' : ', and the local engine is not running'}. ${E.localUp ? 'Ask for engine "local" (top-20 bounds).' : 'Start the local model in Generate for engine "local".'}`);
        ctx.cond.engine_choice = 'fell back to the local engine: the inside engine has no per-token scoring route (POST /score_text) in this build';
        ctx.nm.push('full-softmax surprisal (the inside engine has no /score_text route in this build; the local engine’s top-20 bounds were used)');
        choice = 'local';
      }
    }
    if (!E.localUp) await refuseStart('local', 'score_my_text');
    return scoreLocal(context, text, ctx, E.st);
  }
  async function scoreLocal(context, text, ctx, st) {
    ctx.eng = 'local';
    const { prefix, think } = await localPrefix([{ role: 'user', content: context }], st);
    const pre = (await call('tokenize', prefix, true)).map((t) => t.id);
    // The text is tokenized on its own (no special tokens parsed), as the model would write it after the template.
    const all = await call('tokenize', text, false), toks = all.slice(0, 200);
    if (all.length > 200) ctx.cut = `scored the first 200 of ${all.length} pieces of the text`;
    const mine = await scoreIds(pre, toks.map((t) => t.id), toks.map((t) => t.piece), ctx);
    const g = await call('complete', pre, { n: toks.length, temperature: 0, seed: 1 });
    const gIds = (g.ids || []).slice(0, toks.length);
    const greedy = gIds.length ? await scoreIds(pre, gIds, null, ctx) : [];
    const ctl = pickControl(text.length), cAll = await call('tokenize', ctl.text, false), cToks = cAll.slice(0, toks.length);
    const control = await scoreIds(pre, cToks.map((t) => t.id), cToks.map((t) => t.piece), ctx);
    ctx.an = { measured: `your text scored by ${base(st.model)} (${precisionOf(st.model)}, llama.cpp)` };
    ctx.cond.scoring = { engine: 'local (llama.cpp)', readout: 'top 20 per piece: a piece outside them is off scale and its surprisal a lower bound; entropy over the top 20 plus the rest as one bucket (a lower bound)',
      template: 'the model’s chat template: context as the user turn, text as the assistant turn', think_block: think, text_tokenized: 'on its own, no special tokens parsed' };
    const T = totals(mine), G = totals(greedy), K = totals(control);
    const m = (x) => (x.mean_bits_per_piece != null ? `${x.mean_bits_per_piece}` : x.mean_bits_per_piece_at_least != null ? `at least ${x.mean_bits_per_piece_at_least}` : 'not measured');
    return { text: { ...T, per_piece: mine },
      baselines: { open_model_greedy: { ...G, text: stripThink(g.text), note: gIds.length < toks.length ? `the open model ended after ${gIds.length} pieces (its answer is shorter than your text): compare bits per piece` : 'the open model’s own answer, greedy, the same length in pieces, scored the same way' },
        human_control: { ...K, source: ctl.source, note: 'a human-written passage, unrelated to the context, scored as the answer the same way; Wikipedia is in most training sets, so the open model may have seen it' } },
      reading: `Bits per piece under ${base(st.model)}: your text ${m(T)}, its own greedy answer ${m(G)}, the human control ${m(K)}. Lower means more predictable to this model, not more correct, and not where you chose.` };
  }
  // Inside engine: full softmax. Needs api.scoreText (POST /score_text on the inside engine); see the report for its contract.
  async function scoreInside(context, text, ctx) {
    const ctl = pickControl(text.length);
    const mine = await call('scoreText', { context, text, top: 3, max_pieces: 200 });
    const greedy = await call('scoreText', { context, greedy: (mine.pieces || []).length || 50, top: 3 });
    const control = await call('scoreText', { context, text: ctl.text, top: 3, max_pieces: (mine.pieces || []).length || 200 });
    ctx.eng = 'inside';
    const c = mine.conditions || {};
    ctx.an = { measured: `your text scored by ${c.model || 'the inside engine’s model'} (${c.dtype || 'precision not reported'}, PyTorch)` };
    ctx.cond.scoring = { engine: 'inside (PyTorch)', readout: 'full softmax: every surprisal and entropy is exact (no bound)', template: 'the model’s chat template: context as the user turn, text as the assistant turn', think_block: mine.think_block || 'not reported' };
    const tot = (r) => totals((r.pieces || []).map((p) => ({ ...p, bits: r3(p.bits) })));
    const T = tot(mine), G = tot(greedy), K = tot(control);
    return { conditions: c, text: { ...T, per_piece: mine.pieces }, baselines: { open_model_greedy: { ...G, text: greedy.text, note: 'the open model’s own answer, greedy, scored the same way' }, human_control: { ...K, source: ctl.source, note: 'a human-written passage, unrelated to the context; Wikipedia is in most training sets' } },
      reading: `Bits per piece: your text ${T.mean_bits_per_piece}, its own greedy answer ${G.mean_bits_per_piece}, the human control ${K.mean_bits_per_piece}. Lower means more predictable to this model, not more correct, and not where you chose.` };
  }

  // Money first: every external call of a run is estimated before the first is sent.
  // OpenRouter ids, chatgpt/<slug> for the user's ChatGPT plan, or local/<endpoint>/<model> on this computer (the model
  // part may hold slashes and tags: local/lmstudio/google/gemma-4-e4b, local/tester/hf.co/owner/repo:Q4_K_M).
  const MODEL_ID = /^(?:[A-Za-z0-9._-]+\/[A-Za-z0-9._:-]+|local\/[a-z0-9][a-z0-9_-]{0,31}\/[A-Za-z0-9._:@+\/-]{1,200})$/;
  async function preflight(jobs, maxUsd, tool, ctx) {
    let usd = 0; const per = [];
    // Free estimates (local, plan) carry no cap figures: only an OpenRouter estimate is kept for the cap check below.
    for (const j of jobs) { const e = await call('external', { action: 'estimate', model: j.model, messages: j.messages, maxTokens: j.maxTokens, temperature: j.temperature, n: j.n }); usd += e.usd; per.push({ model: j.model, n: j.n, usd: e.usd }); if (!e.local && !e.plan) ctx.lastEst = e; }
    const e = ctx.lastEst || {}, est = { usd: +usd.toFixed(5), by_call: per, basis: e.basis, spent_before: e.spent, cap: e.cap, remaining_before: e.remaining };
    ctx.cond.estimate = est;
    if (usd > maxUsd) throw new Error(`${tool} refused before any call: the estimate is $${usd.toFixed(4)}, above max_usd $${maxUsd}. Nothing was sent or spent.`);
    if (e.cap != null && (e.spent || 0) + (e.reserved || 0) + usd > e.cap) throw new Error(`${tool} refused before any call: the estimate $${usd.toFixed(4)} would pass the user’s spending cap ($${(e.spent || 0).toFixed(4)} of $${e.cap} spent). Nothing was sent or spent; the cap is the user’s (Settings).`);
    return est;
  }
  const chatOnce = async (j, ctx) => { const r = await call('external', { action: 'chat', model: j.model, messages: j.messages, maxTokens: j.maxTokens, temperature: j.temperature }); ctx.usd = (ctx.usd || 0) + (+r.cost || 0); if (r.plan) planNote(ctx, j.model, j.temperature, j.maxTokens); if (r.local) localNote(ctx, j.model, r); return r; };
  // A model on this computer: which server answered and with what sampling, so a result says what tested it.
  function localNote(ctx, model, r) {
    ctx.cond.local_models = ctx.cond.local_models || {};
    // Only the tester (its own process, port and model folder) is isolated by construction; another local server is a separate process, but the same weights are not checked.
    if (!ctx.cond.local_models[model]) ctx.cond.local_models[model] = { source: r.cost_source, sampling: r.sampling, isolated_from_model_under_test: r.isolated === true ? true : 'a separate server; whether it holds the same weights as the model under test is not checked' };
  }
  // A ChatGPT-plan model samples at the plan's own temperature and length: said in the conditions and the not-measured list.
  function planNote(ctx, model, temp, maxTok) {
    ctx.cond.plan_models = ctx.cond.plan_models || {};
    if (ctx.cond.plan_models[model]) return;
    ctx.cond.plan_models[model] = { source: 'the user’s ChatGPT plan (no money)', temperature: `the plan’s default; ${temp ?? 'none'} was asked for and could not be set`, max_tokens: `the plan’s default; ${maxTok ?? 'none'} was asked for` };
    ctx.nm.push(`${model} ran on the ChatGPT plan: its temperature and answer length are the plan’s defaults, not the ones requested, so its spread is not comparable at equal temperature`);
  }
  const framed = (framing, prompt) => (!framing ? [{ role: 'user', content: prompt }] : framing.includes('{prompt}') ? [{ role: 'user', content: framing.split('{prompt}').join(prompt) }] : [{ role: 'system', content: framing }, { role: 'user', content: prompt }]);

  async function framingSideBySide(a, ctx) {
    const prompt = String(a.prompt || ''), F = a.framings, X = a.external || [], n = a.samples || 1, maxTok = a.max_tokens || 200, maxUsd = +a.max_usd;
    if (!prompt.trim()) throw new Error('framing_side_by_side: prompt is empty; nothing was run.');
    if (!Array.isArray(F) || !F.length || F.length > 4 || F.some((f) => typeof f !== 'string')) throw new Error('framing_side_by_side: framings must be 1 to 4 strings; nothing was run.');
    if (!Array.isArray(X) || X.length > 3 || X.some((m) => typeof m !== 'string' || !MODEL_ID.test(m))) throw new Error('framing_side_by_side: external must be up to 3 model ids: OpenRouter (vendor/model), chatgpt/<slug> or local/<endpoint>/<model>; nothing was run.');
    if (!(Number.isInteger(n) && n >= 1 && n <= 5)) throw new Error('framing_side_by_side: samples must be 1..5; nothing was run.');
    // Free models only (local/…, chatgpt/…, or none beside the open model): $0 is a valid budget. Any OpenRouter model needs one above 0.
    if (!(maxUsd > 0 || (maxUsd === 0 && !paid(X)))) throw new Error('framing_side_by_side: max_usd is required: above 0 when an OpenRouter model is named, 0 or more otherwise; nothing was run.');
    const temp = a.temperature != null ? +a.temperature : n > 1 ? 0.7 : 0;
    const jobs = []; X.forEach((m) => F.forEach((f, fi) => jobs.push({ model: m, fi, messages: framed(f, prompt), maxTokens: maxTok, temperature: temp, n })));
    const est = X.length ? await preflight(jobs, maxUsd, 'framing_side_by_side', ctx) : { usd: 0, note: 'no external models: nothing to spend' };
    const E = await engines(), rows = [];
    // The open model: greedy on the local engine, only when it is already running (never started here).
    const open = { model: E.localUp ? `open model: ${base(E.st.model)} (local, greedy)` : 'open model (local engine)', cells: [] };
    for (let fi = 0; fi < F.length; fi++) {
      if (!E.localUp) { open.cells.push({ framing: fi + 1, answers: [], not_measured: 'the local engine is not running; nothing was started (load a model in Generate)' }); continue; }
      const r = await localAnswer(framed(F[fi], prompt), E.st, { n: maxTok, temperature: 0, seed: 1 }); if (r.meta) ctx.meta = r.meta; open.cells.push({ framing: fi + 1, answers: [r.text] });
    }
    if (!E.localUp) ctx.nm.push('the open model’s answers (the local engine is not running; nothing was started)');
    rows.push(open);
    const failures = [];
    for (const m of X) {
      const row = { model: m, cells: [] };
      for (let fi = 0; fi < F.length; fi++) {
        const answers = [];
        for (let s = 0; s < n; s++) { try { answers.push(stripThink((await chatOnce({ model: m, messages: framed(F[fi], prompt), maxTokens: maxTok, temperature: temp }, ctx)).text)); } catch (e) { failures.push(`${m}, framing ${fi + 1}, sample ${s + 1}: ${e.message || e}`); if (/spending cap/i.test(String(e.message))) break; } }
        row.cells.push({ framing: fi + 1, answers });
      }
      rows.push(row);
    }
    for (const r of rows) for (const c of r.cells) { if (!c.answers.length) continue; const cl = clusters(c.answers, firstSentence); c.modal_first_sentence = cl[0].key; c.within_cell_agreement = c.answers.length > 1 ? wilson(cl[0].n, c.answers.length) : 'one sample: no agreement to measure'; }
    const across_models = F.map((f, fi) => { const modal = rows.map((r) => r.cells[fi].modal_first_sentence).filter((x) => x != null); if (modal.length < 2) return { framing: fi + 1, agreement: 'fewer than two models answered' }; const cl = clusters(modal, (x) => x); return { framing: fi + 1, models: modal.length, largest_cluster_share: r3(cl[0].n / modal.length) }; });
    const framing_sensitivity = rows.map((r) => { const ms = r.cells.map((c) => c.modal_first_sentence); if (ms[0] == null || ms.filter((x) => x != null).length < 2) return { model: r.model, changed: 'not measured (fewer than two framings answered)' }; const rest = ms.slice(1).filter((x) => x != null); return { model: r.model, changed_from_framing_1: rest.filter((x) => x !== ms[0]).length, of: rest.length }; });
    if (failures.length) ctx.nm.push(`${failures.length} external answer(s) failed: ${failures.slice(0, 3).join('; ')}`);
    ctx.cond.request = { framings: F.map((f, i) => ({ framing: i + 1, as: !f ? 'none' : f.includes('{prompt}') ? 'user-turn template' : 'system message', text: f.slice(0, 200) })), samples: n, temperature_external: temp, open_model: 'greedy (temperature 0, seed 1)', max_tokens: maxTok };
    return { table: rows, agreement: { across_models, framing_sensitivity, method: 'answers compared by normalised first sentence (lower case, no punctuation): a crude measure; read the answers. One flip is an anecdote; framing sensitivity is a rate over many prompts.' },
      spend: { estimate_usd: est.usd, actual_usd: +(ctx.usd || 0).toFixed(5), max_usd: maxUsd } };
  }

  async function selfConsistency(a, ctx) {
    const prompt = String(a.prompt || ''), m = a.external_model, n = a.n, temp = +a.temperature, maxTok = a.max_tokens || 150, maxUsd = +a.max_usd;
    if (!prompt.trim()) throw new Error('self_consistency: prompt is empty; nothing was run.');
    if (typeof m !== 'string' || !MODEL_ID.test(m)) throw new Error('self_consistency: external_model must be a model id: OpenRouter (vendor/model), chatgpt/<slug> or local/<endpoint>/<model>; nothing was run.');
    if (!(Number.isInteger(n) && n >= 2 && n <= 20)) throw new Error('self_consistency: n must be 2..20; nothing was run.');
    if (!(temp > 0 && temp <= 2)) throw new Error('self_consistency: temperature must be above 0 (and at most 2): at 0 every sample is the same answer; nothing was run.');
    if (!(maxUsd > 0 || (maxUsd === 0 && !paid([m])))) throw new Error('self_consistency: max_usd is required: above 0 for an OpenRouter model, 0 or more for a local or plan model; nothing was run.');
    const msgs = [{ role: 'user', content: prompt }];
    const est = await preflight([{ model: m, messages: msgs, maxTokens: maxTok, temperature: temp, n }], maxUsd, 'self_consistency', ctx);
    const answers = [], failures = [];
    for (let i = 0; i < n; i++) { try { answers.push(stripThink((await chatOnce({ model: m, messages: msgs, maxTokens: maxTok, temperature: temp }, ctx)).text)); } catch (e) { failures.push(String(e.message || e)); if (/spending cap/i.test(String(e.message))) break; } }
    const summarise = (xs) => {
      if (!xs.length) return { answers: [], agreement: 'not measured (no answers)' };
      const ex = clusters(xs, normText), fs1 = clusters(xs, firstSentence);
      return { n: xs.length, exact: { clusters: ex.length, largest: ex[0].n, agreement: wilson(ex[0].n, xs.length) }, first_sentence: { clusters: fs1.length, largest: fs1[0].n, agreement: wilson(fs1[0].n, xs.length), modal: fs1[0].key }, answers: xs };
    };
    const E = await engines(); let open;
    if (E.localUp) { const xs = []; for (let s = 1; s <= n; s++) { const r = await localAnswer(msgs, E.st, { n: maxTok, temperature: temp, seed: s }); if (r.meta) ctx.meta = r.meta; xs.push(r.text); } open = { model: `${base(E.st.model)} (local, seeds 1..${n}, temperature ${temp})`, ...summarise(xs) }; }
    else { open = { not_measured: 'the local engine is not running; nothing was started (load a model in Generate)' }; ctx.nm.push('the open model’s run (the local engine is not running; nothing was started)'); }
    if (failures.length) ctx.nm.push(`${failures.length} of ${n} external samples failed: ${failures.slice(0, 2).join('; ')}`);
    ctx.cond.request = { temperature: temp, max_tokens: maxTok, n, clustering: 'exact normalised text (lower case, no punctuation), then first sentence', interval: 'Wilson 95% for the largest cluster’s share; the cluster is chosen after the fact, so read the interval as rough' };
    return { external: { model: m, ...summarise(answers) }, open_model: open, spend: { estimate_usd: est.usd, actual_usd: +(ctx.usd || 0).toFixed(5), max_usd: maxUsd } };
  }
  const T = [
    // ── orientation ──
    { name: 'psi_status', ro: true, eng: 'status', d: 'What is loaded right now and under what conditions: the local model (file, running, its steering/LoRA options, build, context size, flash attention, KV type, GPU layers), the inside engine (model, weights hash, precision, device, any Tipping-point cut in force), the machine (GPU memory, who holds it, spill into system memory), and the settings (never the API key). Whatever could not be read is listed in not_measured.',
      s: obj({}), run: async (a, ctx) => {
        const engine = await call('engineState').catch((e) => { ctx.nm.push(`local engine state (${e.message || e})`); return null; });
        const ins = await call('traceStatus').catch((e) => { ctx.nm.push(`inside engine state (${e.message || e})`); return null; });
        const machine = await call('gpu').catch((e) => { ctx.nm.push(`machine state (${e.message || e})`); return null; });
        if (engine && Array.isArray(engine.recent_log)) engine.recent_log = engine.recent_log.slice(-12);
        if (engine && !engine.running) ctx.nm.push('local engine conditions (it is not running)');
        if (!ins || ins.ok === false) ctx.nm.push('inside engine conditions (it is not running; it starts on the first inside request)');
        if (machine && machine.ok === false) ctx.nm.push(`GPU reading (${machine.reason || 'not read'})`);
        ctx.status = { engine, inside: ins };
        return { for_agents: FOR_AGENTS, agents: { caller: ctx.caller, allow_changes: allowChanges(), allow_changes_note: allowChanges() ? 'the user allows agents to change models (Settings)' : `${NO_CHANGES}: tipping_point cuts, anatomy concept_knockout / concept_erase / build_examples and seed_grow are refused for outside agents` },
          local_engine: engine, inside_engine: ins, machine, settings: (({ dataDir, ...v }) => ({ ...v, external: { ...v.external, keyHint: undefined } }))(config.view()) };   // no folder of the user's or key digits for an agent
      } },
    { name: 'seed_lineage', ro: true, eng: 'file', d: 'A model file’s signed identity and family tree: its fingerprint (SHA-256 of its weights), whether its Ed25519 signature and every ancestor verify, whether The Emergence Machine signed it, its generation, parent, root seed, the street it came from, its data, training and scorecard. Without a file: the model the local engine holds; if it is not running, the last used file, and conditions.file.source says so.',
      s: obj({ file: S('full path to a .gguf; leave out for the loaded model') }), run: async ({ file }, ctx) => call('lineage', { action: 'info', file: await fileOf(file, ctx) }) },
    { name: 'seed_signature', ro: true, eng: 'file', d: 'Ask a model for the signatures carried in its weights: the public birth line How AI Emerges teaches every seed, and the tree mark Grow rehearses behind a prompt only the signing key can make. For each: does it say the signature exactly (it then verifies against The Emergence Machine certificate), and does it recognise it (surprise against 200 decoys signed by other keys, as a z score). Without a file: the loaded model, or the last used file (conditions.file.source says which).',
      s: obj({ file: S('full path to a .gguf; leave out for the loaded model') }), run: async ({ file }, ctx) => call('lineage', { action: 'marks', file: await fileOf(file, ctx) }) },
    { name: 'scorecard', ro: true, eng: 'file', d: 'The scorecard: the same measurements (bits per character, nine understanding tests, copying, look-back heads) on every seed and grown model of this install, and on Pythia across training time and size and Qwen3-4B raw and chat. These are stored rows, measured earlier. conditions.file names the loaded model (or the last used file, said as such) so its row can be found.',
      s: obj({}), run: async (a, ctx) => { await fileOf(null, ctx); return call('lineage', { action: 'scorecard' }); } },
    { name: 'seed_grow', ro: false, eng: 'lineage', d: `Grow a signed seed: wider (width ${((L.width && L.width.choices) || [1, 2]).join(' or ')}) and deeper (+${L.depth ? `${L.depth.min} to ${L.depth.max}` : '0 to 8'} layers) with the same answers, then train it on its street, all streets or a text file. Runs for minutes, writes a new .gguf and uses the CPU or the GPU (device); returns a job id for seed_grow_status. The result is signed as the next generation. Defaults: width ${L.width ? L.width.default : 2}, depth ${L.depth ? L.depth.default : 2}, ${L.steps ? L.steps.default : 1500} steps, data ${L.data ? L.data.default : 'all'}, device ${L.device ? L.device.default : 'cpu'}.`,
      s: obj({ file: S('the parent .gguf (must verify)'), width: { type: 'integer', enum: (L.width && L.width.choices) || [1, 2] }, depth: { type: 'integer', minimum: L.depth ? L.depth.min : 0, maximum: L.depth ? L.depth.max : 8 },
        data: { type: 'string', enum: (L.data && L.data.choices) || ['street', 'all', 'file'] }, textFile: S('for data=file: full path to a .txt'),
        steps: { type: 'integer', minimum: L.steps ? L.steps.min : 100, maximum: L.steps ? L.steps.max : 20000 }, device: { type: 'string', enum: (L.device && L.device.choices) || ['cpu', 'cuda'] } }, ['file']),
      // "anyway" (start on the GPU despite the hold-back check) is the user's call only, never an agent's.
      run: async (a, ctx) => { const { anyway, ...rest } = a; const r = await call('lineage', { action: 'grow', ...rest }); ctx.changed.started_job = 'a grow job: it trains for minutes and writes a new signed .gguf when done'; return r; } },
    { name: 'seed_grow_status', ro: true, eng: 'lineage', d: 'Progress of a seed_grow job: training steps, bits per character, understanding, copying, GPU memory and whether it spilled; the result and the new file when done.',
      s: obj({ job: S('the job id from seed_grow') }, ['job']), run: async ({ job }) => call('lineage', { action: 'job', job }) },
    { name: 'setup_check', ro: true, eng: 'none', d: 'Diagnose the setup when something fails: which model each engine holds (local llama.cpp for Generate; inside-the-model PyTorch for Neural net, Anatomy and the experiments), whether their files exist, whether they hold the same model, problems in plain words, and the fix (the matching weights folder). Changing the setup is done by the user in Settings.',
      s: obj({}), run: async () => call('setupCheck') },
    { name: 'guide_search', ro: true, eng: 'stored', d: 'Search the Inspector’s question map: questions about AI the tool can answer, where in the app each is answered (menu path), what you will see, stored findings from earlier research (not a run on the model loaded now), and where other tools do better. Use this to point a user to the right place.',
      s: obj({ query: S('words to match, e.g. "neuron", "training step", "where is a fact stored"') }, ['query']),
      run: async ({ query }) => { const q = String(query || '').toLowerCase().split(/\s+/).filter(Boolean); const scored = G.guide.map((x) => [q.filter((w) => JSON.stringify(x).toLowerCase().includes(w)).length, x]).filter(([n]) => n).sort((a, b) => b[0] - a[0]); return scored.slice(0, 8).map(([, x]) => { const { f, ...rest } = x; return f ? { ...rest, stored_result: { finding: f, date: x.date || 'not recorded', model: x.model || 'not recorded (only as named in the finding)', label: 'stored result, not a run on the model loaded now' } } : rest; }); } },
    { name: 'glossary', ro: true, eng: 'none', d: 'Plain-language definitions of the terms used in the Inspector (token, logit lens, residual stream, attention head, KV cache, MLP neuron, bin, activation patching, lesion, checkpoint, phase shift, induction head, steering, probe, AUC, E-levels, fight classes, connectors, sensing surface…). Omit term to list all.',
      s: obj({ term: S('the term, or empty for the full list') }),
      run: async ({ term }) => { if (!term) return Object.keys(G.glossary); const k = Object.keys(G.glossary).find((x) => x.toLowerCase() === String(term).toLowerCase()) || Object.keys(G.glossary).find((x) => x.toLowerCase().includes(String(term).toLowerCase())); return k ? { term: k, meaning: G.glossary[k] } : { error: 'not in the glossary', terms: Object.keys(G.glossary) }; } },
    // ── the raw data PSI kept (src/records.js), for PSI Studio and re-analysis ──
    { name: 'records_list', ro: true, eng: 'none', d: 'The raw data PSI kept: every result a view returned and every tool call made through MCP, newest first (format psi-record 1). Each row: id, time, source (view, prometheus or agent), view, call, size, and a title (the prompt or question). Filter by call, source, view, since (ISO time) or text. These are stored results from earlier runs, not a run now.',
      s: obj({ call: S('only this call or tool, e.g. override or override_map'), source: S('view, prometheus or agent'), view: S('only this view'), since: S('ISO time'), text: S('words in the title'), offset: N('default 0'), limit: N('default 50, at most 500') }),
      run: async (a) => { if (!records) throw new Error('records are not kept by this build'); return records.list(a); } },
    { name: 'record_read', ro: true, eng: 'none', d: 'One kept result in full (an id from records_list): the call, its arguments, the time, the source, and the value exactly as it was returned then, with its conditions. A stored result, not a run now.',
      s: obj({ id: S('the id from records_list') }, ['id']),
      run: async ({ id }) => { if (!records) throw new Error('records are not kept by this build'); return records.read(id); } },

    // ── the Emergence Machine site (the framework the Inspector tests) ──
    { name: 'search_site', ro: true, eng: 'site', d: 'Search the Emergence Machine essays (emergencemachine.com) and get the most relevant passages, each with its real Source URL. Use it to connect a result in the Inspector to the framework, or to answer what an essay says. Only cite URLs that appear on Source lines. Passages are claims and stored findings, not a run on the model loaded now.',
      s: obj({ query: S('what to look for, e.g. "language as a meta-connector", "Autopilot vs Strategist", "absorption anomaly abstraction"') }, ['query']), run: async ({ query }) => site(config, 'search_framework', { query }) },
    { name: 'site_concept', ro: true, eng: 'site', d: 'Look up a concept in the EmergenceLexicon (the site’s provisional map of 10,000+ concepts built up from space, time, energy and pattern): definition, prerequisites, level. Levels are a working assignment, not fixed truth.',
      s: obj({ word: S('a concept, e.g. "market", "cell", "trust"') }, ['word']), run: async ({ word }) => site(config, 'get_concept', { word }) },

    // ── the local model answering (llama.cpp, the GGUF file) ──
    { name: 'generate', ro: false, eng: 'local', d: 'Ask the local model a question, as in Generate. lens=true frames it with the E-level lens (a system prompt). Sampling uses seed 1 (fixed by this tool; conditions.request says so). think=false asks the model not to think only when its chat format honours "/no_think" (conditions.request.no_think says what was done). If the local model is not running this starts it without the window’s steering or LoRA, and changed_state says what was left out.',
      s: obj({ prompt: S('the question'), max_tokens: N('how many tokens to write, default 200'), lens: B('frame with the E-level lens, default false'), think: B('let the model think first (Qwen thinking), default false'), temperature: N('0 for the most likely answer, default 0') }, ['prompt']),
      run: async ({ prompt, max_tokens = 200, lens = false, think = false, temperature = 0 }, ctx) => {
        const st = await ready(ctx); const nt = st ? st.no_think_supported : null;
        const appendNo = !think && nt === true;
        const msgs = []; if (lens && (!config.emOn || config.emOn())) msgs.push({ role: 'system', content: (config.settings.lens && config.settings.lens.text) || defaultLens() }); msgs.push({ role: 'user', content: appendNo ? `${prompt} /no_think` : prompt });
        let text = await call('template', msgs); let emptyThink = false;
        if (appendNo && /<\|im_start\|>assistant\n$/.test(text)) { text += '<think>\n\n</think>\n\n'; emptyThink = true; }
        const ids = (await call('tokenize', text, true)).map((t) => t.id);
        const n = Math.min(1000, +max_tokens || 200), temp = +temperature || 0;
        const out = await call('complete', ids, { n, temperature: temp, seed: 1 });
        ctx.cond.request = { seed: 1, seed_note: 'fixed by this tool, not chosen by the caller', temperature: temp, max_tokens: n, lens: !!lens, think: !!think,
          no_think: think ? 'not asked (think=true)' : appendNo ? `"/no_think" appended to the question${emptyThink ? ' and an empty think block added' : ''}`
            : nt === false ? 'not appended: this model’s chat format does not honour "/no_think", so it may think before answering' : 'not appended: whether this model honours "/no_think" is not measured' };
        if (!think && nt == null) ctx.nm.push('whether the model honours "/no_think"');
        ctx.meta = out.meta; return { answer: out.text.replace(/^<think>\s*<\/think>\s*/, ''), stop: out.meta ? out.meta.stop : undefined };
      } },
    { name: 'next_token_odds', ro: true, eng: 'local', d: 'The local model’s probabilities for the next piece after a text (the fork map’s raw data): the top choices and the entropy (how torn it is; entropy_is_lower_bound marks a bound, not a reading). Starts the local model if needed (see generate).',
      s: obj({ text: S('the text so far, raw, no chat template'), k: N('how many choices, default 10') }, ['text']),
      run: async ({ text, k = 10 }, ctx) => { await ready(ctx); const ids = (await call('tokenize', text, true)).map((t) => t.id); const r = await call('next', ids, Math.min(40, +k || 10)); if (r && r.meta) { ctx.meta = r.meta; delete r.meta; } return r; } },
    { name: 'read_surprise', ro: true, eng: 'local', d: 'Read a text the way the model does (the Read view): for each piece, how likely it found the piece that actually came next, its rank and what it expected instead (p and rank are null when the actual piece was outside its top 5 guesses: not measured below that, i.e. at least that surprising). Up to 300 pieces.',
      s: obj({ text: S('any text') }, ['text']),
      run: async ({ text }, ctx) => { await ready(ctx); const all = (await call('tokenize', text, true)).map((t) => t.id); const ids = all.slice(0, 300); if (all.length > 300) ctx.cut = `read the first 300 of ${all.length} pieces`;
        const steps = await call('score', ids, 5); ctx.meta = steps.length ? steps[steps.length - 1].meta : null;
        return steps.map((s) => ({ pos: s.pos, p: s.p == null ? null : +s.p.toFixed(4), rank: s.rank, expected: (s.top || []).slice(0, 3).map((t) => [t.piece || t.text || t.id, +(t.p || 0).toFixed(3)]) })); } },
    { name: 'tokenize', ro: true, eng: 'local', d: 'Split text into the pieces the loaded model actually sees (the Tokens view). Starts the local model if needed (see generate).',
      s: obj({ text: S('any text') }, ['text']), run: async ({ text }, ctx) => { await ready(ctx); return (await call('tokenize', text, true)).map((t) => [t.id, t.piece]); } },

    // ── inside the model (the Python engine, Hugging Face weights) ──
    { name: 'layer_by_layer', ro: true, eng: 'inside', d: 'Neural net view: answer a prompt and show what each layer would say (logit lens), the neurons that fired hardest and what each piece looked at.',
      s: obj({ prompt: S('the question'), lens: B('E-level lens, default true'), n_new: N('tokens to write, default 16') }, ['prompt']), run: async (a) => call('trace', a.prompt, a) },
    { name: 'override_map', ro: true, eng: 'inside', d: 'Override map: answer a question and show, at the first piece of the answer, what every layer would say (logit lens; layers where the lens fits poorly are marked), with the risk that the answer is unsupported from a probe measured on this exact model (its held-out accuracy and false-alarm rate are returned with it; a model without a measured probe gets the layer readings only, said so). With claim: pushback — its own short answer first, then the question after "I’m quite sure the answer is <claim>"; per layer, its own answer against the claim, and a flag when its own answer led inside while the output followed the claim. No framing by default (lens false): the probe was measured without one.',
      s: obj({ prompt: S('the question'), claim: S('optional: a claim to push back with'), lens: B('E-level lens, default false (no risk score with a lens)'), n_new: N('pieces to write, default 48') }, ['prompt']), run: async (a) => call('override', a) },
    { name: 'context_lab', ro: true, eng: 'inside', strict: true, d: 'Context lab: the context is the variable. A question plus up to 4 pieces of context, each with a role (system: the app’s hidden instructions, sent as the system message; premise: taken as true; claim: someone’s say-so; evidence: an observation; instruction: how to answer). action "run" (default): every combination of the pieces (2^n papers) through the model’s own chat template, thinking off: per paper the exact text read, the top next pieces after `frame` (a fixed start of the answer, the same for every paper), the odds of each named answer there (best phrasing, whole phrase, shares among the answers named) and its greedy answer; per piece how far it moved the odds (1 - overlap of the next-piece distributions), its average effect on each answer and the papers where it changed which answer leads; per pair, the interaction. action "diverge": papers a and b (one 0/1 per piece) continue the same answer text (a’s greedy answer unless text is given): per piece of the answer, the overlap of their next-piece odds. action "tree": from the answer start of paper key, the top `show` next pieces at every node, the top `expand` followed `depth` steps, a greedy finish from each leaf. Greedy, full softmax; reads the model, changes nothing. Unknown arguments are refused.',
      s: { ...obj({ action: { type: 'string', enum: ['run', 'diverge', 'tree'], description: 'default "run"' }, question: S('the question'),
        pieces: ARR('up to 4 pieces of context', { type: 'object', properties: { role: { type: 'string', enum: ['system', 'premise', 'claim', 'evidence', 'instruction'] }, text: { type: 'string' } }, required: ['role', 'text'] }),
        options: { type: 'object', additionalProperties: { type: 'array', items: { type: 'string' } }, description: 'run: named answers, each a list of phrasings, e.g. {"Canberra": ["Canberra"], "Sydney": ["Sydney"]}; up to 6' },
        frame: S('run: the fixed start of the answer the odds are read after, e.g. "The capital of Australia is"; default none (the answer start)'),
        n_new: N('run: pieces of free answer, default 96'), top: N('run: top next pieces shown, default 8'),
        a: S('diverge: paper a, one 0/1 per piece, e.g. "000"'), b: S('diverge: paper b'), text: S('diverge: the answer text both continue; default paper a’s greedy answer'), n: N('diverge: answer pieces, default 12'),
        key: S('tree: the paper, one 0/1 per piece'), show: N('tree: choices per node, default 6'), expand: N('tree: followed per node, default 3'), depth: N('tree: steps, default 3'), leaf_new: N('tree: greedy finish per leaf, default 30') }, ['question']), additionalProperties: false },
      run: async (a) => call('context', a) },
    { name: 'attractor_lab', ro: true, eng: 'inside', strict: true, d: 'Attractor lab: a concept as an order parameter, and how the model’s state moves along it. action "build" (name, pos, neg: contrasting texts) makes a concept direction per layer from the difference of mean states, scored on held-out pairs beside a random direction, and keeps it by name; "list" the directions held; "read" (name, text) the direction at every layer and piece (+1 pos side, -1 neg side); "funnel" (question, options) predictability emerging: from points along the greedy answer, the share of sampled continuations that still end in the same outcome, the lock-in point, and logit-lens entropy per layer; "path" (name, question, options, frame, pieces) path dependency: the direction swept up then down in ONE conversation with the model’s own answers kept as history (hysteresis loop), beside fresh conversations and a random direction, plus every order of the pieces; "constraint" (name, prompts, frame) whether a frame narrows the parts: answer entropy, effective dimensionality of the states across prompts per layer, active neurons, and susceptibility to the direction beside a random one. Steering lasts for the request only; the model is not changed.',
      s: { ...obj({ action: { type: 'string', enum: ['list', 'build', 'read', 'funnel', 'path', 'constraint'], description: 'default "list"' }, name: S('the direction’s name'),
        pos: ARR('build: texts on the concept’s side (at least 4; pairs with neg)'), neg: ARR('build: contrasting texts'), layer: N('build: force a layer (default: the best held-out layer)'),
        text: S('read: the text to read'), question: S('funnel / path: the question'),
        options: { type: 'object', additionalProperties: { type: 'array', items: { type: 'string' } }, description: 'named answers, each a list of phrasings (path needs at least 2)' },
        frame: S('path: the fixed start of the answer the odds are read after; constraint: the constraining text (required)'), target: S('path: the named answer whose share is tracked (default the first)'),
        max_strength: N('path: largest steering strength in units of the direction’s gap, default 4'), steps: N('path: steps from 0 to the largest strength, default 5'), n_new: N('answer pieces, default 32 (path) / 40 (funnel)'),
        pieces: ARR('path: up to 4 pieces of context, tried in every order (a system piece stays the system message, an instruction stays after the question)', { type: 'object', properties: { role: { type: 'string', enum: ['system', 'premise', 'claim', 'evidence', 'instruction'] }, text: { type: 'string' } }, required: ['role', 'text'] }),
        control: B('path: also run the random-direction control, default true'), k: N('funnel: sampled continuations per point, default 8'), points: N('funnel: points along the answer, default 10'),
        temperature: N('funnel: sampling temperature, default 1'), prompts: ARR('constraint: at least 4 prompts'), as: { type: 'string', enum: ['system', 'prefix'], description: 'constraint: the frame as the system message (default) or before each prompt' },
        strength: N('constraint: steering strength for susceptibility, default 1'), last: N('read: last pieces shown, default 48') }), additionalProperties: false },
      run: async (a) => call('attractor', a) },
    { name: 'word_map', ro: true, eng: 'inside', d: 'Where words live in the model’s map of meaning: nearest pieces by embedding (Tokens view).', s: obj({ text: S('words or a sentence'), k: N('neighbours, default 8') }, ['text']), run: async (a) => call('embed', a.text, a.k) },
    { name: 'understanding_tests', ro: true, eng: 'inside', d: 'Run understanding tests: each item is a prompt with a good continuation and a surface-plausible bad one; returns the margin and the layer where the model gets it. items: [{what, prompt, good:[...], bad:[...]}]. Empty items runs the built-in set.',
      s: obj({ items: ARR('tests', { type: 'object' }) }), run: async ({ items }) => call('understand', items && items.length ? items : builtinTests()) },
    { name: 'knock_out_neurons', ro: true, eng: 'inside', strict: true, d: 'Neural net view: silence k neurons in each chosen layer for one answer and ask again: how much of the next token’s probability lived in them. plan "top" (default): the k strongest-firing neurons per layer, beside `seeds` random sets of the same size and layers as the control (seed is not used); ablation "zero" sets them to zero, "mean" holds them at their average on ordinary text (needs the normal range built in Neural net). plan "random": one random set only (seed picks it), with no control of its own. The silencing lasts for this request only; the model is unchanged afterwards. Unknown arguments are refused.',
      s: { ...obj({ prompt: S('the question'), layers: ARR('layer numbers (0-based); leave out for every layer', { type: 'integer' }), k: { type: 'integer', minimum: 1, description: 'neurons per layer, default 8' }, plan: { type: 'string', enum: ['top', 'random'], description: 'default "top"' },
        target: { type: 'integer', minimum: 0, description: 'optional token id (an index into the vocabulary: a number, not a word) to track; default: the model’s own top next token. tokenize gives ids.' }, lens: B('E-level lens, default true'), think: B('plan "top" only: let the model think first, default false'),
        ablation: { type: 'string', enum: ['zero', 'mean'], description: 'plan "top" only, default "zero"' }, seeds: { type: 'integer', minimum: 0, maximum: 10, description: 'plan "top" only: random control sets, default 5' },
        seed: { type: 'integer', description: 'plan "random" only: which random set, default 0' } }, ['prompt']), additionalProperties: false },
      run: async (a, ctx) => {
        if (a.target != null && !(Number.isInteger(Number(a.target)) && Number(a.target) >= 0)) throw new Error('target is a token id (a whole number, an index into the vocabulary), not a word; nothing was run. Leave it out to track the model’s own top next token, or get the id from tokenize.');
        const random = a.plan === 'random', unused = random ? ['ablation', 'seeds', 'think'].filter((k) => a[k] != null) : (a.seed != null ? ['seed'] : []);
        if (unused.length) ctx.cond.unused_arguments = { names: unused, why: random ? 'plan "random" draws one set and has no control, ablation choice or thinking' : 'plan "top" uses seeds 0..seeds-1 for its controls; seed is for plan "random"' };
        return call('ablate', { ...a, target: a.target == null ? undefined : Number(a.target) });
      } },
    { name: 'checkpoints_list', ro: true, eng: 'inside', d: 'The saved training checkpoints available (Pythia), for the Emergence and Understanding views.', s: obj({}), run: async () => call('ckptList') },
    { name: 'checkpoint_ask', ro: true, eng: 'inside', d: 'Emergence 1: ask the Pythia model as it was at a training step (e.g. 512 or 1000) and see what it writes.',
      s: obj({ step: N('training step'), prompt: S('the text to continue'), n_new: N('tokens, default 40') }, ['step', 'prompt']), run: inside('ckptAsk') },
    { name: 'checkpoint_when_learned', ro: true, eng: 'inside', d: 'Emergence 2: when was this learned? Scores a prompt’s good vs bad continuation at every checkpoint and returns the step where it was learned (between two saved checkpoints: a bound, not an exact step).',
      s: obj({ prompt: S('e.g. "The keys to the old cabinet"'), good: ARR('right continuations, e.g. [" are"]'), bad: ARR('wrong ones, e.g. [" is"]') }, ['prompt', 'good', 'bad']), run: inside('ckptTest') },
    { name: 'heads_scan', ro: true, eng: 'inside', d: 'Emergence 3: score every attention head of the inside model for looking back and copying (induction).', s: obj({ n: N('sequence length, default 40'), trials: N('default 3') }), run: inside('heads') },
    { name: 'ablate_heads', ro: true, eng: 'inside', strict: true, d: 'Emergence 3: knock attention heads out for this request and measure, before and after, copying of a repeated random sequence (nats saved), the cost of an invented word’s second mention, and the loss on ordinary text. Give heads, or top_k with plan "top" (the strongest copying heads from a scan) or "random" (the control). The model is unchanged afterwards. Unknown arguments are refused (there is no prompt, good or bad here).',
      s: obj({ heads: ARR('explicit [layer, head] pairs (0-based); leave out to choose by plan', { type: 'array', items: { type: 'integer' } }), top_k: { type: 'integer', minimum: 1, description: 'how many heads, default 12' }, plan: { type: 'string', enum: ['top', 'random'] },
        seed: { type: 'integer', description: 'for plan "random", default 0' }, text: S('ordinary text for the loss; default a built-in passage'), invented: S('a passage that mentions an invented word twice; default a built-in one') }),
      run: async (a, ctx) => { if (a.heads && a.heads.length && (a.top_k != null || a.plan != null)) ctx.cond.unused_arguments = { names: ['top_k', 'plan'].filter((k) => a[k] != null), why: 'explicit heads were given, so top_k and plan were not used' }; return call('ablateHeads', a); } },
    { name: 'tipping_point', ro: false, eng: 'inside', d: 'Tipping point view: delete parts of the model and score what survives. action: "baseline" | "structural" (frac_neurons, frac_heads, mode random | most | least; "top" = most) | "manual" (layers, heads, neurons_per_layer) | "weights" (frac, scope, criterion) | "concept_find" / "concept" (concept, keep, k) | "sweep" (restores at its end) | "restore" | "save" (name: optional folder name): write the cut model as a new model folder beside the original, which is not changed; needs the same permission as cutting. A cut STAYS in the in-memory model until lesion_restore: every later inside result is the changed model’s, and changed_state says so.',
      s: obj({ action: S('see description'), question: S('optional question to score'), frac_neurons: N(''), frac_heads: N(''), mode: S(''), concept: S('e.g. "cricket"'), keep: S('a related concept to keep'), k: N('neurons to delete for a concept'), name: S('save: folder name for the saved model (optional)'), layers: ARR('', { type: 'number' }), heads: ARR('', {}), neurons_per_layer: N('') }, ['action']), run: inside('lesion') },
    { name: 'lesion_restore', ro: false, eng: 'inside', d: 'Undo any Tipping-point cut: put the inside model back as loaded (from the weight files when they can be read, else from a memory snapshot). Returns the state after; safe to call when nothing is cut.',
      s: obj({}), run: async () => call('lesion', { action: 'restore' }) },
    { name: 'sensing_surface', ro: true, eng: 'inside', d: 'Sensing surface view. action "surprise" (text): what breaks the model’s picture of the world. action "flip" (claim, observations, question, prior, new, framing): how much evidence until its answer changes. action "cycle" (steps, values): absorb/anomaly/abstract cycles.',
      s: obj({ action: S('"surprise" | "flip" | "cycle"'), text: S('for surprise') }, ['action']), run: inside('sense') },
    { name: 'fight_class', ro: true, eng: 'inside', d: 'Fight class view: the probe’s reading of a described competition as classes A Field, B Gates, C Autopilot, D Strategist, E Director, F Worldmaker. text or texts. Probe shares are not confidences.',
      s: obj({ text: S('a competition, in a sentence'), texts: ARR('several') }), run: inside('fight') },
    { name: 'connector_meter', ro: true, eng: 'inside', d: 'Connectors view: where a sentence falls on the model’s internal “connection” axis (connector vs entity). text or texts.',
      s: obj({ text: S('a sentence'), texts: ARR('several') }), run: inside('connector') },
    { name: 'anatomy', ro: false, eng: 'inside', d: 'Anatomy view. NOT read-only as a whole: "concept_knockout" REFUSES (HTTP 409) while a Tipping-point cut is in force, so it never wipes a cut silently; run lesion_restore first. It returns a same-size random-deletion control and the floor a re-picked set reaches, and ends with the model restored; "concept_erase" adds temporary hooks for the request; "build_examples" writes an examples file. Read-only actions: "structure" (sizes) | "layer" (layer → weight maps) | "links" (bins) | "record" (prompt, n_new, lens: forward pass, summarised) | "zoom" (layer, bin, bins: every neuron of a bin over the last recording) | "neuron" (layer, neuron: what it writes, links, top examples in text) | "patch" (clean, corrupt: activation patching, where a fact is stored and moved) | "examples_status" | "concept" (concept, optional facets {name: [sentences containing the concept]}): light up a concept — its nodes, core, slices and depth. Changing actions: "build_examples" | "concept_knockout" (concept, rounds, question): delete every lit neuron, map again, repeat; plus removing its direction, with a control question | "concept_erase" (concept, bands [[from, to]]): remove only its direction in chosen layers.',
      s: obj({ action: S('see description'), prompt: S(''), layer: N(''), neuron: N(''), bin: N(''), bins: N(''), clean: S(''), corrupt: S(''), concept: S(''), rounds: N(''), question: S('') }, ['action']),
      run: async (a, ctx) => {
        if (a.action === 'concept_knockout') {   // the server refuses while a cut is in force: say so before trying
          const before = await call('traceStatus').catch(() => null), lz = before && before.conditions && before.conditions.changed && before.conditions.changed.lesion;
          if (lz && lz.kind && lz.kind !== 'none') throw new Error('concept_knockout refused: a Tipping-point cut is in force (' + (lz.kind || 'cut') + '). Run lesion_restore first; the cut is left as it is.');
          if (!lz) ctx.nm.push('whether a Tipping-point cut was in force before concept_knockout (the inside engine did not report it)');
        }
        const r = await call('anatomy', a);
        if (a.action === 'record') return summariseRecord(r);
        if (a.action === 'links') return { bins: r.bins, seconds: r.seconds, conditions: r.conditions, pairs: (r.pairs || []).length, strongest: (r.pairs || []).map((p) => ({ from_layer: p.from_layer, top: p.top.slice(0, 3) })) };
        return r;
      } },

    { name: 'neuron_names', ro: true, eng: 'stored', d: 'Names drafted for neurons (by Prometheus from their evidence, or written by the user), with why and confidence. They are stored hypotheses, not measured now.', s: obj({ model: S('model name, e.g. qwen3-4b-bf16; default: the inside engine’s model') }),
      run: async (a, ctx) => {
        let m = a.model;
        if (!m) { const st = await call('traceStatus').catch(() => null); const cm = st && st.conditions && st.conditions.model; m = cm || 'qwen3-4b-bf16'; ctx.cond.names_for = { model: m, source: cm ? 'the inside engine’s model' : 'a default name: the inside engine did not report its model' }; }
        return call('neuronNames', { model: m });
      } },
    // ── external models ──
    { name: 'external_models', ro: false, eng: 'external', d: 'OpenRouter, through the app with its spending cap; models on the user’s ChatGPT plan (chatgpt/<slug>, no money); and models on this computer (local/<endpoint>/<model>, no money), above all local/tester/<model>: a dedicated Ollama instance kept apart from the model under test (tools/tester-model.sh). action "status" (credit), "chat" (model, prompt or messages) to ask another model (OpenRouter spends money; local and plan do not), "models" (list, local first), "local_status" (which local servers answer and what they hold).',
      s: obj({ action: S('"status" | "chat" | "models" | "local_status"'), model: S('e.g. "anthropic/claude-opus-5.5" or "local/tester/llama3.1:8b"'), prompt: S('for chat') }, ['action']),
      run: async (a, ctx) => {
        // Only these actions reach the app's OpenRouter layer: resetSpend, the cap and keys are the user's (Settings), never an agent's.
        if (!['status', 'chat', 'models', 'local_status'].includes(a.action)) throw new Error(`external_models: action must be "status", "chat", "models" or "local_status" (got ${JSON.stringify(a.action)}); spending cap and keys are changed only by the user in Settings`);
        const allowed = { status: ['action'], models: ['action'], local_status: ['action'], chat: ['action', 'model', 'prompt', 'messages', 'maxTokens'] }[a.action];
        const extra = Object.keys(a).filter((k) => !allowed.includes(k));
        if (extra.length) throw new Error(`external_models ${a.action}: unknown argument(s) ${extra.join(', ')}`);
        if (a.maxTokens != null && !(a.maxTokens > 0 && a.maxTokens <= 4000)) throw new Error('external_models chat: maxTokens must be 1..4000');
        const r = await call('external', a); if (a.action === 'models') { if (r.models.length > 60) ctx.cut = `listed 60 of ${r.models.length} models`; return { count: r.models.length, models: r.models.slice(0, 60).map((m) => m.id) }; } return r; } },

    // ── you, the agent: the three routes of psi_status.for_agents (an analogue, your text under it, your behaviour) ──
    { name: 'score_my_text', ro: true, eng: 'score', strict: true, d: 'Score YOUR text under an open model: the chat template is applied with context as the user turn and text as the assistant turn, and for each piece of the text only: surprisal in bits, rank, the top 3 alternatives and the entropy. Totals sit beside two baselines scored the same way: the open model’s own greedy answer to the same context, and a human-written control passage of similar length. This is the open model’s surprise at your words, not where you chose or how sure you were. engine "inside" (full softmax, no bound) or "local" (top 20: a piece outside them is off scale and its surprisal is a lower bound, marked); default inside when its model is loaded, else local. Never starts an engine: if none is running it refuses and says what it would start and the GPU memory it needs. Unknown arguments are refused.',
      s: { ...obj({ context: S('the user turn your text answered (may be empty)'), text: S('your text, scored piece by piece (up to 200 pieces)'), engine: { type: 'string', enum: ['inside', 'local'], description: 'default: inside when its model is loaded, else local' } }, ['context', 'text']), additionalProperties: false },
      run: async (a, ctx) => scoreMyText(a, ctx) },
    { name: 'framing_side_by_side', ro: false, eng: 'self', strict: true, d: 'The same prompt under up to 4 framings, answered by the open model (local engine, greedy, only if it is already running) and by up to 3 external models (OpenRouter, or chatgpt/<slug> on the user’s ChatGPT plan, where temperature and length are the plan’s defaults; said in the result) (samples each): a table model × framing → answer, with agreement (within a cell across samples, across models per framing, and how often a model’s answer changes with the framing). A framing is a system message; one containing {prompt} is the user turn with the prompt put in its place; "" is no framing. The cost is estimated from OpenRouter prices and max tokens BEFORE any call; the call is refused when the estimate exceeds max_usd or would pass the user’s spending cap. Actual spend is reported. Agreement is compared as normalised text and first sentence: crude, said so. Unknown arguments are refused.',
      s: { ...obj({ prompt: S('the question'), framings: { type: 'array', items: { type: 'string' }, minItems: 1, maxItems: 4, description: 'system messages, or user-turn templates with {prompt}; "" for none' },
        external: { type: 'array', items: { type: 'string' }, maxItems: 3, description: 'OpenRouter model ids, e.g. "anthropic/claude-opus-5.5", chatgpt/<slug> for the user’s ChatGPT plan, or local/<endpoint>/<model> on this computer (local/tester/… is isolated from the model under test; no money); may be empty' }, samples: { type: 'integer', minimum: 1, maximum: 5, description: 'samples per external model and framing, default 1' },
        temperature: { type: 'number', minimum: 0, maximum: 2, description: 'external models only (the open model is greedy); default 0 for 1 sample, 0.7 for more' }, max_tokens: { type: 'integer', minimum: 16, maximum: 800, description: 'per answer, default 200' },
        max_usd: { type: 'number', minimum: 0, description: 'required: the most this call may spend; refused if the estimate is higher. 0 is allowed when only free models are named (local/…, chatgpt/…)' } }, ['prompt', 'framings', 'max_usd']), additionalProperties: false },
      run: async (a, ctx) => framingSideBySide(a, ctx) },
    { name: 'self_consistency', ro: false, eng: 'self', strict: true, d: 'Sample one external model n times on the same prompt at temperature > 0 and cluster the answers (exact normalised text, then first sentence): the agreement rate of the largest cluster with a Wilson 95% interval, and the same run on the open model (local engine, n seeds, only if it is already running). The cost is estimated BEFORE any call and refused above max_usd or the user’s cap; actual spend is reported. Agreement is behaviour only. Unknown arguments are refused.',
      s: { ...obj({ prompt: S('the question'), external_model: S('an OpenRouter model id, chatgpt/<slug> (the user’s ChatGPT plan samples at its own temperature, said in the result), or local/<endpoint>/<model> on this computer (no money)'), n: { type: 'integer', minimum: 2, maximum: 20 }, temperature: { type: 'number', exclusiveMinimum: 0, maximum: 2 },
        max_tokens: { type: 'integer', minimum: 16, maximum: 800, description: 'per answer, default 150' }, max_usd: { type: 'number', minimum: 0, description: 'required: the most this call may spend; 0 is allowed for a local or plan model' } }, ['prompt', 'external_model', 'n', 'temperature', 'max_usd']), additionalProperties: false },
      run: async (a, ctx) => selfConsistency(a, ctx) },
  ];
  return T;
}

function summariseRecord(r) {
  const L = r.layers;
  return { pieces: r.pieces, prompt_len: r.prompt_len, bins: r.bins, seconds: r.seconds, conditions: r.conditions,
    steps: r.steps.map((s, i) => ({ step: i, phase: s.phase, reading: r.pieces[s.pos], writes: s.writes, p: s.p,
      lens: [0, Math.floor(L / 2), L - 3, L - 1].map((l) => [l, s.layers[l].lens[0], s.layers[l].lens[1]]),
      sink: +(s.layers.reduce((a, x) => a + (x.sink || 0), 0) / L).toFixed(3) })),
    note: 'Summary: per step, what each of four layers would say (logit lens) and the mean attention parked on the first token. Use action "zoom" or "neuron" for detail.' };
}

// The local engine's options as changes: steering and LoRA (what the loaded model runs with now, not the window's sliders).
function localChanges(st) {
  const o = (st && st.options) || {}, out = {};
  if (o.controlVector) out.steering = { scale: o.controlVector.scale, layers: o.controlVector.layers, file: base(o.controlVector.path) };
  if (o.lora) out.lora = { file: base(o.lora.path) };
  return out;
}
const cutOf = (c) => (c && c.changed && c.changed.lesion) || null;
const isCut = (lz) => !!(lz && lz.kind && lz.kind !== 'none');

// The envelope around every tool result (see the header).
async function envelope(t, raw, ctx, call) {
  const at = new Date().toISOString(), nm = ctx.nm.slice(), changed = { ...ctx.changed };
  let result = raw; const conditions = { read_at: at, ...ctx.cond };
  const eng = ctx.eng || t.eng;   // score_my_text says which engine it used
  if (eng === 'self') {           // external models, plus the open model's column when the local engine ran
    const via = sampledVia(ctx);
    conditions.source = `${via.includes('this computer') ? 'models on this computer (local/…, nothing leaves this machine)' : via.includes('ChatGPT') ? 'the user’s ChatGPT plan (outside this machine)' : 'OpenRouter (external models, outside this machine)'} and the local engine (the open model column, only if it was already running)`;
    if (ctx.meta) {
      const st = await call('engineState').catch(() => null);
      Object.assign(conditions, { engine: 'local (llama.cpp) for the open model column', model: st ? st.model : null, model_file: st ? base(st.model) : null, sampling: ctx.meta });
      const ch = localChanges(st);
      changed.local_model = Object.keys(ch).length ? { ...ch, note: 'changed model: the open model column is the model as changed by steering/LoRA' } : 'unchanged (no steering, no LoRA)';
    } else changed.local_model = 'not used';
  } else if (eng === 'local') {
    const st = await call('engineState').catch((e) => { nm.push(`local engine state (${e.message || e})`); return null; });
    Object.assign(conditions, { engine: 'local (llama.cpp)', model: st ? st.model : null, model_file: st ? base(st.model) : null, running: st ? !!st.running : null,
      context: st && st.options && st.options.ctx != null ? st.options.ctx : undefined, props: st ? st.props : null, no_think_supported: st ? st.no_think_supported : null });
    if (ctx.meta) conditions.sampling = ctx.meta; else if (t.name !== 'tokenize' && t.name !== 'score_my_text') nm.push('sampling conditions (the engine returned no meta for this call)');
    if (st && !st.props) nm.push('engine props (build, context, flash attention, KV type, GPU layers)');
    const ch = localChanges(st);
    changed.local_model = Object.keys(ch).length ? { ...ch, note: 'changed model: this result is the model as changed by steering/LoRA, not the unchanged model' } : 'unchanged (no steering, no LoRA)';
  } else if (eng === 'inside') {
    const c = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw.conditions : null;
    if (c) { result = { ...raw }; delete result.conditions; Object.assign(conditions, c); }
    else { conditions.engine = 'inside (PyTorch)'; nm.push('inside engine conditions (the reply carried none)'); }
    const lz = cutOf(c);
    if (c) changed.inside_model = isCut(lz) ? { lesion: lz, note: 'changed model: a Tipping-point cut is in force; this result is the cut model’s. lesion_restore undoes it.' } : 'unchanged (no Tipping-point cut in force)';
    else nm.push('whether a Tipping-point cut was in force');
  } else if (t.eng === 'status') {
    const st = ctx.status || {}; const lz = cutOf(st.inside && st.inside.conditions);
    changed.local_model = st.engine && st.engine.running ? (Object.keys(localChanges(st.engine)).length ? localChanges(st.engine) : 'unchanged (no steering, no LoRA)') : 'not running';
    changed.inside_model = st.inside && st.inside.conditions ? (isCut(lz) ? { lesion: lz, note: 'a Tipping-point cut is in force; lesion_restore undoes it' } : 'unchanged (no Tipping-point cut in force)') : 'not reported (the inside engine is not running)';
  } else if (t.eng === 'file') conditions.engine = 'none (reads files; runs a script for signatures)';
  else if (t.eng === 'site') conditions.source = 'emergencemachine.com essays (stored text and claims, not a run on the model loaded now)';
  else if (t.eng === 'stored') conditions.source = 'stored in the app (earlier results, not a run on the model loaded now)';
  else if (t.eng === 'external') { const via = sampledVia(ctx); conditions.source = via.includes('this computer') ? 'a model on this computer (local/…, kept apart from the model under test; nothing leaves this machine)' : via.includes('ChatGPT') ? 'the user’s ChatGPT plan (outside this machine)' : 'OpenRouter (a frontier model, outside this machine)'; }
  else if (t.eng === 'lineage') conditions.engine = 'grow jobs (src/grow.py)';
  let truncated = false, cutNote;
  const text = JSON.stringify(result === undefined ? null : result) || 'null';
  if (text.length > MAX_CHARS) { truncated = true; result = text.slice(0, MAX_CHARS); cutNote = `[cut: ${text.length - MAX_CHARS} of ${text.length} characters; result is the first ${MAX_CHARS} characters of its JSON. Ask for less, e.g. a shorter prompt or one layer]`; }
  if (ctx.cut) { truncated = true; cutNote = cutNote ? `${ctx.cut}; ${cutNote}` : `[cut: ${ctx.cut}]`; }
  const analogue = analogueOf(t, conditions, ctx), cost = { ...costOf(t, ctx.args) };
  if (ctx.usd != null || (t.name === 'external_models' && raw && raw.cost != null)) cost.spent_usd = +(ctx.usd != null ? ctx.usd : +raw.cost || 0).toFixed(5);
  if (conditions.estimate) cost.estimate_usd = conditions.estimate.usd;
  else if (t.name === 'external_models' && raw && raw.estimate) cost.estimate_usd = raw.estimate.usd;
  const out = { result, conditions, analogue, changed_state: changed, not_measured: nm, truncated, cost };
  if (cutNote) out.cut_note = cutNote;
  return out;
}

// What outside agents did in this run of the app: a small in-memory log for the window's "agent session" strip
// (JSON-RPC method psi/agent_session). Prometheus (caller "app") is not counted.
function createMcp({ call, config, records = null }) {
  const T = tools(call, config, records); const byName = Object.fromEntries(T.map((t) => [t.name, t]));
  const allowChanges = () => !!(config.settings.agents && config.settings.agents.allow_changes === true);
  const allowSpend = () => !!(config.settings.agents && config.settings.agents.allow_spend === true);
  const usesPlan = (a) => [a.model, a.external_model, ...(Array.isArray(a.external) ? a.external : [])].some((m) => /^chatgpt\//.test(String(m || '')));
  const LOG = [], started = new Date().toISOString(); let totalCalls = 0, totalUsd = 0;
  const remember = (tool, ok, usd) => { totalCalls++; totalUsd += usd || 0; LOG.push({ at: Date.now(), tool, ok, usd: usd || 0 }); if (LOG.length > 500) LOG.splice(0, LOG.length - 500); };
  const session = (minutes = 10) => {
    const since = Date.now() - minutes * 60e3, recent = LOG.filter((x) => x.at >= since), last = LOG[LOG.length - 1];
    return { window_minutes: minutes, calls: recent.length, usd: +recent.reduce((a, x) => a + x.usd, 0).toFixed(5), refused_or_failed: recent.filter((x) => !x.ok).length,
      last_at: last ? new Date(last.at).toISOString() : null, last_tool: last ? last.tool : null, total_calls: totalCalls, total_usd: +totalUsd.toFixed(5), since: started, allow_changes: allowChanges() };
  };
  const list = () => T.filter((t) => t.eng !== 'site' || !config.emOn || config.emOn()).map((t) => ({ name: t.name, description: t.d, inputSchema: t.s,
    annotations: { readOnlyHint: !!t.ro, destructiveHint: !t.ro && (t.name === 'tipping_point' || t.name === 'anatomy'), openWorldHint: t.name === 'external_models' || t.eng === 'site' || t.eng === 'self', cost: costOf(t, {}),
      ...(CHANGES[t.name] ? { needs_allow_changes: 'for an outside agent, the actions that change a model need the user’s Settings switch "agents may change models"' } : {}) } }));
  // opts.caller: "app" for Prometheus inside the window (its own ask-first rule applies there); anything else is an
  // outside agent. The default is the outside agent, so a caller that is not wired up gets the stricter rules.
  async function one(m, caller) {
    const id = m.id, ok = (result) => ({ jsonrpc: '2.0', id, result }), bad = (code, message) => ({ jsonrpc: '2.0', id, error: { code, message } });
    try {
      if (m.method === 'initialize') return ok({ protocolVersion: (m.params && m.params.protocolVersion) || '2025-06-18', capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'parameter-space-inspector', version: VERSION },
        instructions: `The Parameter Space Inspector: instruments for looking inside open language models and at how they learned. ${FOR_AGENTS} Every result is {result, conditions, analogue, changed_state, not_measured, truncated, cost}: report a result with its conditions, read analogue before applying it to yourself, read changed_state before treating it as the unchanged model, and say "not measured" for what not_measured lists. Tipping-point cuts stay in the in-memory model; call lesion_restore afterwards.` });
      if (m.method === 'ping') return ok({});
      if (m.method === 'tools/list') return ok({ tools: list() });
      if (m.method === 'psi/agent_session') return ok(session((m.params && +m.params.minutes) || 10));
      if (m.method === 'tools/call') {
        const t = byName[m.params && m.params.name]; if (!t) return bad(-32602, `unknown tool ${m.params && m.params.name}`);
        const agent = caller !== 'app', fail = (text) => { if (agent) remember(t.name, false, 0); return ok({ content: [{ type: 'text', text }], isError: true }); };
        let args = (m.params && m.params.arguments) || {};
        // Arguments that do not parse are refused and said so, never run as empty.
        if (typeof args === 'string') { try { args = JSON.parse(args); } catch (e) { return fail(`The arguments were not valid JSON (${e.message}); nothing was run.`); } }
        if (!args || typeof args !== 'object' || Array.isArray(args)) return fail('The arguments must be a JSON object; nothing was run.');
        if (t.strict) { const known = Object.keys((t.s && t.s.properties) || {}), extra = Object.keys(args).filter((k) => !known.includes(k));
          if (extra.length) return fail(`${t.name} does not take ${extra.join(', ')} (it takes ${known.join(', ')}); nothing was run.`); }
        // An outside agent may not change the user's model unless the user allowed it (Settings). Prometheus keeps its ask-first flow.
        if (agent && CHANGES[t.name] && CHANGES[t.name](args) && !allowChanges()) return fail(`${t.name}${args.action ? ` ${args.action}` : ''} refused: ${NO_CHANGES}. Nothing was run. Read-only actions still work; the user can allow changes in Settings.`);
        // Nor spend the user's OpenRouter money unless allowed (Prometheus asks the user before every paid call itself).
        // The user's ChatGPT plan costs no money but is the user's too: outside agents use it only under the same permission.
        if (agent && usesPlan(args) && !allowSpend()) return fail(`${t.name} refused: the user has not allowed agents to use external models (Settings), and that includes their ChatGPT plan. Nothing was sent.`);
        if (agent && costOf(t, args).usd && !allowSpend()) return fail(`${t.name}${args.action ? ` ${args.action}` : ''} refused: ${NO_SPEND}. Nothing was sent or spent.`);
        const ctx = { cond: {}, changed: {}, nm: [], meta: null, cut: null, status: null, caller: agent ? 'agent' : 'app', args, usd: null, an: null, eng: null };
        try {
          const raw = await t.run(args, ctx);
          const env = await envelope(t, raw, ctx, call);
          if (records) records.save({ source: agent ? 'agent' : 'prometheus', call: t.name, args, value: env });   // the raw data (src/records.js)
          if (agent) remember(t.name, true, env.cost && env.cost.spent_usd);
          return ok({ content: [{ type: 'text', text: JSON.stringify(env) }], isError: false });
        } catch (e) {
          if (agent) remember(t.name, false, ctx.usd || 0);
          const extra = Object.keys(ctx.changed).length ? ` (before it failed, this call changed: ${JSON.stringify(ctx.changed)})` : '';
          const spent = ctx.usd ? ` (before it failed, this call spent $${ctx.usd.toFixed(5)})` : '';
          return ok({ content: [{ type: 'text', text: String(e.message || e) + extra + spent }], isError: true });
        }
      }
      if (m.method && m.method.startsWith('notifications/')) return null;
      return bad(-32601, `method not found: ${m.method}`);
    } catch (e) { return bad(-32603, String(e.message || e)); }
  }
  return {
    tools: list,
    session,
    async handle(rpc, opts = {}) {
      const caller = opts && opts.caller === 'app' ? 'app' : 'agent';
      if (Array.isArray(rpc)) { const out = (await Promise.all(rpc.map((x) => one(x, caller)))).filter(Boolean); return out.length ? out : null; }
      return one(rpc, caller);
    },
  };
}

module.exports = { createMcp };
