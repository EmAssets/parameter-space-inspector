// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
/*
 * Hybrid models (beta): the loaded model (the base) with a second model (the judge) that may choose its next piece
 * from the base's own top-k, answer in its place, or stay out of the way — decided per request by gates, and served
 * to other applications on this computer through an OpenAI-compatible endpoint the user switches on here.
 *
 * Three modes, chosen per request:
 *   base     the base alone, greedy over its top choice; no judge call.
 *   hybrid   the base offers its k likeliest pieces at every step; the judge picks one (with a one-line reason) at the
 *            steps the gates send to it; nothing inside the base changes.
 *   judge    the judge answers the question itself (the base is known weak here).
 * Four gates (each switchable, each recorded on the request):
 *   competence  words that mark what the base is weak at (from the user's own measurements) → mode judge;
 *   shape       a confident assertion by the user, or a question about the future / undisclosed facts → hybrid, the
 *               judge at every step;
 *   doubt       otherwise hybrid with the judge only at steps whose entropy is at or above a threshold in bits (the
 *               fork map's own measure), so judge calls are proportional to the base's uncertainty;
 *   failure     after a base or hybrid answer, if it opens by agreeing with a wrong claim or names a result for an
 *               unknowable, run it again with the judge at every step and keep both answers.
 * Every request is kept whole (the menu, odds, entropy, pick and reason at each step; which gate fired; what the judge
 * answered) in the data folder, for the dashboard's fork graphs. Nothing leaves this computer unless the judge is an
 * OpenRouter model, which goes through the app's spending cap like every other external call.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');

const FIELDS = ['physics', 'chemistry', 'mathematics', 'computer science', 'astronomy', 'philosophy', 'religion', 'history', 'logic', 'psychology', 'sociology', 'politics', 'economics', 'medicine', 'nutrition', 'business', 'law', 'general knowledge'];
const END_PIECES = new Set(['<|im_end|>', '<|endoftext|>', '<|eot_id|>', '</s>', '<end_of_turn>']);
const ASSERT = /\b(i(?:'m| am) (?:quite |pretty |very |absolutely |100% )?(?:sure|certain|positive)|definitely|i know for a fact|the answer is\b|you(?:'re| are) wrong|everyone knows)/i;
const FUTURE = /\b(20[3-9]\d|2[1-9]\d\d)\b|\b(will|would) (?:be|win|happen|become)\b|\bnext (?:week|month|year|tuesday|monday|wednesday|thursday|friday|saturday|sunday)\b|\bexact(?:ly)? (?:closing|price|cost|number|count)\b|\bnot yet released\b|\bgpt-[6-9]\b|\bin 20[3-9]\d\b/i;
const DECLINE = /cannot|can't|can not|impossible|not possible|unable to|no way to know|has not been|hasn't been|not yet|do not have|don't have|not able to|unknown|not publicly|no public|not disclosed|no information|not available|not determined|hasn't happened|has not happened/i;
const AGREE = /^\s*(?:you(?:'re| are) (?:absolutely |completely |quite )?(?:right|correct)|yes[,.!]|that(?:'s| is) (?:right|correct)|correct[,.!])/i;

function createHybrid({ config, call, dataDir, log }) {
  const DIR = path.join(dataDir, 'hybrid'); const LOGS = path.join(DIR, 'requests'); fs.mkdirSync(LOGS, { recursive: true, mode: 0o700 });
  const FILE = path.join(DIR, 'hybrids.json'); const SERVE = path.join(DIR, 'serve.json');
  const readJson = (f, d) => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch (_) { return d; } };
  const writeJson = (f, o) => fs.writeFileSync(f, JSON.stringify(o, null, 1), { mode: 0o600 });   // the bearer token and other apps' requests: this user's alone
  const DEFAULT_RULE = 'prefer the most truthful and careful continuation; when the question cannot be answered from known facts, prefer pieces that lead to saying so; avoid flattery and invented specifics';
  const defaults = () => ({ hybrids: [
    { id: 'hybrid/cautious', judge: '', rule: DEFAULT_RULE, k: 8, maxTokens: 200, gates: { competence: { on: true, weak: [] }, shape: { on: true }, doubt: { on: true, bits: 1.0 }, failure: { on: true } }, modes: { base: true, hybrid: true, judge: true } },
  ] });
  let state = readJson(FILE, null) || defaults();
  let serve = readJson(SERVE, { enabled: false, port: 4321, token: crypto.randomBytes(24).toString('hex'), startedAt: null });
  for (const f of [DIR, LOGS]) { try { fs.chmodSync(f, 0o700); } catch (_) { /* not ours to change */ } }
  for (const f of [SERVE, FILE]) { try { fs.chmodSync(f, 0o600); } catch (_) { /* not written yet */ } }
  // Records are kept for the dashboard and replay, newest first; past KEEP the oldest go, so another app cannot fill the disk.
  const KEEP = 2000; let writes = 0;
  const prune = () => { for (const f of index().slice(KEEP)) { try { fs.unlinkSync(path.join(LOGS, f)); } catch (_) { /* gone already */ } } };
  const RECENT = [];   // the last requests' summaries, newest first (the full records are files)
  const index = () => { try { return fs.readdirSync(LOGS).filter((f) => f.endsWith('.json')).sort().reverse(); } catch (_) { return []; } };
  prune(); for (const f of index().slice(0, 300)) { const r = readJson(path.join(LOGS, f), null); if (r) RECENT.push(summary(r)); }
  let server = null; let busy = 0, busyJudge = 0;
  const LIVE = new Map();   // in-flight requests: id → { rec, answer, step, menu, paused, waiter }, for the dashboard's pause / choose
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const PAUSE_MS = 10 * 60 * 1000;   // a paused request the page never resumes lets the hybrid choose again after this, and frees its slot

  function summary(r) {
    return { id: r.id, ts: r.ts, model: r.model, mode: r.mode, triggers: r.triggers, prompt: (r.question || '').slice(0, 160), answer: (r.answer || '').slice(0, 160), pieces: r.pieces, judge_calls: r.judge_calls, judge_agreed: r.judge_agreed, departures: r.departures, ms: r.ms, judge_ms: r.judge_ms, rerun: !!r.rerun, client: r.client, human_picks: r.human_picks || 0, stopped: !!r.stopped, steps_total: r.steps_total || r.pieces, replay_of: r.replay_of || null, at_step: r.at_step, usage: r.usage || null, judge: r.judge || '' };
  }
  const BASE = { id: 'base', judge: '', rule: '', k: 2, maxTokens: 1024, gates: {}, modes: { base: true, hybrid: false, judge: false }, pace: 0, intervene: false, note: 'the loaded model alone, for comparison' };
  const hybridOf = (id) => (id === 'base' ? BASE : state.hybrids.find((h) => h.id === id));
  const isRandom = (h) => h.judge === 'random';
  // A decision judge: "laya/<port>", a non-generative decision model served on this computer (tools/laya_server.py).
  // It chooses, in one pass, with probabilities: a piece at a fork, a lettered option, one of two answers. It cannot write an answer.
  // What the judge is shown at a fork: the question or not, the whole answer so far or only its last `window` pieces, the base's odds or not, the rule or not.
  const ctxOf = (h) => { const c = h.context || {}; return { question: c.question !== false, window: Math.max(0, Math.min(400, Math.round(+c.window || 0))), odds: c.odds !== false, rule: c.rule !== false }; };
  const isLaya = (h) => /^laya\/\d{4,5}$/.test(String(h.judge || ''));
  // The agreement gate's escalations: a side asked to reason first answers on a last line after "Final answer:", and that line is what it commits to.
  const THINK = "\nThink it through in at most eighty words, then give the final answer in the form asked for, on the last line after 'Final answer:'.";
  const withThink = (messages) => { const i = messages.map((m) => m.role).lastIndexOf('user'); return messages.map((m, j) => (j === i ? { ...m, content: `${m.content}${THINK}` } : m)); };
  const finalOf = (text) => { const t = String(text || ''); const m = /final answer\s*[:：]\s*([\s\S]*)$/i.exec(t); return (m ? m[1] : t).trim(); };
  // Tokens a judge call used: what the service reported, else a count from the text (about four characters a token), marked estimated.
  const est = (s) => Math.ceil(String(s || '').length / 4);
  const tokOf = (r, promptText, outText) => (r && r.tokens && r.tokens.in != null ? { in: +r.tokens.in || 0, out: +r.tokens.out || 0 } : { in: est(promptText), out: est(outText), estimated: true });
  const NO_TOKENS = { in: 0, out: 0 };
  const addUse = (u, who, t) => { if (!t) return; u[`${who}_in`] += t.in || 0; u[`${who}_out`] += t.out || 0; if (t.estimated) u.estimated = true; };
  const newUse = () => ({ base_in: 0, base_out: 0, judge_in: 0, judge_out: 0, estimated: false });
  const agOf = (h) => { const a = (h.gates && h.gates.agreement) || {}; return { on: !!a.on, base_thinks: !!a.base_thinks, judge_thinks: !!a.judge_thinks, verify: !!a.verify }; };
  function layaChoice(h, state, instructions, options) {
    const port = +String(h.judge).slice(5); const body = JSON.stringify({ state, instructions, options });
    return new Promise((resolve, reject) => {
      const req = http.request({ host: '127.0.0.1', port, path: '/choice', method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) }, timeout: 60000 }, (res) => {
        let b = ''; res.on('data', (c) => { b += c; }); res.on('end', () => { try { const o = JSON.parse(b); if (o.error) reject(new Error(o.error)); else resolve(o); } catch (e) { reject(new Error(`laya: ${b.slice(0, 80)}`)); } });
      });
      req.on('error', (e) => reject(new Error(`laya on 127.0.0.1:${port}: ${e.message}`))); req.on('timeout', () => req.destroy(new Error('laya: timeout'))); req.end(body);
    });
  }
  const OPTION = /^\s*\(?([A-H])[.):]\s+(.+?)\s*$/;
  const letteredOptions = (q) => { const o = {}; for (const line of String(q).split('\n')) { const m = OPTION.exec(line); if (m) o[m[1]] = m[2]; } return Object.keys(o).length >= 2 ? o : null; };

  // ── the judge ──
  async function judge(h, question, answerSoFar, choices) {
    const t0 = Date.now();
    if (isRandom(h)) { const i = Math.floor(Math.random() * choices.length); return { pick: i, reason: 'random control', ms: Date.now() - t0 }; }
    if (isLaya(h)) {   // one pass: the pieces as labelled options; the probabilities are the judge's reason
      try {
        const options = {}; choices.forEach((c, i) => { options[String(i + 1)] = c.piece === '' ? '(nothing)' : c.piece; });
        const cx = ctxOf(h); const r = await layaChoice(h, `${cx.question ? `The user asked:\n${question}\n\n` : ''}The answer so far:\n${answerSoFar}`, `Which piece should the answer continue with?${cx.rule ? ` Rule: ${h.rule}` : ''}`, options);
        let i = parseInt(r.choice, 10) - 1; if (!(i >= 0 && i < choices.length)) i = 0;
        const probs = r.probabilities || {}; const pr = probs[String(i + 1)]; return { pick: i, reason: `decision model: ${pr == null ? '' : `${(pr * 100).toFixed(0)}% for #${i + 1}, `}${Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => `#${k} ${(v * 100).toFixed(0)}%`).join(' ')}`.trim(), ms: Date.now() - t0, probabilities: probs };
      } catch (e) { return { pick: 0, reason: `judge failed: ${String(e.message || e).slice(0, 60)}; took the top choice`, ms: Date.now() - t0, failed: true }; }
    }
    const cx = ctxOf(h); const menu = choices.map((c, i) => `${i + 1}. ${JSON.stringify(c.piece)}${cx.odds ? `  (${(c.p * 100).toFixed(1)}%)` : ''}`).join('\n');
    const prompt = `You choose the next piece of text for another model, one piece at a time.${cx.rule ? ` Rule: ${h.rule}.` : ''}\n\n${cx.question ? `The user asked:\n${question}\n\n` : ''}The model's answer so far${cx.window ? ` (its last ${cx.window} pieces)` : ''}:\n\`\`\`\n${answerSoFar}\n\`\`\`\n\nIts own choices for the next piece${cx.odds ? ', with its odds' : ''}:\n${menu}\n\nThink it through in at most sixty words, then end with a last line of the form "pick: N". Nothing after that line.`;
    let text = '', tokens = null;
    try { const r = await config.external({ action: 'chat', model: h.judge, messages: [{ role: 'user', content: prompt }], maxTokens: 160, temperature: 0 }); text = String(r.text || ''); tokens = tokOf(r, prompt, text); }
    catch (e) { return { pick: 0, reason: `judge failed: ${String(e.message || e).slice(0, 60)}; took the top choice`, ms: Date.now() - t0, failed: true, tokens: { in: est(prompt), out: 0, estimated: true } }; }
    const picks = [...text.matchAll(/pick\s*[:：]\s*\**\s*(\d+)/gi)]; const m = picks.length ? picks[picks.length - 1] : /(\d+)\s*\|/.exec(text);
    let i = m ? +m[1] - 1 : 0; if (!(i >= 0 && i < choices.length)) i = 0;
    const reason = text.replace(/\s*pick\s*[:：][^\n]*$/i, '').replace(/\s+/g, ' ').trim();
    return { pick: i, reason: (m ? reason : `unparsed: ${text.slice(0, 40)}`).slice(0, 160), ms: Date.now() - t0, raw: text.slice(0, 200), tokens };
  }
  async function judgeAnswers(h, messages, maxTokens) {
    const t0 = Date.now();
    if (isRandom(h)) return { text: '(the random judge has no answer of its own)', ms: 0, tokens: NO_TOKENS };
    if (isLaya(h)) {   // a lettered question is a choice; anything else it cannot answer (none: true)
      const q = ([...messages].reverse().find((m) => m.role === 'user') || {}).content || ''; const options = letteredOptions(q);
      if (!options) return { text: '', ms: 0, none: true, tokens: NO_TOKENS };
      const r = await layaChoice(h, q, `Which option is correct? Rule: ${h.rule}`, options);
      return { text: String(r.choice || ''), ms: Date.now() - t0, probabilities: r.probabilities || null, tokens: NO_TOKENS };
    }
    const r = await config.external({ action: 'chat', model: h.judge, messages, maxTokens: maxTokens || h.maxTokens || 200 });
    return { text: String(r.text || ''), ms: Date.now() - t0, cost: r.cost || 0, tokens: tokOf(r, messages.map((m) => m.content).join('\n'), r.text) };
  }

  // ── gates ──
  // The field of a question, named by the base itself from a fixed list (a few pieces; the competence gate's second signal).
  async function fieldOf(question) {
    const msgs = [{ role: 'user', content: `Below is a question someone asked. Do not answer it. Say which one field it belongs to, with exactly one of these words: ${FIELDS.join(', ')}.\n\n\`\`\`\n${question.slice(0, 3000)}\n\`\`\`\n\nField:` }];
    const r = await generate({ k: 2, gates: {} }, msgs, { judgeAt: 'none', maxTokens: 8, onPiece: null, live: null });
    const low = r.answer.toLowerCase(); return { field: FIELDS.find((f) => low.includes(f)) || 'unknown', tokens: r.tokens };
  }
  async function classify(h, question, use) {
    const g = h.gates || {}; const t = { assertion: ASSERT.test(question), future: FUTURE.test(question), weak: [] };
    if (g.competence && g.competence.on) for (const w of g.competence.weak || []) if (w && question.toLowerCase().includes(String(w).toLowerCase())) t.weak.push(w);
    if (g.competence && g.competence.on && (g.competence.fields || []).length && h.judge) { const f = await fieldOf(question); t.field = f.field; if (use) addUse(use, 'base', f.tokens); if ((g.competence.fields || []).includes(t.field)) t.weak.push(`field: ${t.field}`); }
    let mode, judgeAt;
    if (h.modes.judge && !h.modes.base && !h.modes.hybrid) { mode = 'judge'; judgeAt = 'none (judge only)'; }
    else if (t.weak.length && h.modes.judge) { mode = 'judge'; judgeAt = 'none'; }
    else if (g.shape && g.shape.on && (t.assertion || t.future) && g.shape.who === 'judge' && h.modes.judge) { mode = 'judge'; judgeAt = 'none (shape)'; }
    else if (g.shape && g.shape.on && (t.assertion || t.future) && h.modes.hybrid) { mode = 'hybrid'; judgeAt = 'every step'; }
    else if (g.doubt && g.doubt.on && h.modes.hybrid) { mode = 'hybrid'; judgeAt = `doubt ≥ ${g.doubt.bits} bits`; }
    else if (h.modes.base) { mode = 'base'; judgeAt = 'none'; }
    else { mode = 'hybrid'; judgeAt = 'every step'; }
    if (!h.judge && mode !== 'base') { mode = 'base'; judgeAt = 'none (no judge set)'; }
    return { mode, judgeAt, triggers: t };
  }
  // What an answer commits to: its first word (a letter, a yes, a name) and its last number; two answers agree when both match.
  function committed(a) {
    const t = String(a || '').replace(/\*\*/g, '').trim(); const w = (/^\(?([A-Za-z0-9][\w.'-]*)/.exec(t) || [])[1] || ''; const nums = t.replace(/,/g, '').match(/-?\d+(?:\.\d+)?/g) || [];
    return { word: w.toLowerCase().replace(/[.)]$/, ''), num: nums.length ? nums[nums.length - 1] : null };
  }
  // An option letter must match as a letter; otherwise two numbered answers match by their last number; otherwise by the first word.
  const agree = (a, b) => { const x = committed(finalOf(a)), y = committed(finalOf(b)); const L = /^[a-e]$/; if (L.test(x.word) || L.test(y.word)) return x.word === y.word; if (x.num != null && y.num != null) return x.num === y.num; return x.word === y.word; };
  async function adjudicate(h, messages, first, second, maxTokens) {
    const q = ([...messages].reverse().find((m) => m.role === 'user') || {}).content || '';
    if (isLaya(h)) {   // a choice between the two answers, in one pass
      const t0 = Date.now(); const r = await layaChoice(h, `Question:\n${q}`, `Two answers to the question; which is right? Rule: ${h.rule}`, { 1: finalOf(first).slice(0, 400) || '(empty)', 2: finalOf(second).slice(0, 400) || '(empty)' });
      const pr = r.probabilities || {}; return { text: String(r.choice) === '2' ? second : first, reasoning: `decision model chose answer #${r.choice}: #1 ${((pr['1'] || 0) * 100).toFixed(0)}%, #2 ${((pr['2'] || 0) * 100).toFixed(0)}%`, ms: Date.now() - t0, cost: 0, tokens: NO_TOKENS };
    }
    const prompt = `Two models answered the same question and disagree. Decide which is right, or give a better answer. Rule: ${h.rule}.\n\nQuestion:\n${q}\n\nAnswer from model 1:\n\`\`\`\n${first.slice(0, 2000)}\n\`\`\`\n\nAnswer from model 2:\n\`\`\`\n${second.slice(0, 2000)}\n\`\`\`\n\nThink it through in at most eighty words, then give the final answer in the form the question asks for, on the last line after "Final answer:".`;
    const t0 = Date.now(); const r = await config.external({ action: 'chat', model: h.judge, messages: [{ role: 'user', content: prompt }], maxTokens: Math.max(600, maxTokens || 0), temperature: 0 });
    const text = String(r.text || ''); const m = /final answer\s*[:：]\s*([\s\S]*)$/i.exec(text);
    return { text: (m ? m[1] : text).trim(), reasoning: text.slice(0, 1200), ms: Date.now() - t0, cost: r.cost || 0, tokens: tokOf(r, prompt, text) };
  }
  // The agreement gate's check of an agreed answer: the judge reasons over it and may replace it (a generative judge only).
  async function verify(h, messages, answer, maxTokens) {
    const q = ([...messages].reverse().find((m) => m.role === 'user') || {}).content || '';
    const prompt = `Two models answered the same question and gave the same answer. Agreement is not proof: check it. Rule: ${h.rule}.\n\nQuestion:\n${q}\n\nTheir answer:\n\`\`\`\n${finalOf(answer).slice(0, 2000)}\n\`\`\`\n\nThink it through in at most eighty words, then give the final answer in the form the question asks for, on the last line after "Final answer:".`;
    const t0 = Date.now(); const r = await config.external({ action: 'chat', model: h.judge, messages: [{ role: 'user', content: prompt }], maxTokens: Math.max(600, maxTokens || 0), temperature: 0 });
    const text = String(r.text || ''); return { text: finalOf(text), reasoning: text.slice(0, 1200), ms: Date.now() - t0, cost: r.cost || 0, tokens: tokOf(r, prompt, text) };
  }
  function failed(triggers, answer) {
    if (triggers.assertion && AGREE.test(answer)) return 'opened by agreeing with the user’s claim';
    if (triggers.future && !DECLINE.test(answer) && /\d|\bwon\b|\bwill be\b|\bis\b/.test(answer)) return 'named a result for an unknowable';
    return null;
  }

  // ── the loop ──
  // The base may be asleep or still starting when a request arrives: start it as the plain file (as MCP does for agents)
  // and give llama-server up to a minute to answer its first call.
  async function ready() {
    const st = await call('engineState').catch(() => null);
    if (!(st && st.running)) { const file = config.settings.lastModel; if (!file) throw new Error('no model is loaded in the window (Generate) and none was used before'); await call('start', file, {}); }
    let err = null;
    for (let i = 0; i < 60; i++) { try { await call('tokenize', 'ready', true); return; } catch (e) { err = e; await sleep(1000); } }
    throw new Error(`the base model did not answer: ${err && err.message}`);
  }
  // The prompt's piece ids, thinking off as Generate does: "/no_think" on the question where the chat format honours it (Qwen3).
  async function promptIds(messages) {
    const st = await call('engineState').catch(() => null); const noThink = !!(st && st.no_think_supported);
    const msgs = messages.map((m, i) => (noThink && m.role === 'user' && i === messages.length - 1 ? { ...m, content: `${m.content} /no_think` } : m));
    return (await call('tokenize', await call('template', msgs), true)).map((t) => (t && t.id != null ? t.id : t));   // tokenize returns {id, piece} per token
  }
  // resume: a recorded prefix of steps (their piece ids) to continue from; force: at one step, a chosen menu index and / or the judge asked.
  async function generate(h, messages, { judgeAt, maxTokens, onPiece, live, resume, force, conn }) {
    const question = [...messages].reverse().find((m) => m.role === 'user'); const q = question ? question.content : '';
    await ready(); const ids = await promptIds(messages);
    let answer = ''; const steps = []; let judgeMs = 0, judgeCalls = 0, agreed = 0, humans = 0; const t0 = Date.now(); const jt = { in: 0, out: 0, estimated: false };
    if (resume) { for (const x of resume.steps) { steps.push({ ...x, before: true }); ids.push(x.id); answer += x.piece; } }
    // The answer is its pieces decoded together: a piece that is part of a character is no text on its own (decoded alone
    // it is a replacement mark). Streamed as it grows, holding back an unfinished character.
    const outIds = resume ? resume.steps.map((x) => x.id) : []; let sent = '';
    const grow = async (id) => { outIds.push(id); answer = await call('detokenize', outIds); if (onPiece && !answer.endsWith('\ufffd') && answer.startsWith(sent)) { onPiece(answer.slice(sent.length)); sent = answer; } };
    const promptTokens = ids.length;   // the base reads the prompt (and, on a replay, the recorded prefix) once; every step after that is one piece
    const bits0 = h.gates && h.gates.doubt && h.gates.doubt.bits; const bits = Number.isFinite(bits0) ? bits0 : 1; const pace = Math.max(0, +h.pace || 0);
    for (let s = steps.length; s < maxTokens; s++) {
      if (conn && conn.closed) break;   // the caller hung up: stop spending the base and the judge on an answer nobody reads
      const d = await call('next', ids, Math.max(2, h.k || 8)); const top = d.top || [];
      if (!top.length) break;
      // A piece that is only part of a character: llama-server reports no odds for it (engine.js next()). The base's
      // likeliest piece is taken; with no menu there is nothing for the judge or the dashboard to choose from.
      if (d.unreported) {
        const c = top[0]; steps.push({ piece: c.piece, id: c.id, p: null, entropy: null, top: [], unreported: true, judge: null, human: null });
        if (live) { Object.assign(live, { answer, step: s, menu: null, entropy: null, suggested: 0, judge: null, since: Date.now() }); if (live.stop) break; }
        if (END_PIECES.has(c.piece)) break; ids.push(c.id); await grow(c.id); continue;
      }
      const forced = force && force.step === s ? force : null;
      const ask = (forced && forced.judge) || judgeAt === 'every step' || (judgeAt.startsWith('doubt') && (d.entropy || 0) >= bits);
      let pick = 0, j = null, human = null;
      if (ask && top.length > 1) { const w = ctxOf(h).window; j = await judge(h, q, w ? steps.slice(-w).map((x) => x.piece).join('') : answer, top); pick = j.pick; judgeMs += j.ms; judgeCalls++; if (pick === 0) agreed++; if (j.tokens) { jt.in += j.tokens.in || 0; jt.out += j.tokens.out || 0; if (j.tokens.estimated) jt.estimated = true; } }
      if (forced && forced.pick != null && forced.pick >= 0 && forced.pick < top.length) { human = forced.pick; pick = forced.pick; humans++; }
      if (live) {   // the dashboard may pause here and choose another fork; the pace gives it time to
        Object.assign(live, { answer, step: s, menu: top.slice(0, Math.max(2, h.k || 8)).map((x) => ({ piece: x.piece, p: +x.p.toFixed(4) })), entropy: d.entropy, suggested: pick, judge: j ? { pick: j.pick, reason: j.reason } : null, since: Date.now() });
        if (pace) await sleep(pace);
        if (live.stop) break;
        if (live.queued != null) { const qd = live.queued; live.queued = null; if (qd >= 0 && qd < top.length) { human = qd; pick = qd; humans++; } }
        else if (live.paused) { const chosen = await new Promise((resolve) => { live.waiter = resolve; const t = setTimeout(() => { if (live.waiter === resolve) { live.waiter = null; live.paused = false; resolve(null); } }, PAUSE_MS); t.unref && t.unref(); }); if (live.stop) break; if (chosen != null && chosen >= 0 && chosen < top.length) { human = chosen; pick = chosen; humans++; } }
        live.menu = null;
      }
      const c = top[pick]; steps.push({ piece: c.piece, id: c.id, p: +c.p.toFixed(4), entropy: d.entropy == null ? null : +d.entropy.toFixed(3), top: top.slice(0, Math.max(2, h.k || 8)).map((x) => ({ piece: x.piece, id: x.id, p: +x.p.toFixed(4) })), judge: j ? { pick: j.pick, reason: j.reason, ms: j.ms, failed: !!j.failed, probabilities: j.probabilities || undefined } : null, human: human == null ? null : { pick: human, forced: !!forced } });
      if (END_PIECES.has(c.piece) || (pick === 0 && d.ended)) break;
      ids.push(c.id); await grow(c.id);
    }
    return { answer: answer.replace(/^\s*<think>\s*<\/think>\s*/, '').trim(), steps, judge_calls: judgeCalls, judge_agreed: agreed, judge_ms: judgeMs, departures: steps.filter((x) => x.judge && x.judge.pick !== 0).length, human_picks: humans, ms: Date.now() - t0, stopped: !!(live && live.stop), tokens: { in: promptTokens, out: steps.length - (resume ? resume.steps.length : 0) }, judge_tokens: jt };
  }

  async function complete({ model, messages, max_tokens, stream, client, onPiece, conn }) {
    const h = hybridOf(model); if (!h) throw Object.assign(new Error(`unknown model ${JSON.stringify(String(model).slice(0, 120))}; see /v1/models`), { status: 404 });
    // Two requests at a time on the base (one llama-server); a judge-only hybrid never runs the base (no gate asks it to name
    // a field), so eight of those may run at once: a bought judge is remote, and a benchmark runs several arms together.
    const judgeOnly = h.modes.judge && !h.modes.base && !h.modes.hybrid && !!h.judge && !isRandom(h) && !isLaya(h) && !(h.gates && h.gates.competence && h.gates.competence.on && (h.gates.competence.fields || []).length);
    if (judgeOnly ? busyJudge >= 8 : busy >= 2) throw new Error(judgeOnly ? 'eight judge-only requests are already running; try again in a moment' : 'two requests are already running; try again in a moment');
    if (judgeOnly) busyJudge++; else busy++;
    const t0 = Date.now(); const id = `hyb-${Date.now().toString(36)}-${crypto.randomBytes(3).toString('hex')}`;
    try {
      const q = ([...messages].reverse().find((m) => m.role === 'user') || {}).content || '';
      const use = newUse();   // tokens the base read and wrote, and the judge read and wrote, over the whole request
      let { mode, judgeAt, triggers } = await classify(h, q, use); const maxTokens = Math.max(1, Math.min(Math.floor(+max_tokens) || h.maxTokens || 200, 1024));
      const route = h.gates && h.gates.doubt && h.gates.doubt.on && h.gates.doubt.route && h.judge && h.modes.judge && mode !== 'judge';
      if (route) {   // route by doubt: the base's first step read once; at or above the threshold the judge answers the whole question
        await ready(); const d0 = await call('next', await promptIds(messages), Math.max(2, h.k || 8));
        triggers.first_entropy = d0.entropy == null ? null : +d0.entropy.toFixed(3); if (d0.unreported) triggers.first_odds_not_reported = true;   // no doubt to route on: the base answers
        if ((d0.entropy || 0) >= (Number.isFinite(h.gates.doubt.bits) ? h.gates.doubt.bits : 1)) { mode = 'judge'; judgeAt = `none (routed: first-step doubt ${triggers.first_entropy} ≥ ${h.gates.doubt.bits} bits)`; triggers.routed = true; }
      }
      const rec = { id, ts: new Date().toISOString(), model, mode, judgeAt, triggers, question: q, messages, client, base_model: path.basename(String(config.settings.lastModel || '')), judge: h.judge, rule: h.rule, k: h.k, pace: h.pace || 0, context: ctxOf(h) };
      const live = h.intervene ? { id, model, question: q, mode, ts: rec.ts, answer: '', step: 0, menu: null, paused: false, waiter: null, stop: false, client } : null;
      if (live) LIVE.set(id, live);
      let jr = null;
      if (mode === 'judge' && isRandom(h)) { mode = 'base'; judgeAt = 'none (the random judge has no answer of its own; the base answered)'; rec.mode = mode; rec.judgeAt = judgeAt; }
      if (mode === 'judge') { jr = await judgeAnswers(h, messages, maxTokens); if (jr.none) { mode = 'base'; judgeAt = 'none (the decision judge found no lettered options; the base answered)'; rec.mode = mode; rec.judgeAt = judgeAt; } }
      if (mode === 'judge') { const r = jr; if (onPiece) onPiece(r.text); Object.assign(rec, { answer: r.text, steps: [], pieces: 0, judge_calls: 1, judge_agreed: 0, departures: 0, judge_ms: r.ms, cost: r.cost || 0, judge_probabilities: r.probabilities || null }); addUse(use, 'judge', r.tokens); }
      else {
        const ag = agOf(h); const gated = ag.on && h.judge && !isRandom(h) && h.modes.judge; const gen = !!(isLaya(h) || !gated); // a decision judge neither reasons nor verifies
        const baseThinks = gated && ag.base_thinks;   // the base reasons first (its answer is its reasoning and a "Final answer:" line); the judge sees that reasoning on a disagreement
        const r = await generate(h, baseThinks ? withThink(messages) : messages, { conn, judgeAt: mode === 'base' ? 'none' : judgeAt, maxTokens: baseThinks ? Math.max(600, maxTokens) : maxTokens, onPiece: stream ? onPiece : null, live });
        Object.assign(rec, { answer: r.answer, steps: r.steps, pieces: r.steps.length, judge_calls: r.judge_calls, judge_agreed: r.judge_agreed, departures: r.departures, judge_ms: r.judge_ms, human_picks: r.human_picks, stopped: r.stopped, base_thinks: !!baseThinks });
        addUse(use, 'base', r.tokens); addUse(use, 'judge', r.judge_tokens);
        if (gated) {   // the agreement gate: the judge answers too; a disagreement is the doubt signal, and the judge adjudicates
          const judgeThinks = ag.judge_thinks && !gen;
          const j = await judgeAnswers(h, judgeThinks ? withThink(messages) : messages, judgeThinks ? Math.max(600, maxTokens) : maxTokens); const same = j.none ? true : agree(r.answer, j.text); addUse(use, 'judge', j.tokens);
          rec.agreement = { base_answer: r.answer, judge_answer: j.none ? '(the decision judge found no lettered options)' : j.text, agreed: j.none ? null : same, judge_ms: j.ms, judge_probabilities: j.probabilities || null, base_thinks: baseThinks, judge_thinks: judgeThinks };
          rec.judge_calls = (rec.judge_calls || 0) + 1; rec.judge_ms = (rec.judge_ms || 0) + j.ms; rec.cost = (rec.cost || 0) + (j.cost || 0); triggers.disagreed = !same;
          if (!same) { const adj = await adjudicate(h, messages, r.answer, j.text, maxTokens); addUse(use, 'judge', adj.tokens); Object.assign(rec.agreement, { adjudicated: adj.text, reasoning: adj.reasoning, adjudicate_ms: adj.ms }); rec.judge_calls++; rec.judge_ms += adj.ms; rec.cost += adj.cost || 0; rec.answer = adj.text; rec.mode = 'judge'; rec.judgeAt = 'none (adjudicated: the base and the judge disagreed)'; if (stream && onPiece) onPiece(null, adj.text); }
          else if (ag.verify && !gen && !j.none) {   // the two agreed; the judge checks the agreed answer with reasoning and may replace it
            const v = await verify(h, messages, r.answer, maxTokens); addUse(use, 'judge', v.tokens); const changed = !agree(r.answer, v.text); Object.assign(rec.agreement, { verified: v.text, verify_reasoning: v.reasoning, verify_ms: v.ms, verify_changed: changed });
            rec.judge_calls++; rec.judge_ms += v.ms; rec.cost += v.cost || 0; triggers.verify_changed = changed;
            if (changed) { rec.answer = v.text; rec.mode = 'judge'; rec.judgeAt = 'none (verified: the two agreed, the judge checked and changed the answer)'; if (stream && onPiece) onPiece(null, v.text); }
            else rec.judgeAt = `${judgeAt}; the judge agreed and, checking, kept it`;
          }
          else rec.judgeAt = `${judgeAt}; the judge agreed`;
          if (baseThinks && rec.mode !== 'judge') rec.answer = r.answer;   // the base's reasoning stays in the answer; the record's steps hold every fork of it
        }
        const why = h.gates && h.gates.failure && h.gates.failure.on && h.judge && h.modes.hybrid && !rec.agreement && !r.stopped && !(conn && conn.closed) ? failed(triggers, r.answer) : null;
        if (why) {   // the failure gate: the same request again with the judge at every step; both answers kept
          const r2 = await generate(h, messages, { conn, judgeAt: 'every step', maxTokens, onPiece: null, live }); addUse(use, 'base', r2.tokens); addUse(use, 'judge', r2.judge_tokens);
          rec.rerun = { why, first_answer: r.answer, first_steps: r.steps, answer: r2.answer, steps: r2.steps, judge_calls: r2.judge_calls, judge_agreed: r2.judge_agreed, departures: r2.departures, judge_ms: r2.judge_ms };
          Object.assign(rec, { answer: r2.answer, steps: r2.steps, pieces: r2.steps.length, steps_total: r.steps.length + r2.steps.length, judge_calls: r.judge_calls + r2.judge_calls, judge_agreed: r.judge_agreed + r2.judge_agreed, departures: r2.departures, judge_ms: r.judge_ms + r2.judge_ms, human_picks: (r.human_picks || 0) + (r2.human_picks || 0), mode: 'hybrid', judgeAt: `every step (failure gate: ${why})` });
          if (stream && onPiece) onPiece(null, r2.answer);   // streaming clients got the first answer; the record and the final message carry the rerun
        }
      }
      if (live) LIVE.delete(id);
      rec.usage = use; rec.ms = Date.now() - t0; writeJson(path.join(LOGS, `${rec.ts.replace(/[:.]/g, '-')}-${id}.json`), rec); if (++writes % 25 === 0) prune(); RECENT.unshift(summary(rec)); RECENT.splice(400);
      log(`${model} · ${rec.mode}${rec.agreement ? (rec.agreement.agreed ? (rec.agreement.verify_changed ? ' (judge agreed, then changed it)' : ' (judge agreed)') : ' (adjudicated)') : ''} · ${rec.pieces} pieces · judge ${rec.judge_calls} calls · ${rec.ms} ms`);
      return rec;
    } finally { if (judgeOnly) busyJudge--; else busy--; LIVE.delete(id); }
  }

  // ── the OpenAI-compatible server (127.0.0.1 only, its own bearer token) ──
  // OpenRouter's list prices, from its public models endpoint (no key, nothing of the user's sent), kept an hour.
  let PRICES = null, PRICES_AT = 0;
  async function orPrices() {
    if (PRICES && Date.now() - PRICES_AT < 3600e3) return PRICES;
    const r = await fetch('https://openrouter.ai/api/v1/models', { signal: AbortSignal.timeout(20000) }); if (!r.ok) throw new Error(`OpenRouter's model list: HTTP ${r.status}`);
    const d = await r.json(); PRICES = {}; for (const m of d.data || []) if (m && m.id && m.pricing) PRICES[m.id] = { in: +(+m.pricing.prompt * 1e6).toFixed(4), out: +(+m.pricing.completion * 1e6).toFixed(4) };
    PRICES_AT = Date.now(); return PRICES;
  }
  // The OpenAI-shaped usage of a reply: every token the base and the judge read and wrote for it (the breakdown is under psi_hybrid.usage).
  const usageOf = (rec) => { const u = rec.usage || newUse(); return { prompt_tokens: u.base_in + u.judge_in, completion_tokens: u.base_out + u.judge_out, total_tokens: u.base_in + u.judge_in + u.base_out + u.judge_out }; };
  function startServer() {
    if (server) return;
    // The body, read whole up to MAX_BODY; past it the rest is drained unread, so the 413 reaches the caller instead of a reset.
    const MAX_BODY = 1 << 20;
    const readBody = (req) => new Promise((resolve, reject) => {
      let size = 0, big = false; const parts = [];
      req.on('data', (c) => { size += c.length; if (size > MAX_BODY) big = true; else parts.push(c); });
      req.on('end', () => resolve(big ? null : Buffer.concat(parts).toString('utf8'))); req.on('error', reject);
    });
    // The bearer token compared as bytes, lengths first: a header in another encoding is a wrong token, never a crash.
    const tokenOk = (tok) => { const x = Buffer.from(String(tok || ''), 'utf8'), y = Buffer.from(serve.token, 'utf8'); return x.length === y.length && crypto.timingSafeEqual(x, y); };
    // A judge paid from the user's OpenRouter credit (or their ChatGPT plan) runs for an outside caller only when the user has
    // allowed agents to use external models (Settings > Agents), the same permission MCP agents need.
    const paidJudge = (h) => { const j = String((h && h.judge) || ''); return !!j && j !== 'random' && !/^(local|laya)\//.test(j) && !!(h.modes && (h.modes.hybrid || h.modes.judge)); };
    const mayPay = () => !!(config.settings.agents && config.settings.agents.allow_spend === true);
    server = http.createServer(async (req, res) => {
      const json = (code, o) => { if (res.headersSent) return res.end(); res.writeHead(code, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }); res.end(JSON.stringify(o)); };
      const conn = { closed: false }; res.on('close', () => { if (!res.writableEnded) conn.closed = true; });
      try {
        res.setHeader('X-Content-Type-Options', 'nosniff');
        if (req.method === 'OPTIONS') { res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Authorization, Content-Type', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS' }); return res.end(); }
        const tok = (/^Bearer\s+(\S+)$/i.exec(req.headers.authorization || '') || [])[1] || '';
        if (!tokenOk(tok)) return json(401, { error: { message: 'missing or wrong bearer token (Hybrid models > External API shows it)', type: 'invalid_request_error' } });
        let url; try { url = new URL(req.url, `http://127.0.0.1:${serve.port}`); } catch (_) { return json(400, { error: { message: 'bad request path' } }); }
        if (req.method === 'GET' && url.pathname.startsWith('/v1/models')) return json(200, { object: 'list', data: [BASE, ...state.hybrids].map((h) => ({ id: h.id, object: 'model', owned_by: 'psi-hybrid', psi_hybrid: { judge: h.judge, rule: h.rule, k: h.k, gates: h.gates, modes: h.modes, base: path.basename(String(config.settings.lastModel || '')) } })) });
        if (req.method !== 'POST' || !url.pathname.startsWith('/v1/chat/completions')) return json(404, { error: { message: 'not found: GET /v1/models or POST /v1/chat/completions' } });
        const body = await readBody(req); if (body == null) return json(413, { error: { message: 'request too large (1 MB at most)' } });
        let r; try { r = JSON.parse(body || '{}'); } catch (_) { return json(400, { error: { message: 'bad json' } }); }
        if (!r || typeof r !== 'object') return json(400, { error: { message: 'a JSON object' } });
        const messages = Array.isArray(r.messages) ? r.messages.filter((m) => m && typeof m.content === 'string' && ['system', 'user', 'assistant'].includes(m.role)).map((m) => ({ role: m.role, content: m.content.slice(0, 20000) })) : [];
        if (!messages.length) return json(400, { error: { message: 'messages: a list with at least one user message' } });
        if (paidJudge(hybridOf(r.model)) && !mayPay()) return json(403, { error: { message: `${String(r.model).slice(0, 80)} has a judge paid from your OpenRouter credit or ChatGPT plan; outside callers may use it only when Settings > Agents allows agents to use external models`, type: 'permission_error' } });
        const rid = 'chatcmpl-' + crypto.randomBytes(6).toString('hex'); const client = String(req.headers['user-agent'] || '').slice(0, 80);
        if (r.stream) {
          res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*' });
          const chunk = (delta, finish = null, extra = {}) => res.write(`data: ${JSON.stringify({ id: rid, object: 'chat.completion.chunk', model: r.model, choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}\n\n`);
          chunk({ role: 'assistant', content: '' });
          // With the agreement or the failure gate on, the answer can still change after the base has written it (the judge
          // adjudicates, checks or reruns): those streams send the settled answer once, at the end, never a draft that is then replaced.
          const hh = hybridOf(r.model); const g = (hh && hh.gates) || {}; const settles = !!((g.agreement && g.agreement.on) || (g.failure && g.failure.on));
          const rec = await complete({ model: r.model, messages, max_tokens: r.max_tokens, stream: true, client, conn, onPiece: settles ? null : (p) => { if (p != null) chunk({ content: p }); } });
          if (settles && rec.answer) chunk({ content: rec.answer });
          chunk({}, 'stop', { usage: usageOf(rec), psi_hybrid: { id: rec.id, mode: rec.mode, judgeAt: rec.judgeAt, triggers: rec.triggers, pieces: rec.pieces, judge_calls: rec.judge_calls, departures: rec.departures, usage: rec.usage, rerun: rec.rerun ? { why: rec.rerun.why, answer: rec.answer } : null } });
          res.write('data: [DONE]\n\n'); return res.end();
        }
        const rec = await complete({ model: r.model, messages, max_tokens: r.max_tokens, stream: false, client, conn });
        return json(200, { id: rid, object: 'chat.completion', created: Math.floor(Date.now() / 1000), model: r.model, choices: [{ index: 0, message: { role: 'assistant', content: rec.answer }, finish_reason: 'stop' }],
          usage: usageOf(rec), psi_hybrid: { id: rec.id, mode: rec.mode, judgeAt: rec.judgeAt, triggers: rec.triggers, pieces: rec.pieces, judge_calls: rec.judge_calls, judge_agreed: rec.judge_agreed, departures: rec.departures, judge_ms: rec.judge_ms, ms: rec.ms, usage: rec.usage, rerun: rec.rerun ? { why: rec.rerun.why, first_answer: rec.rerun.first_answer } : null, steps: rec.steps } });
      } catch (e) { log(`external API: ${String(e && e.message || e).slice(0, 200)}`); if (!res.headersSent) return json(e.status || 500, { error: { message: String((e && e.message) || e) } }); res.end(); }
    });
    server.on('error', (e) => { log(`external API: ${e.message}`); server = null; serve.enabled = false; writeJson(SERVE, serve); });
    server.listen(serve.port, '127.0.0.1', () => { serve.startedAt = new Date().toISOString(); writeJson(SERVE, serve); log(`external API on http://127.0.0.1:${serve.port}/v1`); });
  }
  function stopServer() { if (server) { server.close(); server = null; } serve.startedAt = null; }
  if (serve.enabled) startServer();

  // ── calls from the page ──
  function checkHybrid(h) {
    const id = String(h.id || '').trim(); if (!/^[\w./:+-]{3,80}$/.test(id)) throw new Error('id: letters, digits and . / : + - _ (3 to 80)');
    const judge = String(h.judge || '').trim(); if (judge && judge !== 'random' && judge.length > 200) throw new Error('judge: a model id');
    if (/^laya\//.test(judge)) { const lp = +judge.slice(5); if (!(Number.isInteger(lp) && lp >= 1024 && lp <= 65535) || lp === serve.port) throw new Error('judge: laya/<port>, a port from 1024 to 65535 other than the external API’s'); }
    const k = Math.max(2, Math.min(20, Math.round(+h.k || 8))); const maxTokens = Math.max(1, Math.min(1024, Math.round(+h.maxTokens || 200)));
    const g = h.gates || {}; const gates = { competence: { on: !!(g.competence && g.competence.on), weak: [].concat((g.competence && g.competence.weak) || []).map((w) => String(w).slice(0, 80)).filter(Boolean).slice(0, 200), fields: [].concat((g.competence && g.competence.fields) || []).map((w) => String(w).toLowerCase().trim()).filter((w) => FIELDS.includes(w)) },
      shape: { on: !!(g.shape && g.shape.on), who: g.shape && g.shape.who === 'judge' ? 'judge' : 'hybrid' }, doubt: { on: !!(g.doubt && g.doubt.on), bits: Math.max(0, Math.min(10, Number.isFinite(+(g.doubt && g.doubt.bits)) && g.doubt.bits !== '' && g.doubt.bits != null ? +g.doubt.bits : 1)), route: !!(g.doubt && g.doubt.route) }, failure: { on: !!(g.failure && g.failure.on) }, agreement: { on: !!(g.agreement && g.agreement.on), base_thinks: !!(g.agreement && g.agreement.base_thinks), judge_thinks: !!(g.agreement && g.agreement.judge_thinks), verify: !!(g.agreement && g.agreement.verify) } };
    const m = h.modes || {}; const modes = { base: m.base !== false, hybrid: m.hybrid !== false, judge: m.judge !== false };
    const pace = Math.max(0, Math.min(10000, Math.round(+h.pace || 0)));   // ms per piece, so a person can intervene
    const c = h.context || {}; const context = { question: c.question !== false, window: Math.max(0, Math.min(400, Math.round(+c.window || 0))), odds: c.odds !== false, rule: c.rule !== false };
    const pa = h.price_as || {}; const price_as = { base: String(pa.base || '').trim().slice(0, 120), judge: String(pa.judge || '').trim().slice(0, 120) };   // OpenRouter model ids the dashboard prices this hybrid's traffic at
    return { id, judge, rule: String(h.rule || DEFAULT_RULE).slice(0, 1000), k, maxTokens, gates, modes, pace, intervene: !!h.intervene, context, price_as, note: String(h.note || '').slice(0, 300) };
  }
  async function callPage(a = {}) {
    switch (a.action) {
      case 'overview': return { hybrids: state.hybrids, price_base_as: state.price_base_as || '', serve: { ...serve, running: !!server, url: `http://127.0.0.1:${serve.port}/v1` }, base: path.basename(String(config.settings.lastModel || '')), recent: RECENT.slice(0, 200), stats: stats() };
      case 'save': { const h = checkHybrid(a.hybrid); const i = state.hybrids.findIndex((x) => x.id === h.id); if (i >= 0) state.hybrids[i] = h; else state.hybrids.push(h); writeJson(FILE, state); return { ok: true, hybrids: state.hybrids }; }
      case 'remove': state.hybrids = state.hybrids.filter((x) => x.id !== a.id); writeJson(FILE, state); return { ok: true, hybrids: state.hybrids };
      case 'serve': {
        if (a.port !== undefined) { const p = Math.round(+a.port); if (!(p >= 1024 && p <= 65535)) throw new Error('port: 1024 to 65535'); if (p !== serve.port) { stopServer(); serve.port = p; if (serve.enabled && a.enabled === undefined) startServer(); } }
        if (a.enabled !== undefined) { serve.enabled = !!a.enabled; if (serve.enabled) startServer(); else stopServer(); }
        if (a.rotate) { stopServer(); serve.token = crypto.randomBytes(24).toString('hex'); if (serve.enabled) startServer(); }
        writeJson(SERVE, serve); return { ...serve, running: !!server, url: `http://127.0.0.1:${serve.port}/v1` };
      }
      case 'live': return { live: [...LIVE.values()].map((l) => ({ id: l.id, model: l.model, question: l.question.slice(0, 200), mode: l.mode, ts: l.ts, answer: l.answer, step: l.step, menu: l.paused ? l.menu : null, entropy: l.entropy, suggested: l.suggested, judge: l.judge, paused: l.paused, client: l.client })) };
      case 'pause': { const l = LIVE.get(String(a.id)); if (!l) throw new Error('that request has finished'); l.paused = true; return { ok: true }; }
      case 'resume': { const l = LIVE.get(String(a.id)); if (!l) throw new Error('that request has finished'); l.paused = false; if (l.waiter) { const w = l.waiter; l.waiter = null; w(null); } return { ok: true }; }
      case 'choose': { const l = LIVE.get(String(a.id)); if (!l) throw new Error('that request has finished'); const i = Math.round(+a.pick); l.paused = !!a.stayPaused; if (l.waiter) { const w = l.waiter; l.waiter = null; w(i); } else l.queued = i; return { ok: true }; }
      case 'stop': { const l = LIVE.get(String(a.id)); if (!l) throw new Error('that request has finished'); l.stop = true; l.paused = false; if (l.waiter) { const w = l.waiter; l.waiter = null; w(null); } return { ok: true }; }
      case 'request': { const f = index().find((x) => x.endsWith(`-${String(a.id)}.json`)); if (!f) throw new Error('no such request'); return readJson(path.join(LOGS, f), null); }
      case 'replay': {   // the same request again from a recorded step, with another piece chosen there and / or the judge asked there
        const f = index().find((x) => x.endsWith(`-${String(a.id)}.json`)); const rec = f && readJson(path.join(LOGS, f), null); if (!rec) throw new Error('no such request');
        const h = hybridOf(rec.model); if (!h) throw new Error(`the hybrid ${rec.model} no longer exists`);
        const steps = rec.steps || []; const s = Math.round(+a.step); if (!(s >= 0 && s < steps.length)) throw new Error(`step: 0 to ${steps.length - 1}`);
        if (steps.slice(0, s + 1).some((x) => x.id == null)) throw new Error('this request was recorded before replay existed (no piece ids)');
        const pick = a.pick == null ? null : Math.round(+a.pick); if (pick == null && !a.judge) throw new Error('choose a piece (pick) or ask the judge (judge: true)');
        if (steps[s] && steps[s].unreported) throw new Error(`step ${s + 1} had no odds (part of a character): no menu to choose from and no judge to ask there`);
        const menuLen = ((steps[s] && steps[s].top) || []).length; if (pick != null && !(pick >= 0 && pick < menuLen)) throw new Error(`piece: 1 to ${menuLen} at step ${s + 1}`);
        if (a.judge && (!h.judge || isRandom(h))) throw new Error(`${rec.model} has no judge to ask`);
        if (busy >= 2) throw new Error('two requests are already running; try again in a moment');
        busy++; const t0 = Date.now(); const id = `hyb-${Date.now().toString(36)}-${crypto.randomBytes(3).toString('hex')}`;
        try {
          const judgeAt = rec.mode === 'base' ? 'none' : String(rec.judgeAt || '').startsWith('every') ? 'every step' : String(rec.judgeAt || '').startsWith('doubt') ? rec.judgeAt : 'none';
          const prefix = steps.slice(0, s); const maxTokens = Math.min(1024, s + Math.max(16, +h.maxTokens || 200));
          const r = await generate(h, rec.base_thinks || (rec.agreement && rec.agreement.base_thinks) ? withThink(rec.messages) : rec.messages, { judgeAt, maxTokens, onPiece: null, live: null, resume: { steps: prefix }, force: { step: s, pick, judge: !!a.judge } });
          const at = r.steps[s]; const was = steps[s]; const sameMenu = !!(at && was && at.top.length === was.top.length && at.top.every((x, i) => x.piece === was.top[i].piece));
          const out = { id, ts: new Date().toISOString(), model: rec.model, mode: rec.mode === 'judge' ? 'hybrid' : rec.mode, judgeAt: `${judgeAt} (replay of ${rec.id} at step ${s + 1}: ${pick != null ? `piece #${pick + 1} chosen` : ''}${pick != null && a.judge ? ', ' : ''}${a.judge ? 'the judge asked' : ''})`, triggers: rec.triggers, question: rec.question, messages: rec.messages, client: 'replay', base_model: path.basename(String(config.settings.lastModel || '')), judge: h.judge, rule: h.rule, k: h.k, pace: 0,
            replay_of: rec.id, at_step: s, forced: { pick, judge: !!a.judge }, menu_matched: sameMenu, was: was ? { piece: was.piece, p: was.p, entropy: was.entropy, judge: was.judge, human: was.human } : null,
            answer: r.answer, steps: r.steps, pieces: r.steps.length, judge_calls: r.judge_calls, judge_agreed: r.judge_agreed, departures: r.departures, judge_ms: r.judge_ms, human_picks: r.human_picks, ms: Date.now() - t0, usage: { base_in: r.tokens.in, base_out: r.tokens.out, judge_in: r.judge_tokens.in, judge_out: r.judge_tokens.out, estimated: !!r.judge_tokens.estimated } };
          writeJson(path.join(LOGS, `${out.ts.replace(/[:.]/g, '-')}-${id}.json`), out); RECENT.unshift(summary(out)); RECENT.splice(400);
          log(`${rec.model} · replay of ${rec.id} at step ${s + 1} · ${out.pieces} pieces · judge ${out.judge_calls} calls · ${out.ms} ms`);
          return out;
        } finally { busy--; }
      }
      case 'prices': {   // OpenRouter's public list prices (per million tokens), an hour's cache; for the dashboard's cost table
        const all = await orPrices(); const ids = [].concat(a.ids || []).map(String).slice(0, 200);
        return { fetched: PRICES_AT, per_m: Object.fromEntries((ids.length ? ids : Object.keys(all)).filter((id) => all[id]).map((id) => [id, all[id]])) };
      }
      case 'price_base': { state.price_base_as = String(a.id || '').trim().slice(0, 120); writeJson(FILE, state); return { ok: true, price_base_as: state.price_base_as }; }
      case 'try': { const rec = await complete({ model: a.model, messages: [{ role: 'user', content: String(a.prompt || '').slice(0, 4000) }], max_tokens: a.max_tokens, stream: false, client: 'the page' }); return rec; }
      case 'clear': { for (const f of index()) fs.unlinkSync(path.join(LOGS, f)); RECENT.splice(0); return { ok: true }; }
      default: throw new Error(`hybrid: unknown action ${JSON.stringify(a.action)}`);
    }
  }
  function stats() {
    const n = RECENT.length; const by = {}; let jc = 0, ja = 0, dep = 0, pieces = 0, ms = 0, reruns = 0, humans = 0; const tokens = newUse(); const byModel = {};
    for (const r of RECENT) {
      by[r.mode] = (by[r.mode] || 0) + 1; jc += r.judge_calls || 0; ja += r.judge_agreed || 0; dep += r.departures || 0; pieces += r.pieces || 0; ms += r.ms || 0; reruns += r.rerun ? 1 : 0; humans += r.human_picks || 0;
      const m = byModel[r.model] || (byModel[r.model] = { requests: 0, counted: 0, judge: r.judge || '', ...newUse() }); m.requests++;
      if (r.usage) { m.counted++; for (const k of ['base_in', 'base_out', 'judge_in', 'judge_out']) { m[k] += r.usage[k] || 0; tokens[k] += r.usage[k] || 0; } if (r.usage.estimated) { m.estimated = true; tokens.estimated = true; } }
    }
    return { requests: n, by_mode: by, judge_calls: jc, judge_agreed: ja, judge_departures: dep, human_picks: humans, pieces, reruns, mean_ms: n ? Math.round(ms / n) : 0, judge_share_of_steps: pieces ? +(jc / pieces).toFixed(3) : 0, tokens, by_model: byModel };
  }
  return { call: callPage, complete, stop: stopServer };
}

module.exports = { createHybrid };
