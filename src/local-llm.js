// SPDX-License-Identifier: Apache-2.0
/*
 * A second model on this computer, to test with: any OpenAI-compatible server on a loopback address (Ollama,
 * LM Studio, llama.cpp's llama-server, vLLM, MLX). Models are named local/<endpoint>/<model>, e.g.
 * local/ollama/llama3.1:8b or local/lmstudio/google/gemma-4-e4b, and are usable wherever OpenRouter models are
 * (Prometheus, Compare framings, self-consistency, fact checks, neuron naming). No money: no estimate, no cap.
 *
 * Only loopback hosts are accepted (127.0.0.1, localhost, ::1): this path never sends anything off the computer.
 * Endpoints: settings.local.endpoints [{ name, url }] (Settings > Local test model), url ending in /v1.
 *
 * Isolation: the model that tests must not be the model under test. The default endpoint, "tester", is a dedicated
 * Ollama instance started by tools/tester-model.sh (its own process, port 11435 and model folder ~/.psi-tester),
 * separate from PSI's own llama-server and inside engine and from any Ollama the user runs for other things.
 * A local model is refused when its endpoint is the port PSI's own llama-server listens on.
 */
'use strict';

const PREFIX = 'local/';
const LOOPBACK = new Set(['127.0.0.1', 'localhost', '[::1]', '::1']);
const NAME = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const DEFAULT_ENDPOINTS = [
  { name: 'tester', url: 'http://127.0.0.1:11435/v1' },   // the isolated instance (tools/tester-model.sh)
  { name: 'ollama', url: 'http://127.0.0.1:11434/v1' },
  { name: 'lmstudio', url: 'http://127.0.0.1:1234/v1' },
  { name: 'llamacpp', url: 'http://127.0.0.1:8080/v1' },
];

// A loopback http(s) URL, without a trailing slash; anything else is refused.
function checkUrl(url, what = 'url') {
  let u; try { u = new URL(String(url)); } catch (_) { throw new Error(`${what}: not a URL: ${url}`); }
  if (!['http:', 'https:'].includes(u.protocol)) throw new Error(`${what}: http or https only`);
  if (!LOOPBACK.has(u.hostname)) throw new Error(`${what}: only a server on this computer (127.0.0.1, localhost or ::1), not ${u.hostname}`);
  if (u.username || u.password) throw new Error(`${what}: no user name or password in the URL`);
  return u.toString().replace(/\/+$/, '');
}
function checkEndpoints(list) {
  if (!Array.isArray(list) || list.length > 8) throw new Error('local.endpoints: a list of up to 8 { name, url }');
  const seen = new Set();
  return list.map((e, i) => {
    if (!e || typeof e !== 'object' || Object.keys(e).some((k) => !['name', 'url'].includes(k))) throw new Error(`local.endpoints[${i}]: only { name, url }`);
    if (!NAME.test(String(e.name))) throw new Error(`local.endpoints[${i}].name: lower-case letters, digits, - and _ (e.g. ollama)`);
    if (seen.has(e.name)) throw new Error(`local.endpoints: the name ${e.name} is used twice`); seen.add(e.name);
    return { name: e.name, url: checkUrl(e.url, `local.endpoints[${i}].url`) };
  });
}

function createLocal({ settings, enginePort = () => null }) {
  const endpoints = () => { const s = settings(); return (s.local && Array.isArray(s.local.endpoints) ? s.local.endpoints : DEFAULT_ENDPOINTS); };
  // The model under test answers on PSI's own llama-server: an endpoint on that port would be testing the model with itself.
  const sameAsTested = (url) => { const p = enginePort(); if (!p) return false; const u = new URL(url); return +(u.port || (u.protocol === 'https:' ? 443 : 80)) === +p; };
  const isLocalModel = (m) => typeof m === 'string' && m.startsWith(PREFIX);
  // local/<endpoint>/<model>: the model part may itself hold slashes (google/gemma-4-e4b, hf.co/owner/repo:Q4_K_M).
  function parse(id) {
    const rest = String(id || '').slice(PREFIX.length), i = rest.indexOf('/');
    const name = i > 0 ? rest.slice(0, i) : '', model = i > 0 ? rest.slice(i + 1) : '';
    const ep = endpoints().find((e) => e.name === name);
    if (!ep) throw new Error(`No local endpoint named "${name}" (Settings > Local test model has ${endpoints().map((e) => e.name).join(', ') || 'none'}).`);
    if (!model) throw new Error(`No model named in ${id}: local/${name}/<model>.`);
    const url = checkUrl(ep.url, `local endpoint ${name}`);
    if (sameAsTested(url)) throw new Error(`${id}: ${ep.url} is PSI's own llama-server, the model under test. The model that tests must run apart: start the tester (tools/tester-model.sh) and use local/tester/<model>.`);
    return { ep, model, url, isolated: ep.name === 'tester' };
  }
  async function get(url, ms) {
    // redirect: 'error': a loopback server must not be able to send the request on to another host.
    const r = await fetch(url, { signal: AbortSignal.timeout(ms), redirect: 'error' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  // Which endpoints answer, and what they serve.
  async function status() {
    const rows = await Promise.all(endpoints().map(async (e) => {
      const t0 = Date.now();
      try {
        if (sameAsTested(e.url)) return { name: e.name, url: e.url, up: false, error: 'this is PSI’s own llama-server (the model under test); not used as a tester' }; const d = await get(`${checkUrl(e.url)}/models`, 2500); return { name: e.name, url: e.url, up: true, isolated: e.name === 'tester', models: (d.data || []).map((m) => m.id), ms: Date.now() - t0 }; }
      catch (err) { return { name: e.name, url: e.url, up: false, error: err.name === 'TimeoutError' ? 'no answer in 2.5 s' : /fetch failed|ECONNREFUSED/.test(String(err.message || err.cause)) ? 'not running' : String(err.message) }; }
    }));
    return { endpoints: rows, up: rows.filter((r) => r.up).length };
  }
  // In the same shape as OpenRouter's list (config.external 'models'): $0, marked local.
  async function models() {
    const st = await status();
    return st.endpoints.filter((e) => e.up).flatMap((e) => e.models.filter((m) => !/embed/i.test(m))
      .map((m) => ({ id: `${PREFIX}${e.name}/${m}`, name: `${m} (local · ${e.name})`, in: 0, out: 0, ctx: null, local: true })));
  }

  // Reasoning models served locally (Qwen3, DeepSeek-R1) may put their thinking in the text: kept apart, never as the answer.
  const splitThink = (s) => { const m = /^\s*<think>([\s\S]*?)<\/think>\s*/.exec(s || ''); return m ? { text: s.slice(m[0].length), thinking: m[1].trim() } : { text: s || '', thinking: null }; };

  // Smaller local models often write a tool call as JSON in the text ({"name": …, "parameters": {…}}) instead of a
  // tool_calls entry. Taken as a call only when it names one of the tools offered, with an object of arguments.
  function textToolCall(text, tools) {
    const names = new Set(tools.map((t) => t && t.function && t.function.name).filter(Boolean));
    for (let i = text.indexOf('{'); i >= 0; i = text.indexOf('{', i + 1)) {
      for (let j = text.lastIndexOf('}'); j > i; j = text.lastIndexOf('}', j - 1)) {
        let o; try { o = JSON.parse(text.slice(i, j + 1)); } catch (_) { continue; }
        const args = o && (o.parameters || o.arguments);
        if (o && names.has(o.name) && args && typeof args === 'object' && !Array.isArray(args)) return { id: `call_local_${Date.now().toString(36)}`, type: 'function', function: { name: o.name, arguments: JSON.stringify(args) } };
        break;
      }
    }
    return null;
  }

  async function chat(args) {
    const { ep, model, url } = parse(args.model);
    const s = settings(), ext = s.external || {};
    const body = { model, messages: args.messages || [{ role: 'user', content: args.prompt || '' }], max_tokens: args.maxTokens || ext.maxTokens || 400, stream: false };
    if (args.temperature != null) body.temperature = args.temperature;
    if (Array.isArray(args.tools) && args.tools.length) body.tools = args.tools;
    const t0 = Date.now();
    let r;
    try { r = await fetch(`${url}/chat/completions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(((s.local && s.local.timeoutS) || 300) * 1000), redirect: 'error' }); }
    catch (e) { throw new Error(e.name === 'TimeoutError' ? `${args.model}: no answer from ${ep.name} in ${(s.local && s.local.timeoutS) || 300} s.` : `${args.model}: ${ep.name} at ${ep.url} is not running (start it, or check Settings > Local test model).`); }
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(`${ep.name} ${r.status}: ${(d.error && (d.error.message || d.error)) || r.statusText}${r.status === 404 ? ` (is ${model} loaded or pulled there?)` : ''}`);
    const ch = (d.choices || [])[0] || {}, msg = ch.message || {};
    const { text, thinking } = splitThink(msg.content || '');
    let calls = Array.isArray(msg.tool_calls) && msg.tool_calls.length ? msg.tool_calls : null, fromText = false;
    if (!calls && Array.isArray(body.tools)) { const c = textToolCall(text, body.tools); if (c) { calls = [c]; fromText = true; } }
    const message = { role: 'assistant', content: fromText ? null : text || null }; if (calls) message.tool_calls = calls;
    return { model: args.model, text: fromText ? '' : text, toolCalls: calls, message: args.tools ? message : undefined, finish: calls ? 'tool_calls' : ch.finish_reason || 'stop', cost: 0, local: true, isolated: ep.name === 'tester',
      ...(fromText ? { tool_call_from_text: true } : {}),
      thinking: { text: thinking || msg.reasoning_content || msg.reasoning || null },
      sampling: { temperature: body.temperature ?? `${ep.name}'s default`, max_tokens: body.max_tokens },
      cost_source: `a model on this computer (${ep.name}): no money`,
      tokens: d.usage ? { in: d.usage.prompt_tokens, out: d.usage.completion_tokens } : null, ms: Date.now() - t0 };
  }

  return { chat, status, models, isLocalModel, parse, PREFIX };
}

module.exports = { createLocal, checkEndpoints, checkUrl, DEFAULT_ENDPOINTS, PREFIX };
