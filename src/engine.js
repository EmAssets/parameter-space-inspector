// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
/*
 * The model engine: llama.cpp's own server (llama-server), run in the
 * background on this computer and spoken to over localhost. It gives what the
 * views need from a large GGUF model: tokens with their text pieces, the full
 * next-token distribution (top-k probabilities), and control vectors to steer
 * the model's internal activations.
 *
 *   const e = new Engine({ binDir });
 *   await e.start(modelPath, { controlVector: { path, scale }, ctx: 8192 });
 *   const toks = await e.tokenize('the street');          // [{id, piece}]
 *   const next = await e.next(toks.map(t => t.id), 20);    // {top:[{id,piece,p}], entropy, entropy_is_lower_bound, covered, ended, meta}
 *
 * Every result of next(), complete() and score() carries `meta`: the conditions of that
 * measurement as llama-server reported them (why it stopped, cache hits, speed, the sampling
 * settings and the seed actually used). info() gives the loaded server's conditions
 * (build, context sizes, flash attention, KV cache type, GPU layers).
 */
'use strict';
const { spawn } = require('child_process');
const net = require('net');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

const freePort = () => new Promise((resolve, reject) => {
  const s = net.createServer();
  s.unref();
  s.on('error', reject);
  s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)); });
});
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DEFAULT_CTX = 8192;
const EXE = process.platform === 'win32' ? '.exe' : '';

// A llama.cpp program (llama-server, llama-quantize): in `dir` when one is set, else the first on the PATH; null if neither.
function findLlama(name, dir) {
  const isFile = (f) => { try { return fs.statSync(f).isFile(); } catch (_) { return false; } };
  if (dir) { const f = path.join(dir, name + EXE); return isFile(f) ? f : null; }
  for (const d of String(process.env.PATH || '').split(path.delimiter)) if (d) { const f = path.join(d.replace(/^"|"$/g, ''), name + EXE); if (isFile(f)) return f; }
  return null;
}

// How many layers of a mixture-of-experts model keep their experts on the processor. want: a number chooses it;
// otherwise (auto) as few as make the model fit the GPU's free memory, and none for a dense model, a model that fits,
// a machine without a readable GPU, or unified memory (a Mac: the processor and GPU share it, so nothing is gained).
async function expertsOnCpu(modelPath, ctx, want) {
  const { header } = require('./autoconfig.js'), h = header(modelPath), kv = h.kv || {}, a = kv['general.architecture'];
  const experts = +kv[`${a}.expert_count`] || 0, layers = +kv[`${a}.block_count`] || 0;
  if (!experts || !layers) return { layers: 0, why: 'not a mixture-of-experts model' };
  if (Number.isInteger(want) && want >= 0) return { layers: Math.min(want, layers), why: 'chosen in Settings' };
  const g = await require('./gpu.js').gpuState({ maxAgeMs: 0 });
  if (!g.ok || g.unified || g.freeMB == null) return { layers: 0, why: g.unified ? 'unified memory' : 'GPU not readable' };
  // What the model needs: the file, its KV cache for this context, and room for compute buffers.
  const kvHeads = +kv[`${a}.attention.head_count_kv`] || +kv[`${a}.attention.head_count`] || 0;
  const headDim = +kv[`${a}.attention.key_length`] || ((+kv[`${a}.embedding_length`] || 0) / (+kv[`${a}.attention.head_count`] || 1));
  const needMB = h.size / 2 ** 20 + (layers * kvHeads * headDim * 2 * 2 * ctx) / 2 ** 20 + 900;
  const freeMB = g.freeMB;
  if (needMB <= freeMB) return { layers: 0, why: `fits: needs about ${Math.round(needMB)} MB, ${freeMB} MB free` };
  // One layer's experts: the *_exps tensors over the layers (sizes from the header; where its type is unknown, nine
  // tenths of the file, as experts are most of a mixture-of-experts model).
  const PSIH = require('./autoconfig.js').parser();
  const ex = (h.tensors || []).filter((t) => /_exps(\.|$)/.test(t.name)).map((t) => PSIH.tensorBytes(t));
  const expBytes = ex.length && ex.every((b) => b != null) ? ex.reduce((s, b) => s + b, 0) : h.size * 0.9;
  const perLayerMB = expBytes / layers / 2 ** 20;
  const n = Math.min(layers, Math.ceil((needMB - freeMB) / perLayerMB) + 1);
  return { layers: n, of: layers, why: `needs about ${Math.round(needMB)} MB, ${freeMB} MB free: the experts of ${n} of ${layers} layers stay on the processor (about ${Math.round(perLayerMB)} MB each)` };
}

// What llama-server says about one /completion, in the names the views render. Field names
// changed across llama.cpp builds (stop_type vs stopped_eos/limit; settings nested or flat), so
// both are read; anything the server did not say is null, never guessed.
function metaOf(out, seedUsed) {
  const t = out.timings || {}; const gs = out.generation_settings || {}; const s = gs.params || gs;
  // stop: eos (the model ended) | limit (n_predict reached) | context (the window filled) | unknown. No stop words are ever sent.
  let stop = 'unknown';
  if (typeof out.stop_type === 'string') stop = out.stop_type === 'eos' ? 'eos' : out.stop_type === 'limit' ? (out.truncated ? 'context' : 'limit') : 'unknown';
  else if (out.stopped_eos) stop = 'eos';
  else if (out.stopped_limit) stop = out.truncated ? 'context' : 'limit';
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  // Generation speed. A one-step call (n_predict 0 or 1, as Generate and Read make) has its token picked while the
  // prompt is read, and recent builds report it as 0 tokens in 0.001 ms ("0 tokens/s"): not a measurement. Then the
  // speed is that one forward pass: one step, one new token, so tokens/s = 1000 / its milliseconds. A cached prompt
  // reads only the few tokens after the cache (1 to 2; gpt-oss re-reads a few more, as its sliding window needs); a
  // pass that read a long uncached prompt says nothing about generation, so it is left out.
  let tps = num(t.predicted_per_second);
  if (!(num(t.predicted_ms) >= 1 && (t.predicted_n == null || t.predicted_n > 0))) tps = num(t.prompt_n) != null && t.prompt_n <= 32 && num(t.prompt_ms) > 0 ? +(1000 / t.prompt_ms).toFixed(2) : null;
  return {
    stop,
    tokens_cached: num(out.tokens_cached),
    timings: { prompt_tps: num(t.prompt_per_second), predicted_tps: tps, predicted_ms: num(t.predicted_ms) },
    settings: { temperature: num(s.temperature), top_p: num(s.top_p), top_k: num(s.top_k), min_p: num(s.min_p), seed: seedUsed != null ? seedUsed : num(s.seed) },
  };
}

class Engine {
  constructor({ binDir, log = () => {} }) {
    this.binDir = binDir;
    this.log = log;
    this.proc = null;
    this.port = 0;
    this.model = null;
    this.options = {};
    this.props = {};
    this.meta = {};               // from GET /models: n_ctx_train and the like
    this.logText = '';            // the server's own log (last 128 KB): where n_ctx_train, flash_attn and the KV type are said
    this.no_think_supported = null;
    this.vocabCache = new Map();
  }

  // llama-server: in binDir (settings.llamaDir) when it is set, else on the PATH. null when it cannot be found.
  get exe() { return findLlama('llama-server', this.binDir); }

  async start(modelPath, options = {}) {
    const exe = this.exe;
    if (!exe) throw new Error(this.binDir ? `llama-server was not found in ${this.binDir}. Set llama.cpp's folder in Settings (or PSI_LLAMA_DIR).`
      : "llama-server was not found: set the folder of llama.cpp's llama-server in Settings (or PSI_LLAMA_DIR), or put it on the PATH.");
    if (!modelPath) throw new Error('No model chosen yet: pick a .gguf file in Settings > Models.');
    if (!fs.existsSync(modelPath)) throw new Error(`Model file not found: ${modelPath}`);
    await this.stop();
    const dir = path.dirname(exe);
    this.port = await freePort();
    // A new key each launch: any program on this computer can reach the localhost port, but not use the model
    // without it. Sent by get() and post().
    this.apiKey = crypto.randomBytes(24).toString('hex');
    const ctx = Number.isInteger(options.ctx) && options.ctx > 0 ? options.ctx : DEFAULT_CTX;
    const ngl = options.gpuLayers ?? 99;
    const args = ['-m', modelPath, '--host', '127.0.0.1', '--port', String(this.port),
      '-ngl', String(ngl), '-c', String(ctx), '--no-webui', '-np', '1'];   // the key goes in LLAMA_API_KEY: other programs can read a command line
    // A mixture-of-experts model that does not fit beside what is already on the GPU: keep the experts of just enough
    // layers on the processor (--n-cpu-moe), rather than let Windows spill part of the model into shared memory. On an
    // RTX 4080 SUPER with 3.9 GB held by other programs, gpt-oss-20b spilled 300 MB and generated 18 tokens/s; with 4
    // layers' experts on the processor, 67 tokens/s.
    this.moe = null;
    try { this.moe = await expertsOnCpu(modelPath, ctx, options.cpuMoe); } catch (e) { this.moe = { layers: 0, why: `not decided: ${e.message}` }; }
    if (this.moe && this.moe.layers > 0) args.push('--n-cpu-moe', String(this.moe.layers));
    // A LoRA adapter in GGUF form (convert_lora_to_gguf.py), with an optional scale.
    if (options.lora && options.lora.path) {
      args.push(options.lora.scale != null ? '--lora-scaled' : '--lora');
      args.push(options.lora.scale != null ? `${options.lora.path}:${options.lora.scale}` : options.lora.path);
    }
    const cv = options.controlVector;
    if (cv && cv.path && cv.scale) {
      if (path.extname(cv.path).toLowerCase() !== '.gguf') throw new Error('control vector: only a .gguf file');
      // This llama.cpp build takes FNAME:SCALE and splits on the first colon, so a
      // Windows drive letter ("H:") breaks it. Pass a path relative to the server's
      // folder instead, copying the vector there if it lives on another drive.
      let rel = path.relative(dir, cv.path);
      if (rel.includes(':')) {
        const local = path.join(dir, 'control-vectors', path.basename(cv.path));
        fs.mkdirSync(path.dirname(local), { recursive: true });
        fs.copyFileSync(cv.path, local);
        rel = path.relative(dir, local);
      }
      args.push('--control-vector-scaled', `${rel.replace(/\\/g, '/')}:${cv.scale}`);
      if (cv.layers) args.push('--control-vector-layer-range', String(cv.layers[0]), String(cv.layers[1]));
    }
    this.log(`starting llama-server on port ${this.port} (ctx ${ctx}, ngl ${ngl})`);
    const { OPENROUTER_API_KEY, PSI_TRACE_TOKEN, PSI_MCP_TOKEN, ...env } = process.env;
    this.proc = spawn(exe, args, { cwd: dir, windowsHide: true, env: { ...env, LLAMA_API_KEY: this.apiKey } });
    this.logText = '';
    const keep = (d) => { this.logText = (this.logText + d.toString()).slice(-131072); };
    this.proc.stdout.on('data', keep);
    this.proc.stderr.on('data', keep);
    const exited = new Promise((resolve) => this.proc.once('exit', (code) => resolve(code)));
    // Wait until the model is loaded (the health check turns ok), or the process dies.
    const deadline = Date.now() + (options.timeoutMs ?? 180000);
    for (;;) {
      const done = await Promise.race([exited.then((c) => ({ exit: c })), sleep(400).then(() => null)]);
      if (done) throw new Error(`llama-server stopped (code ${done.exit}). Last output:\n${this.logText.slice(-1500)}`);
      try {
        const r = await fetch(`http://127.0.0.1:${this.port}/health`, { headers: this.auth });
        if (r.ok) break;
      } catch (_) { /* not listening yet */ }
      if (Date.now() > deadline) { await this.stop(); throw new Error(`The model did not finish loading in time. Last output:\n${this.logText.slice(-1500)}`); }
    }
    this.model = modelPath;
    this.options = { ...options, ctx, gpuLayers: ngl };
    this.vocabCache.clear();
    const props = await this.get('/props').catch(() => ({}));
    this.props = props;
    this.meta = await this.get('/models').then((m) => ((m.models || m.data || [])[0] || {}).meta || {}).catch(() => ({}));
    // "/no_think" and the empty <think> block are only meaningful to a chat template that knows them (Qwen3);
    // on any other model they would reach the model as plain text.
    const tmpl = typeof props.chat_template === 'string' ? props.chat_template : '';
    this.no_think_supported = /\/no_think|enable_thinking/.test(tmpl);
    return { ...props, no_think_supported: this.no_think_supported, info: this.info() };
  }

  // The loaded server's conditions: {build, n_ctx, n_ctx_train, flash_attn, kv_type, ngl}, null where the server did not say.
  info() {
    if (!this.proc) return null;
    const p = this.props || {}, dgs = p.default_generation_settings || {}, log = this.logText || '';
    const rx = (re, i = 1) => { const m = re.exec(log); return m ? m[i] : null; };
    const fa = rx(/Flash Attention was auto, set to (\w+)/) || rx(/flash_attn\s*=\s*(\w+)/);
    const kv = (() => { const m = /K \((\w+)\):[^\n]*V \((\w+)\)/.exec(log); if (m) return m[1] === m[2] ? m[1] : `${m[1]}/${m[2]}`; return rx(/type_k\s*=\s*'?(\w+)/); })();
    return {
      moe: this.moe || null,   // mixture-of-experts placement: how many layers keep their experts on the processor, and why
      build: p.build_info || rx(/\bbuild:\s*(\d+\s*\([0-9a-f]+\))/) || null,
      n_ctx: Number.isInteger(dgs.n_ctx) ? dgs.n_ctx : (+rx(/llama_context:\s*n_ctx\s*=\s*(\d+)/) || this.options.ctx || null),
      n_ctx_train: Number.isInteger(this.meta.n_ctx_train) ? this.meta.n_ctx_train : (+rx(/n_ctx_train\s*=\s*(\d+)/) || null),
      flash_attn: fa === 'enabled' || fa === '1' ? true : fa === 'disabled' || fa === '0' ? false : fa || null,
      kv_type: kv || null,
      ngl: this.options.gpuLayers ?? null,
    };
  }

  // Stop the server and wait for it to be gone (up to 5 s), so a restart never races the old process for the GPU.
  async stop() {
    if (this.proc) {
      const p = this.proc;
      this.proc = null;
      try { p.kill(); } catch (_) { /* already gone */ }
      const t0 = Date.now();
      while (p.exitCode == null && p.signalCode == null && Date.now() - t0 < 5000) await sleep(100);
      if (p.exitCode == null && p.signalCode == null) this.log(`llama-server (pid ${p.pid}) did not exit within 5 s`);
    }
  }

  get auth() { return this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}; }

  async get(route) {
    const r = await fetch(`http://127.0.0.1:${this.port}${route}`, { headers: this.auth });
    if (!r.ok) throw new Error(`${route}: HTTP ${r.status}`);
    return r.json();
  }

  async post(route, body) {
    const r = await fetch(`http://127.0.0.1:${this.port}${route}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...this.auth }, body: JSON.stringify(body) });
    const text = await r.text();
    if (!r.ok) throw new Error(`${route}: HTTP ${r.status} ${text.slice(0, 300)}`);
    return JSON.parse(text);
  }

  // Text -> [{id, piece}]. The pieces are what the model actually sees.
  async tokenize(text, special = true) {
    const out = await this.post('/tokenize', { content: text, add_special: false, parse_special: special, with_pieces: true });
    return out.tokens.map((t) => (typeof t === 'number' ? { id: t, piece: null } : { id: t.id, piece: typeof t.piece === 'string' ? t.piece : bytesToText(t.piece) }));
  }

  async detokenize(ids) {
    const out = await this.post('/detokenize', { tokens: ids });
    return out.content;
  }

  async piece(id) {
    if (!this.vocabCache.has(id)) this.vocabCache.set(id, await this.detokenize([id]));
    return this.vocabCache.get(id);
  }

  // The model's next-token distribution after `ids` (raw, before any sampling).
  async next(ids, k = 20) {
    const out = await this.post('/completion', {
      prompt: ids, n_predict: 1, n_probs: k, temperature: 0, cache_prompt: true,
      return_tokens: true, post_sampling_probs: false,
    });
    const meta = metaOf(out, 0);
    const cp = (out.completion_probabilities || [])[0];
    // No probabilities means the model stopped on an end token at once: a normal ending, and no distribution
    // to measure, so the entropy is null (not a reading of 0).
    if (!cp) return { top: [], entropy: null, entropy_is_lower_bound: false, covered: 0, ended: true, meta };
    const list = cp.top_logprobs || cp.probs || [];
    const top = list.map((x) => ({
      id: x.id ?? x.tok_id,
      piece: x.token ?? x.tok_str ?? '',
      p: x.logprob != null ? Math.exp(x.logprob) : x.prob,
    }));
    const covered = top.reduce((a, t) => a + t.p, 0);
    // Entropy from the top-k, with the leftover mass treated as one bucket: a lower bound on the true entropy.
    let entropy = 0;
    top.forEach((t) => { if (t.p > 0) entropy -= t.p * Math.log2(t.p); });
    const rest = Math.max(0, 1 - covered);
    if (rest > 1e-9) entropy -= rest * Math.log2(rest);
    const r = { top, entropy, entropy_is_lower_bound: true, covered, meta };
    if (meta.stop === 'eos') r.ended = true;
    return r;
  }

  // One sampled continuation of n tokens, sampled by the server (fast). Returns {text, ids, meta}.
  // A seed of -1 is replaced by one drawn here, so meta.settings.seed is always the seed that was used.
  async complete(ids, { n = 30, temperature = 0.8, seed = -1, topP = 0.95 } = {}) {
    const used = Number.isInteger(seed) && seed >= 0 ? seed : Math.floor(Math.random() * 0x7fffffff);
    const out = await this.post('/completion', { prompt: ids, n_predict: n, temperature, top_p: topP, seed: used, cache_prompt: true, return_tokens: true });
    return { text: out.content || '', ids: out.tokens || [], meta: metaOf(out, used) };
  }

  // Read a token sequence the way the model does: for each position, how
  // likely was the token that actually came next, and what else did it expect?
  // Returns the steps, each with its own `meta`; `steps.meta` (the last step's) is for in-process callers only:
  // IPC and JSON drop properties of an array, so a view reads `steps[i].meta`.
  async score(ids, k = 5, onStep = () => {}) {
    const steps = [];
    for (let i = 1; i < ids.length; i++) {
      const d = await this.next(ids.slice(0, i), k);
      const hit = d.top.find((t) => t.id === ids[i]);
      steps.push({ pos: i, id: ids[i], p: hit ? hit.p : null, rank: hit ? d.top.indexOf(hit) + 1 : null, top: d.top, entropy: d.entropy, entropy_is_lower_bound: d.entropy_is_lower_bound, meta: d.meta });
      onStep(i, ids.length - 1);
    }
    steps.meta = steps.length ? steps[steps.length - 1].meta : null;
    return steps;
  }

  // The chat template wrapper, so a system message (e.g. the E-level lens) can
  // be placed the way the model was trained to expect.
  async applyTemplate(messages) {
    const out = await this.post('/apply-template', { messages });
    return out.prompt;
  }
}

function bytesToText(b) {
  try { return Buffer.from(b).toString('utf8'); } catch (_) { return ''; }
}

module.exports = { Engine, metaOf, findLlama };
