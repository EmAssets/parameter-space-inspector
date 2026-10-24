// SPDX-License-Identifier: Apache-2.0
// Keeps the inside-the-model engine (PyTorch, Hugging Face weights) on the same model as the local engine
// (llama.cpp, a GGUF file). When a GGUF is loaded, find its weights folder:
//   1. a pairing remembered from before,
//   2. a folder in the models folder whose name matches the file's (quantisation and format suffixes dropped),
//      with the same width and depth,
//   3. for a small model (up to 256 MB, the toy Llama from How AI Emerges), a one-time conversion next to the file.
// A large model with no weights folder is left alone, and the reason is reported.
'use strict';
const { python: PY } = require('./python.js');
const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const SMALL = 256 * 2 ** 20;
const SUFFIX = /([-_.](q\d(_[0-9a-z]+)*|iq\d\w*|f16|f32|bf16|fp16|gguf|hf|safetensors|instruct-gguf))+$/i;
const norm = (name) => name.toLowerCase().replace(/\.gguf$/, '').replace(SUFFIX, '');

let PARSER = null;
function header(file) {
  if (!PARSER) {
    const g = {}; new Function('window', fs.readFileSync(path.join(__dirname, 'ui', 'gguf-header.js'), 'utf8'))(g);
    PARSER = g.PSIHeader;
  }
  const size = fs.statSync(file).size;
  for (const n of [16 << 20, 128 << 20]) {
    const fd = fs.openSync(file, 'r'); const buf = Buffer.alloc(Math.min(n, size));
    try { fs.readSync(fd, buf, 0, buf.length, 0); } finally { fs.closeSync(fd); }
    try { return { ...PARSER.parseHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), size), size }; }
    catch (e) { if (!(e instanceof PARSER.NeedMore)) throw e; }
  }
  throw new Error('header larger than 128 MB');
}

function shape(h) {
  const a = h.kv['general.architecture'];
  return { arch: a, width: +h.kv[`${a}.embedding_length`] || null, layers: +h.kv[`${a}.block_count`] || null };
}

function weightsShape(dir) {
  try {
    const c = JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf8'));
    const t = c.text_config || c;
    return { width: t.hidden_size || null, layers: t.num_hidden_layers || null };
  } catch (_) { return null; }
}
const hasWeights = (dir) => { try { return fs.readdirSync(dir).some((f) => /\.safetensors$|pytorch_model.*\.bin$/.test(f)); } catch (_) { return false; } };
const same = (a, b) => !!(a && b && a.width === b.width && a.layers === b.layers);

function candidates(root) {
  const out = [];
  const walk = (dir, depth) => {
    let ents = []; try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return; }
    for (const e of ents) {
      if (!e.isDirectory()) continue;
      const p = path.join(dir, e.name);
      if (fs.existsSync(path.join(p, 'config.json')) && hasWeights(p)) out.push(p);
      else if (depth < 2) walk(p, depth + 1);
    }
  };
  walk(root, 0);
  return out;
}

function convert(file, out, log) {
  const py = `
import os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
d, f, o, here = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
AutoModelForCausalLM.from_pretrained(d, gguf_file=f).save_pretrained(o)
AutoTokenizer.from_pretrained(d, gguf_file=f).save_pretrained(o)
sys.path.insert(0, here); import gguf_tok
print(gguf_tok.repair(os.path.join(d, f), o))   # the rebuilt tokenizer drops tokens like <think>: put them back
print("ok")`;
  return new Promise((resolve, reject) => {
    const p = spawn(PY(), ['-c', py, path.dirname(file), path.basename(file), out, __dirname], { env: require('./config.js').cacheEnv(), windowsHide: true });
    let tail = ''; const keep = (d) => { tail = (tail + d).slice(-2000); };
    p.stdout.on('data', keep); p.stderr.on('data', keep);
    p.on('exit', (code) => (code === 0 && /ok\s*$/.test(tail) ? resolve(out) : reject(new Error(`conversion failed: ${tail.slice(-400)}`))));
    log(`converting ${path.basename(file)} for the inside engine`);
  });
}

// settings.trace.pairs maps a GGUF path to its weights folder: a string (older settings) or the record sameModel()
// keeps, {gguf, modelDir, date, ids_equal, top5_overlap}.
const pairDir = (v) => (typeof v === 'string' ? v : v && typeof v === 'object' ? v.modelDir || null : null);

// Returns { modelDir, how, note } (modelDir null when nothing fits).
async function pairFor(file, settings, log = () => {}) {
  const key = path.resolve(file).replace(/\\/g, '/');
  const h = header(file); const s = shape(h);
  const pairs = (settings.trace && settings.trace.pairs) || {};
  const remembered = pairDir(pairs[key]);
  if (remembered && hasWeights(remembered) && same(s, weightsShape(remembered))) return { modelDir: remembered, how: 'remembered' };
  const name = norm(path.basename(file)), gname = norm(String(h.kv['general.name'] || ''));
  const roots = [...new Set([settings.modelsDir, path.dirname(file)].filter(Boolean))];
  const found = roots.flatMap(candidates).filter((d) => { const n = norm(path.basename(d)); return (n === name || (gname && n === gname)) && same(s, weightsShape(d)); });
  if (found.length) return { modelDir: found[0].replace(/\\/g, '/'), how: 'matched by name and shape' };
  if (h.size <= SMALL) {
    const out = path.join(path.dirname(file), `${path.basename(file, '.gguf')}-hf`);
    if (hasWeights(out) && same(s, weightsShape(out))) {
      // folders converted before gguf_tok.py existed carry the faulty tokenizer; the repair is idempotent
      spawnSync(PY(), [path.join(__dirname, 'gguf_tok.py'), file, out], { env: require('./config.js').cacheEnv(), windowsHide: true });
      return { modelDir: out.replace(/\\/g, '/'), how: 'converted earlier' };
    }
    await convert(file, out, log);
    return { modelDir: out.replace(/\\/g, '/'), how: 'converted from the GGUF file' };
  }
  return { modelDir: null, note: `No Hugging Face weights folder matches ${path.basename(file)} (${s.layers} layers, width ${s.width}). Neural net and Anatomy keep the model they had; put its weights in a folder named like the file under ${settings.modelsDir || 'the models folder'}.` };
}

// After the local engine loads a model: point the inside engine at the same model, unless the user turned this off.
async function syncTrace({ file, config, tracer, log = () => {} }) {
  const st = config.settings;
  if (st.trace && st.trace.auto === false) return { changed: false, note: 'automatic matching is off' };
  let r;
  try { r = await pairFor(file, st, log); } catch (e) { return { changed: false, note: `could not match the inside engine: ${e.message}` }; }
  if (!r.modelDir) return { changed: false, note: r.note };
  const key = path.resolve(file).replace(/\\/g, '/');
  const old = ((st.trace && st.trace.pairs) || {})[key];
  // Keep an earlier same-model check only if it was about this same folder.
  const rec = old && typeof old === 'object' && old.modelDir === r.modelDir ? old : { gguf: key, modelDir: r.modelDir, date: null, ids_equal: null, top5_overlap: null };
  const pairs = { ...((st.trace && st.trace.pairs) || {}), [key]: rec };
  const cur = (st.trace.modelDir || '').replace(/\\/g, '/');
  if (cur === r.modelDir) { config.update({ trace: { pairs } }); return { changed: false, modelDir: r.modelDir, how: r.how }; }
  await tracer.restart({ modelDir: r.modelDir });
  config.update({ trace: { modelDir: r.modelDir, pairs } });
  log(`inside engine now on ${path.basename(r.modelDir)} (${r.how})`);
  return { changed: true, modelDir: r.modelDir, how: r.how };
}

// What a user (or Prometheus) needs to see when something is off: both engines, and whether they agree.
function check({ config, engine, file }) {
  const st = config.settings, out = { local: file || st.lastModel || null, inside: (st.trace && st.trace.modelDir) || null, auto: !(st.trace && st.trace.auto === false), problems: [], optional: [] };
  if (engine && !engine.exe) out.problems.push(engine.binDir ? `llama-server was not found in ${engine.binDir} (Settings: llama.cpp folder, or PSI_LLAMA_DIR)` : 'llama-server was not found: set the llama.cpp folder in Settings (or PSI_LLAMA_DIR), or put llama-server on the PATH');
  if (!out.local) out.problems.push('no local model chosen yet: pick a .gguf file in Settings > Models');
  else if (!fs.existsSync(out.local)) out.problems.push(`the local model file does not exist: ${out.local}`);
  // Not set up is not broken: Generate and the other basic views work without it. A folder that is set but missing or empty is a problem.
  if (!out.inside) out.optional.push('the inside engine is not set up (optional, for Neural net, Anatomy and the other inside views): no weights folder (Settings: a Hugging Face model folder, or PSI_TRACE_MODEL)');
  else if (!fs.existsSync(out.inside)) out.problems.push(`the inside engine's weights folder does not exist: ${out.inside}`);
  else if (!hasWeights(out.inside)) out.problems.push(`the inside engine's folder has no model weights (no .safetensors): ${out.inside}`);
  if (out.local && fs.existsSync(out.local) && out.inside && fs.existsSync(out.inside)) {
    try {
      const a = shape(header(out.local)), b = weightsShape(out.inside);
      out.localShape = a; out.insideShape = b;
      if (!same(a, b)) out.problems.push(`the two engines hold different models: local ${a.layers} layers × ${a.width}, inside ${b ? `${b.layers} layers × ${b.width}` : 'unreadable'}; Neural net and Anatomy are not showing the model that Generate uses`);
    } catch (e) { out.problems.push(`could not read a header: ${e.message}`); }
  }
  out.engineRunning = !!(engine && engine.proc);
  out.ok = !out.problems.length;
  if ((!out.ok || out.optional.length) && out.local && fs.existsSync(out.local)) {
    // What would fix it: the weights folder that matches the local model, without converting anything here.
    const a = shape(header(out.local)), name = norm(path.basename(out.local));
    const roots = [...new Set([st.modelsDir, path.dirname(out.local)].filter(Boolean))];
    const all = roots.flatMap(candidates).map((d) => d.replace(/\\/g, '/'));
    const match = all.find((d) => norm(path.basename(d)) === name && same(a, weightsShape(d))) || all.find((d) => same(a, weightsShape(d)));
    const small = fs.statSync(out.local).size <= SMALL;
    out.fix = match ? { weights_folder: match, how: 'set the inside engine’s weights folder to this, then save and restart the engine; or turn on "follow the local model" and load the model again' }
      : small ? { how: 'turn on "follow the local model" and load the model again: it is small, so it is converted once automatically' }
      : { how: `no weights folder on this computer matches ${path.basename(out.local)}; it must be downloaded or unpacked into a folder under ${st.modelsDir || 'the models folder'}` };
    out.weight_folders = all.slice(0, 20);
  }
  return out;
}

// Do the two engines hold the same model? Tokenize three fixed English sentences on both (are the ids equal?) and
// compare the five most likely next tokens after each (overlap 0..1, mean over the sentences). Needs both engines
// running; neither is started here. The result is kept in settings.trace.pairs under the GGUF path.
const SAME_TEXTS = ['The cat sat on the mat because it was warm.', 'In 1969, two astronauts walked on the surface of the',
  'Water boils at one hundred degrees Celsius at sea level, and'];
async function sameModel({ engine, tracer, config }) {
  const gguf = engine && engine.model ? path.resolve(engine.model).replace(/\\/g, '/') : null;
  const modelDir = tracer ? tracer.modelDir : null;
  // When it cannot be checked now, the last check of this pair (if any) comes with it as `last`, dated.
  const kept = gguf && config && config.settings.trace && config.settings.trace.pairs ? config.settings.trace.pairs[gguf] : null;
  const last = kept && typeof kept === 'object' && kept.date ? kept : null;
  if (!engine || !engine.proc || !gguf) return { checked: false, reason: 'the local engine is not running', gguf, modelDir, last };
  const st = tracer && (await tracer.state());
  if (!st) return { checked: false, reason: 'the inside engine is not running (it starts on the first Neural net, Anatomy or experiment request)', gguf, modelDir, last };
  const v = tracer.verify(st);
  const per = [];
  // Plus one chat prompt in the model's own template: plain sentences tokenize alike even when the inside engine's
  // tokenizer lacks the template's special tokens (a GGUF-rebuilt Qwen3 tokenizer cut <think> into 3 pieces: ITERATIONS 13).
  let chat = null;
  try { chat = await engine.applyTemplate([{ role: 'user', content: 'Who wrote Hamlet?' }]); } catch (_) { /* no template: plain texts only */ }
  try {
    for (const text of chat ? [...SAME_TEXTS, chat] : SAME_TEXTS) {
      const a = await engine.tokenize(text, text === chat);
      const b = await tracer.nextTop(text, 5);
      const ia = a.map((t) => t.id), ib = b.ids || [];
      const na = await engine.next(ia, 5);
      const ta = new Set(na.top.map((t) => t.id)), tb = new Set((b.top || []).map((t) => t.id));
      const overlap = [...ta].filter((id) => tb.has(id)).length / Math.max(1, Math.min(5, ta.size, tb.size));
      per.push({ text, chat_template: text === chat, ids_equal: ia.length === ib.length && ia.every((x, i) => x === ib[i]), local_ids: ia.length, inside_ids: ib.length,
        top5_overlap: +overlap.toFixed(2), local_top: na.top.map((t) => t.piece), inside_top: (b.top || []).map((t) => t.piece) });
    }
  } catch (e) { return { checked: false, reason: `the check failed: ${e.message}`, gguf, modelDir: (st.conditions && st.conditions.model_dir) || modelDir, last }; }
  const rec = { date: new Date().toISOString(), ids_equal: per.every((x) => x.ids_equal), top5_overlap: +(per.reduce((a, x) => a + x.top5_overlap, 0) / per.length).toFixed(2),
    gguf, modelDir: (st.conditions && st.conditions.model_dir ? String(st.conditions.model_dir).replace(/\\/g, '/') : modelDir) };
  if (config) config.update({ trace: { pairs: { ...((config.settings.trace && config.settings.trace.pairs) || {}), [gguf]: rec } } });
  // Same tokenizer and at least 4 of the top 5 in common on average: the same model (allowing for quantisation).
  const ch = per.find((x) => x.chat_template);
  const chatNote = ch && !ch.ids_equal ? `The two engines split the chat template differently (${ch.local_ids} pieces here, ${ch.inside_ids} inside): the inside engine's tokenizer is missing some of the model's special tokens, so it reads prompts the model was not trained on. Use the weights folder's own tokenizer files from the model's Hugging Face repo.` : null;
  return { checked: true, ...rec, same: rec.ids_equal && rec.top5_overlap >= 0.8, inside_note: v.note || null, chat_note: chatNote, sentences: per };
}

// header/parser: the GGUF header in Node, also for engine.js (mixture-of-experts placement).
const parser = () => { if (!PARSER) { const g = {}; new Function('window', fs.readFileSync(path.join(__dirname, 'ui', 'gguf-header.js'), 'utf8'))(g); PARSER = g.PSIHeader; } return PARSER; };
module.exports = { pairFor, syncTrace, check, norm, sameModel, pairDir, header, parser };
