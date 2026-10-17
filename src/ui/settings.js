// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Settings view: local model, thinking in E-levels (persisted), the inside-the-model engine, and
// external models through OpenRouter (key kept server side, model picker, judge, spending cap).
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const err = (e, o) => PSI.err(e, o);
  let S = null, MODELS = [];
  const ext = () => (S && S.external) || {};
  // A setting that did not save is worth knowing: it would quietly be back to the old value next time.
  const unsaved = (what) => (e) => err(e, { text: `${what} was not saved: ${PSI.explain(e).text}` });
  const effect = (after, text, id) => { let p = id && $(id); if (!p) { p = el('p', 'effect small muted'); if (id) p.id = id; after.after(p); } p.textContent = text; return p; };

  // ── persist the lens and steering controls (they live in this view now) ──
  const LENS = [['lensOn', 'lens', 'on', 'checked'], ['thinkOn', 'lens', 'think', 'checked'], ['lensText', 'lens', 'text', 'value'],
    ['cvScale', 'steering', 'scale', 'value'], ['cvFrom', 'steering', 'from', 'value'], ['cvTo', 'steering', 'to', 'value']];
  function bindLens() {
    LENS.forEach(([id, grp, key, prop]) => {
      const node = $(id); if (!node) return;
      const v = S[grp] && S[grp][key];
      if (v !== undefined && v !== '' && !(prop === 'value' && id === 'lensText' && !v)) { node[prop] = prop === 'checked' ? !!v : v; node.dispatchEvent(new Event(prop === 'checked' ? 'change' : 'input')); }
      node.addEventListener(prop === 'checked' ? 'change' : 'input', () => { const val = prop === 'checked' ? node.checked : (node.type === 'range' || node.type === 'number' ? +node.value : node.value); psi.settings({ [grp]: { [key]: val } }).then((v2) => { S = v2; status(); }).catch(unsaved('The lens and steering setting')); });
    });
  }
  // Emergence Machine content (ui/em.js): the switch saves it; the lens follows it (config.js update), so its switch is set here too.
  function bindEm() {
    const n = $('stEmOn'); if (!n) return; n.checked = !(S.em && S.em.on === false);
    n.addEventListener('change', async () => {
      try {
        S = await psi.settings({ em: { on: n.checked } });
        if ($('lensOn') && $('lensOn').checked !== S.lens.on) { $('lensOn').checked = S.lens.on; $('lensOn').dispatchEvent(new Event('change')); }
        if (window.PSIEM) PSIEM.set(n.checked); status();
      } catch (e) { n.checked = !n.checked; err(e); }
    });
  }
  function status() {
    const g = $('genStatus'); if (!g || !S) return;
    const lens = $('lensOn') ? $('lensOn').checked : S.lens.on, lora = $('loraOn') && $('loraOn').checked, cv = +($('cvScale') ? $('cvScale').value : 0);
    const st = window.PSI ? PSI.state : {}, base = (f) => String(f || '').split(/[\\/]/).pop();
    const now = st.model ? base(st.model) : st.loading ? 'loading the model…' : 'no model loaded';
    // What the loaded model runs with (applied) is kept apart from what the controls say (set): a saved slider is not a steered model.
    const ch = (st.changes || {}), steer = ch.steer, on = steer ? steer.scale : 0;
    // The thinking switch is hidden (app.js thinkSwitch) when the model's chat format has no "/no_think": it thinks as it does.
    const tk = $('thinkOn'), noSwitch = !tk || (tk.closest('label') && tk.closest('label').classList.contains('hidden'));
    const parts = [`lens ${lens ? 'on' : 'off'}`, noSwitch && st.model ? 'thinks as the model does' : `thinking ${tk && tk.checked ? 'shown' : 'hidden'}`];
    if (ch.lora) parts.push('LoRA applied'); else if (lora) parts.push('LoRA on at the next load');
    if (steer) parts.push(steer.short);
    if (cv !== on && st.model) parts.push(`steering slider at ${cv.toFixed(2)}, not applied (press Apply)`); else if (cv && !st.model) parts.push(`steering ${cv.toFixed(2)} not applied (needs a loaded model and Apply)`);
    g.textContent = ''; g.appendChild(document.createTextNode(`${now} · ${parts.join(' · ')} · `));
    g.classList.toggle('cond-changed', !!(steer || ch.lora));
    g.appendChild(st.model || st.loading ? PSI.link('change in Settings', 'settings') : PSI.loadModelLink());
    // Loaded means the engine is running it now; the saved lastModel is only what was used last.
    const moe = st.moe && st.moe.layers > 0 ? ` · the experts of ${st.moe.layers} of ${st.moe.of} layers run on the processor, so it fits the GPU (${st.moe.why.split(': ')[0]})` : '';
    $('stModelNow').textContent = st.model ? `Loaded: ${st.model}${moe}` : st.loading ? 'Loading…' : S.lastModel ? `Last used: ${S.lastModel} (not loaded)` : 'No model loaded.';
    window.PSISettingsModel = st.model || '';
  }

  // ── local model ─────────────────────────────────────────────────────────
  // The list holds GGUF files (the answering engine runs these) and Hugging Face folders, marked ⚠: llama.cpp cannot
  // load a folder, so choosing one swaps Load for Convert to GGUF (src/convert-api.js), which makes a file once.
  const LISTED = new Map();
  const gbTxt = (b) => `${(b / 2 ** 30).toFixed(b < 2 ** 30 ? 2 : 1)} GB`;
  const fmtP = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} billion` : `${Math.round(n / 1e6)} million`);
  async function scan(dir, select) {
    // No folder chosen yet (a new install, or someone who only builds Commons recipes): say so in the list, not as an error.
    if (!(dir || S.modelsDir)) { const sel = $('stModelPick'); sel.textContent = ''; sel.appendChild(el('option', null, 'no models folder yet: enter one above and press Scan')); return; }
    try {
      const r = await psi.listModels(dir || S.modelsDir); const sel = $('stModelPick'); sel.textContent = ''; LISTED.clear();
      const want = (select || S.lastModel || '').replace(/\\/g, '/');
      const add = (host, m, text) => { const o = el('option', null, text); o.value = m.path; if (m.path === want) o.selected = true; host.appendChild(o); LISTED.set(m.path, m); };
      const g1 = el('optgroup'); g1.label = 'GGUF files: the answering engine runs these';
      r.models.forEach((m) => add(g1, m, `${m.name}  (${m.gb} GB)`));
      if (!r.models.length) g1.appendChild(el('option', null, `no .gguf files in ${r.dir}`));
      sel.appendChild(g1);
      if ((r.hf || []).length) {
        const g2 = el('optgroup'); g2.label = '⚠ Hugging Face folders: not GGUF, convert first';
        r.hf.forEach((m) => add(g2, m, `⚠ ${m.name} (Hugging Face folder, ${m.gb} GB): not GGUF${m.has_gguf ? ' · converted file exists' : ''}`));
        sel.appendChild(g2);
      }
      $('stModelsDir').title = `${r.models.length} GGUF files and ${(r.hf || []).length} Hugging Face folders${(r.skipped || []).length ? `; left out: ${r.skipped.map((x) => `${x.path} (${x.why})`).join('; ')}` : ''}`;
      picked();
    } catch (e) { err(e); }
  }
  async function load(path) {
    // lastModel is saved by the app only after the engine has started it (main.js / devserver.js: start).
    if (!path) return; try { if (window.PSIOpenModel) await window.PSIOpenModel(path); S = await psi.settings(); status(); } catch (e) { err(e); }
  }
  // Load a file the app just made, unless that would crowd a long GPU job or not fit (the same check as the reopen at launch).
  async function loadHeld(file, why0) {
    const why = window.PSIMachine ? await window.PSIMachine.holdBack(file).catch(() => null) : null, w = $(why0);
    if (!why) { if (w) w.textContent = ''; return load(file); }
    w.textContent = `not loaded: ${why}. Loading it beside that would spill into system memory and slow both. `;
    const a = el('a', 'psi-link', 'Load anyway'); a.href = '#'; a.addEventListener('click', (e) => { e.preventDefault(); w.textContent = ''; load(file); }); w.appendChild(a);
  }
  const conv = { plan: null, job: null, prog: null, timer: 0, fails: 0, last: null };
  const cur = () => LISTED.get($('stModelPick').value) || null;
  // What the choice in the list allows: Load for a GGUF file, the converter for a folder.
  function picked() {
    const m = cur(), hf = !!(m && m.kind === 'hf');
    $('stModelLoad').classList.toggle('hidden', hf); $('stConvert').classList.toggle('hidden', !hf);
    const le = $('stModelEffect'); if (le) le.classList.toggle('hidden', hf);
    if (!hf) return;
    $('stConvWhy').textContent = `⚠ ${m.name} is a Hugging Face folder (${m.arch || 'architecture not named'}, ${m.gb} GB of ${m.format}), not a GGUF file. The answering engine runs GGUF files. This folder can be converted once; the inside engine (Neural net, Anatomy) can already read it.`;
    const has = m.has_gguf; $('stConvHasRow').classList.toggle('hidden', !has);
    if (has) $('stConvHas').textContent = `already converted: ${has}${(m.converted || []).length > 1 ? ` (and ${m.converted.length - 1} more)` : ''}`;
    if (!conv.job) { $('stConvOut').textContent = ''; $('stConvLoadRow').classList.add('hidden'); PSI.status($('stConvStat'), ''); const c = $('stConvOut').nextElementSibling; if (c && c.classList.contains('cond')) c.remove(); }
    plan();
  }
  async function plan() {
    const m = cur(); if (!m || m.kind !== 'hf') return;
    const fx = $('stConvEffect'); fx.textContent = 'working out the size and the disk space…';
    let p; try { p = await psi.convert({ action: 'plan', folder: m.path, precision: $('stConvPrec').value }); } catch (e) { conv.plan = null; fx.textContent = `Cannot convert: ${PSI.explain(e).text}`; $('stConvGo').disabled = true; return; }
    if (cur() !== m) return;   // the choice changed while this was asked
    conv.plan = p;
    const two = p.intermediate_bytes ? ` First a 16-bit file of about ${gbTxt(p.intermediate_bytes)} is written beside it, then llama-quantize makes the 4-bit file from it and the 16-bit file is deleted.` : '';
    const free = p.free_bytes == null ? 'free space not read' : `${gbTxt(p.free_bytes)} free`;
    fx.textContent = `Writes one new file, ${p.out}: about ${gbTxt(p.est_bytes)} (an estimate from ${fmtP(p.params_estimate)} parameters, ${p.params_basis}). Needs about ${gbTxt(p.need_bytes)} of disk; ${free}.${two} Uses the CPU, not the GPU (llama.cpp’s converter), ${p.params_estimate < 1e9 ? 'for under a minute at this size' : `for several minutes at this size (around 10 to 30 for a 7-billion model)`}. Nothing else changes: the folder, the loaded model and the settings stay as they are; Stop deletes the half-written file.${p.out_exists ? ' That file already exists: converting again replaces it, and you will be asked first.' : ''}${p.problems.length ? ` Cannot convert now: ${p.problems.join('; ')}.` : ''}`;
    $('stConvGo').disabled = !p.ok || !!conv.job;
  }
  $('stModelPick').addEventListener('change', picked);
  $('stConvPrec').addEventListener('change', plan);
  $('stConvLoadExisting').addEventListener('click', () => { const m = cur(); if (m && m.has_gguf) loadHeld(m.has_gguf, 'stConvLoadWhy').then(() => scan(null, m.has_gguf)); });
  $('stConvLoadNew').addEventListener('click', () => { if (conv.last) loadHeld(conv.last.file, 'stConvLoadWhy'); });
  $('stConvGo').addEventListener('click', async () => {
    const m = cur(), p = conv.plan; if (!m || !p || conv.job) return;
    let overwrite = false;
    if (p.out_exists) { if (!confirm(`${p.out} already exists. Replace it with a new conversion? The old file is deleted only when the new one is complete.`)) return; overwrite = true; }
    $('stConvGo').disabled = true; $('stConvOut').textContent = ''; $('stConvLoadRow').classList.add('hidden');
    PSI.status($('stConvStat'), `converting ${m.name} to ${p.label}…`, 'busy');
    try { const r = await psi.convert({ action: 'start', folder: m.path, precision: p.precision, overwrite }); follow(r.job, r.stages); }
    catch (e) { PSI.status($('stConvStat'), `not started: ${PSI.explain(e).text}`, 'error'); $('stConvGo').disabled = false; }
  });
  function convProg(stages, step) {
    if (conv.prog) conv.prog.remove();
    conv.prog = PSI.progress($('stConvProg'), { total: 0, label: stages > 1 ? `step ${step} of 2 · tensor` : 'tensor', view: 'settings', track: false,
      onCancel: () => { if (conv.job) psi.convert({ action: 'cancel', job: conv.job }).catch((e) => err(e)); PSI.status($('stConvStat'), 'stopping: ending the converter and deleting the half-written file…', 'busy'); } });
    conv.prog.stage = step;
  }
  function follow(id, stages) { conv.job = id; conv.fails = 0; convProg(stages || 1, 1); poll(); }
  async function poll() {
    clearTimeout(conv.timer);
    let j; try { j = await psi.convert({ action: 'job', job: conv.job }); conv.fails = 0; } catch (e) {
      if (++conv.fails <= 5) { conv.timer = setTimeout(poll, 1000 * 2 ** conv.fails); return; }
      PSI.status($('stConvStat'), 'lost contact with the conversion; it may still be running', 'error'); if (conv.prog) conv.prog.fail('lost contact'); conv.prog = null; conv.job = null; return;
    }
    if (j.stages > 1 && conv.prog && conv.prog.stage !== j.step && !j.finished) convProg(j.stages, j.step);
    const what = j.stage === 'quantize' ? 'llama-quantize: 16-bit to 4-bit' : j.of ? 'writing the GGUF file' : 'reading the model folder';
    if (!j.finished) { if (conv.prog) { conv.prog.setTotal(j.of); conv.prog.step(j.done, what); } PSI.status($('stConvStat'), j.message || `converting ${j.name}…`, 'busy'); conv.timer = setTimeout(poll, 2000); return; }
    seen(j.id); conv.job = null; const ended = conv.prog; conv.prog = null;
    if (j.cancelled || j.state === 'cancelled') { if (ended) ended.fail('stopped'); PSI.status($('stConvStat'), 'stopped: the half-written file was deleted; the folder is unchanged', 'warn'); plan(); return; }
    if (j.state !== 'done' || !j.result) {
      if (ended) ended.fail('stopped by an error');
      PSI.status($('stConvStat'), `conversion failed: ${j.error || j.message}`, 'error'); $('stConvOut').textContent = j.log && j.log.length ? `last lines: ${j.log.slice(-3).join(' · ')}` : '';
      plan(); return;
    }
    showResult(j.result, ended);
  }
  async function showResult(r, ended) {
    conv.last = r;
    if (ended) ended.finish(`done in ${r.seconds} s`, { markDone: false });
    PSI.status($('stConvStat'), `converted: ${r.file}`, 'ok');
    $('stConvOut').textContent = `${r.name} is now a GGUF file the answering engine can load: ${r.file}, ${r.gb} GB, ${r.tensors} tensors${r.replaced ? ' (the old file was replaced)' : ''}. It is selected in the list above.`;
    PSI.cond($('stConvOut'), [`source folder ${r.source}`, r.label, r.arch, `${r.seconds} s on the CPU`, 'llama.cpp convert_hf_to_gguf.py' + (r.quantized_to ? ` + llama-quantize ${r.quantized_to}` : ''),
      ...(r.restored_keys && r.restored_keys.length ? [{ t: `config keys read from config.json: ${r.restored_keys.join(', ')}`, title: 'A newer transformers renamed these settings; the converter was given the names it expects.' }] : [])], 'converted from');
    $('stConvLoadRow').classList.remove('hidden'); $('stConvLoadWhy').textContent = '';
    await scan(null, r.file);
    $('stConvert').classList.remove('hidden');   // keep the result in view while the new file is selected
    ['stConvWhy', 'stConvHasRow', 'stConvRow', 'stConvEffect'].forEach((id) => $(id).classList.add('hidden'));
  }
  // Back to the folder's own view the next time a folder is chosen.
  $('stModelPick').addEventListener('change', () => ['stConvWhy', 'stConvRow', 'stConvEffect'].forEach((id) => $(id).classList.remove('hidden')));
  // The job lives in the app, not the page. On load: a running conversion gets its progress bar back (Stop still works);
  // one that ended while the page was closed shows its result once (the ids shown are remembered in this browser).
  const SEEN = 'psi-convert-seen';
  const seenIds = () => { try { return JSON.parse(localStorage.getItem(SEEN) || '[]'); } catch (_) { return []; } };
  function seen(id) { try { localStorage.setItem(SEEN, JSON.stringify([id, ...seenIds().filter((x) => x !== id)].slice(0, 50))); } catch (_) { /* private mode: it may show again */ } }
  async function reattach() {
    if (!psi.convertStatus) return;
    let all = []; try { all = await psi.convertStatus(); } catch (_) { return; }
    const run = all.find((j) => j.state === 'running');
    if (run) {
      const m = LISTED.get(run.source); if (m) { $('stModelPick').value = m.path; picked(); }
      $('stConvert').classList.remove('hidden'); $('stConvGo').disabled = true;
      PSI.status($('stConvStat'), run.message, 'busy'); conv.job = run.id; conv.fails = 0; convProg(run.stages, run.step); poll(); return;
    }
    const last = all[0], ids = seenIds(); if (!last || ids.includes(last.id) || !last.started_at) return;
    seen(last.id);
    if (last.state === 'done' && last.result) { await showResult(last.result, null); PSI.status($('stConvStat'), `${last.message} (finished while this page was closed)`, 'ok'); return; }
    // Not done: shown beside the folder it came from (a status line stays; the shared message area may be taken by another message).
    const m = LISTED.get(last.source); if (m) { $('stModelPick').value = m.path; picked(); }
    PSI.status($('stConvStat'), `The last conversion (${last.name} to ${last.label}) ${last.state === 'cancelled' ? 'was stopped' : last.state === 'interrupted' ? 'was interrupted: the app restarted, and the half-written file was deleted' : `failed: ${last.error}`}.`, 'warn');
  }
  $('stModelLoad').addEventListener('click', () => load($('stModelPick').value));
  $('stModelPathLoad').addEventListener('click', () => load($('stModelPath').value.trim()));
  $('stModelsScan').addEventListener('click', async () => { const d = $('stModelsDir').value.trim(); if (d) { try { S = await psi.settings({ modelsDir: d }); } catch (e) { return unsaved('The models folder')(e); } } scan(d); });
  effect($('stModelLoad').closest('.row'), 'Load stops the local model now running (with its steering and LoRA) and starts this one; it takes about its file size in GPU memory and becomes the last used model. With “follow the local model” on, the inside engine switches too, converting a small model once into a folder beside its file.', 'stModelEffect');
  // ── context size: how many tokens the local model can hold at once, passed as `ctx` when the engine starts ──
  const CTX_DEFAULT = 8192, CTX_MIN = 256, CTX_MAX = 1048576;
  const ctxNow = () => { const v = S && S.ctx; return Number.isInteger(v) && v >= CTX_MIN && v <= CTX_MAX ? v : CTX_DEFAULT; };
  // Every start goes through psi.start (app.js openModel); a start that names no context gets the saved one.
  const rawStart = psi.start;
  const startWithCtx = (file, o) => rawStart(file, o && o.ctx != null ? o : { ...(o || {}), ctx: ctxNow() });
  try { psi.start = startWithCtx; } catch (_) { /* a frozen bridge: see ctxWired */ }
  const ctxWired = psi.start === startWithCtx;
  window.PSIStartCtx = ctxNow;   // for a caller that builds its own start options
  function ctxEffect() {
    const v = ctxNow(), kv = (n) => (n * 147456 / 2 ** 30).toFixed(1);
    effect($('stCtx').closest('.row'), `Applies at the next Load (the running model keeps the size it started with). The engine reserves GPU memory for the whole context up front, its KV cache: for Qwen3-4B about 144 KB a token, so ${v.toLocaleString()} tokens take about ${kv(v)} GB beside the weights (8,192 ≈ 1.1 GB, 32,768 ≈ 4.5 GB). If that does not fit, the model spills into system memory and runs many times slower (the strip at the top turns red). More than the model was trained on does not help it read further.${ctxWired ? '' : ' This build cannot pass it to the engine yet: models start with 8,192.'}`, 'stCtxEffect');
  }
  $('stCtx').addEventListener('change', async () => {
    const v = Math.round(+$('stCtx').value);
    if (!(Number.isInteger(v) && v >= CTX_MIN && v <= CTX_MAX)) { $('stCtxStat').textContent = `a whole number from ${CTX_MIN} to ${CTX_MAX.toLocaleString()}; kept ${ctxNow().toLocaleString()}`; $('stCtx').value = ctxNow(); return; }
    try { S = await psi.settings({ ctx: v }); $('stCtxStat').textContent = `saved: ${v.toLocaleString()} tokens at the next Load`; ctxEffect(); } catch (e) { unsaved('The context size')(e); $('stCtx').value = ctxNow(); }
  });

  // ── a settings file that could not be read: the defaults were loaded, and the unreadable file was kept ──
  function recovered() {
    const id = 'stRecovered'; let n = $(id);
    if (!S || !S.recovered_from_corrupt) { if (n) n.remove(); return; }
    if (!n) { n = el('div', 'set-block st-recovered'); n.id = id; n.tabIndex = -1; n.setAttribute('role', 'alert'); const first = $('setModel'); first.parentElement.insertBefore(n, first); }
    n.textContent = '';
    n.append(el('h2', null, 'Settings were recovered'),
      el('p', 'small', `The settings file could not be read, so the defaults were loaded: the models folder, the engines’ folders, the lens, the external models and the spending cap below are the defaults, not what you had set. The unreadable file was kept${S.corrupt_copy ? ` as ${S.corrupt_copy}` : ''}; your OpenRouter key is stored apart and was not affected. Check the values below.`));
    const b = el('button', 'ghost', 'I have checked them'); b.type = 'button';
    b.addEventListener('click', async () => { try { S = await psi.settings({ recovered_from_corrupt: false }); recovered(); } catch (e) { unsaved('Dismissing the notice')(e); } });
    n.appendChild(b);
  }

  const openBtn = $('open'); if (openBtn) { openBtn.classList.add('ghost'); openBtn.replaceWith(openBtn.cloneNode(true)); $('open').addEventListener('click', () => document.querySelector('#tabs button[data-tab=settings]').click()); }

  // ── inside-the-model engine ─────────────────────────────────────────────
  $('stTraceRestart').addEventListener('click', async () => {
    const b = $('stTraceRestart'); b.disabled = true; $('stTraceStat').textContent = 'restarting…';
    try {
      const r = await psi.traceRestart({ modelDir: $('stTraceDir').value.trim(), checkpoints: $('stCkpt').value.trim() }); $('stTraceStat').textContent = `saved; the engine restarts with ${r.modelDir.split(/[\\/]/).pop()} on the next request`;
      PSI.setChange('lesion', null);   // the engine reloads its weights from disk: no cut survives a restart
      S = await psi.settings(); PSI.refreshState().catch(() => { /* benign: it keeps the last known state */ });
    } catch (e) { err(e); $('stTraceStat').textContent = 'failed'; } finally { b.disabled = false; }
  });
  effect($('stTraceRestart').closest('.row'), 'Saves both folders to the settings file. The inside engine stops now (any Tipping-point cut with it) and loads the weights again on its next use.');

  // ── external models ─────────────────────────────────────────────────────
  function showKey() { const e = ext(); if (window.PSI && PSI.state.key !== !!e.keySet) PSI.setState({ key: !!e.keySet }); $('stKeyStat').textContent = e.keySet ? `Key saved (${e.keyHint}).` : 'No key saved.'; $('stKeyStat').className = 'small ' + (e.keySet ? '' : 'muted'); $('stSpent').textContent = `spent through this app: $${(+e.spentUsd || 0).toFixed(3)}`; }
  $('stKeySave').addEventListener('click', async () => { const k = $('stKey').value.trim(); if (!k) return; try { await psi.setKey(k); $('stKey').value = ''; S = await psi.settings(); showKey(); } catch (e) { err(e); } });
  $('stKeyRemove').addEventListener('click', async () => { try { await psi.setKey(''); S = await psi.settings(); showKey(); } catch (e) { err(e); } });
  $('stTest').addEventListener('click', async () => {
    $('stTestOut').textContent = 'checking…';
    try { const r = await psi.external({ action: 'status' }); $('stTestOut').textContent = r.ok ? `connected${r.label ? ` as “${r.label}”` : ''} · ${r.accountCredits != null ? `account credit $${(r.accountCredits - r.accountUsed).toFixed(2)} left` : 'account credit not reported'}${r.keyRemaining != null ? ` · key limit $${(+r.keyRemaining).toFixed(2)} left` : ''}` : `failed: ${r.error}`; }
    catch (e) { $('stTestOut').textContent = `failed: ${e.message || e}`; }
  });
  // Agents (MCP): may outside agents change models? Off unless the user turns it on.
  const agentsShow = () => {
    const on = !!(S.agents && S.agents.allow_changes === true), sp = !!(S.agents && S.agents.allow_spend === true);
    $('stAgents').checked = on; $('stAgentsStat').textContent = on ? 'on: agents can cut and grow models' : 'off: agents can read and measure only';
    $('stAgentsSpend').checked = sp; $('stAgentsSpendStat').textContent = sp ? 'on: agents can spend up to the cap' : 'off: agents cannot ask frontier models';
  };
  $('stAgents').addEventListener('change', async () => { try { S = await psi.settings({ agents: { allow_changes: $('stAgents').checked } }); agentsShow(); } catch (e) { unsaved('The agents setting')(e); agentsShow(); } });
  $('stAgentsSpend').addEventListener('change', async () => { try { S = await psi.settings({ agents: { allow_spend: $('stAgentsSpend').checked } }); agentsShow(); } catch (e) { unsaved('The agents setting')(e); agentsShow(); } });
  $('stCap').addEventListener('change', async () => { try { S = await psi.settings({ external: { capUsd: +$('stCap').value || 0 } }); showKey(); } catch (e) { unsaved('The spending cap')(e); } });
  $('stMaxTok').addEventListener('change', async () => { try { S = await psi.settings({ external: { maxTokens: +$('stMaxTok').value || 400 } }); tryEffect(); } catch (e) { unsaved('Max tokens per answer')(e); } });
  $('stSpentReset').addEventListener('click', async () => { try { await psi.external({ action: 'resetSpend' }); S = await psi.settings(); showKey(); } catch (e) { err(e); } });
  $('stSpentReset').title = 'Clears this app’s counter, so the cap starts over. OpenRouter’s own record and your credit do not change.';
  effect($('stSpentReset').closest('.row'), 'The cap stops the app’s own requests (Compare, Try them, the judge, Prometheus) once the counter reaches it. Reset clears the counter only: OpenRouter’s record and your credit do not change.');

  function chosen() {
    const box = $('stChosen'); box.textContent = '';
    (ext().models || []).forEach((id) => { const c = el('span', 'chip', id + ' ×'); c.style.cursor = 'pointer'; c.title = 'remove'; c.addEventListener('click', () => toggle(id)); box.appendChild(c); });
    if (!(ext().models || []).length) box.appendChild(el('span', 'small muted', 'none chosen'));
  }
  async function toggle(id) { const cur = new Set(ext().models || []); cur.has(id) ? cur.delete(id) : cur.add(id); try { S = await psi.settings({ external: { models: [...cur] } }); } catch (e) { return unsaved('The model choice')(e); } chosen(); table(); tryEffect(); }
  // What "Ask the chosen models" will spend, before it is pressed.
  function tryEffect() {
    const ms = ext().models || [], n = ms.length, p = ms.filter((m) => /^chatgpt\//.test(m)).length, paid = n - p;
    effect($('stTry').closest('.row'), !n ? 'Choose models above first; each one asked is one request (paid on OpenRouter, free on your ChatGPT plan).'
      : `${paid ? `Spends OpenRouter credit: one request to each of the ${paid} OpenRouter model${paid > 1 ? 's' : ''}, up to ${ext().maxTokens || 400} tokens per answer (more for models that reason first).` : ''}${paid && p ? ' ' : ''}${p ? `${p} on your ChatGPT plan: no money, counts against the plan’s limits.` : ''}`, 'stTryEffect');
  }
  function table() {
    const f = $('stFind').value.trim().toLowerCase(), tb = $('stModelTable').querySelector('tbody'); tb.textContent = ''; const on = new Set(ext().models || []);
    MODELS.filter((m) => !f || m.id.toLowerCase().includes(f) || (m.name || '').toLowerCase().includes(f)).slice(0, 200).forEach((m) => {
      const tr = el('tr'); const cb = el('input'); cb.type = 'checkbox'; cb.checked = on.has(m.id); const td = el('td'); td.appendChild(cb); tr.appendChild(td);
      [m.id, m.plan ? 'ChatGPT plan' : m.in, m.plan ? '$0' : m.out, m.ctx ? `${Math.round(m.ctx / 1000)}k` : ''].forEach((v) => tr.appendChild(el('td', null, String(v))));
      tr.addEventListener('click', () => toggle(m.id)); tb.appendChild(tr);
    });
    const j = $('stJudge'); const cur = ext().judge; j.textContent = '';
    const ids = MODELS.length ? MODELS.map((m) => m.id) : [cur]; ids.forEach((id) => { const o = el('option', null, id); o.value = id; if (id === cur) o.selected = true; j.appendChild(o); });
  }
  $('stLoadModels').addEventListener('click', async () => { $('stLoadModels').disabled = true; try { MODELS = (await psi.external({ action: 'models' })).models; table(); } catch (e) { err(e); } finally { $('stLoadModels').disabled = false; } });
  $('stFind').addEventListener('input', table);
  $('stJudge').addEventListener('change', async () => { try { S = await psi.settings({ external: { judge: $('stJudge').value } }); } catch (e) { unsaved('The judge model')(e); } });

  async function askAll(messagesFor, out, label) {
    const models = ext().models || []; if (!models.length) throw new Error('Choose external models in Settings first.');
    out.textContent = '';
    await Promise.all(models.map(async (id) => {
      const card = el('div', 'ext-card'); card.appendChild(el('p', 'k', id)); const body = el('div', null, '…'); card.appendChild(body); out.appendChild(card);
      for (const [name, msgs] of messagesFor()) {
        // The cost is what OpenRouter reports; a reply without usage has no known cost (not $0).
        try { const r = await psi.external({ action: 'chat', model: id, messages: msgs }); if (name) body.appendChild(el('p', 'k', name)); if (body.textContent === '…') body.textContent = ''; body.appendChild(el('p', null, r.text || '(empty)')); card.appendChild(el('p', 'meta', `${name ? name + ': ' : ''}${r.plan ? 'ChatGPT plan, no money' : r.tokens || r.cost ? `$${(+r.cost || 0).toFixed(5)}` : 'cost not reported'} · ${r.ms} ms${r.tokens ? ` · ${r.tokens.in} in / ${r.tokens.out} out` : ''}${r.thinking && r.thinking.tokens ? ` (${r.thinking.tokens} thinking)` : ''}${r.retried ? ` · ran out of thinking room once and was retried with more (the first try cost $${(+r.retried.cost).toFixed(5)}, included)` : ''} · ${r.plan ? 'the plan’s own temperature and length' : 'temperature 0'}${r.finish === 'length' ? ' · cut off at max tokens' : ''}`)); }
        catch (e) { body.textContent = ''; body.appendChild(el('p', 'small', `failed: ${e.message || e}`)); break; }
      }
    }));
    S = await psi.settings(); showKey();
  }
  $('stTry').addEventListener('click', async () => {
    const b = $('stTry'); b.disabled = true; $('stTryStat').textContent = 'asking…';
    try { await askAll(() => [['', [{ role: 'user', content: $('stTryQ').value }]]], $('stTryOut')); $('stTryStat').textContent = ''; } catch (e) { err(e); $('stTryStat').textContent = 'failed'; } finally { b.disabled = false; }
  });
  window.PSISettings = {
    compareExternal: (q, lens, out) => askAll(() => [['lens on', [{ role: 'system', content: lens }, { role: 'user', content: q }]], ['lens off', [{ role: 'user', content: q }]]], out),
  };

  // ── the two engines: follow the local model, and a plain-words check of the setup ──
  $('stTraceAuto').addEventListener('change', async () => { try { S = await psi.settings({ trace: { auto: $('stTraceAuto').checked } }); } catch (e) { unsaved('“Follow the local model”')(e); } });
  const NL = String.fromCharCode(10);
  // What each engine can use here (src/hardware.js): the system's GPU, llama.cpp's devices, PyTorch's device, and advice.
  function hwLines(h) {
    if (!h) return [];
    if (h.error) return ['', `hardware: not checked (${h.error})`];
    const g = h.gpu || {}, l = h.llama || {}, t = h.torch || {};
    const gpuTxt = g.gpus && g.gpus.length ? g.gpus.map((x) => `${x.name}${x.totalMB ? ` (${(x.totalMB / 1024).toFixed(1)} GB${g.unified ? ' unified' : ''})` : ''}`).join(', ') + (g.reader ? ` · read with ${g.reader}` : '') : (g.detail || 'none found');
    const llTxt = l.devices && l.devices.length ? l.devices.map((d) => `${d.backend}: ${d.name}`).join(', ') : l.error ? `not checked (${l.error})` : 'no GPU: runs on the processor';
    const tTxt = t.error ? `not available (${t.error})` : t.accelerator ? `${t.accelerator} · ${t.name || ''} · ${t.dtype || ''} · torch ${t.torch}` : `no GPU: the processor, float32 · torch ${t.torch}`;
    return ['', `hardware (${h.platform}):`, `- GPU: ${gpuTxt}`, `- answering engine (llama.cpp): ${llTxt}`, `- inside engine (PyTorch, ${h.python}): ${tTxt}`,
      ...(h.advice || []).map((a) => `→ ${a}`)];
  }
  async function checkSetup() {
    const r = await psi.setupCheck(), o = $('stSetupOut'); o.classList.remove('hidden');
    o.textContent = [`local model: ${r.local || 'none'}${r.localShape ? ` (${r.localShape.layers} layers × ${r.localShape.width})` : ''}`,
      `inside engine: ${r.inside || 'none'}${r.insideShape ? ` (${r.insideShape.layers} layers × ${r.insideShape.width})` : ''} · ${r.inside_engine && r.inside_engine.ok ? 'running' : 'not running (starts on first use)'}`,
      `follow the local model: ${r.auto ? 'on' : 'off'}`, r.ok ? 'no problems found' : `problems:${NL}- ${r.problems.join(`${NL}- `)}`, ...hwLines(r.hardware)].join(NL);
    return r;
  }
  $('stSetupCheck').addEventListener('click', () => checkSetup().catch(err));
  window.PSISetup = { check: checkSetup, refresh: async () => { S = await psi.settings(); $('stTraceDir').value = S.trace.modelDir || ''; PSI.refreshState().catch(() => {}); } };

  // ── start ───────────────────────────────────────────────────────────────
  psi.settings().then((v) => {
    S = v; bindLens(); bindEm(); scan().then(reattach); status(); showKey(); chosen(); table();
    $('stModelsDir').value = S.modelsDir || ''; $('stTraceDir').value = S.trace.modelDir || ''; $('stCkpt').value = S.trace.checkpoints || ''; $('stTraceAuto').checked = S.trace.auto !== false;
    $('stCap').value = ext().capUsd; $('stMaxTok').value = ext().maxTokens; tryEffect(); agentsShow();
    $('stCtx').value = ctxNow(); ctxEffect(); recovered();
    if (S.recovered_from_corrupt) err('Your settings file could not be read.', { text: `The defaults were loaded and the unreadable file was kept${S.corrupt_copy ? ` (${S.corrupt_copy})` : ''}. Check Settings, including the spending cap.`, action: { label: 'Open Settings', view: 'settings', focus: 'stRecovered' } });
  }).catch(err);
  ['lensOn', 'thinkOn', 'loraOn', 'cvScale'].forEach((id) => $(id) && $(id).addEventListener('change', status));
  PSI.onState(() => { if (S) status(); });
  ['stTest', 'stTry'].forEach((id) => PSI.requires($(id), 'key'));

  // The version, under Help and about (version.js loads after this file; an inline script is blocked by the page's CSP).
  addEventListener('load', () => { const v = window.PSI_VERSION, n = $('stVersion'); if (v && n) n.textContent = `Version ${v.version} (${v.date}). Made by Sail with Claude.`; });
})();
