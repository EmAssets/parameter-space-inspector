// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Getting things from Hugging Face, each where it is used (src/hf-api.js, src/hf_pull.py). Every part here is marked
// data-hf in index.html and stays hidden until Settings > Hugging Face has a working token.
//   Settings > Models    Get a model: search (tagged emergence-machine first; "all of Hugging Face" to widen), its
//                             card (size and whether it fits, licence, which engine reads it, its family, PSI records),
//                             then the weights folder or one GGUF into the models folder, pinned to a commit and checked
//   Settings > inside engine  the repo training checkpoints come from; the local model's weights folder found from its GGUF
//   Emergence                 a training step on demand, and "narrow it": fetch the step in the middle of the bracket
//   Features                  an SAE folder from a repo, then the import there
//   E-levels (LoRA)           a GGUF adapter or control vector, then used at once
//   Commons > Contribute      text from a dataset (with its licence) into the training box
(function () {
  const $ = (id) => document.getElementById(id);
  if (!window.psi || typeof psi.hf !== 'function') return;
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const stat = (node, text, kind) => node && PSI.status(node, text, kind);
  const gb = (b) => (b == null ? '?' : b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GB` : `${Math.max(1, Math.round(b / 2 ** 20))} MB`);
  const hf = (a) => psi.hf(a);
  const link = (repo, kind) => { const a = el('a', null, repo); a.href = `https://huggingface.co/${kind === 'dataset' ? 'datasets/' : ''}${repo}`; a.target = '_blank'; a.rel = 'noopener'; return a; };

  // A long job (download): poll until it ends; the status line shows its stage. Resolves to the finished job.
  async function follow(node, stopBtn) {
    if (stopBtn) { stopBtn.hidden = false; stopBtn.onclick = () => hf({ action: 'cancel' }).catch(() => {}); }
    for (;;) {
      let j; try { j = await hf({ action: 'job' }); } catch (_) { await new Promise((r) => setTimeout(r, 3000)); continue; }
      if (j && j.state === 'running') { stat(node, `${j.stage}${j.of ? ` · ${gb(j.done)} of ${gb(j.of)}` : ''} · ${j.seconds}s`, 'busy'); await new Promise((r) => setTimeout(r, 1500)); continue; }
      if (stopBtn) stopBtn.hidden = true;
      if (!j || j.state !== 'done') { stat(node, j && j.state === 'cancelled' ? 'Stopped; the half-downloaded files were deleted.' : `Failed: ${(j && j.error) || 'unknown'}`, 'error'); return null; }
      return j;
    }
  }
  async function start(a, node, stopBtn) {
    try { await hf(a); } catch (e) { stat(node, `${e.message || e}`, 'error'); return null; }
    return follow(node, stopBtn);
  }

  // ── a search box: tagged emergence-machine by default, all of Hugging Face on request ──
  function finder(box, kind, onPick) {
    const row = el('div', 'row'), q = el('input', 'grow'), all = el('input'), lab = el('label', 'small muted'), go = el('button', 'ghost hf-send', 'Search');
    q.placeholder = kind === 'dataset' ? 'search datasets, or owner/name' : 'search models, or owner/name'; q.setAttribute('aria-label', `search ${kind}s on Hugging Face`);
    all.type = 'checkbox'; lab.title = 'Off: only what PSI published (tagged emergence-machine): models, cut models, recipes, branches.'; lab.append(all, ' all of Hugging Face');
    const say = el('span', 'small muted'), list = el('div', 'cm-list');
    row.append(q, lab, go); box.append(row, say, list);
    const run = async () => {
      const v = q.value.trim(); list.textContent = '';
      if (/^[\w.-]+\/[\w.-]+$/.test(v)) { stat(say, ''); return onPick(v); }
      stat(say, 'searching…', 'busy');
      let r; try { r = await hf({ action: 'search', kind, query: v, all: all.checked, limit: 20 }); } catch (e) { return stat(say, e.message || String(e), 'error'); }
      stat(say, r.items.length ? `${r.items.length} found, ${r.scope}` : `nothing ${r.scope}${all.checked ? '' : ': tick “all of Hugging Face” to look further'}`, r.items.length ? '' : 'warn');
      for (const it of r.items) {
        const b = el('button', 'ghost small'); b.type = 'button';
        b.textContent = `${it.repo}${it.em ? ' · EM' : ''}${it.gated ? ' · gated' : ''} · ${(it.downloads || 0).toLocaleString()} downloads`;
        b.addEventListener('click', () => onPick(it.repo)); list.appendChild(b);
      }
    };
    go.addEventListener('click', run); q.addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
    return { set: (v) => { q.value = v; run(); } };
  }

  // ── Settings > Models: get a model ──
  const mbox = $('hfModelBox');
  if (mbox) {
    const card = el('div', 'out minh-0');
    // Browse on Hugging Face itself (a search prefilled for what the local engine runs: GGUF text-generation models no
    // larger than this GPU holds), then paste the model's address here.
    const browse = el('p', 'small');
    const go = (label, url) => { const a = el('a', null, `${label} ↗`); a.href = url; a.target = '_blank'; a.rel = 'noopener'; return a; };
    const HFM = 'https://huggingface.co/models?';
    const base = { library: 'gguf', pipeline_tag: 'text-generation', sort: 'trending' };
    const linkAll = go('Browse compatible models on Hugging Face', HFM + new URLSearchParams(base));
    browse.append(linkAll, el('span', 'muted', ' · '), go('made with PSI', HFM + new URLSearchParams({ other: 'emergence-machine', sort: 'trending' })));
    psi.gpu().then((g) => {   // cap the size at what fits: at 8 bits a parameter is about a byte, so GB of GPU memory ≈ billions of parameters
      const gbv = g && g.totalMB ? Math.max(1, Math.floor(g.totalMB / 1024)) : null;
      if (gbv) { linkAll.href = HFM + new URLSearchParams({ ...base, num_parameters: `min:0,max:${gbv}B` }); linkAll.title = `GGUF text-generation models up to ${gbv}B parameters (fits ${gbv} GB of GPU memory at 8 bits)`; }
    }).catch(() => {});
    const row = el('div', 'row'), url = el('input', 'grow'), get = el('button', 'hf-send', 'Get');
    url.placeholder = 'paste the model’s address: https://huggingface.co/owner/name (or a link to one .gguf file)'; url.setAttribute('aria-label', 'Hugging Face model address');
    const say = el('span', 'small muted');
    row.append(url, get); mbox.append(browse, row, say, card);
    // https://huggingface.co/owner/name[/blob|resolve|tree/<rev>/<file>][?…], hf://owner/name, or owner/name
    const parse = (v) => {
      const t = String(v || '').trim().replace(/[?#].*$/, '').replace(/^hf:\/\//, '').replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//, '');
      if (/^datasets\/|^spaces\//.test(t)) return { error: 'that is a dataset or a Space, not a model' };
      const m = /^([\w.-]+\/[\w.-]+)(?:\/(?:blob|resolve|tree)\/[^/]+\/(.+))?\/?$/.exec(t);
      return m ? { repo: m[1], file: m[2] && /\.gguf$/i.test(m[2]) ? decodeURIComponent(m[2]) : null } : { error: 'not a Hugging Face model address' };
    };
    const open = () => { const r = parse(url.value); if (r.error) return stat(say, r.error, 'warn'); stat(say, ''); showModel(r.repo, r.file); };
    get.addEventListener('click', open); url.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(); });
    const pick = { set: (repo) => { url.value = `https://huggingface.co/${repo}`; open(); } };
    async function fit(bytes) {
      let g = null; try { g = await psi.gpu(); } catch (_) { /* unknown */ }
      if (!g || !bytes) return 'fit not known';
      const need = bytes / 2 ** 20 * 1.2, free = g.freeMB, ram = g.ram && g.ram.totalMB;
      if (free != null && need < free) return `fits your GPU (${Math.round(free)} MB free)`;
      if (ram && need < ram * 0.6) return `too large for the GPU now${free != null ? ` (${Math.round(free)} MB free)` : ''}; runs on the CPU, slowly`;
      return 'probably too large for this computer';
    }
    async function showModel(repo, want) {
      card.textContent = ''; const say = el('p', 'small muted'); card.appendChild(say); stat(say, `reading ${repo}…`, 'busy');
      let i; try { i = await hf({ action: 'info', repo }); } catch (e) { return stat(say, e.message || String(e), 'error'); }
      say.remove();
      const h = el('p', 'small'); h.append(link(i.repo), ` · ${i.licence ? `licence ${i.licence}` : 'licence not stated'}${i.gated ? ' · gated: accept its terms on its page first' : ''}${i.em ? ' · published from PSI' : ''} · ${(i.downloads || 0).toLocaleString()} downloads`);
      card.appendChild(h);
      card.appendChild(el('p', 'small muted', `${i.architectures ? i.architectures.join(', ') : i.model_type || 'architecture not stated'} · inside engine: ${i.inside_engine}${i.unsafe_files_skipped.length ? ` · pickle files never downloaded: ${i.unsafe_files_skipped.join(', ')}` : ''}`));
      const acts = el('div', 'row'), out = el('p', 'small'), stop = el('button', 'ghost', 'Stop'); stop.hidden = true;
      if (i.folder && !i.remote_code) {
        const b = el('button', 'hf-send', `Get the weights folder (${gb(i.folder.bytes)})`); acts.appendChild(b);
        fit(i.folder.bytes).then((t) => { b.title = t; out.textContent = `Weights folder: ${t}.`; });
        b.addEventListener('click', async () => {
          const j = await start({ action: 'get_model', repo: i.repo }, out, stop); if (!j) return;
          out.textContent = `Downloaded to ${j.result.dest} (commit ${j.result.commit.slice(0, 12)}, every file checked). `;
          const use = el('button', 'ghost small', 'Use it for the inside engine'); use.addEventListener('click', async () => { try { const r = await psi.traceRestart({ modelDir: j.result.dest }); stat(out, `The inside engine now reads ${r.modelDir}.`, 'ok'); } catch (e) { stat(out, e.message || String(e), 'error'); } });
          out.appendChild(use); if ($('stModelsScan')) $('stModelsScan').click();
        });
      }
      if (i.gguf.length) {
        const sel = el('select'); sel.setAttribute('aria-label', 'GGUF file');
        for (const g of i.gguf) { const o = el('option', null, `${g.name} (${gb(g.bytes)})`); o.value = g.name; if (g.name === want) o.selected = true; sel.appendChild(o); }
        const b = el('button', 'hf-send', 'Get this GGUF'); acts.append(sel, b);
        b.addEventListener('click', async () => {
          const j = await start({ action: 'get_model', repo: i.repo, gguf: sel.value }, out, stop); if (!j) return;
          stat(out, `Downloaded ${sel.value} to ${j.result.dest}. It is in the model list above (scanned).`, 'ok'); if ($('stModelsScan')) $('stModelsScan').click();
        });
      }
      if (!i.folder && !i.gguf.length) acts.appendChild(el('span', 'small muted', 'Nothing PSI can load here: no safetensors weights and no GGUF file.'));
      const fam = el('button', 'ghost hf-send', 'Its family'); acts.append(fam, stop); card.append(acts, out);
      fam.addEventListener('click', async () => {
        fam.disabled = true; const f = el('div', 'small'); card.appendChild(f); f.textContent = 'reading its family…';
        let r; try { r = await hf({ action: 'family', repo: i.repo }); } catch (e) { f.textContent = e.message || String(e); return; }
        f.textContent = '';
        const line = (label, items) => { if (!items || !items.length || items.error) return; const p = el('p', 'small'); p.append(`${label}: `); items.forEach((x, k) => { const r2 = typeof x === 'string' ? x : x.repo; const b = el('button', 'ghost small', `${r2}${x.em ? ' · EM' : ''}`); b.type = 'button'; b.addEventListener('click', () => pick.set(r2)); p.append(b); if (k < items.length - 1) p.append(' '); }); f.appendChild(p); };
        line('made from', r.base_models); line('fine-tunes', r.made_from_it.finetune); line('quantised', r.made_from_it.quantized); line('adapters', r.made_from_it.adapter); line('merges', r.made_from_it.merge);
        if (!f.childNodes.length) f.textContent = 'No base model recorded and nothing made from it.';
      });
      for (const rec of i.records) {
        const p = el('p', 'small'); p.append(`PSI record: ${rec} `); card.appendChild(p);
        if (rec.endsWith('recipe.json')) { const b = el('button', 'ghost small hf-send', 'Import the recipe in Commons'); b.addEventListener('click', () => { if ($('cmAddr')) { $('cmAddr').value = `hf://${i.repo}/${rec}`; document.querySelector('[data-tab=commons]').click(); setTimeout(() => $('cmImport').click(), 300); } }); p.appendChild(b); }
        else { const b = el('button', 'ghost small hf-send', 'Read it'); b.addEventListener('click', async () => { try { const r = await hf({ action: 'record', repo: i.repo, file: rec }); const pre = el('pre', 'small'); pre.textContent = JSON.stringify(r.record.cut || r.record, null, 1).slice(0, 4000); p.appendChild(pre); } catch (e) { p.append(` ${e.message || e}`); } }); p.appendChild(b); }
      }
    }
    window.PSIHfModel = (repo) => pick.set(repo);
  }

  // ── Settings > inside engine: the local model's weights folder, found from its GGUF ──
  if ($('hfPairFind')) $('hfPairFind').addEventListener('click', async () => {
    const out = $('hfPairOut'); out.textContent = '';
    let s; try { s = await psi.settings(); } catch (_) { s = {}; }
    if (!s.lastModel || !/\.gguf$/i.test(s.lastModel)) return stat($('hfPairStat'), 'Load a GGUF model first (Local model).', 'warn');
    stat($('hfPairStat'), 'reading the GGUF file…', 'busy');
    let r; try { r = await hf({ action: 'gguf_base', file: s.lastModel }); } catch (e) { return stat($('hfPairStat'), e.message || String(e), 'error'); }
    if (!r.candidates.length) return stat($('hfPairStat'), `Nothing found for ${r.name || 'this file'}. Paste its address under Models > Import from Hugging Face.`, 'warn');
    stat($('hfPairStat'), `${r.name || ''}: ${r.how}`, '');
    for (const c of r.candidates) { const b = el('button', 'ghost small', c); b.type = 'button'; b.addEventListener('click', () => { const h = $('setModelHf'); if (h) h.scrollIntoView({ block: 'start', behavior: 'smooth' }); if (window.PSIHfModel) PSIHfModel(c); }); out.append(b, ' '); }
  });

  // ── Emergence: training steps on demand ──
  let REFS = null;
  async function refs() {
    const st = window.PSIHfState; if (!st || !st.ready || !st.checkpoints) return null;
    if (REFS && REFS.repo === st.checkpoints) return REFS;
    try { REFS = await hf({ action: 'refs', repo: st.checkpoints }); } catch (_) { REFS = null; }
    return REFS;
  }
  async function fillSteps() {
    const sel = $('hfStepPick'); if (!sel) return;
    const r = await refs(); sel.textContent = '';
    if (!r) { stat($('hfStepStat'), (window.PSIHfState || {}).checkpoints ? 'could not list its steps' : 'set the repo in Settings > Inside-the-model engine (checkpoints from)', 'warn'); $('hfStepGet').disabled = true; return; }
    const have = new Set(window.PSIEm ? PSIEm.steps() : []);
    for (const s of r.steps) { const o = el('option', null, `step ${s.toLocaleString()}${have.has(s) ? ' (on disk)' : ''}`); o.value = String(s); sel.appendChild(o); }
    $('hfStepGet').disabled = false; stat($('hfStepStat'), `${r.steps.length} steps in ${r.repo}, ${have.size} on disk`, '');
  }
  async function getStep(step, node) {
    const st = window.PSIHfState || {};
    const j = await start({ action: 'get_ckpt', repo: st.checkpoints, step }, node); if (!j) return false;
    // The engine reads the checkpoints folder it started with: restart it when the folder was set just now.
    try { const l = await psi.ckptList(); const root = j.result.dest.replace(/\/step\d+$/, ''); if (!l || (l.root || '').replace(/\\/g, '/') !== root) await psi.traceRestart({ checkpoints: root }); } catch (_) { /* the listing says */ }
    if (PSI.refreshState) await PSI.refreshState().catch(() => {});   // the checkpoints folder may have been set just now
    if (window.PSIEm) await PSIEm.relist();
    stat(node, `step ${step.toLocaleString()} is on disk (${j.result.dest})`, 'ok'); fillSteps(); return true;
  }
  if ($('hfStepGet')) {
    $('hfStepGet').addEventListener('click', () => getStep(+$('hfStepPick').value, $('hfStepStat')));
    document.addEventListener('psi-hf', fillSteps); document.addEventListener('psi-hf-ckpt', () => { REFS = null; fillSteps(); });
    // After a test: offer to fetch the step in the middle of the bracket when none on disk is left to score there.
    const mo = new MutationObserver(async () => {
      const L = window.PSIEm && PSIEm.learned(), b = $('hfNarrow'); if (!b) return;
      const r = L && L.after_step != null ? await refs() : null;
      const mid = r ? r.steps.filter((s) => s > L.after_step && s < L.by_step && !PSIEm.steps().includes(s)) : [];
      b.classList.toggle('hidden', !(mid.length && !(L.between && L.between.length)));
      b.dataset.step = mid.length ? String(mid[Math.floor(mid.length / 2)]) : '';
      b.textContent = mid.length ? `Narrow it: fetch step ${(+b.dataset.step).toLocaleString()} from Hugging Face` : b.textContent;
    });
    if ($('emTestWhy')) mo.observe($('emTestWhy'), { childList: true, characterData: true, subtree: true });
    $('hfNarrow').addEventListener('click', async () => { const s = +$('hfNarrow').dataset.step; if (!s && s !== 0) return; if (await getStep(s, $('hfStepStat')) && window.PSIEm) PSIEm.score([s]); });
  }

  // ── Features: an SAE ──
  if ($('hfSaeList')) {
    $('hfSaeList').addEventListener('click', async () => {
      const repo = $('hfSaeRepo').value.trim(); stat($('hfSaeStat'), 'reading…', 'busy');
      let i; try { i = await hf({ action: 'info', repo }); } catch (e) { return stat($('hfSaeStat'), e.message || String(e), 'error'); }
      const sel = $('hfSaePick'); sel.textContent = '';
      for (const f of i.sae_folders) { const o = el('option', null, f || '(the top folder)'); o.value = f; sel.appendChild(o); }
      sel.hidden = !i.sae_folders.length; $('hfSaeGet').hidden = !i.sae_folders.length;
      stat($('hfSaeStat'), i.sae_folders.length ? `${i.sae_folders.length} SAE folders: pick the layer you want` : 'no SAE folders found (a folder with cfg.json and weights)', i.sae_folders.length ? '' : 'warn');
    });
    $('hfSaeGet').addEventListener('click', async () => {
      const j = await start({ action: 'get_sae', repo: $('hfSaeRepo').value.trim(), folder: $('hfSaePick').value }, $('hfSaeStat')); if (!j) return;
      $('feImpFolder').value = j.result.folder; stat($('hfSaeStat'), 'Downloaded. Choose its layer and press import.', 'ok');
    });
  }

  // ── E-levels: an adapter or control vector ──
  if ($('hfAdList')) {
    $('hfAdList').addEventListener('click', async () => {
      stat($('hfAdStat'), 'reading…', 'busy');
      let i; try { i = await hf({ action: 'info', repo: $('hfAdRepo').value.trim() }); } catch (e) { return stat($('hfAdStat'), e.message || String(e), 'error'); }
      const files = i.adapters.length ? i.adapters : i.gguf; const sel = $('hfAdPick'); sel.textContent = '';
      for (const f of files) { const o = el('option', null, `${f.name} (${gb(f.bytes)})`); o.value = f.name; sel.appendChild(o); }
      sel.hidden = !files.length; $('hfAdGet').hidden = !files.length;
      stat($('hfAdStat'), files.length ? '' : (i.peft ? 'a safetensors (PEFT) adapter: the local engine needs it as GGUF; convert it with llama.cpp first' : 'no GGUF adapter here'), files.length ? '' : 'warn');
    });
    $('hfAdGet').addEventListener('click', async () => {
      const j = await start({ action: 'get_adapter', repo: $('hfAdRepo').value.trim(), file: $('hfAdPick').value }, $('hfAdStat')); if (!j) return;
      const f = `${j.result.dest}/${$('hfAdPick').value}`; document.dispatchEvent(new CustomEvent('psi-adapter', { detail: { path: f } }));
      stat($('hfAdStat'), 'Downloaded and chosen. Switch it on above.', 'ok');
    });
  }

  // ── Commons > Contribute: text from a dataset ──
  const tbox = $('hfTextBox');
  if (tbox) {
    const chosen = el('div', 'row'); const say = el('span', 'small muted');
    finder(tbox, 'dataset', (repo) => {
      chosen.textContent = ''; const n = el('input', 'w-6em'); n.type = 'number'; n.value = '200000'; n.min = '3000'; n.max = '5000000'; n.setAttribute('aria-label', 'characters');
      const b = el('button', 'hf-send', 'Get its text'); chosen.append(link(repo, 'dataset'), el('span', 'small muted', ' characters '), n, b);
      b.addEventListener('click', async () => {
        const j = await start({ action: 'get_text', repo, chars: +n.value }, say); if (!j) return;
        const r = j.result; let t; try { t = (await hf({ action: 'read_text', repo })).text; } catch (e) { return stat(say, e.message || String(e), 'error'); }
        $('cmText').value = t; $('cmPractice').checked = false;
        if ($('cmSubject') && !$('cmSubject').value) $('cmSubject').value = repo.split('/')[1].slice(0, 60);
        stat(say, `${r.chars.toLocaleString()} characters from ${repo} (${r.split}, column “${r.column}”) · licence: ${r.licence || 'not stated: check it before you publish'}. Choose the matching licence below.`, r.licence ? 'ok' : 'warn');
      });
    });
    tbox.append(chosen, say);
  }

  document.addEventListener('psi-hf', () => { if ((window.PSIHfState || {}).ready) fillSteps(); });
})();
