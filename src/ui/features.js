// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Features view: sparse-autoencoder features of the inside model's layers (src/sae.py through the inside engine,
// src/sae-api.js for the list and for training). Which features fire on a text, a feature's card (name, how often it
// fires, the words it promotes, its strongest examples, steering an answer with it), search, and which features pushed
// the next word. Every text from the model or the sample (examples, answers) is shown as text, never as markup.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const show = (s) => String(s == null ? '' : s).replace(/\n/g, '⏎');
  const f3 = (v) => (v == null ? '—' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) >= 10 ? 1 : 3));
  const VIEW = 'features';
  ['feGo', 'feSearch', 'feWhy', 'feImport'].forEach((id) => PSI.requires($(id), 'engine'));
  let LIST = null, ENC = null, CARD = null;
  const layer = () => +$('feLayer').value;
  async function job(btn, stat, busy, fail, fn) {
    btn.disabled = true; PSI.setRunning(VIEW, true); PSI.status(stat, busy, 'busy');
    try { await fn(); PSI.setDone(VIEW); } catch (e) { PSI.err(e); PSI.status(stat, `${fail}: ${PSI.explain(e).text}`, 'error'); } finally { PSI.setRunning(VIEW, false); btn.disabled = false; }
  }

  // ── the SAEs this model has ──
  async function refresh() {
    try { LIST = await psi.sae({ action: 'list' }); } catch (e) { $('feInfo').textContent = `The SAE list could not be read: ${PSI.explain(e).text}`; return; }
    const sel = $('feLayer'), was = sel.value; sel.textContent = '';
    LIST.saes.forEach((x) => sel.appendChild(new Option(`layer ${x.layer} · ${x.d_sae.toLocaleString()} features`, x.layer)));
    if (was && LIST.saes.some((x) => String(x.layer) === was)) sel.value = was;
    const L = LIST.model_layers || 36, opts = (s, def) => { s.textContent = ''; for (let l = 1; l < L; l++) s.appendChild(new Option(`layer ${l}`, l)); s.value = def; };
    if (!$('feTrLayer').options.length) { opts($('feTrLayer'), Math.round(L / 2)); opts($('feImpLayer'), Math.round(L / 2)); }
    const ss = $('feTrSample'), was2 = ss.value; ss.textContent = '';
    (LIST.samples || []).forEach((x) => ss.appendChild(new Option(`${x.name} · about ${x.approx_pieces >= 1e6 ? `${(x.approx_pieces / 1e6).toFixed(1)} million` : `${Math.round(x.approx_pieces / 1000)},000`} pieces`, x.name)));
    if (!ss.options.length) ss.appendChild(new Option('neuron corpus (default)', ''));
    if (was2 && [...ss.options].some((o) => o.value === was2)) ss.value = was2; else { const w = [...ss.options].find((o) => /wikitext/.test(o.value)); if (w) ss.value = w.value; }
    const none = !LIST.saes.length; $('feLayer').hidden = $('feLayerLab').hidden = none;
    ['feGo', 'feSearch', 'feWhy'].forEach((id) => { $(id).dataset.noSae = none ? '1' : ''; });
    $('feSetup').open = none || $('feSetup').open;
    info();
  }
  function info() {
    const x = LIST && LIST.saes.find((s) => s.layer === layer());
    if (!LIST || !LIST.saes.length) { $('feInfo').textContent = `No SAE yet for ${LIST && LIST.model ? LIST.model : 'the inside model'}: train or import one below.`; return; }
    if (!x) { $('feInfo').textContent = ''; return; }
    const t = x.trained, im = x.imported;
    ringLine(x); qualityLine(x);
    $('feInfo').textContent = `${x.kind === 'topk' ? `TopK (${x.k} active per piece)` : x.kind} · ${t ? `trained here on ${t.tokens.toLocaleString()} pieces of ${t.corpus}, ${t.epochs} passes; unexplained ${(t.fvu * 100).toFixed(0)}% of the variance, ${(t.dead_frac * 100).toFixed(1)}% features never fired` : im ? `imported (${im.as})` : ''}`;
  }
  // How good the SAE is: the splice-in check (loss recovered with its reconstruction in place of the layer's state) and
  // how well its names predict where its features fire (scored by a judge model on held-out snippets).
  function qualityLine(x) {
    let p = $('feQuality'); if (!p) { p = el('div', 'row fe-ring'); p.id = 'feQuality'; ($('feRing') || $('feInfo').parentNode).after(p); }
    p.textContent = ''; const e = x && x.evaluation, n = x && x.name_scores;
    p.appendChild(el('span', 'small', e ? `spliced in: recovers ${(e.loss_recovered * 100).toFixed(1)}% of the loss, adds ${e.loss_added.toFixed(2)} nats per piece, leaves ${(e.unexplained * 100).toFixed(0)}% unexplained, on ${e.pieces.toLocaleString()} pieces of ${e.in_sample ? 'in-sample text (it learned from it)' : 'held-out text'}` : 'not evaluated yet'));
    if (n && n.n) p.appendChild(el('span', 'small', ` · names: ${n.mean.toFixed(2)} balanced accuracy over ${n.n} feature${n.n === 1 ? '' : 's'} (0.5 is chance; judge ${n.judge.split('/').pop()})`));
    const ev = el('button', 'ghost fe-small', e ? 'Evaluate again' : 'Evaluate'); ev.type = 'button'; ev.id = 'feEval';
    ev.title = 'The model’s next-piece loss on held-out text with this SAE’s reconstruction in place of the layer’s state, against the intact model and the state replaced by its mean (about half a minute on a GPU)';
    const sn = el('button', 'ghost fe-small', 'Score 10 names'); sn.type = 'button'; sn.id = 'feScore';
    sn.title = 'A judge model (Settings > OpenRouter, the judge) sees each name and ten held-out snippets and says where the feature fires. Spends a little OpenRouter credit under your cap (about a cent for ten).';
    const st = el('span', 'k m-0'); PSI.requires(ev, 'engine'); PSI.requires(sn, 'key');
    ev.addEventListener('click', () => job(ev, st, 'evaluating on held-out text…', 'Evaluating failed', async () => { const r = await psi.sae({ action: 'evaluate', layer: x.layer }); x.evaluation = r; qualityLine(x); PSI.status($('feEvalStat') || st, 'evaluated', 'ok'); }));
    sn.addEventListener('click', () => job(sn, st, 'scoring ten names with the judge model…', 'Scoring failed', async () => {
      const r = await psi.sae({ action: 'score_names', layer: x.layer, count: 10 }); await refresh(); $('feLayer').value = x.layer; info();
      PSI.status($('feQuality').querySelector('.k'), `${r.scored} scored, mean ${r.mean == null ? '—' : r.mean.toFixed(2)}, $${r.spent_usd.toFixed(4)}`, 'ok');
    }));
    p.append(ev, sn, st);
  }
  // The SAE's ring: signed here by The Emergence Machine, and checked online by emergencemachine.com (or why not).
  function ringLine(x) {
    let p = $('feRing'); if (!p) { p = el('div', 'row fe-ring'); p.id = 'feRing'; $('feInfo').parentNode.after(p); }
    p.textContent = ''; const r = x && x.ring;
    if (!r) { p.appendChild(el('span', 'small muted', 'No ring: this SAE was made before rings existed.')); return; }
    p.appendChild(el('span', `small ${r.signed ? '' : 'cond-warn'}`, `ring ${r.gen} of this model · ${r.signed ? 'signed by The Emergence Machine (signature and certificate checked here)' : `signature does not check: ${r.why}`}`));
    const o = r.online;
    p.appendChild(el('span', `small ${o && o.verified ? 'fe-ok' : 'muted'}`, o ? (o.verified ? ` · verified online by emergencemachine.com, ${new Date(o.at).toLocaleString()}` : ` · not verified online: ${o.why}`) : ' · not checked online yet'));
    const b = el('button', 'ghost fe-small', o && o.verified ? 'Check again' : 'Verify online'); b.type = 'button'; b.title = 'Sends this SAE’s ring records (hashes, names, figures) to emergencemachine.com, never the SAE or the model; the site checks every signature and certificate and signs its answer';
    b.addEventListener('click', async () => { b.disabled = true; try { const v = await psi.sae({ action: 'verify_online', layer: x.layer }); x.ring = v; ringLine(x); } catch (e) { PSI.err(e); b.disabled = false; } });
    p.appendChild(b);
  }
  $('feLayer').addEventListener('change', () => { info(); CARD = null; });
  const needSae = (stat) => { if (LIST && LIST.saes.length) return false; PSI.status(stat, 'No SAE for this model yet: train or import one above.', 'warn'); $('feSetup').open = true; return true; };

  // ── training and importing ──
  let poll = null;
  $('feTrain').addEventListener('click', async () => {
    const lay = +$('feTrLayer').value, have = LIST && LIST.saes.some((x) => x.layer === lay);
    if (!$('feTrain').dataset.sure) {   // two presses: the first says what it will do
      $('feTrain').dataset.sure = '1'; $('feTrain').textContent = 'Press again to start (stops both engines, uses the GPU for minutes)';
      PSI.status($('feTrStat'), have ? `Layer ${lay} has an SAE: training replaces it (names given to its features are kept aside, not carried over).` : '', have ? 'warn' : '');
      setTimeout(() => { delete $('feTrain').dataset.sure; $('feTrain').textContent = 'Train an SAE on this layer'; }, 6000); return;
    }
    delete $('feTrain').dataset.sure; $('feTrain').textContent = 'Train an SAE on this layer';
    try {
      await psi.sae({ action: 'train', confirm: true, layer: lay, expansion: +$('feTrSize').value, tokens: +$('feTrTokens').value, corpus: $('feTrCorpus').value.trim() || $('feTrSample').value || undefined });
      PSI.setRunning(VIEW, true); $('feTrain').disabled = true; $('feTrCancel').hidden = false; watch();
    } catch (e) { PSI.status($('feTrStat'), PSI.explain(e).text, 'error'); }
  });
  // A larger text sample (WikiText-103): a download, so only from this button, after a second press.
  $('feFetch').addEventListener('click', async () => {
    const b = $('feFetch');
    if (!b.dataset.sure) { b.dataset.sure = '1'; b.textContent = 'Press again to download about 150 MB from Hugging Face'; setTimeout(() => { delete b.dataset.sure; b.textContent = 'Get a larger text sample'; }, 6000); return; }
    delete b.dataset.sure; b.textContent = 'Get a larger text sample'; b.disabled = true;
    try {
      await psi.sae({ action: 'fetch_sample', confirm: true });
      const t = setInterval(async () => {
        let s; try { s = await psi.sae({ action: 'fetch_status' }); } catch (_) { return; }
        const p = s.progress || {};
        if (s.fetching) { PSI.status($('feFetchStat'), `${p.stage || 'starting'}${p.note ? ` · ${p.note}` : ''}`, 'busy'); return; }
        clearInterval(t); b.disabled = false;
        if (s.error) PSI.status($('feFetchStat'), `Not fetched: ${s.error}`, 'error');
        else { PSI.status($('feFetchStat'), `Fetched: ${s.done.paragraphs.toLocaleString()} paragraphs, about ${(s.done.approx_pieces / 1e6).toFixed(1)} million pieces.`, 'ok'); await refresh(); }
      }, 2000);
    } catch (e) { b.disabled = false; PSI.status($('feFetchStat'), PSI.explain(e).text, 'error'); }
  });
  $('feTrCancel').addEventListener('click', () => psi.sae({ action: 'cancel' }).catch(() => {}));
  function watch() {
    clearInterval(poll);
    poll = setInterval(async () => {
      let s; try { s = await psi.sae({ action: 'train_status' }); } catch (_) { return; }
      const p = s.progress || {};
      if (s.training) { PSI.status($('feTrStat'), `layer ${s.layer} · ${p.stage || 'starting'}${p.of ? ` ${p.done} of ${p.of}` : ''}${p.fvu != null ? ` · unexplained ${(p.fvu * 100).toFixed(0)}%` : ''}${p.note ? ` · ${p.note}` : ''}`, 'busy'); return; }
      clearInterval(poll); PSI.setRunning(VIEW, false); $('feTrain').disabled = false; $('feTrCancel').hidden = true;
      if (s.error) PSI.status($('feTrStat'), `Training failed: ${s.error}`, 'error');
      else if (s.done) { PSI.status($('feTrStat'), `Trained (ring ${s.done.ring}${s.done.signed ? ', signed' : ', NOT signed: no signing key'}): layer ${s.done.layer}, ${s.done.d_sae.toLocaleString()} features (${s.done.alive_features.toLocaleString()} fired on the sample), unexplained ${(s.done.fvu * 100).toFixed(0)}%, ${Math.round(s.done.seconds / 60)} min.`, 'ok'); PSI.setDone(VIEW); await refresh(); $('feLayer').value = s.done.layer; info(); setTimeout(async () => { await refresh(); $('feLayer').value = s.done.layer; info(); }, 8000); }
    }, 2000);
  }
  psi.sae({ action: 'train_status' }).then((s) => { if (s.training) { PSI.setRunning(VIEW, true); $('feTrain').disabled = true; $('feTrCancel').hidden = false; watch(); } }).catch(() => {});
  $('feImport').addEventListener('click', () => job($('feImport'), $('feImpStat'), 'importing…', 'Import failed', async () => {
    const r = await psi.sae({ action: 'import', folder: $('feImpFolder').value.trim(), layer: +$('feImpLayer').value });
    PSI.status($('feImpStat'), `Imported as ${r.as}: layer ${r.layer}, ${r.d_sae.toLocaleString()} features (ring ${r.ring}${r.signed ? ', signed' : ''}). Its features have no examples yet. Checking the ring online…`, 'ok');
    await psi.sae({ action: 'verify_online', layer: r.layer }).catch(() => {}); await refresh(); $('feLayer').value = r.layer; info();
  }));

  // ── 1. which features fire ──
  const nameOf = (f) => (ENC && ENC.names[String(f)]) || `feature ${f}`;
  function featRow(box, f, v, max, name, onClick) {
    const r = el('button', 'fe-row'); r.type = 'button';
    const bar = el('span', 'fe-bar'); bar.style.width = `${Math.max(2, Math.min(100, (v / (max || v || 1)) * 100))}%`;
    r.append(el('span', 'fe-id', `#${f}`), el('span', 'fe-name', name || `feature ${f}`), el('span', 'fe-v', f3(v)), bar);
    r.addEventListener('click', onClick); box.appendChild(r); return r;
  }
  // The last reading of each part (which features fire, search, why this next word) is kept (PSI.keep): a reload or
  // another device shows it again. The layer picker is not restored (its list comes from the SAEs on disk); each kept
  // result names its own layer.
  function showEnc(r) {
    ENC = r; renderEnc(); PSI.status($('feStat'), `layer ${ENC.layer} · unexplained ${ENC.unexplained == null ? 'not measured (one piece)' : `${(ENC.unexplained * 100).toFixed(0)}%`}`, 'ok');
    PSI.cond($('feStat'), [...(PSI.insideCondParts ? PSI.insideCondParts(ENC.conditions, []) : [])], 'read on');
  }
  const KEEP_ENC = PSI.keep('features-encode', { inputs: ['feText'], status: 'feStat', render: (r) => { if (r && r.tokens) showEnc(r); } });
  $('feGo').addEventListener('click', () => {
    if (needSae($('feStat'))) return; if (!$('feText').value.trim()) return PSI.status($('feStat'), 'Type some text first.', 'warn');
    return job($('feGo'), $('feStat'), 'reading the text…', 'Finding the features failed', async () => {
      showEnc(await KEEP_ENC.call('sae', { action: 'encode', text: $('feText').value, layer: layer(), k: 8 }));
    });
  });
  function renderEnc(sel) {
    const box = $('feTokens'); box.textContent = '';
    const peak = Math.max(1e-9, ...ENC.tokens.filter((t) => !t.first).map((t) => (t.top[0] ? t.top[0][1] : 0)));   // the first piece (an outlier) is left out of the scale
    ENC.tokens.forEach((t, i) => {
      const s = el('span', i === sel ? 'sel' : t.first ? 'fe-first' : null, show(t.piece)); const top = t.top[0];
      s.style.background = top && !t.first ? `rgba(255,195,107,${0.08 + 0.6 * Math.min(1, top[1] / peak)})` : 'transparent';
      s.title = `${top ? `#${top[0]} ${nameOf(top[0])} (${f3(top[1])}) · ${t.active} active` : 'no feature'}${t.first ? ' · the first piece: its state is an outlier (the attention sink), left out of the totals' : ''}`;
      s.tabIndex = 0; const go = () => renderEnc(i); s.addEventListener('click', go); s.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
      box.appendChild(s);
    });
    const list = $('feTop'); list.textContent = '';
    const rows = sel == null ? ENC.overall : ENC.tokens[sel].top;
    $('fePieceHead').textContent = sel == null ? 'strongest features over the whole text (summed over the pieces)' : `features on “${show(ENC.tokens[sel].piece)}” (${ENC.tokens[sel].active} active)`;
    const mx = rows.length ? rows[0][1] : 1;
    rows.forEach(([f, v]) => featRow(list, f, v, mx, nameOf(f), () => card(ENC.layer, f)));
    if (!rows.length) list.appendChild(el('p', 'small muted', 'No feature fires here.'));
  }

  // ── the feature card ──
  async function card(lay, f) {
    const box = $('feCard'); box.textContent = ''; box.appendChild(el('p', 'small muted', `reading feature #${f}…`));
    try { CARD = await psi.sae({ action: 'feature', layer: lay, id: f }); } catch (e) { box.textContent = `The card could not be read: ${PSI.explain(e).text}`; return; }
    const C = CARD; box.textContent = '';
    box.appendChild(el('p', 'fe-title', `#${C.id} · layer ${C.layer}`));
    const nm = el('p'); nm.appendChild(el('b', null, C.name || C.auto_name)); if (!C.name) nm.append(el('span', 'small muted', C.auto_name.startsWith('→') ? '  (no name yet: the words it promotes)' : '  (no name yet: the pieces it fires on most in its strongest examples)')); else nm.append(el('span', 'small muted', `  named by ${C.label.by}`)); box.appendChild(nm);
    if (C.label && C.label.why) box.appendChild(el('p', 'small muted', C.label.why));
    if (C.score) box.appendChild(el('p', 'small', `name score: ${C.score.balanced_accuracy.toFixed(2)} balanced accuracy (${C.score.hits}) for “${C.score.name_scored}”, judged by ${String(C.score.judge).split('/').pop()} on ${C.score.in_sample ? 'in-sample' : 'held-out'} snippets`));
    const nrow = el('div', 'row'), inp = el('input'), save = el('button', 'ghost', 'Save the name'), ask = el('button', 'ghost', 'Name it with Prometheus'), nst = el('span', 'k m-0');
    inp.type = 'text'; inp.id = 'feName'; inp.maxLength = 80; inp.placeholder = 'a short name for this feature'; inp.value = C.name || ''; save.type = ask.type = 'button';
    save.addEventListener('click', async () => { try { await psi.sae({ action: 'label', layer: C.layer, id: C.id, label: inp.value, by: 'you' }); PSI.status(nst, inp.value.trim() ? 'saved' : 'name removed', 'ok'); if (ENC) ENC.names[String(C.id)] = inp.value.trim() || ENC.names[String(C.id)]; } catch (e) { PSI.status(nst, PSI.explain(e).text, 'error'); } });
    ask.addEventListener('click', () => { if (!window.PSIHelper || !PSIHelper.isOn()) return PSI.status(nst, 'Prometheus is off: turn it on in Settings.', 'warn'); PSIHelper.open(); PSIHelper.ask(`Name feature #${C.id} at layer ${C.layer}: read its card with feature_card, then give it a short name with name_feature, and say in a sentence what its examples have in common and how sure you are.`); });
    nrow.append(inp, save, ask, nst); box.appendChild(nrow);
    const stats = el('p', 'small'); stats.textContent = `fires on ${C.freq == null ? '(not measured)' : `${(C.freq * 100).toFixed(C.freq < 0.01 ? 2 : 1)}% of pieces`} · largest activation seen ${f3(C.max_act)}`; box.appendChild(stats);
    const w = el('p', 'small'); w.title = 'The words its direction pushes up when read through the output head (the logit lens). Halfway up the model this reading is often noise; the examples below say more.'; w.appendChild(el('b', null, 'promotes: ')); w.append(C.promotes.map((x) => `“${show(x)}”`).join(' ')); box.appendChild(w);
    const w2 = el('p', 'small muted'); w2.appendChild(el('b', null, 'suppresses: ')); w2.append(C.suppresses.map((x) => `“${show(x)}”`).join(' ')); box.appendChild(w2);
    box.appendChild(el('p', 'k', 'strongest examples'));
    if (!C.examples.length) box.appendChild(el('p', 'small muted', 'No examples kept (an imported SAE, or a feature that never fired on the sample).'));
    const ul = el('ul', 'fe-ex'); C.examples.forEach(([txt, act, a, b]) => { const li = el('li'); li.append(el('span', 'fe-v', f3(act)), ' ', show(txt.slice(0, a))); li.appendChild(el('mark', null, show(txt.slice(a, b)))); li.append(show(txt.slice(b))); ul.appendChild(li); }); box.appendChild(ul);
    // steering
    box.appendChild(el('p', 'k', 'steer an answer with it'));
    box.appendChild(el('p', 'small muted', 'The feature’s direction is added after this layer at every piece while the model answers (greedy), beside the plain answer. Strength is in units of the feature’s largest activation seen: 1 adds it as strongly as it ever fired; much more turns the answer into one repeated word. Negative takes it away.'));
    const sr = el('div', 'row wide'), sp = el('input'), sv = el('input'), sl = el('span', 'small mono'), go = el('button', null, 'Compare answers'), sst = el('span', 'k m-0');
    sp.type = 'text'; sp.id = 'feSteerPrompt'; sp.className = 'grow'; sp.value = 'Tell me about your weekend.'; sv.type = 'range'; sv.id = 'feStrength'; psi.settings().then((v) => { const m = (v.guardrails && v.guardrails.steerMax) || 4; sv.min = -m; sv.max = m; }).catch(() => {}); sv.min = -4; sv.max = 4; sv.step = 0.25; sv.value = 1; sv.setAttribute('aria-label', 'strength');
    const lab = () => { sl.textContent = `strength ${(+sv.value).toFixed(1)}`; }; lab(); sv.addEventListener('input', lab); go.type = 'button'; go.id = 'feSteerGo'; PSI.requires(go, 'engine');
    sr.append(sp); box.appendChild(sr); const r2 = el('div', 'row'); r2.append(sv, sl, go, sst); box.appendChild(r2);
    const out = el('div', 'fe-two fe-steer'); box.appendChild(out);
    go.addEventListener('click', () => job(go, sst, 'answering twice: plain, then with the feature…', 'Steering failed', async () => {
      const r = await psi.sae({ action: 'steer', layer: C.layer, id: C.id, strength: +sv.value, prompt: sp.value, n_new: 48 });
      out.textContent = '';
      const col = (h, t) => { const d = el('div', 'out minh-0'); d.appendChild(el('p', 'k', h)); d.appendChild(el('p', null, t)); out.appendChild(d); };
      col('plain', r.plain); col(`with #${C.id} at strength ${r.strength}`, r.steered); PSI.status(sst, 'done: the feature was removed after', 'ok');
    }));
  }

  // ── 2. search ──
  function showSearch(r) {
    const a = $('feByName'), b = $('feByText'); a.textContent = ''; b.textContent = '';
    r.by_name.forEach((x) => featRow(a, x.id, x.score, r.by_name[0].score, x.name, () => card(r.layer, x.id)));
    r.by_text.forEach((x) => featRow(b, x.id, x.act, r.by_text[0] && r.by_text[0].act, x.name, () => card(r.layer, x.id)));
    if (!r.by_name.length) a.appendChild(el('p', 'small muted', 'none'));
    PSI.status($('feQStat'), `${r.by_name.length} by name, ${r.by_text.length} on the phrase`, 'ok');
  }
  const KEEP_Q = PSI.keep('features-search', { inputs: ['feQuery'], status: 'feQStat', render: (r) => { if (r && r.by_name) showSearch(r); } });
  $('feSearch').addEventListener('click', () => {
    if (needSae($('feQStat'))) return; if (!$('feQuery').value.trim()) return PSI.status($('feQStat'), 'Type what to look for.', 'warn');
    return job($('feSearch'), $('feQStat'), 'searching…', 'Search failed', async () => {
      showSearch(await KEEP_Q.call('sae', { action: 'search', layer: layer(), query: $('feQuery').value.trim() }));
    });
  });

  // ── 3. why this next word ──
  $('feWhy').addEventListener('click', () => {
    if (needSae($('feWhyStat'))) return;
    return job($('feWhy'), $('feWhyStat'), 'reading the prompt and tracing the next word back…', 'Explaining failed', async () => {
      showWhy(await KEEP_WHY.call('sae', { action: 'attribute', prompt: $('feWhyPrompt').value, n: 8 }));
    });
  });
  function showWhy(r) {
    const box = $('feWhyOut'); box.textContent = '';
    const h = el('p'); h.append('next word: '); h.appendChild(el('b', null, `“${show(r.target)}”`)); box.appendChild(h);
    r.layers.forEach((L) => {
      const d = el('div', 'out minh-0 fe-why'); d.appendChild(el('p', 'k', `layer ${L.layer}`));
      const share = Math.abs(L.state_total) > 1e-9 ? L.error / L.state_total : null;
      d.appendChild(el('p', 'small muted', `features together ${f3(L.features_total)} · not explained by the SAE ${f3(L.error)}${share != null ? ` (${(share * 100).toFixed(0)}% of the layer's total ${f3(L.state_total)})` : ''}`));
      const two = el('div', 'fe-two'), up = el('div', 'fe-list'), dn = el('div', 'fe-list');
      up.appendChild(el('p', 'small muted', 'pushed it up')); dn.appendChild(el('p', 'small muted', 'pushed it down'));
      const mx = Math.max(1e-9, ...L.pushed_up.map((x) => x.by), ...L.pushed_down.map((x) => x.by));
      L.pushed_up.forEach((x) => featRow(up, x.id, x.by, mx, x.name, () => { $('feLayer').value = L.layer; info(); card(L.layer, x.id); $('feCard').scrollIntoView({ behavior: 'smooth', block: 'center' }); }));
      L.pushed_down.forEach((x) => featRow(dn, x.id, x.by, mx, x.name, () => { $('feLayer').value = L.layer; info(); card(L.layer, x.id); $('feCard').scrollIntoView({ behavior: 'smooth', block: 'center' }); }));
      two.append(up, dn); d.appendChild(two); box.appendChild(d);
    });
    PSI.status($('feWhyStat'), `${r.layers.length} layer${r.layers.length === 1 ? '' : 's'}`, 'ok');
  }
  const KEEP_WHY = PSI.keep('features-why', { inputs: ['feWhyPrompt'], status: 'feWhyStat', render: (r) => { if (r && r.layers) showWhy(r); } });

  // Opening the view reads the list (from the data folder: the engine does not start for it).
  const sec = $('tab-features');
  new MutationObserver(() => { if (!sec.classList.contains('hidden')) refresh(); }).observe(sec, { attributes: true, attributeFilter: ['class'] });
  if (!sec.classList.contains('hidden')) refresh();
  window.PSIFeatures = { refresh, card, layers: () => (LIST ? LIST.saes.map((x) => x.layer) : []) };
})();
