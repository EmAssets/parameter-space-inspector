// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Commons view: seed lines grown by many people (src/commons-api.js). Recipes (browse Hugging Face, import from any
// address, check every ingredient, build), contributing a branch, verifying anyone's claims, and the profile, keys and
// signing choice. Everything read from a recipe or a record is shown as text (textContent), never as markup.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const VIEW = 'commons';
  const sec = $('tab-commons'); if (!sec || !window.psi) return;
  const api = (a) => psi.commons(a);
  let OV = null, SEL = null, CHECK = null, POLL = null, PLUGINS = [];
  const kb = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : n >= 1024 ? `${Math.round(n / 1024)} KB` : `${n} B`);
  const short = (h) => String(h || '').slice(0, 12);
  // Hugging Face parts show only once a token is saved and checked (ui/hf-set.js).
  const hfOn = () => document.documentElement.classList.contains('hf-on');
  document.addEventListener('psi-hf', () => { if (!sec.classList.contains('hidden')) refresh(); });

  // ── sub-views ──
  function show(which) {
    for (const b of sec.querySelectorAll('[data-cm]')) b.classList.toggle('on', b.dataset.cm === which);
    for (const p of sec.querySelectorAll('[data-cm-panel]')) p.hidden = p.dataset.cmPanel !== which;
    if (which === 'verify') fillVerify();
    try { localStorage.setItem('psi-cm-tab', which); } catch (_) { /* private mode */ }
  }
  for (const b of sec.querySelectorAll('[data-cm]')) b.addEventListener('click', () => show(b.dataset.cm));

  // ── long runs: one at a time, polled ──
  function poll(done) {
    clearTimeout(POLL);
    const tick = async () => {
      let s; try { s = await api({ action: 'status' }); } catch (_) { POLL = setTimeout(tick, 3000); return; }
      const j = s.job; if (!j) { PSI.setRunning(VIEW, false); return; }
      const p = j.progress || {};
      if (!j.finished) {
        const bits = [p.stage, p.step != null ? `step ${p.step} of ${p.of}` : p.i ? `${p.i} of ${p.of}` : '', p.loss != null ? `loss ${p.loss}` : '', p.name || p.note || ''].filter(Boolean).join(' · ');
        PSI.status($('cmStat'), `${j.label}: ${bits || 'starting'}`, 'busy'); $('cmStop').hidden = false; POLL = setTimeout(tick, 1500); return;
      }
      $('cmStop').hidden = true; PSI.setRunning(VIEW, false);
      if (j.error) { PSI.status($('cmStat'), `${j.label}: ${j.error}`, 'error'); if (done) done(null, j.error); return; }
      PSI.status($('cmStat'), `${j.label}: done in ${Math.round((new Date(j.ended) - new Date(j.started)) / 1000)} s`, 'ok'); PSI.setDone(VIEW);
      if (done) await done(j.done || {});
    };
    tick();
  }
  async function start(a, done) {
    try { PSI.setRunning(VIEW, true); await api(a); poll(done); }
    catch (e) { PSI.setRunning(VIEW, false); PSI.status($('cmStat'), e.message, 'error'); }
  }
  $('cmStop').addEventListener('click', () => api({ action: 'cancel' }));

  // ── the line ──
  async function refresh() {
    try { OV = await api({ action: 'overview' }); } catch (e) { PSI.status($('cmStat'), e.message, 'error'); return; }
    const L = OV.line || {};
    $('cmLine').textContent = L.ready ? `${L.title} by ${L.by} (${L.licence}) · ready` : L.error ? `could not read the line: ${L.error}` : `${L.title || 'stories260K'}: not downloaded yet`;
    $('cmFetch').hidden = !!L.ready;
    for (const id of ['cmBuild', 'cmTrain', 'cmMake']) PSI.block($(id), 'line', L.ready ? null : 'Download the seed first (top of this view).');
    fillMe(); await Promise.all([listRecipes(), listBranches(), listModels(), listPlugins()]);
    if (OV.job && OV.job.running) poll(() => refresh());
  }
  $('cmFetch').addEventListener('click', () => start({ action: 'line_fetch', confirm: true }, () => refresh()));

  // ── recipes ──
  function card(parent, title, sub, buttons = []) {
    const c = el('div', 'cm-item'); const t = el('div', 'cm-title'); t.appendChild(el('b', '', title)); if (sub) t.appendChild(el('span', 'small muted', `  ${sub}`)); c.appendChild(t);
    if (buttons.length) { const r = el('div', 'cm-actions'); for (const [label, fn, cls] of buttons) { const b = el('button', `ghost small ${cls || ''}`, label); b.addEventListener('click', fn); r.appendChild(b); } c.appendChild(r); }
    parent.appendChild(c); return c;
  }
  $('cmBrowse').addEventListener('click', async () => {
    const box = $('cmFound'); box.textContent = ''; PSI.status($('cmStat'), 'asking Hugging Face…', 'busy');
    try {
      const r = await api({ action: 'browse', search: $('cmSearch').value, sort: $('cmSort').value });
      const rows = r.results || []; PSI.status($('cmStat'), `${rows.length} public recipe${rows.length === 1 ? '' : 's'} tagged psi-recipe, ${$('cmSort').selectedOptions[0].textContent.toLowerCase()} first`, rows.length ? 'ok' : '');
      if (!rows.length) box.appendChild(el('p', 'small muted', 'No public recipes yet. Be the first: make one from branches (below) and publish it.'));
      for (const m of rows) card(box, m.repo, `${m.downloads || 0} downloads (30 days) · ${m.likes || 0} likes · updated ${String(m.modified || '').slice(0, 10)}`, [['Import', () => { $('cmAddr').value = m.recipe; doImport(); }], ['Open on 🤗', () => window.open(`https://huggingface.co/${m.repo}`, '_blank', 'noopener')]]);
    } catch (e) { PSI.status($('cmStat'), `Hugging Face: ${e.message}`, 'error'); }
  });
  function doImport() { const addr = $('cmAddr').value.trim(); if (!addr) return; start({ action: 'import', addr }, async (d) => { if (!d) return; await listRecipes(); check(d.recipe); }); }
  $('cmImport').addEventListener('click', doImport);

  async function listRecipes() {
    const box = $('cmRecipes'); box.textContent = '';
    let r; try { r = await api({ action: 'recipes' }); } catch (e) { box.appendChild(el('p', 'small bad', e.message)); return; }
    if (!r.recipes.length) { box.appendChild(el('p', 'small muted', 'None yet: import one, or make one from branches.')); return; }
    for (const x of r.recipes) card(box, x.name || x.id, `${x.kind === 'hf' ? `public models: ${x.models.join(' + ')}` : `trained in PSI${x.imported_from ? '' : ' here'}`} · ${x.branches} branches · ${x.plugin} · by ${(x.curator || {}).name || '?'}${x.signed ? ' · signed' : ' · unsigned'}${x.imported_from ? ' · imported' : ''}`, [['Check', () => check(x.id)]]);
  }
  function check(id) {
    SEL = id; $('cmBuildRow').hidden = true; $('cmCheck').textContent = ''; $('cmBuilt').textContent = '';
    start({ action: 'check', id }, (d) => { if (d) { CHECK = d; renderCheck(d); } });
  }
  const STATE = { ready: ['ok', 'ready: every piece is here or downloadable'], recoverable: ['warn', 'recoverable'], partial: ['warn', 'partial'], blocked: ['bad', 'cannot build'] };
  function sigText(sigs) { return !sigs || !sigs.length ? 'unsigned' : sigs.map((s) => `${s.valid ? '✓' : '✗'} ${s.role} ${s.key}`).join(', '); }
  function renderHf(d) {
    const box = $('cmCheck'); box.textContent = '';
    const h = el('div', 'cm-title'); h.appendChild(el('b', '', d.name || d.recipe)); const st = STATE[d.state] || ['', d.state]; h.appendChild(el('span', `cm-badge ${st[0]}`, st[1])); box.appendChild(h);
    if (d.about) box.appendChild(el('p', 'small', d.about));
    box.appendChild(el('p', 'small', d.why));
    box.appendChild(el('p', 'small muted', `public models (${d.line}) · curator ${(d.curator || {}).name || '?'} · ${sigText(d.signatures)} · method ${d.method.plugin} ${JSON.stringify(d.method.options || {})}${d.download_bytes ? ` · to download ${kb(d.download_bytes)}` : ' · everything is here'}`));
    const t = el('table', 'cm-table'); const hr = el('tr'); for (const c of ['', 'role', 'model', 'commit', 'licence', 'strength', 'from base', 'here']) hr.appendChild(el('th', '', c)); t.appendChild(hr);
    for (const m of d.models) {
      const r = el('tr', m.ok || m.here ? '' : 'cm-miss');
      r.appendChild(el('td', m.ok ? 'ok' : 'bad', m.ok ? '✓' : 'gone')); r.appendChild(el('td', '', m.role));
      const a = el('a', '', m.repo); a.href = `https://huggingface.co/${m.repo}/tree/${m.commit}`; a.target = '_blank'; a.rel = 'noopener'; const td = el('td'); td.appendChild(a); r.appendChild(td);
      r.appendChild(el('td', 'small', m.commit.slice(0, 10))); r.appendChild(el('td', '', m.licence || 'not stated')); r.appendChild(el('td', '', m.role === 'base' ? '' : String(m.strength ?? 1)));
      r.appendChild(el('td', m.distance != null && m.distance > d.far ? 'warn' : 'small', m.distance == null ? '' : m.distance.toFixed(3)));
      r.appendChild(el('td', 'small', m.here ? 'yes' : kb(m.bytes))); t.appendChild(r);
    }
    box.appendChild(t);
    const far = d.models.filter((m) => m.distance != null && m.distance > d.far);
    if (far.length) box.appendChild(el('p', 'small warn', `Far from the base: ${far.map((m) => `${m.repo} (${m.distance.toFixed(2)})`).join(', ')}. A light fine-tune moves its base a few hundredths; this is more like another model of the same shape, and merging it usually damages both. Try it, but measure before you trust it.`));
    if (d.models.some((m) => !m.licence)) box.appendChild(el('p', 'small warn', 'A model states no licence: check its card before sharing what you build.'));
    $('cmBuildRow').hidden = d.state === 'blocked';
    for (const id of ['cmRecover', 'cmPartial', 'cmDownload', 'cmBisect']) $(id).hidden = true; $('cmBuild').hidden = false;
    $('cmCosign').hidden = true; $('cmGguf').parentElement.hidden = true;
  }
  function renderCheck(d) {
    if (d.kind === 'hf') return renderHf(d);
    $('cmGguf').parentElement.hidden = false; $('cmBisect').hidden = false;
    const box = $('cmCheck'); box.textContent = '';
    const h = el('div', 'cm-title'); h.appendChild(el('b', '', d.name || d.recipe)); const st = STATE[d.state] || ['', d.state]; h.appendChild(el('span', `cm-badge ${st[0]}`, st[1])); box.appendChild(h);
    if (d.about) box.appendChild(el('p', 'small', d.about));
    box.appendChild(el('p', 'small', d.why));
    if (d.seed_ok === false) {   // the one-time 1 MB seed download, offered where it blocks, then the check runs again
      const b = el('button', 'small', 'Download the seed (1 MB) and check again'); const id = d.recipe;
      b.addEventListener('click', () => start({ action: 'line_fetch', confirm: true }, async () => { await refresh(); check(id); })); box.appendChild(b);
    }
    box.appendChild(el('p', 'small muted', `curator ${(d.curator || {}).name || '?'} · ${sigText(d.signatures)} · method ${d.method.plugin}${d.method.same_code ? '' : d.method.known ? ' (different version here!)' : ' (unknown here)'}${d.method.linear ? ', linear' : ''} · built model published: ${d.output_published ? 'yes' : 'no'}${d.download_bytes ? ` · to download ${kb(d.download_bytes)}` : ''}`));
    const t = el('table', 'cm-table'); const hr = el('tr'); for (const c of ['', 'branch', 'by', 'licence', 'subject', 'signed', 'where']) hr.appendChild(el('th', '', c)); t.appendChild(hr);
    for (const b of d.branches) {
      const r = el('tr', b.status === 'ok' ? '' : 'cm-miss');
      r.appendChild(el('td', b.status === 'ok' ? 'ok' : 'bad', b.status === 'ok' ? '✓' : b.status === 'missing' ? 'missing' : 'refused'));
      r.appendChild(el('td', '', `${b.name || short(b.delta_sha256)}${b.strength !== 1 ? ` ×${b.strength}` : ''}`)); r.appendChild(el('td', '', b.contributor || '?')); r.appendChild(el('td', '', b.licence || '—'));
      r.appendChild(el('td', '', b.subject || '')); r.appendChild(el('td', 'small', sigText(b.signatures))); r.appendChild(el('td', 'small', b.status === 'ok' ? `${b.where}${b.hosts > 1 ? ` (+${b.hosts - 1})` : ''}` : (b.why || '')));
      t.appendChild(r); if (b.note) { const n = el('tr'); const c = el('td', 'small warn', b.note); c.colSpan = 7; n.appendChild(c); t.appendChild(n); }
    }
    box.appendChild(t);
    if (d.only_source.length) box.appendChild(el('p', 'small muted', `Only one branch covers: ${d.only_source.join(', ')}. If it goes, that knowledge goes: worth mirroring.`));
    if (d.lost_subjects.length) box.appendChild(el('p', 'small warn', `No source left for: ${d.lost_subjects.join(', ')}.`));
    const row = $('cmBuildRow'); row.hidden = d.state === 'blocked';
    $('cmBuild').hidden = d.state !== 'ready'; $('cmRecover').hidden = d.state !== 'recoverable'; $('cmPartial').hidden = !['partial', 'recoverable'].includes(d.state);
    $('cmDownload').hidden = !d.output_published; $('cmCosign').hidden = !(OV && OV.em_key_here) || (d.signatures || []).some((s) => s.role === 'emergence-machine');
    const pub = el('div', 'row'); const inp = el('input', 'grow'); inp.placeholder = 'publish this recipe to a Hugging Face repo: owner/name'; inp.setAttribute('aria-label', 'repo for this recipe');
    const go = el('button', 'ghost small', 'Prepare'); go.addEventListener('click', () => prepare('recipe', d.recipe, inp.value.trim(), box)); pub.append(inp, go); pub.dataset.hf = ''; box.appendChild(pub);
  }
  const build = (mode) => start({ action: 'build', id: SEL, mode, gguf: $('cmGguf').checked, confirm: true }, (d) => { if (d) { renderBuilt(d); listModels(); } });
  $('cmBuild').addEventListener('click', () => build('strict')); $('cmRecover').addEventListener('click', () => build('recover'));
  $('cmPartial').addEventListener('click', () => build('partial')); $('cmDownload').addEventListener('click', () => build('download'));
  $('cmBisect').addEventListener('click', () => start({ action: 'bisect', id: SEL, confirm: true }, (d) => { if (d) { const b = $('cmBuilt'); b.textContent = ''; b.appendChild(el('p', '', `Suspect: ${d.suspect} by ${d.contributor || '?'} (${d.builds} builds). With it ${d.full}, without it ${d.without}.`)); b.appendChild(el('p', 'small muted', d.note)); } }));
  $('cmUse').addEventListener('click', async () => { try { const r = await api({ action: 'use', id: SEL, confirm: true }); PSI.status($('cmStat'), `use recorded and signed (${r.use})`, 'ok'); } catch (e) { PSI.status($('cmStat'), e.message, 'error'); } });
  $('cmCosign').addEventListener('click', () => start({ action: 'cosign_em', kind: 'recipe', id: SEL, confirm: true }, () => check(SEL)));
  // ── the second path: public Hugging Face models ──
  $('cmHfFind').addEventListener('click', () => {
    const box = $('cmHfFound'); box.textContent = ''; $('cmHfRow').hidden = true; $('cmHfFx').hidden = true;
    start({ action: 'hf_find', base: $('cmHfBase').value.trim() }, (d) => {
      if (!d) return;
      box.appendChild(el('p', 'small muted', `${d.base} @ ${String(d.commit).slice(0, 10)} (${d.licence || 'no licence stated'}). Tick the models to merge into it:`));
      for (const c of d.candidates) {
        const l = el('label', `small cm-pickrow ${c.ok ? '' : 'cm-no'}`); const x = el('input'); x.type = 'checkbox'; x.value = c.repo; x.disabled = !c.ok;
        l.append(x, ` ${c.repo}`, el('span', 'muted', `  ${c.licence || 'no licence'} · ${c.downloads || 0} downloads · ${c.why}`)); box.appendChild(l);
      }
      const carry = () => { const sel = $('cmHfCarry'); const cur = sel.value; sel.textContent = ''; [...box.querySelectorAll('input:checked')].forEach((x, i) => { const o = el('option', '', x.value); o.value = String(i); sel.appendChild(o); }); sel.value = cur || '0'; };
      box.addEventListener('change', carry); $('cmHfRow').hidden = false; $('cmHfFx').hidden = false;
    });
  });
  $('cmHfMake').addEventListener('click', () => {
    const bs = [...$('cmHfFound').querySelectorAll('input:checked')].map((x) => x.value); if (!bs.length) { PSI.status($('cmStat'), 'tick at least one model', 'warn'); return; }
    start({ action: 'hf_make', confirm: true, base: $('cmHfBase').value.trim(), branches: bs, method: $('cmHfMethod').value, lam: +$('cmHfLam').value, carry: +$('cmHfCarry').value || 0,
      name: $('cmHfName').value || bs.map((x) => x.split('/')[1]).join(' + ') }, async (d) => { if (d) { await listRecipes(); check(d.recipe); if (d.warning) setTimeout(() => PSI.status($('cmStat'), d.warning, 'warn'), 3000); } });
  });
  function renderBuilt(d) {
    const b = $('cmBuilt'); b.textContent = '';
    b.appendChild(el('p', d.matches ? 'ok' : 'warn', d.matches ? `Built: the same bytes the recipe expects (${short(d.output)}).` : `Built ${short(d.output)}; the recipe expects ${short(d.expected)} (${d.mode} build).`));
    for (const n of d.notes || []) b.appendChild(el('p', 'small muted', n));
    if (d.eval) b.appendChild(el('p', 'small', `held-out loss ${d.eval.mean} (${Object.entries(d.eval.parts).map(([k, v]) => `${k} ${v}`).join(', ')}); lower is better`));
    if (d.story) { b.appendChild(el('p', 'k', 'it writes')); b.appendChild(el('p', 'cm-story', d.story)); }
    b.appendChild(el('p', 'small muted', `Saved in ${d.dir}${d.dir ? ' (model.safetensors, record.json, and the GGUF if chosen)' : ''}.`));
    if (d.gguf) { const o = el('button', 'small', 'Open it in Generate'); o.addEventListener('click', () => openBuilt(d.gguf)); b.appendChild(o); }
    else b.appendChild(el('p', 'small muted', 'Tick “also write a GGUF” and build again to run it in Generate and the other views.'));
  }
  function openBuilt(gguf) { if (window.PSIOpenModel) { location.hash = 'generate'; window.PSIOpenModel(gguf); } }

  async function listModels() {
    const box = $('cmModels'); box.textContent = '';
    let r; try { r = await api({ action: 'models' }); } catch (_) { return; }
    if (!r.models.length) { box.appendChild(el('p', 'small muted', 'None yet.')); return; }
    for (const m of r.models) { const x = m.record; card(box, x.recipe_name || short(x.output_sha256), `${short(x.output_sha256)} · ${x.mode} · ${x.matches_recipe ? 'matches its recipe' : 'differs from its recipe'}${x.eval ? ` · loss ${x.eval.mean}` : ''} · ${String(x.built).slice(0, 10)}`, [['Verify', () => { show('verify'); verifyNow('model', m.link || m.dir.split('/').pop()); }], ...(m.gguf ? [['Open in Generate', () => openBuilt(m.gguf)]] : [])]); }
  }

  // ── branches: contribute, pick for a recipe, publish ──
  let BRANCHES = [];
  async function listBranches() {
    let r; try { r = await api({ action: 'branches' }); } catch (_) { return; }
    BRANCHES = r.branches || [];
    const box = $('cmBranches'), pick = $('cmPick'); box.textContent = ''; pick.textContent = '';
    if (!BRANCHES.length) { box.appendChild(el('p', 'small muted', 'None yet.')); pick.appendChild(el('p', 'small muted', 'Train a branch in Contribute, or import a recipe: its branches can be recombined here.')); }
    for (const e of BRANCHES) {
      const x = e.record, ev = x.eval || {}, sub = `${(x.contributor || {}).name || '?'} · ${(x.source || {}).subject || '—'} · ${(x.source || {}).licence} · own text ${ev.own_text ? `${ev.own_text.before} → ${ev.own_text.after}` : '?'} · suite ${ev.before ? `${ev.before.mean} → ${ev.after.mean}` : '?'} · ${sigText(e.signatures_checked)}${e.published ? ` · on ${e.published.repo}` : ''}`;
      card(box, x.name, sub, [['Verify', () => { show('verify'); verifyNow('branch', x.delta_sha256.slice(0, 16)); }], ...(hfOn() ? [['Prepare to publish', () => prepare('branch', x.delta_sha256.slice(0, 16), $('cmRepo').value.trim(), $('cmOutbox'))]] : [])]);
      const l = el('label', 'small cm-pickrow'); const c = el('input'); c.type = 'checkbox'; c.value = x.delta_sha256.slice(0, 16); l.append(c, ` ${x.name} (${(x.contributor || {}).name || '?'}, ${(x.source || {}).subject || '—'})`); pick.appendChild(l);
    }
  }
  $('cmTrain').addEventListener('click', () => {
    const practice = $('cmPractice').checked;
    start({ action: 'train', confirm: true, practice, text: practice ? '' : $('cmText').value, name: $('cmBName').value || (practice ? 'peter pan' : 'my branch'), subject: $('cmSubject').value,
      licence: practice ? 'public domain' : $('cmLicence').value, steps: +$('cmSteps').value || 200 }, (d) => {
      if (!d) return; const b = $('cmTrained'); b.textContent = '';
      b.appendChild(el('p', 'ok', `Branch ${d.name} (${d.branch}) trained in ${d.seconds} s${d.signed ? ', signed' : ', not signed (private)'}.`));
      b.appendChild(el('p', 'small', `On held-out pages of your own text: ${d.own_before} → ${d.own_after}. On the line’s evaluation suite: ${d.before} → ${d.after} (lower is better).`));
      listBranches();
    });
  });
  $('cmPractice').addEventListener('change', () => { $('cmText').disabled = $('cmPractice').checked; });
  function picked() { return [...$('cmPick').querySelectorAll('input:checked')].map((c) => c.value); }
  $('cmMake').addEventListener('click', () => { const bs = picked(); if (!bs.length) { PSI.status($('cmStat'), 'tick at least one branch', 'warn'); return; }
    start({ action: 'recipe_make', confirm: true, branches: bs, plugin: $('cmPlugin').value, name: $('cmRName').value || 'my recipe' }, async (d) => { if (d) { await listRecipes(); check(d.recipe); } }); });
  $('cmSearchBest').addEventListener('click', () => { const bs = picked(); if (bs.length < 2) { PSI.status($('cmStat'), 'tick two or more branches to search among', 'warn'); return; }
    start({ action: 'search', confirm: true, branches: bs, plugin: $('cmPlugin').value }, (d) => { if (!d) return; for (const c of $('cmPick').querySelectorAll('input')) c.checked = d.chosen.includes(c.value);
      const b = $('cmBuilt'); b.textContent = ''; b.appendChild(el('p', '', `Best subset: ${d.chosen.length} of ${bs.length} (score ${d.score}; all of them ${d.everything}). Ticked above.`)); b.appendChild(el('p', 'small muted', d.note)); }); });
  async function listPlugins() {
    try { const r = await api({ action: 'plugins' }); PLUGINS = r.plugins || []; } catch (_) { return; }
    const s = $('cmPlugin'), cur = s.value; s.textContent = ''; for (const p of PLUGINS) { const o = el('option', '', `${p.name}${p.linear ? '' : ' (not linear)'}`); o.value = p.name; s.appendChild(o); }
    s.value = cur || 'average'; about();
  }
  function about() { const p = PLUGINS.find((x) => x.name === $('cmPlugin').value); $('cmPluginAbout').textContent = p ? `${p.about}${p.linear ? ' Linear: if one branch is ever lost, the published model can rebuild it exactly.' : ' Not linear: a lost branch cannot be rebuilt from the published model, so mirror its branches.'}` : ''; }
  $('cmPlugin').addEventListener('change', about);

  // ── publishing: prepare (dry run) shows the files; only then a confirmed push ──
  function prepare(what, id, repo, box) {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repo || '')) { PSI.status($('cmStat'), 'type the repo as owner/name first', 'warn'); return; }
    start({ action: 'publish', what, id, repo, dry_run: true }, (d) => {
      if (!d) return; const o = el('div', 'cm-out');
      o.appendChild(el('p', '', `Ready to send ${d.files.length} files (${kb(d.bytes)}) to huggingface.co/${d.repo}:`)); o.appendChild(el('pre', 'small', d.files.join('\n')));
      o.appendChild(el('p', 'small muted', `Staged in ${d.folder}. Public on Hugging Face unless the repo is private; anyone can then copy these files.`));
      const b = el('button', 'hf-send', `Publish to ${d.repo}`); b.addEventListener('click', () => { if (!OV || !OV.hf_token) { PSI.status($('cmStat'), 'add a Hugging Face token in Settings > Hugging Face first', 'warn'); return; }
        start({ action: 'publish', what, id, repo, confirm: true }, (r) => { if (r) { o.appendChild(el('p', 'ok', `Published: commit ${short(r.commit)} at ${r.time || '?'}. That commit is the public record of when it existed.`)); listBranches(); listRecipes(); } }); });
      o.appendChild(b); box.appendChild(o);
    });
  }

  // ── verify ──
  async function fillVerify() {
    const s = $('cmVItem'), cur = s.value; s.textContent = '';
    const add = (v, t) => { const o = el('option', '', t); o.value = v; s.appendChild(o); };
    for (const e of BRANCHES) add(`branch:${e.record.delta_sha256.slice(0, 16)}`, `branch · ${e.record.name} (${(e.record.contributor || {}).name || '?'})`);
    try { const r = await api({ action: 'recipes' }); for (const x of r.recipes) add(`recipe:${x.id}`, `recipe · ${x.name}`); } catch (_) { /* none */ }
    try { const r = await api({ action: 'models' }); for (const m of r.models) add(`model:${m.link || m.dir.split('/').pop()}`, `model · ${m.record.recipe_name} (${short(m.record.output_sha256)})`); } catch (_) { /* none */ }
    if (cur) s.value = cur;
  }
  function verifyNow(kind, id) {
    const box = $('cmReport'); box.textContent = '';
    start({ action: 'verify', kind, id, replay: +$('cmVReplay').value || 0, online: $('cmVOnline').checked }, (d) => {
      if (!d) return; box.appendChild(el('p', 'cm-title', `${d.name || ''}  (${d.kind} ${d.id})`));
      for (const r of d.rows) {
        const p = el('div', 'cm-claim'); p.appendChild(el('span', r.result === 'yes' ? 'ok' : r.result === 'no' ? 'bad' : 'muted', r.result === 'yes' ? '✓' : r.result === 'no' ? '✗' : '·'));
        p.appendChild(el('span', `cm-strength ${r.strength}`, r.strength)); p.appendChild(el('span', '', r.claim)); box.appendChild(p);
        if (r.detail) box.appendChild(el('div', 'small muted cm-detailrow', r.detail));
      }
    });
  }
  $('cmVerify').addEventListener('click', () => { const [k, id] = String($('cmVItem').value || '').split(':'); if (k) verifyNow(k, id); });

  // ── profile, signing, keys, token ──
  function fillMe() {
    if (!OV) return; const p = OV.identity.profile, s = OV.identity.signing;
    $('cmUser').value = p.username || ''; $('cmAnon').checked = p.anonymous !== false; $('cmDisplay').value = p.display_name || ''; $('cmBio').value = p.bio || '';
    $('cmHfUser').value = ((p.links || []).find((l) => l.host === 'huggingface') || {}).user || ''; $('cmGhUser').value = ((p.links || []).find((l) => l.host === 'github') || {}).user || '';
    for (const c of document.querySelectorAll('[data-cm-private]')) c.checked = (p.visibility || {})[c.dataset.cmPrivate] === 'private';
    for (const r of document.querySelectorAll('input[name=cmLevel]')) r.checked = r.value === s.level;
    const ks = (OV.keys && OV.keys.keys) || []; const sel = $('cmKey'); sel.textContent = '';
    for (const k of ks) { const o = el('option', '', `${k.name} (${k.id})${k.replaced_by ? ' · replaced' : ''}`); o.value = k.id; sel.appendChild(o); }
    if (OV.em_key_here) { const o = el('option', '', 'The Emergence Machine (official releases only)'); o.value = 'em'; sel.appendChild(o); }
    sel.value = s.key || (ks[0] && ks[0].id) || '';
    const kbx = $('cmKeys'); kbx.textContent = '';
    if (!ks.length) kbx.appendChild(el('p', 'small muted', 'No keys yet. You need none to use PSI; make one if you want your work signed.'));
    for (const k of ks) card(kbx, k.name, `${k.id} · made ${String(k.created).slice(0, 10)}${k.replaced_by ? ` · replaced by ${k.replaced_by}` : ''}${s.key === k.id && s.level !== 'private' ? ' · signing with this' : ''}`);
    $('cmTokenState').textContent = OV.hf_token ? 'a token is saved' : 'no token saved';
    if ($('cmTokenWhere')) $('cmTokenWhere').textContent = OV.secrets_where ? `Kept ${OV.secrets_where}; ` : 'Kept on this computer; ';
    $('cmVaultFx').textContent = `One key is one identity: everything signed with it can be linked. Keep several if you want parts of your work kept apart. Keys are encrypted with a secret kept ${OV.secrets_where || 'on this computer'}.`;
    $('cmSigningFx').textContent = `Now: ${s.level === 'private' ? 'private, nothing is signed' : `signing with ${s.key}`}. The choice applies to everything PSI records from now on: branches, recipes, built models, Timeline placements and grafts, feature rings, grown seeds. The Emergence Machine signs only through its own button, and only on a computer that has its key${OV.em_key_here ? ' (this one does)' : ''}.`;
    if (!$('cmIdRepo').value && p.links && p.links.find((l) => l.host === 'huggingface')) $('cmIdRepo').value = `${p.links.find((l) => l.host === 'huggingface').user}/psi-identity`;
  }
  function profileFromForm() {
    const vis = {}; for (const c of document.querySelectorAll('[data-cm-private]')) vis[c.dataset.cmPrivate] = c.checked ? 'private' : 'public';
    const links = []; if ($('cmHfUser').value.trim()) links.push({ host: 'huggingface', user: $('cmHfUser').value.trim(), url: `https://huggingface.co/${$('cmHfUser').value.trim()}` });
    if ($('cmGhUser').value.trim()) links.push({ host: 'github', user: $('cmGhUser').value.trim(), url: `https://github.com/${$('cmGhUser').value.trim()}` });
    return { username: $('cmUser').value, display_name: $('cmDisplay').value, bio: $('cmBio').value, anonymous: $('cmAnon').checked, links, visibility: { ...vis, links: vis.links || 'public' } };
  }
  // Profile & keys and the Hugging Face token live in Settings (setProfile, setHf): their messages show there.
  const say = (m, k) => PSI.status($('cmMeStat'), m, k);
  $('cmSaveProfile').addEventListener('click', async () => { try { await api({ action: 'profile_set', profile: profileFromForm() }); say('profile saved', 'ok'); refresh(); } catch (e) { say(e.message, 'error'); } });
  $('cmImportHf').addEventListener('click', async () => {
    const u = $('cmHfUser').value.trim(); if (!u) { say('type a Hugging Face user name first', 'warn'); return; }
    try { const r = (await api({ action: 'profile_import_hf', user: u })).imported; if (!$('cmUser').value) $('cmUser').value = r.username; $('cmDisplay').value = r.display_name || ''; say(`filled in from Hugging Face (${r.models || 0} models, ${r.datasets || 0} datasets). Check, then Save.`, 'ok'); }
    catch (e) { say(`Hugging Face: ${e.message}`, 'error'); }
  });
  $('cmSaveSigning').addEventListener('click', async () => {
    const lv = (document.querySelector('input[name=cmLevel]:checked') || {}).value || 'private';
    try { await api({ action: 'signing_set', level: lv, key: $('cmKey').value }); say(lv === 'private' ? 'private: nothing will be signed' : `signing with ${$('cmKey').value}`, 'ok'); refresh(); } catch (e) { say(e.message, 'error'); }
  });
  $('cmKCreate').addEventListener('click', async () => {
    try {
      const k = await api({ action: 'key_create', name: $('cmKName').value || 'my key' }); const o = $('cmKOut'); o.textContent = '';
      o.appendChild(el('p', 'ok', `Key ${k.name} (${k.id}) made.`)); o.appendChild(el('p', 'warn', 'Recovery code: write it down now. It is shown once, and it is the only way to bring this key to another computer.'));
      o.appendChild(el('pre', 'cm-code', k.recovery_code)); const hide = el('button', 'ghost small', 'I wrote it down: hide it'); hide.addEventListener('click', () => { o.textContent = ''; }); o.appendChild(hide);
      $('cmKName').value = ''; refresh();
    } catch (e) { say(e.message, 'error'); }
  });
  $('cmKRestore').addEventListener('click', async () => { try { const k = await api({ action: 'key_restore', name: $('cmKName').value || 'restored key', code: $('cmKCode').value }); $('cmKCode').value = ''; say(`restored ${k.name} (${k.id})`, 'ok'); refresh(); } catch (e) { say(e.message, 'error'); } });
  $('cmTokenSave').addEventListener('click', async () => { try { await api({ action: 'hf_token_set', token: $('cmToken').value }); $('cmToken').value = ''; say('token saved', 'ok'); refresh(); } catch (e) { say(e.message, 'error'); } });
  $('cmTokenClear').addEventListener('click', async () => { try { await api({ action: 'hf_token_clear' }); say('token removed', 'ok'); refresh(); } catch (e) { say(e.message, 'error'); } });
  $('cmIdPrepare').addEventListener('click', () => prepare('identity', null, $('cmIdRepo').value.trim(), $('cmKOut')));

  // ── showing the view ──
  let first = true;
  const onShow = () => { if (first) { first = false; let t = 'recipes'; try { t = localStorage.getItem('psi-cm-tab') || t; } catch (_) { /* private mode */ } show(t === 'me' ? 'recipes' : t); } refresh(); };
  // Profile & keys moved to Settings: the Commons button opens it there; Settings loads it when shown.
  $('cmToProfile').addEventListener('click', () => { const t = document.querySelector('[data-tab=settings]'); if (t) t.click(); setTimeout(() => { const b = $('setProfile'); if (b) b.scrollIntoView({ block: 'start' }); }, 50); });
  const setTab = $('tab-settings');
  const loadMe = async () => { try { OV = await api({ action: 'overview' }); fillMe(); } catch (e) { say(e.message, 'error'); } };
  if (setTab) new MutationObserver(() => { if (!setTab.classList.contains('hidden')) loadMe(); }).observe(setTab, { attributes: true, attributeFilter: ['class'] });
  new MutationObserver(() => { if (!sec.classList.contains('hidden')) onShow(); }).observe(sec, { attributes: true, attributeFilter: ['class'] });
  if (!sec.classList.contains('hidden')) onShow();
})();
