// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Shared UI helpers for every view: window.PSI. Loaded before the view scripts (index.html).
//
//   PSI.goto(view, focusId?)              open a view (and optionally scroll to and focus a control)
//   PSI.link(text, view, focusId?)        an <a> that opens a view; PSI.loadModelLink() is the standard
//                                         "Load a model in Settings" link (PSI.NEED_MODEL is its wording)
//   PSI.err(e, {action, kind, text})            one message area: plain-language text for known errors, an action
//                                         link, a dismiss button; stays until dismissed or replaced
//   PSI.explain(e)                        the same mapping without showing it: {text, action, raw}
//   PSI.state / setState / onState        what is ready: model (loaded file or null), loading, engine, checkpoints, key
//   PSI.requires(btn, needs)              disable a button until 'model' | 'engine' | 'checkpoints' | 'key' is ready,
//                                         with a visible reason and link next to it; follows state changes
//   PSI.block(btn, key, reason|null)      the general form: disable for a named reason (e.g. 'busy'), or clear it
//   PSI.setRunning(view, on), PSI.setDone(view), PSI.run(view, fn)
//                                         explicit run state for the nav dots and the "n of 7 run" counter
//   PSI.progress(host, {total, onCancel, label, view, track})
//                                         "step i of n · about M min left" with a Stop button; returns
//                                         {step(i, note), setTotal(n), cancelled, cancel(), finish(text), fail(text), remove()}.
//                                         Marks the view running (track: false to leave the dot alone); finish() marks it done.
//   PSI.status(el, text, kind)            one status-line style: kind '' | 'busy' | 'ok' | 'warn' | 'error'
//   PSI.canvasLabel(canvas, text)         role="img" + aria-label on a drawing
//   PSI.term(text, key?)                  a <span class="term"> with a hover/focus definition (PSI.TERMS)
//   PSI.helpUrl(view)                     help.html#<section> for a view
//   PSI.setChange(key, info|null)         the model as changed (steering, LoRA, a Tipping-point cut): {label, short, engine:
//                                         'local'|'inside', view, focus}; a header chip shows every active change on every view
//   PSI.changesFor(engine), PSI.changeParts(engine)
//                                         the active changes; conditions-line parts for a result from that engine
//   PSI.cond(anchor, parts, lead)         a result's conditions line right after `anchor`: model, engine, sampling, changes
//                                         (parts: strings or {t, chg, frame, est, warn, term, title}); replaced on the next run
//                                         (warn: a condition that did not fit, e.g. slow for this GPU or an answer not reached)
//   (no API) every 20 s while visible: the inside engine's state lights the "changed" chip for a cut made from outside
//                                         (an agent over MCP), and #agentChip shows "agent: N calls · $X spent" (last 10 min)
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const PSI = window.PSI = window.PSI || {};

  // ── navigation ────────────────────────────────────────────────────────────
  const navBtn = (view) => document.querySelector(`#tabs button[data-tab="${CSS.escape(view)}"]`);
  PSI.currentView = () => { const b = document.querySelector('#tabs button[aria-current="page"]'); return b ? b.dataset.tab : 'generate'; };
  PSI.goto = (view, focusId) => {
    if (view === 'paths') { view = 'generate'; focusId = focusId || 'genPaths'; }   // Paths & attractors is now part of Generate
    const b = navBtn(view); if (!b) return false;
    if (b.getAttribute('aria-current') !== 'page') b.click();
    if (focusId) setTimeout(() => { const n = $(focusId); if (n) { n.scrollIntoView({ block: 'center' }); try { n.focus({ preventScroll: true }); } catch (_) { /* not focusable */ } } }, 30);
    return true;
  };
  PSI.link = (text, view, focusId) => {
    const a = el('a', 'psi-link', text); a.href = `#${view}`;
    a.addEventListener('click', (e) => { e.preventDefault(); PSI.goto(view, focusId); });
    return a;
  };
  PSI.NEED_MODEL = 'Load a model in Settings';
  PSI.loadModelLink = () => PSI.link(PSI.NEED_MODEL, 'settings', 'stModelPick');

  // help.html sections for each view (help.html#<view> wins when the manual has that anchor)
  const HELP = { generate: 'watch', neural: 'watch', paths: 'watch', read: 'surface', tokens: 'surface', compare: 'surface',
    understand: 'exp', emergence: 'exp', lesion: 'exp', sensing: 'exp', fight: 'exp', connectors: 'exp', ladder: 'exp',
    anatomy: 'model', features: 'model', timeline: 'exp', file: 'model', lineage: 'seed', commons: 'commons', settings: 'settings' };
  let helpIds = null;
  fetch('help.html').then((r) => (r.ok ? r.text() : '')).then((t) => { helpIds = new Set([...t.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])); updateHelp(); }).catch(() => { /* benign: help links fall back to the section map */ });
  PSI.helpUrl = (view) => { const v = view || PSI.currentView(); return `help.html#${helpIds && helpIds.has(v) ? v : HELP[v] || 'views'}`; };
  function updateHelp() { document.querySelectorAll('a[data-help-link]').forEach((a) => { a.href = PSI.helpUrl(); }); }

  // ── state: what is ready ──────────────────────────────────────────────────
  const state = PSI.state = { model: null, loading: false, engine: false, checkpoints: false, key: false, known: false, changes: {} };
  const watchers = [];
  PSI.onState = (fn) => { watchers.push(fn); try { fn(state); } catch (_) { /* a watcher's own problem */ } return () => { const i = watchers.indexOf(fn); if (i >= 0) watchers.splice(i, 1); }; };
  PSI.setState = (patch) => {
    Object.assign(state, patch || {});
    watchers.slice().forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } });
    refreshGates();
  };
  // Ask the app what is set up. Reads files only: never starts an engine.
  PSI.refreshState = async () => {
    const patch = { known: true };
    try { const s = await psi.settings(); patch.key = !!(s.external && s.external.keySet); patch.checkpointsDir = (s.trace && s.trace.checkpoints) || ''; } catch (_) { /* keep */ }
    try {
      const r = await psi.setupCheck();
      const inside = (r.problems || []).filter((p) => /inside engine/i.test(p));
      patch.engine = !!r.inside && !inside.length;
      patch.insideModel = r.inside ? String(r.inside).split(/[\\/]/).filter(Boolean).pop() : '';   // the folder the inside engine reads
      patch.engineProblem = inside[0] || '';
      patch.checkpoints = patch.engine && !!patch.checkpointsDir;
    } catch (_) { /* keep */ }
    PSI.setState(patch);
    return state;
  };

  // ── gating: disable a control for named reasons, with a visible note ─────
  // A gated control's own `disabled` (set by its view while it runs) is kept apart from the reasons here:
  // the control is disabled while either says so.
  function protoDisabled(n) { let p = Object.getPrototypeOf(n); while (p) { const d = Object.getOwnPropertyDescriptor(p, 'disabled'); if (d) return d; p = Object.getPrototypeOf(p); } return null; }
  function adopt(n) {
    if (n._psiBlocks) return;
    const d = protoDisabled(n); if (!d) return;
    n._psiBlocks = new Map(); n._psiWant = d.get.call(n); n._psiSet = (v) => d.set.call(n, v);
    Object.defineProperty(n, 'disabled', { configurable: true, get() { return d.get.call(this); }, set(v) { this._psiWant = !!v; d.set.call(this, !!v || this._psiBlocks.size > 0); } });
  }
  const NEEDS = {
    model: () => (state.model ? null : state.loading ? { text: 'Loading the model…' } : { text: 'Needs a loaded model:', link: () => PSI.loadModelLink() }),
    engine: () => (state.engine ? null : { text: 'Needs the inside-the-model engine:', link: () => PSI.link('Set its weights folder in Settings', 'settings', 'stTraceDir') }),
    checkpoints: () => (state.checkpoints ? null : { text: 'Needs training checkpoints:', link: () => PSI.link('Set the checkpoints folder in Settings', 'settings', 'stCkpt') }),
    key: () => (state.key ? null : { text: 'Needs an OpenRouter key:', link: () => PSI.link('Add a key in Settings', 'settings', 'stKey') }),
  };
  const gated = new Set();
  PSI.block = (n, key, reason) => {
    if (!n) return; adopt(n); if (!n._psiBlocks) return;
    if (reason) n._psiBlocks.set(key, typeof reason === 'string' ? { text: reason } : reason); else n._psiBlocks.delete(key);
    n._psiSet(n._psiWant || n._psiBlocks.size > 0);
    gated.add(n); notes(n.parentElement);
  };
  PSI.requires = (n, needs) => {
    if (!n) return; adopt(n);
    n._psiNeeds = [...new Set([...(n._psiNeeds || []), ...[].concat(needs)])];
    gated.add(n); gate(n);
  };
  function gate(n) {
    const miss = (n._psiNeeds || []).map((k) => NEEDS[k] && NEEDS[k]()).find(Boolean);
    PSI.block(n, 'needs', state.known || state.model ? miss || null : null);
  }
  function refreshGates() { gated.forEach((n) => { if (!n.isConnected) { gated.delete(n); return; } if (n._psiNeeds) gate(n); }); }
  // One note per row: the first reason among the row's blocked controls, next to the first of them.
  function notes(row) {
    if (!row) return;
    let note = [...row.children].find((c) => c.classList.contains('req-note'));
    const blocked = [...row.children].filter((c) => c._psiBlocks && c._psiBlocks.size);
    const B = blocked.length && blocked[0]._psiBlocks, why = B && (B.get('needs') || [...B.values()].pop());   // a missing requirement explains more than "busy"
    if (!why) { if (note) note.remove(); return; }
    if (!note) note = el('span', 'req-note');
    if (note.previousElementSibling !== blocked[0]) blocked[0].after(note);
    note.textContent = why.text; if (why.link) { note.append(' '); note.appendChild(why.link()); }
  }

  // ── the model as changed: steering, LoRA, Tipping-point cuts ──────────────
  // A result made while a change is active on its engine says so in its conditions line, and the header chip says so on
  // every view, so a changed model is never read as the unchanged one. The two engines change separately: steering and
  // LoRA reload the local model (llama.cpp); a cut stays in the inside engine's model (PyTorch) until it is restored.
  const ENGINE = { local: 'the local model (llama.cpp)', inside: 'the inside engine (PyTorch)' };
  PSI.setChange = (key, info) => { const c = { ...state.changes }; if (info) c[key] = info; else delete c[key]; PSI.setState({ changes: c }); };
  PSI.changesFor = (engine) => Object.values(state.changes || {}).filter((c) => !engine || c.engine === engine);
  PSI.changeParts = (engine) => {
    const on = PSI.changesFor(engine).map((c) => ({ t: c.label, chg: true, term: 'changed model', title: `${ENGINE[c.engine]} is changed: this is not the unchanged model's result` }));
    if (!on.length) on.push({ t: 'model unchanged', title: 'no steering, LoRA or cut was active on this engine' });
    const other = PSI.changesFor().filter((c) => c.engine !== engine);
    if (other.length) on.push(`${other.map((c) => c.short || c.label).join(', ')}: on ${ENGINE[other[0].engine]} only, not here`);
    return on;
  };
  // A view's own "E-level framing" box starts as the app-wide lens says (Settings > Think in E-levels) and follows
  // it until it is changed in the view (or set there: data-own).
  PSI.followLens = (box) => {
    const lo = document.getElementById('lensOn'); if (!box || !lo) return;
    let sync = false;
    const mirror = () => { if (box.dataset.own || box.checked === lo.checked) return; sync = true; box.checked = lo.checked; box.dispatchEvent(new Event('change')); sync = false; };
    box.addEventListener('change', () => { if (!sync) box.dataset.own = '1'; });
    lo.addEventListener('change', mirror); mirror();
  };
  PSI.cond = (anchor, parts, lead = 'run on') => {
    if (!anchor) return null;
    let p = anchor.nextElementSibling; if (!p || !p.classList.contains('cond')) { p = el('p', 'cond'); anchor.after(p); }
    p.textContent = ''; p.appendChild(el('span', 'cond-k', `${lead} `));
    const list = (parts || []).filter(Boolean); p.classList.toggle('cond-changed', list.some((x) => x.chg));
    list.forEach((x, i) => {
      if (i) p.append(' · ');
      const o = typeof x === 'string' ? { t: x } : x, s = o.term ? PSI.term(o.t, o.term) : el('span', null, o.t);
      ['chg', 'frame', 'est', 'warn'].forEach((k) => { if (o[k]) s.classList.add(`cond-${k}`); }); if (o.title) s.title = o.title;
      p.appendChild(s);
    });
    return p;
  };
  let chip = null;
  function changeChip() {
    const host = document.querySelector('header.top .model'); if (!host) return;
    if (!chip) {
      chip = el('button', 'chg-chip'); chip.type = 'button'; chip.id = 'chgChip';
      const mn = $('modelName'); if (mn) mn.after(chip); else host.prepend(chip);
      chip.addEventListener('click', () => { const c = PSI.changesFor()[0]; if (c) PSI.goto(c.view || 'settings', c.focus); });
    }
    const all = PSI.changesFor(); chip.hidden = !all.length; if (!all.length) return;
    chip.textContent = `changed: ${all.map((c) => c.short || c.label).join(' · ')}`;
    chip.title = `${all.map((c) => `${c.label}, on ${ENGINE[c.engine]}`).join('\n')}\nResults made while a change is active are marked “changed”. Click to see where to undo it.`;
  }
  // Every Tipping-point reply carries the cut now in force (src/lesion.py), so watch them all. Best effort: if the app's
  // bridge cannot be wrapped, the Tipping point view can still report with PSI.setChange('lesion', …).
  const pct = (a, b) => `${Math.round((100 * a) / (b || 1))}%`;
  function cutLabel(s) {
    if (s.kind === 'structural') return `cut: ${pct(s.neurons, s.of_neurons)} of neurons, ${pct(s.heads, s.of_heads)} of heads (${s.mode})`;
    if (s.kind === 'weights') return `cut: ${Math.round(s.frac * 100)}% of ${s.scope} weights zeroed (${s.criterion})`;
    if (s.kind === 'manual') return `cut by hand: ${s.heads} heads, ${s.neurons} neurons`;
    if (s.kind === 'concept') return `cut: concept “${s.concept}” deleted`;
    return `cut: ${s.kind}`;
  }
  function watchLesion() {
    const api = window.psi; if (!api || typeof api.lesion !== 'function' || api.lesion._psiWatched) return;
    const raw = api.lesion;
    const wrapped = async (...a) => {
      const r = await raw(...a);
      const s = r && (r.state || (r.points ? { kind: 'none' } : null));   // a server-side sweep restores at its end
      if (s) PSI.setChange('lesion', s.kind && s.kind !== 'none' ? { label: `Tipping-point ${cutLabel(s)}`, short: 'cut', engine: 'inside', view: 'lesion', focus: 'lsRestore' } : null);
      return r;
    };
    wrapped._psiWatched = true;
    try { api.lesion = wrapped; } catch (_) { /* a frozen bridge: try the copy below */ }
    if (api.lesion !== wrapped) try { Object.defineProperty(window, 'psi', { value: { ...api, lesion: wrapped }, configurable: true, writable: true }); } catch (_) { console.warn('[psi] cannot follow Tipping-point cuts: the bridge is frozen'); }
  }

  // ── changes made from outside the window, and the agent session ───────────
  // An agent (over MCP) can cut the inside model without the window seeing a Tipping-point reply, so the inside engine's
  // own state is read every 20 s while the window is visible: a cut in force lights the "changed" chip on every view.
  // Only a positive reading changes the chip: no reading (engine stopped, busy past the 1.5 s probe) leaves it as it is.
  // The same tick asks the MCP server how many calls outside agents made in the last 10 minutes and what they spent.
  const POLL_MS = 20000;
  let agentChip = null, polling = false;
  async function pollInside() {
    const api = window.psi; if (!api || typeof api.traceStatus !== 'function') return;
    let st = null; try { st = await api.traceStatus(); } catch (_) { return; }   // not read: leave the chip as it is
    const c = st && st.ok !== false && st.conditions; if (!c || !c.changed || !('lesion' in c.changed)) return;
    const lz = c.changed.lesion, cur = state.changes.lesion;
    if (lz && lz.kind && lz.kind !== 'none') {
      const label = `Tipping-point ${cutLabel(lz)}`;
      if (!cur || cur.label !== label) PSI.setChange('lesion', { label, short: 'cut', engine: 'inside', view: 'lesion', focus: 'lsRestore', source: 'read from the inside engine (it may have been made by an agent)' });
    } else if (cur) PSI.setChange('lesion', null);
  }
  async function pollAgents() {
    const api = window.psi; if (!api || typeof api.mcp !== 'function') return;
    let s = null; try { const r = await api.mcp({ jsonrpc: '2.0', id: Date.now(), method: 'psi/agent_session', params: { minutes: 10 } }); s = r && r.result; } catch (_) { s = null; }
    if (!agentChip) {
      const host = document.querySelector('header.top .model'); if (!host) return;
      agentChip = el('button', 'agent-chip ghost'); agentChip.type = 'button'; agentChip.id = 'agentChip'; agentChip.hidden = true;
      const after = $('gpuChip') || $('chgChip') || $('modelName'); if (after) after.after(agentChip); else host.appendChild(agentChip);
      agentChip.addEventListener('click', () => PSI.goto('settings', 'stAgents'));
    }
    if (!s || !s.calls) { agentChip.hidden = true; return; }   // no agent calls in 10 minutes (or this build has no session count)
    agentChip.hidden = false;
    agentChip.textContent = `agent: ${s.calls} call${s.calls === 1 ? '' : 's'} · $${(+s.usd || 0).toFixed(s.usd && s.usd < 0.01 ? 4 : 2)} spent`;
    agentChip.classList.toggle('warn', !!s.refused_or_failed);
    agentChip.title = `An outside agent used this Inspector over MCP in the last ${s.window_minutes} minutes: ${s.calls} call(s), ${s.refused_or_failed || 0} refused or failed, $${(+s.usd || 0).toFixed(4)} spent on OpenRouter.\nLast: ${s.last_tool || 'not recorded'}${s.last_at ? ` at ${new Date(s.last_at).toLocaleTimeString()}` : ''}.\nAgents may change models: ${s.allow_changes ? 'yes' : 'no (Settings)'}. Prometheus in this window is not counted.\nClick for Settings.`;
    agentChip.setAttribute('aria-label', `Agent session: ${s.calls} calls, ${(+s.usd || 0).toFixed(2)} dollars spent in the last ${s.window_minutes} minutes. Open Settings.`);
  }
  async function tick() {
    if (polling || document.hidden) return; polling = true;
    try { await Promise.all([pollInside(), pollAgents()]); } finally { polling = false; }
  }
  function watchOutside() {
    if (!window.psi) return;
    setInterval(tick, POLL_MS);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });
    setTimeout(tick, 1500);
  }

  // ── errors ────────────────────────────────────────────────────────────────
  const SET = (label, focus) => ({ label, view: 'settings', focus });
  const KNOWN = [
    // First: the message names an engine ("Inside engine: …") but is about memory, not missing weights.
    [/GPU is overfull|spilled into system memory/i, () => ({ text: 'The GPU is full: a model spilled into system memory and runs many times slower.', action: { label: 'See what holds the GPU', view: 'settings', focus: 'setMachine' } })],
    [/Choose external models in Settings/i, () => ({ text: 'Choose one or more external models first.', action: SET('Choose them in Settings', 'stFind') })],
    [/openrouter|api[ -]?key|no key|\b401\b|unauthori[sz]ed/i, () => ({ text: 'This needs an OpenRouter key, and none is saved (or it was refused).', action: SET('Add a key in Settings', 'stKey') })],
    [/spending cap|cap reached|over the cap/i, () => ({ text: 'The spending cap for external models is reached.', action: SET('Raise or reset it in Settings', 'stCap') })],
    [/checkpoint/i, (m) => (/(no |not |missing|ENOENT|cannot|could not|find)/i.test(m) ? { text: 'No training checkpoints were found where the settings point.', action: SET('Set the checkpoints folder in Settings', 'stCkpt') } : null)],
    [/safetensors|no model weights|weights folder|inside engine/i, () => ({ text: 'The inside-the-model engine has no weights to read.', action: SET('Set its weights folder in Settings', 'stTraceDir') })],
    [/spawn\s+\S*python|python(3)?\s*(was )?not found|no python/i, () => ({ text: 'The inside-the-model engine (Python) could not start on this computer.', action: SET('Check the setup in Settings', 'stSetupCheck') })],
    [/open a gguf model|no model (is )?(loaded|open)|engine is not running|not running|not started|ECONNREFUSED|socket hang up/i, () => ({ text: 'No model is loaded yet, so there is nothing to run.', action: SET(PSI.NEED_MODEL, 'stModelPick') })],
    [/ENOENT|no such file|does not exist|cannot find|not a folder/i, () => ({ text: 'A file the app needs is not where the settings say it is.', action: SET(PSI.NEED_MODEL, 'stModelPick') })],
  ];
  PSI.explain = (e) => {
    const raw = String((e && e.message) || e || 'Something went wrong.');
    for (const [re, f] of KNOWN) if (re.test(raw)) { const r = f(raw); if (r) return { ...r, raw }; }
    return { text: raw, action: null, raw: '' };
  };
  let box = null, timer = 0, lastText = '';
  function errBox() {
    if (box) return box;
    box = $('err');
    if (!box) { box = el('div', 'err hidden'); box.id = 'err'; document.body.appendChild(box); }
    box.textContent = ''; box.setAttribute('role', 'alert');
    const msg = el('p', 'err-msg'); msg.id = 'errMsg';
    const act = el('span', 'err-act');
    const raw = el('details', 'err-raw'); raw.appendChild(el('summary', null, 'technical detail')); raw.appendChild(el('code'));
    const x = el('button', 'ghost err-x', '×'); x.type = 'button'; x.setAttribute('aria-label', 'Dismiss this message'); x.addEventListener('click', () => PSI.clearErr());
    box.append(msg, act, raw, x);
    return box;
  }
  PSI.err = (e, opts = {}) => {
    const b = errBox(), x = PSI.explain(e), action = opts.action || x.action;
    clearTimeout(timer);
    b.classList.toggle('info', opts.kind === 'info'); b.classList.toggle('warn', opts.kind === 'warn');
    if (opts.text) x.text = opts.text;
    b.querySelector('.err-msg').textContent = x.text; lastText = x.text + (x.raw ? ` (${x.raw})` : '');
    const act = b.querySelector('.err-act'); act.textContent = '';
    if (action) {
      let a;
      if (action.view) a = PSI.link(action.label, action.view, action.focus);
      else { a = el('a', 'psi-link', action.label); a.href = action.href || '#'; if (action.onClick) a.addEventListener('click', (ev) => { ev.preventDefault(); action.onClick(); }); }
      a.addEventListener('click', () => PSI.clearErr()); act.appendChild(a);
    }
    const raw = b.querySelector('.err-raw'); raw.hidden = !x.raw; raw.open = false; raw.querySelector('code').textContent = x.raw;
    b.classList.remove('hidden');
    if (opts.kind === 'info') timer = setTimeout(() => PSI.clearErr(), opts.timeout || 4000);
    if (!opts.quiet && opts.kind !== 'info') console.warn('[psi]', x.raw || x.text);
  };
  PSI.notice = (text, opts = {}) => PSI.err(text, { ...opts, kind: 'info' });
  // A warning: it stays until dismissed or acted on (an info notice fades), and is not explained as an error.
  PSI.warn = (text, opts = {}) => { PSI.err(text, { ...opts, kind: 'warn', text, quiet: true }); if (box) box.classList.add('warn'); };

  // ── thinking models ──
  // Where a reply's thinking ends and its answer begins. Thinking blocks: <think>…</think> (Qwen3, DeepSeek-R1, LFM2.5,
  // Granite), <thinking>…</thinking>, gpt-oss's analysis channel (…<|channel|>final). A chat format can open the block
  // in the prompt itself (LFM2.5's ends with "<think>"), so the prompt's end counts. cut: it opened and never closed,
  // so every token went on thinking and none on the answer.
  const T_OPEN = /<think>|<thinking>|<\|channel\|>analysis/g, T_CLOSE = /<\/think>|<\/thinking>|<\|channel\|>final/g;
  const lastAt = (re, s) => { let m, at = -1; re.lastIndex = 0; while ((m = re.exec(s))) at = m.index + m[0].length; return at; };
  PSI.thinking = (prompt, text) => {
    const p = String(prompt || ''), t = String(text || '');
    const promptOpen = lastAt(T_OPEN, p) > lastAt(T_CLOSE, p) && /\S*$/.test(p);
    const open = lastAt(T_OPEN, t), close = lastAt(T_CLOSE, t);
    const opened = promptOpen || open >= 0, closed = close >= 0 && close > open;
    return { opened, closed, cut: opened && !closed, answer: closed ? t.slice(close).replace(/^\s*<\|message\|>/, '').trim() : opened ? '' : t };
  };
  PSI.clearErr = () => { clearTimeout(timer); if (box) box.classList.add('hidden'); lastText = ''; };
  PSI.errText = () => (box && !box.classList.contains('hidden') ? lastText : '');
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && box && !box.classList.contains('hidden') && !e.defaultPrevented) PSI.clearErr(); });

  // ── run state: nav dots and the experiments counter ───────────────────────
  const running = {};
  let done = {}; try { done = JSON.parse(localStorage.getItem('psi-exp-done') || '{}'); } catch (_) { done = {}; }
  PSI.setRunning = (view, on) => { running[view] = Math.max(0, (running[view] || 0) + (on ? 1 : -1)); dots(); };
  PSI.setDone = (view) => { done[view] = true; try { localStorage.setItem('psi-exp-done', JSON.stringify(done)); } catch (_) { /* private mode */ } dots(); };
  PSI.isDone = (view) => !!done[view];
  PSI.isRunning = (view) => !!running[view];
  // Run fn as the view's work: the dot shows it running, success marks it done, a failure goes to PSI.err.
  PSI.run = async (view, fn) => {
    PSI.setRunning(view, true);
    try { const r = await fn(); PSI.setDone(view); return r; } catch (e) { PSI.err(e); return undefined; } finally { PSI.setRunning(view, false); }
  };
  window.addEventListener('psi-em', () => dots());
  function dots() {
    let n = 0, total = 0;
    document.querySelectorAll('#tabs button[data-tab]').forEach((b) => {
      const v = b.dataset.tab, exp = b.hasAttribute('data-exp'), dot = b.querySelector('.dot'); if (!dot) return;
      if (exp && !(b.hasAttribute('data-em') && document.documentElement.classList.contains('em-off'))) { total++; if (done[v]) n++; }
      const st = running[v] ? 'run' : exp ? (done[v] ? 'done' : 'open') : '';
      dot.className = `dot ${st}`.trim();
      let sr = dot.querySelector('.sr-only'); if (!sr) { sr = el('span', 'sr-only'); dot.appendChild(sr); }
      sr.textContent = st === 'run' ? ' (running)' : st === 'done' ? ' (run before)' : st === 'open' ? ' (not run yet)' : '';
    });
    const meta = $('expMeta'); if (meta) meta.textContent = `${n} of ${total} run`;
  }

  // ── progress: "step i of n · about M min left" with a Stop button ─────────
  PSI.progress = (host, { total = 0, onCancel, label = 'step', view, track = true } = {}) => {
    const old = host && host.querySelector(':scope > .psi-progress'); if (old) old.remove();
    const w = el('div', 'psi-progress'); w.setAttribute('role', 'status');
    const bar = el('div', 'psi-bar'), fill = el('i'); bar.appendChild(fill);
    const txt = el('span', 'psi-ptext', 'starting…');
    const stop = el('button', 'ghost psi-stop', 'Stop'); stop.type = 'button';
    w.append(bar, txt, stop); if (host) host.appendChild(w);
    const v = !track ? null : view || (host && host.closest('section.tab') ? host.closest('section.tab').id.replace(/^tab-/, '') : null);
    if (v) PSI.setRunning(v, true);
    const t0 = performance.now(); let ended = false;
    const h = {
      cancelled: false, total,
      step(i, note) {
        if (ended) return;
        const n = h.total; fill.style.width = n ? `${Math.min(100, (i / n) * 100).toFixed(1)}%` : '0';
        let s = n ? `${label} ${i} of ${n}` : `${label} ${i}`;
        if (n && i > 0 && i < n) { const left = ((performance.now() - t0) / i) * (n - i) / 60000; s += left < 1 ? ' · under a minute left' : ` · about ${Math.round(left)} min left`; }
        txt.textContent = s + (note ? ` · ${note}` : '');
      },
      setTotal(n) { h.total = n; },
      cancel() { if (h.cancelled || ended) return; h.cancelled = true; stop.disabled = true; txt.textContent = 'stopping…'; if (onCancel) try { onCancel(); } catch (_) { /* ignore */ } },
      finish(text, { markDone = true } = {}) { end(); if (markDone && v) PSI.setDone(v); w.classList.add('ended'); stop.remove(); txt.textContent = text || (h.cancelled ? 'stopped' : 'done'); fill.style.width = h.cancelled ? fill.style.width : '100%'; },
      fail(text) { end(); w.classList.add('ended', 'failed'); stop.remove(); txt.textContent = text || 'stopped by an error'; },
      remove() { end(); w.remove(); },
      el: w,
    };
    function end() { if (ended) return; ended = true; if (v) PSI.setRunning(v, false); }
    stop.addEventListener('click', () => h.cancel());
    return h;
  };

  // ── status lines, canvases ────────────────────────────────────────────────
  PSI.status = (n, text, kind = '') => {
    if (!n) return;
    n.classList.add('psi-status'); ['busy', 'ok', 'warn', 'error'].forEach((k) => n.classList.toggle(`is-${k}`, k === kind));
    if (!n.hasAttribute('role')) { n.setAttribute('role', 'status'); }
    if (text instanceof Node) { n.textContent = ''; n.appendChild(text); } else n.textContent = text == null ? '' : String(text);
  };
  PSI.canvasLabel = (cv, text) => { if (!cv) return; cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', String(text || '')); };

  // ── terms: a short definition on hover and focus ──────────────────────────
  PSI.TERMS = {
    'logit lens': 'Reading the model’s running state after any layer straight out as a next-word guess. Shows how an answer forms through the layers; a readout, not a cause.',
    'residual stream': 'The running list of numbers for each piece of text that every layer reads from and adds to: what the model “currently thinks” about that position.',
    'kv cache': 'Keys and values kept for every piece read so far, in every layer, so writing the next piece does not re-read the whole prompt. The model’s working memory for context.',
    mlp: 'The feed-forward part of each layer: thousands of neurons that each respond to some pattern and add their result to the running state.',
    probe: 'A small classifier trained to read something (a class, a level, confidence) from the model’s internal numbers.',
    nats: 'A unit of surprise or difference, using natural logarithms. 1 nat is about 1.44 bits.',
    'bits per token': 'How surprised the model is, on average, by each piece of text. Lower means it predicts the text better.',
    auc: 'How well a score separates two groups: 0.5 is chance, 1 is perfect.',
    spearman: 'A correlation of rank orders: 1 means two lists put things in the same order, 0 means no relation.',
    'induction heads': 'Attention heads that find an earlier copy of the current piece and predict what came after it then. The basis of using context.',
    lora: 'A small add-on fine-tune (low-rank adapter) merged into the model as it runs, changing its habits without retraining all of it.',
    'control vector': 'A direction added to the model’s internal numbers at chosen layers: a push inside parameter space, not an instruction.',
    'untrained twin': 'The same architecture with random, untrained weights. Anything it also shows comes from the structure, not from learning.',
    creativity: 'The sampling temperature. At 0 the likeliest piece is always taken; higher flattens the odds so less likely pieces are drawn more often. It changes which road is taken, not the model.',
    seed: 'The starting number of a random draw. The same seed with the same settings writes the same text again; without a fixed seed a rerun can differ.',
    greedy: 'Always taking the single likeliest next piece, with no random draw: the same input gives the same output every time.',
    estimated: 'Worked out from part of the data, not read off directly, so it can be off. Shown with “at least” or “about”.',
    'off scale': 'Outside what the instrument can read, e.g. a piece the model did not rank in its top 20. Shown as “over N bits”, never as a number, and left out of averages.',
    'local engine': 'llama.cpp running the GGUF file. It writes the answers in Generate (with its roads), Read, Tokens (the pieces) and Compare. Steering and LoRA change this one.',
    'inside engine': 'PyTorch running an unpacked copy of the model layer by layer: Neural net, Tokens (where words live), Anatomy and the experiments. Tipping-point cuts change this one.',
    'changed model': 'The model while steering, a LoRA or a Tipping-point cut is active. Results made then are marked: they are not the unchanged model’s.',
  };
  const ALIAS = { 'induction head': 'induction heads', 'mlp neuron': 'mlp', 'bits/token': 'bits per token', 'bits / token': 'bits per token', 'kv-cache': 'kv cache',
    temperature: 'creativity', 'random seed': 'seed', 'inside-the-model engine': 'inside engine', 'off-scale': 'off scale', 'measured vs estimated': 'estimated' };
  PSI.define = (key) => {
    const k = String(key || '').toLowerCase().trim(), k2 = ALIAS[k] || k;
    if (PSI.TERMS[k2]) return PSI.TERMS[k2];
    const G = window.PSI_GLOSSARY || {}; const g = Object.keys(G).find((x) => x.toLowerCase() === k || x.toLowerCase() === k2);
    return g ? G[g] : '';
  };
  PSI.term = (text, key) => { const s = el('span', 'term', text); s.dataset.term = key || text; s.tabIndex = 0; return s; };
  let tip = null, tipFor = null;
  function showTip(t) {
    const def = PSI.define(t.dataset.term || t.textContent); if (!def) return;
    if (!tip) { tip = el('div', 'term-tip'); tip.id = 'termTip'; tip.setAttribute('role', 'tooltip'); document.body.appendChild(tip); }
    tip.textContent = def; tip.hidden = false; tipFor = t; t.setAttribute('aria-describedby', 'termTip');
    const r = t.getBoundingClientRect(), w = Math.min(320, innerWidth - 24);
    tip.style.maxWidth = `${w}px`;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = `${Math.max(12, Math.min(innerWidth - tw - 12, r.left))}px`;
    tip.style.top = `${r.bottom + 8 + th > innerHeight ? r.top - th - 8 : r.bottom + 8}px`;
  }
  function hideTip() { if (tip) tip.hidden = true; if (tipFor) tipFor.removeAttribute('aria-describedby'); tipFor = null; }
  document.addEventListener('mouseover', (e) => { const t = e.target.closest && e.target.closest('.term'); if (t) showTip(t); else if (tipFor && !tipFor.matches(':focus')) hideTip(); });
  document.addEventListener('focusin', (e) => { const t = e.target.closest && e.target.closest('.term'); if (t) showTip(t); });
  document.addEventListener('focusout', (e) => { if (e.target === tipFor) hideTip(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && tipFor) hideTip(); });
  addEventListener('scroll', () => { if (tipFor) hideTip(); }, true);

  // ── start: the nav's current view, terms focusable, dots, help links ──────
  function init() {
    document.querySelectorAll('.term').forEach((t) => { if (!t.hasAttribute('tabindex')) t.tabIndex = 0; });
    const tabs = $('tabs');
    if (tabs) tabs.addEventListener('click', (e) => { if (e.target.closest('button[data-tab]')) setTimeout(updateHelp, 0); });
    dots(); updateHelp(); watchLesion(); PSI.onState(changeChip); watchOutside();
    if (window.psi) PSI.refreshState().catch(() => { /* benign: refreshState keeps the last known state itself */ });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
