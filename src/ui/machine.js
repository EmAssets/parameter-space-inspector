// This computer, as the app sees it (src/gpu.js): the CPU, the RAM, who holds the graphics card, how much is free,
// and whether anything has spilled into system memory. An instrument should show the conditions it measures under:
// a ladder run that spilled ran ~100x slower and looked, from inside, exactly like a run that was merely slow.
// The header strip is never hidden: a sensor that could not read says so ("GPU unreadable", "no GPU", "not read").
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const gb = (mb) => (mb == null ? 'not read' : `${(mb / 1024).toFixed(1)} GB`);
  const g1 = (mb) => (mb / 1024).toFixed(1);
  const g0 = (mb) => String(Math.round(mb / 1024));
  const pct = (v) => (v == null ? null : `${Math.round(v)}%`);
  const POLL = 20000, POLL_HIDDEN = 120000;
  let last = null, lastAt = 0;

  // ── the header strip: CPU · RAM · GPU, quiet when free, amber when a long job holds the card, red on spill ──
  const strip = $('gpuChip') || (() => {
    const b = el('button', 'gpu-chip ghost'); b.type = 'button'; b.id = 'gpuChip';
    const host = document.querySelector('header.top .model'); if (host) host.insertBefore(b, $('open'));
    return b;
  })();
  strip.hidden = false; strip.classList.add('mstrip'); strip.textContent = '';
  // Three meters: an icon, the value and a thin fill bar. The full name and the details are in each meter's tooltip.
  const ICON = {
    cpu: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    ram: '<rect x="2" y="7" width="20" height="10" rx="1.5"/><path d="M6 10v4M10 10v4M14 10v4M18 10v4M5 17v3M19 17v3"/>',
    gpu: '<rect x="2" y="6" width="17" height="11" rx="1.5"/><circle cx="10.5" cy="11.5" r="3"/><path d="M19 9h3M19 14h3M5 17v3M9 17v3"/>' };
  const meter = (kind, name) => {
    const m = el('span', `ms-m ms-${kind}`); m.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON[kind]}</svg>`;
    const v = el('span', 'ms-v'); const bar = el('span', 'ms-bar'); const fill = el('i'); bar.appendChild(fill); const own = el('span', 'ms-own'); own.hidden = true;
    m.append(own, v, bar); strip.appendChild(m);
    return { m, v, set(text, frac, tip, state, mine) {
      v.textContent = text; own.hidden = mine == null; own.textContent = mine == null ? '' : `PSI ${mine} ·`; bar.hidden = frac == null; fill.style.width = frac == null ? '0' : `${Math.max(2, Math.min(100, frac * 100))}%`;
      m.title = `${name}: ${tip}`; m.classList.remove('busy', 'warn', 'unread'); if (state) m.classList.add(state); } };
  };
  const mCpu = meter('cpu', 'Processor'), mRam = meter('ram', 'Memory (RAM)'), mGpu = meter('gpu', 'Graphics card (GPU)');
  mGpu.set(psi.gpu ? '…' : 'not read', null, psi.gpu ? 'reading this computer' : 'this build cannot read the machine', 'unread');
  strip.addEventListener('click', () => PSI.goto('settings', 'setMachine'));

  // ── Settings: "This computer" ──
  const block = el('div', 'set-block'); block.id = 'setMachine'; block.tabIndex = -1;
  block.append(el('h2', null, 'This computer'),
    el('p', 'muted small', 'The processor, the memory and the graphics card as the app reads them, and the app’s own processes. A model that does not fit beside the others on the card keeps running from ordinary memory, without an error, many times slower; this is where that shows. The strip at the top of the app is the short form; it opens this.'));
  const list = el('div', 'gpu-list small');
  const age = el('span', 'small muted ms-age');
  const row = el('div', 'row'); const again = el('button', 'ghost', 'Check again'); again.type = 'button';
  again.addEventListener('click', () => refresh(0)); row.append(again, age);
  block.append(list, row,
    el('p', 'small muted', 'Windows mis-reports some programs (a screen recorder or the desktop compositor can read as the whole card), so programs other than this app are shown as one total, worked out from what the card says is in use. The app’s own processes are listed one by one.'));
  const after = $('setModel'); if (after) after.after(block);

  const ROLE = { app: 'this app', engine: 'local model (answering engine)', inside: 'inside-the-model engine', research: 'research run', training: 'training (Grow)' };
  function procTable(ps) {
    const t = el('table', 'ms-procs'); const hd = el('tr');
    ['process', 'GPU', 'in system memory', 'RAM', 'CPU'].forEach((h) => { const th = el('th', null, h); th.scope = 'col'; hd.appendChild(th); });
    const th = el('thead'); th.appendChild(hd); t.appendChild(th); const tb = el('tbody'); t.appendChild(tb);
    ps.forEach((h) => {
      const tr = el('tr', h.sharedMB >= 256 ? 'gpu-spill' : null);
      const name = el('td'); name.append(el('span', null, h.label || ROLE[h.role] || h.role || 'process'), el('span', 'muted', ` · pid ${h.pid}`));
      tr.append(name,
        el('td', null, h.mb == null ? 'not read' : h.mb ? gb(h.mb) : 'none'),
        el('td', null, h.sharedMB == null ? 'not read' : h.sharedMB >= 256 ? `${gb(h.sharedMB)} spilled (slow)` : h.sharedMB ? `${h.sharedMB} MB (normal)` : 'none'),
        el('td', null, h.ramMB == null ? 'not read' : gb(h.ramMB)),
        el('td', null, h.cpu_pct == null ? 'not read' : `${h.cpu_pct}%`));
      tb.appendChild(tr);
    });
    const wrap = el('div', 'scroll'); wrap.appendChild(t); return wrap;
  }

  function ago() {
    if (!last) { age.textContent = psi.gpu ? 'not read yet' : 'this build cannot read the machine'; return; }
    const s = Math.max(0, Math.round((Date.now() - lastAt) / 1000));
    const when = s < 60 ? `${s} s` : `${Math.round(s / 60)} min`;
    age.textContent = `read ${when} ago${last.poll_ms != null ? ` (the reading took ${(last.poll_ms / 1000).toFixed(1)} s)` : ''} · read again every ${POLL / 1000} s, every ${POLL_HIDDEN / 60000} min while the window is hidden`;
  }
  setInterval(() => { if (!block.closest('.hidden')) ago(); }, 5000);

  function render(g, failed) {
    last = g; lastAt = Date.now(); list.textContent = '';
    renderIdle(g && g.idle);
    strip.classList.remove('warn', 'busy', 'unread');
    // The machine first: CPU and RAM come from the operating system and stand even when the GPU cannot be read.
    const cpu = (g && g.cpu) || {}, ram = (g && g.ram) || {};
    mCpu.set(cpu.pct == null ? '–' : pct(cpu.pct), cpu.pct == null ? null : cpu.pct / 100, cpu.pct == null ? 'not read' : `${pct(cpu.pct)} busy${cpu.cores ? ` across ${cpu.cores} cores` : ''}`, cpu.pct == null ? 'unread' : null);
    const ramOk = ram.usedMB != null && ram.totalMB != null;
    mRam.set(ramOk ? `${g0(ram.usedMB)}/${g0(ram.totalMB)} GB` : '–', ramOk ? ram.usedMB / ram.totalMB : null, ramOk ? `${gb(ram.usedMB)} of ${gb(ram.totalMB)} in use` : 'not read', ramOk ? null : 'unread');
    if (!g) {
      mGpu.set('–', null, 'could not be read', 'unread'); strip.classList.add('unread');
      list.append(el('p', 'bad', `This computer could not be read${failed ? `: ${failed}` : ''}.`));
      strip.title = 'This computer could not be read. Click for Settings > This computer.';
      strip.setAttribute('aria-label', 'This computer: not read. Open Settings, This computer.');
      ago(); return;
    }
    list.append(el('p', null, `Processor: ${cpu.pct == null ? 'not read' : `${pct(cpu.pct)} busy`}${cpu.cores ? ` across ${cpu.cores} cores` : ''}.`));
    list.append(el('p', null, `Memory (RAM): ${ram.usedMB == null ? 'not read' : `${gb(ram.usedMB)} of ${gb(ram.totalMB)} in use`}.`));
    const procs = (g.procs && g.procs.length ? g.procs : g.holders) || [];
    if (!g.ok) {
      const noGpu = g.reason === 'no_gpu';
      mGpu.set(noGpu ? 'none' : '–', null, noGpu ? 'no GPU found' : `unreadable: the sensor failed${g.detail ? ` (${g.detail})` : ''}`, 'unread'); strip.classList.add('unread');
      list.append(el('p', noGpu ? 'muted' : 'bad', noGpu ? (g.note || g.detail || 'No GPU found. Engines run on the CPU.')
        : `GPU unreadable: the sensor failed${g.detail ? ` (${g.detail})` : ''}. The card may be there; how full it is and whether anything spilled are not known.`));
      if (procs.length) { list.append(el('p', 'k ms-sub', 'This app’s processes')); list.append(procTable(procs)); }
    } else {
      const gpus = g.gpus && g.gpus.length ? g.gpus : [{ name: g.gpu, totalMB: g.totalMB, usedMB: g.usedMB, freeMB: g.freeMB, util_pct: g.gpu_util_pct }];
      gpus.forEach((c, i) => list.append(el('p', null, `${gpus.length > 1 ? `GPU ${c.index ?? i}: ` : 'Graphics card: '}${c.name || 'name not read'}: ${gb(c.usedMB)} of ${gb(c.totalMB)} in use, ${gb(c.freeMB)} free${c.util_pct == null ? ', how busy not read' : `, ${pct(c.util_pct)} busy`}.`)));
      if (gpus.length > 1) list.append(el('p', 'muted', `The strip at the top shows GPU ${gpus[0].index ?? 0} (${gpus[0].name || 'name not read'}).`));
      list.append(el('p', 'k ms-sub', 'This app’s processes'));
      if (procs.length) list.append(procTable(procs)); else list.append(el('p', 'muted', 'None of the app’s processes holds the GPU now.'));
      if (g.otherMB != null) list.append(el('p', 'muted', `Other programs (screen, browser, recorder…): ${gb(g.otherMB)} of GPU memory together.`));
      if (g.reader) list.append(el('p', 'muted', `Read with ${g.reader}${g.vendor ? ` (${({ nvidia: 'NVIDIA', amd: 'AMD', apple: 'Apple', intel: 'Intel' })[g.vendor] || g.vendor})` : ''}.`));
      if (g.unified) list.append(el('p', 'muted', 'Unified memory: the GPU and the processor share the RAM, so the figures above are the RAM’s, and a model cannot spill out of the GPU; when the RAM is full, macOS swaps to disk instead.'));
      if (g.spill_detect === 'unavailable') list.append(el('p', 'bad', 'Spill detection unavailable: this system gives no per-process shared-memory counters, so a model running partly from system memory cannot be told from a slow one here.'));
      const spill = g.spill || [], busy = g.busy || [];
      if (spill.length) list.append(el('p', 'bad', `Overfull: ${spill.map((h) => `${h.label} has ${gb(h.sharedMB)} in system memory`).join('; ')}. It keeps running, many times slower. Stop or pause another GPU job, or load a smaller model.`));
      const memOk = g.usedMB != null && g.totalMB != null, frac = memOk ? g.usedMB / g.totalMB : null;
      const used = memOk ? `${g1(g.usedMB)}/${g1(g.totalMB)} GB` : '–';
      const util = g.gpu_util_pct == null ? 'how busy not read' : `${pct(g.gpu_util_pct)} busy`;
      const held = (g.holders || []).filter((h) => h.mb > 0).map((h) => `\n  ${h.label}: ${gb(h.mb)}${h.sharedMB >= 256 ? ` + ${gb(h.sharedMB)} spilled` : ''}`).join('');
      // The bar is the whole card (every program on it); "PSI n" beside it is this app's own share, so a sleeping model is
      // not read as holding what the screen, a browser or a video editor holds.
      const mineMB = (g.holders || []).filter((h) => ['app', 'engine', 'inside', 'training'].includes(h.role)).reduce((a, h) => a + (h.mb || 0), 0);
      const mine = mineMB >= 50 ? `${g1(mineMB)} GB` : '0', mineTip = `This app: ${mine === '0' ? 'nothing on the card' : mine} · the bar is the whole card, every program on it`;
      const base = `${g.gpu || 'card'}: ${memOk ? `${gb(g.usedMB)} of ${gb(g.totalMB)} in use, ${gb(g.freeMB)} free` : 'memory not read'}, ${util}${held}${g.otherMB != null ? `\n  other programs: ${gb(g.otherMB)}` : ''}`;
      if (spill.length) { strip.classList.add('warn'); mGpu.set(`${used} ⚠`, frac, `${mineTip}\nOVERFULL: ${spill.map((h) => `${h.label} spilled ${gb(h.sharedMB)} into system memory, many times slower`).join('; ')}\n${base}`, 'warn', mine); }
      else if (busy.length) { strip.classList.add('busy'); mGpu.set(used, frac, `${mineTip}\nbusy with ${busy[0].label}\n${base}`, 'busy', mine); }
      else mGpu.set(used, frac, `${mineTip}\n${base}`, null, mine);
    }
    strip.removeAttribute('title');                                  // each meter carries its own tooltip
    strip.setAttribute('aria-label', `This computer: ${[mCpu.m.title, mRam.m.title, mGpu.m.title.split('\n')[0]].join('; ')}. Open Settings, This computer.`);
    ago();
  }

  let timer = 0;
  async function refresh(delay = POLL) {
    clearTimeout(timer);
    if (!psi.gpu) { ago(); return null; }
    try { render(await psi.gpu()); } catch (e) { render(null, (e && e.message) || String(e)); }
    if (delay) timer = setTimeout(() => refresh(delay), document.hidden ? POLL_HIDDEN : delay);
    return last;
  }
  if (psi.gpu) { refresh(); document.addEventListener('psi-model', () => setTimeout(() => refresh(0), 3000)); } else ago();

  // Before the app starts an engine by itself (the model reopened at launch): would it crowd out a long GPU job, or
  // not fit? Returns null when it is fine, or a sentence saying what holds the card.
  async function holdBack(file) {
    if (!psi.gpu) return null;
    const g = await refresh(0); if (!g || !g.ok) return null;
    if ((g.busy || []).length) return `${g.busy[0].label} is using the GPU (${gb(g.busy[0].mb)})`;
    let need = 0; try { need = (await psi.readHead(file, 16)).size / 2 ** 20 * 1.05 + 1200; } catch { return null; }
    if (g.freeMB != null && need > g.freeMB) return `it needs about ${gb(need)} of GPU memory and ${gb(g.freeMB)} is free`;
    return null;
  }
  // After a load: did anything spill?
  async function spilled() { const g = await refresh(0); return g && g.ok && (g.spill || []).length ? g.spill : null; }

  // ── Sleep when idle (src/idle.js): after N minutes without use the engines stop and the GPU memory goes back;
  // the next thing run wakes the same model. Said in the header beside the model's name, and set here. ──
  const clock = (t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const mins = (ms) => { const m = Math.max(0, Math.round(ms / 60000)); return m < 1 ? 'under a minute' : m < 90 ? `${m} min` : `${(m / 60).toFixed(1)} h`; };
  const tag = el('span', 'sleep-tag'); tag.id = 'sleepTag'; tag.hidden = true; tag.setAttribute('role', 'status');
  const mn = $('modelName'); if (mn) mn.after(tag);
  const sb = el('div', 'sleep-set');
  const sel = el('select'); sel.id = 'idleMin'; sel.setAttribute('aria-label', 'Free the GPU after this long without use');
  [[15, '15 minutes'], [30, '30 minutes'], [60, '1 hour'], [120, '2 hours'], [240, '4 hours'], [0, 'never']].forEach(([v, t]) => { const o = el('option', null, t); o.value = String(v); sel.appendChild(o); });
  const sleepNow = el('button', 'ghost', 'Sleep now'); sleepNow.type = 'button';
  const idleLine = el('p', 'small muted');
  const lab = el('label', 'row'); lab.append(el('span', null, 'Free the GPU after'), sel, el('span', null, 'without use'), sleepNow);
  sb.append(el('h3', 'k ms-sub', 'Sleep when idle'), el('p', 'small muted', 'A loaded model holds GPU memory and RAM whether or not anyone uses it. After this long with nothing asked of it, the app stops its engines and gives the memory back; the next thing you run reloads the same model (with its steering, LoRA and context) and takes a few seconds longer. Unloading by hand never reloads. The app’s own checks, like this page and the strip at the top, don’t count as use.'), lab, idleLine);
  block.insertBefore(sb, list.nextSibling);
  if (!psi.idle) { sb.hidden = true; }
  sel.addEventListener('change', async () => { try { renderIdle(await psi.idle({ action: 'set', minutes: +sel.value })); } catch (e) { PSI.err(e); } });
  sleepNow.addEventListener('click', async () => { sleepNow.disabled = true; try { renderIdle(await psi.idle({ action: 'sleep' })); refresh(0); } catch (e) { PSI.err(e); } finally { sleepNow.disabled = false; } });
  let idle = null;
  function renderIdle(st) {
    if (st) idle = st; st = idle;
    if (!st) { tag.hidden = true; return; }
    if (document.activeElement !== sel) sel.value = String(st.minutes);
    const a = st.asleep, gave = a && a.freedMB != null ? `${gb(a.freedMB)} of GPU memory` : 'its GPU memory';
    const name = a && a.model ? String(a.model).split(/[\/]/).pop() : null;
    if (st.waking) { tag.hidden = false; tag.className = 'sleep-tag waking'; tag.textContent = 'waking…'; tag.title = `Reloading ${name || 'the model'} for the call that needs it.`; }
    else if (a && !st.running) {
      tag.hidden = !!(mn && /^asleep/.test(mn.textContent));   // the page opened while asleep: the model line already says so, with Wake tag.className = 'sleep-tag'; tag.textContent = 'asleep';
      tag.title = `Asleep since ${clock(a.at)} (${a.why}): gave back ${gave}.${name ? ` The next thing you run reloads ${name}; it takes a few seconds.` : ''}`;
    } else tag.hidden = true;
    sleepNow.disabled = !st.running;
    idleLine.textContent = st.waking ? 'Waking: reloading the model.'
      : a && !st.running ? `Asleep since ${clock(a.at)} (${a.why}); gave back ${gave}.${name ? ` Wakes ${name} on the next use.` : ''}`
      : !st.running ? 'No engine is running, so nothing holds the GPU for this app.'
      : `Last use ${mins(Date.now() - st.last_use_at)} ago${st.in_flight ? ` · ${st.in_flight} call${st.in_flight > 1 ? 's' : ''} running (held awake)` : ''}${st.sleeps_at ? ` · sleeps at ${clock(st.sleeps_at)} if unused` : ' · never sleeps'}.`;
  }

  window.PSIMachine = { refresh, holdBack, spilled, get last() { return last; } };
})();
