// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// Prometheus in motion: when the guide acts for the user, its "hand" is visible.
//   - every action is a trace card in the chat;
//   - a point of light leaves that card and travels, on a damped spring along a curve, to the real
//     control it is about to use (a menu item, a field, a button);
//   - it acts there the way a person would: the view changes with a cross-fade, text is typed,
//     a button is pressed down and released;
//   - while the work runs, a ring traces the button; when it ends, the part of the page that
//     actually changed is found (by watching the DOM) and tied back to the card with a line;
//   - hovering a card later redraws the line to what it touched and what it produced; clicking
//     takes you there.
// All motion is physical (springs with real stiffness and damping), short, and skipped when the
// system asks for reduced motion.
(() => {
  'use strict';
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const reduced = () => RM.matches;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let SPEED = 1;   // 1 normal; larger is slower (the experiment tour's “Slower” button): travel and typing stretch by it
  const NS = 'http://www.w3.org/2000/svg';

  // ── springs ─────────────────────────────────────────────────────────────
  // A damped harmonic oscillator x'' = -k(x-1) - c x', sampled into a CSS linear() easing.
  const SPRINGS = {};
  function spring(stiffness = 170, damping = 22, mass = 1) {
    const key = `${stiffness}/${damping}/${mass}`; if (SPRINGS[key]) return SPRINGS[key];
    let x = 0, v = 0; const dt = 1 / 240, pts = []; let t = 0, still = 0, t99 = 0;
    while (t < 3) { const a = (-stiffness * (x - 1) - damping * v) / mass; v += a * dt; x += v * dt; t += dt; pts.push(x); if (!t99 && x > 0.99) t99 = t; if (Math.abs(x - 1) < 0.001 && Math.abs(v) < 0.01) { if (++still > 24) break; } else still = 0; }
    const n = 64, step = Math.max(1, Math.floor(pts.length / n)), sample = [0];
    for (let i = step; i < pts.length; i += step) sample.push(+pts[i].toFixed(4)); sample.push(1);
    let easing = 'cubic-bezier(.2,.9,.25,1)'; try { const e = `linear(${sample.join(', ')})`; if (CSS.supports('animation-timing-function', e)) easing = e; } catch (_) { /* old engine */ }
    return (SPRINGS[key] = { easing, ms: Math.round(t * 1000), arrive: Math.round((t99 || t) * 1000), at: (sec) => pts[Math.min(pts.length - 1, Math.floor(sec / dt))], fn: (p) => pts[Math.min(pts.length - 1, Math.floor(p * (pts.length - 1)))] });
  }
  const SNAPPY = () => spring(260, 26), GENTLE = () => spring(90, 14), BOUNCY = () => spring(320, 18);   // GENTLE: ~0.35 s to arrive, ~3% overshoot

  // ── the overlay: one SVG for lines, one layer for the point and rings ──
  const layer = document.createElement('div'); layer.className = 'pm-layer'; layer.setAttribute('aria-hidden', 'true');
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('class', 'pm-svg'); layer.appendChild(svg);
  const defs = document.createElementNS(NS, 'defs');
  defs.innerHTML = '<linearGradient id="pmGrad" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#ffc36b" stop-opacity=".15"/><stop offset="1" stop-color="#ffc36b" stop-opacity=".95"/></linearGradient><filter id="pmGlow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>';
  svg.appendChild(defs);
  document.body.appendChild(layer);
  const dot = document.createElement('div'); dot.className = 'pm-dot'; layer.appendChild(dot);

  const center = (r) => ({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
  const rectOf = (n) => n.getBoundingClientRect();
  const onScreen = (r) => r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth && r.width > 0;
  // the point leaves the card from its left edge (the panel is on the right) and arrives at the
  // side of the target nearest to where it came from
  const fromCard = (card) => { const r = rectOf(card.querySelector('.pm-ic') || card); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; };
  function curve(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, bend = Math.min(140, d * 0.28);
    const c = { x: (a.x + b.x) / 2 - (dy / d) * bend, y: (a.y + b.y) / 2 + (dx / d) * bend * 0.6 - bend * 0.35 };
    return { c, at: (t) => ({ x: (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * c.x + t * t * b.x, y: (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * c.y + t * t * b.y }), d: `M${a.x},${a.y} Q${c.x},${c.y} ${b.x},${b.y}` };
  }
  // The part of an element actually visible: clipped by every ancestor that scrolls or hides overflow, and by the viewport.
  function visRect(n) {
    if (!n || !n.isConnected) return null; let r = n.getBoundingClientRect(); if (!r.width && !r.height) return null;
    let L = r.left, T = r.top, R = r.right, B = r.bottom;
    for (let p = n.parentElement; p && p !== document.body; p = p.parentElement) {
      const cs = getComputedStyle(p); if (!/(auto|scroll|hidden|clip)/.test(cs.overflow + cs.overflowX + cs.overflowY)) continue;
      const q = p.getBoundingClientRect(); L = Math.max(L, q.left); T = Math.max(T, q.top); R = Math.min(R, q.right); B = Math.min(B, q.bottom);
    }
    L = Math.max(L, 0); T = Math.max(T, 0); R = Math.min(R, innerWidth); B = Math.min(B, innerHeight);
    return R - L > 2 && B - T > 2 ? { left: L, top: T, right: R, bottom: B, width: R - L, height: B - T } : null;
  }
  // Keep an overlay box on its element every frame (the page may scroll, re-flow or re-lay out after it appears).
  function track(g, n, pad) {
    const tick = () => {
      if (!g.isConnected) return; const r = visRect(n);
      if (!r) g.style.visibility = 'hidden';
      else Object.assign(g.style, { visibility: '', left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` });
      requestAnimationFrame(tick);
    };
    tick();
  }
  function path(dAttr, cls) { const p = document.createElementNS(NS, 'path'); p.setAttribute('d', dAttr); p.setAttribute('class', cls); svg.appendChild(p); return p; }

  async function scrollInto(n) {
    if (!n) return; const r = rectOf(n); if (r.top >= 70 && r.bottom <= innerHeight - 40) return;
    n.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'center' });
    // Finished when the element has stopped moving for a few frames: works for the window and for any scrolling panel.
    await new Promise((res) => { let last = null, still = 0; const t0 = performance.now();
      const tick = () => { const q = rectOf(n); const k = `${Math.round(q.left)},${Math.round(q.top)}`; still = k === last ? still + 1 : 0; last = k;
        if (still >= 4 || performance.now() - t0 > 1500) res(); else requestAnimationFrame(tick); };
      requestAnimationFrame(tick); setTimeout(res, 1600); });   // frames pause in a background tab: never wait forever
  }

  // The point travels from the card to the target along a spring-timed curve, drawing its trail.
  async function travel(card, target) {
    if (!card || !target) return; await scrollInto(target);
    const a0 = fromCard(card), r0 = visRect(target); if (!r0) return;
    let b = center(r0), cv = curve(a0, b);
    if (reduced()) { pulse(target); return; }
    // the move lasts until the point has arrived and come back from its small overshoot; the invisible tail is cut
    const sp = GENTLE(), far = Math.hypot(b.x - a0.x, b.y - a0.y), dur = Math.round(Math.max(sp.arrive * 1.6, 420) * Math.min(1.35, Math.max(0.85, far / 700)) * SPEED);
    const trail = path(cv.d, 'pm-trail'); const len = trail.getTotalLength(); trail.style.strokeDasharray = `${len}`; trail.style.strokeDashoffset = `${len}`;
    dot.classList.add('on'); const t0 = performance.now();
    await new Promise((res) => {
      const tick = (now) => {
        const rr = visRect(target); if (rr) { b = center(rr); cv = curve(a0, b); trail.setAttribute('d', cv.d); }   // follow the target if the page moves
        const p = Math.min(1, (now - t0) / dur), e = p >= 1 ? 1 : sp.at((p * dur) / 1000), q = cv.at(Math.min(1.03, e));
        const ahead = cv.at(Math.min(1, e + 0.02)), ang = Math.atan2(ahead.y - q.y, ahead.x - q.x), speed = Math.min(1, Math.hypot(ahead.x - q.x, ahead.y - q.y) / 9);
        dot.style.transform = `translate(${q.x}px, ${q.y}px) rotate(${ang}rad) scale(${1 + speed * 0.35}, ${1 - speed * 0.18})`;   // stretches along its motion, like a drop
        { const L2 = trail.getTotalLength(); trail.style.strokeDasharray = `${L2}`; trail.style.strokeDashoffset = `${L2 * (1 - Math.min(1, e))}`; }
        if (p < 1) requestAnimationFrame(tick); else res();
      };
      requestAnimationFrame(tick);
    });
    dot.animate([{ transform: dot.style.transform }, { transform: `translate(${b.x}px, ${b.y}px) scale(2.4)`, opacity: 0 }], { duration: 260, easing: 'ease-out' }).finished.then(() => { dot.classList.remove('on'); dot.style.transform = ''; });
    trail.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 700, delay: 120, easing: 'ease-out' }).finished.then(() => trail.remove());
    pulse(target);
  }

  // A ring that settles onto the element (it lands, then breathes out).
  function ring(target, cls = 'pm-ring', hold = 0) {
    const g = document.createElement('div'); g.className = cls; layer.appendChild(g); const pad = 6;
    const br = parseFloat(getComputedStyle(target).borderTopLeftRadius) || 0; g.style.borderRadius = `${Math.min(18, br + pad)}px`;   // concentric with the element's own corners
    track(g, target, pad);
    if (reduced()) { setTimeout(() => g.remove(), 900 + hold); return g; }
    const sp = BOUNCY(); g.animate([{ transform: 'scale(1.18)', opacity: 0 }, { transform: 'scale(1)', opacity: 1 }], { duration: sp.ms, easing: sp.easing, fill: 'forwards' });
    setTimeout(() => g.animate([{ opacity: 1 }, { opacity: 0, transform: 'scale(1.04)' }], { duration: 520, easing: 'ease-in', fill: 'forwards' }).finished.then(() => g.remove()), 700 + hold);
    return g;
  }
  const pulse = (n) => n && ring(n);

  // ── acting ──────────────────────────────────────────────────────────────
  async function typeInto(n, value) {
    const v = String(value);
    if (n.tagName === 'SELECT' || n.type === 'checkbox' || n.type === 'range' || reduced()) {
      if (n.type === 'checkbox') n.checked = value === true || value === 'true'; else n.value = v;
      if (!reduced()) n.animate([{ transform: 'scale(1)' }, { transform: 'scale(1.06)' }, { transform: 'scale(1)' }], { duration: SNAPPY().ms, easing: SNAPPY().easing });
      return;
    }
    n.focus({ preventScroll: true }); n.classList.add('pm-typing');
    const start = n.value; const per = Math.max(8, Math.min(28, 700 / Math.max(1, v.length))) * SPEED;
    if (start && start !== v) { for (let i = start.length; i >= 0; i -= Math.max(1, Math.ceil(start.length / 14))) { n.value = start.slice(0, i); await sleep(12); } }   // a quick select-and-delete
    n.value = ''; for (let i = 1; i <= v.length; i += v.length > 120 ? 3 : 1) { n.value = v.slice(0, i); await sleep(per); } n.value = v;
    n.classList.remove('pm-typing');
  }
  async function pressDown(n) {
    if (reduced()) return; const r = rectOf(n), c = center(r);
    const sp = SNAPPY(); await n.animate([{ transform: 'scale(1)' }, { transform: 'scale(.94)' }], { duration: 110, easing: 'ease-out', fill: 'forwards' }).finished;
    const rip = document.createElement('div'); rip.className = 'pm-ripple'; layer.appendChild(rip); const s = Math.max(r.width, r.height) * 1.6;
    Object.assign(rip.style, { left: `${c.x - s / 2}px`, top: `${c.y - s / 2}px`, width: `${s}px`, height: `${s}px` });
    rip.animate([{ transform: 'scale(.2)', opacity: 0.55 }, { transform: 'scale(1)', opacity: 0 }], { duration: 560, easing: 'cubic-bezier(.2,.8,.2,1)' }).finished.then(() => rip.remove());
    n.animate([{ transform: 'scale(.94)' }, { transform: 'scale(1)' }], { duration: sp.ms, easing: sp.easing, fill: 'forwards' }).finished.then((a) => a.cancel && a.cancel());
  }

  // While work runs: a light runs round the pressed control; it follows the element if the page moves.
  function busy(n) {
    const g = document.createElement('div'); g.className = 'pm-busy'; layer.appendChild(g); track(g, n, 5);
    return () => g.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 260 }).finished.then(() => g.remove());
  }

  // What changed on the page because of an action: watch the view, then pick the most meaningful
  // changed block (a chart, a table, a result paragraph), not the button itself or a status line.
  function watch(scope, except) {
    const hits = new Set(); const mo = new MutationObserver((ms) => ms.forEach((m) => { let n = m.target.nodeType === 1 ? m.target : m.target.parentElement; if (n) hits.add(n); m.addedNodes && m.addedNodes.forEach((a) => a.nodeType === 1 && hits.add(a)); }));
    mo.observe(scope, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['width', 'height', 'class', 'hidden', 'style'] });
    return () => {
      mo.disconnect();
      const ok = [...hits].map((n) => n.closest('canvas, table, .scroll, [id]:not(button):not(input):not(textarea):not(select):not(label), p, div') || n)
        .filter((n) => n && scope.contains(n) && n !== scope && !(except && (n === except || n.contains(except) || except.contains(n))) && !n.closest('#hlp') && n.offsetParent !== null && !/^(BUTTON|INPUT|TEXTAREA|SELECT|LABEL|OPTION)$/.test(n.tagName))
        .filter((n) => { const r = rectOf(n); return r.width > 60 && r.height > 18; });
      const score = (n) => (n.tagName === 'CANVAS' ? 4 : 0) + (n.tagName === 'TABLE' || n.querySelector && n.querySelector('table,canvas') ? 3 : 0) + Math.min(3, (n.textContent || '').length / 200) + (n.id ? 1 : 0) - (/stat|status|k$/i.test(n.id || '') ? 3 : 0);
      const uniq = [...new Set(ok)].filter((n, i, a) => !a.some((m) => m !== n && m.contains(n) && score(m) >= score(n)));
      // Results appear after the control that produced them: prefer what follows it in the page, and what is near it.
      // (Other drawings may repaint at the same time, e.g. on a resize; those sit elsewhere.)
      const ref = except ? rectOf(except) : null;
      const after = (n) => !except || !!(except.compareDocumentPosition(n) & Node.DOCUMENT_POSITION_FOLLOWING);
      const near = (n) => { if (!ref) return 0; const r = rectOf(n) , dy = r.top >= ref.bottom ? r.top - ref.bottom : ref.top - r.bottom; return Math.max(0, dy) / 300; };
      const total = (n) => score(n) + (after(n) ? 4 : 0) - near(n);
      uniq.sort((x, y) => total(y) - total(x));
      return uniq[0] || null;
    };
  }

  // A soft glow that settles onto the result.
  async function reveal(card, n) {
    if (!n) return; await scrollInto(n); ring(n, 'pm-result', 900); link(card, n, 1600);
  }

  // A line between a card and something on the page (drawn on, then fades).
  function link(card, n, hold = 1200) {
    if (!card || !n || !card.isConnected || !n.isConnected || !visRect(n)) return;
    const p = path('', 'pm-link'); const sp = SNAPPY(); const t0 = performance.now(), total = 420 + hold + 400;
    const ends = () => {
      const ic = card.querySelector('.pm-ic') || card, cr = visRect(ic), r = visRect(n); if (!cr || !r) return null;
      const a = { x: cr.left + cr.width / 2, y: cr.top + cr.height / 2 };
      const b = { x: r.right < a.x ? r.right : Math.max(r.left + 8, Math.min(r.right - 8, a.x)), y: Math.max(r.top + 8, Math.min(r.bottom - 8, a.y)) };
      return curve(a, b);
    };
    const tick = (now) => {
      const t = now - t0; if (t > total || !p.isConnected) { p.remove(); return; }
      const cv = ends(); if (!cv) { p.style.visibility = 'hidden'; requestAnimationFrame(tick); return; }
      p.style.visibility = ''; p.setAttribute('d', cv.d); const len = p.getTotalLength();
      const draw = reduced() ? 1 : Math.min(1, sp.at(Math.min(t, 420) / 1000)); p.style.strokeDasharray = `${len}`; p.style.strokeDashoffset = `${len * (1 - draw)}`;
      p.style.opacity = t > 420 + hold ? String(Math.max(0, 1 - (t - 420 - hold) / 400)) : '1';
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ── the trace card in the chat ──────────────────────────────────────────
  const ICON = { open_view: '◧', show_user: '◎', set_control: '⌨', press: '●', current_view: '◌', list_controls: '☰' };
  function card(log, name, label, verbDone) {
    const c = document.createElement('div'); c.className = 'pm-card run'; c.tabIndex = 0;
    c.innerHTML = '<span class="pm-ic"></span><span class="pm-tx"></span><span class="pm-res"></span>';
    c.querySelector('.pm-ic').textContent = ICON[name] || '◆'; c.querySelector('.pm-tx').textContent = label;
    log.appendChild(c); log.scrollTop = log.scrollHeight;
    if (!reduced()) { const sp = BOUNCY(); c.animate([{ opacity: 0, transform: 'translateY(8px) scale(.97)' }, { opacity: 1, transform: 'none' }], { duration: sp.ms, easing: sp.easing }); }
    c.pm = { target: null, result: null, view: null };
    const back = () => { if (c.pm.target && c.pm.target.isConnected) link(c, c.pm.target, 1400); if (c.pm.result && c.pm.result.isConnected) link(c, c.pm.result, 1400); };
    c.addEventListener('mouseenter', back); c.addEventListener('focus', back);
    c.addEventListener('click', async () => {
      if (c.pm.view) { const b = document.querySelector(`#tabs button[data-tab="${c.pm.view}"]`); if (b && b.getAttribute('aria-current') !== 'page') { b.click(); await sleep(250); } }
      const n = (c.pm.result && c.pm.result.isConnected && c.pm.result) || (c.pm.target && c.pm.target.isConnected && c.pm.target); if (n) { await scrollInto(n); ring(n, c.pm.result ? 'pm-result' : 'pm-ring'); back(); }
    });
    c.finish = (ok, text, resultEl) => {
      c.classList.remove('run'); c.classList.add(ok ? 'ok' : 'bad'); if (verbDone && ok) c.querySelector('.pm-tx').textContent = verbDone;
      const res = c.querySelector('.pm-res'); if (text) { res.textContent = text; } if (resultEl) { c.pm.result = resultEl; res.classList.add('has'); }
      if (!reduced()) { const ic = c.querySelector('.pm-ic'); ic.textContent = ok ? '✓' : '!'; const sp = BOUNCY(); ic.animate([{ transform: 'scale(.4)' }, { transform: 'scale(1)' }], { duration: sp.ms, easing: sp.easing }); }
      else c.querySelector('.pm-ic').textContent = ok ? '✓' : '!';
    };
    return c;
  }

  // View change with a cross-fade (View Transitions where the engine has them).
  async function switchView(btn) {
    if (!btn) return; if (reduced() || !document.startViewTransition) { btn.click(); return; }
    const t = document.startViewTransition(() => btn.click()); try { await t.finished; } catch (_) { /* interrupted */ }
  }

  window.PMotion = { card, travel, typeInto, pressDown, busy, watch, reveal, link, ring, switchView, scrollInto, spring, reduced, setSpeed: (f) => { SPEED = Math.max(0.5, Math.min(4, +f || 1)); }, speed: () => SPEED };
})();
