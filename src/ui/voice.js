// SPDX-License-Identifier: Apache-2.0
// Voice input on the boxes that take prose: Generate's prompt, every experiment's text, and the Prometheus box (app and
// Help). A mic button in the box's top-right corner; press it, speak, press it again (or pause) to stop. The words go in
// at the cursor; a box still holding its sample text is replaced. Nothing is sent: you read it, then press Send/Generate.
// It uses the browser's own speech recognition (Web Speech API), so the audio goes to that browser's speech service
// (Chrome: Google, Safari: Apple, Edge: Microsoft); this app never sees or stores it. The note under the box says so the
// first time. Electron (the desktop app) has the API but no service behind it; there the button points to Windows'
// own dictation (Win+H), which types into any box.
(() => {
  'use strict';
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const ELECTRON = /Electron\//.test(navigator.userAgent);
  if (!SR && !ELECTRON) return;                        // no speech recognition here (Firefox, or http on a LAN address): no button
  const SKIP = new Set(['lensText', 'emText']);       // the lens editor and the paste-a-long-text box are not for speaking
  const WHO = /Edg\//.test(navigator.userAgent) ? 'Microsoft' : /Chrome\//.test(navigator.userAgent) ? 'Google'
    : /Safari\//.test(navigator.userAgent) ? 'Apple' : 'your browser’s speech service';
  const WHERE = ELECTRON ? '' : `Speech is turned into text by ${WHO === 'your browser’s speech service' ? WHO : `${WHO}’s speech service`}; this app never sees or stores the audio.`;

  const css = document.createElement('style');
  css.textContent = `.vcw{position:relative;display:block;flex:1;width:100%;min-width:0}
textarea.vc-has{padding-right:42px!important}
.vc-mic{position:absolute;top:5px;right:5px;z-index:2;width:32px;height:32px;min-height:32px!important;min-width:32px;padding:0!important;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;background:transparent!important;color:var(--ink3,#8a877f)!important;border:1px solid transparent!important;cursor:pointer}
.vc-mic:hover{color:var(--ink,#EDEBE4)!important;border-color:var(--line,rgba(237,235,228,.16))!important}
.vc-mic svg{width:17px;height:17px;fill:none;stroke:currentColor;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.vc-mic.on{color:#ff5c5c!important;border-color:#ff5c5c!important;animation:vcPulse 1.4s ease-in-out infinite}
@keyframes vcPulse{0%,100%{box-shadow:0 0 0 0 rgba(255,92,92,.45)}50%{box-shadow:0 0 0 6px rgba(255,92,92,0)}}
@media (prefers-reduced-motion:reduce){.vc-mic.on{animation:none}}
.vc-note{margin:4px 0 0;font:12px/1.4 var(--mono,monospace);color:var(--ink2,#b9b6ad)}
.vc-note.bad{color:#ff7a7a}
.vc-note[hidden]{display:none}`;
  document.head.appendChild(css);

  const MIC = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7"/></svg>';
  const ERR = {
    'not-allowed': 'The microphone is blocked for this page. Allow it in the browser’s site settings (the icon left of the address), then try again.',
    'service-not-allowed': 'This browser won’t run speech recognition here. On iPhone, turn on Siri & Dictation in Settings; elsewhere try Chrome or Safari.',
    'audio-capture': 'No microphone was found.',
    'network': ELECTRON ? 'The desktop app has no speech service. Use Windows dictation instead: click in the box and press Win+H.'
      : 'The speech service couldn’t be reached (it needs the internet).',
    'no-speech': 'Heard nothing. Press the mic and speak.',
    'language-not-supported': `Speech recognition doesn’t support ${navigator.language} here.`,
  };
  let active = null;   // { ta, rec, btn, note } of the box listening now (one at a time)
  let told = false;    // the where-does-the-audio-go line is shown once per page

  function attach(ta) {
    if (!ta || ta.dataset.voice || ta.readOnly || SKIP.has(ta.id)) return;
    ta.dataset.voice = '1'; ta.classList.add('vc-has');
    const wrap = document.createElement('span'); wrap.className = 'vcw';
    ta.parentNode.insertBefore(wrap, ta); wrap.appendChild(ta);
    const btn = document.createElement('button'); btn.type = 'button'; btn.className = 'vc-mic'; btn.innerHTML = MIC;
    btn.setAttribute('aria-label', 'Speak instead of typing');
    btn.title = ELECTRON ? 'Speak instead of typing. In the desktop app: click in the box and press Win+H (Windows dictation).'
      : `Speak instead of typing (press again to stop). ${WHERE}`;
    wrap.appendChild(btn);
    const note = document.createElement('p'); note.className = 'vc-note'; note.hidden = true; note.setAttribute('role', 'status');
    wrap.after(note);
    btn.addEventListener('click', () => (active && active.ta === ta ? stop() : start(ta, btn, note)));
  }

  function say(note, text, bad) { note.textContent = text; note.classList.toggle('bad', !!bad); note.hidden = !text; }

  function start(ta, btn, note) {
    if (active) stop();
    if (!SR || ELECTRON) { say(note, ERR.network, true); return; }
    const untouched = ta.value === ta.defaultValue && ta.value !== '';
    const s = untouched ? 0 : ta.selectionStart, e = untouched ? ta.value.length : ta.selectionEnd;
    let before = ta.value.slice(0, s); const after = ta.value.slice(e);
    if (before && !/\s$/.test(before)) before += ' ';
    const rec = new SR(); rec.lang = navigator.language || 'en-US'; rec.continuous = true; rec.interimResults = true;
    active = { ta, rec, btn, note };
    rec.onresult = (ev) => {
      let text = '';
      for (let i = 0; i < ev.results.length; i++) text += ev.results[i][0].transcript;
      text = text.replace(/^\s+/, '');
      ta.value = before + text + after; const at = (before + text).length;
      try { ta.setSelectionRange(at, at); } catch { /* benign: a box that isn't focused keeps its caret */ }
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    };
    rec.onerror = (ev) => { if (ev.error !== 'aborted') say(note, ERR[ev.error] || `Voice input stopped: ${ev.error}.`, ev.error !== 'no-speech'); };
    rec.onend = () => {
      btn.classList.remove('on'); btn.setAttribute('aria-pressed', 'false');
      if (active && active.rec === rec) active = null;
      if (!note.classList.contains('bad') && note.textContent.startsWith('Listening')) say(note, '');
    };
    try { rec.start(); } catch (err) { active = null; say(note, `Voice input couldn’t start: ${err.message}`, true); return; }
    btn.classList.add('on'); btn.setAttribute('aria-pressed', 'true');
    say(note, `Listening… press the mic again to stop.${told ? '' : ` ${WHERE}`}`); told = true;
  }
  function stop() { const a = active; active = null; if (a) { try { a.rec.stop(); } catch { /* already ended */ } } }

  document.querySelectorAll('textarea').forEach(attach);
  // The Prometheus box is built by its script; on a page where it appears later, pick it up when it does.
  if (!document.getElementById('hlpIn') || !document.getElementById('hlpIn').dataset.voice) {
    const mo = new MutationObserver(() => { const h = document.getElementById('hlpIn'); if (h) { attach(h); mo.disconnect(); } });
    mo.observe(document.body, { childList: true, subtree: true });
  }
  // Sending (or switching view) ends listening, so a reply doesn't get dictated into.
  document.addEventListener('click', (e) => { if (active && e.target.closest && e.target.closest('#hlpSend, button[id$="Go"], #genMore')) stop(); }, true);
  document.addEventListener('keydown', (e) => { if (active && (e.key === 'Escape' || (e.key === 'Enter' && !e.shiftKey && e.target.id === 'hlpIn'))) stop(); }, true);
  window.addEventListener('hashchange', stop);
})();
