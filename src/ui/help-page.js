// SPDX-License-Identifier: Apache-2.0
// The manual (help.html) page's own script, kept in a file so the page can forbid inline script (its Content-Security-Policy).
(() => {
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const slug = (s) => 'term-' + s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  // Terms the manual needs that the shared glossary (guide-data.js) may not have yet. The shared definition wins when both exist.
  const EXTRA = {
    'MLP': 'The feed-forward part of each layer (multi-layer perceptron): it widens each piece’s vector into thousands of neurons, applies a simple non-linearity, and narrows it back, adding the result to the residual stream. Much of what a model stores sits here.',
    'nats': 'A unit of surprise or information, using the natural logarithm. 1 nat is about 1.44 bits. Margins and copying scores in the experiments are in nats: how much more likely one answer is than another, on a log scale.',
    'bits per token': 'The average surprise of a model on a text, in bits, per piece it reads (bits per character divides by letters instead). Lower is better: the model predicted the text more easily. The Tipping point scorecard uses it for ordinary text.',
    'Spearman correlation': 'How well two lists agree in their order, from −1 (opposite order) through 0 (unrelated) to 1 (the same order). The Storage ladder uses it to compare how two models arrange red’s uses (M7) and what a model says against what it holds (M8).',
    'LoRA': 'Low-rank adaptation: a small fine-tune stored as a pair of thin matrices beside the model’s weights and added to them as it runs. Settings > Thinking in E-levels uses one to make the model answer in E-levels without the lens.',
    'control vector': 'A direction added to the model’s internal activations at chosen layers, with a strength. A push inside parameter space, not an instruction. Settings > Thinking in E-levels > e-level steering loads one.',
    'untrained twin': 'The same architecture with random weights, never trained. The Storage ladder measures it beside every model: a level counts only if the trained model clearly beats its twin, so structure that comes from the words alone is not credited.',
    'M0–M8 (storage levels)': 'The Storage ladder’s nine levels of how a model stores a concept: M0 the word’s vector, M1 single neurons, M2 directions, M3 the layout of meanings, M4 the wiring between neurons, M5 combinations built while reading, M6 backups, M7 a layout shared with other labs’ models, M8 self-report coherence (said vs held; named for its design limit until the crossed design runs).',
    'intervention': 'A change made to a model on purpose at one level (swap two word vectors, silence neurons, remove a direction, cut wires), after which everything is measured again. It tests what rests on that level, not just whether it is there.',
    'control': 'A change of the same kind and size as an intervention, made to things unrelated to the concept (random neurons, unrelated words). Whatever moves under the control is the cost of any change; only movement beyond it belongs to the intervention.',
    'zero-ablation': 'Silencing a neuron by setting its output to zero. Simple, but zero can be an unusual value for that neuron, so part of the effect may come from the odd value rather than the missing signal. Compare mean-ablation.',
    'z-score': 'How unusual a value is for that neuron: how many standard deviations it lies from the neuron’s average over ordinary text. Makes neurons in different layers comparable; raw activations are not.',
    'drafted name': 'A neuron name proposed by Prometheus from the neuron’s evidence: a hypothesis, marked “drafted” wherever it appears. Names you write are marked as yours and never overwritten by a draft.',
    'E-level framing': 'A system prompt that asks the model to think in the fifteen E-levels. It changes the input, not the weights; its tokens are part of what the model reads, and attention to them is shown apart.',
    'mean-ablation': 'Silencing a neuron by holding it at its average value over neutral text, instead of at zero. Zero can itself be an unusual signal; the mean removes what the neuron says about this input while keeping the network’s usual level.',
    'creativity (temperature)': 'The sampling temperature. At 0 the likeliest piece is always taken; higher flattens the odds so less likely pieces are drawn more often. It changes which road is taken, not the model.',
    'greedy': 'Always taking the single likeliest next piece, with no random draw: the same input gives the same output every time. Compare and Neural net answer this way.',
    'measured / estimated': 'Measured: read off directly (a probability the model gave). Estimated: worked out from part of the data, such as doubt from only the top 20 choices, so it can be off; shown with “at least”, “≥” or “about”.',
    'off scale': 'A value outside what the instrument can read, such as a piece the model did not rank in its top 20, or an escape point beyond the settings tried. Shown as a bound (“over N bits”) or a range, never as a number, and left out of averages.',
    'local engine': 'llama.cpp running the GGUF file. It writes the answers in Generate, Read, Tokens (the pieces), Generate’s roads and Compare. Steering and the LoRA change this engine only.',
    'inside engine': 'PyTorch running an unpacked copy of the model layer by layer, for Neural net, Tokens (where words live), Anatomy and the experiments. Tipping-point cuts change this engine only, until Restore the model.',
    'changed model': 'The model while steering, a LoRA or a Tipping-point cut is active. A “changed” chip shows in the header, and results made then are marked, because they are not the unchanged model’s.',
    'structural surprise': 'Sensing surface: a word’s surprise given the whole text minus its surprise given only the three words before it, in bits, both read by the same model. Rare words and sentence openings surprise both readings and cancel; what is left is surprise at what the text claims. The first word has no short context and no score. A high rank is only relative to the text until it is compared with ordinary text (research/sensing_null.py).',
    'short-context reading': 'Sensing surface: the same model reading each word with only the three words before it (a 3-word window): an autocomplete’s view. The baseline that structural surprise is measured against.',
    'flip count': 'Sensing surface: after how many contradicting observations the model’s answer moves from the claim to the evidence, by more than 0.5 nats and staying there (0 to 0.5 is leaning, within noise). Read over five shuffled orders of the observations, so it is a range; a framing is compared against a neutral prefix of the same length.',
    'recovery (converge / unlock)': 'Sensing surface, the cycle inside training: after an anomaly (the onset of a drop past the tolerance), the climb back. Converge: back to the earlier level within the tolerance. Unlock: past the earlier peak by more than the tolerance. These are the essay’s two outcomes of a new abstraction; a training curve shows the recovery, not the naming.',
    'connection axis': 'Connectors: the direction inside the model shared by connection sentences across twelve levels. 0 = the average sentence about a thing on its own, 1 = the average sentence about a connection.',
  };
  // Extra ids for a term, so links can use a plural or a shorter name.
  const ALIAS = {
    'induction head': ['term-induction-heads'], 'MLP neuron': ['term-mlp-neurons'], 'ablation / lesion': ['term-ablation', 'term-lesion'],
    'nats': ['term-nat'], 'bits per token': ['term-bits-per-character', 'term-bits'], 'Spearman correlation': ['term-spearman'],
    'M0–M8 (storage levels)': ['term-m0-m8', 'term-m-levels', 'term-storage-levels'], 'intervention': ['term-interventions'],
    'control': ['term-controls'], 'mean-ablation': ['term-mean-ablate', 'term-mean-ablated'], 'untrained twin': ['term-random-twin'],
    'probe': ['term-probes'], 'logit lens': ['term-lens-logit'], 'KV cache': ['term-kv'], 'residual stream': ['term-residual'],
    'control vector': ['term-control-vectors'], 'seed (random)': ['term-seed', 'term-random-seed'], 'seed model': ['term-seeds'], 'loaded / last used': ['term-loaded', 'term-last-used'], 'spill (GPU)': ['term-spill', 'term-spilled'], 'untrained floor': ['term-noise-floor', 'term-floor'],
    'creativity (temperature)': ['term-creativity', 'term-temperature'], 'measured / estimated': ['term-estimated', 'term-measured'], 'off scale': ['term-off-scale-values'],
    'inside engine': ['term-inside-the-model-engine'], 'changed model': ['term-changed'],
    'recovery (converge / unlock)': ['term-recovery', 'term-converge', 'term-unlock'], 'flip count': ['term-flip'], 'short-context reading': ['term-short-context'],
  };
  const EMOFF = window.PSIEM && !PSIEM.on, EMT = /E-level|lens|Emergence Machine|Prometheus/;   // the framework's terms go while its content is off (Settings > Content)
  // The manual's framework-only sections (Fight class, Connectors, the How AI Emerges toy) are hidden with it: each from
  // its heading to the next heading of the same or a higher level.
  if (EMOFF) for (const id of ['fight', 'connectors', 'toy']) {
    const h = document.getElementById(id); if (!h) continue; const lv = +h.tagName[1]; let n = h;
    do { n.setAttribute('data-em', ''); n = n.nextElementSibling; } while (n && !(/^H[1-6]$/.test(n.tagName) && +n.tagName[1] <= lv));
  }
  const G = Object.fromEntries(Object.entries(Object.assign({}, EXTRA, window.PSI_GLOSSARY || {})).filter(([k, v]) => !EMOFF || !/^(E-levels|lens)$/.test(k))), gl = document.getElementById('gl'), used = new Set();
  Object.keys(G).sort((a, b) => a.localeCompare(b)).forEach((k) => {
    const dt = el('dt', null, k), id = slug(k);
    if (!used.has(id)) { dt.id = id; used.add(id); }
    (ALIAS[k] || []).forEach((a) => { if (used.has(a) || document.getElementById(a)) return; const s = el('span'); s.id = a; dt.appendChild(s); used.add(a); });
    gl.appendChild(dt); gl.appendChild(el('dd', null, G[k]));
  });
  const Q = (window.PSI_GUIDE || []).filter((x) => !EMOFF || !EMT.test(`${x.q} ${x.p}`)), K = window.PSI_GUIDE_KINDS || [], qm = document.getElementById('qm');
  K.forEach((k) => {
    const items = Q.filter((x) => x.k === k.id); if (!items.length) return;
    qm.appendChild(el('h3', null, k.t));
    items.forEach((x) => { const d = el('details'); d.appendChild(el('summary', null, x.q)); d.appendChild(el('p', 'p', x.p)); d.appendChild(el('p', null, x.see)); if (x.f) d.appendChild(el('p', null, `Found: ${x.f}`)); if (x.o) d.appendChild(el('p', null, `Elsewhere: ${x.o}`)); qm.appendChild(d); });
  });
  // The glossary is built after the page is parsed, so a deep link to a term needs a nudge.
  const jump = () => { const h = decodeURIComponent(location.hash.slice(1)); if (!h.startsWith('term-')) return; const t = document.getElementById(h); if (t) (t.closest('dt') || t).scrollIntoView({ block: 'start' }); };
  if (document.readyState === 'complete') jump(); else addEventListener('load', jump);
})();

// Back to the app: the manual opens beside the app (its own window in the desktop app, a new tab in the browser), so
// going back brings that app forward and closes this page, with the view asked for; opened on its own, it navigates.
(() => {
  // Delegated, so links added later (Prometheus's "Settings" link) behave the same.
  document.addEventListener('click', (e) => {
    const a = e.target.closest && e.target.closest('[data-to-app]'); if (!a) return;
    const view = a.dataset.toApp; let app = null;
    try { app = window.opener && !window.opener.closed ? window.opener : null; if (app) void app.location.href; } catch { app = null; }
    if (!app) return;                                   // no app behind this page: follow the link (index.html)
    e.preventDefault();
    try { if (view) app.location.hash = view; app.focus(); } catch { /* the app window refused: closing still leaves it */ }
    window.close();
  });
})();
