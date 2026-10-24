# Parameter Space Inspector

Questions about how AI works, answered by opening a real language model on your own computer and
measuring it. How does it choose each word? What does it actually read? Where inside the network does
an answer form? When did it learn to understand? Where does it keep an idea? How much can you remove
before it fails? Does it know when it is about to be wrong?

PSI is a desktop app (Electron) that runs any GGUF model with [llama.cpp](https://github.com/ggml-org/llama.cpp)
and, for the views that look inside, a second engine (PyTorch) on the model's unpacked weights. It was
built on Qwen3-4B (4 billion parameters, 4.3 GB at 8-bit) on an ordinary 16 GB graphics card.

Page, screenshots and findings: <https://emergencemachine.com/tools/parameter-space-inspector/>
From [The Emergence Machine](https://emergencemachine.com).

PSI starts as a plain inspection tool. **Settings > Content** turns on The Emergence Machine's framework content: the
E-level lens (a system prompt that frames answers), the views built on it and character names for models.

> **AI assistant reading this for someone?** Follow [`AGENTS.md`](AGENTS.md): it says how to install PSI with them,
> step by step and only with their yes, or help them use it.

## What it shows

| View | The question | What you see |
|---|---|---|
| **Generate** | How does it write an answer? | The answer coloured by how torn the model was at each piece; a fork map of every step's alternatives with their odds (click one to take that road); the die for a step. Below it, **where the roads go**: sampled continuations from one fork, what that choice closed off, the basins answers settle into, and loops. |
| **Read** | What surprises it? | Any text coloured by surprise, with what the model expected instead. An invented word costs ~37 bits the first time and ~4 the second. |
| **Tokens** | What does it actually read? | The pieces the model sees, how many the same sentence costs in five languages, where your words sit in its embedding space. |
| **Neural net** | Where does the answer form? | For each token it writes: what each layer would say if it stopped there, the neurons that fired, what it looked at; knock-outs with random controls. |
| **Anatomy** | What is the network made of? | Every layer drawn from the weights; activation patching (where a fact is decided); light up a concept, then delete it and look again. |
| **Features** | What is it thinking about, in parts that mean one thing? | A sparse autoencoder (trained in the app or imported) rewrites a layer's state as a few active features: which fire on your text, each feature's card (how often it fires, the words it promotes, its strongest examples, steering an answer with it), search by name or example, and which features pushed the next word. |
| **File** | What is in the file? | Everything the GGUF header says, for files of any size, without reading the weights. |
| **Self-portrait** | What do we know about one model? | Every stored measurement of one model beside its untrained twin, written for a person or an AI agent. |
| **Experiment designer** | Can I test my own idea? | Prometheus drafts a plan (prompts plus a baseline, e.g. one question in six languages beside an unrelated one); you edit it and only you can run it. The app runs the plan with any of its instruments (Neural net, Read, Generate, Tokens, Connectors, Fight class, Sensing surface, Emergence, Understanding, knock-outs, activation patching) and draws the comparison; Prometheus can run it in front of you, view by view. |
| **Compare framings** | Does a framing change the answer? | The same question with and without the E-level lens, side by side. |
| **Understanding** | Does it understand, or only sound right? | Tests that separate understanding from surface statistics, scored against an untrained twin, layer by layer. |
| **Emergence** | When was your question learned? | Your test scored at every saved training checkpoint of Pythia-410m; the heads that look back, and what dies when you remove them. |
| **Timeline** | What emerged when, and where does my model stand? | 19 capabilities and the training step at which each emerged in five Pythia sizes (onset intervals, a mirage check against steady rises); any model folder placed on the same axis with the same tests, the checkpoint it most resembles and what emerged next there; a graft lab that puts one model's layers into another (drop-in, same width, or stitched with fitted maps) against the gap and a random control. Every result carries a signed receipt; checkpoints of one family can be signed as a Merkle series. |
| **Tipping point** | How much can be removed before it fails? | Delete a share of neurons, heads, weights or one concept; the same scorecard after each cut. |
| **Sensing surface** | How does it absorb something new? | Anomalies a text creates, how much evidence flips an answer, the absorb–anomaly–recover cycle in training. |
| **Fight class** | Autopilot, or Strategist? | A probe that reads the kind of competition a sentence describes; whether the model's inner state knows if it will be right. |
| **Connectors** | Does it hold one idea of "a connection"? | Whether different connecting words share one representation. |
| **Storage ladder** | How high does it store "red"? | Nine levels of storage, from the word's vector to a layout shared across labs, each tested against an untrained twin. |
| **Seed & lineage** | Where did this model come from? | Grow small models with a record of their parent, data and training (signed only if you choose a key). |
| **Commons** | Can many people grow one model, and can I check their claims? | Seed lines (first: stories260K, 260K numbers, trains on a CPU in seconds). Train a branch on your own text with a replayable trace; combine branches with recipes pinned by hash (average, weighted, trimmed mean, TIES, DARE-TIES, layer graft: all canonical, so a recipe gives the same bytes anywhere); import recipes from Hugging Face, GitHub, any https link or a file; check every ingredient, then build, recover a lost branch, or build without it; GGUF export. A verifier anyone can run (`src/verify_lineage.py`) marks every claim as proof, evidence or claim. Signing is optional (private by default), with your own keys. |

Some results went against the ideas the app was built to test, and they are kept. The questions the app
answers, with what the measurements showed, are written up at
[parameterspaceinspector.com](https://parameterspaceinspector.com).

## Example: does merging fine-tunes actually help?

What the app is for, in one question. Merging public fine-tunes is popular; whether it gives you anything better than
what you already have is rarely measured. With Commons the question became answerable in an afternoon:

1. **Find candidates.** Find models that merge (Commons > Recipes > 5) lists public fine-tunes of a base with the same shape.
2. **Screen them.** Distance from the base tells a light fine-tune (0.0004–0.04 across 20 public models) from a
   heavily retrained model: Qwen2.5-Coder-0.5B-Instruct is 0.56 from Qwen2.5-0.5B, so it is far more than a light fine-tune.
   It also shows which base a fine-tune is closest to, and it caught a NaN-weight upload.
3. **Build recipes** that rebuild to the same bytes anywhere, so every merge tested can be checked.
4. **Test against the real alternatives**, on held-out data with intervals: the specialists themselves, and one model
   trained on everything. The base model is not the comparison.

The answer, so far: **not as a way to get better quality.**
- Two specialists (addition 80.5%, word reversal 100%) merged into one model that does both, but addition fell to
  49.5% (95% interval 42.6–56.4): significantly below the specialist, and below joint training (86%).
- Qwen Instruct + Coder broke; the distance screen predicted it.
- Qwen Instruct + a medical fine-tune landed between its parents.

What merging can still offer is one model instead of N, and a way to combine models whose training data nobody has.
The next question tests exactly that, with the rule fixed in advance: can one merge of four guard models stay within
2 points of each specialist?

## Requirements

- **Node 22.12+** (Electron 44 needs it) and npm.
- **llama.cpp**: a [release](https://github.com/ggml-org/llama.cpp/releases) with `llama-server`
  (the CUDA build for NVIDIA cards).
- **A GGUF model.** None is included. Built on Qwen3-4B (Apache-2.0):
  [`Qwen3-4B-Q8_0.gguf`](https://huggingface.co/Qwen/Qwen3-4B-GGUF), 4.3 GB.
- **For the views that look inside** (Neural net, Anatomy, Understanding, Emergence, Tipping point,
  Sensing surface, Fight class, Connectors): Python 3.12 with PyTorch and the packages in
  [`requirements.txt`](requirements.txt), an unpacked bf16 copy of the model (8 GB on disk for Qwen3-4B),
  and about 9 GB of free GPU memory (it falls back to the CPU, slowly). Emergence also needs the
  [Pythia-410m](https://huggingface.co/EleutherAI/pythia-410m) checkpoints.

Developed and tested on Windows with a 16 GB NVIDIA GPU. On an Apple Silicon Mac, the browser mode and the basic
views are tested (llama.cpp on Metal, October 2026); the desktop app and the inside views are not yet tested on
macOS. Linux code paths exist.

## Run it

Three ways in, as on [parameterspaceinspector.com](https://parameterspaceinspector.com). Pick the one that fits how
comfortable you are in a terminal.

### 1. Ask an AI assistant (easiest)

Open Claude Code, Codex, Gemini CLI or Cursor in an empty folder and paste:

```
https://github.com/EmAssets/parameter-space-inspector
Help me use this
```

It reads [`AGENTS.md`](AGENTS.md), looks at your computer, suggests a setup that fits, and does each step only after
you say yes. It never asks for passwords or keys. Using a chat assistant that cannot open links? Paste
[`INSTALL_PROMPT.md`](INSTALL_PROMPT.md) instead; it gives you one command at a time.

### 2. Commands

```bash
git clone https://github.com/EmAssets/parameter-space-inspector.git
cd parameter-space-inspector
npm install
npm start
```

Then open **Settings**, choose your GGUF file and press **Load**. Tell PSI where llama.cpp is first: set
`PSI_LLAMA_DIR` to the llama.cpp folder, or `"llamaDir"` in `.userdata/settings.json` (the app runs programs from that
folder, so it is set only there, never from the window). `llama-server` on the PATH also works, but there is no
need to change your PATH. Open only GGUF files you trust: llama.cpp parses them natively.

### 3. By hand, step by step

1. Install **Node.js 22.12+** and **git**. No administrator rights? The nodejs.org `.tar.gz` (macOS, Linux) or `.zip`
   (Windows) unpacked into your home folder works; add its `bin` folder to the PATH for that terminal.
2. Download **llama.cpp** from its [releases](https://github.com/ggml-org/llama.cpp/releases): the newest release named
   b&lt;number&gt; (marked Pre-release; the one marked Latest may have no downloads), the asset for your system and GPU:
   for NVIDIA on Windows, the `cuda` zip plus its `cudart` zip; on an Apple Silicon Mac, `...-bin-macos-arm64`.
   **On a Mac**, a copy downloaded in a browser is quarantined and macOS blocks it (Load then hangs instead of
   failing). After unpacking, run `xattr -dr com.apple.quarantine <llama.cpp folder>`, or download with `curl`.
3. Download a **GGUF model**, for example [Qwen3-4B-Q8_0](https://huggingface.co/Qwen/Qwen3-4B-GGUF) (4.3 GB), or
   Qwen3-1.7B-Q8_0 (1.8 GB) with little GPU memory.
4. Clone PSI, run `npm install`, then `npm start`.
5. In **Settings**, choose the model and press **Load**.
6. For the inside views (Neural net, Anatomy, Understanding and the others):

   ```bash
   python -m venv .venv            # Python 3.12; on a Mac install it first, then: python3.12 -m venv .venv
   # then use .venv's python and pip
   pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA (RTX 50-series: cu128)
   pip install torch               # a Mac (Metal/MPS is included) or no GPU
   pip install -r requirements.txt
   python research/convert_bf16.py --gguf <models>/<model>.gguf --out <models>/<model name>-bf16
   ```

   Named like that, inside your models folder, the unpacked weights are paired with the model the next time you
   press **Load**; set `PSI_PYTHON` to the virtual environment's python.

The manual in the app (**Settings > Help**) walks through every step, every view and every setting.

### Settings and environment

Everything is set in **Settings** while the app runs and saved to `.userdata/settings.json` (or `PSI_DATA_DIR`).
Environment variables give start-up defaults; settings saved in the app take precedence.

| Variable | What | Default |
|---|---|---|
| `PSI_LLAMA_DIR` | the llama.cpp folder | `llama-server` on the PATH |
| `PSI_MODEL` | the GGUF to answer with | none |
| `PSI_MODELS_DIR` | the folder Settings scans for GGUF files | none |
| `PSI_TRACE_MODEL` | the unpacked bf16 weights for the inside engine | none |
| `PSI_CHECKPOINTS` | the Pythia-410m checkpoints folder | none |
| `PSI_CACHE_DIR` | cache for the Python engines (Hugging Face downloads, temporary files); also `"cacheDir"` in settings | system defaults |
| `PSI_DATA_DIR` | settings, keys and names | `.userdata/` in the app folder |
| `PSI_PYTHON` | the Python for the inside engine and jobs (a virtual environment's python, for example) | `python` / `python3` on the PATH |
| `PSI_DEV_PORT`, `PSI_TRACE_PORT` | the browser server's port and the inside engine's port (to run two copies side by side) | 4310, 4311 |
| `PSI_ALLOWED_HOSTS` | extra host names for the browser server, served only through Cloudflare Access; also `"allowedHosts"` | none |
| `PSI_DEVICE` | force the Python engines onto one device: `cpu`, `cuda`, `cuda:1`, `mps` or `xpu` (it must be available) | the GPU when there is room, else the CPU |
| `PSI_LORA` | a LoRA adapter to load with the GGUF | none |
| `PSI_CORPUS` | the text sample (a JSON list of paragraphs) for neuron examples and the normal range | `datasets/neuron-corpus.json` in the cache folder |
| `PSI_EXAMPLES_DIR` | where neuron examples and normal ranges are kept | `psi/` in the cache folder, else `cache/` in the data folder |
| `PSI_SAE_DIR` | where `src/sae.py` keeps SAEs when run by hand (the app always uses `sae/` in the data folder) | `sae/` in the data folder |
| `PSI_SECRETS` | `file` keeps keys in 0600 files instead of the system's keychain | the system's keychain |
| `PSI_ALLOW_LOCAL_HOSTS` | `1` turns off the block on local and private addresses for recipe imports. For testing only | off |

`PSI_SELFTEST=1 npm start` drives the basic views (Generate, Read, Tokens, Compare, File), saves screenshots to `out/` and exits.

## The same UI in a browser

`node src/devserver.js` serves the UI at <http://127.0.0.1:4310>, with the Electron bridge replaced by
local HTTP calls. It answers only on this computer. Every POST needs a token made at each launch (the page
reads it from a `<meta name="psi-token">` tag in the served index.html; the token printed at start opens `/mcp` only) and only Host `127.0.0.1`/`localhost` is served, so other web
pages in the same browser cannot drive it. To use it from another device, put it behind a proxy that
authenticates you first and add that host name to `PSI_ALLOWED_HOSTS`. Requests to such a host are answered only when they
carry Cloudflare Access's `Cf-Access-Jwt-Assertion` header. The server checks only that the header is there, not that it
is valid: a convenience behind Cloudflare Access, not a guarantee. Behind any other tunnel or proxy that does not strip or
set that header, anyone who can reach the server can use the app; with another proxy, forward to `127.0.0.1:4310` instead
(an SSH tunnel, for example). Never expose the port directly. The pages may not be framed by other sites.

## Prometheus (AI guide) and MCP

Optional. With your own [OpenRouter](https://openrouter.ai) key (Settings > External models (OpenRouter)) and the app's
spending cap, **Ask Prometheus** (Ctrl+/) opens a guide that explains terms and results and, if you allow it,
runs the Inspector: it opens views, sets controls and presses buttons in the window, showing every action
and asking first before anything that costs money.

Every instrument is also an MCP server, so any agent can study a model the way you do:

```
claude mcp add --transport http psi http://127.0.0.1:4310/mcp --header "Authorization: Bearer <token>"
```

The token is printed by `node src/devserver.js` at start (new each launch; set `PSI_MCP_TOKEN` to keep one).
Agents cannot change keys, the spending cap or the engine. Two switches in Settings > Agents are off until you turn them
on: changing the model (cuts, concept erase, grow) and spending OpenRouter credit (asking frontier models).

## A second model to test with, on your computer

Instead of OpenRouter, the model that tests (Prometheus, Compare framings, self-consistency, the judge, fact checks,
neuron naming) can be a second open model on the same computer, kept apart from the model under test. The model under
test runs in PSI's own llama-server and inside engine; the tester runs in its own server, with its own process, port and
model folder:

```bash
tools/tester-model.sh start              # a dedicated Ollama on 127.0.0.1:11435, models in ~/.psi-tester
tools/tester-model.sh pull llama3.1:8b   # give it a model (a larger one calls tools more reliably)
```

Then Settings > Prometheus > *Run it on a tester model on this computer* lists it, and the model is
`local/tester/llama3.1:8b` everywhere a model id is asked for (MCP agents included). Other OpenAI-compatible servers on
this computer work the same way: Ollama on 11434 (`local/ollama/…`), LM Studio on 1234 (`local/lmstudio/…`), llama.cpp
on 8080 (`local/llamacpp/…`); set others in settings.json (`local.endpoints`). Only loopback addresses are accepted,
nothing is spent (no cap, no spending permission for agents), and PSI refuses its own llama-server as a tester. For a
container instead, `docker compose -f tools/tester-docker-compose.yml up -d` runs the tester on the same port; on a Mac
Docker has no access to the Metal GPU, so it runs on the CPU.

## What changed

[CHANGELOG.md](CHANGELOG.md): what each release changed in the app, for agents over MCP, and in the manual.

## What leaves your computer

Nothing, unless you use an optional feature:

- **Checking for updates** (Settings > Help and about): when you press Check now, or a few seconds after each start if
  you turn on "Check for updates when the app starts" (off by default), it asks the git repository it was installed from (github.com for a normal install) for the newest commit on its
  branch (git ls-remote). Nothing about you or your models is sent. Downloading and installing an update happen only
  when you press them; an update that changes the app's Node packages also runs npm install (registry.npmjs.org).
- **Prometheus, external comparisons, fact checks, neuron naming**: OpenRouter, with your key and cap. With a model on
  this computer (`local/…`, above) nothing leaves it.
- **Running Prometheus on your ChatGPT plan**: signing in goes to auth.openai.com, questions to api.openai.com. The sign-in
  carries a random identifier for your copy of the app.
- **search_site / site_concept** (the guide's and MCP agents' tools for the site's essays): emergencemachine.com's
  MCP server, only while Emergence Machine content is on (Settings > Content). These calls say they come from PSI and which version, and the site counts them (calls per tool, per version,
  per copy's key: a random identifier created once for your copy and kept in your data folder). The counts hold no
  question text and no IP address.
- **Downloading checkpoints or models**: Hugging Face (its model API and, for dataset previews, datasets-server).
  Checking a recipe's lineage online asks Hugging Face's model API for the published commit.
- **Pushing or publishing to Hugging Face**: uploads the files you chose to your repo, with your token.
- **Importing a recipe** from Hugging Face, GitHub or any https link, and looking up a person's published keys
  (a `psi-identity` repo on Hugging Face or raw.githubusercontent.com).
- **A larger text sample** (Features): downloads part of WikiText-103 from Hugging Face.
- **Growing a model**: fetches street data from storage.googleapis.com/emergence-e8-lens.
- **Verify online** (Features): an SAE's ring records (hashes, names, training figures; never the SAE or the
  model) go to emergencemachine.com, which checks the signatures and certificates and signs its answer. It runs
  after each SAE is trained or imported, and from the view's button. The site counts these checks (no IP address).
  It is off by default; turn it on in Settings > Guardrails (Features). It never runs while Emergence Machine content is off.
- **Voice input** in the browser version: your browser's speech service (Chrome: Google). The desktop app
  points to Windows dictation instead.

## Security

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Licence

Code: [Apache License 2.0](LICENSE). The Emergence Lexicon, the essay texts used as held-out test text and The
Emergence Machine's name and logos are not under that licence, and third-party data and models keep their own licences: see
[NOTICE](NOTICE).
