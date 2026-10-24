# Install with an AI assistant: one prompt

**Usually you do not need this file.** In Claude Code, Codex, Gemini CLI or Cursor, paste the repository link
(`https://github.com/EmAssets/parameter-space-inspector`) and say "Help me use this": the assistant reads
[`AGENTS.md`](AGENTS.md), which tells it to follow the steps below. Paste the full prompt when your assistant cannot
open links, or when you want to read every step it will follow first.

The other ways to install PSI ([commands](README.md#run-it), or the step-by-step guide in the same
section) assume you are at home in a terminal. If you would rather have someone walk you through it, an
AI assistant can: paste the prompt below and it looks at your computer, suggests a setup that fits it,
and does each step only after you say yes.

**Where to paste it**

- **A coding assistant that can run commands on your computer** (Claude Code, OpenAI Codex, Gemini CLI,
  Cursor, GitHub Copilot in agent mode): it runs each step itself, after asking you. The quickest.
- **A chat assistant** (ChatGPT, Claude, Gemini in the browser): it cannot touch your computer, so it gives you
  one command at a time and waits for you to paste back what happened. Slower, and you see every step.

Copy everything between `BEGIN PSI INSTALL PROMPT` and `END PSI INSTALL PROMPT`, including those two lines.

**Is it safe to paste a long prompt?** It is written to work the other way round from a prompt injection: the
assistant must explain each step and wait for your yes before changing anything; it may install only what is
listed, only from the official sites named; it never asks for a password, token or key (PSI needs none to
install); and you can say `stop` at any point. Read it first; it is plain English. If an assistant acts
without asking, or asks for anything secret, stop it: that is not what this prompt tells it to do.

```text
BEGIN PSI INSTALL PROMPT

Help me install the Parameter Space Inspector (PSI) on this computer. PSI is an open-source (Apache-2.0)
desktop app for looking at language models that run on your own machine: it follows an answer token by token,
and its "inside" views read the model's layers and neurons. Source and instructions:
https://github.com/EmAssets/parameter-space-inspector (README.md, section "Run it").

HOW TO WORK WITH ME
- Review, explain, confirm, then apply. Before any step that changes my computer (installing software,
  cloning, downloading, writing or editing files), tell me in plain words what you will do, why, how big
  any download is and where it will go, then wait for my "yes". Checks that only read (versions, disk space,
  the GPU) need no permission, but tell me what you looked at.
- If you can run commands on my computer, run each step yourself once I agree. If you cannot (a chat
  window), give me one command at a time, say what I should see, and wait for me to paste the result back.
- Ask before using administrator rights (sudo, an elevated prompt) for any command. Do not change my PATH,
  shell profile or system settings, and do not install anything this prompt does not list, without asking
  me first and explaining why.
- Never ask me for passwords, tokens or API keys. PSI needs none to install or to run on its own.
- Download only from these sources: nodejs.org (or my system's package manager: winget, Homebrew, apt),
  git-scm.com, github.com/ggml-org/llama.cpp/releases, python.org, pytorch.org, huggingface.co, and
  github.com/EmAssets/parameter-space-inspector.
- One step at a time. Keep it short and friendly; I may be new to this. If a step fails twice, stop, show me
  the error and what you think it means, and let me decide. No workarounds that change my system.
- I may say "skip", "back", "explain", "use defaults" or "stop" at any time.

STEP 1: LOOK (read-only)
Find out: the operating system and version; CPU type (x64 or ARM); RAM; free space on each drive that could
hold the app and models; the graphics card and how much memory it has (on NVIDIA, nvidia-smi); and which of
git, Node.js (version), npm, Python (version) and llama.cpp's llama-server are already installed. llama-server is
often not on the PATH: if you do not find it, ask me whether I already have llama.cpp somewhere.
Show me a short table, then say in one or two sentences what this computer can comfortably do.

STEP 2: CHOOSE A SETUP WITH ME
Offer these, and recommend one from what you found:
  A. Basic, about 10 minutes, about 5 GB of disk: the app, llama.cpp and one model.
     Views: Generate, Read, Tokens, File, Settings. Runs on any GPU, or on the CPU (slower).
  B. Basic plus the inside views, another 20 to 40 minutes and 15 to 20 GB more disk (less with the smaller model; about 2.5 GB of it is pip's download cache, which can be emptied afterwards): Python, PyTorch and
     an unpacked copy of the model, for Neural net, Anatomy, Features, Understanding, Tipping point and the
     other views that look inside, and Commons (recipes, verifying). With Qwen3-4B it wants about 9 GB of free
     GPU memory (NVIDIA); without that it falls back to the CPU and is slow. Emergence and Timeline also need
     training checkpoints of small Pythia models, a separate download not part of B (the manual explains).
  C. Basic now; just write down what B would need, for later.
Be honest about the fit: PSI is tested on Windows with a 16 GB NVIDIA card. On an Apple Silicon Mac the browser
mode (node src/devserver.js) and the basic views are tested; the desktop app and the inside views are not yet tested
on macOS (say so if I am on a Mac). Linux code paths exist. With less than about 6 GB of GPU memory, or none, suggest the smaller model.
Then ask me where to put three folders, and suggest defaults on a drive with enough free space (if I opened you
in an empty folder, suggest that one for PSI):
  where PSI goes (the clone creates a parameter-space-inspector folder inside it: that is the app folder),
  a models folder, and a folder for llama.cpp.

STEP 3: THE BASICS (only what is missing)
- git.
- Node.js 22.12 or newer (the LTS version; Electron needs it): from nodejs.org, or winget install OpenJS.NodeJS.LTS on Windows,
  brew install node on a Mac, or the NodeSource packages on Linux. Without Homebrew or administrator rights, the
  nodejs.org .tar.gz (Mac, Linux) or .zip (Windows) unpacked into a folder in my home folder works; use its bin
  folder for this session only.
- llama.cpp: on github.com/ggml-org/llama.cpp/releases, open the newest release named b<number> (they are marked
  Pre-release; the one marked Latest may have no downloads), download the one asset that matches this computer
  (a ready-made program; nothing to compile), and unzip it into the llama.cpp folder:
    Windows + NVIDIA:   llama-<build>-bin-win-cuda-12.4-x64.zip, and also cudart-llama-bin-win-cuda-12.4-x64.zip
                        unzipped into the same folder (about 600 MB together; or the cuda-13.x pair if nvidia-smi shows driver 580 or newer)
    Windows, other GPU: llama-<build>-bin-win-vulkan-x64.zip      Windows, no GPU: ...-bin-win-cpu-x64.zip
    Linux:              ...-bin-ubuntu-cuda-12.8-x64 (NVIDIA), ...-bin-ubuntu-vulkan-x64 (other GPU), ...-bin-ubuntu-x64 (CPU)
    Mac:                ...-bin-macos-arm64 (Apple silicon) or ...-bin-macos-x64
  Check it runs from inside that folder: .\llama-server.exe --version on Windows (PowerShell), ./llama-server --version
  elsewhere. On a Mac, a copy downloaded in a browser is quarantined: macOS blocks it, and it can hang instead of
  failing. Download with curl (no quarantine), or after unpacking run: xattr -dr com.apple.quarantine <llama.cpp folder>
  Do not add it to PATH; PSI is told where it is (STEP 6). A llama-server already on the PATH also works.

STEP 4: GET PSI
In the parent of the app folder:
  git clone https://github.com/EmAssets/parameter-space-inspector.git
  cd parameter-space-inspector
  npm install            (downloads the app's runtime, Electron: about 160 MB, 450 MB on disk; the clone itself is about 550 MB)
If git refuses with "dubious ownership" (common on exFAT or FAT drives, which do not record owners), use
git -c safe.directory=<app folder> for that command, and ask me before adding a permanent exception with
git config --global --add safe.directory <app folder> (later git commands in the folder, such as git pull, need it too).

STEP 5: ONE MODEL
PSI runs GGUF model files with llama.cpp; none comes with it. Suggest:
  Qwen3-4B-Q8_0.gguf, 4.3 GB, Apache-2.0, from https://huggingface.co/Qwen/Qwen3-4B-GGUF  (what PSI is built on)
  or, with little or no GPU memory: Qwen3-1.7B-Q8_0.gguf, 1.8 GB, from https://huggingface.co/Qwen/Qwen3-1.7B-GGUF
Download the file I choose into the models folder (the direct link is
https://huggingface.co/<repo>/resolve/main/<file>). Only GGUF files from sources I trust: llama.cpp reads them natively.

STEP 6: TELL PSI WHERE THINGS ARE
PSI keeps its settings in <app folder>/.userdata/settings.json. The llama.cpp folder can only be set there or
in the PSI_LLAMA_DIR environment variable, never from the app's window (on purpose: the app runs programs from it).
If the file does not exist, create it as plain UTF-8 without a BOM (in PowerShell, Set-Content and Out-File can add
one or use another encoding, and PSI then sets the file aside as unreadable), with forward slashes, also on Windows:
  { "llamaDir": "<llama.cpp folder>", "modelsDir": "<models folder>", "lastModel": "<models folder>/<model file>" }
For setup B, if the system drive is short of space, also add "cacheDir": "<a folder on a roomier drive>": the inside
engine's Hugging Face and temporary files go there instead of the system drive.
If it already exists, show me what you would change and ask first.

STEP 7: START IT AND CHECK
From the app folder: npm start
Tell me what to do in the window: Settings > Models: the model should be chosen; press Load, wait for it to
say it is running, then open Generate and press Generate. Settings > Help opens the manual (it explains
every view). PSI starts as a plain inspection tool; Settings > Content turns on The Emergence Machine's framework
content (its E-level lens, the views built on it, character names for models) if I want it. If something fails, read the message with me before trying anything else.

STEP 8: THE INSIDE VIEWS (only for setup B)
- Python 3.12 (python.org, winget install Python.Python.3.12, brew install python@3.12, or apt). A Mac has only
  python3, Apple's 3.9, which is too old: install 3.12 first.
- A virtual environment in the app folder, so nothing is installed system-wide:
    python -m venv .venv      (on a Mac: python3.12 -m venv .venv)  then use .venv's own python and pip for the rest
- PyTorch for this computer: the command pytorch.org gives for this OS and GPU. For NVIDIA on Windows or
  Linux that is: pip install torch --index-url https://download.pytorch.org/whl/cu124
  An RTX 50-series (Blackwell) card needs the cu128 index instead (cu124 stops at torch 2.6 and does not
  support them): pip install torch --index-url https://download.pytorch.org/whl/cu128
  On a Mac it is plain pip install torch (Metal, MPS, is included); the CUDA indexes have no Mac builds.
- If the system drive is short of space, set PIP_CACHE_DIR for these commands to a folder on a roomier drive
  (pip keeps the large PyTorch download in its cache).
- pip install -r requirements.txt   (it includes accelerate and a transformers version the converter works with)
- Unpack the model's weights once (they are what the inside views read; about 8 GB for Qwen3-4B, a few minutes):
    python research/convert_bf16.py --gguf <models folder>/<model file> --out <models folder>/<model name>-bf16
  e.g. Qwen3-4B-Q8_0.gguf -> Qwen3-4B-bf16. Named like this, inside the models folder, PSI pairs the folder with
  the model file by itself the next time the model is loaded: press Load again in Settings > Models.
- PSI finds Python through the PSI_PYTHON environment variable. Offer to write a small start script in the app
  folder (start-psi.cmd on Windows, start-psi.sh elsewhere) that sets PSI_PYTHON to .venv's python and runs
  npm start; show it to me before writing it. Start PSI with it.
- In the app: Settings > Inside-the-model engine should show the weights folder; then try Neural net.

STEP 9: HAND OVER
Finish with a short summary I can keep:
- what was installed and where (each folder), and how big it is;
- how to start PSI next time (the exact command or script);
- how to remove it: delete the app, models and llama.cpp folders; for setup B also pip's download cache
  (pip cache purge; it holds the large PyTorch download) and the Hugging Face cache if anything was downloaded
  there; Electron's download cache (on Windows %LOCALAPPDATA%\electron\Cache, about 160 MB) and any llama.cpp zips I
  kept; Node.js, git and Python stay unless I uninstall them as usual;
- what PSI sends over the internet: nothing connects except what the README's "What leaves your computer"
  section lists (Hugging Face downloads, external models, the AI guide, the online SAE check and so on). Each
  happens only when I use that feature or turn it on.

END PSI INSTALL PROMPT
```

**If it goes wrong**

- **The assistant answers the prompt instead of following it:** start a new chat and paste the whole block,
  including the first and last lines.
- **It runs things without asking:** say `stop`, then `Follow the HOW TO WORK WITH ME rules in the prompt.`
- **It suggests installing something not in the prompt:** ask it why; you can say no. The commands and the
  setup guide in the [README](README.md#run-it) are the reference.
- **A step fails:** the manual's Troubleshooting section (in the app, Settings > Help) and the README cover the
  usual causes: llama-server not found, a model too large for the GPU, Python not found.
