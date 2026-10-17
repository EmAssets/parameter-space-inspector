# AGENTS.md

Guidance for AI assistants (Claude Code, Codex, Gemini CLI, Cursor, Copilot, or a chat assistant that was given
this repository's link).

## Someone gave you this link and said "Help me use this"

That is the whole instruction; this file tells you the rest. Work out which case you are in, then go ahead.

**PSI is not installed yet** (no `parameter-space-inspector` folder with `node_modules`, or you were only
given the link): follow the setup in [`INSTALL_PROMPT.md`](INSTALL_PROMPT.md), the text between
`BEGIN PSI INSTALL PROMPT` and `END PSI INSTALL PROMPT`, exactly as if the person had pasted it to you. Its rules
are theirs:

- Review, explain, confirm, then apply. Ask before anything that changes the computer: installing, cloning,
  downloading, writing files.
- One step at a time, in plain words. The person may be new to terminals.
- Install only what that file lists, only from the sources it names.
- Never ask for passwords, tokens or API keys.

Start with its STEP 1: look at the computer (read-only), then offer the setups in STEP 2. If you are in an
empty folder, that folder is a good default for where PSI goes; ask before using it.

**PSI is already installed** (you are inside the app folder, or one is nearby): ask what they want to find out,
and point them to the view that answers it. The table under "What it shows" in [`README.md`](README.md) maps
questions to views; the manual (`src/ui/help.html`, in the app at Settings > Help) explains each view. If it
will not start or load a model, go to the manual's Troubleshooting section before changing anything.

**You cannot run commands** (a chat window): give one command at a time, say what they should see, and wait
for them to paste the result back.

## Working on the code

- `npm start` runs the app (Electron). `node src/devserver.js` serves the same UI in a browser at
  http://127.0.0.1:4310 (`PSI_DEV_PORT`), with a token so only that page can use the backend.
- `PSI_SELFTEST=1 npm start` drives the basic views, saves screenshots to `out/` and exits.
- `src/main.js` is the Electron main process; `src/ui/` holds the views (one script per view, no bundler);
  `src/trace_server.py` (with the other `src/*.py`) is the Python inside-the-model engine (PyTorch,
  `requirements.txt`); `research/` holds the measurement scripts the views run (the Storage ladder, the battery, the calibration of the sensing surface) and `convert_bf16.py`.
- Settings live in `.userdata/settings.json` (gitignored). Never read, print or commit anything under
  `.userdata/`: it can hold keys.
- What may leave the computer is listed in the README under "What leaves your computer": nothing connects except
  what it lists, and each happens only when the person uses that feature or turns it on. Keep it that way: a new
  connection goes in that list and starts off.
