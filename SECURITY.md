# Security

## Reporting a vulnerability

Please report security problems privately, not in a public issue: use **Security > Report a vulnerability** on this
repository's GitHub page (private vulnerability reporting). Say what you found, how to reproduce it, and what an attacker
gains. You will get an answer within 7 days. A fix is released before the details are made public, and you are credited
unless you prefer not to be.

Only the latest release is supported.

## What the app protects, and how

PSI runs on your own computer. What it guards:

| Asset | Where it lives | Protection |
|---|---|---|
| Your OpenRouter key and spending cap | `.userdata/openrouter.key`, `settings.json` | Never sent to the page or to agents; files are owner-only (0600) where the system supports it; every paid request is checked against the cap before it is sent |
| Model signing keys (if you sign seeds) | `.userdata/*.key` | Created owner-only, never overwritten, never committed (`.gitignore`), and checked for and left out whenever the code is published |
| Your files | anywhere | The page can read only `.gguf` files in your models folders or ones you picked in a dialog, never the data folder; no network (UNC) paths |
| Programs the app runs | llama.cpp folder, Python | The llama.cpp folder and the cache folder are set only in the environment or `settings.json`, never from the window or by an agent |

Boundaries:

- **The desktop window** has context isolation, the sandbox and a Content-Security-Policy with no inline script. It
  never navigates away from its own pages; links open in your browser. Only the app's bundled pages can call it.
- **The browser server** (`node src/devserver.js`) listens on 127.0.0.1 only, needs a fresh token for every call, answers
  only its own host names (blocking DNS rebinding), refuses foreign origins and framing, and caps request sizes.
  Extra host names are answered only when a request carries Cloudflare Access's `Cf-Access-Jwt-Assertion` header. The
  server checks that the header is there, not that the token in it is valid: a convenience behind Cloudflare Access,
  not a guarantee. Behind any other tunnel or proxy that does not strip or set that header, anyone who can reach the
  server can use the app.
- **The inside engine** (`trace_server.py`) listens on 127.0.0.1 with its own token. The app sends the token only after
  the server proves it holds it (an HMAC of a fresh nonce), so a program that grabs the port first gets nothing.
- **llama-server** gets a new API key at each start, through its environment (not the command line).
- **Prometheus (the AI guide)** treats what it reads (model output, search results) as untrusted. Its replies are drawn
  as text and links, never HTML. It can use only the tools offered to it, never fills a Settings field or a path, and
  waits for your yes before anything that spends money or changes a model.
- **Outside agents (MCP)** have their own token. They cannot change keys, the cap or the engine. Changing a model and
  spending credit each need a switch in Settings > Agents, both off by default.

## What it does not protect against

- **Untrusted model files.** llama.cpp and PyTorch parse model files natively. Open only GGUF files and weights folders
  you trust, and keep llama.cpp up to date.
- **Other programs running as you.** Anything running under your account can read your files, including the data folder.
- **Agents you connect.** An agent with the MCP token can use every read-only tool and run local models on your GPU.
- **A shared computer.** On Windows, the data folder's protection is the folder's own permissions: keep it in your profile
  or on an NTFS drive, not on a shared or FAT/exFAT drive.
