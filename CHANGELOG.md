# Changelog

Dated releases of the Parameter Space Inspector. Each entry says what changed in the app a person uses, what changed for agents over MCP, and what the manual gained. The public repository receives one commit per release.

## 0.3.0 · 2026-10-24

**The conversation's own history as the variable.** The Attractor lab's `read`, `funnel`, `path` and `constraint` actions take `history` (earlier turns of a conversation, as given, up to 24). `path` also takes `turns` (the user's text at each step, the last repeating) and `strengths` (an explicit list in place of the up-and-down sweep; all 0 watches history alone). Fresh conversations then run each distinct turn text at each strength, and `memory_vs_fresh` compares every given turn with its fresh twin. The order test runs after the history. These are for agents over MCP: the window stays at one question at a time.

**Prometheus knows the level of a question.** Asked what a conversation's earlier turns do to the model (a repeated wrong answer, a correction or apology, "let's start over", praise or blame, topic bleed, an invented fact followed by others, a persona along a long answer), it no longer operates the window or runs a weaker version in a view: it says the question needs the conversation's history and writes out the `attractor_lab` call as a JSON block for an MCP client. Measured on the isolated tester (Qwen3-Coder 30B): before the rule, 16 calls across three views and no answer; with it, 2 calls and the route named.

**Manual.** Prometheus › *What the window answers, and what needs an agent*; the Attractor lab's agent parameters.

**Case studies.** Nine posts for the Rut Watcher (persona 30), measured with this release: whose words hold a model in a rut, whether a correction or an apology takes, praise and blame, topic bleed, a persona along an answer, invention breeding invention, the timing of a correction, one-word answers.

## 0.2.0 · 2026-10-24

**Attractor lab** (Look inside › Attractor lab; MCP `attractor_lab`): a concept as an order parameter. Build a direction from contrasting texts (difference of means per layer; threshold and direction fitted on the even pairs, scored on the odd ones held out, beside a random direction of the same size); read it on any text; funnel (where the answer becomes predictable along the answer and through the layers); path dependency (the direction swept up and down in one conversation with the model's own answers as history, against fresh conversations and a random direction; every order of the context pieces); constraint (free vs framed: next-piece entropy, active neurons, participation-ratio dimensionality, susceptibility as KL in nats beside a random direction). Steering lasts for the request only.

**A second model to test with, on this computer** (Settings › Prometheus › *Run it on a tester model on this computer*; `tools/tester-model.sh`, `tools/tester-docker-compose.yml`): any OpenAI-compatible server on a loopback address, named `local/<endpoint>/<model>`, usable wherever a model id is asked for. The default tester is a dedicated Ollama on 127.0.0.1:11435 with its own model folder, kept apart from PSI's own llama-server, which is refused as a tester. Only loopback URLs are accepted, redirects are refused, nothing is spent (`max_usd` 0 allowed), and every result says which server answered and whether it is the isolated tester.

**Apple Silicon.** The word-map PCA runs on the CPU (`pca_lowrank` never returns on MPS, PyTorch 2.14.1); a trace engine that does not come up in 10 minutes is killed so the next request starts a fresh one; `understanding_tests` with no items runs the built-in battery; fingerprint reads a mark on tokenizers that merge across the prompt boundary; a missing checkpoint folder gives a clear error. The Context lab's answer odds also score bold phrasings.

**Review fixes before release.** Held-out accuracy no longer fits its threshold on the held-out pairs; builds need four pairs; the order test reports distinct papers; MCP envelopes name the model's source (this computer, the ChatGPT plan, or OpenRouter); the Docker image is pinned; the in-memory direction store is bounded.

**Also new in the public repository with this release** (merged into the private master after 0.1.0): the **Context lab** (the context as the variable: every combination of up to four role-tagged pieces, what each piece did, two papers along one answer, the paths; MCP `context_lab`) and the **Override map** (whether an answer was formed inside or made up, read from the model's state before it writes; pushback; the quantization ladder).

**Manual.** Attractor lab; the tester; the needs table.

## 0.1.0 · 2026-10-17

First public release (the private repository's 0.1.0, dated 2026-10-07, published on GitHub on 2026-10-17): the views as the manual of that release lists them, Prometheus, the MCP server, Commons, Seed & lineage and Hugging Face.
