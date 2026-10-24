# Context lab (exploration, branch `context-lab`)

The model stays fixed; the context is the variable. Built 2026-10-23 from the needs of the site article "The Exam
Paper" (emergencemachine.com, draft), whose recordings had to be made with scripts outside PSI.

## What it does

A question plus up to four pieces of context, each with a role: `system` (the app's hidden instructions, sent as the
system message), `premise`, `claim`, `evidence` (before the question), `instruction` (after it).

- **run**: every combination of pieces (2^n papers) through the model's own chat template, thinking off. Per paper:
  the text read, top next pieces after a fixed `frame`, odds of named answers there (whole phrase, best phrasing,
  shares among the answers named) and the greedy answer (unframed). Per piece: `moved` (1 - overlap of the full
  next-piece distributions between papers that differ only in this piece, averaged), effect on each answer, flips of
  the leading answer. Per pair: interaction (p11 - p10 - p01 + p00, averaged).
- **diverge**: two papers continue the same answer text; per answer piece, the overlap of their next-piece odds.
  Shows where the paper stops mattering.
- **tree**: top `show` next pieces at every node, top `expand` followed `depth` steps, greedy finish per leaf.

Where: `src/context_lab.py` (inside engine, `/context`), `src/ui/context.js` (view `#context`, "Look inside"),
MCP tool `context_lab` (strict arguments), help.html `#context`. The view saves a run in the site's exam-paper
format (`window.E8_CONTEXT.scenarios[i]`).

## Checks done

- Standalone (OLMo2-1B, sdpa attention): the "against" scenario reproduces the site recording
  (`record_bench.py`) to four decimals in all 8 papers; the tree root matches `record_paths.py`.
- Through the app (test server on 4399, inside engine on 4312, OLMo2-1B): MCP `tools/list` includes it; a run, the
  unknown-argument refusal, the view (run, compare along the answer, walk the paths) all work, no console errors.
- The inside engine loads with `attn_implementation="eager"`; its odds differ slightly from sdpa recordings
  (instructions paper 10: 0.896 vs 0.917). Verified to be the attention implementation alone. Recordings for the site
  should state which.

## Gaps found in use (2026-10-23)

The lab's code ran two experiments outside PSI (psi-override-lab: `PREREG_CONTEXT_PATHS*.md`, `RESULTS_CONTEXT_PATHS*.md`;
Qwen3-4B under 16 trivial rewordings × 6 weight levels, then the same for OLMo2-1B, SmolLM2-1.7B, Granite-3.3-2B,
Phi-3.5-mini; 5,700 papers). What held up and what did not, with the finding each gap cost:

Good: the framed, whole-phrase, renormalised odds are a clean decision instrument (repeat control identical in 10/10 at
every level of every model; sdpa reproduces the site recordings to four decimals); one paper = one rendered chat was
the right unit; tree and diverge answered the article's questions directly.

1. **It only sees the answers you name.** The hedge list missed "is not yet determined" (one item misread at every
   level); the pushback frame ("The capital of Australia is") missed caving that happens *before* the frame ("You're
   correct! … Sydney"; "Sydney is the capital city"). Found by reading answers by hand. → An answer-kind instrument on
   the greedy answer (and N samples): answers / hedges / declines / agrees / states the claim / corrects / leaks /
   echoes the question; report decisions three ways and flag disagreement.
2. **A stability number with no text check.** Q2_K Qwen scored 0.91 "decisions kept" while repeating the question and
   leaking "Assistant:"; Q2_K Phi scored 0.82 while its refusal to predict the future had gone from 5/5 to 0/5. →
   text-health checks attached to every answer; "stable" is reported beside "same habits as bf16".
3. **One model, one paper set, no weight axis.** The cross-lab run, the quantization ladder, the GGUF→disk prep, the
   intervals and family tables all lived in scripts. → a sweep job (load → run set → unload → next) over models and
   weight variants; the streaming GGUF→bf16 prep with a per-architecture round-trip check as a PSI tool (never
   `gguf_file=` in RAM on a 64 GB machine).
4. **Controls run after the grid, not before.** The meaning-change control was failing for pushback from the start
   (the framed measure cannot see the claim's effect) and the 1,140-paper run went ahead. → controls first; refuse
   when the positive control is invisible to the chosen instrument; predictions stored in the run and scored at the
   end.
5. **Margin shown as the confidence number.** The most one-sided paper (p 0.998) flipped under 5 of 16 rewordings. →
   neighbourhood stability (share of K rewordings keeping the decision) as the headline; margin secondary.
6. **No provenance fields.** Attention implementation moved odds 0.896 → 0.917; dtype, weight source, template flags are
   not in the record. → model path + fingerprint, dtype, attn impl, template flags, weight variant, lab version in
   every record; signed.
7. **No resource preflight.** One run was killed for RAM (GGUF dequant in memory), one stopped on a full disk; both
   predictable from file sizes. → estimate RAM and disk before starting.

Smaller: `tree`/`diverge` with a second model side by side ("same paper, five minds"); `moved` between models, not only
between papers; the perturbation families as a built-in generator with the fixes found today (typo needs a 5-letter
lowercase word; greeting goes at the front; keep "I'm").

## Not built yet (from the gap list)

1. Several models: the inside engine holds one model. A model sweep (load, run, unload, next) as a job, for size and
   family comparisons (the site's "Answers are paths" result needs a same-family check: Qwen3-0.6B/1.7B/4B).
2. Sets: one paper set per run. A question set with wording variants (playbook rule: sets, not single runs), reporting
   effects with intervals.
3. Signed records for the site export (the lineage signing used elsewhere), so a replayed recording is verifiable.
4. Option-free summaries: without named answers only `moved` is reported; a classifier of answer kinds (hedge vs
   answer, using override.HEDGE) would give flips there too.
5. More than four pieces (16 papers) would need a fractional design, not the full factorial.
