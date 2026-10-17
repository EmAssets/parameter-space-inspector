"""
O. The null for structural surprise: what the Sensing surface's anomaly score looks like on ordinary text.

The tab ranks a text's sentences by structural surprise (src/sensing.py: bits given the whole text minus
bits given a 3-word window). A rank is relative; whether rank 1 is anything at all needs a reference:
the same score on ordinary prose with nothing to flag. This script reads ~200 sentences of plain text
with the loaded model exactly as the tab does (each sentence inside its own paragraph), and stores the
sentence scores so a live sentence can be given a percentile and a "beyond ordinary text" mark
(above the 95th percentile of the null). Until it has been run for a model, the tab says
"calibration not yet measured" and shows ranks only.

Needs the GPU model (Qwen3-4B takes ~8 GB). Writes .userdata/sensing-null/<model>.json (PSI_SENSING_NULL_DIR
overrides the folder). The text source is recorded in the file: pass --text with plain-text files of
ordinary prose (news, encyclopaedia, manuals); the default is the held-out essay paragraphs that ship
with the research notes, which are argumentative prose and a weaker null, and the file says so.

  python research/sensing_null.py --model <models>/qwen3-4b-bf16 --text "<corpus>/*.txt" --n 200
"""
import argparse, glob, os, sys, threading, time
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
MODELS = os.environ.get("PSI_MODELS_DIR", "models")   # where the default model folders below live
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))


class _Tracer:
    """The three things src/sensing.py reads from the inside engine: tok, model, device (and a lock)."""
    def __init__(self, path, dtype, device="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.lock = threading.Lock()


def paragraphs(patterns, per=5):
    """Blank-line paragraphs; a block longer than `per` sentences (a file with no paragraph breaks, like the
    held-out essays, one line each) is cut into runs of `per` sentences, so each reads as a paragraph."""
    import sensing
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            txt = open(f, encoding="utf-8", errors="replace").read()
            for p in txt.split("\n\n"):
                sents = sensing._sentences(p)
                for i in range(0, len(sents), per):
                    chunk = " ".join(sents[i:i + per])
                    if len(chunk.split()) >= 25: out.append(chunk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{MODELS}/qwen3-4b-bf16"); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--text", nargs="*", default=[os.path.join(HERE, "results", "heldout-*.txt")], help="plain-text files (globs) of ordinary prose")
    ap.add_argument("--n", type=int, default=200, help="sentences in the null"); ap.add_argument("--window", type=int, default=3); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import sensing
    paras = paragraphs(a.text)
    if not paras: sys.exit("no paragraphs found: pass --text with plain-text files")
    default_src = a.text == [os.path.join(HERE, "results", "heldout-*.txt")]
    source = ("held-out essay paragraphs (research/results/heldout-*.txt): argumentative prose, a weaker null than plain text" if default_src
              else "; ".join(a.text))
    t0 = time.time(); T = _Tracer(a.model, a.dtype, a.device); S = sensing.Sensing(None, T)
    print(f"{S._name()} on {T.device}: {len(paras)} paragraphs, up to {a.n} sentences, window {a.window}", flush=True)
    d = S.calibrate(paras, n_sentences=a.n, window=a.window, out=a.out, source=source)
    print(f"wrote {a.out or S.null_path()}: {d['n_sentences']} sentences from {d['n_paragraphs']} paragraphs, median {d['scores'][len(d['scores']) // 2]} bits, "
          f"95th pct {sorted(d['scores'])[int(0.95 * (len(d['scores']) - 1))]} bits ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
