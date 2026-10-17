# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Convert a Hugging Face model folder to one GGUF file with llama.cpp's own converter (convert_hf_to_gguf.py),
for Settings > Local model's "Convert to GGUF" (src/convert-api.js). The converter does the work; this wrapper only
adds what the app needs around it:

  * progress as JSON lines on stdout: {"stage":"write","done":k,"of":N} after each tensor written;
  * a plain-words refusal for an architecture the converter does not know, before anything is written;
  * keys of config.json that a newer `transformers` renamed (e.g. GPT-NeoX's rotary_pct) are put back beside the new
    names, so the converter finds the names it expects;
  * the last line is {"stage":"done", ...} or {"error": "..."}.

  python to_gguf.py --converter <convert_hf_to_gguf.py> --outfile <file> --outtype q8_0|f16|bf16|f32 -- <model folder>

The model folder comes last, after "--", so it is never read as an option. Runs on the CPU (the caller sets
CUDA_VISIBLE_DEVICES=-1).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def say(**o):
    print(json.dumps(o), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--converter', required=True)
    ap.add_argument('--outfile', required=True)
    ap.add_argument('--outtype', required=True, choices=['q8_0', 'f16', 'bf16', 'f32'])
    ap.add_argument('model')
    a = ap.parse_args()

    conv = Path(a.converter).resolve()
    model = Path(a.model).resolve()
    if not conv.is_file():
        say(error=f"llama.cpp's converter was not found at {conv}")
        return 2
    if not (model / 'config.json').is_file():
        say(error=f'{model} has no config.json: not a Hugging Face model folder')
        return 2
    raw = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    archs = raw.get('architectures') or []
    arch = archs[0] if archs else None

    # The converter puts its own gguf-py first on the path when imported (unless NO_LOCAL_GGUF is set).
    sys.path.insert(0, str(conv.parent))
    sys.argv = [str(conv), str(model), '--outfile', a.outfile, '--outtype', a.outtype]
    errors: list[str] = []

    class Keep(logging.Handler):
        def emit(self, rec):
            if rec.levelno >= logging.ERROR:
                errors.append(rec.getMessage())

    # A model whose vocabulary is a SentencePiece file needs the package; say so in plain words before starting.
    try:
        import sentencepiece  # noqa: F401
        have_sp = True
    except ImportError:
        have_sp = False
    if not have_sp and (model / 'tokenizer.model').is_file():
        say(error="this model's vocabulary is a SentencePiece file (tokenizer.model); the converter needs the Python package for it: pip install sentencepiece")
        return 2

    try:
        import convert_hf_to_gguf as C  # noqa: E402  (module level only; main() is called below)
        import conversion  # the converter's package of model classes
    except Exception as e:  # a broken llama.cpp checkout, or torch missing
        say(error=f"llama.cpp's converter could not start: {type(e).__name__}: {e}")
        return 2
    logging.getLogger().addHandler(Keep())

    # The converter imports sentencepiece before it looks for tokenizer.model, and falls back to the BPE vocabulary
    # (vocab.json + merges.txt) only on FileNotFoundError, so without the package a BPE model (Granite, Qwen,
    # Llama 3...) failed on the import. The file check now comes first.
    base = getattr(conversion, 'base', None) or sys.modules.get('conversion.base')
    TM = getattr(base, 'TextModel', None) or getattr(C, 'TextModel', None)
    sp_orig = getattr(TM, '_create_vocab_sentencepiece', None)
    if sp_orig is not None and not have_sp:
        def _create_vocab_sentencepiece(self, *args, **kw):
            if not (Path(self.dir_model) / 'tokenizer.model').is_file():
                raise FileNotFoundError(f"File not found: {Path(self.dir_model) / 'tokenizer.model'}")
            return sp_orig(self, *args, **kw)
        TM._create_vocab_sentencepiece = _create_vocab_sentencepiece

    text_map = getattr(conversion, 'TEXT_MODEL_MAP', None)
    if arch is None:
        say(error='config.json names no architecture, so the converter cannot tell what model this is', unsupported=None)
        return 3
    if isinstance(text_map, dict) and arch not in text_map:
        say(error=f"llama.cpp's converter does not support {arch} yet", unsupported=arch)
        return 3

    # Keys of config.json that AutoConfig (a newer transformers) renamed or dropped go back in beside the new ones.
    MB = C.ModelBase
    orig = MB.load_hparams
    restored: list[str] = []

    def load_hparams(dir_model, is_mistral_format, guess=True):
        cfg = orig(dir_model, is_mistral_format, guess)
        try:
            if isinstance(cfg, dict) and not is_mistral_format:
                r = json.loads((Path(dir_model) / 'config.json').read_text(encoding='utf-8'))
                for k, v in r.items():
                    if k not in cfg:
                        cfg[k] = v
                        if k not in restored:
                            restored.append(k)
        except Exception:
            pass
        return cfg
    MB.load_hparams = staticmethod(load_hparams)

    # Progress: count each tensor as it is written (the converter's writer walks its tensor dicts in order).
    W = C.gguf.GGUFWriter
    orig_ti = W.write_ti_data_to_file
    state = {'done': 0, 'of': 0, 't': 0.0}

    class Counting(dict):
        def items(self):
            for kv in super().items():
                yield kv
                state['done'] += 1
                now = time.time()
                if now - state['t'] > 0.25 or state['done'] == state['of']:
                    state['t'] = now
                    say(stage='write', done=state['done'], of=state['of'])

    def write_ti_data_to_file(self):
        orig_ti(self)   # the header walks the tensors too: count only the writing that follows
        state['of'] = sum(len(t) for t in self.tensors)
        say(stage='write', done=0, of=state['of'])
        self.tensors = [Counting(t) for t in self.tensors]
    W.write_ti_data_to_file = write_ti_data_to_file

    say(stage='prepare', arch=arch, outtype=a.outtype)
    t0 = time.time()
    try:
        C.main()
    except SystemExit as e:
        if e.code not in (0, None):
            say(error=(errors[-1] if errors else f'the converter stopped (code {e.code})'), arch=arch)
            return 1
    except KeyError as e:
        say(error=f"llama.cpp's converter could not read {arch}: config.json has no {e} (a setting it needs)", arch=arch)
        return 1
    except NotImplementedError as e:
        say(error=f"llama.cpp's converter does not support part of {arch} yet: {e}", arch=arch)
        return 1
    except Exception as e:
        say(error=f'{type(e).__name__}: {e}', arch=arch)
        return 1
    out = Path(a.outfile)
    if not out.is_file():
        say(error=f'the converter finished but wrote no file at {out}', arch=arch)
        return 1
    say(stage='done', file=str(out), bytes=out.stat().st_size, tensors=state['of'], arch=arch, outtype=a.outtype,
        seconds=round(time.time() - t0, 1), restored_keys=restored)
    return 0


if __name__ == '__main__':
    sys.exit(main())
