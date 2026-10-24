# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Repairs a tokenizer that transformers rebuilt from a GGUF file, so the inside engine reads prompts exactly as the
model was trained to. Two faults seen with Qwen3 (transformers 5.3, gguf_file=...):

  - control and user-defined tokens other than the first few (<think>, </think>, <tool_call>, ...) are left out of the
    added tokens, so the text splitter cuts "<think>" into "<", "think", ">": every Qwen3 chat prompt changed
    (the lab's Q8_0 copy, ITERATIONS 13: 5,120 of 5,120 prompts differed from the official tokenizer; 0 after this);
  - the end-of-turn token is set to <|endoftext|> though the GGUF declares <|im_end|>.

The GGUF's own token-type table (3 control, 4 user-defined) and eos id are the source; ids never change. Idempotent.

  python gguf_tok.py <file.gguf> <converted folder>      prints {"added": n, "eos": "..."}
"""
import json, os, sys

KEYS = ("content", "single_word", "lstrip", "rstrip", "normalized", "special")


def repair(gguf_path, out):
    import gguf
    r = gguf.GGUFReader(gguf_path)
    fld, tf = r.fields.get("tokenizer.ggml.tokens"), r.fields.get("tokenizer.ggml.token_type")
    if not fld or not tf: return {"added": 0, "eos": None, "note": "no token table in the GGUF"}
    toks = [bytes(fld.parts[i]).decode("utf-8", "replace") for i in fld.data]
    types = [int(tf.parts[i][0]) for i in tf.data]
    p = os.path.join(out, "tokenizer.json"); tj = json.load(open(p, encoding="utf-8"))
    have = {a["id"] for a in tj["added_tokens"]}
    add = [{"id": i, "content": toks[i], "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": types[i] == 3}
           for i in range(len(toks)) if types[i] in (3, 4) and i not in have]
    if add:
        tj["added_tokens"] = sorted(tj["added_tokens"] + add, key=lambda a: a["id"])
        json.dump(tj, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    eos = None; cp = os.path.join(out, "tokenizer_config.json")
    if os.path.exists(cp):
        c = json.load(open(cp, encoding="utf-8"))
        # transformers reads this list as complete: it must carry every added token, not only the new ones
        c["added_tokens_decoder"] = {str(a["id"]): {k: a[k] for k in KEYS} for a in tj["added_tokens"]}
        e = r.fields.get("tokenizer.ggml.eos_token_id")
        if e is not None:
            eid = int(e.parts[e.data[0]][0])
            if 0 <= eid < len(toks): eos = c["eos_token"] = toks[eid]
        json.dump(c, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return {"added": len(add), "eos": eos}


if __name__ == "__main__":
    print(json.dumps(repair(sys.argv[1], sys.argv[2])))
