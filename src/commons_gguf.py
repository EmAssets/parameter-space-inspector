# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""A built stories260K-line model as a GGUF file, so PSI's answering engine (llama.cpp) can run it.

llama2.c keeps the original Llama layout (rotary on interleaved pairs), which is what GGUF's llama architecture
expects, so no weight is permuted. The 512-piece vocabulary goes in as a SentencePiece-style vocabulary: spaces
written as U+2581, <0xXX> pieces typed as bytes, 0-2 as unknown / begin / end. The output is tied to the input
embedding, so there is no output tensor (llama.cpp then uses token_embd). The model's record rides in the metadata.
"""
import json, os, re
import numpy as np
from gguf import GGUFWriter, TokenType

NAMES = {"attention_norm": "attn_norm", "attention.wq": "attn_q", "attention.wk": "attn_k", "attention.wv": "attn_v", "attention.wo": "attn_output",
         "ffn_norm": "ffn_norm", "feed_forward.w1": "ffn_gate", "feed_forward.w2": "ffn_down", "feed_forward.w3": "ffn_up"}


def gguf_name(k):
    if k == "tok_embeddings.weight": return "token_embd.weight"
    if k == "norm.weight": return "output_norm.weight"
    m = re.match(r"layers\.(\d+)\.(.+)\.weight$", k)
    return f"blk.{m.group(1)}.{NAMES[m.group(2)]}.weight"


def export(model_dir, seed, out=None):
    from safetensors.torch import load_file
    vec = {k: v.float().numpy() for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items()}
    rec = json.load(open(os.path.join(model_dir, "record.json"), encoding="utf-8"))
    a = seed.args; out = out or os.path.join(model_dir, os.path.basename(model_dir.rstrip("/\\")) + ".gguf")
    w = GGUFWriter(out, "llama")
    w.add_name(f"{(rec['record'].get('recipe_name') or 'model')} (stories260K line)")
    w.add_context_length(a["max_seq_len"]); w.add_embedding_length(a["dim"]); w.add_block_count(a["n_layers"]); w.add_feed_forward_length(a["hidden"])
    w.add_head_count(a["n_heads"]); w.add_head_count_kv(a["n_kv_heads"]); w.add_layer_norm_rms_eps(1e-5); w.add_rope_dimension_count(a["dim"] // a["n_heads"])
    w.add_file_type(0)
    toks, scores, types = [], [], []
    for i, (t, s) in enumerate(zip(seed.tok.vocab, seed.tok.scores)):
        if i == 0: toks.append("<unk>"); types.append(TokenType.UNKNOWN)
        elif i == 1: toks.append("<s>"); types.append(TokenType.CONTROL)
        elif i == 2: toks.append("</s>"); types.append(TokenType.CONTROL)
        elif re.fullmatch(r"<0x[0-9A-F]{2}>", t): toks.append(t); types.append(TokenType.BYTE)
        else: toks.append(t.replace(" ", "▁")); types.append(TokenType.NORMAL)
        scores.append(float(s))
    w.add_tokenizer_model("llama"); w.add_token_list(toks); w.add_token_scores(scores); w.add_token_types(types)
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0); w.add_add_bos_token(True); w.add_add_eos_token(False)
    w.add_string("em.commons.record", json.dumps(rec, ensure_ascii=False))
    for k in sorted(vec): w.add_tensor(gguf_name(k), np.ascontiguousarray(vec[k], dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return out
