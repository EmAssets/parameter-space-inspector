# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""A content fingerprint of a weights folder, for the one place where "these exact weights" must hold: matching an
Override-map probe to the model it was measured on.

The weights hash (trace_server._weights_hash) reads only the index file or the first 1 MB of a single file: names,
shapes and the first tensor. A fine-tune keeps all three. A LoRA merge of SmolLM2-1.7B kept the first 1 MB
(the header and the untouched embedding table) byte for byte, so the original's probe scored the tuned copy
(case 27 of the Override playbook). This adds, per weights file: its size, its header, and 256 samples of 16 KB spread
evenly over its data, so any changed tensor larger than 1/256 of a file changes the fingerprint (every LoRA merge
changes all attention and MLP matrices). About 4 MB read per file.

  python weights_fp.py <folder>      prints {"weights_fp": "...", "weights_fp_of": "..."}
"""
import hashlib, json, os, struct, sys

SAMPLES, CHUNK = 256, 16 << 10


def fingerprint(path):
    files = sorted(n for n in os.listdir(path) if n.endswith(".safetensors"))
    if not files: return None, None
    h = hashlib.sha256()
    for n in files:
        p = os.path.join(path, n); size = os.path.getsize(p)
        with open(p, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]; head = f.read(hl)
            h.update(n.encode()); h.update(struct.pack("<Q", size)); h.update(head)
            start, span = 8 + hl, max(0, size - 8 - hl - CHUNK)
            for i in range(SAMPLES):
                f.seek(start + (span * i) // max(1, SAMPLES - 1)); h.update(f.read(CHUNK))
    return h.hexdigest(), f"size, header and {SAMPLES} samples of {CHUNK >> 10} KB of each of {len(files)} .safetensors file(s)"


if __name__ == "__main__":
    fp, of = fingerprint(sys.argv[1]); print(json.dumps({"weights_fp": fp, "weights_fp_of": of}))
