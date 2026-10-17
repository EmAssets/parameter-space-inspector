"""One-time: unpack a GGUF into plain bf16 weights on disk, so later experiments
load straight to the GPU with almost no RAM (safetensors are memory-mapped).
  python research/convert_bf16.py --gguf <models>/Qwen3-4B-Q8_0.gguf --out <models>/Qwen3-4B-bf16
"""
import argparse, os, sys, time
# Downloads and temporary files go under $PSI_CACHE_DIR when it is set (as in the app); otherwise the system defaults.
if os.environ.get("PSI_CACHE_DIR"):
    _c = os.environ["PSI_CACHE_DIR"]
    os.environ.setdefault("HF_HOME", os.path.join(_c, "huggingface")); os.environ.setdefault("TMP", os.path.join(_c, "tmp")); os.environ.setdefault("TEMP", os.path.join(_c, "tmp"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--gguf", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
folder, name = os.path.split(a.gguf); t0 = time.time()
tok = AutoTokenizer.from_pretrained(folder, gguf_file=name)
model = AutoModelForCausalLM.from_pretrained(folder, gguf_file=name, dtype=torch.bfloat16)
model.save_pretrained(a.out, safe_serialization=True, max_shard_size="2GB"); tok.save_pretrained(a.out)
print(f"saved to {a.out} in {time.time() - t0:.0f}s"); sys.exit(0)
