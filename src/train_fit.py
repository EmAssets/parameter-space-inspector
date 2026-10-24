# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""What fits here: can this model be trained on this GPU, and how? An estimate from the model's shape, and a short
measured run that checks it (1 warm-up step and 3 timed optimizer steps; nothing is saved, the weights on disk are
not touched).

  python train_fit.py estimate <model dir> [--free-gb G] [--seq 1024]
  python train_fit.py measure  <model dir> --method full|full8|lora|qlora [--seq 1024] [--cap-gb G]

Methods (batch 1, the given sequence length, gradient checkpointing on):
  full    every weight trains: fp32 master weights, bf16 autocast, AdamW            16 bytes per weight
  full8   every weight trains: bf16 weights and gradients, 8-bit AdamW              6 bytes per weight
  lora    the weights stay bf16 and frozen; rank-16 adapters on every projection train
  qlora   the same, with the frozen weights in 4-bit (NF4); embeddings and head stay bf16

The estimate (written before any measurement, PREREG_ADJACENT E2): weights and optimizer as above; activations kept by
checkpointing (layers x seq x width x 2 bytes), one layer live (seq x (12 x width + 4 x MLP width) x 2 bytes), and the
loss over the vocabulary (seq x vocab x 10 bytes: bf16 logits, their fp32 copy, its gradient). Checked against
measured runs on five models (PREREG_ADJACENT E2): full, 8-bit and LoRA within 13%; 4-bit took 1.3 bytes per quantized
weight, not the 0.56 of the 4-bit format (fp32 copies and dequantisation buffers), so that is the figure used. Speed is
not estimated: a formula from the GPU's peak missed by up to 5x (small models leave the GPU idle; layer count and
vocabulary matter as much as size), so the card shows speed only once measured.

The measured run caps PyTorch at the GPU's free memory (--cap-gb), so a method that does not fit fails with "out of
memory" instead of spilling into system RAM (which Windows drivers do, slowly). Output: one JSON line.
"""
import argparse, glob, json, os, struct, sys, time

GB = 2 ** 30
QLORA_BYTES = 1.3   # per quantized weight, measured (median of five models, 1.21-1.46)
METHODS = ("full", "full8", "lora", "qlora")


def shape(d):
    c = json.load(open(os.path.join(d, "config.json"), encoding="utf-8")); t = c.get("text_config", c)
    n = emb = 0
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with open(f, "rb") as fh:
            h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in h.items():
            if k == "__metadata__": continue
            m = 1
            for s in v["shape"]: m *= s
            n += m
            if "embed" in k or "lm_head" in k or "wte" in k or "embed_out" in k: emb += m
    return {"params": n, "embed_params": emb, "hidden": t.get("hidden_size"), "layers": t.get("num_hidden_layers"),
            "ffn": t.get("intermediate_size") or 4 * t.get("hidden_size"), "vocab": t.get("vocab_size"), "model_type": t.get("model_type")}


def lora_params(s, r=16):   # rank-16 adapters on q, k, v, o and the MLP's three projections (an upper bound for k/v)
    h, f = s["hidden"], s["ffn"]; return s["layers"] * r * (4 * 2 * h + 3 * (h + f))


def estimate(d, free_gb=None, seq=1024, gpu=""):
    s = shape(d); P, E = s["params"], s["embed_params"]; h, L, f, V = s["hidden"], s["layers"], s["ffn"], s["vocab"]
    act = (L * seq * h * 2 + seq * (12 * h + 4 * f) * 2 + seq * V * 10)
    A = lora_params(s)
    weights = {"full": P * 16, "full8": P * 6, "lora": P * 2 + A * 16, "qlora": (P - E) * QLORA_BYTES + E * 2 + A * 16}
    out = {"model": os.path.basename(d.rstrip("/\\")), **s, "seq": seq, "gpu": gpu or None, "free_gb": free_gb, "methods": {}}
    for m in METHODS:
        gb = (weights[m] + act) / GB
        out["methods"][m] = {"gb": round(gb, 2), "fits": (gb + 0.6 <= free_gb) if free_gb else None}
    return out


def text_ids(tok, seq):
    here = os.path.dirname(os.path.abspath(__file__))
    t = " ".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(here, "..", "research", "results", "heldout-*.txt"))))
    ids = tok(t, add_special_tokens=False).input_ids
    while len(ids) < seq + 1: ids = ids + ids
    return ids[: seq + 1]


def measure(d, method, seq=1024, cap_gb=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    if cap_gb: torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * GB / torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(d)
    ids = torch.tensor([text_ids(tok, seq)], device=dev)
    t0 = time.time(); res = {"model": os.path.basename(d.rstrip("/\\")), "method": method, "seq": seq, "cap_gb": cap_gb, "gpu": torch.cuda.get_device_name(0)}
    try:
        if method == "qlora":
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(d, quantization_config=q, device_map=dev, dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32 if method == "full" else torch.bfloat16, device_map=dev)
        model.config.use_cache = False; model.train()   # checkpointing only acts in training mode (from_pretrained returns eval)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if method in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if method == "qlora": model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            else: model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
        train = [p for p in model.parameters() if p.requires_grad]
        if method == "full8":
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(train, lr=1e-5)
        else: opt = torch.optim.AdamW(train, lr=1e-5 if method == "full" else 2e-4)
        res["trainable"] = sum(p.numel() for p in train)
        times = []
        for step in range(4):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=method == "full"):
                loss = model(input_ids=ids[:, :-1], labels=ids[:, :-1]).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.time() - t)
        res.update({"ok": True, "peak_gb": round(torch.cuda.max_memory_allocated() / GB, 2), "reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 2),
                    "tokens_per_s": round(seq / (sum(times[1:]) / 3)), "loss": round(float(loss), 3)})
    except torch.cuda.OutOfMemoryError as e:
        res.update({"ok": False, "out_of_memory": True, "peak_gb": round(torch.cuda.max_memory_allocated() / GB, 2), "error": str(e).splitlines()[0][:200]})
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["estimate", "measure"]); ap.add_argument("model")
    ap.add_argument("--method", choices=METHODS); ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--free-gb", type=float); ap.add_argument("--cap-gb", type=float); ap.add_argument("--gpu", default=""); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "estimate": print(json.dumps(estimate(a.model, a.free_gb, a.seq, a.gpu)))
    else:
        r = measure(a.model, a.method, a.seq, a.cap_gb)
        if a.out:   # from the app: the result is kept for the card, and the job runner reads PSI-DONE
            r["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1); print("PSI-DONE " + json.dumps(r), flush=True)
        else: print(json.dumps(r))
