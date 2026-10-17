# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""Which accelerator the PyTorch engines run on, and the few things that differ between them.

    cuda   NVIDIA (CUDA), or AMD with a ROCm build of PyTorch (ROCm answers as "cuda"; torch.version.hip tells them apart)
    mps    Apple Silicon (Metal): one memory shared with the CPU, so "free" is what macOS lets the GPU take, less what it holds
    xpu    Intel GPUs (Arc, Data Center) with a PyTorch that has XPU
    cpu    anything else, or when the model does not fit; float32

PSI_DEVICE=cpu|cuda|mps|xpu forces one (it must be available). Every engine asks pick(need_gb): the accelerator when it
has room for the model, else the CPU, and the reason is kept (why()). Nothing here assumes a vendor.
"""
import os

import torch

GB = 2 ** 30


def _has(name):
    try:
        if name == "cuda": return torch.cuda.is_available()
        if name == "mps": return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if name == "xpu": return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception: return False
    return name == "cpu"


def accelerator():
    """The accelerator PyTorch can use here (cuda, mps, xpu), or None."""
    want = (os.environ.get("PSI_DEVICE") or "").strip().lower()
    if want:
        base = want.split(":")[0]
        return want if base != "cpu" and _has(base) else None
    for d in ("cuda", "xpu", "mps"):
        if _has(d): return d
    return None


def kind(device):
    return str(device).split(":")[0]


def vendor(device):
    k = kind(device)
    if k == "cuda": return "amd" if getattr(torch.version, "hip", None) else "nvidia"
    return {"mps": "apple", "xpu": "intel"}.get(k)


def name(device):
    k = kind(device)
    try:
        if k == "cuda": return torch.cuda.get_device_name(_index(device))
        if k == "xpu": return torch.xpu.get_device_name(_index(device))
        if k == "mps":
            import platform, subprocess
            try: chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception: chip = platform.processor()
            return f"{chip or 'Apple Silicon'} GPU (unified memory)"
    except Exception: pass
    return None


def _index(device):
    s = str(device)
    if ":" in s: return int(s.split(":")[1])
    k = kind(s)
    try:
        if k == "cuda": return torch.cuda.current_device()
        if k == "xpu": return torch.xpu.current_device()
    except Exception: pass
    return 0


def _system_ram():
    try: return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception: return None


def _smi_free(i):
    """Free bytes on CUDA device i as nvidia-smi reports them, or None (no nvidia-smi, AMD, or any failure)."""
    if getattr(torch.version, "hip", None): return None
    try:
        import subprocess
        u = str(getattr(torch.cuda.get_device_properties(i), "uuid", "") or "")   # the same card whatever the index order
        sel = (u if u.startswith("GPU-") else f"GPU-{u}") if u else str(i)
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", sel], capture_output=True, text=True,
                           timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(r.stdout.strip().splitlines()[0]) * 2 ** 20 if r.returncode == 0 and r.stdout.strip() else None
    except Exception: return None


def _memory(device):
    """(free, total, source) bytes: memory() with where the free figure came from."""
    k = kind(device)
    try:
        if k == "cuda":
            # On Windows (WDDM) PyTorch's free figure can miss what other processes hold (llama-server, the desktop), so
            # nvidia-smi is asked too and the smaller figure counts.
            i = _index(device); f, t = torch.cuda.mem_get_info(i); s = _smi_free(i)
            if s is None: return f, t, "PyTorch (nvidia-smi not available)" if not getattr(torch.version, "hip", None) else "PyTorch"
            return (s, t, f"nvidia-smi; PyTorch said {f / GB:.1f}") if s < f else (f, t, f"PyTorch; nvidia-smi said {s / GB:.1f}")
        else: f, t = memory(device); return f, t, "PyTorch"
    except Exception: pass
    return None, None, None


def memory(device):
    """(free, total) bytes the device can give a model now, or (None, None) when it cannot tell."""
    k = kind(device)
    try:
        if k == "cuda": return _memory(device)[:2]
        if k == "xpu":
            i = _index(device)
            if hasattr(torch.xpu, "mem_get_info"): return tuple(torch.xpu.mem_get_info(i))
            total = torch.xpu.get_device_properties(i).total_memory
            return total - torch.xpu.memory_reserved(i), total
        if k == "mps":
            # Unified memory: macOS caps what the GPU may take (recommended_max_memory); what MPS holds already is used.
            total = torch.mps.recommended_max_memory() if hasattr(torch.mps, "recommended_max_memory") else int((_system_ram() or 0) * 0.65)
            held = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") else 0
            return max(total - held, 0), total
    except Exception: pass
    return None, None


_WHY = {"device": "cpu", "why": "no accelerator"}
MARGIN_GB = 0.75   # kept free beyond the model's estimate: overfilling a GPU spills into shared memory and stalls everything


def pick(need_gb):
    """The device for a model that needs need_gb: the accelerator when it has room, else the CPU (why() says which)."""
    global _WHY
    acc = accelerator()
    if (os.environ.get("PSI_DEVICE") or "").strip().lower() == "cpu":
        _WHY = {"device": "cpu", "why": "PSI_DEVICE=cpu"}; return "cpu"
    if not acc:
        _WHY = {"device": "cpu", "why": "no GPU that PyTorch can use here (CUDA, ROCm, MPS or XPU)"}; return "cpu"
    free, total, src = _memory(acc)
    if free is None:   # cannot tell: try it
        _WHY = {"device": acc, "why": "free memory unknown; trying the GPU"}; return acc
    if need_gb is None or free / GB > need_gb + MARGIN_GB:
        _WHY = {"device": acc, "why": f"{free / GB:.1f} GB free of {total / GB:.1f} GB ({src})", "free_gb": round(free / GB, 2)}; return acc
    _WHY = {"device": "cpu", "why": f"needs {need_gb:.1f} GB (+{MARGIN_GB} GB margin), {acc} has {free / GB:.1f} GB free ({src})", "free_gb": round(free / GB, 2)}
    return "cpu"


def why(): return dict(_WHY)


def free_gb(device=None):
    f, _ = memory(device or accelerator() or "cpu")
    return round(f / GB, 2) if f is not None else 0.0


def dtype(device):
    """bfloat16 where the device does it, else float16 on a GPU; float32 on the CPU."""
    k = kind(device)
    if k == "cpu": return torch.float32
    try:
        if k == "cuda" and torch.cuda.is_bf16_supported(): return torch.bfloat16
        if k == "xpu" and (not hasattr(torch.xpu, "is_bf16_supported") or torch.xpu.is_bf16_supported()): return torch.bfloat16
        if k == "mps":
            torch.ones(1, dtype=torch.bfloat16, device="mps") + 1   # bfloat16 on MPS needs macOS 14 and a recent PyTorch
            return torch.bfloat16
    except Exception: pass
    return torch.float16


def sync(device):
    k = kind(device)
    try:
        if k == "cuda": torch.cuda.synchronize()
        elif k == "mps": torch.mps.synchronize()
        elif k == "xpu": torch.xpu.synchronize()
    except Exception: pass


def empty_cache(device=None):
    k = kind(device or accelerator() or "cpu")
    try:
        if k == "cuda": torch.cuda.empty_cache()
        elif k == "mps": torch.mps.empty_cache()
        elif k == "xpu": torch.xpu.empty_cache()
    except Exception: pass


def reset_peak(device):
    k = kind(device)
    try:
        if k == "cuda": torch.cuda.reset_peak_memory_stats()
        elif k == "xpu": torch.xpu.reset_peak_memory_stats()
    except Exception: pass


def peak_bytes(device):
    """The most memory this process held on the device since reset_peak (MPS has no peak counter: what it holds now)."""
    k = kind(device)
    try:
        if k == "cuda": return torch.cuda.max_memory_allocated()
        if k == "xpu": return torch.xpu.max_memory_allocated()
        if k == "mps": return torch.mps.driver_allocated_memory()
    except Exception: pass
    return None


def total_bytes(device):
    if kind(device) == "cuda":   # no free figure needed, so no nvidia-smi call
        try: return torch.cuda.get_device_properties(_index(device)).total_memory
        except Exception: pass
    return memory(device)[1]


def report():
    """What this machine offers PyTorch, for the setup check."""
    acc = accelerator()
    out = {"torch": torch.__version__, "accelerator": acc, "vendor": vendor(acc) if acc else None, "name": name(acc) if acc else None,
           "rocm": getattr(torch.version, "hip", None), "cuda": getattr(torch.version, "cuda", None),
           "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
           "xpu_built": bool(getattr(torch, "xpu", None))}
    if acc:
        f, t = memory(acc); out.update(free_gb=round(f / GB, 2) if f is not None else None, total_gb=round(t / GB, 2) if t is not None else None,
                                       dtype=str(dtype(acc)).replace("torch.", ""))
    return out


if __name__ == "__main__":
    import json; print(json.dumps(report()))
