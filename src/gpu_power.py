# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
"""GPU power during a piece of work, from NVIDIA's management library (NVML), which ships with the driver
(nvml.dll on Windows, libnvidia-ml.so.1 on Linux). Read through ctypes, so nothing extra is installed; anywhere it is
missing (no NVIDIA card, macOS, CPU only) every figure is None and the work runs as before.

    with PowerMeter(torch_device_index) as pm:
        ...work...
    pm.result()  ->  {"peak_w", "avg_w", "idle_w", "energy_j", "energy_wh", "limit_w", "source", "whole_card": True}

The card's own figures: they include everything else on that GPU at the time (the answering engine, the desktop).
Energy is the driver's energy counter when the card has one (Volta and later), else the sampled power integrated.
Peak is the highest of samples every 20 ms, of instantaneous power where the driver reports it (else its 1 s average).
"""
import ctypes, os, sys, threading, time

_lib = None


def _load():
    global _lib
    if _lib is not None: return _lib or None
    names = ["nvml.dll", os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll"),
             os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvml.dll")] \
        if sys.platform == "win32" else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            if lib.nvmlInit_v2() == 0: _lib = lib; return lib
        except Exception: pass
    _lib = False
    return None


class _Field(ctypes.Structure):   # nvmlFieldValue_t
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", ctypes.c_ulonglong)]


FI_POWER_INSTANT = 186   # NVML_FI_DEV_POWER_INSTANT


def _handle(lib, index, pci_bus_id=None):
    h = ctypes.c_void_p()
    if pci_bus_id and lib.nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id.encode(), ctypes.byref(h)) == 0: return h
    if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(h)) == 0: return h
    return None


class PowerMeter:
    def __init__(self, index=0, pci_bus_id=None, every=0.02):
        self.lib = _load(); self.h = _handle(self.lib, index, pci_bus_id) if self.lib else None
        self.every, self.samples, self._stop = every, [], threading.Event()
        self.instant = None; self.e0 = self.e1 = None; self.t0 = self.t1 = None; self.idle = None

    def _energy(self):   # mJ since the driver loaded, or None
        v = ctypes.c_ulonglong()
        return v.value if self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(v)) == 0 else None

    def _power(self):    # W now: instantaneous where the driver has it, else its averaged reading
        if self.instant is not False:
            f = _Field(fieldId=FI_POWER_INSTANT)
            if self.lib.nvmlDeviceGetFieldValues(self.h, 1, ctypes.byref(f)) == 0 and f.nvmlReturn == 0:
                self.instant = True; return f.value / 1000.0
            self.instant = False
        v = ctypes.c_uint()
        return v.value / 1000.0 if self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(v)) == 0 else None

    def _run(self):
        while not self._stop.is_set():
            p = self._power()
            if p is not None: self.samples.append((time.perf_counter(), p))
            self._stop.wait(self.every)

    def __enter__(self):
        if not self.h: return self
        self.idle = self._power(); self.e0 = self._energy(); self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._run, daemon=True); self._th.start()
        return self

    def __exit__(self, *exc):
        if not self.h: return False
        self._stop.set(); self._th.join(1.0)
        self.e1 = self._energy(); self.t1 = time.perf_counter()
        return False

    def _limit(self):    # W, the card's enforced power limit
        lim = ctypes.c_uint()
        return lim.value / 1000.0 if self.lib.nvmlDeviceGetEnforcedPowerLimit(self.h, ctypes.byref(lim)) == 0 else None

    def result(self):
        if not self.h or self.t1 is None: return None
        secs = max(self.t1 - self.t0, 1e-9); ps = [p for _, p in self.samples]
        if self.e0 is not None and self.e1 is not None and self.e1 >= self.e0: energy, how = (self.e1 - self.e0) / 1000.0, "energy counter"
        else:   # integrate the samples (trapezoids)
            energy = sum((t2 - t1) * (p1 + p2) / 2 for (t1, p1), (t2, p2) in zip(self.samples, self.samples[1:])) if len(self.samples) > 1 else None
            how = "sampled power"
        limit = self._limit()
        r = lambda v, n=1: None if v is None else round(v, n)
        return {"peak_w": r(max(ps)) if ps else None, "avg_w": r(energy / secs) if energy is not None else (r(sum(ps) / len(ps)) if ps else None),
                "idle_w": r(self.idle), "energy_j": r(energy), "energy_wh": r(energy / 3600, 4) if energy is not None else None,
                "limit_w": r(limit), "samples": len(ps), "seconds": r(secs, 2), "source": f"NVML ({how}; peak from {'instantaneous' if self.instant else '1 s averaged'} power)",
                "whole_card": True}


class AmdMeter(PowerMeter):
    """AMD GPUs through amdsmi, the Python library that ships with ROCm (Linux). Same figures as PowerMeter: the
    driver's energy counter when it has one, else the sampled power integrated. Field names changed across ROCm
    releases, so each is looked for under its known names; anything missing is None."""
    def __init__(self, index=0, every=0.05):
        self.lib = self.h = None; self.every, self.samples, self._stop = every, [], threading.Event()
        self.instant = None; self.e0 = self.e1 = None; self.t0 = self.t1 = None; self.idle = None
        try:
            import amdsmi
            amdsmi.amdsmi_init(); hs = amdsmi.amdsmi_get_processor_handles()
            if hs: self.lib, self.h = amdsmi, hs[min(index, len(hs) - 1)]
        except Exception: pass

    @staticmethod
    def _first(d, *keys):
        for k in keys:
            v = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, dict): v = v.get("value")
            if isinstance(v, (int, float)) and v >= 0: return float(v)
        return None

    def _power(self):
        try:
            d = self.lib.amdsmi_get_power_info(self.h)
            self.instant = self._first(d, "current_socket_power") is not None
            return self._first(d, "current_socket_power", "average_socket_power", "socket_power")
        except Exception: return None

    def _energy(self):   # mJ, from the accumulator x its resolution (µJ)
        try:
            d = self.lib.amdsmi_get_energy_count(self.h)
            acc = self._first(d, "energy_accumulator", "power"); res = self._first(d, "counter_resolution") or 15.3
            return acc * res / 1000.0 if acc is not None else None
        except Exception: return None

    def _limit(self):
        try:
            lim = self._first(self.lib.amdsmi_get_power_cap_info(self.h), "power_cap")
            return lim / 1e6 if lim and lim > 10000 else lim   # µW in some releases, W in others
        except Exception: return None

    def result(self):
        r = PowerMeter.result(self)
        if r: r["source"] = r["source"].replace("NVML", "amdsmi")
        return r


def torch_meter(device):
    """A meter for a torch device string: NVML for NVIDIA, amdsmi for AMD (ROCm); none for Apple (macOS shows GPU
    power only to an administrator, through powermetrics), Intel or the CPU."""
    try:
        import devices
        v = devices.vendor(device)
    except Exception:
        v = "nvidia" if str(device).startswith("cuda") else None
    i = int(str(device).split(":")[1]) if ":" in str(device) else 0
    if v == "amd":
        m = AmdMeter(i); return m if m.h else _Null()
    if v != "nvidia": return _Null()
    try:
        import torch
        i = torch.cuda.current_device() if ":" not in str(device) else i
        props = torch.cuda.get_device_properties(i); bus = getattr(props, "pci_bus_id", None)
        pci = f"{getattr(props, 'pci_domain_id', 0):08x}:{bus:02x}:{getattr(props, 'pci_device_id', 0):02x}.0" if isinstance(bus, int) else None
        return PowerMeter(i, pci)
    except Exception:
        return PowerMeter(0)


class _Null:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def result(self): return None


if __name__ == "__main__":
    with PowerMeter(0) as pm: time.sleep(1.0)
    print(pm.result())
