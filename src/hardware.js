// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// What each engine can use on this machine, for Settings > Check the setup: the GPU the system reports (gpu.js), the
// GPUs this llama.cpp build can run on (llama-server --list-devices), and what PyTorch sees (src/devices.py). Where
// an engine would fall back to the CPU although a GPU is there, it says which build to install, for this platform and
// this vendor. Nothing is installed or changed.
const { execFile } = require('child_process');
const path = require('path');
const { python: PY } = require('./python.js');

const run = (cmd, args, opts = {}) => new Promise((resolve) => execFile(cmd, args, { timeout: 60000, windowsHide: true, maxBuffer: 4 << 20, ...opts },
  (e, out, err) => resolve({ error: e ? (e.code === 'ENOENT' ? 'not found' : String(err || e.message).trim().slice(0, 300)) : null, out: `${out || ''}\n${err || ''}` })));

// "  CUDA0: NVIDIA GeForce RTX 4080 SUPER (16375 MiB, 4540 MiB free)", "  Vulkan0: AMD Radeon RX 7900 XTX (…)", "  MTL0: Apple M3 Max (…)"
function parseDevices(text) {
  return String(text).split('\n').map((l) => /^\s+([A-Za-z]+)(\d+):\s+(.+?)(?:\s+\((\d+) MiB(?:,\s*(\d+) MiB free)?\))?\s*$/.exec(l)).filter(Boolean)
    .map(([, backend, i, name, total, free]) => ({ backend, index: +i, name, totalMB: total ? +total : null, freeMB: free ? +free : null }));
}

const LLAMA_BUILD = {
  nvidia: 'the CUDA build (llama-…-bin-win-cuda-… on Windows; on Linux build with -DGGML_CUDA=ON), or the Vulkan build',
  amd: process.platform === 'linux' ? 'the ROCm (HIP) build (-DGGML_HIP=ON), or the Vulkan build' : 'the Vulkan build (llama-…-bin-win-vulkan-x64)',
  intel: 'the Vulkan build, or the SYCL build for Intel GPUs',
  apple: 'any macOS build: Metal is on by default (brew install llama.cpp)',
};
const TORCH_BUILD = {
  nvidia: 'pip install torch --index-url https://download.pytorch.org/whl/cu124 (a CUDA build of PyTorch; an RTX 50-series card needs …/whl/cu128 instead)',
  amd: process.platform === 'linux' ? 'pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 (a ROCm build; Linux only)' : 'PyTorch has no AMD GPU support on Windows yet: the inside engine runs on the CPU (WSL2 with ROCm is the way to use the card)',
  intel: 'pip install torch --index-url https://download.pytorch.org/whl/xpu (an XPU build)',
  apple: 'pip install torch (the standard macOS build has MPS; needs macOS 12.3 or later on Apple Silicon)',
};

async function check({ engine, gpuState }) {
  const out = { platform: `${process.platform} ${process.arch}`, gpu: null, llama: null, torch: null, python: PY(), advice: [] };
  // 1. The system's view.
  try {
    const g = await gpuState({ maxAgeMs: 0 });
    out.gpu = g.ok ? { vendor: g.vendor, reader: g.reader, unified: g.unified, gpus: (g.gpus || []).map((x) => ({ name: x.name, totalMB: x.totalMB })) } : { none: true, detail: g.detail };
  } catch (e) { out.gpu = { error: e.message }; }
  const vendor = out.gpu && out.gpu.vendor;
  // 2. llama.cpp's view.
  const exe = engine && engine.exe;
  if (!exe) out.llama = { error: 'llama-server not found' };
  else {
    const r = await run(exe, ['--list-devices']);
    const devs = parseDevices(r.out);
    out.llama = { exe, devices: devs, backends: [...new Set(devs.map((d) => d.backend))], error: devs.length ? null : r.error };
    if (out.gpu && out.gpu.gpus && out.gpu.gpus.length && !devs.length)
      out.advice.push(`The answering engine (llama.cpp) sees no GPU, so it runs on the processor. Install ${LLAMA_BUILD[vendor] || 'a build for your GPU (Vulkan works on most cards)'} and set its folder in Settings.`);
  }
  // 3. PyTorch's view (the inside engine, experiments, grafts).
  const r = await run(PY(), [path.join(__dirname, 'devices.py')], { cwd: __dirname });
  try { out.torch = JSON.parse(r.out.trim().split('\n').filter((l) => l.startsWith('{')).pop()); }
  catch (_) { out.torch = { error: r.error || 'PyTorch did not answer (is it installed for this Python?)' }; }
  if (out.torch && !out.torch.error && !out.torch.accelerator && out.gpu && out.gpu.gpus && out.gpu.gpus.length)
    out.advice.push(`PyTorch sees no GPU, so the inside engine runs on the processor (slow beyond small models). ${TORCH_BUILD[vendor] || 'Install a PyTorch build for your GPU (pytorch.org/get-started).'}`);
  if (out.torch && out.torch.error) out.advice.push(`The inside engine needs PyTorch: ${out.torch.error}. Python used: ${out.python} (set PSI_PYTHON to choose another).`);
  if (vendor === 'apple') out.advice.push('Apple Silicon: both engines share the RAM with macOS and other apps. A model of about two thirds of the RAM is the practical limit; GPU power is not shown (macOS gives it only to an administrator).');
  if (vendor && vendor !== 'nvidia' && vendor !== 'apple') out.advice.push('Power figures (watts, energy) are read for NVIDIA cards, and for AMD cards on Linux when ROCm’s amdsmi is installed.');
  return out;
}

module.exports = { check, parseDevices };
