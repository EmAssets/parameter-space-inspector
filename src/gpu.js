// The machine's own sensing surface: who holds the GPU, whether anyone has spilled out of it, and what the whole
// machine (CPU, RAM) and the app's own processes are doing.
// On Windows (WDDM) a process that needs more than the card has left keeps running from shared system memory:
// no error, ~100x slower (a gpt-oss ladder run stalled this way on 2026-10-10 when the app's llama-server loaded
// beside it). The app reads this before it starts an engine and shows it in the header.
//
// gpuState({ pids }) -> {
//   ok, reason ('no_gpu' | 'sensor_failed' | null), detail, at (ISO), poll_ms, spill_detect ('ok' | 'unavailable'),
//   gpus: [{index, name, totalMB, usedMB, freeMB, util_pct}],     every GPU
//   gpu, totalMB, usedMB, freeMB, gpu_util_pct,                  the first GPU, as the summary
//   cpu: {pct, cores}, ram: {usedMB, totalMB},                   the machine
//   holders: [{pid, role, label, mb, sharedMB, ramMB, cpu_pct}], processes that are ours or research/training jobs:
//     role 'app' (this process), 'engine', 'inside', 'research', 'training'; the app and the engines in `pids` are
//     listed even with no GPU memory. procs is the same list (older name).
//   otherMB (GPU memory of every other program), spill: [holders...], busy: [holders...]
// }
// ok is about the GPU reading; reason 'sensor_failed' with ok true means the card was read but the per-process
// counters were not (detail says which). Anything a sensor did not say is null, not 0.
'use strict';
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');

// One PowerShell per poll, kept short (target < 1.5 s; the old Get-Counter version took ~4 s): the per-process GPU
// memory (dedicated / shared) from the GPU performance-counter CIM class, and process names, RAM and CPU seconds from one .NET process list (Get-Process -Id on
// ~50 ids took ~1.8 s). Command lines come from Win32_Process, which costs ~1 s per query on a busy machine, so they
// are asked only for pids not seen before (a pid's command line does not change) and only for program names that could
// be one of ours (python, llama-server, node, electron). The machine's CPU % and RAM come from Node (os.cpus deltas over
// the poll, os.totalmem/freemem), and per-process CPU % from CPU-second deltas between polls (the first reading has none).
const PS = (pids, known) => String.raw`
$ErrorActionPreference = 'SilentlyContinue'; $ProgressPreference = 'SilentlyContinue'
$want = @(${pids.join(',') || '0'}); $known = @{}; foreach ($k in @(${known.join(',') || '0'})) { $known[[int]$k] = 1 }
$ded = @{}; $sh = @{}; $counters_ok = $false
$gm = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -Property Name,DedicatedUsage,SharedUsage
foreach ($m in @($gm)) { if ($m -and $m.Name -match 'pid_(\d+)_') { $counters_ok = $true; $id = [int]$matches[1]; $ded[$id] += [double]$m.DedicatedUsage; $sh[$id] += [double]$m.SharedUsage } }
$ids = @(@($ded.Keys | Where-Object { $ded[$_] + $sh[$_] -gt 0 }) + $want | Where-Object { $_ -gt 0 } | Sort-Object -Unique)
$gp = @{}; foreach ($p in [Diagnostics.Process]::GetProcesses()) { $gp[[int]$p.Id] = $p }
$ours = '^(python|pythonw|llama-server|node|electron|psi|parameter)'
$ask = @($ids | Where-Object { $gp[$_] -and -not $known.ContainsKey([int]$_) -and $gp[$_].ProcessName -match $ours })
$cmd = @{}; if ($ask.Count -gt 0) { Get-CimInstance Win32_Process -Filter (($ask | % { "ProcessId=$_" }) -join ' OR ') -Property ProcessId,CommandLine | % { $cmd[[int]$_.ProcessId] = $_.CommandLine } }
$procs = foreach ($id in $ids) {
  $p = $gp[$id]; if (-not $p) { continue }
  # RAM and CPU seconds only for the processes that can be ours: reading them for every GPU client costs seconds.
  $mine = ($want -contains $id) -or ($p.ProcessName -match $ours); $ram = $null; $cs = $null
  if ($mine) { try { $ram = [double]$p.WorkingSet64; $cs = $p.TotalProcessorTime.TotalSeconds } catch { } }
  [pscustomobject]@{ pid = $id; name = $p.ProcessName; cmd = $cmd[$id]; asked = $ask -contains $id; dedicated = [double]$ded[$id]; shared = [double]$sh[$id]; ram = $ram; cpu_s = $cs }
}
[pscustomobject]@{ counters_ok = $counters_ok; procs = @($procs) } | ConvertTo-Json -Depth 4 -Compress
`;

const MB = 2 ** 20;
const base = (s) => String(s || '').replace(/["']/g, '').split(/[\\/]/).filter(Boolean).pop() || '';
const arg = (cmd, flag) => { const m = new RegExp(`${flag}\\s+("[^"]+"|\\S+)`).exec(cmd || ''); return m ? m[1].replace(/"/g, '') : ''; };
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : (typeof v === 'string' && v.trim() !== '' && Number.isFinite(+v) ? +v : null));

// What each process is, in the app's own words. Only named scripts count as research or training: a stray "train"
// in some other program's path is not a training run.
function role(p) {
  const cmd = p.cmd || '';
  if (p.pid === process.pid) return { role: 'app', label: `This app (${process.versions.electron ? 'desktop' : 'dev server'})` };
  if (/llama-server/i.test(p.name) || /llama-server/i.test(cmd)) return { role: 'engine', label: `Local engine: ${base(arg(cmd, '-m'))}` };
  if (/trace_server\.py/.test(cmd)) return { role: 'inside', label: `Inside engine: ${base(arg(cmd, '--model'))}` };
  if (/\b(model_ladder|model_intervene|model_deep)\.py\b/.test(cmd)) return { role: 'research', label: `Storage ladder run: ${arg(cmd, '--tag') || 'a model'}` };
  if (/\b(grow|train)\.py\b/.test(cmd)) return { role: 'training', label: `Training: ${base(arg(cmd, '--out')) || base(arg(cmd, '--model')) || 'a model'}` };
  return { role: 'other', label: p.name || `process ${p.pid}` };
}

const exec = (cmd, args, opts = {}) => new Promise((resolve) => execFile(cmd, args, { timeout: 8000, windowsHide: true, maxBuffer: 4 * MB, ...opts },
  (e, out, err) => resolve(e ? { error: `${e.code || e.message}${err ? `: ${String(err).trim().slice(0, 200)}` : ''}` } : { out: String(out) })));

// nvidia-smi: every GPU, and (where the driver tells) the CUDA processes on them.
async function smi() {
  const g = await exec('nvidia-smi', ['--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits']);
  if (g.error) return { error: g.error };
  const gpus = g.out.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => {
    const [index, name, total, used, util] = l.split(',').map((x) => x.trim());
    const totalMB = num(total), usedMB = num(used);
    return { index: num(index), name, totalMB, usedMB, freeMB: totalMB != null && usedMB != null ? totalMB - usedMB : null, util_pct: num(util) };
  });
  if (!gpus.length) return { error: 'nvidia-smi listed no GPU' };
  const a = await exec('nvidia-smi', ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']);
  const apps = a.error ? [] : a.out.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => { const [pid, mem] = l.split(',').map((x) => x.trim()); return { pid: num(pid), mb: num(mem) }; }).filter((x) => x.pid);
  return { gpus, apps, vendor: 'nvidia', reader: 'nvidia-smi' };
}

// ── Other GPUs: each reader returns { gpus, apps: [], vendor, reader, unified? } or { error } ──
// A key anywhere in a nested JSON whose name matches re (amd-smi and rocm-smi changed their field names across releases).
const deep = (o, re) => { if (!o || typeof o !== 'object') return undefined; for (const [k, v] of Object.entries(o)) { if (re.test(k) && (typeof v !== 'object' || (v && 'value' in v))) return v && typeof v === 'object' ? v.value : v; const d = deep(v, re); if (d !== undefined) return d; } return undefined; };
const toMB = (v, unit) => { const n = num(typeof v === 'string' ? v.replace(/[^\d.]/g, '') : v); if (n == null) return null; const u = String(unit || (typeof v === 'string' ? v : '')).toUpperCase(); return Math.round(/GB/.test(u) ? n * 1024 : /MB/.test(u) ? n : /KB/.test(u) ? n / 1024 : n / MB); };
function gpuRow(index, name, totalMB, usedMB, util) {
  return { index, name: name || 'GPU', totalMB, usedMB, freeMB: totalMB != null && usedMB != null ? Math.max(0, totalMB - usedMB) : null, util_pct: num(util) };
}

// AMD on Linux (ROCm): rocm-smi, else amd-smi. Both print JSON; the fields are looked for by name.
async function amd() {
  const r = await exec('rocm-smi', ['--showproductname', '--showmeminfo', 'vram', '--showuse', '--json']);
  if (!r.error) {
    try {
      const j = JSON.parse(r.out); const cards = Object.entries(j).filter(([k]) => /^card\d+/i.test(k));
      const gpus = cards.map(([k, c], i) => gpuRow(i, c['Card Series'] || c['Card series'] || c['Card model'] || k,
        toMB(num(c['VRAM Total Memory (B)'])), toMB(num(c['VRAM Total Used Memory (B)'])), num(String(c['GPU use (%)'] ?? '').replace('%', ''))));
      if (gpus.length) return { gpus, apps: [], vendor: 'amd', reader: 'rocm-smi' };
    } catch (_) { /* try amd-smi */ }
  }
  const s = await exec('amd-smi', ['static', '--asic', '--vram', '--json']);
  if (s.error) return { error: r.error || s.error };
  const m = await exec('amd-smi', ['metric', '--mem-usage', '--usage', '--json']);
  try {
    const st = [].concat(JSON.parse(s.out)), mt = m.error ? [] : [].concat(JSON.parse(m.out));
    const gpus = st.map((g, i) => {
      const t = deep(g, /^(size|vram_size|total_vram)$/i), u = deep(mt[i], /^used_vram$/i), tv = deep(mt[i], /^total_vram$/i);
      const unit = deep(g, /^unit$/i) || 'MB';
      return gpuRow(i, deep(g, /^(market_name|product_name)$/i), toMB(tv ?? t, unit), toMB(u, unit), deep(mt[i], /^gfx_activity$/i));
    });
    return gpus.length ? { gpus, apps: [], vendor: 'amd', reader: 'amd-smi' } : { error: 'amd-smi listed no GPU' };
  } catch (e) { return { error: `amd-smi: unreadable output (${e.message})` }; }
}

// Windows, any vendor (AMD, Intel, NVIDIA without nvidia-smi): the display adapters' 64-bit memory size from the
// registry and their dedicated memory in use from the GPU performance counters. Adapters are paired largest to largest.
const WIN_ADAPTERS = String.raw`
$ErrorActionPreference = 'SilentlyContinue'
$cls = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
$ads = Get-ChildItem $cls | Where-Object { $_.PSChildName -match '^\d{4}$' } | ForEach-Object { $p = Get-ItemProperty $_.PSPath
  $m = $p.'HardwareInformation.qwMemorySize'; if (-not $m) { $m = $p.'HardwareInformation.MemorySize' }; if ($m -is [byte[]]) { $m = [BitConverter]::ToUInt32($m, 0) }
  if ($p.DriverDesc -and $m -and $p.DriverDesc -notmatch 'Basic|Remote|Virtual|Parsec|Meta') { [pscustomobject]@{ name = $p.DriverDesc; bytes = [double]$m; vendor = $p.ProviderName } } }
$use = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -Property Name,DedicatedUsage | ForEach-Object { [double]$_.DedicatedUsage }
$eng = @{}; Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -Property Name,UtilizationPercentage | ForEach-Object { if ($_.Name -match '(luid_\w+?_phys_\d+).*engtype_3D') { $eng[$matches[1]] += [double]$_.UtilizationPercentage } }
[pscustomobject]@{ adapters = @($ads); used = @($use); util = @($eng.Values) } | ConvertTo-Json -Depth 3 -Compress
`;
function winAdapters() {
  return new Promise((resolve) => execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(WIN_ADAPTERS, 'utf16le').toString('base64')],
    { timeout: 15000, windowsHide: true, maxBuffer: 4 * MB }, (e, out) => {
      if (e) return resolve({ error: `powershell: ${e.code || e.message}` });
      let j; try { j = JSON.parse(String(out).trim()); } catch (_) { return resolve({ error: 'Windows GPU counters: unreadable output' }); }
      const ads = [].concat(j.adapters || []).filter((a) => a && a.bytes > 256 * MB).sort((a, b) => b.bytes - a.bytes);
      const used = [].concat(j.used || []).map(Number).sort((a, b) => b - a), util = [].concat(j.util || []).map(Number).sort((a, b) => b - a);
      if (!ads.length) return resolve({ error: 'no display adapter with its own memory' });
      const vendorOf = (s) => (/amd|advanced micro|radeon/i.test(s) ? 'amd' : /intel/i.test(s) ? 'intel' : /nvidia/i.test(s) ? 'nvidia' : null);
      const gpus = ads.map((a, i) => gpuRow(i, a.name, Math.round(a.bytes / MB), used[i] != null ? Math.round(used[i] / MB) : null, util[i] != null ? Math.min(100, Math.round(util[i])) : null));
      resolve({ gpus, apps: [], vendor: vendorOf(`${ads[0].vendor} ${ads[0].name}`), reader: 'Windows GPU counters' });
    }));
}

// macOS. Apple Silicon has one memory for the CPU and the GPU ("unified"): the GPU's share is the machine's RAM, so
// the figures are the RAM's, and nothing can spill. An Intel Mac's discrete GPU is named with its own memory size.
let MAC_NAME = null;
async function mac() {
  if (MAC_NAME == null) {
    const b = await exec('sysctl', ['-n', 'machdep.cpu.brand_string']); const brand = b.error ? '' : b.out.trim();
    if (/Apple/i.test(brand)) MAC_NAME = { name: `${brand} GPU`, unified: true, totalMB: null };
    else {
      const sp = await exec('system_profiler', ['SPDisplaysDataType', '-json'], { timeout: 20000 });
      let d = null; try { d = [].concat(JSON.parse(sp.out).SPDisplaysDataType || [])[0]; } catch (_) { /* none */ }
      MAC_NAME = d ? { name: d.sppci_model || d._name || 'GPU', unified: false, totalMB: toMB(d.spdisplays_vram || d.spdisplays_vram_shared || null) } : { name: null, unified: false, totalMB: null };
    }
  }
  if (!MAC_NAME.name) return { error: 'No GPU found on this Mac' };
  if (MAC_NAME.unified) {
    const totalMB = Math.round(os.totalmem() / MB), usedMB = Math.round((os.totalmem() - os.freemem()) / MB);
    return { gpus: [gpuRow(0, MAC_NAME.name, totalMB, usedMB, null)], apps: [], vendor: 'apple', reader: 'macOS (unified memory)', unified: true };
  }
  return { gpus: [gpuRow(0, MAC_NAME.name, MAC_NAME.totalMB, null, null)], apps: [], vendor: /amd|radeon/i.test(MAC_NAME.name) ? 'amd' : /intel/i.test(MAC_NAME.name) ? 'intel' : null, reader: 'system_profiler' };
}

// Every GPU reader this platform has, in order: NVIDIA first (it names the processes on the card), then the others.
async function readGpu() {
  const tried = [];
  const order = process.platform === 'darwin' ? [mac] : process.platform === 'win32' ? [smi, winAdapters] : [smi, amd];
  for (const f of order) { const g = await f(); if (!g.error) return g; tried.push(`${f.name}: ${g.error}`); }
  return { error: tried.join('; '), tried };
}

// macOS: the processes from ps (no /proc there); the machine from Node.
function macCounters(pids) {
  return new Promise((resolve) => {
    const base = { counters_ok: false, cpu: null, cores: os.cpus().length || null, ram_total_kb: os.totalmem() / 1024, ram_free_kb: os.freemem() / 1024, procs: [] };
    if (!pids.length) return resolve(base);
    execFile('ps', ['-o', 'pid=,rss=,%cpu=,command=', '-p', pids.join(',')], { timeout: 8000 }, (e, out) => {
      if (e && !out) return resolve(base);
      base.procs = String(out).split('\n').map((l) => /^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(.*)$/.exec(l)).filter(Boolean)
        .map(([, pid, rss, cpu, cmd]) => ({ pid: +pid, name: cmd.split(' ')[0].split('/').pop(), cmd, dedicated: null, shared: null, ram: +rss * 1024, cpu: +cpu }));
      resolve(base);
    });
  });
}

const CMD = new Map();             // pid -> command line ('' when not ours to ask), kept across polls
const LAST_WIN = new Map();        // pid -> {cpu_s, at}
function winCounters(pids) {
  return new Promise((resolve) => {
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(PS(pids, [...CMD.keys()]), 'utf16le').toString('base64')],
      { timeout: 15000, windowsHide: true, maxBuffer: 4 * MB }, (e, out, err) => {
        if (e) return resolve({ error: `powershell: ${e.code || e.message} ${String(err || '').trim().slice(0, 200)}`.trim() });
        let j; try { j = JSON.parse(String(out).trim()); } catch (_) { return resolve({ error: `powershell: unreadable output ${String(out).trim().slice(0, 120)}` }); }
        const procs = (Array.isArray(j.procs) ? j.procs : j.procs ? [j.procs] : []).map((p) => {
          if (p.asked || !CMD.has(p.pid)) CMD.set(p.pid, p.cmd || '');
          const prev = LAST_WIN.get(p.pid), now = Date.now(), cs = num(p.cpu_s);
          if (cs != null) LAST_WIN.set(p.pid, { cpu_s: cs, at: now });
          // CPU % of one core over the time since the previous poll; shown later as a share of the machine.
          const cpu = prev && cs != null && now > prev.at ? Math.max(0, (cs - prev.cpu_s) / ((now - prev.at) / 1000) * 100) : null;
          return { ...p, cmd: CMD.get(p.pid) || '', cpu };
        });
        if (CMD.size > 2000) CMD.clear();
        resolve({ counters_ok: j.counters_ok === true, procs });
      });
  });
}

// Linux: /proc for the machine and the processes (rates from the previous poll, so the first reading has none);
// the GPU holders come from nvidia-smi, and no spill counter exists there.
const readTxt = (f) => { try { return fs.readFileSync(f, 'utf8'); } catch (_) { return ''; } };
let LAST_CPU = null;             // {idle, total} from /proc/stat
const LAST_PROC = new Map();     // pid -> {ticks, at}
function linuxCounters(pids) {
  const stat = readTxt('/proc/stat').split('\n')[0].trim().split(/\s+/).slice(1).map(Number);
  let cpu = null;
  if (stat.length >= 4) {
    const idle = stat[3] + (stat[4] || 0), total = stat.reduce((a, b) => a + b, 0);
    if (LAST_CPU && total > LAST_CPU.total) cpu = +((1 - (idle - LAST_CPU.idle) / (total - LAST_CPU.total)) * 100).toFixed(1);
    LAST_CPU = { idle, total };
  }
  const mem = Object.fromEntries(readTxt('/proc/meminfo').split('\n').map((l) => l.split(/:\s+/)).filter((x) => x.length === 2).map(([k, v]) => [k, parseInt(v, 10)]));
  const hz = 100, now = Date.now();
  const procs = pids.map((pid) => {
    const cmd = readTxt(`/proc/${pid}/cmdline`).replace(/\0/g, ' ').trim(); if (!cmd) return null;
    const st = readTxt(`/proc/${pid}/stat`).split(') ').pop().split(' '), ticks = (+st[11] || 0) + (+st[12] || 0);
    const rss = /VmRSS:\s+(\d+)/.exec(readTxt(`/proc/${pid}/status`)), prev = LAST_PROC.get(pid); LAST_PROC.set(pid, { ticks, at: now });
    const cpuPct = prev && now > prev.at ? +(((ticks - prev.ticks) / hz) / ((now - prev.at) / 1000) * 100).toFixed(1) : null;
    return { pid, name: cmd.split(' ')[0].split('/').pop(), cmd, dedicated: null, shared: null, ram: rss ? +rss[1] * 1024 : null, cpu: cpuPct };
  }).filter(Boolean);
  return { counters_ok: false, cpu, cores: os.cpus().length || null, ram_total_kb: mem.MemTotal || null, ram_free_kb: mem.MemAvailable || mem.MemFree || null, procs };
}

// The machine's CPU % over the poll itself (os.cpus() before and after), and its RAM, from Node: no extra process.
const cpuTimes = () => os.cpus().reduce((a, c) => { const t = c.times; a.idle += t.idle; a.total += t.user + t.nice + t.sys + t.idle + t.irq; return a; }, { idle: 0, total: 0 });
async function run(pids) {
  const c0 = cpuTimes();
  let g, c;
  if (process.platform === 'win32') {
    [g, c] = await Promise.all([readGpu(), winCounters(pids)]);
    const c1 = cpuTimes(), dt = c1.total - c0.total;
    Object.assign(c, { cpu: dt > 0 ? (1 - (c1.idle - c0.idle) / dt) * 100 : null, cores: os.cpus().length || null,
      ram_total_kb: os.totalmem() / 1024, ram_free_kb: os.freemem() / 1024 });
  } else if (process.platform === 'darwin') {
    [g, c] = await Promise.all([readGpu(), macCounters(pids)]);
    const c1 = cpuTimes(), dt = c1.total - c0.total; c.cpu = dt > 0 ? (1 - (c1.idle - c0.idle) / dt) * 100 : null;
  } else {
    g = await readGpu();
    c = linuxCounters([...new Set([...pids, ...(g.apps || []).map((a) => a.pid)])]);
  }
  return { g, c };
}

let cache = null, at = 0, pending = null, lastPids = '';
async function gpuState({ maxAgeMs = 10000, pids = [] } = {}) {
  const want = [...new Set([process.pid, ...pids.filter((p) => Number.isInteger(p) && p > 0)])];
  const key = want.join(',');
  if (cache && key === lastPids && Date.now() - at < maxAgeMs) return cache;
  if (!pending) {
    const t0 = Date.now();
    pending = run(want).then((r) => { pending = null; cache = shape(r, want, Date.now() - t0); at = Date.now(); lastPids = key; return cache; });
  }
  return pending;
}

// Windows' "% Processor Time" for a process is of one core (a process on 4 cores reads 400): shown as a share of the machine.
const cpuPct = (v, cores) => (v == null ? null : +((process.platform === 'win32' || process.platform === 'darwin') && cores ? v / cores : v).toFixed(1));   // ps on macOS counts one core as 100 too

function shape({ g, c }, want, pollMs) {
  const out = { ok: false, reason: null, detail: null, at: new Date().toISOString(), poll_ms: pollMs, spill_detect: 'unavailable', vendor: g.vendor || null, reader: g.reader || null, unified: !!g.unified,
    gpus: [], gpu: null, totalMB: null, usedMB: null, freeMB: null, gpu_util_pct: null, cpu: { pct: null, cores: null }, ram: { usedMB: null, totalMB: null },
    holders: [], otherMB: null, spill: [], busy: [] };
  // The machine, whatever the GPU said.
  out.cpu = { pct: num(c.cpu) != null ? +(+c.cpu).toFixed(1) : null, cores: num(c.cores) };
  const tot = num(c.ram_total_kb), free = num(c.ram_free_kb);
  out.ram = { usedMB: tot != null && free != null ? Math.round((tot - free) / 1024) : null, totalMB: tot != null ? Math.round(tot / 1024) : null };
  if (c.error) out.detail = c.error;
  const cprocs = Array.isArray(c.procs) ? c.procs : c.procs ? [c.procs] : [];
  const byPid = new Map(cprocs.map((p) => [p.pid, p]));
  const ramMB = (p) => (p && p.ram != null ? Math.round(p.ram / MB) : null);
  if (g.error) {
    // No reader found a card (each one's error is in g.tried): no GPU, or one this platform has no reader for.
    const noGpu = (g.tried || [g.error]).every((t) => /ENOENT|not found|not recognized|NVIDIA-SMI has failed|No devices|listed no GPU|no display adapter|No GPU found|cannot find/i.test(t));
    const readers = process.platform === 'darwin' ? 'macOS' : process.platform === 'win32' ? 'nvidia-smi or the Windows GPU counters' : 'nvidia-smi, rocm-smi or amd-smi';
    Object.assign(out, noGpu ? { reason: 'no_gpu', detail: `No GPU found (${readers} found none). The answering engine can still use one through its own build (Vulkan, Metal, ROCm); the inside engine runs on the CPU.` }
      : { reason: 'sensor_failed', detail: `The GPU sensor failed; the card may be there but could not be read: ${g.error}` });
  } else {
    const first = g.gpus[0];
    Object.assign(out, { ok: true, gpus: g.gpus, gpu: first.name, totalMB: first.totalMB, usedMB: first.usedMB, freeMB: first.freeMB, gpu_util_pct: first.util_pct });
    if (c.error) { out.reason = 'sensor_failed'; }   // the card was read, the per-process counters were not: said in detail
    const totalMB = first.totalMB || Infinity;
    // Per-process dedicated/shared memory comes from Windows' counters; without them (Linux, or the counter failed)
    // nvidia-smi's compute apps give a holder's memory where the driver reports it, and a spill cannot be told from slowness.
    const counters = process.platform === 'win32' && c.counters_ok === true;
    out.spill_detect = counters ? 'ok' : 'unavailable';
    const src = counters ? cprocs.filter((p) => (p.dedicated || 0) + (p.shared || 0) > 0) : (g.apps || []).map((a) => ({ ...(byPid.get(a.pid) || {}), pid: a.pid, mb: a.mb }));
    out.holders = src.map((p) => ({ pid: p.pid, ...role(p), mb: counters ? Math.round(Math.min(p.dedicated || 0, totalMB * MB) / MB) : (p.mb ?? null),
      sharedMB: counters ? Math.round((p.shared || 0) / MB) : null, ramMB: ramMB(p), cpu_pct: cpuPct(p.cpu, out.cpu.cores) }))
      .filter((h) => h.role !== 'other').sort((a, b) => (b.mb || 0) - (a.mb || 0));
    // Unified memory (Apple Silicon): the GPU's memory is the RAM, so nothing can spill out of it.
    if (g.unified) out.spill_detect = 'unified';
    // Windows' per-process counter is wrong for some programs (a screen recorder or the compositor can read as the whole
    // card), so other programs are shown together: what nvidia-smi says is used, less the processes named here.
    out.otherMB = first.usedMB != null ? Math.max(0, first.usedMB - out.holders.reduce((s, h) => s + (h.mb || 0), 0)) : null;
  }
  // The app's own processes are holders even when they hold no GPU memory (mb 0 with counters, null without): this
  // process (role 'app') and the engines it was given.
  const seen = new Set(out.holders.map((h) => h.pid));
  for (const pid of want) {
    if (seen.has(pid)) continue;
    const p = byPid.get(pid); if (!p && pid !== process.pid) continue;   // gone, or not readable
    out.holders.push({ pid, ...role(p || { pid, name: 'node', cmd: '' }), mb: out.spill_detect === 'ok' ? 0 : null, sharedMB: out.spill_detect === 'ok' ? 0 : null,
      ramMB: ramMB(p) != null ? ramMB(p) : pid === process.pid ? Math.round(process.memoryUsage().rss / MB) : null, cpu_pct: p ? cpuPct(p.cpu, out.cpu.cores) : null });
  }
  // Spilled: the card is full and this process lives partly in system RAM (shared usage under ~256 MB is normal).
  out.spill = out.spill_detect === 'ok' ? out.holders.filter((h) => h.sharedMB >= 256) : [];
  // Long GPU jobs the app should not crowd out by starting an engine.
  out.busy = out.holders.filter((h) => h.role === 'research' || h.role === 'training');
  out.procs = out.holders;   // older name, same list
  return out;
}

module.exports = { gpuState, _readers: { smi, amd, winAdapters, mac, macCounters, readGpu } };   // _readers: for tests
