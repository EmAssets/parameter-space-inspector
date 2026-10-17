// SPDX-License-Identifier: Apache-2.0
// GGUF header reader, shared with the web Parameter Space Inspector (inspector.html).
(function (root) {
  'use strict';
  const LIMITS = { headMax: 128 << 20, fullBytes: 256 << 20, fullParams: 64e6, tensors: 1e5, kvs: 1e5, arr: 2e7, str: 64 << 20, dims: 8 };
  // ggml tensor types (ggml.h). Gaps are ids ggml retired (4, 5; 31-33 and 36-38 were repacked Q4_0 / IQ4_NL layouts
  // that newer files no longer write, kept here so an older file still reads by name).
  const TYPES = { 0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
    13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S',
    23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16', 31: 'Q4_0_4_4', 32: 'Q4_0_4_8', 33: 'Q4_0_8_8',
    34: 'TQ1_0', 35: 'TQ2_0', 36: 'IQ4_NL_4_4', 37: 'IQ4_NL_4_8', 38: 'IQ4_NL_8_8', 39: 'MXFP4' };
  // [values per block, bytes per block] for each type (ggml's type traits), so bytes per type can be computed from the header.
  const BLOCK = { 0: [1, 4], 1: [1, 2], 2: [32, 18], 3: [32, 20], 6: [32, 22], 7: [32, 24], 8: [32, 34], 9: [32, 36], 10: [256, 84], 11: [256, 110],
    12: [256, 144], 13: [256, 176], 14: [256, 210], 15: [256, 292], 16: [256, 66], 17: [256, 74], 18: [256, 98], 19: [256, 50], 20: [32, 18],
    21: [256, 110], 22: [256, 82], 23: [256, 136], 24: [1, 1], 25: [1, 2], 26: [1, 4], 27: [1, 8], 28: [1, 8], 29: [256, 56], 30: [1, 2],
    31: [32, 18], 32: [32, 18], 33: [32, 18], 34: [256, 54], 35: [256, 66], 36: [32, 18], 37: [32, 18], 38: [32, 18], 39: [32, 17] };
  // general.file_type: the quantisation recipe the file was made with (llama.h, LLAMA_FTYPE_*). It names the
  // recipe, not every tensor: a Q4_K_M file keeps some tensors in Q6_K or F32, so the per-type table is the truth.
  const FILE_TYPES = { 0: 'all F32', 1: 'mostly F16', 2: 'mostly Q4_0', 3: 'mostly Q4_1', 4: 'mostly Q4_1, some F16', 7: 'mostly Q8_0', 8: 'mostly Q5_0',
    9: 'mostly Q5_1', 10: 'mostly Q2_K', 11: 'mostly Q3_K (small)', 12: 'mostly Q3_K (medium)', 13: 'mostly Q3_K (large)', 14: 'mostly Q4_K (small)',
    15: 'mostly Q4_K (medium)', 16: 'mostly Q5_K (small)', 17: 'mostly Q5_K (medium)', 18: 'mostly Q6_K', 19: 'mostly IQ2_XXS', 20: 'mostly IQ2_XS',
    21: 'mostly Q2_K (small)', 22: 'mostly IQ3_XS', 23: 'mostly IQ3_XXS', 24: 'mostly IQ1_S', 25: 'mostly IQ4_NL', 26: 'mostly IQ3_S', 27: 'mostly IQ3_M',
    28: 'mostly IQ2_S', 29: 'mostly IQ2_M', 30: 'mostly IQ4_XS', 31: 'mostly IQ1_M', 32: 'mostly BF16', 33: 'mostly Q4_0_4_4', 34: 'mostly Q4_0_4_8',
    35: 'mostly Q4_0_8_8', 36: 'mostly TQ1_0', 37: 'mostly TQ2_0', 38: 'mostly MXFP4 (mixture-of-experts)' };
  const typeName = (t) => TYPES[t] || `unknown type ${t}`;
  // A tensor's bytes on disk, or null for a type this reader does not know (never guessed).
  const tensorBytes = (t) => { const b = BLOCK[t.type]; return b ? Math.ceil(t.count / b[0]) * b[1] : null; };
  const fileTypeName = (v) => v == null ? null : FILE_TYPES[v] != null ? `${FILE_TYPES[v]} (file_type ${v})` : `file_type ${v} (not in this reader’s list)`;
  // Parameters and bytes per tensor type, largest first; bytes null where the type is unknown.
  function byType(g) {
    const m = new Map();
    for (const t of g.tensors) {
      const k = typeName(t.type), e = m.get(k) || { type: t.type, name: k, tensors: 0, params: 0, bytes: 0, known: !!BLOCK[t.type] };
      e.tensors++; e.params += t.count; const b = tensorBytes(t); e.bytes = b == null || e.bytes == null ? null : e.bytes + b; m.set(k, e);
    }
    return [...m.values()].sort((a, b) => b.params - a.params).map((e) => ({ ...e, bitsPerParam: e.bytes != null && e.params ? (8 * e.bytes) / e.params : null }));
  }
  class NeedMore extends Error {}
  const damaged = (what) => new Error(`${what}: the file looks damaged or is not a model this inspector can read.`);

  // Metadata and tensor list from the first bytes of a file `size` bytes long.
  // Throws NeedMore when the slice ends before the header does.
  function parseHeader(buf, size) {
    const dv = new DataView(buf); let o = 0;
    const td = new TextDecoder();
    const need = (n) => { if (o + n > buf.byteLength) { if (buf.byteLength < size) throw new NeedMore(); throw damaged('The file ends in the middle of its header'); } };
    const num = (n, get) => () => { need(n); const v = get(o); o += n; return v; };
    const u8 = num(1, (p) => dv.getUint8(p)), i8 = num(1, (p) => dv.getInt8(p));
    const u16 = num(2, (p) => dv.getUint16(p, true)), i16 = num(2, (p) => dv.getInt16(p, true));
    const u32 = num(4, (p) => dv.getUint32(p, true)), i32 = num(4, (p) => dv.getInt32(p, true));
    const u64 = num(8, (p) => Number(dv.getBigUint64(p, true))), i64 = num(8, (p) => Number(dv.getBigInt64(p, true)));
    const f32 = num(4, (p) => dv.getFloat32(p, true)), f64 = num(8, (p) => dv.getFloat64(p, true));
    const str = () => {
      const n = u64();
      if (n > LIMITS.str || n > size) throw damaged(`A text field claims ${n.toLocaleString()} bytes`);
      need(n); const v = td.decode(new Uint8Array(buf, o, n)); o += n; return v;
    };
    const SIZE = [1, 1, 2, 2, 4, 4, 4, 1, 8, 0, 8, 8, 8];   // smallest encoded size per value type
    const read = (t) => {
      const f = [u8, i8, u16, i16, u32, i32, f32, () => u8() !== 0, str, null, u64, i64, f64][t];
      if (!f) throw damaged(`Unknown value type ${t}`);
      return f;
    };
    need(4);
    if (td.decode(new Uint8Array(buf, 0, 4)) !== 'GGUF') throw new Error('This is not a GGUF file.');
    o = 4;
    const version = u32();
    if (version !== 2 && version !== 3) throw new Error(`GGUF version ${version} is not supported (2 and 3 are).`);
    const nT = u64(), nKV = u64();
    if (nT > LIMITS.tensors) throw damaged(`The header lists ${nT.toLocaleString()} tensors`);
    if (nKV > LIMITS.kvs) throw damaged(`The header lists ${nKV.toLocaleString()} notes`);
    const kv = Object.create(null);   // a key named __proto__ is just a key
    for (let i = 0; i < nKV; i++) {
      const key = str(), t = u32();
      if (t === 9) {
        const et = u32(), n = u64();
        if (et === 9) throw damaged('Nested lists in the header');
        if (n > LIMITS.arr || n * SIZE[et] > size) throw damaged(`A list claims ${n.toLocaleString()} items`);
        const f = read(et), a = new Array(n);
        for (let j = 0; j < n; j++) a[j] = f();
        kv[key] = a;
      } else kv[key] = read(t)();
    }
    const tensors = [];
    for (let i = 0; i < nT; i++) {
      const name = str(), nd = u32();
      if (nd < 1 || nd > LIMITS.dims) throw damaged(`Tensor ${name} has ${nd} dimensions`);
      const dims = []; for (let j = 0; j < nd; j++) dims.push(u64());
      const count = dims.reduce((a, b) => a * b, 1);
      if (!Number.isSafeInteger(count)) throw damaged(`Tensor ${name} is impossibly large`);
      tensors.push({ name, dims, shape: [...dims].reverse(), count, type: u32(), offset: u64() });
    }
    const align = kv['general.alignment'] || 32;
    if (!Number.isInteger(align) || align < 1 || align > 1 << 20) throw damaged(`Alignment ${align}`);
    const start = Math.ceil(o / align) * align;
    for (const t of tensors) {
      if (start + t.offset > size) throw damaged(`Tensor ${t.name} starts past the end of the file`);
      if (t.type === 0 && start + t.offset + t.count * 4 > size) throw damaged(`Tensor ${t.name} runs past the end of the file`);
    }
    const params = tensors.reduce((a, t) => a + t.count, 0);
    return { version, kv, tensors, start, params, bytes: size };
  }

  // The name to show for a model. general.name is free text, and some publishers put an id there (LiquidAI's LFM2.5
  // GGUFs carry a 40-digit hex hash): then the base name and size label, else the file name without .gguf.
  const idLike = (s) => /^[0-9a-f]{16,}$/i.test(s) || /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(s);
  function displayName(kv, file) {
    const n = String((kv && kv['general.name']) || '').trim();
    if (n && !idLike(n)) return n;
    const b = String((kv && kv['general.basename']) || '').trim(), sz = String((kv && kv['general.size_label']) || '').trim();
    if (b && !idLike(b)) return sz && !b.includes(sz) ? `${b} ${sz}` : b;
    const f = String(file || '').split(/[\\/]/).pop().replace(/\.gguf$/i, '');
    return f || n || 'a model';
  }
  root.PSIHeader = { LIMITS, TYPES, BLOCK, FILE_TYPES, typeName, tensorBytes, fileTypeName, byType, parseHeader, NeedMore, displayName };
})(window);
