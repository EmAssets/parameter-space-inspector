// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
// The Experiment designer's data layer: every number an experiment recorded, as one long table in one shape
// (a row per prompt × measure × layer × token), and what reads it: the data grid in the report, the Excel and
// PowerPoint exports, and the analysis sandbox (sandbox-worker.js), where the user or Prometheus runs code over it.
//
//   columns: prompt (1, 2, …), the plan's factors, measure, layer, token, piece, neuron, value, note
//
// The table is made from the saved results whenever a record is opened, so every experiment in the history has one,
// including those run before this file existed. Nothing here runs code from a plan or an answer: the sandbox runs
// only code typed by the user or written by Prometheus, shown in full beside its result.
(() => {
  'use strict';
  const C = window.PSIDesignCore;
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const AMBER = '#ffc36b', INK = '#f2f1ee', INK3 = '#75726c';
  const PALETTE = [AMBER, '#7fb2c9', '#c99be0', '#9fd39a', INK, '#e08a7f', '#d6c27a', INK3];
  const font = (r, px) => `${px * r}px "IBM Plex Mono", ui-monospace, monospace`;

  // ── the ZIP container (Word, Excel and PowerPoint files are ZIPs of XML parts) ──
  function crc32(u8) { let c, crc = -1; for (let i = 0; i < u8.length; i++) { c = (crc ^ u8[i]) & 0xff; for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0xedb88320 : c >>> 1; crc = (crc >>> 8) ^ c; } return (crc ^ -1) >>> 0; }
  function zip(files) {
    const enc = new TextEncoder(), parts = [], dir = []; let off = 0;
    const u16 = (v) => [v & 255, (v >>> 8) & 255], u32 = (v) => [v & 255, (v >>> 8) & 255, (v >>> 16) & 255, (v >>> 24) & 255];
    for (const f of files) {
      const name = enc.encode(f.name), data = typeof f.data === 'string' ? enc.encode(f.data) : f.data, crc = crc32(data);
      const head = new Uint8Array([...u32(0x04034b50), ...u16(20), ...u16(0x0800), ...u16(0), ...u16(0), ...u16(0x21), ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(name.length), ...u16(0)]);
      parts.push(head, name, data);
      dir.push(new Uint8Array([...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0x0800), ...u16(0), ...u16(0), ...u16(0x21), ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(name.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(0), ...u32(off)]), name);
      off += head.length + name.length + data.length;
    }
    const dsize = dir.reduce((a, x) => a + x.length, 0);
    const end = new Uint8Array([...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length), ...u32(dsize), ...u32(off), ...u16(0)]);
    const all = [...parts, ...dir, end], out = new Uint8Array(all.reduce((a, x) => a + x.length, 0)); let p = 0; for (const x of all) { out.set(x, p); p += x.length; } return out;
  }
  const xml = (t) => String(t == null ? '' : t).replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f￾￿]/g, '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const pngOf = (cv) => { try { return Uint8Array.from(atob(cv.toDataURL('image/png').split(',')[1]), (c) => c.charCodeAt(0)); } catch (_) { return null; } };

  // ── the table ──
  function tableOf(plan, runs) {
    const F = plan.factors, columns = ['prompt', ...F, 'measure', 'layer', 'token', 'piece', 'neuron', 'value', 'note'], rows = [];
    const T = C.TOOLS[plan.instrument];
    runs.forEach((r, i) => {
      const t = r.t || {}, base = { prompt: i + 1 }; F.forEach((f) => { base[f] = r.item[f]; });
      const add = (o) => rows.push({ ...base, measure: o.measure, layer: o.layer ?? null, token: o.token ?? null, piece: o.piece ?? null, neuron: o.neuron ?? null, value: typeof o.value === 'number' && Number.isFinite(o.value) ? o.value : null, note: o.note ?? null });
      if (t.error) { add({ measure: 'error', note: String(t.error).slice(0, 300) }); return; }
      if (T) {
        Object.entries(t.metrics || {}).forEach(([k, v]) => add({ measure: k, value: v, note: (t.atLeast || []).includes(k) ? 'at least' : (t.atMost || []).includes(k) ? 'at most' : null }));
        if (t.cat != null) add({ measure: 'class', note: String(t.cat) });
        if (t.text) add({ measure: 'output', note: String(t.text).slice(0, 500) });
        const raw = t.raw || {};
        if (plan.instrument === 'features' && Array.isArray(raw.tokens)) raw.tokens.forEach((x, j) => (x.top || []).forEach(([f, v]) => add({ measure: 'feature_activation', layer: raw.layer, token: j + 1, piece: x.piece, neuron: f, value: v, note: `${(raw.names || {})[f] || ''}${x.first ? ' (first piece)' : ''}`.trim() || null })));
        if (plan.instrument === 'vectors' && Array.isArray(raw.norms)) raw.norms.forEach((v, l) => add({ measure: 'vector_norm', layer: l, value: v, note: l === 0 ? 'embeddings' : null }));
        return;
      }
      if (plan.instrument === 'read') {
        (t.pieces || []).forEach((x, j) => {
          add({ measure: 'surprise_bits', token: j + 2, piece: x.piece, value: x.surprise != null ? x.surprise : x.floor, note: x.surprise == null ? 'at least (outside the top 20 guesses)' : null });
          if (x.rank != null) add({ measure: 'rank', token: j + 2, piece: x.piece, value: x.rank });
        });
        add({ measure: 'total_bits', value: (t.pieces || []).reduce((s, x) => s + (x.surprise != null ? x.surprise : x.floor || 0), 0), note: (t.pieces || []).some((x) => x.surprise == null) ? 'at least' : null });
        return;
      }
      (t.tokens || []).forEach((x, j) => {   // the Neural net recording: per answer token, per layer
        add({ measure: 'token_p', token: j + 1, piece: x.piece, value: x.p });
        (x.lens || []).forEach((lp, l) => { if (lp) add({ measure: 'lens_p', layer: l + 1, token: j + 1, piece: lp[0], value: lp[1] }); });
        (x.resid || []).forEach((v, l) => add({ measure: 'resid_norm', layer: l + 1, token: j + 1, value: v }));
        (x.mlp || []).forEach((top, l) => (top || []).forEach((m, k) => add({ measure: 'neuron_activation', layer: l + 1, token: j + 1, piece: x.piece, neuron: m[0], value: m[1], note: `rank ${k + 1} in its layer` })));
      });
    });
    return { columns, rows };
  }
  // The vectors an experiment recorded (the Layer vectors instrument), for the sandbox.
  function vectorsOf(plan, runs) {
    if (plan.instrument !== 'vectors') return [];
    return runs.filter((r) => r.t && r.t.raw && r.t.raw.b64).map((r, i) => { const o = { prompt: i + 1, text: r.item.text }; plan.factors.forEach((f) => { o[f] = r.item[f]; }); return { ...o, factors: Object.fromEntries(plan.factors.map((f) => [f, r.item[f]])), layers: r.t.raw.layers, dims: r.t.raw.dims, b64: r.t.raw.b64 }; });
  }

  // ── the data grid ──
  function grid(box, table) {
    const PAGE = 100; let measure = '', q = '', sortK = null, sortD = 1, page = 0;
    const measures = [...new Set(table.rows.map((r) => r.measure))];
    const bar = el('div', 'row dz-gridbar'), sel = el('select'), find = el('input'), info = el('span', 'small muted'), prev = el('button', 'ghost', '‹ Earlier'), next = el('button', 'ghost', 'Later ›');
    sel.id = 'dzGridMeasure'; find.id = 'dzGridFind'; find.type = 'search'; find.placeholder = 'find in any column'; prev.type = next.type = 'button';
    sel.setAttribute('aria-label', 'show one measure'); find.setAttribute('aria-label', 'find in the data');
    sel.appendChild(new Option(`every measure (${measures.length})`, '')); measures.forEach((m) => sel.appendChild(new Option(`${m} (${table.rows.filter((r) => r.measure === m).length})`, m)));
    bar.append(sel, find, prev, next, info);
    const wrap = el('div', 'scroll dz-grid'), tb = el('table', 'dz-gridt');
    wrap.appendChild(tb); box.append(bar, wrap);
    const cols = table.columns.filter((c) => table.rows.some((r) => r[c] != null && r[c] !== ''));
    const fmt = (v) => (v == null ? '' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : Math.abs(v) >= 1000 || Math.abs(v) < 0.001 ? v.toPrecision(4) : v.toFixed(4)) : String(v));
    function draw() {
      let rows = table.rows; if (measure) rows = rows.filter((r) => r.measure === measure);
      if (q) { const s = q.toLowerCase(); rows = rows.filter((r) => cols.some((c) => r[c] != null && String(r[c]).toLowerCase().includes(s))); }
      if (sortK) rows = rows.slice().sort((a, b) => { const x = a[sortK], y = b[sortK]; if (x == null) return 1; if (y == null) return -1; return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * sortD; });
      const pages = Math.max(1, Math.ceil(rows.length / PAGE)); page = Math.min(page, pages - 1);
      tb.textContent = ''; const hr = el('tr');
      cols.forEach((c) => { const th = el('th', 'dz-sort', `${c}${sortK === c ? (sortD > 0 ? ' ▲' : ' ▼') : ''}`); th.tabIndex = 0; th.title = 'sort by this column'; const go = () => { if (sortK === c) sortD = -sortD; else { sortK = c; sortD = 1; } draw(); }; th.addEventListener('click', go); th.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); }); hr.appendChild(th); });
      tb.appendChild(hr);
      rows.slice(page * PAGE, page * PAGE + PAGE).forEach((r) => { const tr = el('tr'); cols.forEach((c) => { const td = el('td', typeof r[c] === 'number' ? 'num' : null, fmt(r[c])); if (c === 'piece' && r[c] != null) td.textContent = C.show(r[c]); tr.appendChild(td); }); tb.appendChild(tr); });
      info.textContent = `${rows.length.toLocaleString()} rows${rows.length > PAGE ? ` · page ${page + 1} of ${pages}` : ''}`; prev.disabled = page === 0; next.disabled = page >= pages - 1;
    }
    sel.addEventListener('change', () => { measure = sel.value; page = 0; draw(); });
    find.addEventListener('input', () => { q = find.value.trim(); page = 0; draw(); });
    prev.addEventListener('click', () => { page--; draw(); }); next.addEventListener('click', () => { page++; draw(); });
    draw();
  }

  // ── Excel (.xlsx): a sheet per part, the header row bold and frozen ──
  const colName = (i) => { let s = ''; for (i += 1; i > 0; i = Math.floor((i - 1) / 26)) s = String.fromCharCode(65 + ((i - 1) % 26)) + s; return s; };
  const sheetName = (s, used) => { let n = String(s).replace(/[[\]:*?/\\]/g, ' ').trim().slice(0, 31) || 'sheet'; let k = 2; const b = n; while (used.has(n.toLowerCase())) n = `${b.slice(0, 28)} ${k++}`; used.add(n.toLowerCase()); return n; };
  function sheetXml(rows) {   // rows: arrays; the first is the header
    const out = rows.map((r, i) => `<row r="${i + 1}">${r.map((v, j) => {
      const ref = `${colName(j)}${i + 1}`, s = i === 0 ? ' s="1"' : '';
      if (typeof v === 'number' && Number.isFinite(v)) return `<c r="${ref}"${s}><v>${v}</v></c>`;
      if (v == null || v === '') return '';
      let t = String(v); if (/^[=+\-@]/.test(t)) t = `'${t}`;   // never a formula
      return `<c r="${ref}" t="inlineStr"${s}><is><t xml:space="preserve">${xml(t.slice(0, 32000))}</t></is></c>`;
    }).join('')}</row>`).join('');
    return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>${out}</sheetData></worksheet>`;
  }
  // sheets: [{name, rows: [[header…], [values…], …]}]
  function xlsx(sheets) {
    const used = new Set(), S = sheets.map((s) => ({ ...s, name: sheetName(s.name, used) }));
    return zip([
      { name: '[Content_Types].xml', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>${S.map((_, i) => `<Override PartName="/xl/worksheets/sheet${i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`).join('')}</Types>` },
      { name: '_rels/.rels', data: '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>' },
      { name: 'xl/workbook.xml', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>${S.map((s, i) => `<sheet name="${xml(s.name)}" sheetId="${i + 1}" r:id="rId${i + 1}"/>`).join('')}</sheets></workbook>` },
      { name: 'xl/_rels/workbook.xml.rels', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${S.map((_, i) => `<Relationship Id="rId${i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet${i + 1}.xml"/>`).join('')}<Relationship Id="rId${S.length + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>` },
      { name: 'xl/styles.xml', data: '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>' },
      ...S.map((s, i) => ({ name: `xl/worksheets/sheet${i + 1}.xml`, data: sheetXml(s.rows) })),
    ]);
  }
  // A summary object as key/value rows (nested keys joined with “.”; long lists cut).
  function flat(o, pre = '', out = []) {
    if (o == null || typeof o !== 'object') { out.push([pre, o]); return out; }
    if (Array.isArray(o) && o.every((x) => x == null || typeof x !== 'object')) { out.push([pre, o.join(', ')]); return out; }
    Object.entries(o).slice(0, 400).forEach(([k, v]) => flat(v, pre ? `${pre}.${k}` : k, out)); return out;
  }

  // ── PowerPoint (.pptx): a title slide, the plan, a slide per chart, the reading and the notes ──
  const EMU = 914400, SW = 12192000, SH = 6858000;
  const THEME = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PSI"><a:themeElements><a:clrScheme name="PSI"><a:dk1><a:srgbClr val="0B0B0C"/></a:dk1><a:lt1><a:srgbClr val="F2F1EE"/></a:lt1><a:dk2><a:srgbClr val="1A1A1C"/></a:dk2><a:lt2><a:srgbClr val="A9A69F"/></a:lt2><a:accent1><a:srgbClr val="FFC36B"/></a:accent1><a:accent2><a:srgbClr val="7FB2C9"/></a:accent2><a:accent3><a:srgbClr val="C99BE0"/></a:accent3><a:accent4><a:srgbClr val="9FD39A"/></a:accent4><a:accent5><a:srgbClr val="E08A7F"/></a:accent5><a:accent6><a:srgbClr val="D6C27A"/></a:accent6><a:hlink><a:srgbClr val="7FB2C9"/></a:hlink><a:folHlink><a:srgbClr val="C99BE0"/></a:folHlink></a:clrScheme><a:fontScheme name="PSI"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="PSI"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>';
  const NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"';
  const GRP = '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>';
  const BG = '<p:bg><p:bgPr><a:solidFill><a:srgbClr val="0B0B0C"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>';
  // A text box: paras [{text, size (pt), bold, color}]
  const textBox = (id, x, y, w, h, paras) => `<p:sp><p:nvSpPr><p:cNvPr id="${id}" name="Text ${id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="${x}" y="${y}"/><a:ext cx="${w}" cy="${h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="0" rIns="0"><a:normAutofit/></a:bodyPr><a:lstStyle/>${paras.map((p) => `<a:p><a:pPr><a:spcAft><a:spcPts val="600"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="${Math.round((p.size || 16) * 100)}"${p.bold ? ' b="1"' : ''} dirty="0"><a:solidFill><a:srgbClr val="${p.color || 'F2F1EE'}"/></a:solidFill><a:latin typeface="${p.mono ? 'Consolas' : 'Calibri'}"/></a:rPr><a:t>${xml(p.text)}</a:t></a:r></a:p>`).join('')}</p:txBody></p:sp>`;
  const picture = (id, rid, x, y, w, h) => `<p:pic><p:nvPicPr><p:cNvPr id="${id}" name="Chart ${id}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="${rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="${x}" y="${y}"/><a:ext cx="${w}" cy="${h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>`;
  // slides: [{title, paras: [{text,…}], image: {png, w, h} | null}]
  function pptx(slides) {
    const files = [], imgs = [];
    const sl = slides.map((s, i) => {
      const parts = [], rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'];
      const M = Math.round(0.5 * EMU); let y = M;
      if (s.title) { parts.push(textBox(2, M, y, SW - 2 * M, Math.round(0.9 * EMU), [{ text: s.title, size: s.big ? 32 : 24, bold: true }])); y += Math.round(1.0 * EMU); }
      if (s.image && s.image.png) {
        imgs.push(s.image.png); const n = imgs.length; rels.push(`<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/image${n}.png"/>`);
        const capH = s.paras && s.paras.length ? Math.round(0.8 * EMU) : 0, maxW = SW - 2 * M, maxH = SH - y - M - capH;
        let w = maxW, h = Math.round(w * s.image.h / s.image.w); if (h > maxH) { h = maxH; w = Math.round(h * s.image.w / s.image.h); }
        parts.push(picture(3, 'rId2', Math.round((SW - w) / 2), y, w, h)); y += h + Math.round(0.15 * EMU);
        if (capH) parts.push(textBox(4, M, y, SW - 2 * M, capH, s.paras.map((p) => ({ size: 12, color: 'A9A69F', ...p }))));
      } else if (s.paras && s.paras.length) parts.push(textBox(4, M, y, SW - 2 * M, SH - y - M, s.paras));
      files.push({ name: `ppt/slides/slide${i + 1}.xml`, data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld ${NS}><p:cSld>${BG}<p:spTree>${GRP}${parts.join('')}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>` });
      files.push({ name: `ppt/slides/_rels/slide${i + 1}.xml.rels`, data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${rels.join('')}</Relationships>` });
      return i + 1;
    });
    return zip([
      { name: '[Content_Types].xml', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>${sl.map((n) => `<Override PartName="/ppt/slides/slide${n}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>`).join('')}</Types>` },
      { name: '_rels/.rels', data: '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>' },
      { name: 'ppt/presentation.xml', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation ${NS}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>${sl.map((n) => `<p:sldId id="${255 + n}" r:id="rId${n + 2}"/>`).join('')}</p:sldIdLst><p:sldSz cx="${SW}" cy="${SH}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>` },
      { name: 'ppt/_rels/presentation.xml.rels', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>${sl.map((n) => `<Relationship Id="rId${n + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide${n}.xml"/>`).join('')}</Relationships>` },
      { name: 'ppt/slideMasters/slideMaster1.xml', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster ${NS}><p:cSld>${BG}<p:spTree>${GRP}</p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>` },
      { name: 'ppt/slideMasters/_rels/slideMaster1.xml.rels', data: '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>' },
      { name: 'ppt/slideLayouts/slideLayout1.xml', data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout ${NS} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>${GRP}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>` },
      { name: 'ppt/slideLayouts/_rels/slideLayout1.xml.rels', data: '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>' },
      { name: 'ppt/theme/theme1.xml', data: THEME },
      ...files, ...imgs.map((png, i) => ({ name: `ppt/media/image${i + 1}.png`, data: png })),
    ]);
  }

  // ── the sandbox ──
  // The time and code-size limits come from the guardrails (Settings > Guardrails); 20 s and 20,000 characters by default.
  function runSandbox(code, input, { seconds = 20, codeChars = 20000 } = {}) {
    const LIMIT_MS = seconds * 1000;
    return new Promise((resolve) => {
      let w; try { w = new Worker('sandbox-worker.js'); } catch (e) { resolve({ ok: false, outputs: [], error: `The sandbox could not start: ${e.message}` }); return; }
      const done = (r) => { clearTimeout(t); try { w.terminate(); } catch (_) { /* ended */ } resolve(r); };
      const t = setTimeout(() => done({ ok: false, outputs: [], error: `Stopped: the code ran longer than ${LIMIT_MS / 1000} s.` }), LIMIT_MS);
      w.onmessage = (e) => done(e.data && typeof e.data === 'object' ? e.data : { ok: false, outputs: [], error: 'The sandbox answered with something other than a result.' });
      w.onerror = (e) => { e.preventDefault(); done({ ok: false, outputs: [], error: `The sandbox failed: ${e.message || 'unknown error'}` }); };
      w.postMessage({ code, input, seed: 1, codeChars });
    });
  }

  // ── the analysis results: text, numbers, tables and charts (drawn here, in the app's style) ──
  const num = (v) => typeof v === 'number' && Number.isFinite(v);
  const fmt = (v) => (v == null ? '—' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toPrecision(4) : v.toFixed(4)) : typeof v === 'object' ? JSON.stringify(v) : String(v));
  function canvasFor(box, h) { const cv = el('canvas', 'dz-chart'); box.appendChild(cv); const r = Math.min(2, devicePixelRatio || 1); cv.style.height = `${h}px`; cv.width = (cv.clientWidth || box.clientWidth || 800) * r; cv.height = h * r; const g = cv.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, cv.width, cv.height); return { cv, g, r, W: cv.width, H: cv.height }; }
  const tick = (v) => String(+v.toPrecision(4));   // an axis label: no trailing zeros
  const ticks = (lo, hi, n = 5) => { if (!(hi > lo)) { hi = lo + 1; } const s = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))), st = [1, 2, 5, 10].map((m) => m * s).find((x) => (hi - lo) / x <= n) || s * 10; const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) out.push(+v.toPrecision(12)); return out; };
  function chart(box, spec) {
    const type = spec.type, title = String(spec.title || '').slice(0, 120);
    if (title) box.appendChild(el('p', 'k', title));
    if (type === 'heatmap') {
      const rows = (spec.rows || []).map(String), cols = (spec.cols || []).map(String), V = spec.values || []; if (!rows.length || !cols.length) throw new Error('a heatmap needs rows, cols and values');
      const flatV = V.flat().filter(num), lo = spec.min ?? Math.min(...flatV), hi = spec.max ?? Math.max(...flatV);
      const { g, r, W } = canvasFor(box, Math.min(560, 40 + rows.length * 22 + 40)); const left = 140 * r, top = 10 * r, cw = (W - left - 10 * r) / cols.length, ch = 22 * r;
      rows.forEach((rl, i) => { g.fillStyle = INK3; g.font = font(r, 9); g.textAlign = 'right'; g.fillText(rl.slice(0, 22), left - 6 * r, top + i * ch + ch * 0.65);
        cols.forEach((_, j) => { const v = (V[i] || [])[j]; if (!num(v)) return; const t = hi > lo ? (v - lo) / (hi - lo) : 0.5; g.fillStyle = `rgba(255,195,107,${0.08 + 0.92 * t})`; g.fillRect(left + j * cw + 1, top + i * ch + 1, cw - 2, ch - 2);
          if (cw > 34 * r) { g.fillStyle = t > 0.6 ? '#000' : INK; g.font = font(r, 8); g.textAlign = 'center'; g.fillText(fmt(v).slice(0, 6), left + j * cw + cw / 2, top + i * ch + ch * 0.65); } }); });
      g.fillStyle = INK3; g.font = font(r, 9); g.textAlign = 'center'; cols.forEach((c, j) => { if (cols.length <= 40 || j % Math.ceil(cols.length / 40) === 0) g.fillText(c.slice(0, 8), left + j * cw + cw / 2, top + rows.length * ch + 14 * r); });
      return;
    }
    const series = (spec.series || []).slice(0, 8); if (!series.length) throw new Error('a chart needs at least one series');
    const { g, r, W, H } = canvasFor(box, 260), L = 54 * r, Rr = 14 * r, T = 14 * r, B = 38 * r;
    let xs, ys = [];
    if (type === 'scatter') { xs = series.flatMap((s) => (s.points || []).map((p) => p[0])).filter(num); ys = series.flatMap((s) => (s.points || []).map((p) => p[1])).filter(num); }
    else { ys = series.flatMap((s) => (s.values || []).filter(num)); }
    if (type === 'bar' || type === 'line') ys.push(0);
    if (Array.isArray(spec.band)) ys.push(...spec.band.flat().filter(num));
    let lo = Math.min(...ys), hi = Math.max(...ys); if (!(hi > lo)) { hi = lo + 1; }
    const yt = ticks(lo, hi); lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);
    const Y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    g.font = font(r, 9); g.textAlign = 'right'; yt.forEach((v) => { g.strokeStyle = 'rgba(242,241,238,.08)'; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - Rr, Y(v)); g.stroke(); g.fillStyle = INK3; g.fillText(tick(v), L - 6 * r, Y(v) + 3 * r); });
    if (type === 'scatter') {
      let xl = Math.min(...xs), xh = Math.max(...xs); if (!(xh > xl)) xh = xl + 1; const xt = ticks(xl, xh); xl = Math.min(xl, xt[0]); xh = Math.max(xh, xt[xt.length - 1]);
      const X = (v) => L + (W - L - Rr) * (v - xl) / (xh - xl);
      g.textAlign = 'center'; xt.forEach((v) => { g.fillStyle = INK3; g.fillText(tick(v), X(v), H - B + 14 * r); });
      series.forEach((s, k) => { g.fillStyle = PALETTE[k % PALETTE.length]; (s.points || []).forEach((p) => { if (num(p[0]) && num(p[1])) { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 3.5 * r, 0, Math.PI * 2); g.fill(); if (p[2] != null) { g.fillStyle = INK3; g.font = font(r, 8); g.textAlign = 'left'; g.fillText(String(p[2]).slice(0, 18), X(p[0]) + 5 * r, Y(p[1]) - 4 * r); g.fillStyle = PALETTE[k % PALETTE.length]; } } }); });
    } else {
      const labels = (spec.labels || spec.x || series[0].values.map((_, i) => i + 1)).map(String), n = labels.length;
      if (type === 'bar') {
        const gw = (W - L - Rr) / n, bw = Math.max(2 * r, (gw * 0.75) / series.length);
        series.forEach((s, k) => (s.values || []).forEach((v, i) => { if (!num(v)) return; g.fillStyle = PALETTE[k % PALETTE.length]; const x = L + i * gw + gw * 0.125 + k * bw, y0 = Y(0), y1 = Y(v); g.fillRect(x, Math.min(y0, y1), bw - 1, Math.abs(y1 - y0)); }));
        if (Array.isArray(spec.errors)) series.forEach((s, k) => (spec.errors[k] || []).forEach((e, i) => { if (!e || !num(e[0]) || !num(e[1])) return; const x = L + i * gw + gw * 0.125 + k * bw + bw / 2; g.strokeStyle = INK; g.beginPath(); g.moveTo(x, Y(e[0])); g.lineTo(x, Y(e[1])); g.stroke(); }));
        g.fillStyle = INK3; g.textAlign = 'center'; labels.forEach((l, i) => { if (n <= 24 || i % Math.ceil(n / 24) === 0) g.fillText(l.slice(0, Math.max(4, Math.floor(gw / (6 * r)))), L + i * gw + gw / 2, H - B + 14 * r); });
      } else {
        const X = (i) => L + (n > 1 ? (W - L - Rr) * i / (n - 1) : (W - L - Rr) / 2);
        if (Array.isArray(spec.band)) { g.fillStyle = 'rgba(242,241,238,.08)'; g.beginPath(); spec.band.forEach((b, i) => { if (b && num(b[1])) g.lineTo(X(i), Y(b[1])); }); for (let i = spec.band.length - 1; i >= 0; i--) { const b = spec.band[i]; if (b && num(b[0])) g.lineTo(X(i), Y(b[0])); } g.fill(); }
        series.forEach((s, k) => { g.strokeStyle = PALETTE[k % PALETTE.length]; g.lineWidth = 1.8 * r; g.beginPath(); let on = false; (s.values || []).forEach((v, i) => { if (!num(v)) { on = false; return; } if (on) g.lineTo(X(i), Y(v)); else { g.moveTo(X(i), Y(v)); on = true; } }); g.stroke(); });
        g.fillStyle = INK3; g.textAlign = 'center'; labels.forEach((l, i) => { if (n <= 20 || i % Math.ceil(n / 20) === 0) g.fillText(l.slice(0, 8), X(i), H - B + 14 * r); });
      }
    }
    if (spec.xlabel || spec.ylabel) { g.fillStyle = INK3; g.font = font(r, 9); g.textAlign = 'center'; if (spec.xlabel) g.fillText(String(spec.xlabel).slice(0, 80), (L + W) / 2, H - 6 * r); if (spec.ylabel) { g.save(); g.translate(12 * r, (T + H - B) / 2); g.rotate(-Math.PI / 2); g.fillText(String(spec.ylabel).slice(0, 50), 0, 0); g.restore(); } }
    if (series.length > 1 || series[0].name) { const lg = el('p', 'small dz-legend'); series.forEach((s, k) => { const sw = el('span', 'dz-sw'); sw.style.background = PALETTE[k % PALETTE.length]; lg.append(sw, ` ${String(s.name || `series ${k + 1}`).slice(0, 40)}   `); }); box.appendChild(lg); }
  }
  // One analysis, shown: who wrote it and when, its code (open it to read it), and everything it showed.
  function showAnalysis(box, a) {
    const card = el('div', 'out minh-0 dz-analysis');
    card.appendChild(el('p', 'small muted', `${a.by === 'prometheus' ? 'written by Prometheus' : 'written by you'} · ${new Date(a.at).toLocaleString()}${a.ms != null ? ` · ran in ${a.ms} ms` : ''}${a.ok === false ? ' · stopped by an error' : ''}`));
    if (a.purpose) card.appendChild(el('p', null, a.purpose));
    const det = el('details', 'dz-code'); det.appendChild(el('summary', 'small', 'the code')); det.appendChild(el('pre', 'dz-detail', a.code)); card.appendChild(det);
    for (const o of a.outputs || []) {
      try {
        if (o.kind === 'text') card.appendChild(el('p', null, o.text));
        else if (o.kind === 'number') { const p = el('p'); p.appendChild(el('b', null, `${o.label}: `)); p.append(fmt(o.value)); if (o.note) p.append(` (${o.note})`); card.appendChild(p); }
        else if (o.kind === 'table') {
          if (o.title) card.appendChild(el('p', 'k', o.title));
          const rows = o.rows || [], cols = [...new Set(rows.flatMap((x) => (x && typeof x === 'object' ? Object.keys(x) : ['value'])))].slice(0, 30);
          const wrap = el('div', 'scroll dz-grid'), tb = el('table', 'dz-gridt'), hr = el('tr'); cols.forEach((c) => hr.appendChild(el('th', null, c))); tb.appendChild(hr);
          rows.forEach((x) => { const tr = el('tr'); cols.forEach((c) => tr.appendChild(el('td', typeof (x && x[c]) === 'number' ? 'num' : null, fmt(x && typeof x === 'object' ? x[c] : x)))); tb.appendChild(tr); });
          wrap.appendChild(tb); card.appendChild(wrap); if (o.total > rows.length) card.appendChild(el('p', 'small muted', `first ${rows.length} of ${o.total} rows`));
        } else if (o.kind === 'chart') chart(card, o.spec || {});
      } catch (e) { card.appendChild(el('p', 'small cond-warn', `This output could not be shown: ${e.message}`)); }
    }
    if (a.value != null) { const p = el('pre', 'dz-detail dz-value'); p.textContent = typeof a.value === 'string' ? a.value : JSON.stringify(a.value, null, 1).slice(0, 6000); card.appendChild(el('p', 'k', 'returned')); card.appendChild(p); }
    if (a.error) card.appendChild(el('p', 'small cond-warn', a.error));
    box.appendChild(card); return card;
  }
  // The reference shown under the code box (and given to Prometheus in its tool description).
  const REFERENCE = [
    'data: the experiment’s table, one object per row: prompt, the factors, measure, layer, token, piece, neuron, value, note.',
    'experiments: when several experiments are given, [{id, name, plan, summary, data}] (data as above).',
    'vectors: for a Layer vectors experiment, [{prompt, text, <each factor as its own key, e.g. v.reality>, factors: {the same factors}, layers: [[numbers] per layer]}] (layer 0 = embeddings). A probe at layer l: stats.probe(vectors.map(v => v.layers[l]), vectors.map(v => v.factors[f])).',
    'stats: mean, sd, median, quantile, ci(values) → {mean, lo, hi}, ttest(a, b) (Welch), pairedT(a, b), permutationTest(a, b, {n, seed}), bootstrapCI(values, fn, {n}), cohenD, corr, spearman, linreg(x, y), groupBy(rows, keys), summarize(rows, keys) → mean ± 95% CI per group, pivot(rows, rowKey, colKey), cosine(a, b), cka(X, Y), pca(X, k), probe(X, labels, {folds, permutations}) → cross-validated accuracy and chance.',
    'show.text(t), show.number(label, value, note), show.table(title, rows), show.chart({type: "bar" | "line" | "scatter" | "heatmap", title, labels, series: [{name, values}] (scatter: points [[x, y, label]]), errors, band, rows/cols/values for a heatmap, xlabel, ylabel}). What the code returns is shown too.',
    'Nothing can be loaded or sent: no network, no files, no “import”. Math.random is seeded; rand(seed) gives another seeded generator. Time and code size are limited (Settings > Guardrails).',
  ];
  const EXAMPLES = {
    tool: '// The primary number by the first factor, with a 95% interval, and a test between the first two groups.\nconst m = data.filter(r => r.measure === data.find(x => x.measure !== "output" && x.value != null).measure);\nconst f = Object.keys(m[0]).find(k => !["prompt","measure","layer","token","piece","neuron","value","note"].includes(k));\nconst rows = stats.summarize(m, f);\nshow.table(`${m[0].measure} by ${f}`, rows);\nshow.chart({ type: "bar", title: `${m[0].measure} by ${f} (mean, 95% interval)`, labels: rows.map(r => r.group), series: [{ name: m[0].measure, values: rows.map(r => r.mean) }], errors: [rows.map(r => [r.ci_lo, r.ci_hi])] });\nconst g = Object.values(stats.groupBy(m, f));\nif (g.length >= 2) show.number("Welch t-test p (first two groups)", stats.ttest(g[0].map(r => r.value), g[1].map(r => r.value)).p);',
    read: '// Total surprise by the first factor, with a 95% interval.\nconst t = data.filter(r => r.measure === "total_bits");\nconst f = Object.keys(t[0]).find(k => !["prompt","measure","layer","token","piece","neuron","value","note"].includes(k));\nconst rows = stats.summarize(t, f);\nshow.table("total surprise (bits)", rows);\nshow.chart({ type: "bar", title: "total surprise by " + f, labels: rows.map(r => r.group), series: [{ name: "bits", values: rows.map(r => r.mean) }], errors: [rows.map(r => [r.ci_lo, r.ci_hi])] });',
    neural: '// How sure the lens is at each layer (first answer token), averaged by the first factor.\nconst f = Object.keys(data[0]).find(k => !["prompt","measure","layer","token","piece","neuron","value","note"].includes(k));\nconst lens = data.filter(r => r.measure === "lens_p" && r.token === 1);\nconst layers = [...new Set(lens.map(r => r.layer))].sort((a, b) => a - b);\nconst groups = stats.groupBy(lens, f);\nshow.chart({ type: "line", title: "lens confidence by layer (first answer token)", labels: layers, xlabel: "layer", series: Object.entries(groups).map(([k, g]) => ({ name: k, values: layers.map(l => stats.mean(g.filter(r => r.layer === l).map(r => r.value))) })) });',
    vectors: '// Can a linear probe read the first factor from each layer? Cross-validated accuracy against chance.\nconst f = Object.keys(vectors[0]).find(k => !["prompt","text","layers","factors","dims"].includes(k));\nconst labels = vectors.map(v => v[f]);\n// The control: the same probe on shuffled labels, averaged over 5 shuffles (it should stay near chance).\nconst g = rand(7), shufs = [...Array(5)].map(() => { const s = labels.slice(); for (let j = s.length - 1; j > 0; j--) { const q = Math.floor(g() * (j + 1)); [s[j], s[q]] = [s[q], s[j]]; } return s; });\nconst L = vectors[0].layers.length, acc = [], chance = [], ctl = [];\nfor (let l = 0; l < L; l++) { const X = vectors.map(v => v.layers[l]); const r = stats.probe(X, labels, { folds: 5 }); acc.push(r.accuracy); chance.push(r.chance); ctl.push(stats.mean(shufs.map((s) => stats.probe(X, s, { folds: 5 }).accuracy))); }\nshow.chart({ type: "line", title: `probe accuracy for “${f}” by layer`, labels: [...Array(L).keys()], xlabel: "layer (0 = embeddings)", series: [{ name: "probe", values: acc }, { name: "shuffled labels", values: ctl }, { name: "chance", values: chance }] });\nconst best = acc.indexOf(Math.max(...acc));\nshow.number("best layer", best, `probe ${Math.round(acc[best] * 100)}% · shuffled labels ${Math.round(ctl[best] * 100)}% (mean of 5) · chance ${Math.round(chance[best] * 100)}% · ${vectors.length} prompts`);\n// How alike are the layers? Linear CKA between every pair of layers.\nconst step = Math.max(1, Math.round(L / 12)), ls = [...Array(L).keys()].filter(l => l % step === 0);\nshow.chart({ type: "heatmap", title: "CKA between layers", rows: ls.map(String), cols: ls.map(String), values: ls.map(a => ls.map(b => stats.cka(vectors.map(v => v.layers[a]), vectors.map(v => v.layers[b])))) , min: 0, max: 1 });',
  };
  const exampleFor = (plan) => (plan.instrument === 'vectors' ? EXAMPLES.vectors : C.TOOLS[plan.instrument] ? EXAMPLES.tool : plan.instrument === 'read' ? EXAMPLES.read : EXAMPLES.neural);

  window.PSIData = { zip, xml, pngOf, tableOf, vectorsOf, grid, xlsx, flat, pptx, runSandbox, showAnalysis, chart, REFERENCE, exampleFor };
})();
