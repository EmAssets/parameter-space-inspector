// SPDX-License-Identifier: Apache-2.0
// The window's only door to the engine: a fixed list of calls, no Node access.
'use strict';
const { contextBridge, ipcRenderer } = require('electron');

const call = async (name, ...args) => {
  const r = await ipcRenderer.invoke(`psi:${name}`, ...args);
  if (!r.ok) throw new Error(r.error);
  return r.value;
};
const NAMES = ['readHead', 'readAll', 'pickModel', 'pickControlVector', 'settings', 'start', 'stop', 'tokenize', 'detokenize', 'next', 'complete', 'score', 'template', 'engineState', 'trace', 'traceStatus', 'embed', 'represent', 'idle', 'quiet', 'scoreText', 'ablate', 'understand', 'ckptList', 'ckptAsk', 'ckptTest', 'heads', 'ablateHeads', 'lesion', 'sense', 'fight', 'connector', 'setKey', 'listModels', 'external', 'traceRestart', 'anatomy', 'mcp', 'neuronNames', 'setupCheck', 'ckptCount', 'lineage', 'ladder', 'gpu', 'convert', 'convertStatus', 'experiments', 'jobs', 'sae', 'vectors', 'timeline', 'commons', 'chatgpt', 'hf', 'savePdf', 'saveFile'];
const api = Object.fromEntries(NAMES.map((n) => [n, (...a) => call(n, ...a)]));
api.on = (channel, fn) => {
  if (!['log', 'progress', 'ladder'].includes(channel)) return () => {};
  const h = (_e, data) => fn(data);
  ipcRenderer.on(`psi:${channel}`, h);
  return () => ipcRenderer.removeListener(`psi:${channel}`, h);
};
contextBridge.exposeInMainWorld('psi', api);
