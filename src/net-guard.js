// SPDX-License-Identifier: Apache-2.0
// Node 24.22.1 (and Electron 44.10.0, which carries it) on macOS: the fetch() built into Node calls
// socket.setTypeOfService(0) before writing each request, and when that socket has just been torn down (a request
// aborted by its timeout while the other side was slow, as when the inside engine is loading) the call fails with
// EINVAL. Node throws that error in a socket callback, where nothing can catch it, and the whole process exits.
// Node already treats the call as best effort on Windows; this does the same everywhere, so a dead socket fails
// its own request (which the caller handles) instead of killing the app. Loaded first by main.js and devserver.js.
'use strict';
const net = require('net');
const orig = net.Socket.prototype.setTypeOfService;
if (typeof orig === 'function' && !orig.psiGuarded) {
  const guarded = function setTypeOfService(tos) {
    try { return orig.call(this, tos); } catch (e) { if (e && e.syscall === 'setTypeOfService') return this; throw e; }
  };
  guarded.psiGuarded = true;
  net.Socket.prototype.setTypeOfService = guarded;
}
