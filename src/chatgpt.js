// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The Emergence Machine (https://emergencemachine.com)
'use strict';
// Prometheus on the user's ChatGPT plan ("Sign in with ChatGPT", developers.openai.com/siwc): a second backend beside
// OpenRouter. The user signs in once in their own browser; the app gets an OAuth token that lets eligible Responses API
// requests count against their ChatGPT plan instead of money. Nothing is spent, so the spending cap does not apply.
//   sign-in   OAuth code flow with PKCE, redirect to http://127.0.0.1:<port>/callback (a one-shot server here), client
//             registered dynamically on first sign-in (client_id=dynamic_agent_client); the ID token is checked against
//             OpenAI's keys (issuer, audience, expiry, nonce) before anything is kept
//   storage   tokens in the secrets store (secrets.js: the OS store in the desktop app, a 0600 file in the dev server),
//             the issued client id, account and host id in <data>/chatgpt.json; the page sees neither token
//   requests  POST /v1/responses with store:false, stream:true; no temperature, no output limit, no system role
//             (instructions go in as a developer message): the plan's rules, not ours. A requested temperature or
//             length is not sent; each answer reports it beside what was used (sampling), so experiments can say so
// Chat Completions shaped messages and tools come in (what helper.js and help-prometheus.js send) and the answer goes back
// in the same shape, so Prometheus itself does not change.
const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');

const AUTH = 'https://auth.openai.com';
const AUTHORIZE = `${AUTH}/api/accounts/authorize`, TOKEN = `${AUTH}/api/accounts/oauth/token`, REVOKE = `${AUTH}/api/accounts/oauth/revoke`, JWKS = `${AUTH}/.well-known/jwks.json`;
const API = 'https://api.openai.com/v1';
const SCOPE = 'openid profile email offline_access resource.invoke chatgpt.tokens.use.direct';
const PLAN_SCOPE = 'chatgpt.tokens.use.direct';
const MANAGE = 'https://chatgpt.com/settings/usage?tab=overview';   // ChatGPT's usage page: what the plan has used
const PREFIX = 'chatgpt/';
const b64u = (b) => Buffer.from(b).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const unb64u = (s) => Buffer.from(String(s).replace(/-/g, '+').replace(/_/g, '/'), 'base64');
const rand = (n = 32) => b64u(crypto.randomBytes(n));

function createChatgpt({ dataDir, secrets, open = null, log = () => {} }) {
  const STATE = path.join(dataDir, 'chatgpt.json');
  const readState = () => { try { return JSON.parse(fs.readFileSync(STATE, 'utf8')) || {}; } catch (_) { return {}; } };
  const writeState = (s) => { fs.writeFileSync(STATE, JSON.stringify(s, null, 1), { mode: 0o600 }); };
  const tokens = () => { try { return JSON.parse(secrets.get('chatgpt-tokens') || 'null'); } catch (_) { return null; } };
  const keep = (t) => secrets.set('chatgpt-tokens', t ? JSON.stringify(t) : '');
  // A stable, opaque id for this computer's install, sent with every sign-in (ext_agent_host_id).
  const hostId = () => { const s = readState(); if (!/^urn:uuid:/.test(s.hostId || '')) { s.hostId = `urn:uuid:${crypto.randomUUID()}`; writeState(s); } return s.hostId; };

  let pending = null;   // the sign-in in progress: { url, server, timer, error }
  let lastError = '';

  // ── ID token: RS256 against OpenAI's published keys ──
  let KEYS = null, KEYS_AT = 0;
  async function jwks(force) {
    if (!force && KEYS && Date.now() - KEYS_AT < 3600e3) return KEYS;
    const r = await fetch(JWKS); if (!r.ok) throw new Error(`OpenAI keys: HTTP ${r.status}`);
    KEYS = (await r.json()).keys || []; KEYS_AT = Date.now(); return KEYS;
  }
  async function checkIdToken(jwt, { clientId, nonce }) {
    const [h, p, s] = String(jwt || '').split('.'); if (!s) throw new Error('no ID token in the answer');
    const head = JSON.parse(unb64u(h)), claims = JSON.parse(unb64u(p));
    if (head.alg !== 'RS256') throw new Error(`ID token: unexpected algorithm ${head.alg}`);
    let jwk = (await jwks()).find((k) => k.kid === head.kid); if (!jwk) jwk = (await jwks(true)).find((k) => k.kid === head.kid);
    if (!jwk) throw new Error('ID token: signed with a key OpenAI does not publish');
    const ok = crypto.verify('RSA-SHA256', Buffer.from(`${h}.${p}`), crypto.createPublicKey({ key: jwk, format: 'jwk' }), unb64u(s));
    if (!ok) throw new Error('ID token: bad signature');
    const now = Date.now() / 1000;
    if (claims.iss !== AUTH) throw new Error(`ID token: issuer ${claims.iss}`);
    if (!(Array.isArray(claims.aud) ? claims.aud.includes(clientId) : claims.aud === clientId)) throw new Error('ID token: not issued to this app');
    if (!(claims.exp > now - 60)) throw new Error('ID token: expired');
    if (claims.nonce !== nonce) throw new Error('ID token: nonce does not match this sign-in');
    if (!claims.sub) throw new Error('ID token: no account id');
    return claims;
  }

  async function tokenCall(form) {
    const r = await fetch(TOKEN, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' }, body: new URLSearchParams(form).toString() });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) { const e = new Error(`ChatGPT sign-in: ${d.error_description || d.error || `HTTP ${r.status}`}`); e.code = d.error; throw e; }
    return d;
  }
  const scopesOf = (d, prev) => String(d.scope || (prev && prev.scope) || '').split(/\s+/).filter(Boolean);

  // ── sign-in ──
  function stopPending() { if (pending) { clearTimeout(pending.timer); try { pending.server.close(); } catch (_) { /* closed */ } } pending = null; }
  async function signIn() {
    stopPending(); lastError = '';
    const st = readState(), verifier = rand(48), state = rand(24), nonce = rand(24);
    const server = http.createServer();
    await new Promise((res, rej) => { server.once('error', rej); server.listen(0, '127.0.0.1', res); });
    const redirect = `http://127.0.0.1:${server.address().port}/callback`;
    const first = !st.clientId;
    const q = { client_id: st.clientId || 'dynamic_agent_client', response_type: 'code', redirect_uri: redirect, scope: SCOPE, resource: `${API}`,
      state, nonce, code_challenge_method: 'S256', code_challenge: b64u(crypto.createHash('sha256').update(verifier).digest()), ext_agent_host_id: hostId() };
    if (first) q.agent_name_hint = 'Parameter Space Inspector';
    const tk = tokens(); if (tk && tk.id_token) q.id_token_hint = tk.id_token;   // skips the account chooser for the same account
    const url = `${AUTHORIZE}?${new URLSearchParams(q).toString()}`;
    const page = (res, code, title, text) => { res.writeHead(code, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(`<!doctype html><meta charset="utf-8"><title>${title}</title><body style="font:16px system-ui;max-width:32rem;margin:4rem auto;color:#222"><h2>${title}</h2><p>${text}</p></body>`); };
    server.on('request', async (req, res) => {
      const u = new URL(req.url, redirect);
      if (u.pathname !== '/callback') return page(res, 404, 'Not found', '');
      const p = u.searchParams;
      if (p.get('state') !== state) return page(res, 400, 'Sign-in not recognised', 'This answer does not belong to the sign-in the Inspector started. Start again from Settings.');
      try {
        if (p.get('error')) throw new Error(`ChatGPT sign-in: ${p.get('error_description') || p.get('error')}`);
        const clientId = p.get('client_id') || st.clientId; if (!clientId) throw new Error('ChatGPT sign-in: no client id came back');
        const code = p.get('code'); if (!code) throw new Error('ChatGPT sign-in: no code came back');
        const d = await tokenCall({ grant_type: 'authorization_code', client_id: clientId, code, code_verifier: verifier, redirect_uri: redirect, resource: API });
        const claims = await checkIdToken(d.id_token, { clientId, nonce });
        const scopes = scopesOf(d);
        keep({ access_token: d.access_token, refresh_token: d.refresh_token, id_token: d.id_token, scope: scopes.join(' '), expires_at: Date.now() + (+d.expires_in || 3600) * 1000 });
        writeState({ ...readState(), clientId, subject: claims.sub, email: claims.email || '', name: claims.name || '', planUsage: scopes.includes(PLAN_SCOPE), signedInAt: new Date().toISOString(), models: null });
        log(`signed in to ChatGPT${claims.email ? ` as ${claims.email}` : ''}; plan usage ${scopes.includes(PLAN_SCOPE) ? 'granted' : 'not granted'}`);
        page(res, 200, 'Signed in', `The Parameter Space Inspector can now use your ChatGPT plan${scopes.includes(PLAN_SCOPE) ? '' : ', but plan usage was not granted, so Prometheus cannot run on it yet'}. You can close this tab.`);
      } catch (e) { lastError = e.message; log(`ChatGPT sign-in failed: ${e.message}`); page(res, 400, 'Sign-in failed', String(e.message).replace(/[<>&]/g, '')); }
      finally { setTimeout(stopPending, 200); }
    });
    pending = { url, server, timer: setTimeout(() => { lastError = 'The sign-in was not finished within 10 minutes; start again.'; stopPending(); }, 600e3) };
    let opened = false; if (open) try { await open(url); opened = true; } catch (_) { /* the page opens it */ }
    return { url, opened, redirect };
  }

  // A fresh access token: refreshed a minute before it expires. A refresh token that no longer works ends the session.
  let refreshing = null;
  async function access() {
    const t = tokens(); if (!t || !t.access_token) throw new Error('Not signed in to ChatGPT: Settings > Prometheus > Continue with ChatGPT.');
    if (t.expires_at - Date.now() > 60e3) return t;
    if (!refreshing) refreshing = (async () => {
      const st = readState();
      try {
        const d = await tokenCall({ grant_type: 'refresh_token', refresh_token: t.refresh_token, client_id: st.clientId, resource: API });
        const scopes = scopesOf(d, t);
        const n = { ...t, access_token: d.access_token, refresh_token: d.refresh_token || t.refresh_token, id_token: d.id_token || t.id_token, scope: scopes.join(' '), expires_at: Date.now() + (+d.expires_in || 3600) * 1000 };
        keep(n); writeState({ ...readState(), planUsage: scopes.includes(PLAN_SCOPE) }); return n;
      } catch (e) {
        if (/invalid_grant|invalid_refresh_token|token_expired|refresh_token_(expired|invalidated|reused)/.test(e.code || e.message)) { keep(null); throw new Error('The ChatGPT session ended: sign in again in Settings > Prometheus.'); }
        throw e;
      } finally { refreshing = null; }
    })();
    return refreshing;
  }

  function status() {
    const st = readState(), t = tokens();
    return { signedIn: !!(t && t.refresh_token), email: st.email || '', name: st.name || '', planUsage: !!st.planUsage, signedInAt: st.signedInAt || null,
      pending: pending ? { url: pending.url } : null, error: lastError || null, storage: secrets.where(),
      manageUsage: MANAGE, models: st.models || null };
  }

  async function signOut() {
    stopPending(); const t = tokens(), st = readState();
    if (t && t.refresh_token && st.clientId) {
      try { await fetch(REVOKE, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ token: t.refresh_token, token_type_hint: 'refresh_token', client_id: st.clientId }).toString() }); }
      catch (e) { log(`ChatGPT sign-out: could not reach OpenAI to revoke (${e.message}); the tokens are deleted here anyway`); }
    }
    keep(null);
    // The issued client id stays: OpenAI asks apps to reuse it when the same account signs in again.
    writeState({ clientId: st.clientId, hostId: st.hostId, subject: st.subject });
    log('signed out of ChatGPT'); return status();
  }

  async function models() {
    const t = await access();
    const r = await fetch(`${API}/models`, { headers: { Authorization: `Bearer ${t.access_token}` } });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(`ChatGPT models: ${(d.error && (d.error.message || d.error.code)) || `HTTP ${r.status}`}`);
    const list = (d.models || d.data || []).filter((m) => !m.visibility || m.visibility === 'list')
      .map((m) => ({ id: PREFIX + (m.slug || m.id), name: m.display_name || m.slug || m.id }));
    writeState({ ...readState(), models: list }); return { models: list };
  }

  // ── chat: Chat Completions shape in, Responses API over SSE, Chat Completions shape out ──
  function toInput(messages) {
    const input = [];
    for (const m of messages || []) {
      const text = typeof m.content === 'string' ? m.content : Array.isArray(m.content) ? m.content.map((c) => c.text || '').join('') : '';
      if (m.role === 'system' || m.role === 'developer') input.push({ role: 'developer', content: text });   // system items are refused on the plan
      else if (m.role === 'user') input.push({ role: 'user', content: text });
      else if (m.role === 'assistant') {
        if (text) input.push({ role: 'assistant', content: text });
        for (const tc of m.tool_calls || []) input.push({ type: 'function_call', call_id: tc.id, name: tc.function.name, arguments: tc.function.arguments || '{}' });
      } else if (m.role === 'tool') input.push({ type: 'function_call_output', call_id: m.tool_call_id, output: text });
    }
    return input;
  }
  const toTools = (tools) => (tools || []).map((t) => (t.type === 'function' && t.function ? { type: 'function', name: t.function.name, description: t.function.description || '', parameters: t.function.parameters || { type: 'object', properties: {} } } : t));
  const PLAN_ERR = {
    subscription_sharing_usage_limit_exceeded: `Your ChatGPT plan's usage limit is reached. Manage usage: ${MANAGE}`,
    subscription_sharing_usage_unavailable: 'ChatGPT plan usage is unavailable right now; try again in a few minutes.',
    subscription_sharing_user_unavailable: 'ChatGPT plan usage is unavailable right now; try again in a few minutes.',
    subscription_sharing_user_not_eligible: 'This ChatGPT account is not eligible for plan usage in other apps (OpenAI names Plus and Pro).',
    subscription_sharing_invalid_user: 'The ChatGPT session is no longer valid: sign in again in Settings > Prometheus.',
    chatpass_v2_scope_not_authorized: 'Plan usage was not granted at sign-in: sign in again and allow it.',
  };
  async function chat(args) {
    if (!(tokens() || {}).refresh_token) throw new Error('Not signed in to ChatGPT: Settings > Prometheus > Continue with ChatGPT.');
    if (!readState().planUsage) throw new Error('ChatGPT plan usage was not granted at sign-in, so Prometheus cannot run on it. Sign in again in Settings > Prometheus and allow it.');
    const model = String(args.model || '').slice(PREFIX.length); if (!model) throw new Error('No ChatGPT model chosen.');
    const body = { model, input: toInput(args.messages || [{ role: 'user', content: args.prompt || '' }]), store: false, stream: true };
    if (Array.isArray(args.tools) && args.tools.length) body.tools = toTools(args.tools);
    const t0 = Date.now();
    const send = async () => { const t = await access(); return fetch(`${API}/responses`, { method: 'POST', headers: { Authorization: `Bearer ${t.access_token}`, 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify(body) }); };
    let r = await send();
    if (r.status === 401) { const t = tokens(); if (t) { t.expires_at = 0; keep(t); } r = await send(); }   // one retry on a fresh token
    if (!r.ok) {
      const d = await r.json().catch(() => ({})), code = (d.error && (d.error.code || d.error.type)) || '';
      throw new Error(PLAN_ERR[code] || `ChatGPT ${r.status}: ${(d.error && d.error.message) || code || r.statusText}`);
    }
    // Server-sent events: keep finished output items; trust the answer only after response.completed.
    const items = []; let done = null, failed = null, buf = '';
    const dec = new TextDecoder();
    const handle = (ev) => {
      if (ev.type === 'response.output_item.done' && ev.item) items.push(ev.item);
      else if (ev.type === 'response.completed') done = ev.response || {};
      else if (ev.type === 'response.failed' || ev.type === 'error') failed = (ev.response && ev.response.error) || ev.error || ev;
      else if (ev.type === 'response.incomplete') failed = { code: 'incomplete', message: `the answer stopped early (${((ev.response || {}).incomplete_details || {}).reason || 'no reason given'})` };
    };
    for await (const chunk of r.body) {
      buf += dec.decode(chunk, { stream: true });
      let i; while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        const data = block.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).join('\n');
        if (data && data !== '[DONE]') { try { handle(JSON.parse(data)); } catch (_) { /* not JSON: skipped */ } }
      }
    }
    if (failed) throw new Error(PLAN_ERR[failed.code] || `ChatGPT: ${failed.message || failed.code || 'the request failed'}`);
    if (!done) throw new Error('ChatGPT: the stream ended without a completed answer; nothing was used.');
    const out = items.length ? items : (done.output || []);
    const text = out.filter((it) => it.type === 'message').flatMap((it) => it.content || []).filter((c) => c.type === 'output_text').map((c) => c.text).join('');
    const calls = out.filter((it) => it.type === 'function_call').map((it) => ({ id: it.call_id, type: 'function', function: { name: it.name, arguments: it.arguments || '{}' } }));
    const msg = { role: 'assistant', content: text || null }; if (calls.length) msg.tool_calls = calls;
    const u = done.usage || null;
    // The plan sets temperature and answer length itself: what was asked for is reported beside what was used, never hidden.
    const sampling = { temperature: 'the plan’s default (not settable)', max_tokens: 'the plan’s default (not settable)', requested_temperature: args.temperature ?? null, requested_max_tokens: args.maxTokens ?? null };
    return { model: args.model, text, toolCalls: calls.length ? calls : null, message: args.tools ? msg : undefined, finish: calls.length ? 'tool_calls' : 'stop', cost: 0, plan: true, sampling,
      cost_source: 'ChatGPT plan (no money; counts against the plan\'s usage limits)', manageUsage: MANAGE,
      tokens: u ? { in: u.input_tokens, out: u.output_tokens } : null, ms: Date.now() - t0 };
  }

  async function call(a = {}) {
    if (a.action === 'status') return status();
    if (a.action === 'signin') return signIn();
    if (a.action === 'cancel') { stopPending(); return status(); }
    if (a.action === 'signout') return signOut();
    if (a.action === 'models') return models();
    throw new Error(`unknown chatgpt action ${a.action}`);
  }
  const isPlanModel = (m) => typeof m === 'string' && m.startsWith(PREFIX);
  return { call, chat, isPlanModel, status, PREFIX, MANAGE, _test: { toInput, toTools, checkIdToken } };
}

module.exports = { createChatgpt };
