import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright-core');

const ROOT = dirname(fileURLToPath(import.meta.url));
const EXT_DIR = join(ROOT, 'extension');
const PROFILE_DIR = join(ROOT, 'agent-profile');
const STAGE_PORT = 9222;
const BRAVE_PORT = 9223;

const control = { port: STAGE_PORT, label: 'stage' };

export function getControlTarget() {
  return { ...control };
}

export function setControlTarget(port, label) {
  control.port = port;
  control.label = label || String(port);
  if (session && session.port !== port) {
    try { session.browser.close(); } catch {}
    session = null;
  }
}

const CHROME_PATHS = [
  process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1223\\chrome-win64\\chrome.exe',
  process.env.LOCALAPPDATA + '\\ms-playwright\\chromium-1223\\chrome-win\\chrome.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
];

let session = null;
let lastUrl = 'https://www.chess.com/play/computer';

async function hardRecover() {
  try { if (session) await session.browser.close().catch(() => {}); } catch {}
  session = null;
  GetKill();
  launchStandaloneChrome();
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (await isDebuggerAlive()) break;
    await new Promise(r => setTimeout(r, 500));
  }
}

function GetKill() {
  try {
    const { execSync } = require('node:child_process');
    const out = execSync('wmic process where "name=\'chrome.exe\'" get ProcessId,ExecutablePath /format:csv', { encoding: 'utf8', timeout: 15000 }).toString();
    out.split('\n').forEach(line => {
      if (line.includes('ms-playwright')) {
        const m = line.trim().match(/(\d+)\s*$/);
        if (m) { try { process.kill(+m[1]); } catch {} }
      }
    });
  } catch {}
}

export function getSession() {
  return session;
}

export function sessionStatus() {
  if (!session) return { active: false };
  return {
    active: true,
    url: session.page ? session.page.url() : null,
    startedAt: session.startedAt,
    steps: session.log.length,
    needsAssist: session.needsAssist || false,
    lastError: session.lastError || null
  };
}

export function sessionLog(n = 12) {
  if (!session) return [];
  return session.log.slice(-n);
}

function log(msg) {
  if (!session) return;
  session.log.push({ t: new Date().toISOString(), msg });
  if (session.log.length > 100) session.log.shift();
}

async function isDebuggerAlive() {
  try {
    const res = await fetch(`http://127.0.0.1:${control.port}/json/version`, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch { return false; }
}

function launchStandaloneChrome() {
  const exe = CHROME_PATHS.find(existsSync);
  if (!exe) throw new Error('no chromium executable found');
  const args = [
    `--remote-debugging-port=${STAGE_PORT}`,
    `--user-data-dir=${PROFILE_DIR}`,
    `--load-extension=${EXT_DIR}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1280,900',
    'about:blank'
  ];
  const child = spawn(exe, args, { detached: true, stdio: 'ignore' });
  child.unref();
}

export async function startSession(startUrl) {
  if (session && session.port === control.port) return sessionStatus();
  if (session) { try { session.browser.close(); } catch {} session = null; }
  if (!(await isDebuggerAlive())) {
    if (control.port !== STAGE_PORT) {
      // linked browser died (normal restart kills debug port) - fall back to stage
      control.port = STAGE_PORT;
      control.label = 'stage-fallback';
    }
    launchStandaloneChrome();
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (await isDebuggerAlive()) break;
      await new Promise(r => setTimeout(r, 500));
    }
    if (!(await isDebuggerAlive())) throw new Error('browser failed to expose debug port');
  }
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${control.port}`, { timeout: 10000 });
  let context = null;
  let page = null;
  for (const attempt of [0, 1, 2]) {
    for (const ctx of browser.contexts()) {
      const pages = ctx.pages ? ctx.pages() : [];
      const good = pages.find(pg => { try { return pg.url() && !pg.isClosed(); } catch { return false; } });
      if (good) { context = ctx; page = good; break; }
    }
    if (page) break;
    if (!context && browser.contexts().length) context = browser.contexts()[0];
    if (!context) context = await browser.newContext();
    page = await context.newPage().catch(() => null);
    if (page) break;
    await new Promise(r => setTimeout(r, 800));
  }
  if (!page) throw new Error('attached but could not acquire a page');
  if (startUrl && !page.url().includes(startUrl)) await page.goto(startUrl, { waitUntil: 'domcontentloaded', timeout: 45000 }).catch(() => {});
  session = { browser, context, page, port: control.port, startedAt: new Date().toISOString(), log: [], needsAssist: false };
  log('attached to standalone chrome' + (startUrl ? ' -> ' + startUrl : ''));
  return sessionStatus();
}

export async function stopSession() {
  if (!session) return { active: false };
  try { await session.browser.close(); } catch {}
  session = null;
  return { active: false, stopped: true, note: 'detached — chrome window stays open' };
}

export async function ensureActive() {
  if (!session) await startSession(null);
  return session;
}

export { evalStable, pickBestPage };

async function pickBestPage() {
  const s = session;
  if (!s || !s.browser) return null;

  // fast path: remembered game tab
  if (s.gamePage) {
    try {
      if (!s.gamePage.isClosed() && /chess\.com/i.test(s.gamePage.url())) { s.page = s.gamePage; return s.page; }
    } catch {}
    s.gamePage = null;
  }

  // cheap URL triage across all tabs (no JS evaluation)
  const candidates = [];
  for (const ctx of s.browser.contexts()) {
    for (const pg of (ctx.pages ? ctx.pages() : [])) {
      let u = '';
      try { u = pg.url(); } catch { continue; }
      if (u === 'about:blank' || u.startsWith('chrome')) continue;
      let score = 5;
      if (/chess\.com\/play/i.test(u)) score = 100;
      else if (/chess\.com/i.test(u)) score = 40;
      candidates.push({ pg, score });
    }
  }
  candidates.sort((a, b) => b.score - a.score);

  // guarded piece-count evals on top 3 only (wake suspended tabs first)
  let best = null;
  let bestN = -1;
  for (const c of candidates.slice(0, 3)) {
    await c.pg.bringToFront().catch(() => {});
    await new Promise(r => setTimeout(r, 700));
    const n = await Promise.race([
      c.pg.evaluate(() => document.querySelectorAll('[class*="piece"][class*="square-"]').length).catch(() => -1),
      new Promise(r => setTimeout(() => r(-1), 4000))
    ]);
    if (n > bestN) { bestN = n; best = c.pg; }
    if (bestN >= 20) break;
  }
  if (!best && candidates.length) best = candidates[0].pg;
  if (best) {
    s.page = best;
    try { if (/chess\.com\/play/i.test(best.url())) s.gamePage = best; } catch {}
  }
  return s.page;
}

async function page() {
  await pickBestPage().catch(() => {});
  const s = await ensureActive();
  return s.page;
}

export async function activePage() {
  return page();
}

async function evalStable(fn, retries = 2) {
  let lastErr;
  for (let i = 0; i <= retries; i++) {
    try {
      const p = await page();
      return await p.evaluate(fn);
    } catch (e) {
      lastErr = e;
      if (!/context|navigation/i.test(e.message)) throw e;
      await pickBestPage().catch(() => {});
      await new Promise(r => setTimeout(r, 1800));
    }
  }
  throw lastErr;
}

export async function actGoto(url) {
  const p = await page();
  await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
  lastUrl = url;
  log('goto ' + url);
  return { ok: true, url: p.url() };
}

export async function actClick(selectorOrText) {
  const p = await page();
  let ok = false;
  if (selectorOrText.startsWith('/') || selectorOrText.startsWith('.') || selectorOrText.startsWith('#') || selectorOrText.includes('[')) {
    ok = await p.locator(selectorOrText).first().click({ timeout: 8000 }).then(() => true).catch(() => false);
    if (!ok) ok = await p.getByText(selectorOrText, { exact: false }).first().click({ timeout: 6000 }).then(() => true).catch(() => false);
  } else {
    ok = await p.getByText(selectorOrText, { exact: false }).first().click({ timeout: 8000 }).then(() => true).catch(() => false);
    if (!ok) ok = await p.locator(selectorOrText).first().click({ timeout: 6000 }).then(() => true).catch(() => false);
  }
  log('click "' + selectorOrText + '" -> ' + (ok ? 'hit' : 'MISS'));
  if (!ok) { session.needsAssist = true; session.lastError = 'click target not found: ' + selectorOrText; }
  return { ok };
}

export async function actType(selector, text) {
  const p = await page();
  await p.locator(selector).first().fill(text, { timeout: 8000 });
  log('typed into ' + selector);
  return { ok: true };
}

export async function actPress(key) {
  const p = await page();
  await p.keyboard.press(key);
  log('press ' + key);
  return { ok: true };
}

export async function actRead() {
  const p = await page();
  const text = await p.evaluate(() => (document.body.innerText || '').replace(/\n{3,}/g, '\n\n').slice(0, 4000));
  log('read ' + text.length + ' chars');
  return { text };
}

export async function actScreenshot() {
  const p = await page();
  const buf = await p.screenshot({ type: 'png' });
  return { image: 'data:image/png;base64,' + buf.toString('base64') };
}

export async function nativeScreenshot() {
  const { execFileSync } = await import('node:child_process');
  const out = execFileSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', join(ROOT, 'capture-screen.ps1')], { encoding: 'ascii', timeout: 15000 });
  return out.trim();
}
