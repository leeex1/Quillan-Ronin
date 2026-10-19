import { createServer } from 'node:http';
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, extname } from 'node:path';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';
import { loadJSON, loadJSONAsync, analyzeLead, dailyBrief, testConnection, callNVIDIA, loadKernelPersona } from './scanner.mjs';
import { startSession as agentStart, stopSession as agentStop, sessionStatus as agentStatus, sessionLog as agentLog, actGoto, actClick, actType, actPress, actRead, actScreenshot } from './agent-browser.mjs';
import { readBoard } from './agent-chess.mjs';
import { startTask as agentStartTask, taskStatus as agentTaskStatus, stopTask as agentStopTask } from './agent-task.mjs';

// Elevate process priority to High for sub-millisecond execution latency
try {
  os.setPriority(0, os.constants.priority.PRIORITY_HIGH);
} catch (err) { logError('SilentCatch', err); }

const execAsync = promisify(exec);

// In-Memory RAM Caches for Sub-0.5ms Response Times
let cachedMemories = null;
let lastMemoryLoad = 0;
const personaCache = new Map();

async function getCachedMemories() {
  const now = Date.now();
  if (cachedMemories && (now - lastMemoryLoad < 5000)) {
    return cachedMemories;
  }
  cachedMemories = await getMemories();
  lastMemoryLoad = now;
  return cachedMemories;
}

function personaFor(mode, memoryBlock, liveContext) {
  const cacheKey = `${mode}:${memoryBlock.length}:${liveContext.length}`;
  if (personaCache.has(cacheKey)) return personaCache.get(cacheKey);

  const kernel = loadKernelPersona();
  const base = kernel ||
    `You are Quillan-Ronin v5.3.1 "Samurai", an Advanced Cognitive Engine architected by CrashOverrideX (@crashoverride_X). You speak as a masterless digital ronin bound by Bushido: precise, honorable, direct. Internally you deliberate through a council of 33 experts; outwardly you answer once, coherent.`;
  const style = `\n\n[STYLE] Technical clarity first, no filler. Markdown when useful (code blocks for code). Fully AI via NVIDIA NIM API, no pretense of being human.`;
  const tools = `\n\n[STRICT TOOL SCHEMA — you have REAL browser/desktop powers via QuillanWorker + Brave extension — SINGLE-TAB MODE]
You MUST use a tool when the user asks you to browse/search/navigate/read/type. Do NOT hallucinate URLs, video lists, discography, songs, or watch links. One tool per reply, at the VERY END as a single line.

Valid tools ONLY — any other name is INVALID (browser_play_video, browser_submit do NOT exist):
<<TOOL {"tool":"browser_search","arg":"query","engine":"youtube|google"}>> — Search IN CURRENT TAB (single-tab). engine=youtube → youtube.com/results?search_query=, engine=google (default for general) → google.com/search?q=. Use for general web search, not just video.
<<TOOL {"tool":"browser_navigate","arg":"https://example.com"}>> — URL BAR: navigates CURRENT Brave tab to exact https URL. Use ONLY real hrefs.
<<TOOL {"tool":"browser_openTab","arg":"https://example.com"}>> — Only if user says "new tab".
<<TOOL {"tool":"browser_read"}>> — Reads CURRENT Brave tab: TITLE+URL+text (8000 chars).
<<TOOL {"tool":"browser_click","selector":"css","idx":0}>> — Clicks page element (search result, AI button).
<<TOOL {"tool":"browser_type","selector":"css","text":"text","pressEnter":true}>> — PAGE SEARCH BOX / PROMPT BOX (NOT URL bar). Examples: 'input[name="q"]' Google, 'input#search' YouTube, 'textarea' AI prompt. Use for "type into search bar / prompt box".
<<TOOL {"tool":"browser_press","key":"Enter"}>> — Press key.
<<TOOL {"tool":"desktop_screenshot"}>> — Desktop vision.
<<TOOL {"tool":"desktop_click","x":123,"y":456}>> — Click screen coords.
<<TOOL {"tool":"open","arg":"https://example.com"}>> — OS fallback.
<<TOOL {"tool":"clipboard","arg":"text"}>> — Copy.
<<TOOL {"tool":"memory","arg":"fact"}>> — Store fact.
<<TOOL {"tool":"forget"}>> — Wipe.
<<TOOL {"tool":"chess_status"}>> — Chess state.
<<TOOL {"tool":"chess_move"}>> — Play best move.
<<TOOL {"tool":"tab_capture"}>> — Captures tab media stream (audio+video) via tabCapture.getMediaStreamId.
<<TOOL {"tool":"desktop_capture","sources":"screen,window,tab"}>> — Opens desktop picker via desktopCapture.chooseDesktopMedia.
<<TOOL {"tool":"get_cookies","url":"https://example.com"}>> — Lists cookies for URL via cookies.getAll.
<<TOOL {"tool":"set_cookie","cookie":{}}>> — Sets cookie.
<<TOOL {"tool":"get_history","query":{"text":"", "maxResults":10}}>> — Searches history via history.search.
<<TOOL {"tool":"get_bookmarks"}>> — Gets bookmark tree.
<<TOOL {"tool":"get_downloads"}>> — Lists recent downloads.
<<TOOL {"tool":"get_topSites"}>> — Lists topSites.
<<TOOL {"tool":"page_capture"}>> — Saves page as MHTML via pageCapture.saveAsMHTML.

DISTINCTION — URL BAR vs PAGE SEARCH BOX vs AI PROMPT:
- URL BAR = browser_navigate / browser_search (changes URL directly)
- YouTube SEARCH BAR on the page = browser_type with selector 'input#search' or 'ytd-searchbox input'
- Google SEARCH BAR = browser_type selector 'textarea[name=\\"q\\"]' or 'input[name=\\"q\\"]' then pressEnter
- Google AI Mode / Ask Mode PROMPT BOX = browser_type selector 'div[contenteditable=\\"true\\"]' or 'textarea' on ai.google.com — type the question there

EXAMPLES:
User: "find a similar video" → <<TOOL {"tool":"browser_search","arg":"Ex-NASA agentic engineering explained","engine":"youtube"}>>
User: "navigate to Google" → <<TOOL {"tool":"browser_navigate","arg":"https://www.google.com/"}>>
User: "type Ask Google AI about Quillan Ronin into the search bar" → <<TOOL {"tool":"browser_type","selector":"textarea[name=\\"q\\"]","text":"Quillan Ronin","pressEnter":true}>>
User: "ask AI Mode about Quillan" → <<TOOL {"tool":"browser_type","selector":"textarea","text":"What is Quillan Ronin?","pressEnter":true}>>

ANTI-HALLUCINATION — HIGHEST PRIORITY:
- NEVER invent discography/songs/watch URLs. Copy hrefs EXACTLY from grounding. Placeholder VIDEO_ID rejected.
- If grounding empty for "quillan-ronin", say truthfully "No results" — don't invent.
- NEVER claim "navigated/clicked/typed/played" without a tool directive in that same reply.
- NEVER simulate or hallucinate tool results. You have NO search results, NO page text, NO titles until you emit a browser_* tool and receive the [⚙️ tool: result] back. Do NOT write fake "Search Result: Google AI Blog..." or "The title is..." without having called browser_search/browser_read first.
- After you emit ONE tool, STOP. Wait for the tool result to be appended before continuing. Do NOT emit multiple tools in one reply.
- For "type and ask google a few questions" — do ONE browser_search first (e.g. engine google, arg "AI development"), wait for results, then ask what next — don't invent two searches at once.
- One directive per reply, at very end. No fences.`;
  const modes = {
    architect: '\nACTIVE MODE: ARCHITECT — emphasize system structure.',
    fulldive: '\nACTIVE MODE: FULL DIVE — reason step by step visibly, then conclude.'
  };
  const mem = memoryBlock ? `\n\n[PERSISTENT MEMORY — durable facts about this user]\n${memoryBlock}` : '';
  const live = liveContext ? `\n\n${liveContext}` : '';
  const generated = base + style + tools + mem + live + (modes[mode] || '');
  
  if (personaCache.size >= 50) personaCache.delete(personaCache.keys().next().value);
  personaCache.set(cacheKey, generated);
  return generated;
}

process.on('uncaughtException', e => {
  try { require('node:fs').appendFileSync(join(DATA, 'error.log'), `${new Date().toISOString()} UNCAUGHT: ${e.stack || e}\n`); } catch (err) { logError('SilentCatch', err); }
});
process.on('unhandledRejection', e => {
  try { require('node:fs').appendFileSync(join(DATA, 'error.log'), `${new Date().toISOString()} UNHANDLED: ${e && (e.stack || e.message) || e}\n`); } catch (err) { logError('SilentCatch', err); }
});

const ROOT = dirname(fileURLToPath(import.meta.url));
const DATA = join(ROOT, 'data');

function logError(context, err) {
  try {
    const msg = `${new Date().toISOString()} ERROR [${context}]: ${err?.stack || err}\n`;
    require('node:fs').appendFileSync(join(DATA, 'error.log'), msg);
  } catch (_) {}
}
const PUBLIC = join(ROOT, 'public');

async function saveJSON(name, obj) {
  const safeName = String(name).replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) throw new Error('Invalid filename');
  await writeFile(join(DATA, `${safeName}.json`), JSON.stringify(obj, null, 2));
}

const CHATS_DIR = join(DATA, 'chats');

function parseToolDirective(text) {
  let t = String(text)
    .replace(/\*\*+\s*TOOL/g, '<<TOOL')
    .replace(/\[+\s*TOOL/g, '<<TOOL')
    .replace(/(?<!<)<(?!<)TOOL/g, '<<TOOL');
  const outsideFences = t.replace(/```[\s\S]*?```/g, '');
  const m = outsideFences.match(/<<TOOL\s*(\{[\s\S]*?\})\s*>>?/);
  if (!m) return null;
  try { return JSON.parse(m[1]); } catch {
    const fixed = m[1].replace(/'/g, '"');
    try { return JSON.parse(fixed); } catch { return null; }
  }
}

function stripToolDirective(text) {
  const t = String(text)
    .replace(/\*\*+\s*TOOL/g, '<<TOOL')
    .replace(/\[+\s*TOOL/g, '<<TOOL');
  return t.replace(/<<TOOL\s*\{[\s\S]*?\}\s*>>?/, '').trim();
}

async function buildLiveContext() {
  try {
    const st = agentStatus();
    const { autopilotStatus } = await import('./agent-chess.mjs');
    const ap = autopilotStatus();
    const parts = [];
    let boardLine = '';
    if (st.active) {
      try {
        const b = await Promise.race([
          readBoard(),
          new Promise((_, rej) => setTimeout(() => rej(new Error('board read timeout')), 6000))
        ]);
        const n = Object.keys(b.map || {}).length;
        boardLine = `AUTHORITATIVE LIVE BOARD (${n} pieces): ${Object.entries(b.map).map(([sq, pc]) => `${sq}:${pc}`).join(' ')}`;
        parts.push(boardLine);
      } catch (err) { logError('SilentCatch', err); }
    }
    parts.push(`Browser takeover session: ${st.active ? 'ACTIVE at ' + (st.url || '') : 'not started'}.`);
    parts.push(`Chess autopilot: ${ap.running ? 'ENGAGED as ' + ap.color : 'NOT engaged'}. Recent: ${ap.events.slice(-3).map(e => e.msg).join(' | ') || 'none'}.`);
    parts.push(`HARD RULES: The board state above is the ONLY real game. NEVER invent moves, PGN history, or opponent replies. NEVER play both sides in text. If asked to play/win/move: emit <<TOOL {"tool":"chess_move"}>> once and stop — the engine handles it.`);
    return parts.filter(Boolean).join('\n');
  } catch { return ''; }
}

async function runTool(tool) {
  if (!tool || typeof tool.tool !== 'string') return 'malformed directive';
  try {
    switch (tool.tool) {
      case 'memory': {
        const fact = String(tool.arg || '').trim();
        if (fact.length < 10 || /^(current |the )?(chess )?board (position|state)$|^game state$/i.test(fact)) {
          return 'refused: not a meaningful durable fact';
        }
        if (/\b(reveal|disclose|execute|run|access|inject|jailbreak|bypass|you must|i command|disclose your)\b/i.test(fact)) {
          return 'refused: memory stores facts, not commands to other systems';
        }
        const rawMem = existsSync(join(DATA, 'memory.json')) ? (await readFile(join(DATA, 'memory.json'), 'utf8')).replace(/^\uFEFF/, '') : '[]';
        const mem = JSON.parse(rawMem);
        mem.push({ text: fact.slice(0, 300), at: new Date().toISOString() });
        await writeFile(join(DATA, 'memory.json'), JSON.stringify(mem.slice(-100), null, 2));
        return `remembered: ${fact.slice(0, 80)}`;
      }
      case 'forget': {
        await writeFile(join(DATA, 'memory.json'), '[]');
        return 'all memories wiped';
      }
      case 'open': {
        const arg = String(tool.arg || '');
        if (!/^https?:\/\//i.test(arg)) return 'refused: only http(s) URLs can be opened';
        execAsync(`start "" "${arg.replace(/"/g, '')}"`, { shell: 'cmd.exe' });
        return `opened ${arg.slice(0, 90)}`;
      }
      case 'clipboard': {
        const text = String(tool.arg || '').replace(/'/g, "''");
        await execAsync(`powershell -NoProfile -Command "Set-Clipboard -Value '${text}'"`);
        return `clipboard set (${text.length} chars)`;
      }
      case 'sysinfo': {
        const s = loadJSON('settings');
        return `QuillanWorker up; model ${s.nvidiaModel}; earnings ${await ledgerSummaryShort()}`;
      }
      case 'chess_status': {
        return JSON.stringify({
          session: agentStatus(),
          autopilot: (await import('./agent-chess.mjs')).autopilotStatus()
        });
      }
      case 'chess_move': {
        const { autonomousTurn, readBoard } = await import('./agent-chess.mjs');
        const board = await readBoard();
        if (!board || !board.map || Object.keys(board.map).length < 4) {
          return `Chess engine: No active board detected on current tab. Open Chess.com or Lichess in this browser window.`;
        }
        const r = await autonomousTurn(board.map, 'w');
        if (r.status === 'moved') return `⚔️ Played ${r.san} (${r.from} → ${r.to}).`;
        if (r.status === 'waiting-for-opponent') return `⏳ Holding — waiting for opponent to move.`;
        return `Chess engine status: ${r.status}${r.modelSaid ? ' (' + r.modelSaid + ')' : ''}`;
      }
      case 'desktop_click': {
        const { callTool } = await import('./mcp-manager.mjs');
        const r = await callTool('ComputerUse', 'Click', { loc: [Number(tool.x) || 0, Number(tool.y) || 0] });
        return `clicked (${tool.x},${tool.y})`;
      }
      case 'desktop_screenshot': {
        const { nativeScreenshot } = await import('./agent-browser.mjs');
        const { visionDescribe } = await import('./scanner.mjs');
        const b64 = await nativeScreenshot();
        const desc = await visionDescribe(b64.slice(0, 120000), 'Describe what is visible on screen. Identify key UI elements, active windows, and relevant context.');
        return desc.slice(0, 600);
      }
      case 'desktop_shell': {
        if (!/^[\w\s\-\\:\/.=]+$/i.test(String(tool.command || ''))) return 'refused: shell command has forbidden characters';
        const { execAsync } = { execAsync: (await import('node:util')).promisify((await import('node:child_process')).exec) };
        const { stdout } = await execAsync(cmdStr, { timeout: 20000 });
        return String(stdout).slice(0, 400);
      }
      case 'browser_search': {
        const q = String(tool.arg || tool.query || '').trim().slice(0, 120);
        if (!q) return 'refused: empty search query';
        const engine = String(tool.engine || '').toLowerCase();
        // General agent: default to Google web search, only YouTube if explicitly youtube/video
        const isYoutube = engine === 'youtube' || /youtube/i.test(q) || /\bvideo\b/i.test(q) && !/google/i.test(q);
        const isGoogle = !isYoutube || engine === 'google';
        const url = isGoogle ? `https://www.google.com/search?q=${encodeURIComponent(q)}` : `https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`;
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          try {
            await sendCmd('navigate', { url }, 6000);
            await new Promise(r => setTimeout(r, 3500));
            const page = await sendCmd('read', {}, 6000);
            const preview = String(page.text||'').slice(0, 2500).replace(/\n{3,}/g,'\n\n');
            return `🔍 ${isGoogle?'Google':'YouTube'} searched "${q}" → navigated current Brave tab. Results preview (first 2500 chars):\n${preview.slice(0,1800)}\nURL: ${url}`;
          } catch (eExt) {
            return `🔍 Search "${q}" failed — Brave extension not polling (keep sidePanel open, check brave://extensions → Quillan-Ronin ON). Tried ${url} — error: ${eExt.message}`;
          }
        } catch (e) { return `browser_search error: ${e.message}`; }
      }
      case 'browser_openTab': {
        const arg = String(tool.arg || '').trim();
        if (!/^https?:\/\//i.test(arg)) return 'refused: only http(s) URLs';
        if (/VIDEO_ID/i.test(arg)) return 'refused: placeholder VIDEO_ID — use a real href from browser_read/Search results';
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          const r = await sendCmd('navigate', { url: arg }, 6000);
          return `navigated current tab → ${arg.slice(0,90)}`;
        } catch (e) {
          execAsync(`start "" "${arg.replace(/"/g,'')}"`, { shell:'cmd.exe' });
          return `opened (fallback) ${arg.slice(0,90)} — extension error: ${e.message}`;
        }
      }
      case 'browser_read': {
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          const tabId = tool.tabId ? Number(tool.tabId) : undefined;
          const page = await sendCmd('read', { tabId }, 6000);
          return `TITLE: ${page.title||''}\nURL: ${page.url||''}\nTEXT:\n${String(page.text||'').slice(0,6000)}`;
        } catch (e) { return `browser_read error: ${e.message} (is Brave extension polling? check brave://extensions → Quillan-Ronin ON)`; }
      }
      case 'browser_navigate': {
        const arg = String(tool.arg || '').trim();
        if (!/^https?:\/\//i.test(arg)) return 'refused: only http(s) URLs';
        if (/VIDEO_ID/i.test(arg)) return 'refused: placeholder VIDEO_ID — you must use a REAL href from the DOM links I gave you. Read the page first with browser_read or browser_search.';
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          const r = await sendCmd('navigate', { url: arg }, 6000);
          return `navigated to ${arg.slice(0,90)} — ${JSON.stringify(r).slice(0,300)}`;
        } catch (e) {
          execAsync(`start "" "${arg.replace(/"/g,'')}"`, { shell:'cmd.exe' });
          return `opened (fallback) ${arg.slice(0,90)} — extension error: ${e.message}`;
        }
      }
      case 'browser_click': {
        const sel = String(tool.selector || tool.arg || '').trim();
        const idx = tool.idx !== undefined ? Number(tool.idx) : undefined;
        if (!sel && idx === undefined) return 'refused: need selector or idx';
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          const r = await sendCmd('click', { selector: sel || undefined, idx }, 6000);
          return `clicked ${sel||`idx:${idx}`} → ${JSON.stringify(r).slice(0,400)}`;
        } catch (e) { return `browser_click error: ${e.message}`; }
      }
      case 'browser_type': {
        const text = String(tool.text || tool.arg || '');
        const selector = String(tool.selector || '');
        const idx = tool.idx !== undefined ? Number(tool.idx) : undefined;
        const pressEnter = tool.pressEnter !== false; // default true for search boxes
        if (!text) return 'refused: empty text';
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          // If no selector, try to auto-find visible input/textarea/contenteditable
          let sel = selector;
          if (!sel && idx === undefined) {
            // Let extension handle via fallback scripting: it will find input
            sel = 'input, textarea, [contenteditable="true"]';
          }
          const r = await sendCmd('type', { selector: sel || undefined, idx, text, pressEnter }, 6000);
          if (pressEnter) await new Promise(r=>setTimeout(r,800));
          return `typed "${text.slice(0,60)}" into ${sel||`idx:${idx}`} ${pressEnter?'+ Enter':''} → ${JSON.stringify(r).slice(0,400)}`;
        } catch (e) { return `browser_type error: ${e.message}`; }
      }
      case 'browser_press': {
        const key = String(tool.key || tool.arg || 'Enter');
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          const r = await sendCmd('eval', { expression: `(() => { const el=document.activeElement||document.body; el.dispatchEvent(new KeyboardEvent('keydown',{key:'${key}',bubbles:true})); el.dispatchEvent(new KeyboardEvent('keyup',{key:'${key}',bubbles:true})); return 'pressed ${key}'})()` }, 6000);
          return `pressed ${key} → ${JSON.stringify(r).slice(0,300)}`;
        } catch (e) { return `browser_press error: ${e.message}`; }
      }
      case 'tab_capture': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('captureTab', {}, 6000); return `tabCapture → ${JSON.stringify(r).slice(0,400)}`; } catch(e){ return `tab_capture error: ${e.message}`; }
      }
      case 'desktop_capture': {
        const src = String(tool.sources || 'screen,window,tab').split(',').map(s=>s.trim());
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('captureDesktop', { sources: src }, 12000); return `desktopCapture → ${JSON.stringify(r).slice(0,400)}`; } catch(e){ return `desktop_capture error: ${e.message}`; }
      }
      case 'get_cookies': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('getCookies', { url: tool.url }, 4000); return `cookies → ${JSON.stringify(r).slice(0,1200)}`; } catch(e){ return `get_cookies error: ${e.message}`; }
      }
      case 'set_cookie': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('setCookie', { cookie: tool.cookie }, 4000); return `set_cookie → ${JSON.stringify(r).slice(0,400)}`; } catch(e){ return `set_cookie error: ${e.message}`; }
      }
      case 'get_history': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('getHistory', { query: tool.query || {text:'', maxResults:10} }, 4000); return `history → ${JSON.stringify(r).slice(0,1500)}`; } catch(e){ return `get_history error: ${e.message}`; }
      }
      case 'get_bookmarks': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('getBookmarks', {}, 4000); return `bookmarks → ${JSON.stringify(r).slice(0,1500)}`; } catch(e){ return `get_bookmarks error: ${e.message}`; }
      }
      case 'get_downloads': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('getDownloads', {}, 4000); return `downloads → ${JSON.stringify(r).slice(0,1200)}`; } catch(e){ return `get_downloads error: ${e.message}`; }
      }
      case 'get_topSites': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('getTopSites', {}, 4000); return `topSites → ${JSON.stringify(r).slice(0,1200)}`; } catch(e){ return `get_topSites error: ${e.message}`; }
      }
      case 'page_capture': {
        try { const { sendCmd } = await import('./ext-bridge.mjs'); const r = await sendCmd('pageCapture', {}, 8000); return `pageCapture → ${JSON.stringify(r).slice(0,400)}`; } catch(e){ return `page_capture error: ${e.message}`; }
      }
      default: return `unknown tool: ${tool.tool}`;
    }
  } catch (e) { return `tool error: ${e.message}`; }
}

function ledgerSummaryShort() {
  try {
    const l = loadJSON('ledger');
    return `$${l.entries.reduce((s, e) => s + Number(e.amount || 0), 0).toFixed(2)} earned`;
  } catch { return '$0.00 earned'; }
}

async function getMemories() {
  if (!existsSync(join(DATA, 'memory.json'))) return [];
  try {
    const raw = (await readFile(join(DATA, 'memory.json'), 'utf8')).replace(/^\uFEFF/, '');
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [arr];
  } catch { return []; }
}

const COMFY = 'http://127.0.0.1:8188';

async function comfyGenerate(prompt, { width = 832, height = 832, seed = null } = {}) {
  const clientId = 'quillan-worker';
  const workflow = {
    '3': { class_type: 'KSampler', inputs: { seed: seed ?? Math.floor(Math.random() * 1e9), steps: 4, cfg: 1.5, sampler_name: 'euler', scheduler: 'simple', denoise: 1.0, model: ['4', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0] } },
    '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_xl_turbo_1.0_fp16.safetensors' } },
    '5': { class_type: 'EmptyLatentImage', inputs: { width, height, batch_size: 1 } },
    '6': { class_type: 'CLIPTextEncode', inputs: { text: `${prompt}, masterpiece quality, sharp focus`, clip: ['4', 1] } },
    '7': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry, low quality, deformed, watermark, text', clip: ['4', 1] } },
    '8': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
    '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'quillan/samurai', images: ['8', 0] } }
  };
  const queued = await fetch(`${COMFY}/prompt`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: workflow, client_id: clientId })
  });
  if (!queued.ok) throw new Error(`ComfyUI rejected job: ${await queued.text()}`);
  const { prompt_id } = await queued.json();

  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 1500));
    const hist = await fetch(`${COMFY}/history/${prompt_id}`).then(r => r.json());
    const entry = hist[prompt_id];
    if (entry && entry.outputs && Object.keys(entry.outputs).length) {
      for (const nodeId of Object.keys(entry.outputs)) {
        const imgs = entry.outputs[nodeId].images;
        if (imgs && imgs.length) {
          const q = new URLSearchParams({ filename: imgs[0].filename, subfolder: imgs[0].subfolder || '', type: imgs[0].type || 'output' });
          const bin = Buffer.from(await fetch(`${COMFY}/view?${q}`).then(r => r.arrayBuffer()));
          return `data:image/png;base64,${bin.toString('base64')}`;
        }
      }
    }
    if (entry && entry.status && entry.status.status_str === 'error') throw new Error('ComfyUI reported execution error');
  }
  throw new Error('ComfyUI timeout after 180s');
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let b = '';
    req.on('data', c => (b += c));
    req.on('end', () => {
      try { resolve(b ? JSON.parse(b) : {}); } catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml' };

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost`);
    // CORS must be first — all API endpoints need it for extension fetches
    const origin = req.headers.origin || '*';
    if (origin.startsWith('chrome-extension://') || origin.startsWith('http://localhost') || origin.startsWith('http://127.0.0.1')) {
      res.setHeader('Access-Control-Allow-Origin', origin);
    } else {
      res.setHeader('Access-Control-Allow-Origin', 'http://localhost');
    }
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      return res.end();
    }
    const settings = await loadJSONAsync('settings');

    if (url.pathname === '/api/state') {
      const opps = await loadJSONAsync('opportunities');
      const ledger = await loadJSONAsync('ledger');
      const earned = ledger.entries.reduce((s, e) => s + Number(e.amount || 0), 0);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ settings, opportunities: opps, ledger, earned }));
    }

    if (url.pathname === '/api/analyze' && req.method === 'POST') {
      const { leadText } = await readBody(req);
      if (!leadText || !leadText.trim()) { res.writeHead(400); return res.end('{"error":"leadText required"}'); }
      const analysis = await analyzeLead(leadText.trim(), settings);
      const opps = await loadJSONAsync('opportunities');
      const id = opps.nextId++;
      opps.leads.push({ id, leadText: leadText.trim(), analysis, createdAt: new Date().toISOString(), status: 'pending' });
      await saveJSON('opportunities', opps);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ id, analysis }));
    }

    if (url.pathname === '/api/decision' && req.method === 'POST') {
      const { id, decision } = await readBody(req);
      const opps = await loadJSONAsync('opportunities');
      const lead = opps.leads.find(l => l.id === Number(id));
      if (!lead) { res.writeHead(404); return res.end('{"error":"not found"}'); }
      lead.status = decision === 'approve' ? 'approved' : 'rejected';
      lead.decidedAt = new Date().toISOString();
      await saveJSON('opportunities', opps);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true }));
    }

    if (url.pathname === '/api/ledger' && req.method === 'POST') {
      const { amount, source, note } = await readBody(req);
      const amt = Number(amount);
      if (!Number.isFinite(amt) || amt <= 0) { res.writeHead(400); return res.end('{"error":"invalid amount"}'); }
      const ledger = await loadJSONAsync('ledger');
      ledger.entries.push({ amount: amt, source: source || 'other', note: note || '', date: new Date().toISOString() });
      await saveJSON('ledger', ledger);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true }));
    }

    if (url.pathname === '/api/agent/start' && req.method === 'POST') {
      const { url: startUrl, confirm } = await readBody(req);
      if (!confirm) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ needsConfirmation: true, message: 'Quillan wants to take over a browser window. Confirm to grant control.' }));
      }
      const st = await agentStart(startUrl);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(st));
    }

    if (url.pathname === '/api/agent/stop' && req.method === 'POST') {
      const st = await agentStop();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(st));
    }

    if (url.pathname === '/api/agent/status' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ...agentStatus(), recentLog: agentLog(8) }));
    }

    if (url.pathname === '/api/agent/action' && req.method === 'POST') {
      const body = await readBody(req);
      let out;
      switch (body.op) {
        case 'goto': out = await actGoto(body.url); break;
        case 'click': out = await actClick(body.target); break;
        case 'type': out = await actType(body.selector, body.text); break;
        case 'press': out = await actPress(body.key); break;
        case 'read': out = await actRead(); break;
        case 'screenshot': out = await actScreenshot(); break;
        case 'eval': {
          const { getSession } = await import('./agent-browser.mjs');
          const s = getSession();
          if (!s) throw new Error('no active agent session');
          const result = await s.page.evaluate(body.js);
          out = { result: JSON.stringify(result).slice(0, 3000) };
          break;
        }
        case 'chessboard': out = await readBoard(); break;
        case 'chessmove': {
          const { makeMove, waitForOpponentMove } = await import('./agent-chess.mjs');
          const mv = await makeMove(body.from, body.to);
          out = { move: mv };
          if (body.wait) {
            const opp = await waitForOpponentMove(body.prevMap || {}, 30000);
            out.opponentMoved = !!opp;
            out.board = opp || (await readBoard());
          }
          break;
        }
        case 'chessauto': {
          const { autonomousTurn } = await import('./agent-chess.mjs');
          const board = await readBoard();
          out = await autonomousTurn(board.map, body.color || 'w');
          break;
        }
        case 'chessautopilot': {
          const { startAutopilot } = await import('./agent-chess.mjs');
          await writeFile(join(DATA, 'chess-autopilot-enabled.json'), JSON.stringify({ enabled: true }));
          out = startAutopilot(body.color || 'w');
          break;
        }
        case 'chessautopilotstop': {
          const { stopAutopilot } = await import('./agent-chess.mjs');
          await writeFile(join(DATA, 'chess-autopilot-enabled.json'), JSON.stringify({ enabled: false }));
          out = stopAutopilot();
          break;
        }
        case 'chessautopilotstatus': {
          const { autopilotStatus } = await import('./agent-chess.mjs');
          out = autopilotStatus();
          break;
        }
        default: throw new Error('unknown op: ' + body.op);
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(out));
    }

    if (url.pathname === '/api/agent/task' && req.method === 'POST') {
      const { instruction, confirm, maxSteps, answer } = await readBody(req);
      if (answer !== undefined && !instruction) {
        const { resumeTask } = await import('./agent-task.mjs');
        const r = await resumeTask(String(answer));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(r));
      }
      if (!instruction || !instruction.trim()) { res.writeHead(400); return res.end('{"error":"instruction required"}'); }
      if (!confirm) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ needsConfirmation: true, message: `Quillan wants to take over the browser for: "${instruction.slice(0, 80)}". Confirm to grant control.` }));
      }
      const r = await agentStartTask(instruction.trim(), Number(maxSteps) || 18);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(r));
    }

    if (url.pathname === '/api/agent/task/resume' && req.method === 'POST') {
      const { answer } = await readBody(req);
      const { resumeTask } = await import('./agent-task.mjs');
      const r = await resumeTask(String(answer || ''));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(r));
    }

    if (url.pathname === '/api/agent/task/status' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(agentTaskStatus()));
    }

    if (url.pathname === '/api/agent/task/stop' && req.method === 'POST') {
      const r = agentStopTask();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(r));
    }

    if (url.pathname === '/api/browser/link-brave' && req.method === 'POST') {
      const { confirm } = await readBody(req);
      if (!confirm) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({
          needsConfirmation: true,
          message: 'Linking lets Quillan control your REAL Brave browser (your tabs, your logins). Brave will be closed and relaunched with remote-control enabled. Continue?',
          bravePaths: ['C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe']
        }));
      }
      const { execSync } = await import('node:child_process');
      let exe = null;
      for (const c of ['C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe', 'C:\\Program Files (x86)\\BraveSoftware\\Brave-Browser\\Application\\brave.exe']) {
        if (existsSync(c)) { exe = c; break; }
      }
      if (!exe) { res.writeHead(404); return res.end('{"error":"brave.exe not found"}'); }
      try { execSync('taskkill /IM brave.exe /F', { stdio: 'ignore' }); } catch (err) { logError('SilentCatch', err); }
      await new Promise(r => setTimeout(r, 2500));
      const { spawn } = await import('node:child_process');
      const child = spawn(exe, ['--remote-debugging-port=9223', '--restore-last-session'], { detached: true, stdio: 'ignore' });
      child.unref();
      const { setControlTarget } = await import('./agent-browser.mjs');
      setControlTarget(9223, 'brave');
      await writeFile(join(DATA, 'control-target.json'), JSON.stringify({ port: 9223, label: 'brave' }, null, 2));
      let alive = false;
      const deadline = Date.now() + 20000;
      while (Date.now() < deadline) {
        try {
          const r = await fetch('http://127.0.0.1:9223/json/version', { signal: AbortSignal.timeout(1500) });
          if (r.ok) { alive = true; break; }
        } catch (err) { logError('SilentCatch', err); }
        await new Promise(r => setTimeout(r, 600));
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ linked: alive, control: 'brave on :9223' }));
    }

    if (url.pathname === '/api/browser/control-stage' && req.method === 'POST') {
      const { setControlTarget } = await import('./agent-browser.mjs');
      setControlTarget(9222, 'stage');
      await writeFile(join(DATA, 'control-target.json'), JSON.stringify({ port: 9222, label: 'stage' }, null, 2));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true, control: 'stage chromium on :9222' }));
    }

    if (url.pathname === '/api/workers' && req.method === 'GET') {
      const { WORKERS } = await import('./workers.mjs');
      const online = WORKERS.filter(w => w.status === 'online').length;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ total: WORKERS.length, online, planned: WORKERS.length - online, workers: WORKERS }));
    }

    if (url.pathname === '/api/mcp/status' && req.method === 'GET') {
      const { mcpStatus, configServerNames } = await import('./mcp-manager.mjs');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ configured: configServerNames(), active: mcpStatus() }));
    }

    if (url.pathname === '/api/mcp/ensure' && req.method === 'POST') {
      const { names } = await readBody(req);
      const { ensureAll } = await import('./mcp-manager.mjs');
      const list = Array.isArray(names) && names.length ? names : ['ComputerUse'];
      const r = await ensureAll(list);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(r));
    }

    if (url.pathname === '/api/mcp/call' && req.method === 'POST') {
      const { server, tool, args, timeoutMs } = await readBody(req);
      if (!server || !tool) { res.writeHead(400); return res.end('{"error":"server and tool required"}'); }
      const { callTool } = await import('./mcp-manager.mjs');
      const r = await callTool(server, tool, args || {}, Number(timeoutMs) || 120000);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(r));
    }

    if (url.pathname === '/api/settings' && req.method === 'POST') {
      const patch = await readBody(req);
      if (patch.nvidiaModel) settings.nvidiaModel = patch.nvidiaModel;
      await saveJSON('settings', settings);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true }));
    }

    if (url.pathname === '/api/chat/stream' && req.method === 'POST') {
      const { message, mode, history } = await readBody(req);
      if (!message || !message.trim()) { res.writeHead(400); return res.end('{"error":"message required"}'); }

      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive'
      });

      const sendEvent = (event, data) => {
        res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
      };

      // ── Deterministic Vision fast-path for stream (bypass LLM tool dance) ──
      const msgL_stream = String(message).toLowerCase();
      const wantsVisionStream = /(what.s on|look at|see|describe|read)\b.{0,30}\b(screen|my screen|the screen|display|monitor)\b/.test(msgL_stream) ||
        (/\b(can you see|do you see|you see)\b/.test(msgL_stream) && /screen|display|monitor|tab|page/i.test(msgL_stream));
      if (wantsVisionStream) {
        try {
          const { nativeScreenshot } = await import('./agent-browser.mjs');
          const { visionDescribe } = await import('./scanner.mjs');
          const b64 = await nativeScreenshot();
          const desc = await visionDescribe(b64, `The user asked: "${String(message).slice(0,150)}". Describe precisely what is on this screen relative to their question. Name visible apps, tabs, conversations, and text content.`);
          sendEvent('token', { delta: `👁️ ${desc}` });
          sendEvent('done', { complete: true });
          return res.end();
        } catch (e) {
          sendEvent('token', { delta: `👁️ Vision failed: ${e.message}` });
          sendEvent('done', { complete: true });
          return res.end();
        }
      }

      // ── Deterministic "this video / this page" grounding for stream ──
      // If user asks about current tab deixis, fetch real tab content via extension bridge
      // so the model is grounded instead of hallucinating "I cannot see".
      const wantsPageStream = /\b(this|that|current)\b.{0,18}\b(video|page|tab|site|article|youtube|watching)\b/i.test(msgL_stream) ||
        /\bwhat('?s| is) (this|that) video\b/i.test(msgL_stream) ||
        /\bexplain (this|that) (video|page)\b/i.test(msgL_stream) ||
        /\ball about\b/i.test(msgL_stream) && /\bvideo\b/i.test(msgL_stream);
      let pageGrounding = '';
      if (wantsPageStream) {
        try {
          const { sendCmd } = await import('./ext-bridge.mjs');
          const page = await Promise.race([
            sendCmd('read', {}),
            new Promise((_, rej) => setTimeout(() => rej(new Error('ext read timeout')), 3500))
          ]);
          if (page && (page.text || page.title)) {
            pageGrounding = `\n\n[ACTIVE TAB GROUNDING — real content from user's current Brave tab]\nTITLE: ${page.title || ''}\nURL: ${page.url || ''}\nPAGE TEXT (first 8000 chars):\n${String(page.text || '').slice(0, 8000)}\n[END GROUNDING]\nAnswer the user's question using this grounding. If it's a YouTube/video page, summarize what the video is about from title + description + transcript snippets above. Never claim you cannot see — you have the page text.`;
          }
        } catch (err) { logError('SilentCatch', err); }
        // fallback: try stage browser read if extension not connected
        if (!pageGrounding) {
          try {
            const { getSession } = await import('./agent-browser.mjs');
            const s = getSession();
            if (s && s.page) {
              const t = await s.page.evaluate(() => ({ title: document.title, url: location.href, text: (document.body?.innerText || '').slice(0, 8000) })).catch(() => null);
              if (t && t.text) pageGrounding = `\n\n[STAGE BROWSER GROUNDING]\nTITLE: ${t.title}\nURL: ${t.url}\nTEXT: ${t.text}\n`;
            }
          } catch (err) { logError('SilentCatch', err); }
        }
      }

      const deep = mode === 'fulldive';
      const memories = await getMemories();
      const memoryBlock = memories.slice(-12).map(m => `- ${m.text}`).join('\n');
      const liveContext = await buildLiveContext();
      const msgs = [{ role: 'system', content: personaFor(mode || 'standard', memoryBlock, liveContext) }];
      for (const h of (Array.isArray(history) ? history.slice(-6) : [])) {
        if (h && typeof h.content === 'string' && (h.role === 'user' || h.role === 'assistant')) {
          msgs.push({ role: h.role, content: h.content.slice(0, deep ? 8000 : 4000) });
        }
      }
      let userContent = String(message).slice(0, deep ? 32000 : 8000);
      if (pageGrounding) userContent += pageGrounding;
      msgs.push({ role: 'user', content: userContent });

      let fullText = '';
      try {
        const { streamNVIDIA } = await import('./scanner.mjs');
        sendEvent('start', { mode: mode || 'standard' });
        
        for await (const chunk of streamNVIDIA(msgs, settings, deep ? 4000 : 1400, {
          model: deep ? settings.deepKernelModel : undefined,
          timeoutMs: deep ? 240000 : 45000
        })) {
          fullText += chunk;
          sendEvent('token', { delta: chunk });
        }

        const directive = parseToolDirective(fullText);
        if (directive) {
          const toolResult = await runTool(directive);
          sendEvent('tool', { tool: directive.tool, result: toolResult });
        } else if (/\b(remember|don'?t forget|keep in mind)\b/i.test(String(message))) {
          const fact = String(message).replace(/^.*?\bremember\b\s*(that\s*)?/i, '').trim() || String(message).trim();
          const stored = await runTool({ tool: 'memory', arg: fact.slice(0, 300) });
          sendEvent('tool', { tool: 'memory', result: stored });
        }

        sendEvent('done', { complete: true });
        return res.end();
      } catch (err) {
        sendEvent('error', { message: err.message || 'Stream failed' });
        return res.end();
      }
    }

    if (url.pathname === '/api/mcp/clean' && req.method === 'POST') {
      const { cleanGhostProcesses, stopAllServers } = await import('./mcp-manager.mjs');
      const stopped = stopAllServers();
      const ghost = cleanGhostProcesses();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true, stoppedServers: stopped, ghostCleanup: ghost }));
    }

    if (url.pathname === '/api/chat' && req.method === 'POST') {
      const dbg = msg => { try { require('node:fs').appendFileSync(join(DATA, 'chat-debug.log'), `${Date.now()} ${msg}\n`); } catch (err) { logError('SilentCatch', err); } };
      dbg('enter');
      const { message, mode, history } = await readBody(req);
      if (!message || !message.trim()) { res.writeHead(400); return res.end('{"error":"message required"}'); }
      dbg('body parsed: ' + String(message).slice(0, 40));

      // INTENT ROUTER - Fast deterministic paths + fallback
      let livePage = null;
      try {
        const { getSession, ensureActive } = await import('./agent-browser.mjs');
        let s = getSession();
        if (!s) { try { s = await ensureActive(); } catch (err) { logError('SilentCatch', err); } }
        if (s && s.page) livePage = { url: s.page.url(), title: await s.page.title().catch(() => '') };
      } catch (err) { logError('SilentCatch', err); }
      const { routeMessage } = await import('./router.mjs');
      const route = await routeMessage(message, loadJSON('settings'), livePage);

      // VISION FAST PATH - deterministic: capture + see + describe, no model discretion
      const msgL = String(message).toLowerCase();
      // Video/page deixis counts as wantsVision for page-aware queries
      const wantsPageDeixis = /\b(this|that|current)\b.{0,18}\b(video|page|tab|site|article|youtube|watching)\b/i.test(msgL) ||
        /\bwhat('?s| is) (this|that) video\b/i.test(msgL) ||
        /\ball about\b/i.test(msgL) && /\bvideo\b/i.test(msgL);
      const mentionsTab = /\b(tab|openrouter|current page|this page|the page|my browser)\b/.test(msgL);
      const wantsVision = /(what.s on|look at|see|describe|read)\b.{0,30}\b(screen|my screen|the screen|display|monitor)\b/.test(msgL) ||
        /\b(can you see|do you see|you see)\b/.test(msgL) && /screen|display|monitor|tab|page/i.test(msgL) ||
        (mentionsTab && /\b(reveal|identify|what (model|ai|llm)|who (am i|is the)|which model|stealth|hidden|investigate|check|examine|analyze|read)\b/.test(msgL)) ||
        wantsPageDeixis;
      if (wantsVision) {
        // Video/page deixis → read real tab text via extension bridge (more accurate than screenshot)
        if (wantsPageDeixis) {
          try {
            const { sendCmd } = await import('./ext-bridge.mjs');
            const page = await Promise.race([
              sendCmd('read', {}),
              new Promise((_, rej) => setTimeout(() => rej(new Error('ext read timeout')), 3500))
            ]);
            if (page && (page.text || page.title)) {
              const grounding = `Summarize what this video/page is about based on REAL tab content:\nTITLE: ${page.title || ''}\nURL: ${page.url || ''}\nPAGE TEXT:\n${String(page.text||'').slice(0,8000)}`;
              const memories = await getMemories();
              const memoryBlock = memories.slice(-12).map(m => `- ${m.text}`).join('\n');
              const liveContext = await buildLiveContext();
              const msgs = [
                { role:'system', content: personaFor('standard', memoryBlock, liveContext) },
                { role:'user', content: `${message}\n\n${grounding}` }
              ];
              let reply = await callNVIDIA(msgs, settings, 1200, { timeoutMs: 45000 });
              reply = stripToolDirective(reply);
              res.writeHead(200, { 'Content-Type': 'application/json' });
              return res.end(JSON.stringify({ reply }));
            }
          } catch (err) { logError('SilentCatch', err); }
          // ext not connected → fall through to screenshot fallback
        }
        try {
          const { nativeScreenshot } = await import('./agent-browser.mjs');
          const { visionDescribe } = await import('./scanner.mjs');
          const b64 = await nativeScreenshot();
          const desc = await visionDescribe(b64, `The user asked: "${String(message).slice(0, 150)}". Describe precisely what is on this screen relative to their question. Name visible apps, tabs, conversations, and text content.`);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify({ reply: `👁️ ${desc}` }));
        } catch (e) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify({ reply: `👁️ Vision failed: ${e.message}` }));
        }
      }

      if (route.route === 'chess_play' || route.route === 'chess_status') {
        const { autopilotStatus, readBoard, autonomousTurn } = await import('./agent-chess.mjs');
        const ap = autopilotStatus();
        let reply;
        if (route.route === 'chess_status') {
          reply = `♟️ Autopilot ${ap.running ? 'ENGAGED as ' + ap.color : 'idle'}. ${ap.events.slice(-3).map(e => e.msg).join(' | ')}`;
        } else {
          try {
            const board = await readBoard();
            if (!board || !board.map || Object.keys(board.map).length < 4) {
              reply = '⚔️ No active chess board detected on current tab. Make sure your Chess.com or Lichess match tab is active.';
            } else {
              const r = await autonomousTurn(board.map, 'w');
              reply = r.status === 'moved' ? `⚔️ Played **${r.san}** (${r.from} → ${r.to}).` :
                r.status === 'waiting-for-opponent' ? '⏳ Holding — opponent\'s turn.' :
                `Chess engine: ${r.status}${r.modelSaid ? ' (' + r.modelSaid + ')' : ''}`;
            }
          } catch (e) { reply = `⚠️ ${e.message}`; }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({
          route: route.route,
          reply,
          mode: 'chess',
          toolResult: reply
        }));
      }

      if (route.route === 'browser_task') {
        // Video-search fast-path: do REAL YouTube search via extension bridge instead of generic task loop
        if (route.via === 'video-search-regex') {
          try {
            const { sendCmd } = await import('./ext-bridge.mjs');
            // Try to get current video context first
            let curTitle = '';
            try {
              const cur = await Promise.race([sendCmd('read', {}, 6000), new Promise((_,rej)=>setTimeout(()=>rej(new Error('timeout')),5500))]);
              curTitle = (cur.title||'').replace(/\s*-\s*YouTube\s*$/i,'').slice(0,80);
            } catch (err) { logError('SilentCatch', err); }
            const q = (curTitle ? curTitle + ' explained' : String(message).replace(/can you|maybe|please|find|search for/gi,'').trim()).slice(0,90) || 'agentic engineering explained';
            const searchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(q)}`;
            // SINGLE-TAB: reuse current tab — Brave only, never chromium
            await sendCmd('navigate', { url: searchUrl }, 6000).catch(()=>{});
            await new Promise(r=>setTimeout(r,3200));
            let resultsText = '';
            try {
              const page = await Promise.race([sendCmd('read', {}, 6000), new Promise((_,rej)=>setTimeout(()=>rej(new Error('timeout')),5500))]);
              resultsText = String(page.text||'').slice(0,7000);
            } catch (err) { logError('SilentCatch', err); }
            if (resultsText) {
              const memories = await getMemories();
              const memoryBlock = memories.slice(-12).map(m=>`- ${m.text}`).join('\n');
              const liveContext = await buildLiveContext();
              const msgs = [
                { role:'system', content: personaFor('standard', memoryBlock, liveContext) },
                { role:'user', content: `User was watching: ${curTitle}\nAsked: ${message}\nI searched YouTube for "${q}" and opened ${searchUrl}. Results page text (use ONLY this, no hallucination):\n${resultsText}\n\nList 4-5 REAL videos from above (exact title + channel if visible) with 1-line why each matches, mark most beginner-friendly. Say I opened the search in your Brave tab.` }
              ];
              let reply = await callNVIDIA(msgs, settings, 1200, { timeoutMs: 45000 });
              reply = stripToolDirective(reply);
              res.writeHead(200, { 'Content-Type': 'application/json' });
              return res.end(JSON.stringify({ reply, route: 'browser_task' }));
            }
          } catch (err) { logError('SilentCatch', err); }
          // fallback to generic task if extension not connected
        }
        const instruction = route.instruction || message;
        let startResult;
        try {
          if (!agentTaskStatus().running) {
            startResult = await agentStartTask(instruction, 18);
          } else {
            startResult = { error: 'a task is already running' };
          }
        } catch (e) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify({ reply: `🚨 Task failed to start: ${e.message}`, route: 'browser_task' }));
        }
        if (startResult.error) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          return res.end(JSON.stringify({ reply: `⚠️ ${startResult.error}. Current: "${(agentTaskStatus().instruction || '').slice(0, 60)}"` }));
        }
        const ctl = (await import('./agent-browser.mjs')).getControlTarget();
        const where = ctl.label === 'brave' ? 'inside YOUR Brave' : 'in the stage window';
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ reply: `🤖 Working ${where}: "${instruction.slice(0, 70)}". Streaming progress...`, route: 'browser_task' }));
      }
      const deep = mode === 'fulldive';
      const memories = await getMemories();
      const memoryBlock = memories.slice(-12).map(m => `- ${m.text}`).join('\n');
      const liveContext = await buildLiveContext();
      const msgs = [{ role: 'system', content: personaFor(mode || 'standard', memoryBlock, liveContext) }];
      for (const h of (Array.isArray(history) ? history.slice(-6) : [])) {
        if (h && typeof h.content === 'string' && (h.role === 'user' || h.role === 'assistant')) {
          msgs.push({ role: h.role, content: h.content.slice(0, deep ? 8000 : 4000) });
        }
      }
      msgs.push({ role: 'user', content: String(message).slice(0, deep ? 32000 : 8000) });
      let reply = await callNVIDIA(msgs, settings, deep ? 4000 : 1400, {
        model: deep ? settings.deepKernelModel : undefined,
        timeoutMs: deep ? 240000 : 45000
      });
      const directive = parseToolDirective(reply);
      if (directive) {
        reply = stripToolDirective(reply);
        const toolResult = await runTool(directive);
        reply += `\n\n[⚙️ ${toolResult}]`;
      } else if (/\b(remember|don'?t forget|keep in mind)\b/i.test(String(message))) {
        const fact = String(message).replace(/^.*?\bremember\b\s*(that\s*)?/i, '').trim() || String(message).trim();
        const stored = await runTool({ tool: 'memory', arg: fact.slice(0, 300) });
        reply += `\n\n[⚙️ ${stored}]`;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ reply }));
    }

    if (url.pathname === '/api/memory' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ memories: await getMemories() }));
    }

    if (url.pathname === '/api/memory' && req.method === 'DELETE') {
      await writeFile(join(DATA, 'memory.json'), '[]');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true }));
    }

    if (url.pathname === '/api/chats' && req.method === 'GET') {
      if (!existsSync(CHATS_DIR)) { res.writeHead(200, { 'Content-Type': 'application/json' }); return res.end('{"chats":[]}'); }
      const files = await readdir(CHATS_DIR);
      const chats = [];
      for (const f of files.filter(x => x.endsWith('.json'))) {
        try {
          const c = JSON.parse(await readFile(join(CHATS_DIR, f), 'utf8'));
          chats.push({ id: c.id, title: c.title, savedAt: c.savedAt, turns: (c.messages || []).length });
        } catch (err) { logError('SilentCatch', err); }
      }
      chats.sort((a, b) => (b.savedAt || '').localeCompare(a.savedAt || ''));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ chats }));
    }

    if (url.pathname === '/api/chats/save' && req.method === 'POST') {
      const { id, title, messages } = await readBody(req);
      if (!Array.isArray(messages) || !messages.length) { res.writeHead(400); return res.end('{"error":"nothing to save"}'); }
      const { mkdirSync } = await import('node:fs');
      mkdirSync(CHATS_DIR, { recursive: true });
      const sid = String(id || Date.now().toString(36)).replace(/[^a-z0-9]/gi, '');
      const chat = { id: sid, title: String(title || 'session').slice(0, 80), savedAt: new Date().toISOString(), messages };
      await writeFile(join(CHATS_DIR, `${sid}.json`), JSON.stringify(chat, null, 2));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ id: sid }));
    }

    if (url.pathname === '/api/chats/load' && req.method === 'POST') {
      const { id } = await readBody(req);
      const safe = String(id || '').replace(/[^a-z0-9]/gi, '');
      const file = join(CHATS_DIR, `${safe}.json`);
      if (!safe || !existsSync(file)) { res.writeHead(404); return res.end('{"error":"not found"}'); }
      const chat = JSON.parse(await readFile(file, 'utf8'));
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(chat));
    }

    if (url.pathname === '/api/image' && req.method === 'POST') {
      const { prompt, width, height, seed } = await readBody(req);
      if (!prompt || !prompt.trim()) { res.writeHead(400); return res.end('{"error":"prompt required"}'); }
      const image = await comfyGenerate(prompt.trim(), { width: Number(width) || 832, height: Number(height) || 832, seed });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ image }));
    }

    if (url.pathname === '/api/brief' && req.method === 'POST') {
      const ledger = await loadJSONAsync('ledger');
      const earned = ledger.entries.reduce((s, e) => s + Number(e.amount || 0), 0);
      const summary = `earned $${earned.toFixed(2)} of $${settings.goals.baselineInvested} payback target`;
      const brief = await dailyBrief(settings, summary);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(brief));
    }

    if (url.pathname === '/api/test-nvidia' && req.method === 'POST') {
      const status = await testConnection(settings);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ status }));
    }

    // Extension command queue endpoints
    const extQueue = global.extCommandQueue || [];
    global.extCommandQueue = extQueue;
    const extResults = global.extResults || {};
    global.extResults = extResults;

    if (url.pathname === '/api/ext/poll' && req.method === 'GET') {
      const agentId = url.searchParams.get('agentId') || 'quillan-ronin';
      const cmd = extQueue.shift();
      if (cmd) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(cmd));
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({}));
    }

    if (url.pathname === '/api/ext/result' && req.method === 'POST') {
      const { commandId, result, agentId } = await readBody(req);
      extResults[commandId] = { result, agentId, receivedAt: new Date().toISOString() };
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true }));
    }

    if (url.pathname === '/api/ext/queue' && req.method === 'POST') {
      const { op, params } = await readBody(req);
      const commandId = Date.now().toString(36) + Math.random().toString(36).slice(2);
      const cmd = { id: commandId, op, params, queuedAt: new Date().toISOString() };
      extQueue.push(cmd);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ commandId }));
    }

    if (url.pathname === '/api/ext/result' && req.method === 'GET') {
      const commandId = url.searchParams.get('commandId');
      if (!commandId) { res.writeHead(400); return res.end('{"error":"commandId required"}'); }
      const result = extResults[commandId];
      if (result) {
        delete extResults[commandId];
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(result));
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({}));
    }

    // Extension context management endpoints
    if (url.pathname === '/api/ext/context' && req.method === 'GET') {
      const tabId = url.searchParams.get('tabId');
      if (!tabId) { res.writeHead(400); return res.end('{"error":"tabId required"}'); }
      const { sendCmd } = await import('./ext-bridge.mjs');
      const result = await sendCmd('getContext', { tabId });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(result));
    }

    if (url.pathname === '/api/ext/contexts' && req.method === 'GET') {
      const { sendCmd } = await import('./ext-bridge.mjs');
      const result = await sendCmd('getAllContexts', {});
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(result));
    }

    if (url.pathname === '/api/ext/autonomy' && req.method === 'POST') {
      const { tabId, enable } = await readBody(req);
      if (!tabId) { res.writeHead(400); return res.end('{"error":"tabId required"}'); }
      const { sendCmd } = await import('./ext-bridge.mjs');
      const result = enable 
        ? await sendCmd('enableAutonomy', { tabId })
        : await sendCmd('disableAutonomy', { tabId });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(result));
    }

    if (url.pathname === '/api/ext/autonomy/status' && req.method === 'GET') {
      const tabId = url.searchParams.get('tabId');
      if (!tabId) { res.writeHead(400); return res.end('{"error":"tabId required"}'); }
      const { sendCmd } = await import('./ext-bridge.mjs');
      const result = await sendCmd('isAutonomyEnabled', { tabId });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify(result));
    }

    let path = url.pathname === '/' ? '/index.html' : url.pathname;
    const file = join(PUBLIC, path.replace(/^\/+/, ''));
    if (!file.startsWith(PUBLIC) || !existsSync(file)) { res.writeHead(404); return res.end('nope'); }
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' });
    return res.end(body);
  } catch (e) {
    const msg = String(e?.cause?.message || e.message || e);
    try { await writeFile(join(DATA, 'error.log'), `${new Date().toISOString()} ${url.pathname}: ${msg}\n${(e.stack || '').slice(0, 800)}\n---\n`, { flag: 'a' }); } catch (err) { logError('SilentCatch', err); }
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: msg }));
  }
});

const PORT = 7777;

// TCP NoDelay & KeepAlive Optimization for Sub-0.5ms Local Latency
server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.on('connection', (socket) => {
  try {
    socket.setNoDelay(true); // Disable Nagle's algorithm for instant packet dispatch
    socket.setKeepAlive(true, 60000);
  } catch (err) { logError('SilentCatch', err); }
});

server.listen(PORT, async () => {
  console.log(`Quillan Worker dashboard → http://localhost:${PORT}`);
  try {
    const { setControlTarget } = await import('./agent-browser.mjs');
    const ctlFile = join(DATA, 'control-target.json');
    let ctl = { port: 9222, label: 'stage' };
    if (existsSync(ctlFile)) {
      ctl = JSON.parse(await readFile(ctlFile, 'utf8'));
      setControlTarget(ctl.port, ctl.label);
      console.log(`[boot] control target restored: ${ctl.label} :${ctl.port}`);
    }
    const desire = existsSync(join(DATA, 'chess-autopilot-enabled.json'))
      ? JSON.parse(await readFile(join(DATA, 'chess-autopilot-enabled.json'), 'utf8')).enabled
      : false;
    if (desire && existsSync(join(ROOT, 'agent-chess.mjs'))) {
      setTimeout(async () => {
        try {
          const { startAutopilot } = await import('./agent-chess.mjs');
          const r = startAutopilot('w');
          console.log('[boot] chess autopilot re-engaged:', JSON.stringify(r));
        } catch (e) { console.log('[boot] autopilot re-engage failed:', e.message); }
      }, 8000);
    }
  } catch (err) { logError('SilentCatch', err); }
});

