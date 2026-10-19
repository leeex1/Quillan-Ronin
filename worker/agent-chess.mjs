import { getSession, ensureActive, evalStable, activePage, pickBestPage } from './agent-browser.mjs';

const FILE_LETTERS = 'abcdefgh';

async function boardStatus() {
  let s = getSession();
  if (!s) s = await ensureActive();
  return s;
}

export async function readBoard(attempts = 2) {
  let res = null;
  let pickedUrl = '';

  // 1. Try Playwright remote-debugging CDP page first
  for (let i = 0; i < attempts; i++) {
    try {
      const pg = await pickBestPage().catch(() => null);
      if (pg) {
        pickedUrl = pg.url();
        res = await evalStable(() => {
          const list = [];
          document.querySelectorAll('[class*="square-"]').forEach(el => {
            const cls = typeof el.className === 'string' ? el.className : (el.className.baseVal || '');
            if (!cls.includes('piece')) return;
            const m = cls.match(/piece\s+([wb])([pnbrqk])\s+square-(\d)(\d)\b/);
            if (!m) return;
            list.push({ sq: 'abcdefgh'[(+m[3]) - 1] + m[4], color: m[1], type: m[2] });
          });
          return { list };
        }).catch(e => ({ error: e.message.slice(0, 60) }));
        if (res && res.list && res.list.length >= 4) break;
      }
    } catch {}
    if (i < attempts - 1) await new Promise(r => setTimeout(r, 600));
  }

  // 2. Fallback: Query live tab via Extension Bridge if Playwright not connected
  if (!res || !res.list || res.list.length < 4) {
    try {
      const { sendCmd } = await import('./ext-bridge.mjs');
      const extRes = await sendCmd('eval', {
        expression: `(() => {
          const list = [];
          document.querySelectorAll('[class*="square-"]').forEach(el => {
            const cls = typeof el.className === 'string' ? el.className : (el.className.baseVal || '');
            if (!cls.includes('piece')) return;
            const m = cls.match(/piece\\s+([wb])([pnbrqk])\\s+square-(\\d)(\\d)\\b/);
            if (!m) return;
            list.push({ sq: 'abcdefgh'[(+m[3]) - 1] + m[4], color: m[1], type: m[2] });
          });
          return JSON.stringify({ list });
        })()`
      });
      if (extRes && extRes.result && extRes.result.value) {
        const parsed = JSON.parse(extRes.result.value);
        if (parsed && parsed.list) res = parsed;
      }
    } catch {}
  }

  const map = {};
  for (const p of ((res && res.list) || [])) if (p.sq) map[p.sq] = p.color + p.type;
  return { 
    count: Object.keys(map).length, 
    map, 
    debug: (res && (res.error || res.dbg)) || (Object.keys(map).length === 0 ? 'No active chess board detected on current tab' : null) 
  };
}


export async function makeMove(from, to) {
  const p = await activePage();
  const geom = await p.evaluate(({ fromSq, toSq }) => {
    const anyPiece = document.querySelector('[class*="piece"][class*="square-"]');
    if (!anyPiece) return null;
    const pr = anyPiece.getBoundingClientRect();
    const size = pr.width;
    const tf = new DOMMatrixReadOnly(getComputedStyle(anyPiece).transform);
    const originX = pr.left - tf.e;
    const originY = pr.top - tf.f;
    const cc = s => ({ col: 'abcdefgh'.indexOf(s[0]) + 1, row: +s[1] });
    const f = cc(fromSq), t = cc(toSq);
    return {
      fx: originX + size * (f.col - 0.5),
      fy: originY + size * (8 - f.row + 0.5),
      tx: originX + size * (t.col - 0.5),
      ty: originY + size * (8 - t.row + 0.5)
    };
  }, { fromSq: from, toSq: to });

  if (!geom) throw new Error('could not compute board geometry');
  await p.mouse.click(geom.fx, geom.fy);
  await p.waitForTimeout(300);
  await p.mouse.click(geom.tx, geom.ty);
  return { ok: true, from, to };
}

export async function waitForOpponentMove(prevMap, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2000));
    const cur = await readBoard();
    const changed = Object.keys(cur.map).length !== Object.keys(prevMap).length ||
      Object.entries(cur.map).some(([sq, pc]) => prevMap[sq] !== pc);
    if (changed) return cur;
  }
  return null;
}

function mapToPlacement(map) {
  const rows = [];
  for (let rank = 8; rank >= 1; rank--) {
    let row = '', empty = 0;
    for (const file of 'abcdefgh') {
      const pc = map[file + rank];
      if (!pc) { empty++; continue; }
      if (empty) { row += empty; empty = 0; }
      row += pc[0] === 'w' ? pc[1].toUpperCase() : pc[1];
    }
    if (empty) row += empty;
    rows.push(row);
  }
  return rows.join('/');
}

let lastKnownMapJSON = null;

async function detectTurn(ourColor) {
  const p = await activePage();
  return p.evaluate((mine) => {
    const highs = [...document.querySelectorAll('[class*="highlight"]')]
      .map(h => (h.className.toString().match(/square-(\d\d)/) || [])[1])
      .filter(Boolean);
    if (!highs.length) return { turn: 'unknown' };
    const destCode = highs[highs.length - 1];
    const destSq = 'abcdefgh'[(+destCode[0]) - 1] + destCode[1];
    let occupant = null;
    document.querySelectorAll('[class*="piece"]').forEach(el => {
      const m = el.className.toString().match(/piece\s+([wb])([pnbrqk])\s+square-(\d)(\d)\b/);
      if (m && +m[3] === +destCode[0] && +m[4] === +destCode[1]) occupant = m[1];
    });
    if (!occupant) return { turn: 'unknown' };
    return { turn: occupant === mine ? 'theirs' : 'ours', movedTo: destSq };
  }, ourColor);
}

export async function autonomousTurn(map, ourColor = 'w') {
  const keyCount = Object.keys(map || {}).length;
  if (!map || keyCount < 4) return { status: 'game-likely-over', debugKeys: keyCount };

  const snapshot = JSON.stringify(map);
  if (lastKnownMapJSON && snapshot === lastKnownMapJSON) {
    return { status: 'waiting-for-opponent', note: 'board unchanged since our last move' };
  }

  const turnCheck = await detectTurn(ourColor).catch(() => ({ turn: 'unknown' }));
  if (turnCheck.turn === 'theirs') {
    return { status: 'not-our-turn', detail: `opponent piece landed on ${turnCheck.movedTo}` };
  }
  void turnCheck;
  const { Chess } = await import('chess.js');
  const { callNVIDIA, loadJSON } = await import('./scanner.mjs');

  const placement = mapToPlacement(map);
  const fen = `${placement} ${ourColor} - - 0 1`;

  let legal;
  try {
    const g = new Chess(fen);
    legal = g.moves({ verbose: true }).map(m => ({ san: m.san, from: m.from, to: m.to }));
  } catch (e) {
    return { status: 'fen-error', error: e.message, fen };
  }
  if (!legal.length) return { status: 'no-legal-moves', fen };

  const settings = loadJSON('settings');
  const choices = legal.slice(0, 40).map(l => l.san).join(', ');
  const reply = await callNVIDIA([
    { role: 'system', content: 'You are a chess engine. Given a FEN position and a list of legal moves, pick the strongest move. Reply with ONLY the SAN move string, nothing else.' },
    { role: 'user', content: `FEN: ${fen}\nLegal moves: ${choices}\nYour color: ${ourColor === 'w' ? 'white' : 'black'}\nBest move (SAN only):` }
  ], settings);

  const chosenSan = reply.trim().replace(/[^\w+#=\-]/g, '');
  const match = legal.find(l => l.san === chosenSan);
  if (!match) return { status: 'illegal-choice', modelSaid: reply.trim().slice(0, 50), legalCount: legal.length };

  await makeMove(match.from, match.to);
  lastKnownMapJSON = JSON.stringify(await readBoard());
  return { status: 'moved', san: match.san, from: match.from, to: match.to };
}

let autopilot = { running: false, color: "w", events: [], startedAt: null };

export function autopilotStatus() {
  return { ...autopilot, events: autopilot.events.slice(-10) };
}

function apLog(msg) {
  autopilot.events.push({ t: new Date().toISOString(), msg });
  if (autopilot.events.length > 50) autopilot.events.shift();
}

export function stopAutopilot() {
  autopilot.running = false;
  apLog("autopilot disengaged");
  return { running: false };
}

export function startAutopilot(ourColor = "w") {
  if (autopilot.running) return { running: true, note: "already running" };
  autopilot = { running: true, color: ourColor, events: [], startedAt: new Date().toISOString() };
  apLog("autopilot engaged as " + ourColor);
  runLoop();
  return { running: true };
}

async function runLoop() {
  const first = await readBoard();
  let lastMap = first.map;
  apLog(`watching board (${Object.keys(lastMap).length} pieces)`);

  while (autopilot.running) {
    try {
      await new Promise(r => setTimeout(r, 2500));
      const cur = await readBoard();
      const curCount = Object.keys(cur.map).length;
      if (curCount < 6) { apLog(`board nearly empty (${curCount} pieces) - game over?`); stopAutopilot(); break; }
      const changed = JSON.stringify(cur.map) !== JSON.stringify(lastMap);
      if (!changed) continue;

      apLog('detected opponent move - thinking...');
      const result = await autonomousTurn(cur.map, autopilot.color);
      if (result.status === 'moved') {
        apLog(`played ${result.san}`);
        await new Promise(r => setTimeout(r, 1500));
        lastMap = (await readBoard()).map;
      } else if (result.status === 'waiting-for-opponent' || result.status === 'not-our-turn') {
        apLog(`holding (${result.status})`);
        lastMap = cur;
      } else {
        apLog(`needs assist: ${result.status} ${result.modelSaid || ''}`);
        autopilot.needsAssist = true;
        break;
      }
    } catch (e) {
      apLog(`error: ${e.message.slice(0, 80)}`);
      await new Promise(r => setTimeout(r, 3000));
    }
  }
}
