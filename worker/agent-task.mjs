import { callNVIDIA, loadJSON, callToolViaMcp } from './task-support.mjs';

let task = null;

export function taskStatus() {
  if (!task) return { exists: false };
  return {
    running: task.running,
    paused: !!task.paused,
    question: task.question || null,
    instruction: task.instruction,
    plan: task.plan || [],
    step: task.step,
    maxSteps: task.maxSteps,
    events: task.events.slice(-14),
    needsHelp: task.needsHelp || null,
    result: task.result || null
  };
}

export function stopTask(reason = 'user requested stop') {
  if (!task) return { stopped: false };
  task.running = false;
  task.result = { outcome: 'stopped', reason };
  return { stopped: true };
}

function tLog(msg) {
  if (!task) return;
  task.events.push({ t: Date.now(), msg });
  if (task.events.length > 120) task.events.shift();
}

const CAPTCHA_SIGNS = ['captcha', 'verify you are human', 'prove you are human', 'cf-challenge', 'g-recaptcha', 'h-captcha', 'are-you-a-robot'];

function looksLikeCaptcha(snapshotText) {
  const hay = (snapshotText || '').toLowerCase();
  return CAPTCHA_SIGNS.some(s => hay.includes(s));
}

export async function startTask(instruction, maxSteps = 18, answers = null, resumeData = null) {
  if (task && task.running) return { error: 'a task is already running', current: task.instruction };

  const fresh = !resumeData;
  task = fresh
    ? { running: true, instruction, step: 0, maxSteps, events: [], transcript: [], plan: [], answers: {}, paused: null }
    : Object.assign(resumeData.task, { running: true, paused: null });

  if (answers && task.pendingQuestion) {
    task.answers[task.pendingQuestion.field || 'q' + task.step] = answers.answer || '';
    task.transcript.push({ role: 'user', text: '[answered] ' + (answers.answer || '') });
  }

  runTaskLoop().catch(e => {
    tLog('fatal: ' + e.message.slice(0, 100));
    task.running = false;
    task.needsHelp = e.message.slice(0, 140);
  });
  return { started: fresh ? 'new' : 'resumed', instruction: task.instruction };
}

export async function resumeTask(answer) {
  if (!task) return { error: 'no task' };
  if (task.paused && task.paused.kind === 'question') {
    task.answers[task.paused.field] = answer;
    task.transcript.push({ role: 'user', text: '[answer] ' + answer });
    task.paused = null;
    task.running = true;
    tLog(`human answered: ${String(answer).slice(0, 60)}`);
    runTaskLoop().catch(e => { tLog('fatal: ' + e.message); task.running = false; });
    return { resumed: true };
  }
  if (task.paused && task.paused.kind === 'captcha') {
    task.paused = null;
    task.running = true;
    tLog('human handled captcha - resuming');
    runTaskLoop().catch(e => { tLog('fatal: ' + e.message); task.running = false; });
    return { resumed: true };
  }
  return { error: 'nothing paused' };
}

async function pauseFor(kind, question, field) {
  task.paused = { kind, question: question || '', field: field || '' };
  task.running = false;
  tLog(kind === 'captcha' ? 'CAPTCHA detected - human required' : `asking human: ${String(question).slice(0, 80)}`);
}

async function mcp(tool, args) {
  const { callTool } = await import('./mcp-manager.mjs');
  const r = await callTool('ChromeDevTools', tool, args, 60000);
  return r.text || '';
}

async function visionClick(targetDescription) {
  const { nativeScreenshot } = await import('./agent-browser.mjs');
  const { visionDescribe } = await import('./scanner.mjs');
  const { callTool } = await import('./mcp-manager.mjs');
  const settings = loadJSON('settings');

  const b64 = await nativeScreenshot();
  const q = `Find this on screen: "${targetDescription}". The image is HALF-SCALE of the real screen (real screen is 2x larger). Reply ONLY JSON: {"found":true,"x":<center x in IMAGE pixels>,"y":<center y in IMAGE pixels>} or {"found":false,"reason":"..."}`;
  const raw = await visionDescribe(b64, q, settings);
  let coords = null;
  try { coords = JSON.parse(raw.slice(raw.indexOf('{'), raw.lastIndexOf('}') + 1)); } catch {}
  if (!coords || !coords.found) return { ok: false, note: raw.slice(0, 120) };
  const x = Math.round((coords.x || 0) * 2);
  const y = Math.round((coords.y || 0) * 2);
  await callTool('ComputerUse', 'Click', { loc: [x, y] });
  return { ok: true, x, y, target: targetDescription };
}

async function observe() {
  const snap = await mcp('take_snapshot', {});
  if (looksLikeCaptcha(snap)) return { snap, captcha: true };
  return { snap, captcha: false };
}

async function runTaskLoop() {
  const settings = loadJSON('settings');
  await mcp('navigate_page', { url: 'about:blank', type: 'url' }).catch(() => {});

  if (!(task.plan || []).length) {
    tLog('planning...');
    let memLine = '';
    try {
      const { readFileSync } = await import('node:fs');
      const memPath = new URL('./data/memory.json', import.meta.url);
      const mems = JSON.parse(readFileSync(memPath, 'utf8').replace(/^\uFEFF/, ''));
      if (Array.isArray(mems) && mems.length) memLine = 'Known user facts: ' + mems.slice(-10).map(m => m.text).join('; ');
    } catch {}
    const planRaw = await callNVIDIA([
      { role: 'system', content: 'Turn vague user goals into short concrete browser checklists. Reply ONLY JSON: {"plan":["step1","step2"],"questions":[{"field":"name","question":"..."}]}. Max 8 steps. Ask a question ONLY if the goal is impossible without the answer.' },
      { role: 'user', content: `GOAL: ${task.instruction}\n${memLine}` }
    ], settings);
    try {
      const j = JSON.parse(planRaw.slice(planRaw.indexOf('{'), planRaw.lastIndexOf('}') + 1));
      task.plan = j.plan || [];
      for (const q of (j.questions || []).slice(0, 2)) {
        if (!(q.field in (task.answers || {}))) {
          task.pendingQuestion = q;
          await pauseFor('question', q.question, q.field);
          return;
        }
      }
      task.plan.forEach((p2, i) => tLog(`plan ${i + 1}: ${p2}`));
    } catch { tLog('planning parse failed - freestyle'); }
    task.planDone = true;
  }

  let lastSnap = '';
  let failStreak = 0;

  while (task.running && task.step < task.maxSteps) {
    task.step++;
    const { snap, captcha } = await observe();
    if (captcha) { await pauseFor('captcha', 'CAPTCHA appeared - solve it in the browser, then Resume.'); return; }

    const decisionRaw = await callNVIDIA([
      { role: 'system', content: `You are Quillan's browser-hands, driving a real Chrome via the Chrome DevTools protocol. You get an accessibility-tree snapshot where every interactive element has a uid (like "1_8").

Output EXACTLY ONE JSON directive:
{"op":"click","uid":"1_8","expect":"what changes after"}
{"op":"vision_click","target":"description of thing to click","expect":"..."}
{"op":"fill","uid":"1_5","text":"...","expect":"field now contains..."}
{"op":"navigate","url":"https://...","expect":"..."}
{"op":"press","key":"Enter","expect":"..."}
{"op":"done","summary":"evidence the GOAL is achieved"}
{"op":"help","reason":"why stuck"}

Rules: one action per turn. Use uids EXACTLY as listed in the snapshot - never invent them. If the last verify said NO CHANGE, change strategy. done requires evidence the GOAL is met, not just that an action succeeded.` },
      { role: 'user', content: `GOAL: ${task.instruction}\nPLAN: ${(task.plan || []).join(' | ') || 'freestyle'}\nSTEP ${task.step}/${task.maxSteps}\nKNOWN: ${JSON.stringify(task.answers || {})}\n${task.lastVerify ? 'LAST VERIFY: ' + task.lastVerify + '\n(If NO CHANGE: critique and change strategy - RCI loop)\n' : ''}\nPAGE SNAPSHOT (a11y tree with uids):\n${snap.slice(0, 6000)}` }
    ], settings, 700, { model: settings.deepKernelModel, timeoutMs: 90000 });

    let d = null;
    try { d = JSON.parse(decisionRaw.match(/\{[\s\S]*\}/)[0]); } catch { tLog('unparseable directive'); continue; }

    const op = (d.op || '').toLowerCase();
    let acted = false;

    try {
      if (op === 'navigate') { await mcp('navigate_page', { url: d.url, type: 'url' }); acted = true; tLog('goto ' + d.url); await new Promise(r => setTimeout(r, 2000)); }
      else if (op === 'click') { await mcp('click', { uid: d.uid }); acted = true; tLog(`clicked ${d.uid}`); await new Promise(r => setTimeout(r, 1800)); }
      else if (op === 'fill') { await mcp('fill', { uid: d.uid, value: String(d.text || '') }); acted = true; tLog(`filled ${d.uid}`); }
      else if (op === 'press') { await mcp('press_key', { key: d.key || 'Enter' }); acted = true; tLog(`pressed ${d.key}`); await new Promise(r => setTimeout(r, 1500)); }
      else if (op === 'vision_click') { const r = await visionClick(d.target || d.text || ''); tLog(r.ok ? `vision-clicked (${r.x},${r.y})` : `vision miss: ${r.note}`); acted = r.ok; }
      else if (op === 'done') { task.running = false; task.result = { outcome: 'completed', summary: d.summary || '' }; tLog('GOAL COMPLETE: ' + (d.summary || '').slice(0, 90)); return; }
      else if (op === 'help') { task.running = false; task.needsHelp = d.reason || 'blocked'; tLog('NEEDS HELP: ' + task.needsHelp); return; }
      else { tLog('unknown op ' + d.op); }
    } catch (e) {
      tLog(`action ${op} failed: ${e.message.slice(0, 60)}`);
      failStreak++;
    }

    if (acted) {
      await new Promise(r => setTimeout(r, 900));
      const fresh = await observe().catch(() => ({ snap: '' }));
      const changed = fresh.snap !== lastSnap;
      task.lastVerify = `expected "${(d.expect || '').slice(0, 60)}" | snapshot changed: ${changed}`;
      if (!changed) {
        failStreak++;
        if (failStreak >= 2) {
          task.running = false;
          task.needsHelp = `stuck: two actions, no page change. Intent: ${d.expect || op}`;
          tLog('STUCK - escalating to human');
          return;
        }
      } else failStreak = 0;
      lastSnap = fresh.snap;
      tLog('verify: ' + task.lastVerify);
    }
  }

  if (task.running) {
    task.running = false;
    task.needsHelp = 'max steps reached';
    tLog('max steps reached');
  }
}
