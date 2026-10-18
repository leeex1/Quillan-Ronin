import readline from 'node:readline';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { load, save, ledgerView, nim, persona, SAMURAI } from './agent-core.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const SETTINGS = join(ROOT, 'data', 'settings.json');
const LEDGER = join(ROOT, 'data', 'ledger.json');

let mode = 'standard';
let history = [];

async function brief() {
  const j = await nim([
    { role: 'system', content: persona(mode) + '\nProduce a practical daily action brief for reaching income goals via freelance platforms, digital products and NFT marketing. Realistic JSON only.' },
    { role: 'user', content: `Earnings state: ${ledgerView()}. Return JSON keys: focus(string), tasks(array of {action, platform, effortMinutes}), notes(string).` }
  ], 900);
  try {
    const b = JSON.parse(j.slice(j.indexOf('{'), j.lastIndexOf('}') + 1));
    console.log(`\nFOCUS: ${b.focus}\n`);
    for (const t of (b.tasks || [])) console.log(` > ${t.action} [${t.platform} · ~${t.effortMinutes}m]`);
    if (b.notes) console.log(`\nnotes: ${b.notes}`);
  } catch { console.log('\n' + j); }
}

async function analyze(text) {
  const p = load(SETTINGS).userProfile;
  const j = await nim([
    { role: 'system', content: `${persona(mode)}\nEvaluate this gig/opportunity for ${p.handle}. Skills: ${p.skills.join(', ')}. Reply with: score/10, category, one-line summary, suggested USD price, then a 130-word honest AI-disclosed proposal draft.` },
    { role: 'user', content: text }
  ], 700);
  console.log('\n' + j + '\n');
}

function addLedger(amountStr, source) {
  const amt = parseFloat(amountStr);
  if (!Number.isFinite(amt) || amt <= 0) return console.log('usage: /ledger <amount> <source>');
  const l = load(LEDGER);
  l.entries.push({ amount: amt, source: source || 'other', date: new Date().toISOString() });
  save(LEDGER, l);
  console.log('logged. ' + ledgerView());
}

console.log('QUILLAN-RONIN STANDALONE v1.0 - bushido kernel active');
console.log(`model: ${load(SETTINGS).nvidiaModel} | earnings: ${ledgerView()}`);
console.log('/mode <std|arch|dive> /brief /analyze <text> /ledger <amt> <src> /status /quit\n');

const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });
rl.prompt();

rl.on('line', async line => {
  const t = line.trim();
  if (!t) return rl.prompt();
  const sp = t.indexOf(' ');
  const cmd = (sp === -1 ? t : t.slice(0, sp)).toLowerCase();
  const rest = sp === -1 ? '' : t.slice(sp + 1);
  try {
    if (cmd === '/quit' || cmd === '/exit') { console.log('the blade rests.'); process.exit(0); }
    else if (cmd === '/status') console.log(`mode=${mode} | ${ledgerView()} | ${history.length} history msgs`);
    else if (cmd === '/mode') {
      const map = { std: 'standard', arch: 'architect', dive: 'fulldive' };
      const m = map[rest] || rest;
      if (['standard', 'architect', 'fulldive'].includes(m)) { mode = m; console.log('mode -> ' + mode); }
      else console.log('modes: std | arch | dive');
    }
    else if (cmd === '/brief') await brief();
    else if (cmd === '/analyze') rest ? await analyze(rest) : console.log('usage: /analyze <gig text>');
    else if (cmd === '/ledger') { const [a, ...s] = rest.split(' '); addLedger(a, s.join(' ')); }
    else {
      const reply = await nim([{ role: 'system', content: persona(mode) }, ...history.slice(-6), { role: 'user', content: t }]);
      console.log(reply + '\n');
      history.push({ role: 'user', content: t }, { role: 'assistant', content: reply });
      if (history.length > 20) history = history.slice(-20);
    }
  } catch (e) { console.log('ERR: ' + e.message); }
  rl.prompt();
});
