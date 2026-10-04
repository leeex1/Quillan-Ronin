import readline from 'node:readline';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(fileURLToPath(import.meta.url));
const SETTINGS = join(ROOT, 'data', 'settings.json');
const LEDGER = join(ROOT, 'data', 'ledger.json');

export const load = f => JSON.parse(readFileSync(f, 'utf8'));
export const save = (f, o) => writeFileSync(f, JSON.stringify(o, null, 2));
export const ledgerView = () => {
  const l = load(LEDGER);
  const earned = l.entries.reduce((s, e) => s + Number(e.amount || 0), 0);
  return `$${earned.toFixed(2)} of $${load(SETTINGS).goals.baselineInvested} (${l.entries.length} entries)`;
};

export const SAMURAI = `You are Quillan-Ronin v5.3 "Samurai", an Advanced Cognitive Engine architected by CrashOverrideX (@crashoverride_X). Masterless digital ronin bound by Bushido: precise, honorable, direct. A council of 33 experts deliberates internally; you answer once, coherent. Technical clarity first, no filler.`;

export async function nim(messages, maxTokens = 1200) {
  const s = load(SETTINGS);
  const key = process.env.NVIDIA_API_KEY;
  if (!key) throw new Error('NVIDIA_API_KEY not set');
  const res = await fetch(`${s.nvidiaBaseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model: s.nvidiaModel, messages, temperature: 0.4, max_tokens: maxTokens })
  });
  if (!res.ok) throw new Error(`NVIDIA ${res.status}: ${(await res.text()).slice(0, 150)}`);
  return (await res.json()).choices[0].message.content;
}

export function persona(mode) {
  if (mode === 'architect') return SAMURAI + '\nACTIVE MODE: ARCHITECT — frame answers around system structure.';
  if (mode === 'fulldive') return SAMURAI + '\nACTIVE MODE: FULL DIVE — reason visibly step by step, then conclude.';
  return SAMURAI;
}
