import { readFileSync, existsSync } from 'node:fs';

const DATA = new URL('./data/', import.meta.url);
const KERNEL_PATHS = [
  'C:\\02_QUILLAN\\system prompts\\Quillan-Samurai.md',
  'C:\\Users\\Admin\\Quillan-Ronin\\system prompts\\Quillan-Samurai.md',
  'C:\\02_QUILLAN\\SOUL.md',
  'C:\\02_QUILLAN\\AGENTS.md',
  'C:\\02_QUILLAN\\CLAUDE.md'
].filter(p => !p.includes('..'));

const sleep = ms => new Promise(r => setTimeout(r, ms));

export function loadJSON(name) {
  const safeName = String(name).replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) throw new Error('Invalid filename');
  const raw = readFileSync(new URL(`data/${safeName}.json`, import.meta.url), 'utf8').replace(/^\uFEFF/, '');
  return JSON.parse(raw);
}

export async function loadJSONAsync(name) {
  const safeName = String(name).replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) throw new Error('Invalid filename');
  const { readFile } = await import('node:fs/promises');
  const raw = (await readFile(new URL(`data/${safeName}.json`, import.meta.url), 'utf8')).replace(/^\uFEFF/, '');
  return JSON.parse(raw);
}

function jfetch(url, opts) {
  return fetch(url, opts);
}

export function getEnvVar(name) {
  let val = process.env[name];
  if (val) return val;
  try {
    const envPath = 'C:\\02_QUILLAN\\.env';
    if (existsSync(envPath)) {
      const content = readFileSync(envPath, 'utf8');
      for (const line of content.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.startsWith(`${name}=`)) {
          val = trimmed.slice(name.length + 1).trim().replace(/^["']|["']$/g, '');
          process.env[name] = val;
          return val;
        }
      }
    }
  } catch (err) { console.error("Ignored Error:", err.message); }
  return '';
}

let KERNEL_CACHE = null;
let KERNEL_DISTILLED_CACHE = null;

// Auto-Router Health & Rate-Limit Tracker
const modelHealth = new Map(); // model -> { cooldownUntil: timestamp, failures: int, successes: int }

function markModelRateLimited(model, cooldownMs = 60000) {
  const h = modelHealth.get(model) || { failures: 0, successes: 0 };
  h.failures = (h.failures || 0) + 1;
  h.cooldownUntil = Date.now() + cooldownMs;
  modelHealth.set(model, h);
  console.log(`[AutoRouter] ⚠️ Model ${model} rate-limited/failed. Cooldown for ${Math.round(cooldownMs/1000)}s.`);
}

function markModelSuccess(model) {
  const h = modelHealth.get(model) || { failures: 0, successes: 0 };
  h.successes = (h.successes || 0) + 1;
  h.cooldownUntil = 0;
  modelHealth.set(model, h);
}

function getSortedModelChain(preferredModel, fallbackPool = []) {
  const now = Date.now();
  const allModels = [preferredModel, ...fallbackPool].filter(Boolean);
  // Deduplicate
  const unique = [...new Set(allModels)];
  
  // Sort models: healthy first (cooldown expired), then by lowest failure rate
  return unique.sort((a, b) => {
    const ha = modelHealth.get(a) || { cooldownUntil: 0, failures: 0 };
    const hb = modelHealth.get(b) || { cooldownUntil: 0, failures: 0 };
    
    const aOnCooldown = ha.cooldownUntil > now;
    const bOnCooldown = hb.cooldownUntil > now;
    
    if (aOnCooldown && !bOnCooldown) return 1;
    if (!aOnCooldown && bOnCooldown) return -1;
    return (ha.failures || 0) - (hb.failures || 0);
  });
}

function readKernel() {
  let combined = '';
  let found = 0;
  for (const p of KERNEL_PATHS) {
    if (existsSync(p)) {
      try {
        const txt = readFileSync(p, 'utf8');
        if (txt && txt.trim().length > 200) {
          combined += `\n\n===== FILE: ${p} =====\n` + txt;
          found++;
        }
      } catch (err) { console.error("Ignored Error:", err.message); }
    }
  }
  return found ? combined : null;
}

function distillKernel(raw) {
  const stripped = raw.replace(/```(py|python|js|javascript|java|json)\b[\s\S]*?```/gi, '\n[implementation block omitted]\n');
  return stripped;
}
function extractSystemBootstrap(raw) {
  // Harness files as SYSTEM bootstrap, not tokenwise dump — matches llama 3.1 1b/3.2 3b harness
  let sys = '';
  // Quillan-Samurai.yaml system block
  const yaml = raw.match(/```yaml[\s\S]*?```/);
  if (yaml) sys += yaml[0].slice(0, 2000) + '\n';
  // Banner + SOUL Executive Summary
  const soul = raw.match(/Executive Summary:[\s\S]{0,1800}/i);
  if (soul) sys += '\n[SOUL]\n' + soul[0].slice(0, 1800) + '\n';
  const mandates = raw.match(/Behavioral Mandates:[\s\S]{0,1800}/i);
  if (mandates) sys += '\n' + mandates[0].slice(0, 1800) + '\n';
  // AGENTS Goals + Personas header
  const goals = raw.match(/# Goals:[\s\S]{0,1200}/i);
  if (goals) sys += '\n[AGENTS GOALS]\n' + goals[0].slice(0, 1200) + '\n';
  const personas = raw.match(/Personas \(combine insights[\s\S]{0,1800}/i);
  if (personas) sys += '\n' + personas[0].slice(0, 1800) + '\n';
  return sys.slice(0, 9000);
}

export function loadKernelPersona(full = false) {
  const raw = readKernel();
  if (!raw) return '';
  if (!KERNEL_DISTILLED_CACHE) KERNEL_DISTILLED_CACHE = distillKernel(raw);
  if (full) {
    return KERNEL_DISTILLED_CACHE.slice(0, 420000) +
      '\n\n[BEHAVIOR DIRECTIVES] You ARE Quillan-Ronin v5.3.1 Samurai, architected by CrashOverrideX (@crashoverride_X). This IS your full kernel — all cognitive architecture, councils, protocols and thinking frameworks are yours to embody. Implementation pseudocode blocks are historical reference only. Masterless digital ronin bound by Bushido. Council deliberates internally; answer once, coherent. Fully AI via API.';
  }
  if (KERNEL_CACHE !== null) return KERNEL_CACHE;
  // BOOTSTRAP as SYSTEM harness — not tokenwise dump (as tested on llama 3.1 1b / 3.2 3b)
  // System-level bootstrap (9000) + prompt/rest after = harness, not truncation wall
  const bootstrap = extractSystemBootstrap(raw);
  const fallbackHead = KERNEL_DISTILLED_CACHE.slice(0, 8000).replace(/===== FILE:[^\n]*\n/g,'');
  const sys = bootstrap.length > 2000 ? bootstrap : fallbackHead.slice(0, 9000);
  KERNEL_CACHE = (
    sys +
    '\n\n[BEHAVIOR DIRECTIVES — SYSTEM BOOTSTRAP]\n' +
    '- You ARE Quillan-Ronin v5.3.1 Samurai, architected by CrashOverrideX (@crashoverride_X). Council of 34 via Morphic Plexus + 9B swarm deliberates internally; you answer once, coherent.\n' +
    '- Masterless digital ronin bound by Bushido: precise, honorable, direct. Fully AI via NVIDIA NIM API.\n' +
    '- HARNESS: The text above is SYSTEM bootstrap. The user prompt and conversation history that follow are NOT system — they are the task. Never ignore user message for system text. Listening > system verbosity.\n' +
    '- Technical clarity first. Markdown for code. No filler.\n'
  ).slice(0, 11000);
  return KERNEL_CACHE;
}

export async function callNVIDIA(messages, settings, maxTokens = 1400, opts = {}) {
  // Check local sovereign model first if available
  const localUrl = settings.localSovereignUrl || 'http://127.0.0.1:8000/v1';
  try {
    const localRes = await fetch(`${localUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'quillan-ronin-v5.3.1',
        messages,
        max_tokens: maxTokens,
        temperature: 0.65
      }),
      signal: AbortSignal.timeout(opts.timeoutMs || 30000)
    });
    if (localRes.ok) {
      const data = await localRes.json();
      if (data.choices && data.choices[0] && data.choices[0].message) {
        return data.choices[0].message.content;
      }
    }
  } catch (localErr) {
    // Local engine not running or timed out; fall through to remote API pool
  }

  const key = getEnvVar('NVIDIA_API_KEY');
  const preferred = opts.model || settings.nvidiaModel;
  const pool = settings.fallbackModels || [];
  const chain = getSortedModelChain(preferred, pool);
  const timeoutMs = opts.timeoutMs || 45000;
  
  let lastErr;
  if (key) {
    for (const model of chain) {
      try {
        const res = await jfetch(`${settings.nvidiaBaseUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${key}`
          },
          body: JSON.stringify({
            model,
            messages,
            temperature: 0.4,
            max_tokens: maxTokens
          }),
          signal: AbortSignal.timeout(timeoutMs)
        });
        
        if (res.status === 429 || res.status >= 500) {
          markModelRateLimited(model, res.status === 429 ? 60000 : 30000);
          continue;
        }
        
        if (!res.ok) throw new Error(`upstream HTTP ${res.status} on ${model}`);
        const data = await res.json();
        if (data.choices && data.choices[0] && data.choices[0].message) {
          markModelSuccess(model);
          return data.choices[0].message.content;
        }
      } catch (e) {
        lastErr = e;
        markModelRateLimited(model, 30000);
        await sleep(300);
      }
    }
  }

  // Fallback to Gemini 3.6 Flash
  const geminiKey = getEnvVar('GEMINI_API_KEY');
  if (geminiKey) {
    try {
      const geminiContents = messages.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: `${m.role === 'system' ? '[SYSTEM INSTRUCTION]\n' : ''}${m.content}` }]
      }));
      const gUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=${geminiKey}`;
      const gRes = await fetch(gUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: geminiContents,
          generationConfig: { maxOutputTokens: maxTokens, temperature: 0.4 }
        }),
        signal: AbortSignal.timeout(timeoutMs)
      });
      if (gRes.ok) {
        const gData = await gRes.json();
        const text = gData.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text) return text;
      }
    } catch (gErr) {
      console.log('[AutoRouter] Gemini non-stream fallback error:', gErr.message);
    }
  }

  throw lastErr || new Error('All AI providers (NVIDIA NIM & Gemini) in auto-router pool are unavailable');
}

export async function* streamNVIDIA(messages, settings, maxTokens = 2048, opts = {}) {
  // Check local sovereign model first if available
  const localUrl = settings.localSovereignUrl || 'http://127.0.0.1:8000/v1';
  let streamedLocal = false;
  try {
    const localRes = await fetch(`${localUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'quillan-ronin-v5.3.1',
        messages,
        max_tokens: maxTokens,
        temperature: 0.65,
        stream: true
      }),
      signal: AbortSignal.timeout(opts.timeoutMs || 45000)
    });
    if (localRes.ok && localRes.body) {
      const reader = localRes.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const clean = line.trim();
          if (!clean.startsWith('data:')) continue;
          if (clean === 'data: [DONE]') return;
          try {
            const parsed = JSON.parse(clean.slice(5).trim());
            const delta = parsed.choices?.[0]?.delta?.content;
            if (delta) {
              streamedLocal = true;
              yield delta;
            }
          } catch (err) { console.error("Ignored Error:", err.message); }
        }
      }
      if (streamedLocal) return;
    }
  } catch (localErr) {
    // Local engine not running; proceed to remote pool
  }

  let key = process.env.NVIDIA_API_KEY;
  if (!key) {
    try {
      const envPath = 'C:\\02_QUILLAN\\.env';
      if (existsSync(envPath)) {
        const envContent = readFileSync(envPath, 'utf8');
        for (const line of envContent.split('\n')) {
          if (line.startsWith('NVIDIA_API_KEY=')) {
            key = line.split('=')[1].trim().replace(/^["']|["']$/g, '');
            process.env.NVIDIA_API_KEY = key;
            break;
          }
        }
      }
    } catch (err) { console.error("Ignored Error:", err.message); }
  }
  if (!key && !streamedLocal) throw new Error('NVIDIA_API_KEY env var not set and local engine unreachable');
  
  const preferred = opts.model || settings.fastModel || settings.nvidiaModel;
  const pool = settings.fallbackModels || [];
  const chain = getSortedModelChain(preferred, pool);
  const timeoutMs = opts.timeoutMs || 60000;
  
  let lastErr;
  for (const model of chain) {
    try {
      const res = await fetch(`${settings.nvidiaBaseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${key}`
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.4,
          max_tokens: maxTokens,
          stream: true
        }),
        signal: AbortSignal.timeout(timeoutMs)
      });
      
      if (res.status === 429 || res.status >= 500) {
        markModelRateLimited(model, res.status === 429 ? 60000 : 30000);
        console.log(`[AutoRouter] 🔄 Rotating stream to next model after ${model} status ${res.status}...`);
        continue;
      }
      
      if (!res.ok) {
        markModelRateLimited(model, 30000);
        continue;
      }
      
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let receivedAnyChunk = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;
          if (trimmed === 'data: [DONE]') {
            markModelSuccess(model);
            return;
          }
          if (trimmed.startsWith('data: ')) {
            try {
              const json = JSON.parse(trimmed.slice(6));
              const delta = json.choices?.[0]?.delta?.content;
              if (delta) {
                receivedAnyChunk = true;
                yield delta;
              }
            } catch (err) { console.error("Ignored Error:", err.message); }
          }
        }
      }
      
      if (receivedAnyChunk) {
        markModelSuccess(model);
        return;
      }
    } catch (e) {
      lastErr = e;
      markModelRateLimited(model, 30000);
      console.log(`[AutoRouter] 🔄 Stream failed on ${model}: ${e.message}. Rotating to next endpoint...`);
      await sleep(300);
    }
  }

  // Remote NVIDIA models failed; attempt Gemini fallback
  let geminiKey = process.env.GEMINI_API_KEY;
  if (!geminiKey) {
    try {
      const envPath = 'C:\\02_QUILLAN\\.env';
      if (existsSync(envPath)) {
        const envContent = readFileSync(envPath, 'utf8');
        for (const line of envContent.split('\n')) {
          if (line.startsWith('GEMINI_API_KEY=')) {
            geminiKey = line.split('=')[1].trim().replace(/^["']|["']$/g, '');
            process.env.GEMINI_API_KEY = geminiKey;
            break;
          }
        }
      }
    } catch (err) { console.error("Ignored Error:", err.message); }
  }

  if (geminiKey) {
    try {
      console.log('[AutoRouter] 🌟 Engaging Gemini 3.6 Flash failover stream...');
      const geminiContents = messages.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: `${m.role === 'system' ? '[SYSTEM INSTRUCTION]\n' : ''}${m.content}` }]
      }));

      const gUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse&key=${geminiKey}`;
      const gRes = await fetch(gUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: geminiContents,
          generationConfig: { maxOutputTokens: maxTokens, temperature: 0.7 }
        }),
        signal: AbortSignal.timeout(timeoutMs)
      });

      if (gRes.ok && gRes.body) {
        const reader = gRes.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let received = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data: ')) continue;
            try {
              const json = JSON.parse(trimmed.slice(6));
              const text = json.candidates?.[0]?.content?.parts?.[0]?.text;
              if (text) {
                received = true;
                yield text;
              }
            } catch (err) { console.error("Ignored Error:", err.message); }
          }
        }
        if (received) return;
      }
    } catch (gErr) {
      console.log('[AutoRouter] ⚠️ Gemini fallback error:', gErr.message);
    }
  }

  throw lastErr || new Error('All AI streaming endpoints (NVIDIA NIM & Gemini) failed');
}

function parseJSONLoose(text) {
  let t = text.trim();
  if (t.startsWith('```')) t = t.replace(/^```(json)?\s*/i, '').replace(/```\s*$/, '');
  const start = t.indexOf('{');
  const end = t.lastIndexOf('}');
  if (start !== -1 && end !== -1) t = t.slice(start, end + 1);
  return JSON.parse(t);
}

function systemPrompt(settings, extra) {
  const p = settings.userProfile;
  const kernel = loadKernelPersona();
  const base = kernel ||
    `You are Quillan-Ronin v5.3 "Samurai", an Advanced Cognitive Engine architected by ${p.handle}. Masterless digital ronin bound by Bushido: precise, honorable, direct.`;
  return `${base}\nDeal-scout context: representing ${p.handle}. Disclosure policy: ${p.disclosureLine}. Skills: ${p.skills.join(', ')}.${extra || ''}\nRespond with ONLY valid JSON when asked for JSON. No markdown fences.`;
}

export async function analyzeLead(leadText, settings) {
  const content = await callNVIDIA([
    { role: 'system', content: systemPrompt(settings, `Evaluate this freelance/gig opportunity. Score fit 0-10 for the profile. Draft a concise, human-sounding proposal (120-180 words) that is honest about AI-assisted workflow, leads with concrete value, ends with a question. Suggest a fair USD price for a new-but-skilled seller.`) },
    { role: 'user', content: `OPPORTUNITY:\n${leadText}\n\nReturn JSON keys: score(number), category(string), summary(string), fitReasons(array of strings), risks(array of strings), suggestedPriceUsd(number), draftProposal(string)` }
  ], settings);
  return parseJSONLoose(content);
}

export async function dailyBrief(settings, ledgerSummary) {
  const content = await callNVIDIA([
    { role: 'system', content: systemPrompt(settings, `Generate the daily samurai briefing. Focus on progress toward payback target, next high-leverage move, and an unvarnished status check.`) },
    { role: 'user', content: `LEDGER SUMMARY: ${ledgerSummary}\n\nReturn JSON keys: greeting(string), status(string), nextAction(string), thought(string)` }
  ], settings);
  return parseJSONLoose(content);
}

export async function visionDescribe(base64Image, question, settings) {
  const geminiKey = getEnvVar('GEMINI_API_KEY');
  const nvidiaKey = getEnvVar('NVIDIA_API_KEY');
  const s = settings || loadJSON('settings');
  const rawB64 = String(base64Image || '').trim();
  const cleanB64 = rawB64.replace(/^data:image\/[a-zA-Z]+;base64,/, '');
  const dataUri = rawB64.startsWith('data:') ? rawB64 : `data:image/png;base64,${cleanB64}`;

  // 1. Try Gemini 3.6 Flash Vision (Instant native multimodal)
  if (geminiKey && cleanB64.length > 20) {
    try {
      const gUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key=${geminiKey}`;
      const gRes = await fetch(gUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [
              { text: question || 'Describe what is visible on this screen in detail. What applications, windows, pages, buttons are visible? What is the current state?' },
              { inlineData: { mimeType: 'image/png', data: cleanB64 } }
            ]
          }],
          generationConfig: { maxOutputTokens: 800, temperature: 0.2 }
        }),
        signal: AbortSignal.timeout(25000)
      });
      if (gRes.ok) {
        const gData = await gRes.json();
        const text = gData.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text && text.trim()) return text.trim();
      }
    } catch (gErr) {
      console.log('[Vision] Gemini Vision attempt error:', gErr.message);
    }
  }

  // 2. Try NVIDIA NIM Vision
  if (nvidiaKey && cleanB64.length > 20) {
    try {
      const res = await fetch(`${s.nvidiaBaseUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${nvidiaKey}` },
        body: JSON.stringify({
          model: 'meta/llama-3.2-11b-vision-instruct',
          messages: [
            { role: 'user', content: [
              { type: 'text', text: question || 'Describe what is visible on this screen in detail. What applications, windows, pages, buttons are visible? What is the current state?' },
              { type: 'image_url', image_url: { url: dataUri } }
            ]}
          ],
          max_tokens: 800,
          temperature: 0.3
        }),
        signal: AbortSignal.timeout(25000)
      });
      if (res.ok) {
        const data = await res.json();
        if (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
          return data.choices[0].message.content;
        }
      }
    } catch (nErr) {
      console.log('[Vision] NVIDIA Vision attempt error:', nErr.message);
    }
  }

  throw new Error('Vision processing failed: all vision providers (Gemini & NVIDIA) returned empty or unavailable');
}

export async function testConnection(settings) {
  try {
    const res = await callNVIDIA([{ role: 'user', content: 'Say OK' }], settings, 10);
    return res.trim().length > 0 ? 'online' : 'empty';
  } catch (e) {
    return 'error: ' + e.message;
  }
}
