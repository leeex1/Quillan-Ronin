const fs = require("fs");
const path = require("path");

// LLM client — streaming with proper cleanup (fixes stream reader mem leak + AbortController leaks)
// Priority: 1) Ollama local  2) NVIDIA NIM  3) fallback mock
// SECURITY: no hardcoded keys — read from env or llm.json only

let cfgCache = null;
function loadCfg(){
  if(cfgCache) return cfgCache;
  const defaults = {
    ollama: { base:"http://localhost:11434/v1", model:"falcon3:1b-instruct-q8_0" },
    nvidia: { base:"https://integrate.api.nvidia.com/v1", key: (process.env.NVIDIA_API_KEY||"").trim(), model:"nvidia/nemotron-3.5-lightning-30b-a3b" },
    openai: { base:"https://api.openai.com/v1", key:(process.env.OPENAI_API_KEY||"").trim(), model:"gpt-4o-mini" }
  };
  try {
    const brainPath = "C:\\02_QUILLAN\\configs\\llm.json";
    if(fs.existsSync(brainPath)){
      const j = JSON.parse(fs.readFileSync(brainPath,"utf-8"));
      if(j.nvidia_key) defaults.nvidia.key = j.nvidia_key;
      if(j.model) defaults.nvidia.model = j.model;
      if(j.ollama_model) defaults.ollama.model = j.ollama_model;
    }
  } catch(e){}
  // SECURITY: removed hardcoded nvapi key fallback — must come from env/file
  cfgCache = defaults;
  return defaults;
}

async function tryFetch(base, key, model, prompt, callbacks){
  const url = base.replace(/\/$/, "") + "/chat/completions";
  const headers = { "Content-Type":"application/json" };
  if(key && key!=="unused") headers["Authorization"] = "Bearer " + key;
  const body = JSON.stringify({
    model,
    messages: [
      { role:"system", content:"You are Quillan, a concise, warm, slightly playful desktop assistant (Clippy-reborn). Keep replies under 90 words unless asked for code. Be helpful." },
      { role:"user", content: prompt }
    ],
    temperature: 0.35,
    max_tokens: 700,
    stream: true
  });

  const ctrl = new AbortController();
  const to = setTimeout(()=> { try{ ctrl.abort(); }catch(e){} }, 45000);
  let full = "";
  let reader = null;
  const MAX_STREAM = 8000; // FIX: bound response size to prevent OOM
  try {
    const res = await fetch(url, { method:"POST", headers, body, signal: ctrl.signal });
    clearTimeout(to);
    if(!res.ok){
      const txt = await res.text().catch(()=>"");
      throw new Error(`HTTP ${res.status} ${txt.slice(0,220)}`);
    }
    if(res.body && res.body.getReader){
      reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      try {
        while(true){
          const { done, value } = await reader.read();
          if(done) break;
          if (full.length > MAX_STREAM) { try{ await reader.cancel(); }catch(e){}; throw new Error("response too large"); }
          buf += decoder.decode(value, { stream:true });
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for(const line of lines){
            const t = line.trim();
            if(!t || t==="data: [DONE]") continue;
            if(!t.startsWith("data:")) continue;
            try {
              const j = JSON.parse(t.slice(5).trim());
              const delta = j.choices && j.choices[0] && (j.choices[0].delta && j.choices[0].delta.content || j.choices[0].text || "");
              if(delta){
                full += delta;
                if (full.length > MAX_STREAM) { try{ await reader.cancel(); }catch(e){}; throw new Error("response too large"); }
                callbacks.onToken(delta, full);
              }
            } catch(e){}
          }
        }
        if(buf.trim().startsWith("data:")){
          try { const j=JSON.parse(buf.trim().slice(5)); const d=j.choices?.[0]?.delta?.content||""; if(d){ full+=d; callbacks.onToken(d,full);} }catch(e){}
        }
      } finally {
        // FIX: always release reader lock to free mem
        try { reader.releaseLock(); } catch(e){}
        try { await reader.cancel().catch(()=>{}); } catch(e){}
      }
      if(!full) throw new Error("empty stream");
      callbacks.onDone(full);
      return full;
    } else {
      const j = await res.json();
      full = j.choices?.[0]?.message?.content || j.choices?.[0]?.text || "";
      if (full.length > MAX_STREAM) full = full.slice(0, MAX_STREAM);
      callbacks.onToken(full, full);
      callbacks.onDone(full);
      return full;
    }
  } catch(e){
    clearTimeout(to);
    // FIX: ensure reader is cancelled on error path
    if (reader) { try{ await reader.cancel(); }catch(_e){} try{ reader.releaseLock(); }catch(_e){} }
    throw e;
  }
}

let _activeAbort = null; // FIX: allow cancelling previous send if user spams
async function send(prompt, { onToken=()=>{}, onDone=()=>{}, onError=()=>{} }={}){
  // cancel previous pending mock timers if any
  if (_activeAbort) { try{ _activeAbort.abort(); }catch(e){} _activeAbort=null; }
  const cfg = loadCfg();
  const safe = String(prompt).slice(0, 900);
  let lastErr = "";
  try {
    await tryFetch(cfg.ollama.base, "unused", cfg.ollama.model, safe, { onToken, onDone, onError });
    return;
  } catch(e){ lastErr = "ollama: "+e.message; }
  if(cfg.nvidia.key){
    try {
      await tryFetch(cfg.nvidia.base, cfg.nvidia.key, cfg.nvidia.model, safe, { onToken, onDone, onError });
      return;
    } catch(e){ lastErr += " | nvidia: "+e.message; }
  }
  if(cfg.openai.key){
    try {
      await tryFetch(cfg.openai.base, cfg.openai.key, cfg.openai.model, safe, { onToken, onDone, onError });
      return;
    } catch(e){ lastErr += " | openai: "+e.message; }
  }
  // Fallback mock (so UI never appears dead)
  const mocks = [
    "Hey — I'm here! My LLM is waking up. Ask me about code, files, or just say hi.",
    `You said: "${safe.slice(0,60)}" — cool! Hook up Ollama on :11434 or set NVIDIA_API_KEY for full brain power.`,
    "I'm Quillan, your Clippy-reborn. I can help with code, chat, or just vibe on your desktop."
  ];
  const mock = mocks[Math.floor(Math.random()*mocks.length)];
  // FIX: cancellable mock streaming — store controller to abort on next send
  const ctrl = new AbortController();
  _activeAbort = ctrl;
  let cur="";
  for(const ch of mock){
    if (ctrl.signal.aborted) break;
    cur+=ch; onToken(ch, cur);
    // cancellable delay
    await new Promise(r=>{
      const t=setTimeout(r, 12);
      ctrl.signal.addEventListener("abort", ()=>{ clearTimeout(t); r(); }, {once:true});
    });
  }
  if (!ctrl.signal.aborted) onDone(cur);
  _activeAbort=null;
  if(lastErr) console.log("[llm] fallback used —", lastErr);
}

// Legacy compat
let legacyHandler = null;
function setHandler(fn){ legacyHandler = fn; }
function resetCfg(){ cfgCache=null; }
module.exports = { send, setHandler, _tryFetch: tryFetch, resetCfg };
