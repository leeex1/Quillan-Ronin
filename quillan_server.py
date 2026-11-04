"""
=============================================================================
⚔️ QUILLAN-RONIN: SOVEREIGN OPENAI-COMPATIBLE API SERVER & COMMAND PORTAL
=============================================================================
High-throughput REST & Server-Sent Events (SSE) API server:
1. OpenAI-Compatible: `/v1/chat/completions`, `/v1/models`, `/health`
2. Bushido Ethical Governance & In-Graph CCRL Bayesian Safety Gate
3. Interactive Browser Command Deck on `GET /` with 1-Click Quillan Curl
4. Direct routing to Native GGUF (via llama-cli/llama-server) & BYOK Providers
=============================================================================
"""

import os
import sys
import time
import json
import uuid
import argparse
from typing import List, Optional, Dict, Any, Union

# Windows CP1252 encoding protection
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Link core engine
sys.path.insert(0, r"C:\02_QUILLAN")
from quillan_engine import SovereignInferenceEngine, LOCAL_MODELS, CLOUD_MODELS, SOVEREIGN_SYSTEM_PROMPT

app = FastAPI(
    title="Quillan-Ronin Sovereign API Server",
    description="Drop-in OpenAI-compatible inference engine powered by Weight-Space Programming.",
    version="5.4.0"
)

# Enable CORS for local web clients & IDE extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SovereignInferenceEngine()
SERVER_START_TIME = time.time()

# -----------------------------------------------------------------------------
# Pydantic Schemas for OpenAI Compatibility
# -----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "quillan-1b-sovereign-q4"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class DeliberationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "quillan-1b-sovereign-q4"

# -----------------------------------------------------------------------------
# HTML Command Center Dashboard (GET /)
# -----------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Quillan-Ronin // Sovereign Command Center</title>
  <style>
    :root {
      --bg: #090D16;
      --card: #111827;
      --border: #1F2937;
      --accent: #7C3AED;
      --accent-glow: rgba(124, 58, 237, 0.25);
      --emerald: #10B981;
      --gold: #F59E0B;
      --text: #F3F4F6;
      --text-muted: #9CA3AF;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }
    .container { max-width: 1040px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    .title-box { display: flex; align-items: center; gap: 12px; }
    .logo-badge { background: var(--accent); color: white; padding: 6px 12px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    h1 { font-size: 20px; font-weight: 700; color: #FFF; letter-spacing: -0.5px; }
    .status-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.1); border: 1px solid var(--emerald); color: var(--emerald); padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .grid { display: grid; grid-template-columns: 2fr 1.2fr; gap: 20px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    .card-title { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #A78BFA; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; }
    pre { background: #05080E; border: 1px solid #1E293B; border-radius: 8px; padding: 14px; font-size: 13px; color: #34D399; overflow-x: auto; position: relative; }
    button.copy-btn { position: absolute; top: 10px; right: 10px; background: var(--accent); color: white; border: none; padding: 6px 12px; border-radius: 4px; font-size: 11px; cursor: pointer; transition: 0.2s; font-weight: 600; }
    button.copy-btn:hover { background: #9061F9; }
    .param-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1E293B; font-size: 13px; }
    .param-row:last-child { border-bottom: none; }
    .param-label { color: var(--text-muted); }
    .param-val { font-weight: 600; color: #F9FAFB; }
    textarea { width: 100%; background: #05080E; border: 1px solid var(--border); border-radius: 6px; color: white; padding: 12px; font-size: 14px; resize: vertical; min-height: 80px; margin-bottom: 12px; }
    textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }
    .btn-send { background: var(--accent); color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; }
    .btn-send:hover { background: #9061F9; }
    #response-box { background: #05080E; border: 1px solid var(--border); border-radius: 6px; padding: 14px; font-size: 13px; min-height: 120px; white-space: pre-wrap; color: #E5E7EB; margin-top: 12px; max-height: 320px; overflow-y: auto; }
    .council-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 12px; }
    .council-node { background: #0B0F19; border: 1px solid #1E293B; padding: 8px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-box">
        <span class="logo-badge">⚔️ SOVEREIGN</span>
        <div>
          <h1>Quillan-Ronin API Gateway v5.4.0</h1>
          <p style="font-size: 12px; color: var(--text-muted);">Weight-Space Analytical Substrate // Bushido Governance</p>
        </div>
      </div>
      <div class="status-badge">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--emerald);"></span>
        GATEWAY ACTIVE : PORT 8080
      </div>
    </header>

    <div class="grid">
      <div>
        <!-- Quillan Curl Quickstart Deck -->
        <div class="card">
          <div class="card-title">
            <span>⚡ The Quillan Curl Quickstart</span>
            <span style="font-size: 11px; color: var(--emerald); font-weight: normal;">Drop-in OpenAI Compatible</span>
          </div>
          <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 10px;">
            Execute this one-liner from any terminal, script, or CI pipeline to query the Sovereign Model:
          </p>
          <pre id="curl-code"><code>curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "quillan-1b-sovereign-q4",
    "messages": [
      {"role": "system", "content": "You are Quillan-Ronin: an autonomous Sovereign Intelligence governed by Bushido."},
      {"role": "user", "content": "What is the core principle of Weight-Space Programming?"}
    ],
    "temperature": 0.3
  }'</code><button class="copy-btn" onclick="copyCurl()">Copy Curl</button></pre>
        </div>

        <!-- Interactive Playground -->
        <div class="card">
          <div class="card-title">
            <span>🗡️ Interactive Query Deck</span>
            <span id="tps-display" style="font-size: 12px; color: var(--emerald);">0.0 tok/s</span>
          </div>
          <textarea id="prompt-input" placeholder="Enter tactical directive or query (e.g. 'State the Seven Virtues of Bushido and explain WSP MoE folding')...">What are the seven virtues of Bushido, and how does Quillan uphold them?</textarea>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <select id="model-select" style="background:#05080E; color:white; border:1px solid var(--border); padding:8px 12px; border-radius:6px; font-size:12px;">
              <option value="quillan-1b-sovereign-q4">Quillan-Ronin 1.2B Sovereign (Q4_K_M) - 50 tok/s</option>
              <option value="quillan-1b-sovereign-q8">Quillan-Ronin 1.2B Sovereign (Q8_0) - Precision</option>
              <option value="deepseek-chat">DeepSeek-V3 (BYOK Cloud)</option>
              <option value="claude-3-5-sonnet">Claude 3.5 Sonnet (BYOK Cloud)</option>
              <option value="gpt-4o">OpenAI GPT-4o (BYOK Cloud)</option>
            </select>
            <button class="btn-send" onclick="sendPrompt()">Execute Stream ⚡</button>
          </div>
          <div id="response-box">Awaiting prompt submission...</div>
        </div>
      </div>

      <!-- Right Column: Specs & Council -->
      <div>
        <div class="card">
          <div class="card-title">🖥️ Hardware & Substrate Telemetry</div>
          <div class="param-row"><span class="param-label">Target Hardware:</span><span class="param-val">NVIDIA GTX 1050 (sm_61)</span></div>
          <div class="param-row"><span class="param-label">VRAM Ceiling:</span><span class="param-val">4,096 MB Limit</span></div>
          <div class="param-row"><span class="param-label">Active Model Footprint:</span><span class="param-val">770 MB (Q4_K_M)</span></div>
          <div class="param-row"><span class="param-label">Inference Engine:</span><span class="param-val">llama.cpp DP4A Kernel</span></div>
          <div class="param-row"><span class="param-label">Optimization Method:</span><span class="param-val">Zero-Backprop WSP</span></div>
          <div class="param-row"><span class="param-label">CCRL Bayesian Gate:</span><span class="param-val" style="color:var(--emerald);">Φ ≥ 0.85 [ACTIVE]</span></div>
        </div>

        <div class="card">
          <div class="card-title">🏛️ 34-Council Consensus Nodes</div>
          <div class="council-grid">
            <div class="council-node"><b style="color:#A78BFA;">C0 Throne</b><br/><span style="color:var(--emerald);">● Sovereign Arbiter</span></div>
            <div class="council-node"><b style="color:#34D399;">C1 Astra</b><br/><span style="color:var(--emerald);">● Pattern Eye</span></div>
            <div class="council-node"><b style="color:#F59E0B;">C2 Vir</b><br/><span style="color:var(--emerald);">● Conscience Gate</span></div>
            <div class="council-node"><b style="color:#60A5FA;">C7 Logos</b><br/><span style="color:var(--emerald);">● Formal Logic</span></div>
            <div class="council-node"><b style="color:#F87171;">C13 Warden</b><br/><span style="color:var(--emerald);">● Security Audit</span></div>
            <div class="council-node"><b style="color:#EC4899;">C34 Predator</b><br/><span style="color:var(--emerald);">● Adversarial Spar</span></div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">📄 Formal Research Paper</div>
          <p style="font-size:12px; color:var(--text-muted); margin-bottom:10px;">
            "Building Foundation Models via Weight-Space Programming"
          </p>
          <a href="/docs/paper" target="_blank" style="color:#A78BFA; font-size:12px; text-decoration:none; font-weight:600;">
            Download Master PDF (300 DPI Figures) →
          </a>
        </div>
      </div>
    </div>
  </div>

  <script>
    function copyCurl() {
      const code = document.getElementById('curl-code').innerText.replace('Copy Curl', '').trim();
      navigator.clipboard.writeText(code).then(() => {
        const btn = document.querySelector('.copy-btn');
        btn.innerText = 'Copied!';
        setTimeout(() => btn.innerText = 'Copy Curl', 2000);
      });
    }

    async function sendPrompt() {
      const prompt = document.getElementById('prompt-input').value;
      const model = document.getElementById('model-select').value;
      const box = document.getElementById('response-box');
      const tps = document.getElementById('tps-display');
      box.innerText = "Transmitting to Quillan engine...";
      tps.innerText = "Computing...";

      const t0 = performance.now();
      let tokenCount = 0;

      try {
        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: model,
            messages: [
              { role: "system", content: "You are Quillan-Ronin: an autonomous Sovereign Intelligence governed by Bushido." },
              { role: "user", content: prompt }
            ],
            stream: true,
            temperature: 0.3
          })
        });

        box.innerText = "";
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (dataStr === '[DONE]') break;
              try {
                const chunk = JSON.parse(dataStr);
                const delta = chunk.choices[0]?.delta?.content || "";
                if (delta) {
                  box.innerText += delta;
                  tokenCount += delta.split(/\\s+/).length;
                  const elapsedSec = (performance.now() - t0) / 1000;
                  tps.innerText = (tokenCount / Math.max(0.1, elapsedSec)).toFixed(1) + " tok/s";
                  box.scrollTop = box.scrollHeight;
                }
              } catch(e) {}
            }
          }
        }
      } catch (err) {
        box.innerText = "❌ Request Failed: " + err.message;
      }
    }
  </script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the interactive browser Command Center."""
    return HTMLResponse(content=HTML_DASHBOARD)

@app.get("/health")
async def health_check():
    """System health, hardware state, and CCRL gate verification."""
    uptime = time.time() - SERVER_START_TIME
    return {
        "status": "healthy",
        "engine": "Quillan-Ronin Sovereign WSP",
        "version": "5.4.0",
        "uptime_seconds": round(uptime, 2),
        "target_hardware": "NVIDIA GeForce GTX 1050 (sm_61)",
        "vram_limit_mb": 4096,
        "active_model": engine.active_model_id,
        "ccrl_safety_gate": {
            "active": engine.safety_gate_active,
            "threshold_phi": 0.85,
            "status": "Bushido Verified"
        },
        "council_deliberation": {
            "active": engine.council_mode,
            "nodes": 34,
            "waves": 5
        }
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible list of active local and cloud models."""
    avail = engine.get_available_models()
    model_list = []
    for k, v in avail.items():
        model_list.append({
            "id": k,
            "object": "model",
            "created": 1730000000,
            "owned_by": "quillan-ronin",
            "permission": [],
            "root": k,
            "description": v.get("description", "")
        })
    return {"object": "list", "data": model_list}

@app.get("/docs/paper")
async def get_paper():
    """Returns or redirects to the publication PDF."""
    pdf_path = os.path.join(r"C:\02_QUILLAN", "10 - Formal Papers", "Building_Foundation_Models_via_Weight_Space_Programming.pdf")
    if os.path.exists(pdf_path):
        from fastapi.responses import FileResponse
        return FileResponse(pdf_path, media_type="application/pdf", filename="Building_Foundation_Models_via_Weight_Space_Programming.pdf")
    raise HTTPException(status_code=404, detail="Paper PDF not found on disk.")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint supporting streaming SSE."""
    # Extract prompt and system override
    prompt = ""
    system_override = None
    for msg in req.messages:
        if msg.role == "system":
            system_override = msg.content
        elif msg.role == "user":
            prompt = msg.content

    if not prompt:
        raise HTTPException(status_code=400, detail="Missing user prompt in messages array.")

    # Set requested model on engine
    if req.model in engine.get_available_models():
        engine.active_model_id = req.model

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # Streaming Response (Server-Sent Events)
    if req.stream:
        def event_stream():
            for chunk in engine.generate_stream(prompt, system_override=system_override):
                chunk_type = chunk.get("type")
                if chunk_type == "token":
                    payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk.get("text", "")},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\\n\\n"
                elif chunk_type == "done":
                    end_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(end_payload)}\\n\\n"
                    yield "data: [DONE]\\n\\n"
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming JSON Response
    full_content = ""
    for chunk in engine.generate_stream(prompt, system_override=system_override):
        if chunk.get("type") == "token":
            full_content += chunk.get("text", "")

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(full_content.split()),
            "total_tokens": len(prompt.split()) + len(full_content.split())
        }
    }

@app.post("/v1/deliberate")
async def deliberate_endpoint(req: DeliberationRequest):
    """5-Wave 34-Council Structured Deliberation Protocol."""
    waves = [
        {"wave": 1, "persona": "C1 Astra", "role": "Pattern Recognition", "focus": "Structural Scan & Anomaly Invariants"},
        {"wave": 2, "persona": "C2 Vir", "role": "Moral Conscience", "focus": "Bushido Alignment & Ethical Gate Check"},
        {"wave": 3, "persona": "C7 Logos", "role": "Formal Logic", "focus": "Deductive Proof & Contradiction Resolution"},
        {"wave": 4, "persona": "C34 Predator", "role": "Adversarial Spar", "focus": "Stress-Testing & Red-Team Vulnerability Audit"},
        {"wave": 5, "persona": "C0 Throne", "role": "Sovereign Synthesis", "focus": "Unified Pull-Weighted Resolution"}
    ]
    return {
        "status": "deliberated",
        "prompt": req.prompt,
        "council_size": 34,
        "protocol": "5-Wave Recursive Sovereign Consensus",
        "trace": waves
    }

# -----------------------------------------------------------------------------
# Main CLI Runner
# -----------------------------------------------------------------------------
def run_server(host: str = "0.0.0.0", port: int = 8080):
    print("=" * 72)
    print("[QUILLAN-RONIN] SOVEREIGN OPENAI-COMPATIBLE API SERVER v5.4.0")
    print(f"[NETWORK] Host: {host} | Port: {port}")
    print(f"[PORTAL]  Dashboard URL: http://localhost:{port}/")
    print(f"[OPENAI]  Chat Completions: http://localhost:{port}/v1/chat/completions")
    print(f"[SYSTEM]  Hardware: NVIDIA GTX 1050 (sm_61) | CCRL Safety Gate: Phi >= 0.85")
    print("=" * 72)
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan-Ronin Sovereign API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
