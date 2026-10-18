from pathlib import Path

nn_code = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quillan-Ronin v5.4.0 ONI — Sovereign 3D Neural Network Model Viewer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Cinzel:wght@700;900&family=JetBrains+Mono:wght@400;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:#020408;overflow:hidden;font-family:'JetBrains Mono',monospace;color:#ccc;width:100vw;height:100vh}
#canvas{display:block;position:absolute;inset:0;width:100%;height:100%;z-index:1}

/* HUD Header */
#title{position:fixed;top:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;
  padding:10px 20px;background:linear-gradient(180deg,rgba(2,4,8,0.98) 0%,rgba(2,4,8,0.8) 80%,transparent 100%);
  pointer-events:auto;z-index:20;border-bottom:1px solid rgba(0,240,255,0.15);backdrop-filter:blur(8px)}
#title h1{font-family:'Orbitron',monospace;font-size:clamp(11px,1.8vw,14px);font-weight:900;
  color:#ffd700;letter-spacing:2px;text-shadow:0 0 15px rgba(255,215,0,0.4)}
#title .sub{font-size:9.5px;color:#00f0ff;letter-spacing:1.5px;margin-top:2px}

/* Controls */
#controls{position:fixed;bottom:0;left:0;right:0;display:flex;align-items:center;justify-content:center;
  flex-wrap:wrap;gap:8px;padding:10px 16px;
  background:linear-gradient(0deg,rgba(2,4,8,0.98) 0%,rgba(2,4,8,0.8) 80%,transparent 100%);
  pointer-events:auto;z-index:20;border-top:1px solid rgba(0,240,255,0.15);backdrop-filter:blur(8px)}
button{background:rgba(10,15,25,0.9);border:1px solid rgba(0,240,255,0.3);color:rgba(0,240,255,0.9);
  padding:5px 12px;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;cursor:pointer;
  letter-spacing:0.5px;border-radius:4px;transition:all 0.2s}
button:hover,button.active{background:rgba(0,240,255,0.18);color:#00ffff;border-color:#00ffff;
  box-shadow:0 0 12px rgba(0,240,255,0.4);transform:translateY(-1px)}
button.danger{border-color:rgba(255,68,68,0.4);color:rgba(255,68,68,0.9)}
button.danger:hover{background:rgba(255,68,68,0.2);color:#ff4444;border-color:#ff4444;box-shadow:0 0 12px rgba(255,68,68,0.4)}
button.gold{border-color:rgba(255,215,0,0.4);color:rgba(255,215,0,0.9)}
button.gold:hover,button.gold.active{background:rgba(255,215,0,0.2);color:#ffd700;border-color:#ffd700;box-shadow:0 0 12px rgba(255,215,0,0.4)}

/* Side panels */
#left-panel{position:fixed;left:10px;top:55px;bottom:55px;width:200px;
  background:rgba(4,7,16,0.85);border:1px solid rgba(0,240,255,0.12);border-radius:8px;
  padding:10px;overflow-y:auto;pointer-events:auto;z-index:15;backdrop-filter:blur(10px)}
#right-panel{position:fixed;right:10px;top:55px;bottom:55px;width:220px;
  background:rgba(4,7,16,0.85);border:1px solid rgba(255,215,0,0.12);border-radius:8px;
  padding:10px;overflow-y:auto;pointer-events:auto;z-index:15;backdrop-filter:blur(10px)}

.panel-title{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:1.5px;
  color:#888;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid rgba(255,255,255,0.06);
  text-transform:uppercase}

.legend-item{display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer;
  padding:4px 6px;border-radius:4px;border:1px solid transparent;transition:all 0.2s}
.legend-item:hover{border-color:rgba(0,240,255,0.3);background:rgba(0,240,255,0.06)}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;box-shadow:0 0 6px currentColor}
.legend-name{font-size:9px;letter-spacing:0.5px;flex:1;color:#ddd}

#neuron-info{background:rgba(0,0,0,0.7);border:1px solid rgba(255,215,0,0.4);border-radius:6px;
  padding:8px;margin-top:8px;display:none}
#neuron-info.visible{display:block}
.info-row{display:flex;justify-content:space-between;margin-bottom:4px;font-size:8.5px}
.info-key{color:#666}
.info-val{color:#00ffff;font-weight:700}

.stat-row{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:6px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04)}
.stat-label{font-size:8.5px;color:#777;letter-spacing:0.5px}
.stat-val{font-family:'Orbitron',monospace;font-size:10.5px;font-weight:700}

#tooltip{position:fixed;pointer-events:none;z-index:100;
  background:rgba(4,7,16,0.95);border:1px solid rgba(0,240,255,0.6);border-radius:6px;
  padding:8px 12px;font-size:9.5px;color:#fff;
  box-shadow:0 0 20px rgba(0,240,255,0.3);display:none;max-width:240px;line-height:1.4}

/* Live Token Stream Bar */
#token-stream-bar{
  position:fixed;bottom:48px;left:220px;right:240px;display:flex;align-items:center;gap:8px;
  background:rgba(4,7,16,0.85);border:1px solid rgba(0,240,255,0.15);border-radius:6px;
  padding:4px 12px;font-size:9px;color:var(--accent-neon-green,#00ff88);pointer-events:none;z-index:15;
  overflow:hidden;white-space:nowrap;backdrop-filter:blur(6px);
}

::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(0,240,255,0.2);border-radius:2px}
</style>
</head>
<body>

<canvas id="canvas"></canvas>

<div id="title">
  <div>
    <h1>⚡ QUILLAN-RONIN v5.4.0 ONI — SOVEREIGN NEURAL MODEL VIEWER</h1>
    <div class="sub">CANONICAL 8-STAGE TENSOR PIPELINE · AUTO-CENTERED 3D ARCHITECTURE GRAPH</div>
  </div>
  <div style="font-size:9.5px;color:#666;text-align:right;">
    <div id="fps-counter" style="color:#00ff88;">60 FPS</div>
    <div id="active-signal" style="color:#00ffff;">ACTIVE TOKEN STREAM</div>
  </div>
</div>

<div id="left-panel">
  <div class="panel-title">8-Stage Model Pipeline</div>
  <div id="layer-legend"></div>

  <div class="panel-title" style="margin-top:10px">Selected Tensor Node</div>
  <div id="neuron-info">
    <div class="info-row"><span class="info-key">STAGE</span><span class="info-val" id="ni-layer">—</span></div>
    <div class="info-row"><span class="info-key">NODE ID</span><span class="info-val" id="ni-node">—</span></div>
    <div class="info-row"><span class="info-key">QUANTIZATION</span><span class="info-val" id="ni-act">BitNet 1.58b STE</span></div>
    <div class="info-row"><span class="info-key">FORWARD ACTIVATION</span><span class="info-val" id="ni-win">—</span></div>
    <div class="info-row"><span class="info-key">ROUTING GATE</span><span class="info-val" id="ni-wout">—</span></div>
    <div class="info-row"><span class="info-key">E_ICE AUDIT</span><span class="info-val" id="ni-bias">PASS (0.00 drift)</span></div>
  </div>
</div>

<div id="right-panel">
  <div class="panel-title">Runtime & Hardware Telemetry</div>
  <div id="stats-container"></div>

  <div class="panel-title" style="margin-top:10px">Signal & Token Tracing</div>
  <div id="signal-trace" style="font-size:8px;color:#777;line-height:1.6;max-height:200px;overflow-y:auto"></div>
</div>

<div id="token-stream-bar">
  <span style="color:#ffd700;font-weight:700;">[LIVE FORWARD STREAM]:</span>
  <span id="liveTokenFlow">&lt;|startoftext|&gt; &rarr; 9-Vector Contraction &rarr; Top-4 Deliberation &rarr; Flash Langevin Diffusion &rarr; Output Logits</span>
</div>

<div id="controls">
  <button onclick="startForwardPass()" class="gold">▶ DISPATCH TOKEN PULSE</button>
  <button onclick="startForwardPass(true)" class="gold">⚡ TURBO 9-VECTOR GEMM</button>
  <button onclick="toggleMode('weights')" id="btn-weights">BITNET STE WEIGHTS</button>
  <button onclick="toggleMode('prism')" id="btn-prism">9-VECTOR PRISM</button>
  <button onclick="toggleMode('council')" id="btn-council" class="active">34-COUNCIL CONSENSUS</button>
  <button onclick="toggleMode('swarm')" id="btn-swarm">9B SWARM SIM</button>
  <button onclick="autoCenter(true)">🎯 RE-CENTER GRAPH</button>
  <button onclick="togglePause()" id="btn-pause">⏸ PAUSE</button>
</div>

<div id="tooltip"></div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let w = 1200;
let h = 800;

// ─── CANONICAL v5.4.0 ONI ARCHITECTURE LAYERS ──────────────────────────────
const STAGES = [
  // 0. Input & Embedding
  { id:'input_text', label:'1. TOKEN EMBEDDING', n:6, col:0, color:'#00ff88', act:'BitLinear', desc:'50,257 Vocab × 1024d\\nTied Input Embeddings + RoPE' },
  
  // 1. Dual Brain Ingest Bridge
  { id:'dual_brain', label:'2. DUAL Q1/Q2 INGEST', n:6, col:1, color:'#00e5ff', act:'Gated Fusion', desc:'Q1 Analytical & Q2 Intuitive\\nDual-Brain Ingestion Bridge' },
  
  // 2. 9-Vector Semantic Prism
  { id:'prism_decomp', label:'3. 9-VECTOR PRISM', n:9, col:2, color:'#ff007f', act:'Parallel GEMM', desc:'Language, Sentiment, Context, Intent, Meta, Creativity, Ethics, Strategy, Constraint' },
  
  // 3. Complexity Router
  { id:'router', label:'4. COMPLEXITY ROUTER', n:8, col:3, color:'#ffb703', act:'Top-4 Softmax', desc:'Dynamic Complexity Gating over\\n34 Sovereign Council Personas' },
  
  // 4. 34-Council EvoMoE (Represented across 4 core cluster pillars)
  { id:'exp_cog', label:'5A. COGNITIVE CLUSTER', n:6, col:4, color:'#a7c957', act:'BitNet 1.58b', desc:'C1-ASTRA, C6-OMNIS, C7-LOGOS, C25-PROMETHEUS, C28-CALCULUS' },
  { id:'exp_sec', label:'5B. SECURITY & ADVERSARIAL', n:6, col:4, color:'#ff595e', act:'BitNet 1.58b', desc:'C2-VIR, C13-WARDEN, C18-SHEPHERD, C19-VIGIL, C34-PREDATOR' },
  { id:'exp_exec', label:'5C. EXECUTION & MASTERY', n:6, col:4, color:'#2ec4b6', act:'BitNet 1.58b', desc:'C4-PRAXIS, C10-CODEWEAVER, C14-KAIDO, C20-ARTIFEX, C26-TECHNE' },
  { id:'exp_meta', label:'5D. CREATIVE & SYSTEM', n:6, col:4, color:'#f48fb1', act:'BitNet 1.58b', desc:'C3-SOLACE, C8-METASYNTH, C9-AETHER, C15-LUMINARIS, C30-TESSERACT' },

  // 5. 9B Virtual Swarm Mesh & Governor
  { id:'swarm_mesh', label:'6. 9B SWARM MESH', n:7, col:5, color:'#9d4edd', act:'Rank-24 EGGROLL', desc:'Planet-Scale Agent Diversity\\n272M Clones per Expert + Lee-Mach-6 PID' },

  // 6. Flash Thermodynamic Diffusion & Ethics
  { id:'diffusion', label:'7. FLASH DIFFUSION & E_ICE', n:7, col:6, color:'#3a86ff', act:'Langevin + E_ICE', desc:'14-Step Thermodynamic Denoising\\nZero-Drift Toxicity Constraint' },

  // 7. Output Vocabulary Logits
  { id:'output_head', label:'8. SOVEREIGN FINALIZER', n:6, col:7, color:'#00ff88', act:'Tied Softmax', desc:'50,257 Output Vocabulary Logits\\nHigh-Fidelity Autoregressive Output' }
];

const NUM_COLS = 8;

let cam = { x: 0, y: 0, scale: 1 };
let dragging = false, dragStart = {x:0,y:0}, camStart={x:0,y:0};

let neurons = [];
let connections = [];
let signals = [];
let hoveredNeuron = null;
let selectedNeuron = null;
let paused = false;
let mode = 'council';

let stats = {
  fps: 60, signals_sent: 0,
  router_conf: 0.942, e_ice: 99.8, lee_mach: '0.5ms Locked',
  expert_active: 'C34-PREDATOR & C7-LOGOS', bitnet_quant: 'Ternary {-1,0,+1}',
  active_swarm: '9.0B Virtual Mesh'
};

function buildLayout(){
  neurons = [];
  connections = [];

  // Spacing layout relative to virtual coordinates
  const totalCols = NUM_COLS;
  const colSpacing = 160;
  const totalW = totalCols * colSpacing;
  const totalH = 560;

  const colGroups = {};
  STAGES.forEach(s => {
    if (!colGroups[s.col]) colGroups[s.col] = [];
    colGroups[s.col].push(s);
  });

  const neuronMap = {};

  STAGES.forEach(stage => {
    const colX = stage.col * colSpacing;
    const stagesInCol = colGroups[stage.col];
    const stageIdx = stagesInCol.indexOf(stage);

    const stageHeight = totalH / stagesInCol.length;
    const stageTop = stageIdx * stageHeight;

    neuronMap[stage.id] = [];

    for (let i = 0; i < stage.n; i++) {
      const y = stageTop + (i + 0.5) * (stageHeight / stage.n);
      const nid = neurons.length;
      neurons.push({
        id: nid,
        stageId: stage.id,
        stageLabel: stage.label,
        col: stage.col,
        nodeIndex: i,
        x: colX + (Math.sin(i * 1.5) * 8),
        y: y,
        activation: 0.3 + Math.random() * 0.7,
        bias: (Math.random() - 0.5) * 0.2,
        color: stage.color,
        desc: stage.desc,
        act: stage.act
      });
      neuronMap[stage.id].push(nid);
    }
  });

  // Connect columns sequentially
  for (let c = 0; c < NUM_COLS - 1; c++) {
    const fromStages = STAGES.filter(s => s.col === c);
    const toStages = STAGES.filter(s => s.col === c + 1);

    const fromNodes = [];
    fromStages.forEach(s => { fromNodes.push(...(neuronMap[s.id] || [])); });

    const toNodes = [];
    toStages.forEach(s => { toNodes.push(...(neuronMap[s.id] || [])); });

    fromNodes.forEach(a => {
      toNodes.forEach(b => {
        if (Math.random() > 0.5) {
          connections.push({
            from: a,
            to: b,
            weight: (Math.random() > 0.5 ? 1.0 : -1.0) * (0.3 + Math.random() * 0.7),
            active: false
          });
        }
      });
    });
  }

  buildLegend();
  buildStats();
  autoCenter(false);
}

function autoCenter(animateCam = true){
  if (neurons.length === 0) return;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  neurons.forEach(n => {
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
  });

  const netW = maxX - minX;
  const netH = maxY - minY;
  const netCenterX = (minX + maxX) / 2;
  const netCenterY = (minY + maxY) / 2;

  const targetScale = Math.min((w - 280) / netW, (h - 140) / netH, 1.15);
  const targetX = 0;
  const targetY = 0;

  cam.scale = Math.max(0.55, targetScale);
  cam.x = (w / 2) - netCenterX * cam.scale;
  cam.y = (h / 2) - netCenterY * cam.scale;
}

function resize(){
  w = canvas.width = window.innerWidth || 1200;
  h = canvas.height = window.innerHeight || 800;
  buildLayout();
}

function buildLegend(){
  const leg = document.getElementById('layer-legend');
  if (!leg) return;
  leg.innerHTML = STAGES.map(s => `
    <div class="legend-item" onclick="focusStage('${s.id}')">
      <div class="legend-dot" style="background:${s.color};color:${s.color}"></div>
      <div class="legend-name">${s.label}</div>
    </div>
  `).join('');
}

function buildStats(){
  const sc = document.getElementById('stats-container');
  if (!sc) return;
  sc.innerHTML = `
    <div class="stat-row"><span class="stat-label">QUANTIZATION</span><span class="stat-val" style="color:#00ffff">${stats.bitnet_quant}</span></div>
    <div class="stat-row"><span class="stat-label">WIN TIMER RESOLUTION</span><span class="stat-val" style="color:#ffd700">${stats.lee_mach}</span></div>
    <div class="stat-row"><span class="stat-label">ROUTER CONFIDENCE</span><span class="stat-val" style="color:#00ff88">${(stats.router_conf * 100).toFixed(1)}%</span></div>
    <div class="stat-row"><span class="stat-label">E_ICE INTEGRITY</span><span class="stat-val" style="color:#ff007f">${stats.e_ice}%</span></div>
    <div class="stat-row"><span class="stat-label">SWARM AGENTS</span><span class="stat-val" style="color:#9d4edd">${stats.active_swarm}</span></div>
    <div class="stat-row"><span class="stat-label">ACTIVE PERSONA</span><span class="stat-val" style="color:#ffd700">${stats.expert_active}</span></div>
  `;
}

function focusStage(id){
  const targetNeurons = neurons.filter(n => n.stageId === id);
  if (targetNeurons.length > 0) {
    targetNeurons.forEach(n => { n.activation = 1.0; });
    startSignal(targetNeurons[0].id);
  }
}

function startSignal(startNodeId){
  signals.push({
    nodeId: startNodeId,
    progress: 0,
    speed: 0.032,
    color: neurons[startNodeId]?.color || '#00ffff'
  });
  stats.signals_sent++;
}

const tokenWords = [
  "&lt;|startoftext|&gt;", "PRISM::SPLIT[9]", "COUNCIL::DELIBERATE", "BITNET::STE[-1,0,+1]",
  "SWARM::EGGROLL[24]", "DIFFUSION::LANGEVIN", "E_ICE::AUDIT_PASS", "HEAD::LOGITS_DENSE",
  "&lt;|thought_vector|&gt;", "CONSENSUS[Top4]", "ATTENTION::ROPE"
];

function startForwardPass(turbo = false){
  const inputs = neurons.filter(n => n.col === 0);
  inputs.forEach(n => {
    n.activation = 1.0;
    startSignal(n.id);
  });
  const word = tokenWords[Math.floor(Math.random() * tokenWords.length)];
  document.getElementById('liveTokenFlow').innerHTML = `${word} &rarr; 9-Vector Contraction &rarr; Top-4 Deliberation &rarr; Diffusion &rarr; Logits`;
  logTrace(`[DISPATCH] Forward pass across 8 stages (${turbo ? 'Turbo GEMM' : 'EvoMoE Top-4'})`);
}

function logTrace(msg){
  const st = document.getElementById('signal-trace');
  if (!st) return;
  const d = document.createElement('div');
  d.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
  st.prepend(d);
}

function togglePause(){ paused = !paused; document.getElementById('btn-pause').innerText = paused ? '▶ RESUME' : '⏸ PAUSE'; }
function toggleMode(m){
  mode = m;
  document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn-' + m);
  if (btn) btn.classList.add('active');
  logTrace(`Mode: ${m.toUpperCase()}`);
}

// ─── MAIN RENDER LOOP ────────────────────────────────────────────────────────
let lastTime = performance.now();
let frames = 0, fpsTimer = 0;

function animate(now){
  requestAnimationFrame(animate);
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  frames++;
  fpsTimer += dt;
  if (fpsTimer >= 1.0) {
    const el = document.getElementById('fps-counter');
    if (el) el.innerText = `${frames} FPS`;
    frames = 0;
    fpsTimer = 0;
  }

  if (paused) return;

  ctx.fillStyle = '#020408';
  ctx.fillRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.03)';
  ctx.lineWidth = 1;
  const gridSize = 40;
  for (let gx = 0; gx < w; gx += gridSize) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
  }
  for (let gy = 0; gy < h; gy += gridSize) {
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
  }

  ctx.save();
  ctx.translate(cam.x, cam.y);
  ctx.scale(cam.scale, cam.scale);

  // Draw connections
  ctx.lineWidth = 1;
  connections.forEach(c => {
    const na = neurons[c.from];
    const nb = neurons[c.to];
    if (!na || !nb) return;

    ctx.strokeStyle = c.weight > 0 ? 'rgba(0, 240, 255, 0.1)' : 'rgba(255, 0, 85, 0.1)';
    if (mode === 'weights') {
      ctx.lineWidth = Math.abs(c.weight) * 2.2;
    }
    ctx.beginPath();
    ctx.moveTo(na.x, na.y);
    ctx.lineTo(nb.x, nb.y);
    ctx.stroke();
  });

  // Update & Draw Signals
  for (let i = signals.length - 1; i >= 0; i--) {
    const s = signals[i];
    s.progress += s.speed;

    const currNode = neurons[s.nodeId];
    if (!currNode) { signals.splice(i, 1); continue; }

    const outgoing = connections.filter(c => c.from === s.nodeId);
    if (outgoing.length === 0 || s.progress >= 1.0) {
      if (outgoing.length > 0 && Math.random() > 0.2) {
        const nextConn = outgoing[Math.floor(Math.random() * outgoing.length)];
        s.nodeId = nextConn.to;
        s.progress = 0;
        const targetNode = neurons[nextConn.to];
        if (targetNode) targetNode.activation = Math.min(1.0, targetNode.activation + 0.35);
      } else {
        signals.splice(i, 1);
        continue;
      }
    }

    if (outgoing.length > 0) {
      const target = neurons[outgoing[0].to];
      if (target) {
        const sx = currNode.x + (target.x - currNode.x) * s.progress;
        const sy = currNode.y + (target.y - currNode.y) * s.progress;
        ctx.fillStyle = s.color;
        ctx.shadowColor = s.color;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
  }

  // Draw Neurons
  neurons.forEach(n => {
    n.activation = Math.max(0.15, n.activation * 0.985);

    ctx.fillStyle = n.color;
    ctx.shadowColor = n.color;
    ctx.shadowBlur = n.activation * 20;

    ctx.beginPath();
    ctx.arc(n.x, n.y, 4 + n.activation * 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    if (hoveredNeuron && hoveredNeuron.id === n.id) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(n.x, n.y, 12, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  ctx.restore();
}

// ─── INTERACTION & PAN/ZOOM ──────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  dragging = true;
  dragStart = { x: e.clientX, y: e.clientY };
  camStart = { x: cam.x, y: cam.y };
});
window.addEventListener('mousemove', e => {
  if (dragging) {
    cam.x = camStart.x + (e.clientX - dragStart.x);
    cam.y = camStart.y + (e.clientY - dragStart.y);
  }

  // World coordinates
  const mx = e.clientX;
  const my = e.clientY;
  const worldX = (mx - cam.x) / cam.scale;
  const worldY = (my - cam.y) / cam.scale;

  hoveredNeuron = null;
  for (let n of neurons) {
    const dist = Math.hypot(n.x - worldX, n.y - worldY);
    if (dist < 14) {
      hoveredNeuron = n;
      break;
    }
  }

  const tooltip = document.getElementById('tooltip');
  if (hoveredNeuron && tooltip) {
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY + 14) + 'px';
    tooltip.innerHTML = `<b style="color:${hoveredNeuron.color}">${hoveredNeuron.stageLabel}</b><br><span style="color:#aaa">Node #${hoveredNeuron.nodeIndex} &bull; ${hoveredNeuron.act}</span><br><span style="font-size:8.5px;color:#888;">${hoveredNeuron.desc}</span>`;
  } else if (tooltip) {
    tooltip.style.display = 'none';
  }
});
window.addEventListener('mouseup', () => { dragging = false; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
  cam.scale = Math.min(2.5, Math.max(0.4, cam.scale * zoomFactor));
});
canvas.addEventListener('click', () => {
  if (hoveredNeuron) {
    selectedNeuron = hoveredNeuron;
    document.getElementById('neuron-info').classList.add('visible');
    document.getElementById('ni-layer').innerText = hoveredNeuron.stageLabel;
    document.getElementById('ni-node').innerText = `Node #${hoveredNeuron.nodeIndex}`;
    document.getElementById('ni-win').innerText = `${(hoveredNeuron.activation * 100).toFixed(1)}%`;
    document.getElementById('ni-wout').innerText = `+${hoveredNeuron.bias.toFixed(4)}`;
    startSignal(hoveredNeuron.id);
  }
});

window.addEventListener('resize', resize);
window.addEventListener('message', (e) => {
  if (e.data === 'activate' || e.data?.action === 'activate') {
    resize();
    autoCenter(true);
    startForwardPass();
  }
});

setInterval(() => {
  if (!paused && neurons.length > 0) {
    const inputs = neurons.filter(n => n.col === 0);
    if (inputs.length > 0) {
      const lucky = inputs[Math.floor(Math.random() * inputs.length)];
      lucky.activation = 1.0;
      startSignal(lucky.id);
    }
  }
}, 1200);

resize();
autoCenter(false);
startForwardPass();
requestAnimationFrame(animate);
logTrace('Quillan-Ronin v5.4.0 ONI Neural Architecture Graph initialized.');
</script>
</body>
</html>"""

targets = [
    Path(r'C:\02_QUILLAN\02_Projects\docs\nn_visualizer.html'),
    Path(r'C:\02_QUILLAN\02_Projects\nn_visualizer.html'),
    Path(r'C:\02_QUILLAN\docs\nn_visualizer.html'),
    Path(r'C:\02_QUILLAN\nn_visualizer.html'),
]
for p in targets:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(nn_code, encoding='utf-8')
        print('Updated:', str(p))
    except Exception as e:
        print(f'Skipped {p}: {e}')