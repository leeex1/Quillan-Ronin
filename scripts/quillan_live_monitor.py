#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: REAL-TIME 3D LIVE FLEET MONITOR & BENCHMARK HUD
=============================================================================
Runs a lightweight, zero-dependency local HTTP server (port 8899) serving a
Ronin-grade Three.js 3D Tripartite Core visualization + cyberpunk HUD that
streams live benchmark progress, active probe counts, model leaderboard stats,
and real-time logs.
=============================================================================
"""

import os
import re
import sys
import json
import time
import glob
import logging
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveMonitor")

ROOT_DIR = r"C:\02_QUILLAN"
EXPORTS_DIR = os.path.join(ROOT_DIR, "exports")
RESULTS_FILE = os.path.join(EXPORTS_DIR, "benchmark_100_per_section_results.json")
BRAIN_TASKS_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\290bbfdb-f3fe-47b2-a61b-c3fb5c016750\.system_generated\tasks"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QUILLAN-RONIN // Tripartite Fleet Command & Live Monitor</title>
<style>
  :root {
    --bg-dark: #08090d;
    --panel-bg: rgba(14, 17, 27, 0.82);
    --border-color: rgba(0, 240, 255, 0.25);
    --cyan: #00f0ff;
    --purple: #8b5cf6;
    --gold: #f59e0b;
    --green: #10b981;
    --red: #ef4444;
    --text-main: #e2e8f0;
    --text-dim: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "JetBrains Mono", monospace;
    overflow: hidden;
    height: 100vh;
    width: 100vw;
  }
  #webgl-canvas {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1;
  }
  .hud-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 2;
    pointer-events: none;
    display: grid;
    grid-template-rows: 64px 1fr 220px;
    padding: 16px;
    gap: 12px;
  }
  .pointer-active { pointer-events: auto; }
  .header {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.08);
  }
  .title-block { display: flex; align-items: center; gap: 12px; }
  .badge-pulse {
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 10px var(--cyan);
    animation: pulse 1.6s infinite;
  }
  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.4); opacity: 0.5; }
  }
  .title { font-size: 18px; font-weight: 700; letter-spacing: 1.5px; color: #fff; }
  .subtitle { font-size: 11px; color: var(--cyan); letter-spacing: 1px; }
  .status-pills { display: flex; gap: 12px; align-items: center; font-size: 12px; }
  .pill {
    padding: 6px 12px; border-radius: 20px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    display: flex; gap: 6px; align-items: center;
  }
  .pill.active { border-color: var(--green); color: var(--green); background: rgba(16,185,129,0.1); }
  
  .main-content {
    display: grid;
    grid-template-columns: 360px 1fr 380px;
    gap: 12px;
    height: 100%;
    overflow: hidden;
  }
  .panel {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }
  .panel-title {
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    color: var(--cyan); text-transform: uppercase;
    margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid rgba(0, 240, 255, 0.15);
    padding-bottom: 8px;
  }

  /* Progress Bar */
  .active-run-card {
    background: rgba(0, 240, 255, 0.04);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
  }
  .run-model-title { font-size: 14px; font-weight: bold; color: #fff; margin-bottom: 4px; }
  .run-category { font-size: 12px; color: var(--purple); font-family: monospace; }
  .progress-wrap { margin-top: 10px; }
  .progress-text { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; font-family: monospace; }
  .progress-track {
    width: 100%; height: 8px; background: rgba(255,255,255,0.08);
    border-radius: 4px; overflow: hidden; position: relative;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, var(--cyan), var(--purple));
    width: 0%; transition: width 0.3s ease;
  }

  /* Model Leaderboard Cards */
  .model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 10px;
    transition: transform 0.2s, border-color 0.2s;
  }
  .model-card:hover { border-color: var(--cyan); transform: translateY(-2px); }
  .model-card.current { border-color: var(--cyan); background: rgba(0, 240, 255, 0.05); }
  .mc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .mc-name { font-size: 13px; font-weight: 700; color: #fff; }
  .mc-score { font-size: 16px; font-weight: 800; color: var(--cyan); font-family: monospace; }
  .mc-ci { font-size: 10px; color: var(--text-dim); }
  
  .cat-bars { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-top: 6px; }
  .cb-item { text-align: center; }
  .cb-name { font-size: 8px; color: var(--text-dim); text-transform: uppercase; margin-bottom: 2px; }
  .cb-bar-bg { height: 28px; width: 100%; background: rgba(255,255,255,0.05); border-radius: 3px; position: relative; overflow: hidden; display: flex; align-items: flex-end; }
  .cb-bar-fill { width: 100%; background: var(--cyan); transition: height 0.4s ease; }
  .cb-val { font-size: 9px; font-family: monospace; margin-top: 2px; }

  /* Live Terminal */
  .terminal-box {
    flex: 1;
    background: rgba(0,0,0,0.6);
    border-radius: 6px;
    padding: 10px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    line-height: 1.45;
    overflow-y: auto;
    color: #a0aec0;
  }
  .log-line { margin-bottom: 2px; word-break: break-all; }
  .log-pass { color: var(--green); font-weight: bold; }
  .log-fail { color: var(--gold); font-weight: bold; }
  .log-id { color: var(--cyan); }
  .log-cat { color: var(--purple); }

  /* 3D Center Area Overlay */
  .center-display {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    padding-bottom: 20px;
  }
  .core-metrics {
    display: flex;
    gap: 16px;
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    padding: 10px 20px;
    border-radius: 30px;
    backdrop-filter: blur(8px);
  }
  .cm-stat { text-align: center; }
  .cm-label { font-size: 9px; color: var(--text-dim); letter-spacing: 1px; text-transform: uppercase; }
  .cm-val { font-size: 16px; font-weight: bold; color: var(--cyan); font-family: monospace; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>

<canvas id="webgl-canvas"></canvas>

<div class="hud-overlay">
  <!-- Top Header -->
  <header class="header pointer-active">
    <div class="title-block">
      <div class="badge-pulse"></div>
      <div>
        <div class="title">QUILLAN-RONIN // FLEET COMMAND</div>
        <div class="subtitle">TRIPARTITE FRONTIER CALIBRATION & BENCHMARK HUD</div>
      </div>
    </div>
    <div class="status-pills">
      <div class="pill active" id="system-status"><span class="badge-pulse" style="width:8px; height:8px;"></span> ACTIVE BENCHMARK</div>
      <div class="pill" id="gpu-status">GPU: GTX 1050 (3.2 GB / 4.0 GB)</div>
      <div class="pill" id="clock-pill">12:21:00 EST</div>
    </div>
  </header>

  <!-- Middle Center HUD -->
  <div class="main-content">
    <!-- Left Column: Active Run + Fleet Stats -->
    <div class="panel pointer-active">
      <div class="panel-title">
        <span>CURRENT BENCHMARK PROBE</span>
        <span id="active-probe-badge">0 / 500</span>
      </div>

      <div class="active-run-card">
        <div class="run-model-title" id="run-model-name">Model C Dragon (1.0B WSP v8.0)</div>
        <div class="run-category" id="run-cat-name">Category: GSM8K_Math</div>
        <div class="progress-wrap">
          <div class="progress-text">
            <span id="progress-percent">0.0%</span>
            <span id="probe-ratio">0 / 500</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" id="probe-progress-bar"></div>
          </div>
        </div>
      </div>

      <div class="panel-title" style="margin-top: 10px;">
        <span>FLEET LEADERBOARD (100 / SECTION)</span>
        <span style="font-size:10px; color:var(--text-dim);">Wilson 95% CI</span>
      </div>
      
      <div style="flex: 1; overflow-y: auto;" id="models-list">
        <!-- Rendered dynamically -->
      </div>
    </div>

    <!-- Center 3D Space (Click-through to orbit controls) -->
    <div class="center-display">
      <div class="core-metrics pointer-active">
        <div class="cm-stat">
          <div class="cm-label">TOTAL PROBES</div>
          <div class="cm-val" id="cm-total-probes">1,500</div>
        </div>
        <div class="cm-stat" style="border-left: 1px solid rgba(255,255,255,0.1); padding-left: 16px;">
          <div class="cm-label">RCI VERIFICATIONS</div>
          <div class="cm-val" id="cm-rci-count">ACTIVE</div>
        </div>
        <div class="cm-stat" style="border-left: 1px solid rgba(255,255,255,0.1); padding-left: 16px;">
          <div class="cm-label">WSP GROWTH</div>
          <div class="cm-val">&Delta;&theta; = 0 PARAMS</div>
        </div>
      </div>
    </div>

    <!-- Right Column: Live Terminal Ticker -->
    <div class="panel pointer-active">
      <div class="panel-title">
        <span>LIVE EXECUTION STREAM</span>
        <span style="color:var(--green); font-size:10px;">LIVE (1s POLL)</span>
      </div>
      <div class="terminal-box" id="terminal-content">
        <div class="log-line">Initializing Quillan-Ronin HUD telemetry stream...</div>
      </div>
    </div>
  </div>

  <!-- Bottom Bar Info -->
  <div class="panel pointer-active" style="display:flex; flex-direction:row; align-items:center; justify-content:space-between; padding: 10px 20px;">
    <div style="font-size:11px; color:var(--text-dim);">
      [Theorem 1 WSP v8.0 Closed-Form Null-Space Engine] &bull; 
      <span style="color:var(--cyan);">Model A (Flagship 1.2B)</span> &bull; 
      <span style="color:var(--purple);">Model B (Mini 0.5B)</span> &bull; 
      <span style="color:var(--gold);">Model C (Dragon 1.0B)</span>
    </div>
    <div style="font-size:11px; color:var(--text-dim); font-family:monospace;">
      Server: http://127.0.0.1:8899 | Auto-Refresh: 1000ms
    </div>
  </div>
</div>

<script>
// =============================================================================
// THREE.JS 3D TRIPARTITE CORE SCENE
// =============================================================================
const canvas = document.getElementById('webgl-canvas');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x08090d, 0.025);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 3, 14);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// Ambient & Point Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const lightCyan = new THREE.PointLight(0x00f0ff, 2.5, 50);
lightCyan.position.set(-6, 4, 4);
scene.add(lightCyan);

const lightPurple = new THREE.PointLight(0x8b5cf6, 2.5, 50);
lightPurple.position.set(6, 4, 4);
scene.add(lightPurple);

const lightGold = new THREE.PointLight(0xf59e0b, 2.0, 50);
lightGold.position.set(0, -5, 4);
scene.add(lightGold);

// Tripartite Orbiting Nodes: Model A, Model B, Model C
const coreGroup = new THREE.Group();
scene.add(coreGroup);

// Central Pillar / Energy Sphere
const centerGeo = new THREE.IcosahedronGeometry(1.2, 2);
const centerMat = new THREE.MeshStandardMaterial({
  color: 0x00f0ff,
  wireframe: true,
  emissive: 0x00f0ff,
  emissiveIntensity: 0.6
});
const centerCore = new THREE.Mesh(centerGeo, centerMat);
coreGroup.add(centerCore);

// Outer Energy Rings
const ringGeo1 = new THREE.TorusGeometry(3.5, 0.04, 16, 100);
const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x00f0ff, opacity: 0.5, transparent: true });
const ring1 = new THREE.Mesh(ringGeo1, ringMat1);
ring1.rotation.x = Math.PI / 3;
coreGroup.add(ring1);

const ringGeo2 = new THREE.TorusGeometry(4.2, 0.03, 16, 100);
const ringMat2 = new THREE.MeshBasicMaterial({ color: 0x8b5cf6, opacity: 0.4, transparent: true });
const ring2 = new THREE.Mesh(ringGeo2, ringMat2);
ring2.rotation.y = Math.PI / 4;
coreGroup.add(ring2);

// Model Nodes
function createModelNode(color, radius, name) {
  const group = new THREE.Group();
  const geo = new THREE.DodecahedronGeometry(radius, 1);
  const mat = new THREE.MeshStandardMaterial({
    color: color,
    emissive: color,
    emissiveIntensity: 0.7,
    roughness: 0.2,
    metalness: 0.8
  });
  const mesh = new THREE.Mesh(geo, mat);
  group.add(mesh);
  
  // Outer Wireframe Halo
  const haloGeo = new THREE.DodecahedronGeometry(radius * 1.35, 1);
  const haloMat = new THREE.MeshBasicMaterial({ color: color, wireframe: true, transparent: true, opacity: 0.35 });
  const halo = new THREE.Mesh(haloGeo, haloMat);
  group.add(halo);

  return { group, mesh, halo };
}

const nodeA = createModelNode(0x00f0ff, 0.65, "Model A Flagship");
const nodeB = createModelNode(0x8b5cf6, 0.48, "Model B Mini");
const nodeC = createModelNode(0xf59e0b, 0.58, "Model C Dragon");

coreGroup.add(nodeA.group);
coreGroup.add(nodeB.group);
coreGroup.add(nodeC.group);

// Background Particle Starfield
const particleCount = 700;
const particleGeo = new THREE.BufferGeometry();
const particlePos = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i += 3) {
  particlePos[i] = (Math.random() - 0.5) * 40;
  particlePos[i + 1] = (Math.random() - 0.5) * 40;
  particlePos[i + 2] = (Math.random() - 0.5) * 40;
}
particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
const particleMat = new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.08, transparent: true, opacity: 0.5 });
const particles = new THREE.Points(particleGeo, particleMat);
scene.add(particles);

// Window resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animation Loop
let clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  coreGroup.rotation.y = t * 0.12;
  coreGroup.rotation.x = Math.sin(t * 0.08) * 0.1;

  ring1.rotation.z = t * 0.25;
  ring2.rotation.x = t * 0.2;

  centerCore.rotation.y = -t * 0.3;
  centerCore.rotation.z = t * 0.15;

  // Orbit node A (Radius 3.2)
  const angleA = t * 0.6;
  nodeA.group.position.set(Math.cos(angleA) * 3.2, Math.sin(angleA * 1.5) * 0.8, Math.sin(angleA) * 3.2);
  nodeA.mesh.rotation.y = t * 0.8;
  nodeA.halo.rotation.x = -t * 0.5;

  // Orbit node B (Radius 4.2)
  const angleB = t * 0.45 + (Math.PI * 2 / 3);
  nodeB.group.position.set(Math.cos(angleB) * 4.2, Math.cos(angleB * 1.3) * 1.1, Math.sin(angleB) * 4.2);
  nodeB.mesh.rotation.y = t * 1.1;

  // Orbit node C (Radius 3.8)
  const angleC = t * 0.5 + (Math.PI * 4 / 3);
  nodeC.group.position.set(Math.cos(angleC) * 3.8, Math.sin(angleC * 1.1) * 0.9, Math.sin(angleC) * 3.8);
  nodeC.mesh.rotation.y = t * 0.7;

  particles.rotation.y = t * 0.02;

  renderer.render(scene, camera);
}
animate();

// =============================================================================
// TELEMETRY HUD POLLING & UI UPDATER
// =============================================================================
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    renderHUD(data);
  } catch (err) {
    console.error("Telemetry poll failed:", err);
  }
}

function renderHUD(data) {
  // Clock
  const now = new Date();
  document.getElementById('clock-pill').textContent = now.toLocaleTimeString();

  // Active run & probe
  const active = data.active_run || {};
  const probeIdx = active.probe_index || 0;
  const probeTotal = active.probe_total || 500;
  const pct = ((probeIdx / Math.max(1, probeTotal)) * 100).toFixed(1);

  document.getElementById('run-model-name').textContent = active.model_name || "Quillan-Ronin Engine";
  document.getElementById('run-cat-name').textContent = "Current Category: " + (active.category || "Testing...");
  document.getElementById('active-probe-badge').textContent = `[${probeIdx} / ${probeTotal}]`;
  document.getElementById('probe-ratio').textContent = `${probeIdx} / ${probeTotal}`;
  document.getElementById('progress-percent').textContent = `${pct}%`;
  document.getElementById('probe-progress-bar').style.width = `${pct}%`;

  // Status badge
  const statusPill = document.getElementById('system-status');
  if (data.is_running) {
    statusPill.className = "pill active";
    statusPill.innerHTML = '<span class="badge-pulse" style="width:8px; height:8px;"></span> ACTIVE BENCHMARK RUNNING';
  } else {
    statusPill.className = "pill";
    statusPill.innerHTML = '<span style="width:8px; height:8px; border-radius:50%; background:#94a3b8; display:inline-block;"></span> SUITE COMPLETED';
  }

  // Model Leaderboard Cards
  const modelsContainer = document.getElementById('models-list');
  modelsContainer.innerHTML = '';
  const results = data.benchmark_results || {};
  
  for (const [mname, mdata] of Object.entries(results)) {
    const isCurrent = active.model_name && mname.includes(active.model_name.slice(0, 7));
    const card = document.createElement('div');
    card.className = `model-card ${isCurrent ? 'current' : ''}`;
    
    const acc = (mdata.overall_accuracy !== undefined ? mdata.overall_accuracy : 0).toFixed(1);
    const ciLow = mdata.ci_95 ? mdata.ci_95[0] : 0;
    const ciHigh = mdata.ci_95 ? mdata.ci_95[1] : 0;
    const cats = mdata.category_summary || {};

    let catBarsHtml = '';
    const catKeys = ["HumanEval_Code", "MMLU_Academic", "GSM8K_Math", "ARC_AGI_Inductive", "MCP_ToolCalling"];
    const catShort = {"HumanEval_Code": "CODE", "MMLU_Academic": "MMLU", "GSM8K_Math": "GSM", "ARC_AGI_Inductive": "ARC", "MCP_ToolCalling": "MCP"};

    for (const ck of catKeys) {
      const cinfo = cats[ck] || { accuracy: 0 };
      const cacc = Math.round(cinfo.accuracy || 0);
      catBarsHtml += `
        <div class="cb-item">
          <div class="cb-name">${catShort[ck]}</div>
          <div class="cb-bar-bg"><div class="cb-bar-fill" style="height:${cacc}%;"></div></div>
          <div class="cb-val">${cacc}%</div>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="mc-header">
        <div>
          <div class="mc-name">${mname}</div>
          <div class="mc-ci">95% CI: [${ciLow}%, ${ciHigh}%] &bull; ${mdata.total_passed || 0}/${mdata.total_probes || 500}</div>
        </div>
        <div class="mc-score">${acc}%</div>
      </div>
      <div class="cat-bars">${catBarsHtml}</div>
    `;
    modelsContainer.appendChild(card);
  }

  // Live Terminal Logs
  const term = document.getElementById('terminal-content');
  if (data.recent_logs && data.recent_logs.length) {
    term.innerHTML = data.recent_logs.map(line => {
      let styled = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      if (line.includes("PASS")) {
        styled = styled.replace(/PASS/, '<span class="log-pass">PASS</span>');
      } else if (line.includes("FAIL")) {
        styled = styled.replace(/FAIL/, '<span class="log-fail">FAIL</span>');
      }
      styled = styled.replace(/\[(HumanEval_Code|MMLU_Academic|GSM8K_Math|ARC_AGI_Inductive|MCP_ToolCalling)\]/, '<span class="log-cat">[$1]</span>');
      styled = styled.replace(/ID:\s*([A-Za-z0-9_\-\/]+)/, 'ID: <span class="log-id">$1</span>');
      return `<div class="log-line">${styled}</div>`;
    }).join('');
    term.scrollTop = term.scrollHeight;
  }
}

setInterval(fetchStatus, 1000);
fetchStatus();
</script>
</body>
</html>
"""


def get_latest_task_log() -> Tuple[str, List[str]]:
    """Retrieves the most recent task log file in brain tasks."""
    logs = glob.glob(os.path.join(BRAIN_TASKS_DIR, "*.log"))
    if not logs:
        return "", []
    latest = max(logs, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return latest, lines[-40:]
    except Exception:
        return latest, []


def extract_current_probe_info(log_lines: List[str]) -> Dict[str, Any]:
    """Parses current probe, category, model name, and progress from recent log lines."""
    info: Dict[str, Any] = {
        "model_name": "Model C Dragon (1.0B WSP v8.0)",
        "category": "Benchmarking",
        "probe_index": 0,
        "probe_total": 500
    }
    for l in reversed(log_lines):
        if "EVALUATING MODEL:" in l:
            info["model_name"] = l.split("EVALUATING MODEL:")[-1].strip()
        m_probe = re.search(r"\[(\d+)\/(\d+)\]\s*\[\s*([A-Za-z0-9_]+)\s*\]", l)
        if m_probe:
            info["probe_index"] = int(m_probe.group(1))
            info["probe_total"] = int(m_probe.group(2))
            info["category"] = m_probe.group(3).strip()
            break
    return info


def check_if_running() -> bool:
    """Checks if llama-server or run_100_per_section_suite is currently running."""
    try:
        res = subprocess.run(["powershell", "-Command", "Get-Process llama-server,python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessName"], capture_output=True, text=True, timeout=3)
        return "llama-server" in res.stdout
    except Exception:
        return False


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/api/status":
            latest_path, recent_logs = get_latest_task_log()
            probe_info = extract_current_probe_info(recent_logs)
            is_active = check_if_running()

            results: Dict[str, Any] = {}
            if os.path.exists(RESULTS_FILE):
                try:
                    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                        results = json.load(f)
                except Exception:
                    pass

            payload = {
                "is_running": is_active,
                "active_run": probe_info,
                "benchmark_results": results,
                "recent_logs": recent_logs[-25:],
                "log_file": os.path.basename(latest_path) if latest_path else ""
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence normal request logs
        return


def run_server(port: int = 8899):
    server = HTTPServer(("127.0.0.1", port), MonitorHandler)
    logger.info("Quillan Fleet Monitor live on http://127.0.0.1:%d", port)
    server.serve_forever()


if __name__ == "__main__":
    port = 8899
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
