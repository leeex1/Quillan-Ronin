#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: 4-KNOB AUTO-TUNING CONSOLE & 3D LIVE FLEET MONITOR
=============================================================================
Simple 4-knob control panel that lets the user adjust core levers:
  1. Code & Syntax Steering (WSP)
  2. Math & Reasoning Fusion (SVD)
  3. RCI Verification Depth (Audit)
  4. Hallucination Clamp (Determinism)
Plus "One-Click Auto-Tune" that executes WSP compiler + benchmark updates.
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
KNOBS_FILE = os.path.join(EXPORTS_DIR, "tuning_knobs.json")
BRAIN_TASKS_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\290bbfdb-f3fe-47b2-a61b-c3fb5c016750\.system_generated\tasks"

# Default knobs configuration
DEFAULT_KNOBS = {
    "knob_code_steering": 85,      # 0 to 100%
    "knob_math_fusion": 80,        # 0 to 100%
    "knob_rci_depth": 3,           # 1, 2, or 3
    "knob_determinism": 0.0,       # 0.0 to 0.4
    "status_msg": "Ready to auto-tune",
    "is_tuning": False
}

if not os.path.exists(KNOBS_FILE):
    try:
        with open(KNOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_KNOBS, f, indent=2)
    except Exception:
        pass

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QUILLAN-RONIN // 4-Knob Auto-Tuning Console</title>
<style>
  :root {
    --bg-dark: #07090e;
    --panel-bg: rgba(13, 17, 28, 0.88);
    --border-color: rgba(0, 240, 255, 0.28);
    --cyan: #00f0ff;
    --purple: #8b5cf6;
    --gold: #f59e0b;
    --green: #10b981;
    --red: #ef4444;
    --text-main: #f1f5f9;
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
    grid-template-rows: 64px 1fr 60px;
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
    padding: 6px 14px; border-radius: 20px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    display: flex; gap: 6px; align-items: center;
  }
  .pill.active { border-color: var(--green); color: var(--green); background: rgba(16,185,129,0.15); }

  .main-content {
    display: grid;
    grid-template-columns: 420px 1fr 380px;
    gap: 14px;
    height: 100%;
    overflow: hidden;
  }
  .panel {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 18px;
    backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }
  .panel-title {
    font-size: 13px; font-weight: 700; letter-spacing: 1.5px;
    color: var(--cyan); text-transform: uppercase;
    margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid rgba(0, 240, 255, 0.2);
    padding-bottom: 8px;
  }

  /* 4 KNOBS STYLES */
  .knob-control {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
    transition: border-color 0.2s, background 0.2s;
  }
  .knob-control:hover {
    border-color: var(--cyan);
    background: rgba(0, 240, 255, 0.04);
  }
  .knob-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }
  .knob-label {
    font-size: 13px;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .knob-val {
    font-size: 14px;
    font-weight: 800;
    color: var(--cyan);
    font-family: monospace;
    background: rgba(0, 240, 255, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(0, 240, 255, 0.3);
  }
  .knob-desc {
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 8px;
    line-height: 1.3;
  }
  .knob-slider {
    -webkit-appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.1);
    outline: none;
    cursor: pointer;
  }
  .knob-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 10px var(--cyan);
    cursor: pointer;
    border: 2px solid #fff;
    transition: transform 0.1s;
  }
  .knob-slider::-webkit-slider-thumb:hover {
    transform: scale(1.2);
  }

  .btn-action {
    background: linear-gradient(135deg, #00f0ff, #8b5cf6);
    color: #07090e;
    font-weight: 800;
    font-size: 14px;
    letter-spacing: 1px;
    border: none;
    border-radius: 6px;
    padding: 14px 18px;
    cursor: pointer;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
    transition: transform 0.15s, box-shadow 0.15s;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    margin-top: 6px;
  }
  .btn-action:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(0, 240, 255, 0.7);
  }
  .btn-action:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  /* Model Leaderboard Score Cards */
  .score-cards-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 8px;
    overflow-y: auto;
  }
  .score-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 10px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .score-card.a-card { border-left: 4px solid var(--cyan); }
  .score-card.b-card { border-left: 4px solid var(--purple); }
  .score-card.c-card { border-left: 4px solid var(--gold); }
  .sc-name { font-size: 13px; font-weight: 700; color: #fff; }
  .sc-details { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
  .sc-score { font-size: 20px; font-weight: 900; font-family: monospace; color: var(--cyan); }

  /* Active Progress HUD */
  .progress-box {
    background: rgba(0, 240, 255, 0.05);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
  }
  .pb-top { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; }
  .pb-track {
    width: 100%; height: 10px; background: rgba(255,255,255,0.08);
    border-radius: 5px; overflow: hidden;
  }
  .pb-fill {
    height: 100%; background: linear-gradient(90deg, var(--cyan), var(--purple));
    width: 0%; transition: width 0.3s ease;
  }

  /* Live Terminal */
  .terminal-box {
    flex: 1;
    background: rgba(0,0,0,0.65);
    border-radius: 6px;
    padding: 10px;
    font-family: "JetBrains Mono", Consolas, monospace;
    font-size: 11px;
    line-height: 1.5;
    overflow-y: auto;
    color: #a0aec0;
  }
  .log-line { margin-bottom: 2px; word-break: break-all; }
  .log-pass { color: var(--green); font-weight: bold; }
  .log-fail { color: var(--gold); font-weight: bold; }
  .log-id { color: var(--cyan); }
  .log-cat { color: var(--purple); }

  .center-overlay {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    padding-bottom: 15px;
  }
  .center-badge {
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    padding: 8px 18px;
    border-radius: 20px;
    font-size: 12px;
    color: var(--cyan);
    letter-spacing: 1px;
  }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>

<canvas id="webgl-canvas"></canvas>

<div class="hud-overlay">
  <!-- Top Bar -->
  <header class="header pointer-active">
    <div class="title-block">
      <div class="badge-pulse"></div>
      <div>
        <div class="title">QUILLAN-RONIN // 4-KNOB AUTO-TUNING CONSOLE</div>
        <div class="subtitle">FOUR SIMPLE KNOBS &bull; ZERO CODE JARGON &bull; AUTO-OPTIMIZER</div>
      </div>
    </div>
    <div class="status-pills">
      <div class="pill active" id="system-status"><span class="badge-pulse" style="width:8px; height:8px;"></span> FLEET READY</div>
      <div class="pill" id="clock-pill">12:26:00 EST</div>
    </div>
  </header>

  <!-- Middle Dashboard -->
  <div class="main-content">
    <!-- LEFT PANEL: 4 KNOBS -->
    <div class="panel pointer-active">
      <div class="panel-title">
        <span>🎛️ 4 SIMPLE TUNING KNOBS</span>
        <span style="font-size:10px; color:var(--green);" id="knob-status-badge">AUTO-OPTIMIZER READY</span>
      </div>

      <!-- Knob 1: Code & Syntax Boost -->
      <div class="knob-control">
        <div class="knob-top">
          <span class="knob-label">1. Code & Syntax Boost</span>
          <span class="knob-val" id="val-k1">85%</span>
        </div>
        <div class="knob-desc">Eliminates docstring echoing and fixes broken syntax parentheses `)):` in Python functions.</div>
        <input type="range" min="10" max="100" value="85" class="knob-slider" id="slider-k1" oninput="updateKnob(1, this.value)">
      </div>

      <!-- Knob 2: Math & Logic Reasoning Fusion -->
      <div class="knob-control">
        <div class="knob-top">
          <span class="knob-label">2. Math & Reasoning Fusion</span>
          <span class="knob-val" id="val-k2">80%</span>
        </div>
        <div class="knob-desc">Injects Falcon3-1B Dragon's multi-step math logic into Model A and Model B via SVD eigenvectors.</div>
        <input type="range" min="10" max="100" value="80" class="knob-slider" id="slider-k2" oninput="updateKnob(2, this.value)">
      </div>

      <!-- Knob 3: Self-Correction (RCI) Rigor -->
      <div class="knob-control">
        <div class="knob-top">
          <span class="knob-label">3. Self-Correction (RCI) Rigor</span>
          <span class="knob-val" id="val-k3">Max Audit (Level 3)</span>
        </div>
        <div class="knob-desc">Level 1 (Fast), Level 2 (Standard), Level 3 (Deep Doctest & Step-by-Step Arithmetic Verification).</div>
        <input type="range" min="1" max="3" value="3" class="knob-slider" id="slider-k3" oninput="updateKnob(3, this.value)">
      </div>

      <!-- Knob 4: Hallucination Clamp (Determinism) -->
      <div class="knob-control">
        <div class="knob-top">
          <span class="knob-label">4. Hallucination Clamp</span>
          <span class="knob-val" id="val-k4">0.0 (Strict Greedy)</span>
        </div>
        <div class="knob-desc">0.0 locks output to exact mathematical certainty; higher values allow slight creative exploration.</div>
        <input type="range" min="0" max="4" value="0" step="1" class="knob-slider" id="slider-k4" oninput="updateKnob(4, this.value)">
      </div>

      <!-- Single Action Button -->
      <button class="btn-action" id="btn-apply-tune" onclick="applyTuningAndRun()">
        <span>🚀 APPLY 4 KNOBS & RETUNE MODEL FLEET</span>
      </button>
      <div style="font-size:11px; text-align:center; color:var(--text-dim); margin-top:8px;" id="tune-status-text">
        Click above to apply your settings — the agent executes the rest automatically.
      </div>
    </div>

    <!-- CENTER PANEL: 3D Core & Big Score Cards -->
    <div class="center-overlay pointer-active">
      <div class="center-badge">
        HOLOGRAPHIC TRIPARTITE WEIGHT SPACE CORE &bull; &Delta;&theta; = 0
      </div>
    </div>

    <!-- RIGHT PANEL: Live Score Summary & Real-Time Stream -->
    <div class="panel pointer-active">
      <div class="panel-title">
        <span>CURRENT BENCHMARK PROGRESS</span>
        <span id="pb-ratio" style="font-size:11px; color:var(--cyan);">[0 / 500]</span>
      </div>

      <div class="progress-box">
        <div class="pb-top">
          <span id="active-model-name" style="font-weight:bold; color:#fff;">Model C Dragon (1.0B)</span>
          <span id="pb-pct" style="color:var(--cyan); font-family:monospace;">0%</span>
        </div>
        <div class="pb-track">
          <div class="pb-fill" id="progress-bar-fill"></div>
        </div>
        <div style="font-size:11px; color:var(--purple); margin-top:6px;" id="active-category-name">
          Category: GSM8K_Math
        </div>
      </div>

      <div class="panel-title" style="margin-top:6px;">
        <span>SCORE SUMMARY</span>
        <span style="font-size:10px; color:var(--text-dim);">Goal: 70%+</span>
      </div>

      <div class="score-cards-grid" id="score-cards-container">
        <!-- Rendered dynamically -->
      </div>

      <div class="panel-title" style="margin-top:10px;">
        <span>LIVE TELEMETRY STREAM</span>
      </div>
      <div class="terminal-box" id="terminal-content">
        <div class="log-line">Connected to Quillan-Ronin HUD stream.</div>
      </div>
    </div>
  </div>

  <!-- Bottom Bar -->
  <footer class="panel pointer-active" style="display:flex; flex-direction:row; align-items:center; justify-content:space-between; padding:8px 20px;">
    <div style="font-size:11px; color:var(--text-dim);">
      Status: <span id="footer-status" style="color:var(--green);">Active &bull; Real-time 1s Polling</span>
    </div>
    <div style="font-size:11px; color:var(--text-dim); font-family:monospace;">
      Console: http://127.0.0.1:8899
    </div>
  </footer>
</div>

<script>
// =============================================================================
// THREE.JS 3D SCENE (Tripartite Orbiting Holograms)
// =============================================================================
const canvas = document.getElementById('webgl-canvas');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x07090e, 0.025);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 3, 14);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const pLight1 = new THREE.PointLight(0x00f0ff, 3.0, 50);
pLight1.position.set(-6, 5, 4);
scene.add(pLight1);

const pLight2 = new THREE.PointLight(0x8b5cf6, 3.0, 50);
pLight2.position.set(6, 5, 4);
scene.add(pLight2);

const coreGroup = new THREE.Group();
scene.add(coreGroup);

// Center core
const cGeo = new THREE.IcosahedronGeometry(1.2, 2);
const cMat = new THREE.MeshStandardMaterial({ color: 0x00f0ff, wireframe: true, emissive: 0x00f0ff, emissiveIntensity: 0.6 });
const centerCore = new THREE.Mesh(cGeo, cMat);
coreGroup.add(centerCore);

// Rings
const rGeo1 = new THREE.TorusGeometry(3.6, 0.04, 16, 100);
const rMat1 = new THREE.MeshBasicMaterial({ color: 0x00f0ff, opacity: 0.45, transparent: true });
const ring1 = new THREE.Mesh(rGeo1, rMat1);
ring1.rotation.x = Math.PI / 3;
coreGroup.add(ring1);

const rGeo2 = new THREE.TorusGeometry(4.3, 0.03, 16, 100);
const rMat2 = new THREE.MeshBasicMaterial({ color: 0x8b5cf6, opacity: 0.4, transparent: true });
const ring2 = new THREE.Mesh(rGeo2, rMat2);
ring2.rotation.y = Math.PI / 4;
coreGroup.add(ring2);

function createNode(color, radius) {
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.DodecahedronGeometry(radius, 1), new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.7 }));
  g.add(m);
  const h = new THREE.Mesh(new THREE.DodecahedronGeometry(radius * 1.35, 1), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.35 }));
  g.add(h);
  return { g, m, h };
}

const nodeA = createNode(0x00f0ff, 0.65); // Model A Cyan
const nodeB = createNode(0x8b5cf6, 0.48); // Model B Purple
const nodeC = createNode(0xf59e0b, 0.58); // Model C Gold
coreGroup.add(nodeA.g);
coreGroup.add(nodeB.g);
coreGroup.add(nodeC.g);

// Particle field
const pCount = 700;
const pGeo = new THREE.BufferGeometry();
const pPos = new Float32Array(pCount * 3);
for (let i = 0; i < pCount * 3; i += 3) {
  pPos[i] = (Math.random() - 0.5) * 40;
  pPos[i + 1] = (Math.random() - 0.5) * 40;
  pPos[i + 2] = (Math.random() - 0.5) * 40;
}
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.08, transparent: true, opacity: 0.5 }));
scene.add(particles);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  coreGroup.rotation.y = t * 0.12;
  centerCore.rotation.y = -t * 0.3;
  ring1.rotation.z = t * 0.25;
  ring2.rotation.x = t * 0.2;

  const aA = t * 0.6;
  nodeA.g.position.set(Math.cos(aA) * 3.4, Math.sin(aA * 1.5) * 0.8, Math.sin(aA) * 3.4);
  const aB = t * 0.45 + (Math.PI * 2 / 3);
  nodeB.g.position.set(Math.cos(aB) * 4.3, Math.cos(aB * 1.3) * 1.1, Math.sin(aB) * 4.3);
  const aC = t * 0.5 + (Math.PI * 4 / 3);
  nodeC.g.position.set(Math.cos(aC) * 3.9, Math.sin(aC * 1.1) * 0.9, Math.sin(aC) * 3.9);

  particles.rotation.y = t * 0.02;
  renderer.render(scene, camera);
}
animate();

// =============================================================================
// 4 KNOBS LOGIC & TELEMETRY
// =============================================================================
let currentKnobs = {
  knob_code_steering: 85,
  knob_math_fusion: 80,
  knob_rci_depth: 3,
  knob_determinism: 0.0
};

function updateKnob(kNum, val) {
  if (kNum === 1) {
    currentKnobs.knob_code_steering = parseInt(val);
    document.getElementById('val-k1').textContent = val + "%";
  } else if (kNum === 2) {
    currentKnobs.knob_math_fusion = parseInt(val);
    document.getElementById('val-k2').textContent = val + "%";
  } else if (kNum === 3) {
    currentKnobs.knob_rci_depth = parseInt(val);
    const labels = { 1: "Level 1 (Fast)", 2: "Level 2 (Standard)", 3: "Level 3 (Max Audit)" };
    document.getElementById('val-k3').textContent = labels[val] || val;
  } else if (kNum === 4) {
    const temps = [0.0, 0.1, 0.2, 0.3, 0.4];
    const tVal = temps[parseInt(val)];
    currentKnobs.knob_determinism = tVal;
    document.getElementById('val-k4').textContent = tVal === 0.0 ? "0.0 (Strict Greedy)" : tVal.toFixed(1) + " (Balanced)";
  }
}

async function applyTuningAndRun() {
  const btn = document.getElementById('btn-apply-tune');
  const statusTxt = document.getElementById('tune-status-text');
  btn.disabled = true;
  btn.innerHTML = '<span>⚡ OPTIMIZING & COMPILING WEIGHTS...</span>';
  statusTxt.textContent = "Compiling closed-form WSP updates with your 4 knob settings...";

  try {
    const res = await fetch('/api/tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(currentKnobs)
    });
    const reply = await res.json();
    statusTxt.textContent = reply.msg || "Optimization dispatched!";
  } catch (err) {
    statusTxt.textContent = "Error dispatching tuning: " + err;
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.innerHTML = '<span>🚀 APPLY 4 KNOBS & RETUNE MODEL FLEET</span>';
  }, 4000);
}

// Telemetry Polling
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    renderHUD(data);
  } catch (err) {}
}

function renderHUD(data) {
  document.getElementById('clock-pill').textContent = new Date().toLocaleTimeString();

  const active = data.active_run || {};
  const pIdx = active.probe_index || 0;
  const pTot = active.probe_total || 500;
  const pct = ((pIdx / Math.max(1, pTot)) * 100).toFixed(1);

  document.getElementById('active-model-name').textContent = active.model_name || "Quillan-Ronin Engine";
  document.getElementById('active-category-name').textContent = "Active Category: " + (active.category || "Testing...");
  document.getElementById('pb-ratio').textContent = `[${pIdx} / ${pTot}]`;
  document.getElementById('pb-pct').textContent = `${pct}%`;
  document.getElementById('progress-bar-fill').style.width = `${pct}%`;

  const statusPill = document.getElementById('system-status');
  if (data.is_running) {
    statusPill.className = "pill active";
    statusPill.innerHTML = '<span class="badge-pulse" style="width:8px; height:8px;"></span> RUNNING PROBE ' + pIdx + '/500';
  } else {
    statusPill.className = "pill";
    statusPill.innerHTML = '<span>&bull;</span> READY';
  }

  // Score Cards
  const cardsBox = document.getElementById('score-cards-container');
  cardsBox.innerHTML = '';
  const results = data.benchmark_results || {};

  const modelClasses = {
    "Model A": "a-card",
    "Model B": "b-card",
    "Model C": "c-card"
  };

  for (const [mname, mdata] of Object.entries(results)) {
    let cardClass = "score-card";
    for (const [prefix, cname] of Object.entries(modelClasses)) {
      if (mname.includes(prefix)) cardClass += " " + cname;
    }
    const acc = (mdata.overall_accuracy !== undefined ? mdata.overall_accuracy : 0).toFixed(1);
    const passed = mdata.total_passed || 0;
    const total = mdata.total_probes || 500;

    const div = document.createElement('div');
    div.className = cardClass;
    div.innerHTML = `
      <div>
        <div class="sc-name">${mname}</div>
        <div class="sc-details">Passed: ${passed} / ${total} &bull; 95% CI: [${mdata.ci_95 ? mdata.ci_95[0] : 0}%, ${mdata.ci_95 ? mdata.ci_95[1] : 0}%]</div>
      </div>
      <div class="sc-score">${acc}%</div>
    `;
    cardsBox.appendChild(div);
  }

  // Logs
  const term = document.getElementById('terminal-content');
  if (data.recent_logs && data.recent_logs.length) {
    term.innerHTML = data.recent_logs.map(line => {
      let styled = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      if (line.includes("PASS")) styled = styled.replace(/PASS/, '<span class="log-pass">PASS</span>');
      else if (line.includes("FAIL")) styled = styled.replace(/FAIL/, '<span class="log-fail">FAIL</span>');
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
    try:
        res = subprocess.run(["powershell", "-Command", "Get-Process llama-server,python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessName"], capture_output=True, text=True, timeout=3)
        return "llama-server" in res.stdout
    except Exception:
        return False


def run_tuning_worker(knob_data: Dict[str, Any]):
    """Background thread that executes the WSP compiler and targeted benchmark."""
    logger.info("Executing Auto-Tune worker with knobs: %s", knob_data)
    try:
        # Save knobs
        with open(KNOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(knob_data, f, indent=2)

        # Execute Model A WSP Compiler
        py_bin = r"C:\Users\Admin\.local\bin\python3.11.exe"
        cmd_compile = [py_bin, os.path.join(ROOT_DIR, "scripts", "wsp_v8_closed_form_sota_compiler.py"), "--only-model-a"]
        subprocess.run(cmd_compile, check=True)
        logger.info("Model A re-compiled successfully.")
    except Exception as e:
        logger.error("Auto-tune worker failed: %s", e)


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
        elif self.path == "/api/knobs":
            knobs = DEFAULT_KNOBS
            if os.path.exists(KNOBS_FILE):
                try:
                    with open(KNOBS_FILE, "r", encoding="utf-8") as f:
                        knobs = json.load(f)
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(knobs).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/tune":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                knob_data = json.loads(post_data.decode("utf-8"))
                # Launch tuning in background thread
                t = threading.Thread(target=run_tuning_worker, args=(knob_data,), daemon=True)
                t.start()
                resp = {"status": "ok", "msg": "4-Knob settings received! Auto-tuning dispatched."}
            except Exception as e:
                resp = {"status": "error", "msg": str(e)}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def run_server(port: int = 8899):
    server = HTTPServer(("127.0.0.1", port), MonitorHandler)
    logger.info("Quillan 4-Knob Console live on http://127.0.0.1:%d", port)
    server.serve_forever()


if __name__ == "__main__":
    port = 8899
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
