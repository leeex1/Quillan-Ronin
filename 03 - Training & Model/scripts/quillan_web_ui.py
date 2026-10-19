#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN SOVEREIGN NEURAL REASONING STUDIO (WEB UI)
===========================================================
Ultra-fast, responsive, zero-external-dependency dark-mode client interface for:
  - System 1 Mini (6L, 577M) rapid edge reasoning
  - System 2 Main Flagship (12L, 1-3B Class) deliberative cognitive reasoning
  - Live token streaming, reasoning trace isolation (<think>), and hardware telemetries.
"""

def get_web_ui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>👑 Quillan-Ronin Sovereign AI Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #06090e;
      --bg-surface: #0c121c;
      --bg-card: #121b2b;
      --bg-card-hover: #172439;
      --border-subtle: #1e2c44;
      --border-glow: #00f0ff44;
      --cyan: #00f0ff;
      --cyan-dim: #00f0ff22;
      --gold: #ffbe1a;
      --gold-dim: #ffbe1a22;
      --green: #00e676;
      --purple: #b388ff;
      --text-main: #f0f4fc;
      --text-muted: #8ca0bd;
      --font-ui: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-ui);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 15% 10%, #00f0ff08 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, #ffbe1a06 0%, transparent 40%);
    }

    /* HEADER */
    header {
      background: rgba(12, 18, 28, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-logo {
      font-size: 26px;
      filter: drop-shadow(0 0 8px #00f0ff88);
      animation: float 3s ease-in-out infinite alternate;
    }

    @keyframes float {
      from { transform: translateY(0px); }
      to { transform: translateY(-2px); }
    }

    .brand-text h1 {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #ffffff 40%, var(--cyan) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-text p {
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
      letter-spacing: 0.5px;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      font-family: var(--font-mono);
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
    }

    .badge-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.85); }
    }

    .btn-secondary {
      background: var(--bg-card);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s ease;
    }

    .btn-secondary:hover {
      background: var(--bg-card-hover);
      border-color: var(--cyan);
      color: var(--cyan);
      box-shadow: 0 0 12px var(--cyan-dim);
    }

    /* MAIN LAYOUT */
    .container {
      max-width: 1380px;
      margin: 0 auto;
      width: 100%;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      padding: 24px;
      flex: 1;
    }

    /* SIDEBAR */
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px;
      position: relative;
    }

    .card-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* MODEL TOGGLE */
    .model-selector {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .model-option {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 12px 14px;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
    }

    .model-option:hover {
      border-color: #3b5072;
      background: var(--bg-card-hover);
    }

    .model-option.active {
      border-color: var(--cyan);
      background: rgba(0, 240, 255, 0.06);
      box-shadow: 0 0 16px rgba(0, 240, 255, 0.12);
    }

    .model-option.active.main-model {
      border-color: var(--gold);
      background: rgba(255, 190, 26, 0.06);
      box-shadow: 0 0 16px rgba(255, 190, 26, 0.12);
    }

    .model-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
    }

    .model-name {
      font-size: 14px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .model-tag {
      font-size: 10px;
      font-family: var(--font-mono);
      padding: 2px 7px;
      border-radius: 12px;
      background: #1c2a40;
      color: var(--text-muted);
    }

    .model-desc {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
    }

    /* TELEMETRY GAUGES */
    .stat-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .stat-box {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px;
    }

    .stat-label {
      font-size: 10px;
      color: var(--text-muted);
      font-family: var(--font-mono);
      text-transform: uppercase;
    }

    .stat-value {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-main);
      margin-top: 2px;
      font-family: var(--font-mono);
    }

    /* CHIPS */
    .chip-container {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .prompt-chip {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 11px;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.2s ease;
      line-height: 1.2;
    }

    .prompt-chip:hover {
      background: var(--bg-card-hover);
      border-color: var(--cyan);
      color: #ffffff;
      transform: translateY(-1px);
    }

    /* CHAT PANEL */
    .chat-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      height: calc(100vh - 120px);
      overflow: hidden;
    }

    .chat-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(18, 27, 43, 0.4);
    }

    .chat-status {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      font-weight: 600;
    }

    .active-indicator {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--cyan);
      background: var(--cyan-dim);
      padding: 3px 8px;
      border-radius: 6px;
      border: 1px solid #00f0ff44;
    }

    .active-indicator.main {
      color: var(--gold);
      background: var(--gold-dim);
      border-color: #ffbe1a44;
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      scroll-behavior: smooth;
    }

    .message {
      display: flex;
      flex-direction: column;
      max-width: 85%;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message.user {
      align-self: flex-end;
    }

    .message.assistant {
      align-self: flex-start;
      max-width: 95%;
      width: 100%;
    }

    .msg-bubble {
      padding: 14px 18px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.6;
    }

    .message.user .msg-bubble {
      background: linear-gradient(135deg, #0052cc 0%, #0080ff 100%);
      color: #ffffff;
      border-bottom-right-radius: 2px;
      box-shadow: 0 4px 14px rgba(0, 128, 255, 0.25);
    }

    .message.assistant .msg-bubble {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-bottom-left-radius: 2px;
      width: 100%;
    }

    .msg-meta {
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--text-muted);
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* REASONING ACCORDION */
    .reasoning-block {
      background: rgba(6, 9, 14, 0.85);
      border: 1px solid #1a2a44;
      border-radius: 8px;
      margin-bottom: 12px;
      overflow: hidden;
    }

    .reasoning-toggle {
      padding: 8px 12px;
      font-size: 11px;
      font-family: var(--font-mono);
      color: var(--cyan);
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      background: rgba(0, 240, 255, 0.04);
      user-select: none;
    }

    .reasoning-content {
      padding: 12px;
      font-size: 12px;
      font-family: var(--font-mono);
      color: #8da4c4;
      line-height: 1.5;
      white-space: pre-wrap;
      border-top: 1px dashed #1a2a44;
      max-height: 260px;
      overflow-y: auto;
    }

    /* INPUT BAR */
    .chat-input-wrapper {
      padding: 16px 20px;
      border-top: 1px solid var(--border-subtle);
      background: var(--bg-surface);
    }

    .chat-form {
      display: flex;
      gap: 12px;
    }

    .chat-input {
      flex: 1;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 14px 18px;
      border-radius: 10px;
      font-size: 14px;
      font-family: var(--font-ui);
      outline: none;
      resize: none;
      height: 52px;
      line-height: 1.4;
      transition: all 0.2s ease;
    }

    .chat-input:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 16px var(--cyan-dim);
    }

    .btn-send {
      background: linear-gradient(135deg, var(--cyan) 0%, #00a8ff 100%);
      color: #050b14;
      border: none;
      padding: 0 24px;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }

    .btn-send:hover:not(:disabled) {
      box-shadow: 0 0 20px rgba(0, 240, 255, 0.5);
      transform: translateY(-1px);
    }

    .btn-send:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* BENCHMARK MODAL */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 24px;
    }

    .modal-overlay.open {
      display: flex;
    }

    .modal-content {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      max-width: 900px;
      width: 100%;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
    }

    .modal-header {
      padding: 20px 24px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .modal-body {
      padding: 24px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .bench-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 16px;
    }

    .bench-cat {
      font-size: 13px;
      font-weight: 700;
      color: var(--cyan);
      margin-bottom: 4px;
    }

    .bench-q {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 10px;
      font-style: italic;
    }

    .bench-resp {
      font-size: 13px;
      font-family: var(--font-mono);
      background: #080d14;
      padding: 10px;
      border-radius: 6px;
      white-space: pre-wrap;
      color: #e2eafc;
      border: 1px solid #162438;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #1a2a44; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #284068; }
  </style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="brand-logo">👑</div>
      <div class="brand-text">
        <h1>Quillan-Ronin Sovereign Studio</h1>
        <p>DUAL COGNITIVE REASONING ARCHITECTURE • PORT 8000</p>
      </div>
    </div>
    <div class="header-actions">
      <div class="badge">
        <span class="badge-dot"></span>
        <span id="gateway-status">LIVE GATEWAY</span>
      </div>
      <button class="btn-secondary" onclick="openBenchmarkModal()">
        📊 10Q Audit
      </button>
      <button class="btn-secondary" onclick="compactMemory()">
        🧹 Trim RAM
      </button>
      <button class="btn-secondary" onclick="reloadModels()">
        🔄 Reload
      </button>
    </div>
  </header>

  <!-- MAIN APP CONTAINER -->
  <div class="container">

    <!-- SIDEBAR -->
    <aside class="sidebar">

      <!-- MODEL SELECTOR -->
      <div class="card">
        <div class="card-title">
          <span>Active Neural Model</span>
          <span style="color:var(--cyan); font-family:var(--font-mono); font-size:11px;">MoE TOP-4</span>
        </div>
        <div class="model-selector">

          <!-- MINI MODEL -->
          <div class="model-option active" id="opt-mini" onclick="selectModel('quillan-oni-mini-6l')">
            <div class="model-header">
              <span class="model-name">⚡ System 1: Mini</span>
              <span class="model-tag">6 Layers</span>
            </div>
            <div class="model-desc">
              Rapid intuitive reasoning engine. 577M parameter scale with 34 routed experts. Peak tok/s.
            </div>
          </div>

          <!-- MAIN MODEL -->
          <div class="model-option main-model" id="opt-main" onclick="selectModel('quillan-oni-main-12l')">
            <div class="model-header">
              <span class="model-name">🧠 System 2: Main</span>
              <span class="model-tag" style="background:#3d2f09; color:var(--gold);">12 Layers</span>
            </div>
            <div class="model-desc">
              Flagship deliberative reasoning. 1–3B capacity class with 726.7M elements. Deep cognitive synthesis.
            </div>
          </div>

        </div>
      </div>

      <!-- TELEMETRY -->
      <div class="card">
        <div class="card-title">Real-Time Telemetry</div>
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-label">RAM Working Set</div>
            <div class="stat-value" id="stat-ram">14.5 GB</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">CPU Cores Used</div>
            <div class="stat-value" id="stat-threads">3 Threads</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Last Velocity</div>
            <div class="stat-value" id="stat-toks" style="color:var(--cyan);">-- tok/s</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Last Latency</div>
            <div class="stat-value" id="stat-latency">-- ms</div>
          </div>
        </div>
      </div>

      <!-- DOMAIN BENCHMARK PRESETS -->
      <div class="card">
        <div class="card-title">Domain Benchmark Queries</div>
        <div class="chip-container">
          <div class="prompt-chip" onclick="applyPreset('Hello! Who are you, and what are your primary capabilities?')">🚀 Identity</div>
          <div class="prompt-chip" onclick="applyPreset('A right triangle has legs of length 5 and 12. What is the length of the hypotenuse and its area?')">📐 5-12-13 Math</div>
          <div class="prompt-chip" onclick="applyPreset('Write a Python function to check if a string is a palindrome.')">🐍 Python Palindrome</div>
          <div class="prompt-chip" onclick="applyPreset('Explain the primary function of photosynthesis in plants.')">🌿 Photosynthesis</div>
          <div class="prompt-chip" onclick="applyPreset('What is the key difference between SIGTERM and SIGKILL in Linux?')">🐧 Linux Signals</div>
          <div class="prompt-chip" onclick="applyPreset('What is SQL Injection and what is the standard method to prevent it?')">🛡️ Cybersecurity</div>
          <div class="prompt-chip" onclick="applyPreset('Explain the core mechanics of the Raft consensus algorithm, specifically leader election.')">⚙️ Raft Consensus</div>
          <div class="prompt-chip" onclick="applyPreset('What are the main trade-offs between monolithic and microservices architectures?')">🏛️ Architecture</div>
        </div>
      </div>

    </aside>

    <!-- CHAT WORKSPACE -->
    <main class="chat-panel">
      
      <div class="chat-header">
        <div class="chat-status">
          <span style="font-size:16px;">💬 Reasoning Playground</span>
          <span class="active-indicator" id="current-model-badge">quillan-oni-mini-6l</span>
        </div>
        <div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">
          Deterministic AVX2 SIMD Engine
        </div>
      </div>

      <!-- MESSAGES SCROLL -->
      <div class="chat-messages" id="chat-stream">
        <div class="message assistant">
          <div class="msg-bubble">
            Greetings. I am Quillan-Ronin. Both <strong>System 1 (6L Mini, 577M)</strong> and <strong>System 2 (12L Main, 1–3B Class)</strong> neural reasoning engines are fully loaded and synchronized in memory. Select your model and submit any technical, mathematical, or algorithmic inquiry to begin.
          </div>
          <div class="msg-meta">
            <span>⚡ Ready for inference</span>
          </div>
        </div>
      </div>

      <!-- INPUT BAR -->
      <div class="chat-input-wrapper">
        <form class="chat-form" onsubmit="submitChat(event)">
          <input 
            type="text" 
            class="chat-input" 
            id="prompt-input" 
            placeholder="Ask Quillan anything (e.g. distributed systems, math proofs, code synthesis)..." 
            autocomplete="off"
            required
          />
          <button type="submit" class="btn-send" id="btn-submit">
            <span>Generate</span>
            <span>➔</span>
          </button>
        </form>
      </div>

    </main>

  </div>

  <!-- BENCHMARK MODAL -->
  <div class="modal-overlay" id="bench-modal" onclick="closeBenchmarkModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h2 style="font-size:18px; font-weight:700;">🏆 10-Question Master Domain Benchmark Results</h2>
        <button class="btn-secondary" onclick="closeBenchmarkModal()">✕ Close</button>
      </div>
      <div class="modal-body" id="modal-bench-list">
        <div style="text-align:center; padding:20px; color:var(--text-muted); font-family:var(--font-mono);">
          Loading benchmark records from checkpoints/benchmark_10q_results.json...
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentModel = 'quillan-oni-mini-6l';
    let isGenerating = false;

    function selectModel(modelId) {
      currentModel = modelId;
      const optMini = document.getElementById('opt-mini');
      const optMain = document.getElementById('opt-main');
      const badge = document.getElementById('current-model-badge');

      if (modelId === 'quillan-oni-main-12l') {
        optMain.classList.add('active');
        optMini.classList.remove('active');
        badge.textContent = 'quillan-oni-main-12l (Flagship 12L)';
        badge.className = 'active-indicator main';
      } else {
        optMini.classList.add('active');
        optMain.classList.remove('active');
        badge.textContent = 'quillan-oni-mini-6l (Mini 6L)';
        badge.className = 'active-indicator';
      }
    }

    function applyPreset(text) {
      const input = document.getElementById('prompt-input');
      input.value = text;
      input.focus();
    }

    async function submitChat(e) {
      e.preventDefault();
      if (isGenerating) return;

      const input = document.getElementById('prompt-input');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      isGenerating = true;
      document.getElementById('btn-submit').disabled = true;

      const chatStream = document.getElementById('chat-stream');

      // 1. Append user message
      const userDiv = document.createElement('div');
      userDiv.className = 'message user';
      userDiv.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
      chatStream.appendChild(userDiv);

      // 2. Append assistant placeholder
      const asstDiv = document.createElement('div');
      asstDiv.className = 'message assistant';
      asstDiv.innerHTML = `
        <div class="msg-bubble">
          <div style="display:flex; align-items:center; gap:8px; color:var(--text-muted); font-family:var(--font-mono); font-size:12px;">
            <span class="badge-dot"></span>
            <span>Reasoning across 34 MoE expert manifolds...</span>
          </div>
        </div>
      `;
      chatStream.appendChild(asstDiv);
      chatStream.scrollTop = chatStream.scrollHeight;

      const startTime = performance.now();

      try {
        const response = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: currentModel,
            messages: [{ role: 'user', content: text }],
            max_tokens: currentModel.includes('12') ? 110 : 80,
            temperature: 0.65
          })
        });

        const data = await response.json();
        const elapsed = (performance.now() - startTime).toFixed(0);

        if (data.choices && data.choices[0] && data.choices[0].message) {
          const content = data.choices[0].message.content;
          const usage = data.usage || {};
          const tokens = usage.completion_tokens || 0;
          const tokPerSec = (tokens / (elapsed / 1000)).toFixed(1);

          // Update telemetries
          document.getElementById('stat-latency').textContent = `${elapsed} ms`;
          document.getElementById('stat-toks').textContent = `${tokPerSec} tok/s`;

          // Format reasoning if present
          let htmlContent = '';
          if (content.includes('<think>')) {
            const parts = content.split('</think>');
            const thinkPart = parts[0].replace('<think>', '').trim();
            const answerPart = parts[1] ? parts[1].trim() : '';

            htmlContent += `
              <div class="reasoning-block">
                <div class="reasoning-toggle" onclick="toggleReasoning(this)">
                  <span>🧠 Deliberative Reasoning Trace</span>
                  <span>▼</span>
                </div>
                <div class="reasoning-content">${escapeHtml(thinkPart)}</div>
              </div>
            `;
            if (answerPart) {
              htmlContent += `<div>${escapeHtml(answerPart)}</div>`;
            }
          } else {
            htmlContent = `<div>${escapeHtml(content)}</div>`;
          }

          asstDiv.querySelector('.msg-bubble').innerHTML = htmlContent;
          asstDiv.innerHTML += `
            <div class="msg-meta">
              <span>Model: ${currentModel}</span>
              <span>⚡ ${tokPerSec} tok/s</span>
              <span>⏱️ ${elapsed} ms</span>
              <span>🔢 ${tokens} tokens</span>
            </div>
          `;
        } else {
          asstDiv.querySelector('.msg-bubble').innerHTML = `<span style="color:#ff5252;">Error: ${escapeHtml(JSON.stringify(data))}</span>`;
        }
      } catch (err) {
        asstDiv.querySelector('.msg-bubble').innerHTML = `<span style="color:#ff5252;">Connection failed: ${escapeHtml(err.message)}</span>`;
      } finally {
        isGenerating = false;
        document.getElementById('btn-submit').disabled = false;
        chatStream.scrollTop = chatStream.scrollHeight;
        updateHealth();
      }
    }

    function toggleReasoning(el) {
      const content = el.nextElementSibling;
      const arrow = el.querySelector('span:last-child');
      if (content.style.display === 'none') {
        content.style.display = 'block';
        arrow.textContent = '▼';
      } else {
        content.style.display = 'none';
        arrow.textContent = '▶';
      }
    }

    async function updateHealth() {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data = await res.json();
          const ram = (data.working_set_mb / 1024).toFixed(1);
          document.getElementById('stat-ram').textContent = `${ram} GB`;
        }
      } catch (e) {}
    }

    async function compactMemory() {
      try {
        const res = await fetch('/api/hardware/compact', { method: 'POST' });
        if (res.ok) {
          const d = await res.json();
          alert(`Memory reclaimed: ${d.memory_reclaimed_mb.toFixed(1)} MB trimmed.`);
          updateHealth();
        }
      } catch (e) {
        alert('Trim failed: ' + e);
      }
    }

    async function reloadModels() {
      try {
        const res = await fetch('/api/reload', { method: 'POST', body: '{}', headers: { 'Content-Type': 'application/json' } });
        if (res.ok) {
          alert('Both 6L Mini and 12L Main models reloaded from disk.');
          updateHealth();
        }
      } catch (e) {
        alert('Reload failed: ' + e);
      }
    }

    async function openBenchmarkModal() {
      const modal = document.getElementById('bench-modal');
      modal.classList.add('open');
      const list = document.getElementById('modal-bench-list');

      try {
        const res = await fetch('/api/benchmark');
        if (res.ok) {
          const items = await res.json();
          list.innerHTML = items.map((it, idx) => `
            <div class="bench-card">
              <div class="bench-cat">${escapeHtml(it.category)}</div>
              <div class="bench-q">Prompt: "${escapeHtml(it.prompt)}"</div>
              <div class="bench-resp">${escapeHtml(it.response)}</div>
              <div class="msg-meta" style="margin-top:10px;">
                <span>⏱️ ${it.latency ? it.latency.toFixed(2) : 0}s</span>
                <span>⚡ ${it.tok_per_sec ? it.tok_per_sec.toFixed(1) : 0} tok/s</span>
                <span>🔢 ${it.tokens_generated || 0} tokens</span>
              </div>
            </div>
          `).join('');
        } else {
          list.innerHTML = `<div style="color:#ff5252;">Failed to load benchmark data.</div>`;
        }
      } catch (e) {
        list.innerHTML = `<div style="color:#ff5252;">Error loading benchmark data: ${escapeHtml(e.message)}</div>`;
      }
    }

    function closeBenchmarkModal() {
      document.getElementById('bench-modal').classList.remove('open');
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    // Initialize telemetry polling
    setInterval(updateHealth, 8000);
    updateHealth();
  </script>
</body>
</html>
"""
