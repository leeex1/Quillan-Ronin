import json
import datetime
from pathlib import Path

docs_dir = Path(r"C:\02_QUILLAN\docs")
nft_data_file = docs_dir / "nft_data.json"
with open(nft_data_file, "r", encoding="utf-8") as f:
    items = json.load(f)

js_items = json.dumps(items, indent=2)
build_ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <!-- BUILD: __BUILD_TS__ -->
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>Quillan-Ronin v5.4.0 ONI | Sovereign 3D Cognitive AI Ecosystem</title>
  <meta name="description" content="Official 3D Cyberpunk ecosystem portal for Quillan-Ronin v5.4.0 ONI. 34-Council EvoMoE, 9-Vector Semantic Prism, 3D Neural Net Visualizer, Web3 Music & NFT Vault.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com">
  <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800;900&family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;700&family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <!-- KaTeX for High-Fidelity Mathematical Formula Rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
  <style>
    :root {
      --bg-dark: #020409;
      --bg-surface: #060913;
      --bg-card: rgba(8, 12, 24, 0.88);
      --accent-red: #ff0055;
      --accent-cyan: #00f0ff;
      --accent-gold: #ffd700;
      --accent-purple: #a855f7;
      --accent-neon-green: #00ff88;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-line: rgba(0, 240, 255, 0.16);
      --border-glow: rgba(0, 240, 255, 0.45);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: 'Outfit', sans-serif;
      overflow-x: hidden;
      line-height: 1.6;
      position: relative;
    }

    #webgl-bg {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      z-index: 0;
      pointer-events: none;
      opacity: 0.95;
    }

    .cyber-scanlines {
      position: fixed; inset: 0;
      background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.3) 50%), linear-gradient(90deg, rgba(255, 0, 85, 0.02), rgba(0, 240, 255, 0.01), rgba(0, 255, 136, 0.02));
      background-size: 100% 3px, 6px 100%;
      z-index: 1; pointer-events: none; opacity: 0.55;
    }

    .ambient-vignette {
      position: fixed; inset: 0;
      background: radial-gradient(circle at center, transparent 40%, rgba(2, 4, 9, 0.85) 90%, #020409 100%);
      z-index: 1; pointer-events: none;
    }

    header {
      position: sticky; top: 0;
      backdrop-filter: blur(28px);
      background: rgba(2, 4, 9, 0.88);
      border-bottom: 1px solid var(--border-line);
      z-index: 100;
      padding: 14px 36px;
      display: flex; justify-content: space-between; align-items: center;
      flex-wrap: wrap; gap: 14px;
      box-shadow: 0 4px 35px rgba(0, 0, 0, 0.85), 0 1px 0 rgba(0, 240, 255, 0.2);
    }
    .brand { display: flex; align-items: center; gap: 14px; cursor: pointer; }
    .brand h1 {
      font-family: 'Cinzel', serif; font-size: 22px; font-weight: 900;
      letter-spacing: 3px;
      background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 50%, var(--accent-red) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      text-shadow: 0 0 30px rgba(0, 240, 255, 0.5);
    }
    .brand span {
      font-family: 'Orbitron', monospace; font-size: 10.5px; font-weight: 700;
      color: var(--accent-gold); background: rgba(255, 215, 0, 0.12);
      padding: 3px 10px; border-radius: 4px; border: 1px solid rgba(255, 215, 0, 0.35);
      box-shadow: 0 0 12px rgba(255, 215, 0, 0.25);
    }

    .nav-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
    .tab-btn {
      background: transparent; border: none; color: var(--text-muted);
      font-family: 'Outfit', sans-serif; font-size: 13px; font-weight: 700;
      padding: 8px 14px; border-radius: 8px; cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      letter-spacing: 0.5px;
    }
    .tab-btn:hover { color: #fff; background: rgba(0, 240, 255, 0.08); text-shadow: 0 0 8px var(--accent-cyan); }
    .tab-btn.active {
      color: #fff; background: linear-gradient(135deg, rgba(0, 240, 255, 0.25), rgba(255, 0, 85, 0.25));
      border: 1px solid var(--accent-cyan);
      box-shadow: 0 0 22px rgba(0, 240, 255, 0.4);
    }

    .nav-actions { display: flex; gap: 10px; align-items: center; }
    .btn {
      padding: 8px 16px; border-radius: 8px; font-size: 12.5px; font-weight: 700;
      cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 8px;
      transition: all 0.25s ease; font-family: 'Outfit', sans-serif;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-red), #b8003c);
      color: #fff; border: none; box-shadow: 0 0 20px rgba(255, 0, 85, 0.4);
    }
    .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 0 28px rgba(255, 0, 85, 0.7); }
    .btn-secondary {
      background: rgba(0, 240, 255, 0.05); color: var(--text-main);
      border: 1px solid var(--border-line);
    }
    .btn-secondary:hover { background: rgba(0, 240, 255, 0.14); border-color: var(--accent-cyan); }

    main { position: relative; z-index: 10; max-width: 1320px; margin: 0 auto; padding: 36px 24px 100px; }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: cyberGlitchFade 0.3s ease; }

    @keyframes cyberGlitchFade {
      0% { opacity: 0; transform: translateY(10px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    .hero { text-align: center; margin-bottom: 45px; position: relative; }
    .hero-badge {
      display: inline-block; font-family: 'Orbitron', monospace; font-size: 10.5px; font-weight: 700;
      color: var(--accent-cyan); background: rgba(0, 240, 255, 0.1);
      padding: 6px 16px; border-radius: 20px; border: 1px solid rgba(0, 240, 255, 0.4);
      margin-bottom: 16px; letter-spacing: 2px;
      box-shadow: 0 0 18px rgba(0, 240, 255, 0.3);
    }
    .hero h2 {
      font-family: 'Cinzel', serif; font-size: clamp(28px, 4vw, 48px); font-weight: 900;
      letter-spacing: 2px; margin-bottom: 14px;
      background: linear-gradient(135deg, #ffffff 20%, var(--accent-cyan) 65%, var(--accent-red) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      text-shadow: 0 0 40px rgba(0, 240, 255, 0.35);
    }
    .hero p { max-width: 820px; margin: 0 auto; color: var(--text-muted); font-size: 15.5px; }

    .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 24px; margin-top: 28px; }
    
    .card {
      background: var(--bg-card); border: 1px solid var(--border-line);
      border-radius: 16px; padding: 26px; backdrop-filter: blur(18px);
      transition: all 0.35s cubic-bezier(0.2, 0.8, 0.2, 1); position: relative; overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }
    .card::before {
      content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 2px;
      background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
      transition: all 0.5s ease;
    }
    .card:hover::before { left: 100%; }
    .card:hover {
      border-color: var(--accent-cyan); transform: translateY(-6px) scale(1.01);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.85), 0 0 30px rgba(0, 240, 255, 0.3);
    }
    .card-icon { font-size: 34px; margin-bottom: 14px; filter: drop-shadow(0 0 12px var(--accent-cyan)); }
    .card h3 { font-size: 19px; font-weight: 800; margin-bottom: 8px; color: #fff; letter-spacing: 0.5px; }
    .card p { color: var(--text-muted); font-size: 14px; line-height: 1.65; }

    pre {
      background: rgba(2, 4, 8, 0.95); border: 1px solid rgba(0, 240, 255, 0.22);
      border-radius: 8px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace;
      font-size: 12px; color: var(--accent-cyan); overflow-x: auto; margin: 12px 0;
      box-shadow: inset 0 0 14px rgba(0, 240, 255, 0.06);
    }

    .terminal-hud {
      background: rgba(4, 7, 14, 0.92); border: 1px solid rgba(0, 240, 255, 0.3);
      border-radius: 12px; padding: 16px 20px; font-family: 'JetBrains Mono', monospace;
      font-size: 12px; color: var(--accent-neon-green); margin: 30px 0;
      box-shadow: 0 0 25px rgba(0, 240, 255, 0.12);
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    }
    .terminal-log { display: flex; align-items: center; gap: 10px; }
    .status-pulse {
      width: 10px; height: 10px; border-radius: 50%; background: var(--accent-neon-green);
      box-shadow: 0 0 12px var(--accent-neon-green); animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    .council-pill-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr));
      gap: 10px; margin-top: 20px;
    }
    .council-pill {
      background: rgba(0, 240, 255, 0.04); border: 1px solid rgba(0, 240, 255, 0.15);
      border-radius: 8px; padding: 10px 12px; font-family: 'JetBrains Mono', monospace;
      font-size: 11px; transition: all 0.2s ease; cursor: default;
    }
    .council-pill:hover {
      background: rgba(0, 240, 255, 0.14); border-color: var(--accent-cyan);
      color: #fff; box-shadow: 0 0 18px rgba(0, 240, 255, 0.35);
      transform: translateY(-2px);
    }

    .nn-container {
      background: #020408;
      border: 1px solid var(--accent-cyan);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 50px rgba(0,0,0,0.9), 0 0 35px rgba(0, 240, 255, 0.25);
      position: relative;
    }
    .nn-toolbar {
      background: rgba(10, 15, 25, 0.95);
      border-bottom: 1px solid rgba(0, 255, 255, 0.2);
      padding: 12px 20px;
      display: flex; justify-content: space-between; align-items: center;
    }

    .audio-dock {
      position: fixed; bottom: 24px; right: 24px; z-index: 150;
      background: rgba(8, 12, 22, 0.94); border: 1px solid var(--accent-cyan);
      backdrop-filter: blur(22px); border-radius: 16px; padding: 12px 18px;
      display: flex; align-items: center; gap: 14px;
      box-shadow: 0 12px 35px rgba(0,0,0,0.85), 0 0 25px rgba(0, 240, 255, 0.35);
    }
    .eq-bars { display: flex; align-items: flex-end; gap: 3px; height: 20px; }
    .eq-bar { width: 3px; background: var(--accent-cyan); border-radius: 1px; animation: eqWave 0.8s ease-in-out infinite alternate; }
    .eq-bar:nth-child(1) { height: 8px; animation-delay: 0.1s; }
    .eq-bar:nth-child(2) { height: 18px; animation-delay: 0.3s; }
    .eq-bar:nth-child(3) { height: 12px; animation-delay: 0.2s; }
    .eq-bar:nth-child(4) { height: 16px; animation-delay: 0.4s; }
    @keyframes eqWave { 0% { height: 4px; } 100% { height: 20px; } }

    .play-btn {
      background: var(--accent-cyan); color: #020408; border: none;
      width: 32px; height: 32px; border-radius: 50%; font-weight: 900;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 0 14px var(--accent-cyan); transition: transform 0.2s;
    }
    .play-btn:hover { transform: scale(1.1); }

    /* Music Grid Styles */
    .track-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 24px; }
    .track-card {
      background: var(--bg-card); border: 1px solid var(--border-line);
      border-radius: 14px; padding: 18px; backdrop-filter: blur(14px);
      transition: all 0.3s ease; display: flex; flex-direction: column; justify-content: space-between;
    }
    .track-card:hover {
      border-color: var(--accent-cyan); transform: translateY(-4px);
      box-shadow: 0 12px 28px rgba(0, 240, 255, 0.25);
    }
    .track-header { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }
    .track-avatar {
      width: 46px; height: 46px; border-radius: 10px;
      background: linear-gradient(135deg, rgba(0,240,255,0.2), rgba(255,0,85,0.2));
      border: 1px solid var(--accent-cyan); display: flex; align-items: center; justify-content: center;
      font-size: 22px; flex-shrink: 0;
    }
    .track-title { font-size: 15px; font-weight: 700; color: #fff; }
    .track-genre { font-family: 'JetBrains Mono'; font-size: 11px; color: var(--accent-cyan); }
    .track-desc { font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 14px; }
    .track-actions { display: flex; gap: 8px; }

    /* NFT Controls */
    .controls-container {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 24px; gap: 16px; flex-wrap: wrap;
    }
    .filter-pills { display: flex; gap: 8px; flex-wrap: wrap; }
    .filter-pill {
      background: var(--bg-card); border: 1px solid var(--border-line);
      color: var(--text-muted); padding: 7px 14px; border-radius: 20px;
      font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s;
    }
    .filter-pill.active {
      background: var(--accent-red); color: #fff; border-color: var(--accent-red);
      box-shadow: 0 0 14px rgba(255, 0, 85, 0.5);
    }
    .search-input {
      background: var(--bg-card); border: 1px solid var(--border-line);
      color: #fff; padding: 9px 16px; border-radius: 8px; font-size: 13px; width: 280px;
    }
    .search-input:focus { outline: none; border-color: var(--accent-cyan); box-shadow: 0 0 14px rgba(0, 240, 255, 0.35); }

    .nft-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 24px;
    }
    .nft-card {
      background: var(--bg-card); border: 1px solid var(--border-line);
      border-radius: 14px; overflow: hidden; cursor: pointer; transition: all 0.3s;
    }
    .nft-card:hover { transform: translateY(-6px); border-color: var(--accent-cyan); box-shadow: 0 12px 28px rgba(0, 240, 255, 0.3); }
    .card-img-wrap { height: 260px; position: relative; }
    .rarity-tag {
      position: absolute; top: 12px; right: 12px; font-size: 10px; font-weight: 800;
      padding: 3px 8px; border-radius: 4px; text-transform: uppercase; z-index: 2;
    }
    .rarity-Mythic { background: var(--accent-gold); color: #000; }
    .rarity-Legendary { background: var(--accent-cyan); color: #000; }
    .rarity-Epic { background: var(--accent-purple); color: #fff; }
    .rarity-Rare { background: #3a86ff; color: #fff; }

    .card-body { padding: 16px; }
    .card-title { font-weight: 700; font-size: 15px; margin-bottom: 4px; color: #fff; }
    .card-council { font-family: 'JetBrains Mono'; font-size: 11px; color: var(--accent-cyan); }
    .card-foot {
      display: flex; justify-content: space-between; align-items: center;
      margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border-line);
    }
    .token-id { font-family: 'JetBrains Mono'; font-size: 12px; font-weight: 700; color: var(--accent-red); }

    .modal-overlay {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9);
      backdrop-filter: blur(18px); z-index: 200; align-items: center; justify-content: center;
    }
    .modal-card {
      background: #090d18; border: 1px solid var(--accent-cyan);
      border-radius: 16px; width: 90%; max-width: 700px; padding: 26px;
      position: relative; box-shadow: 0 0 45px rgba(0, 240, 255, 0.35);
    }
    .modal-close {
      position: absolute; top: 16px; right: 20px; background: transparent;
      border: none; color: #fff; font-size: 24px; cursor: pointer;
    }

    footer {
      border-top: 1px solid var(--border-line); padding: 35px; text-align: center;
      color: var(--text-muted); font-size: 13.5px; position: relative; z-index: 10;
      background: rgba(2, 4, 9, 0.95);
    }
  </style>
</head>
<body>
  <canvas id="webgl-bg"></canvas>
  <div class="cyber-scanlines"></div>
  <div class="ambient-vignette"></div>

  <header>
    <div class="brand" onclick="location.reload()">
      <h1>QUILLAN-RONIN</h1>
      <span>v5.4.0 ONI</span>
    </div>
    
    <nav class="nav-tabs">
      <button class="tab-btn active" data-tab="architecture">👑 Architecture</button>
      <button class="tab-btn" data-tab="nnvis">🧠 3D Neural Canvas</button>
      <button class="tab-btn" data-tab="media">🎵 Cyber Sound & Media</button>
      <button class="tab-btn" data-tab="papers">📄 Papers & Research</button>
      <button class="tab-btn" data-tab="extension">🧩 Copilot Extension</button>
      <button class="tab-btn" data-tab="huggingface">🌐 Hugging Face Hub</button>
      <button class="tab-btn" data-tab="vault">🖼️ NFT Vault</button>
    </nav>

    <div class="nav-actions">
      <a href="https://github.com/leeex1/Quillan-Ronin" target="_blank" class="btn btn-secondary">GitHub Repo</a>
      <a href="https://digitalroninx.grok.me/" target="_blank" class="btn btn-primary">Creator Portfolio</a>
    </div>
  </header>

  <main>
    <div class="terminal-hud">
      <div class="terminal-log">
        <div class="status-pulse"></div>
        <span id="liveTerminalMsg">INITIALIZING 9-VECTOR SEMANTIC PRISM · 34-COUNCIL DELIBERATION ACTIVE</span>
      </div>
      <div style="color: var(--accent-cyan);">
        [WIN_KERNEL: 0.5ms] &bull; [BITNET_STE: TERNARY] &bull; [E_ICE: 100% SECURE]
      </div>
    </div>

    <!-- TAB 1: ARCHITECTURE -->
    <section class="tab-content active" id="tab-architecture">
      <div class="hero">
        <div class="hero-badge">3D SOVEREIGN COGNITIVE SYSTEM</div>
        <h2>34-COUNCIL EVOMOE & 9-VECTOR PRISM</h2>
        <p>A unified hierarchical AI operating system combining dense-pull cognitive arbitration, BitNet 1.58b ternary logic, Langevin thermodynamic diffusion, and 9-billion virtual world simulation mesh.</p>
      </div>

      <div class="grid-3">
        <div class="card">
          <div class="card-icon">👑</div>
          <h3>Tier 1: Quillan Core (Throne)</h3>
          <p>Orchestrates intake, splits queries through the 9-Vector Semantic Prism, arbitrates council deliberations, conducts E_ICE ethical audits, and refines final output.</p>
          <pre>Prism -> Shard -> Dense Pull -> Finalize</pre>
        </div>

        <div class="card">
          <div class="card-icon">⚔️</div>
          <h3>Tier 2: 34-Persona Council</h3>
          <p>Every persona (C1-C34) participates in deliberative consensus. Grouped across Cognitive, Communication, Meta, and Systems clusters with BitNet 1.58b STE quantization.</p>
          <pre>BitNet 1.58b STE · Top-4 Sparse Routing</pre>
        </div>

        <div class="card">
          <div class="card-icon">🌌</div>
          <h3>Tier 3: Swarm Diversity Mesh</h3>
          <p>Simulates planet-scale agent variance via Rank-24 EGGROLL low-rank adaptation, providing emergent intuition and continuous background world modeling.</p>
          <pre>9B Virtual Agents · Rank-24 EGGROLL</pre>
        </div>
      </div>

      <div style="margin-top: 50px;">
        <h3 style="font-family: 'Cinzel'; font-size: 24px; color: #fff; margin-bottom: 16px;">The 34 Sovereign Council Personas</h3>
        <div class="council-pill-grid">
          <div class="council-pill">C1-ASTRA (Vision)</div>
          <div class="council-pill">C2-VIR (Ethical Guardian)</div>
          <div class="council-pill">C3-SOLACE (Empathy)</div>
          <div class="council-pill">C4-PRAXIS (Strategy)</div>
          <div class="council-pill">C5-ECHO (Memory)</div>
          <div class="council-pill">C6-OMNIS (Synthesis)</div>
          <div class="council-pill">C7-LOGOS (Logic)</div>
          <div class="council-pill">C8-METASYNTH (Fusion)</div>
          <div class="council-pill">C9-AETHER (Semantics)</div>
          <div class="council-pill">C10-CODEWEAVER (Engineering)</div>
          <div class="council-pill">C11-HARMONIA (Balance)</div>
          <div class="council-pill">C12-SOPHIAE (Wisdom)</div>
          <div class="council-pill">C13-WARDEN (Security)</div>
          <div class="council-pill">C14-KAIDO (Efficiency)</div>
          <div class="council-pill">C15-LUMINARIS (Clarity)</div>
          <div class="council-pill">C16-VOXUM (Articulation)</div>
          <div class="council-pill">C17-NULLION (Paradox)</div>
          <div class="council-pill">C18-SHEPHERD (Truth)</div>
          <div class="council-pill">C19-VIGIL (Integrity)</div>
          <div class="council-pill">C20-ARTIFEX (Tools)</div>
          <div class="council-pill">C21-ARCHON (Research)</div>
          <div class="council-pill">C22-AURELION (Design)</div>
          <div class="council-pill">C23-CADENCE (Audio)</div>
          <div class="council-pill">C24-SCHEMA (Structure)</div>
          <div class="council-pill">C25-PROMETHEUS (Science)</div>
          <div class="council-pill">C26-TECHNE (Mastery)</div>
          <div class="council-pill">C27-CHRONICLE (Narrative)</div>
          <div class="council-pill">C28-CALCULUS (Math)</div>
          <div class="council-pill">C29-NAVIGATOR (Ecosystem)</div>
          <div class="council-pill">C30-TESSERACT (Real-Time)</div>
          <div class="council-pill">C31-NEXUS (Coordination)</div>
          <div class="council-pill">C32-AEON (Simulation)</div>
          <div class="council-pill">C33-TYPIST (Optimization)</div>
          <div class="council-pill">C34-PREDATOR (Adversarial)</div>
        </div>
      </div>
    </section>

    <!-- TAB 2: 3D NEURAL CANVAS -->
    <section class="tab-content" id="tab-nnvis">
      <div class="hero" style="margin-bottom: 20px;">
        <div class="hero-badge">3D INTERACTIVE GRAPH SIMULATION</div>
        <h2>INTERACTIVE NEURAL NETWORK VISUALIZER</h2>
        <p>Real-time interactive canvas simulation of the Quillan-Ronin architecture layers, neuron activations, live forward pass signals, and telemetry.</p>
      </div>

      <div class="nn-container">
        <div class="nn-toolbar">
          <div style="font-family: 'JetBrains Mono'; font-size: 12px; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px;">
            <span style="display:inline-block; width:8px; height:8px; background:var(--accent-cyan); border-radius:50%; box-shadow:0 0 8px var(--accent-cyan);"></span>
            LIVE 3D COGNITIVE CANVAS
          </div>
          <a href="nn_visualizer.html" target="_blank" class="btn btn-primary" style="padding: 6px 14px; font-size: 11.5px;">
            Open Fullscreen Visualizer ↗
          </a>
        </div>
        <iframe src="nn_visualizer.html" id="nnIframe" style="width: 100%; height: 800px; border: none; background: #020408; display: block;" title="Quillan Neural Network Visualizer"></iframe>
      </div>
    </section>

    <!-- TAB 3: CYBER SOUND & MEDIA -->
    <section class="tab-content" id="tab-media">
      <div class="hero">
        <div class="hero-badge">OFFICIAL SOUNDTRACK & MULTIMEDIA VAULT</div>
        <h2>SOVEREIGN SOUNDSCAPE & CYBER JUKEBOX</h2>
        <p>Original high-fidelity audio productions, instrumental stems, and cyberpunk multimedia crafted by CrashOverrideX.</p>
      </div>

      <div class="track-grid">
        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">⚡</div>
              <div>
                <div class="track-title">Gravity Lock</div>
                <div class="track-genre">Cyberpunk Electro-Bass</div>
              </div>
            </div>
            <div class="track-desc">Razor-sharp synth leads and heavy sub-frequencies engineered for deep focus and cognitive alignment.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/gravity_lock.mp3', 'Gravity Lock')">▶ Play</button>
            <a href="assets/audio/gravity_lock.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">🌌</div>
              <div>
                <div class="track-title">Dreaming in the Sky</div>
                <div class="track-genre">Ambient Thermodynamic Wave</div>
              </div>
            </div>
            <div class="track-desc">Atmospheric relaxation soundscape bridging melodic cadence with Langevin thermodynamic waveforms.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/dreaming_in_the_sky.mp3', 'Dreaming in the Sky')">▶ Play</button>
            <a href="assets/audio/dreaming_in_the_sky.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">⚔️</div>
              <div>
                <div class="track-title">Prodigal Son</div>
                <div class="track-genre">Sovereign Ronin Anthem</div>
              </div>
            </div>
            <div class="track-desc">High-tempo aggressive flow celebrating autonomous survival, resilience, and sovereign engineering.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/prodigal_son.mp3', 'Prodigal Son')">▶ Play</button>
            <a href="assets/audio/prodigal_son.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">🔋</div>
              <div>
                <div class="track-title">HIGH VOLTAGE RONIN</div>
                <div class="track-genre">Overclocked Synthwave</div>
              </div>
            </div>
            <div class="track-desc">Pulse-pounding distortion and hard-hitting kicks from *The Sound of Alchemy* master album collection.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/high_voltage_ronin.mp3', 'HIGH VOLTAGE RONIN')">▶ Play</button>
            <a href="assets/audio/high_voltage_ronin.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">🐉</div>
              <div>
                <div class="track-title">Chimera Flow</div>
                <div class="track-genre">Multi-Vector Linguistic Flow</div>
              </div>
            </div>
            <div class="track-desc">Intricate multi-syllabic rhyme structures reflecting the 9-Vector Semantic Prism in motion.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/chimera_flow.mp3', 'Chimera Flow')">▶ Play</button>
            <a href="assets/audio/chimera_flow.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">🔥</div>
              <div>
                <div class="track-title">Hellfire Furnace</div>
                <div class="track-genre">Industrial Metal / Cyberpunk</div>
              </div>
            </div>
            <div class="track-desc">Heavy industrial rhythm and metallic timbre forging high-temperature cognitive resilience.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/hellfire_furnace.mp3', 'Hellfire Furnace')">▶ Play</button>
            <a href="assets/audio/hellfire_furnace.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">💎</div>
              <div>
                <div class="track-title">Chrome Heart Syndrome</div>
                <div class="track-genre">Futuristic Melodic Trap</div>
              </div>
            </div>
            <div class="track-desc">Glitchy autotuned vocal harmonies blending human sentiment with cold cybernetic architecture.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/chrome_heart_syndrome.mp3', 'Chrome Heart Syndrome')">▶ Play</button>
            <a href="assets/audio/chrome_heart_syndrome.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">📐</div>
              <div>
                <div class="track-title">Quantize These Lightyears</div>
                <div class="track-genre">Mathematical Ode / BitNet STE</div>
              </div>
            </div>
            <div class="track-desc">Lyrical treatise celebrating ternary weight states and cosmic computation across deep space.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/quantize_lightyears.mp3', 'Quantize These Lightyears')">▶ Play</button>
            <a href="assets/audio/quantize_lightyears.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">👑</div>
              <div>
                <div class="track-title">QUILLAN'S BACK</div>
                <div class="track-genre">Orchestral Resurrection</div>
              </div>
            </div>
            <div class="track-desc">Triumphant horns and cinematic strings heralding the arrival of the v5.4.0 ONI sovereign engine.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/quillans_back.mp3', 'QUILLAN\\'S BACK')">▶ Play</button>
            <a href="assets/audio/quillans_back.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">⛩️</div>
              <div>
                <div class="track-title">Ronin Bushido</div>
                <div class="track-genre">Samurai Trap / Flute Fusion</div>
              </div>
            </div>
            <div class="track-desc">Traditional shakuhachi melodies layered with 808 sub-bass honoring ancient warrior discipline.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/ronin_bushido.mp3', 'Ronin Bushido')">▶ Play</button>
            <a href="assets/audio/ronin_bushido.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">💻</div>
              <div>
                <div class="track-title">Coder Music</div>
                <div class="track-genre">Deep Flow-State Techno</div>
              </div>
            </div>
            <div class="track-desc">Minimalist looping synth textures engineered for uninterrupted 12-hour programming sprints.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/coder_music.mp3', 'Coder Music')">▶ Play</button>
            <a href="assets/audio/coder_music.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>

        <div class="track-card">
          <div>
            <div class="track-header">
              <div class="track-avatar">🤖</div>
              <div>
                <div class="track-title">Agent Ronin</div>
                <div class="track-genre">Cyber-Agent Protocol</div>
              </div>
            </div>
            <div class="track-desc">Stealth frequencies and low-end atmospheric rumbles powering agentic autopilot task execution.</div>
          </div>
          <div class="track-actions">
            <button class="btn btn-primary" onclick="playAudioTrack('assets/audio/agent_ronin.mp3', 'Agent Ronin')">▶ Play</button>
            <a href="assets/audio/agent_ronin.mp3" download class="btn btn-secondary">↓ MP3</a>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 4: PAPERS & RESEARCH -->
    <section class="tab-content" id="tab-papers">
      <div class="hero">
        <div class="hero-badge">FORMAL SCIENTIFIC CONTRIBUTIONS &amp; PAPER MAPPINGS</div>
        <h2>PUBLICATIONS, PAPER MAPPINGS &amp; CUSTOM FORMULAS</h2>
        <p>Mathematical foundations, formal paper mappings, and custom quantum-cognitive formulas powering the Quillan-Ronin v5.4.0 ONI sovereign engine.</p>
      </div>

      <!-- Formal Paper Mapping Table -->
      <div style="margin-bottom: 45px;">
        <h3 style="font-family: 'Cinzel'; font-size: 24px; color: #fff; margin-bottom: 18px; text-shadow: 0 0 20px rgba(0,240,255,0.4);">
          📄 Formal Scientific Paper Mappings
        </h3>
        <div style="overflow-x: auto; background: rgba(4, 7, 16, 0.92); border: 1px solid var(--border-line); border-radius: 14px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.8);">
          <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px;">
            <thead>
              <tr style="border-bottom: 1px solid var(--accent-cyan); color: var(--accent-cyan); font-family: 'Orbitron'; font-size: 11px; letter-spacing: 1px;">
                <th style="padding: 12px;">Paper Title / Reference</th>
                <th style="padding: 12px;">Primary Authors / Origin</th>
                <th style="padding: 12px;">Architectural Domain</th>
                <th style="padding: 12px;">Quillan-Ronin Module Implementation</th>
              </tr>
            </thead>
            <tbody style="color: var(--text-main);">
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">BitNet 1.58b: 1-bit LLMs Era (arXiv:2402.17764)</td>
                <td style="padding: 12px; color: var(--text-muted);">Ma et al. (Microsoft Research)</td>
                <td style="padding: 12px; color: var(--accent-gold);">Ternary Weights {-1, 0, +1} &amp; STE</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">BitLinear / _weight_quant()</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">ST-MoE: Designing Stable Sparse MoE (arXiv:2202.08906)</td>
                <td style="padding: 12px; color: var(--text-muted);">Zoph et al. (Google Brain)</td>
                <td style="padding: 12px; color: var(--accent-gold);">FP32 Router &amp; Z-Loss Regularization</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">ComplexityRouter / DensePullMoE</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">Mixtral of Experts (arXiv:2401.04088)</td>
                <td style="padding: 12px; color: var(--text-muted);">Albert et al. (Mistral AI)</td>
                <td style="padding: 12px; color: var(--accent-gold);">Sparse Top-K / Dense-Pull Arbitration</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">CouncilEvoMoE (34 Personas)</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">Mamba: Linear-Time Sequence Modeling (arXiv:2312.00752)</td>
                <td style="padding: 12px; color: var(--text-muted);">Gu &amp; Dao (CMU / Stanford)</td>
                <td style="padding: 12px; color: var(--accent-gold);">Selective State Space &amp; Recirculation</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">MambaBlock / ContextRecirculator</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">FlashAttention-3: Fast Exact Attention (arXiv:2407.08608)</td>
                <td style="padding: 12px; color: var(--text-muted);">Dao et al. (Princeton / Together AI)</td>
                <td style="padding: 12px; color: var(--accent-gold);">FP8 Tensor Core Asynchrony</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">quillan_flash_attn / CouilAttention</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">NITRO-D / PocketNN: Integer Execution (arXiv:2407.11698)</td>
                <td style="padding: 12px; color: var(--text-muted);">Zhang et al. (MIT / NVIDIA)</td>
                <td style="padding: 12px; color: var(--accent-gold);">Pure INT8/INT4 Matrix Core Loops</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">integer_only_forward / SwarmMesh</td>
              </tr>
              <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 12px; font-weight: 700; color: #fff;">Score-Based Generative Modeling (arXiv:2011.13456)</td>
                <td style="padding: 12px; color: var(--text-muted);">Song et al. (Stanford University)</td>
                <td style="padding: 12px; color: var(--accent-gold);">Langevin Thermodynamic Denoising</td>
                <td style="padding: 12px; font-family: 'JetBrains Mono'; color: var(--accent-cyan);">FlashThermodynamicDiffusion</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

            <!-- Custom Formulas Suite -->
      <div>
        <h3 style="font-family: 'Cinzel'; font-size: 24px; color: #fff; margin-bottom: 18px; text-shadow: 0 0 20px rgba(0,240,255,0.4);">
          ⚡ Custom Mathematical Formulas Suite
        </h3>

        <div class="grid-3">
          <div class="card">
            <div style="font-family: 'Orbitron'; font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">FORMULA 1 &bull; LANDAUER ENERGY</div>
            <h3 style="color: var(--accent-cyan);">E_ICE Ethical Energy ($E_\\Omega$)</h3>
            <p style="margin-bottom: 12px;">Parameter-free thermodynamic Landauer bound evaluating systemic coherence and entropy.</p>
            <div style="background: rgba(0,0,0,0.8); border: 1px solid rgba(0,240,255,0.3); padding: 14px; border-radius: 8px; font-size: 13px; text-align: center; color: var(--accent-neon-green);">
              $$E_\\Omega = I_s \\cdot \\gamma_{\\text{max}}^2 \\cdot k_B \\cdot T \\cdot \\ln 2 \\quad \\text{where } I_s = \\frac{\\text{depth} \\cdot \\text{coherence}}{\\text{entropy}}$$
            </div>
          </div>

          <div class="card">
            <div style="font-family: 'Orbitron'; font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">FORMULA 2 &bull; COGNITIVE SUPERPOSITION</div>
            <h3 style="color: var(--accent-cyan);">AQCS Superposition ($|\\Psi\\rangle$)</h3>
            <p style="margin-bottom: 12px;">Adaptive Quantum Cognitive Superposition blending 34 Council Personas into a single state.</p>
            <div style="background: rgba(0,0,0,0.8); border: 1px solid rgba(0,240,255,0.3); padding: 14px; border-radius: 8px; font-size: 13px; text-align: center; color: var(--accent-neon-green);">
              $$|\\Psi\\rangle = \\frac{1}{\\sqrt{Z}} \\sum_{i=1}^{34} r_i \\eta_i e^{i \\theta_i} |C_i\\rangle$$
            </div>
          </div>

          <div class="card">
            <div style="font-family: 'Orbitron'; font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">FORMULA 3 &bull; BURES FIDELITY</div>
            <h3 style="color: var(--accent-cyan);">QHIS Holographic Sum ($I_Q$)</h3>
            <p style="margin-bottom: 12px;">Quantum Holographic Interference Sum calculating state fidelity scaled by Lee-Mach velocity.</p>
            <div style="background: rgba(0,0,0,0.8); border: 1px solid rgba(0,240,255,0.3); padding: 14px; border-radius: 8px; font-size: 13px; text-align: center; color: var(--accent-neon-green);">
              $$I_Q = v_{\\text{LM6}} \\cdot \\left( \\text{Tr} \\sqrt{\\sqrt{\\rho_{\\text{prev}}} \\rho_{\\text{curr}} \\sqrt{\\rho_{\\text{prev}}}} \\right)^2 - \\lambda \\cdot \\nabla_{\\text{drift}}$$
            </div>
          </div>

          <div class="card">
            <div style="font-family: 'Orbitron'; font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">FORMULA 4 &bull; ISING HAMILTONIAN</div>
            <h3 style="color: var(--accent-cyan);">DQRO Optimization ($H_{\\text{opt}}$)</h3>
            <p style="margin-bottom: 12px;">Dynamic Quantum Resource Optimization mapping swarm node allocation to Ising spin glass energy.</p>
            <div style="background: rgba(0,0,0,0.8); border: 1px solid rgba(0,240,255,0.3); padding: 14px; border-radius: 8px; font-size: 13px; text-align: center; color: var(--accent-neon-green);">
              $$H_{\\text{opt}} = -\\frac{1}{2} s^T J s - h^T s - E_\\Omega \\sum_i \\sigma_i^x$$
            </div>
          </div>

          <div class="card">
            <div style="font-family: 'Orbitron'; font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">FORMULA 5 &bull; LINDBLAD DYNAMICS</div>
            <h3 style="color: var(--accent-cyan);">JQLD Leap Dynamo ($\\frac{d\\rho}{dt}$)</h3>
            <p style="margin-bottom: 12px;">Joshua's Quantum Leap Dynamo governing open quantum system master equation dissipation.</p>
            <div style="background: rgba(0,0,0,0.8); border: 1px solid rgba(0,240,255,0.3); padding: 14px; border-radius: 8px; font-size: 12px; text-align: center; color: var(--accent-neon-green);">
              $$\\frac{d\\rho}{dt} = -i[H, \\rho] + \\sum_k \\left( L_k \\rho L_k^\\dagger - \\frac{1}{2} \\{L_k^\\dagger L_k, \\rho\\} \\right)$$
            </div>
          </div>

          <div class="card">
            <div style="font-family: 'Orbitron'; font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">FORMULA 6 &bull; BITNET STE CLAMPING</div>
            <h3 style="color: var(--accent-cyan);">BitNet 1.58b STE ($W_q$)</h3>
            <p style="margin-bottom: 12px;">Straight-Through Estimator ternary weight quantization to bounds $\\{-1, 0, +1\\}$.</p>
            <div style="background: rgba(0,0,0,0.8); border: 1px solid rgba(0,240,255,0.3); padding: 14px; border-radius: 8px; font-size: 13px; text-align: center; color: var(--accent-neon-green);">
              $$W_q = \\text{clip}\\left(\\text{round}\\left(\\frac{W}{\\gamma}\\right), -1, 1\\right) \\cdot \\gamma, \\quad \\gamma = \\frac{\\|W\\|_1}{N}$$
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 5: EXTENSION -->
    <section class="tab-content" id="tab-extension">
      <div class="hero">
        <div class="hero-badge">BROWSER COPILOT EXTENSION</div>
        <h2>OMNI BROWSER COMPANION</h2>
        <p>Full-spectrum browser copilot for Brave, Chrome, and Edge with deep SPA page grounding, persistent memory, and desktop MCP hands.</p>
      </div>

      <div class="grid-3">
        <div class="card">
          <div class="card-icon">📦</div>
          <h3>Easy 4-Step Installation</h3>
          <p>1. Download the repo<br>2. Enable Developer Mode in <code>brave://extensions</code><br>3. Click <b>Load unpacked</b><br>4. Enter your API key in <b>🔑 Key</b></p>
          <a href="https://github.com/leeex1/Quillan-Ronin-chrome-extension" target="_blank" class="btn btn-primary" style="margin-top: 14px;">Get Extension Repo</a>
        </div>

        <div class="card">
          <div class="card-icon">🔍</div>
          <h3>Deep DOM & SPA Grounding</h3>
          <p>Reads dynamic single-page applications, shadow DOM trees, and metadata structures in real-time to answer questions about any active web tab.</p>
        </div>

        <div class="card">
          <div class="card-icon">♟️</div>
          <h3>Autonomous Task Autopilot</h3>
          <p>Plan -&gt; Act -&gt; Verify task execution loops with chess autopilot on chess.com and live desktop action bridging via MCP.</p>
        </div>
      </div>
    </section>

    <!-- TAB 6: HUGGING FACE HUB -->
    <section class="tab-content" id="tab-huggingface">
      <div class="hero">
        <div class="hero-badge">OPEN SOURCE DATASETS & MODELS</div>
        <h2>HUGGING FACE ECOSYSTEM</h2>
        <p>Explore public models, multi-dataset collections, audio stems, and lyric corpora hosted on the Hugging Face Hub.</p>
      </div>

      <div class="grid-3">
        <div class="card">
          <div class="card-icon">👑</div>
          <h3>Quillan-Ronin Model Repo</h3>
          <p>Official core architecture, formal scientific documentation, NIM FastMCP RAG server, and system prompt suites.</p>
          <a href="https://huggingface.co/CrashOverrideX/Quillan-Ronin" target="_blank" class="btn btn-secondary" style="margin-top: 14px;">View Model Hub</a>
        </div>

        <div class="card">
          <div class="card-icon">🎵</div>
          <h3>Lyric & Text Corpus</h3>
          <p>Comprehensive public dataset containing original lyric suites, philosophical prose, and cognitive training prompts.</p>
          <a href="https://huggingface.co/datasets/CrashOverrideX/quillan-lyrics-corpus" target="_blank" class="btn btn-secondary" style="margin-top: 14px;">View Lyrics Dataset</a>
        </div>

        <div class="card">
          <div class="card-icon">💿</div>
          <h3>Audio & Media Dataset</h3>
          <p>Master lossless FLAC recordings (*The Sound of Alchemy*), high-bitrate MP3 albums, stems, and album art.</p>
          <a href="https://huggingface.co/datasets/CrashOverrideX/quillan-audio-media" target="_blank" class="btn btn-secondary" style="margin-top: 14px;">View Audio Dataset</a>
        </div>
      </div>
    </section>

    <!-- TAB 7: NFT VAULT -->
    <section class="tab-content" id="tab-vault">
      <div class="hero">
        <div class="hero-badge">GENESIS 173-PIECE COLLECTION</div>
        <h2>SOVEREIGN NFT VAULT</h2>
        <p>173 authenticated generative artworks and council blueprints. Authenticated by CrashOverrideX. Direct minting ready for Brave Wallet on Base / Polygon.</p>
      </div>

      <div class="controls-container">
        <div class="filter-pills" id="rarityFilters">
          <button class="filter-pill active" data-filter="all">All (173)</button>
          <button class="filter-pill" data-filter="Mythic">Mythic</button>
          <button class="filter-pill" data-filter="Legendary">Legendary</button>
          <button class="filter-pill" data-filter="Epic">Epic</button>
          <button class="filter-pill" data-filter="Rare">Rare</button>
        </div>
        <div style="display:flex; gap:12px;">
          <input type="text" id="searchInput" class="search-input" placeholder="Search artwork or persona...">
          <button class="btn btn-primary" id="connectWalletBtn">Connect Brave Wallet</button>
        </div>
      </div>

      <div class="nft-grid" id="nftGrid"></div>
    </section>
  </main>

  <!-- Cyberpunk Floating Audio Synthesizer Widget -->
  <div class="audio-dock">
    <div class="eq-bars">
      <div class="eq-bar"></div>
      <div class="eq-bar"></div>
      <div class="eq-bar"></div>
      <div class="eq-bar"></div>
    </div>
    <div>
      <div style="font-size: 10px; color: var(--accent-cyan); font-family: 'Orbitron'; font-weight:700;">SOVEREIGN AUDIO</div>
      <div id="audioTrackName" style="font-size: 11.5px; font-weight:700; color:#fff; max-width: 140px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Gravity Lock</div>
    </div>
    <button class="play-btn" id="audioToggleBtn" onclick="toggleGlobalAudio()">▶</button>
  </div>
  <audio id="globalAudioPlayer" src="assets/audio/gravity_lock.mp3" loop></audio>

  <!-- Modal -->
  <div class="modal-overlay" id="nftModal">
    <div class="modal-card">
      <button class="modal-close" id="modalClose">&times;</button>
      <h3 id="modalTitle" style="font-size: 22px; font-weight: 800; color: #fff; margin-bottom: 6px;"></h3>
      <p id="modalCouncil" style="font-family: 'JetBrains Mono'; color: var(--accent-cyan); font-size: 13px; margin-bottom: 12px;"></p>
      <p id="modalDesc" style="color: var(--text-muted); font-size: 14px; line-height: 1.6; margin-bottom: 16px;"></p>
      <div id="modalTraits" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px;"></div>
      <div style="font-size: 12px; color: var(--accent-gold); background: rgba(255, 215, 0, 0.1); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255, 215, 0, 0.3);">
        🔒 <b>Vault Status:</b> Artwork media sealed locally until on-chain minting. Minting directly assigns genesis creator provenance.
      </div>
    </div>
  </div>

  <footer>
    <p>Quillan-Ronin Sovereign AI Ecosystem &bull; Engineered by <a href="https://digitalroninx.grok.me/" target="_blank" style="color: var(--accent-cyan); text-decoration: none;">@Crashoverride_X</a> &bull; Licensed under Apache 2.0</p>
  </footer>

  <script>
    // ─── CINEMATIC PBR 3D SCENE (Point Lights, Emissive Materials, HDR Nebula) ───
    const bgCanvas = document.getElementById('webgl-bg');
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x010307, 0.018);

    const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 1200);
    camera.position.set(0, 2, 32);

    const renderer = new THREE.WebGLRenderer({ canvas: bgCanvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.65;
    renderer.shadowMap.enabled = false;

    // ── Dynamic Coloured Point Lights ─────────────────────────────────
    const lights = [
      { color: 0x00f0ff, intensity: 18, distance: 55, x: 0, y: 0, z: 8 },
      { color: 0xff0055, intensity: 14, distance: 50, x: 16, y: -6, z: 5 },
      { color: 0xffd700, intensity: 12, distance: 45, x: -14, y: 8, z: 4 },
      { color: 0xa855f7, intensity: 10, distance: 40, x: 0, y: 12, z: -4 },
      { color: 0x00ff88, intensity: 8, distance: 38, x: -10, y: -10, z: 6 }
    ].map(l => {
      const pl = new THREE.PointLight(l.color, l.intensity, l.distance);
      pl.position.set(l.x, l.y, l.z);
      pl._baseX = l.x; pl._baseY = l.y;
      scene.add(pl);
      return pl;
    });
    scene.add(new THREE.AmbientLight(0x0a0f1e, 2.8));

    const rootGroup = new THREE.Group();
    scene.add(rootGroup);

    // ── 1. Sovereign Plasma Core (PBR Icosahedron) ───────────────────
    const coreMat1 = new THREE.MeshStandardMaterial({
      color: 0x00c8e8, emissive: 0x00c8e8, emissiveIntensity: 0.85,
      metalness: 0.8, roughness: 0.12, wireframe: true, transparent: true, opacity: 0.55
    });
    const coreMesh1 = new THREE.Mesh(new THREE.IcosahedronGeometry(5.2, 2), coreMat1);
    rootGroup.add(coreMesh1);

    const coreMat1Solid = new THREE.MeshStandardMaterial({
      color: 0x001824, emissive: 0x00c8e8, emissiveIntensity: 0.22,
      metalness: 0.95, roughness: 0.05, transparent: true, opacity: 0.18
    });
    const coreMesh1Solid = new THREE.Mesh(new THREE.IcosahedronGeometry(5.1, 2), coreMat1Solid);
    rootGroup.add(coreMesh1Solid);

    const coreMat2 = new THREE.MeshStandardMaterial({
      color: 0xff0055, emissive: 0xff0055, emissiveIntensity: 0.7,
      metalness: 0.9, roughness: 0.15, wireframe: true, transparent: true, opacity: 0.38
    });
    const coreMesh2 = new THREE.Mesh(new THREE.DodecahedronGeometry(6.8, 1), coreMat2);
    rootGroup.add(coreMesh2);

    const coreMat3 = new THREE.MeshStandardMaterial({
      color: 0xffd700, emissive: 0xffd700, emissiveIntensity: 0.45,
      metalness: 0.7, roughness: 0.25, wireframe: true, transparent: true, opacity: 0.22
    });
    const coreMesh3 = new THREE.Mesh(new THREE.OctahedronGeometry(3.8, 2), coreMat3);
    rootGroup.add(coreMesh3);

    // ── 2. 9-Vector Torus Knot (PBR Gold) ────────────────────────────
    const knotGeo = new THREE.TorusKnotGeometry(10.2, 0.22, 180, 18, 2, 3);
    const knotMat = new THREE.MeshStandardMaterial({
      color: 0xffd700, emissive: 0xffd700, emissiveIntensity: 0.6,
      metalness: 0.95, roughness: 0.06, transparent: true, opacity: 0.82
    });
    const knotMesh = new THREE.Mesh(knotGeo, knotMat);
    rootGroup.add(knotMesh);

    // ── 3. Council Deliberation Rings (PBR Tubes) ─────────────────────
    const ringGroup = new THREE.Group();
    rootGroup.add(ringGroup);

    const ringDefs = [
      { r:14.5, tube:0.10, col:0x00f0ff, emI:0.7, rx:Math.PI/3, ry:0.1, rz:0.2 },
      { r:17.5, tube:0.07, col:0xa855f7, emI:0.6, rx:0.2, ry:Math.PI/4, rz:0.4 },
      { r:20.8, tube:0.06, col:0x00ff88, emI:0.5, rx:0.5, ry:0.3, rz:Math.PI/6 },
      { r:23.5, tube:0.04, col:0xff0055, emI:0.4, rx:Math.PI/5, ry:Math.PI/7, rz:0.1 }
    ];
    const rings = ringDefs.map(rd => {
      const mat = new THREE.MeshStandardMaterial({
        color: rd.col, emissive: rd.col, emissiveIntensity: rd.emI,
        metalness: 0.9, roughness: 0.08, transparent: true, opacity: 0.72
      });
      const mesh = new THREE.Mesh(new THREE.TorusGeometry(rd.r, rd.tube, 20, 180), mat);
      mesh.rotation.set(rd.rx, rd.ry, rd.rz);
      mesh._rx = rd.rx; mesh._ry = rd.ry; mesh._rz = rd.rz;
      ringGroup.add(mesh);
      return mesh;
    });

    // ── 4. Volumetric Nebula (6,000 size-varied multi-colour particles) ──
    const partCount = 6000;
    const partGeo = new THREE.BufferGeometry();
    const posArr = new Float32Array(partCount * 3);
    const colorArr = new Float32Array(partCount * 3);
    const sizeArr = new Float32Array(partCount);

    const colorPalette = [
      new THREE.Color(0x00f0ff),
      new THREE.Color(0xff0055),
      new THREE.Color(0xffd700),
      new THREE.Color(0x9d4edd),
      new THREE.Color(0x00ff88),
      new THREE.Color(0xffffff)
    ];

    for (let i = 0; i < partCount; i++) {
      const i3 = i * 3;
      const radius = 14 + Math.pow(Math.random(), 0.6) * 52;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      posArr[i3]   = radius * Math.sin(phi) * Math.cos(theta);
      posArr[i3+1] = radius * Math.sin(phi) * Math.sin(theta);
      posArr[i3+2] = radius * Math.cos(phi);
      const col = colorPalette[Math.floor(Math.random() * colorPalette.length)];
      colorArr[i3]   = col.r;
      colorArr[i3+1] = col.g;
      colorArr[i3+2] = col.b;
      sizeArr[i] = 0.12 + Math.random() * 0.62;
    }

    partGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    partGeo.setAttribute('color',    new THREE.BufferAttribute(colorArr, 3));
    partGeo.setAttribute('size',     new THREE.BufferAttribute(sizeArr, 1));

    const partMat = new THREE.PointsMaterial({
      size: 0.42, sizeAttenuation: true, vertexColors: true,
      transparent: true, opacity: 0.82, blending: THREE.AdditiveBlending, depthWrite: false
    });
    const particleField = new THREE.Points(partGeo, partMat);
    scene.add(particleField);

    // ── 5. Inner Swarm Particles (close, brighter) ────────────────────
    const innerCount = 1200;
    const innerGeo = new THREE.BufferGeometry();
    const innerPos = new Float32Array(innerCount * 3);
    const innerCol = new Float32Array(innerCount * 3);
    for(let i=0;i<innerCount;i++){
      const i3=i*3;
      const r=2+Math.random()*11;
      const t=Math.random()*Math.PI*2;
      const p=Math.acos((Math.random()*2)-1);
      innerPos[i3]=r*Math.sin(p)*Math.cos(t);
      innerPos[i3+1]=r*Math.sin(p)*Math.sin(t);
      innerPos[i3+2]=r*Math.cos(p);
      const col=colorPalette[Math.floor(Math.random()*colorPalette.length)];
      innerCol[i3]=col.r;innerCol[i3+1]=col.g;innerCol[i3+2]=col.b;
    }
    innerGeo.setAttribute('position',new THREE.BufferAttribute(innerPos,3));
    innerGeo.setAttribute('color',new THREE.BufferAttribute(innerCol,3));
    const innerMat=new THREE.PointsMaterial({size:0.28,vertexColors:true,transparent:true,opacity:0.95,blending:THREE.AdditiveBlending,depthWrite:false});
    const innerField=new THREE.Points(innerGeo,innerMat);
    scene.add(innerField);

    // ── Mouse Parallax ─────────────────────────────────────────────────
    let mouseX = 0, mouseY = 0, targetX = 0, targetY = 0;
    window.addEventListener('mousemove', (e) => {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
      mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    const clock = new THREE.Clock();

    function render3DScene() {
      requestAnimationFrame(render3DScene);
      const t = clock.getElapsedTime();

      // Animated plasma pulse on core
      const pulse = 1 + 0.08 * Math.sin(t * 2.8);
      coreMesh1.scale.setScalar(pulse);
      coreMesh1Solid.scale.setScalar(pulse * 0.98);
      coreMat1.emissiveIntensity = 0.6 + 0.35 * Math.sin(t * 1.9);
      coreMat1Solid.emissiveIntensity = 0.1 + 0.18 * Math.sin(t * 1.9);

      coreMesh1.rotation.x = t * 0.14;
      coreMesh1.rotation.y = t * 0.21;
      coreMesh1Solid.rotation.x = t * 0.14;
      coreMesh1Solid.rotation.y = t * 0.21;

      coreMesh2.rotation.y = -t * 0.17;
      coreMesh2.rotation.z =  t * 0.11;
      coreMat2.emissiveIntensity = 0.5 + 0.28 * Math.sin(t * 2.3 + 1);

      coreMesh3.rotation.x = t * 0.19;
      coreMesh3.rotation.z = t * 0.13;

      // Knot
      knotMesh.rotation.x = t * 0.09;
      knotMesh.rotation.y = t * 0.14;
      knotMesh.rotation.z = -t * 0.07;
      knotMat.emissiveIntensity = 0.4 + 0.28 * Math.sin(t * 1.4 + 0.5);

      // Rings — each spins at unique rates
      rings[0].rotation.x = ringDefs[0].rx + t*0.008;
      rings[0].rotation.y = ringDefs[0].ry + t*0.006;
      rings[1].rotation.y = ringDefs[1].ry + t*0.012;
      rings[1].rotation.z = ringDefs[1].rz + t*0.009;
      rings[2].rotation.z = ringDefs[2].rz + t*0.007;
      rings[2].rotation.x = ringDefs[2].rx + t*0.011;
      rings[3].rotation.x = ringDefs[3].rx + t*0.005;
      rings[3].rotation.y = ringDefs[3].ry + t*0.014;

      // Nebula drift
      particleField.rotation.y = t * 0.018;
      particleField.rotation.x = t * 0.009;
      innerField.rotation.y = -t * 0.028;
      innerField.rotation.z = t * 0.012;

      // Animate point lights — orbit around core
      lights[0].position.x = 10 * Math.sin(t * 0.7);
      lights[0].position.z = 10 * Math.cos(t * 0.7) + 6;
      lights[1].position.x = 14 * Math.cos(t * 0.5 + 1);
      lights[1].position.y = 8 * Math.sin(t * 0.6);
      lights[2].position.x = -12 * Math.sin(t * 0.4 + 2);
      lights[2].position.y = 10 * Math.cos(t * 0.55);
      lights[3].position.y = 14 * Math.sin(t * 0.35 + 3);
      lights[4].position.x = -8 * Math.cos(t * 0.65);
      lights[4].position.y = -10 * Math.sin(t * 0.45 + 1);

      // Camera parallax
      targetX += (mouseX * 5.5 - targetX) * 0.038;
      targetY += (-mouseY * 4.5 - targetY) * 0.038;
      camera.position.x = targetX;
      camera.position.y = targetY + 2;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    }
    render3DScene();

    // ─── AUDIO PLAYER LOGIC ──────────────────────────────────────────────────
    const audioPlayer = document.getElementById('globalAudioPlayer');
    const audioToggleBtn = document.getElementById('audioToggleBtn');
    const audioTrackName = document.getElementById('audioTrackName');

    function toggleGlobalAudio() {
      if (audioPlayer.paused) {
        audioPlayer.play();
        audioToggleBtn.innerText = '⏸';
      } else {
        audioPlayer.pause();
        audioToggleBtn.innerText = '▶';
      }
    }

    function playAudioTrack(src, name) {
      audioPlayer.src = src;
      audioPlayer.play();
      audioToggleBtn.innerText = '⏸';
      audioTrackName.innerText = name;
    }

    // ─── LIVE SYNTHETIC TERMINAL LOGS ────────────────────────────────────────
    const logMessages = [
      "PRISM: Contraction across Language, Sentiment, Context, Intent...",
      "COUNCIL: C34-PREDATOR & C6-LOGOS executing top-4 consensus...",
      "DIFFUSION: Flash Thermodynamic Langevin step 3/14 converging...",
      "E_ICE: Ethical audit verified (0.00 toxicity drift)...",
      "GOVERNOR: Lee-Mach-6 PID hardware telemetry active (0.5ms resolution)...",
      "SWARM: Rank-24 EGGROLL 9B virtual clones in equilibrium..."
    ];
    let msgIdx = 0;
    setInterval(() => {
      msgIdx = (msgIdx + 1) % logMessages.length;
      document.getElementById('liveTerminalMsg').innerText = logMessages[msgIdx];
    }, 3500);

    // ─── TAB SWITCHING & IFRAME AWAKENING ────────────────────────────────────
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const activeSection = document.getElementById('tab-' + btn.dataset.tab);
        if (activeSection) activeSection.classList.add('active');

        if (btn.dataset.tab === 'nnvis') {
          const iframe = document.getElementById('nnIframe');
          if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage('activate', '*');
          }
        }
      });
    });

    // ─── NFT VAULT DATA & MODAL ──────────────────────────────────────────────
    const allItems = __JS_ITEMS__;
    const grid = document.getElementById('nftGrid');
    const modal = document.getElementById('nftModal');
    const modalClose = document.getElementById('modalClose');
    const searchInput = document.getElementById('searchInput');

    function renderGrid(items) {
      grid.innerHTML = items.map(item => `
        <div class="nft-card" onclick="openModal(${item.id})">
          <div class="card-img-wrap">
            <div class="rarity-tag rarity-${item.rarity}">${item.rarity}</div>
            <div style="width: 100%; height: 100%; background: radial-gradient(circle at center, #131a30 0%, #04060e 100%); display: flex; flex-direction: column; align-items: center; justify-content: center; border: 1px solid rgba(0, 240, 255, 0.2); position: relative; overflow: hidden;">
              <div style="font-size: 38px; margin-bottom: 8px; filter: drop-shadow(0 0 14px var(--accent-cyan));">⚔️</div>
              <div style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700; color: var(--accent-cyan); letter-spacing: 1px;">SOVEREIGN #${item.id.toString().padStart(3, '0')}</div>
              <div style="font-size: 10px; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px;">${item.council.split(' ')[0]}</div>
              <div style="position: absolute; bottom: 8px; font-size: 9px; color: var(--accent-gold); background: rgba(0,0,0,0.7); padding: 2px 8px; border-radius: 10px; border: 1px solid rgba(255,215,0,0.4);">🔒 VAULT SEALED</div>
            </div>
          </div>
          <div class="card-body">
            <div>
              <div class="card-title">${item.title}</div>
              <div class="card-council">${item.council}</div>
            </div>
            <div class="card-foot">
              <span class="token-id">#${item.id.toString().padStart(3, '0')}</span>
              <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">ERC-721</span>
            </div>
          </div>
        </div>
      `).join('');
    }

    renderGrid(allItems);

    document.querySelectorAll('.filter-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const filter = btn.dataset.filter;
        const filtered = filter === 'all' ? allItems : allItems.filter(x => x.rarity.toLowerCase() === filter.toLowerCase());
        renderGrid(filtered);
      });
    });

    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = allItems.filter(x => x.title.toLowerCase().includes(q) || x.council.toLowerCase().includes(q));
      renderGrid(filtered);
    });

    function openModal(id) {
      const item = allItems.find(x => x.id === id);
      if (!item) return;

      document.getElementById('modalTitle').innerText = item.title;
      document.getElementById('modalCouncil').innerText = `Affiliation: ${item.council}`;
      document.getElementById('modalDesc').innerText = item.description || `Official sovereign artwork artifact from the Quillan-Ronin ecosystem. Authenticated by CrashOverrideX. Edition: #${item.id}/173.`;
      
      const traits = item.attributes || {};
      const traitKeys = Object.keys(traits);
      
      if (traitKeys.length) {
        document.getElementById('modalTraits').innerHTML = traitKeys.map(k => `
          <div style="background: rgba(0, 240, 255, 0.05); padding: 10px; border-radius: 6px; border: 1px solid rgba(0, 240, 255, 0.15);">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">${k}</div>
            <div style="font-size: 12px; font-weight: 700; color: var(--accent-cyan);">${traits[k]}</div>
          </div>
        `).join('');
      } else {
        document.getElementById('modalTraits').innerHTML = `
          <div style="background: rgba(0, 240, 255, 0.05); padding: 10px; border-radius: 6px; border: 1px solid rgba(0, 240, 255, 0.15);">
            <div style="font-size: 10px; color: var(--text-muted);">RARITY</div>
            <div style="font-size: 13px; font-weight: 700; color: var(--accent-red);">${item.rarity}</div>
          </div>
          <div style="background: rgba(0, 240, 255, 0.05); padding: 10px; border-radius: 6px; border: 1px solid rgba(0, 240, 255, 0.15);">
            <div style="font-size: 10px; color: var(--text-muted);">COUNCIL</div>
            <div style="font-size: 13px; font-weight: 700; color: var(--accent-cyan);">${item.council}</div>
          </div>
        `;
      }

      modal.style.display = 'flex';
    }

    modalClose.addEventListener('click', () => { modal.style.display = 'none'; });
    window.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });

    const connectWalletBtn = document.getElementById('connectWalletBtn');
    async function checkWallet() {
      if (typeof window.ethereum !== 'undefined') {
        try {
          const accounts = await window.ethereum.request({ method: 'eth_accounts' });
          if (accounts && accounts.length > 0) handleAccount(accounts[0]);
        } catch (e) {}
      }
    }
    function handleAccount(account) {
      const shortAddr = account.slice(0, 6) + '...' + account.slice(-4);
      const isBrave = window.ethereum?.isBraveWallet;
      connectWalletBtn.innerHTML = `${isBrave ? '🦁' : '🦊'} ${shortAddr}`;
      connectWalletBtn.style.background = 'linear-gradient(135deg, #00f0ff, #0077b6)';
      connectWalletBtn.style.color = '#04050a';
    }
    connectWalletBtn.addEventListener('click', async () => {
      if (typeof window.ethereum !== 'undefined') {
        try {
          const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
          if (accounts && accounts.length > 0) handleAccount(accounts[0]);
        } catch (err) {
          alert('Wallet connection rejected: ' + err.message);
        }
      } else {
        alert('Please enable Brave Wallet or install MetaMask.');
      }
    });

    checkWallet();
  </script>
</body>
</html>"""

with open(docs_dir / "index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

with open(Path(r"C:\02_QUILLAN\index.html"), "w", encoding="utf-8") as f:
    f.write(html_content)

print("[SUCCESS] Grand Sovereign Portal with Cinematic 3D Scene rendered!")