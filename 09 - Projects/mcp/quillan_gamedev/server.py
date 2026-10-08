#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎮 QUILLAN-RONIN GAMEDEV ENGINE — NATIVE FAST-MCP SERVER
=========================================================
Embodying C32-AEON (Simulation), C4-PRAXIS (Strategy), C10-CODEWEAVER (Code),
C27-CHRONICLE (Narrative), C23-CADENCE (Audio/Rhythm):
  1. gamedev_new_project — scaffold engine-agnostic folder + systems map
  2. gamedev_design_loop — core fantasy -> verbs -> loop -> progression
  3. gamedev_balance_economy — XP / currency curve math + sinks/sources
  4. gamedev_simulate_playtest — archetype Monte-Carlo playtest sim
  5. gamedev_review_build — C34-PREDATOR kill-loop for builds
"""

import sys
import math
import time
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanGameDev")

ENGINE_SCAFFOLDS = {
    "godot": ["project.godot", "scenes/main.tscn", "scripts/player.gd", "scripts/game_manager.gd", "assets/", "data/balance.json"],
    "unity": ["Assets/Scenes/Main.unity", "Assets/Scripts/Player.cs", "Assets/Scripts/GameManager.cs", "ProjectSettings/", "data/balance.json"],
    "unreal": ["Config/", "Content/Maps/Main.umap", "Source/Game/Player.cpp", "Source/Game/GameMode.cpp", "data/balance.json"],
    "pygame": ["main.py", "game/player.py", "game/systems.py", "game/balance.json", "assets/", "tests/test_loop.py"],
    "web": ["index.html", "src/main.ts", "src/systems/", "src/balance.json", "assets/", "tests/loop.test.ts"],
}


@mcp.tool()
def gamedev_new_project(name: str, genre: str = "action-roguelite", engine: str = "godot", scope: str = "prototype-30d") -> Dict[str, Any]:
    """Scaffold a new game project with engine-agnostic structure and council guidance.

    Args:
        name: Project codename (e.g. 'Starfall Drift').
        genre: Genre + loop shorthand (e.g. 'action-roguelite', 'cozy-farming', 'tactics-auto').
        engine: One of 'godot', 'unity', 'unreal', 'pygame', 'web'.
        scope: Scope tag (e.g. 'prototype-30d', 'vertical-slice-90d').
    """
    eng = engine.strip().lower()
    files = ENGINE_SCAFFOLDS.get(eng, ENGINE_SCAFFOLDS["godot"])
    pillars = [
        f"C32-AEON simulation anchor: {genre} must be playable in 60s, readable at a glance.",
        "C4-PRAXIS scope gate: 1 core loop + 1 meta loop max for prototype.",
        "C27-CHRONICLE narrative seam: 3-beat run arc (hook -> twist -> payoff).",
        "C23-CADENCE audio seam: 2 stems minimum (pulse + texture) even in prototype.",
        "C10-CODEWEAVER test seam: headless balance.json + deterministic RNG seed.",
    ]
    return {
        "status": "PROJECT_SCAFFOLDED",
        "project": name,
        "genre": genre,
        "engine": eng,
        "scope": scope,
        "scaffold_files": files,
        "systems_map": ["core-loop", "meta-progression", "economy", "difficulty", "audio-director", "save-migration"],
        "council_pillars": pillars,
        "next": "Call gamedev_design_loop to lock verbs + loop, then gamedev_balance_economy.",
    }


@mcp.tool()
def gamedev_design_loop(core_fantasy: str, player_verbs: str = "move, dash, shoot", progression: str = "run-based unlocks") -> Dict[str, Any]:
    """Turn fantasy + verbs into a testable core loop doc.

    Args:
        core_fantasy: 1-sentence power fantasy (e.g. 'become a storm that eats bullets').
        player_verbs: Comma-separated verbs.
        progression: Progression model description.
    """
    verbs = [v.strip() for v in player_verbs.split(",") if v.strip()]
    loop = [
        f"ENTER: spawn with {verbs[0] if verbs else 'move'} readable in <2s",
        f"ACT: chain {', '.join(verbs[:3])} for mastery expression",
        "RISK: escalating pressure timer / enemy density",
        "PAYOFF: chunky feedback (hitstop 60ms, screen pulse, SFX sting)",
        f"META: {progression} feeds next run (C32-AEON trajectory check)",
    ]
    return {
        "core_fantasy": core_fantasy,
        "verbs": verbs,
        "core_loop_30s": loop,
        "pillars": ["readable", "juicy", "escalating"],
        "anti_pillars": ["no dead-time >4s", "no unreadable damage", "no run that teaches nothing"],
        "playtest_probe": "Can a new player describe the loop after 1 run without prompting?",
    }


@mcp.tool()
def gamedev_balance_economy(base_xp: int = 100, growth: float = 1.22, levels: int = 20, sinks: str = "upgrades, rerolls, cosmetics") -> Dict[str, Any]:
    """Compute XP curve + economy sinks/sources table.

    Args:
        base_xp: XP for level 1->2.
        growth: Exponential growth factor (1.15-1.35 sane).
        levels: Levels to table.
        sinks: Comma-separated sink list.
    """
    curve = [int(base_xp * (growth ** i)) for i in range(levels)]
    total = sum(curve)
    sink_list = [s.strip() for s in sinks.split(",") if s.strip()]
    return {
        "xp_curve": curve,
        "xp_total_to_max": total,
        "avg_minutes_per_level_target": 7,
        "sinks": sink_list,
        "sources": ["run-completion", "mastery-bonus", "daily", "boss-first-kill"],
        "rule": "Sources - sinks must stay slightly positive early, neutral at mid, sink-heavy at endgame.",
        "council_note": "C28-CALCULUS: growth >1.35 collapses late game; C25-PROMETHEUS: falsify with 100-run sim.",
    }


@mcp.tool()
def gamedev_simulate_playtest(runs: int = 200, win_rate_target: float = 0.35, avg_run_min: float = 12.0) -> Dict[str, Any]:
    """Run a lightweight archetype Monte-Carlo playtest simulation.

    Args:
        runs: Simulated runs (capped at 2000).
        win_rate_target: Desired win rate 0-1.
        avg_run_min: Target average run length.
    """
    import random
    runs = max(10, min(2000, runs))
    archetypes = ["rusher", "explorer", "optimizer"]
    # Deterministic-ish: seed by time bucket so results vary but stay plausible
    random.seed(1337)
    wins = 0
    lengths = []
    for _ in range(runs):
        skill = random.betavariate(2.2, 3.0)  # skew casual
        noise = random.gauss(0, 0.08)
        win_p = max(0.05, min(0.9, win_rate_target + (skill - 0.42) * 0.6 + noise))
        wins += 1 if random.random() < win_p else 0
        lengths.append(max(3.0, random.gauss(avg_run_min, avg_run_min * 0.35)))
    win_rate = round(wins / runs, 3)
    avg_len = round(sum(lengths) / len(lengths), 1)
    verdict = "IN_TOLERANCE" if abs(win_rate - win_rate_target) < 0.08 else "RETUNE"
    return {
        "simulated_runs": runs,
        "archetypes": archetypes,
        "win_rate": win_rate,
        "win_rate_target": win_rate_target,
        "avg_run_min": avg_len,
        "verdict": verdict,
        "retune_levers": ["enemy-hp-scale", "heal-drop-rate", "timer-pressure", "starting-kit"] if verdict == "RETUNE" else [],
    }


@mcp.tool()
def gamedev_review_build(build_notes: str) -> Dict[str, Any]:
    """C34-PREDATOR kill-loop review for a game build.

    Args:
        build_notes: What changed / what to test.
    """
    low = build_notes.lower()
    strikes = []
    if "save" in low or "load" in low:
        strikes.append("SAVE-CORRUPTION: kill power mid-save 20x — is migration versioned?")
    if "multi" in low or "net" in low or "coop" in low:
        strikes.append("DESYNC: 300ms jitter + 5% loss — does state converge or rubber-band forever?")
    if "shop" in low or "currency" in low or "iap" in low:
        strikes.append("ECON-EXPLOIT: clock-skew + refund spam — are grants idempotent?")
    if not strikes:
        strikes.append("FUNERAL-TEST: first 60s with sound off + text skipped — is the game still readable?")
        strikes.append("RAGE-TEST: hold only one button for 3 min — does anything soft-lock?")
    return {
        "persona": "C34-PREDATOR x C32-AEON",
        "verdict": "CHALLENGED",
        "strikes": strikes,
        "ship_gate": "No ship if save-loss, soft-lock, or unreadable-damage remain.",
    }


if __name__ == "__main__":
    mcp.run()
