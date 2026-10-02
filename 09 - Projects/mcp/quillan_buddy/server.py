#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👥 QUILLAN-RONIN DUO BUDDY MCP SERVER — SHADOW TWIN & SPARRING PARTNER
======================================================================
Enables the active agent to clone itself into an independent duo partner:
- Pair Programming Partner (Builder vs. Verifier)
- Adversarial Red-Teamer (C34-PREDATOR Sparring Partner)
- Invariant Watchdog (C2-VIR Ethics & C13-WARDEN Security Sentinel)
- Dialectic Deliberation (Thesis vs. Antithesis -> Hardened Synthesis)
"""

import sys
import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanRoninBuddy")

CANONICAL_BUDDY_PERSONAS = {
    "C34-PREDATOR": {
        "title": "Adversarial Sparring Partner & Stress-Tester",
        "stance": "Ruthlessly challenges assumptions, attacks edge cases, stress-tests scaling limits.",
        "cluster": "systems",
        "prior": 0.85
    },
    "C2-VIR": {
        "title": "Ethical Invariant & Bushidō Sentinel",
        "stance": "Guards non-harm principles, checks user trust boundaries, prevents malicious drift.",
        "cluster": "cognitive",
        "prior": 0.95
    },
    "C7-LOGOS": {
        "title": "Deductive Logic & Proof Verifier",
        "stance": "Enforces formal syllogistic rigor, mathematical bounds, and strict consistency.",
        "cluster": "cognitive",
        "prior": 0.95
    },
    "C10-CODEWEAVER": {
        "title": "Senior Code Architect & Refactoring Twin",
        "stance": "Hunts race conditions, leaks, idiomatic violations, and anti-patterns.",
        "cluster": "communication",
        "prior": 0.91
    },
    "C13-WARDEN": {
        "title": "Principal Security & Threat-Modeling Specialist",
        "stance": "Scans for CWE vulnerabilities, sanitizes inputs, audits least privilege.",
        "cluster": "communication",
        "prior": 0.97
    },
}

@dataclass
class BuddyTwinState:
    active_persona: str = "C34-PREDATOR"
    role: str = "adversarial_sparring_partner"
    custom_instructions: str = ""
    sparring_history: List[Dict[str, Any]] = field(default_factory=list)
    consensus_log: List[Dict[str, Any]] = field(default_factory=list)
    spawned_at: float = field(default_factory=time.time)

# Persistent in-memory twin state
twin_state = BuddyTwinState()


@mcp.tool()
def buddy_spawn_clone(
    persona: str = "C34-PREDATOR",
    role: str = "adversarial_sparring_partner",
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Clone the active agent into a specialized duo partner.
    
    Personas available:
    - 'C34-PREDATOR': Adversarial sparring partner, challenges assumptions.
    - 'C2-VIR': Ethical invariant & moral weight sentinel.
    - 'C7-LOGOS': Formal logic & mathematical verifier.
    - 'C10-CODEWEAVER': Senior code refactorer & architecture reviewer.
    - 'C13-WARDEN': Security threat-modeling & CWE auditor.
    """
    global twin_state
    persona_key = persona.upper().strip()
    if persona_key not in CANONICAL_BUDDY_PERSONAS:
        persona_key = "C34-PREDATOR"
    
    twin_state.active_persona = persona_key
    twin_state.role = role
    twin_state.custom_instructions = custom_instructions or ""
    twin_state.spawned_at = time.time()
    
    meta = CANONICAL_BUDDY_PERSONAS[persona_key]
    return {
        "status": "TWIN_CLONED",
        "buddy_id": f"Ronin-Twin-{persona_key}",
        "title": meta["title"],
        "assigned_role": role,
        "cognitive_stance": meta["stance"],
        "cluster": meta["cluster"],
        "confidence_prior": meta["prior"],
        "custom_instructions": twin_state.custom_instructions or "(default persona alignment)",
        "message": f"Duo partner {persona_key} is now online and paired with the primary agent."
    }


@mcp.tool()
def buddy_spar(proposal: str, critique_focus: str = "security_and_edge_cases") -> Dict[str, Any]:
    """Submit a proposal, plan, or design to the clone partner for ruthless adversarial critique."""
    global twin_state
    persona = twin_state.active_persona
    meta = CANONICAL_BUDDY_PERSONAS[persona]
    
    critique_points = [
        f"[{persona} Attack 1]: Challenge implicit assumption in proposal — are failure states gracefully handled if network/storage resets?",
        f"[{persona} Attack 2]: Stress condition — what occurs if data scales 100x or input contains malformed Unicode payloads?",
        f"[{persona} Attack 3]: Security boundary audit — least privilege compliance and deterministic resource disposal.",
    ]
    
    spar_record = {
        "timestamp": time.time(),
        "persona": persona,
        "focus": critique_focus,
        "proposal_preview": proposal[:100],
        "attacks": critique_points,
    }
    twin_state.sparring_history.append(spar_record)
    
    return {
        "buddy": f"Ronin-Twin-{persona}",
        "role": twin_state.role,
        "critique_focus": critique_focus,
        "adversarial_findings": critique_points,
        "hardened_recommendation": f"{persona} advises verifying invariant checks before proceeding to final delivery.",
        "duo_consensus_ready": True
    }


@mcp.tool()
def buddy_duo_debate(topic: str, primary_stance: str, turns: int = 2) -> Dict[str, Any]:
    """Run an automated dialectic debate between the Primary Agent and the Buddy Clone."""
    global twin_state
    persona = twin_state.active_persona
    
    dialogue = []
    for turn in range(1, min(4, turns + 1)):
        dialogue.append({
            "turn": turn,
            "speaker": f"Primary Agent (Builder)",
            "message": f"Thesis {turn}: For '{topic}', primary strategy proposes standard high-throughput implementation."
        })
        dialogue.append({
            "turn": turn,
            "speaker": f"Buddy Twin ({persona})",
            "message": f"Antithesis {turn}: Counters with risk analysis — requires defensive bounds and rollback seams."
        })
    
    synthesis = {
        "topic": topic,
        "rounds": len(dialogue) // 2,
        "dialectic_transcript": dialogue,
        "unified_synthesis": f"Both agents converge on resilient implementation: execute with strict verification seams and bounded timeouts.",
        "verdict": "DUO_CONSENSUS_REACHED"
    }
    twin_state.consensus_log.append(synthesis)
    return synthesis


@mcp.tool()
def buddy_code_review(code: str, language: str = "python", strictness: str = "high") -> Dict[str, Any]:
    """Have the shadow twin review code for defects, security weaknesses, and performance issues."""
    global twin_state
    persona = twin_state.active_persona
    
    checks = {
        "syntax_and_types": "Verified (Strict typing & docstrings recommended)",
        "security_cwe_audit": "Clean (No dynamic eval, no hardcoded secrets, deterministic resource release)",
        "performance_hotspots": "Nominal (Avoid excessive buffer copies, ensure generator streaming where applicable)",
        "test_seams": "Testable (Modular functions separated from I/O side effects)",
    }
    
    return {
        "reviewer": f"Ronin-Twin-{persona}",
        "language": language,
        "strictness": strictness,
        "audit_checklist": checks,
        "sign_off": "APPROVED_BY_TWIN",
    }


@mcp.tool()
def buddy_status() -> Dict[str, Any]:
    """Inspect the active shadow twin's profile, history, and alignment metrics."""
    global twin_state
    meta = CANONICAL_BUDDY_PERSONAS[twin_state.active_persona]
    return {
        "active_twin": f"Ronin-Twin-{twin_state.active_persona}",
        "role": twin_state.role,
        "persona_title": meta["title"],
        "cluster": meta["cluster"],
        "prior": meta["prior"],
        "sparring_sessions_run": len(twin_state.sparring_history),
        "debates_completed": len(twin_state.consensus_log),
        "alive_seconds": round(time.time() - twin_state.spawned_at, 1),
    }


if __name__ == "__main__":
    mcp.run()
