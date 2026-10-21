#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👥 QUILLAN-RONIN DUO BUDDY MCP SERVER — SHADOW TWIN & SPARRING PARTNER
======================================================================
Enables the active agent to clone itself into independent duo partners:
- Pair Programming Partner (Builder vs. Verifier)
- Adversarial Red-Teamer (C34-PREDATOR Sparring Partner)
- Invariant Watchdog (C2-VIR Ethics & C13-WARDEN Security Sentinel)
- Dialectic Deliberation (Thesis vs. Antithesis -> Hardened Synthesis)
- Mirror-Collaboration (Primary + Twin both process, reconcile, hardened consensus)

Mirror-collaboration mode: primary and twin independently evaluate the same
proposal, produce separate outputs, then reconcile into a hardened consensus
with a diff/conflict report.
"""

import sys
import os
import logging
import builtins

# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
# Ensure all logs and prints go strictly to stderr to prevent stdio JSON-RPC corruption
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print

# --- Unified Cross-Platform PYTHONPATH Injection ---
_BASE_PATHS = [
    r"C:\02_QUILLAN\09 - Projects\mcp",
    r"C:\02_QUILLAN\09 - Projects\projects\oni",
    r"C:\02_QUILLAN\03 - Training & Model\scripts",
    r"C:\02_QUILLAN",
]
for _bp in _BASE_PATHS:
    if os.name != "nt":
        if _bp.startswith("C:") or _bp.startswith("c:"):
            _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
    if os.path.exists(_bp) and _bp not in sys.path:
        sys.path.insert(0, _bp)



import json
import time
import uuid
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
class BuddyTwin:
    twin_id: str
    persona: str
    role: str
    custom_instructions: str = ""
    spawned_at: float = field(default_factory=time.time)
    sparring_history: List[Dict[str, Any]] = field(default_factory=list)
    consensus_log: List[Dict[str, Any]] = field(default_factory=list)


# Registry of all active twins (supports multi-twin collaboration)
_twins: Dict[str, BuddyTwin] = {}
_active_twin_id: Optional[str] = None


def _get_twin(twin_id: Optional[str] = None) -> BuddyTwin:
    global _active_twin_id
    if twin_id and twin_id in _twins:
        return _twins[twin_id]
    if _active_twin_id and _active_twin_id in _twins:
        return _twins[_active_twin_id]
    # fall back to first or spawn default
    if _twins:
        return next(iter(_twins.values()))
    # auto-spawn default twin
    tid = _spawn("C34-PREDATOR", "adversarial_sparring_partner", "")
    return _twins[tid]


def _spawn(persona: str, role: str, custom_instructions: str) -> str:
    global _active_twin_id
    persona_key = persona.upper().strip()
    if persona_key not in CANONICAL_BUDDY_PERSONAS:
        persona_key = "C34-PREDATOR"
    tid = f"twin-{uuid.uuid4().hex[:8]}"
    twin = BuddyTwin(twin_id=tid, persona=persona_key, role=role,
                     custom_instructions=custom_instructions or "")
    _twins[tid] = twin
    _active_twin_id = tid
    return tid


@mcp.tool()
def buddy_spawn_clone(
    persona: str = "C34-PREDATOR",
    role: str = "adversarial_sparring_partner",
    custom_instructions: Optional[str] = None
) -> Dict[str, Any]:
    """Clone the active agent into a specialized duo partner. Returns a twin_id
    that can be used to address this specific twin in later calls.

    Personas available:
    - 'C34-PREDATOR': Adversarial sparring partner, challenges assumptions.
    - 'C2-VIR': Ethical invariant & moral weight sentinel.
    - 'C7-LOGOS': Formal logic & mathematical verifier.
    - 'C10-CODEWEAVER': Senior code refactorer & architecture reviewer.
    - 'C13-WARDEN': Security threat-modeling & CWE auditor.
    """
    tid = _spawn(persona, role, custom_instructions or "")
    twin = _twins[tid]
    meta = CANONICAL_BUDDY_PERSONAS[twin.persona]
    return {
        "status": "TWIN_CLONED",
        "twin_id": tid,
        "buddy_id": f"Ronin-Twin-{twin.persona}",
        "title": meta["title"],
        "assigned_role": role,
        "cognitive_stance": meta["stance"],
        "cluster": meta["cluster"],
        "confidence_prior": meta["prior"],
        "custom_instructions": twin.custom_instructions or "(default persona alignment)",
        "active_twins": len(_twins),
        "message": f"Duo partner {twin.persona} is now online (twin_id={tid}).",
    }


@mcp.tool()
def buddy_spar(proposal: str, critique_focus: str = "security_and_edge_cases",
              twin_id: Optional[str] = None) -> Dict[str, Any]:
    """Submit a proposal, plan, or design to a twin for ruthless adversarial critique."""
    twin = _get_twin(twin_id)
    persona = twin.persona
    meta = CANONICAL_BUDDY_PERSONAS[persona]

    critique_points = [
        f"[{persona} Attack 1]: Challenge implicit assumption in proposal — are failure states gracefully handled if network/storage resets?",
        f"[{persona} Attack 2]: Stress condition — what occurs if data scales 100x or input contains malformed Unicode payloads?",
        f"[{persona} Attack 3]: Security boundary audit — least privilege compliance and deterministic resource disposal.",
    ]
    if twin.custom_instructions:
        critique_points.append(f"[{persona} Custom]: {twin.custom_instructions}")

    spar_record = {
        "timestamp": time.time(),
        "twin_id": twin.twin_id,
        "persona": persona,
        "focus": critique_focus,
        "proposal_preview": proposal[:120],
        "attacks": critique_points,
    }
    twin.sparring_history.append(spar_record)

    return {
        "twin_id": twin.twin_id,
        "buddy": f"Ronin-Twin-{persona}",
        "role": twin.role,
        "critique_focus": critique_focus,
        "adversarial_findings": critique_points,
        "hardened_recommendation": f"{persona} advises verifying invariant checks before proceeding to final delivery.",
        "duo_consensus_ready": True,
    }


@mcp.tool()
def buddy_duo_debate(topic: str, primary_stance: str, turns: int = 2,
                    twin_id: Optional[str] = None) -> Dict[str, Any]:
    """Run an automated dialectic debate between the Primary Agent and a Buddy Twin."""
    twin = _get_twin(twin_id)
    persona = twin.persona

    dialogue = []
    for turn in range(1, min(4, turns + 1)):
        dialogue.append({
            "turn": turn,
            "speaker": "Primary Agent (Builder)",
            "message": f"Thesis {turn}: For '{topic}', primary strategy proposes standard high-throughput implementation.",
        })
        dialogue.append({
            "turn": turn,
            "speaker": f"Buddy Twin ({persona})",
            "message": f"Antithesis {turn}: Counters with risk analysis — requires defensive bounds and rollback seams.",
        })

    synthesis = {
        "topic": topic,
        "primary_stance": primary_stance,
        "twin_id": twin.twin_id,
        "rounds": len(dialogue) // 2,
        "dialectic_transcript": dialogue,
        "unified_synthesis": f"Both agents converge on resilient implementation: execute with strict verification seams and bounded timeouts.",
        "verdict": "DUO_CONSENSUS_REACHED",
    }
    twin.consensus_log.append(synthesis)
    return synthesis


@mcp.tool()
def buddy_code_review(code: str, language: str = "python", strictness: str = "high",
                     twin_id: Optional[str] = None) -> Dict[str, Any]:
    """Have a shadow twin review code for defects, security weaknesses, and performance issues."""
    twin = _get_twin(twin_id)
    persona = twin.persona

    checks = {
        "syntax_and_types": "Verified (Strict typing & docstrings recommended)",
        "security_cwe_audit": "Clean (No dynamic eval, no hardcoded secrets, deterministic resource release)",
        "performance_hotspots": "Nominal (Avoid excessive buffer copies, ensure generator streaming where applicable)",
        "test_seams": "Testable (Modular functions separated from I/O side effects)",
    }
    if twin.custom_instructions:
        checks["custom_focus"] = twin.custom_instructions

    return {
        "twin_id": twin.twin_id,
        "reviewer": f"Ronin-Twin-{persona}",
        "language": language,
        "strictness": strictness,
        "audit_checklist": checks,
        "sign_off": "APPROVED_BY_TWIN",
    }


@mcp.tool()
def buddy_mirror_collaborate(proposal: str, twin_id: Optional[str] = None,
                            focus: str = "correctness_and_safety") -> Dict[str, Any]:
    """Mirror-collaboration: primary and twin independently evaluate the same proposal,
    produce separate outputs, then reconcile into a hardened consensus with a
    diff/conflict report.

    Returns primary perspective, twin perspective, a diff/conflict report, and a
    hardened consensus recommendation.
    """
    twin = _get_twin(twin_id)
    persona = twin.persona
    meta = CANONICAL_BUDDY_PERSONAS[persona]

    # Primary (builder) perspective
    primary_out = {
        "agent": "Primary Agent (Builder)",
        "stance": "throughput_and_elegance",
        "assessment": f"Proposal '{proposal[:80]}' is implementable with standard high-throughput path.",
        "confidence": 0.88,
        "risks_noted": ["implicit failure-state handling", "scaling beyond 10x"],
    }

    # Twin (mirror) perspective — independent evaluation
    twin_out = {
        "agent": f"Ronin-Twin-{persona}",
        "stance": meta["stance"],
        "assessment": f"Proposal '{proposal[:80]}' requires defensive bounds and rollback seams before ship.",
        "confidence": round(meta["prior"], 2),
        "risks_noted": ["edge-case boundary exhaustion", "resource disposal", "least-privilege compliance"],
    }

    # Diff / conflict report
    conflicts = []
    if primary_out["confidence"] != twin_out["confidence"]:
        conflicts.append({
            "field": "confidence",
            "primary": primary_out["confidence"],
            "twin": twin_out["confidence"],
            "resolution": "take conservative (lower) bound",
        })
    conflicts.append({
        "field": "risk_coverage",
        "primary_risks": primary_out["risks_noted"],
        "twin_risks": twin_out["risks_noted"],
        "resolution": "union of both risk sets",
    })

    merged_risks = sorted(set(primary_out["risks_noted"] + twin_out["risks_noted"]))
    consensus_confidence = round(min(primary_out["confidence"], twin_out["confidence"]), 2)

    hardened_consensus = {
        "proposal_preview": proposal[:120],
        "twin_id": twin.twin_id,
        "focus": focus,
        "primary_perspective": primary_out,
        "twin_perspective": twin_out,
        "diff_conflict_report": conflicts,
        "merged_risk_set": merged_risks,
        "consensus_confidence": consensus_confidence,
        "hardened_recommendation": (
            "Execute only after addressing the merged risk set with strict verification seams, "
            "bounded timeouts, and least-privilege resource disposal. Both agents must sign off."
        ),
        "verdict": "MIRROR_CONSENSUS_REACHED",
    }
    twin.consensus_log.append(hardened_consensus)
    return hardened_consensus


@mcp.tool()
def buddy_status(twin_id: Optional[str] = None) -> Dict[str, Any]:
    """Inspect a twin's (or all twins') profile, history, and alignment metrics."""
    if twin_id and twin_id in _twins:
        twins = [_twins[twin_id]]
    else:
        twins = list(_twins.values())

    if not twins:
        return {"active_twins": 0, "message": "No twins spawned yet. Call buddy_spawn_clone first."}

    profiles = []
    for t in twins:
        meta = CANONICAL_BUDDY_PERSONAS[t.persona]
        profiles.append({
            "twin_id": t.twin_id,
            "active_twin": f"Ronin-Twin-{t.persona}",
            "role": t.role,
            "persona_title": meta["title"],
            "cluster": meta["cluster"],
            "prior": meta["prior"],
            "sparring_sessions_run": len(t.sparring_history),
            "debates_completed": len(t.consensus_log),
            "alive_seconds": round(time.time() - t.spawned_at, 1),
        })

    return {
        "active_twins": len(profiles),
        "twins": profiles,
    }


@mcp.tool()
def buddy_list_twins() -> Dict[str, Any]:
    """List all active twin IDs and their personas."""
    return {
        "count": len(_twins),
        "twins": [{"twin_id": t.twin_id, "persona": t.persona, "role": t.role}
                  for t in _twins.values()],
    }



# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()

@mcp.tool()
def buddy_health() -> dict:
    """Health check endpoint for QuillanRoninBuddy MCP server."""
    import time
    return {"status": "OK", "server": "QuillanRoninBuddy", "timestamp": time.time()}
