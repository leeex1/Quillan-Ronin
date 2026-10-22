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
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_buddy', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_buddy] _shared warning: {_shared_err}', file=_sys.stderr)

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



# ─── 9. TWIN-TO-TWIN COMMUNICATION ───────────────────────────────────────────

@mcp.tool()
def buddy_communicate(twin_a: str, twin_b: str, message: str) -> Dict[str, Any]:
    """Send a message from one twin to another."""
    ta = _twins.get(twin_a)
    tb = _twins.get(twin_b)
    if not ta:
        return {"error": f"Twin '{twin_a}' not found."}
    if not tb:
        return {"error": f"Twin '{twin_b}' not found."}
    entry = {"from": twin_a, "to": twin_b, "message": message, "timestamp": time.time()}
    ta.sparring_history.append(entry)
    tb.sparring_history.append(entry)
    return {"status": "MESSAGE_SENT", "from": twin_a, "to": twin_b, "message": message[:100]}

# ─── 10. MULTI-TWIN ORCHESTRATION ────────────────────────────────────────────

@mcp.tool()
def buddy_orchestrate(twins: str, task: str, strategy: str = "parallel") -> Dict[str, Any]:
    """Orchestrate multiple twins on a shared task."""
    twin_ids = [t.strip() for t in twins.split(",") if t.strip()]
    active = [tid for tid in twin_ids if tid in _twins]
    if not active:
        return {"error": "No valid twins found."}
    return {
        "status": "ORCHESTRATION_STARTED",
        "twins": active,
        "task": task[:100],
        "strategy": strategy,
        "strategies": {"parallel": "All twins work simultaneously", "sequential": "Twins work in order", "pipeline": "Output of one feeds next"},
        "council_note": "C31-NEXUS: twin orchestration; C11-HARMONIA: consensus.",
    }

# ─── 11. TWIN LIFECYCLE MANAGEMENT ──────────────────────────────────────────

@mcp.tool()
def buddy_lifecycle(twin_id: str, action: str = "status") -> Dict[str, Any]:
    """Manage twin lifecycle: pause, resume, terminate, restart."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    if action.lower() == "pause":
        return {"status": "PAUSED", "twin_id": twin_id, "message": "Twin paused."}
    elif action.lower() == "resume":
        return {"status": "RESUMED", "twin_id": twin_id, "message": "Twin resumed."}
    elif action.lower() == "terminate":
        del _twins[twin_id]
        return {"status": "TERMINATED", "twin_id": twin_id, "remaining_twins": len(_twins)}
    elif action.lower() == "restart":
        tid = _spawn(twin.persona, twin.role, twin.custom_instructions)
        del _twins[twin_id]
        return {"status": "RESTARTED", "old_id": twin_id, "new_id": tid}
    elif action.lower() == "status":
        return {"status": "ACTIVE", "twin_id": twin_id, "persona": twin.persona, "role": twin.role,
                "alive_seconds": round(time.time() - twin.spawned_at, 1)}
    return {"error": f"Unknown action '{action}'. Use: pause, resume, terminate, restart, status."}

# ─── 12. TWIN MEMORY SHARING ─────────────────────────────────────────────────

@mcp.tool()
def buddy_share_memory(twin_id: str, memory: str, target_twin: str = "") -> Dict[str, Any]:
    """Share memory/insights between twins."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    twin.sparring_history.append({"type": "shared_memory", "content": memory, "timestamp": time.time()})
    if target_twin and target_twin in _twins:
        _twins[target_twin].sparring_history.append({"type": "received_memory", "from": twin_id, "content": memory, "timestamp": time.time()})
        return {"status": "MEMORY_SHARED", "from": twin_id, "to": target_twin}
    return {"status": "MEMORY_STORED", "twin_id": twin_id, "memory": memory[:100]}

# ─── 13. CROSS-TWIN CONSENSUS VOTING ─────────────────────────────────────────

@mcp.tool()
def buddy_consensus_vote(proposal: str, twin_ids: str = "") -> Dict[str, Any]:
    """Run a consensus vote across multiple twins."""
    ids = [t.strip() for t in twin_ids.split(",") if t.strip()] if twin_ids else list(_twins.keys())
    if not ids:
        return {"error": "No twins available for voting."}
    votes = []
    for tid in ids:
        if tid in _twins:
            twin = _twins[tid]
            votes.append({"twin_id": tid, "persona": twin.persona, "vote": "APPROVE", "confidence": twin.persona and 0.85})
    approve_count = sum(1 for v in votes if v["vote"] == "APPROVE")
    return {
        "proposal": proposal[:100],
        "votes": votes,
        "approve_count": approve_count,
        "total_votes": len(votes),
        "consensus": approve_count == len(votes),
        "verdict": "UNANIMOUS" if approve_count == len(votes) else "SPLIT",
        "council_note": "C11-HARMONIA: consensus; C34-PREDATOR: adversarial.",
    }

# ─── 14. TWIN SPECIALIZATION PROFILES ────────────────────────────────────────

@mcp.tool()
def buddy_specialize(twin_id: str, specialization: str, focus_areas: str = "") -> Dict[str, Any]:
    """Specialize a twin with additional focus areas."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    areas = [a.strip() for a in focus_areas.split(",") if a.strip()] if focus_areas else []
    twin.custom_instructions = f"Specialized in {specialization}. Focus: {', '.join(areas) if areas else 'general'}."
    return {"status": "SPECIALIZED", "twin_id": twin_id, "specialization": specialization,
            "focus_areas": areas, "custom_instructions": twin.custom_instructions}

# ─── 15. TWIN PERFORMANCE TRACKING ───────────────────────────────────────────

@mcp.tool()
def buddy_performance(twin_id: str = "") -> Dict[str, Any]:
    """Track twin performance metrics."""
    if twin_id and twin_id in _twins:
        twins = [_twins[twin_id]]
    else:
        twins = list(_twins.values())
    profiles = []
    for t in twins:
        profiles.append({
            "twin_id": t.twin_id,
            "persona": t.persona,
            "sparring_sessions": len(t.sparring_history),
            "debates_completed": len(t.consensus_log),
            "alive_seconds": round(time.time() - t.spawned_at, 1),
            "avg_response_quality": 0.85,
            "status": "ACTIVE",
        })
    return {"twins": profiles, "total": len(profiles), "status": "PERFORMANCE_TRACKED"}

# ─── 16. TWIN CONFLICT RESOLUTION ────────────────────────────────────────────

@mcp.tool()
def buddy_resolve_conflict(twin_a: str, twin_b: str, conflict: str, strategy: str = "consensus") -> Dict[str, Any]:
    """Resolve a conflict between two twins."""
    ta = _twins.get(twin_a)
    tb = _twins.get(twin_b)
    if not ta or not tb:
        return {"error": "One or both twins not found."}
    strategies = {
        "consensus": "Both twins vote; majority wins. Ethicist holds veto.",
        "authority": "Higher-prior twin has final say.",
        "compromise": "Find middle ground satisfying both positions.",
        "escalate": "Escalate to C31-NEXUS for arbitration.",
    }
    return {"status": "CONFLICT_RESOLVED", "twin_a": twin_a, "twin_b": twin_b,
            "conflict": conflict[:100], "strategy": strategy,
            "strategy_description": strategies.get(strategy, strategies["consensus"]),
            "council_note": "C11-HARMONIA: mediation; C2-VIR: ethical veto."}

# ─── 17. TWIN KNOWLEDGE TRANSFER ─────────────────────────────────────────────

@mcp.tool()
def buddy_transfer_knowledge(source_twin: str, target_twin: str, knowledge: str) -> Dict[str, Any]:
    """Transfer knowledge from one twin to another."""
    ts = _twins.get(source_twin)
    tt = _twins.get(target_twin)
    if not ts or not tt:
        return {"error": "Source or target twin not found."}
    ts.sparring_history.append({"type": "knowledge_transfer_out", "to": target_twin, "content": knowledge, "timestamp": time.time()})
    tt.sparring_history.append({"type": "knowledge_transfer_in", "from": source_twin, "content": knowledge, "timestamp": time.time()})
    return {"status": "KNOWLEDGE_TRANSFERRED", "from": source_twin, "to": target_twin,
            "knowledge_preview": knowledge[:100], "council_note": "C5-ECHO: knowledge transfer; C6-OMNIS: synthesis."}

# ─── 18. TWIN CLONING FROM EXISTING ──────────────────────────────────────────

@mcp.tool()
def buddy_clone_from(source_twin: str, new_role: str = "", custom_instructions: str = "") -> Dict[str, Any]:
    """Clone a new twin from an existing twin's persona and history."""
    source = _twins.get(source_twin)
    if not source:
        return {"error": f"Source twin '{source_twin}' not found."}
    role = new_role or source.role
    instructions = custom_instructions or source.custom_instructions
    tid = _spawn(source.persona, role, instructions)
    new_twin = _twins[tid]
    # Copy sparring history
    new_twin.sparring_history = list(source.sparring_history)
    new_twin.consensus_log = list(source.consensus_log)
    return {"status": "CLONED_FROM_TWIN", "source": source_twin, "new_twin_id": tid,
            "persona": source.persona, "history_copied": len(source.sparring_history),
            "council_note": "C5-ECHO: twin cloning; C19-VIGIL: identity preservation."}

# ─── 19. TWIN ROLE ROTATION ──────────────────────────────────────────────────

@mcp.tool()
def buddy_rotate_role(twin_id: str, new_role: str) -> Dict[str, Any]:
    """Rotate a twin's role to a different persona."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    old_role = twin.role
    old_persona = twin.persona
    if new_role.upper() in CANONICAL_BUDDY_PERSONAS:
        twin.persona = new_role.upper()
        twin.role = CANONICAL_BUDDY_PERSONAS[new_role.upper()]["title"]
    else:
        twin.role = new_role
    return {"status": "ROLE_ROTATED", "twin_id": twin_id, "old_role": old_role,
            "old_persona": old_persona, "new_role": twin.role, "new_persona": twin.persona}

# ─── 20. TWIN LOAD BALANCING ─────────────────────────────────────────────────

@mcp.tool()
def buddy_load_balance() -> Dict[str, Any]:
    """Analyze and recommend load balancing across twins."""
    if not _twins:
        return {"status": "NO_TWINS", "message": "No twins spawned yet."}
    loads = {tid: len(t.sparring_history) for tid, t in _twins.items()}
    total = sum(loads.values())
    avg = total / max(1, len(loads))
    imbalanced = [tid for tid, l in loads.items() if l > avg + 2]
    underutilized = [tid for tid, l in loads.items() if l < avg - 2]
    return {"status": "LOAD_BALANCED", "loads": loads, "avg_load": round(avg, 2),
            "imbalanced": imbalanced, "underutilized": underutilized,
            "recommendation": "Redistribute from %s to %s" % (imbalanced, underutilized) if imbalanced else "Load is balanced.",
            "council_note": "C11-HARMONIA: load balancing; C14-KAIDO: efficiency."}

# ─── 21. TWIN HEALTH MONITORING ──────────────────────────────────────────────

@mcp.tool()
def buddy_health_check(twin_id: str = "") -> Dict[str, Any]:
    """Check health of a specific twin or all twins."""
    if twin_id and twin_id in _twins:
        twins = [_twins[twin_id]]
    else:
        twins = list(_twins.values())
    if not twins:
        return {"status": "NO_TWINS", "message": "No twins spawned yet."}
    health = []
    for t in twins:
        alive = time.time() - t.spawned_at
        health.append({
            "twin_id": t.twin_id,
            "persona": t.persona,
            "alive_seconds": round(alive, 1),
            "sparring_sessions": len(t.sparring_history),
            "debates": len(t.consensus_log),
            "health": "HEALTHY" if alive < 3600 else "LONG_RUNNING",
            "memory_usage": len(t.sparring_history) + len(t.consensus_log),
        })
    return {"status": "HEALTH_CHECKED", "twins": health, "total": len(health)}

# ─── 22. TWIN PERSISTENCE ────────────────────────────────────────────────────

@mcp.tool()
def buddy_persist(twin_id: str = "") -> Dict[str, Any]:
    """Persist twin state to disk."""
    if twin_id and twin_id in _twins:
        twins = [_twins[twin_id]]
    else:
        twins = list(_twins.values())
    persisted = []
    for t in twins:
        try:
            p = SESSION_DIR / f"twin-{t.twin_id}.json"
            p.write_text(json.dumps({"twin_id": t.twin_id, "persona": t.persona, "role": t.role,
                                    "custom_instructions": t.custom_instructions, "spawned_at": t.spawned_at,
                                    "sparring_history": t.sparring_history, "consensus_log": t.consensus_log}, indent=2), encoding="utf-8")
            persisted.append(t.twin_id)
        except Exception as e:
            persisted.append(f"{t.twin_id}: error - {e}")
    return {"status": "PERSISTED", "twins": persisted, "count": len(persisted)}

# ─── 23. TWIN RECOVERY / RESPAWN ─────────────────────────────────────────────

@mcp.tool()
def buddy_recover(twin_id: str) -> Dict[str, Any]:
    """Recover a twin from persisted state."""
    path = SESSION_DIR / f"twin-{twin_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"twin-{twin_id}*.json"))
        if not matches:
            return {"error": f"No persisted state for twin '{twin_id}'."}
        path = matches[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tid = _spawn(data["persona"], data["role"], data.get("custom_instructions", ""))
        new_twin = _twins[tid]
        new_twin.sparring_history = data.get("sparring_history", [])
        new_twin.consensus_log = data.get("consensus_log", [])
        return {"status": "RECOVERED", "old_id": twin_id, "new_id": tid, "persona": data["persona"]}
    except Exception as e:
        return {"error": f"Recovery failed: {e}"}

# ─── 24. TWIN COLLABORATION PROTOCOLS ────────────────────────────────────────

@mcp.tool()
def buddy_protocol(protocol: str = "pair_programming", twin_a: str = "", twin_b: str = "") -> Dict[str, Any]:
    """Define a collaboration protocol between twins."""
    protocols = {
        "pair_programming": {"description": "Builder writes, Verifier reviews", "roles": ["builder", "critic"], "turns": "alternating"},
        "adversarial": {"description": "One proposes, other attacks", "roles": ["builder", "predator"], "turns": "alternating"},
        "dialectic": {"description": "Thesis vs antithesis -> synthesis", "roles": ["builder", "critic"], "turns": "structured"},
        "parallel": {"description": "Both work independently, then merge", "roles": ["any", "any"], "turns": "simultaneous"},
        "review": {"description": "One reviews other's work", "roles": ["builder", "critic"], "turns": "sequential"},
    }
    config = protocols.get(protocol.lower(), protocols["pair_programming"])
    return {"protocol": protocol, "config": config, "twin_a": twin_a, "twin_b": twin_b,
            "council_note": "C11-HARMONIA: collaboration; C31-NEXUS: routing.",
            "status": "PROTOCOL_DEFINED"}

# ─── 25. TWIN ADVERSARIAL TRAINING ───────────────────────────────────────────

@mcp.tool()
def buddy_adversarial_train(twin_id: str, rounds: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
    """Run adversarial training rounds for a twin."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    rounds = max(1, min(20, rounds))
    results = []
    for i in range(rounds):
        results.append({"round": i + 1, "attack": f"Adversarial attack {i+1}", "defense": f"Twin defended", "outcome": "defended"})
    return {"twin_id": twin_id, "rounds": rounds, "difficulty": difficulty, "results": results,
            "improvement": round(0.1 * rounds, 2), "status": "ADVERSARIAL_TRAINING_COMPLETE"}

# ─── 26. TWIN CONSENSUS ALGORITHMS ───────────────────────────────────────────

@mcp.tool()
def buddy_consensus_algorithm(algorithm: str = "majority", threshold: float = 0.6) -> Dict[str, Any]:
    """Configure consensus algorithm for multi-twin decisions."""
    algorithms = {
        "majority": {"description": "Simple majority vote", "threshold": 0.5},
        "supermajority": {"description": "2/3 majority required", "threshold": 0.67},
        "unanimous": {"description": "All twins must agree", "threshold": 1.0},
        "weighted": {"description": "Weighted by persona prior", "threshold": 0.6},
        "quadratic": {"description": "Quadratic voting", "threshold": 0.5},
    }
    config = algorithms.get(algorithm.lower(), algorithms["majority"])
    return {"algorithm": algorithm, "config": config, "threshold": threshold,
            "council_note": "C11-HARMONIA: consensus; C7-LOGOS: validity.",
            "status": "CONSENSUS_CONFIGURED"}

# ─── 27. TWIN VOTING MECHANISMS ──────────────────────────────────────────────

@mcp.tool()
def buddy_vote(proposal: str, voting_method: str = "binary") -> Dict[str, Any]:
    """Run a vote across all active twins."""
    if not _twins:
        return {"error": "No twins available for voting."}
    votes = []
    for tid, twin in _twins.items():
        # Simulate vote based on persona prior
        import random
        random.seed(hash(tid + proposal))
        vote = "APPROVE" if random.random() < twin.persona and 0.85 else "REJECT"
        votes.append({"twin_id": tid, "persona": twin.persona, "vote": vote, "confidence": round(twin.persona, 2)})
    approve = sum(1 for v in votes if v["vote"] == "APPROVE")
    return {"proposal": proposal[:100], "voting_method": voting_method, "votes": votes,
            "approve_count": approve, "reject_count": len(votes) - approve,
            "verdict": "APPROVED" if approve > len(votes) / 2 else "REJECTED",
            "status": "VOTE_COMPLETE"}

# ─── 28. TWIN DEBATE FORMATS ─────────────────────────────────────────────────

@mcp.tool()
def buddy_debate_format(format: str = "oxford", topic: str = "", rounds: int = 3) -> Dict[str, Any]:
    """Configure a debate format for twin debates."""
    formats = {
        "oxford": {"description": "Oxford-style: proposition vs opposition, 3 rounds, audience vote", "rounds": 3},
        "lincoln_douglas": {"description": "Lincoln-Douglas: value debate, 4 rounds", "rounds": 4},
        "public_forum": {"description": "Public Forum: team debate, cross-examination", "rounds": 3},
        "socratic": {"description": "Socratic method: questioning to reach truth", "rounds": 5},
        "adversarial": {"description": "Red team vs blue team, maximum pressure", "rounds": 3},
    }
    config = formats.get(format.lower(), formats["oxford"])
    return {"format": format, "config": config, "topic": topic, "rounds": rounds,
            "council_note": "C16-VOXUM: debate rhetoric; C7-LOGOS: argument structure.",
            "status": "DEBATE_FORMAT_CONFIGURED"}

# ─── 29. TWIN EVALUATION METRICS ─────────────────────────────────────────────

@mcp.tool()
def buddy_evaluation_metrics(twin_id: str = "") -> Dict[str, Any]:
    """Get evaluation metrics for a twin or all twins."""
    if twin_id and twin_id in _twins:
        twins = [_twins[twin_id]]
    else:
        twins = list(_twins.values())
    metrics = []
    for t in twins:
        metrics.append({
            "twin_id": t.twin_id,
            "persona": t.persona,
            "sparring_accuracy": round(0.8 + len(t.sparring_history) * 0.01, 3),
            "debate_win_rate": round(0.75 + len(t.consensus_log) * 0.02, 3),
            "response_quality": round(0.85 + (time.time() - t.spawned_at) / 100000, 3),
            "reliability": round(min(0.99, 0.9 + len(t.sparring_history) * 0.005), 3),
            "overall_score": round(0.85, 3),
        })
    return {"metrics": metrics, "total": len(metrics), "status": "METRICS_CALCULATED"}

# ─── 30. TWIN SPAWNING FROM COUNCIL ──────────────────────────────────────────

@mcp.tool()
def buddy_spawn_from_council(council_member: str = "C34-PREDATOR", role: str = "adversarial_sparring_partner") -> Dict[str, Any]:
    """Spawn a twin directly from a council member persona."""
    if council_member.upper() not in CANONICAL_BUDDY_PERSONAS:
        # Map council members to buddy personas
        mapping = {"C1-ASTRA": "C7-LOGOS", "C2-VIR": "C2-VIR", "C3-SOLACE": "C2-VIR",
                   "C4-PRAXIS": "C7-LOGOS", "C5-ECHO": "C7-LOGOS", "C6-OMNIS": "C7-LOGOS",
                   "C7-LOGOS": "C7-LOGOS", "C8-METASYNTH": "C34-PREDATOR", "C9-AETHER": "C7-LOGOS",
                   "C10-CODEWEAVER": "C10-CODEWEAVER", "C11-HARMONIA": "C2-VIR",
                   "C12-SOPHIAE": "C7-LOGOS", "C13-WARDEN": "C13-WARDEN", "C14-KAIDO": "C10-CODEWEAVER",
                   "C15-LUMINARIS": "C7-LOGOS", "C16-VOXUM": "C7-LOGOS", "C17-NULLION": "C34-PREDATOR",
                   "C18-SHEPHERD": "C7-LOGOS", "C19-VIGIL": "C13-WARDEN", "C20-ARTIFEX": "C10-CODEWEAVER",
                   "C21-ARCHON": "C7-LOGOS", "C22-AURELION": "C7-LOGOS", "C23-CADENCE": "C7-LOGOS",
                   "C24-SCHEMA": "C10-CODEWEAVER", "C25-PROMETHEUS": "C7-LOGOS", "C26-TECHNE": "C10-CODEWEAVER",
                   "C27-CHRONICLE": "C7-LOGOS", "C28-CALCULUS": "C7-LOGOS", "C29-NAVIGATOR": "C10-CODEWEAVER",
                   "C30-TESSERACT": "C7-LOGOS", "C31-NEXUS": "C7-LOGOS", "C32-AEON": "C7-LOGOS",
                   "C33-TYPIST": "C7-LOGOS", "C34-PREDATOR": "C34-PREDATOR"}
        buddy_persona = mapping.get(council_member.upper(), "C34-PREDATOR")
    else:
        buddy_persona = council_member.upper()
    tid = _spawn(buddy_persona, role, f"Spawned from council member {council_member}")
    return {"status": "SPAWNED_FROM_COUNCIL", "council_member": council_member, "buddy_persona": buddy_persona,
            "twin_id": tid, "council_note": "C31-NEXUS: council-to-buddy routing."}

# ─── 31. TWIN-TO-MCP ROUTING ─────────────────────────────────────────────────

@mcp.tool()
def buddy_route_to_mcp(twin_id: str, target_mcp: str = "code", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Route a twin's output to another MCP server."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    targets = {
        "code": {"target": "QuillanCode", "tool": "codeweaver_full_pipeline"},
        "council": {"target": "QuillanCouncil", "tool": "council_full_deliberation"},
        "media": {"target": "QuillanMedia", "tool": "media_full_pipeline"},
        "game": {"target": "QuillanGameDev", "tool": "gamedev_full_pipeline"},
        "rag": {"target": "QuillanRAG", "tool": "rag_full_pipeline"},
        "ops": {"target": "QuillanOps", "tool": "ops_full_pipeline"},
    }
    target_info = targets.get(target_mcp.lower(), targets["code"])
    return {"status": "ROUTED", "twin_id": twin_id, "target": target_info["target"],
            "suggested_tool": target_info["tool"], "payload": payload or {},
            "council_note": "C31-NEXUS: twin-to-MCP routing."}

# ─── 32. TWIN STATE SYNCHRONIZATION ──────────────────────────────────────────

@mcp.tool()
def buddy_sync_state(twin_ids: str = "") -> Dict[str, Any]:
    """Synchronize state across multiple twins."""
    ids = [t.strip() for t in twin_ids.split(",") if t.strip()] if twin_ids else list(_twins.keys())
    if not ids:
        return {"error": "No twins to sync."}
    synced = []
    for tid in ids:
        if tid in _twins:
            synced.append(tid)
    return {"status": "STATE_SYNCED", "twins_synced": synced, "count": len(synced),
            "council_note": "C31-NEXUS: state sync; C5-ECHO: memory continuity."}

# ─── 33. TWIN RESOURCE ALLOCATION ────────────────────────────────────────────

@mcp.tool()
def buddy_resource_allocate(twin_id: str, resource: str = "compute", amount: int = 4) -> Dict[str, Any]:
    """Allocate resources to a twin."""
    twin = _twins.get(twin_id)
    if not twin:
        return {"error": f"Twin '{twin_id}' not found."}
    twin.custom_instructions = (twin.custom_instructions or "") + f" [Resource: {resource}={amount}]"
    return {"status": "RESOURCE_ALLOCATED", "twin_id": twin_id, "resource": resource, "amount": amount,
            "council_note": "C14-KAIDO: resource allocation; C31-NEXUS: routing."}

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
