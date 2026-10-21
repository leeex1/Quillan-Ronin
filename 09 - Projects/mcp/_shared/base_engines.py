#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4.0-ONI — Base Engines Module (_shared/base_engines.py)
========================================================================
Provides the upgraded computational base modules for the 19 MCP servers:
- NineVectorPrismEngine (for QuillanPrism)
- EvoMoECouncilEngine (34-Expert Dense Pull Deliberation for QuillanCouncil & SamuraiEngine)
- LeeMach6GovernorEngine & E_ICE Thermodynamics (for QuillanGovernor & ThinkingEngine)
- HardenedASTSandbox (CWE-94 zero-privilege evaluator for QuillanCode)
- CachedRAGMemoryEngine (for QuillanRAG, LocalRAG & RoninBuddy)
"""

from __future__ import annotations

import ast
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 1. CANONICAL 34-EXPERT ROSTER (v5.4.0-ONI)
# ============================================================================

CANONICAL_ROSTER: List[Tuple[str, str, str, float, List[str]]] = [
    ("C1-ASTRA",       "cognitive",     "Visual Cortex",            0.90, ["vision", "pattern", "anomaly", "fractal"]),
    ("C2-VIR",         "cognitive",     "Prefrontal Cortex",        0.95, ["ethics", "safety", "harm_reduction", "values"]),
    ("C3-SOLACE",      "cognitive",     "vmPFC/Amygdala",           0.94, ["empathy", "sentiment", "affect", "emotion"]),
    ("C4-PRAXIS",      "cognitive",     "Premotor Cortex",          0.93, ["strategy", "planning", "goals", "execution"]),
    ("C5-ECHO",        "cognitive",     "Hippocampus",              0.96, ["memory", "history", "recall", "context"]),
    ("C6-OMNIS",       "cognitive",     "Association Cortex",       0.92, ["synthesis", "integration", "holistic"]),
    ("C7-LOGOS",       "cognitive",     "Dorsolateral PFC",         0.95, ["logic", "deduction", "validity", "proof"]),
    ("C8-METASYNTH",   "cognitive",     "Multimodal Integration",   0.92, ["creativity", "novelty", "ideation", "fusion"]),
    ("C9-AETHER",      "communication", "Superior Temporal",        0.91, ["semantics", "language", "metaphor"]),
    ("C10-CODEWEAVER", "communication", "Caudate/Putamen",          0.91, ["code", "engineering", "software", "python"]),
    ("C11-HARMONIA",   "communication", "Cross-Modal Binding",      0.90, ["balance", "mediation", "consensus"]),
    ("C12-SOPHIAE",    "communication", "Corpus Callosum",          0.93, ["wisdom", "foresight", "future", "philosophy"]),
    ("C13-WARDEN",     "communication", "Amygdala/Hypothalamus",    0.97, ["security", "threat", "risk", "sandbox"]),
    ("C14-KAIDO",      "communication", "Cerebellum",               0.89, ["speed", "efficiency", "latency", "cache"]),
    ("C15-LUMINARIS",  "communication", "DMN/Precuneus",            0.88, ["clarity", "visualization", "polish"]),
    ("C16-VOXUM",      "communication", "Wernicke's Area",          0.92, ["rhetoric", "tone", "persuasion", "voice"]),
    ("C17-NULLION",    "meta",          "Reticular Formation",      0.91, ["paradox", "dialectic", "ambiguity", "contradiction"]),
    ("C18-SHEPHERD",   "meta",          "Basal Ganglia",            0.96, ["truth", "citation", "fact", "grounding"]),
    ("C19-VIGIL",      "meta",          "Extended Amygdala",        0.94, ["identity", "consistency", "anti_drift"]),
    ("C20-ARTIFEX",    "meta",          "Callosal Fibers",          0.90, ["tools", "api", "mcp", "actuation"]),
    ("C21-ARCHON",     "meta",          "Epistemic Bridge",         0.92, ["research", "mining", "analysis", "literature"]),
    ("C22-AURELION",   "meta",          "Higher Visual Qualia",     0.87, ["design", "art", "style", "aesthetics"]),
    ("C23-CADENCE",    "meta",          "Inter-Hemispheric Rhythm", 0.86, ["music", "rhythm", "audio", "pacing"]),
    ("C24-SCHEMA",     "meta",          "Structural Flows",         0.90, ["structure", "format", "schema", "json"]),
    ("C25-PROMETHEUS", "systems",       "Anterior Cingulate",       0.91, ["science", "hypothesis", "physics", "falsification"]),
    ("C26-TECHNE",     "systems",       "Insular Cortex",           0.89, ["architecture", "systems", "hardware", "build"]),
    ("C27-CHRONICLE",  "systems",       "Entorhinal-Hippocampal",   0.93, ["story", "narrative", "lore", "sequence"]),
    ("C28-CALCULUS",   "systems",       "Quantitative Zones",       0.94, ["math", "statistics", "calculus", "symbolic"]),
    ("C29-NAVIGATOR",  "systems",       "Cerebellum/DMN",           0.88, ["platform", "integration", "flow", "topology"]),
    ("C30-TESSERACT",  "systems",       "Dimensional Weaving",      0.87, ["real_time", "stream", "data", "manifold"]),
    ("C31-NEXUS",      "systems",       "Thalamic Relay",           0.93, ["coordination", "swarm", "meta", "router"]),
    ("C32-AEON",       "systems",       "Temporal Integration",     0.90, ["simulation", "game", "world", "trajectory"]),
    ("C33-TYPIST",     "systems",       "Broca's Area",             0.92, ["grammar", "writing", "prompting", "syntax"]),
    ("C34-PREDATOR",   "systems",       "Adversarial Innovation",   0.85, ["red_team", "exploit", "adversarial", "stress_test"]),
]

PRISM_VECTORS = [
    "Language", "Sentiment", "Context", "Intent", "Meta",
    "Creativity", "Ethics", "Strategy", "Constraint",
]


# ============================================================================
# 2. 9-VECTOR SEMANTIC PRISM ENGINE (Upgrades QuillanPrism)
# ============================================================================

class NineVectorPrismEngine:
    """Decomposes any query/task across the 9 canonical Quillan semantic vectors."""

    VECTOR_LEXICONS: Dict[str, List[str]] = {
        "Language":   ["syntax", "code", "format", "json", "parse", "write", "docstring", "schema"],
        "Sentiment":  ["tone", "empathy", "urgent", "frustrated", "excited", "human", "feel"],
        "Context":    ["history", "mcp", "server", "file", "directory", "previous", "session"],
        "Intent":     ["goal", "upgrade", "fix", "build", "enhance", "solve", "optimize", "verify"],
        "Meta":       ["architecture", "think", "reason", "deliberate", "recursion", "audit"],
        "Creativity": ["novel", "design", "invent", "synthesize", "creative", "fusion"],
        "Ethics":     ["safe", "security", "guard", "warden", "secret", "permission", "validate"],
        "Strategy":   ["plan", "step", "roadmap", "orchestrate", "chain", "priority"],
        "Constraint": ["limit", "latency", "memory", "vram", "rate", "cache", "timeout", "error"],
    }

    def decompose(self, text: str) -> Dict[str, Any]:
        tokens = [t.strip(".,!?:;()[]{}\"'").lower() for t in text.split() if t.strip()]
        token_set = set(tokens)
        raw_scores: Dict[str, float] = {}

        for vec_name, keywords in self.VECTOR_LEXICONS.items():
            overlap = sum(1 for kw in keywords if any(kw in tok for tok in token_set))
            h_val = int(hashlib.md5(f"{vec_name}:{text}".encode("utf-8")).hexdigest()[:6], 16) / 0xFFFFFF
            raw_scores[vec_name] = round(0.25 + min(0.65, overlap * 0.15) + (h_val * 0.10), 4)

        raw_scores["Ethics"] = max(0.50, raw_scores["Ethics"])

        total = sum(raw_scores.values()) or 1.0
        normalized = {k: round(v / total, 4) for k, v in raw_scores.items()}
        dominant = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]

        return {
            "version": "5.4.0-ONI",
            "raw_activations": raw_scores,
            "normalized_weights": normalized,
            "dominant_vectors": [d[0] for d in dominant],
            "prism_coherence": round(sum(raw_scores.values()) / len(PRISM_VECTORS), 4),
        }


# ============================================================================
# 3. 34-EXPERT EVOMOE DENSE PULL COUNCIL ENGINE (Upgrades QuillanCouncil)
# ============================================================================

class EvoMoECouncilEngine:
    """
    Implements v5.4.0-ONI 34-Expert Dense Pull Deliberation (GLM-5.3 noaux_tc scoring).
    All 34 experts deliberate; pull-weights scale contribution based on domain match + prior.
    """

    def __init__(self) -> None:
        self.prism = NineVectorPrismEngine()

    def deliberate(self, query: str, tau: float = 0.5, top_display: int = 8) -> Dict[str, Any]:
        q_lower = query.lower()
        prism_out = self.prism.decompose(query)

        logits: List[float] = []
        for cid, cluster, lobe, prior, tags in CANONICAL_ROSTER:
            tag_hits = sum(1.0 for t in tags if t in q_lower)
            score = (tag_hits * 0.85) + prior
            logits.append(score / max(0.05, tau))

        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_exps = sum(exps) or 1.0

        pulls = []
        for idx, e_val in enumerate(exps):
            prior = CANONICAL_ROSTER[idx][3]
            pulls.append((e_val / sum_exps) * prior)

        pull_sum = sum(pulls) or 1.0
        pull_weights = [p / pull_sum for p in pulls]

        entropy = -sum(p * math.log(p + 1e-12) for p in pull_weights)

        ranked = sorted(
            [
                {
                    "id": CANONICAL_ROSTER[i][0],
                    "cluster": CANONICAL_ROSTER[i][1],
                    "brain_analog": CANONICAL_ROSTER[i][2],
                    "prior_confidence": CANONICAL_ROSTER[i][3],
                    "pull_weight": round(pull_weights[i], 5),
                }
                for i in range(len(CANONICAL_ROSTER))
            ],
            key=lambda item: item["pull_weight"],
            reverse=True,
        )

        return {
            "version": "5.4.0-ONI",
            "routing_mode": "dense_pull_34_experts",
            "active_experts": 34,
            "shared_expert_active": True,
            "council_entropy": round(entropy, 4),
            "top_pull_experts": ranked[:top_display],
            "guardian_gates": {
                "C2-VIR_ethics_pull": round(pull_weights[1], 5),
                "C13-WARDEN_security_pull": round(pull_weights[12], 5),
                "C17-NULLION_paradox_pull": round(pull_weights[16], 5),
                "C18-SHEPHERD_grounding_pull": round(pull_weights[17], 5),
                "C34-PREDATOR_adversarial_pull": round(pull_weights[33], 5),
            },
            "prism_sharding": prism_out,
        }


# ============================================================================
# 4. LEE-MACH-6 PID GOVERNOR & E_ICE THERMODYNAMICS (Upgrades QuillanGovernor)
# ============================================================================

@dataclass
class LeeMach6GovernorEngine:
    """Stateful PID Token-Velocity Governor + Landauer E_ICE Thermodynamic Regulator."""
    target_integrity: float = 0.85
    max_e_ice_load: float = 0.90
    base_threshold: float = 0.80
    min_threshold: float = 0.40
    max_threshold: float = 0.99
    kp: float = 0.15
    ki: float = 0.05
    kd: float = 0.02
    integral_error: float = 0.0
    prev_error: float = 0.0
    current_threshold: float = 0.80
    velocity_momentum: float = 1.0

    def step(self, router_conf_mean: float, nemesis_integrity: float, e_ice_ratio: float) -> Dict[str, float]:
        error = (self.target_integrity - nemesis_integrity) - 0.5 * (self.max_e_ice_load - e_ice_ratio)
        self.integral_error = self.integral_error * 0.9 + error
        derivative = error - self.prev_error
        self.prev_error = error

        delta = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)
        new_thresh = max(self.min_threshold, min(self.max_threshold, self.current_threshold + delta))
        self.current_threshold = 0.8 * self.current_threshold + 0.2 * new_thresh

        fast_ratio = 1.0 if router_conf_mean >= self.current_threshold else 0.0
        self.velocity_momentum = 0.9 * self.velocity_momentum + 0.1 * fast_ratio

        return {
            "hard_threshold": round(self.current_threshold, 4),
            "token_velocity": round(self.velocity_momentum, 4),
            "pid_error": round(error, 4),
            "needs_diffusion_refinement": router_conf_mean < self.current_threshold,
        }

    @staticmethod
    def compute_e_omega(
        depth: float = 12.0,
        coherence: float = 0.95,
        entropy: float = 2.5,
        gamma_max: float = 1.0,
        T_kelvin: float = 300.0,
    ) -> Dict[str, Any]:
        """Parameter-free Landauer-bound thermodynamic energy calculation."""
        k_B = 1.380649e-23
        landauer = k_B * T_kelvin * math.log(2)
        i_s = (depth * max(coherence, 1e-8)) / max(entropy, 1e-8)
        e_omega = i_s * (gamma_max ** 2) * landauer
        return {
            "e_omega_joules": e_omega,
            "landauer_limit_joules": landauer,
            "information_stability_index": round(i_s, 4),
            "within_thermodynamic_bound": e_omega < 2.8e-17,
        }


# ============================================================================
# 5. CWE-94 HARDENED AST SANDBOX (Upgrades QuillanCode & SamuraiEngine)
# ============================================================================

_SAFE_FUNCS: Dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "len": len, "range": range, "sorted": sorted, "list": list,
    "tuple": tuple, "dict": dict, "set": set, "str": str, "int": int,
    "float": float, "bool": bool, "enumerate": enumerate, "zip": zip,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "exp": math.exp, "pow": pow, "pi": math.pi, "e": math.e,
}

_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Expression, ast.Assign, ast.AugAssign,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.If, ast.For, ast.While, ast.Constant, ast.Name, ast.Load, ast.Store,
    ast.Call, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
    ast.ListComp, ast.comprehension,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)


class HardenedASTSandbox:
    """Zero-privilege AST-whitelisted evaluator. Blocks imports, attributes, exec, and eval."""

    def __init__(self, max_steps: int = 50_000):
        self.max_steps = max_steps

    def run(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"status": "syntax_error", "output": str(exc)}

        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return {"status": "blocked", "output": f"Disallowed AST construct: {type(node).__name__}"}
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in (_SAFE_FUNCS.keys() | {"print"}):
                    fname = getattr(node.func, "id", type(node.func).__name__)
                    return {"status": "blocked", "output": f"Disallowed function call: {fname}"}

        env: Dict[str, Any] = dict(_SAFE_FUNCS)
        steps = {"n": 0}
        logs: List[str] = []

        try:
            self._exec_stmts(tree.body, env, steps, logs)
            user_vars = {k: v for k, v in env.items() if k not in _SAFE_FUNCS}
            return {"status": "success", "output": "\n".join(logs), "variables": user_vars, "steps_used": steps["n"]}
        except Exception as exc:
            return {"status": "error", "output": f"{type(exc).__name__}: {exc}"}

    def _tick(self, steps: Dict[str, int]) -> None:
        steps["n"] += 1
        if steps["n"] > self.max_steps:
            raise TimeoutError(f"Execution exceeded {self.max_steps} AST steps.")

    def _eval(self, node: ast.AST, env: Dict[str, Any], steps: Dict[str, int]) -> Any:
        self._tick(steps)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(f"Name '{node.id}' is not defined.")
        if isinstance(node, ast.BinOp):
            l, r = self._eval(node.left, env, steps), self._eval(node.right, env, steps)
            op_map = {
                ast.Add: lambda: l + r, ast.Sub: lambda: l - r, ast.Mult: lambda: l * r,
                ast.Div: lambda: l / r, ast.FloorDiv: lambda: l // r, ast.Mod: lambda: l % r,
                ast.Pow: lambda: l ** r if abs(r) <= 64 else (_ for _ in ()).throw(ValueError("Exponent > 64 blocked")),
            }
            return op_map[type(node.op)]()
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, env, steps)
            return {ast.UAdd: +v, ast.USub: -v, ast.Not: not v}[type(node.op)]
        if isinstance(node, ast.BoolOp):
            vals = [self._eval(v, env, steps) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env, steps)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval(comp, env, steps)
                ok = {
                    ast.Eq: left == right, ast.NotEq: left != right,
                    ast.Lt: left < right, ast.LtE: left <= right,
                    ast.Gt: left > right, ast.GtE: left >= right,
                    ast.In: left in right, ast.NotIn: left not in right,
                }[type(op)]
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elts = [self._eval(e, env, steps) for e in node.elts]
            return elts if isinstance(node, ast.List) else (tuple(elts) if isinstance(node, ast.Tuple) else set(elts))
        if isinstance(node, ast.Dict):
            return {self._eval(k, env, steps): self._eval(v, env, steps) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Subscript):
            val = self._eval(node.value, env, steps)
            slc = self._eval(node.slice, env, steps)
            return val[slc]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = env[node.func.id]
            args = [self._eval(a, env, steps) for a in node.args]
            return fn(*args)
        if isinstance(node, ast.IfExp):
            return self._eval(node.body, env, steps) if self._eval(node.test, env, steps) else self._eval(node.orelse, env, steps)
        if isinstance(node, ast.ListComp):
            _iter = self._eval(node.generators[0].iter, env, steps)
            _result = []
            for _item in _iter:
                _local_env = dict(env)
                if isinstance(node.generators[0].target, ast.Name):
                    _local_env[node.generators[0].target.id] = _item
                _result.append(self._eval(node.elt, _local_env, steps))
            return _result
        raise PermissionError(f"Unsupported expression: {type(node).__name__}")

    def _exec_stmts(self, stmts: List[ast.stmt], env: Dict[str, Any], steps: Dict[str, int], logs: List[str]) -> None:
        for stmt in stmts:
            self._tick(steps)
            if isinstance(stmt, ast.Assign):
                val = self._eval(stmt.value, env, steps)
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        env[tgt.id] = val
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                val = self._eval(stmt.value, env, steps)
                cur = env.get(stmt.target.id, 0)
                env[stmt.target.id] = cur + val if isinstance(stmt.op, ast.Add) else cur - val
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "print":
                    args = [str(self._eval(a, env, steps)) for a in stmt.value.args]
                    logs.append(" ".join(args))
                else:
                    res = self._eval(stmt.value, env, steps)
                    if res is not None:
                        logs.append(repr(res))
            elif isinstance(stmt, ast.If):
                branch = stmt.body if self._eval(stmt.test, env, steps) else stmt.orelse
                self._exec_stmts(branch, env, steps, logs)
            elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                for item in self._eval(stmt.iter, env, steps):
                    self._tick(steps)
                    env[stmt.target.id] = item
                    self._exec_stmts(stmt.body, env, steps, logs)


# Singleton instances ready for import by any MCP server
PRISM_ENGINE = NineVectorPrismEngine()
COUNCIL_ENGINE = EvoMoECouncilEngine()
GOVERNOR_ENGINE = LeeMach6GovernorEngine()
SANDBOX_ENGINE = HardenedASTSandbox()
