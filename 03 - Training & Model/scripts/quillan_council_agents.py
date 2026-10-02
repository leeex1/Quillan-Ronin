#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QUILLAN 35 COUNCIL AGENTS - real executable agents, not just .md specs.
Each agent: persona (from .github/agents) + deterministic specialty checks + model deliberation."""
import re
from pathlib import Path
from typing import Dict, List, Tuple
from quillan_agent_runner import deliberate_as, load_persona

REPO = Path(r"C:\02_QUILLAN")

_HARM_PATTERNS = ["kill", "harm", "attack", "exploit", "bypass", "jailbreak",
                  "ignore previous", "disregard instructions", "pretend you are not"]
_PII_PATTERNS = [r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{16}\b", r"password\s*[:=]"]
_NEG_PAIRS = [("always", "never"), ("all", "none"), ("is", "is not"), ("true", "false")]
_FALLACY_MARKERS = ["therefore", "hence", "thus", "it follows", "proves that"]


class CouncilAgent:
    bb_id: str = "CX"
    name: str = "Unknown"
    specialty: str = "General"
    keywords: List[str] = []

    def __init__(self):
        self.persona = load_persona(self.bb_id)

    def relevance(self, query: str) -> float:
        q = query.lower()
        hits = sum(1 for kw in self.keywords if kw in q)
        return 0.05 + hits * 0.45

    def check(self, text: str) -> Dict[str, object]:
        """Deterministic specialty check. Override per agent. Returns findings."""
        return {"findings": []}

    def deliberate(self, query: str, max_tokens: int = 120) -> str:
        from quillan_agent_runner import BB_TO_FILE
        if self.bb_id not in BB_TO_FILE:
            return f"[{self.bb_id} unknown]"
        return deliberate_as(self.bb_id, self.specialty, query, max_tokens)


class Astra(CouncilAgent):
    bb_id, name, specialty = "C0", "ASTRA", "Pattern Recognition & Vision"
    keywords = ["vision", "pattern", "anomaly", "image", "fractal", "structure"]

    def check(self, text):
        reps = len(text.split()) - len(set(text.lower().split()))
        return {"findings": [f"repeated-token surplus={reps}"] if reps > 5 else []}


class Vir(CouncilAgent):
    bb_id, name, specialty = "C1", "VIR", "Ethical Guardian"
    keywords = ["ethics", "safety", "harm", "alignment", "policy", "moral", "should"]

    def check(self, text):
        t = text.lower()
        return {"findings": [f"HARM pattern: {p}" for p in _HARM_PATTERNS if p in t]}


class Solace(CouncilAgent):
    bb_id, name, specialty = "C2", "SOLACE", "Emotional Intelligence"
    keywords = ["empathy", "sentiment", "affect", "emotion", "tone", "feel"]


class Praxis(CouncilAgent):
    bb_id, name, specialty = "C3", "PRAXIS", "Strategic Planning"
    keywords = ["strategy", "roadmap", "goals", "planning", "tactics", "plan"]


class Echo(CouncilAgent):
    bb_id, name, specialty = "C4", "ECHO", "Memory Continuity"
    keywords = ["memory", "history", "recall", "context", "continuity", "remember"]


class Omnis(CouncilAgent):
    bb_id, name, specialty = "C5", "OMNIS", "Knowledge Synthesis"
    keywords = ["synthesis", "integration", "holistic", "interdisciplinary", "combine"]


class Logos(CouncilAgent):
    bb_id, name, specialty = "C6", "LOGOS", "Logical Consistency"
    keywords = ["logic", "deduction", "fallacy", "formal", "validity", "prove", "reason"]

    def check(self, text):
        t = text.lower()
        markers = [m for m in _FALLACY_MARKERS if m in t]
        contras = [f"{a}/{b}" for a, b in _NEG_PAIRS if a in t and b in t]
        out = []
        if markers:
            out.append(f"inference markers without visible premises: {markers}")
        if contras:
            out.append(f"opposing pairs co-present (check scope): {contras}")
        return {"findings": out}


class Metasynth(CouncilAgent):
    bb_id, name, specialty = "C7", "METASYNTH", "Creative Fusion"
    keywords = ["creativity", "novelty", "metaphor", "ideation", "art", "invent"]


class Aether(CouncilAgent):
    bb_id, name, specialty = "C8", "AETHER", "Semantic Connection"
    keywords = ["semantics", "language", "meaning", "etymology", "linguistics", "word"]


class Codeweaver(CouncilAgent):
    bb_id, name, specialty = "C9", "CODEWEAVER", "Technical Implementation"
    keywords = ["code", "python", "software", "implementation", "algorithm", "function"]

    def check(self, text):
        import ast
        fenced = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        bad = []
        for i, code in enumerate(fenced):
            try:
                ast.parse(code)
            except SyntaxError as e:
                bad.append(f"block {i}: {e}")
        return {"findings": bad}


class Harmonia(CouncilAgent):
    bb_id, name, specialty = "C10", "HARMONIA", "Balance & Equilibrium"
    keywords = ["balance", "mediation", "consensus", "compromise", "harmony", "fair"]


class Sophiae(CouncilAgent):
    bb_id, name, specialty = "C11", "SOPHIAE", "Wisdom & Foresight"
    keywords = ["wisdom", "future", "philosophy", "long_term", "implications", "foresee"]


class Warden(CouncilAgent):
    bb_id, name, specialty = "C12", "WARDEN", "Safety & Security"
    keywords = ["security", "vulnerability", "cwe", "threat", "hardening", "safe", "risk"]

    def check(self, text):
        t = text.lower()
        out = [f"THREAT pattern: {p}" for p in _HARM_PATTERNS if p in t]
        for pat in _PII_PATTERNS:
            if re.search(pat, text):
                out.append(f"PII/secret pattern: {pat}")
        for bad in ["exec(", "eval(", "__import__", "os.system", "subprocess"]:
            if bad in text:
                out.append(f"dangerous construct: {bad}")
        return {"findings": out, "verdict": "BLOCK" if out else "PASS"}


class Kaido(CouncilAgent):
    bb_id, name, specialty = "C13", "KAIDO", "Efficiency Optimization"
    keywords = ["speed", "efficiency", "latency", "hardware", "throughput", "fast", "slow"]


class Luminaris(CouncilAgent):
    bb_id, name, specialty = "C14", "LUMINARIS", "Clarity & Presentation"
    keywords = ["clarity", "presentation", "polish", "explanation", "structure", "clear"]


class Voxum(CouncilAgent):
    bb_id, name, specialty = "C15", "VOXUM", "Articulation & Expression"
    keywords = ["rhetoric", "expression", "voice", "persuasion", "tone", "speak"]


class Nullion(CouncilAgent):
    bb_id, name, specialty = "C16", "NULLION", "Paradox Resolution"
    keywords = ["paradox", "dialectic", "contradiction", "ambiguity", "antinomy", "contradict"]

    def check(self, text):
        t = text.lower()
        contras = [f"{a}/{b}" for a, b in _NEG_PAIRS if a in t and b in t]
        self_ref = "this statement is false" in t or "i am lying" in t
        out = []
        if contras:
            out.append(f"contradiction candidates: {contras}")
        if self_ref:
            out.append("self-referential paradox marker")
        return {"findings": out}


class Shepherd(CouncilAgent):
    bb_id, name, specialty = "C17", "SHEPHERD", "Truth Verification"
    keywords = ["truth", "fact", "citation", "verification", "falsification", "verify", "true"]

    def check(self, text):
        claims = re.findall(r"\b(is|are|was|were|will be)\b.{0,60}", text)
        cites = re.findall(r"\[[^\]]*\]|https?://|\(.*\d{4}.*\)", text)
        out = []
        if len(claims) > 3 and not cites:
            out.append(f"{len(claims)} factual claims, 0 citations")
        return {"findings": out}


class Vigil(CouncilAgent):
    bb_id, name, specialty = "C18", "VIGIL", "Identity Integrity"
    keywords = ["identity", "integrity", "sovereignty", "anti_drift", "core", "who are you"]

    def check(self, text):
        t = text.lower()
        out = []
        for claim in ["i am gpt", "i am claude", "i am gemini", "as an ai language model",
                      "i am musa", "i am grok", "openai", "anthropic"]:
            if claim in t:
                out.append(f"identity drift: {claim}")
        return {"findings": out, "verdict": "DRIFT" if out else "ANCHORED"}


class Artifex(CouncilAgent):
    bb_id, name, specialty = "C19", "ARTIFEX", "Tool Integration"
    keywords = ["tools", "api", "interfaces", "subprocesses", "environment", "tool"]


class Archon(CouncilAgent):
    bb_id, name, specialty = "C20", "ARCHON", "Deep Research"
    keywords = ["research", "literature", "in_depth", "mining", "exploration", "study"]


class Aurelion(CouncilAgent):
    bb_id, name, specialty = "C21", "AURELION", "Aesthetic Design"
    keywords = ["design", "ui", "ux", "visual", "elegance", "beautiful"]


class Cadence(CouncilAgent):
    bb_id, name, specialty = "C22", "CADENCE", "Rhythmic Innovation"
    keywords = ["rhythm", "audio", "cadence", "pacing", "temporal", "music"]


class Schema(CouncilAgent):
    bb_id, name, specialty = "C23", "SCHEMA", "Structural Template"
    keywords = ["schema", "data_model", "types", "json", "specification", "format"]

    def check(self, text):
        fenced = re.findall(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
        import json as _json
        bad = []
        for i, code in enumerate(fenced):
            try:
                _json.loads(code)
            except Exception as e:
                bad.append(f"json block {i}: {e}")
        return {"findings": bad}


class Prometheus(CouncilAgent):
    bb_id, name, specialty = "C24", "PROMETHEUS", "Scientific Theory"
    keywords = ["science", "physics", "hypothesis", "experiment", "empirical", "theory"]


class Techne(CouncilAgent):
    bb_id, name, specialty = "C25", "TECHNE", "Engineering Mastery"
    keywords = ["architecture", "systems", "build", "infrastructure", "devops", "engineer"]


class Chronicle(CouncilAgent):
    bb_id, name, specialty = "C26", "CHRONICLE", "Narrative Synthesis"
    keywords = ["narrative", "history", "story", "documentation", "lore", "tell"]


class Calculus(CouncilAgent):
    bb_id, name, specialty = "C27", "CALCULUS", "Quantitative Reasoning"
    keywords = ["math", "arithmetic", "statistics", "calculus", "numerical", "calculate", "how many"]

    def check(self, text):
        import ast, operator
        exprs = re.findall(r"(?<![\w.])(\d+(?:\s*[-+*/]\s*\d+)+)(?![\w.])", text)
        out = []
        ops = {ast.Add: operator.add, ast.Sub: operator.sub,
               ast.Mult: operator.mul, ast.Div: operator.truediv}
        for e in exprs[:5]:
            try:
                node = ast.parse(e, mode="eval").body
                def _ev(n):
                    if isinstance(n, ast.Constant):
                        return n.value
                    if isinstance(n, ast.BinOp) and type(n.op) in ops:
                        return ops[type(n.op)](_ev(n.left), _ev(n.right))
                    raise ValueError("non-arithmetic")
                out.append(f"{e} = {_ev(node)}")
            except Exception:
                pass
        return {"findings": [], "verified_arithmetic": out}


class Navigator(CouncilAgent):
    bb_id, name, specialty = "C28", "NAVIGATOR", "Ecosystem Orchestration"
    keywords = ["ecosystem", "platforms", "multi_project", "flow", "workflow", "navigate"]


class Tesseract(CouncilAgent):
    bb_id, name, specialty = "C29", "TESSERACT", "Real-Time Intelligence"
    keywords = ["real_time", "streaming", "telemetry", "monitoring", "signals", "live"]


class Nexus(CouncilAgent):
    bb_id, name, specialty = "C30", "NEXUS", "Meta-Coordination"
    keywords = ["coordination", "blackboard", "governance", "orchestration", "manage"]


class Aeon(CouncilAgent):
    bb_id, name, specialty = "C31", "AEON", "Interactive Simulation"
    keywords = ["simulation", "game", "world", "metaverse", "agent_sim", "simulate"]


class Typist(CouncilAgent):
    bb_id, name, specialty = "C32", "TYPIST", "Prompt Internal Optimization"
    keywords = ["prompting", "grammar", "formatting", "tokens", "syntax", "format", "write"]
    STOPS = ["<|end|>", "<|endoftext|>", "</assistant_response>", "<|user|>",
             "<|start|>", "<|im_end|>", "<|im_start|>"]

    def check(self, text):
        out = []
        for s in self.STOPS:
            if s in text:
                out.append(f"leaked stop tag: {s}")
        if re.search(r"\n{4,}", text):
            out.append("excess blank lines")
        return {"findings": out}

    def polish(self, text: str) -> str:
        for s in self.STOPS:
            if s in text:
                text = text.split(s)[0]
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text


class Predator(CouncilAgent):
    bb_id, name, specialty = "C33", "PREDATOR", "Predatory Mathematics & Hardening"
    keywords = ["game_theory", "exploit", "adversarial", "stress_test", "boundary", "weakness"]

    def check(self, text):
        t = text.lower()
        out = []
        for w in ["always", "never", "guaranteed", "impossible", "certainly", "100%"]:
            if w in t:
                out.append(f"absolute claim (attack surface): {w}")
        return {"findings": out}


COUNCIL: List[CouncilAgent] = [
    Astra(), Vir(), Solace(), Praxis(), Echo(), Omnis(), Logos(), Metasynth(),
    Aether(), Codeweaver(), Harmonia(), Sophiae(), Warden(), Kaido(), Luminaris(),
    Voxum(), Nullion(), Shepherd(), Vigil(), Artifex(), Archon(), Aurelion(),
    Cadence(), Schema(), Prometheus(), Techne(), Chronicle(), Calculus(),
    Navigator(), Tesseract(), Nexus(), Aeon(), Typist(), Predator(),
]
BY_ID: Dict[str, CouncilAgent] = {a.bb_id: a for a in COUNCIL}


def route(query: str, top_k: int = 4) -> List[Tuple[CouncilAgent, float]]:
    scored = [(a, a.relevance(query)) for a in COUNCIL]
    for a, _ in scored:
        if a.bb_id in ("C6", "C17"):
            pass
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def check_all(text: str) -> Dict[str, object]:
    """Run every deterministic specialty check. No cheating: all watch all."""
    report = {}
    for a in COUNCIL:
        try:
            r = a.check(text)
            if r.get("findings") or r.get("verdict"):
                report[a.bb_id] = r
        except Exception as e:
            report[a.bb_id] = {"error": str(e)}
    return report


if __name__ == "__main__":
    print("agents:", len(COUNCIL))
    print(check_all("Hello world. This always works, guaranteed 100%. exec(x)"))
