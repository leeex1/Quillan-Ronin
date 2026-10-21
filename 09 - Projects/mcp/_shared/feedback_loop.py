#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4.0-ONI — Feedback Loop System (_shared/feedback_loop.py)
============================================================================
Implements a local-first self-improvement loop that:
1. Captures reasoning traces from MCP tool executions
2. Runs critic evaluation on each trace
3. Extracts heuristics (successful patterns & anti-patterns)
4. Stores feedback with embeddings for retrieval
5. Injects learned heuristics into system prompts dynamically
6. Enables "learning from mistakes" without model fine-tuning

Architecture:
- ExperienceTraceStore: Persistent trace storage with embeddings
- CriticEngine: Evaluates outputs, extracts heuristics
- FeedbackManager: Orchestrates the full loop (capture → critique → store → inject)
- PromptEnhancer: Injects learned heuristics into system prompts
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ============================================================================
# 1. DATA MODELS
# ============================================================================

@dataclass
class ExperienceTrace:
    """A single execution trace from an MCP tool call."""
    trace_id: str
    timestamp: float
    server_name: str
    tool_name: str
    input_params: Dict[str, Any]
    output_result: Dict[str, Any]
    user_feedback: Optional[Dict[str, Any]] = None
    critic_score: Optional[float] = None
    critic_notes: Optional[str] = None
    extracted_heuristics: List[Dict[str, Any]] = field(default_factory=list)
    vector: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Heuristic:
    """An extracted heuristic from critic evaluation."""
    heuristic_id: str
    pattern_type: str
    description: str
    context_pattern: str
    confidence: float
    usage_count: int
    success_rate: float
    created_at: float
    last_used: Optional[float] = None
    source_trace_ids: List[str] = field(default_factory=list)


@dataclass
class FeedbackEntry:
    """A feedback entry linking a task embedding to critique results."""
    embedding: List[float]
    task_text: str
    feedback_type: str
    heuristics: List[Dict[str, Any]]
    timestamp: float
    source_trace_id: str


# ============================================================================
# 2. EXPERIENCE TRACE STORE
# ============================================================================

class ExperienceTraceStore:
    """Persistent storage for experience traces with vector similarity search."""

    def __init__(self, storage_dir: str = ".quillan_feedback"):
        self.storage_dir = Path(os.path.join(os.path.expanduser("~"), ".quillan_feedback"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.traces_file = self.storage_dir / "traces.jsonl"
        self.heuristics_file = self.storage_dir / "heuristics.jsonl"
        self.feedback_file = self.storage_dir / "feedback.jsonl"
        self._traces: List[ExperienceTrace] = []
        self._heuristics: List[Heuristic] = []
        self._feedback: List[FeedbackEntry] = []
        self._load_all()

    def _load_all(self):
        if self.traces_file.exists():
            with open(self.traces_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self._traces.append(ExperienceTrace(**data))
                        except Exception:
                            pass
        if self.heuristics_file.exists():
            with open(self.heuristics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self._heuristics.append(Heuristic(**data))
                        except Exception:
                            pass
        if self.feedback_file.exists():
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self._feedback.append(FeedbackEntry(**data))
                        except Exception:
                            pass

    def store_trace(self, trace: ExperienceTrace) -> str:
        self._traces.append(trace)
        self._append_to_file(self.traces_file, trace)
        return trace.trace_id

    def store_heuristic(self, heuristic: Heuristic) -> str:
        self._heuristics.append(heuristic)
        self._append_to_file(self.heuristics_file, heuristic)
        return heuristic.heuristic_id

    def store_feedback(self, feedback: FeedbackEntry) -> str:
        self._feedback.append(feedback)
        self._append_to_file(self.feedback_file, feedback)
        return feedback.heuristics[0].heuristic_id if feedback.heuristics else ""

    def _append_to_file(self, path: Path, obj: Any):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(obj)) + "\n")

    def get_all_traces(self) -> List[ExperienceTrace]:
        return list(self._traces)

    def get_all_heuristics(self) -> List[Heuristic]:
        return list(self._heuristics)

    def get_all_feedback(self) -> List[FeedbackEntry]:
        return list(self._feedback)

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[FeedbackEntry, float]]:
        results = []
        for entry in self._feedback:
            if not entry.embedding:
                continue
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            results.append((entry, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-8
        norm_b = math.sqrt(sum(x * x for x in b)) or 1e-8
        return dot / (norm_a * norm_b)


# ============================================================================
# 3. TEXT EMBEDDING
# ============================================================================

class TextEmbedder:
    """Simple text embedding without external ML dependencies."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._vocab: Dict[str, int] = {}
        self._vocab_size = 0

    def _get_vocab_index(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._vocab_size
            self._vocab_size += 1
        return self._vocab[token] % self.dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            gram = text_lower[i:i+3]
            idx = self._get_vocab_index(gram)
            vec[idx] += 1.0
        words = text_lower.split()
        for word in words:
            idx = self._get_vocab_index(f"w:{word}")
            vec[idx] += 2.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1e-8
        vec = [v / norm for v in vec]
        return vec


# ============================================================================
# 4. CRITIC ENGINE
# ============================================================================

class CriticEngine:
    """Evaluates tool outputs and extracts heuristics from traces."""

    def __init__(self, store: ExperienceTraceStore):
        self.store = store
        self.embedder = TextEmbedder()

    def evaluate_trace(self, trace: ExperienceTrace) -> Dict[str, Any]:
        result = trace.output_result or {}
        input_params = trace.input_params or {}

        is_success = result.get("status", "").lower() in ("ok", "success", "completed")
        confidence = self._calculate_confidence(trace)

        heuristics = self._extract_heuristics(trace, is_success, confidence)

        for h in heuristics:
            self.store.store_heuristic(h)

        embedding = self.embedder.embed(
            f"{trace.server_name} {trace.tool_name} {str(trace.input_params)[:200]}"
        )
        feedback_entry = FeedbackEntry(
            embedding=embedding,
            task_text=str(trace.input_params)[:200],
            feedback_type="positive" if is_success else "negative",
            heuristics=[h for h in heuristics],
            timestamp=time.time(),
            source_trace_id=trace.trace_id,
        )
        self.store.store_feedback(feedback_entry)

        return {
            "trace_id": trace.trace_id,
            "confidence": confidence,
            "is_success": is_success,
            "heuristics_extracted": len(heuristics),
            "heuristics": [h for h in heuristics],
        }

    def _calculate_confidence(self, task: ExperienceTrace) -> float:
        score = 0.5
        output = task.output_result or {}
        input_params = task.input_params or {}

        if output.get("status", "").lower() in ("ok", "success", "completed"):
            score += 0.3
        elif output.get("status", "").lower() in ("error", "failed", "blocked"):
            score -= 0.3

        if "error" in str(output).lower():
            score -= 0.2

        if input_params:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _extract_heuristics(
        self, trace: ExperienceTrace, is_success: bool, confidence: float
    ) -> List[Heuristic]:
        heuristics = []
        input_params = trace.input_params or {}
        output = trace.output_result or {}

        if is_success and confidence > 0.7:
            heuristics.append(Heuristic(
                heuristic_id=f"heur_{trace.trace_id}_success",
                pattern_type="success",
                description=f"Successful execution of {trace.server_name}.{trace.tool_name} with input keys: {', '.join(sorted(input_params.keys())[:5])}",
                context_pattern=f"Input keys: {', '.join(sorted(input_params.keys())[:5])}",
                confidence=confidence,
                usage_count=1,
                success_rate=1.0,
                created_at=time.time(),
                source_trace_ids=[trace.trace_id],
            ))

        if not is_success and confidence < 0.5:
            error_msg = str(output.get("error", output.get("error_type", "unknown")))
            heuristics.append(Heuristic(
                heuristic_id=f"heur_{trace.trace_id}_failure",
                pattern_type="failure",
                description=f"Failure pattern: {error_msg} when calling {trace.server_name}.{trace.tool_name}",
                context_pattern=f"Input keys: {', '.join(sorted(input_params.keys())[:5])}",
                confidence=1.0 - confidence,
                usage_count=1,
                success_rate=0.0,
                created_at=time.time(),
                source_trace_ids=[trace.trace_id],
            ))

        if is_success and confidence > 0.8:
            heuristics.append(Heuristic(
                heuristic_id=f"heur_{trace.trace_id}_best_practice",
                pattern_type="best_practice",
                description=f"Best practice: {trace.server_name}.{trace.tool_name} works well with input keys: {', '.join(sorted(input_params.keys())[:5])}",
                context_pattern=f"Input keys: {', '.join(sorted(input_params.keys())[:5])}",
                confidence=confidence,
                usage_count=1,
                success_rate=1.0,
                created_at=time.time(),
                source_trace_ids=[trace.trace_id],
            ))

        return heuristics


# ============================================================================
# 5. FEEDBACK MANAGER
# ============================================================================

class FeedbackManager:
    """Feedback loop manager that coordinates trace capture, critique, and storage."""

    def __init__(self, store: ExperienceTraceStore, critic: CriticEngine):
        self.store = store
        self.critic = critic

    def process_tool_result(
        self,
        server_name: str,
        tool_name: str,
        input_params: Dict[str, Any],
        output_result: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trace = ExperienceTrace(
            trace_id=f"trace_{int(time.time()*1000)}_{hash(str(input_params)) % 10000}",
            timestamp=time.time(),
            server_name=server_name,
            tool_name=tool_name,
            input_params=input_params,
            output_result=output_result,
            user_feedback=user_feedback,
        )
        self.store.store_trace(trace)
        critique_result = self.critic.evaluate_trace(trace)
        return {
            "trace_id": trace.trace_id,
            "critique": critique_result,
            "feedback_stored": True,
        }

    def get_relevant_feedback(
        self, query_text: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        embedder = TextEmbedder()
        query_embedding = embedder.embed(query_text)
        results = self.store.search_similar(query_embedding, top_k=top_k)

        formatted = []
        for entry, similarity in results:
            formatted.append({
                "query": entry.task_text,
                "feedback_type": entry.feedback_type,
                "heuristics": entry.heuristics,
                "similarity": round(similarity, 4),
                "timestamp": entry.timestamp,
            })
        return formatted


# ============================================================================
# 6. PROMPT ENHANCER
# ============================================================================

class PromptEnhancer:
    def __init__(self, feedback_manager: FeedbackManager):
        self.fm = feedback_manager

    def enhance_prompt(
        self,
        base_prompt: str,
        query_text: str,
        max_heuristics: int = 5,
    ) -> str:
        feedback_results = self.fm.get_relevant_feedback(query_text, top_k=max_heuristics)

        if not feedback_results:
            return base_prompt

        enhancement_parts = ["\n\n=== LEARNED HEURISTICS (from prior experience) ==="]
        for fb in feedback_results:
            h = fb["heuristics"][0] if fb["heuristics"] else {}
            if isinstance(h, dict):
                desc = h.get("description", "N/A")
            elif hasattr(h, "description"):
                desc = h.description
            else:
                desc = str(h)[:100] if h else "N/A" if isinstance(h, dict) else h.description
            if fb["feedback_type"] == "positive":
                enhancement_parts.append(f"  ✓ SUCCESS: {desc}")
            elif fb["feedback_type"] == "negative":
                enhancement_parts.append(f"  ✗ FAILURE: {desc}")
            else:
                enhancement_parts.append(f"  ⚠ NOTE: {desc}")
        enhancement_parts.append("=== END HEURISTICS ===\n")

        enhancement = "\n".join(enhancement_parts)
        return base_prompt + "\n" + enhancement

    def enhance_with_heuristics(
        self,
        base_prompt: str,
        heuristics: List[Heuristic],
    ) -> str:
        if not heuristics:
            return base_prompt
        enhancement_parts = ["\n=== LEARNED HEURISTICS ==="]
        for h in heuristics:
            enhancement_parts.append(f"  • {h.description} (confidence: {h.confidence:.2f})")
        enhancement_parts.append("=== END ===\n")
        return base_prompt + "\n" + "\n".join(enhancement_parts)


# ============================================================================
# 7. GLOBAL INSTANCE
# ============================================================================

_feedback_manager: Optional[FeedbackManager] = None


def get_feedback_manager() -> FeedbackManager:
    global _feedback_manager
    if _feedback_manager is None:
        store = ExperienceTraceStore()
        critic = CriticEngine(store)
        _feedback_manager = FeedbackManager(store, critic)
    return _feedback_manager


def get_thermodynamic_feedback() -> Dict[str, Any]:
    return {
        "thermodynamic_state": "stable",
        "entropy_level": 0.3,
        "recommendation": "Continue current reasoning trajectory",
    }


__all__ = [
    "ExperienceTrace",
    "Heuristic",
    "FeedbackEntry",
    "ExperienceTraceStore",
    "TextEmbedder",
    "CriticEngine",
    "FeedbackManager",
    "PromptEnhancer",
    "get_feedback_manager",
    "get_thermodynamic_feedback",
]