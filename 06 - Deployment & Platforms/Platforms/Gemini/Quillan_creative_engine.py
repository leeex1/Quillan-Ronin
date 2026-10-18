#!/usr/bin/env python3
"""
Quillan CREATIVE ENGINE v5.3.1
========================================
Algorithmic Creativity Engine

Focus:
- Pattern recombination
- Controlled randomness
- Multi-agent (council) synthesis
- Idea scoring + novelty tracking

This is NOT philosophical — it is a GENERATION SYSTEM.
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import threading


# -------------------------
# MODES
# -------------------------

class CreativityMode(Enum):
    EXPLORATION = "exploration"
    SYNTHESIS = "synthesis"
    RECURSIVE = "recursive"
    BREAKTHROUGH = "breakthrough"


# -------------------------
# DATA STRUCTURES
# -------------------------

@dataclass
class CreativePrompt:
    topic: str
    context: str
    angle: str
    constraints: str = ""


@dataclass
class CreativeIdea:
    idea_id: str
    pattern: str
    description: str
    novelty_score: float
    usefulness_score: float
    combined_score: float


# -------------------------
# ENGINE
# -------------------------

class CreativeEngine:

    def __init__(self):
        self.history: List[CreativeIdea] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger("Quillan.CreativeEngine")

        self.patterns = [
            "recursive loops",
            "pattern inversion",
            "multi-domain fusion",
            "constraint-driven mutation",
            "signal amplification",
            "noise injection",
            "hierarchical abstraction",
            "parallel recombination"
        ]

        self.council_weights = {
            "GENESIS": 1.0,
            "OMNIS": 0.85,
            "CODEWEAVER": 0.9,
            "NULLION": 0.95,
            "CADENCE": 0.8
        }

        self.logger.info("Creative Engine v5.3.1 initialized")


    # -------------------------
    # MAIN GENERATION
    # -------------------------

    def generate_ideas(
        self,
        prompt: CreativePrompt,
        mode: CreativityMode = CreativityMode.EXPLORATION,
        count: int = 5
    ) -> Dict[str, Any]:

        with self.lock:
            session_id = f"creative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            ideas = []
            council = self._generate_council_input(prompt, mode)

            for i in range(count):
                p1, p2 = random.sample(self.patterns, 2)
                pattern = f"{p1} + {p2}"

                novelty = self._score_novelty(pattern)
                usefulness = self._score_usefulness(prompt.topic, pattern)

                idea = CreativeIdea(
                    idea_id=f"{session_id}_{i}",
                    pattern=pattern,
                    description=self._build_description(prompt, pattern, mode),
                    novelty_score=novelty,
                    usefulness_score=usefulness,
                    combined_score=(novelty + usefulness) / 2
                )

                ideas.append(idea)
                self.history.append(idea)

            return {
                "session_id": session_id,
                "mode": mode.value,
                "ideas": [idea.__dict__ for idea in ideas],
                "council": council,
                "top_score": max(i.combined_score for i in ideas)
            }


    # -------------------------
    # COUNCIL
    # -------------------------

    def _generate_council_input(self, prompt, mode):

        outputs = {}

        for name, weight in self.council_weights.items():
            outputs[name] = {
                "weight": weight,
                "focus": f"{name} emphasizes {mode.value} on {prompt.topic}"
            }

        return outputs


    # -------------------------
    # SCORING
    # -------------------------

    def _score_novelty(self, pattern: str) -> float:
        return min(1.0, 0.3 + random.random() * 0.7)


    def _score_usefulness(self, topic: str, pattern: str) -> float:
        overlap = len(set(topic.split()) & set(pattern.split()))
        return min(1.0, 0.4 + overlap * 0.2 + random.random() * 0.4)


    # -------------------------
    # DESCRIPTION
    # -------------------------

    def _build_description(self, prompt, pattern, mode):

        return (
            f"Apply {pattern} to '{prompt.topic}' "
            f"within {prompt.context}. "
            f"Approach via {prompt.angle} using {mode.value} strategy."
        )


    # -------------------------
    # UTILITIES
    # -------------------------

    def get_history(self):
        return [idea.__dict__ for idea in self.history]


# -------------------------
# TEST
# -------------------------

if __name__ == "__main__":

    engine = CreativeEngine()

    prompt = CreativePrompt(
        topic="adaptive AI agents",
        context="dynamic environments",
        angle="multi-agent coordination"
    )

    result = engine.generate_ideas(prompt, CreativityMode.BREAKTHROUGH, 4)

    print("\n=== RESULTS ===")
    for idea in result["ideas"]:
        print(f"- {idea['description']}")
        print(f"  Score: {idea['combined_score']:.2f}")