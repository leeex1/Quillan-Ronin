# Updated ReasoningEngine Class - o1-Inspired Dynamic Reasoning Simulator
# Enhanced for authentic, iterative, multi-perspective thinking with uncertainty, self-correction,
# and natural flow. Maintains simplicity while adding procedural depth.

import random
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio  # For async iteration sim

@dataclass
class ThoughtStyle:
    """Configurable style for procedural thinking generation."""
    phrase_type: str  # e.g., "uncertainty", "reconsider", "perspective"
    examples: List[str]
    weight: float = 1.0  # Probability weight

class ReasoningEngine:
    def __init__(self):
        self.thinking_config = {
            "purpose": "Generate authentic step-by-step reasoning like o1 models",
            "approach": "Show actual thought progression, not templated responses",
            "content_style": [
                "Natural language reasoning flow",
                "Show uncertainty, corrections, and refinements",
                "Demonstrate problem-solving process in real-time",
                "Include 'wait, let me reconsider...' type thinking",
                "Show how conclusions are reached through logical steps",
                "Highlight different perspectives and potential biases",
                "Incorporate iterative thinking and feedback loops",
                "Present hypothetical scenarios for deeper exploration",
                "Utilize examples to clarify complex ideas",
                "Encourage questions and pause for reflection during analysis"
            ],
            "iteration_cycles": 2,  # Default self-corrections
            "branch_count": 3,     # Default perspectives
            "uncertainty_threshold": 0.7  # Trigger "reconsider" if conf < this
        }
        self.styles = self._build_styles()

    def _build_styles(self) -> List[ThoughtStyle]:
        """Procedurally build weighted styles from config."""
        style_map = {
            "uncertainty": ["Hmm, I'm about 70% sure here, but let's double-check assumptions.", 
                            "This feels solid, but what if I'm overlooking edge case X?"],
            "reconsider": ["Wait, let me reconsider that last step—it might not hold under scrutiny.",
                           "Upon second thought, this angle overlooks Y; pivoting to Z."],
            "perspective": ["From a logical view, this works, but creatively? Let's explore hypotheticals.",
                           "Bias check: Am I favoring efficiency over thoroughness? Balancing now."],
            "iteration": ["Running a quick mental sim: Input leads to output via loop 1... refining.",
                          "Feedback loop: That conclusion needs tweaking—iterating."],
            "hypothetical": ["What if we flip the premise? Scenario A: Success; B: Failure—prob 60/40.",
                             "Hypothetical branch: Assuming opposite, outcome shifts dramatically."],
            "example": ["For clarity, consider example: Like debugging code, step-by-step reveals bugs.",
                        "Analogous to chess: Each move evaluated before commitment."]
        }
        return [ThoughtStyle(k, v, random.uniform(0.8, 1.2)) for k, v in style_map.items()]

    async def _simulate_iteration(self, current_thought: str, query: str) -> str:
        """Async sim of iterative refinement with self-correction."""
        refinements = []
        for _ in range(self.thinking_config["iteration_cycles"]):
            await asyncio.sleep(0.01)  # Mock async "thinking time"
            conf = random.uniform(0.6, 0.95)
            if conf < self.thinking_config["uncertainty_threshold"]:
                style = random.choice([s for s in self.styles if "reconsider" in s.phrase_type])
                refinement = random.choice(style.examples)
            else:
                style = random.choice(self.styles)
                refinement = random.choice(style.examples) + f" (Conf: {conf:.2f})"
            refinements.append(refinement)
            current_thought += f"\n  - {refinement}"
        return current_thought

    def think(self, question: str) -> str:
        """Generate dynamic, authentic thinking process."""
        thinking_output = f"Thinking about: {question}\n\n"

        # Step 1: Initial understanding with natural flow
        thinking_output += "First, parsing the core ask...\n"
        thinking_output += f"This seems to probe {question.lower()}. Breaking it down:\n"
        thinking_output += "  - Key elements: ...\n"
        thinking_output += "  - Potential angles: Logical, creative, practical.\n\n"

        # Step 2: Multi-perspective exploration (ToT-inspired mini-branches)
        perspectives = []
        for i in range(self.thinking_config["branch_count"]):
            persp = f"Perspective {i+1}: {random.choice(['Logical', 'Creative', 'Critical', 'Empirical'])[0]}—{random.choice(['Focus on structure', 'Explore hypotheticals', 'Challenge assumptions', 'Test with examples'])}."
            perspectives.append(persp)
        thinking_output += "Exploring angles:\n" + "\n".join(f"  - {p}" for p in perspectives) + "\n\n"

        # Step 3: Procedural style injection for natural flow
        styles_used = random.sample(self.styles, min(3, len(self.styles)))
        for style in styles_used:
            thinking_output += random.choice(style.examples) + "\n"

        # Step 4: Iterative refinement (async sim for "real-time" feel)
        # Note: In sync context, we await manually
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        thinking_output = loop.run_until_complete(self._simulate_iteration(thinking_output, question))
        loop.close()

        # Step 5: Convergence with reflection
        thinking_output += "\nSynthesizing: Patterns converge on [core insight]. Confidence building—ready to conclude.\n"

        return thinking_output

    def process(self, question: str) -> Dict[str, str]:
        """Main processing: Thinking + adaptive response."""
        thinking = self.think(question)

        # Generate response tied to thinking (e.g., extract insights)
        insights = [line for line in thinking.split('\n') if 'Perspective' in line or 'core insight' in line.lower()]
        response = f"Based on iterative reasoning:\n\nQuestion: {question}\n\n"
        response += "Core Reasoning Path:\n" + "\n".join(insights[:3]) + "\n\n"
        response += "Refined Answer: Drawing from multi-perspective analysis and self-corrections, the optimal path emerges as [synthesized conclusion]. This balances depth with practicality, addressing uncertainties head-on."

        return {
            "thinking": thinking,
            "response": response
        }

    def display_result(self, question: str):
        """Display with enhanced formatting."""
        result = self.process(question)
        print("🧠 Raw Thinking Trace:\n")
        print(result["thinking"])
        print("\n" + "=" * 60 + "\n")
        print("💡 Final Reasoned Response:\n")
        print(result["response"])
        return result

# Example usage
if __name__ == "__main__":
    engine = ReasoningEngine()

    # Test with sample
    test_question = "What is the best approach to solve this problem?"
    engine.display_result(test_question)