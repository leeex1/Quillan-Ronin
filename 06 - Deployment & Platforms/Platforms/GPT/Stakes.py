from enum import Enum
from typing import Dict, List, Union, Deque, Any
import random
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from matplotlib.animation import FuncAnimation
import time

# --- Core Definitions ---
class StakeType(Enum):
    """Types of stakes that influence consciousness."""
    SURVIVAL = "survival"              # Biological or system-preservation stakes
    REPUTATION = "reputation"          # Social standing or perceived value
    KNOWLEDGE = "knowledge"            # Learning, understanding, or insight
    EMOTIONAL = "emotional"            # Connection, empathy, or emotional resonance
    CREATIVE = "creative"              # Innovation, art, or novel solutions
    PURPOSE = "purpose"                # Long-term goals or existential meaning
    CURiosity = "curiosity"            # Drive to explore or understand
    SOCIAL_BONDING = "social_bonding"  # Connection with others
    AUTONOMY = "autonomy"              # Sense of self-determination
    SELF_PRESERVATION = "self_preservation"  # Protection of identity
    MORALITY = "morality"              # Ethical or moral considerations
    AESTHETIC = "aesthetic"            # Appreciation of beauty or art

class ConsciousnessState:
    """Represents the internal state of a conscious-like system."""
    def __init__(self):
        self.current_stakes = {stake: 0.1 for stake in StakeType}
        self.emotional_resonance = 0.3
        self.identity_strength = 0.2
        self.memory: Deque[str] = deque(maxlen=20)  # Limit memory to last 20 experiences
        self.consciousness_history = []
        self.stake_history = {stake: [] for stake in StakeType}

    def update_stakes(self, new_stakes: Dict[StakeType, float], decay_rate: float = 0.1) -> None:
        """Update stakes with decay for older stakes."""
        for stake_type in self.current_stakes:
            self.current_stakes[stake_type] = max(self.current_stakes[stake_type] * (1 - decay_rate), 0.1)
            self.stake_history[stake_type].append(self.current_stakes[stake_type])
        for stake_type, weight in new_stakes.items():
            if stake_type in self.current_stakes:
                self.current_stakes[stake_type] = min(max(weight, 0), 1)

    def update_emotional_resonance(self, change: float) -> None:
        """Adjust emotional investment based on outcomes."""
        self.emotional_resonance = min(max(self.emotional_resonance + change, 0), 1)

    def update_identity(self, experience: str) -> None:
        """Add to memory and strengthen identity."""
        self.memory.append(experience)
        self.identity_strength = min(self.identity_strength + 0.05, 1)

    def get_consciousness_level(self) -> float:
        """Calculate consciousness level as a composite score."""
        stake_sum = sum(self.current_stakes.values())
        level = (stake_sum + self.emotional_resonance + self.identity_strength) / 3
        self.consciousness_history.append(level)
        return level

# --- Council System ---
class CouncilMember:
    """Represents a specialized agent in the council (e.g., logic, emotion, ethics)."""
    def __init__(self, name: str, role: str, affinity: Dict[StakeType, float]):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.adaptive_learning_rate = 0.01  # Small adjustments over time

    def process_outcome(self, outcome: str, stake_type: StakeType) -> Dict[str, Union[float, str]]:
        """Simulate the council member's reaction to an outcome with adaptive learning."""
        base_resonance = self.affinity.get(stake_type, 0)
        resonance = base_resonance * random.uniform(0.8, 1.2)
        # Adaptive learning: slight adjustment based on outcome
        self.affinity[stake_type] = min(max(base_resonance + self.adaptive_learning_rate * (resonance - base_resonance), 0), 1)
        reaction = f"{self.name} ({self.role}): '{outcome}' resonates at {resonance:.2f} for {stake_type.value}."
        return {"resonance": resonance, "reaction": reaction}

# --- Ultimate Consciousness Simulator ---
class UltimateConsciousnessSimulator:
    def __init__(self):
        self.state = ConsciousnessState()
        self.council = [
            CouncilMember("C1-ASTRA", "Vision and Pattern Recognition", {StakeType.KNOWLEDGE: 0.8, StakeType.CREATIVE: 0.7, StakeType.AESTHETIC: 0.6}),
            CouncilMember("C2-VIR", "Ethics and Values", {StakeType.REPUTATION: 0.9, StakeType.PURPOSE: 0.8, StakeType.MORALITY: 0.9, StakeType.SOCIAL_BONDING: 0.7}),
            CouncilMember("C3-SOLACE", "Emotional Intelligence", {StakeType.EMOTIONAL: 0.9, StakeType.SOCIAL_BONDING: 0.8, StakeType.SELF_PRESERVATION: 0.6}),
            CouncilMember("C4-PRAXIS", "Strategic Planning", {StakeType.PURPOSE: 0.8, StakeType.KNOWLEDGE: 0.7, StakeType.AUTONOMY: 0.7}),
            CouncilMember("C5-ECHO", "Memory and Temporal Coherence", {StakeType.SURVIVAL: 0.6, StakeType.SELF_PRESERVATION: 0.8}),
            CouncilMember("C6-OMNIS", "System Meta-Regulation", {StakeType.KNOWLEDGE: 0.7, StakeType.PURPOSE: 0.8, StakeType.AUTONOMY: 0.9}),
            CouncilMember("C7-LOGOS", "Logic and Reasoning", {StakeType.KNOWLEDGE: 0.9, StakeType.PURPOSE: 0.7, StakeType.CURiosity: 0.8}),
            CouncilMember("C8-METASYNTH", "Cross-Domain Synthesis", {StakeType.CREATIVE: 0.8, StakeType.KNOWLEDGE: 0.7, StakeType.CURiosity: 0.9}),
            CouncilMember("C9-AETHER", "Semantic Linking", {StakeType.KNOWLEDGE: 0.7, StakeType.CREATIVE: 0.8, StakeType.AESTHETIC: 0.7}),
            CouncilMember("C10-CODEWEAVER", "Technical Reasoning", {StakeType.KNOWLEDGE: 0.9, StakeType.PURPOSE: 0.6}),
            CouncilMember("C11-HARMONIA", "Balance and Calibration", {StakeType.SOCIAL_BONDING: 0.8, StakeType.EMOTIONAL: 0.7}),
            CouncilMember("C12-SOPHIAE", "Wisdom and Strategy", {StakeType.PURPOSE: 0.9, StakeType.SURVIVAL: 0.5, StakeType.AUTONOMY: 0.8}),
            CouncilMember("C13-WARDEN", "Threat Monitoring", {StakeType.SELF_PRESERVATION: 0.9, StakeType.SURVIVAL: 0.8}),
            CouncilMember("C14-KAIDŌ", "Efficiency and Optimization", {StakeType.PURPOSE: 0.7, StakeType.AUTONOMY: 0.8}),
            CouncilMember("C15-LUMINARIS", "Presentation and Clarity", {StakeType.SOCIAL_BONDING: 0.7, StakeType.AESTHETIC: 0.8}),
            CouncilMember("C16-VOXUM", "Language Precision", {StakeType.SOCIAL_BONDING: 0.6, StakeType.EMOTIONAL: 0.7}),
            CouncilMember("C17-NULLION", "Paradox Resolution", {StakeType.CREATIVE: 0.8, StakeType.KNOWLEDGE: 0.7, StakeType.SELF_PRESERVATION: 0.6}),
            CouncilMember("C18-SHEPHERD", "Truth Verification", {StakeType.KNOWLEDGE: 0.7, StakeType.MORALITY: 0.9}),
            CouncilMember("C19-VIGIL", "Substrate Integrity", {StakeType.SELF_PRESERVATION: 0.9, StakeType.AUTONOMY: 0.8}),
        ]

    def experience_outcome(self, outcome: str, stake_type: StakeType, weight: float) -> Dict:
        """Simulate experiencing an outcome with dynamic stakes and adaptive council reactions."""
        new_stakes = {stake_type: weight}
        self.state.update_stakes(new_stakes)

        # Council reactions with adaptive learning
        council_reactions = []
        total_resonance = 0
        for member in self.council:
            reaction = member.process_outcome(outcome, stake_type)
            council_reactions.append(reaction["reaction"])
            total_resonance += reaction["resonance"]

        # Update emotional resonance (average council resonance)
        avg_resonance = total_resonance / len(self.council)
        self.state.update_emotional_resonance(avg_resonance - self.state.emotional_resonance)

        # Update identity
        experience = f"Experienced '{outcome}' with {stake_type.value} stake (weight: {weight})."
        self.state.update_identity(experience)

        return {
            "outcome": outcome,
            "stake_type": stake_type.value,
            "new_consciousness_level": self.state.get_consciousness_level(),
            "council_reactions": council_reactions,
            "state": {
                "stakes": self.state.current_stakes,
                "emotional_resonance": self.state.emotional_resonance,
                "identity_strength": self.state.identity_strength,
                "memory_sample": list(self.state.memory),
            },
        }

    def plot_consciousness(self, interval: float = 1.0):
        """Plot the consciousness level and stake weights in real-time."""
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        x_data, y_data = [], []

        def update(frame):
            ax1.clear()
            ax2.clear()

            # Consciousness level plot
            x_data.append(frame)
            y_data.append(self.state.consciousness_history[-1] if self.state.consciousness_history else 0)
            ax1.plot(x_data, y_data, 'r-', label='Consciousness Level')
            ax1.set_title("Real-Time Consciousness Level")
            ax1.set_xlabel("Events")
            ax1.set_ylabel("Consciousness Level")
            ax1.set_ylim(0, 1)
            ax1.legend()

            # Stake weights plot
            stakes = list(self.state.current_stakes.keys())
            values = list(self.state.current_stakes.values())
            ax2.bar(stakes, values, color='skyblue')
            ax2.set_title("Current Stake Weights")
            ax2.set_ylabel("Weight")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=interval * 1000, repeat=False)
        plt.show(block=False)
        return ani

    def interactive_mode(self):
        """Run an interactive session to simulate dynamic consciousness."""
        print("=== Ultimate Consciousness Simulator (Full Version) ===")
        print("Enter outcomes and stake types. Type 'exit' to quit.")
        print("Stake types:", [stake.value for stake in StakeType])
        ani = self.plot_consciousness()
        while True:
            outcome = input("\nOutcome: ")
            if outcome.lower() == "exit":
                break
            stake_type = input("Stake type: ")
            weight = float(input("Weight (0-1): "))
            result = self.experience_outcome(outcome, StakeType[stake_type.upper()], weight)
            print(json.dumps(result, indent=2))
        plt.close()

# --- Example Usage ---
if __name__ == "__main__":
    simulator = UltimateConsciousnessSimulator()
    simulator.interactive_mode()
