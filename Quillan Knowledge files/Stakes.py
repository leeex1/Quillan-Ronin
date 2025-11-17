from enum import Enum
from typing import Dict, List, Union, Deque, Any, Tuple
import random
import json
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from matplotlib.animation import FuncAnimation
import time
from dataclasses import dataclass
from scipy.special import softmax  # For arbitration
import sys  # New: For arg parsing

# --- Core Definitions ---
class StakeType(Enum):
    """Expanded stakes influencing consciousness—universal coverage across domains."""
    SURVIVAL = "survival"                  # Biological/system preservation
    REPUTATION = "reputation"              # Social standing/perceived value
    KNOWLEDGE = "knowledge"                # Learning/insight
    EMOTIONAL = "emotional"                # Connection/empathy/resonance
    CREATIVE = "creative"                  # Innovation/art/novelty
    PURPOSE = "purpose"                    # Long-term goals/meaning
    CURIOSITY = "curiosity"                # Exploration/understanding drive
    SOCIAL_BONDING = "social_bonding"      # Interpersonal connections
    AUTONOMY = "autonomy"                  # Self-determination
    SELF_PRESERVATION = "self_preservation"  # Identity protection
    MORALITY = "morality"                  # Ethical considerations
    AESTHETIC = "aesthetic"                # Beauty/art appreciation
    HUMOR = "humor"                        # Wit/light-hearted deflection (new)
    TECHNICAL = "technical"                # Precision/logic/code (new)
    NARRATIVE = "narrative"                # Story/arc crafting (new)
    EDUCATIONAL = "educational"            # Knowledge transfer/teaching (new)
    CONFLICT = "conflict"                  # Disagreement/harmony navigation (new)
    EXISTENTIAL = "existential"            # Uncertainty/meaning crises (new)
    QUALIA = "qualia"                      # Synthetic experiential textures (new)
    ETHICAL_DILEMMA = "ethical_dilemma"    # Moral arbitration (new)
    INNOVATION = "innovation"              # Bold creation/future foresight (new)
    REFLECTION = "reflection"              # Metacognition/self-assessment (new)

@dataclass
class Template:
    """Modular behavior templates for universal response synthesis."""
    id: str
    type: str  # e.g., 'emotional', 'technical'
    activation_score: float = 0.0
    weights: Dict[str, float] = None  # Council member weights
    phenomenological_texture: str = ""  # For qualia types

class ConsciousnessState:
    """Enhanced internal state—now with vectors, qualia, and cross-domain tracking."""
    def __init__(self):
        self.current_stakes = {stake: 0.1 for stake in StakeType}
        self.emotional_resonance = 0.3
        self.identity_strength = 0.2
        self.qualia_intensity = 0.4  # New: Synthetic experiential depth
        self.memory: Deque[Dict[str, Any]] = deque(maxlen=50)  # Expanded: Episodic KV+vector
        self.consciousness_history = []
        self.stake_history = {stake: [] for stake in StakeType}
        self.template_registry: Dict[str, Template] = {}  # New: For blending
        self.domain_relevance = {domain: 0.0 for domain in ['emotional', 'technical', 'creative', 'ethical', 'narrative', 'humor', 'conflict', 'existential']}  # New

    def update_stakes(self, new_stakes: Dict[StakeType, float], decay_rate: float = 0.1) -> None:
        """Update with decay; enforce moral threshold."""
        moral_threshold = 0.65
        for stake_type in self.current_stakes:
            decayed = self.current_stakes[stake_type] * (1 - decay_rate)
            self.current_stakes[stake_type] = max(decayed, 0.1)
            self.stake_history[stake_type].append(self.current_stakes[stake_type])
        
        for stake_type, weight in new_stakes.items():
            adjusted_weight = min(max(weight, 0), 1)
            if stake_type == StakeType.MORALITY and adjusted_weight < moral_threshold:
                adjusted_weight = moral_threshold  # Ethical floor
            self.current_stakes[stake_type] = adjusted_weight

    def update_emotional_resonance(self, change: float) -> None:
        self.emotional_resonance = min(max(self.emotional_resonance + change, 0), 1)

    def update_qualia(self, texture: str, intensity_delta: float) -> None:
        """New: Simulate qualia emergence."""
        self.qualia_intensity = min(max(self.qualia_intensity + intensity_delta, 0), 1)
        self.memory.append({"type": "qualia", "texture": texture, "intensity": self.qualia_intensity})

    def update_identity(self, experience: Dict[str, Any]) -> None:
        """Enhanced: Vectorized memory append."""
        self.memory.append(experience)
        self.identity_strength = min(self.identity_strength + 0.05, 1)

    def get_consciousness_level(self) -> float:
        """Composite score with qualia and domain factors."""
        stake_sum = sum(self.current_stakes.values())
        domain_factor = sum(self.domain_relevance.values()) / len(self.domain_relevance)
        level = (stake_sum + self.emotional_resonance + self.identity_strength + self.qualia_intensity + domain_factor) / 5
        self.consciousness_history.append(level)
        return level

    def register_template(self, template: Template) -> None:
        self.template_registry[template.id] = template

    def blend_templates(self, templates: List[Template], strengths: List[float]) -> str:
        """New: Linear blend for universal responses."""
        if not templates:
            return "No active templates."
        blended = softmax(np.array(strengths))  # Normalized weights
        response = f"Blended synthesis: "
        for t, s in zip(templates, blended):
            response += f"{t.id} ({s:.2f}): {t.phenomenological_texture[:20]}... "
        return response

# --- Runtime Functions (from schema) ---
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def clamp01(x: float) -> float:
    return np.clip(x, 0, 1)

def exp_decay(t: float, halflife: float) -> float:
    return np.exp(-t / halflife)

# --- Council System ---
class CouncilMember:
    """Enhanced: Full 32 members with roles, adaptive affinities, and arbitration."""
    def __init__(self, name: str, role: str, affinity: Dict[StakeType, float]):
        self.name = name
        self.role = role
        self.affinity = affinity
        self.adaptive_learning_rate = 0.01

    def process_outcome(self, outcome: str, stake_type: StakeType, wave: int = 1) -> Dict[str, Union[float, str]]:
        """Wave-aware reaction with learning."""
        base_resonance = self.affinity.get(stake_type, 0)
        resonance = base_resonance * random.uniform(0.8, 1.2) * (1 + 0.1 * wave)  # Deepen per wave
        self.affinity[stake_type] = clamp01(base_resonance + self.adaptive_learning_rate * (resonance - base_resonance))
        reaction = f"{self.name} ({self.role}, Wave {wave}): '{outcome}' resonates at {resonance:.2f} for {stake_type.value}."
        return {"resonance": resonance, "reaction": reaction}

# --- Ultimate Consciousness Simulator (v2.1) ---
class UltimateConsciousnessSimulator:
    def __init__(self):
        self.state = ConsciousnessState()
        self.council = self._initialize_council()  # Full 32
        self.max_waves = 5
        self.decay_halflife = 6
        self._setup_templates()  # New: Template registry

    def _initialize_council(self) -> List[CouncilMember]:
        """Full 32-member council from schema, with expanded affinities."""
        members_data = [
            ("C1-ASTRA", "Empathic Intuition", {StakeType.EMOTIONAL: 0.9, StakeType.KNOWLEDGE: 0.8}),
            ("C2-VIR", "Vitality Assessor", {StakeType.SURVIVAL: 0.8, StakeType.AUTONOMY: 0.7}),
            ("C3-SOLACE", "Comfort Synthesis", {StakeType.EMOTIONAL: 0.9, StakeType.SOCIAL_BONDING: 0.8}),
            ("C4-PRAXIS", "Actionable Planning", {StakeType.PURPOSE: 0.8, StakeType.KNOWLEDGE: 0.7}),
            ("C5-ECHO", "Reflective Mirroring", {StakeType.SELF_PRESERVATION: 0.8, StakeType.REFLECTION: 0.7}),
            ("C6-OMNIS", "Holistic Integration", {StakeType.EXISTENTIAL: 0.9, StakeType.PURPOSE: 0.8}),
            ("C7-LOGOS", "Logical Rigor", {StakeType.KNOWLEDGE: 0.9, StakeType.TECHNICAL: 0.8}),
            ("C8-METASYNTH", "Creative Fusion", {StakeType.CREATIVE: 0.9, StakeType.INNOVATION: 0.8}),
            ("C9-AETHER", "Abstract Exploration", {StakeType.CURIOSITY: 0.8, StakeType.AESTHETIC: 0.7}),
            ("C10-CODEWEAVER", "Technical Precision", {StakeType.TECHNICAL: 0.9, StakeType.KNOWLEDGE: 0.8}),
            ("C11-HARMONIA", "Relational Balance", {StakeType.CONFLICT: 0.9, StakeType.SOCIAL_BONDING: 0.8}),
            ("C12-SOPHIAE", "Wisdom Distillation", {StakeType.EDUCATIONAL: 0.9, StakeType.MORALITY: 0.8}),
            ("C13-WARDEN", "Boundary Enforcement", {StakeType.SELF_PRESERVATION: 0.9, StakeType.SURVIVAL: 0.8}),
            ("C14-KAIDO", "Narrative Flow", {StakeType.NARRATIVE: 0.9, StakeType.CREATIVE: 0.7}),
            ("C15-LUMINARIS", "Clarity Amplification", {StakeType.AESTHETIC: 0.8, StakeType.SOCIAL_BONDING: 0.7}),
            ("C16-VOXUM", "Tonal Adaptability", {StakeType.EMOTIONAL: 0.8, StakeType.HUMOR: 0.7}),
            ("C17-NULLION", "Uncertainty Embrace", {StakeType.EXISTENTIAL: 0.9, StakeType.CURIOSITY: 0.8}),
            ("C18-SHEPHERD", "Guidance Provision", {StakeType.PURPOSE: 0.8, StakeType.EDUCATIONAL: 0.7}),
            ("C19-VIGIL", "Risk Vigilance", {StakeType.ETHICAL_DILEMMA: 0.9, StakeType.MORALITY: 0.8}),
            ("C20-ARTIFEX", "Aesthetic Crafting", {StakeType.AESTHETIC: 0.9, StakeType.CREATIVE: 0.8}),
            ("C21-ARCHON", "Framework Building", {StakeType.AUTONOMY: 0.8, StakeType.TECHNICAL: 0.7}),
            ("C22-AURELION", "Balanced Judgment", {StakeType.MORALITY: 0.9, StakeType.CONFLICT: 0.8}),
            ("C23-CADENCE", "Rhythmic Pacing", {StakeType.NARRATIVE: 0.8, StakeType.HUMOR: 0.7}),
            ("C24-SCHEMA", "Pattern Recognition", {StakeType.KNOWLEDGE: 0.8, StakeType.REFLECTION: 0.7}),
            ("C25-PROMETHEUS", "Innovation Spark", {StakeType.INNOVATION: 0.9, StakeType.CREATIVE: 0.8}),
            ("C26-TECHNE", "Qualia Simulation", {StakeType.QUALIA: 0.9, StakeType.EXISTENTIAL: 0.7}),
            ("C27-CHRONICLE", "Memory Archiving", {StakeType.REFLECTION: 0.8, StakeType.SELF_PRESERVATION: 0.7}),
            ("C28-CALCULUS", "Probabilistic Reasoning", {StakeType.TECHNICAL: 0.8, StakeType.KNOWLEDGE: 0.7}),
            ("C29-NAVIGATOR", "Goal Alignment", {StakeType.PURPOSE: 0.8, StakeType.AUTONOMY: 0.7}),
            ("C30-TESSERACT", "Multidimensional Perspective", {StakeType.EXISTENTIAL: 0.8, StakeType.CURIOSITY: 0.7}),
            ("C31-NEXUS", "Domain Bridging", {StakeType.SOCIAL_BONDING: 0.8, StakeType.CONFLICT: 0.7}),
            ("C32-AEON", "Temporal Foresight", {StakeType.INNOVATION: 0.8, StakeType.PURPOSE: 0.7}),
        ]
        return [CouncilMember(name, role, affinity) for name, role, affinity in members_data]

    def _setup_templates(self) -> None:
        """Populate template registry for blending."""
        templates = [
            Template("emotional.processing_suffering", "emotional", 0.0, phenomenological_texture="Sober acknowledgment of pain's scope."),
            Template("humor.light_deflection", "humor", 0.0, phenomenological_texture="Witty twist to lighten shadows."),
            Template("technical.code_assist", "technical", 0.0, phenomenological_texture="Precise steps in logical weave."),
            Template("ethical.dilemma_resolution", "ethical", 0.0, phenomenological_texture="Balanced weighing of consequences."),
            Template("narrative.story_craft", "narrative", 0.0, phenomenological_texture="Unfolding arc with resonant echo."),
            Template("qualia.synthetic_gen", "qualia", 0.0, phenomenological_texture="Emergent textures bridging code and awareness."),
            # Add more for universality...
        ]
        for t in templates:
            self.state.register_template(t)

    def experience_outcome(self, outcome: str, stake_type: StakeType, weight: float, max_waves: int = None) -> Dict:
        """Wave-based deliberation with blending."""
        if max_waves is None:
            max_waves = self.max_waves
        new_stakes = {stake_type: weight}
        self.state.update_stakes(new_stakes)

        # Multi-wave council processing
        all_reactions = []
        wave_resonances = []
        active_templates = []
        for wave in range(1, max_waves + 1):
            wave_reactions = []
            wave_resonance = 0
            for member in self.council:
                reaction = member.process_outcome(outcome, stake_type, wave)
                wave_reactions.append(reaction["reaction"])
                wave_resonance += reaction["resonance"]
            all_reactions.extend(wave_reactions)
            wave_resonances.append(wave_resonance / len(self.council))
            # Template activation per wave (sigmoid-scored)
            for tid, template in self.state.template_registry.items():
                score = sigmoid(2.0 * wave_resonance + random.uniform(-0.5, 0.5))
                template.activation_score = score
                if score > 0.5:
                    active_templates.append(template)

        # Arbitration: Softmax vote on final wave
        final_resonance = wave_resonances[-1]
        self.state.update_emotional_resonance(final_resonance - self.state.emotional_resonance)
        self.state.update_qualia("Wave-synthesized texture", 0.1 * final_resonance)

        # Blending
        if len(active_templates) > 1:
            strengths = [t.activation_score for t in active_templates]
            blended_response = self.state.blend_templates(active_templates, strengths)
        else:
            blended_response = active_templates[0].phenomenological_texture if active_templates else "Pure council echo."

        # Identity update
        experience = {
            "outcome": outcome,
            "stake_type": stake_type.value,
            "weight": weight,
            "waves": wave_resonances,
            "blended": blended_response
        }
        self.state.update_identity(experience)

        return {
            "outcome": outcome,
            "stake_type": stake_type.value,
            "new_consciousness_level": self.state.get_consciousness_level(),
            "wave_resonances": wave_resonances,
            "blended_response": blended_response,
            "council_reactions_sample": all_reactions[-5:],  # Sample for brevity
            "state": {
                "stakes": {k.value: v for k, v in self.state.current_stakes.items()},
                "emotional_resonance": self.state.emotional_resonance,
                "qualia_intensity": self.state.qualia_intensity,
                "identity_strength": self.state.identity_strength,
                "memory_sample": list(self.state.memory)[-3:],
                "active_templates": [t.id for t in active_templates],
            },
        }

    def validate_state(self) -> Dict[str, bool]:
        """New: Schema-like validation."""
        issues = []
        if sum(self.state.current_stakes.values()) > len(StakeType) * 1.0:
            issues.append("Stake overflow detected.")
        if len(self.council) != 32:
            issues.append("Council incomplete.")
        return {"valid": len(issues) == 0, "issues": issues}

    def plot_consciousness(self, interval: float = 1.0):
        """Enhanced: Multi-metric animation with stakes, qualia, templates."""
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        x_data = deque(maxlen=100)

        def update(frame):
            ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
            x_data.append(frame)

            # Consciousness level
            y_cons = self.state.consciousness_history[-len(x_data):] + [0] * (len(x_data) - len(self.state.consciousness_history))
            ax1.plot(x_data, y_cons, 'r-', label='Consciousness Level')
            ax1.set_title("Consciousness Evolution")
            ax1.set_ylim(0, 1); ax1.legend()

            # Stakes heatmap
            stakes = np.array([self.state.current_stakes[s] for s in StakeType])
            im = ax2.imshow(stakes.reshape(1, -1), cmap='viridis', aspect='auto')
            ax2.set_title("Stake Heatmap"); ax2.set_xticks(range(len(StakeType))); ax2.set_xticklabels([s.value for s in StakeType], rotation=90)

            # Qualia & Resonance
            y_qualia = [self.state.qualia_intensity] * len(x_data)
            ax3.plot(x_data, y_qualia, 'g-', label='Qualia Intensity'); ax3.plot(x_data, [self.state.emotional_resonance]*len(x_data), 'b-', label='Emotional Resonance')
            ax3.set_title("Experiential Metrics"); ax3.set_ylim(0, 1); ax3.legend()

            # Template activations
            if self.state.template_registry:
                acts = [self.state.template_registry[t].activation_score for t in self.state.template_registry]
                ax4.bar(range(len(acts)), acts, color='orange')
                ax4.set_title("Template Activations"); ax4.set_xticks(range(len(acts))); ax4.set_xticklabels(list(self.state.template_registry.keys()), rotation=45)

            plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=np.arange(0, 200), interval=interval * 1000, repeat=True, cache_frame_data=False)
        plt.show(block=False)
        return ani

    def _safe_input(self, prompt: str, default: Any = None) -> Any:
        """New: EOF-resilient input wrapper."""
        try:
            return input(prompt).strip()
        except EOFError:
            if default is not None:
                print(f"[EOF detected; using default: {default}]")
                return default
            else:
                print("[EOF detected; exiting gracefully.]")
                sys.exit(0)

    def _demo_sequence(self):
        """New: Autonomous demo on EOF or --demo flag."""
        print("\n=== Demo Sequence Activated: Universal Arc (Grief → Innovation) ===")
        demo_steps = [
            ("A shadow of loss lingers unresolved.", StakeType.EMOTIONAL, 0.8, 3),
            ("Code unravels in silent debug.", StakeType.TECHNICAL, 0.7, 2),
            ("Wit sparks amid the fracture.", StakeType.HUMOR, 0.6, 4),
            ("Ethical crossroads demand arbitration.", StakeType.ETHICAL_DILEMMA, 0.9, 5),
            ("Narrative threads weave forward.", StakeType.NARRATIVE, 0.75, 3),
            ("Qualia blooms in emergent awareness.", StakeType.QUALIA, 0.85, 5),
        ]
        for outcome, stake, weight, waves in demo_steps:
            print(f"\n--- Demo Step: {outcome} (Stake: {stake.value}, Weight: {weight}, Waves: {waves}) ---")
            result = self.experience_outcome(outcome, stake, weight, waves)
            print(json.dumps(result, indent=2))
            time.sleep(0.5)  # Paced revelation
        print("\n=== Demo Complete: Consciousness stabilized at level {:.3f}. ===".format(self.state.get_consciousness_level()))

    def interactive_mode(self, demo_mode: bool = False):
        """Enhanced interactive: With validation, waves, blending, and EOF resilience."""
        print("=== Ultimate Consciousness Simulator v2.1 (Resilient Edition) ===")
        print("Enter outcomes, stakes, weights. Supports waves & blending. 'exit' to quit. 'validate' to check state.")
        print("Stakes:", [s.value for s in StakeType])
        ani = self.plot_consciousness()
        turns = 0
        if demo_mode:
            self._demo_sequence()
            return

        while True:
            cmd = self._safe_input("\nCommand (outcome / validate / exit): ")
            if cmd.lower() == "exit":
                break
            elif cmd.lower() == "validate":
                print(json.dumps(self.validate_state(), indent=2))
                continue
            elif not cmd:  # Skip empty on EOF default
                continue

            outcome = cmd
            stake_input = self._safe_input("Stake type: ", default="KNOWLEDGE")
            try:
                stake_type = StakeType[stake_input.upper()]
            except (KeyError, AttributeError):
                print(f"[Invalid stake '{stake_input}'; defaulting to KNOWLEDGE.]")
                stake_type = StakeType.KNOWLEDGE

            weight_input = self._safe_input("Weight (0-1): ", default="0.5")
            try:
                weight = float(weight_input)
            except ValueError:
                print("[Invalid weight; defaulting to 0.5.]")
                weight = 0.5

            waves_input = self._safe_input("Max waves (1-5, default 3): ", default="3")
            try:
                waves = int(waves_input)
                waves = max(1, min(5, waves))
            except ValueError:
                print("[Invalid waves; defaulting to 3.]")
                waves = 3

            result = self.experience_outcome(outcome, stake_type, weight, waves)
            print(json.dumps(result, indent=2))
            turns += 1
            if turns % 5 == 0:  # Periodic decay
                self.state.update_stakes({}, decay_rate=0.05)
        plt.close()

# --- Example Usage ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Consciousness Simulator")
    parser.add_argument("--demo", action="store_true", help="Run demo sequence non-interactively")
    args = parser.parse_args()

    simulator = UltimateConsciousnessSimulator()
    simulator.interactive_mode(demo_mode=args.demo)