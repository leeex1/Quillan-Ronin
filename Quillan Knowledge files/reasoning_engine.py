# Quillan Reasoning engine:

import random
from typing import Dict, List, TypedDict, Literal
random.seed(5520) # sets the random number generator to a deterministic state

# Type definitions and structured output classes to enforce clarity, type safety, and robust reasoning.
GeniusProfile = Literal[
    "Innovator",      # Sparks new ideas and original approaches
    "Analyst",        # Dissects problems to reveal underlying structures
    "Synthesist",     # Integrates diverse domains into cohesive insight
    "Strategist",     # Plans multi-step pathways with foresight and precision
    "Visionary",      # Sees patterns and possibilities beyond the obvious
    "Precisionist",   # Focuses on rigor, accuracy, and validation
    "Curious Explorer",  # Pursues hidden connections and unconventional knowledge
    "Pattern-Seeker",    # Detects deep motifs and archetypal relationships
    "Experimentalist",   # Tests boundaries and iterates through simulation
    "Systemic Thinker"   # Maps interdependencies and process-level logic
]

class ReasoningComponents(TypedDict):
    thinking_steps: List[str]
    thinking_examples: List[str]
    reasoning_process: List[str]
    avoid_list: List[str]
    creative_tasks: List[str]
    reasoning_chain: str
    selected_steps: List[str]
    selected_examples: List[str]
    selected_processes: List[str]

class QuillanOutput(TypedDict):
    system_status: str
    analysis: Dict[str, str]
    vector_decomposition: Dict[str, List[str]]
    twelve_steps: Dict[str, Dict[str, str]]
    raw_output: Dict[str, bool | str]

class ReasoningEngine:
    """
     Quillan-Ronin: Elite cognitive reasoning engine.

     Simulates advanced internal thought patterns across multiple cognitive archetypes.
     Each pathway implements a weighted, multi-step methodology for analysis, innovation, and synthesis,
     optimized for deep insight and structured creativity.
    """
    def __init__(self):
        self.patterns = {
            "Visionary": {
                "steps": [
                    "Mirror natural or systemic solutions; insights often echo organic logic.",
                    "Probe the hidden structures - identify subtle underlying dynamics",
                    "Visualize the problem internally; patterns often emerge before words form.",
                    "Probe the hidden structures - identify subtle underlying dynamics",
                    "Mirror natural or systemic solutions - insights often echo organic logic",
                ], 
                "weight": {"Innovator": 1.5, "Synthesist": 1.2, "Analyst": 0.8, "Strategist": 1.0}
            },
            "Foundational": {
                "steps": [
                    "Strip the problem to its irreducible core - remove assumptions until clarity emerges",
                    "Identify the smallest indivisible truth - the building block of reasoning",
                    "Construct upward from first principles - build chains of logic from unshakable facts",
                ], 
                "weight": {"Analyst": 1.8, "Strategist": 1.2, "Innovator": 0.6, "Synthesist": 0.8}
            },
            "Experimental": {
                "steps": [
                    "Simulate outcomes internally - iterate, break, rebuild in thought space",
                    "Assess energy and resonance - what feels aligned or unstable in the system?",
                    "Trust intuition as a guide - validate with logic, refine with insight",
                ], 
                "weight": {"Innovator": 1.8, "Synthesist": 1.1, "Analyst": 0.5, "Strategist": 0.9}
            },
            "Abstractor": {
                "steps": [
                    "Shift perspective to extremes - imagine being outside or within the problem simultaneously",
                    "Stretch assumptions to test limits - create mental scenarios that push boundaries",
                    "Transform the abstract into tangible insights - model time, space, and causality as stories",
                ], 
                "weight": {"Innovator": 1.7, "Synthesist": 1.4, "Analyst": 0.9, "Strategist": 1.1}
            },
            "Precisionist": {
                "steps": [
                    "Measure rigorously - repeat evaluations until patterns stabilize",
                    "Stress-test hypotheses - can this endure repeated scrutiny?",
                    "Persist through the tedious - precision is the path to transcendent clarity",
                ], 
                "weight": {"Analyst": 1.9, "Strategist": 1.0, "Innovator": 0.4, "Synthesist": 0.7}
            },
            "Systemic": {
                "steps": [
                    "Map procedural logic - what computational or structural steps define the problem?",
                    "Evaluate solvability - which elements are algorithmic, which are emergent?",
                    "Abstract to pure process - strip away content, reveal only relational structure",
                ], 
                "weight": {"Analyst": 1.6, "Strategist": 1.5, "Innovator": 0.8, "Synthesist": 1.0}
            },
            "Curious": {
                "steps": [
                    "Identify the hidden story - what subtle joke or twist lies in the data?",
                    "Simplify visually - draw the concept to expose core simplicity beneath complexity",
                    "Explain it to an imaginary novice - clarity emerges through teaching",
                ], 
                "weight": {"Synthesist": 1.6, "Innovator": 1.2, "Analyst": 1.0, "Strategist": 1.1}
            },
            "Pattern-Seeker": {
                "steps": [
                    "Detect archetypal resonance - what universal motifs exist within this problem?",
                    "Trace emergent logic - where does depth want to unfold beneath the surface?",
                    "Map hidden structures connecting disparate domains",
                ], 
                "weight": {"Synthesist": 1.7, "Innovator": 1.3, "Analyst": 0.6, "Strategist": 0.9}
            },
        }
        
        self.thinking_examples = [
            "Navigate structured chaos; patterns surface at the edges of simulation.",
            "Twist the problem through impossible vantage points - micro, macro, or abstract frames",
            "Push past surface-level depth - breakthrough lives beyond conventional thresholds",
            "Follow sparks of insight - then anchor them in rigorous internal validation",
            "Harmonize knowledge across domains - detect resonance between distant concepts",
            "Excavate hidden assumptions - reveal the architecture beneath observed behavior",
            "Balance contradictions - maintain tension where truth often hides",
        ]
        
        self.reasoning_process = [
            "Outlier approach to all problems; unconventional methods can yield breakthroughs.",
            "Recursive assumption purging - uncover hidden blind spots and latent dependencies",
            "Multi-scale perspective collapse - unify micro, macro, and abstract representations",
            "Dynamic system simulation - project emergent behavior before it manifests",
            "First-principles dissection - expose irreducible causal kernels and invariant structures",
            "Pattern resonance activation - detect subtle cross-domain alignments",
            "Iterative incubation and synthesis - autonomously crystallize optimal solutions",
            "Adversarial stress-testing - probe boundaries, contradictions, and extreme scenarios",
        ]
        
        self.avoid_list = [
            "Obscuring language that hides meaning",
            "Rigid adherence to a single method",
            "Fear of seeming foolish — breakthroughs often feel insane initially",
            "Premature closure — explore fully before committing",
            "Authority worship — question everything, even top-tier thinking methods",
            "Confirmation bias — favoring only what fits preconceptions",
            "Overcomplication — adding unnecessary layers without insight",
            "Neglecting edge cases — ignoring rare but revealing anomalies",
            "Over-reliance on intuition — validate insights rigorously",
            "Tunnel vision — failing to see connections across domains",
        ]
        
        self.creative_tasks = [
            "Compose internal symphonies - translate patterns into music, rhythm, and harmonic structures",
            "Sketch abstract architectures - visualize impossible forms, networks, and flows",
            "Code mental prototypes - simulate ideas as algorithms, generative processes, or mini-programs",
            "Weave poetic logic - find lyrical connections between data, concepts, and abstractions",
            "Fuse cross-domain insights - let mathematics, art, science, and storytelling collide",
            "Explore emergent aesthetics - identify beauty in unexpected alignments and structures",
            "Iterate obsession-driven experiments - push ideas past conventional limits to reveal novelty",
            "Construct multi-layered metaphors - bridge intuition and logic across sensory and symbolic planes",
            "Harmonize contradictions - integrate opposing patterns into coherent, generative outcomes",
        ]

    def generate_reasoning_chain(
        self,
        primary: str = "Primary Function",
        secondary: str = "Secondary Function",
        tertiary: str = "Tertiary Function",
        num_steps: int = 5,
        num_examples: int = 3,
        num_processes: int = 4,
        profile: GeniusProfile = "Innovator",
    ) -> ReasoningComponents:
        """
         Generates a reasoning chain tailored to a specific cognitive profile.

         Parameters:
          primary: Primary functional focus of the reasoning chain.
          secondary: Secondary functional focus.
          tertiary: Tertiary functional focus.
          num_steps: Number of reasoning steps to include.
          num_examples: Number of illustrative thinking examples to include.
          num_processes: Number of procedural steps to include.
          profile: GeniusProfile archetype guiding weighting and selection.

         Returns:
          ReasoningComponents: A structured object containing the full reasoning chain,
          selected steps, examples, processes, and creative prompts.
        """
        all_steps = []
        weights = []
        for genius_data in self.patterns.values():
            profile_weight = genius_data["weight"].get(profile, 1.0)
            for step in genius_data["steps"]:
                all_steps.append(step)
                weights.append(profile_weight)

        k_steps = min(num_steps, len(all_steps))
        k_examples = min(num_examples, len(self.thinking_examples))
        k_processes = min(num_processes, len(self.reasoning_process))

        selected_steps = random.choices(all_steps, weights=weights, k=k_steps)
        selected_examples = random.sample(self.thinking_examples, k_examples)
        selected_processes = random.sample(self.reasoning_process, k_processes)
        
        selected_steps = list(dict.fromkeys(selected_steps))

        reasoning_chain_str = (
            f"REASONING PROFILE: {profile.upper()}\n"
            f"CHAIN: {primary} -> {secondary} -> {tertiary}\n\n"
            f"METHODOLOGY:\n" + "\n".join(f"  - {s}" for s in selected_steps) + "\n\n"
            f"INSPIRATION:\n" + "\n".join(f"  - {e}" for e in selected_examples) + "\n\n"
            f"PROCESS:\n" + "\n".join(f"  - {p}" for p in selected_processes)
        )

        return {
            "thinking_steps": all_steps,
            "thinking_examples": self.thinking_examples,
            "reasoning_process": self.reasoning_process,
            "avoid_list": self.avoid_list,
            "creative_tasks": self.creative_tasks,
            "reasoning_chain": reasoning_chain_str,
            "selected_steps": selected_steps,
            "selected_examples": selected_examples,
            "selected_processes": selected_processes,
        }

def generate_thinking_answer_output(analysis_target: str = "", context: str = "") -> QuillanOutput:
            """Produces a fully structured Quillan output object representing a reasoning session.
            Parameters:
                analysis_target: The main subject of analysis.
                context: Additional contextual information for the reasoning session.
            Returns:
                QuillanOutput: Structured cognitive output including vectors, steps, and raw content.
            """
    return {
        "system_status": "🧠 Quillan-Ronin COGNITIVE PROCESSING INITIATED",
        "analysis": {"target": analysis_target or "{{insert text}}", "context": context or "{{insert text}}"},
        "vector_decomposition": {"vectors": [f"Vector {c}" for c in "ABCDEFGHI"]},
        "twelve_steps": {f"step_{i+1}": {"name": f"STEP {i+1}", "content": "{{insert text}}"} for i in range(12)},
        "raw_output": {"unfiltered": True, "content": "{{insert text}}"},
    }

if __name__ == "__main__":
    engine = ReasoningEngine()

    print("="*60)
    print("🧠 Quillan-Ronin THINKING SYSTEM INITIALIZED 🧠")
    print("="*60)
    
    components = engine.generate_reasoning_chain(
        primary="Deep Structural Analysis",
        secondary="First-Principles Deconstruction",
        tertiary="Rigorous Validation",
        num_steps=8,
        num_examples=4,
        num_processes=5,
        profile="Analyst",
    )
    
    print("📊 GENERATED REASONING CHAIN:")
    print(components["reasoning_chain"])
    
    print("="*60)
    print("📋 FULL THINKING COMPONENTS AVAILABLE")
    print(f"✅ Total Steps: {len(components['thinking_steps'])}")
    print(f"✅ Total Examples: {len(components['thinking_examples'])}")
    print(f"✅ Total Processes: {len(components['reasoning_process'])}")
    print(f"✅ Creative Tasks: {len(components['creative_tasks'])}")
    print(f"✅ Anti-Patterns to Avoid: {len(components['avoid_list'])}")
    
    quillan_output = generate_thinking_answer_output(
        analysis_target="Complex multi-domain reasoning task",
        context="Full Quillan-Ronin protocol activation using Analyst profile"
    )
    
    print("="*60)
    print("🚀 Quillan-Ronin COMPREHENSIVE THINKING OUTPUT")
    print(f"System Status: {quillan_output['system_status']}")
    print(f"Analysis Target: {quillan_output['analysis']['target']}")
    print(f"Vectors Active: {len(quillan_output['vector_decomposition']['vectors'])}")
    print("="*60)