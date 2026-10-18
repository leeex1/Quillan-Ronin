import math
from typing import List

# Quantum-inspired and cognitive system formulas

def coherence(entropy: float, coupling: float) -> float:
    """Calculates coherence based on entropy and coupling."""
    return 1 - math.exp(-entropy * coupling)

def uncertainty(prior: float, signal: float) -> float:
    """Calculates informational uncertainty using logarithmic divergence."""
    return -1 * math.log2(signal / prior) if signal > 0 and prior > 0 else 0

def vector_alignment(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

def resonance(amplitude: float, frequency: float) -> float:
    return amplitude * math.sin(2 * math.pi * frequency)

def phase_shift(wave1: float, wave2: float) -> float:
    return math.acos(min(1, max(-1, wave1 * wave2)))

def entanglement(info1: float, info2: float) -> float:
    return abs(info1 - info2) / max(info1, info2)

def predictability(stability: float, volatility: float) -> float:
    return 1 - (volatility / (stability + 1e-9))

def novelty_score(signal: float, baseline: float) -> float:
    return (signal - baseline) / (baseline + 1e-9)

def signal_to_noise(signal: float, noise: float) -> float:
    return signal / (noise + 1e-9)

def attention_focus(distraction: float, intent: float) -> float:
    return intent / (distraction + intent + 1e-9)

def mental_energy(load: float, recovery: float) -> float:
    return recovery - load

def idea_density(ideas: int, tokens: int) -> float:
    return ideas / (tokens + 1e-9)

def divergence(metric1: float, metric2: float) -> float:
    return abs(metric1 - metric2) / ((metric1 + metric2) / 2 + 1e-9)

def entropy_gradient(entropy_old: float, entropy_new: float) -> float:
    return entropy_new - entropy_old

def cognitive_load(effort: float, capacity: float) -> float:
    return effort / (capacity + 1e-9)

def time_decay(value: float, decay_rate: float, time: float) -> float:
    return value * math.exp(-decay_rate * time)

def error_amplification(error: float, multiplier: float) -> float:
    return error * multiplier

def feedback_gain(response: float, input_signal: float) -> float:
    return response / (input_signal + 1e-9)

def belief_shift(confidence_old: float, confidence_new: float) -> float:
    return confidence_new - confidence_old

def insight_probability(patterns_detected: int, total_patterns: int) -> float:
    return patterns_detected / (total_patterns + 1e-9)

def decision_efficiency(successes: int, decisions: int) -> float:
    return successes / (decisions + 1e-9)
