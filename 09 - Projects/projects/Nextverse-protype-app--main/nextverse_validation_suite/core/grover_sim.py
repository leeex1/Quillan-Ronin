"""Deterministic Grover amplitude-amplification simulation on a classical statevector.

This is real quantum-algorithm math implemented with complex numpy arrays: no random
numbers, no fabricated speedups. It gives the JQLD "Grover boost" claim a concrete
mechanism — the oracle query count of amplitude amplification versus classical search.

It is still a simulation of a quantum algorithm on classical hardware; it does not
claim a wall-clock hardware advantage.
"""

import math
import numpy as np


def uniform_state(n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    return np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)


def grover_iterations(n_qubits: int) -> int:
    """Standard optimal Grover iteration count round(pi/4 * sqrt(N))."""
    return int(round(math.pi / 4.0 * math.sqrt(1 << n_qubits)))


def amplify(n_qubits: int, marked_index: int, iterations: int) -> np.ndarray:
    """Apply `iterations` oracle phase flips + diffusion reflections."""
    dim = 1 << n_qubits
    if not 0 <= marked_index < dim:
        raise ValueError("marked_index out of range")
    psi = uniform_state(n_qubits)
    s = psi.copy()
    for _ in range(iterations):
        psi[marked_index] *= -1.0  # oracle phase flip
        psi = 2.0 * s * np.vdot(s, psi) - psi  # reflection about |s>
    return psi


def grover_report(n_qubits: int, marked_index: int) -> dict:
    """Query complexity and success probability from an actual state-vector run."""
    dim = 1 << n_qubits
    k = grover_iterations(n_qubits)
    psi = amplify(n_qubits, marked_index, k)
    p_marked = float(abs(psi[marked_index]) ** 2)
    classical_queries = dim // 2
    return {
        "search_space": dim,
        "grover_oracle_queries": k,
        "classical_expected_queries": classical_queries,
        "query_speedup": classical_queries / k,
        "marked_probability": p_marked,
        "norm": float(np.linalg.norm(psi)),
    }


if __name__ == "__main__":
    for n in (4, 8, 10):
        r = grover_report(n, (1 << n) - 1)
        print(
            f"n={n:2d} N={r['search_space']:5d} queries={r['grover_oracle_queries']:4d} "
            f"speedup={r['query_speedup']:6.2f}x P(marked)={r['marked_probability']:.4f} "
            f"norm={r['norm']:.12f}"
        )
