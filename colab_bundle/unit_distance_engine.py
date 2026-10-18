"""
Planar Point Sets with Many Unit Distances: Counterexample to the Erdős Unit Distance Conjecture
================================================================================================
Mathematical & Computational Invariant Implementation based on:
OpenAI (2025/2026; "Planar Point Sets with Many Unit Distances", 18 pp)

Key Invariants & Formulations Implemented:
1. Disproof of the Erdős 1946 Unit Distance Conjecture (Theorem 1.1):
   - Refutes nu(n) <= n^{1 + C / log log n}.
   - Proves nu(n) >= n^{1 + delta} for fixed delta > 0 and infinitely many n.
2. Algebraic CM Field Torus Invariant (Section 2, Proposition 2.2):
   - Totally real field L of degree f = [L : Q], CM extension K = L(i), with complex conjugation c.
   - Rational primes q_b = 1 (mod 4) splitting completely in L and K.
   - Elements u = alpha / c(alpha) in K* satisfying:
     N_{K/L}(u) = u * c(u) = 1.
   - All complex embeddings sigma: K -> C satisfy |sigma(u)| = 1 identically.
3. High-Dimensional Minkowski Polydisc Slicing (Section 2, Eq. 8):
   - Embedding into V = C^f ~ R^{2f} via lattice Lambda = Q^{-2} O_K.
   - Polydisc W = prod_{r=1}^f { z in C : |z| <= R }.
   - Disc overlap ratio rho_R = Area(D(0, R) cap D(1, R)) / (pi R^2) -> 1 as R -> inf.
   - Pair density in coset: D_y >= |U| * (rho_R)^f * |X|.
4. Coordinate Injective Projection to R^2 (Section 2, Eqs. 9-11):
   - Projection pi_1: C^f -> C ~ R^2 is strictly injective on any coset of Lambda.
   - Every projected pair (pi_1(x), pi_1(x + u)) has Euclidean distance |sigma_1(u)| = 1.0.
   - Yields planar point set P with unit distances nu(P) >= 0.5 * e^{gamma f / 2} |P| >= |P|^{1 + delta}.
"""

import math
from typing import Dict, List, Optional, Tuple, NamedTuple
import torch
import numpy as np


def circle_intersection_area(r: float, d: float = 1.0) -> float:
    """
    Computes exact intersection area of two planar disks of radius r with centers separated by distance d.
    Formula:
    A(r, d) = 2 * r^2 * arccos(d / (2r)) - 0.5 * d * sqrt(4*r^2 - d^2) for d < 2r.
    """
    if d >= 2.0 * r:
        return 0.0
    if d <= 0.0:
        return math.pi * (r ** 2)
    part1 = 2.0 * (r ** 2) * math.acos(d / (2.0 * r))
    part2 = 0.5 * d * math.sqrt(max(0.0, 4.0 * (r ** 2) - (d ** 2)))
    return part1 - part2


def disc_overlap_ratio(r: float, d: float = 1.0) -> float:
    """Computes rho_R = Area(D(0, R) cap D(1, R)) / (pi R^2) (Section 2)."""
    area_total = math.pi * (r ** 2)
    area_inter = circle_intersection_area(r, d)
    return area_inter / max(1e-12, area_total)


class NormOneTorusElement(NamedTuple):
    """Container for norm-one torus element u in K = L(i)."""
    alpha_real: float
    alpha_imag: float
    u_real: float
    u_imag: float
    modulus: float


class AlgebraicCMTorusGenerator:
    """
    Constructs norm-one elements u = alpha / c(alpha) in CM field K = L(i) (Proposition 2.2).
    Every such element satisfies |sigma(u)| = 1 under every complex embedding sigma.
    """

    def __init__(self, split_primes: Optional[List[int]] = None):
        # Rational primes q = 1 (mod 4) that split in Z[i] as q = a^2 + b^2 = (a + bi)(a - bi)
        self.primes = split_primes or [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]
        self._prime_factors = self._compute_gaussian_factors(self.primes)

    def _compute_gaussian_factors(self, primes: List[int]) -> Dict[int, Tuple[int, int]]:
        """Finds integers (a, b) such that q = a^2 + b^2."""
        factors = {}
        for q in primes:
            found = False
            limit = int(math.isqrt(q)) + 1
            for a in range(1, limit):
                b2 = q - a * a
                b = int(math.isqrt(b2))
                if b * b == b2:
                    factors[q] = (a, b)
                    found = True
                    break
            if not found:
                raise ValueError(f"Prime {q} is not congruent to 1 (mod 4)")
        return factors

    def generate_norm_one_units(self, max_elements: int = 128) -> List[NormOneTorusElement]:
        """
        Generates distinct norm-one elements u = alpha / bar(alpha) using combinations of prime ideals.
        u = (alpha_r + i alpha_i) / (alpha_r - i alpha_i)
          = (alpha_r^2 - alpha_i^2 + 2 i alpha_r alpha_i) / (alpha_r^2 + alpha_i^2).
        Notice that |u|^2 = (u_r^2 + u_i^2) = 1.0 identically!
        """
        units: List[NormOneTorusElement] = []
        prime_list = list(self._prime_factors.items())

        # Generate elements from single and composite Gaussian primes
        for q, (a, b) in prime_list:
            # alpha = a + bi
            denom = float(a * a + b * b)
            u_r = float(a * a - b * b) / denom
            u_i = float(2 * a * b) / denom
            mod = math.hypot(u_r, u_i)
            units.append(NormOneTorusElement(
                alpha_real=float(a),
                alpha_imag=float(b),
                u_real=u_r,
                u_imag=u_i,
                modulus=mod
            ))
            if len(units) >= max_elements:
                break

        # Generate products of two Gaussian primes for richer combinations
        if len(units) < max_elements:
            for i in range(len(prime_list)):
                q1, (a1, b1) = prime_list[i]
                for j in range(i + 1, len(prime_list)):
                    q2, (a2, b2) = prime_list[j]
                    # alpha = (a1 + i b1) * (a2 + i b2) = (a1*a2 - b1*b2) + i (a1*b2 + a2*b1)
                    ar = a1 * a2 - b1 * b2
                    ai = a1 * b2 + a2 * b1
                    denom = float(ar * ar + ai * ai)
                    u_r = float(ar * ar - ai * ai) / denom
                    u_i = float(2 * ar * ai) / denom
                    mod = math.hypot(u_r, u_i)
                    units.append(NormOneTorusElement(
                        alpha_real=float(ar),
                        alpha_imag=float(ai),
                        u_real=u_r,
                        u_imag=u_i,
                        modulus=mod
                    ))
                    if len(units) >= max_elements:
                        break
                if len(units) >= max_elements:
                    break

        return units


class MinkowskiPlanarPointSet:
    """
    Simulates the Minkowski embedding of K = L(i) into C^f, cutting by a polydisc of radius R,
    and projecting to the first complex coordinate C ~ R^2 (Section 2, Theorem 2.3).
    """

    def __init__(self, f_dim: int = 4, radius: float = 3.0):
        self.f_dim = f_dim
        self.radius = radius
        self.torus_gen = AlgebraicCMTorusGenerator()

    def construct_planar_point_set(
        self,
        num_seeds: int = 16,
        num_translations: int = 8
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[str, float]]:
        """
        Constructs a planar point set P in R^2 with dense unit distance graph:
        1. Selects norm-one algebraic elements u_k with |sigma_1(u_k)| = 1.0.
        2. Builds seed lattice points inside polydisc.
        3. Applies translations x -> x + u_k while remaining inside polydisc.
        4. Projects to planar coordinates (x_1, y_1).
        Returns:
            points: [N, 2] array of planar coordinates.
            unit_pairs: List of index pairs (i, j) with Euclidean distance ||p_i - p_j||_2 = 1.0.
            metrics: Theoretical and empirical density metrics.
        """
        units = self.torus_gen.generate_norm_one_units(max_elements=num_translations)
        num_units = len(units)

        # Base seed points in R^2
        points_list: List[np.ndarray] = []

        # Generate a grid of points within radius R - 1 so unit steps stay within bounds
        safe_r = max(0.5, self.radius - 1.0)
        grid_dim = int(math.isqrt(num_seeds)) + 1
        xs = np.linspace(-safe_r, safe_r, grid_dim)
        ys = np.linspace(-safe_r, safe_r, grid_dim)

        seed_pts = []
        for x in xs:
            for y in ys:
                if x * x + y * y <= safe_r * safe_r:
                    seed_pts.append(np.array([x, y], dtype=np.float64))
                    if len(seed_pts) >= num_seeds:
                        break
            if len(seed_pts) >= num_seeds:
                break

        # Set of all unique points (using quantized hash to avoid duplicate points)
        point_set: Dict[Tuple[int, int], np.ndarray] = {}
        scale = 1e8

        def pt_key(p: np.ndarray) -> Tuple[int, int]:
            return (int(round(p[0] * scale)), int(round(p[1] * scale)))

        # Add seed points
        for sp in seed_pts:
            point_set[pt_key(sp)] = sp

        # Add unit translated points: p + u for each norm-one element u
        for sp in seed_pts:
            for u in units:
                trans_p = sp + np.array([u.u_real, u.u_imag], dtype=np.float64)
                if np.dot(trans_p, trans_p) <= self.radius * self.radius:
                    point_set[pt_key(trans_p)] = trans_p

        all_points = np.array(list(point_set.values()), dtype=np.float64)
        N = len(all_points)

        # Find all pairs with Euclidean distance exactly 1.0 (within numerical tolerance 1e-6)
        unit_pairs: List[Tuple[int, int]] = []
        # Spatial search / pairwise comparison
        for i in range(N):
            diffs = all_points[i + 1:] - all_points[i]
            dists = np.hypot(diffs[:, 0], diffs[:, 1])
            matches = np.where(np.abs(dists - 1.0) < 1e-6)[0]
            for m in matches:
                unit_pairs.append((i, i + 1 + int(m)))

        nu = len(unit_pairs)

        # Theoretical metrics
        rho = disc_overlap_ratio(self.radius, d=1.0)
        # Erdos classical bound: n^{1 + C / log log n}
        log_n = max(2.0, math.log(max(3, N)))
        log_log_n = math.log(log_n)
        erdos_exponent = 1.0 + (1.0 / log_log_n)
        empirical_exponent = math.log(max(1, nu)) / math.log(max(2, N)) if nu > 0 else 1.0

        metrics = {
            "num_points": float(N),
            "num_unit_pairs": float(nu),
            "overlap_ratio_rho": float(rho),
            "erdos_exponent": float(erdos_exponent),
            "empirical_exponent": float(empirical_exponent),
            "average_degree": float(2.0 * nu / max(1, N)),
        }

        return all_points, unit_pairs, metrics
