"""NextVerse formula engine — deterministic computations only.

2026-11-26 remediation: the previous engine returned random dummy results and drew
"speedups" from random.uniform(), with time.sleep() standing in for benchmarks.
Every method below now evaluates the arithmetic of its published formula on real
numbers. `measured_performance` equals the computed value unless a real benchmark
is supplied by the caller; no result is fabricated. These are heuristic engineering
models, not physical laws, and their speedup factors are model outputs, not evidence
of measured hardware gains.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional


@dataclass
class FormulaResult:
    formula_name: str
    theoretical_value: float
    measured_performance: float
    speedup_factor: float
    execution_time: float
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    benchmarked: bool = False


DEFAULTS: Dict[str, Dict[str, float]] = {
    "JQLD": {"C": 1.1, "N_G": 20, "N_R": 15, "N_c": 10, "delta_q": 0.08},
    "LVVM": {"R_vm": 2.0, "psi_vm": 0.2, "mu_vm": 0.03, "tau_vm": 0.02},
    "DESS": {"E_avg": 0.90, "w_context": 1.0, "phi_grover": 0.90, "R_min": 0.80},
    "JRRN": {"D_c": 1000, "P_t": 500, "F_c": 0.90, "gamma_c": 0.25, "sigma_c": 5.0},
    "LRPP": {
        "C_prev": 88116,
        "N_agents": 10,
        "A_a": 2000,
        "alpha": 0.90,
        "rho_a": 0.90,
        "kappa_a": 0.015,
    },
    "DVVE": {"P_core": 85000, "F_v": 1.0, "omega_v": 0.20, "nu_v": 0.03},
    "JSSC": {
        "N_NPC": 200,
        "N_players": 50,
        "beta": 0.5,
        "chi": 100,
        "Q_ai": 1.0,
        "zeta_ai": 0.20,
    },
    "LSSS": {
        "P_core_audio": 100,
        "Delta_audio": 1.0,
        "theta_audio": 0.25,
        "iota_audio": 0.015,
    },
    "DNNL": {
        "D_n": 10,
        "B_w": 1000,
        "V_n": 0.10,
        "xi_n": 0.30,
        "sum_P": 500,
        "pi_n": 5.0,
    },
    "JHFR": {"H_int": 0.15, "H_eth": 0.10, "H_net": 0.10, "phi_sys": 0.20},
    "LMCB": {"M_avg": 0.92, "W_context": 1.0, "psi_grover": 0.90, "E_min": 0.85},
}


def _result(name, value, speedup, params, start, note=None, benchmark=None):
    return FormulaResult(
        formula_name=name,
        theoretical_value=value,
        measured_performance=value if benchmark is None else benchmark,
        speedup_factor=speedup,
        execution_time=time.process_time() - start,
        parameters=params,
        success=True,
        error_message=note,
        benchmarked=benchmark is not None,
    )


class NextVerseFormulaEngine:
    def __init__(self):
        self.formulas = list(DEFAULTS)
        self.baseline_metrics: Dict[str, float] = {}

    @staticmethod
    def _p(name, overrides):
        p = dict(DEFAULTS[name])
        if overrides:
            p.update({k: v for k, v in overrides.items() if k in p})
        return p

    def calculate_jqld(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("JQLD", kw)
        bits = kw.get("search_space_bits")
        if bits is not None:
            # Real Grover state-vector simulation: query count is computed, not
            # a calibrated constant. Still a simulation, not a wall-clock gain.
            try:
                from .grover_sim import grover_report
            except ImportError:
                from grover_sim import grover_report
            g = grover_report(int(bits), (1 << int(bits)) - 1)
            q = dict(p)
            q["search_space_bits"] = int(bits)
            return _result(
                "JQLD",
                p["C"] * g["query_speedup"],
                g["query_speedup"],
                q,
                t0,
                f"Grover sim: P(marked)={g['marked_probability']:.4f}, "
                f"queries={g['grover_oracle_queries']}/{g['classical_expected_queries']}",
            )
        exp = (p["N_G"] * 0.6 * 0.4 + p["N_R"] * 0.4 * 0.3 + p["N_c"] * 0.7 * 0.5) / (
            1 + p["delta_q"]
        )
        speedup = 2**exp
        return _result("JQLD", p["C"] * speedup, speedup, p, t0)

    def calculate_lvvm(self, q_value: Optional[float] = None, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("LVVM", kw)
        q = (
            q_value
            if isinstance(q_value, (int, float)) and q_value > 0
            else self.calculate_jqld().theoretical_value
        )
        mult = (p["R_vm"] + p["psi_vm"] * (1 - p["mu_vm"])) / (1 + p["tau_vm"])
        return _result("LVVM", q * mult, mult, p, t0)

    def calculate_dess(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("DESS", kw)
        E, w, phi = p["E_avg"], p["w_context"], p["phi_grover"]
        r_t = (0.4 * E * phi + 0.3 * (E * 0.944) * phi + 0.3 * (E * 1.055) * phi) * w
        return _result(
            "DESS",
            r_t,
            phi * 1.5,
            p,
            t0,
            None if r_t >= p["R_min"] else "threshold breach",
        )

    def calculate_jrrn(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("JRRN", kw)
        t_ms = (
            p["D_c"] / (p["P_t"] * 1000 * p["F_c"] * (1 + p["gamma_c"]))
        ) * 1000 + p["sigma_c"]
        return _result("JRRN", t_ms, 80 / max(t_ms, 1), p, t0)

    def calculate_lrpp(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("LRPP", kw)
        c_t = p["C_prev"] + (p["N_agents"] * p["A_a"] * p["alpha"] * p["rho_a"]) / (
            1 + p["kappa_a"]
        )
        return _result("LRPP", c_t, c_t / p["C_prev"], p, t0)

    def calculate_dvve(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("DVVE", kw)
        fac = (1 + p["omega_v"]) / (1 + p["nu_v"])
        return _result("DVVE", p["P_core"] * p["F_v"] * fac, fac, p, t0)

    def calculate_jssc(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("JSSC", kw)
        inside = p["N_NPC"] + p["beta"] * p["N_players"] + p["chi"]
        s = math.sqrt(inside) * p["Q_ai"] * (1 + p["zeta_ai"]) * 850
        return _result("JSSC", s, 1 + p["zeta_ai"], p, t0)

    def calculate_lsss(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("LSSS", kw)
        fac = (1 + p["theta_audio"]) / (1 + p["iota_audio"])
        return _result("LSSS", p["P_core_audio"] * p["Delta_audio"] * fac, fac, p, t0)

    def calculate_dnnl(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("DNNL", kw)
        bw = p["B_w"] * (1 - p["V_n"]) * (1 + p["xi_n"]) + p["sum_P"]
        l_ms = (p["D_n"] / bw) * 1000 + p["pi_n"]
        return _result("DNNL", l_ms, 50 / max(l_ms, 1), p, t0)

    def calculate_jhfr(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("JHFR", kw)
        denom = p["H_int"] + p["H_eth"] + p["H_net"] * (1 - p["phi_sys"])
        factors = [1.8, 2.0, 1.3, 1.4, 1.25, 1.5, 1.2]
        product = 1.0
        for f in factors:
            product *= f * 0.95
        o_sys = product / max(denom, 0.05)
        return _result("JHFR", o_sys, o_sys, p, t0)

    def calculate_lmcb(self, **kw) -> FormulaResult:
        t0 = time.process_time()
        p = self._p("LMCB", kw)
        e_t = p["M_avg"] * p["W_context"] * p["psi_grover"]
        rate = min(100.0, max(0.0, (e_t / p["E_min"]) * 99.5))
        return _result(
            "LMCB",
            rate,
            1 + p["psi_grover"] * 0.5,
            p,
            t0,
            None if e_t >= p["E_min"] else "moral drift",
        )

    def run_complete_validation(self) -> Tuple[Dict[str, FormulaResult], float]:
        order = [
            "JQLD",
            "LVVM",
            "DESS",
            "JRRN",
            "LRPP",
            "DVVE",
            "JSSC",
            "LSSS",
            "DNNL",
            "JHFR",
            "LMCB",
        ]
        results = {}
        q = None
        for name in order:
            if name == "LVVM":
                results[name] = self.calculate_lvvm(q_value=q)
            else:
                results[name] = getattr(self, "calculate_" + name.lower())()
                if name == "JQLD":
                    q = results[name].theoretical_value
        factor = 1.0
        for r in results.values():
            factor *= r.speedup_factor
        # These are heuristic models with no empirical validation against real
        # hardware or measured workloads. A product of model factors is NOT a
        # measured speedup and must not be reported as one.
        return results, 1.0


if __name__ == "__main__":
    res, compound = NextVerseFormulaEngine().run_complete_validation()
    for k, v in res.items():
        print(
            f"{k}: {v.theoretical_value:.4f} (deterministic, benchmarked={v.benchmarked})"
        )
    print(
        f"measured speedup: unavailable (no empirical validation; model factor reported as 1.0)"
    )
