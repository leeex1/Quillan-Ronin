// Self-contained native verification for the NextVerse Quantum Core Alpha.
// Mirrors the numeric expectations of test_qcc_alpha.cpp / test_qsvm_alpha.cpp
// without requiring GoogleTest (which the CMake file would fetch from network).
#include <cassert>
#include <cmath>
#include <iostream>

#include "qcc_alpha.hpp"
#include "qsvm_alpha.hpp"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "[FAIL] " #cond " (line " << __LINE__ << ")\n";             \
            return 1;                                                                \
        }                                                                            \
    } while (0)

static bool near(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

int main() {
    using namespace NextVerse;

    QuantumComputationalCore::JQLDParams defaults;
    QuantumComputationalCore qcc(defaults);

    // QCC: detailed falls back to simplified with no algorithm list.
    double q = qcc.calculateJQLD_detailed().value();
    CHECK(near(q, 1.1e9 * 1024.0, 1e5));
    std::cout << "[PASS] QCC detailed/default = " << q << "\n";

    // QCC: detailed with a custom optimisation.
    QuantumComputationalCore::JQLDParams custom;
    custom.classical_base_performance_C = 1.0e9;
    custom.algo_optimizations.push_back({"TestOpt", 2, 0.5, 1.5});
    custom.omega_q_divisor = 2.0;
    QuantumComputationalCore qcc_custom(custom);
    double qc = qcc_custom.calculateJQLD_detailed().value();
    CHECK(near(qc, (1.0e9 * std::pow(2.0, 1.5)) / 2.0, 1e2));
    std::cout << "[PASS] QCC detailed/custom = " << qc << "\n";

    // QCC: Q_lite default.
    double qlite = qcc.calculateJQLD_Qlite().value();
    CHECK(near(qlite, 0.25 * 1.1e9 * std::log2(11.0) * 0.6 * 0.74, 1e2));
    std::cout << "[PASS] QCC Q_lite/default = " << qlite << "\n";

    // QCC: low RAM constraint returns nullopt.
    QuantumComputationalCore::JQLDParams low_ram;
    low_ram.free_ram_gb_lite = 0.5;
    QuantumComputationalCore qcc_low(low_ram);
    CHECK(!qcc_low.calculateJQLD_Qlite().has_value());
    std::cout << "[PASS] QCC Q_lite low-RAM rejected\n";

    // QSVM: detailed / simplified / VMSlim.
    QuantumSuperVM::LVVMParams lvparams;
    QuantumSuperVM qsvm(qcc, lvparams, 0);
    double expected_detailed = q * (0.98 + 0.15 * (1.0 - 0.03)) / 1.02;
    CHECK(near(qsvm.getAmplifiedPerformance().value(), expected_detailed, 1e7));
    std::cout << "[PASS] QSVM detailed = " << qsvm.getAmplifiedPerformance().value() << "\n";

    double expected_simplified = q * 0.98;
    CHECK(near(qsvm.calculateLVVM_simplified().value(), expected_simplified, 1e7));
    std::cout << "[PASS] QSVM simplified = " << expected_simplified << "\n";

    double expected_slim = 0.85 * q * std::sqrt(0.98) * 0.9;
    CHECK(near(qsvm.calculateLVVM_VMSlim().value(), expected_slim, 1e6));
    std::cout << "[PASS] QSVM VMSlim = " << expected_slim << "\n";

    // QSVM: high swap is rejected.
    QuantumSuperVM::LVVMParams bad_swap = lvparams;
    bad_swap.swap_usage_percent_slim = 150.0;
    QuantumSuperVM qsvm_bad(qcc, bad_swap, 1);
    CHECK(!qsvm_bad.calculateLVVM_VMSlim().has_value());
    std::cout << "[PASS] QSVM high-swap rejected\n";

    std::cout << "[PASS] All native quantum-core checks passed\n";
    return 0;
}
