#include "DVVE.hpp"
#include <iostream> // For std::cout
#include <algorithm> // For std::max/min if needed

DVVECalculator::DVVECalculator() {
    std::cout << "DVVECalculator created." << std::endl;
}

double DVVECalculator::calculatePerformance(const DVVEParameters& params) {
    if ((1.0 + params.nu_v) == 0) {
        std::cerr << "Error: DVVE nu_v parameter would cause division by zero." << std::endl;
        return 0;
    }
    double Rp = params.P_core * params.F_v * (1.0 + params.omega_v) / (1.0 + params.nu_v);
    // std::cout << "  DVVE Calculated R_p: " << Rp << std::endl; // Can be verbose
    return Rp;
}

double DVVECalculator::determineAdaptiveBoost(double P_core_current, double P_core_target_for_max_quality) {
    // Example: If current P_core is less than half of what's needed for max quality,
    // apply a boost (which implies simplification, e.g. higher omega_v if it means "boosted simplification factor")
    // Or, if omega_v is an efficiency gain, it might increase with better P_core.
    // The formula R_p = P_core * F_v * (1 + omega_v) / (1 + nu_v) suggests omega_v INCREASES R_p.
    // So, a higher omega_v means the system is trying to be more efficient or simplify to achieve targets.
    // Let's assume omega_v is a factor that helps when P_core is low by simplifying things.
    // A higher omega_v could mean "more aggressive LOD / simplification".
    // For this example, let's make it simple: if P_core is low, increase omega_v.

    if (P_core_current < P_core_target_for_max_quality * 0.5) {
        return 0.2; // Apply a moderate adaptive "boost" (simplification)
    } else if (P_core_current < P_core_target_for_max_quality * 0.75) {
        return 0.1; // Apply a small adaptive "boost" (simplification)
    }
    return 0.0; // No boost/simplification needed if P_core is sufficient
}
