#ifndef ETHICAL_FRAMEWORK_HPP
#define ETHICAL_FRAMEWORK_HPP

#include <boost/container/vector.hpp>

class EthicalFramework {
public:
    // Ethical check: E_t = sum(M_i * W_i(context)) >= E_min
    bool check_ethics(const std::string& context, double M_i) {
        double W_i = context == "combat" ? 0.9 : 0.8; // Context weight
        double E_t = M_i * W_i;
        double E_min = 0.75;
        return E_t >= E_min;
    }
};

#endif 