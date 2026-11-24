// Native, dependency-free verification of the portable game-engine units.
// Covers the compile blockers fixed in 2026: 5-field Particle, local
// QuantumCore.hpp, portable AICommandServer, and the DVVE calculator.
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "Particle.hpp"
#include "PhysicsEngine.hpp"
#include "DVVE.hpp"
#include "AICommandServer.hpp"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "[FAIL] " #cond " (line " << __LINE__ << ")\n";             \
            return 1;                                                                \
        }                                                                            \
    } while (0)

int main() {
    // B4: the test suite aggregate-initialises five fields (x, y, vx, vy, mass).
    std::vector<Particle> particles;
    particles.push_back({1.0f, 2.0f, 0.1f, -0.1f, 1.0f});
    particles.push_back({3.0f, 4.0f, -0.1f, 0.1f, 1.5f});
    CHECK(particles.size() == 2);
    CHECK(std::abs(particles[0].mass - 1.0f) < 1e-6);
    std::cout << "[PASS] Particle 5-field aggregate initialisation\n";

    // B5: PhysicsEngine compiles against the local QuantumCore.hpp and runs.
    PhysicsEngine physics;
    physics.performAdvancedCollisions(particles);
    std::cout << "[PASS] PhysicsEngine advanced collision offload\n";

    // DVVE calculation.
    DVVECalculator calc;
    DVVEParameters params;
    double rp = calc.calculatePerformance(params);
    CHECK(rp > 0.0);
    double expected = params.P_core * params.F_v * (1.0 + params.omega_v) / (1.0 + params.nu_v);
    if (std::abs(rp - expected) >= 1e-3) {
        std::cerr << "[FAIL] DVVE formula mismatch\n";
        return 1;
    }
    std::cout << "[PASS] DVVE performance = " << rp << "\n";

    // B6: ThreadSafeQueue and the portable AICommand contract.
    ThreadSafeQueue<AICommand> queue;
    AICommand command;
    command.avatar_id = "avatar-1";
    command.action = "move_to";
    command.x = 1.0f;
    queue.push(command);
    AICommand popped;
    CHECK(queue.try_pop(popped));
    CHECK(popped.avatar_id == "avatar-1" && popped.action == "move_to");
    CHECK(std::abs(popped.x - 1.0f) < 1e-6);
    CHECK(queue.empty());
    AICommand missing;
    CHECK(!queue.try_pop(missing));
    std::cout << "[PASS] AICommandServer portable queue contract\n";

    std::cout << "[PASS] All native game-engine checks passed\n";
    return 0;
}
