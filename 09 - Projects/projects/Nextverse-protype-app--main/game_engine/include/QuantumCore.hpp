#ifndef QUANTUMCORE_HPP
#define QUANTUMCORE_HPP

// Minimal, self-contained contract for the simulated quantum task offload used
// by PhysicsEngine. The previous include pointed at a non-existent external
// ../quantum_core tree, which made the engine fail to configure.

#include <map>
#include <string>

enum class QuantumTaskType {
    GENERAL_QUANTUM_OPTIMIZATION,
    JQLD_CALCULATION,
    ADVANCED_PHYSICS_COLLISION,
    LVVM_AMPLIFICATION,
};

struct QuantumTask {
    std::string task_id;
    QuantumTaskType type = QuantumTaskType::GENERAL_QUANTUM_OPTIMIZATION;
    std::map<std::string, std::string> parameters;
    int qubits = 0;
    double complexity = 0.0;
};

#endif // QUANTUMCORE_HPP
