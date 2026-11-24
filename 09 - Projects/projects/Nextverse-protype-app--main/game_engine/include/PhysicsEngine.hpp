#ifndef PHYSICSENGINE_HPP
#define PHYSICSENGINE_HPP

#include <vector>
#include <string>
#include <iostream>
#include "Particle.hpp" // From our own include directory
#include "QuantumCore.hpp" // Local minimal contract for QuantumTask (no external quantum_core tree required)

// Placeholder for scene objects if needed by other methods
struct SceneObjectPlaceholder {
    std::string id;
};

class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();

    void update(double deltaTime, std::vector<SceneObjectPlaceholder>& sceneObjects);
    void performAdvancedCollisions(const std::vector<Particle>& particles); // Takes const ref to particles

private:
    // ...
};

#endif // PHYSICSENGINE_HPP
