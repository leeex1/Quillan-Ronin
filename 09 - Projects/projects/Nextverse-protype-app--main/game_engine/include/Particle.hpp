#ifndef PARTICLE_HPP
#define PARTICLE_HPP

struct Particle {
    float x = 0.0f;
    float y = 0.0f;
    float vx = 0.0f; // Velocity x
    float vy = 0.0f; // Velocity y
    float mass = 1.0f; // Mass (tests aggregate-initialise x, y, vx, vy, mass)
    // unsigned char r=255, g=255, b=255, a=255; // Optional color
};

#endif // PARTICLE_HPP
