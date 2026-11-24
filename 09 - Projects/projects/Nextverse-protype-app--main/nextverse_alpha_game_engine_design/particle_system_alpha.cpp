#include "particle_system_alpha.hpp"
#include <SDL2/SDL.h> // For SDL_Renderer if drawing directly, or for screen dimensions
#include <iostream>   // For logging
#include <algorithm>  // For std::min/max

namespace NextVerse {

ParticleSystem::ParticleSystem(size_t max_particles)
    : max_particles_(max_particles),
      num_active_particles_(0), // Initially no active particles until initialized
      random_engine_(std::random_device{}()) { // Seed the random engine
    particles_.resize(max_particles_); // Pre-allocate memory for all potential particles
    std::cout << "[ParticleSystem_Alpha] Initialized with capacity for " << max_particles_ << " particles." << std::endl;
}

void ParticleSystem::initializeParticles(size_t num_to_spawn, int world_width, int world_height) {
    num_active_particles_ = std::min(num_to_spawn, max_particles_);
    std::cout << "[ParticleSystem_Alpha] Initializing/respawning " << num_active_particles_ << " particles." << std::endl;

    // Update random distributions based on current world dimensions
    pos_dist_x_ = std::uniform_real_distribution<float>(0.0f, static_cast<float>(world_width));
    pos_dist_y_ = std::uniform_real_distribution<float>(0.0f, static_cast<float>(world_height));
    vel_dist_ = std::uniform_real_distribution<float>(-50.0f, 50.0f); // pixels/sec

    for (size_t i = 0; i < num_active_particles_; ++i) {
        particles_[i].x = pos_dist_x_(random_engine_);
        particles_[i].y = pos_dist_y_(random_engine_);
        particles_[i].z = 0; // Keep Z at 0 for 2D SDL rendering
        particles_[i].vx = vel_dist_(random_engine_);
        particles_[i].vy = vel_dist_(random_engine_);
        particles_[i].vz = 0;
        particles_[i].color = {
            static_cast<uint8_t>(random_engine_() % 156 + 100), // Brighter colors
            static_cast<uint8_t>(random_engine_() % 156 + 100),
            static_cast<uint8_t>(random_engine_() % 156 + 100),
            static_cast<uint8_t>(random_engine_() % 100 + 155) // Mostly opaque
        };
        particles_[i].size = 1.0f + static_cast<float>(random_engine_() % 30) / 10.0f; // size 1.0 to 3.9 pixels
        particles_[i].lifetime = 5.0f + static_cast<float>(random_engine_() % 100) / 10.0f; // 5 to 15 seconds
    }
}

void ParticleSystem::updateParticles(float deltaTime, double qcc_performance_factor, int world_width, int world_height) {
    // Conceptual: JQLD (via qcc_performance_factor) influences how many particles are updated in detail
    // or the complexity of their physics.
    // For Alpha, we can use it to determine the number of particles to process fully.
    // A higher factor means more particles can be processed accurately.

    // For this Alpha, we'll update all active particles but log the factor.
    // A more advanced version might do:
    // size_t particles_to_update_fully = static_cast<size_t>(num_active_particles_ * (qcc_performance_factor / SOME_BASELINE_PERFORMANCE_TARGET));
    // particles_to_update_fully = std::min(num_active_particles_, particles_to_update_fully);

    if (deltaTime <= 0.0f) return; // Avoid issues with zero or negative delta time

    // std::cout << "[ParticleSystem_Alpha] Updating " << num_active_particles_ << " particles. DeltaTime: " << deltaTime
    //           << "s. QCC_Performance_Factor (conceptual for physics scaling): " << qcc_performance_factor << std::endl;

    size_t current_active_count = num_active_particles_;
    size_t processed_this_frame = 0;

    for (size_t i = 0; i < num_active_particles_; ) { // Manual increment of i
        particles_[i].lifetime -= deltaTime;

        if (particles_[i].lifetime <= 0) {
            // "Kill" particle by swapping with the last truly active particle
            if (i < num_active_particles_ - 1) { // If not already the last one
                 std::swap(particles_[i], particles_[num_active_particles_ - 1]);
            }
            num_active_particles_--;
            if (num_active_particles_ == 0) break; // All particles died
            // Don't increment 'i', the new particle at current 'i' needs processing.
            // If i was the one that got swapped to the end, it's fine, loop will exit or i will be > num_active_particles_
            continue;
        }

        // Basic Euler integration for position
        particles_[i].x += particles_[i].vx * deltaTime;
        particles_[i].y += particles_[i].vy * deltaTime;
        // particles_[i].z += particles_[i].vz * deltaTime; // z not used for rendering in 2D

        // Simple boundary collision (bounce)
        if (particles_[i].x <= 0) {
            particles_[i].x = 0; particles_[i].vx *= -0.9f; // Dampen velocity on bounce
        } else if (particles_[i].x >= world_width) {
            particles_[i].x = static_cast<float>(world_width); particles_[i].vx *= -0.9f;
        }
        if (particles_[i].y <= 0) {
            particles_[i].y = 0; particles_[i].vy *= -0.9f;
        } else if (particles_[i].y >= world_height) {
            particles_[i].y = static_cast<float>(world_height); particles_[i].vy *= -0.9f;
        }

        processed_this_frame++;
        i++; // Move to next particle only if current one wasn't swapped
    }

    // Respawn some particles if count drops too low to maintain visual density
    if (num_active_particles_ < max_particles_ / 2 && max_particles_ > 0) {
        size_t to_respawn = std::min(max_particles_ - num_active_particles_, max_particles_ / 10); // Respawn 10% of max
        if(to_respawn > 0) {
            std::cout << "[ParticleSystem_Alpha] Low particle count (" << num_active_particles_ << "), attempting to respawn " << to_respawn << std::endl;
            // Re-initialize a portion of "dead" particles
            std::uniform_real_distribution<float> r_pos_dist_x(0.0f, static_cast<float>(world_width));
            std::uniform_real_distribution<float> r_pos_dist_y(0.0f, static_cast<float>(world_height));

            for(size_t k=0; k < to_respawn; ++k) {
                if (num_active_particles_ >= max_particles_) break;

                Particle& p = particles_[num_active_particles_]; // Get ref to the first inactive particle
                p.x = r_pos_dist_x(random_engine_);
                p.y = r_pos_dist_y(random_engine_);
                p.vx = vel_dist_(random_engine_);
                p.vy = vel_dist_(random_engine_);
                p.color = {
                    static_cast<uint8_t>(random_engine_() % 156 + 100),
                    static_cast<uint8_t>(random_engine_() % 156 + 100),
                    static_cast<uint8_t>(random_engine_() % 156 + 100),
                    static_cast<uint8_t>(random_engine_() % 100 + 155)
                };
                p.size = 1.0f + static_cast<float>(random_engine_() % 30) / 10.0f;
                p.lifetime = 5.0f + static_cast<float>(random_engine_() % 100) / 10.0f;
                num_active_particles_++;
            }
        }
    }

    // Conceptual JQLD-accelerated physics (stubs)
    // buildSpatialPartition();
    // queryCollisions();
}

void ParticleSystem::renderParticles(SDL_Renderer* renderer) const {
    if (!renderer) {
        std::cerr << "[ParticleSystem_Alpha] ERROR: Renderer is null in renderParticles." << std::endl;
        return;
    }

    for (size_t i = 0; i < num_active_particles_; ++i) {
        const auto& p = particles_[i];
        SDL_SetRenderDrawColor(renderer, p.color.r, p.color.g, p.color.b, p.color.a);

        if (p.size <= 1.5f) { // Threshold for drawing point vs rect for better visibility of small particles
            SDL_RenderDrawPoint(renderer, static_cast<int>(p.x), static_cast<int>(p.y));
        } else {
            SDL_FRect particle_rect = { // Use FRect for float precision positioning
                p.x - p.size / 2.0f,
                p.y - p.size / 2.0f,
                p.size,
                p.size
            };
            SDL_RenderFillRectF(renderer, &particle_rect);
        }
    }
}

size_t ParticleSystem::getActiveParticleCount() const {
    return num_active_particles_;
}

void ParticleSystem::buildSpatialPartition() {
    // std::cout << "[ParticleSystem_Alpha] STUB: Building spatial partition (e.g., Octree)." << std::endl;
}

void ParticleSystem::queryCollisions() {
    // std::cout << "[ParticleSystem_Alpha] STUB: Querying collisions within spatial partition." << std::endl;
}

} // namespace NextVerse
