#ifndef PARTICLE_SYSTEM_ALPHA_HPP
#define PARTICLE_SYSTEM_ALPHA_HPP

#include <vector>
#include <random> // For basic particle initialization/movement
#include "game_engine_common_alpha.hpp"
#include "qcc_alpha.hpp" // For conceptual JQLD access

// Forward declare SDL_Renderer if only used for rendering method signature
struct SDL_Renderer;

namespace NextVerse {

    /**
     * @brief Manages a large number of particles for the Game Engine.
     * Physics updates are conceptually accelerated by JQLD via a performance factor.
     */
    class ParticleSystem {
    public:
        ParticleSystem(size_t max_particles = 100000); // As per DVVE benchmark

        /**
         * @brief Updates particle physics (movement, simple collisions).
         * The qcc_performance_factor (derived from JQLD) determines how many particles
         * can be updated or how complex their updates can be within a frame time.
         * @param deltaTime Time since last frame.
         * @param qcc_performance_factor Factor representing QCC's current processing power.
         * @param world_width Width of the simulation area.
         * @param world_height Height of the simulation area.
         */
        void updateParticles(float deltaTime, double qcc_performance_factor, int world_width, int world_height);

        /**
         * @brief Renders all active particles.
         * @param renderer SDL renderer instance.
         */
        void renderParticles(SDL_Renderer* renderer) const;

        /**
         * @brief Resets or initializes particles to random states.
         * @param num_to_spawn Number of particles to initialize/respawn.
         * @param world_width Width of the spawn area.
         * @param world_height Height of the spawn area.
         */
        void initializeParticles(size_t num_to_spawn, int world_width, int world_height);

        size_t getActiveParticleCount() const;

    private:
        std::vector<Particle> particles_;
        size_t max_particles_;
        size_t num_active_particles_ = 0; // Live count maintained by the alpha .cpp

        // For random generation in particle initialization
        std::mt19937 random_engine_;
        std::uniform_real_distribution<float> pos_dist_x_;
        std::uniform_real_distribution<float> pos_dist_y_;
        std::uniform_real_distribution<float> vel_dist_;

        // Conceptual Octree for spatial partitioning (Alpha: could be simplified or stubbed)
        // OctreeAlpha octree_;
        void buildSpatialPartition(); // Placeholder
        void queryCollisions();       // Placeholder
    };

} // namespace NextVerse

#endif // PARTICLE_SYSTEM_ALPHA_HPP
