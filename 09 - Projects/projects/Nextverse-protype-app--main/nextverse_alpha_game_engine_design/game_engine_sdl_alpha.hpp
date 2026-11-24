#ifndef GAME_ENGINE_SDL_ALPHA_HPP
#define GAME_ENGINE_SDL_ALPHA_HPP

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include <SDL2/SDL.h>
#include <iostream> // For logging
#include <cmath>    // For std::abs

#include "quantum_common_alpha.hpp"       // For AIAvatarPlaceholderAlpha
#include "game_engine_common_alpha.hpp"   // For Particle
#include "particle_system_alpha.hpp"      // For ParticleSystem
#include "qcc_alpha.hpp"                  // For QuantumComputationalCore
#include "qsvm_alpha.hpp"                 // For QuantumSuperVM

namespace NextVerse {

    class GameEngineSDL_Alpha {
    public:
        struct DVVEParams {
            double phi_v_vertex_flux;       // phi_v: Represents rendering throughput/efficiency factor
            double nu_v_vortical_resistance; // nu_v: Represents resistance/overhead to rendering (should be < 1)
            double F_v_visual_complexity;   // F_v: Factor for overall scene/particle complexity

            DVVEParams() : phi_v_vertex_flux(1.0),
                           nu_v_vortical_resistance(0.1), // Must be < 1.0
                           F_v_visual_complexity(1.0) {}
        };

        GameEngineSDL_Alpha(QuantumComputationalCore& qcc, QuantumSuperVM& qsvm);
        ~GameEngineSDL_Alpha();

        bool initialize(const std::string& title, int width, int height);
        void runMainLoop();

    private:
        void handleEvents();
        void update(float deltaTime);
        void render();

        /**
         * @brief Calculates the rendering performance capacity based on the new DVVE formula:
         * Rp = P_core * (phi_v / (1 - nu_v)) * F_v
         * @param p_core The P_core value obtained from QCC/QSVM (getAmplifiedPerformance).
         * @return A metric representing rendering capacity (e.g., sustainable particle updates/renders per second).
         */
        double calculateDVVERenderingPerformance(double p_core) const;

        void loadGameAssets();
        void updateAIAvatars(float deltaTime);
        void renderAIAvatars();
        void processAICommands(); // Placeholder

        SDL_Window* window_ = nullptr;
        SDL_Renderer* renderer_ = nullptr;
        bool is_running_ = false;
        int screen_width_ = 800;
        int screen_height_ = 600;

        QuantumComputationalCore& qcc_;
        QuantumSuperVM& qsvm_;
        DVVEParams dvve_params_;

        std::unique_ptr<ParticleSystem> particle_system_;
        std::vector<AIAvatarPlaceholderAlpha> ai_avatars_;

        Uint64 last_tick_ = 0;
        float target_fps_ = 50.0f; // Target for DVVE benchmark

        // This will be dynamically adjusted based on calculateDVVERenderingPerformance
        size_t particle_render_budget_;
        static constexpr size_t MAX_PARTICLES_TOTAL = 100000; // Compile-time constant: avoids member init-order UB
    };

} // namespace NextVerse

#endif // GAME_ENGINE_SDL_ALPHA_HPP
