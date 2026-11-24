#ifndef GAME_ENGINE_SDL_GROK2_HPP
#define GAME_ENGINE_SDL_GROK2_HPP

// This snippet is from docs/Documentation/grok 2 code/game_engine.txt
// It depicts a simple GameEngine using SDL2 for rendering particles.

#include "quantum_vm.hpp" // Assuming quantum_vm.hpp defines QuantumVM
#include <SDL2/SDL.h> // Make sure SDL2 is correctly linked and include paths are set up in a real build
#include <boost/container/vector.hpp> // Included in original, but not directly used in this snippet
#include <cstdlib> // For rand()

class GameEngine_SDL_Grok2 {
public:
    GameEngine_SDL_Grok2(QuantumVM& vm) : vm_(vm), window_(nullptr), renderer_(nullptr) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            // Handle SDL initialization error, e.g., log SDL_GetError()
            return;
        }
        window_ = SDL_CreateWindow("NextVerse SDL Example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
        if (!window_) {
            // Handle window creation error
            SDL_Quit();
            return;
        }
        // Using software renderer as in snippet. For performance, SDL_RENDERER_ACCELERATED is preferred.
        renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
        if (!renderer_) {
            // Handle renderer creation error
            SDL_DestroyWindow(window_);
            SDL_Quit();
        }
    }

    ~GameEngine_SDL_Grok2() {
        if (renderer_) {
            SDL_DestroyRenderer(renderer_);
        }
        if (window_) {
            SDL_DestroyWindow(window_);
        }
        SDL_Quit();
    }

    // Render particles: R_p = P_core * F_v
    void render_particles(int num_particles) {
        if (!renderer_) return;

        // vm_.offload_task is conceptual. Direct rendering shown.
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255); // Black background
        SDL_RenderClear(renderer_);
        SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255); // White particles
        for (int i = 0; i < num_particles; ++i) {
            int x = std::rand() % 800;
            int y = std::rand() % 600;
            SDL_RenderDrawPoint(renderer_, x, y);
        }
        SDL_RenderPresent(renderer_);
    }

private:
    QuantumVM& vm_;
    SDL_Window* window_;
    SDL_Renderer* renderer_;
};

#endif // GAME_ENGINE_SDL_GROK2_HPP 