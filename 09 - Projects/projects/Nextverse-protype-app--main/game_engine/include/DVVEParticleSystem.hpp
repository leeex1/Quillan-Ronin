#ifndef DVVEPARTICLESYSTEM_HPP
#define DVVEPARTICLESYSTEM_HPP

#include <SDL.h>
#include <string>
#include <vector>
#include "Particle.hpp"

class DVVEParticleSystem {
public:
    DVVEParticleSystem();
    ~DVVEParticleSystem();

    bool initialize(const char* window_title = "DVVE Particles (SDL2)", int screen_width = 320, int screen_height = 240);
    void handleEvents();
    void update(double deltaTime);
    void render();
    void shutdown();

    void setTargetParticleCount(int count);
    int getActiveParticleCount() const;

    bool isInitialized() const { return is_sdl_initialized_; }
    bool shouldQuit() const { return quit_sdl_loop_; }

    // Read-only view of the live particle buffer (required by Game.cpp collision offload).
    const std::vector<Particle>& getParticles() const { return particles_; }

    // Pixel data access
    int getPixelDataWidth() const { return screen_width_; }
    int getPixelDataHeight() const { return screen_height_; }
    SDL_PixelFormatEnum getPixelDataFormat() const; // To confirm format
    const std::vector<unsigned char>& getPixelDataBuffer() const;

private:
    void initParticles();
    void updatePixelBuffer(); // New private method

    SDL_Window* sdl_window_ = nullptr;
    SDL_Renderer* sdl_renderer_ = nullptr;
    SDL_Texture* sdl_render_target_texture_ = nullptr;

    std::vector<Particle> particles_;
    std::vector<SDL_FPoint> points_buffer_;
    std::vector<unsigned char> cpu_pixel_buffer_; // For SDL_RenderReadPixels

    int current_target_particle_count_ = 1000;
    int screen_width_ = 0;
    int screen_height_ = 0;

    bool is_sdl_initialized_ = false;
    bool quit_sdl_loop_ = false;
};

#endif // DVVEPARTICLESYSTEM_HPP
