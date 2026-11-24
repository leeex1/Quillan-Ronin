#include "DVVEParticleSystem.hpp"
#include <iostream>
#include <random>
#include <algorithm> // For std::max
// Ensure SDL_render.h is effectively included via SDL.h for SDL_RenderReadPixels

DVVEParticleSystem::DVVEParticleSystem()
    : sdl_window_(nullptr), sdl_renderer_(nullptr), sdl_render_target_texture_(nullptr),
      is_sdl_initialized_(false), quit_sdl_loop_(false),
      current_target_particle_count_(1000),
      screen_width_(0), screen_height_(0) {}

DVVEParticleSystem::~DVVEParticleSystem() {
    if (is_sdl_initialized_) {
        shutdown();
    }
}

bool DVVEParticleSystem::initialize(const char* window_title, int S_width, int S_height) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return false;
    }
    screen_width_ = S_width;
    screen_height_ = S_height;

    sdl_window_ = SDL_CreateWindow(window_title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                screen_width_, screen_height_, SDL_WINDOW_SHOWN);
    if (!sdl_window_) {
        std::cerr << "SDL Window Error: " << SDL_GetError() << std::endl;
        SDL_Quit(); return false;
    }

    sdl_renderer_ = SDL_CreateRenderer(sdl_window_, -1, SDL_RENDERER_ACCELERATED);
    if (!sdl_renderer_) {
        std::cerr << "SDL Renderer Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(sdl_window_); sdl_window_ = nullptr;
        SDL_Quit(); return false;
    }

    SDL_SetRenderDrawBlendMode(sdl_renderer_, SDL_BLENDMODE_BLEND);

    sdl_render_target_texture_ = SDL_CreateTexture(sdl_renderer_, SDL_PIXELFORMAT_RGBA8888,
                                                 SDL_TEXTUREACCESS_TARGET, screen_width_, screen_height_);
    if (!sdl_render_target_texture_) {
        std::cerr << "Failed to create render target texture! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(sdl_renderer_); sdl_renderer_ = nullptr;
        SDL_DestroyWindow(sdl_window_); sdl_window_ = nullptr;
        SDL_Quit();
        return false;
    }
    SDL_SetTextureBlendMode(sdl_render_target_texture_, SDL_BLENDMODE_BLEND);

    // Resize CPU pixel buffer
    cpu_pixel_buffer_.resize(screen_width_ * screen_height_ * 4); // 4 bytes for RGBA8888

    initParticles();

    is_sdl_initialized_ = true;
    quit_sdl_loop_ = false;
    std::cout << "DVVEParticleSystem (SDL2) initialized. CPU pixel buffer size: " << cpu_pixel_buffer_.size() << std::endl;
    return true;
}

void DVVEParticleSystem::initParticles() {
    particles_.assign(current_target_particle_count_, Particle());
    points_buffer_.resize(current_target_particle_count_);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distX(0.0f, static_cast<float>(screen_width_));
    std::uniform_real_distribution<float> distY(0.0f, static_cast<float>(screen_height_));
    std::uniform_real_distribution<float> distVX(-50.0f, 50.0f);
    std::uniform_real_distribution<float> distVY(-50.0f, 50.0f);
    for (auto& p : particles_) {
        p.x = distX(rng); p.y = distY(rng); p.vx = distVX(rng); p.vy = distVY(rng);
    }
}

void DVVEParticleSystem::setTargetParticleCount(int count) {
    int new_count = std::max(0, count);
    bool target_changed = (new_count != current_target_particle_count_);

    if (target_changed) {
        current_target_particle_count_ = new_count;
    }

    if ((static_cast<int>(particles_.size()) != current_target_particle_count_ ||
         points_buffer_.size() != current_target_particle_count_) &&
        is_sdl_initialized_) {
        initParticles();
    } else if (target_changed && !is_sdl_initialized_) {
        // Handled by current_target_particle_count_ update
    } else if (!target_changed && static_cast<int>(particles_.size()) == current_target_particle_count_ && points_buffer_.size() != current_target_particle_count_ && is_sdl_initialized_) {
        points_buffer_.resize(current_target_particle_count_);
    }
}

int DVVEParticleSystem::getActiveParticleCount() const { return static_cast<int>(particles_.size()); }

void DVVEParticleSystem::handleEvents() {
    if (!is_sdl_initialized_ || quit_sdl_loop_) return; SDL_Event e;
    while (SDL_PollEvent(&e) != 0) { if (e.type == SDL_QUIT) { quit_sdl_loop_ = true; std::cout << "DVVEParticleSystem: SDL_QUIT event received." << std::endl;}}}

void DVVEParticleSystem::update(double deltaTime) {
    if (!is_sdl_initialized_ || quit_sdl_loop_ || particles_.empty()) return; float dt = static_cast<float>(deltaTime);
    for (auto& p : particles_) { p.x += p.vx * dt; p.y += p.vy * dt;
        if (p.x < 0) { p.x = 0; p.vx *= -1.0f; } else if (p.x > static_cast<float>(screen_width_)) { p.x = static_cast<float>(screen_width_); p.vx *= -1.0f; }
        if (p.y < 0) { p.y = 0; p.vy *= -1.0f; } else if (p.y > static_cast<float>(screen_height_)) { p.y = static_cast<float>(screen_height_); p.vy *= -1.0f; }}}

void DVVEParticleSystem::updatePixelBuffer() {
    if (!is_sdl_initialized_ || !sdl_render_target_texture_ || !sdl_renderer_ || cpu_pixel_buffer_.empty()) return;

    // The render target should already be sdl_render_target_texture_ if this is called after drawing particles to it.
    int pitch = screen_width_ * 4; // 4 bytes for RGBA8888
    if (SDL_RenderReadPixels(sdl_renderer_, nullptr, SDL_PIXELFORMAT_RGBA8888,
                             cpu_pixel_buffer_.data(), pitch) != 0) {
        std::cerr << "Failed to read pixels from render target texture! SDL_Error: " << SDL_GetError() << std::endl;
    }
}

SDL_PixelFormatEnum DVVEParticleSystem::getPixelDataFormat() const {
    return SDL_PIXELFORMAT_RGBA8888; // The format we are using for the buffer and texture
}

const std::vector<unsigned char>& DVVEParticleSystem::getPixelDataBuffer() const {
    return cpu_pixel_buffer_;
}

void DVVEParticleSystem::render() {
    if (!is_sdl_initialized_ || quit_sdl_loop_ || !sdl_render_target_texture_) return;

    SDL_SetRenderTarget(sdl_renderer_, sdl_render_target_texture_);
    SDL_SetRenderDrawColor(sdl_renderer_, 0, 0, 0, 0);
    SDL_RenderClear(sdl_renderer_);

    if (!particles_.empty()) {
        SDL_SetRenderDrawColor(sdl_renderer_, 255, 255, 255, 150);
        if (points_buffer_.size() != particles_.size()) { points_buffer_.resize(particles_.size()); }
        for (size_t i = 0; i < particles_.size(); ++i) {
            points_buffer_[i] = {particles_[i].x, particles_[i].y};
        }
        if (!points_buffer_.empty()){
           SDL_RenderDrawPointsF(sdl_renderer_, points_buffer_.data(), static_cast<int>(points_buffer_.size()));
        }
    }

    updatePixelBuffer(); // Update the CPU buffer with the texture content

    SDL_SetRenderTarget(sdl_renderer_, nullptr); // Reset to window
    SDL_SetRenderDrawColor(sdl_renderer_, 50, 50, 50, 255);
    SDL_RenderClear(sdl_renderer_);
    SDL_RenderCopy(sdl_renderer_, sdl_render_target_texture_, nullptr, nullptr);
    SDL_RenderPresent(sdl_renderer_);
}

void DVVEParticleSystem::shutdown() {
    if (is_sdl_initialized_) {
        std::cout << "Shutting down DVVEParticleSystem (SDL2)..." << std::endl;
        particles_.clear();
        points_buffer_.clear();
        cpu_pixel_buffer_.clear(); // Clear the CPU pixel buffer
        if (sdl_render_target_texture_) { SDL_DestroyTexture(sdl_render_target_texture_); sdl_render_target_texture_ = nullptr; }
        if (sdl_renderer_) { SDL_DestroyRenderer(sdl_renderer_); sdl_renderer_ = nullptr; }
        if (sdl_window_) { SDL_DestroyWindow(sdl_window_); sdl_window_ = nullptr; }
        SDL_Quit();
        is_sdl_initialized_ = false;
        quit_sdl_loop_ = true;
    }
}
