#include "game_engine_sdl_alpha.hpp"
#include <stdexcept> // For error handling
#include <algorithm> // For std::min/max
#include <iomanip>   // For std::fixed, std::setprecision

namespace NextVerse {

GameEngineSDL_Alpha::GameEngineSDL_Alpha(QuantumComputationalCore& qcc, QuantumSuperVM& qsvm)
    : qcc_(qcc), qsvm_(qsvm), particle_render_budget_(MAX_PARTICLES_TOTAL) { // Initialize with max budget
    std::cout << "[GameEngineSDL_Alpha] Initializing with QCC and QSVM references." << std::endl;
}

GameEngineSDL_Alpha::~GameEngineSDL_Alpha() {
    if (renderer_) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
    std::cout << "[GameEngineSDL_Alpha] SDL resources released." << std::endl;
}

bool GameEngineSDL_Alpha::initialize(const std::string& title, int width, int height) {
    screen_width_ = width;
    screen_height_ = height;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "[GameEngineSDL_Alpha] ERROR: SDL_Init failed: " << SDL_GetError() << std::endl;
        return false;
    }

    window_ = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                             screen_width_, screen_height_, SDL_WINDOW_SHOWN);
    if (!window_) {
        std::cerr << "[GameEngineSDL_Alpha] ERROR: SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return false;
    }

    // Per CrashOverrideX, using software rendering.
    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer_) {
        std::cerr << "[GameEngineSDL_Alpha] ERROR: SDL_CreateRenderer (software) failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window_);
        SDL_Quit();
        return false;
    }
    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND); // Enable alpha blending

    std::cout << "[GameEngineSDL_Alpha] SDL initialized successfully. Window: " << width << "x" << height << std::endl;

    loadGameAssets(); // Alpha: Initializes placeholders

    particle_system_ = std::make_unique<ParticleSystem>(MAX_PARTICLES_TOTAL);
    particle_system_->initializeParticles(MAX_PARTICLES_TOTAL, screen_width_, screen_height_);
    std::cout << "[GameEngineSDL_Alpha] Particle system initialized with " << MAX_PARTICLES_TOTAL << " particles." << std::endl;

    // Initialize a placeholder AI avatar
    ai_avatars_.emplace_back("AlphaAvatar01", static_cast<float>(screen_width_/2), static_cast<float>(screen_height_/2));
    ai_avatars_[0].color = {0, 255, 0, 150}; // Green, semi-transparent
    ai_avatars_[0].size = 20.0f;
    std::cout << "[GameEngineSDL_Alpha] Placeholder AI avatar initialized." << std::endl;

    is_running_ = true;
    return true;
}

void GameEngineSDL_Alpha::loadGameAssets() {
    std::cout << "[GameEngineSDL_Alpha] STUB: loadGameAssets() called. (No actual assets loaded for Alpha)" << std::endl;
    // For Alpha, this could set up default particle properties or avatar appearances if not done elsewhere.
}

// DVVE: Rp = P_core * (phi_v / (1 - nu_v)) * F_v
double GameEngineSDL_Alpha::calculateDVVERenderingPerformance(double p_core) const {
    std::cout << "[GameEngineSDL_Alpha] Calculating DVVE Performance (New Formula)..." << std::endl;
    if (p_core <= 0 && !(std::abs(p_core) < 1e-9)) { // Allow p_core = 0
        std::cerr << "[GameEngineSDL_Alpha] Warning: P_core (QSVM performance = " << p_core << ") is non-positive. DVVE result will be <= 0." << std::endl;
    }
    double resistance_term = 1.0 - dvve_params_.nu_v_vortical_resistance;
    if (std::abs(resistance_term) < 1e-9) {
        std::cerr << "[GameEngineSDL_Alpha] ERROR: DVVE nu_v_vortical_resistance (" << dvve_params_.nu_v_vortical_resistance
                  << ") makes (1 - nu_v) zero. Division by zero." << std::endl;
        return 0.0;
    }
    if (dvve_params_.nu_v_vortical_resistance >= 1.0) {
         std::cerr << "[GameEngineSDL_Alpha] ERROR: DVVE nu_v_vortical_resistance (" << dvve_params_.nu_v_vortical_resistance
                   << ") must be less than 1 for positive denominator." << std::endl;
        return 0.0;
    }
    if (dvve_params_.phi_v_vertex_flux < 0) {
        std::cerr << "[GameEngineSDL_Alpha] Warning: DVVE phi_v_vertex_flux (" << dvve_params_.phi_v_vertex_flux <<") is negative." << std::endl;
    }
     if (dvve_params_.F_v_visual_complexity < 0) {
        std::cerr << "[GameEngineSDL_Alpha] Warning: DVVE F_v_visual_complexity (" << dvve_params_.F_v_visual_complexity <<") is negative." << std::endl;
    }

    double performance_metric = p_core *
                                (dvve_params_.phi_v_vertex_flux / resistance_term) *
                                dvve_params_.F_v_visual_complexity;

    std::cout << "[GameEngineSDL_Alpha] DVVE Calculation: P_core=" << std::fixed << std::setprecision(0) << p_core
              << ", phi_v=" << dvve_params_.phi_v_vertex_flux
              << ", nu_v=" << dvve_params_.nu_v_vortical_resistance
              << ", F_v=" << dvve_params_.F_v_visual_complexity
              << ", Resulting Rp (metric)=" << std::fixed << std::setprecision(2) << performance_metric << std::endl;
    return performance_metric;
}


void GameEngineSDL_Alpha::handleEvents() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            is_running_ = false;
            std::cout << "[GameEngineSDL_Alpha] SDL_QUIT event received. Shutting down." << std::endl;
        }
        // Add other event handling (keyboard, mouse) here if needed for Alpha
    }
}

void GameEngineSDL_Alpha::update(float deltaTime) {
    // 1. Get Performance Core value from QSVM
    // QSVM returns std::optional<double>; fall back to 0.0 when unavailable.
    double p_core = qsvm_.getAmplifiedPerformance().value_or(0.0);
    std::cout << "[GameEngineSDL_Alpha] Update Cycle - DeltaTime: " << deltaTime
              << "s, Fetched P_core from QSVM: " << std::fixed << std::setprecision(0) << p_core << std::endl;

    // 2. Calculate DVVE Rendering Performance/Capacity
    //    F_v could be dynamically tied to active particle count relative to max capacity.
    if (MAX_PARTICLES_TOTAL > 0) {
        dvve_params_.F_v_visual_complexity = static_cast<double>(particle_system_ ? particle_system_->getActiveParticleCount() : 0) / MAX_PARTICLES_TOTAL;
        if (dvve_params_.F_v_visual_complexity < 0.1) dvve_params_.F_v_visual_complexity = 0.1; // Minimum complexity if some particles exist
    } else {
        dvve_params_.F_v_visual_complexity = 1.0; // Default if no particles defined
    }

    double dvve_rendering_capacity_metric = calculateDVVERenderingPerformance(p_core);

    // 3. Adjust particle render/update budget based on DVVE capacity (conceptual for Alpha)
    //    This logic needs careful calibration. For Alpha, it's a simplified interpretation.
    //    Rp is a capacity metric. Higher Rp allows more complex rendering/more particles at target FPS.
    //    Let's assume Rp directly correlates to how many particles we can afford to process/render.
    //    The CrashOverrideX target is 100k particles @ 50 FPS with P_core from a 1.1GHz CPU.
    //    A very high Rp might mean we can render all MAX_PARTICLES_TOTAL.
    //    A lower Rp means we should render fewer to maintain FPS.

    // Example: If Rp is some factor times MAX_PARTICLES_TOTAL * target_fps, we can render all.
    // This is highly conceptual for Alpha.
    double base_render_load_per_particle = 50000; // Arbitrary cycles per particle for rendering/updates.
    size_t affordable_particles = 0;
    if (base_render_load_per_particle > 1e-9 && target_fps_ > 1e-9) {
         affordable_particles = static_cast<size_t>(dvve_rendering_capacity_metric / (base_render_load_per_particle * target_fps_));
    }

    particle_render_budget_ = std::min(MAX_PARTICLES_TOTAL, affordable_particles);
    // particle_render_budget_ = MAX_PARTICLES_TOTAL; // For Alpha, force max to test system load

    std::cout << "[GameEngineSDL_Alpha] DVVE Capacity Metric: " << dvve_rendering_capacity_metric
              << ". Affordable Particles (conceptual): " << affordable_particles
              << ". Current Particle Render Budget: " << particle_render_budget_ << std::endl;

    // 4. Update game systems
    processAICommands();
    updateAIAvatars(deltaTime);

    if (particle_system_) {
        // Pass p_core (or a derivative) to particle system for JQLD-like physics scaling
        particle_system_->updateParticles(deltaTime, p_core, screen_width_, screen_height_);
    }
}

void GameEngineSDL_Alpha::render() {
    SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
    SDL_RenderClear(renderer_);

    if (particle_system_) {
        // Potentially use particle_render_budget_ in renderParticles if implemented there
        particle_system_->renderParticles(renderer_);
    }
    renderAIAvatars();

    SDL_RenderPresent(renderer_);
}

void GameEngineSDL_Alpha::runMainLoop() {
    std::cout << "[GameEngineSDL_Alpha] Starting main loop." << std::endl;
    last_tick_ = SDL_GetPerformanceCounter();

    while (is_running_) {
        Uint64 current_tick = SDL_GetPerformanceCounter();
        float deltaTime = static_cast<float>(current_tick - last_tick_) / SDL_GetPerformanceFrequency();
        if (deltaTime == 0.0f) { // Avoid division by zero if ticks are too fast / same
             std::this_thread::sleep_for(std::chrono::microseconds(100)); // Small sleep
             current_tick = SDL_GetPerformanceCounter();
             deltaTime = static_cast<float>(current_tick - last_tick_) / SDL_GetPerformanceFrequency();
        }
        last_tick_ = current_tick;

        // Cap deltaTime to prevent spiral of death on major hitches, e.g., max 100ms (10 FPS)
        deltaTime = std::min(0.1f, deltaTime);


        handleEvents();
        update(deltaTime);
        render();

        // Optional: Simple FPS display
        // static int frame_count = 0;
        // static float time_accumulator = 0.0f;
        // frame_count++;
        // time_accumulator += deltaTime;
        // if (time_accumulator >= 1.0f) {
        //     std::cout << "FPS: " << frame_count / time_accumulator << std::endl;
        //     frame_count = 0;
        //     time_accumulator = 0.0f;
        // }
    }
    std::cout << "[GameEngineSDL_Alpha] Main loop ended." << std::endl;
}


void GameEngineSDL_Alpha::processAICommands() {
    // STUB: Simulate receiving a command to change avatar state
    static int command_timer = 0;
    command_timer++;
    if (command_timer % 300 == 0 && !ai_avatars_.empty()) { // Every ~5-6 seconds at 50 FPS
        ai_avatars_[0].current_action_text = "speaking_simulated";
        std::cout << "[GameEngineSDL_Alpha] STUB: AI Command Received - Avatar '" << ai_avatars_[0].id << "' to 'speaking_simulated'" << std::endl;
    } else if (command_timer % 300 == 150 && !ai_avatars_.empty()) {
        ai_avatars_[0].current_action_text = "idle_simulated";
         std::cout << "[GameEngineSDL_Alpha] STUB: AI Command Received - Avatar '" << ai_avatars_[0].id << "' to 'idle_simulated'" << std::endl;
    }
}

void GameEngineSDL_Alpha::updateAIAvatars(float deltaTime) {
    for (auto& avatar : ai_avatars_) {
        if (avatar.current_action_text == "speaking_simulated") {
            // Simple visual feedback: slight size oscillation
            avatar.size = 20.0f + 5.0f * std::abs(std::sin(static_cast<float>(SDL_GetTicks()) / 200.0f));
        } else {
            avatar.size = 20.0f;
        }
        // Keep avatar centered for this simple demo
        avatar.x = static_cast<float>(screen_width_ / 2);
        avatar.y = static_cast<float>(screen_height_ / 2);
    }
}

void GameEngineSDL_Alpha::renderAIAvatars() {
    if (!renderer_) return;
    for (const auto& avatar : ai_avatars_) {
        SDL_Rect avatar_rect = {
            static_cast<int>(avatar.x - avatar.size / 2),
            static_cast<int>(avatar.y - avatar.size / 2),
            static_cast<int>(avatar.size),
            static_cast<int>(avatar.size)
        };
        SDL_SetRenderDrawColor(renderer_, avatar.color.r, avatar.color.g, avatar.color.b, avatar.color.a);
        SDL_RenderFillRect(renderer_, &avatar_rect);
        // In a real version, you'd render text for current_action_text using SDL_ttf
        // For Alpha, visual change (like size oscillation) indicates state change.
    }
}

} // namespace NextVerse
