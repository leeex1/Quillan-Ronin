#ifndef GAME_ENGINE_COMMON_ALPHA_HPP
#define GAME_ENGINE_COMMON_ALPHA_HPP

#include <cstdint> // uint8_t
#include <utility> // std::move
#include <string>
#include <vector>

namespace NextVerse {

    // As defined in CrashOverrideX guide
    struct Particle {
        float x, y, z; // z might be used for depth sorting or simple effects in 2D
        float vx, vy, vz;
        // Potentially add color, size, lifetime for Alpha
        struct {
            uint8_t r, g, b, a;
        } color;
        float size;
        float lifetime;

        Particle() : x(0), y(0), z(0), vx(0), vy(0), vz(0),
                     color{255,255,255,255}, size(1.0f), lifetime(10.0f) {}
    };

    // Simple representation for an AI Avatar for Alpha
    struct AIAvatarPlaceholderAlpha {
        std::string id;
        float x, y; // Position in 2D space
        struct {
            uint8_t r, g, b, a;
        } color;
        float size;
        std::string current_action_text; // e.g., "idle", "speaking"

        AIAvatarPlaceholderAlpha(std::string id_ = "avatar1", float x_ = 0, float y_ = 0)
            : id(std::move(id_)), x(x_), y(y_), color{0, 255, 0, 255}, size(10.0f) {}
    };


} // namespace NextVerse


#endif // GAME_ENGINE_COMMON_ALPHA_HPP
