#ifndef GAME_HPP
#define GAME_HPP

// Panda3D Headers
#include "pandaFramework.h"
#include "pandaSystem.h"
#include "windowProperties.h"
#include "nodePath.h"
#include "genericAsyncTask.h"
#include "clockObject.h" // For ClockObject::get_global_clock()
#include "lpoint3.h"     // For LPoint3f
#include "keyboardButton.h" // For KeyboardButton
#include "event.h"          // For Event

// Panda3D HTTP Client related headers
#include "httpClient.h"
#include "httpChannel.h"
#include "urlSpec.h"
#include "ramfile.h"        // For reading response body into Ramfile
#include "filename.h"       // May not be strictly needed for POST, but often with HTTPClient

// Engine Components
#include "DVVE.hpp"
#include "PhysicsEngine.hpp"
#include "AssetManager.hpp"
#include "DVVEParticleSystem.hpp" // Ensure this is present
#include "AICommandServer.hpp"    // For AICommandServer and AICommand, ThreadSafeQueue

#include <string>
#include <vector> // For scene_objects_placeholders_
#include <map>    // For ai_avatar_nodes_
#include <memory> // For std::unique_ptr

// Forward declaration for AICommand might not be needed if AICommandServer.hpp includes it
// struct AICommand;
// class AICommandServer; // Forward declare

class Game {
public:
    Game();
    ~Game();

    // Initialize now takes argc and argv for PandaFramework
    bool initialize(int argc, char* argv[]);
    void runLoop(); // Will call framework_.main_loop()
    void shutdown();

    // Public methods for tests or external control (if any)
    void setMockPCore(double p_core_val);
    void setVisualComplexity(double f_v_val);
    double getCurrentRp() const { return current_Rp_; }
    int getNumObjectsToRender() const { return num_objects_to_render_; }
    const std::string& getDetailLevel() const { return detail_level_; }
    const DVVEParameters& getDVVEParams() const { return dvve_params_; }

    // Static callback for Panda3D task
    static AsyncTask::DoneStatus gameUpdateTask(GenericAsyncTask* task, void* data);

    bool shouldQuitPandaLoop() const { return quit_panda_loop_; }

    // Control/AI surface (definitions live in src/Game.cpp; declared here so
    // callbacks and the test-suite translation units can compile).
    void update(double deltaTime) { updateInternal(deltaTime); }
    void sendContextualQueryToAI(const std::string& object_id);
    void processAICommands(double dt);
    static void userRequestQuit(const Event* event, void* data);

private:
    // Internal game logic methods called by gameUpdateTask
    void processInput();
    void updateInternal(double deltaTime); // Renamed from update to avoid conflict if any base class has update
    void renderSceneLogic(); // Placeholder for logic that affects rendering

    PandaFramework framework_;
    WindowFramework* window_ = nullptr; // Panda's window
    NodePath scene_root_; // Root of our game scene elements
    NodePath camera_; // Game camera

    // Models for demonstrating complexity changes
    NodePath model_base_environment_; // The main environment model from step 1
    NodePath model_complex_detail_;   // An additional model to show when performance is high
    NodePath model_medium_detail_;    // An alternative or simpler additional model
    std::vector<NodePath> optional_decorative_objects_; // A few small objects

    // Engine components
    DVVEParameters dvve_params_;
    DVVECalculator dvve_calculator_;
    PhysicsEngine physics_engine_;
    AssetManager asset_manager_;

    // State variables
    double current_Rp_ = 0.0;
    int num_objects_to_render_ = 10;
    std::string detail_level_ = "Medium";
    double target_p_core_for_max_quality_ = 2.0e9;
    int loop_iteration_count_ = 0;

    std::vector<SceneObjectPlaceholder> scene_objects_placeholders_;

    DVVEParticleSystem dvve_particle_system_;
    bool enable_particle_system_ = true;
    bool quit_panda_loop_ = false; // For SDL/particle system quit requests
    bool user_quit_requested_ = false; // For user-initiated quit (e.g., Esc key)

    // Player Avatar
    NodePath player_avatar_node_;
    LPoint3f player_pos_;

    // AI Controlled Avatars
    std::map<std::string, NodePath> ai_avatar_nodes_;
    // std::map<std::string, LPoint3f> ai_avatar_targets_; // For more complex movement later

    // AI Command Server & Queue
    std::unique_ptr<AICommandServer> ai_command_server_;
    ThreadSafeQueue<AICommand> ai_command_queue_;


    // For particle system profiling
    double particle_event_time_ms_ = 0.0;
    double particle_update_time_ms_ = 0.0;
    double particle_render_time_ms_ = 0.0;
    long long particle_total_frames_processed_ = 0;
    long long particle_profile_log_interval_ = 180;

    PT(Texture) panda_particle_texture_; // For Panda3D representation of particle texture
    NodePath particle_card_node_; // For displaying the particle texture in Panda3D

    // HTTP Client for AI Interaction
    PT(HTTPClient) http_client_;
};

#endif // GAME_HPP
