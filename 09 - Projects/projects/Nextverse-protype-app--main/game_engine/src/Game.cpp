#include "Game.hpp"
#include "load_prc_file.h" // For load_prc_file_data
#include "keyboardButton.h" // For KeyboardButton, already in Game.hpp but good for explicitness
#include "event.h"          // For Event, already in Game.hpp
#include <iostream>
#include <iomanip> // For logging output formatting
#include <chrono>  // For profiling
#include "texture.h"        // For Texture
#include "samplerState.h"   // For SamplerState
#include "pta_uchar.h"      // For PTA_uchar
#include "cardMaker.h"
#include "transparencyAttrib.h"

Game::Game() : window_(nullptr), loop_iteration_count_(0),
             particle_event_time_ms_(0.0), particle_update_time_ms_(0.0), particle_render_time_ms_(0.0),
             particle_total_frames_processed_(0), particle_profile_log_interval_(180) { // Initialize members
    // panda_particle_texture_ is default constructed (nullptr)
    std::cout << "Game instance created." << std::endl;
    // Default DVVE params (can be changed by setters)
    dvve_params_.P_core = 1.5e9;
    dvve_params_.F_v = 1.0;
    dvve_params_.nu_v = 0.05;
    // target_p_core_for_max_quality_ is already initialized in Game.hpp
    // current_Rp_, num_objects_to_render_, detail_level_ also have defaults in Game.hpp
}

Game::~Game() {
    std::cout << "Game instance destroyed." << std::endl;
}

// Initialize now takes argc and argv for PandaFramework
bool Game::initialize(int argc, char* argv[]) {
    std::cout << "Game initializing with Panda3D..." << std::endl;

    // Optionally load Panda3D config data
    // load_prc_file_data("", "window-title NextVerse Game Engine");
    // load_prc_file_data("", "fullscreen #f");
    // load_prc_file_data("", "win-size 1280 720");

    framework_.open_framework(argc, argv);
    framework_.set_window_title("NextVerse Panda3D Engine");

    window_ = framework_.open_window();
    if (!window_) {
        std::cerr << "Could not open Panda3D window!" << std::endl;
        framework_.close_framework(); // Clean up framework if window fails
        return false;
    }
    camera_ = window_->get_camera_group();
    scene_root_ = window_->get_render();

    // AI Command Server Initialization
    const int GAME_ENGINE_COMMAND_PORT = 8081; // Define port for AI commands
    // ai_command_queue_ is already a member, no need to pass a pointer to a local.
    // The AICommandServer will get a pointer to the member queue.
    ai_command_server_ = std::make_unique<AICommandServer>(GAME_ENGINE_COMMAND_PORT, &ai_command_queue_);
    ai_command_server_->start_listening();
    std::cout << "AICommandServer initialized and listening on port " << GAME_ENGINE_COMMAND_PORT << "." << std::endl;

    // Window properties (alternative to prc file)
    WindowProperties props = window_->get_graphics_window()->get_properties();
    props.set_size(1280, 720);
    // props.set_title("NextVerse Panda3D Engine"); // Title already set by framework_.set_window_title
    window_->get_graphics_window()->request_properties(props);

    std::cout << "Panda3D window opened." << std::endl;

    asset_manager_ = AssetManager(); // Initialize asset manager
    std::cout << "AssetManager initialized for Game." << std::endl;

    // Load base environment model
    model_base_environment_ = asset_manager_.loadModel("models/environment");
    if (!model_base_environment_.is_empty()) {
        model_base_environment_.reparent_to(scene_root_);
        model_base_environment_.set_scale(0.25, 0.25, 0.25);
        model_base_environment_.set_pos(-8, 42, 0);
        // camera_.look_at(model_base_environment_); // Look at the main model - will be done after all models potentially loaded
    } else {
        std::cerr << "Error: Base environment model failed to load via AssetManager." << std::endl;
        // Attempt to load panda as a fallback if environment fails
        model_base_environment_ = asset_manager_.loadModel("panda.egg");
        if (!model_base_environment_.is_empty()) {
            model_base_environment_.reparent_to(scene_root_);
            model_base_environment_.set_scale(0.005);
            model_base_environment_.set_pos(0,0,0);
            // camera_.set_pos(0, -30, 6); // General position for panda
            // camera_.look_at(model_base_environment_);
            std::cout << "Loaded panda.egg as fallback base environment." << std::endl;
        } else {
             std::cerr << "Error: Fallback panda.egg also failed to load." << std::endl;
        }
    }

    // Load additional models for DVVE demonstration
    model_complex_detail_ = asset_manager_.loadModel("panda.egg"); // Try loading panda again for this specific role
    if (!model_complex_detail_.is_empty()) {
        model_complex_detail_.reparent_to(scene_root_);
        model_complex_detail_.set_pos(model_base_environment_.get_x() + 5, model_base_environment_.get_y(), 0);
        model_complex_detail_.set_scale(0.003);
        model_complex_detail_.hide();
    } else {
        std::cout << "Warning: Could not load model_complex_detail_ (panda.egg) via AssetManager." << std::endl;
    }

    model_medium_detail_ = asset_manager_.loadModel("models/misc/rgbCube.egg"); // Standard Panda3D model
    if (model_medium_detail_.is_empty()){ // try box.egg as an alternative name for cube
         std::cout << "Warning: Could not load models/misc/rgbCube.egg. Trying box.egg..." << std::endl;
         model_medium_detail_ = asset_manager_.loadModel("box.egg");
    }
    if (!model_medium_detail_.is_empty()) {
        model_medium_detail_.reparent_to(scene_root_);
        model_medium_detail_.set_pos(model_base_environment_.get_x() - 5, model_base_environment_.get_y(), 1);
        model_medium_detail_.set_scale(1);
        model_medium_detail_.hide();
    } else {
        std::cout << "Warning: Could not load model_medium_detail_ (rgbCube.egg or box.egg) via AssetManager." << std::endl;
    }

    optional_decorative_objects_.clear(); // Clear previous if any
    for (int i = 0; i < 3; ++i) {
        NodePath deco = asset_manager_.loadModel("models/misc/sphere.egg"); // Standard Panda3D model
        if (deco.is_empty()){ // try another name
            std::cout << "Warning: Could not load models/misc/sphere.egg. Trying frowney.egg..." << std::endl;
            deco = asset_manager_.loadModel("frowney.egg"); // another simple model
        }
        if (!deco.is_empty()) {
            deco.reparent_to(scene_root_);
            deco.set_pos(model_base_environment_.get_x() + (i * 3.0 - 3.0), model_base_environment_.get_y() + 5, 1);
            deco.set_scale(0.5);
            deco.hide();
            optional_decorative_objects_.push_back(deco);
        } else {
            std::cout << "Warning: Could not load decorative sphere/frowney model " << i << " via AssetManager." << std::endl;
        }
    }

    std::cout << "Attempted to load models via AssetManager." << std::endl;
    // asset_manager_.listLoadedAssets(); // List assets again after texture loading

    // Adjust camera after loading the base model.
    if (!model_base_environment_.is_empty()) {
        camera_.set_pos(model_base_environment_.get_x(), model_base_environment_.get_y() - 30, model_base_environment_.get_z() + 6);
        camera_.look_at(model_base_environment_);
    } else { // Default camera if no base model
         camera_.set_pos(0, -40, 6);
         camera_.look_at(0,0,0);
    }

    // Initialize other game components
    physics_engine_ = PhysicsEngine();
    scene_objects_placeholders_.push_back({"object_1_from_game"});

    std::cout << "Attempting to load textures via AssetManager..." << std::endl;
    PT(Texture) panda_texture = asset_manager_.getTexture("maps/panda-rgb.png"); // Often comes with panda.egg
    if (panda_texture) {
        std::cout << "  Game: Successfully loaded 'panda-rgb.png' via AssetManager." << std::endl;
    }

    PT(Texture) env_texture = asset_manager_.getTexture("maps/envir-ground.jpg"); // Example, might not exist
    if (env_texture) {
        std::cout << "  Game: Successfully loaded 'envir-ground.jpg' via AssetManager." << std::endl;
    }

    PT(Texture) missing_texture = asset_manager_.getTexture("textures/non_existent_texture.png");
    if (!missing_texture) {
        std::cout << "  Game: Correctly reported failure to load 'non_existent_texture.png'." << std::endl;
    }

    // Example: Apply panda_texture to model_complex_detail_ (which is usually panda.egg)
    if (!model_complex_detail_.is_empty() && panda_texture) {
        model_complex_detail_.set_texture(panda_texture);
        std::cout << "  Game: Applied texture '" << panda_texture->get_name()
                  << "' to model '" << model_complex_detail_.get_name() << "'." << std::endl;
    } else {
        if (model_complex_detail_.is_empty()) {
            std::cout << "  Game: Skipped applying texture to model_complex_detail_ as model is empty." << std::endl;
        }
        if (!panda_texture) { // Check specifically for texture nullness for logging
            std::cout << "  Game: Skipped applying panda_texture as texture is null (failed to load)." << std::endl;
        }
    }

    // Example: Attempt to apply env_texture to model_base_environment_
    if (!model_base_environment_.is_empty() && env_texture) {
        model_base_environment_.set_texture(env_texture, 1); // Apply with an override, default is 0 but good to be explicit if replacing
        std::cout << "  Game: Applied texture '" << env_texture->get_name()
                  << "' to model '" << model_base_environment_.get_name() << "'." << std::endl;
    } else {
        if (model_base_environment_.is_empty()){
             std::cout << "  Game: Skipped applying env_texture to model_base_environment_ as model is empty." << std::endl;
        }
        if (!env_texture) {
             std::cout << "  Game: Skipped applying env_texture as texture is null (failed to load)." << std::endl;
        }
    }

    asset_manager_.listLoadedAssets(); // List assets again to see textures and their load status

    // Load placeholder AI avatars
    NodePath ai_avatar1 = asset_manager_.loadModel("models/misc/rgbCube.egg"); // Using a simple cube
    if (!ai_avatar1.is_empty()) {
        ai_avatar1.reparent_to(scene_root_);
        ai_avatar1.set_pos(5, 10, 0); // Initial position for ai_1
        ai_avatar1.set_scale(0.5);
        ai_avatar_nodes_["ai_1"] = ai_avatar1;
        std::cout << "Loaded AI avatar 'ai_1' (rgbCube.egg)." << std::endl;
    } else {
        std::cerr << "Warning: Failed to load model for AI avatar 'ai_1'." << std::endl;
    }
    NodePath ai_avatar2 = asset_manager_.loadModel("box.egg"); // Another simple model
     if (!ai_avatar2.is_empty()) {
        ai_avatar2.reparent_to(scene_root_);
        ai_avatar2.set_pos(-5, 10, 0); // Initial position for ai_2
        ai_avatar2.set_scale(0.7);
        ai_avatar_nodes_["ai_2"] = ai_avatar2;
        std::cout << "Loaded AI avatar 'ai_2' (box.egg)." << std::endl;
    } else {
        std::cerr << "Warning: Failed to load model for AI avatar 'ai_2'." << std::endl;
    }


    // Initial DVVE calculation
    dvve_params_.omega_v = dvve_calculator_.determineAdaptiveBoost(dvve_params_.P_core, target_p_core_for_max_quality_);
    current_Rp_ = dvve_calculator_.calculatePerformance(dvve_params_);
    std::cout << "Initial DVVE R_p: " << current_Rp_ << std::endl;
    dvve_params_.log();

    // Add the game update task to Panda3D's task manager
    AsyncTaskManager::get_global_ptr()->add(new GenericAsyncTask("GameUpdateTask", &Game::gameUpdateTask, this));
    std::cout << "Game update task added to Panda3D task manager." << std::endl;

    // Player Avatar Initialization
    player_avatar_node_ = asset_manager_.loadModel("panda"); // "panda" is a common default model
    if (player_avatar_node_.is_empty()) {
        std::cout << "Warning: Failed to load 'panda' model for avatar. Trying 'box.egg'..." << std::endl;
        player_avatar_node_ = asset_manager_.loadModel("box.egg"); // Try another simple model
    }
    if (!player_avatar_node_.is_empty()) {
        player_pos_ = LPoint3f(0, 10, 0); // Start a bit forward from camera's initial gaze if camera is at origin
        player_avatar_node_.reparent_to(scene_root_);
        player_avatar_node_.set_pos(player_pos_);
        player_avatar_node_.set_scale(0.002); // Panda model is large
        std::cout << "Player avatar loaded and placed." << std::endl;
    } else {
        std::cerr << "Error: Failed to load placeholder model for player avatar." << std::endl;
    }

    // Enable keyboard input and define quit key
    if (window_ != nullptr && window_->get_graphics_window() != nullptr) { // Ensure window is valid
        window_->get_graphics_window()->enable_keyboard(); // Recommended way to enable keyboard for the window
        std::cout << "Keyboard input enabled for game window." << std::endl;
        framework_.define_key("escape", "user_request_quit_event", &Game::userRequestQuit, this);
        std::cout << "Defined 'escape' key for user quit request." << std::endl;

        // Define 'i' key for contextual AI query
        framework_.define_key("i", "inspect_object_ai_query_event",
                              [](const Event* event, void* data) {
                                  // Could add logic here to determine which object_id based on game state
                                  static_cast<Game*>(data)->sendContextualQueryToAI("crystal_A"); // Hardcode for MVP
                              },
                              this);
        std::cout << "Defined 'i' key for contextual AI query (e.g., inspect 'crystal_A')." << std::endl;

    } else {
        std::cerr << "Error: Window not available for enabling keyboard or defining keys." << std::endl;
    }

    // Initialize HTTP client for AI communication
    http_client_ = HTTPClient::get_global_ptr();
    if (!http_client_) {
        std::cerr << "Error: Failed to get global HTTPClient instance from Panda3D." << std::endl;
    } else {
        std::cout << "HTTPClient initialized for AI communication." << std::endl;
    }

    if (enable_particle_system_) {
        if (!dvve_particle_system_.initialize()) {
            std::cerr << "Failed to initialize DVVE Particle System (SDL2)!" << std::endl;
            enable_particle_system_ = false;
        } else {
            // Setup Panda3D texture for particle system if SDL initialized
            int p_width = dvve_particle_system_.getPixelDataWidth();
            int p_height = dvve_particle_system_.getPixelDataHeight();
            if (p_width > 0 && p_height > 0) {
                panda_particle_texture_ = new Texture("particle_sdl_tex"); // Panda Texture name
                panda_particle_texture_->setup_texture(Texture::TT_2d_texture, p_width, p_height, 1,
                                                      Texture::T_unsigned_byte, Texture::F_rgba); // Use F_rgba
                panda_particle_texture_->set_ram_image_compression(Texture::CM_off);
                panda_particle_texture_->set_minfilter(SamplerState::FT_linear);
                panda_particle_texture_->set_magfilter(SamplerState::FT_linear);
                std::cout << "  Game: Initialized Panda3D texture for particle system." << std::endl;

                std::cout << "  Game: Setting up particle card for Panda3D scene..." << std::endl;
                CardMaker cm("particle_card");

                float p_tex_width = static_cast<float>(dvve_particle_system_.getPixelDataWidth());
                float p_tex_height = static_cast<float>(dvve_particle_system_.getPixelDataHeight());
                float card_aspect_ratio = (p_tex_height > 0) ? p_tex_width / p_tex_height : 1.0f;

                float desired_card_height_in_aspect2d = 0.5f;
                float desired_card_width_in_aspect2d = desired_card_height_in_aspect2d * card_aspect_ratio;

                cm.set_frame(-1.0, -1.0 + desired_card_width_in_aspect2d,  // left, right
                             -1.0, -1.0 + desired_card_height_in_aspect2d); // bottom, top

                particle_card_node_ = NodePath(cm.generate());
                if (particle_card_node_.is_empty()) {
                    std::cerr << "  Game: Failed to generate particle card!" << std::endl;
                } else {
                    particle_card_node_.reparent_to(window_->get_aspect_2d());
                    particle_card_node_.set_texture(panda_particle_texture_);
                    particle_card_node_.set_transparency(TransparencyAttrib::M_alpha);
                    std::cout << "  Game: Particle card created and textured." << std::endl;
                }
            }
        }
    }

    std::cout << "Game initialized successfully with Panda3D." << std::endl;
    return true;
}

void Game::runLoop() {
    std::cout << "Starting Panda3D main loop..." << std::endl;
    framework_.main_loop();
    std::cout << "Exited Panda3D main loop." << std::endl;
    // After framework_.main_loop() exits, we should call shutdown
    shutdown();
}

void Game::shutdown() {
    std::cout << "Game shutting down..." << std::endl;
    if (enable_particle_system_ && dvve_particle_system_.isInitialized()) {
        dvve_particle_system_.shutdown();
    }
    if (!particle_card_node_.is_empty()) { // Remove the card node
        particle_card_node_.remove_node();
        std::cout << "  Game: Particle card removed." << std::endl;
    }

    if(ai_command_server_) { // Stop the AI command server
        ai_command_server_->stop_listening();
        // The unique_ptr will handle deletion, no need to explicitly delete.
        // ai_command_server_.reset(); // Or let it go out of scope.
        std::cout << "AICommandServer stopped and cleaned up." << std::endl;
    }

    if (framework_.is_initialized()) { // Check before closing
       framework_.close_framework();
    }
    std::cout << "Game shutdown complete." << std::endl;
}


void Game::sendContextualQueryToAI(const std::string& object_id) {
    if (!http_client_) {
        std::cerr << "Game: HTTPClient not initialized. Cannot send AI query." << std::endl;
        return;
    }

    // URL should point to where AI Assistant's Flask app is listening.
    // In Docker Compose, service name "ai-assistant" on its internal port (e.g., 5000).
    // For local dev (GameEngine local, AI Assistant in Docker mapped to 3002): "http://localhost:3002/contextual_query"
    // For local dev (Both local): "http://localhost:5000/contextual_query"
    // Using "http://ai-assistant:5000/contextual_query" assuming Docker Compose inter-service communication.
    std::string url_str = "http://ai-assistant:5000/contextual_query";

    std::string player_x_str = std::to_string(player_pos_.get_x());
    std::string player_y_str = std::to_string(player_pos_.get_y());
    std::string player_z_str = std::to_string(player_pos_.get_z());

    // Manual JSON construction for MVP
    std::string json_payload =
        "{\"event_type\": \"inspect_object\", "
        "\"object_id\": \"" + object_id + "\", "
        "\"player_location\": {\"x\": " + player_x_str + ", \"y\": " + player_y_str + ", \"z\": " + player_z_str + "}}";

    std::cout << "Game: Sending contextual query to AI: " << json_payload << " to " << url_str << std::endl;

    URLSpec url(url_str);
    PT(HTTPChannel) channel = http_client_->make_channel(false); // false for non-SSL (http)

    PT(Ramfile) post_data_ramfile = new Ramfile();
    post_data_ramfile->append_data(json_payload);

    channel->set_url(url);
    channel->set_method(HTTPChannel::CM_post);
    channel->set_request_body(post_data_ramfile);
    channel->set_header("Content-Type", "application/json");

    // This is a simplified blocking attempt. Panda's HTTPClient is async.
    // For a robust solution, a task chain or callback would be better.
    channel->send_request(); // Initiate request

    // Loop to wait for header (simulate blocking) - very basic
    int attempts = 0;
    while (!channel->is_status_known() && attempts < 100) { // Timeout after ~1s if 10ms sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Non-blocking sleep
        Thread::consider_yield(); // Panda3D yield
        attempts++;
    }

    if (channel->is_status_known() && HTTPClient::is_success(channel->get_status_code())) {
        PT(Ramfile) response_body_rf = new Ramfile();
        // Loop to download body (simulate blocking)
        while (channel->download_to_ram(response_body_rf)) {
            Thread::consider_yield(); // Important for cooperative multitasking in Panda3D
        }
        // Check if download_to_ram completed fully or if there was an issue
        if (channel->is_download_complete()){
            std::string response_str = response_body_rf->get_data();
            std::cout << "Game: AI Contextual Response (Raw): " << response_str << std::endl;

            // Basic JSON parsing for "ai_generated_message" (MVP: string search)
            const std::string key_to_find = "\"ai_generated_message\": \"";
            size_t msg_start = response_str.find(key_to_find);
            if (msg_start != std::string::npos) {
                msg_start += key_to_find.length();
                size_t msg_end = response_str.find("\"", msg_start);
                if (msg_end != std::string::npos) {
                    std::string ai_msg = response_str.substr(msg_start, msg_end - msg_start);
                    std::cout << "Game (AI Says - Contextual): " << ai_msg << std::endl;
                    // Here you would typically display this message in an in-game UI element.
                } else {
                    std::cerr << "Game: AI Contextual Response - Could not parse message end from: " << response_str << std::endl;
                }
            } else {
                std::cerr << "Game: AI Contextual Response - No 'ai_generated_message' field found in: " << response_str << std::endl;
            }
        } else {
             std::cerr << "Game: AI Contextual Query - Download did not complete. Channel valid: " << channel->is_valid() << std::endl;
        }
    } else if (channel->is_status_known()) {
        std::cerr << "Game: AI Contextual Query - HTTP Error Status: " << channel->get_status_code() << std::endl;
        PT(Ramfile) error_body_rf = new Ramfile();
        channel->download_to_ram(error_body_rf); // Try to get error body
        std::cerr << "Game: AI Contextual Query - Error Body: " << error_body_rf->get_data() << std::endl;
    } else {
        std::cerr << "Game: AI Contextual Query - Failed to send request or get status header after " << attempts << " attempts." << std::endl;
    }
    // http_client_->release_channel(channel); // Not necessary for PT(HTTPChannel)
}


void Game::processAICommands(double dt) {
    AICommand command;
    while (ai_command_queue_.try_pop(command)) {
        std::cout << "Game: Processing AI Command for avatar '" << command.avatar_id
                  << "' action '" << command.action << "'." << std::endl;
        auto it = ai_avatar_nodes_.find(command.avatar_id);
        if (it != ai_avatar_nodes_.end()) {
            NodePath& avatarNode = it->second;
            if (!avatarNode.is_empty()) {
                if (command.action == "move_to") {
                    // For MVP, directly set position.
                    // Later, this could set a target, and movement handled over time.
                    avatarNode.set_pos(command.x, command.y, command.z);
                    std::cout << "  Game: Moved AI avatar '" << command.avatar_id << "' to ("
                              << command.x << "," << command.y << "," << command.z << ")." << std::endl;
                } else if (command.action == "play_animation") {
                    // Placeholder for animation logic
                    std::cout << "  Game: AI avatar '" << command.avatar_id << "' would play animation: "
                              << " (not implemented)" << std::endl;
                } else {
                    std::cerr << "  Game: Unknown action '" << command.action << "' for AI avatar '"
                              << command.avatar_id << "'." << std::endl;
                }
            } else {
                std::cerr << "  Game: AI avatar NodePath for '" << command.avatar_id << "' is empty. Cannot process command." << std::endl;
            }
        } else {
            std::cerr << "  Game: AI avatar '" << command.avatar_id << "' not found. Cannot process command." << std::endl;
        }
    }
}

// Static callback for Panda3D task
AsyncTask::DoneStatus Game::gameUpdateTask(GenericAsyncTask* task, void* data) {
    Game* game_ptr = static_cast<Game*>(data);
    if (!game_ptr || game_ptr->quit_panda_loop_) {
        if (game_ptr && game_ptr->framework_.is_playing() && game_ptr->quit_panda_loop_) {
           std::cout << "SDL requested quit or game_ptr->quit_panda_loop_ is true, stopping Panda framework." << std::endl;
           game_ptr->framework_.make_exit_request();
        }
        return AsyncTask::DS_done;
    }
    // Panda Window check - this was the original check
    if (!game_ptr->window_ || !game_ptr->window_->get_graphics_window() || !game_ptr->window_->get_graphics_window()->is_active()) {
        std::cout << "Panda3D window is inactive, stopping gameUpdateTask." << std::endl;
        // game_ptr->framework_.make_exit_request(); // Request exit if window is lost
        return AsyncTask::DS_done;
    }

    if (game_ptr->user_quit_requested_) {
        game_ptr->framework_.make_exit_request();
        return AsyncTask::DS_done; // Stop this task
    }


    double dt = ClockObject::get_global_clock()->get_dt();

    game_ptr->processInput(dt); // Pass dt
    game_ptr->processAICommands(dt); // Process AI commands each frame
    game_ptr->updateInternal(dt);
    game_ptr->renderSceneLogic();

    if (game_ptr->enable_particle_system_ && game_ptr->dvve_particle_system_.isInitialized()) {
        auto p_ev_start = std::chrono::steady_clock::now();
        game_ptr->dvve_particle_system_.handleEvents();
        auto p_ev_end = std::chrono::steady_clock::now();
        game_ptr->particle_event_time_ms_ += std::chrono::duration<double, std::milli>(p_ev_end - p_ev_start).count();

        if (game_ptr->dvve_particle_system_.shouldQuit()) {
            game_ptr->quit_panda_loop_ = true;
        }

        if (!game_ptr->quit_panda_loop_) {
            auto p_up_start = std::chrono::steady_clock::now();
            game_ptr->dvve_particle_system_.update(dt);
            auto p_up_end = std::chrono::steady_clock::now();
            game_ptr->particle_update_time_ms_ += std::chrono::duration<double, std::milli>(p_up_end - p_up_start).count();

            auto p_ren_start = std::chrono::steady_clock::now();
            game_ptr->dvve_particle_system_.render();
            auto p_ren_end = std::chrono::steady_clock::now();
            game_ptr->particle_render_time_ms_ += std::chrono::duration<double, std::milli>(p_ren_end - p_ren_start).count();
        }
        game_ptr->particle_total_frames_processed_++;

        if (game_ptr->loop_iteration_count_ > 0 && (game_ptr->loop_iteration_count_ % game_ptr->particle_profile_log_interval_ == 0)) {
            if (game_ptr->particle_total_frames_processed_ > 0) {
                double avg_event = game_ptr->particle_event_time_ms_ / game_ptr->particle_total_frames_processed_;
                double avg_update = game_ptr->particle_update_time_ms_ / game_ptr->particle_total_frames_processed_;
                double avg_render = game_ptr->particle_render_time_ms_ / game_ptr->particle_total_frames_processed_;
                std::cout << "[Particle Profiler] Avg over last " << game_ptr->particle_total_frames_processed_ << " frames (Game frame: " << game_ptr->loop_iteration_count_ << "): "
                          << "Event: " << std::fixed << std::setprecision(3) << avg_event << "ms, "
                          << "Update: " << avg_update << "ms, "
                          << "Render: " << avg_render << "ms " // This render time is for SDL window preview + pixel read
                          << "(Particles: " << game_ptr->dvve_particle_system_.getActiveParticleCount() << ")"
                          << std::endl;

                game_ptr->particle_event_time_ms_ = 0.0;
                game_ptr->particle_update_time_ms_ = 0.0;
                game_ptr->particle_render_time_ms_ = 0.0;
                game_ptr->particle_total_frames_processed_ = 0;
            }
        }

        // Copy SDL particle texture data to Panda3D texture
        if (!game_ptr->quit_panda_loop_ && game_ptr->panda_particle_texture_ && game_ptr->dvve_particle_system_.isInitialized()) {
            const auto& sdl_pixels = game_ptr->dvve_particle_system_.getPixelDataBuffer();
            int sdl_w = game_ptr->dvve_particle_system_.getPixelDataWidth();
            int sdl_h = game_ptr->dvve_particle_system_.getPixelDataHeight();

            if (!sdl_pixels.empty() &&
                game_ptr->panda_particle_texture_->get_x_size() == sdl_w &&
                game_ptr->panda_particle_texture_->get_y_size() == sdl_h) {

                std::vector<unsigned char> flipped_pixels(sdl_pixels.size());
                int pitch = sdl_w * 4;
                for(int y = 0; y < sdl_h; ++y) {
                    memcpy(flipped_pixels.data() + (sdl_h - 1 - y) * pitch,
                           sdl_pixels.data() + y * pitch,
                           pitch);
                }

                CPTA_uchar cpta_pixel_data = CPTA_uchar::make_const(flipped_pixels.data(), flipped_pixels.size());
                game_ptr->panda_particle_texture_->set_ram_image(cpta_pixel_data, Texture::CM_off);

            } else if (!sdl_pixels.empty()) {
                // Log mismatch periodically if it occurs
                 if (game_ptr->loop_iteration_count_ > 0 && (game_ptr->loop_iteration_count_ % game_ptr->particle_profile_log_interval_ == 0)) {
                    std::cerr << "  Game: Mismatch Panda texture and SDL pixel buffer sizes. Panda: "
                              << game_ptr->panda_particle_texture_->get_x_size() << "x" << game_ptr->panda_particle_texture_->get_y_size()
                              << ", SDL: " << sdl_w << "x" << sdl_h
                              << ". Skipping copy." << std::endl;
                 }
            }
        }
    }
    return AsyncTask::DS_cont;
}

void Game::setMockPCore(double p_core_val) {
    // std::cout << "  Setting Mock P_core to " << p_core_val << std::endl;
    dvve_params_.P_core = p_core_val;
}

void Game::userRequestQuit(const Event* event, void* data) {
    Game* self = static_cast<Game*>(data);
    if (self) {
        std::cout << "User requested quit via Escape key." << std::endl;
        self->user_quit_requested_ = true;
    }
}

void Game::setVisualComplexity(double f_v_val) {
    // std::cout << "  Setting Visual Complexity (F_v) to " << f_v_val << std::endl;
    dvve_params_.F_v = f_v_val;
}

void Game::processInput(double dt) { // dt is now passed
    // DVVE parameter cycling (existing logic)
    int P_core_change_interval = 40;
    int F_v_change_interval = 20; // Adjusted to be different from P_core for variety

    if (loop_iteration_count_ > 0) { // Avoid changing on the very first frame if desired
        if (loop_iteration_count_ % P_core_change_interval == 0) {
            int p_core_state = (loop_iteration_count_ / P_core_change_interval) % 3;
            double new_P_core = dvve_params_.P_core;
            if (p_core_state == 0) new_P_core = 2.0e9; // High
            else if (p_core_state == 1) new_P_core = 1.0e9; // Medium
            else new_P_core = 0.5e9; // Low
            if (new_P_core != dvve_params_.P_core) setMockPCore(new_P_core);
        }
        if (loop_iteration_count_ % F_v_change_interval == 0) {
            int f_v_state = (loop_iteration_count_ / F_v_change_interval) % 2;
            double new_F_v = dvve_params_.F_v;
            if (f_v_state == 0) new_F_v = 0.8; // Low F_v
            else new_F_v = 1.5; // High F_v
            if (new_F_v != dvve_params_.F_v) setVisualComplexity(new_F_v);
        }
    }

    // Avatar movement
    const float MOVEMENT_SPEED = 5.0f; // Units per second
    if (window_ && window_->get_graphics_window() && window_->get_graphics_window()->get_keyboard_map()->is_button_down(KeyboardButton::ascii_key('w'))) {
        player_pos_.set_y(player_pos_.get_y() + MOVEMENT_SPEED * dt);
        // To make it more visible if avatar is small or far:
        // std::cout << "W pressed, new player_pos_ Y: " << player_pos_.get_y() << std::endl;
    }
    // Add S, A, D similarly if needed
    // if (window_ && window_->get_graphics_window() && window_->get_graphics_window()->get_keyboard_map()->is_button_down(KeyboardButton::ascii_key('s'))) {
    //     player_pos_.set_y(player_pos_.get_y() - MOVEMENT_SPEED * dt);
    // }
    // if (window_ && window_->get_graphics_window() && window_->get_graphics_window()->get_keyboard_map()->is_button_down(KeyboardButton::ascii_key('a'))) {
    //     player_pos_.set_x(player_pos_.get_x() - MOVEMENT_SPEED * dt);
    // }
    // if (window_ && window_->get_graphics_window() && window_->get_graphics_window()->get_keyboard_map()->is_button_down(KeyboardButton::ascii_key('d'))) {
    //     player_pos_.set_x(player_pos_.get_x() + MOVEMENT_SPEED * dt);
    // }
}

void Game::updateInternal(double deltaTime) {
    // Update DVVE parameters and performance calculation (existing logic)
    dvve_params_.omega_v = dvve_calculator_.determineAdaptiveBoost(dvve_params_.P_core, target_p_core_for_max_quality_);
    current_Rp_ = dvve_calculator_.calculatePerformance(dvve_params_);

    if (current_Rp_ > 1.8e9) {
        detail_level_ = "Very High"; num_objects_to_render_ = 100;
    } else if (current_Rp_ > 1.2e9) {
        detail_level_ = "High"; num_objects_to_render_ = 70;
    } else if (current_Rp_ > 0.8e9) {
        detail_level_ = "Medium"; num_objects_to_render_ = 50;
    } else {
        detail_level_ = "Low"; num_objects_to_render_ = 30;
    }

    std::string suffix = "";
    if (dvve_params_.omega_v > 0.15) suffix = " (Simplified due to low P_core)";
    else if (dvve_params_.omega_v > 0.05) suffix = " (Slightly Simplified)";
    detail_level_ += suffix;

    // --- Temporary override for particle count benchmarking ---
    bool benchmark_particle_counts = true; // Set to true to enable this benchmark
    if (benchmark_particle_counts && enable_particle_system_ && dvve_particle_system_.isInitialized()) {
        int benchmark_interval = 300; // Change count every ~5 seconds if 60fps (300 frames)
        int current_benchmark_stage = (loop_iteration_count_ / benchmark_interval) % 4; // Cycle through 4 stages
        int benchmark_count = 1000;
        switch (current_benchmark_stage) {
            case 0: benchmark_count = 1000; break;
            case 1: benchmark_count = 10000; break;
            case 2: benchmark_count = 25000; break;
            case 3: benchmark_count = 50000; break;
        }
        // Only set if it's the start of a new benchmark stage to avoid constant calls
        if (loop_iteration_count_ % benchmark_interval == 0) {
           std::cout << "== BENCHMARKING PARTICLE COUNT: " << benchmark_count << " ==" << std::endl;
           dvve_particle_system_.setTargetParticleCount(benchmark_count);
        }
    } else if (enable_particle_system_ && dvve_particle_system_.isInitialized()) {
        // Original DVVE-driven particle count logic (from sub-step 3.3)
        int target_particles = 0;
        if (num_objects_to_render_ >= 100) target_particles = 20000;
        else if (num_objects_to_render_ >= 70) target_particles = 10000;
        else if (num_objects_to_render_ >= 50) target_particles = 5000;
        else target_particles = 1000;

        if (loop_iteration_count_ % 15 == 0) {
            dvve_particle_system_.setTargetParticleCount(target_particles);
        }
    }
    // --- End of particle count logic ---

    physics_engine_.update(deltaTime, scene_objects_placeholders_);
    if (loop_iteration_count_ % 60 == 0) { // Keep this less frequent
        if (enable_particle_system_ && dvve_particle_system_.isInitialized()) {
            // Pass a const reference to the actual particles
            physics_engine_.performAdvancedCollisions(dvve_particle_system_.getParticles());
        } else {
            // Call with an empty vector or a simulated count if particle system is off
            // For simplicity, let's only call if particle system is active.
            // Or, could use the old simulated_particle_count_for_physics_ if system is off:
            // physics_engine_.performAdvancedCollisions(simulated_particle_count_for_physics_); // This would need an overload
            std::cout << "  GameUpdate: Advanced collisions skipped as particle system is not active." << std::endl;
        }
    }

    // Logging for the current frame's state
    if (loop_iteration_count_ % 30 == 0) {
        std::cout << "Frame " << loop_iteration_count_
                  << " (dt: " << std::fixed << std::setprecision(4) << deltaTime << "s)"
                  << " PlayerPos: (" << std::fixed << std::setprecision(1) << player_pos_.get_x() << "," << player_pos_.get_y() << "," << player_pos_.get_z() << ")" // Log player position
                  << " DVVE Inputs: [P_core:" << std::fixed << std::setprecision(1) << dvve_params_.P_core
                  << ", F_v:" << std::fixed << std::setprecision(1) << dvve_params_.F_v
                  << ", omega_v:" << std::fixed << std::setprecision(2) << dvve_params_.omega_v
                  << ", nu_v:" << std::fixed << std::setprecision(2) << dvve_params_.nu_v << "]"
                  << " => R_p: " << std::fixed << std::setprecision(2) << current_Rp_
                  << " | Panda3D Sim: " << num_objects_to_render_
                  << " objs at '" << detail_level_ << "' detail."
                  << " | SDL Particles: " << dvve_particle_system_.getActiveParticleCount() // Add this
                  << std::endl;
        if (loop_iteration_count_ % 120 == 0 && loop_iteration_count_ > 0) asset_manager_.listLoadedAssets();
    }
    loop_iteration_count_++;
}

void Game::renderSceneLogic() {
    // Default state: base environment is always visible if loaded
    if (!model_base_environment_.is_empty()) {
        model_base_environment_.show();
    }

    // Manage detail models: only one of these should be visible or none
    if (!model_complex_detail_.is_empty()) model_complex_detail_.hide();
    if (!model_medium_detail_.is_empty()) model_medium_detail_.hide();

    std::string msg_prefix = "DVVE Scene: ";

    if (detail_level_.find("Very High") != std::string::npos) {
        if (!model_complex_detail_.is_empty()) {
            model_complex_detail_.show();
            // msg_prefix += "Complex ON. "; // Logging done below
        }
    } else if (detail_level_.find("High") != std::string::npos) { // "High" or "High (Slightly Simplified)"
        if (!model_medium_detail_.is_empty()) {
            model_medium_detail_.show();
            // msg_prefix += "MediumDetail ON. ";
        }
    } else {
        // msg_prefix += "BaseOnly (No extra detail model). ";
    }

    // Manage optional decorative objects based on num_objects_to_render_
    int deco_to_show = 0;
    if (num_objects_to_render_ >= 100) deco_to_show = 3;       // Very High
    else if (num_objects_to_render_ >= 70) deco_to_show = 2;  // High
    else if (num_objects_to_render_ >= 50) deco_to_show = 1;  // Medium

    for (size_t i = 0; i < optional_decorative_objects_.size(); ++i) {
        if (i < deco_to_show) {
            if (!optional_decorative_objects_[i].is_empty()) optional_decorative_objects_[i].show();
        } else {
            if (!optional_decorative_objects_[i].is_empty()) optional_decorative_objects_[i].hide();
        }
    }

    // Update player avatar model position
    if (!player_avatar_node_.is_empty()) {
        player_avatar_node_.set_pos(player_pos_);
    }


    // Log the scene state periodically
    if (loop_iteration_count_ % 30 == 0) {
        if (!model_complex_detail_.is_empty() && model_complex_detail_.is_hidden()) msg_prefix += "Complex OFF. ";
        else if (!model_complex_detail_.is_empty()) msg_prefix += "Complex ON. ";

        if (!model_medium_detail_.is_empty() && model_medium_detail_.is_hidden()) msg_prefix += "MedDetail OFF. ";
        else if (!model_medium_detail_.is_empty()) msg_prefix += "MedDetail ON. ";

        msg_prefix += std::to_string(deco_to_show) + "/" + std::to_string(optional_decorative_objects_.size()) + " deco items. ";
        std::cout << msg_prefix << std::endl;
    }
}
