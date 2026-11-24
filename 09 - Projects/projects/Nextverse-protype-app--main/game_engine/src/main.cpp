#include "Game.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "Starting Game Engine (Panda3D Integration)..." << std::endl;

    Game game_engine_instance;

    if (game_engine_instance.initialize(argc, argv)) { // Pass argc, argv
        game_engine_instance.runLoop(); // Calls framework_.main_loop()
    } else {
        std::cerr << "Failed to initialize game engine with Panda3D!" << std::endl;
        return 1;
    }

    // shutdown() is called by Game::runLoop() when Panda's main_loop exits.
    // If Game::gameUpdateTask could signal an exit condition back to Game::runLoop
    // other than window close, then Game::runLoop might exit and shutdown would be called here.
    // For now, Panda's main_loop exiting (e.g., by window close) means Game::runLoop will call Game::shutdown().

    std::cout << "Game Engine finished." << std::endl;
    return 0;
}
