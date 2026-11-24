# Main CMakeLists.txt for the NextVerse Platform
cmake_minimum_required(VERSION 3.10) # Or a suitable minimum version
project(NextVersePlatform LANGUAGES CXX)

# Option for enabling/disabling Quantum Offload features
option(WITH_QUANTUM_OFFLOAD "Enable Quantum Offload specific code paths" OFF)

if(WITH_QUANTUM_OFFLOAD)
    add_compile_definitions(WITH_QUANTUM_OFFLOAD) # Pass definition to C++ code
    message(STATUS "Quantum Offload: ENABLED")
else()
    message(STATUS "Quantum Offload: DISABLED")
endif()

# Add subdirectories for different modules within the platform
add_subdirectory(network)
add_subdirectory(ai-assistant/cpp)
add_subdirectory(quantum_avatars)
# Add other modules like daw, mmorpg as they are integrated
# add_subdirectory(daw/conceptual_logic) # Example, adjust path as needed
# add_subdirectory(mmorpg) # Example, adjust path as needed
