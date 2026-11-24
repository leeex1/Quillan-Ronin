cmake_minimum_required(VERSION 3.10)
project(NextVerseGameEngineSDLAlpha LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# --- Dependencies ---
find_package(SDL2 REQUIRED)
if(NOT SDL2_FOUND)
    message(FATAL_ERROR "SDL2 not found. Please ensure SDL2 is installed and discoverable by CMake.")
endif()

# Assuming QCC_Alpha and QSVM_Alpha headers are in a known relative path for Alpha.
# In a larger project, these would be proper targets found via find_package or add_subdirectory.
set(NEXTVERSE_CORE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nextverse_alpha_core_design) # Adjust if structure differs
include_directories(SYSTEM ${NEXTVERSE_CORE_INCLUDE_DIR}) # Use SYSTEM for external-like headers


# --- Common Game Engine Headers (Interface Library) ---
add_library(GameEngineCommonAlpha INTERFACE)
target_include_directories(GameEngineCommonAlpha INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR} # For game_engine_common_alpha.hpp
    ${NEXTVERSE_CORE_INCLUDE_DIR} # For quantum_common_alpha.hpp if it's used by game_engine_common_alpha.hpp
)

# --- ParticleSystem_Alpha Library ---
add_library(ParticleSystemAlpha particle_system_alpha.cpp)
target_include_directories(ParticleSystemAlpha PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) # For particle_system_alpha.hpp
target_link_libraries(ParticleSystemAlpha PUBLIC GameEngineCommonAlpha)
# If ParticleSystem uses QCC/QSVM types directly from their headers (not just passed as double):
target_include_directories(ParticleSystemAlpha INTERFACE ${NEXTVERSE_CORE_INCLUDE_DIR})


# --- GameEngineSDL_Alpha Library ---
set(GAME_ENGINE_SDL_SOURCES
    game_engine_sdl_alpha.cpp
    # particle_system_alpha.cpp is now part of ParticleSystemAlpha library
)

add_library(GameEngineSDLAlpha ${GAME_ENGINE_SDL_SOURCES})
target_include_directories(GameEngineSDLAlpha PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}      # For game_engine_sdl_alpha.hpp
    ${NEXTVERSE_CORE_INCLUDE_DIR}    # For qcc_alpha.hpp, qsvm_alpha.hpp
)
target_link_libraries(GameEngineSDLAlpha PUBLIC
    GameEngineCommonAlpha
    ParticleSystemAlpha
    SDL2::SDL2 # Imported target from find_package(SDL2)
    # Link against QCC_Alpha and QSVM_Alpha if GameEngineSDL_Alpha uses their compiled code
    # For Alpha, if QCC/QSVM are header-only or their .cpp files are added directly to an executable,
    # this might not be needed. But if they are separate libs, they must be linked.
    # Assuming QCC_Alpha and QSVM_Alpha are libraries defined elsewhere (e.g., in a parent CMakeLists or found via find_package)
    # For this standalone example, if QCC/QSVM .cpp files were here, they'd be part of a lib or this target.
    # If QCC/QSVM are built as libraries QCC_Alpha_Lib and QSVM_Alpha_Lib:
    # QCC_Alpha_Lib
    # QSVM_Alpha_Lib
)

# --- Example Executable for the Game Engine (CrashOverrideX style demo) ---
# This executable would need a main_game_engine_demo.cpp file.
# It needs to link GameEngineSDLAlpha, and also QCC_Alpha and QSVM_Alpha if they are compiled libraries.
# add_executable(DVVE_Sonic_Armageddon_Demo main_game_engine_demo.cpp)
# target_link_libraries(DVVE_Sonic_Armageddon_Demo PRIVATE
#     GameEngineSDLAlpha
    # QCC_Alpha_Lib  # Example if QCC/QSVM are actual compiled libraries
    # QSVM_Alpha_Lib # Example
# )


message(STATUS "NextVerse SDL Game Engine Alpha configured.")
message(STATUS "SDL2 found: ${SDL2_LIBRARIES}")
message(STATUS "Ensure QCC_Alpha and QSVM_Alpha headers are accessible (path: ${NEXTVERSE_CORE_INCLUDE_DIR}).")
message(STATUS "If QCC/QSVM are compiled libraries, ensure they are linked correctly if used by an executable target.")

# Note for building this module:
# 1. This CMakeLists.txt assumes it's part of a larger structure where QCC/QSVM are handled.
#    If building standalone, you'd need to add source files for QCC/QSVM or link precompiled libs.
# 2. A main_game_engine_demo.cpp would be required to create an executable.
# 3. SDL2 development libraries must be installed on the system.
#    - Windows: Set SDL2_DIR or use vcpkg.
#    - Linux: sudo apt-get install libsdl2-dev.
#    - macOS: brew install sdl2.
# 4. The include path for QCC/QSVM headers (NEXTVERSE_CORE_INCLUDE_DIR) needs to be correct relative
#    to this CMakeLists.txt file's location.
#    Example: If this 'nextverse_alpha_game_engine_design' directory is parallel to 'nextverse_alpha_core_design',
#    then NEXTVERSE_CORE_INCLUDE_DIR should be something like "${CMAKE_CURRENT_SOURCE_DIR}/../nextverse_alpha_core_design".
#    Using SYSTEM include_directories for these helps treat them like external library headers.
