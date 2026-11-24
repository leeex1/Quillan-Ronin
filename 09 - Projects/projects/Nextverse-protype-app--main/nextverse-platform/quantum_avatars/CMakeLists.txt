# CMakeLists.txt for the QuantumAvatars Module C++ Components
cmake_minimum_required(VERSION 3.10)
project(QuantumAvatarsModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ensure compiler can find headers in the 'include' directory for this module
# This is useful if other targets within this CMakeLists.txt (e.g., an executable)
# need to include AvatarAdapter.h using #include "AvatarAdapter.h"
# For add_library, target_include_directories is more precise for exporting includes.
# Removed: # include_directories(include) # Less modern, but simple for self-contained module

add_library(QuantumAvatarsModule STATIC
    src/AvatarAdapter.cpp
    # Add other C++ source files for Quantum Avatars here
    # e.g., src/QuantumAvatar.cpp if it becomes a .cpp file
)

if(MSVC)
    target_compile_options(QuantumAvatarsModule PRIVATE /W4)
else()
    target_compile_options(QuantumAvatarsModule PRIVATE -Wall -Wextra -Wpedantic)
endif()

# This correctly specifies that QuantumAvatarsModule itself uses its 'include' directory,
# and more importantly, any target linking against QuantumAvatarsModule will
# automatically get this include directory.
target_include_directories(QuantumAvatarsModule PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # For build tree
    $<INSTALL_INTERFACE:include>                           # For install tree (if installing)
)

# Link AIAssistantCpp module
target_link_libraries(QuantumAvatarsModule PRIVATE AIAssistantCpp)

# Example of linking other libraries if needed:
# target_link_libraries(QuantumAvatarsModule PRIVATE SomeOtherLib::SomeOtherTarget)

message(STATUS "QuantumAvatarsModule configured.")
