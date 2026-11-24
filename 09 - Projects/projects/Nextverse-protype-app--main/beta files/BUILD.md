# Nextverse Build System Documentation

## Overview

This document explains the build configuration for the Nextverse platform, which uses CMake to manage dependencies and compile the codebase.

## Prerequisites

- Visual Studio 2022 with C++ development tools
- CMake version 3.10 or higher
- Panda3D 1.10.15 SDK (installed at `C:/development platform/Nextverse/Panda3D-1.10.15-x64`)
- PortAudio (optional) for audio functionality

## Build Process

1. **Configure the project**:
   ```bash
   cmake -S . -B build
   ```

2. **Build the project**:
   ```bash
   cmake --build build --config Debug
   ```

3. **Run the executable**:
   ```bash
   ./build/bin/Debug/NextVerse.exe
   ```

## CMake Configuration Explanation

The `CMakeLists.txt` file handles:

1. **C++ Standard**: Using C++14 for compatibility with Eigen library included in Panda3D
   ```cmake
   set(CMAKE_CXX_STANDARD 14)
   set(CMAKE_CXX_STANDARD_REQUIRED True)
   ```

2. **Panda3D Integration**: Manually finding Panda3D libraries and headers
   ```cmake
   set(PANDA3D_DIR "C:/development platform/Nextverse/Panda3D-1.10.15-x64")
   find_path(PANDA3D_INCLUDE pandaFramework.h PATHS ${PANDA3D_INCLUDE_DIR})
   find_library(PANDA3D_FRAMEWORK_LIBRARY NAMES p3framework libp3framework)
   ```

3. **PortAudio (Optional)**: Conditionally links PortAudio if found
   ```cmake
   find_package(portaudio CONFIG QUIET)
   if(portaudio_FOUND)
       set(PORTAUDIO_LIBRARIES portaudio_static)
   else()
       set(PORTAUDIO_LIBRARIES "")
   endif()
   ```

4. **Dependency Linking**: Consolidated in one target_link_libraries command
   ```cmake
   target_link_libraries(NextVerse PRIVATE
       ${PANDA3D_LIBRARIES}
       $<$<BOOL:${portaudio_FOUND}>:${PORTAUDIO_LIBRARIES}>
       opengl32
   )
   ```

## Troubleshooting

- **Header File Issues**: If Panda3D headers aren't found, verify the installation path and that header files exist in the include directory.
- **Linker Errors**: Make sure all required libraries are available at the specified paths.
- **Eigen Issues**: The C++14 standard is used to avoid compatibility issues with the Eigen library included with Panda3D.

## Testing the Build

After building, run the executable to verify that:
1. The application launches without errors
2. The Panda3D rendering window appears
3. The basic 3D scene can be visualized
