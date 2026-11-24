cmake_minimum_required(VERSION 3.15)
project(ConceptualDAW VERSION 1.0 LANGUAGES CXX)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # Optional: disable GNU extensions if pure C++20 is desired

# Removed global: include_directories(include)

# Explicitly list source files
set(SOURCES
    src/AudioEffect.cpp
    src/AudioTrack.cpp
    src/DelayEffect.cpp
    src/EditSession.cpp
    src/MidiTrack.cpp
    src/ReverbEffect.cpp
    src/Transport.cpp
    src/main.cpp
)

# Add the executable target
add_executable(ConceptualDAW_App ${SOURCES})
target_include_directories(ConceptualDAW_App PRIVATE include)

# Optional: Add compiler warnings (good practice)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(ConceptualDAW_App PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        # -Werror # Uncomment to treat warnings as errors
    )
elseif(MSVC)
    target_compile_options(ConceptualDAW_App PRIVATE
        /W4
        # /WX # Uncomment to treat warnings as errors
    )
endif()

# Print a message upon configuration
message(STATUS "Configuring ConceptualDAW project...")
message(STATUS "Sources: ${SOURCES}")
message(STATUS "Target: ConceptualDAW_App")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# Native signal-path verification (no external dependencies).
enable_testing()
set(LIB_SOURCES
    src/AudioEffect.cpp
    src/AudioTrack.cpp
    src/DelayEffect.cpp
    src/EditSession.cpp
    src/MidiTrack.cpp
    src/ReverbEffect.cpp
    src/Transport.cpp
)
add_executable(ConceptualDAW_NativeVerify tests/native_verify.cpp ${LIB_SOURCES})
target_include_directories(ConceptualDAW_NativeVerify PRIVATE include)
add_test(NAME ConceptualDAW_NativeVerify COMMAND ConceptualDAW_NativeVerify)
