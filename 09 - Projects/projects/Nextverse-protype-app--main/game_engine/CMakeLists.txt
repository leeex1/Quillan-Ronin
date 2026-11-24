cmake_minimum_required(VERSION 3.15)
project(NextVerseGameEngine LANGUAGES CXX)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
# Heavyweight SDKs are optional: configure must succeed on hosts without them.
find_package(Panda3D QUIET)
find_package(SDL2 QUIET)
find_package(GTest QUIET)

# Manual SDK roots are supported for hosts without CMake config packages.
# Pass -DPanda3D_ROOT_DIR=<sdk> -DSDL2_ROOT_DIR=<sdk> at configure time.
if(NOT Panda3D_FOUND AND Panda3D_ROOT_DIR)
    set(Panda3D_INCLUDE_DIRS "${Panda3D_ROOT_DIR}/include")
    set(Panda3D_LIBRARIES
        "${Panda3D_ROOT_DIR}/lib/libp3framework.lib"
        "${Panda3D_ROOT_DIR}/lib/libpanda.lib"
        "${Panda3D_ROOT_DIR}/lib/libpandaexpress.lib"
        "${Panda3D_ROOT_DIR}/lib/libp3dtool.lib"
        "${Panda3D_ROOT_DIR}/lib/libp3dtoolconfig.lib"
        "${Panda3D_ROOT_DIR}/lib/libp3direct.lib"
    )
    set(Panda3D_FOUND TRUE)
    message(STATUS "Using manual Panda3D SDK root: ${Panda3D_ROOT_DIR}")
endif()
if(NOT SDL2_FOUND AND SDL2_ROOT_DIR)
    set(SDL2_INCLUDE_DIRS "${SDL2_ROOT_DIR}/include")
    if(NOT TARGET SDL2::SDL2)
        add_library(SDL2::SDL2 UNKNOWN IMPORTED)
        set_target_properties(SDL2::SDL2 PROPERTIES
            IMPORTED_LOCATION "${SDL2_ROOT_DIR}/lib/x64/SDL2.lib")
    endif()
    set(SDL2_FOUND TRUE)
    message(STATUS "Using manual SDL2 SDK root: ${SDL2_ROOT_DIR}")
endif()

set(PORTABLE_SOURCES
    src/DVVE.cpp
    src/PhysicsEngine.cpp
    src/AICommandServer.cpp
)
set(FULL_SOURCES
    ${PORTABLE_SOURCES}
    src/Game.cpp
    src/AssetManager.cpp
    src/DVVEParticleSystem.cpp
)

if(Panda3D_FOUND AND SDL2_FOUND)
    add_library(GameEngineLib STATIC ${FULL_SOURCES})
    target_include_directories(GameEngineLib PUBLIC include)
    target_include_directories(GameEngineLib SYSTEM PRIVATE
        ${Panda3D_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS})
    target_link_libraries(GameEngineLib PUBLIC
        ${Panda3D_LIBRARIES} SDL2::SDL2 Threads::Threads)

    add_executable(NextVerseGameEngineApp src/main.cpp)
    target_link_libraries(NextVerseGameEngineApp PRIVATE GameEngineLib)
    set(ENGINE_TARGET GameEngineLib)
else()
    message(WARNING
        "Panda3D and/or SDL2 not found: configuring the portable engine subset "
        "(PhysicsEngine, DVVE, AICommandServer) only. The full app is skipped.")
    add_library(GameEnginePortable STATIC ${PORTABLE_SOURCES})
    target_include_directories(GameEnginePortable PUBLIC include)
    target_link_libraries(GameEnginePortable PUBLIC Threads::Threads)
    if(WIN32)
        target_link_libraries(GameEnginePortable PUBLIC ws2_32)
    endif()
    set(ENGINE_TARGET GameEnginePortable)
endif()

enable_testing()

# Compiler-only verification that always builds, independent of external SDKs.
add_executable(GameEngineNativeVerify tests/native_verify.cpp)
target_link_libraries(GameEngineNativeVerify PRIVATE ${ENGINE_TARGET})
add_test(NAME GameEngineNativeVerify COMMAND GameEngineNativeVerify)

if(GTest_FOUND AND TARGET GameEngineLib)
    add_executable(GameEngineTests
        tests/TestGameDVVE.cpp
        tests/TestPhysicsEngine.cpp
        tests/TestAssetManager.cpp
    )
    target_include_directories(GameEngineTests SYSTEM PRIVATE
        ${Panda3D_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS})
    target_link_libraries(GameEngineTests PRIVATE
        GameEngineLib GTest::gtest GTest::gtest_main Threads::Threads)
    include(GoogleTest)
    gtest_discover_tests(GameEngineTests)
endif()

message(STATUS "Game Engine configured. Full SDK build: ${Panda3D_FOUND}/${SDL2_FOUND}; GTest: ${GTest_FOUND}")
