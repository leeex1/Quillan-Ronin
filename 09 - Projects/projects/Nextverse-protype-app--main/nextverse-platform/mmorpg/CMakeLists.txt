cmake_minimum_required(VERSION 3.10)
project(MMORPG_Server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find PkgConfig for finding libraries like ZeroMQ and libpqxx if needed
find_package(PkgConfig REQUIRED)

# Find ZeroMQ using pkg-config
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq) # IMPORTED_TARGET creates PkgConfig::ZMQ

# Find PostgreSQL (libpq - C library)
find_package(PostgreSQL REQUIRED) # For the C library libpq

# Find libpqxx (C++ wrapper for PostgreSQL) using pkg-config
pkg_check_modules(PQXX REQUIRED IMPORTED_TARGET libpqxx) # IMPORTED_TARGET creates PkgConfig::PQXX

# Removed global include_directories block that was here.

# Add executable sources
add_executable(mmorpg_server
    src/main.cpp
    src/game_data_store.cpp
    src/PlayerManager.cpp      # New
    src/Synchronization.cpp    # New
    # db_connection.hpp is header only
    # database_manager.hpp is header only for methods currently defined (or was)
    # quantum_vm_mock.hpp is header only
    # NPCBehavior.h contains SimpleIdleNPCBehavior definition, no separate .cpp for it yet
)

# Add compiler warnings
if(MSVC)
    target_compile_options(mmorpg_server PRIVATE /W4)
else()
    target_compile_options(mmorpg_server PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add target-specific include directories
target_include_directories(mmorpg_server PRIVATE
    include # Local include directory
    SYSTEM ${ZMQ_INCLUDE_DIRS}
    SYSTEM ${PQXX_INCLUDE_DIRS}
    SYSTEM ${POSTGRESQL_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(mmorpg_server PRIVATE
    NetworkModule             # Link our network module
    PkgConfig::ZMQ            # Link imported target from pkg_check_modules for ZeroMQ
    PkgConfig::PQXX           # Link imported target from pkg_check_modules for libpqxx
    ${POSTGRESQL_LIBRARIES}   # Link libpq (C library) - libpqxx might link this itself, but explicit can be safer
)

message(STATUS "CMake configured for MMORPG_Server.")
message(STATUS "ZMQ Include Dirs: ${ZMQ_INCLUDE_DIRS}")
message(STATUS "ZMQ Libraries: ${ZMQ_LIBRARIES}")
message(STATUS "PQXX Include Dirs: ${PQXX_INCLUDE_DIRS}")
message(STATUS "PQXX Libraries: ${PQXX_LIBRARIES}")
message(STATUS "PostgreSQL Include Dirs: ${POSTGRESQL_INCLUDE_DIRS}")
message(STATUS "PostgreSQL Libraries: ${POSTGRESQL_LIBRARIES}")
