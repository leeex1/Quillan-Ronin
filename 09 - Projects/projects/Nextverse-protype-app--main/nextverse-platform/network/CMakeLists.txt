cmake_minimum_required(VERSION 3.10) # Or a suitable version
# CMakeLists.txt for the Network Module
project(NetworkModule)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the library
add_library(NetworkModule STATIC
    src/NetworkHandler.cpp
    src/SecureConnection.cpp
    src/LoadBalancer.cpp
    # Add other source files here as they are created (e.g., src/SecureConnection.cpp, src/LoadBalancer.cpp)
)

if(MSVC)
    target_compile_options(NetworkModule PRIVATE /W4)
else()
    target_compile_options(NetworkModule PRIVATE -Wall -Wextra -Wpedantic)
endif()

# We need to tell CMake where to find the header files for this library
target_include_directories(NetworkModule PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add any necessary link libraries later, for example:
# target_link_libraries(NetworkModule PRIVATE SomeOtherLib)

# Install rules (optional, for packaging)
# install(TARGETS NetworkModule DESTINATION lib)
# install(DIRECTORY include/ DESTINATION include)
