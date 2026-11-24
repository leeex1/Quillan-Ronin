# CMakeLists.txt for the AI Assistant C++ Components
cmake_minimum_required(VERSION 3.10)
project(AIAssistantCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the library
add_library(AIAssistantCpp STATIC
    src/DecisionTree.cpp
    # Add other C++ source files for the AI assistant here
)

if(MSVC)
    target_compile_options(AIAssistantCpp PRIVATE /W4)
else()
    target_compile_options(AIAssistantCpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_include_directories(AIAssistantCpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Example: If AIInterface was a separate library target it might be linked
# target_link_libraries(AIAssistantCpp PRIVATE AIInterfaceLib)
