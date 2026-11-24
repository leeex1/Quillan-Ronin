cmake_minimum_required(VERSION 3.20)
project(NextVerseBrowserPlatform VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6 packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets WebEngineWidgets WebChannel)

# Enable Qt auto MOC, UIC, and RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Source files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/LocalProxy.cpp
    src/LocalProxy.h
    src/WebBridge.cpp
    src/WebBridge.h
)

# Resources
qt6_add_resources(SOURCES
    resources/resources.qrc
)

# Create executable
qt6_add_executable(NextVerseBrowserPlatform ${SOURCES})

# Link Qt libraries
target_link_libraries(NextVerseBrowserPlatform
    Qt6::Core
    Qt6::Widgets
    Qt6::WebEngineWidgets
    Qt6::WebChannel
)

# Set target properties
set_target_properties(NextVerseBrowserPlatform PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)

# Copy React build files to resources during build
add_custom_target(copy_react_build ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/react-frontend/build
    ${CMAKE_BINARY_DIR}/resources/web
    COMMENT "Copying React build files to resources"
)

add_dependencies(NextVerseBrowserPlatform copy_react_build)
