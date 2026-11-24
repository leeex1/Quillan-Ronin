cmake_minimum_required(VERSION 3.15)
project(NextVerseDAW VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# JUCE is expected at ThirdParty/JUCE beneath this directory (or on the
# system prefix path). Set -DJUCE_DIR=<path to JUCEConfig.cmake> to override.
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/JUCE")
find_package(JUCE CONFIG QUIET)

# A source checkout can be supplied with -DJUCE_SOURCE_DIR=<path to JUCE root>.
if(NOT JUCE_FOUND AND JUCE_SOURCE_DIR)
    add_subdirectory("${JUCE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/JUCE_build")
    set(JUCE_FROM_SOURCE TRUE)
elseif(NOT JUCE_FOUND)
    find_package(JUCE CONFIG REQUIRED)
endif()

# Tracktion Engine module root. The checked-in vendored header is a one-line
# placeholder. The JUCE-only audio graph target below does not depend on it;
# the GUI target is built only when a real Tracktion checkout is present.
if(TRACKTION_ROOT_DIR)
    set(TRACKTION_MODULE_ROOT "${TRACKTION_ROOT_DIR}")
else()
    set(TRACKTION_MODULE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/modules/tracktion_engine")
endif()
get_filename_component(TRACKTION_MODULES_PARENT "${TRACKTION_MODULE_ROOT}" DIRECTORY)
set(TRACKTION_OK TRUE)
if(NOT EXISTS "${TRACKTION_MODULE_ROOT}/tracktion_engine.h")
    set(TRACKTION_OK FALSE)
else()
    file(READ "${TRACKTION_MODULE_ROOT}/tracktion_engine.h" _tracktion_header)
    if(_tracktion_header MATCHES "placeholder")
        set(TRACKTION_OK FALSE)
    endif()
endif()

if(TRACKTION_OK)
    # Standard JUCE CMake API. The previous juce_add_juce_project_from_jucer()
    # call is not part of JUCE's CMake API and always failed at configure time.
    juce_add_gui_app(NextVerseDAW
        PRODUCT_NAME "NextVerse DAW"
        COMPANY_NAME "NextVerse"
    )

    target_sources(NextVerseDAW PRIVATE
        Source/Main.cpp
        Source/MainComponent.cpp
        Source/MainComponent.h
    )
    # Tracktion ships as JUCE modules; register the three modules it needs.
    juce_add_module("${TRACKTION_MODULES_PARENT}/tracktion_core")
    juce_add_module("${TRACKTION_MODULES_PARENT}/tracktion_graph")
    juce_add_module("${TRACKTION_MODULE_ROOT}")
    target_include_directories(NextVerseDAW PRIVATE "${TRACKTION_MODULE_ROOT}")

    target_compile_definitions(NextVerseDAW PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:NextVerseDAW,JUCE_PRODUCT_NAME>"
        JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
    )

    target_link_libraries(NextVerseDAW PRIVATE
        tracktion_engine
        tracktion_graph
        tracktion_core
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_gui_extra
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
    )
else()
    message(WARNING
        "Tracktion Engine is still a placeholder; skipping the NextVerseDAW GUI "
        "target. The JUCE audio-graph target and its offline test still build.")
endif()

# Real JUCE audio graph: a tone generator feeding the graph output.
# This target needs JUCE only (no Tracktion) and is regression-tested offline.
juce_add_console_app(NextVerseDAW_AudioGraphTest
    PRODUCT_NAME "NextVerse DAW Audio Graph Test")
target_sources(NextVerseDAW_AudioGraphTest PRIVATE
    tests/juce_audio_graph_test.cpp
    Source/NextVerseToneGenerator.h
    Source/NextVerseSynthProcessor.h)
target_compile_definitions(NextVerseDAW_AudioGraphTest PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0)
target_link_libraries(NextVerseDAW_AudioGraphTest PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_recommended_config_flags)

enable_testing()
add_test(NAME NextVerseDAW_AudioGraphTest COMMAND NextVerseDAW_AudioGraphTest)

message(STATUS "NextVerseDAW configured with JUCE ${JUCE_VERSION}")
