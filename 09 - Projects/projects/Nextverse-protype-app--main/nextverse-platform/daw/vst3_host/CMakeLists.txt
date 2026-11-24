cmake_minimum_required(VERSION 3.22)
project(NextVerseVST3Host LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE source is already present on this machine inside the ronineq project.
if(NOT JUCE_SOURCE_DIR)
    message(FATAL_ERROR "Pass -DJUCE_SOURCE_DIR=<path to JUCE root>")
endif()
add_subdirectory("${JUCE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/JUCE_build")

juce_add_console_app(NVVST3Host
    PRODUCT_NAME "NextVerse VST3 Host"
    VERSION 0.1.0)

target_sources(NVVST3Host PRIVATE vst3_host.cpp)
target_compile_definitions(NVVST3Host PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGIN_HOSTING=1
    JUCE_PLUGINHOST_VST3=1)
target_link_libraries(NVVST3Host PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_core
    juce::juce_recommended_config_flags)

juce_add_console_app(NVVST3Graph
    PRODUCT_NAME "NextVerse VST3 Graph"
    VERSION 0.1.0)
target_sources(NVVST3Graph PRIVATE vst3_graph.cpp)
target_compile_definitions(NVVST3Graph PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGIN_HOSTING=1
    JUCE_PLUGINHOST_VST3=1)
target_link_libraries(NVVST3Graph PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_core
    juce::juce_recommended_config_flags)

juce_add_console_app(NVVST3Parity
    PRODUCT_NAME "NextVerse VST3 Parity Benchmark"
    VERSION 0.1.0)
target_sources(NVVST3Parity PRIVATE vst3_parity.cpp)
target_compile_definitions(NVVST3Parity PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGIN_HOSTING=1
    JUCE_PLUGINHOST_VST3=1)
target_link_libraries(NVVST3Parity PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_core
    juce::juce_recommended_config_flags)

juce_add_console_app(NVVST3Mixer
    PRODUCT_NAME "NextVerse VST3 Mixer Benchmark"
    VERSION 0.1.0)
target_sources(NVVST3Mixer PRIVATE vst3_mixer.cpp)
target_compile_definitions(NVVST3Mixer PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGIN_HOSTING=1
    JUCE_PLUGINHOST_VST3=1)
target_link_libraries(NVVST3Mixer PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_basics
    juce::juce_core
    juce::juce_recommended_config_flags)
