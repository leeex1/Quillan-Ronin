cmake_minimum_required(VERSION 3.22)
project(NextVerseDAWGui VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT JUCE_SOURCE_DIR)
    message(FATAL_ERROR "Pass -DJUCE_SOURCE_DIR=<path to JUCE root>")
endif()
add_subdirectory("${JUCE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/JUCE_build")

juce_add_gui_app(NextVerseDAWGui
    PRODUCT_NAME "NextVerse DAW"
    VERSION 0.1.0)

target_sources(NextVerseDAWGui PRIVATE Main.cpp)
target_compile_definitions(NextVerseDAWGui PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGIN_HOSTING=1
    JUCE_PLUGINHOST_VST3=1)
target_link_libraries(NextVerseDAWGui PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_basics
    juce::juce_recommended_config_flags)
