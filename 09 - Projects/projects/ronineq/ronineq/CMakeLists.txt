# RoninEQ — clean-room dynamic equalizer (VST3 + Standalone)
# License: AGPL-3.0. See LICENSE.
#
# Windows / Visual Studio quick start:
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
# The VST3 is copied to the system VST3 folder automatically
# (COPY_PLUGIN_AFTER_BUILD). Run the DSP smoke test with:
#   ctest --test-dir build -C Release --output-on-failure

cmake_minimum_required (VERSION 3.22)

# ---- Rename the plugin in exactly one place ----
set (RONINEQ_NAME "RoninEQ" CACHE STRING "Plugin display/product name")
set (RONINEQ_MANUFACTURER "CrashOverrideX" CACHE STRING "Plugin manufacturer name")
set (RONINEQ_VERSION "1.0.0" CACHE STRING "Plugin version (x.y.z)")

project (RoninEQ VERSION ${RONINEQ_VERSION} LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Version hex for the JUCE plugin header, e.g. 1.0.0 -> 0x010000.
string (REPLACE "." ";" _ver_parts "${RONINEQ_VERSION}")
list (GET _ver_parts 0 _ver_major)
list (GET _ver_parts 1 _ver_minor)
list (GET _ver_parts 2 _ver_patch)
math (EXPR RONINEQ_VERSION_HEX
      "(${_ver_major} << 16) | (${_ver_minor} << 8) | ${_ver_patch}")
set (RONINEQ_VERSION_STRING "${RONINEQ_VERSION}")

# ---- JUCE 8 via FetchContent ----
include (FetchContent)
FetchContent_Declare (
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.8   # any 8.x tag works; pinned for reproducible builds
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable (JUCE)

# Generated header carrying the plugin name (single rename point).
configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PluginInfo.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/PluginInfo.h"
    @ONLY)

# ---- The plugin ----
juce_add_plugin (RoninEQ
    PRODUCT_NAME                "${RONINEQ_NAME}"
    COMPANY_NAME                "${RONINEQ_MANUFACTURER}"
    VERSION                     "${RONINEQ_VERSION}"
    FORMATS                     VST3 Standalone
    PLUGIN_MANUFACTURER_CODE    CoX1
    PLUGIN_CODE                 RnEQ
    IS_SYNTH                    FALSE
    NEEDS_MIDI_INPUT            FALSE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    VST3_CATEGORIES             "Fx|EQ|Dynamics"
    COPY_PLUGIN_AFTER_BUILD     TRUE)

# Brand-new plugin: no VST2 ever released, so the VST3 must NOT claim to
# replace a VST2 (JUCE errors out otherwise — see juce_audio_plugin_client_VST3.cpp).
# PUBLIC so the define reaches the RoninEQ_VST3 wrapper target.
target_compile_definitions (RoninEQ PUBLIC JUCE_VST3_CAN_REPLACE_VST2=0)

target_sources (RoninEQ PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/DSP/Biquad.cpp
    Source/DSP/EnvelopeFollower.cpp
    Source/DSP/DynamicBand.cpp)

target_include_directories (RoninEQ PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Source"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_compile_definitions (RoninEQ PRIVATE
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0)

target_link_libraries (RoninEQ PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

juce_generate_juce_header (RoninEQ)

# ---- DSP smoke test (no JUCE; runs anywhere, incl. this machine) ----
add_executable (dsp_smoke
    Source/Tests/dsp_smoke.cpp
    Source/DSP/Biquad.cpp
    Source/DSP/EnvelopeFollower.cpp
    Source/DSP/DynamicBand.cpp)
target_include_directories (dsp_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Source")

include (CTest)
add_test (NAME dsp_smoke COMMAND dsp_smoke)
