# JDXX Chains — AI DAW Copilot: the "hands" plugin (Phase 1)
# AGPL-3.0. Derivative of RoninEQ (AGPL-3.0) by CrashOverrideX. See LICENSE.
#
# Windows / Visual Studio quick start:
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
#   cmake --build build --config Release
#   ctest --test-dir build -C Release --output-on-failure
#
# NOTE: JDXX reference values below (the five §4 chains) are the authoritative
# spec. BandLab's built-in internals are proprietary; every effect here is a
# clean-room implementation matched to the documented parameters, not copied.

cmake_minimum_required (VERSION 3.22)

# ---- The plugin, renamed in exactly one place ----
set (JDXC_NAME "JDXX Chains" CACHE STRING "Plugin display/product name")
set (JDXC_MANUFACTURER "CrashOverrideX" CACHE STRING "Plugin manufacturer name")
set (JDXX_CHAINS_VERSION "0.1.0" CACHE STRING "Plugin version (x.y.z)")

project (JdxxChains VERSION ${JDXX_CHAINS_VERSION} LANGUAGES C CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

string (REPLACE "." ";" _ver_parts "${JDXX_CHAINS_VERSION}")
list (GET _ver_parts 0 _ver_major)
list (GET _ver_parts 1 _ver_minor)
list (GET _ver_parts 2 _ver_patch)
math (EXPR JDXX_CHAINS_VERSION_HEX
      "(${_ver_major} << 16) | (${_ver_minor} << 8) | ${_ver_patch}")
set (JDXX_CHAINS_VERSION_STRING "${JDXX_CHAINS_VERSION}")

# ---- JUCE 8 via FetchContent (or reuse a pre-fetched checkout) ----
include (FetchContent)
FetchContent_Declare (
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.8
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable (JUCE)

configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PluginInfo.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/PluginInfo.h"
    @ONLY)

# ---- The plugin ----
juce_add_plugin (JdxxChains
    PRODUCT_NAME                "${JDXC_NAME}"
    COMPANY_NAME                "${JDXC_MANUFACTURER}"
    VERSION                     "${JDXX_CHAINS_VERSION}"
    FORMATS                     VST3 Standalone
    PLUGIN_MANUFACTURER_CODE    CoX1
    PLUGIN_CODE                 JDXX
    IS_SYNTH                    FALSE
    NEEDS_MIDI_INPUT            FALSE
    NEEDS_MIDI_OUTPUT           FALSE
    IS_MIDI_EFFECT              FALSE
    VST3_CATEGORIES             "Fx|Mastering"
    COPY_PLUGIN_AFTER_BUILD     TRUE)

# Brand-new plugin: no VST2 lineage — do NOT claim VST2 replaceability.
target_compile_definitions (JdxxChains PUBLIC JUCE_VST3_CAN_REPLACE_VST2=0)

target_sources (JdxxChains PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/DSP/Biquad.cpp
    Source/DSP/EnvelopeFollower.cpp
    Source/Chains/Modules.cpp
    Source/Chains/ChainEngine.cpp
    Source/Chains/ChainSpecs.cpp)

target_include_directories (JdxxChains PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Source"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_compile_definitions (JdxxChains PRIVATE
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0)

target_link_libraries (JdxxChains PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

juce_generate_juce_header (JdxxChains)

# ---- Chain smoke test (no JUCE; runs anywhere) ----
add_executable (chain_smoke
    Source/Tests/chain_smoke.cpp
    Source/DSP/Biquad.cpp
    Source/DSP/EnvelopeFollower.cpp
    Source/Chains/Modules.cpp
    Source/Chains/ChainEngine.cpp
    Source/Chains/ChainSpecs.cpp)
target_include_directories (chain_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Source")

include (CTest)
add_test (NAME chain_smoke COMMAND chain_smoke)
