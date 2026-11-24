cmake_minimum_required(VERSION 3.15)
project(NextVerseCoreAlpha LANGUAGES CXX)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The core simulation modules depend only on the C++ standard library. The
# previous FetchContent calls pulled nlohmann/json, cpp-httplib and GoogleTest
# from the network at configure time, which made offline builds impossible.
add_library(QCC_Alpha qcc_alpha.cpp)
target_include_directories(QCC_Alpha PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(QSVM_Alpha qsvm_alpha.cpp)
target_include_directories(QSVM_Alpha PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(QSVM_Alpha PUBLIC QCC_Alpha)
find_package(Threads REQUIRED)
target_link_libraries(QSVM_Alpha PUBLIC Threads::Threads)

enable_testing()

# Dependency-free verification, always available.
add_executable(CoreAlphaNativeVerify tests/native_verify.cpp)
target_link_libraries(CoreAlphaNativeVerify PRIVATE QSVM_Alpha)
add_test(NAME CoreAlphaNativeVerify COMMAND CoreAlphaNativeVerify)

# The original GTest suites are built only when GTest is already installed.
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(RunQCCAlphaTests tests/test_qcc_alpha.cpp)
    target_link_libraries(RunQCCAlphaTests PRIVATE QCC_Alpha GTest::gtest GTest::gtest_main)
    add_test(NAME QCC_Alpha_Tests COMMAND RunQCCAlphaTests)

    add_executable(RunQSVMAlphaTests tests/test_qsvm_alpha.cpp)
    target_link_libraries(RunQSVMAlphaTests PRIVATE QSVM_Alpha GTest::gtest GTest::gtest_main)
    add_test(NAME QSVM_Alpha_Tests COMMAND RunQSVMAlphaTests)
else()
    message(STATUS "GTest not installed; building the dependency-free native verification only.")
endif()

message(STATUS "NextVerse Core Alpha configured (offline).")
