cmake_minimum_required(VERSION 3.15)
project(QuillanCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/arch:AVX2 /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-mavx2 -mfma -O3 -Wall)
endif()

include_directories(include)

add_executable(quillan_engine src/quillan_cli.cpp)

if(WIN32)
    target_link_libraries(quillan_engine PRIVATE psapi)
endif()
