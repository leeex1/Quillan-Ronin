# Modern CMake Best Practices

## Target-Based Approach
- Use modern target-based commands instead of global variables
- Prefer `target_include_directories()`, `target_link_libraries()`, and `target_compile_definitions()` over their global equivalents
- Example:
  ```cmake
  add_library(mylib STATIC source1.cpp source2.cpp)
  target_include_directories(mylib PUBLIC include)
  target_link_libraries(mylib PUBLIC dependency)
  ```

## Configuration Time Optimizations
1. **Minimize Glob Operations**
   - Avoid `file(GLOB ...)` for source files
   - List source files explicitly for better control and faster configuration
   ```cmake
   set(SOURCES main.cpp widget.cpp utils.cpp)
   add_executable(myApp ${SOURCES})
   ```

2. **Cache Results Properly**
   - Use `find_package` efficiently and don't repeat expensive checks
   ```cmake
   find_package(Boost REQUIRED COMPONENTS filesystem system)
   ```

3. **Reduce Unnecessary Dependencies**
   - Keep optional features truly optional
   - Minimize dependency tree complexity

4. **Use Ninja Build Generator**
   - Significantly faster than Makefiles or Visual Studio generators
   ```bash
   cmake -G Ninja ..
   ```

## Build Time Optimizations
1. **Enable Parallel Builds**
   - Ninja does this automatically
   - For Makefiles: `make -j$(nproc)`

2. **Use Unity Builds**
   - Combine multiple source files into single compilation units
   ```cmake
   set_target_properties(myTarget PROPERTIES UNITY_BUILD ON)
   ```

3. **Precompiled Headers (PCH)**
   - Cache commonly used headers to speed up compilation
   ```cmake
   target_precompile_headers(myTarget PRIVATE <vector> <string> "pch.h")
   ```

4. **Use Compiler Caching**
   - Configure ccache or sccache to avoid redundant compilations
   ```bash
   CC="ccache gcc" CXX="ccache g++" cmake ..
   ```

## Dependency Management
1. **Use find_package in CONFIG mode first**
   ```cmake
   find_package(Boost CONFIG REQUIRED COMPONENTS filesystem)
   ```

2. **Utilize FetchContent for third-party dependencies**
   ```cmake
   include(FetchContent)
   FetchContent_Declare(
     fmt
     GIT_REPOSITORY https://github.com/fmtlib/fmt.git
     GIT_TAG 9.1.0
   )
   FetchContent_MakeAvailable(fmt)
   ```

3. **Consider vcpkg or Conan for comprehensive dependency management**
   ```cmake
   # CMake toolchain file approach
   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path/to/vcpkg]/scripts/buildsystems/vcpkg.cmake
   ```

## Best Practices for Structure
1. **Organize with add_subdirectory**
   - Keep logical modules in separate directories
   - Use add_subdirectory for each module

2. **Create INTERFACE targets for header-only libraries**
   ```cmake
   add_library(header_only INTERFACE)
   target_include_directories(header_only INTERFACE include)
   ```

3. **Use export and install for libraries**
   ```cmake
   install(TARGETS mylib
     EXPORT mylib-targets
     LIBRARY DESTINATION lib
     ARCHIVE DESTINATION lib
     RUNTIME DESTINATION bin
     INCLUDES DESTINATION include
   )
   ```

4. **Follow a consistent layout**
   - /src - Source files
   - /include - Public headers
   - /cmake - CMake modules
   - /test - Test files

## Testing & Quality
1. **Enable CTest**
   ```cmake
   enable_testing()
   add_test(NAME test1 COMMAND mytest arg1 arg2)
   ```

2. **Set up proper packaging**
   ```cmake
   set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
   set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
   include(CPack)
   ```

3. **Configure continuous integration**
   - Generate compilation database: `cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
   - Use with clang-tidy, clang-format, etc.

## Common Pitfalls to Avoid
1. Mixing old and new CMake styles
2. Over-reliance on global variables and properties
3. Hard-coding paths
4. Not handling platform differences correctly
5. Ignoring the dependency graph structure 