# C++ Optimization and Modern Techniques

## Compile-Time Optimizations

1. **Template Metaprogramming**
   - Perform calculations at compile time with constexpr
   - Use type traits for compile-time code selection
   ```cpp
   template <typename T>
   constexpr bool is_numeric = std::is_arithmetic_v<T>;
   
   template <typename T>
   std::enable_if_t<is_numeric<T>, T> square(T value) {
       return value * value;
   }
   ```

2. **Constant Expressions**
   - Use constexpr for compile-time evaluation
   - Create constexpr functions for compile-time computation
   ```cpp
   constexpr int factorial(int n) {
       return (n <= 1) ? 1 : (n * factorial(n - 1));
   }
   ```

3. **Link-Time Optimization (LTO)**
   - Enable in compiler settings to optimize across translation units
   - Compiler flags: -flto (GCC/Clang), /GL (MSVC)

## Runtime Optimizations

1. **Data Structures**
   - Choose appropriate containers based on access patterns
   - Optimize for specific operations (insertion, deletion, lookup)
   - Consider memory layout and cache effects
   ```cpp
   // For frequent lookups by key
   std::unordered_map<Key, Value> fast_lookup;
   
   // For sorted data with binary search
   std::vector<Item> sorted_items; // with std::sort and std::lower_bound
   
   // For frequent insertion/removal at both ends
   std::deque<Item> flexible_container;
   ```

2. **Algorithms**
   - Use standard library algorithms where possible
   - Avoid redundant work (early returns, short-circuiting)
   - Consider algorithmic complexity
   ```cpp
   // Use algorithms for clearer, optimized code
   std::sort(vec.begin(), vec.end());
   auto it = std::lower_bound(vec.begin(), vec.end(), value);
   ```

3. **Memory Management**
   - Reduce dynamic allocations
   - Use preallocated buffers or memory pools
   - Consider custom allocators for hot paths
   ```cpp
   // Preallocate to avoid reallocations
   vec.reserve(expected_size);
   
   // Use placement new for custom memory management
   char buffer[sizeof(MyClass)];
   MyClass* obj = new(buffer) MyClass();
   ```

4. **Multi-threading**
   - Use std::async for simple parallelization
   - Consider thread pools for frequent tasks
   - Prefer lock-free structures when appropriate
   ```cpp
   auto future = std::async(std::launch::async, compute_intensive_function, arg);
   result = future.get();
   ```

## Modern C++ Features for Performance

1. **Move Semantics**
   - Use rvalue references and move operations to avoid copies
   - Implement move constructors and assignment operators
   ```cpp
   class Buffer {
   public:
       Buffer(Buffer&& other) noexcept 
           : data_(std::exchange(other.data_, nullptr))
           , size_(std::exchange(other.size_, 0)) {}
   private:
       char* data_;
       size_t size_;
   };
   ```

2. **Perfect Forwarding**
   - Use universal references and std::forward for zero-overhead
   ```cpp
   template<typename... Args>
   void emplace(Args&&... args) {
       new_element = new T(std::forward<Args>(args)...);
   }
   ```

3. **Structured Bindings**
   - Efficiently unpack tuples and structures
   ```cpp
   auto [iter, success] = my_map.insert({key, value});
   ```

4. **Lambda Expressions**
   - Use for small function objects without overhead
   - Capture only what you need (preferably by value)
   ```cpp
   auto sorter = [](const auto& a, const auto& b) { return a.value < b.value; };
   std::sort(items.begin(), items.end(), sorter);
   ```

## Low-Level Optimizations

1. **SIMD Instructions**
   - Use compiler intrinsics or libraries like Eigen
   - Consider auto-vectorization hints
   ```cpp
   #include <immintrin.h>
   __m256 a = _mm256_set1_ps(1.0f);
   __m256 b = _mm256_set1_ps(2.0f);
   __m256 result = _mm256_add_ps(a, b);
   ```

2. **Cache Optimization**
   - Design data structures for cache locality
   - Use prefetching for predictable access patterns
   ```cpp
   struct AoS { // Array of Structures (often worse for SIMD)
       float x, y, z;
   };
   
   // Versus:
   struct SoA { // Structure of Arrays (often better for SIMD)
       float* x;
       float* y;
       float* z;
   };
   ```

3. **Branch Prediction**
   - Make the common case fast (hint with [[likely]] attribute)
   - Consider branch-free alternatives for hot paths
   ```cpp
   if (likely_condition) [[likely]] {
       // Common path
   } else {
       // Uncommon path
   }
   ```

## Measurement and Profiling

1. **Benchmarking**
   - Use Google Benchmark or similar libraries
   - Measure actual performance, don't guess
   ```cpp
   static void BenchmarkFunction(benchmark::State& state) {
       for (auto _ : state) {
           // Code to benchmark
       }
   }
   BENCHMARK(BenchmarkFunction);
   ```

2. **Profiling Tools**
   - Use CPU profilers (perf, VTune, etc.)
   - Memory profilers (Valgrind, Dr. Memory)
   - Analyze hotspots and bottlenecks
   
3. **Compiler Explorer**
   - Check generated assembly for critical sections
   - Compare different optimization levels
   - Experiment with compiler flags

## Best Practices

1. **Optimize Selectively**
   - Focus on the 20% of code that takes 80% of time
   - Profile before optimizing
   - Document performance-critical code

2. **Maintain Readability**
   - Avoid obscure optimizations that harm maintainability
   - Comment the reason for optimization techniques
   - Consider abstraction cost carefully

3. **Test Thoroughly**
   - Verify correctness after optimizations
   - Benchmark across different platforms
   - Consider edge cases

4. **Stay Modern**
   - Keep up with compiler improvements
   - Revisit old optimizations that may be outdated
   - Leverage new language features 