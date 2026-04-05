cmake_minimum_required(VERSION 3.16)
project(SpikePropagation VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults to Release for benchmarking.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# ── Library: sparse matrix formats + LIF model + topology generators ──────────
add_library(snn_lib STATIC
    src/coo_matrix.cpp
    src/csr_matrix.cpp
    src/csc_matrix.cpp
    src/ell_matrix.cpp
    src/lif_neuron.cpp
    src/network.cpp
)
target_include_directories(snn_lib PUBLIC include)

# ── Benchmark executable ──────────────────────────────────────────────────────
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE snn_lib)

# ── Unit tests ────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_formats tests/test_formats.cpp)
target_link_libraries(test_formats PRIVATE snn_lib)

add_test(NAME format_tests COMMAND test_formats)
