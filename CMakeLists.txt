cmake_minimum_required(VERSION 3.14)
project(SpikePropagationSNN VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
endif()

# ---------- library target ----------
add_library(spike_lib
    src/coo_matrix.cpp
    src/csr_matrix.cpp
    src/csc_matrix.cpp
    src/ell_matrix.cpp
    src/topology.cpp
    src/lif_neuron.cpp
    src/benchmark.cpp
    src/csv_io.cpp
)
target_include_directories(spike_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---------- main executable ----------
add_executable(spike_benchmark src/main.cpp)
target_link_libraries(spike_benchmark PRIVATE spike_lib)

# ---------- tests (Catch2 via FetchContent) ----------
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
add_subdirectory(tests)
