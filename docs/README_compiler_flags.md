# C++ Compiler Flags — Reference & Justification

This document lists every flag that GCC (`/usr/bin/c++`, the system g++) sees
when building the `spike_benchmark` binary and the unit-test executables, and
explains *why* each one is needed for the experimental protocol described in
the project proposal (Stetsenko, *Benchmarking Sparse Matrix Storage Formats
for SNN Spike Propagation*, Spring 2026).

The flag set is defined declaratively in [CMakeLists.txt](CMakeLists.txt) and
verified by inspecting `build/compile_commands.json`. A representative
compile line for a source file in Release mode is:

```text
/usr/bin/c++  -I.../include  -O3 -DNDEBUG  -std=gnu++17 \
    -Wall -Wextra -Wpedantic  -O2 -DNDEBUG  -c src/csr_matrix.cpp
```

The duplication of `-O3 -DNDEBUG` (CMake's Release default) followed by
`-O2 -DNDEBUG` (this project's explicit override) is intentional and
documented below — the **last** `-O` on the command line is the one GCC
honours, so the effective optimisation level is `-O2`.

---

## Language standard

### `-std=gnu++17`

Selected by `set(CMAKE_CXX_STANDARD 17)` + `CMAKE_CXX_STANDARD_REQUIRED ON`
in [CMakeLists.txt](CMakeLists.txt).

**Why it matters.**
- `std::chrono::high_resolution_clock`, `std::mt19937`, structured bindings,
  `if constexpr`, and `<filesystem>` (used in CSV I/O) all require C++17.
- The proposal §6 explicitly lists "C++17, CMake, std::chrono" as the core
  benchmark stack — pinning the standard guarantees that the compiled
  binary matches the proposal's claim and prevents reviewers from getting a
  C++20 build with different `<chrono>` semantics.
- `gnu++17` (rather than strict `c++17`) is GCC's default and enables a few
  GNU extensions that CMake itself relies on; it does *not* introduce any
  non-portable code into the project sources.

---

## Optimisation flags (Release)

Set in [CMakeLists.txt](CMakeLists.txt):

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -DNDEBUG)
```

### `-O2`

**Why it matters.**
- The benchmark *measures* the inner scatter / gather loops in
  [src/csr_matrix.cpp](src/csr_matrix.cpp),
  [src/csc_matrix.cpp](src/csc_matrix.cpp),
  [src/coo_matrix.cpp](src/coo_matrix.cpp), and
  [src/ell_matrix.cpp](src/ell_matrix.cpp). At `-O0` the compiler keeps every
  variable on the stack and the timings would be dominated by spurious
  loads, not by the format's intrinsic cache behaviour — the very thing the
  thesis is trying to isolate.
- `-O2` is the standard SNN-simulator baseline (NEST and GeNN both compile
  generated code at `-O2`/`-O3`), which makes the head-to-head comparison
  in RQ3 a like-for-like one.
- `-O2` was chosen over `-O3` deliberately: `-O3` enables aggressive
  auto-vectorisation and loop interchange that can *hide* the format
  differences the project wants to expose (e.g. a vectorised COO loop can
  artificially close the gap to CSR). `-O2` keeps the compiler honest:
  inlining and basic CSE are on, but the data-access pattern of each
  format is preserved in the emitted code.

> Note: CMake's built-in `Release` default (`-O3 -DNDEBUG`) appears earlier
> on the command line, but the explicit `-O2` we add wins because GCC
> applies the *last* `-O` flag.

### `-DNDEBUG`

**Why it matters.**
- Disables `assert()` in release builds. The sparse-format implementations
  use `assert` for invariants such as "row index in range" and
  "`row_ptr` is monotone"; leaving them on would inject a branch into
  every inner-loop iteration and again pollute the timing.
- It is also required to match the standard SNN-simulator build setting,
  per the rationale above for `-O2`.

---

## Optimisation flags (Debug)

```cmake
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0)
```

### `-g`

**Why it matters.** Emits DWARF debug info so that:
- `gdb` / `lldb` can step through the scatter/gather kernels when a unit
  test in [tests/](tests/) fails;
- `perf record` and `perf annotate` (used by `scripts/run_benchmarks.sh
  --perf`) can attribute cache misses to specific source lines — required
  for the cache analysis in
  [docs/README_cache_and_metrics.md](docs/README_cache_and_metrics.md).

### `-O0`

**Why it matters.** Disables every optimisation. This is the *correct*
setting for Debug builds because optimised stack frames lose variables and
make breakpoints inaccurate. Debug builds are **never** used for timing
runs — only for test failures and gdb sessions — so the performance hit is
irrelevant.

---

## Warning flags (always on)

```cmake
add_compile_options(-Wall -Wextra -Wpedantic)
```

These three flags catch a deliberately broad class of issues that would
otherwise threaten the validity of the measurements.

### `-Wall`

Enables the standard set of warnings GCC considers high-confidence
(unused variables, missing returns, signed/unsigned comparisons, …).

**Why it matters for an SNN benchmark.** The hot loops manipulate `int`
indices into `std::vector<double>` weight arrays. A silent
`int`/`size_t` mismatch in `row_ptr[i]` access can either crash on large
networks or — worse — silently mask out high-bit edges. `-Wall` catches
those at compile time, before they corrupt benchmark CSVs.

### `-Wextra`

Adds further lints (e.g. `-Wunused-parameter`,
`-Wmissing-field-initializers`, `-Wempty-body`).

**Why it matters.** The `BenchmarkConfig` and `BenchmarkResult` structs in
[include/benchmark.h](include/benchmark.h) are aggregate-initialised. A
forgotten field initialiser would default to a garbage measurement column
in the output CSV; `-Wextra`'s `-Wmissing-field-initializers` flags that
immediately.

### `-Wpedantic`

Enforces strict ISO C++ compliance and warns on GNU extensions.

**Why it matters.**
- The project must build identically on any C++17-conforming compiler so
  reviewers can reproduce the results without GCC-specific flags. Catching
  GNU extensions at compile time keeps that contract.
- The proposal claims portability to "C++17, CMake, std::chrono"; `-Wpedantic`
  is the static check that the source actually honours that claim.

---

## Include path

### `-I<repo>/include`

Added by `target_include_directories(spike_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)`.

**Why it matters.** Public headers
([include/sparse_matrix.h](include/sparse_matrix.h),
[include/topology.h](include/topology.h),
[include/lif_neuron.h](include/lif_neuron.h),
[include/benchmark.h](include/benchmark.h),
[include/csv_io.h](include/csv_io.h)) are reachable both from sources in
`src/` and from the unit-test translation units in `tests/`, ensuring that
every component sees the same struct layouts and that benchmark and tests
compile against identical declarations. Without a single shared include
root, `BenchmarkResult` could drift between `spike_lib` and the tests, and
the validation in [tests/test_formats.cpp](tests/test_formats.cpp) would no
longer prove anything about the binary that produces the CSVs.

---

## Implicitly-set compile/link flags worth documenting

These are not in [CMakeLists.txt](CMakeLists.txt) explicitly but are
relevant to the experimental protocol:

| Flag | Source | Relevance |
|------|--------|-----------|
| `-fPIC` | CMake default for archives on Linux x86-64 | Ensures `spike_lib.a` (and the test binaries that link it) load at any address; immaterial for timing because all hot symbols are inlined into the executable. |
| Default linker | system `ld` | No `-static`, no `-lnuma`: the binary stays portable across the lab's machines, which is required by Step 6 ("repeat 10 trials" on a documented host). |
| No `-march=native` | (deliberately *not* set) | Disabling `-march=native` is a methodological choice — using it would make the binary CPU-specific and the published numbers non-reproducible across machines. The proposal calls for a portable benchmark, so we accept the small SIMD performance loss. |
| No `-fopenmp` | (deliberately *not* set) | The benchmark is single-threaded by design (see *Scaling* section in [docs/README_topology_flags.md](docs/README_topology_flags.md)). Adding OpenMP would change the experimental object from "format quality" to "format + parallelisation quality" and is out of scope for the proposal. |

---

## Mapping to the research questions

| Research question (proposal §2) | Compiler flags that protect the result |
|---------------------------------|----------------------------------------|
| RQ1 — which format is fastest? | `-O2 -DNDEBUG` (uniform, realistic baseline; no `-O3` vectorisation that would mask format differences); `-Wall -Wextra` (no silent integer truncations corrupting timings). |
| RQ2 — do synthetic results hold on NEST connectivity? | `-std=gnu++17` (stable `<chrono>` and `<filesystem>` semantics across the synthetic and NEST-CSV code paths in [src/csv_io.cpp](src/csv_io.cpp)). |
| RQ3 — does the CPU ranking agree with GeNN? | Absence of `-march=native` and `-fopenmp` (CPU pipeline stays portable and single-threaded so the comparison object matches what GeNN measures on the GPU). |

---

## Summary table

| Flag | Set in | Purpose |
|------|--------|---------|
| `-std=gnu++17` | CMake `CXX_STANDARD 17` | Pin the language version the proposal commits to. |
| `-O2` | `add_compile_options` (Release) | Realistic optimisation, deliberately *not* `-O3`. |
| `-DNDEBUG` | `add_compile_options` (Release) | Strip `assert()` from inner loops. |
| `-g` | `add_compile_options` (Debug) | Symbols for `gdb` and `perf annotate`. |
| `-O0` | `add_compile_options` (Debug) | Faithful debugging (never used for timing). |
| `-Wall` | always on | Catch index-arithmetic mistakes that would corrupt CSVs. |
| `-Wextra` | always on | Catch missing struct-field initialisers in `BenchmarkResult`. |
| `-Wpedantic` | always on | Enforce ISO C++17, keep the build reproducible across compilers. |
| `-I.../include` | `target_include_directories` | Single source of truth for shared headers between binary and tests. |
