# Benchmark Harness

## Overview

The benchmark harness orchestrates the end-to-end measurement of spike
propagation performance.  It connects topology generation, sparse-format
construction, and LIF simulation into a single timed loop, recording
wall-clock time, peak resident memory, and spike count.

---

## Methodology

Each benchmark trial proceeds as follows:

1. **Topology generation** — Create a `COOTriplets` using the specified
   graph model and parameters.
2. **Format construction** — Build the target sparse matrix format from the
   COO data.  Construction time is *excluded* from the measurement.
3. **Neuron initialization** — Create a `LIFPopulation` of $N$ neurons.
   Approximately 1% of neurons are given initial spikes (seeded
   deterministically).
4. **Timed simulation loop** — For each of the $T$ timesteps:
   - Clear the synaptic current buffer.
   - Call `matrix.scatter(spikes, I_syn)` to propagate spikes through
     the weight matrix.
   - Call `population.step(I_syn, spikes)` to integrate the LIF dynamics
     and detect new spikes.
5. **Measurement** — Record wall-clock time (`std::chrono::steady_clock`),
   peak RSS (from `/proc/self/status` VmHWM), and cumulative spike count.

### Rationale for Excluding Construction

Format construction is a one-time cost that depends on sort algorithms
(for CSR/CSC) and is not representative of runtime performance.  By
measuring only the simulation loop, we isolate the effect of the sparse
format's access pattern on spike propagation throughput.

---

## Profiling

### Programmatic (Built-in)

The function `get_peak_rss_kb()` reads `/proc/self/status` on Linux and
reports `VmHWM` (high-water mark of resident set size) in kilobytes.
This captures the true physical memory footprint of the process, including
the sparse matrix, neuron state, and all auxiliary buffers.

### Cache Hierarchy Detection

At startup, the benchmark reads `/sys/devices/system/cpu/cpu0/cache/`
to detect L1d, L1i, L2, and L3 cache sizes. These are:
- Printed to stdout so the cache hierarchy is always visible.
- Written to `results/cache_info.csv` for automated analysis.
- Used to compute **matrix-to-cache size ratios** (see Derived Metrics).

This lets you determine which sparse matrix fits in which cache level —
a central question for understanding format-dependent performance.

### Derived Metrics (Built-in)

The benchmark computes the following derived metrics for each configuration:

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| Effective bandwidth (GB/s) | `matrix_bytes / mean_time` | Upper-bound memory bandwidth utilisation; reveals memory-bound vs compute-bound behaviour |
| Scatter throughput (edges/ms) | `spikes_per_step × avg_out_degree × timesteps / mean_time` | Direct measure of spike-propagation efficiency per format |
| Bytes per spike | `matrix_bytes / total_spikes` | Memory cost per propagated spike event |
| Cache ratio L1d | `matrix_bytes / L1d_size` | >1 means the matrix spills out of L1 |
| Cache ratio L2 | `matrix_bytes / L2_size` | >1 means the matrix spills out of L2 |
| Cache ratio L3 | `matrix_bytes / L3_size` | >1 means the matrix must fetch from DRAM |

### External (perf stat)

The shell script `scripts/run_benchmarks.sh --perf` wraps each benchmark
invocation with `perf stat` collecting 8 hardware counters:

| Counter | What it measures |
|---------|-----------------|
| `cache-misses` | Total cache misses (combined L1+LLC) |
| `cache-references` | Total cache references (= hits + misses) |
| `instructions` | Retired instructions |
| `cycles` | CPU cycles consumed |
| `L1-dcache-load-misses` | L1 data cache misses — most performance-critical for sparse ops |
| `LLC-load-misses` | Last-level cache misses — actual DRAM traffic |
| `dTLB-load-misses` | TLB misses — reveals scattered vs sequential access patterns |
| `branch-misses` | Branch mispredictions — higher for irregular sparsity patterns |

**Key derived ratios from perf data:**
- **Cache miss rate** = `cache-misses / cache-references`
- **IPC** = `instructions / cycles` — pipeline efficiency
- **L1 miss rate** = `L1-dcache-load-misses / cache-references`

---

## Sweep Parameters

The `--sweep` CLI option runs a full grid over:

| Dimension | Values |
|-----------|--------|
| Network size $N$ | 1000, 2000, 5000, 10 000, 20 000 |
| Connection density $p$ | 0.01, 0.02, 0.05, 0.1, 0.2 |
| Topology | `erdos_renyi`, `fixed_indegree`, `barabasi_albert`, `watts_strogatz` |
| Format | `coo`, `csr`, `csc`, `ell` |

Each configuration is run with 3 trials; the output CSV contains one row
per trial.  Downstream analysis (plotting scripts) computes means and
confidence intervals.

---

## CSV Output Schema

### benchmark_results.csv

| Column | Type | Description |
|--------|------|-------------|
| `format` | string | Sparse matrix format name |
| `topology` | string | Graph model name |
| `N` | int | Number of neurons |
| `density` | float | Connection density parameter |
| `timesteps` | int | Number of simulation steps |
| `trials` | int | Number of repeat trials |
| `mean_time_ms` | float | Mean wall-clock time across trials |
| `std_time_ms` | float | Standard deviation of time |
| `peak_rss_kb` | int | Peak resident set size in kilobytes |
| `total_spikes` | int | Mean total spike events per trial |
| `spikes_per_step` | float | Average spikes per timestep |
| `memory_bytes` | int | Sparse matrix memory footprint |
| `nnz` | int | Number of non-zero entries |
| `effective_bw_gbps` | float | Effective memory bandwidth (GB/s) |
| `scatter_throughput_edges_per_ms` | float | Edges scattered per millisecond |
| `bytes_per_spike` | float | Matrix bytes per propagated spike |
| `cache_ratio_L1` | float | Matrix size / L1d cache size |
| `cache_ratio_L2` | float | Matrix size / L2 cache size |
| `cache_ratio_L3` | float | Matrix size / L3 cache size |

### perf_results.csv

| Column | Type | Description |
|--------|------|-------------|
| `format` | string | Sparse matrix format |
| `topology` | string | Graph model |
| `N` | int | Network size |
| `density` | float | Connection density |
| `cache_misses` | int | Total cache misses |
| `cache_refs` | int | Total cache references |
| `instructions` | int | Retired instructions |
| `cycles` | int | CPU cycles |
| `L1d_load_misses` | int | L1 data cache load misses |
| `LLC_load_misses` | int | Last-level cache load misses |
| `dTLB_load_misses` | int | Data TLB load misses |
| `branch_misses` | int | Branch mispredictions |

### cache_info.csv

| Column | Type | Description |
|--------|------|-------------|
| `level` | string | Cache level (L1d, L1i, L2, L3) |
| `size_bytes` | int | Cache size in bytes |

---

## API Reference

### Configuration

```cpp
struct BenchmarkConfig {
    std::string format;      // "coo", "csr", "csc", "ell"
    std::string topology;    // "erdos_renyi", etc.
    int         N;
    double      density;
    int         timesteps;
    unsigned    seed;
};
```

### Result

```cpp
struct BenchmarkResult {
    std::string format, topology;
    int    N, nnz, timesteps, trial;
    double density, time_ms;
    long   peak_rss_kb;
    long   total_spikes;
};
```

### Core Functions

```cpp
BenchmarkResult run_benchmark(const BenchmarkConfig& config, int trial);
long get_peak_rss_kb();

void write_csv_header(std::ostream& os);
void write_csv_row(std::ostream& os, const BenchmarkResult& r);
```

### Files

| File | Purpose |
|------|---------|
| `include/benchmark.h` | Config/Result structs, function declarations |
| `src/benchmark.cpp`   | Benchmark loop, RSS measurement, CSV I/O |
| `src/main.cpp`        | CLI argument parser, single-run and sweep modes |
