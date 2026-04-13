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
2. **Weight normalisation** — All recurrent weights are set to
   $w = G / \sqrt{K_{\text{avg}}}$, where $G$ is the coupling strength
   (default 2.0) and $K_{\text{avg}}$ is the mean degree.  This ensures
   consistent network dynamics across all topologies.
3. **Format construction** — Build the target sparse matrix format from the
   COO data.  Construction time is *excluded* from the measurement.
4. **Neuron initialization** — Create a `LIFPopulation` of $N$ neurons.
   Approximately 1% of neurons are given initial spikes (seeded
   deterministically).
5. **Scatter benchmark (push-based)** — For each of the $T$ timesteps:
   - Clear the synaptic current buffer.
   - Call `matrix.scatter(spikes, I_syn)` to propagate spikes through
     the weight matrix.
   - Add **external Poisson drive**: each neuron receives
     $\text{Poisson}(\lambda)$ external spikes per timestep, each weighted
     by $w_{\text{ext}}$ (default $\lambda = 15$, $w_{\text{ext}} = 1.5$,
     modelling background cortical input à la Brunel 2000).
   - Call `population.step(I_syn)` to integrate the LIF dynamics
     and detect new spikes.
6. **Gather benchmark (pull-based)** — The same simulation is replayed
   with identical spike patterns, but using `matrix.gather_all(spikes, I_syn)`
   instead of `scatter`.  This provides a symmetric timing comparison
   where CSC's column-oriented layout is the natural fit.
7. **Gather-only benchmark (pure matrix timing)** — When `--gather-only`
   is enabled, an additional benchmark loop runs `gather_all` in isolation:
   spikes are generated at a controlled fixed rate (from `--inject-rate`,
   or 5% by default) and only the matrix operation is timed — no LIF
   integration, no Poisson drive.  This eliminates the LIF overhead that
   otherwise dominates the measurement (999/1000 timesteps are LIF-only
   when the network is nearly silent) and directly measures CSC's
   column-indexed access advantage.
8. **Measurement** — Record scatter wall-clock time, gather wall-clock time,
   gather-only wall-clock time, peak RSS (from `/proc/self/status` VmHWM),
   and cumulative spike count.
9. **Robust statistics** — Apply IQR-based outlier rejection to scatter,
   gather, and gather-only trial timings before computing mean and standard
   deviation.  Report median times alongside mean/std for robustness.

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
| Scatter throughput (edges/ms) | `spikes_per_step × avg_out_degree × timesteps / scatter_time` | Direct measure of push-based spike-propagation efficiency per format |
| Gather throughput (edges/ms) | `spikes_per_step × avg_out_degree × timesteps / gather_time` | Direct measure of pull-based spike-propagation efficiency per format |
| Bytes per spike | `matrix_bytes / total_spikes` | Memory cost per propagated spike event |
| Median time (ms) | Median of per-trial scatter times | Robust central tendency, unaffected by outliers |
| Outliers removed | Count of trials rejected by IQR filter | Indicates measurement reliability |
| Cache ratio L1d | `matrix_bytes / L1d_size` | >1 means the matrix spills out of L1 |
| Cache ratio L2 | `matrix_bytes / L2_size` | >1 means the matrix spills out of L2 |
| Cache ratio L3 | `matrix_bytes / L3_size` | >1 means the matrix must fetch from DRAM |

### External Poisson Drive

To sustain realistic network activity throughout the simulation, each
neuron receives external Poisson-distributed input at every timestep:

$$I_{\text{ext},i}(t) = w_{\text{ext}} \times \text{Poisson}(\lambda)$$

With default parameters $\lambda = 15$, $w_{\text{ext}} = 1.5$, the mean
external current is $\bar{I}_{\text{ext}} = 22.5$ mV, which is near the
LIF rheobase of 15 mV ($V_{\text{thresh}} - V_{\text{rest}}$).  This
produces a sustained firing rate of approximately 4% across all topologies
and sizes.  The Poisson rate can be adjusted via `--poisson-rate` and
`--poisson-weight` to control network activity.

### Robust Statistics (IQR Outlier Rejection)

Long-running benchmark sweeps are susceptible to occasional system
interference (OS scheduling, swapping, thermal throttling) that can
produce single-trial times 100–1000× above the true value.  Because the
arithmetic mean is not robust to such outliers, a single bad trial can
corrupt the reported timing for an entire configuration.

**Method:** Before computing mean and standard deviation, both scatter
and gather trial-time vectors are filtered using the Interquartile Range
(IQR) method:

1. Sort the trial times and compute $Q_1$ (25th percentile) and $Q_3$
   (75th percentile).
2. Compute $\text{IQR} = Q_3 - Q_1$.
3. Reject any trial outside $[Q_1 - 1.5 \cdot \text{IQR},\; Q_3 + 1.5 \cdot \text{IQR}]$.
4. Compute mean and standard deviation from the remaining trials only.

Additionally:

- **Bessel's correction** is applied when computing standard deviation
  (dividing by $n - 1$ instead of $n$), yielding an unbiased sample
  estimator.
- **Median time** is computed from the *full* (pre-rejection) trial set
  and reported alongside mean/std as a second robust central tendency
  measure.
- The CSV column `outliers_removed` records how many trials were
  rejected per configuration so data quality can be audited.

### Spike-Rate Sweeps

The `--sweep-rates` option enables characterisation of scatter/gather cost
as a function of network activity.  By varying the Poisson rate (e.g.,
`--sweep-rates "5 10 15 20 25 30"`), the benchmark measures how each
format's throughput scales with the fraction of active neurons.

### Per-Process RSS (`--subprocess`)

By default, `VmHWM` is a process-global high-water mark, so sequential
configurations inherit the largest allocation from earlier runs.  The
`--subprocess` flag forks a separate child process for each configuration
via `popen()`.  Each child runs `--single-config` mode, producing a CSV
row on stdout, then exits — ensuring that `peak_rss_kb` reflects only
that configuration's memory footprint.

---

## Network Activity Controls

Three mechanisms are provided to ensure the network sustains meaningful
spiking activity throughout the benchmark, addressing the silent-network
problem where weight normalisation suppresses firing after the initial seed.

### Background Current (`--bg-current`)

A constant DC current $I_{\text{bg}}$ is injected into every neuron at each
timestep, added to the synaptic current buffer after scatter/gather and
Poisson drive:

$$
I_{\text{syn},i}(t) \leftarrow I_{\text{syn},i}(t) + I_{\text{bg}}
$$

**Recommended value:** `--bg-current 14.0` places the equilibrium potential
at $V_{\text{eq}} = V_{\text{rest}} + 14 = -51$ mV, just above $V_{\text{thresh}}$,
producing a sustained ~5–8% firing rate.  This is the simplest way to
keep the network active while preserving realistic LIF dynamics.

**Expected outcomes:**
- Consistent non-zero `total_spikes` across all configurations
- Valid `scatter_throughput`, `gather_throughput`, and `bytes_per_spike` metrics
- Meaningful activity-dependent differences between formats (CSR's spike-
  proportional cost becomes measurable)

### Controlled Spike Injection (`--inject-rate`)

When set to a value in $(0, 1]$, each timestep's spikes are **overridden**
with a uniformly random subset of neurons at the specified fraction.  The
LIF integration still runs (preserving its compute/cache overhead) but its
threshold crossings are ignored.

This decouples the sparse-matrix workload from network dynamics, enabling:
- **Cost-vs-activity curves**: measure scatter/gather time at 1%, 5%, 10%,
  25% spike rates to find format crossover points
- **Reproducible comparisons**: identical spike counts across formats and
  topologies, isolating the access-pattern effect

**Example:** `--inject-rate 0.05` produces exactly ~5% of neurons spiking
per timestep regardless of topology, density, or Poisson drive settings.

### Dedicated Gather-Only Benchmark (`--gather-only`)

Adds a separate timed benchmark loop that measures `gather_all()` in
isolation.  In this loop:

1. Spikes are generated at a fixed rate (from `--inject-rate`, or 5%
   by default).
2. Only `matrix.gather_all(spikes, I_syn)` is called and timed — no LIF
   integration, no Poisson drive, no scatter.
3. Results are reported as `gather_only_mean_time_ms`, `gather_only_std_time_ms`,
   `gather_only_median_time_ms`, and `gather_only_throughput`.

**Purpose:** The standard scatter and gather benchmarks include full LIF
simulation in the timing loop.  When the network is mostly silent, 999 of
1000 timesteps are pure LIF updates (identical across formats), drowning
out format-specific differences.  The gather-only benchmark eliminates this
confound and directly tests CSC's column-indexed access pattern advantage
against CSR's row-indexed layout.

**Expected outcomes:**
- CSC should outperform CSR on gather-only (column-indexed access is CSC's
  natural strength)
- ELL should show consistent gather-only timing (fixed $K_{\max}$ per row)
- COO should be slowest (full-matrix scan regardless of spike count)
- Performance differences between formats will be more pronounced than in
  the mixed scatter+LIF benchmark

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

| Dimension | Default values | CLI override |
|-----------|----------------|---------------|
| Network size $N$ | 1000, 5000, 10 000 | `--sweep-sizes "1000 5000 10000"` |
| Connection density $p$ | 0.01, 0.05, 0.1 | `--sweep-densities "0.01 0.05 0.1"` |
| Poisson rate $\lambda$ | 15.0 (single) | `--sweep-rates "5 10 15 20 25 30"` |
| Topology | er, fi, ba, ws | (all four) |
| Format | coo, csr, csc, ell | (all four) |

Each configuration is run with 10 trials (configurable via `--trials`);
the output CSV contains one row per configuration with mean, std, and
median across trials, after IQR-based outlier rejection.

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
| `mean_time_ms` | float | Mean wall-clock time across trials (after outlier rejection) |
| `std_time_ms` | float | Sample standard deviation of time (Bessel-corrected, after outlier rejection) |
| `median_time_ms` | float | Median scatter trial time (computed before outlier rejection) |
| `peak_rss_kb` | int | Peak resident set size in kilobytes |
| `total_spikes` | int | Mean total spike events per trial |
| `spikes_per_step` | float | Average spikes per timestep |
| `memory_bytes` | int | Sparse matrix memory footprint |
| `nnz` | int | Number of non-zero entries |
| `effective_bw_gbps` | float | Effective memory bandwidth (GB/s) |
| `scatter_throughput_edges_per_ms` | float | Edges scattered per millisecond |
| `bytes_per_spike` | float | Matrix bytes per propagated spike |
| `gather_mean_time_ms` | float | Mean gather trial time (ms, after outlier rejection) |
| `gather_std_time_ms` | float | Sample std of gather trial time (ms, Bessel-corrected) |
| `gather_median_time_ms` | float | Median gather trial time (ms) |
| `gather_throughput_edges_per_ms` | float | Edges gathered per millisecond |
| `cache_ratio_L1` | float | Matrix size / L1d cache size |
| `cache_ratio_L2` | float | Matrix size / L2 cache size |
| `cache_ratio_L3` | float | Matrix size / L3 cache size |
| `poisson_rate` | float | External Poisson drive rate used |
| `outliers_removed` | int | Number of scatter trials rejected by IQR filter |
| `gather_only_mean_time_ms` | float | Mean gather-only trial time (ms, after outlier rejection) |
| `gather_only_std_time_ms` | float | Std of gather-only trial time (ms) |
| `gather_only_median_time_ms` | float | Median gather-only trial time (ms) |
| `gather_only_throughput_edges_per_ms` | float | Edges gathered per ms (gather-only benchmark) |
| `background_current` | float | Background DC current injected per neuron per step (mV) |
| `inject_spike_rate` | float | Controlled spike injection rate (0 = LIF dynamics) |

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
    std::string format   = "csr";   // coo, csr, csc, ell
    std::string topology = "er";    // er, fi, ba, ws
    int         N        = 1000;
    double      density  = 0.05;
    int         timesteps = 1000;
    int         trials    = 10;
    unsigned    seed      = 42;

    // External Poisson drive
    double      poisson_rate      = 15.0;  // Mean external spikes/neuron/step
    double      poisson_weight    = 1.5;   // Weight (mV) per external spike
    // Recurrent weight normalisation: w = coupling_strength / sqrt(K_avg)
    double      coupling_strength = 2.0;

    // Background current injected into every neuron each timestep (mV).
    // Set to ~14.0 to sustain near-threshold activity.
    double      background_current = 0.0;

    // Controlled spike injection: fixed fraction of neurons spike each step.
    // Overrides LIF dynamics when > 0. Value in [0,1].
    double      inject_spike_rate = 0.0;

    // Run a dedicated gather-only benchmark (pure matrix timing).
    bool        gather_only_benchmark = false;
};
```

### Result

```cpp
struct BenchmarkResult {
    std::string format, topology;
    int         N, timesteps, trials;
    double      density;
    double      mean_time_ms, std_time_ms;     // Scatter timing (after outlier rejection)
    double      median_time_ms;                // Scatter median (pre-rejection)
    double      gather_median_time_ms;         // Gather median (pre-rejection)
    int         outliers_removed;              // Trials rejected by IQR filter
    long        peak_rss_kb;
    long        total_spikes;
    double      spikes_per_step;
    size_t      memory_bytes, nnz;

    // Scatter metrics
    double      effective_bw_gbps;
    double      scatter_throughput;             // edges/ms
    double      bytes_per_spike;

    // Gather metrics (after outlier rejection)
    double      gather_mean_time_ms, gather_std_time_ms;
    double      gather_throughput;              // edges/ms

    // Cache ratios
    double      matrix_cache_ratio_L1, matrix_cache_ratio_L2, matrix_cache_ratio_L3;

    // Drive parameters
    double      poisson_rate;

    // Gather-only benchmark results
    double      gather_only_mean_time_ms, gather_only_std_time_ms;
    double      gather_only_median_time_ms;
    double      gather_only_throughput;         // edges/ms

    // Background / injection parameters
    double      background_current;
    double      inject_spike_rate;
};
```

### Core Functions

```cpp
BenchmarkResult run_benchmark(const BenchmarkConfig& config);
long get_peak_rss_kb();

void write_csv_header(const std::string& filename);
void append_csv_row(const std::string& filename, const BenchmarkResult& r);
```

### Files

| File | Purpose |
|------|---------|
| `include/benchmark.h` | Config/Result structs, function declarations |
| `src/benchmark.cpp`   | Benchmark loop, RSS measurement, CSV I/O |
| `src/main.cpp`        | CLI argument parser, single-run and sweep modes |
