#pragma once

#include "sparse_matrix.h"
#include <string>
#include <vector>

/// CPU cache hierarchy information (sizes in bytes, 0 if unavailable).
struct CacheInfo {
    long L1d_bytes = 0;
    long L1i_bytes = 0;
    long L2_bytes  = 0;
    long L3_bytes  = 0;
};

/// Configuration for a single benchmark run.
struct BenchmarkConfig {
    std::string format   = "csr";     // coo, csr, csc, ell
    std::string topology = "er";      // er, fi, ba, ws
    int         N        = 1000;      // number of neurons
    double      density  = 0.05;      // connection probability / degree param
    int         timesteps = 1000;
    int         trials    = 10;
    unsigned    seed      = 42;

    // External Poisson drive — models background cortical input.
    double      poisson_rate      = 15.0;  // Mean external spikes per neuron per step
    double      poisson_weight    = 1.5;   // Weight (mV) per external spike
    // Recurrent weight normalisation: w = coupling_strength / sqrt(K_avg).
    double      coupling_strength = 2.0;

    // Background current injected into every neuron each timestep (mV).
    // Set to ~14.0 to sustain near-threshold activity (V_rest + I_bg ≈ V_thresh).
    double      background_current = 0.0;

    // Controlled spike injection: if > 0, inject spikes at this fixed
    // fraction of neurons per timestep (independent of LIF dynamics).
    // Value in [0,1], e.g. 0.05 = 5% of neurons spike each step.
    double      inject_spike_rate = 0.0;

    // Run a dedicated gather-only benchmark (separate from scatter).
    bool        gather_only_benchmark = false;

    // Path to a NEST-exported CSV connectivity file.  When non-empty the
    // topology generator is bypassed and the matrix is loaded from this file.
    std::string nest_csv;
};

/// Results from a single benchmark configuration.
struct BenchmarkResult {
    std::string format;
    std::string topology;
    int         N              = 0;
    double      density        = 0.0;
    int         timesteps      = 0;
    int         trials         = 0;
    double      mean_time_ms   = 0.0;
    double      std_time_ms    = 0.0;
    long        peak_rss_kb    = 0;
    long        total_spikes   = 0;
    double      spikes_per_step = 0.0;
    size_t      memory_bytes   = 0;
    size_t      nnz            = 0;

    // --- Robust statistics ---
    double      median_time_ms = 0.0;         // Median scatter trial time
    double      gather_median_time_ms = 0.0;  // Median gather trial time
    int         outliers_removed = 0;         // Number of outlier trials removed

    // --- Scatter metrics ---
    double      effective_bw_gbps  = 0.0;  // Effective bandwidth (GB/s)
    double      scatter_throughput = 0.0;  // Edges processed per ms
    double      bytes_per_spike    = 0.0;  // Memory cost per propagated spike

    // --- Gather metrics ---
    double      gather_mean_time_ms = 0.0; // Mean gather trial time (ms)
    double      gather_std_time_ms  = 0.0; // Std of gather trial time (ms)
    double      gather_throughput   = 0.0; // Edges processed per ms (gather)

    // --- Cache ratios ---
    double      matrix_cache_ratio_L1 = 0.0; // matrix_bytes / L1d size
    double      matrix_cache_ratio_L2 = 0.0; // matrix_bytes / L2 size
    double      matrix_cache_ratio_L3 = 0.0; // matrix_bytes / L3 size

    // --- Drive parameters (recorded for spike-rate sweep analysis) ---
    double      poisson_rate   = 0.0;

    // --- Gather-only benchmark results ---
    double      gather_only_mean_time_ms = 0.0;
    double      gather_only_std_time_ms  = 0.0;
    double      gather_only_median_time_ms = 0.0;
    double      gather_only_throughput   = 0.0;

    // --- Background / injection parameters ---
    double      background_current = 0.0;
    double      inject_spike_rate  = 0.0;
};

/// Run a single benchmark configuration and return aggregated results.
BenchmarkResult run_benchmark(const BenchmarkConfig& config);

/// Build a SparseMatrix of the requested format from COO triplets.
std::unique_ptr<SparseMatrix> build_matrix(const std::string& format,
                                           const COOTriplets& triplets);

/// Read peak resident set size (VmHWM) from /proc/self/status (Linux).
/// Returns value in KB, or -1 if unavailable.
long get_peak_rss_kb();

/// Detect CPU cache hierarchy from /sys/devices/system/cpu/cpu0/cache/.
CacheInfo detect_cache_info();

/// Print cache info to stdout.
void print_cache_info(const CacheInfo& info);

/// Write a CSV header line for benchmark results.
void write_csv_header(const std::string& filename);

/// Append a BenchmarkResult row to a CSV file.
void append_csv_row(const std::string& filename, const BenchmarkResult& result);

/// Write cache info to a separate CSV file.
void write_cache_info_csv(const std::string& filename, const CacheInfo& info);
