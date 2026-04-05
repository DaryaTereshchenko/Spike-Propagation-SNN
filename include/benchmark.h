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

    // --- Derived metrics ---
    double      effective_bw_gbps  = 0.0;  // Effective bandwidth (GB/s)
    double      scatter_throughput = 0.0;  // Edges processed per ms
    double      bytes_per_spike    = 0.0;  // Memory cost per propagated spike
    double      matrix_cache_ratio_L1 = 0.0; // matrix_bytes / L1d size
    double      matrix_cache_ratio_L2 = 0.0; // matrix_bytes / L2 size
    double      matrix_cache_ratio_L3 = 0.0; // matrix_bytes / L3 size
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
