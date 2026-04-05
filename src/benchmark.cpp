#include "benchmark.h"
#include "coo_matrix.h"
#include "csr_matrix.h"
#include "csc_matrix.h"
#include "ell_matrix.h"
#include "topology.h"
#include "lif_neuron.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Helper: read VmHWM from /proc/self/status
// ---------------------------------------------------------------------------
long get_peak_rss_kb()
{
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return -1;

    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb;
            if (iss >> kb) return kb;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Cache info detection from /sys/devices/system/cpu/cpu0/cache/
// ---------------------------------------------------------------------------
static long read_sysfs_long(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return 0;
    std::string val;
    std::getline(f, val);
    if (val.empty()) return 0;

    // Parse values like "32K", "1024K", "8192K", or plain numbers
    long multiplier = 1;
    if (val.back() == 'K' || val.back() == 'k') {
        multiplier = 1024;
        val.pop_back();
    } else if (val.back() == 'M' || val.back() == 'm') {
        multiplier = 1024 * 1024;
        val.pop_back();
    }
    try {
        return std::stol(val) * multiplier;
    } catch (...) {
        return 0;
    }
}

static std::string read_sysfs_string(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string val;
    std::getline(f, val);
    return val;
}

CacheInfo detect_cache_info()
{
    CacheInfo info;
    const std::string base = "/sys/devices/system/cpu/cpu0/cache/";

    // Iterate over index0, index1, index2, index3, ...
    for (int i = 0; i < 10; ++i) {
        std::string idx_dir = base + "index" + std::to_string(i) + "/";
        std::string level_str = read_sysfs_string(idx_dir + "level");
        std::string type_str  = read_sysfs_string(idx_dir + "type");
        long size = read_sysfs_long(idx_dir + "size");

        if (level_str.empty()) break;
        int level = 0;
        try { level = std::stoi(level_str); } catch (...) { continue; }

        if (level == 1 && type_str == "Data")        info.L1d_bytes = size;
        else if (level == 1 && type_str == "Instruction") info.L1i_bytes = size;
        else if (level == 2)                         info.L2_bytes  = size;
        else if (level == 3)                         info.L3_bytes  = size;
    }

    return info;
}

void print_cache_info(const CacheInfo& info)
{
    auto fmt_kb = [](long bytes) -> std::string {
        if (bytes == 0) return "N/A";
        if (bytes >= 1024 * 1024)
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        return std::to_string(bytes / 1024) + " KB";
    };

    std::cout << "=== CPU Cache Hierarchy ===\n"
              << "  L1 Data:        " << fmt_kb(info.L1d_bytes) << "\n"
              << "  L1 Instruction: " << fmt_kb(info.L1i_bytes) << "\n"
              << "  L2 Unified:     " << fmt_kb(info.L2_bytes)  << "\n"
              << "  L3 Unified:     " << fmt_kb(info.L3_bytes)  << "\n"
              << std::endl;
}

void write_cache_info_csv(const std::string& filename, const CacheInfo& info)
{
    std::ofstream f(filename);
    f << "level,size_bytes\n"
      << "L1d," << info.L1d_bytes << "\n"
      << "L1i," << info.L1i_bytes << "\n"
      << "L2,"  << info.L2_bytes  << "\n"
      << "L3,"  << info.L3_bytes  << "\n";
}

// ---------------------------------------------------------------------------
// Build matrix
// ---------------------------------------------------------------------------
std::unique_ptr<SparseMatrix> build_matrix(const std::string& format,
                                           const COOTriplets& triplets)
{
    if (format == "coo") return std::make_unique<COOMatrix>(triplets);
    if (format == "csr") return std::make_unique<CSRMatrix>(triplets);
    if (format == "csc") return std::make_unique<CSCMatrix>(triplets);
    if (format == "ell") return std::make_unique<ELLMatrix>(triplets);
    throw std::invalid_argument("Unknown format: " + format);
}

// ---------------------------------------------------------------------------
// Run benchmark
// ---------------------------------------------------------------------------
BenchmarkResult run_benchmark(const BenchmarkConfig& config)
{
    // 1. Generate topology.
    COOTriplets triplets = generate_topology(config.topology, config.N,
                                             config.density, config.seed);

    // 2. Build sparse matrix.
    auto matrix = build_matrix(config.format, triplets);

    // 3. Create LIF population.
    LIFPopulation lif(config.N);

    const int N = config.N;
    std::vector<double> I_syn(N, 0.0);

    // Storage for per-trial timings and spike counts.
    std::vector<double> trial_times(config.trials);
    long total_spikes_all = 0;

    for (int trial = 0; trial < config.trials; ++trial) {
        lif.reset();
        long trial_spikes = 0;

        // Seed initial spikes: ~1% of neurons fire randomly.
        std::vector<int> spikes;
        {
            std::mt19937 init_rng(config.seed + trial);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < N; ++i) {
                if (dist(init_rng) < 0.01) spikes.push_back(i);
            }
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < config.timesteps; ++t) {
            // Clear synaptic input buffer.
            std::fill(I_syn.begin(), I_syn.end(), 0.0);

            // Scatter spikes from previous timestep.
            if (!spikes.empty()) {
                matrix->scatter(spikes, I_syn);
            }

            // Step LIF and collect new spikes for next iteration.
            spikes = lif.step(I_syn);
            trial_spikes += static_cast<long>(spikes.size());
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        trial_times[trial] = ms;
        total_spikes_all += trial_spikes;
    }

    // Compute mean and std of trial times.
    double sum  = std::accumulate(trial_times.begin(), trial_times.end(), 0.0);
    double mean = sum / config.trials;
    double sq_sum = 0.0;
    for (double t : trial_times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / config.trials);

    BenchmarkResult result;
    result.format        = config.format;
    result.topology      = config.topology;
    result.N             = config.N;
    result.density       = config.density;
    result.timesteps     = config.timesteps;
    result.trials        = config.trials;
    result.mean_time_ms  = mean;
    result.std_time_ms   = std_dev;
    result.peak_rss_kb   = get_peak_rss_kb();
    result.total_spikes  = total_spikes_all / config.trials;
    result.spikes_per_step = static_cast<double>(result.total_spikes)
                           / config.timesteps;
    result.memory_bytes  = matrix->memory_bytes();
    result.nnz           = matrix->num_nonzeros();

    // --- Compute derived metrics ---
    // Effective bandwidth: during scatter, each spike reads its row of the
    // matrix (pointer + column indices + values).  Approximate total bytes
    // read per trial: memory_bytes touched proportional to mean spikes.
    // Simplified: memory_bytes / mean_time gives upper-bound bandwidth.
    if (mean > 0.0) {
        double bytes_per_trial = static_cast<double>(result.memory_bytes);
        result.effective_bw_gbps = (bytes_per_trial / (mean * 1e-3)) / 1e9;
    }

    // Scatter throughput: total NNZ edges traversed per ms.
    // Each spike scatters to its outgoing edges; approximate as
    // (avg_spikes_per_step * avg_edges_per_neuron * timesteps) / mean_time.
    if (mean > 0.0 && result.N > 0) {
        double avg_out_degree = static_cast<double>(result.nnz) / result.N;
        double edges_traversed = result.spikes_per_step * avg_out_degree
                               * config.timesteps;
        result.scatter_throughput = edges_traversed / mean;
    }

    // Bytes per spike: matrix memory / total spikes propagated.
    if (result.total_spikes > 0) {
        result.bytes_per_spike = static_cast<double>(result.memory_bytes)
                               / result.total_spikes;
    }

    // Cache ratios: how many times does the matrix exceed each cache level?
    CacheInfo cache = detect_cache_info();
    double mb = static_cast<double>(result.memory_bytes);
    if (cache.L1d_bytes > 0) result.matrix_cache_ratio_L1 = mb / cache.L1d_bytes;
    if (cache.L2_bytes  > 0) result.matrix_cache_ratio_L2 = mb / cache.L2_bytes;
    if (cache.L3_bytes  > 0) result.matrix_cache_ratio_L3 = mb / cache.L3_bytes;

    return result;
}

// ---------------------------------------------------------------------------
// CSV I/O for results
// ---------------------------------------------------------------------------
void write_csv_header(const std::string& filename)
{
    std::ofstream f(filename);
    f << "format,topology,N,density,timesteps,trials,"
         "mean_time_ms,std_time_ms,peak_rss_kb,"
         "total_spikes,spikes_per_step,memory_bytes,nnz,"
         "effective_bw_gbps,scatter_throughput_edges_per_ms,"
         "bytes_per_spike,cache_ratio_L1,cache_ratio_L2,cache_ratio_L3\n";
}

void append_csv_row(const std::string& filename, const BenchmarkResult& r)
{
    std::ofstream f(filename, std::ios::app);
    f << r.format   << "," << r.topology << "," << r.N << ","
      << r.density  << "," << r.timesteps << "," << r.trials << ","
      << r.mean_time_ms  << "," << r.std_time_ms << ","
      << r.peak_rss_kb   << ","
      << r.total_spikes  << "," << r.spikes_per_step << ","
      << r.memory_bytes  << "," << r.nnz << ","
      << r.effective_bw_gbps << "," << r.scatter_throughput << ","
      << r.bytes_per_spike << ","
      << r.matrix_cache_ratio_L1 << ","
      << r.matrix_cache_ratio_L2 << ","
      << r.matrix_cache_ratio_L3 << "\n";
}
