#include "benchmark.h"
#include "coo_matrix.h"
#include "csr_matrix.h"
#include "csc_matrix.h"
#include "csv_io.h"
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
    // 1. Generate topology (or load from NEST CSV).
    COOTriplets triplets;
    if (!config.nest_csv.empty()) {
        triplets = load_coo_from_csv(config.nest_csv);
    } else {
        triplets = generate_topology(config.topology, config.N,
                                     config.density, config.seed);
    }

    // When loading from CSV, infer N from the matrix.
    const int N = config.nest_csv.empty() ? config.N : triplets.N;

    // Normalise recurrent weights: w = coupling_strength / sqrt(K_avg).
    // Ensures consistent network dynamics across all topologies.
    {
        double K_avg = static_cast<double>(triplets.nnz()) / N;
        double w = config.coupling_strength / std::sqrt(std::max(K_avg, 1.0));
        for (auto& v : triplets.vals) {
            v = w;
        }
    }

    // 2. Build sparse matrix.
    auto matrix = build_matrix(config.format, triplets);

    // 3. Create LIF population.
    LIFPopulation lif(N);

    std::vector<double> I_syn(N, 0.0);

    // Storage for per-trial timings and spike counts.
    std::vector<double> trial_times(config.trials);
    std::vector<double> gather_times(config.trials);
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

        // RNG for external Poisson drive.
        std::mt19937 ext_rng(config.seed + trial + 1000);
        std::poisson_distribution<int> poisson_dist(config.poisson_rate);

        // ---- Scatter benchmark (push-based spike delivery) ----
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < config.timesteps; ++t) {
            // Clear synaptic input buffer.
            std::fill(I_syn.begin(), I_syn.end(), 0.0);

            // Scatter spikes from previous timestep.
            if (!spikes.empty()) {
                matrix->scatter(spikes, I_syn);
            }

            // External Poisson drive: each neuron receives
            // Poisson-distributed external spikes per timestep,
            // modelling background cortical input (Brunel 2000).
            for (int i = 0; i < N; ++i) {
                I_syn[i] += poisson_dist(ext_rng) * config.poisson_weight;
            }

            // Background current: constant DC injection to sustain activity.
            if (config.background_current != 0.0) {
                for (int i = 0; i < N; ++i) {
                    I_syn[i] += config.background_current;
                }
            }

            // Step LIF and collect new spikes for next iteration.
            spikes = lif.step(I_syn);

            // Controlled spike injection: override LIF spikes with a fixed
            // fraction of randomly selected neurons (independent of dynamics).
            if (config.inject_spike_rate > 0.0) {
                spikes.clear();
                std::uniform_real_distribution<double> inj_dist(0.0, 1.0);
                for (int i = 0; i < N; ++i) {
                    if (inj_dist(ext_rng) < config.inject_spike_rate) {
                        spikes.push_back(i);
                    }
                }
            }

            trial_spikes += static_cast<long>(spikes.size());
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        trial_times[trial] = ms;
        total_spikes_all += trial_spikes;

        // ---- Gather benchmark (pull-based synaptic input) ----
        // Re-seed and replay the same simulation to get matching spike
        // patterns, but time gather_all instead of scatter.
        lif.reset();
        std::vector<int> g_spikes;
        {
            std::mt19937 init_rng(config.seed + trial);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < N; ++i) {
                if (dist(init_rng) < 0.01) g_spikes.push_back(i);
            }
        }
        std::mt19937 g_ext_rng(config.seed + trial + 1000);
        std::poisson_distribution<int> g_poisson_dist(config.poisson_rate);

        auto g0 = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < config.timesteps; ++t) {
            std::fill(I_syn.begin(), I_syn.end(), 0.0);

            if (!g_spikes.empty()) {
                matrix->gather_all(g_spikes, I_syn);
            }

            for (int i = 0; i < N; ++i) {
                I_syn[i] += g_poisson_dist(g_ext_rng) * config.poisson_weight;
            }

            // Background current (must match scatter loop).
            if (config.background_current != 0.0) {
                for (int i = 0; i < N; ++i) {
                    I_syn[i] += config.background_current;
                }
            }

            g_spikes = lif.step(I_syn);

            // Controlled spike injection (must match scatter loop).
            if (config.inject_spike_rate > 0.0) {
                g_spikes.clear();
                std::uniform_real_distribution<double> inj_dist(0.0, 1.0);
                for (int i = 0; i < N; ++i) {
                    if (inj_dist(g_ext_rng) < config.inject_spike_rate) {
                        g_spikes.push_back(i);
                    }
                }
            }
        }

        auto g1 = std::chrono::high_resolution_clock::now();
        gather_times[trial] = std::chrono::duration<double, std::milli>(g1 - g0).count();
    }

    // ---- Dedicated gather-only benchmark ----
    // Times gather_all in isolation, with controlled spike injection at a
    // fixed rate.  This measures pure column-indexed access performance
    // (CSC's natural advantage) without scatter/LIF overhead.
    std::vector<double> gather_only_times;
    if (config.gather_only_benchmark) {
        gather_only_times.resize(config.trials);
        // Use inject_spike_rate if set, otherwise default to 5% activity.
        double go_rate = (config.inject_spike_rate > 0.0)
                       ? config.inject_spike_rate : 0.05;

        for (int trial = 0; trial < config.trials; ++trial) {
            std::mt19937 go_rng(config.seed + trial + 2000);
            std::uniform_real_distribution<double> go_dist(0.0, 1.0);

            auto go_t0 = std::chrono::high_resolution_clock::now();

            for (int t = 0; t < config.timesteps; ++t) {
                // Generate spikes at fixed rate.
                std::vector<int> go_spikes;
                for (int i = 0; i < N; ++i) {
                    if (go_dist(go_rng) < go_rate) go_spikes.push_back(i);
                }

                // Pure gather: time only the matrix operation.
                std::fill(I_syn.begin(), I_syn.end(), 0.0);
                if (!go_spikes.empty()) {
                    matrix->gather_all(go_spikes, I_syn);
                }
            }

            auto go_t1 = std::chrono::high_resolution_clock::now();
            gather_only_times[trial] =
                std::chrono::duration<double, std::milli>(go_t1 - go_t0).count();
        }
    }

    // ---- IQR-based outlier rejection for robust statistics ----
    // Removes trials beyond Q1 - 1.5*IQR .. Q3 + 1.5*IQR, then computes
    // mean/std from the remaining ("trimmed") set.  This guards against
    // occasional system hiccups (scheduling, swapping) that can produce
    // single-trial times 100-1000x higher than the true value.
    auto reject_outliers = [](std::vector<double>& times) -> int {
        if (times.size() < 4) return 0; // Need at least 4 for meaningful IQR
        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        double Q1 = sorted[n / 4];
        double Q3 = sorted[3 * n / 4];
        double IQR = Q3 - Q1;
        double lo = Q1 - 1.5 * IQR;
        double hi = Q3 + 1.5 * IQR;
        std::vector<double> clean;
        clean.reserve(n);
        for (double t : times) {
            if (t >= lo && t <= hi) clean.push_back(t);
        }
        int removed = static_cast<int>(times.size() - clean.size());
        if (!clean.empty()) times = std::move(clean);
        return removed;
    };

    auto compute_median = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        if (n % 2 == 1) return v[n / 2];
        return (v[n / 2 - 1] + v[n / 2]) / 2.0;
    };

    // Compute medians before outlier removal (on full data).
    double scatter_median = compute_median(trial_times);
    double gather_median  = compute_median(gather_times);

    // Remove outliers.
    int scatter_outliers = reject_outliers(trial_times);
    int gather_outliers  = reject_outliers(gather_times);
    (void)gather_outliers;

    // Compute mean and std of scatter trial times (sample std, Bessel's correction).
    double sum  = std::accumulate(trial_times.begin(), trial_times.end(), 0.0);
    double mean = sum / static_cast<double>(trial_times.size());
    double sq_sum = 0.0;
    for (double t : trial_times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = (trial_times.size() > 1)
                   ? std::sqrt(sq_sum / (trial_times.size() - 1))
                   : 0.0;

    // Compute mean and std of gather trial times.
    double g_sum  = std::accumulate(gather_times.begin(), gather_times.end(), 0.0);
    double g_mean = g_sum / static_cast<double>(gather_times.size());
    double g_sq_sum = 0.0;
    for (double t : gather_times) {
        g_sq_sum += (t - g_mean) * (t - g_mean);
    }
    double g_std = (gather_times.size() > 1)
                 ? std::sqrt(g_sq_sum / (gather_times.size() - 1))
                 : 0.0;

    // Compute gather-only statistics.
    double go_mean = 0.0, go_std = 0.0, go_median = 0.0;
    if (!gather_only_times.empty()) {
        go_median = compute_median(gather_only_times);
        reject_outliers(gather_only_times);
        double go_sum = std::accumulate(gather_only_times.begin(),
                                        gather_only_times.end(), 0.0);
        go_mean = go_sum / static_cast<double>(gather_only_times.size());
        double go_sq = 0.0;
        for (double t : gather_only_times) go_sq += (t - go_mean) * (t - go_mean);
        go_std = (gather_only_times.size() > 1)
               ? std::sqrt(go_sq / (gather_only_times.size() - 1))
               : 0.0;
    }

    BenchmarkResult result;
    result.format        = config.format;
    result.topology      = config.nest_csv.empty() ? config.topology : "nest";
    result.N             = N;
    result.density       = config.nest_csv.empty() ? config.density
                           : static_cast<double>(triplets.nnz()) / (static_cast<double>(N) * N);
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
    result.poisson_rate  = config.poisson_rate;

    // --- Robust statistics ---
    result.median_time_ms        = scatter_median;
    result.gather_median_time_ms = gather_median;
    result.outliers_removed      = scatter_outliers;

    // --- Gather metrics ---
    result.gather_mean_time_ms = g_mean;
    result.gather_std_time_ms  = g_std;

    // --- Gather-only benchmark results ---
    result.gather_only_mean_time_ms   = go_mean;
    result.gather_only_std_time_ms    = go_std;
    result.gather_only_median_time_ms = go_median;

    // --- Config parameters ---
    result.background_current = config.background_current;
    result.inject_spike_rate  = config.inject_spike_rate;

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

    // Gather throughput: same edge counting, divided by gather time.
    if (g_mean > 0.0 && result.N > 0) {
        double avg_out_degree = static_cast<double>(result.nnz) / result.N;
        double edges_traversed = result.spikes_per_step * avg_out_degree
                               * config.timesteps;
        result.gather_throughput = edges_traversed / g_mean;
    }

    // Gather-only throughput: uses the controlled injection rate.
    if (go_mean > 0.0 && result.N > 0) {
        double go_rate = (config.inject_spike_rate > 0.0)
                       ? config.inject_spike_rate : 0.05;
        double avg_out_degree = static_cast<double>(result.nnz) / result.N;
        double go_edges = go_rate * result.N * avg_out_degree * config.timesteps;
        result.gather_only_throughput = go_edges / go_mean;
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
         "mean_time_ms,std_time_ms,median_time_ms,peak_rss_kb,"
         "total_spikes,spikes_per_step,memory_bytes,nnz,"
         "effective_bw_gbps,scatter_throughput_edges_per_ms,"
         "bytes_per_spike,"
         "gather_mean_time_ms,gather_std_time_ms,gather_median_time_ms,"
         "gather_throughput_edges_per_ms,"
         "cache_ratio_L1,cache_ratio_L2,cache_ratio_L3,"
         "poisson_rate,outliers_removed,"
         "gather_only_mean_time_ms,gather_only_std_time_ms,"
         "gather_only_median_time_ms,gather_only_throughput_edges_per_ms,"
         "background_current,inject_spike_rate\n";
}

void append_csv_row(const std::string& filename, const BenchmarkResult& r)
{
    std::ofstream f(filename, std::ios::app);
    f << r.format   << "," << r.topology << "," << r.N << ","
      << r.density  << "," << r.timesteps << "," << r.trials << ","
      << r.mean_time_ms  << "," << r.std_time_ms << ","
      << r.median_time_ms << ","
      << r.peak_rss_kb   << ","
      << r.total_spikes  << "," << r.spikes_per_step << ","
      << r.memory_bytes  << "," << r.nnz << ","
      << r.effective_bw_gbps << "," << r.scatter_throughput << ","
      << r.bytes_per_spike << ","
      << r.gather_mean_time_ms << "," << r.gather_std_time_ms << ","
      << r.gather_median_time_ms << ","
      << r.gather_throughput << ","
      << r.matrix_cache_ratio_L1 << ","
      << r.matrix_cache_ratio_L2 << ","
      << r.matrix_cache_ratio_L3 << ","
      << r.poisson_rate << ","
      << r.outliers_removed << ","
      << r.gather_only_mean_time_ms << ","
      << r.gather_only_std_time_ms << ","
      << r.gather_only_median_time_ms << ","
      << r.gather_only_throughput << ","
      << r.background_current << ","
      << r.inject_spike_rate << "\n";
}
