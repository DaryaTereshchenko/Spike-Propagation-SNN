#include "benchmark.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal argument parser
// ---------------------------------------------------------------------------
static std::string get_arg(int argc, char* argv[], const std::string& flag,
                           const std::string& default_val = "")
{
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == flag) return argv[i + 1];
    }
    return default_val;
}

static bool has_flag(int argc, char* argv[], const std::string& flag)
{
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------
static void print_result(const BenchmarkResult& r)
{
    std::cout << "  Format:          " << r.format          << "\n"
              << "  Topology:        " << r.topology        << "\n"
              << "  N:               " << r.N               << "\n"
              << "  Density/param:   " << r.density         << "\n"
              << "  Timesteps:       " << r.timesteps       << "\n"
              << "  Trials:          " << r.trials          << "\n"
              << "  NNZ:             " << r.nnz             << "\n"
              << "  Matrix memory:   " << r.memory_bytes    << " bytes\n"
              << "  Mean time:       " << r.mean_time_ms    << " ms\n"
              << "  Std time:        " << r.std_time_ms     << " ms\n"
              << "  Peak RSS:        " << r.peak_rss_kb     << " kB\n"
              << "  Avg spikes/step: " << r.spikes_per_step << "\n"
              << "  Total spikes:    " << r.total_spikes    << "\n"
              << "  --- Derived metrics ---\n"
              << "  Eff. bandwidth:  " << r.effective_bw_gbps  << " GB/s\n"
              << "  Scatter thruput: " << r.scatter_throughput << " edges/ms\n"
              << "  Bytes/spike:     " << r.bytes_per_spike    << "\n"
              << "  Cache ratio L1d: " << r.matrix_cache_ratio_L1 << "x\n"
              << "  Cache ratio L2:  " << r.matrix_cache_ratio_L2 << "x\n"
              << "  Cache ratio L3:  " << r.matrix_cache_ratio_L3 << "x\n"
              << std::endl;
}

static void print_usage()
{
    std::cerr
        << "Usage: spike_benchmark [options]\n"
        << "\n"
        << "Single-run options:\n"
        << "  --format    {coo|csr|csc|ell}    Sparse matrix format   (default: csr)\n"
        << "  --topology  {er|fi|ba|ws}        Network topology       (default: er)\n"
        << "  --size      N                    Number of neurons      (default: 1000)\n"
        << "  --density   p                    Density / degree param (default: 0.05)\n"
        << "  --timesteps T                    Simulation timesteps   (default: 1000)\n"
        << "  --trials    R                    Repeat count           (default: 10)\n"
        << "  --seed      S                    Random seed            (default: 42)\n"
        << "\n"
        << "Output:\n"
        << "  --output-csv FILE                Write results as CSV\n"
        << "\n"
        << "Sweep mode:\n"
        << "  --sweep                          Run full parameter sweep\n"
        << "  --sweep-sizes   \"1000 5000 ...\"  Sizes to sweep        (default: 1000 5000 10000)\n"
        << "  --sweep-densities \"0.01 0.05 ..\" Densities to sweep    (default: 0.01 0.05 0.1)\n"
        << "\n"
        << "Other:\n"
        << "  --help                           Show this help message\n";
}

// Parse a space-separated list of doubles from a string.
static std::vector<double> parse_doubles(const std::string& s)
{
    std::vector<double> v;
    std::istringstream iss(s);
    double d;
    while (iss >> d) v.push_back(d);
    return v;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_usage();
        return 0;
    }

    const std::string output_csv = get_arg(argc, argv, "--output-csv");
    const std::string cache_csv  = get_arg(argc, argv, "--cache-csv");

    // Detect and print cache hierarchy at startup.
    CacheInfo cache_info = detect_cache_info();
    print_cache_info(cache_info);
    if (!cache_csv.empty()) {
        write_cache_info_csv(cache_csv, cache_info);
        std::cout << "Cache info written to " << cache_csv << "\n\n";
    }

    if (has_flag(argc, argv, "--sweep")) {
        // ---- Sweep mode ----
        auto sizes_str     = get_arg(argc, argv, "--sweep-sizes", "1000 5000 10000");
        auto densities_str = get_arg(argc, argv, "--sweep-densities", "0.01 0.05 0.1");
        int  timesteps     = std::stoi(get_arg(argc, argv, "--timesteps", "1000"));
        int  trials        = std::stoi(get_arg(argc, argv, "--trials", "10"));
        unsigned seed      = static_cast<unsigned>(std::stoul(get_arg(argc, argv, "--seed", "42")));

        auto sizes     = parse_doubles(sizes_str);
        auto densities = parse_doubles(densities_str);

        std::vector<std::string> formats    = {"coo", "csr", "csc", "ell"};
        std::vector<std::string> topologies = {"er", "fi", "ba", "ws"};

        if (!output_csv.empty()) {
            write_csv_header(output_csv);
        }

        int total = static_cast<int>(formats.size() * topologies.size()
                    * sizes.size() * densities.size());
        int done  = 0;

        for (auto& fmt : formats) {
            for (auto& topo : topologies) {
                for (double sz : sizes) {
                    for (double dens : densities) {
                        BenchmarkConfig cfg;
                        cfg.format    = fmt;
                        cfg.topology  = topo;
                        cfg.N         = static_cast<int>(sz);
                        cfg.density   = dens;
                        cfg.timesteps = timesteps;
                        cfg.trials    = trials;
                        cfg.seed      = seed;

                        ++done;
                        std::cout << "[" << done << "/" << total << "] "
                                  << fmt << " / " << topo
                                  << " / N=" << cfg.N
                                  << " / d=" << dens << " ..." << std::flush;

                        try {
                            auto result = run_benchmark(cfg);
                            std::cout << " " << result.mean_time_ms
                                      << " ms (±" << result.std_time_ms << ")\n";
                            print_result(result);
                            if (!output_csv.empty()) {
                                append_csv_row(output_csv, result);
                            }
                        } catch (const std::exception& e) {
                            std::cout << " SKIPPED: " << e.what() << "\n";
                        }
                    }
                }
            }
        }

    } else {
        // ---- Single run mode ----
        BenchmarkConfig cfg;
        cfg.format    = get_arg(argc, argv, "--format",    "csr");
        cfg.topology  = get_arg(argc, argv, "--topology",  "er");
        cfg.N         = std::stoi(get_arg(argc, argv, "--size",      "1000"));
        cfg.density   = std::stod(get_arg(argc, argv, "--density",   "0.05"));
        cfg.timesteps = std::stoi(get_arg(argc, argv, "--timesteps", "1000"));
        cfg.trials    = std::stoi(get_arg(argc, argv, "--trials",    "10"));
        cfg.seed      = static_cast<unsigned>(
                            std::stoul(get_arg(argc, argv, "--seed", "42")));

        std::cout << "Running benchmark...\n";
        auto result = run_benchmark(cfg);
        print_result(result);

        if (!output_csv.empty()) {
            write_csv_header(output_csv);
            append_csv_row(output_csv, result);
            std::cout << "Results written to " << output_csv << "\n";
        }
    }

    return 0;
}
