#include "benchmark.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
              << "  Poisson rate:    " << r.poisson_rate    << "\n"
              << "  Background I:    " << r.background_current << " mV\n"
              << "  Inject rate:     " << r.inject_spike_rate  << "\n"
              << "  Timesteps:       " << r.timesteps       << "\n"
              << "  Trials:          " << r.trials          << "\n"
              << "  Outliers removed:" << r.outliers_removed << "\n"
              << "  NNZ:             " << r.nnz             << "\n"
              << "  Matrix memory:   " << r.memory_bytes    << " bytes\n"
              << "  Mean time:       " << r.mean_time_ms    << " ms\n"
              << "  Median time:     " << r.median_time_ms  << " ms\n"
              << "  Std time:        " << r.std_time_ms     << " ms\n"
              << "  Peak RSS:        " << r.peak_rss_kb     << " kB\n"
              << "  Avg spikes/step: " << r.spikes_per_step << "\n"
              << "  Total spikes:    " << r.total_spikes    << "\n"
              << "  --- Scatter metrics ---\n"
              << "  Eff. bandwidth:  " << r.effective_bw_gbps  << " GB/s\n"
              << "  Scatter thruput: " << r.scatter_throughput << " edges/ms\n"
              << "  Bytes/spike:     " << r.bytes_per_spike    << "\n"
              << "  --- Gather metrics ---\n"
              << "  Gather time:     " << r.gather_mean_time_ms << " ms (±"
                                       << r.gather_std_time_ms  << ")\n"
              << "  Gather median:   " << r.gather_median_time_ms << " ms\n"
              << "  Gather thruput:  " << r.gather_throughput   << " edges/ms\n"
              << "  --- Gather-only benchmark ---\n"
              << "  GO time:         " << r.gather_only_mean_time_ms << " ms (±"
                                       << r.gather_only_std_time_ms  << ")\n"
              << "  GO median:       " << r.gather_only_median_time_ms << " ms\n"
              << "  GO throughput:   " << r.gather_only_throughput << " edges/ms\n"
              << "  --- Cache ratios ---\n"
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
        << "Network drive:\n"
        << "  --poisson-rate   R               External Poisson rate  (default: 15.0)\n"
        << "  --poisson-weight W               External spike weight  (default: 1.5)\n"
        << "  --coupling       G               Recurrent w=G/sqrt(K) (default: 2.0)\n"
        << "  --bg-current     I               Background DC current  (default: 0; ~14 for sustained activity)\n"
        << "  --inject-rate    F               Fixed spike injection rate [0,1] (default: 0 = use LIF)\n"
        << "\n"
        << "Benchmark modes:\n"
        << "  --gather-only                    Run dedicated gather-only benchmark\n"
        << "\n"
        << "Output:\n"
        << "  --output-csv FILE                Write results as CSV\n"
        << "\n"
        << "Sweep modes:\n"
        << "  --sweep                          Run full parameter sweep\n"
        << "  --sweep-sizes   \"1000 5000 ...\"  Sizes to sweep        (default: 1000 5000 10000)\n"
        << "  --sweep-densities \"0.01 0.05 ..\" Densities to sweep    (default: 0.01 0.05 0.1)\n"
        << "  --sweep-rates   \"5 10 15 20 30\"  Poisson rates to sweep (default: single rate)\n"
        << "  --subprocess                     Fork a child process per config for accurate RSS\n"
        << "\n"
        << "Internal (used by --subprocess):\n"
        << "  --single-config                  Run one config, print CSV row to stdout\n"
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

    // -------------------------------------------------------------------
    // Single-config mode: run ONE config and print CSV row to stdout.
    // Used internally by --subprocess sweep.
    // -------------------------------------------------------------------
    if (has_flag(argc, argv, "--single-config")) {
        BenchmarkConfig cfg;
        cfg.format    = get_arg(argc, argv, "--format",    "csr");
        cfg.topology  = get_arg(argc, argv, "--topology",  "er");
        cfg.N         = std::stoi(get_arg(argc, argv, "--size",      "1000"));
        cfg.density   = std::stod(get_arg(argc, argv, "--density",   "0.05"));
        cfg.timesteps = std::stoi(get_arg(argc, argv, "--timesteps", "1000"));
        cfg.trials    = std::stoi(get_arg(argc, argv, "--trials",    "10"));
        cfg.seed      = static_cast<unsigned>(
                            std::stoul(get_arg(argc, argv, "--seed", "42")));
        cfg.poisson_rate      = std::stod(get_arg(argc, argv, "--poisson-rate",   "15.0"));
        cfg.poisson_weight    = std::stod(get_arg(argc, argv, "--poisson-weight", "1.5"));
        cfg.coupling_strength = std::stod(get_arg(argc, argv, "--coupling",       "2.0"));
        cfg.background_current = std::stod(get_arg(argc, argv, "--bg-current",    "0.0"));
        cfg.inject_spike_rate  = std::stod(get_arg(argc, argv, "--inject-rate",   "0.0"));
        cfg.gather_only_benchmark = has_flag(argc, argv, "--gather-only");

        auto result = run_benchmark(cfg);

        // Print CSV row to stdout (no header — parent will write that).
        std::cout << result.format   << "," << result.topology << ","
                  << result.N        << "," << result.density  << ","
                  << result.timesteps << "," << result.trials  << ","
                  << result.mean_time_ms  << "," << result.std_time_ms << ","
                  << result.median_time_ms << ","
                  << result.peak_rss_kb   << ","
                  << result.total_spikes  << "," << result.spikes_per_step << ","
                  << result.memory_bytes  << "," << result.nnz << ","
                  << result.effective_bw_gbps << "," << result.scatter_throughput << ","
                  << result.bytes_per_spike << ","
                  << result.gather_mean_time_ms << "," << result.gather_std_time_ms << ","
                  << result.gather_median_time_ms << ","
                  << result.gather_throughput << ","
                  << result.matrix_cache_ratio_L1 << ","
                  << result.matrix_cache_ratio_L2 << ","
                  << result.matrix_cache_ratio_L3 << ","
                  << result.poisson_rate << ","
                  << result.outliers_removed << ","
                  << result.gather_only_mean_time_ms << ","
                  << result.gather_only_std_time_ms << ","
                  << result.gather_only_median_time_ms << ","
                  << result.gather_only_throughput << ","
                  << result.background_current << ","
                  << result.inject_spike_rate << "\n";

        return 0;
    }

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
        auto rates_str     = get_arg(argc, argv, "--sweep-rates", "");
        int  timesteps     = std::stoi(get_arg(argc, argv, "--timesteps", "1000"));
        int  trials        = std::stoi(get_arg(argc, argv, "--trials", "10"));
        unsigned seed      = static_cast<unsigned>(std::stoul(get_arg(argc, argv, "--seed", "42")));
        double poisson_rate   = std::stod(get_arg(argc, argv, "--poisson-rate",   "15.0"));
        double poisson_weight = std::stod(get_arg(argc, argv, "--poisson-weight", "1.5"));
        double coupling       = std::stod(get_arg(argc, argv, "--coupling",       "2.0"));
        double bg_current     = std::stod(get_arg(argc, argv, "--bg-current",     "0.0"));
        double inject_rate    = std::stod(get_arg(argc, argv, "--inject-rate",    "0.0"));
        bool gather_only      = has_flag(argc, argv, "--gather-only");
        bool use_subprocess   = has_flag(argc, argv, "--subprocess");

        auto sizes     = parse_doubles(sizes_str);
        auto densities = parse_doubles(densities_str);

        // Spike-rate sweep: if --sweep-rates is given, iterate over multiple
        // Poisson rates to characterise scatter/gather cost vs. activity.
        std::vector<double> rates;
        if (!rates_str.empty()) {
            rates = parse_doubles(rates_str);
        } else {
            rates.push_back(poisson_rate);
        }

        std::vector<std::string> formats    = {"coo", "csr", "csc", "ell"};
        std::vector<std::string> topologies = {"er", "fi", "ba", "ws"};

        if (!output_csv.empty()) {
            write_csv_header(output_csv);
        }

        int total = static_cast<int>(formats.size() * topologies.size()
                    * sizes.size() * densities.size() * rates.size());
        int done  = 0;

        // Resolve path to self for subprocess mode.
        std::string self_exe;
        if (use_subprocess) {
            // argv[0] may be relative; resolve it.
            self_exe = argv[0];
        }

        for (auto& fmt : formats) {
            for (auto& topo : topologies) {
                for (double sz : sizes) {
                    for (double dens : densities) {
                        for (double rate : rates) {
                            ++done;
                            std::cout << "[" << done << "/" << total << "] "
                                      << fmt << " / " << topo
                                      << " / N=" << static_cast<int>(sz)
                                      << " / d=" << dens;
                            if (rates.size() > 1) {
                                std::cout << " / rate=" << rate;
                            }
                            std::cout << " ..." << std::flush;

                            try {
                                BenchmarkResult result;

                                if (use_subprocess) {
                                    // Fork a child process for accurate RSS
                                    // measurement (VmHWM is process-global).
                                    std::ostringstream cmd;
                                    cmd << self_exe
                                        << " --single-config"
                                        << " --format "    << fmt
                                        << " --topology "  << topo
                                        << " --size "      << static_cast<int>(sz)
                                        << " --density "   << dens
                                        << " --timesteps " << timesteps
                                        << " --trials "    << trials
                                        << " --seed "      << seed
                                        << " --poisson-rate "   << rate
                                        << " --poisson-weight " << poisson_weight
                                        << " --coupling "       << coupling
                                        << " --bg-current "     << bg_current
                                        << " --inject-rate "    << inject_rate;
                                    if (gather_only) cmd << " --gather-only";

                                    FILE* pipe = popen(cmd.str().c_str(), "r");
                                    if (!pipe) throw std::runtime_error("popen failed");

                                    char buf[4096];
                                    std::string csv_line;
                                    while (fgets(buf, sizeof(buf), pipe)) {
                                        csv_line += buf;
                                    }
                                    int status = pclose(pipe);
                                    if (status != 0) {
                                        throw std::runtime_error("child exited with " + std::to_string(status));
                                    }

                                    // Parse CSV line back into result.
                                    // Remove trailing newline.
                                    while (!csv_line.empty() &&
                                           (csv_line.back() == '\n' || csv_line.back() == '\r')) {
                                        csv_line.pop_back();
                                    }

                                    // Append directly to output file and print.
                                    if (!output_csv.empty()) {
                                        std::ofstream f(output_csv, std::ios::app);
                                        f << csv_line << "\n";
                                    }

                                    // Extract mean_time and std_time for
                                    // console display (fields 7 and 8).
                                    std::istringstream iss(csv_line);
                                    std::string field;
                                    double mean_t = 0, std_t = 0;
                                    for (int fi = 0; fi < 8 && std::getline(iss, field, ','); ++fi) {
                                        if (fi == 6) mean_t = std::stod(field);
                                        if (fi == 7) std_t  = std::stod(field);
                                    }
                                    std::cout << " " << mean_t
                                              << " ms (±" << std_t << ")\n";

                                } else {
                                    // In-process mode.
                                    BenchmarkConfig cfg;
                                    cfg.format    = fmt;
                                    cfg.topology  = topo;
                                    cfg.N         = static_cast<int>(sz);
                                    cfg.density   = dens;
                                    cfg.timesteps = timesteps;
                                    cfg.trials    = trials;
                                    cfg.seed      = seed;
                                    cfg.poisson_rate      = rate;
                                    cfg.poisson_weight    = poisson_weight;
                                    cfg.coupling_strength = coupling;
                                    cfg.background_current = bg_current;
                                    cfg.inject_spike_rate  = inject_rate;
                                    cfg.gather_only_benchmark = gather_only;

                                    result = run_benchmark(cfg);
                                    std::cout << " " << result.mean_time_ms
                                              << " ms (±" << result.std_time_ms << ")\n";
                                    print_result(result);
                                    if (!output_csv.empty()) {
                                        append_csv_row(output_csv, result);
                                    }
                                }
                            } catch (const std::exception& e) {
                                std::cout << " SKIPPED: " << e.what() << "\n";
                            }
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
        cfg.poisson_rate      = std::stod(get_arg(argc, argv, "--poisson-rate",   "15.0"));
        cfg.poisson_weight    = std::stod(get_arg(argc, argv, "--poisson-weight", "1.5"));
        cfg.coupling_strength = std::stod(get_arg(argc, argv, "--coupling",       "2.0"));
        cfg.background_current = std::stod(get_arg(argc, argv, "--bg-current",    "0.0"));
        cfg.inject_spike_rate  = std::stod(get_arg(argc, argv, "--inject-rate",   "0.0"));
        cfg.gather_only_benchmark = has_flag(argc, argv, "--gather-only");

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
