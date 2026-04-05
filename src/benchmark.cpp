// Benchmark harness for sparse matrix spike-propagation formats.
//
// Usage:
//   ./benchmark [--N <size>] [--p <density>] [--topology <er|fi|ba|ws>]
//               [--timesteps <T>] [--trials <K>] [--output <csv_file>]
//
// Defaults: N=1000, p=0.05, topology=er, T=1000, K=10, output=results.csv

#include "coo_matrix.hpp"
#include "csr_matrix.hpp"
#include "csc_matrix.hpp"
#include "ell_matrix.hpp"
#include "lif_neuron.hpp"
#include "network.hpp"
#include "sparse_matrix.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/resource.h>
static long peak_rss_kb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  // kB on Linux
}
#else
static long peak_rss_kb() { return -1; }
#endif

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ── CLI helpers ───────────────────────────────────────────────────────────────

static std::string arg_value(int argc, char** argv, const std::string& flag,
                              const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == flag) return argv[i + 1];
    }
    return def;
}

// ── Run one trial: T timesteps with the given matrix and return elapsed ms ───

static double run_trial(SparseMatrix& mat, LifNeuronPop& pop, int T,
                         bool use_scatter) {
    const std::size_t N = pop.size();
    std::vector<double> I_syn(N, 0.0);

    pop.reset(1234);
    auto t0 = Clock::now();
    for (int t = 0; t < T; ++t) {
        std::fill(I_syn.begin(), I_syn.end(), 0.0);
        std::vector<bool> spikes = pop.step(I_syn);   // intrinsic drive
        // Spike propagation through synaptic matrix.
        std::fill(I_syn.begin(), I_syn.end(), 0.0);
        if (use_scatter) {
            mat.scatter(spikes, I_syn);
        } else {
            mat.gather(spikes, I_syn);
        }
    }
    auto t1 = Clock::now();
    return Ms(t1 - t0).count();
}

// ── statistics ────────────────────────────────────────────────────────────────

static void stats(const std::vector<double>& v, double& mean, double& sd) {
    mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    sd = std::sqrt(sq / v.size());
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const std::size_t N    = std::stoul(arg_value(argc, argv, "--N",         "1000"));
    const double      p    = std::stod (arg_value(argc, argv, "--p",         "0.05"));
    const std::string topo = arg_value(argc, argv, "--topology",  "er");
    const int         T    = std::stoi (arg_value(argc, argv, "--timesteps", "1000"));
    const int         K    = std::stoi (arg_value(argc, argv, "--trials",    "10"));
    const std::string csv  = arg_value(argc, argv, "--output",    "results.csv");

    std::cout << "=== Spike-Propagation Benchmark ===\n"
              << "  N=" << N << "  p=" << p << "  topology=" << topo
              << "  T=" << T << "  trials=" << K << "\n\n";

    // ── Generate connectivity edges ───────────────────────────────────────────
    std::vector<Edge> edges;
    if (topo == "er") {
        edges = erdos_renyi(N, p, 1.0, 0);
    } else if (topo == "fi") {
        std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(p * N));
        edges = fixed_indegree(N, k, 1.0, 0);
    } else if (topo == "ba") {
        std::size_t m = std::max<std::size_t>(1, static_cast<std::size_t>(p * N));
        edges = barabasi_albert(N, m, 1.0, 0);
    } else if (topo == "ws") {
        std::size_t k = std::max<std::size_t>(2, static_cast<std::size_t>(p * N));
        if (k % 2 != 0) ++k;
        edges = watts_strogatz(N, k, 0.1, 1.0, 0);
    } else {
        std::cerr << "Unknown topology '" << topo << "'. Use er|fi|ba|ws.\n";
        return 1;
    }

    // Separate into COO components.
    std::vector<std::size_t> ri, ci;
    std::vector<double>      wv;
    ri.reserve(edges.size());
    ci.reserve(edges.size());
    wv.reserve(edges.size());
    for (const auto& e : edges) {
        ri.push_back(e.src);
        ci.push_back(e.dst);
        wv.push_back(e.weight);
    }

    std::cout << "  edges=" << edges.size()
              << "  density=" << static_cast<double>(edges.size()) / (N * N) << "\n\n";

    // ── Build matrices ────────────────────────────────────────────────────────
    CooMatrix coo(N, N);
    for (std::size_t k = 0; k < ri.size(); ++k) coo.add_entry(ri[k], ci[k], wv[k]);
    auto csr = CsrMatrix::from_coo(N, N, ri, ci, wv);
    auto csc = CscMatrix::from_coo(N, N, ri, ci, wv);
    auto ell = EllMatrix::from_coo(N, N, ri, ci, wv);

    std::cout << "  Memory (bytes):\n"
              << "    COO: " << coo.memory_bytes() << "\n"
              << "    CSR: " << csr.memory_bytes() << "\n"
              << "    CSC: " << csc.memory_bytes() << "\n"
              << "    ELL: " << ell.memory_bytes()
              << "  (max_cols=" << ell.max_cols() << ")\n\n";

    // ── LIF population ────────────────────────────────────────────────────────
    LifNeuronPop::Params lif_params;
    LifNeuronPop pop(N, lif_params);

    // ── Benchmark loop ────────────────────────────────────────────────────────
    struct FormatEntry {
        std::string  name;
        SparseMatrix* mat;
        bool         scatter_mode;
    };

    std::vector<FormatEntry> formats = {
        {"COO_scatter", &coo, true},
        {"COO_gather",  &coo, false},
        {"CSR_scatter", &csr, true},
        {"CSR_gather",  &csr, false},
        {"CSC_scatter", &csc, true},
        {"CSC_gather",  &csc, false},
        {"ELL_scatter", &ell, true},
        {"ELL_gather",  &ell, false},
    };

    std::ofstream out(csv);
    out << "format,N,p,topology,T,trial,elapsed_ms,peak_rss_kb,mem_bytes\n";

    for (auto& fmt : formats) {
        std::vector<double> times(K);
        for (int trial = 0; trial < K; ++trial) {
            times[trial] = run_trial(*fmt.mat, pop, T, fmt.scatter_mode);
        }
        double mean, sd;
        stats(times, mean, sd);

        long rss = peak_rss_kb();
        std::size_t mem = fmt.mat->memory_bytes();

        std::printf("  %-15s  mean=%8.2f ms  sd=%7.2f ms  mem=%zu B  rss=%ld kB\n",
                    fmt.name.c_str(), mean, sd, mem, rss);

        for (int trial = 0; trial < K; ++trial) {
            out << fmt.name << ',' << N << ',' << p << ','
                << topo << ',' << T << ',' << trial << ','
                << times[trial] << ',' << rss << ',' << mem << '\n';
        }
    }

    std::cout << "\nResults written to " << csv << "\n";
    return 0;
}
