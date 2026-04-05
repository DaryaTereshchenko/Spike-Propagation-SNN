#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "topology.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// Erdős–Rényi
// ---------------------------------------------------------------------------
TEST_CASE("Erdős–Rényi: edge count and no self-loops", "[topology][er]")
{
    const int N = 500;
    const double p = 0.05;
    auto coo = generate_erdos_renyi(N, p, 42);

    // No self-loops.
    for (size_t i = 0; i < coo.nnz(); ++i) {
        REQUIRE(coo.rows[i] != coo.cols[i]);
    }

    // Expected edges: N*(N-1)*p.  Allow 3-sigma tolerance.
    double expected = N * (N - 1) * p;
    double sigma    = std::sqrt(N * (N - 1) * p * (1 - p));
    double actual   = static_cast<double>(coo.nnz());

    REQUIRE(actual > expected - 3 * sigma);
    REQUIRE(actual < expected + 3 * sigma);
}

TEST_CASE("Erdős–Rényi: weights are normalized", "[topology][er]")
{
    const int N = 200;
    const double p = 0.1;
    auto coo = generate_erdos_renyi(N, p, 123);

    double expected_weight = 1.0 / std::sqrt(N * p);
    for (size_t i = 0; i < coo.nnz(); ++i) {
        REQUIRE_THAT(coo.vals[i],
                     Catch::Matchers::WithinRel(expected_weight, 1e-10));
    }
}

// ---------------------------------------------------------------------------
// Fixed in-degree
// ---------------------------------------------------------------------------
TEST_CASE("Fixed in-degree: each column has exactly K entries", "[topology][fi]")
{
    const int N = 300;
    const int K = 10;
    auto coo = generate_fixed_indegree(N, K, 42);

    // Total edges must be N * K.
    REQUIRE(coo.nnz() == static_cast<size_t>(N * K));

    // Count in-degree per column.
    std::vector<int> indeg(N, 0);
    for (size_t i = 0; i < coo.nnz(); ++i) {
        indeg[coo.cols[i]]++;
    }
    for (int j = 0; j < N; ++j) {
        REQUIRE(indeg[j] == K);
    }
}

TEST_CASE("Fixed in-degree: no self-loops", "[topology][fi]")
{
    const int N = 100;
    const int K = 5;
    auto coo = generate_fixed_indegree(N, K, 99);

    for (size_t i = 0; i < coo.nnz(); ++i) {
        REQUIRE(coo.rows[i] != coo.cols[i]);
    }
}

TEST_CASE("Fixed in-degree: no duplicate sources per target", "[topology][fi]")
{
    const int N = 100;
    const int K = 10;
    auto coo = generate_fixed_indegree(N, K, 77);

    // For each target, collect its sources and check uniqueness.
    std::vector<std::set<int>> sources(N);
    for (size_t i = 0; i < coo.nnz(); ++i) {
        sources[coo.cols[i]].insert(coo.rows[i]);
    }
    for (int j = 0; j < N; ++j) {
        REQUIRE(static_cast<int>(sources[j].size()) == K);
    }
}

// ---------------------------------------------------------------------------
// Barabási–Albert
// ---------------------------------------------------------------------------
TEST_CASE("Barabási–Albert: basic structure", "[topology][ba]")
{
    const int N = 200;
    const int m = 3;
    auto coo = generate_barabasi_albert(N, m, 42);

    // Should have edges and all indices in range.
    REQUIRE(coo.nnz() > 0);
    for (size_t i = 0; i < coo.nnz(); ++i) {
        REQUIRE(coo.rows[i] >= 0);
        REQUIRE(coo.rows[i] <  N);
        REQUIRE(coo.cols[i] >= 0);
        REQUIRE(coo.cols[i] <  N);
        REQUIRE(coo.rows[i] != coo.cols[i]);
    }
}

TEST_CASE("Barabási–Albert: degree distribution is heavy-tailed", "[topology][ba]")
{
    const int N = 1000;
    const int m = 2;
    auto coo = generate_barabasi_albert(N, m, 42);

    // Compute degree (sum of in + out since it's undirected stored as directed).
    std::vector<int> degree(N, 0);
    for (size_t i = 0; i < coo.nnz(); ++i) {
        degree[coo.rows[i]]++;
    }

    // Maximum degree should be significantly larger than average.
    double avg = static_cast<double>(coo.nnz()) / N;
    int max_deg = *std::max_element(degree.begin(), degree.end());
    REQUIRE(max_deg > 2 * avg);  // power-law produces hubs
}

// ---------------------------------------------------------------------------
// Watts–Strogatz
// ---------------------------------------------------------------------------
TEST_CASE("Watts–Strogatz: beta=0 yields ring lattice with degree K", "[topology][ws]")
{
    const int N = 100;
    const int K = 4;
    auto coo = generate_watts_strogatz(N, K, 0.0, 42);

    // Total edges for ring lattice with K neighbors: N * K (directed both ways).
    REQUIRE(coo.nnz() == static_cast<size_t>(N * K));

    // Each node should have degree K.
    std::vector<int> degree(N, 0);
    for (size_t i = 0; i < coo.nnz(); ++i) {
        degree[coo.rows[i]]++;
    }
    for (int i = 0; i < N; ++i) {
        REQUIRE(degree[i] == K);
    }
}

TEST_CASE("Watts–Strogatz: no self-loops", "[topology][ws]")
{
    const int N = 50;
    const int K = 4;
    auto coo = generate_watts_strogatz(N, K, 0.5, 42);

    for (size_t i = 0; i < coo.nnz(); ++i) {
        REQUIRE(coo.rows[i] != coo.cols[i]);
    }
}

TEST_CASE("Watts–Strogatz: edge count preserved under rewiring", "[topology][ws]")
{
    const int N = 100;
    const int K = 6;
    auto coo = generate_watts_strogatz(N, K, 0.3, 42);

    // Edge count should be N * K regardless of rewiring.
    REQUIRE(coo.nnz() == static_cast<size_t>(N * K));
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------
TEST_CASE("generate_topology dispatcher works", "[topology]")
{
    auto er = generate_topology("er", 100, 0.1, 42);
    REQUIRE(er.N == 100);
    REQUIRE(er.nnz() > 0);

    auto fi = generate_topology("fi", 100, 0.05, 42);
    REQUIRE(fi.nnz() == 500);  // K = max(1, int(0.05*100)) = 5, nnz = 100*5

    REQUIRE_THROWS(generate_topology("unknown", 100, 0.1, 42));
}
