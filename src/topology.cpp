#include "topology.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <numeric>

// ---------------------------------------------------------------------------
// Erdős–Rényi G(N, p)
// ---------------------------------------------------------------------------
COOTriplets generate_erdos_renyi(int N, double p, unsigned seed)
{
    COOTriplets coo;
    coo.N = N;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const double w = 1.0 / std::sqrt(N * p);   // weight normalization

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;               // no self-loops
            if (dist(rng) < p) {
                coo.rows.push_back(i);
                coo.cols.push_back(j);
                coo.vals.push_back(w);
            }
        }
    }
    return coo;
}

// ---------------------------------------------------------------------------
// Fixed in-degree K
// ---------------------------------------------------------------------------
COOTriplets generate_fixed_indegree(int N, int K, unsigned seed)
{
    if (K >= N) {
        throw std::invalid_argument("K must be < N for fixed in-degree");
    }

    COOTriplets coo;
    coo.N = N;

    std::mt19937 rng(seed);
    const double w = 1.0 / K;

    // For each target neuron j, sample K distinct sources ≠ j.
    std::vector<int> candidates(N);
    std::iota(candidates.begin(), candidates.end(), 0);

    for (int j = 0; j < N; ++j) {
        // Fisher–Yates partial shuffle: pick K elements from candidates
        // excluding j.
        // Build a pool without j.
        std::vector<int> pool;
        pool.reserve(N - 1);
        for (int i = 0; i < N; ++i) {
            if (i != j) pool.push_back(i);
        }

        // Partial shuffle to pick K elements.
        for (int k = 0; k < K; ++k) {
            std::uniform_int_distribution<int> pick(k, static_cast<int>(pool.size()) - 1);
            int idx = pick(rng);
            std::swap(pool[k], pool[idx]);
        }

        for (int k = 0; k < K; ++k) {
            coo.rows.push_back(pool[k]);   // source
            coo.cols.push_back(j);         // target
            coo.vals.push_back(w);
        }
    }
    return coo;
}

// ---------------------------------------------------------------------------
// Barabási–Albert preferential attachment
// ---------------------------------------------------------------------------
COOTriplets generate_barabasi_albert(int N, int m, unsigned seed)
{
    if (m < 1 || m >= N) {
        throw std::invalid_argument("m must be in [1, N)");
    }

    COOTriplets coo;
    coo.N = N;

    std::mt19937 rng(seed);

    // Start with a fully connected clique of m+1 nodes.
    std::vector<int> degree(N, 0);
    std::set<std::pair<int,int>> edge_set;

    auto add_edge = [&](int u, int v) {
        if (u == v) return;
        if (edge_set.count({u, v})) return;
        edge_set.insert({u, v});
        edge_set.insert({v, u});
        coo.rows.push_back(u);
        coo.cols.push_back(v);
        coo.vals.push_back(1.0);
        coo.rows.push_back(v);
        coo.cols.push_back(u);
        coo.vals.push_back(1.0);
        degree[u]++;
        degree[v]++;
    };

    for (int i = 0; i <= m; ++i) {
        for (int j = i + 1; j <= m; ++j) {
            add_edge(i, j);
        }
    }

    // Add remaining nodes one at a time.
    // Repeated-degree list for efficient weighted sampling.
    std::vector<int> degree_list;
    for (int i = 0; i <= m; ++i) {
        for (int d = 0; d < degree[i]; ++d) {
            degree_list.push_back(i);
        }
    }

    for (int new_node = m + 1; new_node < N; ++new_node) {
        // Select m distinct targets via preferential attachment.
        std::set<int> targets;
        while (static_cast<int>(targets.size()) < m) {
            std::uniform_int_distribution<int> pick(0, static_cast<int>(degree_list.size()) - 1);
            int target = degree_list[pick(rng)];
            if (target != new_node) {
                targets.insert(target);
            }
        }

        for (int t : targets) {
            add_edge(new_node, t);
            degree_list.push_back(new_node);
            degree_list.push_back(t);
        }
    }

    return coo;
}

// ---------------------------------------------------------------------------
// Watts–Strogatz small-world
// ---------------------------------------------------------------------------
COOTriplets generate_watts_strogatz(int N, int K, double beta, unsigned seed)
{
    if (K % 2 != 0 || K < 2) {
        throw std::invalid_argument("K must be even and >= 2");
    }
    if (K >= N) {
        throw std::invalid_argument("K must be < N");
    }

    COOTriplets coo;
    coo.N = N;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    const int half_K = K / 2;

    // Build ring lattice as an edge set for efficient duplicate checking.
    // edges[i] = set of neighbors of node i.
    std::vector<std::set<int>> adj(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= half_K; ++j) {
            int target = (i + j) % N;
            adj[i].insert(target);
            adj[target].insert(i);
        }
    }

    // Rewire: for each node i, for each edge to i's clockwise neighbors,
    // rewire with probability beta.
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= half_K; ++j) {
            if (uniform01(rng) < beta) {
                int old_target = (i + j) % N;
                // Pick a new random target ≠ i, not already a neighbor.
                int new_target;
                int attempts = 0;
                do {
                    std::uniform_int_distribution<int> pick(0, N - 1);
                    new_target = pick(rng);
                    attempts++;
                    if (attempts > N * 10) break;  // safety valve
                } while (new_target == i || adj[i].count(new_target));

                if (attempts <= N * 10 && new_target != i) {
                    // Remove old edge, add new one.
                    adj[i].erase(old_target);
                    adj[old_target].erase(i);
                    adj[i].insert(new_target);
                    adj[new_target].insert(i);
                }
            }
        }
    }

    // Convert adjacency sets to COO (both directions already in adj).
    for (int i = 0; i < N; ++i) {
        for (int j : adj[i]) {
            if (i < j) {   // avoid duplicates — add both directions once
                coo.rows.push_back(i);
                coo.cols.push_back(j);
                coo.vals.push_back(1.0);
                coo.rows.push_back(j);
                coo.cols.push_back(i);
                coo.vals.push_back(1.0);
            }
        }
    }

    return coo;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------
COOTriplets generate_topology(const std::string& name, int N, double param,
                              unsigned seed)
{
    if (name == "er") {
        return generate_erdos_renyi(N, param, seed);
    } else if (name == "fi") {
        int K = std::max(1, static_cast<int>(param * N));
        return generate_fixed_indegree(N, K, seed);
    } else if (name == "ba") {
        int m = std::max(1, static_cast<int>(param * N));
        return generate_barabasi_albert(N, m, seed);
    } else if (name == "ws") {
        int K = std::max(2, static_cast<int>(param * N));
        return generate_watts_strogatz(N, K, 0.3, seed);
    } else {
        throw std::invalid_argument("Unknown topology: " + name);
    }
}
