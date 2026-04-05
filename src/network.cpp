#include "network.hpp"
#include <random>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// ── helpers ──────────────────────────────────────────────────────────────────
// ── Erdős–Rényi G(N, p) ──────────────────────────────────────────────────────

std::vector<Edge> erdos_renyi(std::size_t N, double p, double weight,
                               unsigned seed) {
    std::mt19937 rng(seed);
    std::bernoulli_distribution coin(p);
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(N * N * p * 1.1));
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if (i != j && coin(rng)) {
                edges.push_back({i, j, weight});
            }
        }
    }
    return edges;
}

// ── Fixed in-degree ───────────────────────────────────────────────────────────

std::vector<Edge> fixed_indegree(std::size_t N, std::size_t k, double weight,
                                  unsigned seed) {
    if (k >= N) {
        throw std::invalid_argument("fixed_indegree: k must be < N");
    }
    std::mt19937 rng(seed);
    std::vector<Edge> edges;
    edges.reserve(N * k);

    std::vector<std::size_t> candidates(N);
    for (std::size_t i = 0; i < N; ++i) candidates[i] = i;

    for (std::size_t j = 0; j < N; ++j) {
        // Sample k distinct sources != j using partial Fisher-Yates.
        std::vector<std::size_t> pool(candidates);
        pool.erase(pool.begin() + j);    // remove self
        for (std::size_t s = 0; s < k; ++s) {
            std::uniform_int_distribution<std::size_t> pick(s, pool.size() - 1);
            std::size_t idx = pick(rng);
            std::swap(pool[s], pool[idx]);
            edges.push_back({pool[s], j, weight});
        }
    }
    return edges;
}

// ── Barabási–Albert preferential attachment ───────────────────────────────────

std::vector<Edge> barabasi_albert(std::size_t N, std::size_t m, double weight,
                                   unsigned seed) {
    if (m == 0 || m >= N) {
        throw std::invalid_argument("barabasi_albert: need 0 < m < N");
    }
    std::mt19937 rng(seed);
    std::vector<Edge> edges;
    edges.reserve(N * m);

    // Degree sequence used to implement preferential attachment via repeated sampling.
    std::vector<std::size_t> degree_list;
    degree_list.reserve(2 * N * m);

    // Seed: fully connect first m+1 nodes.
    for (std::size_t i = 0; i <= m; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            edges.push_back({i, j, weight});
            edges.push_back({j, i, weight});
            degree_list.push_back(i);
            degree_list.push_back(i);
            degree_list.push_back(j);
            degree_list.push_back(j);
        }
    }

    // Add remaining nodes one at a time.
    for (std::size_t i = m + 1; i < N; ++i) {
        std::unordered_set<std::size_t> chosen;
        while (chosen.size() < m) {
            std::uniform_int_distribution<std::size_t> pick(0, degree_list.size() - 1);
            std::size_t target = degree_list[pick(rng)];
            if (target != i && chosen.find(target) == chosen.end()) {
                chosen.insert(target);
                edges.push_back({i, target, weight});
                edges.push_back({target, i, weight});
                degree_list.push_back(i);
                degree_list.push_back(i);
                degree_list.push_back(target);
                degree_list.push_back(target);
            }
        }
    }
    return edges;
}

// ── Watts–Strogatz small-world ────────────────────────────────────────────────

std::vector<Edge> watts_strogatz(std::size_t N, std::size_t k, double beta,
                                  double weight, unsigned seed) {
    if (k % 2 != 0) {
        throw std::invalid_argument("watts_strogatz: k must be even");
    }
    if (k >= N) {
        throw std::invalid_argument("watts_strogatz: k must be < N");
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> any_node(0, N - 1);

    // Build regular ring lattice: each node i -> i±1 ... i±(k/2) (mod N).
    // Store as adjacency set for rewiring.
    std::vector<std::unordered_set<std::size_t>> adj(N);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t d = 1; d <= k / 2; ++d) {
            std::size_t j = (i + d) % N;
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }

    // Rewire: for each node i, for each right-neighbour j, rewire with prob beta.
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t d = 1; d <= k / 2; ++d) {
            std::size_t j = (i + d) % N;
            if (prob(rng) < beta) {
                // Choose a new target != i and not already connected.
                std::size_t newj;
                do {
                    newj = any_node(rng);
                } while (newj == i || adj[i].count(newj));
                adj[i].erase(j);
                adj[j].erase(i);
                adj[i].insert(newj);
                adj[newj].insert(i);
            }
        }
    }

    // Collect directed edges.
    std::vector<Edge> edges;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j : adj[i]) {
            edges.push_back({i, j, weight});
        }
    }
    return edges;
}
