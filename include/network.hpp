#pragma once
#include <vector>
#include <cstddef>
#include <string>
#include <utility>  // pair

// Connectivity generator for synthetic network topologies.
// All functions return a list of (source, target, weight) triples,
// which can be fed directly to the sparse-matrix factory functions.

struct Edge {
    std::size_t src;
    std::size_t dst;
    double      weight;
};

// Erdős–Rényi G(N, p): each directed edge exists independently with probability p.
std::vector<Edge> erdos_renyi(std::size_t N, double p, double weight = 1.0,
                               unsigned seed = 0);

// Fixed in-degree: every neuron receives exactly k randomly chosen distinct inputs.
std::vector<Edge> fixed_indegree(std::size_t N, std::size_t k, double weight = 1.0,
                                  unsigned seed = 0);

// Barabási–Albert preferential-attachment graph (directed, m new edges per step).
std::vector<Edge> barabasi_albert(std::size_t N, std::size_t m, double weight = 1.0,
                                   unsigned seed = 0);

// Watts–Strogatz small-world graph (each node connected to k nearest neighbours,
// then each edge is rewired with probability beta).
std::vector<Edge> watts_strogatz(std::size_t N, std::size_t k, double beta,
                                  double weight = 1.0, unsigned seed = 0);
