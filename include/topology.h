#pragma once

#include "sparse_matrix.h"   // COOTriplets
#include <string>

/// Generate an Erdős–Rényi random graph G(N, p).
/// Each directed edge (i→j, i≠j) exists independently with probability @p p.
/// Weights are set to 1.0 / sqrt(N * p) for input normalization.
COOTriplets generate_erdos_renyi(int N, double p, unsigned seed);

/// Generate a random graph with fixed in-degree K.
/// Every neuron j receives exactly K incoming edges from K distinct sources
/// chosen uniformly at random (no self-loops).
/// Weights are set to 1.0 / K.
COOTriplets generate_fixed_indegree(int N, int K, unsigned seed);

/// Generate a Barabási–Albert preferential-attachment graph.
/// Start with a fully connected clique of (m+1) nodes.  Each new node
/// attaches to @p m existing nodes with probability proportional to their
/// current degree.  Produces an undirected-style graph stored as directed
/// edges (both directions).
/// Weights are 1.0.
COOTriplets generate_barabasi_albert(int N, int m, unsigned seed);

/// Generate a Watts–Strogatz small-world graph.
/// Start with a ring lattice where each node connects to its K/2 nearest
/// neighbors on each side.  Each edge is then rewired with probability
/// @p beta to a random target (no self-loops, no duplicate edges).
/// Weights are 1.0.
COOTriplets generate_watts_strogatz(int N, int K, double beta, unsigned seed);

/// Parse a topology name string ("er", "fi", "ba", "ws") and generate the
/// corresponding graph.  @p param is interpreted as:
///   - er: density p
///   - fi: in-degree K (cast to int)
///   - ba: attachments m (cast to int)
///   - ws: uses K = (int)param, beta = 0.3 (hardcoded default)
COOTriplets generate_topology(const std::string& name, int N, double param,
                              unsigned seed);
