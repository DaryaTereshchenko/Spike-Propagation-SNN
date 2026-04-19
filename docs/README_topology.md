# Network Topology Generators

> **Implementation Status (April 2026):** All four topology generators —
> Erdős–Rényi, Fixed In-Degree, Barabási–Albert, and Watts–Strogatz — are
> fully implemented in `src/topology.cpp` / `include/topology.h`. Unit tests
> in `tests/test_topology.cpp` verify edge counts, degree distributions, and
> self-loop exclusion. All topologies have been benchmarked across the full
> sweep grid (N ∈ {1000, 5000, 10000}, density ∈ {0.01, 0.05, 0.1}).

## Overview

The topology of a neural network—i.e., which neurons are connected and with
what weights—has a profound effect on spike-propagation cost.  This module
provides four well-studied random graph models used extensively in
computational neuroscience.  Each generator produces a `COOTriplets` struct
that can be converted to any supported sparse matrix format.

---

## Graph Models

### 1. Erdős–Rényi (ER)

**Key:** `"erdos_renyi"`

**Definition:** Each possible directed edge $(j \to i)$ for $i \neq j$ is
included independently with probability $p$.  Self-connections are excluded.

**Expected number of edges:** $\mathrm{E}[\text{nnz}] = N(N-1)p$

**Degree distribution:** Each neuron's in-degree follows
$\text{Bin}(N-1, p)$, which for large $N$ and moderate $p$ approximates a
Poisson distribution with mean $(N-1)p$.

**Weight normalization:** $w = 1 / \sqrt{Np}$ — scales total synaptic
input so that total expected input current $\approx \sqrt{Np} \cdot w = 1$,
preventing runaway excitation.

**Biological relevance:** The ER model is arguably the simplest null model for
cortical connectivity. While real cortical circuits exhibit structured
connectivity, the ER model's statistical homogeneity makes it an essential
baseline.  It corresponds to Brunel's (2000) balanced random network model.

**Parameters:**
| Parameter | Meaning | CLI name |
|-----------|---------|----------|
| `N`       | Number of neurons | `--size` |
| `p`       | Connection probability | `--density` |
| `seed`    | RNG seed | `--seed` |

---

### 2. Fixed In-Degree (FI)

**Key:** `"fixed_indegree"`

**Definition:** Every neuron receives exactly $K = \lfloor p \times N \rfloor$
incoming connections, each chosen uniformly at random (*without* replacement
and excluding self-connections).  Selection uses the Fisher–Yates shuffle
(Knuth's Algorithm P) for unbiased sampling.

**Exact number of edges:** $\text{nnz} = N \times K$

**Degree distribution:** In-degree is deterministically $K$; out-degree is
approximately $\text{Bin}(N, K/N)$.

**Weight normalization:** $w = 1 / K$ — ensures the sum of incoming weights
to every neuron is exactly 1.

**Biological relevance:** In cortical circuits, the variability of in-degree
is often much lower than predicted by an ER model.  Fixed in-degree networks
are used in the NEST reference model (Potjans & Diesmann, 2014) and provide
a controlled setting where every neuron receives exactly the same total
excitation, isolating the effect of topology from input heterogeneity.

**ELLPACK advantage:** Because all rows have exactly $K$ non-zeros, ELLPACK
stores this topology with zero padding waste ($K_{\max} = K$), making it the
ideal test case for ELLPACK performance.

**Parameters:**
| Parameter | Meaning | CLI name |
|-----------|---------|----------|
| `N`       | Number of neurons | `--size` |
| `K` (via $p$) | In-degree per neuron | `--density` |
| `seed`    | RNG seed | `--seed` |

---

### 3. Barabási–Albert (BA)

**Key:** `"barabasi_albert"`

**Definition:** Starting from a fully connected seed of $m$ nodes (where
$m = \max(1, \lfloor p \times N \rfloor)$), each new node connects to $m$
existing nodes with probability proportional to their current degree
(*preferential attachment*).  The implementation uses a flat degree list for
$O(1)$ weighted sampling.

**Number of edges:** $\text{nnz} = m(m-1) + (N-m) \times m$  (seed clique +
new edges)

**Degree distribution:** Power-law tail $P(k) \sim k^{-3}$, yielding a
scale-free network with hub neurons.

**Weight normalization:** $w = 1.0$ (uniform).

**Biological relevance:** Although cortical connectivity does not follow a
strict power-law, hub neurons do appear in the connectome (e.g., cortical
"rich-club" organization; van den Heuvel & Sporns, 2011).  The heavy-tailed
degree distribution of BA graphs stress-tests ELLPACK (which must pad to
the maximum degree) and reveals how hub neurons create load imbalance during
scatter.

**Parameters:**
| Parameter | Meaning | CLI name |
|-----------|---------|----------|
| `N`       | Number of neurons | `--size` |
| `m` (via $p$) | Edges per new node | `--density` |
| `seed`    | RNG seed | `--seed` |

---

### 4. Watts–Strogatz (WS)

**Key:** `"watts_strogatz"`

**Definition:** Start from a directed ring lattice where each node connects
to $K = \max(2, \lfloor p \times N \rfloor)$ nearest neighbors (half forward,
half backward).  Then rewire each edge with probability $\beta = 0.3$:
replace the target with a uniformly random node (avoiding self-loops and
duplicate edges; limited retries with safety valve).

**Number of edges:** $\text{nnz} = N \times K$ (edge count is preserved
during rewiring).

**Degree distribution:** Approximately regular with small variance (between
ring lattice and random graph extremes).

**Weight normalization:** $w = 1.0$ (uniform).

**Biological relevance:** Cortical networks exhibit the "small-world"
property—high clustering coefficient with short average path length (Watts &
Strogatz, 1998).  At $\beta = 0.3$, the WS model balances local clustering
from the ring lattice with the short paths introduced by random rewiring.
This models local cortical column connectivity with long-range inter-area
projections.

**Parameters:**
| Parameter | Meaning | CLI name |
|-----------|---------|----------|
| `N`       | Number of neurons | `--size` |
| `K` (via $p$) | Ring lattice degree | `--density` |
| `\beta = 0.3` | Rewiring probability | (hardcoded) |
| `seed`    | RNG seed | `--seed` |

---

## API Reference

### Generator Function

```cpp
COOTriplets generate_topology(const std::string& name,
                              int N, double density, unsigned seed);
```

Dispatcher that selects the generator by `name`.  Valid names:
`"erdos_renyi"`, `"fixed_indegree"`, `"barabasi_albert"`, `"watts_strogatz"`.

### Individual Generators

```cpp
COOTriplets generate_erdos_renyi(int N, double p, std::mt19937& rng);
COOTriplets generate_fixed_indegree(int N, double p, std::mt19937& rng);
COOTriplets generate_barabasi_albert(int N, double p, std::mt19937& rng);
COOTriplets generate_watts_strogatz(int N, double p, std::mt19937& rng);
```

### Files

| File | Purpose |
|------|---------|
| `include/topology.h` | Generator declarations |
| `src/topology.cpp`   | Generator implementations |
