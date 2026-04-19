# Sparse Matrix Formats for Spike Propagation

> **Implementation Status (April 2026):** All four formats — COO, CSR, CSC,
> and ELLPACK — are fully implemented with `scatter()`, `gather()`, and
> `gather_all()` methods behind the `SparseMatrix` abstract interface.
> Source files: `src/{coo,csr,csc,ell}_matrix.cpp` with headers in
> `include/`. Unit tests in `tests/test_formats.cpp` verify correctness
> of scatter/gather operations across all formats. Memory footprint
> reporting via `memory_bytes()` is validated against analytical formulas.

## Overview

Spike propagation in spiking neural networks (SNNs) fundamentally requires
multiplying a sparse connectivity (weight) matrix by a binary spike vector.
The choice of sparse storage format has a dramatic impact on both memory
consumption and runtime performance.  This module implements four classical
sparse matrix formats—**COO**, **CSR**, **CSC**, and **ELLPACK**—and exposes
them through a common abstract interface (`SparseMatrix`) so that benchmarks
can compare them under identical conditions.

---

## The Spike-Propagation Primitive

Given $N$ neurons, let $\mathbf{W} \in \mathbb{R}^{N \times N}$ denote the
weight matrix where $W_{ij}$ is the synaptic weight **from** neuron $j$
**to** neuron $i$.  At each timestep, a subset $S \subseteq \{0, \ldots, N-1\}$
of neurons fire.  The synaptic current delivered to neuron $i$ is:

$$
I_i = \sum_{j \in S} W_{ij}
$$

This is equivalent to $\mathbf{I} = \mathbf{W} \mathbf{s}$, where
$\mathbf{s}$ is the binary spike indicator vector.

Two algorithmic approaches exist:

| Approach | Traversal direction | Storage access pattern | Best format |
|----------|---------------------|------------------------|-------------|
| **Scatter** (push) | Iterate over spiking neurons; for each, iterate over its post-synaptic targets and *add* the weight to the target's current | Column-oriented access per spiking neuron | **CSR** (row = target, column = source; rows of W indexed by target) |
| **Gather** (pull) | Iterate over all neurons; for each target, iterate over its pre-synaptic sources and check if they spiked | Row-oriented access per target | **CSC** (quick column traversal for gather) |

In our implementation, **scatter** is the primary benchmark operation (push-based:
iterate spiking neurons, distribute weights to targets) and **gather** is the
symmetric counterpart (pull-based: for each target, sum incoming weights from
spiking sources).  Both operations are benchmarked at every timestep to enable
fair scatter-vs-gather comparison across formats.

---

## Format Descriptions

### 1. COO (Coordinate)

**Storage:**  Three parallel arrays `row[k]`, `col[k]`, `val[k]` for
$k = 0, \ldots, \text{nnz}-1$.

**Memory:**

$$
M_{\text{COO}} = \text{nnz} \times (2 \times \text{sizeof(int)} + \text{sizeof(double)})
$$

**Scatter complexity:** $O(\text{nnz})$ per timestep — every element is
visited regardless of the spike count.

**Gather complexity:** $O(\text{nnz})$ for the same reason.

**Use case:** Data exchange format.  Easy to construct and convert to other
formats. Poor computational performance.

### 2. CSR (Compressed Sparse Row)

**Storage:**
- `row_ptr[N+1]`:  `row_ptr[i]` is the index into `col_idx` / `values`
  where row $i$ begins.
- `col_idx[nnz]`:  Column indices of non-zero elements.
- `values[nnz]`:   Non-zero values.

**Memory:**

$$
M_{\text{CSR}} = (N + 1) \times \text{sizeof(int)} + \text{nnz} \times (\text{sizeof(int)} + \text{sizeof(double)})
$$

**Scatter complexity:** For scatter in our convention (W rows = targets),
we iterate over each spiking source $j$ and need the column $j$ of W.
Our CSR stores row-compressed data; to scatter from column $j$, we scan
all entries where `col_idx[k] == j`.  However, our implementation
re-indexes: the CSR is stored such that row $i$ contains the outgoing
synapses of neuron $i$ (transposed convention). This gives
$O\bigl(\sum_{j \in S} \deg_{\text{out}}(j)\bigr)$ per timestep.

**Gather complexity:** Efficient: $O(\deg_{\text{in}}(i))$ per target neuron $i$.

**Use case:** Best format for scatter-based propagation when rows are indexed
by source neuron.

### 3. CSC (Compressed Sparse Column)

**Storage:**
- `col_ptr[N+1]`:  `col_ptr[j]` is the index into `row_idx` / `values`
  where column $j$ begins.
- `row_idx[nnz]`:  Row indices of non-zero elements.
- `values[nnz]`:   Non-zero values.

**Memory:**

$$
M_{\text{CSC}} = (N + 1) \times \text{sizeof(int)} + \text{nnz} \times (\text{sizeof(int)} + \text{sizeof(double)})
$$

**Scatter complexity:** For scatter from spiking neuron $j$, column $j$
gives all post-synaptic targets in $O(\deg(j))$.

**Gather complexity:** $O(\text{nnz})$ — must scan all columns to find
entries in a given row.

**Use case:** Best for gather-based propagation; also efficient for
scatter when columns correspond to source neurons (which is our default
`W_{ij}` convention).

### 4. ELLPACK (ELL)

**Storage:**  Two 2D arrays of size $N \times K$ where $K$ is the
maximum number of non-zeros per row:
- `col_indices[i][k]`:  Column index of the $k$-th non-zero in row $i$
  (padded with $-1$).
- `values[i][k]`:  Corresponding weight value (padded with $0$).

**Memory:**

$$
M_{\text{ELL}} = N \times K \times (\text{sizeof(int)} + \text{sizeof(double)})
$$

**Scatter complexity:** $O(|S| \times K)$ — regular strided access makes
this very cache-friendly.

**Gather complexity:** $O(K)$ per target — constant regardless of topology.

**Use case:** Best cache performance when the degree distribution is narrow
(e.g., fixed in-degree topologies).  Wastes memory when the degree
distribution has high variance (e.g., Barabási–Albert).

---

## Complexity and Memory Summary

| Format | Memory | Scatter (push) | Gather (pull) | Cache friendliness |
|--------|--------|-----------------|---------------|---------------------|
| COO    | $O(\text{nnz})$ (3 arrays) | $O(\text{nnz})$ | $O(\text{nnz})$ | Poor (random access) |
| CSR    | $O(N + \text{nnz})$ | $O(\sum \deg_{\text{out}})$ | $O(\deg_{\text{in}})$ | Good (row-sequential) |
| CSC    | $O(N + \text{nnz})$ | $O(\sum \deg_{\text{out}})$ | $O(\deg_{\text{in}})$ | Good (column-sequential) |
| ELL    | $O(N \cdot K)$ | $O(|S| \cdot K)$ | $O(K)$ | Best (strided) |

Note: $K = \max_i \deg(i)$ for ELLPACK.

---

## API Reference

### Abstract Base Class: `SparseMatrix`

```cpp
class SparseMatrix {
public:
    virtual ~SparseMatrix() = default;

    // Push-based spike delivery (scatter benchmark).
    virtual void scatter(const std::vector<int>& spike_sources,
                         std::vector<double>&    out_buffer) const = 0;

    // Pull-based synaptic input for a single target.
    virtual double gather(int target,
                          const std::vector<int>& spike_sources) const = 0;

    // Pull-based synaptic input for ALL targets (gather benchmark).
    virtual void gather_all(const std::vector<int>& spike_sources,
                            std::vector<double>&    out_buffer) const = 0;

    virtual int    num_rows()     const = 0;
    virtual int    num_cols()     const = 0;
    virtual size_t num_nonzeros() const = 0;
    virtual size_t memory_bytes() const = 0;
};
```

`scatter` and `gather_all` are the two operations benchmarked in the timed
loop.  `gather` (single-target) is used in unit tests.

### Interchange Format: `COOTriplets`

```cpp
struct COOTriplets {
    std::vector<int>    rows, cols;
    std::vector<double> vals;
    int N;                // Matrix dimension (N×N)
};
```

All topology generators produce `COOTriplets`.  Each sparse matrix
constructor accepts `const COOTriplets&` and converts to its internal
representation.

### Construction Pattern

```cpp
// 1. Generate topology.
COOTriplets coo = generate_topology("erdos_renyi", N, density, seed);

// 2. Construct any format.
CSRMatrix  csr(coo);
CSCMatrix  csc(coo);
ELLMatrix  ell(coo);
COOMatrix  coo_mat(coo);
```

### Files

| File | Purpose |
|------|---------|
| `include/sparse_matrix.h` | Abstract base class + `COOTriplets` |
| `include/coo_matrix.h`, `src/coo_matrix.cpp` | COO implementation |
| `include/csr_matrix.h`, `src/csr_matrix.cpp` | CSR implementation |
| `include/csc_matrix.h`, `src/csc_matrix.cpp` | CSC implementation |
| `include/ell_matrix.h`, `src/ell_matrix.cpp` | ELLPACK implementation |
