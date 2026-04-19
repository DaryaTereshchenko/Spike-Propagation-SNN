# Plotting and Visualisation

> **Implementation Status (April 2026):** The plotting script
> `scripts/plot_results.py` is fully implemented with 9 plotting functions
> covering time-vs-N, time-vs-density, memory-vs-N, NEST comparison, cache
> heatmaps, effective bandwidth, scatter throughput, cache ratio bars, and
> TLB/branch heatmaps. CPU benchmark data in `results/benchmark_results.csv`
> (144 rows) is ready for visualisation.

## Overview

The plotting module (`scripts/plot_results.py`) generates publication-quality
figures from the benchmark CSV output.  It produces five analysis views that
together characterise the performance trade-offs of each sparse format.

---

## Plots

### 1. Simulation Time vs. Network Size

**Function:** `plot_time_vs_N()`

**File:** `results/time_vs_N.png`

A 2×2 subplot grid (one panel per topology).  Each line shows a sparse
format's mean wall-clock time as a function of network size $N$, with
shaded ±1 standard deviation bands.

**Interpretation:** This plot reveals the scaling behaviour of each format.
CSR and CSC should show near-linear scaling with the product $N \cdot p$
(number of non-zeros), while COO shows uniformly worse performance because
it traverses **all** non-zeros on every timestep regardless of spike count.
ELLPACK may outperform CSR/CSC for fixed-indegree topologies due to its
cache-friendly strided access, but degrade on Barabási–Albert networks
where padding waste dominates.

---

### 2. Simulation Time vs. Density

**Function:** `plot_time_vs_density()`

**File:** `results/time_vs_density.png`

Mean time plotted against connection density $p$ (averaged over all
topologies and network sizes).

**Interpretation:** For scatter-based propagation, time should grow roughly
linearly with $p$ because the number of synapses scales as $\Theta(N^2 p)$.
Deviations from linearity at high density may indicate cache saturation
(working set exceeds L2/L3 cache).

---

### 3. Memory vs. Network Size

**Function:** `plot_memory_vs_N()`

**File:** `results/memory_vs_N.png`

Peak RSS (in MB) as a function of $N$, one line per format.

**Interpretation:** CSR and CSC have identical memory footprints
($O(N + \text{nnz})$).  COO uses slightly more ($3 \times \text{nnz}$ arrays
vs. $2 + N$).  ELLPACK memory is $O(N \cdot K_{\max})$ and can be
significantly larger than CSR when the degree distribution has high variance
(Barabási–Albert).

---

### 4. NEST Comparison

**Function:** `plot_nest_comparison()`

**File:** `results/nest_comparison.png`

A grouped bar chart comparing simulation time across formats when using
the NEST-exported connectivity matrix ($N = 10\,000$, balanced E/I column).

**Interpretation:** This validates that our C++ benchmark produces
consistent spike dynamics with a reference NEST simulation, and reveals
the format-dependent overhead on a biologically structured connectivity.

---

### 5. Cache Miss Heatmap

**Function:** `plot_cache_heatmap()`

**File:** `results/cache_heatmap.png`

A heatmap with formats on the $y$-axis and topologies on the $x$-axis,
coloured by cache miss rate (cache-misses / cache-references × 100, from
`perf stat` data).

**Interpretation:** ELLPACK should show the lowest miss rate for regular
topologies (fixed-indegree, Watts–Strogatz) due to stride-1 access.  COO
should show the highest miss rate because its random access pattern defeats
hardware prefetching.  CSR/CSC should lie between the two.

---

## Usage

```bash
# After running benchmarks:
python3 scripts/plot_results.py

# Specify custom data and output directories:
python3 scripts/plot_results.py --data results/benchmark_results.csv \
    --perf results/perf_results.csv --outdir results/
```

### Dependencies

- `matplotlib >= 3.5`
- `pandas >= 1.4`
- `numpy >= 1.21`

### Files

| File | Purpose |
|------|---------|
| `scripts/plot_results.py` | All 5 plotting functions + CLI |
