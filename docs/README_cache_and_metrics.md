# Cache Hierarchy and Derived Performance Metrics

## Overview

Sparse matrix spike propagation performance is fundamentally governed by
how well a format's memory access pattern interacts with the CPU cache
hierarchy. This module detects the cache configuration at runtime and
computes derived metrics that quantify this interaction.

---

## CPU Cache Hierarchy

### Background

Modern CPUs use a multi-level cache hierarchy to bridge the speed gap
between the processor and main memory (DRAM):

| Level | Typical Size | Latency | Role |
|-------|-------------|---------|------|
| **L1d** (data) | 32–64 KB per core | ~1 ns (~4 cycles) | Fastest; holds the hot working set |
| **L1i** (instruction) | 32–64 KB per core | ~1 ns | Instruction fetch cache |
| **L2** (unified) | 256 KB – 2 MB per core | ~3–5 ns (~12 cycles) | Bridges L1 and L3 |
| **L3** (unified) | 4–64 MB shared | ~10–20 ns (~40 cycles) | Last on-chip level before DRAM |
| **DRAM** | GBs | ~50–100 ns (~200 cycles) | Main memory, orders of magnitude slower |

When a sparse matrix **fits entirely in L1d**, scatter operations run at
processor speed. When it spills into L2 or L3, each cache miss adds
latency. When it exceeds L3, the CPU must fetch from DRAM — a ~50×
penalty relative to L1.

### Detection Method

At startup, the benchmark reads the Linux sysfs interface:

```
/sys/devices/system/cpu/cpu0/cache/index{0,1,2,3,...}/
├── level    → 1, 2, or 3
├── type     → "Data", "Instruction", or "Unified"
└── size     → e.g. "48K", "2048K", "24576K"
```

The implementation iterates over `index0` through `index9`, classifying
each entry by its level and type. Sizes are parsed from the `<N>K` / `<N>M`
format into bytes. The result is stored in the `CacheInfo` struct:

```cpp
struct CacheInfo {
    long L1d_bytes = 0;   // L1 data cache size
    long L1i_bytes = 0;   // L1 instruction cache size
    long L2_bytes  = 0;   // L2 unified cache size
    long L3_bytes  = 0;   // L3 unified (last-level) cache size
};
```

### Example Output

```
=== CPU Cache Hierarchy ===
  L1 Data:        48 KB
  L1 Instruction: 64 KB
  L2 Unified:     2 MB
  L3 Unified:     24 MB
```

Cache sizes are also written to `results/cache_info.csv` for automated
analysis.

---

## Derived Metrics

Seven derived metrics are computed for every benchmark configuration. They
transform raw timing and memory data into quantities that explain **why**
one format outperforms another.

### 1. Cache Ratio (matrix_bytes / cache_level_bytes)

$$R_{Lk} = \frac{M_{\text{matrix}}}{C_{Lk}}$$

where $M_{\text{matrix}}$ is `memory_bytes()` of the sparse matrix and
$C_{Lk}$ is the size of cache level $k \in \{L1d, L2, L3\}$.

**Interpretation:**
- $R < 1$ — the matrix fits in this cache level
- $R \approx 1$ — borderline; performance cliff expected
- $R > 1$ — the matrix spills; higher levels or DRAM will be hit

| Format | N=1000, d=0.05 | N=5000, d=0.05 |
|--------|---------------|----------------|
| **CSR** (606 KB) | L1: 12.3×, L2: 0.29×, L3: 0.02× | L1: 305×, L2: 7.2×, L3: 0.60× |

This tells us that at N=1000 CSR fits in L2 (ratio 0.29), but at N=5000
it spills past L2 (ratio 7.2) and approaches L3 capacity (ratio 0.60).

### Memory Formulas Per Format

The `memory_bytes()` function returns the exact in-memory footprint used
to compute cache ratios:

| Format | Formula | Components |
|--------|---------|------------|
| **COO** | $\text{nnz} \times (4 + 4 + 8) = 16 \cdot \text{nnz}$ | `row[nnz]` (int) + `col[nnz]` (int) + `val[nnz]` (double) |
| **CSR** | $(N+1) \times 4 + \text{nnz} \times (4 + 8) = 4(N+1) + 12 \cdot \text{nnz}$ | `row_ptr[N+1]` (int) + `col_idx[nnz]` (int) + `val[nnz]` (double) |
| **CSC** | $(N+1) \times 4 + \text{nnz} \times (4 + 8) = 4(N+1) + 12 \cdot \text{nnz}$ | `col_ptr[N+1]` (int) + `row_idx[nnz]` (int) + `val[nnz]` (double) |
| **ELL** | $N \times K_{\max} \times (4 + 8) = 12 \cdot N \cdot K_{\max}$ | `indices[N×K_max]` (int) + `values[N×K_max]` (double) |

where $K_{\max}$ is the maximum number of non-zeros in any row. For
networks with uneven degree distributions (e.g., Barabási–Albert), ELL
can be significantly larger than CSR because it pads every row to the
maximum degree.

### 2. Effective Bandwidth (GB/s)

$$B_{\text{eff}} = \frac{M_{\text{matrix}}}{t_{\text{mean}} \times 10^{-3}} \times 10^{-9}$$

This is an **upper-bound** estimate of how much memory bandwidth the
simulation uses. It assumes the entire matrix is read once per trial.

**Interpretation:**
- Compare against the hardware's peak memory bandwidth
  (e.g., ~50 GB/s for DDR4, ~100 GB/s for DDR5, ~200 GB/s for HBM).
- If $B_{\text{eff}}$ is close to peak bandwidth, the workload is
  **memory-bound** — faster computation won't help.
- If $B_{\text{eff}}$ is much lower than peak, the workload may be
  **latency-bound** (random access pattern) or **compute-bound**.

### 3. Scatter Throughput (edges/ms)

$$S_{\text{scatter}} = \frac{\bar{s} \times \bar{d}_{\text{out}} \times T}{t_{\text{scatter}}}$$

where $\bar{s}$ is the average number of spikes per timestep,
$\bar{d}_{\text{out}} = \text{nnz}/N$ is the average out-degree, and
$T$ is the number of timesteps.

**Interpretation:**
- This directly measures how many synaptic connections are processed
  per millisecond of wall-clock time during push-based propagation.
- Higher values indicate a more cache-friendly access pattern.
- CSR and ELL excel here because they store rows contiguously — each
  spiking neuron's outgoing connections are read sequentially.

### 4. Gather Throughput (edges/ms)

$$S_{\text{gather}} = \frac{\bar{s} \times \bar{d}_{\text{out}} \times T}{t_{\text{gather}}}$$

Same edge-count formula, but divided by the gather trial time.

**Interpretation:**
- Measures pull-based synaptic input collection efficiency.
- CSC should excel here because `gather_all` iterates each column's
  entries sequentially — its native access pattern.
- Comparing $S_{\text{scatter}}$ vs $S_{\text{gather}}$ per format
  reveals the scatter/gather asymmetry inherent in each layout.

### 5. Bytes per Spike

$$B_s = \frac{M_{\text{matrix}}}{S_{\text{total}}}$$

where $S_{\text{total}}$ is the total number of spike events across all
timesteps (averaged over trials).

**Interpretation:**
- Lower is better — indicates more efficient use of memory per
  propagated spike event.
- This is only meaningful when the network sustains spiking activity
  (i.e., $S_{\text{total}} > 0$).

---

## Hardware Performance Counters (perf stat)

When the benchmark is run with `--perf`, the shell script wraps each
configuration with `perf stat` collecting 8 hardware counters:

### Counter Descriptions

| Counter | What it measures | Why it matters for sparse SNN |
|---------|-----------------|------------------------------|
| `cache-misses` | Total cache misses | Overall cache-unfriendliness |
| `cache-references` | Total cache references (hits + misses) | Baseline for miss rate computation |
| `instructions` | Retired instructions | Total work done |
| `cycles` | CPU cycles consumed | Total time cost |
| `L1-dcache-load-misses` | L1 data cache misses | Most critical — every miss costs ~12 cycles to L2 |
| `LLC-load-misses` | Last-level cache misses | Each miss = DRAM access (~200 cycles) |
| `dTLB-load-misses` | Data TLB misses | Reveals fragmented/scattered access patterns |
| `branch-misses` | Branch mispredictions | Higher for irregular sparsity (variable row lengths) |

### Derived Ratios from perf Data

| Ratio | Formula | Interpretation |
|-------|---------|----------------|
| **Cache miss rate** | `cache-misses / cache-references` | Fraction of cache accesses that miss |
| **IPC** | `instructions / cycles` | Pipeline efficiency; modern CPUs can sustain 2–4 IPC |
| **L1 miss rate** | `L1-dcache-load-misses / cache-references` | Fraction of data accesses that miss L1 |
| **DRAM access rate** | `LLC-load-misses / cache-references` | Fraction of accesses going all the way to DRAM |

### Expected Behaviour by Format

| Metric | COO | CSR (scatter) | CSC (gather) | ELL |
|--------|-----|---------------|--------------|-----|
| **L1 miss rate** | High (random row,col) | Medium (sequential col in row) | Low-Medium (sequential row in col) | Low (strided, predictable) |
| **dTLB misses** | Highest (3 arrays, random) | Medium (2 arrays, sequential) | Medium | Low (dense strided) |
| **Branch misses** | Low (uniform loop) | Medium (variable row length) | Medium (variable col length) | Lowest (fixed K_max iteration) |
| **LLC misses** | Depends on N | Depends on N | Depends on N | Higher if padding is large |

---

## Interpreting Results: A Decision Framework

Use the following framework to explain format performance differences:

1. **Check cache ratios** — Does the matrix fit in L1? L2? L3?
   - If $R_{L1} < 1$: all formats will be fast; differences are minimal.
   - If $R_{L2} < 1 < R_{L1}$: L1 miss rate becomes the differentiator.
   - If $R_{L3} < 1 < R_{L2}$: LLC misses dominate; sequential formats win.
   - If $R_{L3} > 1$: DRAM-bound; effective bandwidth is the key metric.

2. **Check effective bandwidth** — Is it close to peak?
   - If yes → memory-bound; format layout determines performance.
   - If no → latency-bound (random access) or compute-bound.

3. **Check scatter & gather throughput** — Which format processes edges fastest?
   - CSR should dominate for scatter (push-based propagation).
   - CSC should dominate for gather (pull-based propagation).
   - Comparing $S_{\text{scatter}}$ vs $S_{\text{gather}}$ per format
     reveals the inherent layout asymmetry.
   - ELL trades memory for regularity (no branch mispredictions).

4. **Check dTLB misses** — Is address translation a bottleneck?
   - High TLB misses indicate the matrix spans too many virtual pages.
   - ELL's strided layout and CSR's sequential rows help here.

---

## Output Files

| File | Contents |
|------|----------|
| `results/cache_info.csv` | L1d, L1i, L2, L3 sizes in bytes |
| `results/benchmark_results.csv` | All metrics including derived columns |
| `results/perf_results.csv` | Hardware counter data from `perf stat` |

### Example cache_info.csv

```csv
level,size_bytes
L1d,49152
L1i,65536
L2,2097152
L3,25165824
```

---

## Plots Generated

The plotting script (`scripts/plot_results.py`) produces four additional
plots from the new metrics:

| Plot | File | Description |
|------|------|-------------|
| Effective bandwidth vs N | `bandwidth_vs_N.png` | Shows whether the workload becomes memory-bound at larger N |
| Scatter throughput vs N | `scatter_throughput_vs_N.png` | Compares edge processing efficiency across formats |
| Cache ratio bar chart | `cache_ratios.png` | Grouped bars showing L1/L2/L3 ratios per format at max N |
| TLB & branch heatmap | `tlb_branch_heatmap.png` | Reveals access pattern differences between formats |
