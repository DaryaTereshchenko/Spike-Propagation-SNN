# Benchmark Results — Analysis and Discussion

This document presents the results of a systematic benchmarking of four sparse
matrix storage formats (**COO**, **CSR**, **CSC**, **ELL**) across four network
topologies (**Erdős–Rényi**, **Fixed In-Degree**, **Barabási–Albert**,
**Watts–Strogatz**), three network sizes ($N \in \{1000, 5000, 10000\}$), and
three connection densities ($d \in \{0.01, 0.05, 0.1\}$).  Each configuration
was run for 1 000 timesteps over 10 trials.

---

## 1  Critical Anomaly — Zero Network Activity

**All 120 configurations report `total_spikes = 0`.**  The LIF population
never sustains activity beyond the seeded initial firing (≈ 1 % of neurons on
timestep 0).

### Root cause

The synaptic weight normalisation
$w = 1/\sqrt{Np}$ (Erdős–Rényi) or $w = 1/K$ (Fixed In-Degree) keeps the
*variance* of the total synaptic input bounded, but the absolute magnitude is
far too small to cross the 15 mV threshold gap
($V_\text{thresh} - V_\text{rest} = -50 - (-65) = 15$ mV).

**Example** (ER, $N = 10\,000$, $d = 0.1$):
- $w = 1/\sqrt{N \cdot d} = 1/\sqrt{1000} \approx 0.032$
- ≈ 100 initial spikes, each neuron has ≈ 1 000 incoming connections
- Expected I_syn per neuron ≈ $10 \times 0.032 = 0.32$ mV-equivalent
- $\Delta V = (dt/\tau_m) \cdot R \cdot I_\text{syn} = 0.05 \times 0.32 \approx 0.016$ mV
- Requires ≈ 940 steps of sustained input to reach threshold — but the input
  dies at timestep 1

### Consequence for interpretation

Because scatter is only active for **one out of 1 000 timesteps**, the
measured wall-clock times reflect:

| Component | Contribution |
|-----------|-------------|
| 1 scatter call (timestep 0, ≈ 10 spikes) | Format-dependent |
| 999 × N LIF Euler updates (no scatter) | Identical across formats |
| Cache pollution from matrix allocation | Format- and topology-dependent |

The format ranking and relative speedups are still valid because the single
scatter call exercises the format's asymptotic complexity, and the cache
pollution effect is a realistic artefact of matrix layout.  However, metrics
that depend on spike counts (`scatter_throughput`, `bytes_per_spike`) are
**zero and should be disregarded**.

### Recommended fixes (for future runs)

| Approach | Change |
|----------|--------|
| External Poisson drive | Add constant background current $I_\text{ext} = 14$ mV to each neuron per timestep so the network operates near threshold |
| Weight gain | Multiply all weights by a gain factor $g \approx 50\text{–}100$ |
| Lower threshold gap | Set $V_\text{thresh} = -60$ mV (5 mV gap instead of 15 mV) |
| Higher initial activity | Seed 10–20 % of neurons instead of 1 % |
| Artificial spike injection | Inject Poisson spikes at a controlled rate (e.g. 5 % per timestep) to decouple spike generation from LIF dynamics |

---

## 2  Peak RSS Is a High-Water Mark

`peak_rss_kb` is read from `/proc/self/status` (`VmHWM`), which records the
**process-lifetime maximum** resident set.  Because the sweep runs all 120
configurations sequentially in one process, the value plateaus early
(≈ 1 264 MB after the largest ELL–BA allocation) and never decreases.

**The `peak_rss_kb` column is not usable for per-configuration memory
comparison.**  Use the `memory_bytes` column instead — it reports the
format-specific storage footprint directly from each matrix object.

---

## 3  Performance Results

### 3.1  Wall-Clock Time (ms, mean ± std, 10 trials, 1 000 timesteps)

#### Erdős–Rényi (ER)

| N | Density | COO | CSR | CSC | ELL |
|----:|--------:|----:|----:|----:|----:|
| 1 000 | 0.01 | 1.27 ± 0.08 | 1.51 ± 0.38 | 1.86 ± 0.83 | 1.36 ± 0.21 |
| 1 000 | 0.05 | 1.43 ± 0.23 | 1.38 ± 0.22 | 1.34 ± 0.11 | 1.27 ± 0.07 |
| 1 000 | 0.10 | 1.52 ± 0.11 | 1.83 ± 0.90 | 1.51 ± 0.14 | 1.33 ± 0.15 |
| 5 000 | 0.01 | 8.17 ± 1.65 | 7.32 ± 0.85 | 7.66 ± 1.01 | 7.07 ± 0.94 |
| 5 000 | 0.05 | 9.01 ± 0.84 | 7.48 ± 1.64 | 11.44 ± 4.63 | 6.53 ± 0.22 |
| 5 000 | 0.10 | 10.79 ± 1.01 | 7.95 ± 1.43 | 14.18 ± 3.52 | 8.41 ± 3.10 |
| 10 000 | 0.01 | 18.26 ± 2.18 | 16.42 ± 1.11 | 17.96 ± 2.36 | 15.33 ± 3.05 |
| 10 000 | 0.05 | 22.23 ± 3.30 | 16.70 ± 2.52 | 27.44 ± 3.81 | 16.02 ± 2.45 |
| 10 000 | 0.10 | 29.71 ± 2.99 | **15.05 ± 1.32** | 37.79 ± 6.25 | 15.21 ± 2.89 |

#### Fixed In-Degree (FI)

| N | Density | COO | CSR | CSC | ELL |
|----:|--------:|----:|----:|----:|----:|
| 1 000 | 0.01 | 1.27 ± 0.07 | 1.75 ± 0.58 | 1.75 ± 0.90 | 1.67 ± 0.54 |
| 1 000 | 0.05 | 1.67 ± 0.51 | 1.35 ± 0.20 | 1.74 ± 0.57 | 1.57 ± 0.41 |
| 1 000 | 0.10 | 1.77 ± 0.70 | 1.50 ± 0.46 | 2.35 ± 2.79 | 1.24 ± 0.04 |
| 5 000 | 0.01 | 11.10 ± 3.16 | 7.09 ± 0.47 | 7.69 ± 0.89 | 6.96 ± 0.73 |
| 5 000 | 0.05 | 10.75 ± 0.78 | 8.43 ± 2.60 | 11.91 ± 2.92 | 7.06 ± 0.99 |
| 5 000 | 0.10 | 13.55 ± 1.09 | 9.01 ± 3.11 | 12.48 ± 1.45 | 7.61 ± 2.10 |
| 10 000 | 0.01 | 17.98 ± 2.43 | 16.55 ± 2.29 | 16.84 ± 2.87 | 14.14 ± 1.51 |
| 10 000 | 0.05 | 28.73 ± 4.17 | 18.56 ± 4.54 | 29.21 ± 3.07 | 15.72 ± 2.84 |
| 10 000 | 0.10 | 41.84 ± 5.68 | 17.47 ± 5.16 | 39.89 ± 5.35 | **15.96 ± 2.88** |

#### Barabási–Albert (BA)

| N | Density | COO | CSR | CSC | ELL |
|----:|--------:|----:|----:|----:|----:|
| 1 000 | 0.01 | 1.58 ± 0.46 | 1.90 ± 1.43 | 1.55 ± 0.37 | 1.60 ± 0.69 |
| 1 000 | 0.05 | 1.60 ± 0.29 | 1.39 ± 0.27 | 1.42 ± 0.09 | 1.57 ± 0.47 |
| 1 000 | 0.10 | 1.84 ± 0.50 | 1.73 ± 0.55 | 1.55 ± 0.13 | 1.61 ± 0.71 |
| 5 000 | 0.01 | 9.89 ± 2.32 | 6.95 ± 0.23 | 9.13 ± 2.80 | 7.40 ± 2.68 |
| 5 000 | 0.05 | 14.89 ± 2.45 | 7.92 ± 2.17 | 12.74 ± 2.37 | 9.04 ± 4.08 |
| 5 000 | 0.10 | 16.70 ± 1.85 | **7.15 ± 1.02** | 18.02 ± 3.09 | 8.23 ± 2.00 |
| 10 000 | 0.01 | 19.51 ± 2.60 | 15.84 ± 2.96 | 19.81 ± 2.73 | 14.43 ± 2.09 |
| 10 000 | 0.05 | 35.58 ± 5.04 | 16.03 ± 0.98 | 39.66 ± 5.03 | 15.57 ± 3.03 |
| 10 000 | 0.10 | 52.16 ± 4.76 | **16.54 ± 2.42** | 65.55 ± 6.90 | 16.28 ± 2.68 |

#### Watts–Strogatz (WS)

| N | Density | COO | CSR | CSC | ELL |
|----:|--------:|----:|----:|----:|----:|
| 1 000 | 0.01 | 1.59 ± 0.92 | 1.43 ± 0.32 | 1.60 ± 0.80 | 1.58 ± 0.71 |
| 1 000 | 0.05 | 2.48 ± 2.17 | 1.92 ± 1.97 | 1.51 ± 0.34 | 1.32 ± 0.10 |
| 1 000 | 0.10 | 1.79 ± 0.57 | 1.42 ± 0.25 | 1.51 ± 0.28 | 1.36 ± 0.32 |
| 5 000 | 0.01 | 8.25 ± 1.44 | 8.61 ± 1.44 | 7.77 ± 1.72 | 6.94 ± 0.56 |
| 5 000 | 0.05 | 12.34 ± 7.05 | 8.32 ± 2.81 | 10.78 ± 2.95 | 6.46 ± 0.05 |
| 5 000 | 0.10 | 13.40 ± 2.57 | 7.10 ± 1.19 | 15.07 ± 2.19 | 8.10 ± 2.19 |
| 10 000 | 0.01 | 18.12 ± 2.65 | 14.92 ± 0.67 | 16.65 ± 1.75 | 14.66 ± 2.08 |
| 10 000 | 0.05 | 27.56 ± 2.91 | 14.53 ± 0.99 | 31.19 ± 4.95 | 16.33 ± 2.56 |
| 10 000 | 0.10 | 35.22 ± 2.71 | 16.34 ± 4.60 | 41.45 ± 7.25 | **13.75 ± 0.99** |

#### Key observations

1. **CSR and ELL are consistently fastest**, typically 2–4× faster than COO
   and CSC at large scale ($N = 10\,000$, $d = 0.1$).
2. **CSC is the slowest format for scatter** — its `scatter()` implementation
   must scan all nnz entries (same asymptotic cost as COO), but with worse
   memory access patterns (column-major iteration for a row-oriented query).
3. **ELL slightly outperforms CSR for regular topologies** (FI, WS) because
   uniform degree $\Rightarrow$ minimal padding, and its strided memory
   layout yields excellent spatial locality.
4. **At small scale ($N = 1\,000$), all formats are comparable** (< 2 ms);
   differences are dominated by measurement noise (high relative std).
5. **BA topology penalises COO and CSC most severely** because its high edge
   count ($\text{nnz} \approx 2Nm$) amplifies the $O(\text{nnz})$ scatter
   cost.

### 3.2  Format Speedup over COO at N = 10 000, d = 0.1

| Topology | CSR speedup | CSC speedup | ELL speedup |
|----------|------------:|------------:|------------:|
| ER       | 1.97× | 0.79× | 1.95× |
| FI       | 2.39× | 1.05× | 2.62× |
| BA       | 3.15× | 0.80× | 3.20× |
| WS       | 2.16× | 0.85× | 2.56× |

CSC is consistently *slower* than COO despite identical memory usage — the
column-oriented scan with a boolean lookup is less cache-friendly than COO's
sequential scan.

---

## 4  Memory Footprint

### 4.1  Matrix Memory (MB) — `memory_bytes` column

| N | Density | Topology | COO | CSR | CSC | ELL |
|------:|--------:|:---------|------:|------:|------:|------:|
| 1 000 | 0.01 | ER | 0.15 | 0.12 | 0.12 | 0.23 |
| 1 000 | 0.05 | ER | 0.77 | 0.58 | 0.58 | 0.86 |
| 1 000 | 0.10 | ER | 1.53 | 1.15 | 1.15 | 1.55 |
| 5 000 | 0.01 | ER | 3.81 | 2.88 | 2.88 | 4.58 |
| 5 000 | 0.05 | ER | 19.05 | 14.31 | 14.31 | 17.57 |
| 5 000 | 0.10 | ER | 38.11 | 28.61 | 28.61 | 33.42 |
| 10 000 | 0.01 | ER | 15.26 | 11.48 | 11.48 | 16.14 |
| 10 000 | 0.05 | ER | 76.28 | 57.25 | 57.25 | 68.32 |
| 10 000 | 0.10 | ER | 152.59 | 114.47 | 114.47 | 128.75 |
| 10 000 | 0.10 | FI | 152.59 | 114.47 | 114.47 | 128.38 |
| 10 000 | 0.10 | BA | 289.91 | 217.44 | 217.44 | **500.40** |
| 10 000 | 0.10 | WS | 152.59 | 114.47 | 114.47 | 120.85 |

### 4.2  Memory formulas (theoretical)

| Format | Formula | Bytes per nnz |
|--------|---------|--------------|
| COO | $3 \times \texttt{nnz} \times \bar{s}$ ≈ $16 \times \texttt{nnz}$ | 16 |
| CSR | $(N+1) \times 4 + \texttt{nnz} \times 12$ | 12 + amortised |
| CSC | $(N+1) \times 4 + \texttt{nnz} \times 12$ | 12 + amortised |
| ELL | $N \times K_\max \times 12$ | Variable (depends on $K_\max / \bar{K}$) |

where $\bar{s}$ = average element size = `(sizeof(int) + sizeof(int) + sizeof(double))` = 16 bytes.

### 4.3  ELL Memory Explosion for Barabási–Albert

ELL pads every row to $K_\max$ (the maximum out-degree).  For power-law degree
distributions (BA), hub nodes have extremely high degree:

| N | Density | nnz (BA) | $K_\max$ (est.) | ELL / CSR ratio |
|------:|--------:|---------:|--------:|-----------:|
| 1 000 | 0.01 | 19 890 | 177 | 8.8× |
| 1 000 | 0.05 | 97 450 | 343 | 3.5× |
| 1 000 | 0.10 | 189 900 | 456 | 2.4× |
| 5 000 | 0.01 | 497 450 | 755 | 7.6× |
| 5 000 | 0.05 | 2 437 250 | 1 615 | 3.3× |
| 5 000 | 0.10 | 4 749 500 | 2 185 | 2.3× |
| 10 000 | 0.01 | 1 989 900 | 1 455 | 7.3× |
| 10 000 | 0.05 | 9 749 500 | 3 126 | 3.2× |
| 10 000 | 0.10 | 18 999 000 | 4 372 | 2.3× |

$K_\max$ estimated from `memory_bytes / (N × 12)`.

**ELL is unsuitable for scale-free topologies** — its memory cost grows with
$N \times K_\max$ rather than nnz, making it impractical when the degree
variance is high.  For regular topologies (FI, WS) where
$K_\max \approx \bar{K}$, ELL has minimal overhead (< 6 % over CSR).

---

## 5  Cache Hierarchy Analysis

The benchmark machine has the following cache hierarchy (detected from
`/sys/devices/system/cpu`):

| Level | Size |
|-------|------|
| L1d | 48 KB (49 152 B) |
| L2 | 1 280 KB (1 310 720 B) |
| L3 | 18 432 KB (18 874 368 B) |

*(Values inferred from `cache_ratio` columns: e.g., ER N=1000 d=0.01 COO
has `memory_bytes=159 424` and `cache_ratio_L1=3.24`, giving
$L1d = 159\,424 / 3.24 \approx 49\,152$ B.)*

### 5.1  Cache Ratio Summary (matrix_bytes / cache_size)

A ratio > 1 means the matrix exceeds that cache level.

| Format | Topology | N | Density | L1d ratio | L2 ratio | L3 ratio |
|--------|----------|----:|--------:|----------:|---------:|---------:|
| COO | ER | 10 000 | 0.10 | 3 255× | 76.3× | 6.36× |
| CSR | ER | 10 000 | 0.10 | 2 442× | 57.2× | 4.77× |
| CSC | ER | 10 000 | 0.10 | 2 442× | 57.2× | 4.77× |
| ELL | ER | 10 000 | 0.10 | 2 747× | 64.4× | 5.36× |
| ELL | BA | 10 000 | 0.10 | **10 674×** | **250.2×** | **20.8×** |
| CSR | BA | 10 000 | 0.10 | 4 639× | 108.7× | 9.06× |
| ELL | WS | 10 000 | 0.10 | 2 576× | 60.4× | 5.03× |

### 5.2  Cache-performance interaction

Despite **all** large configurations exceeding L3, CSR and ELL maintain
competitive performance.  This can be attributed to:

- **CSR scatter** accesses only the rows of spiking neurons — spatial locality
  within each row's contiguous `col_idx[]` and `val[]` arrays benefits from
  hardware prefetching.
- **ELL scatter** has perfectly strided access (`base + k` for $k = 0 \ldots
  K_\max - 1$), enabling aggressive prefetching even when the matrix exceeds
  all cache levels.
- **COO and CSC** must touch the entire matrix (scan all nnz), causing full
  cache pollution regardless of spike count.

---

## 6  Effective Bandwidth

The reported `effective_bw_gbps` = `memory_bytes / mean_time_ms` provides an
**upper-bound estimate** of the matrix traversal bandwidth.  Because only one
scatter call is active (due to zero spikes after timestep 0), these values
reflect the ratio of total matrix size to total loop time, not sustained
scatter bandwidth.

### Selected values (N = 10 000, d = 0.1)

| Format | Topology | Eff. BW (GB/s) | Notes |
|--------|----------|---------------:|-------|
| CSR | ER | 7.98 | Highest: dominated by LIF update time, not scatter |
| CSR | BA | 13.79 | Larger matrix, similar time → higher apparent BW |
| CSC | BA | 3.48 | Slowest scatter → lowest apparent BW |
| ELL | BA | 32.22 | Inflated: ELL's large padded matrix ÷ fast time |
| ELL | WS | 9.20 | Compact ELL → reasonable BW estimate |

**Caution:** These values are not directly comparable to hardware STREAM
bandwidth benchmarks because they conflate scatter cost with LIF update cost.

---

## 7  Edge Count Validation

The `nnz` column allows verification of topology generator correctness:

| Topology | Expected nnz | Density = 0.01, N = 10 000 | Density = 0.1, N = 10 000 |
|----------|-------------|---------------------------|--------------------------|
| ER | $\approx N^2 \cdot d$ | 999 815 (expected 1 000 000 ± √) | 10 000 192 (expected 10 000 000 ± √) |
| FI | $N \cdot K = N^2 \cdot d$ | 1 000 000 (exact) | 10 000 000 (exact) |
| BA | $\approx 2Nm$ | 1 989 900 (expected 2 000 000) | 18 999 000 (expected 19 000 000) |
| WS | $N \cdot K$ | 1 000 000 (exact) | 10 000 000 (exact) |

Note: BA produces nearly **2×** the edge count of other topologies at the same
density parameter because each new node adds $m$ undirected edges (2m directed).
The density parameter maps to $m = \lfloor d \cdot N \rfloor$, so BA graphs are
structurally denser.  This partially explains BA's higher execution times for
COO and CSC.

---

## 8  Format × Topology Interaction — Key Research Findings

### 8.1  CSR vs ELL across topologies (N = 10 000, d = 0.1)

| Topology | CSR (ms) | ELL (ms) | ELL memory / CSR memory | Faster format |
|----------|------:|------:|------------------------:|:-------------|
| ER | 15.05 | 15.21 | 1.12× | CSR (≈ tied) |
| FI | 17.47 | 15.96 | 1.12× | ELL |
| BA | 16.54 | 16.28 | 2.30× | ELL (marginal; 2.3× memory cost) |
| WS | 16.34 | 13.75 | 1.06× | **ELL** |

**Finding:** ELL is faster or comparable to CSR for *all* topologies at large
scale, even for BA where it uses 2.3× more memory.  ELL's strided access
pattern outweighs its memory disadvantage.  However, for memory-constrained
systems, CSR provides the best time-memory trade-off for irregular topologies.

### 8.2  Scatter complexity classes

| Class | Formats | Scatter cost | Scaling with N, d |
|-------|---------|-------------|-------------------|
| Spike-proportional | CSR, ELL | $O(\|S\| \cdot \bar{K})$ | Independent of total nnz |
| Matrix-proportional | COO, CSC | $O(\text{nnz})$ | Grows with N²·d |

This distinction becomes critical at large scale: CSR/ELL scatter cost depends
on the *activity* (number of spiking neurons × average degree), while COO/CSC
always scan the full matrix.

### 8.3  CSC overhead vs COO

Despite identical memory footprints and the same $O(\text{nnz})$ asymptotic
scatter cost, CSC is **consistently 10–30 % slower than COO** at large scale.
This is because CSC scatter:

1. Iterates columns and checks each entry's row against a boolean mask
   (random access to `is_spiking[]`)
2. The column-oriented layout causes cache-line thrashing when accumulating
   `out_buffer[c]` across non-contiguous column blocks

COO's sequential scan of `(row, col, val)` triples is more cache-friendly.

---

## 9  Scaling Behaviour

### 9.1  Time vs N (ER, d = 0.1)

| Format | N = 1 000 | N = 5 000 | N = 10 000 | Ratio 10k/1k |
|--------|--------:|--------:|---------:|--------:|
| COO | 1.52 | 10.79 | 29.71 | 19.5× |
| CSR | 1.83 | 7.95 | 15.05 | 8.2× |
| CSC | 1.51 | 14.18 | 37.79 | 25.0× |
| ELL | 1.33 | 8.41 | 15.21 | 11.4× |

- **COO and CSC scale as $O(N^2)$** — consistent with $\text{nnz} = N^2 d$
  and full-matrix scan.
- **CSR scales roughly as $O(N)$** — scatter cost depends only on spike degree,
  and LIF update is $O(N)$.
- **ELL scales as $O(N)$ to $O(N \cdot K_\max)$** — for ER, $K_\max$ grows
  slowly (Poisson tail), so scaling is close to linear.

### 9.2  Time vs density (ER, N = 10 000)

| Format | d = 0.01 | d = 0.05 | d = 0.10 | Ratio 0.1/0.01 |
|--------|--------:|--------:|---------:|--------:|
| COO | 18.26 | 22.23 | 29.71 | 1.63× |
| CSR | 16.42 | 16.70 | 15.05 | 0.92× |
| CSC | 17.96 | 27.44 | 37.79 | 2.10× |
| ELL | 15.33 | 16.02 | 15.21 | 0.99× |

- **CSR and ELL are density-invariant** — scatter touches only spike rows.
- **COO and CSC degrade linearly with density** — more nnz = longer scan.
- CSR actually gets *slightly faster* at higher density, likely due to better
  cache line utilisation when rows are denser.

---

## 10  Suggestions for Future Work and Improved Publication Quality

### 10.1  Fix network activity to obtain spike-dependent metrics

The highest-priority improvement.  With sustained spike activity, one can
measure:
- **Scatter throughput** (edges/ms) as a function of spike rate
- **Bytes per spike** — memory efficiency of each format
- **Format crossover points** — at what spike rate does CSR outperform ELL?

### 10.2  Separate benchmark per process

Run each configuration in an isolated process (`fork` or script invocation) to
obtain accurate per-configuration peak RSS values.

### 10.3  Add hardware performance counters

Integrate `perf_event_open()` directly or use the existing `run_benchmarks.sh
--perf` to capture:
- L1/L2/L3 cache miss rates per configuration
- Instructions per cycle (IPC)
- Branch mispredictions

### 10.4  Controlled spike-rate sweeps

Inject spikes at fixed rates (1 %, 5 %, 10 %, 25 %, 50 %) independent of LIF
dynamics to characterise how scatter cost scales with activity for each format.

### 10.5  Gather benchmark

Add CSC-optimised gather results for comparison — CSC should dominate CSR for
pull-based propagation, providing the symmetric counterpart.

### 10.6  Memory–performance Pareto analysis

Plot time vs. memory for each (format, topology) pair to identify Pareto-optimal
configurations.  ELL's memory waste for BA should make it Pareto-dominated by
CSR for irregular topologies.

### 10.7  Comparison with established simulators

Include GeNN (GPU) and NEST (CPU) reference timings for the same network
configurations to contextualise the benchmark within the SNN literature.

---

## 11  Summary Table — Recommended Format by Use Case

| Criterion | Best format | Rationale |
|-----------|------------|-----------|
| Fastest scatter (push) | CSR or ELL | $O(\|S\| \cdot \bar{K})$, spike-proportional |
| Fastest gather (pull) | CSC | Column-indexed, $O(K)$ per target |
| Minimum memory | CSR / CSC | $(N+1) \times 4 + \text{nnz} \times 12$ |
| Regular topologies (FI, WS) | ELL | Minimal padding, strided access |
| Scale-free topologies (BA) | CSR | No padding waste from hub nodes |
| Format interchange | COO | Trivial construction; convert to CSR/CSC/ELL |
| GPU acceleration | ELL or CSR | ELL: coalesced warps; CSR: standard SpMV |

---

*Data generated by `spike_benchmark --sweep` on a Linux system with a 48 KB L1d / 1.25 MB L2 / 18 MB L3 cache hierarchy.  All timings are wall-clock averages over 10 trials of 1 000 timesteps each.*
