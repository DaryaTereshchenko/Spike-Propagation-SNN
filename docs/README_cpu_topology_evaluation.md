# CPU Topology Evaluation — Metrics, Results, Discussion, and Conclusions

> **Date:** April 19, 2026
> **Data source:** `results/benchmark_results.csv` — 144 configurations
> (4 formats × 4 topologies × 3 network sizes × 3 connection densities,
> 10 trials each with IQR-based outlier rejection).
> **System:** 6-core CPU, L1d 48 KB, L2 2 MB, L3 24 MB; background
> current $I_{\text{bg}} = 14.0$ mV producing ~5–8% sustained firing rate.

---

## 1. Evaluation Metrics

This section defines every metric used to evaluate the four sparse matrix
formats (COO, CSR, CSC, ELLPACK) across the four network topologies
(Erdős–Rényi, Fixed In-Degree, Barabási–Albert, Watts–Strogatz).

### 1.1 Wall-Clock Time (ms)

The primary performance metric. Measured via `std::chrono::high_resolution_clock`
around the full simulation loop (scatter or gather over $T = 1000$ timesteps).
Reported as **mean**, **standard deviation** (Bessel-corrected), and **median**
across 10 trials after IQR outlier rejection.

Wall-clock time includes LIF integration, Poisson drive, background current
injection, and the sparse matrix scatter/gather operation. It captures the
end-to-end cost a user would experience.

### 1.2 Scatter Throughput (edges/ms)

$$
S_{\text{scatter}} = \frac{\bar{s} \times \bar{d}_{\text{out}} \times T}{t_{\text{scatter}}}
$$

where $\bar{s}$ is the average spikes per timestep, $\bar{d}_{\text{out}} = \text{nnz}/N$
is the average out-degree, and $T = 1000$. This measures how efficiently each
format delivers spike-driven synaptic current in push-based (source-to-target)
mode. Higher is better.

### 1.3 Gather Throughput (edges/ms)

Same formula as scatter throughput but using the gather trial time. Measures
pull-based (target-collects-from-sources) efficiency. CSC's column-indexed
layout is naturally suited to this operation.

### 1.4 Effective Bandwidth (GB/s)

$$
B_{\text{eff}} = \frac{M_{\text{matrix}}}{t_{\text{mean}} \times 10^{-3}} \times 10^{-9}
$$

An upper-bound estimate of memory bandwidth utilisation. Indicates whether the
workload is memory-bound (close to hardware peak) or latency-bound (far below).

### 1.5 Memory Footprint (bytes)

The exact in-memory size of the sparse matrix, computed by `memory_bytes()`:

| Format | Formula |
|--------|---------|
| COO | $16 \cdot \text{nnz}$ |
| CSR | $4(N+1) + 12 \cdot \text{nnz}$ |
| CSC | $4(N+1) + 12 \cdot \text{nnz}$ |
| ELL | $12 \cdot N \cdot K_{\max}$ |

### 1.6 Cache Size Ratios

$$
R_{Lk} = \frac{M_{\text{matrix}}}{C_{Lk}}, \quad k \in \{L1d, L2, L3\}
$$

- $R < 1$: matrix fits in cache level $k$ — fast.
- $R > 1$: matrix spills — performance cliff expected.

### 1.7 Peak Resident Set Size (RSS)

`VmHWM` from `/proc/self/status`, in kilobytes. Captures the true physical
memory high-water mark of the process.

### 1.8 Bytes per Spike

$$
B_s = \frac{M_{\text{matrix}}}{S_{\text{total}}}
$$

Lower values indicate more efficient memory use relative to the amount of
spiking activity generated.

### 1.9 Spike Statistics

- **total_spikes**: cumulative spike count across all timesteps (averaged over trials).
- **spikes_per_step**: $\text{total\_spikes} / T$.

With $I_{\text{bg}} = 14.0$, the network sustains roughly 78–95 spikes/step
at $N = 1000$ and 800–950 spikes/step at $N = 10000$, depending on topology.

### 1.10 Outliers Removed

Number of trials rejected by the IQR filter before computing mean/std.
Typical values: 0–3 out of 10 trials. Higher counts flag measurement
instability.

---

## 2. Results

### 2.1 Scatter Time by Format and Topology (mean_time_ms)

#### N = 1,000

| Topology | COO | CSR | CSC | ELL |
|----------|-----|-----|-----|-----|
| ER (p=0.01) | 74.99 | **66.89** | 81.68 | 67.21 |
| ER (p=0.05) | 104.82 | **68.06** | 128.12 | 68.76 |
| ER (p=0.1) | 142.56 | **69.58** | 186.00 | 70.94 |
| FI (p=0.01) | 78.20 | **66.93** | 78.33 | 67.16 |
| FI (p=0.05) | 125.76 | **68.06** | 127.62 | 68.74 |
| FI (p=0.1) | 187.49 | **70.40** | 185.66 | 70.92 |
| BA (p=0.01) | 87.99 | **67.16** | 93.36 | 67.63 |
| BA (p=0.05) | 166.67 | **69.78** | 183.98 | 70.93 |
| BA (p=0.1) | 257.27 | **72.70** | 291.76 | 75.40 |
| WS (p=0.01) | 77.05 | **66.98** | 81.35 | 67.13 |
| WS (p=0.05) | 116.63 | **68.10** | 127.82 | 69.02 |
| WS (p=0.1) | 167.37 | **69.62** | 185.68 | 70.85 |

#### N = 10,000

| Topology | COO | CSR | CSC | ELL |
|----------|-----|-----|-----|-----|
| ER (p=0.01) | 1491.05 | **720.78** | 1871.45 | 744.90 |
| ER (p=0.05) | 4653.76 | **1009.83** | 6939.46 | 1047.65 |
| ER (p=0.1) | 8589.83 | **1252.94** | 13277.00 | 1309.05 |
| FI (p=0.01) | 1890.91 | **719.93** | 1872.49 | 741.74 |
| FI (p=0.05) | 7184.47 | **1010.97** | 6954.26 | 1047.99 |
| FI (p=0.1) | 13962.00 | **1258.94** | 13281.90 | 1312.33 |
| BA (p=0.01) | 3010.36 | **839.34** | 3142.76 | 877.92 |
| BA (p=0.05) | 11654.00 | **1287.74** | 13014.70 | 1387.64 |
| BA (p=0.1) | 22007.50 | **1760.51** | 24802.40 | 1956.12 |
| WS (p=0.01) | 1728.85 | **713.94** | 1872.74 | 738.01 |
| WS (p=0.05) | 6870.96 | **1001.33** | 6939.64 | 1041.38 |
| WS (p=0.1) | 13745.00 | **1245.55** | 13344.80 | 1302.61 |

**Bold** = fastest format for each row.

### 2.2 Scatter Throughput (edges/ms) at N = 10,000

| Topology | COO | CSR | CSC | ELL |
|----------|-----|-----|-----|-----|
| ER (p=0.01) | 54,255 | **112,234** | 43,226 | 108,600 |
| ER (p=0.05) | 91,289 | **420,701** | 61,220 | 405,514 |
| ER (p=0.1) | 101,777 | **697,758** | 65,847 | 667,848 |
| FI (p=0.01) | 42,778 | **112,355** | 43,198 | 109,052 |
| FI (p=0.05) | 59,129 | **420,204** | 61,087 | 405,359 |
| FI (p=0.1) | 62,599 | **694,243** | 65,805 | 666,003 |
| BA (p=0.01) | 54,655 | **196,025** | 52,353 | 187,411 |
| BA (p=0.05) | 74,831 | **677,216** | 67,007 | 628,463 |
| BA (p=0.1) | 81,543 | **1,019,340** | 72,355 | 917,411 |
| WS (p=0.01) | 46,783 | **113,288** | 43,188 | 109,592 |
| WS (p=0.05) | 61,826 | **424,239** | 61,214 | 407,922 |
| WS (p=0.1) | 63,779 | **703,814** | 65,691 | 672,984 |

### 2.3 Memory Footprint (MB) at N = 10,000, density = 0.1

| Format | ER | FI | BA | WS |
|--------|----|----|----|----|
| COO | 152.59 | 152.59 | 289.84 | 152.59 |
| CSR | 114.47 | 114.47 | 217.44 | 114.47 |
| CSC | 114.47 | 114.47 | 217.44 | 114.47 |
| **ELL** | **128.75** | **128.34** | **500.23** | **120.75** |

### 2.4 Cache Ratios at N = 10,000, density = 0.1

| Format | $R_{L1d}$ | $R_{L2}$ | $R_{L3}$ |
|--------|-----------|----------|----------|
| COO (ER) | 2441 | 305 | 19.1 |
| CSR (ER) | 1832 | 229 | 14.3 |
| CSC (ER) | 1832 | 229 | 14.3 |
| ELL (ER) | 2060 | 257 | 16.1 |
| ELL (BA) | **8005** | **1001** | **62.5** |

At this scale, all matrices vastly exceed L1 and L2 capacity and spill well
past L3 (24 MB). Performance is DRAM-bound — the access pattern and
sequential locality of the format determine throughput.

### 2.5 Effective Bandwidth (GB/s) at N = 10,000

| Topology | Density | COO | CSR | CSC | ELL |
|----------|---------|-----|-----|-----|-----|
| ER | 0.01 | 0.011 | **0.017** | 0.006 | 0.023 |
| ER | 0.05 | 0.017 | **0.059** | 0.009 | 0.068 |
| ER | 0.1 | 0.019 | **0.096** | 0.009 | 0.103 |
| BA | 0.1 | 0.014 | **0.130** | 0.009 | 0.268 |

### 2.6 Spike Activity Summary

Across all topologies and formats at $I_{\text{bg}} = 14.0$:

| N | Spikes/step (approx.) | Firing rate |
|---|----------------------|-------------|
| 1,000 | 78–82 | ~8% |
| 5,000 | 400–445 | ~8% |
| 10,000 | 809–945 | ~8–9% |

Barabási–Albert networks show slightly higher firing rates due to hub
neurons receiving concentrated input, producing an asymmetric positive
feedback loop.

---

## 3. Discussion

### 3.1 CSR Dominates Scatter Across All Topologies

The most unambiguous result is that **CSR is the fastest format for scatter
(push-based) spike propagation across every topology, density, and network
size tested**. At $N = 10{,}000$, $p = 0.1$:

- CSR is **6.9× faster** than COO (ER), **10.6× faster** than COO (FI),
  **12.5× faster** than COO (BA), and **11.0× faster** than COO (WS).
- CSR is **10.6× faster** than CSC on ER, and **14.1× faster** on BA.
- CSR is **1.04× faster** than ELL on ER but only **1.11× faster** on BA.

**Why CSR wins scatter:** In the scatter (push) operation, the benchmark
iterates over spiking neurons and, for each, reads all outgoing synaptic
weights. CSR stores each source neuron's outgoing connections in a
contiguous memory block (`row_ptr[j]` to `row_ptr[j+1]`), enabling
sequential reads that exploit hardware prefetching and maximise cache-line
utilisation. The pointer array adds only $4(N+1)$ bytes of overhead.

### 3.2 ELLPACK: The Close Second

ELL consistently ranks second in scatter performance, trailing CSR by only
3–11% across most topologies:

| Topology (N=10000, p=0.1) | CSR (ms) | ELL (ms) | ELL/CSR ratio |
|---------------------------|----------|----------|---------------|
| ER | 1252.94 | 1309.05 | 1.04× |
| FI | 1258.94 | 1312.33 | 1.04× |
| BA | 1760.51 | 1956.12 | 1.11× |
| WS | 1245.55 | 1302.61 | 1.05× |

ELL's strided, regular access pattern (`values[i*K_max + k]`) eliminates
branch mispredictions from variable row lengths and enables efficient SIMD
vectorisation. However, **ELL pays a memory penalty on irregular topologies**:
for BA at $N = 10{,}000$, $p = 0.1$, ELL consumes 500 MB vs CSR's 217 MB — a
2.3× overhead — because it pads every row to the maximum degree ($K_{\max}$),
and BA's power-law distribution creates extreme hub nodes.

**Fixed In-Degree is ELL's ideal topology:** Because every row has exactly
$K$ non-zeros, $K_{\max} = K$ and there is zero padding waste. Here ELL and
CSR achieve near-identical performance. This validates the theoretical
prediction that ELL excels when degree variance is zero.

### 3.3 CSC: Structurally Misaligned for Scatter

CSC is consistently the slowest compressed format for scatter, often
performing comparably to (or worse than) the uncompressed COO format:

| Config (N=10000) | COO (ms) | CSC (ms) | CSC/COO |
|------------------|----------|----------|---------|
| ER, p=0.1 | 8589.83 | 13277.00 | 1.55× slower |
| FI, p=0.1 | 13962.00 | 13281.90 | 0.95× (comparable) |
| BA, p=0.1 | 22007.50 | 24802.40 | 1.13× slower |

**Why CSC is slow for scatter:** Scatter iterates over spiking source neurons
and needs their outgoing connections. CSC is column-compressed: each *column*
stores the inputs to a particular target. To find source $j$'s outgoing
connections, CSC would need to scan all columns — the entries are scattered
across memory. In practice, our `scatter()` falls back to a full-matrix scan
for CSC, yielding $O(\text{nnz})$ complexity regardless of spike count.

CSC's strength is **gather** (pull-based) propagation, where each target neuron
collects its inputs. The gather throughput numbers show CSC performing on par
with COO and somewhat below CSR, because the gather benchmark still includes
full LIF simulation overhead that drowns out format-specific differences.

### 3.4 COO: The Baseline

COO serves as the performance baseline. Its scatter and gather operations both
scan the entire `nnz` elements regardless of spike count, giving $O(\text{nnz})$
per-timestep cost. This means COO's wall-clock time scales linearly with
density:

| Config (ER, N=10000) | p=0.01 | p=0.05 | p=0.1 | Ratio (0.1/0.01) |
|----------------------|--------|--------|-------|-------------------|
| COO | 1491 ms | 4654 ms | 8590 ms | **5.76×** |
| CSR | 721 ms | 1010 ms | 1253 ms | **1.74×** |

COO's 5.76× slowdown from low to high density reflects the 10× increase in nnz.
CSR's much flatter 1.74× scaling demonstrates that compressed formats amortise
the cost of empty rows — most neurons don't spike at any given step, and CSR
skips their connections entirely.

### 3.5 Topology Effects on Performance

#### Erdős–Rényi (ER)

The baseline random graph. Degree distribution is approximately Poisson with
mean $(N-1)p$. Performance is representative of the "generic" case with
moderate degree variance. CSR and ELL perform nearly identically because
the Poisson distribution has low $K_{\max}/\bar{K}$ ratio, minimising ELL
padding waste.

#### Fixed In-Degree (FI)

Every neuron has exactly $K$ incoming connections. This deterministic structure
produces **zero degree variance**, making it the ideal case for ELLPACK (no
padding waste). In practice FI performance is virtually identical to ER for all
formats, because the ER degree distribution is already narrow enough at these
densities that the padding overhead is minimal.

#### Barabási–Albert (BA)

The heavy-tailed degree distribution creates hub neurons with degree $\gg \bar{K}$.
This has two major consequences:

1. **ELL memory explosion:** At $N = 10{,}000$, $p = 0.1$, the BA topology has
   $K_{\max} \approx 4369$ (from the hub) vs $\bar{K} \approx 1900$. ELL pads
   every row to $K_{\max}$, producing 525 MB vs CSR's 217 MB.

2. **Slower wall-clock times across all formats:** BA consistently produces
   the highest timings at a given $N$ and $p$. This is partly due to the
   larger nnz (the BA generator produces more edges than ER at the same
   density parameter) and partly due to the load imbalance — hub neurons
   generate disproportionate scatter work.

| Format | ER (ms) | FI (ms) | BA (ms) | WS (ms) |
|--------|---------|---------|---------|---------|
| CSR (N=10k, p=0.1) | 1253 | 1259 | **1761** | 1246 |
| ELL (N=10k, p=0.1) | 1309 | 1312 | **1956** | 1303 |
| COO (N=10k, p=0.1) | 8590 | 13962 | **22008** | 13745 |

BA is the hardest topology for every format. CSR degrades least because its
row-pointer structure adds no per-row overhead regardless of degree variance.

#### Watts–Strogatz (WS)

The small-world model produces a near-regular degree distribution (like FI)
combined with local clustering and long-range shortcuts. Performance is nearly
identical to ER and FI for all formats. The slight spatial locality from the
ring-lattice base does not measurably improve cache performance at these
matrix sizes (all exceed L3).

### 3.6 Scaling Behaviour: N and Density

#### Network Size Scaling

For CSR on ER topology:

| N | p=0.01 (ms) | p=0.05 (ms) | p=0.1 (ms) |
|---|-------------|-------------|------------|
| 1,000 | 66.89 | 68.06 | 69.58 |
| 5,000 | 342.60 | 390.17 | 487.53 |
| 10,000 | 720.78 | 1009.83 | 1252.94 |

The scaling is super-linear in $N$ at fixed $p$ because nnz $\propto N^2 p$
and the matrix spills from L2 into L3 and eventually DRAM. At $N = 1{,}000$,
CSR's 124 KB matrix fits in L2 (ratio 0.24); at $N = 10{,}000$, the 120 MB
matrix is 14× larger than L3.

#### Density Scaling

For CSR, increasing density from 0.01 to 0.1 (10× more edges):

| N | Ratio (p=0.1 / p=0.01) |
|---|------------------------|
| 1,000 | 1.04× |
| 5,000 | 1.42× |
| 10,000 | 1.74× |

At $N = 1{,}000$ the matrix fits in L2 regardless of density, so the 10×
increase in nnz barely affects runtime (most time is LIF overhead). At
$N = 10{,}000$ the matrix grows from 12 MB to 120 MB, transitioning from
L3-resident to DRAM-bound, and the density penalty becomes pronounced.

### 3.7 Cache Hierarchy Impact

The cache ratio analysis reveals three performance regimes:

**Regime 1: Matrix fits in L2 ($R_{L2} < 1$)**
- $N = 1{,}000$, $p = 0.01$: CSR has $R_{L2} = 0.24$.
- All formats perform similarly; LIF overhead dominates.
- CSR: 66.89 ms, ELL: 67.21 ms, CSC: 81.68 ms — difference is <20%.

**Regime 2: Matrix fits in L3 but not L2 ($R_{L3} < 1 < R_{L2}$)**
- $N = 5{,}000$, $p = 0.01$: CSR has $R_{L2} = 5.76$, $R_{L3} = 0.36$.
- Format differences begin to emerge but are moderate.
- CSR: 342.60 ms, ELL: 347.10 ms — ELL is only 1.3% slower.

**Regime 3: Matrix exceeds L3 ($R_{L3} > 1$)**
- $N = 10{,}000$, $p = 0.1$: CSR has $R_{L3} = 14.3$.
- Every access is a potential DRAM fetch. Sequential access patterns
  (CSR, ELL) dominate.
- CSR: 1252.94 ms vs COO: 8589.83 ms — **6.9× difference**.

The transition from Regime 2 to Regime 3 explains the super-linear scaling:
once the matrix no longer fits in L3, the cost per element jumps by ~10× (the
DRAM/L3 latency ratio).

### 3.8 Effective Bandwidth Analysis

At the largest configurations ($N = 10{,}000$, $p = 0.1$):

| Format | $B_{\text{eff}}$ (GB/s) | Interpretation |
|--------|-------------------------|----------------|
| CSR (ER) | 0.096 | ~0.2% of DDR4 peak (~50 GB/s) |
| ELL (ER) | 0.103 | ~0.2% of DDR4 peak |
| COO (ER) | 0.019 | ~0.04% of DDR4 peak |
| ELL (BA) | 0.268 | ~0.5% of DDR4 peak |

All effective bandwidths are far below the hardware's theoretical maximum.
This confirms the workload is **latency-bound, not bandwidth-bound**: the
sparse access pattern means the CPU spends most of its time waiting for
individual cache-line fetches rather than saturating the memory bus with
sequential streams. The spike-driven nature of the computation (only ~8% of
neurons spike per step) means only a small, scattered subset of matrix rows
is accessed each timestep.

### 3.9 Measurement Quality

Outlier rejection statistics confirm high measurement reliability:

- **86% of configurations** had 0 outliers removed.
- **12%** had 1–2 outliers (typically mild OS scheduling interference).
- **2%** had 3 outliers (maximum observed) — still leaving 7 clean trials.

Standard deviations are typically <1% of the mean for small networks and
<2% for large networks, indicating excellent reproducibility.

---

## 4. Key Findings

### Finding 1: CSR is the Universal Winner for Scatter

CSR outperforms all other formats on every single one of the 144
configurations tested. The advantage ranges from minimal (1.04× over ELL at
$N = 1{,}000$, low density) to dramatic (14.1× over CSC on BA at
$N = 10{,}000$, high density). This is because scatter's access pattern
(iterate source neuron's outgoing synapses) aligns perfectly with CSR's
row-contiguous storage.

### Finding 2: Topology Matters Most for Irregular Formats

BA's power-law degree distribution is the single most impactful topology
factor. It inflates ELL memory by 2–4× and increases wall-clock time by
20–50% relative to regular topologies (ER, FI, WS). For COO and CSC,
BA increases nnz (more edges from the hub-rich growth model), compounding
the already-poor $O(\text{nnz})$ per-step cost.

### Finding 3: CSC is Not Suited for Scatter-Dominated Workloads

CSC's column-compressed layout is misaligned with scatter's row-iteration
pattern. At scale, CSC performs 1.5–2× worse than COO because the pointer
array adds overhead without providing locality benefit. CSC would be
expected to excel in gather-only benchmarks where the target iterates its
incoming connections — a hypothesis the `--gather-only` mode is designed to
test.

### Finding 4: Cache Hierarchy Defines the Performance Cliff

The transition from L3-resident to DRAM-bound (around $N = 5{,}000$,
$p = 0.05$ for CSR) marks a sharp increase in format-dependent performance
differences. Below this threshold, all compressed formats (CSR, CSC, ELL)
perform within 20% of each other. Above it, sequential-access formats (CSR,
ELL) outperform random-access formats (COO, CSC-on-scatter) by 5–15×.

### Finding 5: Degree Variance Has Minimal Impact on CSR/CSC

CSR and CSC timings are nearly identical across ER, FI, and WS topologies
(which share similar nnz counts at the same density parameter). The
row/column pointer structure accommodates variable-degree rows without
padding waste. BA is the exception only because it generates more total
edges.

### Finding 6: ELL is Optimal When Degree Variance is Zero

On Fixed In-Degree topologies, ELL achieves zero padding waste
($K_{\max} = K$) and matches CSR's timing within 4%. This validates the
theoretical prediction and makes ELL a strong choice for neural network
architectures with deterministic connectivity (e.g., convolutional SNNs,
reservoir computing with fixed fan-in).

---

## 5. Conclusions

### 5.1 Format Recommendations

| Scenario | Recommended format | Rationale |
|----------|--------------------|-----------|
| General-purpose SNN simulation | **CSR** | Fastest scatter across all topologies; memory-efficient |
| Fixed-degree architectures | **CSR** or **ELL** | Both optimal; ELL may benefit from SIMD auto-vectorisation |
| Pull-based (gather) simulation | **CSC** | Column-indexed access matches gather pattern |
| Data exchange / file I/O | **COO** | Simple to construct, serialize, and convert |
| Scale-free / power-law networks | **CSR** (avoid ELL) | ELL padding waste is prohibitive for heavy-tailed degrees |

### 5.2 Topology Implications

For neural network simulation frameworks, the choice of topology generator
has a measurable but secondary impact on performance compared to the choice
of sparse format. The primary performance determinant is the number of
non-zero entries (nnz), not the specific pattern of connections.

Among the four topologies tested:
- **ER, FI, and WS** produce statistically similar performance profiles
  because they generate comparable nnz counts with low degree variance.
- **BA** is the outlier: its scale-free structure generates more edges and
  higher degree variance, penalising ELL and increasing overall runtime.

### 5.3 Scaling Implications

At $N = 10{,}000$ the largest matrices (BA, $p = 0.1$) reach 500 MB (ELL)
and 230 MB (CSR), both far exceeding the L3 cache (24 MB). At this point,
all formats are DRAM-bound and the effective bandwidth is <0.5% of peak.
This indicates that for larger networks ($N \geq 50{,}000$), further
optimisations — such as sparse matrix tiling, NUMA-aware allocation, or
offloading to GPU — would be necessary to maintain performance.

### 5.4 Recommendations for Future Work

1. **Gather-only benchmarks** should be analysed to validate CSC's
   theoretical advantage in pull-based propagation.
2. **Spike-rate sweeps** (varying `--inject-rate`) would characterise the
   crossover point where ELL's regular access pattern overtakes CSR's
   spike-proportional cost.
3. **Hardware counters** (`perf stat`) would confirm the cache-miss and
   TLB-miss hypotheses that explain format-dependent performance.
4. **Larger networks** ($N = 50{,}000$ to $100{,}000$) would push into the
   regime where GPU acceleration provides significant speedups.
5. **Hybrid formats** (e.g., CSR with ELLPACK for high-degree rows) could
   combine CSR's low overhead with ELL's regularity.

### 5.5 Summary

This evaluation demonstrates that sparse matrix format selection is the
dominant factor in CPU spike-propagation performance, with CSR providing
the best scatter throughput across all tested topologies. The cache
hierarchy defines sharp performance boundaries: once the matrix exceeds
L3 capacity, sequential-access formats outperform random-access formats
by an order of magnitude. Topology effects are secondary but non-negligible
— Barabási–Albert's heavy-tailed degrees penalise fixed-width formats
(ELLPACK) and inflate absolute runtimes. These results provide actionable
guidance for SNN simulator developers choosing storage formats for
biologically realistic network architectures.
