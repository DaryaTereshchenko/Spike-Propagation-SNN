# Sparse Matrix Storage Formats for Spike Propagation in Spiking Neural Networks: A Comparative Analysis Across Network Topologies

---

## Abstract

Spiking neural networks (SNNs) are the computational backbone of the third generation of neural network models, where information is encoded in the precise timing of discrete spike events rather than continuous activation values. The core computational primitive of SNN simulation — spike propagation — reduces to a sparse matrix–vector multiplication (SpMV), where a connectivity (weight) matrix is multiplied by a binary spike indicator vector at every simulation timestep. The choice of sparse matrix storage format therefore has a direct and measurable impact on simulation performance. This work investigates a unified research question: **Which sparse matrix storage format is most suitable for spike propagation across biologically motivated spiking network architectures, and how do format-topology interactions govern performance and memory trade-offs in implemented SNN simulators?** To answer this question, we implement a controlled benchmarking framework in C++17 that evaluates four classical sparse formats — Coordinate (COO), Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), and ELLPACK (ELL) — across four network topologies drawn from computational neuroscience: Erdős–Rényi random graphs, fixed in-degree networks, Barabási–Albert scale-free networks, and Watts–Strogatz small-world graphs. The framework integrates a Leaky Integrate-and-Fire (LIF) neuron model, cache hierarchy detection, derived performance metrics, and cross-validation against the NEST simulator and GPU-accelerated GeNN framework. Results demonstrate that format performance is not universal but depends critically on the interaction between the storage layout's memory access pattern and the degree distribution of the network topology.

---

## 1. Introduction

### 1.1 Context and Motivation

Biological neural computation operates through discrete electrochemical impulses — action potentials, or *spikes* — that propagate along axonal projections to post-synaptic targets. Unlike the rate-coded activations of conventional artificial neural networks (ANNs), biological neurons communicate through the precise timing of these all-or-none events, enabling temporal coding, coincidence detection, and energy-efficient event-driven computation (Maass, 2002; Gerstner et al., 2014). Spiking neural networks (SNNs) — sometimes termed the *third generation* of neural networks (Maass, 1997) — model this spike-based communication explicitly, offering both biological fidelity and the potential for neuromorphic hardware implementations that exploit event-driven sparsity for orders-of-magnitude energy savings over conventional deep learning accelerators (Merolla et al., 2014; Davies et al., 2018).

The central computational bottleneck of SNN simulation is **spike propagation**: at each discrete timestep, the simulator must identify which neurons have fired and deliver their synaptic weights to all post-synaptic targets. For a network of $N$ neurons with connectivity described by a weight matrix $\mathbf{W} \in \mathbb{R}^{N \times N}$, this operation is formally:

$$
I_i = \sum_{j \in S} W_{ij}
$$

where $S$ is the set of neurons that spiked at the current timestep, and $I_i$ is the synaptic current delivered to neuron $i$. In vector form, this is $\mathbf{I} = \mathbf{W} \mathbf{s}$, where $\mathbf{s}$ is a binary spike indicator vector.

Biological neural networks are overwhelmingly **sparse**: cortical neurons typically have $10^3$–$10^4$ synaptic connections out of a potential $10^{10}$ targets, yielding connection densities on the order of $10^{-6}$–$10^{-4}$ (Abeles, 1991; Braitenberg & Schüz, 1998). Storing and traversing connectivity as a dense matrix is therefore both memory-prohibitive and computationally wasteful. Sparse matrix storage formats — which store only the non-zero entries — are essential for tractable SNN simulation.

However, not all sparse formats are equally suited to the spike propagation primitive. The specific **memory access pattern** induced by a format during the scatter (push) or gather (pull) operation interacts with the **degree distribution** of the network topology and the **CPU cache hierarchy** to determine actual runtime performance. This interaction is the central focus of the present work.

### 1.2 Research Question

This thesis addresses a single unified research question that subsumes three interconnected sub-questions:

> **Which sparse matrix storage format is most suitable for spike propagation in spiking neural networks across biologically motivated network architectures, and how do the structural properties of the network topology — degree distribution, regularity, and scale-free characteristics — interact with the format's memory access pattern to determine performance and memory efficiency in practice?**

This question unifies three constituent inquiries:

1. **Format comparison**: How do COO, CSR, CSC, and ELLPACK compare in wall-clock time, memory consumption, and cache behaviour for the spike propagation primitive?
2. **Topology dependence**: How does the network structure — specifically, the variance and shape of the degree distribution — modulate format performance?
3. **Simulator validation**: Do these findings generalise to established SNN simulators (NEST, GeNN), and how does CPU sparse-format performance compare to GPU-accelerated propagation?

### 1.3 Contributions

This work makes the following contributions:

1. A **controlled benchmarking framework** in C++17 that isolates the spike propagation kernel from other simulation overheads, enabling fair comparison of four sparse formats under identical conditions.
2. A **systematic experimental design** crossing 4 formats × 4 topologies × 3 network sizes × 3 connection densities, with robust statistical treatment (IQR outlier rejection, Bessel-corrected standard deviations, median reporting).
3. A **cache-aware analysis** that detects the hardware cache hierarchy at runtime and computes matrix-to-cache size ratios, directly linking format performance to memory hierarchy behaviour.
4. **Cross-validation** against the NEST reference simulator (CPU) and the GeNN code-generation framework (GPU), contextualising the results within the broader SNN simulation ecosystem.
5. Identification of **format-topology interaction effects** — in particular, the conditions under which ELLPACK's cache-friendly strided access compensates for its memory overhead, and the conditions under which it fails catastrophically (scale-free networks with power-law degree distributions).

---

## 2. Literature Overview

### 2.1 Spiking Neural Networks

Spiking neural networks are biophysically motivated computational models where neurons communicate via discrete spike events. Unlike conventional ANNs where activations are real-valued (rate-coded), SNNs encode information in the timing of individual spikes, enabling temporal coding strategies such as time-to-first-spike, spike-phase coding, and population synchrony (Gerstner & Kistler, 2002).

**Maass (1997)** established the theoretical framework for SNNs as the "third generation" of neural networks, proving that networks of spiking neurons with temporal coding are strictly more powerful than sigmoidal networks (second generation) and threshold circuits (first generation) in terms of computational complexity. This result motivates the study of SNN simulation efficiency: if SNNs can solve problems inaccessible to conventional ANNs, then the computational cost of simulating them becomes a first-order concern.

**Brunel (2000)** provided a foundational analytical treatment of random balanced networks of integrate-and-fire neurons, showing that Erdős–Rényi connectivity with balanced excitation and inhibition produces asynchronous irregular firing — a hallmark of cortical activity. Brunel's model is the template for the benchmark network used in this work, including the external Poisson drive that maintains network activity.

**Gerstner et al. (2014)** provide a comprehensive treatment of neuronal dynamics, from biophysical Hodgkin-Huxley models to simplified integrate-and-fire models, establishing the LIF neuron as the standard computational building block for large-scale network simulations. The simplicity of the LIF model — a single first-order ODE with a threshold reset — ensures that simulation runtime is dominated by the spike propagation kernel rather than per-neuron dynamics, making it the ideal choice for benchmarking sparse format performance.

### 2.2 Sparse Matrix Storage Formats

Sparse matrix storage is a well-studied topic in numerical linear algebra and high-performance computing (Saad, 2003; Davis, 2006). The four formats evaluated in this work represent the classical spectrum of space-time trade-offs:

**COO (Coordinate)** stores each non-zero as a `(row, col, value)` triplet. It is the simplest representation and serves as the universal interchange format — any sparse matrix can be constructed as COO and converted to other formats. However, COO provides no structural acceleration: both scatter and gather operations require a full scan of all $\text{nnz}$ entries, making it $O(\text{nnz})$ per timestep regardless of spike activity. Its random memory access pattern (indices appear in arbitrary order) defeats hardware prefetching, leading to poor cache behaviour.

**CSR (Compressed Sparse Row)** stores a row-pointer array of size $N+1$, a column-index array, and a values array, both of size $\text{nnz}$. The row-pointer enables $O(1)$ access to any row's non-zeros, making row-traversal operations — such as scatter from a spiking source neuron to all its targets — efficient at $O(\text{deg}(j))$ per spiking neuron $j$. CSR is the standard format in scientific computing libraries (Eigen, PETSc, SciPy) and the most widely used format in SNN simulators including NEST (Eppler et al., 2009). Its storage cost is $4(N+1) + 12 \cdot \text{nnz}$ bytes (with 4-byte integers and 8-byte doubles).

**CSC (Compressed Sparse Column)** is the column-oriented counterpart to CSR: it stores a column-pointer array of size $N+1$, a row-index array, and a values array. CSC enables $O(1)$ access to any column's non-zeros, making it the natural format for gather-based (pull) propagation, where each target neuron iterates over its incoming connections. CSC's scatter operation, however, requires iterating all columns to find spiking sources — an $O(\text{nnz})$ operation. CSC has the same storage cost as CSR.

**ELLPACK (ELL)** stores two dense 2D arrays of size $N \times K_{\max}$, where $K_{\max}$ is the maximum number of non-zeros in any row. Rows with fewer non-zeros are padded with sentinel values ($-1$ for indices, $0$ for weights). The resulting regular, strided memory layout is highly cache-friendly: scatter for a spiking neuron accesses a contiguous block of $K_{\max}$ entries at a known offset, enabling aggressive hardware prefetching. However, ELLPACK's memory cost is $12 \cdot N \cdot K_{\max}$ bytes — when the degree distribution has high variance (as in scale-free networks), $K_{\max}$ can be orders of magnitude larger than the mean degree, leading to catastrophic memory waste. ELLPACK was originally developed for vector processors (Grimes et al., 1979) and has found renewed interest in GPU computing due to its coalesced memory access pattern (Bell & Garland, 2009).

### 2.3 Network Topology Models

The performance of a sparse format for SNN simulation depends critically on the network topology — specifically, the **degree distribution** and its moments (mean, variance, maximum). This work evaluates four topology models chosen to span the spectrum from perfectly regular connectivity to power-law degree distributions:

**Erdős–Rényi (ER) random graphs** (Erdős & Rényi, 1959) are the simplest random network model: each directed edge exists independently with probability $p$. The degree distribution is Binomial (approximated by Poisson for large $N$), yielding moderate variance ($\sigma^2 = Np(1-p)$) and $K_{\max}$ growing as $O(\log N / \log \log N)$ above the mean. ER graphs serve as the null model for cortical connectivity (Brunel, 2000) and provide a baseline for format comparison.

**Fixed in-degree (FI) networks** assign every neuron exactly $K = \lfloor pN \rfloor$ incoming connections, sampled uniformly without replacement via the Fisher–Yates shuffle. The in-degree distribution is a point mass at $K$ — zero variance. This topology is used in the NEST cortical microcircuit model (Potjans & Diesmann, 2014) and is the **ideal case for ELLPACK**: because all rows have exactly $K$ non-zeros, no padding is wasted ($K_{\max} = K$).

**Barabási–Albert (BA) scale-free networks** (Barabási & Albert, 1999) are generated by preferential attachment: each new node connects to $m$ existing nodes with probability proportional to their current degree. This produces a power-law degree distribution $P(k) \sim k^{-3}$, with "hub" neurons whose degree can be orders of magnitude larger than the mean. Hub neurons are observed in cortical "rich-club" organisation (van den Heuvel & Sporns, 2011). BA graphs are the **worst case for ELLPACK**: $K_{\max}$ scales with network size, and every row must be padded to the hub degree, leading to memory explosion (up to 8.8× the memory of CSR in our experiments).

**Watts–Strogatz (WS) small-world networks** (Watts & Strogatz, 1998) start from a ring lattice with $K$ nearest-neighbour connections, then rewire each edge with probability $\beta$ to a random target. The resulting graphs have high clustering (local structure) with short path lengths (global efficiency) — the "small-world" property observed in cortical connectivity. The degree distribution is approximately regular (similar to FI, with small variance from rewiring), making WS networks a good candidate for ELLPACK.

### 2.4 SNN Simulators

**NEST** (NEural Simulation Tool; Gewaltig & Diesmann, 2007; Eppler et al., 2009) is a widely-used open-source simulator for large-scale spiking networks. NEST uses a CSR-like representation for synaptic connectivity and distributes computation across multiple MPI processes and threads. Our benchmark validates against NEST by importing NEST-generated connectivity matrices (balanced E/I columns with 10,000 neurons following Potjans & Diesmann, 2014) via CSV export.

**GeNN** (GPU-Enhanced Neuronal Networks; Yavuz et al., 2016; Knight et al., 2021) is a code-generation framework that compiles SNN model descriptions into optimised CUDA kernels. GeNN supports three connectivity modes — DENSE (full matrix), SPARSE (CSR-like), and BITMASK (bit-packed) — each with different GPU memory and access characteristics. We use GeNN as a GPU reference point to contextualise CPU sparse-format performance.

### 2.5 Cache Hierarchy and Memory Performance

Modern CPUs employ a multi-level cache hierarchy to hide main memory latency. A typical configuration consists of: L1 data cache (32–64 KB, ~4 cycle latency), L2 unified cache (256 KB–2 MB, ~12 cycles), L3 shared cache (4–64 MB, ~40 cycles), and DRAM (~200 cycles). When a sparse matrix fits entirely in L1, scatter/gather operations run at near-processor speed. When it spills beyond L3, every access incurs a ~50× penalty relative to L1.

The **cache ratio** — the ratio of matrix memory footprint to cache level size — is a simple predictor of performance cliffs. If $R_{Lk} = M_{\text{matrix}} / C_{Lk} > 1$, the matrix exceeds cache level $k$ and performance is expected to degrade. This metric, computed at runtime by our benchmark, directly explains the observed performance ordering: at $N = 10{,}000$, $d = 0.1$, the CSR matrix for Erdős–Rényi connectivity has an L2 ratio of 57.2× and an L3 ratio of 4.77× — deep into DRAM territory — yet CSR's sequential row access pattern enables hardware prefetching that partially compensates.

Bell & Garland (2009) and Williams et al. (2007) provide detailed analyses of sparse format performance on GPUs and multicore CPUs, respectively, establishing that memory access regularity (stride, locality) often matters more than algorithmic complexity for in-practice performance.

---

## 3. Key Terms and Definitions

This section provides precise definitions for all technical concepts used throughout the thesis.

### 3.1 Neuroscience Terms

**Spike (action potential):** A discrete, all-or-none electrical impulse (~1 ms duration, ~100 mV amplitude) that propagates along a neuron's axon. In computational models, spikes are represented as binary events: neuron $i$ either fires ($s_i = 1$) or does not ($s_i = 0$) at each timestep.

**Synapse:** The junction between a pre-synaptic neuron (sender) and a post-synaptic neuron (receiver). Each synapse has an associated *weight* $W_{ij}$ that determines the magnitude of current delivered to neuron $i$ when neuron $j$ fires.

**Synaptic current ($I_{\text{syn}}$):** The total input current arriving at a neuron from all its pre-synaptic sources that fired at the current timestep: $I_{\text{syn},i} = \sum_{j \in S} W_{ij}$.

**Membrane potential ($V$):** The electrical potential difference across a neuron's cell membrane. In the LIF model, $V$ integrates synaptic input over time and triggers a spike upon reaching the threshold $V_{\text{thresh}}$.

**Refractory period ($t_{\text{ref}}$):** The interval after a spike during which a neuron cannot fire again. This models the biological recovery of sodium/potassium ion channels and prevents unrealistically high firing rates.

**Firing rate:** The fraction of neurons that spike per timestep. Biologically realistic cortical firing rates are 1–10 Hz (0.1–1% per ms timestep). Our benchmark targets 4–8% per timestep with background current injection, representing a high-activity regime that stresses the spike propagation kernel.

### 3.2 Sparse Matrix Terms

**Non-zero (nnz):** The number of non-zero entries in the sparse matrix — equivalently, the number of synaptic connections in the network.

**Connection density ($p$, $d$):** The fraction of possible connections that exist: $d \approx \text{nnz} / N^2$. For ER graphs, $d$ is the per-edge inclusion probability. For other topologies, $d$ parameterises the degree (e.g., $K = \lfloor dN \rfloor$ for fixed in-degree).

**Degree distribution:** The probability distribution of the number of connections per neuron. In-degree is the number of incoming connections; out-degree is the number of outgoing ones. The degree distribution's **variance** and **maximum** ($K_{\max}$) are critical determinants of ELLPACK efficiency.

**Scatter (push-based propagation):** The operation of iterating over spiking neurons and distributing ("scattering") their synaptic weights to all post-synaptic targets. For neuron $j$ that spiked: for each target $i$ connected to $j$, add $W_{ij}$ to $I_{\text{syn},i}$.

**Gather (pull-based propagation):** The operation of iterating over all target neurons and collecting ("gathering") incoming weights from spiking sources. For each target $i$: sum $W_{ij}$ over all $j \in S$ that are connected to $i$.

**Row-pointer ($\text{row\_ptr}$):** In CSR format, an array of size $N+1$ where $\text{row\_ptr}[i]$ gives the starting index of row $i$'s non-zeros in the column-index and values arrays. Enables $O(1)$ random access to any row.

**Padding (sentinel):** In ELLPACK, rows shorter than $K_{\max}$ are filled with dummy entries (index $= -1$, value $= 0$) to maintain the regular $N \times K_{\max}$ array shape. Padding entries are skipped during computation but still consume memory and cache space.

### 3.3 Performance Terms

**Wall-clock time:** The elapsed real time for the simulation loop, measured via `std::chrono::high_resolution_clock`. Excludes topology generation and format construction (one-time costs).

**Peak RSS (Resident Set Size):** The maximum physical memory used by the process, read from `/proc/self/status` (`VmHWM`) on Linux. Represents the true memory footprint including the sparse matrix, neuron state, and all auxiliary buffers.

**Cache miss rate:** The fraction of memory accesses that miss at a given cache level, requiring data to be fetched from a higher (slower) level. Measured via hardware performance counters (`perf stat`).

**Effective bandwidth (GB/s):** Upper-bound estimate of memory throughput: $B_{\text{eff}} = M_{\text{matrix}} / (t_{\text{mean}} \times 10^{-3}) \times 10^{-9}$. Comparing this to hardware peak bandwidth reveals whether the workload is memory-bound or latency-bound.

**Scatter throughput (edges/ms):** The number of synaptic connections processed per millisecond during push-based propagation: $S = (\bar{s} \times \bar{d}_{\text{out}} \times T) / t_{\text{scatter}}$, where $\bar{s}$ is the mean spikes per step and $\bar{d}_{\text{out}}$ is the mean out-degree.

**IQR outlier rejection:** A robust statistical method for removing measurement outliers. Trials outside $[Q_1 - 1.5 \cdot \text{IQR},\; Q_3 + 1.5 \cdot \text{IQR}]$ are excluded before computing mean and standard deviation, where $Q_1$ and $Q_3$ are the 25th and 75th percentiles and $\text{IQR} = Q_3 - Q_1$.

---

## 4. Experimental Architecture

### 4.1 Design Rationale

The experimental framework is designed around a single principle: **isolate the sparse format's access pattern as the independent variable** while controlling for all other factors — neuron model, network activity, random seed, compiler optimisations, and measurement methodology. This requires:

1. **A common abstract interface** (`SparseMatrix`) through which all formats expose identical `scatter()` and `gather_all()` operations, ensuring that the benchmark loop is format-agnostic.
2. **A lightweight neuron model** (LIF) whose per-neuron update cost is negligible relative to the spike propagation cost, so that measured timing differences reflect format performance rather than neuron dynamics.
3. **Controlled network activity** via background current injection ($I_{\text{bg}}$) and controlled spike injection (`--inject-rate`), ensuring consistent spike rates across configurations and enabling activity-dependent analysis.
4. **Per-format memory instrumentation** via the `memory_bytes()` virtual method, providing exact storage footprints independent of process-level memory measurements.
5. **Cache-aware analysis** via runtime detection of L1d/L2/L3 sizes and computation of cache ratios per configuration.

### 4.2 System Architecture

The framework consists of five interconnected modules:

```
┌─────────────────┐     COOTriplets     ┌───────────────────┐
│  Topology        │ ──────────────────► │  Sparse Matrix     │
│  Generators      │                     │  (COO/CSR/CSC/ELL) │
│  (ER/FI/BA/WS)   │                     └────────┬──────────┘
└─────────────────┘                              │
                                                  │ scatter() / gather_all()
                                                  ▼
┌─────────────────┐     spike indices   ┌───────────────────┐
│  LIF Population  │ ◄─────────────────  │  Benchmark Harness │
│  (N neurons)     │ ──────────────────► │  (timing loop)      │
└─────────────────┘     I_syn vector    └────────┬──────────┘
                                                  │
                                                  ▼
                                         BenchmarkResult (CSV)
```

**Module 1: Topology Generators** produce a `COOTriplets` struct — three parallel vectors `(rows, cols, vals)` plus the matrix dimension $N$. Each generator implements a well-defined random graph model with explicit seed control for reproducibility. The density parameter $p$ is interpreted differently by each topology (edge probability for ER, in-degree fraction for FI, attachment count fraction for BA, nearest-neighbour fraction for WS), but always produces an $N \times N$ directed graph.

**Module 2: Sparse Matrix Formats** accept `COOTriplets` and construct their internal representation. The conversion cost (sorting for CSR/CSC, max-degree computation for ELL) is excluded from timing. All four formats implement the `SparseMatrix` abstract interface, ensuring the benchmark calls identical virtual methods.

**Module 3: LIF Population** implements the Leaky Integrate-and-Fire neuron model with forward Euler integration. The LIF model was chosen for three reasons:

- **Computational simplicity**: The per-neuron update is a single multiply-add plus a threshold comparison — $O(1)$ per neuron, $O(N)$ total. This ensures that the spike propagation kernel ($O(\text{nnz})$ or $O(|S| \cdot K)$) dominates runtime.
- **Biological fidelity**: Despite its simplicity, the LIF model captures membrane integration, threshold firing, and refractory periods — the essential dynamics of cortical neurons (Gerstner et al., 2014).
- **Standardisation**: The LIF model is the default neuron type in NEST (`iaf_psc_delta`/`iaf_psc_alpha`) and GeNN (`LIF`), enabling direct parameter-matched cross-validation.

The continuous-time LIF equation:

$$\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + R \cdot I_{\text{syn}}(t)$$

is discretised via forward Euler with $\Delta t = 1$ ms:

$$V^{(n+1)} = V^{(n)} + \frac{\Delta t}{\tau_m} \left[ -(V^{(n)} - V_{\text{rest}}) + R \cdot I_{\text{syn}}^{(n)} \right]$$

The stability condition $0 < \Delta t / \tau_m < 2$ is satisfied with $\Delta t / \tau_m = 1/20 = 0.05 \ll 2$.

**Module 4: Benchmark Harness** orchestrates each trial:
1. Generate topology → 2. Build sparse format → 3. Initialise LIF population → 4. Run timed loop ($T = 1000$ timesteps, $R = 10$ trials) → 5. Record statistics with IQR outlier rejection.

**Module 5: Cross-Validation** provides two independent reference points:
- **NEST integration**: A PyNEST script exports a balanced E/I cortical column (8,000 excitatory + 2,000 inhibitory neurons, fixed in-degree, Brunel-style parameters) as a CSV connectivity matrix, which is loaded into the C++ benchmark for format comparison on biologically structured connectivity.
- **GeNN integration**: A PyGeNN script defines an identical LIF network and benchmarks three GPU connectivity modes (DENSE, SPARSE, BITMASK), providing GPU speedup estimates for contextualisation.

### 4.3 Parameter Space

The full factorial experimental design sweeps:

| Dimension | Values | Count |
|-----------|--------|-------|
| Sparse format | COO, CSR, CSC, ELL | 4 |
| Network topology | Erdős–Rényi, Fixed In-Degree, Barabási–Albert, Watts–Strogatz | 4 |
| Network size ($N$) | 1,000; 5,000; 10,000 | 3 |
| Connection density ($d$) | 0.01, 0.05, 0.1 | 3 |
| **Total configurations** | | **144** |

Each configuration runs $T = 1{,}000$ timesteps over $R = 10$ trials, with IQR-based outlier rejection. The random seed is fixed at 42 for all configurations to ensure identical network instances across formats.

### 4.4 Activity Controls

A key design challenge is ensuring that the network sustains meaningful spiking activity throughout all 1,000 timesteps. Without external drive, weight normalisation ($w = G / \sqrt{K_{\text{avg}}}$) causes the network to fall silent after the initial seed spikes — the synaptic input is far too weak to bridge the 15 mV threshold gap ($V_{\text{thresh}} - V_{\text{rest}} = -50 - (-65) = 15$ mV). Silent networks yield degenerate benchmarks where only a single scatter call (at timestep 0) exercises the sparse format, and the remaining 999 timesteps measure only LIF update cost.

Three complementary mechanisms address this:

1. **Background current** ($I_{\text{bg}} = 14$ mV): A constant DC current injected into every neuron, placing the equilibrium potential at $V_{\text{eq}} = V_{\text{rest}} + I_{\text{bg}} = -51$ mV — just above threshold. Combined with external Poisson drive and recurrent input, this sustains a 5–8% firing rate.

2. **External Poisson drive** ($\lambda = 15$, $w_{\text{ext}} = 1.5$ mV): Each neuron receives $\text{Poisson}(\lambda)$ external spikes per timestep, providing stochastic background input that models thalamic and inter-area cortical input (Brunel, 2000).

3. **Controlled spike injection** (`--inject-rate $F$`): Overrides LIF spike output with a random fraction $F$ of neurons per timestep, enabling controlled cost-vs-activity experiments independent of network dynamics. This is essential for determining format crossover points (e.g., at what activity level does ELL's strided access outperform CSR's row-indexed access?).

### 4.5 Measurement Methodology

**Timing:** Each trial measures the wall-clock time of the full simulation loop (scatter + LIF integration + Poisson drive, for all $T$ timesteps) using `std::chrono::high_resolution_clock`. Format construction time is excluded because it is a one-time cost that depends on sorting algorithms rather than runtime access patterns.

**Robust statistics:** Raw trial times are filtered via the Interquartile Range method before computing mean and standard deviation. Bessel's correction ($\div (n-1)$) is applied for unbiased variance estimation. Median time is reported alongside mean as a second robust central tendency measure.

**Memory:** The `memory_bytes()` method returns the exact in-memory footprint of each format's internal arrays, enabling precise per-format comparison independent of process-level overhead. Peak RSS from `/proc/self/status` provides a complementary whole-process measurement.

**Cache analysis:** At startup, the benchmark reads the Linux sysfs interface (`/sys/devices/system/cpu/cpu0/cache/`) to detect L1d, L2, and L3 cache sizes. For each configuration, the matrix-to-cache ratio $R_{Lk} = M_{\text{matrix}} / C_{Lk}$ is computed and recorded.

**Derived metrics:** Seven performance metrics are computed from raw data:
- Cache ratios (L1d, L2, L3)
- Effective bandwidth (GB/s)
- Scatter throughput (edges/ms)
- Gather throughput (edges/ms)
- Bytes per spike
- Median time (ms)
- Outliers removed (count)

### 4.6 Justification of Design Choices

**Why C++17?** C++ provides deterministic memory layout, manual memory management (essential for measuring exact storage footprints), and template/polymorphism support for the abstract interface — without the garbage collection overhead of managed languages that would corrupt timing measurements.

**Why forward Euler?** Forward Euler is the simplest stable integrator for the LIF equation with our parameters ($\Delta t / \tau_m = 0.05$). Higher-order methods (Runge-Kutta, exact integration) would increase per-neuron cost, diluting the format-dependent signal in the timing data. For benchmarking purposes, $O(\Delta t)$ global error is acceptable.

**Why exclude construction time?** Format construction (sorting COO entries into CSR/CSC order, computing $K_{\max}$ for ELL) is a one-time cost amortised over many simulation steps in production simulators. Including it would conflate format conversion efficiency with runtime propagation performance.

**Why 10 trials with outlier rejection?** Long-running benchmarks on shared systems are susceptible to OS scheduling interference, thermal throttling, and DRAM refresh pauses. Without outlier rejection, a single corrupted trial (100–1000× above the true value) can dominate the arithmetic mean. IQR rejection eliminates these artefacts while preserving statistical validity.

**Why these four topologies?** The four topology models span the biologically relevant spectrum from regular ($K_{\max} \approx \bar{K}$: FI, WS) to highly irregular ($K_{\max} \gg \bar{K}$: BA), with ER as the statistical baseline. This spectrum is precisely what determines the ELLPACK padding overhead — the central format-topology interaction under investigation.

---

## 5. Sparse Matrix Formats: Detailed Analysis

### 5.1 COO (Coordinate List)

**Structure:** Three parallel arrays — `row[nnz]`, `col[nnz]`, `val[nnz]` — store each non-zero element as a triplet.

**Memory:** $M_{\text{COO}} = \text{nnz} \times (4 + 4 + 8) = 16 \cdot \text{nnz}$ bytes.

**Scatter implementation:** Linear scan of all nnz entries. For each entry $(r, c, v)$: if neuron $r$ is in the spiking set $S$, add $v$ to $\text{out}[c]$. Complexity: $O(\text{nnz})$ per timestep — independent of spike count.

**Gather implementation:** Same linear scan, but checking column membership in $S$. Complexity: $O(\text{nnz})$.

**Rationale for inclusion:** COO is the universal interchange format. Every topology generator outputs `COOTriplets`, and every other format is constructed from COO. Including it in benchmarks establishes a worst-case performance baseline. COO's simplicity also makes it trivially correct — if all four formats produce identical spike sequences from COO-initialised connectivity, format conversion is verified.

### 5.2 CSR (Compressed Sparse Row)

**Structure:** `row_ptr[N+1]` (row offsets), `col_idx[nnz]` (column indices), `values[nnz]` (weights).

**Memory:** $M_{\text{CSR}} = 4(N+1) + 12 \cdot \text{nnz}$ bytes. Approximately 25% less than COO.

**Scatter implementation:** For each spiking neuron $j \in S$, iterate over the contiguous range $[\text{row\_ptr}[j], \text{row\_ptr}[j+1])$ in `col_idx` and `values`, adding each weight to the corresponding target in `out_buffer`. Complexity: $O(\sum_{j \in S} \text{deg}_{\text{out}}(j))$ — proportional to the number of spiking neurons times their average degree, not to the total nnz.

**Gather implementation (gather_all):** For each target neuron $i$ (all $N$), iterate over row $i$ of the transposed view. Since CSR is row-compressed, gathering *incoming* connections requires either iterating all rows or using a CSR of the transposed matrix. Our implementation iterates columns, checking membership — effectively $O(\text{nnz})$.

**Rationale for inclusion:** CSR is the de facto standard in numerical computing and SNN simulators. It provides $O(1)$ row access, making scatter spike-proportional rather than matrix-proportional. This distinction — the core finding of this thesis — becomes critical at large scale where only a small fraction of neurons spike per timestep.

### 5.3 CSC (Compressed Sparse Column)

**Structure:** `col_ptr[N+1]` (column offsets), `row_idx[nnz]` (row indices), `values[nnz]` (weights).

**Memory:** Identical to CSR: $M_{\text{CSC}} = 4(N+1) + 12 \cdot \text{nnz}$ bytes.

**Scatter implementation:** Must iterate all columns (or all nnz) to find spiking sources — $O(\text{nnz})$, same as COO but with worse cache behaviour (column-oriented layout induces random access to `out_buffer` entries across non-contiguous column blocks).

**Gather implementation (gather_all):** For each spiking neuron $j \in S$, column $j$ gives all post-synaptic targets in $O(\text{deg}(j))$. This is the symmetric advantage: CSC is to gather what CSR is to scatter.

**Rationale for inclusion:** CSC provides the symmetric counterpart to CSR, enabling scatter-vs-gather comparison. The inclusion of both demonstrates that format suitability depends on the propagation direction (push vs. pull), a design choice faced by every SNN simulator.

### 5.4 ELLPACK (ELL)

**Structure:** Two dense 2D arrays of size $N \times K_{\max}$: `col_indices[N][K_max]` and `values[N][K_max]`. Rows shorter than $K_{\max}$ are padded with index $= -1$, value $= 0$.

**Memory:** $M_{\text{ELL}} = 12 \cdot N \cdot K_{\max}$ bytes. When $K_{\max} \approx \bar{K}$ (regular topologies), this is comparable to CSR. When $K_{\max} \gg \bar{K}$ (scale-free topologies), ELL can be several times larger.

**Scatter implementation:** For spiking neuron $j$, access row $j$ as a contiguous block of $K_{\max}$ entries at offset $j \times K_{\max}$. Skip padded entries (index $= -1$). Complexity: $O(|S| \times K_{\max})$.

**Cache advantage:** The regular stride-1 access pattern — $K_{\max}$ consecutive `(index, value)` pairs per spiking neuron, with each neuron's row at a predictable offset — enables hardware prefetchers to load the next cache line before it is needed. This is in contrast to CSR, where `row_ptr[j]` must be dereferenced to find row $j$'s starting location (one level of indirection).

**Rationale for inclusion:** ELLPACK represents the cache-optimised extreme of the format spectrum. Its performance advantage over CSR depends entirely on the degree distribution's variance: zero-waste for FI, minimal waste for WS, catastrophic waste for BA. This topology-dependent behaviour is a central finding.

---

## 6. Network Topologies: Biological Motivation and Format Implications

### 6.1 Erdős–Rényi (ER) — The Null Model

**Construction:** Each of the $N(N-1)$ possible directed edges is included independently with probability $p$.

**Degree distribution:** $\text{In-degree} \sim \text{Bin}(N-1, p) \approx \text{Poisson}(Np)$ for large $N$.

**$K_{\max}$ growth:** $O(\log N)$ above the mean — moderate tail.

**Weight normalisation:** $w = 1/\sqrt{Np}$ — prevents runaway excitation while allowing fluctuation-driven firing.

**Format implications:** ER's moderate degree variance makes it a middle ground for ELLPACK: some padding waste ($K_{\max} / \bar{K} \approx 1.1$–$1.3$ for our sizes), but not catastrophic. All formats are fairly tested.

**Biological relevance:** ER is arguably the simplest null model for cortical connectivity. While the real cortex exhibits structured connectivity (distance-dependent, laminar, modular), ER's statistical homogeneity makes it the canonical starting point (Brunel, 2000). Any format comparison that does not include ER is incomplete.

### 6.2 Fixed In-Degree (FI) — The Regular Network

**Construction:** Each neuron receives exactly $K = \lfloor pN \rfloor$ incoming connections, sampled without replacement via Fisher–Yates shuffle.

**Degree distribution:** In-degree is deterministic ($\delta_K$); out-degree is approximately $\text{Bin}(N, K/N)$.

**$K_{\max}$ for rows (outgoing):** Slightly above $K$ due to out-degree variability, but close to $K$.

**Weight normalisation:** $w = 1/K$ — exact input balance.

**Format implications:** The near-zero degree variance makes FI the ideal ELLPACK topology: $K_{\max} \approx K$, so padding overhead is less than 6% over CSR in our experiments.

**Biological relevance:** The cortical microcircuit model of Potjans & Diesmann (2014) — the most widely-cited NEST reference model — uses fixed in-degree connectivity. Our FI topology directly emulates this model.

### 6.3 Barabási–Albert (BA) — The Scale-Free Network

**Construction:** Preferential attachment starting from a fully-connected seed of $m$ nodes. Each new node connects to $m$ existing nodes with probability proportional to current degree.

**Degree distribution:** Power-law $P(k) \sim k^{-3}$ with hub neurons whose degree grows as $O(\sqrt{N})$.

**$K_{\max}$ growth:** $O(\sqrt{N})$ or higher — extreme tail. At $N = 10{,}000$, $d = 0.1$, the estimated $K_{\max} \approx 4{,}372$.

**Edge count:** BA produces $\approx 2Nm$ directed edges (both directions of each undirected edge), making it structurally denser than ER/FI/WS at the same nominal density parameter.

**Weight normalisation:** $w = 1.0$ (uniform) — BA's heterogeneous connectivity makes normalisation-by-degree ill-defined for hub nodes.

**Format implications:** BA is the **worst case for ELLPACK**: at $N = 10{,}000$, $d = 0.01$, ELL uses **7.3× more memory than CSR** because every row is padded to the hub degree. Despite this, ELL's runtime can still be competitive (strided access compensates for memory waste), though CSR achieves better memory-performance trade-offs. BA is the **stress test** for any sparse format claiming generality.

**Biological relevance:** While pure power-law connectivity is not observed in cortex, hub neurons appear in the connectome's "rich-club" organisation (van den Heuvel & Sporns, 2011), where a small number of highly-connected nodes provide backbone routing.

### 6.4 Watts–Strogatz (WS) — The Small-World Network

**Construction:** Ring lattice with $K = \max(2, \lfloor pN \rfloor)$ nearest-neighbour connections, with each edge rewired to a random target with probability $\beta = 0.3$.

**Degree distribution:** Approximately regular — the ring lattice has uniform degree, and rewiring introduces only small perturbations. The degree variance is much lower than ER and far lower than BA.

**$K_{\max}$ growth:** Slowly — close to $K$ with small deviations from rewiring.

**Weight normalisation:** $w = 1.0$ (uniform).

**Format implications:** WS's near-regular degree distribution makes it a strong candidate for ELLPACK (minimal padding), and our results confirm that ELL achieves its best performance on WS at large scale ($N = 10{,}000$, $d = 0.1$: 13.75 ms, the fastest single configuration in the entire sweep).

**Biological relevance:** Cortical networks exhibit the small-world property — high local clustering coefficient (reflecting cortical column structure) with short average path length (reflecting long-range inter-area projections) — as characterised by Watts & Strogatz (1998). The $\beta = 0.3$ rewiring probability balances these two properties.

---

## 7. Experimental Results Overview

This section provides a summary of the key empirical findings without a full discussion (the discussion of which format is "best" overall is deferred to future work). The complete numerical results are available in the companion documents.

### 7.1 Scatter Performance Ranking

At large scale ($N = 10{,}000$, $d = 0.1$), the consistent performance ordering across topologies is:

$$\text{ELL} \approx \text{CSR} \ll \text{COO} < \text{CSC}$$

CSR and ELL achieve 2–3× speedup over COO and 2–4× speedup over CSC. This ordering arises from the fundamental distinction between **spike-proportional** formats (CSR, ELL: $O(|S| \cdot K)$ scatter cost) and **matrix-proportional** formats (COO, CSC: $O(\text{nnz})$ scatter cost).

### 7.2 Format-Topology Interaction

The relative performance of CSR vs. ELL depends on topology regularity:

| Topology | Degree regularity | ELL vs. CSR (time) | ELL vs. CSR (memory) |
|----------|-------------------|---------------------|----------------------|
| Fixed In-Degree | Perfect | ELL faster (2.62× over COO vs. 2.39×) | 1.12× overhead |
| Watts–Strogatz | Near-regular | **ELL fastest overall** (13.75 ms) | 1.06× overhead |
| Erdős–Rényi | Moderate variance | ELL ≈ CSR | 1.12× overhead |
| Barabási–Albert | Power-law | ELL marginally faster | **2.30× overhead** |

**For regular topologies (FI, WS):** ELL is the optimal choice — minimal memory overhead with superior cache performance.

**For scale-free topologies (BA):** CSR provides the best memory-performance trade-off — competitive speed at 2.3× less memory than ELL.

### 7.3 Scaling Behaviour

Two distinct scaling classes emerge:

- **CSR and ELL** scale approximately linearly with $N$ (at fixed density), consistent with their spike-proportional scatter cost and the $O(N)$ LIF update.
- **COO and CSC** scale as $O(N^2)$ at fixed density, because $\text{nnz} = N^2 d$ and their scatter cost is proportional to nnz.

CSR and ELL are also **density-invariant** for scatter: at $N = 10{,}000$, CSR's time changes by a factor of only 0.92× from $d = 0.01$ to $d = 0.1$, while COO increases by 1.63× and CSC by 2.10×.

### 7.4 Cache Hierarchy Effects

At $N = 10{,}000$, $d = 0.1$, all formats exceed L3 cache capacity (cache ratios of 4.8× to 20.8×). Despite operating entirely from DRAM, CSR and ELL maintain competitive performance because:

- **CSR:** Sequential access within each row's `col_idx[]` and `values[]` arrays enables hardware prefetching.
- **ELL:** Perfectly strided access (row $j$'s data at offset $j \times K_{\max}$) enables even more aggressive prefetching.
- **COO and CSC:** Irregular access patterns defeat prefetching, resulting in higher effective memory latency.

### 7.5 Memory Footprint

CSR and CSC are the most memory-efficient formats, with identical footprints ($4(N+1) + 12 \cdot \text{nnz}$ bytes). ELL's overhead is topology-dependent: 6% over CSR for WS, 12% for ER, but up to **730% for BA at low density** ($N = 10{,}000$, $d = 0.01$).

---

## 8. Simulator Cross-Validation

### 8.1 NEST Reference

The NEST integration validates our benchmark against the most widely-used production SNN simulator. A balanced E/I cortical column (10,000 neurons: 8,000 excitatory, 2,000 inhibitory, fixed in-degree $C_E = 800$, $C_I = 200$, following Potjans & Diesmann, 2014) is exported from PyNEST as a CSV connectivity matrix and loaded into the C++ benchmark. This enables format comparison on biologically structured (non-random) connectivity while ensuring that the connectivity is identical to what NEST would simulate internally.

### 8.2 GeNN GPU Reference

The GeNN integration provides GPU context for the CPU format comparison. Three GPU connectivity modes — DENSE ($O(N^2)$ memory, simple access), SPARSE (CSR-like, $O(\text{nnz})$ memory), and BITMASK (1-bit per potential connection, $O(N^2/8)$ memory) — are benchmarked with identical LIF parameters. Expected GPU speedups are 5–100× depending on mode and problem size, highlighting that format choice on GPUs involves different trade-offs (warp divergence, coalesced access) than on CPUs (cache hierarchy, prefetching).

---

## 9. Summary of Format Suitability by Use Case

| Use case | Recommended format | Rationale |
|----------|-------------------|-----------|
| Scatter (push) on regular networks | **ELL** | Strided access, minimal padding waste |
| Scatter (push) on irregular networks | **CSR** | Spike-proportional cost without memory waste |
| Gather (pull) | **CSC** | Column-indexed, $O(K)$ per target |
| Memory-constrained systems | **CSR / CSC** | Smallest footprint ($12 \cdot \text{nnz} + 4(N+1)$) |
| GPU acceleration | **ELL** or **CSR** | Coalesced warps (ELL), standard SpMV (CSR) |
| Data interchange | **COO** | Trivial construction, universal converter |
| Biologically structured connectivity (mixed regularity) | **CSR** | Best general-purpose trade-off |

---

## 10. Conclusion and Outlook

This thesis demonstrates that the question "Which sparse matrix storage format is best for SNN simulation?" has no universal answer. Instead, **format suitability is determined by the interaction between the storage layout's memory access pattern and the network topology's degree distribution**. For networks with regular or near-regular connectivity (fixed in-degree, small-world) — which are precisely the topologies used by major computational neuroscience reference models — ELLPACK's cache-friendly strided access provides the best performance at minimal memory overhead. For networks with power-law degree distributions (scale-free, preferential attachment) — which model hub-dominated connectivity observed in the connectome — CSR provides the best memory-performance trade-off, avoiding the catastrophic padding overhead that ELLPACK incurs. COO and CSC are consistently slower for scatter-dominant workloads and should be relegated to data interchange (COO) or gather-optimised use cases (CSC).

These findings are directly relevant to the design of next-generation SNN simulators: a simulator that supports multiple sparse formats and selects automatically based on the degree distribution of the target network could achieve optimal performance across the full range of biologically motivated architectures. The controlled spike-injection mechanism developed in this work further enables systematic characterisation of format crossover points as a function of network activity — a critical parameter that varies by orders of magnitude between resting-state and stimulus-driven cortical activity.

---

## References

- Abeles, M. (1991). *Corticonics: Neural Circuits of the Cerebral Cortex.* Cambridge University Press.
- Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.
- Bell, N., & Garland, M. (2009). Implementing sparse matrix-vector multiplication on throughput-oriented processors. *Proceedings of SC '09*.
- Braitenberg, V., & Schüz, A. (1998). *Cortex: Statistics and Geometry of Neuronal Connectivity.* Springer.
- Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *Journal of Computational Neuroscience*, 8(3), 183–208.
- Davies, M., et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1), 82–99.
- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems.* SIAM.
- Eppler, J. M., et al. (2009). PyNEST: A convenient interface to the NEST simulator. *Frontiers in Neuroinformatics*, 2, 12.
- Erdős, P., & Rényi, A. (1959). On random graphs I. *Publicationes Mathematicae Debrecen*, 6, 290–297.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models.* Cambridge University Press.
- Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal Dynamics.* Cambridge University Press.
- Gewaltig, M.-O., & Diesmann, M. (2007). NEST (NEural Simulation Tool). *Scholarpedia*, 2(4), 1430.
- Grimes, R. G., Kincaid, D. R., & Young, D. M. (1979). ITPACK 2.0 User's Guide. CNA-150, University of Texas at Austin.
- Knight, J. C., Nowotny, T. (2021). Larger GPU-accelerated brain simulations with procedural connectivity. *Nature Computational Science*, 1, 136–142.
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659–1671.
- Merolla, P. A., et al. (2014). A million spiking-neuron integrated circuit with a scalable communication network and interface. *Science*, 345(6197), 668–673.
- Potjans, T. C., & Diesmann, M. (2014). The cell-type specific cortical microcircuit: Relating structure and activity in a full-scale spiking network model. *Cerebral Cortex*, 24(3), 785–806.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
- van den Heuvel, M. P., & Sporns, O. (2011). Rich-club organization of the human connectome. *Journal of Neuroscience*, 31(44), 15775–15786.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442.
- Williams, S., Oliker, L., Vuduc, R., Shalf, J., Yelick, K., & Demmel, J. (2007). Optimization of sparse matrix-vector multiplication on emerging multicore platforms. *Proceedings of SC '07*.
- Yavuz, E., Turner, J., & Nowotny, T. (2016). GeNN: A code generation framework for accelerated brain simulations. *Scientific Reports*, 6, 18854.
