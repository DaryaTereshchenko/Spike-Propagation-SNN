# GeNN GPU Benchmark — Library, Topology, and Experiments

> **Implementation Status (April 2026):** GPU benchmarks have been executed
> on NVIDIA GB10 (Blackwell architecture, CUDA 13.0.2) using PyGeNN 5.4.0.
> 18 of 27 configurations completed (9 DENSE + 9 SPARSE); 9 BITMASK configs
> skipped due to a GeNN 5.4.0 array-reshape bug with non-power-of-two N.
> Results archived in `results/gpu_results.csv` and
> `results/gpu_benchmark_log_20260419_200640.txt`.

## 1. The GeNN Library

[GeNN](https://github.com/genn-team/genn) (GPU-Enhanced Neuronal Networks) is an
open-source C++/CUDA framework for simulating spiking neural networks on NVIDIA
GPUs.  The version used in this project is **PyGeNN 5.4.0**, the Python front-end
for GeNN 5.x, built from source with the CUDA back-end.

### 1.1 Code-Generation Pipeline

GeNN does **not** ship a pre-compiled GPU simulator.  Instead, it follows a
*code-generation* workflow:

1. **Model definition** — The user describes neuron populations, synapse
   populations, connectivity, and parameters through the PyGeNN Python API.
2. **Code generation** — GeNN emits tailored CUDA C++ kernels for every
   combination of neuron model, synapse model, and connectivity type.  Generated
   sources are placed in a per-model build directory (e.g.
   `bench_SPARSE_5000_0_CODE/`).
3. **Compilation** — GeNN invokes `make` (or MSVC on Windows) to compile the
   generated CUDA kernels into a shared library.
4. **Load & run** — The shared library is loaded into the Python process via
   `model.load()`.  Subsequent `model.step_time()` calls execute one simulation
   timestep entirely on the GPU.

This approach produces highly specialised kernels — there is no runtime
branching on neuron type or synapse format — at the cost of a one-time
compilation step per configuration.

### 1.2 Why GeNN?

| Feature | Benefit |
|---------|---------|
| Code-generation | Minimal GPU branching; kernel code matches the exact model |
| Multiple connectivity modes | DENSE, SPARSE (CSR-like), BITMASK — directly comparable to the CPU sparse formats |
| Built-in timing counters | Per-kernel GPU-side profiling (`timing_enabled = True`) without external tooling |
| Spike recording | On-device circular buffers; avoids per-timestep device→host transfers |
| Batch mode | Native support for batched simulations (set `batch_size`) |

### 1.3 Platform

| Component | Value |
|-----------|-------|
| System | NVIDIA DGX Spark |
| CPU | NVIDIA Grace (ARM aarch64) |
| GPU | NVIDIA GB10 |
| Driver | 580.142 |
| CUDA Toolkit | 13.0.2 (nvcc V13.0.88) |
| Python | 3.13.12 (Miniconda) |
| PyGeNN | 5.4.0 (built from source) |

> **Note:** GeNN 5.4.0 does not yet include SM 12.1 in its architecture table.
> A cosmetic warning is emitted and the runtime falls back to SM 12.0 parameters.
> All benchmarks execute correctly.

---

## 2. LIF Neuron Model on the GPU

The GPU benchmark uses the same Leaky Integrate-and-Fire (LIF) model as the
CPU implementation, ensuring an apples-to-apples comparison.

### 2.1 Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Membrane capacitance | $C$ | 1.0 | — |
| Membrane time constant | $\tau_m$ | 20.0 | ms |
| Resting potential | $V_\text{rest}$ | −65.0 | mV |
| Reset potential | $V_\text{reset}$ | −65.0 | mV |
| Threshold | $V_\text{thresh}$ | −50.0 | mV |
| Background current | $I_\text{offset}$ | 14.0 | mV |
| Refractory period | $\tau_\text{ref}$ | 2.0 | ms |

GeNN's built-in `LIF` neuron model expects **seven** parameters (including
`Ioffset`).  The background current of 14.0 mV places the resting equilibrium
at $V_\text{eq} = -51$ mV, just above threshold, sustaining spontaneous
activity without external Poisson input.

### 2.2 Synapse Model

- **Weight update:** `StaticPulse` — a single fixed weight `g` applied
  instantaneously on each pre-synaptic spike.
- **Post-synaptic model:** `DeltaCurr` — incoming current is added directly to
  the membrane potential (no exponential filtering).
- **Weight value:** $w = 1 / \sqrt{N \cdot p}$, where $N$ is the network size
  and $p$ the connection density.  This normalisation keeps the expected
  recurrent drive consistent across network sizes.

---

## 3. Network Topology Construction on the GPU

Unlike the CPU benchmark — which builds explicit COO/CSR/CSC/ELLPACK matrices
in user code — GeNN constructs the connectivity **on-device** during the
`model.load()` call, using a declarative connectivity initialiser.

### 3.1 Connectivity Initialiser

All GPU benchmarks use GeNN's `FixedProbabilityNoAutapse` initialiser:

```python
conn = init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": density})
```

This is equivalent to an **Erdős–Rényi** random graph:

- Each directed edge $(j \to i)$, $i \neq j$, is included independently with
  probability $p$ (the `density` parameter).
- Self-connections (autapses) are excluded.
- Expected edge count: $N(N-1) \cdot p$.
- The random seed is set per-trial (`model.seed = 42 + trial`) for
  reproducibility.

The initialiser runs entirely on the GPU during `model.load()`.  GeNN
internally generates per-thread random streams and builds the chosen storage
format in device memory — no host-side adjacency matrix is ever materialised.

### 3.2 Connectivity / Storage Modes

Three GeNN `SynapseMatrixType` modes were benchmarked:

| Mode | GPU Storage | Memory | Description |
|------|------------|--------|-------------|
| **DENSE** | Full $N \times N$ weight matrix | $O(N^2)$ | Every possible synapse stored; zero weights represent absent connections. Fast kernel dispatch (coalesced reads) but memory-intensive. |
| **SPARSE** | CSR-like compressed format | $O(N + \text{nnz})$ | Only non-zero synapses stored. GeNN maintains row pointers and column indices on-device, analogous to the CPU CSR format. |
| **BITMASK** | Bit-packed connectivity mask + shared weight | $O(N^2 / 8)$ | One bit per potential synapse. All existing synapses share the same weight. Memory-efficient for uniform-weight networks. |

> **BITMASK limitation:** In GeNN 5.4.0 BITMASK mode encounters an internal
> array reshape error when the neuron count is not a power of two.  All nine
> BITMASK configurations were skipped; the issue has been reported upstream.

### 3.3 Comparison with CPU Topology

| Aspect | CPU benchmark | GPU benchmark (GeNN) |
|--------|--------------|---------------------|
| Topologies | ER, Fixed In-Degree, Barabási–Albert, Watts–Strogatz | ER only (`FixedProbabilityNoAutapse`) |
| Storage formats | COO, CSR, CSC, ELLPACK | DENSE, SPARSE, BITMASK |
| Matrix construction | Host-side C++ generators | On-device GPU initialiser |
| Activity driver | Poisson input + background current | Background current only (`Ioffset = 14.0`) |

The GPU benchmark deliberately uses a single topology (ER) because the primary
goal is to measure **GPU storage-format performance**, not topology effects.
The ER model is the natural match for `FixedProbabilityNoAutapse` and produces
statistically equivalent graphs to the CPU ER generator at the same density.

---

## 4. Experimental Design

### 4.1 Parameter Sweep

The GPU sweep mirrors the CPU benchmark grid along three dimensions:

| Dimension | Values |
|-----------|--------|
| Network size $N$ | 1 000, 5 000, 10 000 |
| Connection density $p$ | 0.01, 0.05, 0.1 |
| Storage mode | DENSE, SPARSE (BITMASK skipped) |

Total configurations: $3 \times 3 \times 3 = 27$ attempted, **18 completed**
(9 BITMASK skipped).

### 4.2 Timing Protocol

Each configuration follows a strict protocol:

1. **Build & load** — GeNN generates CUDA code, compiles, and loads the model.
   GPU memory usage is sampled before and after via `nvidia-smi`.
2. **Warm-up** — 5 simulation timesteps are executed and discarded.  This
   ensures GPU caches, TLBs, and the CUDA runtime are in steady state.
3. **Timer reset** — GeNN's per-kernel timing accumulators are zeroed.
4. **Timed run** — 1 000 timesteps are executed.  Both wall-clock time
   (`time.perf_counter`) and GeNN kernel times are recorded.
5. **Spike retrieval** — Spike recording buffers are pulled from device memory.
   Total spike count and spikes-per-step are computed.
6. **Repeat** — Steps 1–5 are repeated for 10 independent trials (different
   random seeds).  Mean, standard deviation, and median are reported.
7. **Cleanup** — `model.unload()` frees GPU memory between trials.

### 4.3 GeNN Kernel Timing Breakdown

With `timing_enabled = True`, GeNN reports cumulative GPU kernel time across
all timesteps for each kernel category:

| Timer | Kernel(s) measured |
|-------|--------------------|
| `neuron_update_time` | LIF membrane integration + threshold detection |
| `presynaptic_update_time` | Spike propagation through synapses (main computational kernel) |
| `postsynaptic_update_time` | Post-synaptic current accumulation (zero for `DeltaCurr`) |
| `synapse_dynamics_time` | Continuous synapse dynamics (zero for `StaticPulse`) |
| `init_time` | One-time neuron/synapse state initialisation |
| `init_sparse_time` | One-time sparse connectivity generation |

### 4.4 Reproducibility

```bash
# Single configuration
CUDA_PATH=/usr/local/cuda python3 scripts/genn_benchmark.py \
    --size 1000 --density 0.05 --mode SPARSE \
    --timesteps 1000 --trials 10

# Full sweep (as executed)
CUDA_PATH=/usr/local/cuda python3 -u scripts/genn_benchmark.py \
    --sweep --timesteps 1000 --trials 10 \
    --output results/gpu_results.csv
```

Results are written to [`results/gpu_results.csv`](../results/gpu_results.csv)
(21 columns).  The full session log is archived at
[`results/gpu_benchmark_log_20260419_200640.txt`](../results/gpu_benchmark_log_20260419_200640.txt).
