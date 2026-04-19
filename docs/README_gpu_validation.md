# GPU Validation with GeNN

> **Implementation Status (April 2026):** The GeNN validation script
> (`scripts/genn_benchmark.py`) is fully implemented and has been executed.
> The LIF parameters match the C++ implementation exactly (τ_m = 20 ms,
> V_rest = −65 mV, V_thresh = −50 mV, t_ref = 2 ms, I_offset = 14 mV).
> Spike-per-step counts are consistent across CPU and GPU, confirming
> behavioural equivalence.

## Overview

To compare CPU sparse-format performance against GPU-accelerated spike
propagation, we use **GeNN** (GPU-Enhanced Neuronal Networks), a
code-generation framework that compiles spiking neural network models
to optimised CUDA kernels.  The validation script
(`scripts/genn_benchmark.py`) defines a LIF network with identical
parameters to the C++ implementation and benchmarks three GPU connectivity
modes.

---

## GeNN Connectivity Modes

### DENSE

The full $N \times N$ weight matrix is stored in GPU global memory.
Spike propagation reads one full row per spiking neuron.

**GPU Memory:** $O(N^2 \cdot 4)$ bytes (float32).

**Characteristics:** Simple access pattern, high bandwidth utilisation,
but memory-prohibitive for large $N$.  Equivalent to a dense matrix–vector
multiply using CUDA cores.

### SPARSE

GeNN's sparse mode uses a CSR-like representation on the GPU: a row-pointer
array and a column-index array.  Only non-zero weights are stored.

**GPU Memory:** $O(N + \text{nnz})$ float32/int32 values.

**Characteristics:** Memory-efficient for low-density networks.  Irregular
access due to indirect indexing; warp divergence possible when neighbour
counts vary across neurons in the same warp.

### BITMASK

GeNN-specific mode that stores each potential connection as a single bit.
The weight is a single scalar shared by all connections (homogeneous
weights).

**GPU Memory:** $O(N^2 / 8)$ bytes.

**Characteristics:** Extremely memory-efficient, but limited to binary
or uniform-weight connectivity.  Suitable for our benchmark since all
connections within a topology use the same normalised weight.

---

## Neuron Model Correspondence

The PyGeNN script uses GeNN's built-in `"LIF"` model with parameters
matching the C++ LIFParams:

| PyGeNN parameter | C++ equivalent | Value |
|------------------|----------------|-------|
| `TauM`           | `tau_m`        | 20.0 ms |
| `Vrest`          | `V_rest`       | −65.0 mV |
| `Vreset`         | `V_rest` (reset = rest) | −65.0 mV |
| `Vthresh`        | `V_thresh`     | −50.0 mV |
| `TauRefrac`      | `t_ref`        | 2.0 ms |
| `C`              | 1.0 (normalized) | 1.0 |

The synaptic model is `"StaticPulse"` with `"DeltaCurr"` post-synaptic
model (instantaneous current injection), matching our C++ delta-current
implementation.

---

## Profiling

### GPU Timing

Wall-clock time is measured around the `model.step_time()` loop using
Python's `time.perf_counter()`.  This captures the full GPU execution
including kernel launch overhead and implicit synchronisation at the end
of each step.

For more granular GPU profiling, wrap the script with NVIDIA tools:

```bash
# Nsight Systems timeline
nsys profile python3 scripts/genn_benchmark.py --size 5000

# Nsight Compute kernel analysis
ncu --set full python3 scripts/genn_benchmark.py --size 5000 --mode SPARSE
```

### Spike Recording

GeNN's built-in spike recording mechanism is enabled via
`pop.spike_recording_enabled = True`.  After simulation, spike counts are
pulled from the GPU to verify that the GPU model produces biologically
plausible activity.

---

## Usage

```bash
# Single run (all 3 modes)
python3 scripts/genn_benchmark.py --size 5000 --density 0.05

# Specific mode
python3 scripts/genn_benchmark.py --size 5000 --mode SPARSE

# Full sweep with CSV output
python3 scripts/genn_benchmark.py --sweep --output results/gpu_results.csv
```

### Requirements

- Python 3.8+
- PyGeNN >= 4.8.0 (`pip install pygenn`)
- NVIDIA CUDA toolkit + compatible GPU driver
- NumPy

### Setup

```bash
# Install PyGeNN (requires CUDA to be installed first)
pip install pygenn

# Verify installation
python3 -c "from pygenn import GeNNModel; print('PyGeNN OK')"
```

---

## Expected Results

| Mode | Typical speedup vs. CPU CSR | Memory scaling |
|------|----------------------------|----------------|
| DENSE | 10–50× for $N \leq 10\,000$ | $O(N^2)$ — GPU VRAM limited |
| SPARSE | 5–20× | $O(\text{nnz})$ — efficient |
| BITMASK | 20–100× | $O(N^2/8)$ — compact |

Speedups are highly dependent on GPU model, $N$, and density.  BITMASK
excels when the connectivity is dense enough that the bit-packed
representation fits in GPU cache.

### Files

| File | Purpose |
|------|---------|
| `scripts/genn_benchmark.py` | PyGeNN LIF benchmark with DENSE/SPARSE/BITMASK |
