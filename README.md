# Spike-Propagation-SNN

Benchmarking sparse matrix storage formats for spiking neural network (SNN)
spike propagation, as described in the project proposal
*"Benchmarking Sparse Matrix Storage Formats for Spiking Neural Network Spike
Propagation"* (Daria Stetsenko, Spring 2026).

## Project overview

Spiking neural networks communicate through discrete spikes over connectivity
matrices that are typically 1–20 % non-zero — a classic sparse matrix problem.
Each timestep, every spiking neuron must notify its post-synaptic targets,
making connector traversal the dominant compute cost.

This project implements and benchmarks four standard sparse formats:

| Format | Description |
|--------|-------------|
| **COO** | Coordinate: `(row, col, value)` triples |
| **CSR** | Compressed Sparse Row: contiguous column indices per source neuron — optimal for **scatter** (push) |
| **CSC** | Compressed Sparse Column: contiguous row indices per target neuron — optimal for **gather** (pull) |
| **ELL** | ELLPACK: padded row storage with fixed width — GPU-friendly coalesced access |

The benchmark measures **wall-clock latency**, **peak RSS**, and (on Linux)
**cache-miss rates** via `perf stat`.

## Repository structure

```
Spike-Propagation-SNN/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── sparse_matrix.hpp   # Abstract SparseMatrix interface
│   ├── coo_matrix.hpp
│   ├── csr_matrix.hpp
│   ├── csc_matrix.hpp
│   ├── ell_matrix.hpp
│   ├── lif_neuron.hpp      # Leaky Integrate-and-Fire neuron population
│   └── network.hpp         # Topology generators
├── src/
│   ├── coo_matrix.cpp
│   ├── csr_matrix.cpp
│   ├── csc_matrix.cpp
│   ├── ell_matrix.cpp
│   ├── lif_neuron.cpp
│   ├── network.cpp
│   └── benchmark.cpp       # Main benchmark harness
├── tests/
│   └── test_formats.cpp    # Unit tests (no external framework required)
└── python/
    ├── plot_results.py     # Matplotlib visualisation
    └── run_sweep.py        # Full parameter sweep driver
```

## Requirements

* C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
* CMake ≥ 3.16
* Python ≥ 3.8 with `matplotlib` and `numpy` (for plots)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This produces:
* `build/benchmark`     – the main benchmark executable
* `build/test_formats`  – the unit-test binary

## Run unit tests

```bash
cd build
ctest --output-on-failure
# or directly:
./test_formats
```

## Run a single benchmark

```bash
./build/benchmark --N 1000 --p 0.05 --topology er --timesteps 1000 --trials 10 --output results.csv
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--N` | `1000` | Number of neurons |
| `--p` | `0.05` | Connection probability / density |
| `--topology` | `er` | `er` Erdős–Rényi, `fi` fixed in-degree, `ba` Barabási–Albert, `ws` Watts–Strogatz |
| `--timesteps` | `1000` | Simulation timesteps per trial |
| `--trials` | `10` | Number of repeated trials |
| `--output` | `results.csv` | CSV output file |

## Run the full parameter sweep

```bash
cd build
python3 ../python/run_sweep.py --binary ./benchmark --output sweep_results.csv
```

This iterates over all (N, p, topology) combinations from the proposal:
* N ∈ {10³, 10⁴, 5×10⁴, 10⁵}
* p ∈ {0.01, 0.05, 0.10, 0.20}
* topology ∈ {er, fi, ba, ws}

## Visualise results

```bash
python3 python/plot_results.py results.csv
```

Produces three PNG files:
* `scatter_gather_comparison.png` – mean latency per format/operation
* `memory_footprint.png`          – memory consumption per format
* `latency_vs_N.png`              – scaling curves (when multiple N present)

## Cache profiling (Linux)

```bash
perf stat -e L1-dcache-load-misses,LLC-load-misses,instructions \
    ./build/benchmark --N 10000 --p 0.05 --topology er
```

## LIF neuron model

The test harness uses a minimal Leaky Integrate-and-Fire neuron to generate
realistic spike patterns:

```
τ_m · dV/dt = −(V − V_rest) + R · I_syn
```

Solved with forward Euler; spikes occur when V ≥ V_thresh, followed by a
reset to V_reset and an absolute refractory period.

Default parameters: τ_m = 20 ms, V_rest = V_reset = −65 mV,
V_thresh = −50 mV, R = 1 MΩ, dt = 0.1 ms, refrac = 20 steps (2 ms).

## References

1. Brette et al. (2007). Simulation of networks of spiking neurons: a review of tools and strategies. *J. Comput. Neurosci.* 23(3):349–398.
2. Eppler et al. (2024). NEST 3.0. *Front. Neuroinf.* 3:12.
3. Knight & Nowotny (2021). PyGeNN. *Front. Neuroinf.* 15:659005.
4. Bell & Garland (2009). Implementing sparse matrix–vector multiplication on throughput-oriented processors. *SC'09*, ACM.
