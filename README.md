# Spike Propagation SNN — Sparse Format Benchmark

A C++17 benchmark framework for evaluating sparse matrix storage formats
in the context of spiking neural network (SNN) spike propagation.  Compares
**COO**, **CSR**, **CSC**, and **ELLPACK** formats across four network
topologies, measuring wall-clock time, memory consumption, and cache
behaviour.

## Quick Start

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure && cd ..

# Single benchmark
./build/spike_benchmark --format csr --topology erdos_renyi \
    --size 5000 --density 0.05 --timesteps 1000

# Full sweep
./build/spike_benchmark --sweep --output-csv results/benchmark_results.csv

# Plot results
python3 scripts/plot_results.py
```

## Project Structure

```
├── CMakeLists.txt              # Build system
├── include/
│   ├── sparse_matrix.h         # Abstract base class + COOTriplets
│   ├── coo_matrix.h            # COO format
│   ├── csr_matrix.h            # CSR format
│   ├── csc_matrix.h            # CSC format
│   ├── ell_matrix.h            # ELLPACK format
│   ├── topology.h              # Network topology generators
│   ├── lif_neuron.h            # LIF neuron model
│   ├── benchmark.h             # Benchmark harness
│   └── csv_io.h                # CSV import/export
├── src/
│   ├── main.cpp                # CLI entry point
│   ├── coo_matrix.cpp          # COO implementation
│   ├── csr_matrix.cpp          # CSR implementation
│   ├── csc_matrix.cpp          # CSC implementation
│   ├── ell_matrix.cpp          # ELLPACK implementation
│   ├── topology.cpp            # Topology generators
│   ├── lif_neuron.cpp          # LIF integration
│   ├── benchmark.cpp           # Benchmark loop + profiling
│   └── csv_io.cpp              # CSV I/O
├── tests/
│   ├── test_formats.cpp        # Sparse format unit tests
│   ├── test_topology.cpp       # Topology generator tests
│   └── test_lif.cpp            # LIF neuron tests
├── scripts/
│   ├── run_benchmarks.sh       # Automated sweep with perf stat
│   ├── plot_results.py         # Publication-quality plots
│   ├── nest_export.py          # PyNEST reference model export
│   └── genn_benchmark.py       # GPU validation with GeNN
├── docs/                       # Module documentation (for paper)
│   ├── README_sparse_matrix.md
│   ├── README_topology.md
│   ├── README_lif_neuron.md
│   ├── README_benchmark.md
│   ├── README_csv_io.md
│   ├── README_plotting.md
│   └── README_gpu_validation.md
└── results/                    # Output directory (git-ignored)
```

## Sparse Matrix Formats

| Format | Memory | Scatter Cost | Best For |
|--------|--------|-------------|----------|
| COO    | 3 × nnz arrays | O(nnz) | Interchange |
| CSR    | N+1 + 2×nnz | O(Σ deg_out) | Scatter (push) |
| CSC    | N+1 + 2×nnz | O(Σ deg_out) | Gather (pull) |
| ELL    | N × K_max × 2 | O(\|S\| × K) | Regular topologies |

## Network Topologies

- **Erdős–Rényi** — iid edge probability, Poisson degree distribution
- **Fixed in-degree** — deterministic in-degree via Fisher–Yates sampling
- **Barabási–Albert** — preferential attachment, power-law degree distribution
- **Watts–Strogatz** — small-world: ring lattice + random rewiring (β=0.3)

## Profiling

- **Built-in:** Peak RSS via `/proc/self/status` VmHWM
- **perf stat:** Cache misses, instructions, cycles (via `run_benchmarks.sh --perf`)
- **GPU:** GeNN DENSE/SPARSE/BITMASK modes with Nsight compatibility

## Requirements

- C++17 compiler (GCC 8+, Clang 7+)
- CMake 3.14+
- Python 3.8+ with matplotlib, pandas, numpy (for plotting)
- Optional: NEST 3.x (for reference export), PyGeNN 4.8+ (for GPU validation)

## Documentation

Detailed module descriptions are in `docs/`:

| Document | Topic |
|----------|-------|
| [docs/README_sparse_matrix.md](docs/README_sparse_matrix.md) | Storage formats, complexity analysis |
| [docs/README_topology.md](docs/README_topology.md) | Graph models, biological relevance |
| [docs/README_lif_neuron.md](docs/README_lif_neuron.md) | LIF equations, Euler discretisation |
| [docs/README_benchmark.md](docs/README_benchmark.md) | Methodology, sweep parameters, CSV schema |
| [docs/README_csv_io.md](docs/README_csv_io.md) | CSV format, NEST integration |
| [docs/README_plotting.md](docs/README_plotting.md) | Plot descriptions, interpretation |
| [docs/README_gpu_validation.md](docs/README_gpu_validation.md) | GeNN setup, GPU profiling |
