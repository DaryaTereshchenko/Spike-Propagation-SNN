## AI Agent Implementation Plan

### Target Directory Structure
```
Spike-Propagation-SNN/
├── CMakeLists.txt
├── .gitignore
├── include/
│   ├── sparse_matrix.h        # Abstract interface
│   ├── coo_matrix.h
│   ├── csr_matrix.h
│   ├── csc_matrix.h
│   ├── ell_matrix.h
│   ├── topology.h             # Graph generators
│   ├── lif_neuron.h           # LIF population
│   ├── benchmark.h            # Benchmark harness
│   └── csv_io.h               # CSV import/export
├── src/
│   ├── coo_matrix.cpp
│   ├── csr_matrix.cpp
│   ├── csc_matrix.cpp
│   ├── ell_matrix.cpp
│   ├── topology.cpp
│   ├── lif_neuron.cpp
│   ├── benchmark.cpp
│   ├── csv_io.cpp
│   └── main.cpp               # CLI entry point
├── tests/
│   ├── CMakeLists.txt
│   ├── test_formats.cpp
│   ├── test_topology.cpp
│   └── test_lif.cpp
├── scripts/
│   ├── nest_export.py         # PyNEST → CSV
│   ├── genn_benchmark.py      # GPU validation
│   ├── plot_results.py        # Visualization
│   └── run_benchmarks.sh
└── results/
    └── .gitkeep
```

### Phase 1: Project Skeleton & Build System *(no dependencies)*
1. Create `CMakeLists.txt` — C++17, Release/Debug configurations, `-O2` for Release, `-Wall -Wextra`.
2. Create `.gitignore` — build/, results/*.csv, *.o.
3. Scaffold all empty header/source files so `cmake --build` succeeds.

### Phase 2: Sparse Matrix Format Library *(depends on Phase 1)*
4. Define abstract `SparseMatrix` base class with pure virtual methods:
   - `scatter(spike_sources, out_buffer)` — push synaptic input from spiking neurons to their targets
   - `gather(target, spike_sources) → double` — pull synaptic input for a target neuron
   - `memory_bytes() → size_t`
   - `num_rows()`, `num_cols()`, `num_nonzeros()`
5. Implement `COOMatrix` — store `row[], col[], val[]`; scatter must scan all nnz entries.
6. Implement `CSRMatrix` — store `row_ptr[N+1], col_idx[nnz], val[nnz]`; scatter is O(degree) per spiking row. Build from COO by sorting.
7. Implement `CSCMatrix` — store `col_ptr[N+1], row_idx[nnz], val[nnz]`; gather is O(degree) per target column.
8. Implement `ELLMatrix` — store `indices[N][max_nnz_per_row]`, `values[N][max_nnz_per_row]`, pad with -1 sentinel. `max_nnz_per_row` determined from data at construction.
9. Write `tests/test_formats.cpp` — small 5×5 matrix, verify all 4 formats produce identical scatter/gather results and correct `memory_bytes`.

### Phase 3: Topology Generators *(parallel with Phase 2)*
10. Implement in `topology.h/.cpp`:
    - `generate_erdos_renyi(N, p, seed)` — edge with probability *p*, returns COO triplets.
    - `generate_fixed_indegree(N, K, seed)` — exactly *K* incoming edges per neuron.
    - `generate_barabasi_albert(N, m, seed)` — preferential attachment, *m* edges per new node.
    - `generate_watts_strogatz(N, K, beta, seed)` — ring lattice + rewiring.
    - All use `<random>` with explicit seeds.
11. Write `tests/test_topology.cpp` — verify edge counts, degree distributions, structural invariants.

### Phase 4: LIF Neuron Model *(depends on Phases 2 & 3)*
12. Implement `LIFPopulation` class:
    - Params: $\tau_m$=20ms, $V_{\text{rest}}$=-65mV, $V_{\text{thresh}}$=-50mV, $V_{\text{reset}}$=-65mV, R=1.0, dt=1.0ms, $t_{\text{ref}}$=2ms.
    - `step(I_syn) → vector<int>` — update all neurons via forward Euler, return spike indices.
13. Write `tests/test_lif.cpp` — constant input spike rate test, zero-input decay test, refractory period test.

### Phase 5: Benchmark Harness *(depends on Phases 2–4)*
14. Implement `BenchmarkConfig` struct and `run_benchmark()` function:
    - Generate topology → build sparse format → create LIF population.
    - Time inner loop (1000 timesteps): compute `I_syn` via scatter, step LIF, collect spikes.
    - Record wall-clock (excl. construction), peak RSS via status.
    - Repeat 10 trials, compute mean ± std.
15. Implement `main.cpp` CLI with flags: `--format`, `--size`, `--density`, `--topology`, `--timesteps`, `--trials`, `--seed`, `--output-csv`, `--sweep`.
16. **Verify**: run small sweep (N=1000, p=0.05, ER, all 4 formats), confirm identical spike counts across formats and valid CSV output.

### Phase 6: CSV I/O for NEST Connectivity *(parallel with Phase 5)*
17. Implement `csv_io.h/.cpp` — `load_coo_from_csv()` and `save_coo_to_csv()`.
18. Write `scripts/nest_export.py` — build balanced E/I cortical column in PyNEST (8000 exc / 2000 inh, `iaf_psc_alpha`), extract connectivity via `nest.GetConnections()`, export as CSV.

### Phase 7: Plotting & Full Sweep *(depends on Phases 5 & 6)*
19. Create `scripts/run_benchmarks.sh` — build Release, run full sweep across all (format × N × density × topology) combinations, optionally wrap with `perf stat`.
20. Create `scripts/plot_results.py`:
    - Time vs N (grouped by format, subplots per topology)
    - Time vs density (grouped by format)
    - Memory vs N (grouped by format)
    - NEST biological connectivity comparison (bar chart)
    - Cache-miss heatmap (if perf data available)

### Phase 8: GeNN GPU Validation *(optional, independent)*
21. Create `scripts/genn_benchmark.py` — define LIF network in PyGeNN, test DENSE / SPARSE / BITMASK modes, compare GPU timings with CPU results.

### Phase 9: Cleanup
22. README.md with build instructions, usage, results summary.
23. Verify all tests pass, clean compile with `-Wall -Wextra`.

---

### Verification Checklist
1. `cmake --build build && ctest --test-dir build` — all unit tests pass
2. All 4 formats produce **identical spike counts** for a given (topology, seed) pair
3. Small sweep (N=1000, all formats, ER+BA) completes in < 5 min, produces valid CSV
4. Plots generate readable figures from the CSV data
5. NEST CSV loads into C++ without errors
6. `perf stat` shows cache counter output for at least one configuration

### Key Decisions
- **Scatter is primary benchmark** (models spike delivery); gather is secondary
- Weights stored as `double` for consistency with NEST
- ELL `max_nnz_per_row` determined from actual data at construction time
- Topology generators in C++ for speed; NEST export is a separate Python script
- GeNN validation is optional — requires CUDA GPU + GeNN installation

### Out of Scope
- Hybrid formats (HYB, BSR), multi-threaded parallelism, custom CUDA kernels, dynamic rewiring, synaptic plasticity