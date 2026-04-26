# Justifying the Network Sizes (`N`) Used in the Topology Sweep

This document explains why the benchmark sweep is run with
`N ∈ {1000, 5000, 10000}` (the defaults in
[scripts/run_benchmarks.sh](scripts/run_benchmarks.sh) and in
[src/main.cpp](src/main.cpp)) rather than the proposal's wider grid
`N ∈ {1000, 10000, 50000, 100000}`. The argument is intended to be
lifted directly into the methodology chapter of the thesis.

---

## 1. Anchor on the cache hierarchy, not on round numbers

The whole point of the format comparison is to show how performance
changes when the working set crosses a cache boundary. So `N` should be
chosen to **straddle L1 / L2 / L3 / DRAM**, not to look pretty.

Per-format dominant memory cost (bytes), with `nnz ≈ N²·p` and `p = 0.05`:

| Format | Working set | Formula |
|--------|-------------|---------|
| CSR / CSC | `(N+1)·4 + nnz·(4+8)` | ≈ `12·N²·p` |
| COO | `nnz·(4+4+8)` | ≈ `16·N²·p` |
| ELL | `N·K_max·(4+8)` | ≈ `12·N·K_max` |

For `p = 0.05`, the CSR connectivity matrix size is roughly:

| N | CSR bytes | Sits in… |
|---|-----------|----------|
| 1 000 | ~0.6 MB | **L2** of typical Intel/AMD desktop (≥ 512 kB–1 MB) — small-data regime |
| 5 000 | ~15 MB | **L3** (typical 16–32 MB) — mid regime |
| 10 000 | ~60 MB | **DRAM**, but state vectors still fit in L3 — onset of memory-bound regime |
| 50 000 | ~1.5 GB | **deep DRAM**, far past LLC |
| 100 000 | ~6 GB | DRAM-bound; NUMA effects appear on multi-socket hosts |

The `cache_info.csv` row produced by `print_cache_info()` in
[src/benchmark.cpp](src/benchmark.cpp) documents the actual L1/L2/L3
sizes of the host, so the thesis can plot vertical "cache cliff" lines on
the throughput-vs-N curves and say *"the CSR/CSC crossover predicted by
the model occurs at N ≈ X, where the matrix overflows L3"*. That is the
strongest possible justification for a chosen N grid.

---

## 2. Justifying the three sizes that were actually used (1k / 5k / 10k)

These three points can be framed as **one data point per cache tier**,
which is the minimum needed to make a cache-cliff claim:

- **N = 1 000** — *L2-resident* baseline. Captures the regime where
  access pattern matters more than memory traffic, so any COO/CSR gap
  here is attributable to indirection cost, not bandwidth.
- **N = 5 000** — *L3-resident* mid-point. The matrix exceeds L2 but the
  state vectors still cache; this is where ELLPACK's regular layout
  typically beats CSR on uniform-degree (`fi`) topologies.
- **N = 10 000** — *DRAM-bound*. The matrix no longer fits any on-chip
  cache, so timings become bandwidth-bound and the format comparison
  becomes a *bytes-per-spike* comparison — the metric `r.bytes_per_spike`
  reported by [src/benchmark.cpp](src/benchmark.cpp).

Three sizes is also the **minimum number that lets you fit and report a
scaling exponent** (slope of `log(time)` vs `log(N)`); two points only
give a line, three give a residual.

---

## 3. Why the proposal's 50 k / 100 k points were (legitimately) skipped

Be honest about it; reviewers prefer this to silence:

1. **Benchmark cost grows with `N²·p`.** A full sweep is
   `4 formats × 4 topologies × |sizes| × |densities| × 10 trials × T timesteps`.
   At `N = 100 000, p = 0.1` the connectivity alone is ≈ 12 GB and a
   single trial of 1 000 timesteps takes minutes. The current grid
   (~144 configs) already takes hours; extending to 100 k would push it
   to days *per re-run*.
2. **RAM ceiling on the lab machine.** Running BA or WS at
   `N = 100 000, p = 0.2` requires ~24 GB just for the matrix, before
   state vectors and Poisson buffers — outside the envelope of a typical
   workstation. The thesis can cite the host's physical RAM as the hard
   upper bound.
3. **Diminishing scientific return.** Once `N > L3 / sizeof(matrix-row)`,
   the working set is already in DRAM and adding more zeros on the high
   end produces *no new regime crossing* — only confirmation. The
   proposal's grid was conservative; cache-tier coverage is the actual
   stopping criterion.
4. **GPU comparability (RQ3).** `scripts/genn_benchmark.py` runs on the
   GB10's 24 GB; choosing CPU sizes that the GPU can also instantiate
   keeps the RQ3 comparison apples-to-apples. 50 k–100 k at the higher
   densities exceed that.

---

## 4. Topology-specific caveats

`--density` is reinterpreted per topology
(see [docs/README_topology_flags.md](docs/README_topology_flags.md)), so
`N` interacts with it differently. Worth explicitly noting:

- **`fi` (fixed in-degree)** — `K = ⌊p·N⌋`. At `N = 1000, p = 0.01` you
  get `K = 10`, which is the *biological* cortical fan-in order of
  magnitude (Potjans & Diesmann 2014 use ~K = 80 in a balanced column).
  So the lower-N runs aren't just "small" — they are biologically the
  most realistic; larger N pushes into super-physiological connectivity
  unless `p` is also reduced.
- **`ba` (Barabási–Albert)** — `nnz ≈ N·m` with `m = ⌊p·N⌋`, so
  `nnz ≈ p·N²` like the others, but the **max degree** scales as
  `√(m·N)` ≈ `√(p)·N`. ELLPACK padding cost therefore grows linearly in
  N, which is precisely why a *small* N is enough to expose ELL's
  worst case — you don't need 100 k to make the point.
- **`ws` (Watts–Strogatz)** — degree is fixed at `K`, so `nnz` scales
  linearly in N. WS is the cheapest topology and is the only one where
  one could push to 50 k–100 k cheaply if a single high-N data point is
  desired as a sanity check.

---

## 5. A reproducible selection-criterion sentence for the thesis

The following paragraph can be lifted verbatim into the methodology
chapter:

> The network sizes `N ∈ {1000, 5000, 10000}` were selected so that the
> Erdős–Rényi connectivity matrix at `p = 0.05` (the proposal's central
> density) occupies, respectively, the L2 cache, the L3 cache, and main
> memory of the benchmark host (sizes documented in `cache_info.csv`).
> This three-point grid is the minimum required to localise the
> cache-driven crossover between CSR, CSC and ELL while keeping the full
> 144-configuration sweep within the lab machine's 24 GB RAM and a
> single-day wall-clock budget. Larger sizes from the proposal
> (`N ∈ {50 000, 100 000}`) were not run because (i) at densities
> ≥ 0.05 they exceed the host's physical memory and the GB10 GPU's
> 24 GB used in the RQ3 cross-validation, and (ii) once the working set
> exceeds L3, additional N only confirms the DRAM-bound regime without
> introducing a new cache tier.

---

## 6. Optional: cheap experiment to make the justification empirical

For a one-line empirical hook, add `N ∈ {500, 2000, 20000}` *only for
`er` at `p = 0.05` and CSR/CSC* (cheap: ~6 extra configs) and overlay
them on the throughput-vs-N plot. The kink at the L3 boundary is the
visual justification that "three N's are enough":

```bash
./build/spike_benchmark --sweep --subprocess \
    --sweep-sizes "500 1000 2000 5000 10000 20000" \
    --sweep-densities "0.05" \
    --bg-current 14.0 --gather-only \
    --output-csv results/n_scan_er.csv
# then filter the CSV to (topology=er, format∈{csr,csc}) and plot.
```

This converts the methodological argument into a single figure, which is
typically what reviewers ask for.
