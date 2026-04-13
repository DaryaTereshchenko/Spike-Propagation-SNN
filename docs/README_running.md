# Running Benchmarks — CPU and GPU

## CPU Benchmarks

```bash
# 1. Clone the repo
git clone https://github.com/DaryaTereshchenko/Spike-Propagation-SNN.git
cd Spike-Propagation-SNN

# 2. Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 3. Run tests
cd build && ctest --output-on-failure && cd ..

# 4. Single benchmark run
./build/spike_benchmark --format csr --topology er --size 5000 --density 0.05

# 5. Single run with background current (sustains ~5-8% firing rate)
./build/spike_benchmark --format csr --topology er --size 5000 --density 0.05 \
    --bg-current 14.0 --gather-only

# 6. Single run with controlled spike injection (5% fixed rate, LIF-independent)
./build/spike_benchmark --format csc --topology er --size 5000 --density 0.05 \
    --inject-rate 0.05 --gather-only

# 7. Full parameter sweep with background current + gather-only benchmark
./build/spike_benchmark --sweep --bg-current 14.0 --gather-only \
    --output-csv results/sweep_bg14.csv

# 8. Full sweep with per-config RSS (subprocess mode — recommended)
./build/spike_benchmark --sweep --subprocess --bg-current 14.0 --gather-only \
    --output-csv results/benchmark_results.csv

# 9. Spike-rate sweep (characterise cost vs. activity level)
./build/spike_benchmark --sweep --subprocess \
    --inject-rate 0.05 --gather-only \
    --sweep-sizes "1000 5000" --sweep-densities "0.05" \
    --sweep-rates "5 10 15 20 25 30" \
    --output-csv results/rate_sweep_results.csv

# 10. Full sweep WITH hardware counters (cache misses, IPC)
./scripts/run_benchmarks.sh --perf

# 11. Generate plots
pip install matplotlib pandas numpy
python3 scripts/plot_results.py
```

### CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--format {coo\|csr\|csc\|ell}` | Sparse matrix format | `csr` |
| `--topology {er\|fi\|ba\|ws}` | Network topology | `er` |
| `--size N` | Number of neurons | `1000` |
| `--density p` | Connection density / degree param | `0.05` |
| `--timesteps T` | Simulation timesteps | `1000` |
| `--trials R` | Repeat count | `10` |
| `--seed S` | Random seed | `42` |
| `--poisson-rate R` | External Poisson rate (spikes/neuron/step) | `15.0` |
| `--poisson-weight W` | External spike weight (mV) | `1.5` |
| `--coupling G` | Recurrent weight = G/sqrt(K) | `2.0` |
| `--bg-current I` | Background DC current per neuron (mV) | `0.0` |
| `--inject-rate F` | Fixed spike injection rate [0,1] (overrides LIF) | `0.0` |
| `--gather-only` | Run dedicated gather-only benchmark | (off) |
| `--output-csv FILE` | Write results as CSV | (none) |
| `--sweep` | Run full parameter sweep | (off) |
| `--sweep-sizes "..."` | Space-separated sizes to sweep | `"1000 5000 10000"` |
| `--sweep-densities "..."` | Space-separated densities | `"0.01 0.05 0.1"` |
| `--sweep-rates "..."` | Space-separated Poisson rates | (single rate) |
| `--subprocess` | Fork child process per config for accurate RSS | (off) |
| `--single-config` | Internal: run one config, print CSV to stdout | (off) |

**Output files:**
- `results/benchmark_results.csv` — scatter/gather timing, memory, spike counts
- `results/rate_sweep_results.csv` — spike-rate sweep data (if run)
- `results/perf_results.csv` — cache misses, instructions, cycles (if `--perf`)
- `results/*.png` — publication-quality plots

> **Note on `perf`:** If `perf stat` fails with a permissions error, run:
> ```bash
> sudo sysctl -w kernel.perf_event_paranoid=-1
> ```

---

## GPU Benchmarks (NVIDIA GB10)

```bash
# 1. Clone the repo
git clone https://github.com/DaryaTereshchenko/Spike-Propagation-SNN.git
cd Spike-Propagation-SNN

# 2. Verify CUDA is available
nvidia-smi
nvcc --version

# 3. Install PyGeNN and dependencies
pip install numpy pygenn

# 4. Verify GeNN installation
python3 -c "from pygenn import GeNNModel; print('PyGeNN OK')"

# 5. Single quick test (all 3 GPU modes)
python3 scripts/genn_benchmark.py --size 1000 --density 0.05 --timesteps 1000

# 6. Full GPU sweep with CSV output
python3 scripts/genn_benchmark.py --sweep --output results/gpu_results.csv

# 7. (Optional) Detailed GPU profiling with Nsight Systems
nsys profile -o results/gpu_profile python3 scripts/genn_benchmark.py --size 5000 --density 0.05

# 8. (Optional) Kernel-level analysis with Nsight Compute
ncu --set full python3 scripts/genn_benchmark.py --size 5000 --mode SPARSE
```

**Output files:**
- `results/gpu_results.csv` — GPU timing per mode (DENSE/SPARSE/BITMASK)
- `results/gpu_profile.nsys-rep` — Nsight timeline (optional)

---

## Comparing CPU vs GPU Results

Both CSVs share the same network size/density parameter grid. To compare them side-by-side after collecting both:

```bash
# Copy gpu_results.csv from GPU machine into results/ on CPU machine, then:
python3 -c "
import pandas as pd
cpu = pd.read_csv('results/benchmark_results.csv')
gpu = pd.read_csv('results/gpu_results.csv')

# CPU: best format per (N, density) config
cpu_best = cpu.groupby(['N','density'])['mean_time_ms'].min().reset_index()
cpu_best.columns = ['N','density','cpu_ms']

# GPU: pivot by mode
for mode in gpu['mode'].unique():
    g = gpu[gpu['mode']==mode][['N','density','time_ms']].copy()
    g.columns = ['N','density',f'gpu_{mode}_ms']
    cpu_best = cpu_best.merge(g, on=['N','density'], how='outer')

print(cpu_best.to_string(index=False))
"
```

The GB10 is a **Blackwell-architecture** GPU, so BITMASK mode should show particularly strong performance due to the large L2 cache.
