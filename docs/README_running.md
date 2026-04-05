# Running Benchmarks — CPU and GPU

## CPU Benchmarks (ARM64)

```bash
# 1. Clone the repo
git clone https://github.com/DaryaTereshchenko/Spike-Propagation-SNN.git
cd Spike-Propagation-SNN

# 2. Install dependencies
sudo apt update && sudo apt install -y build-essential cmake linux-tools-generic

# 3. Quick smoke test first
./scripts/run_benchmarks.sh --small

# 4. Full benchmark sweep (takes ~30-60 min)
./scripts/run_benchmarks.sh

# 5. Full sweep WITH hardware counters (cache misses, IPC)
./scripts/run_benchmarks.sh --perf

# 6. Generate plots
pip install matplotlib pandas numpy
python3 scripts/plot_results.py
```

**Output files:**
- `results/benchmark_results.csv` — timing, memory, spike counts
- `results/perf_results.csv` — cache misses, instructions, cycles (if `--perf`)
- `results/*.png` — 5 publication-quality plots

> **Note on `perf` with ARM64:** If `perf stat` fails with a permissions error, run:
> ```bash
> sudo sysctl -w kernel.perf_event_paranoid=-1
> ```
> Or run the benchmark with `sudo`.

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
