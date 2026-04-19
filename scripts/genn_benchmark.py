#!/usr/bin/env python3
"""
genn_benchmark.py -- GPU benchmark using PyGeNN 5.x on NVIDIA GB10 (DGX Spark).

Defines a LIF spiking neural network with identical parameters to the C++
implementation and benchmarks three GeNN connectivity modes:
  - DENSE:    full weight matrix on GPU
  - SPARSE:   CSR-like sparse storage on GPU
  - BITMASK:  bit-packed connectivity mask (GeNN-specific)

Measures GPU kernel times (neuron update, presynaptic update, init),
wall-clock times, spike counts, and memory for comparison with
the CPU sparse format benchmarks.

Requirements:
  - GeNN 5.x (built with CUDA backend: CUDA_PATH=/usr/local/cuda pip install ...)
  - CUDA GPU + driver
  - Python 3.8+

Usage:
    python3 scripts/genn_benchmark.py --help
    python3 scripts/genn_benchmark.py --size 1000 --density 0.05 --timesteps 1000
    python3 scripts/genn_benchmark.py --sweep --output results/gpu_results.csv
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from itertools import product
from statistics import mean, stdev, median

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy")
    sys.exit(1)

try:
    from pygenn import (
        GeNNModel,
        init_sparse_connectivity,
        init_weight_update,
        init_postsynaptic,
    )
    GENN_AVAILABLE = True
except ImportError:
    GENN_AVAILABLE = False
    print("WARNING: PyGeNN not available. Script will validate syntax only.")


# --------------------------------------------------------------------------
# LIF neuron model parameters (matching the C++ implementation)
# --------------------------------------------------------------------------
LIF_PARAMS = {
    "C": 1.0,
    "TauM": 20.0,
    "Vrest": -65.0,
    "Vreset": -65.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0,
}

LIF_INIT = {
    "V": -65.0,
    "RefracTime": 0.0,
}

BACKGROUND_CURRENT = 14.0  # matching CPU benchmark


# --------------------------------------------------------------------------
# nvidia-smi helpers
# --------------------------------------------------------------------------
def query_gpu_info():
    """Return a dict of static GPU properties via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_name": parts[0],
            "gpu_mem_total_mb": float(parts[1]),
            "driver_version": parts[2],
            "max_sm_clock_mhz": parts[3],
            "max_mem_clock_mhz": parts[4],
            "compute_cap": parts[5],
        }
    except Exception:
        return {}


def query_gpu_mem_used():
    """Return current GPU memory used in MB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip()
        return float(out)
    except Exception:
        return -1.0


# --------------------------------------------------------------------------
# Core benchmark
# --------------------------------------------------------------------------
def run_genn_benchmark(N, density, connectivity_mode, timesteps, trials,
                       seed=42, bg_current=BACKGROUND_CURRENT):
    if not GENN_AVAILABLE:
        return None

    nnz_expected = int(N * N * density)

    trial_results = []

    for trial in range(trials):
        mem_before = query_gpu_mem_used()

        model = GeNNModel("float", f"bench_{connectivity_mode}_{N}_{trial}",
                          backend="cuda")
        model.dt = 1.0
        model.seed = seed + trial
        model.timing_enabled = True
        model.batch_size = 1

        # Neuron population (use Ioffset for background current).
        params = dict(LIF_PARAMS)
        params["Ioffset"] = bg_current
        pop = model.add_neuron_population("pop", N, "LIF", params, LIF_INIT)
        pop.spike_recording_enabled = True

        # Synapse weight.
        w = 1.0 / max(1.0, math.sqrt(N * density))

        # Synapse population (GeNN 5.x API).
        wu_init = init_weight_update("StaticPulse", {}, {"g": w})
        ps_init = init_postsynaptic("DeltaCurr")

        if connectivity_mode == "DENSE":
            model.add_synapse_population(
                "syn", "DENSE", pop, pop, wu_init, ps_init)
        elif connectivity_mode == "SPARSE":
            conn = init_sparse_connectivity("FixedProbabilityNoAutapse",
                                            {"prob": density})
            model.add_synapse_population(
                "syn", "SPARSE", pop, pop, wu_init, ps_init, conn)
        elif connectivity_mode == "BITMASK":
            conn = init_sparse_connectivity("FixedProbabilityNoAutapse",
                                            {"prob": density})
            model.add_synapse_population(
                "syn", "BITMASK", pop, pop, wu_init, ps_init, conn)
        else:
            raise ValueError(f"Unknown mode: {connectivity_mode}")

        # Build, load, allocate.
        model.build()
        model.load(num_recording_timesteps=timesteps)

        mem_after = query_gpu_mem_used()
        gpu_mem_delta_mb = mem_after - mem_before if mem_before >= 0 else -1.0

        # Warm-up (5 steps, not measured).
        for _ in range(5):
            model.step_time()

        # Reset timing counters after warm-up.
        model.timestep = 0
        try:
            model.neuron_update_time = 0.0
            model.presynaptic_update_time = 0.0
            model.postsynaptic_update_time = 0.0
            model.synapse_dynamics_time = 0.0
            model.init_time = 0.0
            model.init_sparse_time = 0.0
        except Exception:
            pass

        # Timed run.
        t_start = time.perf_counter()
        for _ in range(timesteps):
            model.step_time()
        t_end = time.perf_counter()
        wall_ms = (t_end - t_start) * 1000.0

        # Retrieve GeNN kernel times (ms).
        neuron_ms = model.neuron_update_time
        presyn_ms = model.presynaptic_update_time
        postsyn_ms = model.postsynaptic_update_time
        syndyn_ms = model.synapse_dynamics_time
        init_ms = model.init_time
        init_sparse_ms = model.init_sparse_time

        # Spike data (GeNN 5: returns list of (times, ids) per batch).
        model.pull_recording_buffers_from_device()
        rec = pop.spike_recording_data
        if isinstance(rec, list) and len(rec) > 0:
            spike_times, spike_ids = rec[0]
        else:
            spike_times, spike_ids = rec
        total_spikes = len(spike_ids) if spike_ids is not None else 0
        spikes_per_step = total_spikes / timesteps if timesteps > 0 else 0

        trial_results.append({
            "wall_ms": wall_ms,
            "neuron_ms": neuron_ms,
            "presyn_ms": presyn_ms,
            "postsyn_ms": postsyn_ms,
            "syndyn_ms": syndyn_ms,
            "init_ms": init_ms,
            "init_sparse_ms": init_sparse_ms,
            "total_spikes": total_spikes,
            "spikes_per_step": spikes_per_step,
            "gpu_mem_delta_mb": gpu_mem_delta_mb,
        })

        # Cleanup to free GPU memory between trials.
        model.unload()

    # Aggregate across trials.
    walls = [t["wall_ms"] for t in trial_results]
    result = {
        "mode": connectivity_mode,
        "N": N,
        "density": density,
        "timesteps": timesteps,
        "trials": trials,
        "nnz": nnz_expected,
        "mean_time_ms": mean(walls),
        "std_time_ms": stdev(walls) if len(walls) > 1 else 0.0,
        "median_time_ms": median(walls),
        "neuron_update_ms": mean([t["neuron_ms"] for t in trial_results]),
        "presynaptic_update_ms": mean([t["presyn_ms"] for t in trial_results]),
        "postsynaptic_update_ms": mean([t["postsyn_ms"] for t in trial_results]),
        "synapse_dynamics_ms": mean([t["syndyn_ms"] for t in trial_results]),
        "init_time_ms": mean([t["init_ms"] for t in trial_results]),
        "init_sparse_time_ms": mean([t["init_sparse_ms"] for t in trial_results]),
        "total_spikes": trial_results[-1]["total_spikes"],
        "spikes_per_step": trial_results[-1]["spikes_per_step"],
        "gpu_mem_delta_mb": trial_results[-1]["gpu_mem_delta_mb"],
        "background_current": bg_current,
    }

    # Derived throughput metrics.
    total_kernel_ms = (result["neuron_update_ms"]
                       + result["presynaptic_update_ms"]
                       + result["postsynaptic_update_ms"])
    result["total_kernel_ms"] = total_kernel_ms
    if total_kernel_ms > 0:
        result["gpu_throughput_edges_per_ms"] = nnz_expected / (total_kernel_ms / timesteps)
    else:
        result["gpu_throughput_edges_per_ms"] = 0.0

    return result


def print_result(r):
    print(f"  Mode:               {r['mode']}")
    print(f"  N:                  {r['N']}")
    print(f"  Density:            {r['density']}")
    print(f"  NNZ:                {r['nnz']}")
    print(f"  Wall time:          {r['mean_time_ms']:.2f} +/- {r['std_time_ms']:.2f} ms")
    print(f"  Neuron update:      {r['neuron_update_ms']:.2f} ms")
    print(f"  Presynaptic:        {r['presynaptic_update_ms']:.2f} ms")
    print(f"  Postsynaptic:       {r['postsynaptic_update_ms']:.2f} ms")
    print(f"  Total kernel:       {r['total_kernel_ms']:.2f} ms")
    print(f"  GPU throughput:     {r['gpu_throughput_edges_per_ms']:.1f} edges/ms")
    print(f"  Total spikes:       {r['total_spikes']}")
    print(f"  Spikes/step:        {r['spikes_per_step']:.1f}")
    print(f"  GPU mem delta:      {r['gpu_mem_delta_mb']:.1f} MB")
    print()


CSV_FIELDS = [
    "mode", "N", "density", "timesteps", "trials", "nnz",
    "mean_time_ms", "std_time_ms", "median_time_ms",
    "neuron_update_ms", "presynaptic_update_ms", "postsynaptic_update_ms",
    "synapse_dynamics_ms", "init_time_ms", "init_sparse_time_ms",
    "total_kernel_ms", "gpu_throughput_edges_per_ms",
    "total_spikes", "spikes_per_step",
    "gpu_mem_delta_mb", "background_current",
]


def write_csv(results, output_path):
    if not results:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GeNN 5.x GPU benchmark for spike propagation (DGX Spark / GB10)")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bg-current", type=float, default=BACKGROUND_CURRENT)
    parser.add_argument("--mode", choices=["DENSE", "SPARSE", "BITMASK"],
                        default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep (matches CPU grid)")
    parser.add_argument("--output", default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    if not GENN_AVAILABLE:
        print("PyGeNN not installed. Cannot run GPU benchmarks.")
        return

    # Print GPU info.
    gpu_info = query_gpu_info()
    print("=" * 60)
    print("GeNN 5 GPU Benchmark -- DGX Spark (GB10)")
    print("=" * 60)
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    print()

    results = []

    if args.sweep:
        sizes = [1000, 5000, 10000]
        densities = [0.01, 0.05, 0.1]
        modes = ["DENSE", "SPARSE", "BITMASK"]

        total = len(sizes) * len(densities) * len(modes)
        done = 0

        for n, d, m in product(sizes, densities, modes):
            done += 1
            print(f"[{done}/{total}] {m} / N={n} / d={d} ...")
            try:
                result = run_genn_benchmark(
                    n, d, m, args.timesteps, args.trials,
                    args.seed, args.bg_current)
                if result:
                    results.append(result)
                    print_result(result)
            except Exception as e:
                print(f"  SKIPPED: {e}\n")
    else:
        modes = [args.mode] if args.mode else ["DENSE", "SPARSE", "BITMASK"]
        for m in modes:
            print(f"Running {m} / N={args.size} / d={args.density} ...")
            result = run_genn_benchmark(
                args.size, args.density, m, args.timesteps, args.trials,
                args.seed, args.bg_current)
            if result:
                results.append(result)
                print_result(result)

    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        write_csv(results, args.output)

    print("=" * 60)
    print(f"Done. {len(results)} benchmarks completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
