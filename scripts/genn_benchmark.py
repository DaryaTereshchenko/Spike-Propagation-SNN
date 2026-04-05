#!/usr/bin/env python3
"""
genn_benchmark.py — GPU validation benchmark using PyGeNN.

Defines a LIF spiking neural network with identical parameters to the C++
implementation and benchmarks three GeNN connectivity modes:
  - DENSE:    full weight matrix on GPU
  - SPARSE:   CSR-like sparse storage on GPU
  - BITMASK:  bit-packed connectivity mask (GeNN-specific)

Measures GPU wall-clock times and spike counts for comparison with the CPU
sparse format benchmarks.

Requirements:
  - GeNN >= 4.8 (pip install pygenn)
  - CUDA GPU + driver
  - Python 3.8+

Usage:
    python3 scripts/genn_benchmark.py --help
    python3 scripts/genn_benchmark.py --size 1000 --density 0.05 --timesteps 1000
    python3 scripts/genn_benchmark.py --sweep --output results/gpu_results.csv
"""

import argparse
import csv
import os
import sys
import time
from itertools import product

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from pygenn import (
        GeNNModel,
        init_connectivity,
        init_var,
    )
    GENN_AVAILABLE = True
except ImportError:
    GENN_AVAILABLE = False
    print("WARNING: PyGeNN not available. Script will validate syntax only.")


# --------------------------------------------------------------------------
# LIF neuron model parameters (matching the C++ implementation)
# --------------------------------------------------------------------------
LIF_PARAMS = {
    "C": 1.0,            # Membrane capacitance (normalized; tau_m = R*C = 20ms)
    "TauM": 20.0,        # Membrane time constant (ms)
    "Vrest": -65.0,      # Resting potential (mV)
    "Vreset": -65.0,     # Reset potential (mV)
    "Vthresh": -50.0,    # Spike threshold (mV)
    "TauRefrac": 2.0,    # Refractory period (ms)
}

LIF_INIT = {
    "V": -65.0,          # Initial membrane potential
    "RefracTime": 0.0,   # Initial refractory time remaining
}


def run_genn_benchmark(N, density, connectivity_mode, timesteps, seed=42):
    """
    Run a single GeNN benchmark.

    Args:
        N:                  Number of neurons
        density:            Connection probability
        connectivity_mode:  "DENSE", "SPARSE", or "BITMASK"
        timesteps:          Number of simulation timesteps
        seed:               Random seed

    Returns:
        dict with keys: mode, N, density, timesteps, time_ms, total_spikes
    """
    if not GENN_AVAILABLE:
        return {
            "mode": connectivity_mode,
            "N": N,
            "density": density,
            "timesteps": timesteps,
            "time_ms": -1.0,
            "total_spikes": -1,
        }

    model = GeNNModel("float", f"benchmark_{connectivity_mode}_{N}")
    model.dT = 1.0  # timestep in ms
    model._model.set_seed(seed)

    # Create neuron population using built-in LIF model.
    pop = model.add_neuron_population(
        "pop", N, "LIF",
        LIF_PARAMS, LIF_INIT
    )
    pop.spike_recording_enabled = True

    # Weight for normalized input.
    w = 1.0 / max(1.0, (N * density) ** 0.5)

    # Synapse connectivity.
    if connectivity_mode == "DENSE":
        model.add_synapse_population(
            "syn", "DENSE",
            pop, pop,
            "StaticPulse", {}, {"g": w}, {}, {},
            "DeltaCurr", {}, {}
        )
    elif connectivity_mode == "SPARSE":
        model.add_synapse_population(
            "syn", "SPARSE",
            pop, pop,
            "StaticPulse", {}, {"g": w}, {}, {},
            "DeltaCurr", {}, {},
            init_connectivity("FixedProbabilityNoAutapse", {"prob": density})
        )
    elif connectivity_mode == "BITMASK":
        model.add_synapse_population(
            "syn", "BITMASK",
            pop, pop,
            "StaticPulse", {}, {"g": w}, {}, {},
            "DeltaCurr", {}, {},
            init_connectivity("FixedProbabilityNoAutapse", {"prob": density})
        )
    else:
        raise ValueError(f"Unknown connectivity mode: {connectivity_mode}")

    # Build and load model.
    model.build()
    model.load(num_recording_timesteps=timesteps)

    # Run simulation and measure wall-clock time.
    t_start = time.perf_counter()
    for _ in range(timesteps):
        model.step_time()
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0

    # Pull spike recording data.
    model.pull_recording_buffers_from_device()
    spike_times, spike_ids = pop.spike_recording_data
    total_spikes = len(spike_ids) if spike_ids is not None else 0

    return {
        "mode": connectivity_mode,
        "N": N,
        "density": density,
        "timesteps": timesteps,
        "time_ms": elapsed_ms,
        "total_spikes": total_spikes,
    }


def print_result(result):
    """Pretty-print a single benchmark result."""
    print(f"  Mode:          {result['mode']}")
    print(f"  N:             {result['N']}")
    print(f"  Density:       {result['density']}")
    print(f"  Timesteps:     {result['timesteps']}")
    print(f"  GPU Time:      {result['time_ms']:.2f} ms")
    print(f"  Total Spikes:  {result['total_spikes']}")
    print()


def write_csv(results, output_path):
    """Write results list to CSV."""
    if not results:
        return
    fieldnames = ["mode", "N", "density", "timesteps", "time_ms", "total_spikes"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GeNN GPU benchmark for spike propagation")
    parser.add_argument("--size", type=int, default=1000,
                        help="Number of neurons (default: 1000)")
    parser.add_argument("--density", type=float, default=0.05,
                        help="Connection density (default: 0.05)")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="Simulation timesteps (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--mode", choices=["DENSE", "SPARSE", "BITMASK"],
                        default=None,
                        help="Specific connectivity mode (default: run all)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep")
    parser.add_argument("--output", default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    if not GENN_AVAILABLE:
        print("PyGeNN is not installed. Syntax check passed.")
        print("Install GeNN and re-run on a CUDA-capable machine.")
        return

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
                result = run_genn_benchmark(n, d, m, args.timesteps, args.seed)
                results.append(result)
                print_result(result)
            except Exception as e:
                print(f"  SKIPPED: {e}\n")

    else:
        modes = [args.mode] if args.mode else ["DENSE", "SPARSE", "BITMASK"]
        for m in modes:
            print(f"Running {m} / N={args.size} / d={args.density} ...")
            result = run_genn_benchmark(
                args.size, args.density, m, args.timesteps, args.seed)
            results.append(result)
            print_result(result)

    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        write_csv(results, args.output)


if __name__ == "__main__":
    main()
