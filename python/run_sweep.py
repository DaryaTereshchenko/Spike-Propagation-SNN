"""
run_sweep.py – Run the C++ benchmark across all (N, p, topology) combinations
               defined in the project proposal and combine results into a single CSV.

Usage:
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
    python3 ../python/run_sweep.py [--binary ./benchmark] [--output sweep_results.csv]
"""

import argparse
import subprocess
import os
import csv
import sys

NETWORK_SIZES    = [1_000, 10_000, 50_000, 100_000]
DENSITIES        = [0.01, 0.05, 0.10, 0.20]
TOPOLOGIES       = ["er", "fi", "ba", "ws"]
TIMESTEPS        = 1_000
TRIALS           = 10

def run_benchmark(binary, N, p, topology, timesteps, trials, tmp_csv):
    cmd = [
        binary,
        "--N",         str(N),
        "--p",         str(p),
        "--topology",  topology,
        "--timesteps", str(timesteps),
        "--trials",    str(trials),
        "--output",    tmp_csv,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {cmd}:\n{result.stderr}", file=sys.stderr)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary",  default="./benchmark")
    parser.add_argument("--output",  default="sweep_results.csv")
    args = parser.parse_args()

    tmp_csv  = "/tmp/snn_tmp_result.csv"
    combined = []
    header   = None

    total = len(NETWORK_SIZES) * len(DENSITIES) * len(TOPOLOGIES)
    done  = 0

    for N in NETWORK_SIZES:
        for p in DENSITIES:
            for topo in TOPOLOGIES:
                done += 1
                print(f"[{done}/{total}] N={N} p={p} topo={topo} ...", flush=True)
                ok = run_benchmark(args.binary, N, p, topo,
                                   TIMESTEPS, TRIALS, tmp_csv)
                if not ok:
                    continue
                with open(tmp_csv, newline="") as f:
                    reader = csv.DictReader(f)
                    if header is None:
                        header = reader.fieldnames
                    for row in reader:
                        combined.append(row)

    if not combined:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nSweep complete. {len(combined)} rows written to {args.output}")

if __name__ == "__main__":
    main()
