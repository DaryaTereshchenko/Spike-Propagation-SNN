"""
plot_results.py – Visualise benchmark results from the C++ benchmark harness.

Usage:
    python3 python/plot_results.py results.csv

Produces:
  - scatter_gather_comparison.png  : mean latency per format, scatter vs gather
  - memory_footprint.png           : memory usage per format
  - latency_vs_N.png               : (when multiple N are present) scaling curves
"""

import sys
import os
import csv
import statistics
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["N"]           = int(row["N"])
            row["p"]           = float(row["p"])
            row["T"]           = int(row["T"])
            row["trial"]       = int(row["trial"])
            row["elapsed_ms"]  = float(row["elapsed_ms"])
            row["peak_rss_kb"] = int(row["peak_rss_kb"])
            row["mem_bytes"]   = int(row["mem_bytes"])
            rows.append(row)
    return rows

# ── Aggregate helpers ─────────────────────────────────────────────────────────

def group_by(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return groups

def mean_sd(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)

# ── Plot 1: scatter vs gather latency ─────────────────────────────────────────

def plot_scatter_gather(rows, outdir):
    by_format = group_by(rows, "format")
    formats   = sorted(by_format.keys())

    names, means, sds = [], [], []
    for fmt in formats:
        times = [r["elapsed_ms"] for r in by_format[fmt]]
        m, s  = mean_sd(times)
        names.append(fmt)
        means.append(m)
        sds.append(s)

    x       = np.arange(len(names))
    colours = [cm.tab10(i % 10) for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=sds, capsize=4, color=colours, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("Spike-propagation latency per format and operation")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(outdir, "scatter_gather_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

# ── Plot 2: memory footprint ──────────────────────────────────────────────────

def plot_memory(rows, outdir):
    # Memory is per format (not per trial), deduplicate.
    seen   = {}
    for r in rows:
        base = r["format"].rsplit("_", 1)[0]  # e.g. "CSR_scatter" -> "CSR"
        seen[base] = r["mem_bytes"]

    formats = sorted(seen.keys())
    mems    = [seen[f] / 1024 for f in formats]   # convert to KB

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = [cm.tab10(i % 10) for i in range(len(formats))]
    ax.bar(formats, mems, color=colours, alpha=0.85)
    ax.set_ylabel("Memory (KB)")
    ax.set_title("Sparse matrix memory footprint by format")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(outdir, "memory_footprint.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

# ── Plot 3: latency vs N (if multiple sizes present) ─────────────────────────

def plot_latency_vs_N(rows, outdir):
    ns = sorted({r["N"] for r in rows})
    if len(ns) < 2:
        return   # only one network size – skip

    # Focus on scatter mode for clarity.
    scatter_rows = [r for r in rows if r["format"].endswith("_scatter")]
    by_format    = group_by(scatter_rows, "format")

    fig, ax = plt.subplots(figsize=(8, 5))
    for fmt, frows in sorted(by_format.items()):
        by_n = group_by(frows, "N")
        xs, ys, errs = [], [], []
        for n in ns:
            if n in by_n:
                times = [r["elapsed_ms"] for r in by_n[n]]
                m, s  = mean_sd(times)
                xs.append(n)
                ys.append(m)
                errs.append(s)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=fmt)

    ax.set_xlabel("Network size N")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("Scatter latency vs network size")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(outdir, "latency_vs_N.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <results.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    outdir   = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    os.makedirs(outdir, exist_ok=True)

    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    plot_scatter_gather(rows, outdir)
    plot_memory(rows, outdir)
    plot_latency_vs_N(rows, outdir)

if __name__ == "__main__":
    main()
