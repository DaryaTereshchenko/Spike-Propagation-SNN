#!/usr/bin/env python3
"""
plot_results.py — Generate publication-quality plots from benchmark CSV data.

Produces:
  1. Time vs N (line plot, per format, subplots per topology)
  2. Time vs density (line plot, per format)
  3. Memory vs N (line plot, per format)
  4. NEST biological connectivity comparison (bar chart)
  5. Cache-miss heatmap (if perf data available)
  6. Effective bandwidth vs N (line plot, per format)
  7. Scatter throughput vs N (line plot, per format)
  8. Cache ratio vs N (grouped bar chart — L1/L2/L3 per format)
  9. TLB & branch miss heatmap (if perf data available)

Usage:
    python3 scripts/plot_results.py [--results results/benchmark_results.csv]
                                     [--perf results/perf_results.csv]
                                     [--nest results/nest_benchmark.csv]
                                     [--outdir results/]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)


FORMAT_COLORS = {
    "coo": "#e74c3c",
    "csr": "#2ecc71",
    "csc": "#3498db",
    "ell": "#f39c12",
}
FORMAT_MARKERS = {"coo": "o", "csr": "s", "csc": "^", "ell": "D"}


def plot_time_vs_n(df, outdir):
    """Plot 1: Time vs N, one line per format, subplots per topology."""
    topologies = sorted(df["topology"].unique())
    n_topo = len(topologies)
    fig, axes = plt.subplots(1, n_topo, figsize=(5 * n_topo, 4), squeeze=False)

    for idx, topo in enumerate(topologies):
        ax = axes[0, idx]
        subset = df[df["topology"] == topo]

        for fmt in sorted(subset["format"].unique()):
            fmt_data = subset[subset["format"] == fmt].sort_values("N")
            ax.errorbar(
                fmt_data["N"], fmt_data["mean_time_ms"],
                yerr=fmt_data["std_time_ms"],
                label=fmt.upper(), color=FORMAT_COLORS.get(fmt, "gray"),
                marker=FORMAT_MARKERS.get(fmt, "x"), capsize=3, linewidth=1.5
            )

        ax.set_xlabel("N (neurons)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Topology: {topo.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Spike Propagation Time vs Network Size", fontsize=14)
    fig.tight_layout()
    path = os.path.join(outdir, "time_vs_N.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_time_vs_density(df, outdir):
    """Plot 2: Time vs density, one line per format."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for fmt in sorted(df["format"].unique()):
        fmt_data = df[df["format"] == fmt]
        # Average across topologies and sizes for each density.
        grouped = fmt_data.groupby("density")["mean_time_ms"].mean().reset_index()
        grouped = grouped.sort_values("density")

        ax.plot(grouped["density"], grouped["mean_time_ms"],
                label=fmt.upper(), color=FORMAT_COLORS.get(fmt, "gray"),
                marker=FORMAT_MARKERS.get(fmt, "x"), linewidth=1.5)

    ax.set_xlabel("Connection Density")
    ax.set_ylabel("Mean Time (ms)")
    ax.set_title("Spike Propagation Time vs Connection Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "time_vs_density.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_memory_vs_n(df, outdir):
    """Plot 3: Memory vs N, one line per format."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for fmt in sorted(df["format"].unique()):
        fmt_data = df[df["format"] == fmt]
        grouped = fmt_data.groupby("N")["memory_bytes"].mean().reset_index()
        grouped = grouped.sort_values("N")
        mem_mb = grouped["memory_bytes"] / (1024 * 1024)

        ax.plot(grouped["N"], mem_mb,
                label=fmt.upper(), color=FORMAT_COLORS.get(fmt, "gray"),
                marker=FORMAT_MARKERS.get(fmt, "x"), linewidth=1.5)

    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Matrix Memory (MB)")
    ax.set_title("Sparse Matrix Memory vs Network Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "memory_vs_N.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_nest_comparison(nest_csv, outdir):
    """Plot 4: NEST biological connectivity benchmark comparison."""
    if nest_csv is None or not os.path.exists(nest_csv):
        print("  Skipping NEST comparison plot (no data)")
        return

    df = pd.read_csv(nest_csv)
    fig, ax = plt.subplots(figsize=(7, 5))

    formats = sorted(df["format"].unique())
    x = np.arange(len(formats))
    width = 0.6

    times = [df[df["format"] == f]["mean_time_ms"].values[0] for f in formats]
    colors = [FORMAT_COLORS.get(f, "gray") for f in formats]

    bars = ax.bar(x, times, width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in formats])
    ax.set_ylabel("Time (ms)")
    ax.set_title("NEST Biological Connectivity: Format Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(outdir, "nest_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_cache_heatmap(perf_csv, outdir):
    """Plot 5: Cache-miss rate heatmap from perf data."""
    if perf_csv is None or not os.path.exists(perf_csv):
        print("  Skipping cache-miss heatmap (no perf data)")
        return

    df = pd.read_csv(perf_csv)

    # Compute cache miss rate.
    df["miss_rate"] = df["cache_misses"] / df["cache_refs"].replace(0, np.nan)

    # Pivot: rows = format, columns = (topology, N)
    df["label"] = df["topology"].str.upper() + " N=" + df["N"].astype(str)
    pivot = df.pivot_table(values="miss_rate", index="format", columns="label")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f.upper() for f in pivot.index])

    # Annotate cells with percentages.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7)

    ax.set_title("Cache Miss Rate by Format and Configuration")
    fig.colorbar(im, ax=ax, label="Miss Rate")
    fig.tight_layout()

    path = os.path.join(outdir, "cache_miss_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_effective_bandwidth(df, outdir):
    """Plot 6: Effective bandwidth vs N, one line per format."""
    if "effective_bw_gbps" not in df.columns:
        print("  Skipping bandwidth plot (column not found)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for fmt in sorted(df["format"].unique()):
        fmt_data = df[df["format"] == fmt]
        grouped = fmt_data.groupby("N")["effective_bw_gbps"].mean().reset_index()
        grouped = grouped.sort_values("N")

        ax.plot(grouped["N"], grouped["effective_bw_gbps"],
                label=fmt.upper(), color=FORMAT_COLORS.get(fmt, "gray"),
                marker=FORMAT_MARKERS.get(fmt, "x"), linewidth=1.5)

    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("Effective Memory Bandwidth vs Network Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "bandwidth_vs_N.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_scatter_throughput(df, outdir):
    """Plot 7: Scatter throughput vs N, one line per format."""
    if "scatter_throughput_edges_per_ms" not in df.columns:
        print("  Skipping scatter throughput plot (column not found)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for fmt in sorted(df["format"].unique()):
        fmt_data = df[df["format"] == fmt]
        grouped = (fmt_data.groupby("N")["scatter_throughput_edges_per_ms"]
                   .mean().reset_index())
        grouped = grouped.sort_values("N")

        ax.plot(grouped["N"], grouped["scatter_throughput_edges_per_ms"],
                label=fmt.upper(), color=FORMAT_COLORS.get(fmt, "gray"),
                marker=FORMAT_MARKERS.get(fmt, "x"), linewidth=1.5)

    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Scatter Throughput (edges/ms)")
    ax.set_title("Edge Scatter Throughput vs Network Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "scatter_throughput_vs_N.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_cache_ratios(df, outdir):
    """Plot 8: Cache ratio (matrix size / cache level) grouped bar chart."""
    ratio_cols = ["cache_ratio_L1", "cache_ratio_L2", "cache_ratio_L3"]
    if not all(c in df.columns for c in ratio_cols):
        print("  Skipping cache ratio plot (columns not found)")
        return

    # Use the largest N with er topology for a clear comparison
    subset = df[df["N"] == df["N"].max()]
    if len(subset) == 0:
        return

    formats = sorted(subset["format"].unique())
    x = np.arange(len(formats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (col, label) in enumerate(zip(ratio_cols, ["L1d", "L2", "L3"])):
        vals = [subset[subset["format"] == f][col].mean() for f in formats]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f.upper() for f in formats])
    ax.set_ylabel("Matrix Size / Cache Size (ratio)")
    ax.set_title(f"Matrix-to-Cache Size Ratio (N={df['N'].max()})")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Fits in cache")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")
    fig.tight_layout()

    path = os.path.join(outdir, "cache_ratios.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_tlb_branch_heatmap(perf_csv, outdir):
    """Plot 9: TLB miss + branch miss heatmap from perf data."""
    if perf_csv is None or not os.path.exists(perf_csv):
        print("  Skipping TLB/branch heatmap (no perf data)")
        return

    df = pd.read_csv(perf_csv)

    # Check if new columns exist
    if "dTLB_load_misses" not in df.columns or "branch_misses" not in df.columns:
        print("  Skipping TLB/branch heatmap (columns not found)")
        return

    df["label"] = df["topology"].str.upper() + " N=" + df["N"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    for ax, metric, title in zip(
        axes,
        ["dTLB_load_misses", "branch_misses"],
        ["dTLB Load Misses", "Branch Mispredictions"],
    ):
        pivot = df.pivot_table(values=metric, index="format", columns="label")
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([f.upper() for f in pivot.index])

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    txt = f"{val/1e6:.1f}M" if val >= 1e6 else f"{val/1e3:.0f}K"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7)

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Count")

    fig.suptitle("TLB and Branch Misses by Format and Configuration", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, "tlb_branch_heatmap.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--results", default="results/benchmark_results.csv",
                        help="Main benchmark CSV")
    parser.add_argument("--perf", default=None,
                        help="Perf stat CSV (optional)")
    parser.add_argument("--nest", default=None,
                        help="NEST benchmark CSV (optional)")
    parser.add_argument("--outdir", default="results/",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.results} ...")
    df = pd.read_csv(args.results)
    print(f"  {len(df)} rows loaded")

    print("Generating plots:")
    plot_time_vs_n(df, args.outdir)
    plot_time_vs_density(df, args.outdir)
    plot_memory_vs_n(df, args.outdir)
    plot_nest_comparison(args.nest, args.outdir)
    plot_cache_heatmap(args.perf, args.outdir)
    plot_effective_bandwidth(df, args.outdir)
    plot_scatter_throughput(df, args.outdir)
    plot_cache_ratios(df, args.outdir)
    plot_tlb_branch_heatmap(args.perf, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
