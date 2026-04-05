#!/usr/bin/env bash
#
# run_benchmarks.sh — Build the project in Release mode and run the full
# parameter sweep, optionally wrapping each run with `perf stat`.
#
# Usage:
#   ./scripts/run_benchmarks.sh                   # basic run
#   ./scripts/run_benchmarks.sh --perf             # with perf stat counters
#   ./scripts/run_benchmarks.sh --small            # quick smoke test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_DIR="$PROJECT_DIR/results"
BINARY="$BUILD_DIR/spike_benchmark"

USE_PERF=false
SMALL=false

for arg in "$@"; do
    case "$arg" in
        --perf)  USE_PERF=true ;;
        --small) SMALL=true ;;
        *)       echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ---- Build ----
echo "=== Building Release ==="
cmake -B "$BUILD_DIR" -S "$PROJECT_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j "$(nproc)"

mkdir -p "$RESULTS_DIR"

# ---- Define sweep grid ----
if $SMALL; then
    SIZES="1000"
    DENSITIES="0.05"
    TOPOLOGIES="er fi"
    FORMATS="coo csr csc ell"
    TIMESTEPS=100
    TRIALS=2
else
    SIZES="1000 5000 10000"
    DENSITIES="0.01 0.05 0.1"
    TOPOLOGIES="er fi ba ws"
    FORMATS="coo csr csc ell"
    TIMESTEPS=1000
    TRIALS=10
fi

SEED=42
OUTPUT_CSV="$RESULTS_DIR/benchmark_results.csv"
PERF_CSV="$RESULTS_DIR/perf_results.csv"
CACHE_CSV="$RESULTS_DIR/cache_info.csv"

# ---- Run sweep ----
echo "=== Running parameter sweep ==="
echo "  Sizes:      $SIZES"
echo "  Densities:  $DENSITIES"
echo "  Topologies: $TOPOLOGIES"
echo "  Formats:    $FORMATS"
echo "  Timesteps:  $TIMESTEPS"
echo "  Trials:     $TRIALS"
echo "  Output:     $OUTPUT_CSV"
echo ""

# Write CSV header
echo "format,topology,N,density,timesteps,trials,mean_time_ms,std_time_ms,peak_rss_kb,total_spikes,spikes_per_step,memory_bytes,nnz,effective_bw_gbps,scatter_throughput_edges_per_ms,bytes_per_spike,cache_ratio_L1,cache_ratio_L2,cache_ratio_L3" > "$OUTPUT_CSV"

if $USE_PERF; then
    echo "format,topology,N,density,cache_misses,cache_refs,instructions,cycles,L1d_load_misses,LLC_load_misses,dTLB_load_misses,branch_misses" > "$PERF_CSV"
fi

RUN_COUNT=0
CACHE_LOGGED=false
for fmt in $FORMATS; do
    for topo in $TOPOLOGIES; do
        for size in $SIZES; do
            for dens in $DENSITIES; do
                RUN_COUNT=$((RUN_COUNT + 1))
                echo -n "[$RUN_COUNT] $fmt / $topo / N=$size / d=$dens ... "

                # On first run, also write cache info CSV
                CACHE_FLAG=""
                if ! $CACHE_LOGGED; then
                    CACHE_FLAG="--cache-csv $CACHE_CSV"
                    CACHE_LOGGED=true
                fi

                # Run the benchmark
                "$BINARY" \
                    --format "$fmt" \
                    --topology "$topo" \
                    --size "$size" \
                    --density "$dens" \
                    --timesteps "$TIMESTEPS" \
                    --trials "$TRIALS" \
                    --seed "$SEED" \
                    --output-csv "$OUTPUT_CSV" \
                    $CACHE_FLAG \
                    2>/dev/null || { echo "SKIPPED"; continue; }

                echo "done"

                # Optionally run with expanded perf stat counters
                if $USE_PERF && command -v perf &>/dev/null; then
                    PERF_OUTPUT=$(perf stat -e cache-misses,cache-references,instructions,cycles,L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses,branch-misses \
                        "$BINARY" \
                        --format "$fmt" \
                        --topology "$topo" \
                        --size "$size" \
                        --density "$dens" \
                        --timesteps "$TIMESTEPS" \
                        --trials 1 \
                        --seed "$SEED" \
                        2>&1 >/dev/null || true)

                    # Parse perf output
                    cache_misses=$(echo "$PERF_OUTPUT" | grep "cache-misses" | awk '{gsub(/,/,"",$1); print $1}')
                    cache_refs=$(echo "$PERF_OUTPUT" | grep "cache-references" | awk '{gsub(/,/,"",$1); print $1}')
                    instructions=$(echo "$PERF_OUTPUT" | grep "instructions" | awk '{gsub(/,/,"",$1); print $1}')
                    cycles=$(echo "$PERF_OUTPUT" | grep "cycles" | head -1 | awk '{gsub(/,/,"",$1); print $1}')
                    l1d_misses=$(echo "$PERF_OUTPUT" | grep "L1-dcache-load-misses" | awk '{gsub(/,/,"",$1); print $1}')
                    llc_misses=$(echo "$PERF_OUTPUT" | grep "LLC-load-misses" | awk '{gsub(/,/,"",$1); print $1}')
                    dtlb_misses=$(echo "$PERF_OUTPUT" | grep "dTLB-load-misses" | awk '{gsub(/,/,"",$1); print $1}')
                    branch_misses=$(echo "$PERF_OUTPUT" | grep "branch-misses" | awk '{gsub(/,/,"",$1); print $1}')

                    echo "$fmt,$topo,$size,$dens,$cache_misses,$cache_refs,$instructions,$cycles,$l1d_misses,$llc_misses,$dtlb_misses,$branch_misses" >> "$PERF_CSV"
                fi
            done
        done
    done
done

echo ""
echo "=== Sweep complete ==="
echo "  Results:    $OUTPUT_CSV"
echo "  Cache info: $CACHE_CSV"
if $USE_PERF; then
    echo "  Perf data:  $PERF_CSV"
fi
echo "  Total runs: $RUN_COUNT"
