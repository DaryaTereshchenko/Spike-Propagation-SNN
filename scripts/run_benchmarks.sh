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
BG_CURRENT=14.0
GATHER_ONLY=true
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
echo "  Bg current: $BG_CURRENT"
echo "  Gather-only: $GATHER_ONLY"
echo "  Output:     $OUTPUT_CSV"
echo ""

# Build sweep args
SWEEP_ARGS="--sweep"
SWEEP_ARGS="$SWEEP_ARGS --sweep-sizes \"$SIZES\""
SWEEP_ARGS="$SWEEP_ARGS --sweep-densities \"$DENSITIES\""
SWEEP_ARGS="$SWEEP_ARGS --timesteps $TIMESTEPS"
SWEEP_ARGS="$SWEEP_ARGS --trials $TRIALS"
SWEEP_ARGS="$SWEEP_ARGS --seed $SEED"
SWEEP_ARGS="$SWEEP_ARGS --bg-current $BG_CURRENT"
SWEEP_ARGS="$SWEEP_ARGS --output-csv $OUTPUT_CSV"
if $GATHER_ONLY; then
    SWEEP_ARGS="$SWEEP_ARGS --gather-only"
fi

eval "$BINARY" $SWEEP_ARGS

# ---- Optional perf stat pass ----
if $USE_PERF && command -v perf &>/dev/null; then
    echo ""
    echo "=== Running perf stat pass ==="
    echo "format,topology,N,density,cache_misses,cache_refs,instructions,cycles,L1d_load_misses,LLC_load_misses,dTLB_load_misses,branch_misses" > "$PERF_CSV"

    RUN_COUNT=0
    for fmt in $FORMATS; do
        for topo in $TOPOLOGIES; do
            for size in $SIZES; do
                for dens in $DENSITIES; do
                    RUN_COUNT=$((RUN_COUNT + 1))
                    echo -n "[perf $RUN_COUNT] $fmt / $topo / N=$size / d=$dens ... "

                    PERF_OUTPUT=$(perf stat -e cache-misses,cache-references,instructions,cycles,L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses,branch-misses \
                        "$BINARY" \
                        --format "$fmt" \
                        --topology "$topo" \
                        --size "$size" \
                        --density "$dens" \
                        --timesteps "$TIMESTEPS" \
                        --trials 1 \
                        --seed "$SEED" \
                        --bg-current "$BG_CURRENT" \
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

                    echo "done"
                done
            done
        done
    done
fi

echo ""
echo "=== Sweep complete ==="
echo "  Results:    $OUTPUT_CSV"
if $USE_PERF; then
    echo "  Perf data:  $PERF_CSV"
fi
