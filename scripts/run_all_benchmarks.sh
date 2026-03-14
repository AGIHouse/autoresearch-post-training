#!/bin/bash
# ============================================================================
# Run all benchmark experiments sequentially on the VM.
# Each run is 10 minutes. Total: ~50 minutes for 5 experiments.
#
# Usage (run ON the VM):
#   cd ~/post_train && bash scripts/run_all_benchmarks.sh
#
# Or from local (via deploy script):
#   ./scripts/deploy_and_run.sh bench-all
# ============================================================================

set -euo pipefail

cd ~/post_train
source venv/bin/activate 2>/dev/null || true

RESULTS_FILE="$HOME/benchmark_results.txt"
echo "============================================" > "$RESULTS_FILE"
echo "Benchmark Results - $(date)" >> "$RESULTS_FILE"
echo "============================================" >> "$RESULTS_FILE"

run_one() {
    local config="$1"
    local name="$2"
    echo ""
    echo "############################################################"
    echo "# Experiment: $name"
    echo "# Config: $config"
    echo "# Started: $(date)"
    echo "############################################################"
    echo ""

    # Upgrade deps if needed (for vllm version bump)
    # pip install -e ".[dev]" -q 2>/dev/null || true

    bash scripts/run_benchmark.sh "$config" "$name" 2>&1 | tee -a "$RESULTS_FILE"

    echo "" >> "$RESULTS_FILE"
    echo "---" >> "$RESULTS_FILE"
    echo ""
}

echo "Running experiments 1-4 (10 min each, ~40 min total)..."
echo ""

run_one configs/bench_1_baseline.yaml   "1_baseline"
run_one configs/bench_2_subprocess.yaml "2_subprocess"
run_one configs/bench_3_sleep.yaml      "3_sleep"
run_one configs/bench_4_reuse.yaml      "4_reuse"

echo ""
echo "============================================"
echo "ALL BENCHMARKS COMPLETE"
echo "============================================"
echo "Results saved to: $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE"
