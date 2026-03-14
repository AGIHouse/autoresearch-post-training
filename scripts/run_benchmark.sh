#!/bin/bash
# ============================================================================
# Run a 10-minute benchmark training run and extract performance metrics.
#
# Usage:
#   ./scripts/run_benchmark.sh [config_yaml] [experiment_name]
#
# Example:
#   ./scripts/run_benchmark.sh configs/benchmark.yaml baseline
#
# Output:
#   - Training log: ~/bench_<name>.log
#   - Metrics summary printed to stdout
# ============================================================================

set -euo pipefail

CONFIG="${1:-configs/benchmark.yaml}"
NAME="${2:-experiment}"
DURATION=600  # 10 minutes in seconds
LOGFILE="$HOME/bench_${NAME}.log"

echo "============================================"
echo "Benchmark: $NAME"
echo "Config:    $CONFIG"
echo "Duration:  ${DURATION}s (10 min)"
echo "Log:       $LOGFILE"
echo "============================================"

cd ~/post_train

# Activate venv if it exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Clean previous outputs
rm -rf outputs/benchmark

# Start training in background
echo "Starting training at $(date)..."
python -m src.train --config "$CONFIG" > "$LOGFILE" 2>&1 &
TRAIN_PID=$!

# Wait for duration or until training finishes
sleep "$DURATION" || true

# Check if still running
if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Time limit reached. Stopping training..."
    kill "$TRAIN_PID" 2>/dev/null || true
    sleep 5
    kill -9 "$TRAIN_PID" 2>/dev/null || true
else
    echo "Training finished before time limit."
fi

echo ""
echo "============================================"
echo "Results for: $NAME"
echo "============================================"

# Extract metrics from log
# TRL logs lines like: {'loss': 0.0, 'grad_norm': 0.5, ... 'step': 10}
# Also logs timing info

# Count training steps completed (look for 'step' in logged dicts)
STEPS=$(grep -oP "'step':\s*\K[0-9]+" "$LOGFILE" | tail -1 || echo "0")
echo "Steps completed: $STEPS"

# Extract timestamps of first and last step to compute avg step time
FIRST_STEP_TIME=$(grep -m1 "'step':" "$LOGFILE" | head -1 || true)
LAST_STEP_TIME=$(grep "'step':" "$LOGFILE" | tail -1 || true)

# Look for TRL's training speed log
SAMPLES_PER_SEC=$(grep -oP "'train_samples_per_second':\s*\K[0-9.]+" "$LOGFILE" | tail -1 || echo "N/A")
STEPS_PER_SEC=$(grep -oP "'train_steps_per_second':\s*\K[0-9.]+" "$LOGFILE" | tail -1 || echo "N/A")

echo "Samples/sec: $SAMPLES_PER_SEC"
echo "Steps/sec: $STEPS_PER_SEC"

if [ "$STEPS_PER_SEC" != "N/A" ] && [ "$STEPS_PER_SEC" != "0" ]; then
    SEC_PER_STEP=$(python3 -c "print(f'{1/float(\"$STEPS_PER_SEC\"):.1f}')" 2>/dev/null || echo "N/A")
    echo "Sec/step: $SEC_PER_STEP"
fi

# GPU utilization snapshot
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi failed)"

echo ""
echo "Last 20 lines of log:"
tail -20 "$LOGFILE"

echo ""
echo "============================================"
echo "Benchmark complete: $NAME"
echo "============================================"
