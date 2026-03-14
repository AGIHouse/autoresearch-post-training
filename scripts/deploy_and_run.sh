#!/bin/bash
# ============================================================================
# Deploy code to GCP VM and run training + evaluation
# ============================================================================
#
# Usage:
#   ./scripts/deploy_and_run.sh deploy        # Sync code to VM
#   ./scripts/deploy_and_run.sh train         # Deploy + run training
#   ./scripts/deploy_and_run.sh eval          # Deploy + run eval (baseline + trained)
#   ./scripts/deploy_and_run.sh eval-rlvr     # Deploy + run RLVR eval
#   ./scripts/deploy_and_run.sh status        # Check running processes
#   ./scripts/deploy_and_run.sh logs [file]   # Tail logs (default: train.log)
#   ./scripts/deploy_and_run.sh results       # Pull eval results from VM
#   ./scripts/deploy_and_run.sh save-model     # Upload trained model to GCS
#   ./scripts/deploy_and_run.sh stop          # Stop the VM (prompts to save model first)
#
# Environment:
#   GCP_PROJECT_ID  (default: agent-rl-lift)
#   GCP_ZONE        (default: us-central1-a)
#   GCP_INSTANCE    (default: coding-agent-rl)
#   GCS_BUCKET      (default: gs://agent-rl-lift-models)
# ============================================================================

set -euo pipefail

PROJECT="${GCP_PROJECT_ID:-agent-rl-lift}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-coding-agent-rl}"
BUCKET="${GCS_BUCKET:-gs://agent-rl-lift-models}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="~/post_train"

SSH_FLAGS="--zone=$ZONE --project=$PROJECT --ssh-flag=-o --ssh-flag=ConnectTimeout=15"

ssh_cmd() {
    gcloud compute ssh "$INSTANCE" $SSH_FLAGS --command="$1"
}

scp_to() {
    gcloud compute scp "$1" "$INSTANCE:$2" --zone="$ZONE" --project="$PROJECT"
}

scp_from() {
    gcloud compute scp "$INSTANCE:$1" "$2" --zone="$ZONE" --project="$PROJECT"
}

deploy() {
    echo "Syncing code to $INSTANCE..."
    gcloud compute scp --recurse "$REPO_DIR/src/" "$INSTANCE:$REMOTE_DIR/src/" --zone="$ZONE" --project="$PROJECT"
    gcloud compute scp --recurse "$REPO_DIR/configs/" "$INSTANCE:$REMOTE_DIR/configs/" --zone="$ZONE" --project="$PROJECT"
    gcloud compute scp --recurse "$REPO_DIR/docker/" "$INSTANCE:$REMOTE_DIR/docker/" --zone="$ZONE" --project="$PROJECT"
    scp_to "$REPO_DIR/pyproject.toml" "$REMOTE_DIR/pyproject.toml"
    scp_to "$REPO_DIR/Makefile" "$REMOTE_DIR/Makefile"
    echo "Deploy complete."
}

run_train() {
    deploy
    echo "Starting training on $INSTANCE..."
    ssh_cmd "cd $REMOTE_DIR && source venv/bin/activate && \
        nohup python -m src.train --config configs/dr_grpo.yaml > ~/train.log 2>&1 &"
    echo "Training started in background. Monitor with:"
    echo "  $0 logs train.log"
}

run_bench() {
    local config="${2:-configs/benchmark.yaml}"
    local name="${3:-experiment}"
    deploy
    echo "Starting benchmark '$name' on $INSTANCE (10 min)..."
    ssh_cmd "cd $REMOTE_DIR && source venv/bin/activate && \
        bash scripts/run_benchmark.sh $config $name"
}

run_eval() {
    deploy
    echo "Starting evaluation (baseline + trained) on $INSTANCE..."
    ssh_cmd "cd $REMOTE_DIR && source venv/bin/activate && \
        export VLLM_WORKER_MULTIPROC_METHOD=spawn && \
        nohup bash -c ' \
            python -m src.evaluate --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark mbpp --output outputs/eval_baseline.json 2>&1 | tee eval_baseline.log && \
            python -m src.evaluate --model ./outputs/final --benchmark mbpp --is-adapter --output outputs/eval_trained.json 2>&1 | tee eval_trained.log && \
            echo EVAL_COMPLETE \
        ' > ~/eval.log 2>&1 &"
    echo "Eval started in background. Monitor with:"
    echo "  $0 logs eval.log"
}

run_eval_rlvr() {
    deploy
    echo "Starting RLVR evaluation on $INSTANCE..."
    ssh_cmd "cd $REMOTE_DIR && source venv/bin/activate && \
        export VLLM_WORKER_MULTIPROC_METHOD=spawn && \
        nohup bash -c ' \
            python -m src.evaluate --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark rlvr --output outputs/eval_baseline_rlvr.json 2>&1 | tee eval_baseline_rlvr.log && \
            python -m src.evaluate --model ./outputs/final --benchmark rlvr --is-adapter --output outputs/eval_trained_rlvr.json 2>&1 | tee eval_trained_rlvr.log && \
            echo EVAL_RLVR_COMPLETE \
        ' > ~/eval_rlvr.log 2>&1 &"
    echo "RLVR eval started in background. Monitor with:"
    echo "  $0 logs eval_rlvr.log"
}

show_status() {
    echo "=== GPU ==="
    ssh_cmd "nvidia-smi | head -20" 2>/dev/null || echo "(SSH failed)"
    echo ""
    echo "=== Processes ==="
    ssh_cmd "ps aux | grep -E 'src\.(train|evaluate)' | grep -v grep" 2>/dev/null || echo "No training/eval processes running."
}

show_logs() {
    local logfile="${1:-train.log}"
    ssh_cmd "tail -30 ~/$logfile 2>/dev/null || echo 'Log not found: $logfile'"
}

pull_results() {
    echo "Pulling eval results from $INSTANCE..."
    mkdir -p "$REPO_DIR/outputs"
    for f in eval_baseline.json eval_trained.json eval_baseline_rlvr.json eval_trained_rlvr.json; do
        scp_from "$REMOTE_DIR/outputs/$f" "$REPO_DIR/outputs/$f" 2>/dev/null && echo "  $f" || true
    done
    echo ""
    cd "$REPO_DIR" && make compare
}

save_model() {
    local run_name="${1:-}"

    # Check if model exists on VM
    if ! ssh_cmd "test -d $REMOTE_DIR/outputs/final && echo exists" 2>/dev/null | grep -q exists; then
        echo "No trained model found at $REMOTE_DIR/outputs/final on $INSTANCE."
        return 1
    fi

    # Get model size
    local size
    size=$(ssh_cmd "du -sh $REMOTE_DIR/outputs/final 2>/dev/null | cut -f1" 2>/dev/null)
    echo "Found trained model on VM ($size)"

    # Generate a run name if not provided
    if [ -z "$run_name" ]; then
        run_name="run-$(date +%Y%m%d-%H%M%S)"
        read -p "Enter a name for this run [$run_name]: " input_name
        run_name="${input_name:-$run_name}"
    fi

    local gcs_path="$BUCKET/$run_name"
    echo "Uploading to: $gcs_path"

    # Upload model
    ssh_cmd "gsutil -m cp -r $REMOTE_DIR/outputs/final $gcs_path/model/" 2>&1
    echo "Model uploaded."

    # Upload eval results if they exist
    ssh_cmd "gsutil -m cp $REMOTE_DIR/outputs/eval_*.json $gcs_path/eval/ 2>/dev/null" 2>&1 || true

    # Upload training config
    ssh_cmd "gsutil cp $REMOTE_DIR/configs/dr_grpo.yaml $gcs_path/config.yaml 2>/dev/null" 2>&1 || true

    echo ""
    echo "Saved to: $gcs_path"
    echo "  Model:   $gcs_path/model/"
    echo "  Eval:    $gcs_path/eval/"
    echo "  Config:  $gcs_path/config.yaml"
    echo ""
    echo "To download later:"
    echo "  gsutil -m cp -r $gcs_path/model/ ./outputs/final"
}

stop_vm() {
    # Check if there's a trained model we should save first
    if ssh_cmd "test -d $REMOTE_DIR/outputs/final && echo exists" 2>/dev/null | grep -q exists; then
        echo ""
        echo "WARNING: Trained model found on VM."
        echo "Stopping the VM may discard it (local SSD data is lost)."
        echo ""
        read -p "Upload model to GCS before stopping? [Y/n]: " save_answer
        if [ "${save_answer:-Y}" != "n" ] && [ "${save_answer:-Y}" != "N" ]; then
            save_model
        else
            echo "Skipping model upload."
        fi
        echo ""
    fi

    echo "Stopping VM $INSTANCE..."
    gcloud compute instances stop "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --discard-local-ssd=true
    echo "VM stopped."
}

# ── Main ──────────────────────────────────────────────────────────────
case "${1:-}" in
    deploy)      deploy ;;
    train)       run_train ;;
    bench)       run_bench "$@" ;;
    bench-all)   deploy; echo "Running all benchmarks on $INSTANCE (~50 min)..."; ssh_cmd "cd $REMOTE_DIR && source venv/bin/activate && nohup bash scripts/run_all_benchmarks.sh > ~/bench_all.log 2>&1 &"; echo "Benchmarks started. Monitor with: $0 logs bench_all.log" ;;
    eval)        run_eval ;;
    eval-rlvr)   run_eval_rlvr ;;
    status)      show_status ;;
    logs)        show_logs "${2:-train.log}" ;;
    results)     pull_results ;;
    save-model)  save_model "${2:-}" ;;
    stop)        stop_vm ;;
    *)
        echo "Usage: $0 {deploy|train|bench|eval|eval-rlvr|status|logs|results|save-model|stop}"
        echo ""
        echo "  deploy      - Sync code to VM"
        echo "  train       - Deploy + start training"
        echo "  bench CFG N - Deploy + run 10-min benchmark (config, experiment name)"
        echo "  eval        - Deploy + run MBPP eval (baseline + trained)"
        echo "  eval-rlvr   - Deploy + run RLVR eval (baseline + trained)"
        echo "  status      - Check GPU and running processes"
        echo "  logs FILE   - Tail log file (default: train.log)"
        echo "  results     - Pull eval results and print comparison"
        echo "  save-model  - Upload trained model + eval results to GCS"
        echo "  stop        - Stop the VM (prompts to save model first)"
        exit 1
        ;;
esac
