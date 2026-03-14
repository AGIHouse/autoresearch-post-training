#!/bin/bash
# ============================================================================
# GCP GPU Instance Setup Script
# ============================================================================
#
# Provisions a GCP VM with an H100 80GB GPU and sets up the training env.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GPU quota in the target region
#   - Billing enabled
#
# Usage:
#   ./scripts/setup_gcp.sh create          # Create H100 instance (default)
#   ./scripts/setup_gcp.sh create a100     # Create A100 instance instead
#   ./scripts/setup_gcp.sh install         # Install deps (run inside instance)
#
# Cost:
#   a3-highgpu-1g (1x H100 80GB): ~$7.35/hr on-demand
#   a2-highgpu-1g (1x A100 80GB): ~$3.67/hr on-demand
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-agent-rl-lift}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-coding-agent-rl}"
BOOT_DISK_SIZE="200GB"
IMAGE_FAMILY="common-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# ── Functions ──────────────────────────────────────────────────────────────

create_instance() {
    local gpu_type="${1:-h100}"

    if [ -z "$PROJECT_ID" ]; then
        echo "ERROR: Set GCP_PROJECT_ID environment variable"
        exit 1
    fi

    if [ "$gpu_type" = "h100" ]; then
        MACHINE_TYPE="a3-highgpu-1g"
        ACCELERATOR="type=nvidia-h100-80gb,count=1"
        GPU_DESC="1x H100 80GB (~$7.35/hr)"
    elif [ "$gpu_type" = "a100" ]; then
        MACHINE_TYPE="a2-highgpu-1g"
        ACCELERATOR="type=nvidia-tesla-a100,count=1"
        GPU_DESC="1x A100 40GB (~$3.67/hr)"
    elif [ "$gpu_type" = "a100-80gb" ]; then
        MACHINE_TYPE="a2-ultragpu-1g"
        ACCELERATOR="type=nvidia-a100-80gb,count=1"
        GPU_DESC="1x A100 80GB (~$5.00/hr)"
    else
        echo "ERROR: Unknown GPU type '$gpu_type'. Use: h100, a100, a100-80gb"
        exit 1
    fi

    echo "Creating GCP instance: $INSTANCE_NAME"
    echo "  GPU: $GPU_DESC"
    echo "  Machine: $MACHINE_TYPE"
    echo "  Zone: $ZONE"
    echo "  Project: $PROJECT_ID"
    echo ""

    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator="$ACCELERATOR" \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type="pd-ssd" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --maintenance-policy=TERMINATE \
        --metadata="install-nvidia-driver=True" \
        --scopes="default,storage-rw"

    echo ""
    echo "Instance created! SSH into it with:"
    echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
    echo ""
    echo "Then run: ./scripts/setup_gcp.sh install"
}

install_dependencies() {
    echo "============================================"
    echo "Installing training environment..."
    echo "============================================"

    # Update system
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker.io

    # Ensure Docker is running and accessible
    sudo systemctl start docker
    sudo usermod -aG docker "$USER"
    echo "NOTE: You may need to log out and back in for Docker group to take effect."
    echo "Or run: newgrp docker"

    # Install Python dependencies
    pip install --upgrade pip
    pip install -e ".[dev]"

    # Build the Docker sandbox
    echo "Building Docker sandbox image..."
    sudo docker build -t coding-sandbox:latest -f docker/Dockerfile.sandbox docker/

    # Verify GPU
    echo ""
    echo "============================================"
    echo "Verifying GPU..."
    echo "============================================"
    nvidia-smi

    echo ""
    echo "============================================"
    echo "Setup complete!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "  1. wandb login"
    echo "  2. python src/train.py  (or: make train)"
    echo ""
}

# ── Main ───────────────────────────────────────────────────────────────────

case "${1:-}" in
    create)
        create_instance "${2:-h100}"
        ;;
    install)
        install_dependencies
        ;;
    *)
        echo "Usage: $0 {create|install} [gpu_type]"
        echo ""
        echo "  create [h100|a100|a100-80gb]  - Create GPU instance (default: h100)"
        echo "  install                        - Install dependencies (run inside instance)"
        exit 1
        ;;
esac
