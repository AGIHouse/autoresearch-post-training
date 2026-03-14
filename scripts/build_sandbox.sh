#!/bin/bash
# Build the Docker sandbox image for code execution
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building sandbox image..."
docker build -t coding-sandbox:latest -f "$PROJECT_DIR/docker/Dockerfile.sandbox" "$PROJECT_DIR/docker/"
echo "Done! Image: coding-sandbox:latest"
