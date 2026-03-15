GPU_HOST := ubuntu@38.128.232.129

.PHONY: ssh sandbox install train run eval eval-baseline eval-trained compare test clean

# SSH into the GPU instance
ssh:
	ssh $(GPU_HOST)

# Build the Docker sandbox image
sandbox:
	docker build -t coding-sandbox:latest -f docker/Dockerfile.sandbox docker/

# Install project dependencies (bootstraps uv if missing)
install:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$$HOME/.local/bin:$$PATH"; }
	PATH="$$HOME/.local/bin:$$PATH" uv sync --all-extras

# Run GRPO training with default config (cleans outputs first)
train:
	rm -rf outputs/
	WANDB_MODE=disabled uv run train.py --config configs/default.yaml

# Run autonomous self-improvement loop
run:
	uv run run.py

# Run training with built-in defaults
train-default:
	uv run train.py

# Run unit tests
test:
	uv run pytest tests/ -v

# Clean outputs
clean:
	rm -rf outputs/ wandb/ __pycache__
