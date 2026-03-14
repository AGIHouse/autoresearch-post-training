GPU_HOST := root@165.245.141.38

.PHONY: ssh sandbox install train eval eval-baseline eval-trained compare test clean

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

# Run GRPO training with dr_grpo config
train:
	uv run train.py --config configs/dr_grpo.yaml

# Run training with default config
train-default:
	uv run train.py

# Run unit tests
test:
	uv run pytest tests/ -v

# Clean outputs
clean:
	rm -rf outputs/ wandb/ __pycache__
