.PHONY: sandbox install train eval eval-baseline eval-trained eval-rlvr compare test clean

# Build the Docker sandbox image
sandbox:
	docker build -t coding-sandbox:latest -f docker/Dockerfile.sandbox docker/

# Install project dependencies
install:
	pip install -e ".[dev]"

# Run GRPO training with dr_grpo config
train:
	python -m src.train --config configs/dr_grpo.yaml

# Run training with default config
train-default:
	python -m src.train

# ── Evaluation (uses vLLM + Docker sandbox) ─────────────────────────
# Set VLLM_WORKER_MULTIPROC_METHOD=spawn if you hit CUDA fork errors

# Evaluate baseline model on MBPP
eval-baseline:
	VLLM_WORKER_MULTIPROC_METHOD=spawn python -m src.evaluate \
		--model Qwen/Qwen2.5-Coder-7B-Instruct \
		--benchmark mbpp \
		--output outputs/eval_baseline.json

# Evaluate trained LoRA adapter on MBPP
eval-trained:
	VLLM_WORKER_MULTIPROC_METHOD=spawn python -m src.evaluate \
		--model ./outputs/final \
		--benchmark mbpp \
		--is-adapter \
		--output outputs/eval_trained.json

# Evaluate both models on RLVR
eval-rlvr:
	VLLM_WORKER_MULTIPROC_METHOD=spawn python -m src.evaluate \
		--model Qwen/Qwen2.5-Coder-7B-Instruct \
		--benchmark rlvr \
		--output outputs/eval_baseline_rlvr.json
	VLLM_WORKER_MULTIPROC_METHOD=spawn python -m src.evaluate \
		--model ./outputs/final \
		--benchmark rlvr \
		--is-adapter \
		--output outputs/eval_trained_rlvr.json

# Full eval: baseline + trained on MBPP, then compare
eval: eval-baseline eval-trained compare

# Print comparison of baseline vs trained results
compare:
	@python3 scripts/compare_results.py outputs

# Run unit tests
test:
	pytest tests/ -v

# Clean outputs
clean:
	rm -rf outputs/ wandb/ __pycache__ src/__pycache__
