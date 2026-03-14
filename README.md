# post_train: RL Training for Coding Agents

A complete system for improving a coding LLM's ability to solve programming problems using **reinforcement learning**. The model generates Python solutions, executes them against test cases in a secure sandbox, and learns from the results using GRPO (Group Relative Policy Optimization).

## What This Does

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GRPO Training Loop                           │
│                                                                      │
│  ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐  │
│  │ Dataset  │────▶│  Model   │────▶│ Sandbox  │────▶│   Reward    │  │
│  │ (MBPP/  │     │ (vLLM)   │     │ (Docker/ │     │ Computation │  │
│  │  RLVR)  │     │          │     │  subproc)│     │             │  │
│  └─────────┘     │ Generate │     │ Execute  │     │ pass/fail   │  │
│                  │ 8 sols   │     │ + test   │     │ per group   │  │
│                  │ per prob │     │          │     │             │  │
│                  └──────────┘     └──────────┘     └──────┬──────┘  │
│                                                           │         │
│                  ┌──────────┐     ┌──────────────────────┐│         │
│                  │  Update  │◀────│ Group-Relative       ││         │
│                  │  Policy  │     │ Advantages           ││         │
│                  │  (LoRA)  │     │                      │◀         │
│                  │          │     │ A_i = (r_i - μ) / σ  │          │
│                  └──────────┘     └──────────────────────┘          │
│                                                                      │
│  The model learns: "for this problem, solution 3 scored 1.0 while   │
│  solutions 1,2,4-8 scored 0.0-0.3, so generate more like solution 3"│
└──────────────────────────────────────────────────────────────────────┘
```

### Results (First Run)

| Benchmark | Baseline | Trained | Delta |
|-----------|----------|---------|-------|
| **MBPP** (500 problems) | 78.0% (390/500) | 79.0% (395/500) | **+1.0 pp** |
| **RLVR** (2550 problems) | 20.5% (523/2550) | 30.7% (784/2550) | **+10.2 pp** |

Trained with Dr.GRPO (binary rewards, no KL penalty) for 200 steps on MBPP. The large RLVR improvement (+10.2 pp) shows meaningful generalization to harder, unseen competitive programming problems.

## Quick Start

```bash
# 1. Clone and install
git clone <repo_url> && cd post_train
pip install -e ".[dev]"

# 2. Build the code execution sandbox (optional — subprocess fallback available)
make sandbox

# 3. Run tests
make test

# 4. Train
make train
# or with custom config:
python src/train.py --config configs/default.yaml
# or with Dr.GRPO (removes length/std bias, binary rewards):
python src/train.py --config configs/dr_grpo.yaml
```

## Architecture

### Components

| Component | File | Purpose |
|-----------|------|---------|
| **Config** | `src/config.py` | All hyperparameters in one dataclass |
| **Dataset** | `src/dataset.py` | Loads MBPP, RLVR, or OpenCoder for TRL |
| **Sandbox** | `src/sandbox.py` | Docker or subprocess sandbox with parallel pool |
| **Reward** | `src/reward.py` | Scores completions via test execution (partial or binary) |
| **Training** | `src/train.py` | Orchestrates GRPOTrainer |
| **Evaluation** | `src/evaluate.py` | Measures pass@1 on benchmarks |

### Standing on the Shoulders of Giants

This system's design is informed by several production RL-for-code systems:

| Project | What We Learned |
|---------|----------------|
| **[Open-R1](https://github.com/huggingface/open-r1)** (HuggingFace) | TRL GRPOTrainer + vLLM colocate + binary rewards. Our training loop mirrors theirs. |
| **[DeepSWE](https://www.together.ai/blog/deepswe)** (Together AI) | GRPO++ with no KL, no std normalization, binary rewards. Inspired our `dr_grpo` config and binary reward mode. |
| **[AceCoder](https://github.com/TIGER-AI-Lab/AceCoder)** (TIGER-AI-Lab) | Synthesized ~16 test cases per problem, +25% on HumanEval+ in 80 steps. Informed our partial credit reward design. |
| **[Modal GRPO Example](https://modal.com/docs/examples/grpo_trl)** | End-to-end TRL + sandbox reference. Validated our architecture choices. |
| **[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)** | Found vanilla GRPO didn't always work — sometimes PPO+GAE is better. Good to know as a fallback. |
| **[EvalPlus](https://github.com/evalplus/evalplus)** | Docker sandbox with configurable resource limits. Our sandbox design follows their patterns. |

### Key Design Decisions

#### Why GRPO over PPO?

PPO requires a **critic network** (value function) that's roughly the same size as the policy model. For a 7B model, that's an extra ~14GB of GPU memory. GRPO eliminates this by using a clever trick:

Instead of learning V(s) to compute advantages, GRPO **samples G completions per prompt** and uses the group's mean reward as the baseline:

```
A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
```

This tells us: "completion i scored 0.3 above the group average" or "completion i scored 0.2 below average." The model then increases the probability of above-average completions and decreases below-average ones.

**Result**: Same quality as PPO, half the memory, simpler code.

Note: [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) found that PPO with GAE can sometimes outperform vanilla GRPO. If training stalls, this is worth exploring.

#### Why LoRA instead of full fine-tuning?

Full GRPO fine-tuning of a 7B model needs:
- Model weights: 14GB (bf16)
- Reference model: 14GB (bf16) — needed for KL penalty
- Optimizer states: 28GB+ (AdamW stores momentum + variance)
- Gradients + activations: 20-30GB

**Total: >76GB** — exceeds A100 80GB.

LoRA freezes the base model and adds small adapter matrices (~67M parameters for r=64). Only the adapters are trained, so optimizer states shrink to ~0.5GB. Total memory drops to ~55-65GB.

#### Sandbox: Docker vs Subprocess

The model generates arbitrary Python code. Two sandbox backends are supported:

| Backend | Isolation | Speed | Use Case |
|---------|-----------|-------|----------|
| **Docker** | Full (network, fs, memory, PID limits) | ~100ms startup | Production training |
| **Subprocess** | Timeout only | Instant | Local dev, CI, debugging |

Docker is preferred for production (EvalPlus, SWE-bench, Open-R1 all use Docker or microVMs). The subprocess backend enables local development without Docker.

The pool auto-detects: tries Docker first, falls back to subprocess.

For even stronger isolation, consider [E2B](https://e2b.dev/) (Firecracker microVMs, 80ms boot) or [Modal Sandboxes](https://modal.com/blog/what-is-ai-code-sandbox).

#### Partial Credit vs Binary Rewards

Both modes are supported, configurable via `reward_mode`:

| Mode | Reward | When to Use | Who Uses It |
|------|--------|-------------|-------------|
| **`partial`** | passed/total (0.0-1.0) | Few test cases (MBPP's 3), early experiments | AceCoder |
| **`binary`** | 1.0 if all pass, else 0.0 | Many test cases (16+), production training | DeepSWE, Open-R1 |

DeepSWE (Together AI) found that **sparse binary rewards outperform dense feedback** for coding. However, with MBPP's 3 tests per problem, partial credit provides more learning signal.

**Recommendation**: Start with `partial` for MBPP, switch to `binary` if using RLVR or datasets with many tests.

### Reward Function Design

Two reward signals are summed (with weights):

```
total_reward = 1.0 * code_execution_reward + 0.1 * format_reward
```

**Code execution reward** (`src/reward.py:code_execution_reward`):
| Outcome | Partial Mode | Binary Mode |
|---------|-------------|-------------|
| All tests pass | 1.0 | 1.0 |
| Some tests pass | passed/total | 0.0 |
| Code errors | -0.5 | -0.5 |
| No code extracted | -0.5 | -0.5 |

**Format reward** (`src/reward.py:format_reward`):
| Outcome | Reward |
|---------|--------|
| Has ` ```python ... ``` ` block | 1.0 (weighted to 0.1) |
| No proper block | 0.0 |

### Supported Datasets

| Dataset | Size | Tests/Problem | Best For |
|---------|------|---------------|----------|
| **MBPP** (`mbpp`) | 374 train | 3-6 | Quick experiments, debugging |
| **RLVR** (`rlvr`) | Larger | More | Production training with verified rewards |
| **OpenCoder** (`opencoder`) | Large | Validated | TRL's official example dataset |

Set via `dataset_name` in config.

## Training Configurations

Two configs are provided:

### `configs/default.yaml` — Standard GRPO
Standard setup with partial credit rewards. Good starting point.

### `configs/dr_grpo.yaml` — Dr. GRPO (recommended for production)
Based on DeepSWE and Dr.GRPO findings:
- `loss_type: "dr_grpo"` — removes length and std normalization biases
- `reward_mode: "binary"` — sparse rewards (proven better by DeepSWE)
- `scale_rewards: false` — no std normalization within group
- `beta: 0.0` — no KL penalty

### Key Parameters

| Parameter | Default | What It Controls |
|-----------|---------|-----------------|
| `num_generations` | 8 | Completions per prompt (G in GRPO). Higher = better advantage estimates but slower |
| `beta` | 0.0 | KL penalty coefficient. 0 = disabled (per DeepSWE, Open-Reasoner-Zero) |
| `loss_type` | "grpo" | Loss variant: "grpo", "dr_grpo" (no length/std bias), "dapo" |
| `reward_mode` | "partial" | "partial" (fractional) or "binary" (all-or-nothing) |
| `learning_rate` | 5e-6 | Conservative for RL stability |
| `temperature` | 0.9 | Sampling temp during rollouts |
| `dataset_name` | "mbpp" | Dataset: "mbpp", "rlvr", or "opencoder" |
| `sandbox_backend` | "auto" | "auto", "docker", or "subprocess" |

### Performance (Benchmarked)

On a single H100 80GB with the optimized config:

| Metric | Value |
|--------|-------|
| **Median step time** | 2.35 s |
| **Steps/hour** | ~255 |
| **200-step run** | ~13 min |
| **Completions/step** | 64 (8 prompts × 8 generations) |

See `EXPERIMENT_LOG.md` for the full optimization study (subprocess sandbox, vLLM memory tuning, iteration reuse).

### Memory Budget

#### Single GPU (7B model, 1x H100 80GB)

```
Base model (bf16):              14 GB
Reference model (bf16):         14 GB
LoRA adapters + optimizer:       1 GB
vLLM KV cache (0.45 util):     36 GB
Activations (grad checkpoint):  10 GB
Overhead:                        5 GB
─────────────────────────────────────
Total:                         ~80 GB  ✓ fits in 80GB
```

#### Multi-GPU Scaling (LoRA GRPO)

| Model Size | GPUs Needed | Key Requirements |
|------------|-------------|------------------|
| **7B** | 1x H100 | Colocate mode, LoRA |
| **14B** | 2x H100 | ZeRO-2 or colocate TP=2 |
| **32B** | 4x H100 | ZeRO-3, TP=4 for vLLM |
| **70-72B** | 8x H100 | ZeRO-3, TP=8, vLLM sleep mode, batch_size=1 |

**72B is the max for 8x H100** with LoRA GRPO (confirmed by HuggingFace with TRL + vLLM colocate).

## Infrastructure: GCP Setup

### Create a GPU Instance

```bash
export GCP_PROJECT_ID=your-project-id

# H100 (recommended)
./scripts/setup_gcp.sh create h100

# A100 (budget option)
./scripts/setup_gcp.sh create a100
```

Available configurations:

| Instance | GPU | Cost (on-demand) |
|----------|-----|-------------------|
| `a3-highgpu-1g` | 1x H100 80GB | ~$7.35/hr |
| `a2-highgpu-1g` | 1x A100 40GB | ~$3.67/hr |
| `a2-ultragpu-1g` | 1x A100 80GB | ~$5.00/hr |

### Install Dependencies

```bash
# SSH into the instance
gcloud compute ssh coding-agent-rl --zone=us-central1-a

# Clone the repo and install
git clone <repo_url> && cd post_train
./scripts/setup_gcp.sh install
```

### Cost Estimate (H100)

| Phase | Hours | Cost |
|-------|-------|------|
| Setup + debugging | 2 | $14.70 |
| Training (200 steps) | 0.25 | $1.84 |
| Evaluation | 0.5 | $3.68 |
| Benchmarking | 1 | $7.35 |
| **Total** | **~4** | **~$28** |

Well within the $1000 budget.

## Monitoring Training

Training logs to [Weights & Biases](https://wandb.ai). Key metrics to watch:

### Healthy Training Signals

| Metric | Expected Trend | Problem If... |
|--------|---------------|---------------|
| `reward/code_execution_reward/mean` | 0.1 → 0.5-0.8 over 200 steps | Flat = no learning. Drops = instability |
| `reward/format_reward/mean` | Quickly stabilizes near 1.0 | Stays at 0 = model ignoring format |
| `completions/mean_length` | Stable, slight decrease | Collapses to 0 = mode collapse. Explodes = verbosity gaming |
| `entropy` | Gradual decrease | Sharp drop = mode collapse |

### Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM | Reduce `vllm_gpu_memory_utilization` to 0.2, reduce batch size to 1 |
| Reward not improving | Check sandbox is working (`make test`). Try `loss_type: "dr_grpo"` |
| Response length inflation | Switch to `configs/dr_grpo.yaml` (removes length normalization) |
| Mode collapse | Increase temperature to 1.0. Set `beta: 0.01` for light KL penalty |
| Loss spikes | Reduce `learning_rate` to 1e-6. Check `max_grad_norm` is 1.0 |
| GRPO not converging | Consider PPO+GAE (per Open-Reasoner-Zero findings) |

## Deploy & Run (GCP VM)

A single script handles the full workflow:

```bash
# Sync code to VM
./scripts/deploy_and_run.sh deploy

# Deploy + start training
./scripts/deploy_and_run.sh train

# Deploy + run eval (baseline + trained on MBPP)
./scripts/deploy_and_run.sh eval

# Deploy + run RLVR eval
./scripts/deploy_and_run.sh eval-rlvr

# Monitor
./scripts/deploy_and_run.sh status       # GPU usage + running processes
./scripts/deploy_and_run.sh logs         # Tail train.log (or: logs eval.log)

# Pull results and print comparison table
./scripts/deploy_and_run.sh results

# Save trained model to GCS before stopping
./scripts/deploy_and_run.sh save-model

# Stop VM (prompts to save model if one exists)
./scripts/deploy_and_run.sh stop
```

Configure via environment variables: `GCP_PROJECT_ID`, `GCP_ZONE`, `GCP_INSTANCE`, `GCS_BUCKET`.

## Evaluation

Uses vLLM for fast batched inference and Docker sandbox pool for parallel test execution.

```bash
# Evaluate baseline on MBPP (~3s for 500 completions on H100)
python -m src.evaluate --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark mbpp

# Evaluate trained LoRA adapter (auto-merges into base model)
python -m src.evaluate --model ./outputs/final --benchmark mbpp --is-adapter

# Evaluate on RLVR (~2550 competitive programming problems)
python -m src.evaluate --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark rlvr

# Compare results
make compare
```

## Extending the System

### Adding a New Dataset

Add a loader function in `src/dataset.py`:

```python
def _load_my_dataset(split: str = "train") -> Dataset:
    ds = load_dataset("my-org/my-dataset", split=split)
    def format_row(row):
        return {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": row["problem"]}],
            "test_list": row["tests"],
            "test_setup_code": "",
        }
    return ds.map(format_row, remove_columns=ds.column_names)
```

Then add it to the `loaders` dict in `load_dataset_for_grpo()`.

### Adding a New Reward Function

1. Define a function in `src/reward.py`:
   ```python
   def my_reward(prompts: list, completions: list, **kwargs) -> list[float]:
   ```

2. Add it to the `reward_funcs` list in `src/train.py`.
3. Add its weight to `reward_weights` in config.

### Scaling to Multi-GPU

```bash
# 4x H100 with DeepSpeed ZeRO-3
accelerate launch \
    --num_processes 4 \
    --config_file configs/deepspeed_zero3.yaml \
    src/train.py
```

For 70B+ models on 8x H100, key config changes:

```yaml
# configs/dr_grpo_72b.yaml
model_name: "Qwen/Qwen2.5-Coder-72B-Instruct"
per_device_train_batch_size: 1
vllm_gpu_memory_utilization: 0.5
vllm_sleep_enabled: true      # essential for 70B+ — frees training memory during generation
# Requires: DeepSpeed ZeRO-3, vLLM tensor_parallel_size=8
```

For purpose-built distributed RL frameworks (async gen/train, Ray scheduling):
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — fastest, Ray-based
- [veRL](https://github.com/volcengine/verl) (ByteDance) — best docs, FSDP-native

## Project Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| TRL | >= 0.23.0 | GRPOTrainer with vLLM sleep support |
| transformers | >= 4.48.0 | Model loading, tokenization |
| PEFT | >= 0.14.0 | LoRA adapters |
| vLLM | >= 0.8.5 | Fast generation with sleep() API |
| docker (Python) | >= 7.0.0 | Docker sandbox (optional) |
| datasets | >= 3.2.0 | Dataset loading |
| wandb | >= 0.19.0 | Experiment tracking |
| evalplus | >= 0.3.1 | MBPP+ / HumanEval+ evaluation |

## References

### Papers
- [GRPO: DeepSeekMath](https://arxiv.org/abs/2402.03300) — The algorithm we use
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO applied to reasoning at scale
- [AceCoder](https://arxiv.org/abs/2502.01718) — Test case synthesis for RL (+25% on HumanEval+)
- [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186) — Base model technical report

### Implementations We Built On
- [Open-R1](https://github.com/huggingface/open-r1) — HuggingFace's DeepSeek-R1 reproduction
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — Framework docs
- [DeepSWE](https://www.together.ai/blog/deepswe) — GRPO++ for software engineering
- [Modal GRPO Example](https://modal.com/docs/examples/grpo_trl) — End-to-end reference
- [EvalPlus](https://github.com/evalplus/evalplus) — Evaluation framework
- [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) — PPO vs GRPO findings
- [NousResearch RLVR](https://huggingface.co/datasets/NousResearch/RLVR_Coding_Problems) — RL-ready coding dataset
