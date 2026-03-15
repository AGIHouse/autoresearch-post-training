# autoresearch-post-training

Autonomous self-improvement loop for GRPO post-training on MBPP++.

## Architecture

- **`run.py`** — Orchestration loop. Calls Claude (Opus) to propose changes, runs experiments on remote GPU, keeps only improvements. Runs indefinitely.
- **`train.py`** — The ONLY file the LLM modifies. Contains TrainingConfig, LoRA/GRPO setup, callbacks, and training loop.
- **`prepare.py`** — Fixed infrastructure (READ-ONLY). Sandbox, MBPP++ dataset, reward functions, evaluation.
- **`configs/default.yaml`** — Default hyperparameters.
- **`results.tsv`** — Experiment log (tab-separated). Tracks pass@1, memory, keep/discard per commit.

## How it works

```
LOOP FOREVER:
  1. Read current best train.py + experiment history
  2. Call Claude (Opus) to propose ONE focused change
  3. Write modified train.py, git commit
  4. Push to remote GPU, run training (5 min budget)
  5. Parse results: pass@1 on MBPP++ test split (147 problems)
  6. If pass@1 improved → KEEP (update best). Otherwise → DISCARD (revert to best).
  7. Log result to results.tsv
```

## Running

```bash
export ANTHROPIC_API_KEY=...
uv run run.py                    # new branch autoresearch/<date>
uv run run.py --tag experiment1  # custom branch name
uv run run.py --resume           # resume on current branch
```

## What the LLM can change in train.py

- GRPO hyperparameters (num_generations, temperature, beta, loss_type, scale_rewards)
- LoRA configuration (rank, alpha, dropout, target modules)
- Training hyperparameters (LR, batch size, gradient accumulation, warmup, weight decay)
- Reward mode (partial vs binary), reward weights
- Callbacks and monitoring
- vLLM settings
- Model choice (must fit in ~42GB VRAM with vLLM colocate)

## Constraints

- Do NOT modify `prepare.py` — it's fixed infrastructure
- Do NOT add new dependencies
- Stay within ~42GB VRAM on an 80GB A100
- Each experiment must complete within the time budget
- The metric is pass@1 on MBPP++ test split (147 problems)
