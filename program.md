# autoresearch-post-training

This is an experiment to have an LLM autonomously research post-training techniques for coding agents using GRPO on MBPP++ (EvalPlus).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed infrastructure: sandbox, MBPP++ dataset loading, reward functions, evaluation. Do not modify.
   - `train.py` — the file you modify. GRPO config, LoRA setup, callbacks, training loop.
   - `configs/default.yaml` — default hyperparameters.
4. **Verify sandbox**: Check that Docker sandbox is built (`docker images | grep coding-sandbox`). If not, tell the human to run `make sandbox`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs on a single GPU. Launch with: `uv run train.py` or `uv run train.py --config configs/default.yaml`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - GRPO hyperparameters (num_generations, temperature, beta, loss_type)
  - LoRA configuration (rank, alpha, target modules)
  - Training hyperparameters (LR, batch size, gradient accumulation)
  - Reward mode (partial vs binary)
  - Callbacks and monitoring
  - vLLM settings

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed sandbox, MBPP++ dataset loading, reward functions, and evaluation.
- Install new packages or add dependencies.
- Modify the reward functions or evaluation harness.

**The goal is simple: maximize pass@1 on MBPP++.** Try different GRPO configurations, LoRA settings, reward modes, etc.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Establish the baseline by running the training script as is.

## Output format

The training logs metrics to WandB. After training, evaluate with the evaluation functions in `prepare.py`.

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	pass_at_1	memory_gb	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Tune `train.py` with an experimental idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Check results
6. If improved, keep. If not, git reset back.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Continue indefinitely.
