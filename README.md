# autoresearch-post-training

Autonomous AI agents conducting post-training research for coding agents. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for GRPO (Group Relative Policy Optimization) training.

The agent modifies `train.py`, runs experiments, and iterates to maximize coding benchmark performance (pass@1).

## Architecture

GRPO training loop (per step):
1. Sample batch of coding prompts from dataset
2. Generate G=8 completions per prompt using vLLM
3. Execute each completion in a sandboxed environment against test cases
4. Compute group-relative advantages
5. Update policy using clipped surrogate objective

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                       # install dependencies
make sandbox                  # build Docker sandbox image
uv run train.py               # run with defaults
uv run train.py --config configs/dr_grpo.yaml  # run with Dr.GRPO config
```

## Structure

| File | Role | Who edits |
|------|------|-----------|
| `prepare.py` | Sandbox, datasets, rewards, evaluation | Human only (read-only for agent) |
| `train.py` | GRPO config, LoRA setup, callbacks, training loop | Agent |
| `program.md` | Agent instructions | Human |
| `configs/` | YAML config presets (default, dr_grpo) | Either |
| `docker/` | Sandbox container (Dockerfile + runner) | Human only |

## How it works

1. Agent creates a branch `autoresearch/<tag>`
2. Establishes baseline by running `train.py` unmodified
3. Loops forever: edit `train.py` → commit → run → evaluate → keep/discard
4. Results logged to `results.tsv` (untracked)

See `program.md` for full agent instructions.
