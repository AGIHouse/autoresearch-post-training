# autoresearch-post-training

Autonomous AI agents conducting post-training research. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for post-training (SFT, DPO, RLHF).

The agent modifies `train.py`, runs 5-minute experiments, and iterates to minimize `val_bpb`.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py      # download model + data (~5 min, one-time)
uv run train.py        # ~5 min per experiment
```

## Structure

| File | Role | Who edits |
|------|------|-----------|
| `prepare.py` | Data prep, evaluation, constants | Human only (read-only for agent) |
| `train.py` | SFT/DPO training loop, hyperparams, LoRA config | Agent |
| `program.md` | Agent instructions | Human |

## How it works

1. Agent creates a branch `autoresearch/<tag>`
2. Establishes baseline by running `train.py` unmodified
3. Loops forever: edit `train.py` → commit → run → evaluate → keep/discard
4. Results logged to `results.tsv` (untracked)

See `program.md` for full agent instructions.
