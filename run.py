"""
Autonomous self-improvement loop for post-training.

Uses Claude (Opus) to propose changes to train.py, runs experiments
locally, and keeps only improvements. Designed to run on the GPU machine.

Usage:
    export ANTHROPIC_API_KEY=...
    python3 -u run.py
"""

import os
import re
import subprocess
import time
from datetime import datetime

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_FILE = "train.py"
RESULTS_FILE = "results.tsv"
RUN_TIMEOUT = 720  # 12 min max per experiment

SYSTEM_PROMPT = """\
You are an expert ML researcher optimizing GRPO post-training for a coding agent.

You will be given train.py, the ONLY file you can modify. Your goal is to \
maximize pass@1 on MBPP++ (147 test problems).

## Architecture (DO NOT modify — defined in prepare.py)
- SandboxPool: executes generated code in subprocess/Docker sandboxes
- Dataset: MBPP++ from EvalPlus. Train split (231 problems), test split (147 problems)
- Reward functions (called by GRPOTrainer):
  - code_execution_reward(prompts, completions, test_list, test_setup_code):
    Executes code in sandbox. Returns -0.5 (error/no code), 0.0-1.0 (partial credit), 1.0 (all pass)
  - format_reward(prompts, completions): 1.0 if ```python fenced, else 0.0
- Evaluation: vLLM batched inference in subprocess → sandbox execution → pass@1
- merge_adapter(): merges LoRA into base model for eval

## What you CAN change in train.py
- TrainingConfig fields: model, LoRA (r, alpha, dropout, targets), GRPO \
(num_generations, temperature, beta, loss_type, scale_rewards, num_iterations), \
training (LR, batch size, grad accum, warmup, weight decay, max_grad_norm), \
reward_mode ("partial" or "binary"), reward_weights, vLLM settings, time budget
- Callbacks: early stopping, monitoring, custom callbacks
- Any other logic in train.py

## Constraints
- Do NOT add new imports/dependencies beyond what's in pyproject.toml
- Stay within ~42GB VRAM on an 80GB A100 (training + vLLM colocate)
- Training has a wall-clock time budget (see time_budget_seconds in config)
- The metric is pass@1 on the 147 MBPP++ test problems

## Strategy
- Make ONE focused change per iteration
- Consider what worked/failed in the experiment history
- Think about what matters: reward signal quality, exploration, learning rate, \
batch composition, loss formulation
- Simpler is better when performance is equal
- If many experiments haven't improved, try a bigger/different change
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def ensure_results_header():
    if not os.path.exists(RESULTS_FILE):
        write_file(RESULTS_FILE, "iteration\tpass_at_1\tmemory_gb\tstatus\tdescription\n")


def read_results():
    if os.path.exists(RESULTS_FILE):
        return read_file(RESULTS_FILE)
    return ""


def find_best_pass_at_1():
    """Parse results.tsv to find the best pass@1 so far."""
    best = 0.0
    if not os.path.exists(RESULTS_FILE):
        return best
    for line in read_file(RESULTS_FILE).strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                val = float(parts[1])
                if val > best:
                    best = val
            except ValueError:
                pass
    return best


def parse_output(output):
    """Extract pass@1 and memory from training stdout."""
    baseline = re.search(r"baseline_pass_at_1:\s+([0-9.]+)", output)
    trained = re.search(r"trained_pass_at_1:\s+([0-9.]+)", output)
    memory = re.search(r"memory_gb:\s+([0-9.]+)", output)
    delta = re.search(r"delta_pass_at_1:\s+([+\-0-9.]+)", output)
    steps = re.search(r"num_steps:\s+(\d+)", output)

    if not trained:
        return None

    return {
        "baseline": float(baseline.group(1)) if baseline else None,
        "trained": float(trained.group(1)),
        "memory_gb": float(memory.group(1)) if memory else 0.0,
        "delta": float(delta.group(1)) if delta else None,
        "steps": int(steps.group(1)) if steps else None,
    }


# ---------------------------------------------------------------------------
# LLM: propose a change
# ---------------------------------------------------------------------------

def propose_change(client, train_py, results_history, prev_log=None):
    """Ask Claude to propose a modification to train.py."""

    user_parts = [
        f"Here is the current train.py:\n\n```python\n{train_py}\n```\n",
        f"Experiment history (tab-separated):\n```\n{results_history}\n```\n",
    ]

    if prev_log:
        trimmed = prev_log[-4000:]
        user_parts.append(
            f"Log from the most recent experiment:\n```\n{trimmed}\n```\n"
        )

    user_parts.append(
        "Propose ONE focused change to improve pass@1. "
        "First, in 2-3 sentences explain what you're changing and why. "
        "Then return the COMPLETE modified train.py inside ```python ... ``` markers."
    )

    text = ""
    with client.messages.stream(
        model="claude-opus-4-20250514",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(user_parts)}],
    ) as stream:
        for chunk in stream.text_stream:
            text += chunk

    explanation = text.split("```python")[0].strip()
    explanation_oneline = " ".join(explanation.replace("\n", " ").split())[:200]

    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError("No ```python code block found in LLM response")

    new_code = match.group(1)
    return new_code, explanation_oneline


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_experiment():
    """Run training locally and return output."""
    print("  Running training...")
    r = subprocess.run(
        ["uv", "run", "train.py", "--config", "configs/default.yaml"],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )
    return r.stdout + r.stderr, r.returncode


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    client = anthropic.Anthropic()

    ensure_results_header()

    best_train_py = read_file(TRAIN_FILE)
    best_pass_at_1 = find_best_pass_at_1()
    prev_log = None
    iteration = 0
    consecutive_failures = 0

    print(f"Best pass@1 so far: {best_pass_at_1:.4f}")
    print(f"Starting autonomous improvement loop...\n")

    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}  |  best={best_pass_at_1:.4f}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        # Always start from the best version
        write_file(TRAIN_FILE, best_train_py)

        # 1. Propose change
        print("  Asking Claude for a change...")
        try:
            new_code, explanation = propose_change(
                client, best_train_py, read_results(), prev_log
            )
        except Exception as e:
            print(f"  ERROR proposing change: {e}")
            time.sleep(10)
            continue

        print(f"  Change: {explanation[:120]}")

        # 2. Write new train.py and run
        write_file(TRAIN_FILE, new_code)

        t0 = time.time()
        try:
            # Clean outputs to avoid stale merge cache
            subprocess.run(["rm", "-rf", "outputs"], check=False)
            output, returncode = run_experiment()
        except subprocess.TimeoutExpired:
            output = "TIMEOUT: experiment exceeded time limit"
            returncode = 1
        elapsed = time.time() - t0
        prev_log = output

        # 3. Parse results
        results = parse_output(output)

        if results is None or returncode != 0:
            status = "crash"
            pass_at_1 = 0.0
            memory_gb = 0.0
            consecutive_failures += 1
            print(f"  CRASH (took {elapsed:.0f}s, consecutive={consecutive_failures})")
            for line in output.strip().split("\n")[-5:]:
                print(f"    {line}")
        else:
            pass_at_1 = results["trained"]
            memory_gb = results.get("memory_gb", 0.0)
            consecutive_failures = 0

            if pass_at_1 > best_pass_at_1:
                status = "keep"
                old_best = best_pass_at_1
                best_pass_at_1 = pass_at_1
                best_train_py = new_code
                print(f"  KEEP: pass@1={pass_at_1:.4f} (was {old_best:.4f}) "
                      f"memory={memory_gb}GB steps={results.get('steps')} ({elapsed:.0f}s)")
            else:
                status = "discard"
                # Restore best train.py
                write_file(TRAIN_FILE, best_train_py)
                print(f"  DISCARD: pass@1={pass_at_1:.4f} <= best={best_pass_at_1:.4f} "
                      f"memory={memory_gb}GB steps={results.get('steps')} ({elapsed:.0f}s)")

        # 4. Log to results.tsv
        with open(RESULTS_FILE, "a") as f:
            desc = explanation[:100].replace("\t", " ")
            f.write(f"{iteration}\t{pass_at_1:.6f}\t{memory_gb}\t{status}\t{desc}\n")

        # Safety: if too many consecutive crashes, pause
        if consecutive_failures >= 5:
            print("\n  WARNING: 5 consecutive crashes. Pausing 60s...")
            time.sleep(60)
            consecutive_failures = 0


if __name__ == "__main__":
    main()
