"""
Reward functions for GRPO training of a coding agent.

Two reward signals are combined:

1. **Code execution reward** (weight=1.0):
   Runs the model's generated code against test cases in a sandbox.
   Supports two modes:
   - "partial": reward = passed_tests / total_tests (denser signal)
   - "binary":  reward = 1.0 if all tests pass, 0.0 otherwise (sparser but
                 DeepSWE/Together AI found this works better for coding)

2. **Format reward** (weight=0.1):
   Small bonus for wrapping code in ```python ... ``` markers.

Design decisions informed by research:
- DeepSWE (Together AI): Sparse binary rewards outperform dense feedback
- AceCoder: Partial credit works well with many test cases (16+/problem)
- Open-R1: Uses binary pass/fail with E2B sandboxes
- Recommendation: Use "binary" with many tests, "partial" with few tests (MBPP)

These functions follow TRL's reward function signature:
    def reward_fn(prompts, completions, **kwargs) -> list[float]
    where **kwargs receives extra dataset columns (test_list, test_setup_code).
"""

import logging
import re
from typing import Any

from src.sandbox import SandboxPool, ExecutionResult
from src.utils import extract_code_from_completion

logger = logging.getLogger(__name__)

# Module-level sandbox pool, initialized by train.py before training starts
_sandbox_pool: SandboxPool | None = None

# Reward mode: "partial" (default) or "binary"
_reward_mode: str = "partial"


def set_sandbox_pool(pool: SandboxPool | None) -> None:
    """Set the global sandbox pool used by the reward function."""
    global _sandbox_pool
    _sandbox_pool = pool


def set_reward_mode(mode: str) -> None:
    """Set the reward mode: 'partial' for fractional credit, 'binary' for all-or-nothing."""
    global _reward_mode
    if mode not in ("partial", "binary"):
        raise ValueError(f"reward_mode must be 'partial' or 'binary', got '{mode}'")
    _reward_mode = mode
    logger.info(f"Reward mode set to: {mode}")


def get_sandbox_pool() -> SandboxPool:
    """Get the global sandbox pool, raising if not initialized."""
    if _sandbox_pool is None:
        raise RuntimeError(
            "Sandbox pool not initialized. Call set_sandbox_pool() before training."
        )
    return _sandbox_pool


def _score_result(result: ExecutionResult) -> float:
    """
    Compute reward from an execution result based on the current mode.

    Binary mode (DeepSWE-style):
        1.0 if all tests pass, 0.0 otherwise.
        Simpler signal, avoids rewarding partially-correct solutions
        that might learn wrong patterns.

    Partial mode (AceCoder-style):
        passed / total. Denser gradient signal, especially useful when
        you have few test cases per problem (like MBPP's 3).
    """
    if result.total == 0:
        return 0.0

    if result.passed == 0 and result.errors:
        # Code errored out (syntax error, runtime crash, timeout)
        return -0.5

    if _reward_mode == "binary":
        return 1.0 if result.all_passed else 0.0
    else:
        return result.pass_rate


def code_execution_reward(
    prompts: list,
    completions: list,
    test_list: list[list[str]] | None = None,
    test_setup_code: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """
    Primary reward: execute generated code against test cases.

    How GRPO uses this:
        For each prompt, GRPO samples G=8 completions. This function receives
        ALL completions across ALL prompts in the batch (flattened).
        The test_list and test_setup_code are repeated/aligned by TRL to match.

    Reward scale (partial mode):
        -0.5  : code doesn't parse or raises an exception before tests run
         0.0  : code runs but passes 0 tests
         0.0-1.0 : partial credit (passed / total)
         1.0  : all tests pass

    Reward scale (binary mode):
        -0.5  : code doesn't parse or crashes
         0.0  : code runs but doesn't pass all tests
         1.0  : all tests pass

    Args:
        prompts: Input prompts (not used, but required by TRL signature)
        completions: Model-generated completions
        test_list: Test assertions for each completion's corresponding prompt
        test_setup_code: Setup code for each completion's corresponding prompt

    Returns:
        List of float rewards, one per completion
    """
    pool = get_sandbox_pool()

    if test_list is None:
        logger.warning("No test_list provided to reward function, returning 0.0 for all")
        return [0.0] * len(completions)

    # Extract code from each completion
    codes = [extract_code_from_completion(c) for c in completions]

    # Build execution items
    items = []
    for i, code in enumerate(codes):
        if code is None:
            items.append(None)  # will skip execution
        else:
            tests = test_list[i] if i < len(test_list) else []
            setup = test_setup_code[i] if test_setup_code and i < len(test_setup_code) else ""
            items.append((code, tests, setup))

    # Execute non-None items in parallel
    exec_items = [item for item in items if item is not None]
    exec_indices = [i for i, item in enumerate(items) if item is not None]

    if exec_items:
        exec_results = pool.execute_batch(exec_items)
    else:
        exec_results = []

    # Map results back to original indices
    result_map: dict[int, ExecutionResult] = dict(zip(exec_indices, exec_results))

    # Compute rewards
    rewards = []
    for i in range(len(completions)):
        if i not in result_map:
            # Code extraction failed — couldn't even parse a code block
            rewards.append(-0.5)
        else:
            rewards.append(_score_result(result_map[i]))

    logger.debug(
        f"Batch rewards ({_reward_mode}): mean={sum(rewards)/len(rewards):.3f}, "
        f"pass_rate>0: {sum(1 for r in rewards if r > 0)}/{len(rewards)}"
    )
    return rewards


def format_reward(
    prompts: list,
    completions: list,
    **kwargs: Any,
) -> list[float]:
    """
    Secondary reward: bonus for proper output formatting.

    Returns 1.0 if the completion contains a properly formatted
    ```python ... ``` block, 0.0 otherwise.

    This is weighted at 0.1 (via reward_weights in config), so it adds
    a small nudge toward structured output without dominating the
    correctness signal.
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else ""
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    text = msg["content"]
                    break

        if re.search(r"```python\s*\n(.*?)```", text, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
