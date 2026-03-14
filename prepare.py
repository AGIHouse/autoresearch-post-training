"""
Fixed utilities, data prep, sandbox, reward functions, and evaluation for
post-training experiments using MBPP++ (EvalPlus).

This file is READ-ONLY for the agent — do not modify.

Contains:
    - ExecutionResult / SubprocessSandbox / DockerSandbox / SandboxPool
    - Dataset loading (MBPP++ via evalplus) formatted for GRPOTrainer
    - Reward functions (code_execution_reward, format_reward)
    - Evaluation (vLLM batched inference + sandbox execution)
    - Utilities (code extraction, seeding, logging)
"""

import json
import logging
import math
import random
import re
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from evalplus.data import get_mbpp_plus

logger = logging.getLogger(__name__)

# ---- Constants ----
SYSTEM_PROMPT = """\
You are an expert Python programmer. Given a programming task, write a \
correct Python solution.

Rules:
- Output ONLY the Python code inside ```python ... ``` markers
- Do not include test cases, examples, or explanations
- Write clean, correct, and complete code
- Include all necessary imports inside your code block"""


# =============================================================================
# Utilities
# =============================================================================

def extract_code_from_completion(completion: str | list[dict]) -> str | None:
    """
    Extract Python code from an LLM completion.

    Handles:
      1. ```python\\n<code>\\n``` fences
      2. Generic ```\\n<code>\\n``` fences
      3. Raw code (fallback)
    """
    # Handle conversational format from TRL
    if isinstance(completion, list):
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                completion = msg["content"]
                break
        else:
            return None

    if not isinstance(completion, str) or not completion.strip():
        return None

    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: treat entire completion as code
    return completion.strip()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging format for the training system."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Sandbox
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing code against test cases in the sandbox."""
    passed: int
    total: int
    errors: list[str]

    @property
    def pass_rate(self) -> float:
        """Fraction of tests passed (0.0 to 1.0)."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total and self.total > 0


class SubprocessSandbox:
    """
    Lightweight sandbox using subprocess for local development.

    NOT suitable for production training with untrusted model outputs.
    Use DockerSandbox for that.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(self, code: str, tests: list[str], setup_code: str = "") -> ExecutionResult:
        """Execute code + tests in a subprocess."""
        script = self._build_script(code, tests, setup_code)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=True
            ) as f:
                f.write(script)
                f.flush()

                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip().split("\n")[-1])
                return ExecutionResult(
                    passed=data.get("passed", 0),
                    total=data.get("total", len(tests)),
                    errors=data.get("errors", []),
                )
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return ExecutionResult(passed=0, total=len(tests), errors=[error_msg])

        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=0, total=len(tests), errors=["Execution timed out"])
        except json.JSONDecodeError as e:
            return ExecutionResult(passed=0, total=len(tests), errors=[f"Output parse error: {e}"])
        except Exception as e:
            return ExecutionResult(passed=0, total=len(tests), errors=[str(e)])

    def _build_script(self, code: str, tests: list[str], setup_code: str) -> str:
        """Build a Python script that runs code + tests and outputs JSON."""
        return textwrap.dedent(f"""\
            import json
            import sys

            results = {{"passed": 0, "total": {len(tests)}, "errors": []}}
            namespace = {{}}

            try:
                # Setup code
                exec({repr(setup_code)}, namespace)
                # Solution code
                exec({repr(code)}, namespace)
                # Run tests
                tests = {repr(tests)}
                for i, test in enumerate(tests):
                    try:
                        exec(test, namespace)
                        results["passed"] += 1
                    except AssertionError as e:
                        results["errors"].append(f"Test {{i}}: AssertionError: {{e}}")
                    except Exception as e:
                        results["errors"].append(f"Test {{i}}: {{type(e).__name__}}: {{e}}")
            except SyntaxError as e:
                results["errors"].append(f"SyntaxError: {{e}}")
            except Exception as e:
                results["errors"].append(f"{{type(e).__name__}}: {{e}}")

            print(json.dumps(results))
        """)


class DockerSandbox:
    """
    Production sandbox using Docker containers via subprocess.

    Each execution runs in an ephemeral container (--rm) with:
    - No network access (--network none)
    - Limited memory (--memory 256m)
    - Limited PIDs (--pids-limit 64)
    - CPU throttling (--cpus 0.5)
    - Wall-clock timeout via subprocess.run()
    """

    def __init__(self, timeout: int = 10, memory_limit: str = "256m",
                 image: str = "coding-sandbox:latest"):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.image = image

        # Verify docker is available and image exists
        result = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Sandbox image '{self.image}' not found. "
                f"Run 'make sandbox' to build it."
            )

    def execute(self, code: str, tests: list[str], setup_code: str = "") -> ExecutionResult:
        """Execute code + tests in a Docker container via subprocess."""
        payload = json.dumps({
            "code": code,
            "tests": tests,
            "setup_code": setup_code,
            "timeout": self.timeout,
        })

        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", "none",
            "--memory", self.memory_limit,
            "--pids-limit", "64",
            "--cpus", "0.5",
            self.image,
        ]

        try:
            result = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )

            if result.stdout.strip():
                parsed = json.loads(result.stdout.strip().split("\n")[-1])
                return ExecutionResult(
                    passed=parsed.get("passed", 0),
                    total=parsed.get("total", len(tests)),
                    errors=parsed.get("errors", []),
                )
            else:
                error_msg = result.stderr.strip()[:500] if result.stderr else "No output"
                return ExecutionResult(passed=0, total=len(tests), errors=[error_msg])

        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=0, total=len(tests), errors=["Execution timed out"])
        except json.JSONDecodeError as e:
            return ExecutionResult(passed=0, total=len(tests), errors=[f"Output parse error: {e}"])
        except Exception as e:
            logger.debug(f"Sandbox error: {type(e).__name__}: {e}")
            return ExecutionResult(passed=0, total=len(tests), errors=[str(e)])


class SandboxPool:
    """
    Pool for parallel code execution with automatic backend selection.

    Tries Docker first (production). Falls back to subprocess (development).
    Uses ThreadPoolExecutor for parallel execution.
    """

    def __init__(self, pool_size: int = 8, timeout: int = 10, memory_limit: str = "256m",
                 image: str = "coding-sandbox:latest", backend: str = "auto"):
        self.pool_size = pool_size
        self.backend_name = backend

        if backend == "auto":
            try:
                self._backend = DockerSandbox(timeout, memory_limit, image)
                self.backend_name = "docker"
                logger.info("Sandbox backend: Docker")
            except Exception as e:
                logger.warning(f"Docker unavailable ({e}), falling back to subprocess")
                self._backend = SubprocessSandbox(timeout)
                self.backend_name = "subprocess"
        elif backend == "docker":
            self._backend = DockerSandbox(timeout, memory_limit, image)
            logger.info("Sandbox backend: Docker")
        elif backend == "subprocess":
            self._backend = SubprocessSandbox(timeout)
            logger.info("Sandbox backend: subprocess")
        else:
            raise ValueError(f"Unknown sandbox backend: {backend}")

    def execute(self, code: str, tests: list[str], setup_code: str = "") -> ExecutionResult:
        """Execute code + tests using the selected backend."""
        return self._backend.execute(code, tests, setup_code)

    def execute_batch(self, items: list[tuple[str, list[str], str]]) -> list[ExecutionResult]:
        """Execute multiple (code, tests, setup_code) tuples in parallel."""
        results: list[ExecutionResult | None] = [None] * len(items)
        per_future_timeout = self._backend.timeout * 3
        total_timeout = math.ceil(len(items) / self.pool_size) * per_future_timeout + 60

        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            future_to_idx = {
                executor.submit(self.execute, code, tests, setup): idx
                for idx, (code, tests, setup) in enumerate(items)
            }

            for future in as_completed(future_to_idx, timeout=total_timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=per_future_timeout)
                except Exception as e:
                    logger.error(f"Batch execution error at index {idx}: {e}")
                    results[idx] = ExecutionResult(
                        passed=0,
                        total=len(items[idx][1]),
                        errors=[str(e)],
                    )

        # Fill any remaining None results
        for i, r in enumerate(results):
            if r is None:
                results[i] = ExecutionResult(
                    passed=0, total=len(items[i][1]), errors=["Execution timed out"]
                )

        return results  # type: ignore[return-value]


# =============================================================================
# Dataset — MBPP++ (EvalPlus)
# =============================================================================

# MBPP original splits: train=task_id 11-510, test=task_id 511-974
_MBPP_TRAIN_MAX_ID = 510


def _parse_task_id(task_id: str) -> int:
    """Extract numeric ID from 'Mbpp/123' format."""
    return int(task_id.split("/")[1])


def _format_problem(task_id: str, problem: dict) -> dict:
    """Format a single MBPP++ problem into prompt + test_list + setup_code."""
    assertion_str = problem.get("assertion", "")
    test_list = [line.strip() for line in assertion_str.strip().split("\n") if line.strip()]
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]
    return {
        "task_id": task_id,
        "prompt": prompt,
        "test_list": test_list,
        "setup_code": "",
    }


def load_mbpp_plus_for_grpo() -> Dataset:
    """
    Load MBPP++ train split (task_id <= 510) and format for GRPOTrainer.

    Uses the standard MBPP train/test split: train=11-510, test=511-974.

    Returns:
        HuggingFace Dataset with columns: prompt, test_list, test_setup_code
    """
    logger.info("Loading MBPP++ train split from EvalPlus...")
    mbpp_data = get_mbpp_plus()

    prompts = []
    test_lists = []
    test_setup_codes = []

    for task_id, problem in mbpp_data.items():
        if _parse_task_id(task_id) > _MBPP_TRAIN_MAX_ID:
            continue

        formatted = _format_problem(task_id, problem)
        prompts.append(formatted["prompt"])
        test_lists.append(formatted["test_list"])
        test_setup_codes.append(formatted["setup_code"])

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "test_list": test_lists,
        "test_setup_code": test_setup_codes,
    })

    logger.info(f"Formatted {len(dataset)} MBPP++ train problems for GRPO")
    return dataset


def load_mbpp_plus_test() -> tuple[list[list[dict]], list[dict]]:
    """
    Load MBPP++ test split (task_id > 510) for evaluation.

    Returns:
        (prompts, metadata) where prompts are OpenAI-format messages
        and metadata contains task_id, tests, setup_code per problem.
    """
    logger.info("Loading MBPP++ test split for evaluation...")
    mbpp_data = get_mbpp_plus()

    prompts = []
    metadata = []

    for task_id, problem in mbpp_data.items():
        if _parse_task_id(task_id) <= _MBPP_TRAIN_MAX_ID:
            continue

        formatted = _format_problem(task_id, problem)
        prompts.append(formatted["prompt"])
        metadata.append({
            "task_id": task_id,
            "tests": formatted["test_list"],
            "setup_code": formatted["setup_code"],
        })

    logger.info(f"Built {len(prompts)} MBPP++ test evaluation prompts")
    return prompts, metadata


# =============================================================================
# Reward
# =============================================================================

# Module-level sandbox pool, initialized by train.py before training starts
_sandbox_pool: SandboxPool | None = None
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
    """Compute reward from an execution result based on the current mode."""
    if result.total == 0:
        return 0.0

    if result.passed == 0 and result.errors:
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

    Reward scale (partial mode):
        -0.5  : code doesn't parse or raises an exception
         0.0  : code runs but passes 0 tests
         0.0-1.0 : partial credit (passed / total)
         1.0  : all tests pass

    Reward scale (binary mode):
        -0.5  : code doesn't parse or crashes
         0.0  : code runs but doesn't pass all tests
         1.0  : all tests pass
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
            items.append(None)
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

    result_map: dict[int, ExecutionResult] = dict(zip(exec_indices, exec_results))

    # Compute rewards
    rewards = []
    for i in range(len(completions)):
        if i not in result_map:
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
    Secondary reward: bonus for proper ```python ... ``` formatting.
    Returns 1.0 if properly formatted, 0.0 otherwise.
    Weighted at 0.1 via reward_weights in config.
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


# =============================================================================
# Evaluation
# =============================================================================

def generate_batch_vllm(model_path: str, prompts: list[list[dict]],
                        max_tokens: int = 1024) -> list[str]:
    """
    Generate solutions for all prompts using vLLM batched inference.

    Args:
        model_path: Path to model or HF model ID
        prompts: List of conversation messages (OpenAI format)
        max_tokens: Max new tokens per completion

    Returns:
        List of generated text strings
    """
    import gc
    import os
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # vLLM needs absolute paths for local models
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)

    logger.info(f"Loading model into vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Format prompts using chat template
    formatted = []
    for messages in prompts:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)

    sampling_params = SamplingParams(
        temperature=0,  # greedy for deterministic eval
        max_tokens=max_tokens,
    )

    logger.info(f"Generating {len(formatted)} completions with vLLM...")
    outputs = llm.generate(formatted, sampling_params)

    # Extract generated text, sorted by request order
    results = [""] * len(formatted)
    for output in outputs:
        idx = output.request_id
        results[int(idx)] = output.outputs[0].text

    # Free GPU memory — must shut down vLLM engine to release CUDA contexts
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Generation complete")
    return results


def merge_adapter(model_path: str, base_model: str, output_dir: str) -> str:
    """Merge a LoRA adapter into the base model and save. Returns path to merged model."""
    import gc
    import os
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_path = os.path.abspath(os.path.join(output_dir, "merged_model"))
    if os.path.exists(merged_path):
        logger.info(f"Merged model already exists at {merged_path}, skipping merge")
        return merged_path

    logger.info(f"Merging adapter {model_path} into {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(merged_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_path


def evaluate_solutions(completions: list[str], metadata: list[dict],
                       pool: SandboxPool) -> dict:
    """
    Execute generated solutions against test cases using the sandbox pool.

    Returns:
        Dict with pass_at_1, passed, total, and per-problem results
    """
    codes = [extract_code_from_completion(comp) for comp in completions]

    batch_items = []
    no_code_indices = set()
    for i, (code, meta) in enumerate(zip(codes, metadata)):
        if code is None:
            no_code_indices.add(i)
            batch_items.append(("", meta["tests"], meta["setup_code"]))
        else:
            batch_items.append((code, meta["tests"], meta["setup_code"]))

    logger.info(f"Executing {len(batch_items)} solutions in sandbox pool "
                f"({len(no_code_indices)} had no extractable code)...")
    exec_results = pool.execute_batch(batch_items)

    passed = 0
    total = 0
    results = []
    for i, (exec_result, meta) in enumerate(zip(exec_results, metadata)):
        if i in no_code_indices:
            results.append({"task_id": meta["task_id"], "passed": False, "error": "no code extracted"})
            total += 1
            continue

        problem_passed = exec_result.all_passed
        if problem_passed:
            passed += 1
        total += 1

        results.append({
            "task_id": meta["task_id"],
            "passed": problem_passed,
            "tests_passed": exec_result.passed,
            "tests_total": exec_result.total,
            "errors": exec_result.errors[:3],
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  Scored {i+1}/{len(metadata)} | Running pass@1: {passed/total:.1%}")

    pass_at_1 = passed / total if total > 0 else 0.0
    logger.info(f"pass@1: {pass_at_1:.1%} ({passed}/{total})")

    return {
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "results": results,
    }
