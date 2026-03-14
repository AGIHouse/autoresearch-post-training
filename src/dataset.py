"""
Dataset loading and formatting for GRPO training.

Supports multiple datasets:

1. **MBPP** (google-research-datasets/mbpp):
   374 train problems, 3+ tests each. Good baseline, well-studied.
   Small but clean — ideal for debugging and quick experiments.

2. **RLVR Coding Problems** (NousResearch/RLVR_Coding_Problems):
   Larger dataset with verified test cases, designed specifically for
   RL with verifiable rewards. Better test coverage than MBPP.

3. **OpenCoder** (OpenCoder-LLM/opc-sft-stage2):
   Instruction-code-test triples validated by Python compiler.
   Used by Modal's GRPO example and TRL's official coding example.

All datasets are formatted into TRL's GRPOTrainer expected structure:
    Required column:
        - "prompt": list[dict] — conversation in OpenAI format
    Extra columns (forwarded to reward function as kwargs):
        - "test_list": list[str] — assertion strings
        - "test_setup_code": str — setup code run before tests

The system prompt instructs the model to output code inside ```python```
fences for reliable extraction.
"""

import logging

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert Python programmer. Given a programming task, write a \
correct Python solution.

Rules:
- Output ONLY the Python code inside ```python ... ``` markers
- Do not include test cases, examples, or explanations
- Write clean, correct, and complete code
- Include all necessary imports inside your code block"""


def load_dataset_for_grpo(dataset_name: str = "mbpp", split: str = "train") -> Dataset:
    """
    Load a coding dataset and format it for GRPOTrainer.

    Args:
        dataset_name: One of "mbpp", "rlvr", "opencoder"
        split: "train" or "test"

    Returns:
        HuggingFace Dataset with columns: prompt, test_list, test_setup_code
    """
    loaders = {
        "mbpp": _load_mbpp,
        "rlvr": _load_rlvr,
        "opencoder": _load_opencoder,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(loaders.keys())}")

    return loaders[dataset_name](split)


def _load_mbpp(split: str = "train") -> Dataset:
    """
    Load MBPP (Mostly Basic Python Problems).

    374 train problems with 3 test assertions + challenge tests each.
    Simple entry-level problems — good starting difficulty for 7B models.
    """
    logger.info(f"Loading MBPP ({split} split)...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    logger.info(f"Loaded {len(ds)} problems")

    def format_row(row):
        # Include test cases in prompt so model knows expected function name/signature
        task_desc = row["text"]
        tests_str = "\n".join(row["test_list"])
        user_content = f"{task_desc}\n\nYour code must satisfy these test cases:\n{tests_str}"

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Combine standard tests with challenge tests for richer reward signal
        all_tests = row["test_list"]
        if row.get("challenge_test_list"):
            all_tests = all_tests + row["challenge_test_list"]

        return {
            "prompt": prompt,
            "test_list": all_tests,
            "test_setup_code": row.get("test_setup_code", ""),
        }

    formatted = ds.map(format_row, remove_columns=ds.column_names)
    logger.info(f"Formatted {len(formatted)} MBPP problems for GRPO")
    return formatted


def _load_rlvr(split: str = "train") -> Dataset:
    """
    Load NousResearch RLVR Coding Problems (func-type only).

    Filters to function-based problems (2,580 of 24,287) and converts
    JSON input/output test pairs into assert statements compatible with
    our sandbox. Harder than MBPP — competitive programming difficulty.

    Reference: https://huggingface.co/datasets/NousResearch/RLVR_Coding_Problems
    """
    import json

    logger.info(f"Loading RLVR Coding Problems ({split} split)...")
    ds = load_dataset("NousResearch/RLVR_Coding_Problems", split=split)
    logger.info(f"Loaded {len(ds)} total problems")

    # Filter to func-type problems (assert-compatible)
    ds = ds.filter(lambda row: row["problem_type"] == "func")
    logger.info(f"Filtered to {len(ds)} function-type problems")

    def format_row(row):
        fn_name = row["fn_name"]
        task_desc = row["problem"]

        # Convert JSON tests to assert statements
        tests_data = json.loads(row["tests"])
        inputs = tests_data.get("input", [])
        outputs = tests_data.get("output", [])
        test_asserts = []
        for inp, out in zip(inputs, outputs):
            assert_str = f"assert {fn_name}({', '.join(repr(a) for a in inp)}) == {repr(out)}"
            # Skip individual tests with huge literals (>500 chars)
            if len(assert_str) <= 500:
                test_asserts.append(assert_str)

        # Include function name and sample tests in prompt
        tests_preview = "\n".join(test_asserts[:3])
        user_content = (
            f"{task_desc}\n\n"
            f"Your function must be named `{fn_name}`.\n\n"
            f"Your code must satisfy these test cases:\n{tests_preview}"
        )

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return {
            "prompt": prompt,
            "test_list": test_asserts,
            "test_setup_code": "",
            "_prompt_len": len(user_content),
        }

    formatted = ds.map(format_row, remove_columns=ds.column_names)

    # Filter out problems with prompts too long for the model context
    # (Qwen 32K context - 1024 completion tokens = ~30K for prompt)
    # Rough estimate: 1 token ≈ 4 chars
    max_prompt_chars = 4000  # ~1000 tokens, conservative
    before = len(formatted)
    formatted = formatted.filter(lambda row: row["_prompt_len"] <= max_prompt_chars)
    formatted = formatted.filter(lambda row: len(row["test_list"]) > 0)
    formatted = formatted.remove_columns(["_prompt_len"])
    logger.info(f"Formatted {len(formatted)} RLVR problems for GRPO (filtered {before - len(formatted)} oversized)")
    return formatted


def _load_opencoder(split: str = "train") -> Dataset:
    """
    Load OpenCoder instruction-code-test triples.

    Used by TRL's official GRPO coding example and Modal's GRPO example.
    Each row has instruction, code, and test cases validated by execution.

    Reference: https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2
    """
    logger.info(f"Loading OpenCoder opc-sft-stage2 ({split} split)...")
    ds = load_dataset(
        "OpenCoder-LLM/opc-sft-stage2",
        "educational_instruct",
        split=split,
    )
    logger.info(f"Loaded {len(ds)} problems")

    def format_row(row):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
        ]

        # OpenCoder stores test cases as a single string of assertions
        test_str = row.get("test_cases", "")
        if isinstance(test_str, str):
            tests = [line.strip() for line in test_str.strip().split("\n") if line.strip()]
        else:
            tests = test_str

        return {
            "prompt": prompt,
            "test_list": tests,
            "test_setup_code": "",
        }

    formatted = ds.map(format_row, remove_columns=ds.column_names)
    logger.info(f"Formatted {len(formatted)} OpenCoder problems for GRPO")
    return formatted


# Convenience aliases
def load_mbpp_for_grpo(split: str = "train") -> Dataset:
    """Load MBPP and format for GRPOTrainer."""
    return load_dataset_for_grpo("mbpp", split)


def load_mbpp_test() -> Dataset:
    """Load the MBPP test split for validation during training."""
    return load_dataset_for_grpo("mbpp", "test")
