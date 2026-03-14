"""
Evaluation script for measuring the trained model's coding ability.

Uses vLLM for fast batched inference and Docker sandbox pool for parallel
test execution. Supports evaluating on MBPP and RLVR benchmarks.

Usage:
    # Evaluate baseline on MBPP
    python -m src.evaluate --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark mbpp

    # Evaluate trained LoRA adapter on RLVR
    python -m src.evaluate --model ./outputs/final --benchmark rlvr --is-adapter

    # Evaluate on both benchmarks
    python -m src.evaluate --model ./outputs/final --benchmark both --is-adapter

    # Evaluate baseline (no adapter merge needed)
    python -m src.evaluate --model Qwen/Qwen2.5-Coder-7B-Instruct --benchmark both
"""

import argparse
import json
import logging
import os
from pathlib import Path

# vLLM v1 uses fork by default, which breaks if CUDA is initialized before fork.
# Force spawn to avoid "Cannot re-initialize CUDA in forked subprocess" errors.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from datasets import load_dataset

from src.dataset import SYSTEM_PROMPT
from src.sandbox import SandboxPool
from src.utils import extract_code_from_completion, set_seed, setup_logging

logger = logging.getLogger(__name__)


def merge_adapter(model_path: str, base_model: str, output_dir: str) -> str:
    """
    Merge a LoRA adapter into the base model and save the merged model.

    Returns the path to the merged model (for vLLM to load).
    """
    import torch
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

    # Free memory before vLLM loads
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_path


def build_prompts_mbpp():
    """Load MBPP test set and build prompts + test metadata."""
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    logger.info(f"Loaded {len(ds)} MBPP test problems")

    prompts = []
    metadata = []
    for row in ds:
        tests = row["test_list"]
        tests_str = "\n".join(tests)
        user_content = f"{row['text']}\n\nYour code must satisfy these test cases:\n{tests_str}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompts.append(messages)
        metadata.append({
            "task_id": row["task_id"],
            "tests": tests,
            "setup_code": row.get("test_setup_code", ""),
        })

    return prompts, metadata


def build_prompts_rlvr():
    """Load RLVR func-type problems and build prompts + test metadata."""
    import json as json_mod

    # RLVR only has a train split — use it for evaluation
    ds = load_dataset("NousResearch/RLVR_Coding_Problems", split="train")
    logger.info(f"Loaded {len(ds)} total RLVR problems")

    ds = ds.filter(lambda row: row["problem_type"] == "func")
    logger.info(f"Filtered to {len(ds)} function-type problems")

    prompts = []
    metadata = []
    for i, row in enumerate(ds):
        fn_name = row["fn_name"]
        tests_data = json_mod.loads(row["tests"])
        inputs = tests_data.get("input", [])
        outputs = tests_data.get("output", [])
        test_asserts = []
        for inp, out in zip(inputs, outputs):
            assert_str = f"assert {fn_name}({', '.join(repr(a) for a in inp)}) == {repr(out)}"
            if len(assert_str) <= 500:
                test_asserts.append(assert_str)

        if not test_asserts:
            continue

        tests_preview = "\n".join(test_asserts[:3])
        user_content = (
            f"{row['problem']}\n\n"
            f"Your function must be named `{fn_name}`.\n\n"
            f"Your code must satisfy these test cases:\n{tests_preview}"
        )

        # Skip oversized prompts
        if len(user_content) > 4000:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompts.append(messages)
        metadata.append({
            "task_id": f"rlvr_{i}",
            "tests": test_asserts,
            "setup_code": "",
        })

    logger.info(f"Built {len(prompts)} RLVR evaluation prompts")
    return prompts, metadata


def generate_batch_vllm(model_path: str, prompts: list[list[dict]], max_tokens: int = 1024) -> list[str]:
    """
    Generate solutions for all prompts using vLLM batched inference.

    Args:
        model_path: Path to model or HF model ID
        prompts: List of conversation messages (OpenAI format)
        max_tokens: Max new tokens per completion

    Returns:
        List of generated text strings
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # vLLM needs absolute paths for local models (relative paths get treated as HF repo IDs)
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)

    logger.info(f"Loading model into vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
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
        idx = output.request_id  # vLLM assigns sequential int IDs starting from 0
        results[int(idx)] = output.outputs[0].text

    # Free GPU memory
    del llm
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Generation complete")
    return results


def evaluate_solutions(completions: list[str], metadata: list[dict], pool: SandboxPool) -> dict:
    """
    Execute generated solutions against test cases using the sandbox pool.

    Args:
        completions: Raw model outputs
        metadata: List of dicts with task_id, tests, setup_code
        pool: SandboxPool for parallel execution

    Returns:
        Dict with pass_at_1, passed, total, and per-problem results
    """
    # Extract code from completions
    codes = []
    for comp in completions:
        code = extract_code_from_completion(comp)
        codes.append(code)

    # Build batch items for parallel execution
    batch_items = []
    no_code_indices = set()
    for i, (code, meta) in enumerate(zip(codes, metadata)):
        if code is None:
            no_code_indices.add(i)
            # Placeholder — won't actually execute meaningfully
            batch_items.append(("", meta["tests"], meta["setup_code"]))
        else:
            batch_items.append((code, meta["tests"], meta["setup_code"]))

    logger.info(f"Executing {len(batch_items)} solutions in sandbox pool "
                f"({len(no_code_indices)} had no extractable code)...")
    exec_results = pool.execute_batch(batch_items)

    # Compile results
    passed = 0
    total = 0
    results = []
    for i, (exec_result, meta) in enumerate(zip(exec_results, metadata)):
        if i in no_code_indices:
            results.append({
                "task_id": meta["task_id"],
                "passed": False,
                "error": "no code extracted",
            })
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
            "errors": exec_result.errors[:3],  # truncate for readability
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate coding model on benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model ID")
    parser.add_argument("--benchmark", type=str, default="mbpp", choices=["mbpp", "rlvr", "both"])
    parser.add_argument("--is-adapter", action="store_true", help="Model path is a LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--sandbox-pool-size", type=int, default=8)
    parser.add_argument("--sandbox-timeout", type=int, default=10)
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    # If adapter, merge into base model first
    model_path = args.model
    if args.is_adapter:
        model_path = merge_adapter(args.model, args.base_model, os.path.dirname(args.model) or ".")

    # Set up sandbox pool for test execution
    pool = SandboxPool(
        pool_size=args.sandbox_pool_size,
        timeout=args.sandbox_timeout,
        memory_limit="256m",
    )

    all_results = {}

    if args.benchmark in ("mbpp", "both"):
        logger.info("=== Evaluating on MBPP ===")
        prompts, metadata = build_prompts_mbpp()
        completions = generate_batch_vllm(model_path, prompts)
        mbpp_results = evaluate_solutions(completions, metadata, pool)
        mbpp_results["benchmark"] = "mbpp"
        all_results["mbpp"] = mbpp_results

    if args.benchmark in ("rlvr", "both"):
        logger.info("=== Evaluating on RLVR ===")
        prompts, metadata = build_prompts_rlvr()
        completions = generate_batch_vllm(model_path, prompts)
        rlvr_results = evaluate_solutions(completions, metadata, pool)
        rlvr_results["benchmark"] = "rlvr"
        all_results["rlvr"] = rlvr_results

    # Save results
    output_path = args.output or f"outputs/eval_{args.benchmark}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 50)
    for bench_name, bench_results in all_results.items():
        print(f"{bench_name.upper():>10} pass@1: {bench_results['pass_at_1']:.1%} "
              f"({bench_results['passed']}/{bench_results['total']})")
    print("=" * 50)


if __name__ == "__main__":
    main()
