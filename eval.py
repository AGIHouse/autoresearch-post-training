"""
Standalone evaluation script for a trained Qwen3.5-0.8B checkpoint on MBPP++.

Generates one greedy solution per problem and counts how many pass all tests.
Can evaluate the base model, a LoRA checkpoint, or a merged model.

Usage:
    # Evaluate the base model (baseline)
    python eval.py

    # Evaluate a LoRA checkpoint
    python eval.py --checkpoint outputs/best

    # Evaluate on more problems, with verbose output
    python eval.py --checkpoint outputs/iter3 --n 378 --verbose
"""

import argparse
import logging
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sandbox import run_code

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
_CODE_RE = re.compile(r"```python\s*\n(.*?)(?:```|$)", re.DOTALL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-0.8B on MBPP++ (pass@1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="LoRA adapter directory (None = evaluate base model)")
    parser.add_argument("--n",          type=int, default=100,
                        help="number of MBPP++ test problems to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout",    type=float, default=10.0,
                        help="per-problem execution timeout (seconds)")
    parser.add_argument("--verbose",    action="store_true",
                        help="print each problem's result")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.checkpoint:
        log.info(f"Loading LoRA adapter: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()  # merge for faster inference
        log.info("  LoRA merged into base model weights")

    model.eval()
    device = next(model.parameters()).device

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading evalplus/mbppplus [test]...")
    raw = load_dataset("evalplus/mbppplus", split="test")
    PROMPT_TEMPLATE = (
        "# Python programming task\n"
        "# {description}\n"
        "#\n"
        "# Your solution must pass:\n"
        "{tests_commented}\n\n"
        "```python\n"
    )
    def fmt(row):
        tests = row.get("test_list") or []
        tests_commented = "\n".join(f"# {t}" for t in tests[:3])
        return {
            "prompt": PROMPT_TEMPLATE.format(
                description=row["prompt"].strip(),
                tests_commented=tests_commented,
            ),
            "test_list": tests,
            "test_imports": (row.get("test_imports") or "").strip(),
        }
    test_data = raw.map(fmt, remove_columns=raw.column_names)
    subset = test_data.select(range(min(args.n, len(test_data))))
    log.info(f"Evaluating on {len(subset)} problems...")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    passed = 0
    results = []

    for i, row in enumerate(subset):
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tok = out[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_tok, skip_special_tokens=True)

        m = _CODE_RE.search(row["prompt"] + completion)
        code = m.group(1).strip() if m else None
        if code is None:
            status = "NO_CODE"
            error = "no ```python block found"
        else:
            result = run_code(
                code,
                row["test_list"],
                row["test_imports"],
                timeout=args.timeout,
            )
            if result.all_passed:
                status = "PASS"
                error = ""
                passed += 1
            else:
                status = f"FAIL {result.passed}/{result.total}"
                error = result.error

        results.append((status, error))

        if args.verbose:
            # Show task description (first 80 chars)
            desc = row["prompt"].split("\n")[1].replace("# ", "")[:80]
            log.info(f"  [{i+1:3d}] {status:<15}  {desc}")
            if error and args.verbose:
                log.info(f"         error: {error[:100]}")

        # Progress every 25 problems
        if (i + 1) % 25 == 0:
            log.info(f"  {i+1}/{len(subset)}  running pass@1 = {passed/(i+1):.1%}")

    # ── Results ───────────────────────────────────────────────────────────────
    pass_at_1 = passed / len(subset)
    no_code = sum(1 for s, _ in results if s == "NO_CODE")
    fails   = sum(1 for s, _ in results if s.startswith("FAIL"))

    log.info("")
    log.info(f"{'='*50}")
    log.info(f"  pass@1    = {pass_at_1:.1%}  ({passed}/{len(subset)})")
    log.info(f"  no code   = {no_code/len(subset):.1%}  ({no_code}/{len(subset)})")
    log.info(f"  wrong     = {fails/len(subset):.1%}  ({fails}/{len(subset)})")
    log.info(f"{'='*50}")
    if args.checkpoint:
        log.info(f"  checkpoint: {args.checkpoint}")
    else:
        log.info(f"  model: {MODEL_ID} (base, no fine-tuning)")


if __name__ == "__main__":
    main()
