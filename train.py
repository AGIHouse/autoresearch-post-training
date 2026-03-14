"""
autoresearch: continuously train Qwen3.5-0.8B-Base on MBPP++ coding problems.

The training loop runs for N iterations. Each iteration has two phases:

    Phase 1 — SFT (Supervised Fine-Tuning)
        Train on MBPP++ reference solutions with teacher forcing.
        This seeds the model with a correct output format and a reasonable
        starting policy before RL kicks in.

    Phase 2 — GRPO (Group Relative Policy Optimization)
        Sample 8 candidate solutions per problem, execute them against the
        test suite, and update the model to produce more solutions like the
        high-scoring ones. No critic network needed — advantages are computed
        group-relative within each prompt's batch:

            A_i = (r_i - mean(group)) / std(group)

        Why GRPO over PPO?  No value network → half the GPU memory.
        Why beta=0.0?       No KL penalty. Verifiable rewards (code execution)
                            make the KL term unnecessary, per DeepSWE / Open-
                            Reasoner-Zero findings.
        Why LoRA?           Only ~50M trainable params out of 800M → fast
                            updates, small optimizer state, easy to reset.

    After each iteration we run a quick pass@1 eval on the MBPP++ test set
    and save the best checkpoint.

Usage:
    python train.py                           # defaults: 3 iterations
    python train.py --iterations 5 --sft_steps 300 --rl_steps 300
    python train.py --output_dir runs/exp1 --no_wandb
    python train.py --rl_only                 # skip SFT (if already fine-tuned)

Dataset:  evalplus/mbppplus  (MBPP with more test cases; 374 train, 500 test)
Model:    Qwen/Qwen3.5-0.8B-Base
Hardware: fits in ~6 GB VRAM (bf16 + LoRA)
"""

import argparse
import logging
import os
import re

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from sandbox import ExecResult, run_batch, run_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"

# The prompt primes the model to write a ```python block.
# For a base model we use completion-style formatting (no chat template).
PROMPT_TEMPLATE = """\
# Python programming task
# {description}
#
# Your solution must pass:
{tests_commented}

```python
"""

# Matches a ```python ... ``` block (used in extraction)
_CODE_RE = re.compile(r"```python\s*\n(.*?)(?:```|$)", re.DOTALL)


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_mbppplus(split: str = "train") -> Dataset:
    """
    Load MBPP++ and format for training.

    MBPP++ (evalplus/mbppplus) extends the original MBPP with:
      - More test cases per problem (avg ~7 vs MBPP's 3)
      - Challenge tests covering edge cases
    This gives a richer reward signal for RL.

    NOTE: evalplus/mbppplus only has a single "test" split (378 problems).
    We manually split: first 300 → train, last 78 → eval.

    Returned columns:
        prompt        str        — problem prompt ending with ```python\n
        code          str        — reference solution (for SFT)
        test_list     list[str]  — assert strings (for RL reward)
        test_imports  str        — import statements needed by tests
    """
    log.info(f"Loading evalplus/mbppplus [{split}]...")
    full = load_dataset("evalplus/mbppplus", split="test")
    if split == "train":
        ds = full.select(range(300))
    else:
        ds = full.select(range(300, len(full)))
    log.info(f"  {len(ds)} problems")

    def fmt(row):
        tests = row.get("test_list") or []
        # Show first 3 tests in the prompt so the model knows function signatures
        tests_commented = "\n".join(f"# {t}" for t in tests[:3])
        prompt = PROMPT_TEMPLATE.format(
            description=row["prompt"].strip(),
            tests_commented=tests_commented,
        )
        return {
            "prompt": prompt,
            "code": (row.get("code") or "").strip(),
            "test_list": tests,
            "test_imports": "\n".join(row.get("test_imports") or []).strip(),
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def make_sft_dataset(data: Dataset) -> Dataset:
    """
    Convert MBPP++ rows into full text sequences for SFT.

    Each sequence is:
        [prompt ending with ```python\n] + [reference code] + [\n```\n]

    SFT trains the model to predict this entire sequence (standard causal LM).
    The model learns both the output format (python fences) and correct solutions.
    """
    def to_text(row):
        return {"text": row["prompt"] + row["code"] + "\n```\n"}

    return data.map(to_text, remove_columns=data.column_names)


# ── Code extraction ────────────────────────────────────────────────────────────

def extract_code(text: str | list) -> str | None:
    """
    Pull Python code out of a completion.

    Handles both string completions and TRL's chat-dict format.
    Returns None if no code block found.
    """
    if isinstance(text, list):
        # TRL sometimes wraps completions as message dicts
        for msg in reversed(text):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                text = msg.get("content", "")
                break
        else:
            return None

    m = _CODE_RE.search(text)
    if m:
        return m.group(1).strip() or None

    # Fallback: if prompt primed with ```python\n, the completion IS the code
    # (model may not include closing ```)
    stripped = text.strip()
    return stripped if stripped else None


# ── Reward functions ────────────────────────────────────────────────────────────
# TRL's GRPOTrainer calls these with signature:
#   fn(prompts, completions, **extra_columns) -> list[float]
# Extra dataset columns (test_list, test_imports) are passed as kwargs.

def reward_execution(
    prompts: list,
    completions: list,
    test_list: list[list[str]] | None = None,
    test_imports: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """
    Execute each completion's code against the problem's test cases.

    Reward scale (partial credit mode):
        -0.5   code can't be parsed, or crashes before running any test
         0.0   runs but passes 0 tests
         0–1   passes / total (partial credit — useful when problems have few tests)
         1.0   all tests pass

    Why partial credit?  MBPP++ has ~7 tests/problem. Giving 3/7 vs 0/7 credit
    provides a denser gradient signal than binary pass/fail, which helps the
    0.8B model bootstrap from near-zero performance.
    """
    if test_list is None:
        return [0.0] * len(completions)

    # Build (code, tests, setup_imports) for each completion
    exec_items: list[tuple[str, list[str], str] | None] = []
    for i, comp in enumerate(completions):
        code = extract_code(comp)
        tests = test_list[i] if i < len(test_list) else []
        imports = test_imports[i] if test_imports and i < len(test_imports) else ""
        exec_items.append((code, tests, imports) if code else None)

    # Execute non-None items in parallel; preserve order
    valid_items = [(it, i) for i, it in enumerate(exec_items) if it is not None]
    if valid_items:
        results: list[ExecResult] = run_batch([it for it, _ in valid_items])
        result_map = {idx: res for (_, idx), res in zip(valid_items, results)}
    else:
        result_map = {}

    rewards = []
    for i, comp in enumerate(completions):
        fmt_bonus = 0.1 if _CODE_RE.search(comp if isinstance(comp, str) else "") else 0.0
        if i not in result_map:
            rewards.append(-0.5 + fmt_bonus)  # no code extracted
        else:
            r = result_map[i]
            if r.total == 0:
                rewards.append(fmt_bonus)
            elif r.passed == 0 and r.error:
                rewards.append(-0.5 + fmt_bonus)  # crashed
            else:
                rewards.append(r.pass_rate + fmt_bonus)
    return rewards


def reward_format(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    Small bonus (+0.1 weight) for wrapping code in a ```python block.

    This nudges the model toward structured output without overpowering the
    correctness signal. In practice it mainly matters in the first few steps
    before the model learns the format from SFT.
    """
    return [1.0 if _CODE_RE.search(c if isinstance(c, str) else "") else 0.0
            for c in completions]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    test_data: Dataset,
    n: int = 100,
    max_new_tokens: int = 512,
    exec_timeout: float = 10.0,
) -> float:
    """
    Estimate pass@1 on the MBPP++ test set using greedy decoding.

    We generate one solution per problem (greedy, deterministic) and check
    if it passes all test cases. Returns fraction of problems solved.
    """
    model.eval()
    device = next(model.parameters()).device
    subset = test_data.select(range(min(n, len(test_data))))

    passed = 0
    for row in subset:
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tok = out[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_tok, skip_special_tokens=True)

        # The prompt ends with ```python\n so prepend it to help extraction
        code = extract_code(row["prompt"] + completion)
        if code:
            result = run_code(
                code,
                row["test_list"],
                row["test_imports"],
                timeout=exec_timeout,
            )
            if result.all_passed:
                passed += 1

    model.train()
    return passed / len(subset)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="autoresearch: continuously train Qwen3.5-0.8B on MBPP++",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations",  type=int,   default=3,
                        help="number of SFT→GRPO cycles")
    parser.add_argument("--sft_steps",   type=int,   default=200,
                        help="gradient steps per SFT phase")
    parser.add_argument("--rl_steps",    type=int,   default=200,
                        help="gradient steps per GRPO phase")
    parser.add_argument("--eval_n",      type=int,   default=100,
                        help="problems to evaluate after each iteration")
    parser.add_argument("--output_dir",  type=str,   default="./outputs")
    parser.add_argument("--no_wandb",    action="store_true",
                        help="disable Weights & Biases logging")
    parser.add_argument("--rl_only",     action="store_true",
                        help="skip SFT phase (use when resuming from a checkpoint)")
    parser.add_argument("--lora_r",      type=int,   default=16,
                        help="LoRA rank — higher = more capacity, more memory")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report_to = "none" if args.no_wandb else "wandb"

    torch.manual_seed(args.seed)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    log.info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for decoder-only generation

    # Try flash attention; silently fall back if not available
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "eager"
        log.info("flash_attn not found, using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # Apply LoRA ONCE before the loop.
    # Both SFT and GRPO will update the same adapter weights each iteration —
    # no need to re-apply LoRA, and it avoids stacking adapters by accident.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,   # standard heuristic: alpha = 2 * r
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data = load_mbppplus("train")
    test_data  = load_mbppplus("test")
    sft_data   = make_sft_dataset(train_data)  # full text for teacher-forcing
    # train_data already has prompt/test_list/test_imports for GRPO

    log.info(f"Train: {len(train_data)} problems  |  Test: {len(test_data)} problems")

    # ── Baseline evaluation ───────────────────────────────────────────────────
    log.info("Running baseline eval (greedy, pass@1)...")
    baseline = evaluate(model, tokenizer, test_data, n=args.eval_n)
    log.info(f"  Baseline pass@1 = {baseline:.1%}")
    best_score = baseline

    # ── Continuous training loop ───────────────────────────────────────────────
    #
    # The loop runs `--iterations` times. Each iteration:
    #   1. SFT   — teaches format + good starting policy
    #   2. GRPO  — explores via sampling, rewards correct solutions
    #   3. Eval  — measure progress, save best checkpoint
    #
    # Starting from iteration 2, the model already knows the format well from
    # the previous SFT phase, so GRPO gets a head start. The cycle continues
    # to improve the policy as long as GRPO finds new solutions to learn from.

    for it in range(1, args.iterations + 1):
        log.info("")
        log.info(f"{'='*60}")
        log.info(f"  ITERATION {it} / {args.iterations}")
        log.info(f"{'='*60}")

        # ── Phase 1: SFT ──────────────────────────────────────────────────────
        if not args.rl_only:
            log.info(f"  [Phase 1] SFT  ({args.sft_steps} steps)")
            sft_trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                args=SFTConfig(
                    output_dir=f"{args.output_dir}/iter{it}_sft",
                    max_steps=args.sft_steps,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    learning_rate=2e-4,
                    lr_scheduler_type="cosine",
                    warmup_steps=5,
                    bf16=True,
                    gradient_checkpointing=True,
                    logging_steps=5,
                    save_steps=args.sft_steps,  # save once at the end
                    report_to=report_to,
                    run_name=f"sft-qwen0.8b-iter{it}",
                    dataset_text_field="text",
                    max_length=512,
                    seed=args.seed,
                ),
                train_dataset=sft_data,
            )
            sft_trainer.train()
            del sft_trainer
            torch.cuda.empty_cache()

        # ── Phase 2: GRPO ─────────────────────────────────────────────────────
        #
        # For each training step GRPO:
        #   1. Samples num_generations=8 completions per prompt
        #   2. Scores each with reward_execution + reward_format
        #   3. Computes group-relative advantages within each prompt's 8 completions
        #   4. Updates LoRA params to push toward high-advantage completions
        #
        # With per_device_train_batch_size=4 and grad_accum=2:
        #   8 prompts/step × 8 completions = 64 code executions/step
        #   At ~0.3s/execution (parallel): ~3s overhead/step
        log.info(f"  [Phase 2] GRPO ({args.rl_steps} steps)")
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=GRPOConfig(
                output_dir=f"{args.output_dir}/iter{it}_grpo",
                max_steps=args.rl_steps,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                num_generations=8,           # G: completions sampled per prompt
                max_completion_length=512,
                temperature=0.9,             # enough entropy for exploration
                learning_rate=5e-6,          # conservative — RL is noisy
                lr_scheduler_type="constant_with_warmup",
                warmup_steps=5,
                beta=0.0,                    # no KL penalty (beta=0 is current best practice)
                loss_type="grpo",            # standard GRPO (group relative advantage)
                scale_rewards=True,          # normalize reward std within each group
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                save_steps=args.rl_steps,
                report_to=report_to,
                run_name=f"grpo-qwen0.8b-iter{it}",
                seed=args.seed,
            ),
            reward_funcs=[reward_execution],
            train_dataset=train_data,
        )
        grpo_trainer.train()
        del grpo_trainer
        torch.cuda.empty_cache()

        # ── Evaluate ──────────────────────────────────────────────────────────
        log.info(f"  Evaluating (greedy, pass@1, n={args.eval_n})...")
        score = evaluate(model, tokenizer, test_data, n=args.eval_n)
        delta = score - baseline
        log.info(
            f"  Iteration {it}  pass@1 = {score:.1%}  "
            f"({'↑' if delta >= 0 else '↓'}{abs(delta):.1%} vs baseline)"
        )

        # Save every iteration checkpoint
        ckpt = f"{args.output_dir}/iter{it}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        log.info(f"  Saved → {ckpt}")

        if score > best_score:
            best_score = score
            model.save_pretrained(f"{args.output_dir}/best")
            tokenizer.save_pretrained(f"{args.output_dir}/best")
            log.info(f"  ★ New best!  Saved → {args.output_dir}/best")

    # ── Done ─────────────────────────────────────────────────────────────────
    log.info("")
    log.info(f"Training complete.")
    log.info(f"  Baseline  pass@1 = {baseline:.1%}")
    log.info(f"  Final     pass@1 = {score:.1%}")
    log.info(f"  Best      pass@1 = {best_score:.1%}")
    log.info(f"  Best checkpoint → {args.output_dir}/best")


if __name__ == "__main__":
    main()
