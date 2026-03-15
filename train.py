"""
autoresearch: continuously train Qwen3.5-0.8B-Base on MBPP++ coding problems.

The training loop runs for N iterations (or until a wall-clock budget is hit).
Each iteration has two phases:

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

    After each iteration we evaluate pass@1 on the MBPP++ test set, log the
    result to a JSONL file, and save the best checkpoint.

Usage:
    python train.py                                   # defaults
    python train.py --sft_budget 30 --grpo_budget 90 --iterations 12
    python train.py --total_budget 1200               # stop after 20 min total
    python train.py --output_dir runs/exp1 --no_wandb
    python train.py --rl_only                         # skip SFT phase

Dataset:  evalplus/mbppplus  (378 problems; first 300 train, last 78 eval)
Model:    Qwen/Qwen3.5-0.8B-Base
Hardware: fits in ~6 GB VRAM (bf16 + LoRA)
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from sandbox import ExecResult, run_batch, run_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Base-model prompt: completion-style, primed with ```python
BASE_PROMPT_TEMPLATE = """\
# Python programming task
# {description}
#
# Your solution must pass:
{tests_commented}

```python
"""

_CODE_RE = re.compile(r"```python\s*\n(.*?)(?:```|$)", re.DOTALL)


def build_prompt(row: dict, tokenizer) -> str:
    """
    Build a prompt for the given dataset row.
    - Instruct models (has chat_template): use apply_chat_template with a user message.
    - Base models: use the completion-style BASE_PROMPT_TEMPLATE.
    """
    tests = row.get("test_list") or []
    tests_str = "\n".join(tests[:3])
    description = row["prompt"].strip() if isinstance(row["prompt"], str) else row["prompt"]

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content":
            f"Write a Python function to solve this task:\n{description}\n\n"
            f"The function must pass these tests:\n{tests_str}\n\n"
            f"Respond with only a ```python code block."}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        tests_commented = "\n".join(f"# {t}" for t in tests[:3])
        return BASE_PROMPT_TEMPLATE.format(
            description=description, tests_commented=tests_commented
        )


# ── Time-budget callback ────────────────────────────────────────────────────────

class TimeBudgetCallback(TrainerCallback):
    """Stop training when a wall-clock budget (seconds) is exhausted."""

    def __init__(self, budget_seconds: float):
        self.budget = budget_seconds
        self.start_time: float | None = None
        self.steps_run = 0

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.steps_run += 1
        if self.budget > 0 and time.time() - self.start_time >= self.budget:
            control.should_training_stop = True
        return control


# ── Dataset ────────────────────────────────────────────────────────────────────

def load_mbppplus(split: str, tokenizer) -> Dataset:
    """
    Load MBPP++ and format for training.

    evalplus/mbppplus only has a single "test" split (378 problems).
    We manually split: first 300 → train, last 78 → eval.

    Keeps 'description' and 'tests_str' as raw text so make_sft_dataset
    can build the full instruct conversation without re-parsing.
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
        description = row["prompt"].strip()
        return {
            "prompt": build_prompt(row, tokenizer),
            "description": description,
            "code": (row.get("code") or "").strip(),
            "test_list": tests,
            "tests_str": "\n".join(tests[:3]),
            "test_imports": "\n".join(row.get("test_imports") or []).strip(),
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def make_sft_dataset(data: Dataset, tokenizer) -> Dataset:
    """Build full supervised sequences for SFT.
    Instruct models: user task message + assistant ```python code``` response.
    Base models: raw completion-style text.
    """
    def to_text(row):
        if getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "user", "content":
                    f"Write a Python function to solve this task:\n{row['description']}\n\n"
                    f"The function must pass these tests:\n{row['tests_str']}\n\n"
                    f"Respond with only a ```python code block."},
                {"role": "assistant", "content": f"```python\n{row['code']}\n```"},
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
        else:
            return {"text": row["prompt"] + row["code"] + "\n```\n"}
    return data.map(to_text, remove_columns=data.column_names)


# ── Code extraction ────────────────────────────────────────────────────────────

def extract_code(text: str | list) -> str | None:
    if isinstance(text, list):
        for msg in reversed(text):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                text = msg.get("content", "")
                break
        else:
            return None

    m = _CODE_RE.search(text)
    if m:
        return m.group(1).strip() or None

    stripped = text.strip()
    return stripped if stripped else None


# ── Reward function ────────────────────────────────────────────────────────────

def reward_execution(
    prompts: list,
    completions: list,
    test_list: list[list[str]] | None = None,
    test_imports: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """
    Reward scale:
        -0.5   syntax error or runtime crash        (+ 0.1 format bonus if ```python present)
         0.0   runs but passes 0 tests
         0–1   passed / total  (partial credit)
         1.0   all tests pass
    """
    if test_list is None:
        return [0.0] * len(completions)

    exec_items: list[tuple[str, list[str], str] | None] = []
    for i, comp in enumerate(completions):
        code = extract_code(comp)
        tests = test_list[i] if i < len(test_list) else []
        imports = test_imports[i] if test_imports and i < len(test_imports) else ""
        exec_items.append((code, tests, imports) if code else None)

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
            rewards.append(-0.5 + fmt_bonus)
        else:
            r = result_map[i]
            if r.total == 0:
                rewards.append(fmt_bonus)
            elif r.passed == 0 and r.error:
                rewards.append(-0.5 + fmt_bonus)
            else:
                rewards.append(r.pass_rate + fmt_bonus)
    return rewards


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, test_data: Dataset, n: int = 50) -> float:
    """Estimate pass@1 on MBPP++ test set using greedy decoding."""
    model.eval()
    device = next(model.parameters()).device
    subset = test_data.select(range(min(n, len(test_data))))

    passed = 0
    for row in subset:
        inputs = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tok = out[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(new_tok, skip_special_tokens=True)

        # For instruct models the completion contains the full response incl. ```python
        # For base models prepend the prompt to help extraction
        search_text = completion if getattr(tokenizer, "chat_template", None) else row["prompt"] + completion
        code = extract_code(search_text)
        if code:
            result = run_code(code, row["test_list"], row["test_imports"], timeout=10.0)
            if result.all_passed:
                passed += 1

    model.train()
    return passed / len(subset)


# ── JSONL experiment logger ────────────────────────────────────────────────────

def log_experiment(log_file: str, entry: dict):
    """Append one experiment result to the JSONL log."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"  Logged → {log_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="autoresearch: continuously train Qwen3.5-0.8B on MBPP++",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations",    type=int,   default=20,
                        help="max SFT→GRPO cycles (may be cut short by --total_budget)")
    parser.add_argument("--sft_budget",    type=float, default=30.0,
                        help="wall-clock seconds per SFT phase (0 = use --sft_steps)")
    parser.add_argument("--sft_steps",     type=int,   default=200,
                        help="gradient steps per SFT phase (used when sft_budget=0)")
    parser.add_argument("--sft_lr",        type=float, default=5e-5,
                        help="SFT learning rate (lower = less catastrophic forgetting)")
    parser.add_argument("--grpo_budget",   type=float, default=300.0,
                        help="wall-clock seconds per GRPO phase (0 = use --rl_steps)")
    parser.add_argument("--grpo_lr",       type=float, default=5e-6,
                        help="GRPO learning rate")
    parser.add_argument("--rl_steps",      type=int,   default=200,
                        help="gradient steps per GRPO phase (used when grpo_budget=0)")
    parser.add_argument("--total_budget",  type=float, default=0.0,
                        help="total wall-clock seconds for the whole run (0 = no limit)")
    parser.add_argument("--eval_n",        type=int,   default=30,
                        help="problems to evaluate after each iteration")
    parser.add_argument("--output_dir",    type=str,   default="./outputs")
    parser.add_argument("--log_file",      type=str,   default="./experiment_log.jsonl",
                        help="path to append per-iteration JSONL results")
    parser.add_argument("--no_wandb",      action="store_true")
    parser.add_argument("--rl_only",       action="store_true",
                        help="skip SFT phase")
    parser.add_argument("--model_id",      type=str,   default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID")
    parser.add_argument("--lora_r",        type=int,   default=16)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report_to = "none" if args.no_wandb else "wandb"
    run_start = time.time()

    torch.manual_seed(args.seed)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    log.info(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "eager"
        log.info("flash_attn not found, using eager attention")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data = load_mbppplus("train", tokenizer)
    test_data  = load_mbppplus("test",  tokenizer)
    sft_data   = make_sft_dataset(train_data, tokenizer)
    log.info(f"Train: {len(train_data)} problems  |  Test: {len(test_data)} problems")

    # ── Baseline evaluation ───────────────────────────────────────────────────
    log.info("Running baseline eval...")
    t0 = time.time()
    baseline = evaluate(model, tokenizer, test_data, n=args.eval_n)
    log.info(f"  Baseline pass@1 = {baseline:.1%}  ({time.time()-t0:.0f}s)")
    best_score = baseline

    # Log baseline as iteration 0
    log_experiment(args.log_file, {
        "iteration": 0,
        "pass_at_1": round(baseline * 100, 1),
        "baseline": round(baseline * 100, 1),
        "delta": 0.0,
        "is_best": True,
        "description": "baseline (no fine-tuning)",
        "sft_steps_run": 0,
        "grpo_steps_run": 0,
        "elapsed_s": round(time.time() - run_start, 1),
        "timestamp": datetime.now().isoformat(),
    })

    # ── Training loop ─────────────────────────────────────────────────────────
    for it in range(1, args.iterations + 1):

        # Check total budget
        if args.total_budget > 0 and (time.time() - run_start) >= args.total_budget:
            log.info(f"Total budget ({args.total_budget}s) reached — stopping after {it-1} iterations.")
            break

        log.info("")
        log.info(f"{'='*60}")
        log.info(f"  ITERATION {it} / {args.iterations}  (elapsed {time.time()-run_start:.0f}s)")
        log.info(f"{'='*60}")

        # ── Phase 1: SFT ──────────────────────────────────────────────────────
        sft_cb = TimeBudgetCallback(args.sft_budget)
        if not args.rl_only:
            max_sft = args.sft_steps if args.sft_budget == 0 else 99999
            sft_label = f"{args.sft_budget}s" if args.sft_budget > 0 else f"{args.sft_steps} steps"
            log.info(f"  [Phase 1] SFT  ({sft_label})")
            sft_trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                args=SFTConfig(
                    output_dir=f"{args.output_dir}/iter{it}_sft",
                    max_steps=max_sft,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    learning_rate=args.sft_lr,
                    lr_scheduler_type="cosine",
                    warmup_steps=3,
                    bf16=True,
                    gradient_checkpointing=True,
                    logging_steps=999999,   # suppress per-step output
                    save_steps=99999,
                    report_to=report_to,
                    run_name=f"sft-qwen0.8b-iter{it}",
                    dataset_text_field="text",
                    max_length=512,
                    seed=args.seed,
                ),
                train_dataset=sft_data,
                callbacks=[sft_cb],
            )
            sft_trainer.train()
            log.info(f"    SFT ran {sft_cb.steps_run} steps")
            del sft_trainer
            torch.cuda.empty_cache()

        # ── Phase 2: GRPO ─────────────────────────────────────────────────────
        grpo_cb = TimeBudgetCallback(args.grpo_budget)
        max_grpo = args.rl_steps if args.grpo_budget == 0 else 99999
        grpo_label = f"{args.grpo_budget}s" if args.grpo_budget > 0 else f"{args.rl_steps} steps"
        log.info(f"  [Phase 2] GRPO ({grpo_label})")
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=GRPOConfig(
                output_dir=f"{args.output_dir}/iter{it}_grpo",
                max_steps=max_grpo,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                num_generations=8,
                max_completion_length=512,
                temperature=0.9,
                learning_rate=args.grpo_lr,
                lr_scheduler_type="constant_with_warmup",
                warmup_steps=3,
                beta=0.0,
                loss_type="grpo",
                scale_rewards=True,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=999999,   # suppress per-step output
                save_steps=99999,
                report_to=report_to,
                run_name=f"grpo-qwen0.8b-iter{it}",
                seed=args.seed,
            ),
            reward_funcs=[reward_execution],
            train_dataset=train_data,
            callbacks=[grpo_cb],
        )
        grpo_trainer.train()
        log.info(f"    GRPO ran {grpo_cb.steps_run} steps")
        del grpo_trainer
        torch.cuda.empty_cache()

        # ── Evaluate ──────────────────────────────────────────────────────────
        log.info(f"  Evaluating (pass@1, n={args.eval_n})...")
        t0 = time.time()
        score = evaluate(model, tokenizer, test_data, n=args.eval_n)
        delta = score - baseline
        is_best = score > best_score
        sign = "↑" if delta >= 0 else "↓"
        star = "  ★ NEW BEST" if is_best else ""
        log.info(
            f"  Iteration {it}  pass@1 = {score:.1%}  "
            f"({sign}{abs(delta):.1%} vs baseline){star}  ({time.time()-t0:.0f}s)"
        )

        if is_best:
            best_score = score
            model.save_pretrained(f"{args.output_dir}/best")
            tokenizer.save_pretrained(f"{args.output_dir}/best")
            log.info(f"  Saved best → {args.output_dir}/best")

        # ── Log to JSONL ──────────────────────────────────────────────────────
        model_short = args.model_id.split("/")[-1]
        description = (
            f"{model_short}: SFT {sft_cb.steps_run}s(lr={args.sft_lr:.0e}) "
            f"+ GRPO {grpo_cb.steps_run}s(lr={args.grpo_lr:.0e}, G=8)"
        )
        log_experiment(args.log_file, {
            "iteration": it,
            "pass_at_1": round(score * 100, 1),
            "baseline": round(baseline * 100, 1),
            "delta": round(delta * 100, 1),
            "is_best": is_best,
            "description": description,
            "sft_steps_run": sft_cb.steps_run,
            "grpo_steps_run": grpo_cb.steps_run,
            "elapsed_s": round(time.time() - run_start, 1),
            "timestamp": datetime.now().isoformat(),
        })

    # ── Done ─────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    log.info("")
    log.info(f"Training complete in {total_elapsed/60:.1f} min.")
    log.info(f"  Baseline  pass@1 = {baseline:.1%}")
    log.info(f"  Best      pass@1 = {best_score:.1%}  (+{(best_score-baseline):.1%})")
    log.info(f"  Best checkpoint → {args.output_dir}/best")
    log.info(f"  Experiment log  → {args.log_file}")


if __name__ == "__main__":
    main()
