"""
Main training script for RL-training a coding agent with GRPO.

This is the entry point that orchestrates everything:
    1. Load config (from YAML or defaults)
    2. Initialize the Docker sandbox pool
    3. Load and format the MBPP dataset
    4. Configure LoRA (parameter-efficient fine-tuning)
    5. Configure GRPOTrainer with custom reward functions
    6. Train
    7. Save the final LoRA adapter

How GRPO training works (high-level):
    For each training step:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Sample a batch of prompts from the dataset                  │
    │ 2. For each prompt, generate G=8 completions using vLLM        │
    │ 3. Run each completion through the reward functions:            │
    │    - Execute code in Docker sandbox → correctness score        │
    │    - Check formatting → format bonus                           │
    │ 4. Compute group-relative advantages:                          │
    │    A_i = (r_i - mean(r_group)) / std(r_group)                  │
    │    This tells us which completions are better/worse than        │
    │    average FOR THAT SPECIFIC PROMPT                             │
    │ 5. Update the policy using clipped surrogate objective:         │
    │    L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]          │
    │    where ratio = π_new(a|s) / π_old(a|s)                      │
    └─────────────────────────────────────────────────────────────────┘
    The model learns to produce more completions like the high-reward
    ones and fewer like the low-reward ones, without needing a
    separate critic/value network.

Usage:
    # With defaults
    python src/train.py

    # With custom config
    python src/train.py --config configs/default.yaml

    # With accelerate (for multi-GPU or DeepSpeed)
    accelerate launch --num_processes 1 src/train.py
"""

import argparse
import logging
import sys

from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

from src.callbacks import (
    SampleLoggerCallback,
    RewardStatsCallback,
    EarlyStoppingCallback,
    CompletionLengthMonitor,
)
from src.config import TrainingConfig
from src.dataset import load_dataset_for_grpo
from src.reward import code_execution_reward, format_reward, set_sandbox_pool, set_reward_mode
from src.sandbox import SandboxPool
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def build_grpo_config(config: TrainingConfig) -> GRPOConfig:
    """Convert our TrainingConfig into TRL's GRPOConfig."""
    return GRPOConfig(
        output_dir=config.output_dir,
        # GRPO-specific
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        beta=config.beta,
        loss_type=config.loss_type,
        scale_rewards=config.scale_rewards,
        num_iterations=config.num_iterations,
        reward_weights=config.reward_weights,
        # Training
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        # Precision & memory
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        # vLLM
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_sleep_enabled=config.vllm_sleep_enabled,
        # Logging & saving
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to="wandb",
        run_name=f"grpo-{config.model_name.split('/')[-1]}",
        seed=config.seed,
    )


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    """Build LoRA configuration for parameter-efficient fine-tuning."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def main():
    parser = argparse.ArgumentParser(description="RL-train a coding agent with GRPO")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    setup_logging()

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig()
        logger.info("Using default config")

    set_seed(config.seed)

    # Save config for reproducibility
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    config.to_yaml(f"{config.output_dir}/config.yaml")

    # ── Initialize sandbox ────────────────────────────────────────────────
    logger.info(
        f"Initializing sandbox pool: {config.sandbox_pool_size} workers, "
        f"{config.sandbox_timeout}s timeout, backend={config.sandbox_backend}"
    )
    pool = SandboxPool(
        pool_size=config.sandbox_pool_size,
        timeout=config.sandbox_timeout,
        memory_limit=config.sandbox_memory_limit,
        image=config.sandbox_image,
        backend=config.sandbox_backend,
    )
    set_sandbox_pool(pool)

    # ── Configure reward mode ──────────────────────────────────────────────
    set_reward_mode(config.reward_mode)

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset = load_dataset_for_grpo(config.dataset_name)
    logger.info(f"Training on {len(dataset)} {config.dataset_name} problems")

    # ── Configure LoRA ────────────────────────────────────────────────────
    peft_config = build_lora_config(config)
    logger.info(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}")

    # ── Configure GRPO ────────────────────────────────────────────────────
    grpo_config = build_grpo_config(config)

    # ── Configure callbacks ────────────────────────────────────────────────
    callbacks = [
        SampleLoggerCallback(log_every_n_steps=10, num_samples=4),
        RewardStatsCallback(),
        EarlyStoppingCallback(patience=200, min_reward_threshold=-0.3),
        CompletionLengthMonitor(
            alert_if_below=10,
            alert_if_above=int(config.max_completion_length * 0.9),
        ),
    ]
    logger.info(f"Monitoring: {[type(c).__name__ for c in callbacks]}")

    # ── Create trainer ────────────────────────────────────────────────────
    logger.info(f"Creating GRPOTrainer with model: {config.model_name}")
    trainer = GRPOTrainer(
        model=config.model_name,
        args=grpo_config,
        reward_funcs=[code_execution_reward, format_reward],
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting GRPO training...")
    logger.info(
        f"  Steps: {config.max_steps} | "
        f"Batch: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = "
        f"{config.per_device_train_batch_size * config.gradient_accumulation_steps} prompts/step | "
        f"Completions/step: {config.per_device_train_batch_size * config.gradient_accumulation_steps * config.num_generations}"
    )
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    final_path = f"{config.output_dir}/final"
    trainer.save_model(final_path)
    logger.info(f"Model saved to {final_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
