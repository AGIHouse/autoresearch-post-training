"""
GRPO training script for RL-training a coding agent on MBPP++.
This is the file the agent modifies. Everything is fair game:
  - GRPO hyperparameters (num_generations, temperature, beta, loss_type)
  - LoRA configuration (rank, alpha, target modules)
  - Training hyperparameters (LR, batch size, gradient accumulation)
  - vLLM settings (memory utilization, sleep)
  - Sandbox configuration
  - Reward mode (partial vs binary)
  - Callbacks and monitoring

Usage:
    uv run train.py
    uv run train.py --config configs/default.yaml
"""

import argparse
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from peft import LoraConfig, TaskType
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import GRPOConfig, GRPOTrainer

from prepare import (
    SandboxPool,
    code_execution_reward,
    format_reward,
    load_mbpp_plus_for_grpo,
    set_reward_mode,
    set_sandbox_pool,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Central configuration — all hyperparameters live here."""
    # Model
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # GRPO
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    temperature: float = 0.9
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = 200
    beta: float = 0.0
    loss_type: str = "grpo"
    scale_rewards: bool = True
    num_iterations: int = 1

    # vLLM
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.45
    vllm_sleep_enabled: bool = False

    # Training
    bf16: bool = True
    gradient_checkpointing: bool = True
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 50
    logging_steps: int = 1
    seed: int = 42

    # Sandbox
    sandbox_pool_size: int = 8
    sandbox_timeout: int = 10
    sandbox_memory_limit: str = "256m"
    sandbox_image: str = "coding-sandbox:latest"
    sandbox_backend: str = "auto"

    # Paths
    output_dir: str = "./outputs"
    wandb_project: str = "coding-agent-rl"

    # Reward
    reward_weights: list[float] = field(default_factory=lambda: [1.0, 0.1])
    reward_mode: str = "partial"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config from a YAML file, overriding defaults."""
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in overrides.items() if hasattr(cls, k)})

    def to_yaml(self, path: str | Path) -> None:
        """Save current config to YAML."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Callbacks
# =============================================================================

def _try_wandb_log(data: dict, step: int | None = None) -> None:
    """Log to WandB if available, otherwise log to console."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(data, step=step)
            return
    except ImportError:
        pass
    logger.info(f"[step {step}] {data}")


class SampleLoggerCallback(TrainerCallback):
    """Logs sample completions to WandB Tables at regular intervals."""

    def __init__(self, log_every_n_steps: int = 10, num_samples: int = 4):
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return

    def log_samples(self, step: int, samples: list[dict]) -> None:
        """Log sample completions. Called from the reward function."""
        try:
            import wandb
            if wandb.run is None:
                return

            table = wandb.Table(columns=["step", "prompt_excerpt", "completion_excerpt", "reward"])
            for s in samples[:self.num_samples]:
                prompt_text = s.get("prompt", "")
                if isinstance(prompt_text, list):
                    prompt_text = prompt_text[-1].get("content", "") if prompt_text else ""
                table.add_data(
                    step,
                    prompt_text[:200],
                    s.get("completion", "")[:500],
                    s.get("reward", 0.0),
                )
            wandb.log({"samples/completions": table}, step=step)
        except (ImportError, Exception) as e:
            logger.debug(f"Failed to log samples: {e}")


class RewardStatsCallback(TrainerCallback):
    """Tracks detailed reward statistics beyond TRL's built-in mean/std."""

    def __init__(self):
        self._recent_rewards: deque[list[float]] = deque(maxlen=10)

    def record_rewards(self, step: int, rewards: list[float]) -> None:
        self._recent_rewards.append(rewards)
        if not rewards:
            return

        n = len(rewards)
        stats = {
            "reward_stats/fraction_perfect": sum(1 for r in rewards if r >= 1.0) / n,
            "reward_stats/fraction_error": sum(1 for r in rewards if r <= -0.5) / n,
            "reward_stats/fraction_zero": sum(1 for r in rewards if r == 0.0) / n,
            "reward_stats/fraction_positive": sum(1 for r in rewards if r > 0) / n,
            "reward_stats/unique_reward_values": len(set(round(r, 4) for r in rewards)),
            "reward_stats/min": min(rewards),
            "reward_stats/max": max(rewards),
        }
        _try_wandb_log(stats, step=step)


class EarlyStoppingCallback(TrainerCallback):
    """Stops training if reward collapses or stagnates."""

    def __init__(self, patience: int = 50, min_reward_threshold: float = -0.3,
                 min_improvement: float = 0.01):
        self.patience = patience
        self.min_reward_threshold = min_reward_threshold
        self.min_improvement = min_improvement
        self._best_reward: float = float("-inf")
        self._steps_without_improvement: int = 0
        self._steps_below_threshold: int = 0
        self._reward_history: deque[float] = deque(maxlen=patience)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        reward = logs.get("reward", logs.get("reward/mean"))
        if reward is None:
            for key in logs:
                if "reward" in key and "mean" in key:
                    reward = logs[key]
                    break
        if reward is None:
            return

        self._reward_history.append(reward)

        # Check for collapse
        if reward < self.min_reward_threshold:
            self._steps_below_threshold += 1
        else:
            self._steps_below_threshold = 0

        if self._steps_below_threshold >= self.patience:
            logger.warning(
                f"EARLY STOPPING: Reward below {self.min_reward_threshold} "
                f"for {self.patience} consecutive steps. Current: {reward:.4f}"
            )
            control.should_training_stop = True
            _try_wandb_log({"early_stopping/reason": "reward_collapse"}, step=state.global_step)
            return

        # Check for stagnation
        if reward > self._best_reward + self.min_improvement:
            self._best_reward = reward
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if self._steps_without_improvement >= self.patience:
            logger.warning(
                f"EARLY STOPPING: No improvement for {self.patience} steps. "
                f"Best: {self._best_reward:.4f}, Current: {reward:.4f}"
            )
            control.should_training_stop = True
            _try_wandb_log({"early_stopping/reason": "stagnation"}, step=state.global_step)
            return

        _try_wandb_log({
            "early_stopping/best_reward": self._best_reward,
            "early_stopping/steps_without_improvement": self._steps_without_improvement,
        }, step=state.global_step)


class CompletionLengthMonitor(TrainerCallback):
    """Monitors completion lengths to detect mode collapse or length gaming."""

    def __init__(self, alert_if_below: int = 10, alert_if_above: int = 900):
        self.alert_if_below = alert_if_below
        self.alert_if_above = alert_if_above
        self._alerted_collapse = False
        self._alerted_explosion = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        mean_len = None
        for key in logs:
            if "length" in key.lower() and "completion" in key.lower():
                mean_len = logs[key]
                break
        if mean_len is None:
            return

        if mean_len < self.alert_if_below and not self._alerted_collapse:
            logger.warning(
                f"ALERT: Mean completion length collapsed to {mean_len:.0f} tokens. "
                f"Consider increasing temperature."
            )
            self._alerted_collapse = True

        if mean_len > self.alert_if_above and not self._alerted_explosion:
            logger.warning(
                f"ALERT: Mean completion length at {mean_len:.0f} tokens (near max). "
                f"Consider using dr_grpo loss."
            )
            self._alerted_explosion = True


# =============================================================================
# Training
# =============================================================================

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
    """Main entry point: load config, init sandbox, load data, train, save."""
    parser = argparse.ArgumentParser(description="RL-train a coding agent with GRPO on MBPP++")
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

    # ── Load MBPP++ dataset ───────────────────────────────────────────────
    dataset = load_mbpp_plus_for_grpo()
    logger.info(f"Training on {len(dataset)} MBPP++ problems")

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
    logger.info("Starting GRPO training on MBPP++...")
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
