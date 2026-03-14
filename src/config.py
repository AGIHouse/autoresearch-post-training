"""
Central configuration for the RL coding agent training system.

All hyperparameters live here as a single dataclass. This makes it easy to:
- See every tunable knob in one place
- Override via CLI, YAML, or environment variables
- Serialize the exact config used for each training run

The defaults are tuned for: Qwen2.5-Coder-7B-Instruct + LoRA + GRPO
on a single A100 80GB GPU (GCP a2-highgpu-1g instance).
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ── GRPO ───────────────────────────────────────────────────────────────
    num_generations: int = 8            # G: completions sampled per prompt
    max_prompt_length: int = 512        # MBPP prompts are short (~50-100 tokens)
    max_completion_length: int = 1024   # most solutions <200 tokens, but allow more
    temperature: float = 0.9            # sampling temperature for rollouts
    learning_rate: float = 5e-6         # conservative for RL stability
    per_device_train_batch_size: int = 2  # prompts per device per step
    gradient_accumulation_steps: int = 4  # effective batch = 2*4 = 8 prompts = 64 completions
    num_train_epochs: int = 3
    max_steps: int = 200                # ~200 steps * 8 prompts/step ≈ 4 epochs over MBPP
    beta: float = 0.0                   # KL penalty coefficient (0 = disabled, per recent best practice)
    loss_type: str = "grpo"             # "grpo", "dr_grpo", or "dapo"
    scale_rewards: bool = True          # normalize rewards by std within group
    num_iterations: int = 1              # μ: reuse same generations for N gradient updates (saves generation time)

    # ── vLLM ───────────────────────────────────────────────────────────────
    use_vllm: bool = True
    vllm_mode: str = "colocate"         # share GPU between vLLM and training
    vllm_gpu_memory_utilization: float = 0.45  # KV cache allocation (0.45 max without sleep, 0.9 with sleep)
    vllm_sleep_enabled: bool = False          # Free vLLM VRAM during training via sleep() API (TRL >= 0.23)

    # ── Training ───────────────────────────────────────────────────────────
    bf16: bool = True
    gradient_checkpointing: bool = True
    warmup_steps: int = 10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 50
    logging_steps: int = 1
    seed: int = 42

    # ── Sandbox ────────────────────────────────────────────────────────────
    sandbox_pool_size: int = 8          # parallel workers
    sandbox_timeout: int = 10           # seconds per code execution
    sandbox_memory_limit: str = "256m"  # per container (Docker only)
    sandbox_image: str = "coding-sandbox:latest"
    sandbox_backend: str = "auto"       # "auto", "docker", or "subprocess"

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_name: str = "mbpp"          # "mbpp", "rlvr", or "opencoder"

    # ── Paths ──────────────────────────────────────────────────────────────
    output_dir: str = "./outputs"
    wandb_project: str = "coding-agent-rl"

    # ── Reward ─────────────────────────────────────────────────────────────
    reward_weights: list[float] = field(default_factory=lambda: [1.0, 0.1])
    reward_mode: str = "partial"        # "partial" (fractional) or "binary" (all-or-nothing)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load config from a YAML file, overriding defaults."""
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in overrides.items() if hasattr(cls, k)})

    def to_yaml(self, path: str | Path) -> None:
        """Save current config to YAML."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
