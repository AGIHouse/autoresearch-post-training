"""
Training callbacks for monitoring, sample logging, and early stopping.

These plug into TRL's GRPOTrainer via the `callbacks` argument.
All custom metrics are logged to WandB alongside TRL's built-in metrics.

What TRL logs automatically (we get for free):
    - loss, learning_rate
    - reward/mean, reward/std (per reward function)
    - kl (KL divergence from reference model)
    - completion lengths
    - clip_ratio

What we add:
    1. SampleLoggerCallback: Logs actual generated code to WandB Tables
       so you can READ what the model is producing at each checkpoint.
    2. RewardStatsCallback: Tracks reward distribution, error rates,
       and execution statistics beyond just mean/std.
    3. EvalCallback: Runs held-out evaluation every N steps.
    4. EarlyStoppingCallback: Detects reward collapse or mode collapse
       and stops training before wasting compute.

Usage in train.py:
    trainer = GRPOTrainer(
        ...,
        callbacks=[
            SampleLoggerCallback(log_every_n_steps=10),
            RewardStatsCallback(),
            EarlyStoppingCallback(patience=50),
        ],
    )

All callbacks use WandB for logging. If WandB is not available,
they fall back to console logging.
"""

import logging
from collections import deque
from typing import Any

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


def _try_wandb_log(data: dict, step: int | None = None) -> None:
    """Log to WandB if available, otherwise log to console."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(data, step=step)
            return
    except ImportError:
        pass
    # Fallback: log to console
    logger.info(f"[step {step}] {data}")


class SampleLoggerCallback(TrainerCallback):
    """
    Logs sample completions to WandB Tables at regular intervals.

    This is the most important monitoring tool. Reading actual model
    outputs tells you things metrics can't:
    - Is the model producing valid Python?
    - Is it wrapping code in ```python blocks?
    - Is it generating diverse solutions or mode-collapsing?
    - Is it gaming the reward (e.g., hardcoding test outputs)?

    Logs a WandB Table with columns:
        step, prompt, completion, reward, code_extracted
    """

    def __init__(self, log_every_n_steps: int = 10, num_samples: int = 4):
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples
        self._pending_samples: list[dict] = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict | None = None, **kwargs):
        """Called by TRL after each logging step."""
        if state.global_step % self.log_every_n_steps != 0:
            return

        # TRL stores recent completions in the trainer's internal state.
        # We access them via the model's generate outputs stored during the step.
        # Since we can't directly access completions from the callback,
        # we log whatever metrics are available and rely on the
        # reward function's own logging for sample inspection.
        #
        # For full sample logging, we use a custom reward wrapper (see below).

    def log_samples(self, step: int, samples: list[dict]) -> None:
        """
        Log sample completions. Called from the reward function.

        Args:
            step: Current training step
            samples: List of dicts with keys: prompt, completion, reward
        """
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
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to log samples: {e}")


class RewardStatsCallback(TrainerCallback):
    """
    Tracks detailed reward statistics beyond TRL's built-in mean/std.

    Logs:
        - reward_stats/fraction_perfect: % of completions scoring 1.0
        - reward_stats/fraction_error: % of completions scoring -0.5
        - reward_stats/fraction_zero: % of completions scoring exactly 0.0
        - reward_stats/unique_reward_values: diversity of rewards in batch

    These help detect:
        - Mode collapse: fraction_perfect approaches 1.0 too quickly (memorization)
        - Reward hacking: high mean reward but low fraction_perfect (gaming partial credit)
        - Training failure: fraction_error stays high (model can't produce valid code)
    """

    def __init__(self):
        self._recent_rewards: deque[list[float]] = deque(maxlen=10)

    def record_rewards(self, step: int, rewards: list[float]) -> None:
        """
        Record a batch of rewards for stats computation.
        Call this from the reward function after each batch.
        """
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
    """
    Stops training if reward collapses or stagnates.

    Monitors:
    1. Reward collapse: mean reward drops below threshold for `patience` steps
    2. Reward stagnation: mean reward doesn't improve for `patience` steps

    This prevents wasting GPU hours on a run that has diverged.
    Training on GCP A100 costs ~$3.67/hr — catching a bad run early
    at step 20 instead of step 200 saves real money.
    """

    def __init__(
        self,
        patience: int = 50,
        min_reward_threshold: float = -0.3,
        min_improvement: float = 0.01,
    ):
        self.patience = patience
        self.min_reward_threshold = min_reward_threshold
        self.min_improvement = min_improvement

        self._best_reward: float = float("-inf")
        self._steps_without_improvement: int = 0
        self._steps_below_threshold: int = 0
        self._reward_history: deque[float] = deque(maxlen=patience)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if logs is None:
            return

        # TRL logs reward as "reward" or "reward/mean" or per-function
        reward = logs.get("reward", logs.get("reward/mean"))
        if reward is None:
            # Try to find any reward key
            for key in logs:
                if "reward" in key and "mean" in key:
                    reward = logs[key]
                    break

        if reward is None:
            return

        self._reward_history.append(reward)

        # Check for collapse: reward consistently below threshold
        if reward < self.min_reward_threshold:
            self._steps_below_threshold += 1
        else:
            self._steps_below_threshold = 0

        if self._steps_below_threshold >= self.patience:
            logger.warning(
                f"EARLY STOPPING: Reward below {self.min_reward_threshold} "
                f"for {self.patience} consecutive steps. "
                f"Current reward: {reward:.4f}"
            )
            control.should_training_stop = True
            _try_wandb_log({"early_stopping/reason": "reward_collapse"}, step=state.global_step)
            return

        # Check for stagnation: no improvement for patience steps
        if reward > self._best_reward + self.min_improvement:
            self._best_reward = reward
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if self._steps_without_improvement >= self.patience:
            logger.warning(
                f"EARLY STOPPING: No reward improvement for {self.patience} steps. "
                f"Best: {self._best_reward:.4f}, Current: {reward:.4f}"
            )
            control.should_training_stop = True
            _try_wandb_log({"early_stopping/reason": "stagnation"}, step=state.global_step)
            return

        # Log monitoring stats
        _try_wandb_log({
            "early_stopping/best_reward": self._best_reward,
            "early_stopping/steps_without_improvement": self._steps_without_improvement,
        }, step=state.global_step)


class CompletionLengthMonitor(TrainerCallback):
    """
    Monitors completion lengths to detect mode collapse or length gaming.

    Mode collapse: lengths collapse toward 0 (model gives up)
    Length gaming: lengths explode to max (model pads output to game
                  length normalization — the exact issue Dr.GRPO fixes)

    Alerts are logged as WandB alerts (show up as notifications).
    """

    def __init__(self, alert_if_below: int = 10, alert_if_above: int = 900):
        self.alert_if_below = alert_if_below
        self.alert_if_above = alert_if_above
        self._alerted_collapse = False
        self._alerted_explosion = False

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if logs is None:
            return

        # TRL logs completion length as "completions/mean_length" or similar
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
                f"This may indicate mode collapse. Consider increasing temperature."
            )
            self._alerted_collapse = True
            try:
                import wandb
                if wandb.run:
                    wandb.alert(
                        title="Completion Length Collapse",
                        text=f"Mean completion length dropped to {mean_len:.0f} tokens at step {state.global_step}",
                        level=wandb.AlertLevel.WARN,
                    )
            except (ImportError, Exception):
                pass

        if mean_len > self.alert_if_above and not self._alerted_explosion:
            logger.warning(
                f"ALERT: Mean completion length at {mean_len:.0f} tokens "
                f"(near max). Consider using dr_grpo loss to fix length gaming."
            )
            self._alerted_explosion = True
            try:
                import wandb
                if wandb.run:
                    wandb.alert(
                        title="Completion Length Explosion",
                        text=f"Mean completion length hit {mean_len:.0f} tokens at step {state.global_step}. Try dr_grpo loss.",
                        level=wandb.AlertLevel.WARN,
                    )
            except (ImportError, Exception):
                pass
