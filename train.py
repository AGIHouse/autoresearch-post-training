"""
Post-training script for SFT experiments.
This is the file the agent modifies. Everything is fair game:
  - Training hyperparameters
  - LoRA / full fine-tuning config
  - Data formatting and chat templates
  - Optimizer and scheduler
  - Batch size, gradient accumulation
"""

import time
import json
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

from prepare import (
    CACHE_DIR, DATA_DIR, MODEL_DIR, BASE_MODEL, MAX_SEQ_LEN,
    evaluate_model,
)

# ---- Hyperparameters (edit these!) ----
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
TIME_BUDGET_SECONDS = 300  # 5 minute wall-clock budget

# LoRA config (set USE_LORA=False for full fine-tuning)
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ---- Setup ----

def load_model_and_tokenizer():
    """Load base model and tokenizer, optionally apply LoRA."""
    pass


def load_train_data():
    """Load the SFT training dataset from prepared JSON."""
    pass


class TimeBudgetCallback:
    """Stop training after TIME_BUDGET_SECONDS."""

    def __init__(self):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass


# ---- Training ----

def train():
    """Main training loop: load model, train, evaluate, print results."""
    pass


if __name__ == "__main__":
    train()
