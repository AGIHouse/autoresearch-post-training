"""
Fixed utilities, one-time data prep, and evaluation for post-training experiments.
This file is READ-ONLY for the agent — do not modify.
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ---- Constants ----
CACHE_DIR = Path.home() / ".cache" / "autoresearch-post-training"
DATA_DIR = CACHE_DIR / "data"
MODEL_DIR = CACHE_DIR / "models"
MAX_SEQ_LEN = 2048
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"  # small model for fast iteration

# ---- Data Preparation ----

def download_base_model():
    """Download and cache the base model and tokenizer."""
    pass


def prepare_sft_data():
    """Download and prepare SFT dataset."""
    pass


def prepare_dpo_data():
    """Download and prepare DPO preference dataset."""
    pass


# ---- Evaluation ----

def evaluate_model(model, tokenizer, eval_data_path, max_samples=200):
    """
    Evaluate a post-trained model. Returns a dict of metrics.
    This is the ground truth evaluation — do not modify.
    """
    pass


# ---- Main ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for post-training experiments")
    parser.add_argument("--sft", action="store_true", help="Prepare SFT data")
    parser.add_argument("--dpo", action="store_true", help="Prepare DPO data")
    parser.add_argument("--all", action="store_true", help="Prepare everything")
    args = parser.parse_args()

    if args.all or (not args.sft and not args.dpo):
        args.sft = True
        args.dpo = True

    download_base_model()

    if args.sft:
        prepare_sft_data()
    if args.dpo:
        prepare_dpo_data()

    print("Done! Data preparation complete.")
