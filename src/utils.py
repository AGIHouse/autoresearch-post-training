"""
Utility functions for the RL coding agent training system.

Handles:
- Extracting Python code from LLM completions (markdown fences, raw code)
- Reproducibility (seeding)
- Logging setup
"""

import logging
import random
import re

import numpy as np
import torch


logger = logging.getLogger(__name__)


def extract_code_from_completion(completion: str | list[dict]) -> str | None:
    """
    Extract Python code from an LLM completion.

    The model is prompted to wrap code in ```python ... ``` blocks.
    This function handles multiple formats:
      1. Proper markdown fence: ```python\n<code>\n```
      2. Generic fence: ```\n<code>\n```
      3. Raw code (no fences) — used as fallback

    Args:
        completion: Either a string or a list of message dicts (conversational format).
                    If conversational, extracts content from the assistant's last message.

    Returns:
        The extracted code string, or None if the completion is empty.
    """
    # Handle conversational format from TRL
    if isinstance(completion, list):
        # Get the last assistant message
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                completion = msg["content"]
                break
        else:
            return None

    if not isinstance(completion, str) or not completion.strip():
        return None

    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: treat entire completion as code
    return completion.strip()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging format for the training system."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
