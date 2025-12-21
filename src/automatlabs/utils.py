"""Utility functions."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def save_json(data: dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


