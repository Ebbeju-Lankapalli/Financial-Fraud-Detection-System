"""
Reproducibility utilities for the Financial Fraud Detection System.

Machine-learning pipelines contain several sources of randomness, including:

- Python's random module
- NumPy
- PyTorch CPU operations
- PyTorch CUDA operations

This module provides one function for configuring those random-number
generators consistently across dataset preparation, training, and evaluation.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .constants import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Configure random seeds used throughout the project.

    The function always configures Python and NumPy.

    If PyTorch is installed, it also configures CPU and CUDA random-number
    generators. PyTorch is imported inside the function so that lightweight
    data-preparation code can still use this module in environments where
    PyTorch has not been installed.

    Args:
        seed:
            Integer seed used by supported random-number generators.

    Returns:
        None.

    Raises:
        TypeError:
            If seed is not an integer.

        ValueError:
            If seed is negative.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be greater than or equal to 0.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_numpy_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """
    Create an independent NumPy random-number generator.

    Using an independent Generator is preferred when a function requires
    local randomness without modifying NumPy's process-wide random state.

    Args:
        seed:
            Integer seed for the generator.

    Returns:
        A NumPy Generator initialized with the supplied seed.

    Raises:
        TypeError:
            If seed is not an integer.

        ValueError:
            If seed is negative.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be greater than or equal to 0.")

    return np.random.default_rng(seed)