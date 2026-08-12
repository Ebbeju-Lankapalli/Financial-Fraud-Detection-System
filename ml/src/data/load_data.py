"""
Dataset loading utilities for the Financial Fraud Detection System.

This module retrieves the Cifer fraud-detection dataset from Hugging Face
and creates a reproducible working subset for the ML pipeline.

Validation, preprocessing, splitting, balancing, and conversation formatting
are intentionally handled by separate modules.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

from ..utils.constants import (
    DATASET_ID,
    DATASET_SPLIT,
    RANDOM_SEED,
    WORKING_SUBSET_SIZE,
)


def load_fraud_dataset(
    dataset_id: str = DATASET_ID,
    split: str = DATASET_SPLIT,
) -> Dataset:
    """
    Load the fraud-detection dataset from Hugging Face Hub.

    Args:
        dataset_id:
            Hugging Face dataset repository identifier.

        split:
            Dataset split to load.

    Returns:
        Hugging Face Dataset containing the requested split.

    Raises:
        ValueError:
            If dataset_id or split is empty.

        RuntimeError:
            If the dataset cannot be loaded or contains no rows.
    """

    if not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string.")

    if not dataset_id.strip():
        raise ValueError(
            "dataset_id must be a non-empty string."
        )

    if not isinstance(split, str):
        raise TypeError("split must be a string.")

    if not split.strip():
        raise ValueError(
            "split must be a non-empty string."
        )

    normalized_dataset_id = dataset_id.strip()
    normalized_split = split.strip()

    try:
        dataset = load_dataset(
            normalized_dataset_id,
            split=normalized_split,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load dataset "
            f"'{normalized_dataset_id}' "
            f"with split '{normalized_split}'."
        ) from exc

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "Expected load_dataset() to return a Dataset "
            "for the requested split."
        )

    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset '{normalized_dataset_id}' "
            f"split '{normalized_split}' contains no rows."
        )

    return dataset


def select_working_subset(
    dataset: Dataset,
    subset_size: int = WORKING_SUBSET_SIZE,
    seed: int = RANDOM_SEED,
) -> Dataset:
    """
    Create a deterministic working subset from a source dataset.

    The source dataset is shuffled before selection so that the working
    subset is not simply the first N records in source-file order.

    Args:
        dataset:
            Source Hugging Face Dataset.

        subset_size:
            Maximum number of rows to retain.

        seed:
            Seed used for deterministic shuffling.

    Returns:
        Dataset containing at most subset_size transactions.

    Raises:
        TypeError:
            If argument types are invalid.

        ValueError:
            If subset_size is not positive, seed is negative, or the
            source dataset is empty.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "dataset must be an instance of datasets.Dataset."
        )

    if not isinstance(subset_size, int):
        raise TypeError(
            "subset_size must be an integer."
        )

    if subset_size <= 0:
        raise ValueError(
            "subset_size must be greater than 0."
        )

    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be greater than or equal to 0."
        )

    if len(dataset) == 0:
        raise ValueError(
            "dataset must contain at least one row."
        )

    selected_size = min(
        subset_size,
        len(dataset),
    )

    shuffled_dataset = dataset.shuffle(
        seed=seed
    )

    return shuffled_dataset.select(
        range(selected_size)
    )


def load_working_dataset(
    dataset_id: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    subset_size: int = WORKING_SUBSET_SIZE,
    seed: int = RANDOM_SEED,
) -> Dataset:
    """
    Load the source dataset and create the project working subset.

    Args:
        dataset_id:
            Hugging Face dataset repository identifier.

        split:
            Dataset split to load.

        subset_size:
            Maximum number of rows to retain.

        seed:
            Seed used for deterministic shuffling.

    Returns:
        Reproducible working Hugging Face Dataset.
    """

    dataset = load_fraud_dataset(
        dataset_id=dataset_id,
        split=split,
    )

    return select_working_subset(
        dataset=dataset,
        subset_size=subset_size,
        seed=seed,
    )