"""
Train/test splitting and class-balancing utilities.

This module creates independent training and evaluation datasets for the
Financial Fraud Detection System.

The source dataset is split before balancing so that examples selected for
fine-tuning cannot also be selected for evaluation.

Fraud datasets are naturally class-imbalanced. Therefore, balanced sampling
uses an adaptive strategy: the requested sample size is used when enough
examples exist, otherwise the minority-class availability becomes the
effective per-class sample size.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, concatenate_datasets

from ..utils.constants import (
    FRAUD_CLASS,
    FRAUD_LABEL_COLUMN,
    LEGITIMATE_CLASS,
    RANDOM_SEED,
    TEST_SAMPLES_PER_CLASS,
    TEST_SPLIT_RATIO,
    TRAIN_SAMPLES_PER_CLASS,
)


def get_class_counts(dataset: Dataset) -> dict[int, int]:
    """
    Count legitimate and fraudulent transactions.

    Args:
        dataset:
            Hugging Face Dataset containing the fraud-label column.

    Returns:
        Dictionary mapping class labels to their counts.

    Raises:
        TypeError:
            If dataset is not a Hugging Face Dataset.

        ValueError:
            If the dataset is empty or does not contain the fraud label.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "dataset must be an instance of datasets.Dataset."
        )

    if len(dataset) == 0:
        raise ValueError(
            "dataset must contain at least one row."
        )

    if FRAUD_LABEL_COLUMN not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain '{FRAUD_LABEL_COLUMN}'."
        )

    labels = dataset[FRAUD_LABEL_COLUMN]

    legitimate_count = sum(
        label == LEGITIMATE_CLASS
        for label in labels
    )

    fraud_count = sum(
        label == FRAUD_CLASS
        for label in labels
    )

    return {
        LEGITIMATE_CLASS: legitimate_count,
        FRAUD_CLASS: fraud_count,
    }


def _filter_class(
    dataset: Dataset,
    class_label: int,
) -> Dataset:
    """
    Return rows belonging to one fraud class.

    Args:
        dataset:
            Source Hugging Face Dataset.

        class_label:
            Fraud class to retain.

    Returns:
        Filtered Hugging Face Dataset.
    """

    return dataset.filter(
        lambda example: (
            example[FRAUD_LABEL_COLUMN] == class_label
        ),
        desc=f"Selecting class {class_label}",
    )


def split_dataset(
    dataset: Dataset,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> DatasetDict:
    """
    Create independent train and test partitions.

    Each fraud class is split independently before the class partitions are
    recombined. This preserves representation of both classes even when the
    original dataset is highly imbalanced.

    Args:
        dataset:
            Clean Hugging Face Dataset.

        test_ratio:
            Fraction of each class assigned to the test partition.

        seed:
            Deterministic random seed.

    Returns:
        DatasetDict containing ``train`` and ``test``.

    Raises:
        TypeError:
            If argument types are invalid.

        ValueError:
            If the ratio is invalid, the dataset is empty, the fraud-label
            column is missing, or either class has fewer than two examples.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "dataset must be an instance of datasets.Dataset."
        )

    if len(dataset) == 0:
        raise ValueError(
            "dataset must contain at least one row."
        )

    if FRAUD_LABEL_COLUMN not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain '{FRAUD_LABEL_COLUMN}'."
        )

    if isinstance(test_ratio, bool) or not isinstance(
        test_ratio,
        (int, float),
    ):
        raise TypeError(
            "test_ratio must be a numeric value."
        )

    test_ratio = float(test_ratio)

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be greater than or equal to 0."
        )

    legitimate_dataset = _filter_class(
        dataset,
        LEGITIMATE_CLASS,
    )

    fraud_dataset = _filter_class(
        dataset,
        FRAUD_CLASS,
    )

    if len(legitimate_dataset) < 2:
        raise ValueError(
            "At least two legitimate transactions are required "
            "to create train and test partitions."
        )

    if len(fraud_dataset) < 2:
        raise ValueError(
            "At least two fraud transactions are required "
            "to create train and test partitions."
        )

    legitimate_split = legitimate_dataset.train_test_split(
        test_size=test_ratio,
        seed=seed,
        shuffle=True,
    )

    fraud_split = fraud_dataset.train_test_split(
        test_size=test_ratio,
        seed=seed + 1,
        shuffle=True,
    )

    train_dataset = concatenate_datasets(
        [
            legitimate_split["train"],
            fraud_split["train"],
        ]
    ).shuffle(seed=seed)

    test_dataset = concatenate_datasets(
        [
            legitimate_split["test"],
            fraud_split["test"],
        ]
    ).shuffle(seed=seed + 1)

    return DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )


def balance_dataset(
    dataset: Dataset,
    requested_samples_per_class: int,
    seed: int = RANDOM_SEED,
) -> Dataset:
    """
    Create a balanced binary fraud dataset.

    The target sample count is adaptive. When both classes contain at least
    the requested number of examples, that requested size is used. When the
    minority class contains fewer rows, every available minority example is
    used and the majority class is downsampled to the same count.

    Args:
        dataset:
            Source Hugging Face Dataset containing both classes.

        requested_samples_per_class:
            Preferred number of examples from each class.

        seed:
            Deterministic random seed.

    Returns:
        Balanced and shuffled Hugging Face Dataset.

    Raises:
        TypeError:
            If arguments have invalid types.

        ValueError:
            If the requested size is invalid or either fraud class is absent.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "dataset must be an instance of datasets.Dataset."
        )

    if not isinstance(
        requested_samples_per_class,
        int,
    ):
        raise TypeError(
            "requested_samples_per_class must be an integer."
        )

    if requested_samples_per_class <= 0:
        raise ValueError(
            "requested_samples_per_class must be greater than 0."
        )

    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be greater than or equal to 0."
        )

    legitimate_dataset = _filter_class(
        dataset,
        LEGITIMATE_CLASS,
    )

    fraud_dataset = _filter_class(
        dataset,
        FRAUD_CLASS,
    )

    if len(legitimate_dataset) == 0:
        raise ValueError(
            "Dataset contains no legitimate transactions."
        )

    if len(fraud_dataset) == 0:
        raise ValueError(
            "Dataset contains no fraud transactions."
        )

    effective_samples_per_class = min(
        requested_samples_per_class,
        len(legitimate_dataset),
        len(fraud_dataset),
    )

    legitimate_sample = (
        legitimate_dataset
        .shuffle(seed=seed)
        .select(
            range(effective_samples_per_class)
        )
    )

    fraud_sample = (
        fraud_dataset
        .shuffle(seed=seed + 1)
        .select(
            range(effective_samples_per_class)
        )
    )

    balanced_dataset = concatenate_datasets(
        [
            fraud_sample,
            legitimate_sample,
        ]
    )

    return balanced_dataset.shuffle(
        seed=seed + 2
    )


def prepare_balanced_splits(
    dataset: Dataset,
    test_ratio: float = TEST_SPLIT_RATIO,
    train_samples_per_class: int = TRAIN_SAMPLES_PER_CLASS,
    test_samples_per_class: int = TEST_SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> DatasetDict:
    """
    Split a clean dataset and independently balance each partition.

    This is the primary entry point used by later notebooks and training code.

    Processing order:

    1. Split the clean working dataset into independent train/test partitions.
    2. Balance only the training partition.
    3. Balance only the test partition.
    4. Return both datasets.

    Because balancing occurs after splitting, evaluation examples cannot be
    selected from the training partition.

    Args:
        dataset:
            Clean source Hugging Face Dataset.

        test_ratio:
            Fraction assigned to the independent test partition.

        train_samples_per_class:
            Preferred number of examples per class for fine-tuning.

        test_samples_per_class:
            Preferred number of examples per class for evaluation.

        seed:
            Deterministic random seed.

    Returns:
        DatasetDict with balanced ``train`` and ``test`` datasets.
    """

    split_data = split_dataset(
        dataset=dataset,
        test_ratio=test_ratio,
        seed=seed,
    )

    balanced_train = balance_dataset(
        dataset=split_data["train"],
        requested_samples_per_class=(
            train_samples_per_class
        ),
        seed=seed,
    )

    balanced_test = balance_dataset(
        dataset=split_data["test"],
        requested_samples_per_class=(
            test_samples_per_class
        ),
        seed=seed + 10,
    )

    return DatasetDict(
        {
            "train": balanced_train,
            "test": balanced_test,
        }
    )