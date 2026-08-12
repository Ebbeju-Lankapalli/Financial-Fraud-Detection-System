"""
Preprocessing and validation utilities for fraud transaction data.

This module validates dataset structure and removes records that cannot be
safely used by the model-training pipeline.

It deliberately does not perform train/test splitting or class balancing.
Those responsibilities belong to split_data.py.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

from datasets import Dataset

from ..utils.constants import (
    AMOUNT_COLUMN,
    EXPECTED_TRANSACTION_TYPES,
    FEATURE_COLUMNS,
    FRAUD_CLASS,
    FRAUD_LABEL_COLUMN,
    LEGITIMATE_CLASS,
    MIN_BALANCE,
    MIN_TRANSACTION_AMOUNT,
    RECIPIENT_BALANCE_AFTER_COLUMN,
    RECIPIENT_BALANCE_BEFORE_COLUMN,
    REQUIRED_COLUMNS,
    SENDER_BALANCE_AFTER_COLUMN,
    SENDER_BALANCE_BEFORE_COLUMN,
    TRANSACTION_TYPE_COLUMN,
)

BALANCE_COLUMNS = (
    SENDER_BALANCE_BEFORE_COLUMN,
    SENDER_BALANCE_AFTER_COLUMN,
    RECIPIENT_BALANCE_BEFORE_COLUMN,
    RECIPIENT_BALANCE_AFTER_COLUMN,
)


def validate_dataset_schema(dataset: Dataset) -> None:
    """
    Validate that the dataset contains every column required by the project.

    Args:
        dataset:
            Hugging Face Dataset to validate.

    Raises:
        TypeError:
            If dataset is not a Hugging Face Dataset.

        ValueError:
            If the dataset is empty or required columns are missing.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be an instance of datasets.Dataset.")

    if len(dataset) == 0:
        raise ValueError("dataset must contain at least one row.")

    available_columns = set(dataset.column_names)
    required_columns = set(REQUIRED_COLUMNS)

    missing_columns = sorted(required_columns - available_columns)

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise ValueError(
            f"Dataset is missing required columns: {missing}."
        )


def _is_valid_numeric_value(
    value: object,
    minimum: float,
) -> bool:
    """
    Check whether a value is a finite real number above a minimum.

    Boolean values are rejected even though bool is technically a subclass
    of int in Python.

    Args:
        value:
            Value to validate.

        minimum:
            Inclusive minimum accepted value.

    Returns:
        True when valid, otherwise False.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, Real):
        return False

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return False

    return numeric_value >= minimum


def _is_valid_transaction_type(value: object) -> bool:
    """
    Check whether a transaction type is supported by the project.

    Args:
        value:
            Candidate transaction type.

    Returns:
        True for a supported transaction type, otherwise False.
    """

    if not isinstance(value, str):
        return False

    normalized_value = value.strip().upper()

    return normalized_value in EXPECTED_TRANSACTION_TYPES


def _is_valid_fraud_label(value: object) -> bool:
    """
    Check whether a fraud label represents class 0 or class 1.

    Args:
        value:
            Candidate fraud label.

    Returns:
        True for a valid fraud label, otherwise False.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, Real):
        return False

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return False

    if not numeric_value.is_integer():
        return False

    return int(numeric_value) in {
        LEGITIMATE_CLASS,
        FRAUD_CLASS,
    }


def is_valid_transaction(
    example: Mapping[str, object],
) -> bool:
    """
    Determine whether one transaction can be used by the ML pipeline.

    A valid record must contain:

    - every required field
    - a supported transaction type
    - a finite non-negative transaction amount
    - finite non-negative balances
    - a fraud label of 0 or 1

    Args:
        example:
            Transaction represented by a mapping of column names to values.

    Returns:
        True when the transaction is valid, otherwise False.
    """

    if not isinstance(example, Mapping):
        return False

    for column in REQUIRED_COLUMNS:
        if column not in example:
            return False

    if not _is_valid_transaction_type(
        example[TRANSACTION_TYPE_COLUMN]
    ):
        return False

    if not _is_valid_numeric_value(
        example[AMOUNT_COLUMN],
        MIN_TRANSACTION_AMOUNT,
    ):
        return False

    for column in BALANCE_COLUMNS:
        if not _is_valid_numeric_value(
            example[column],
            MIN_BALANCE,
        ):
            return False

    return _is_valid_fraud_label(
        example[FRAUD_LABEL_COLUMN]
    )


def normalize_transaction(
    example: Mapping[str, object],
) -> dict[str, object]:
    """
    Normalize one valid transaction into consistent project data types.

    Transaction type is converted to uppercase, financial values become
    floats, and the fraud label becomes an integer.

    Args:
        example:
            Valid transaction mapping.

    Returns:
        Dictionary containing normalized feature and label values.

    Raises:
        ValueError:
            If the supplied transaction is invalid.
    """

    if not is_valid_transaction(example):
        raise ValueError(
            "Cannot normalize an invalid transaction."
        )

    return {
        TRANSACTION_TYPE_COLUMN: str(
            example[TRANSACTION_TYPE_COLUMN]
        )
        .strip()
        .upper(),
        AMOUNT_COLUMN: float(
            example[AMOUNT_COLUMN]
        ),
        SENDER_BALANCE_BEFORE_COLUMN: float(
            example[SENDER_BALANCE_BEFORE_COLUMN]
        ),
        SENDER_BALANCE_AFTER_COLUMN: float(
            example[SENDER_BALANCE_AFTER_COLUMN]
        ),
        RECIPIENT_BALANCE_BEFORE_COLUMN: float(
            example[RECIPIENT_BALANCE_BEFORE_COLUMN]
        ),
        RECIPIENT_BALANCE_AFTER_COLUMN: float(
            example[RECIPIENT_BALANCE_AFTER_COLUMN]
        ),
        FRAUD_LABEL_COLUMN: int(
            float(example[FRAUD_LABEL_COLUMN])
        ),
    }


def preprocess_dataset(dataset: Dataset) -> Dataset:
    """
    Validate, filter, normalize, and reduce a transaction dataset.

    Processing order:

    1. Validate source schema.
    2. Remove invalid transactions.
    3. Normalize retained transactions.
    4. Keep only model features and the fraud label.

    Args:
        dataset:
            Source Hugging Face Dataset.

    Returns:
        Clean Hugging Face Dataset ready for train/test splitting.

    Raises:
        ValueError:
            If preprocessing removes every transaction.
    """

    validate_dataset_schema(dataset)

    cleaned_dataset = dataset.filter(
        is_valid_transaction,
        desc="Filtering invalid transactions",
    )

    if len(cleaned_dataset) == 0:
        raise ValueError(
            "No valid transactions remain after preprocessing."
        )

    cleaned_dataset = cleaned_dataset.map(
        normalize_transaction,
        desc="Normalizing transactions",
    )

    columns_to_keep = set(FEATURE_COLUMNS) | {
        FRAUD_LABEL_COLUMN
    }

    columns_to_remove = [
        column
        for column in cleaned_dataset.column_names
        if column not in columns_to_keep
    ]

    if columns_to_remove:
        cleaned_dataset = cleaned_dataset.remove_columns(
            columns_to_remove
        )

    return cleaned_dataset