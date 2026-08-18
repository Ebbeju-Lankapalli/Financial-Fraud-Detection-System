"""
Leakage-aware chronological splitting for IEEE-CIS fraud data.

Transactions are ordered by timestamp before splitting so validation
and test transactions occur strictly after their corresponding
training periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological fraud-model datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    dataframe: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> TemporalSplit:
    """Split transactions chronologically into train/validation/test."""

    if dataframe.empty:
        raise ValueError(
            "dataframe must contain transactions."
        )

    if "transaction_ts" not in dataframe.columns:
        raise ValueError(
            "transaction_ts column is required."
        )

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    ordered = (
        dataframe
        .sort_values(
            "transaction_ts"
        )
        .reset_index(
            drop=True
        )
    )

    train_end = int(
        len(ordered)
        * train_ratio
    )

    validation_end = int(
        len(ordered)
        * (
            train_ratio
            + validation_ratio
        )
    )

    train = (
        ordered
        .iloc[:train_end]
        .copy()
    )

    validation = (
        ordered
        .iloc[
            train_end:
            validation_end
        ]
        .copy()
    )

    test = (
        ordered
        .iloc[
            validation_end:
        ]
        .copy()
    )

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )
