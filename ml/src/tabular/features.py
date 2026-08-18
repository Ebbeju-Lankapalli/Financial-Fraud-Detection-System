"""
Feature engineering for the tabular fraud detector.

These features are derived only from information available in the
transaction request. The fraud label is never used as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.src.utils.constants import (
    AMOUNT_COLUMN,
    FRAUD_LABEL_COLUMN,
    RECIPIENT_BALANCE_AFTER_COLUMN,
    RECIPIENT_BALANCE_BEFORE_COLUMN,
    SENDER_BALANCE_AFTER_COLUMN,
    SENDER_BALANCE_BEFORE_COLUMN,
    TRANSACTION_TYPE_COLUMN,
)

NUMERIC_COLUMNS = [
    AMOUNT_COLUMN,
    SENDER_BALANCE_BEFORE_COLUMN,
    SENDER_BALANCE_AFTER_COLUMN,
    RECIPIENT_BALANCE_BEFORE_COLUMN,
    RECIPIENT_BALANCE_AFTER_COLUMN,
]


def build_tabular_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert normalized transaction rows into tabular ML features.

    The transformation does not use isFraud except to leave it available
    to the caller as the target column.
    """

    required = {
        TRANSACTION_TYPE_COLUMN,
        *NUMERIC_COLUMNS,
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Missing tabular feature columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    features = dataframe[
        [
            TRANSACTION_TYPE_COLUMN,
            *NUMERIC_COLUMNS,
        ]
    ].copy()

    features[
        TRANSACTION_TYPE_COLUMN
    ] = (
        features[
            TRANSACTION_TYPE_COLUMN
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Balance movement features.
    features[
        "sender_delta"
    ] = (
        features[
            SENDER_BALANCE_AFTER_COLUMN
        ]
        - features[
            SENDER_BALANCE_BEFORE_COLUMN
        ]
    )

    features[
        "recipient_delta"
    ] = (
        features[
            RECIPIENT_BALANCE_AFTER_COLUMN
        ]
        - features[
            RECIPIENT_BALANCE_BEFORE_COLUMN
        ]
    )

    # Expected-vs-observed balance consistency.
    features[
        "sender_expected_after"
    ] = (
        features[
            SENDER_BALANCE_BEFORE_COLUMN
        ]
        - features[
            AMOUNT_COLUMN
        ]
    )

    features[
        "sender_balance_error"
    ] = (
        features[
            SENDER_BALANCE_AFTER_COLUMN
        ]
        - features[
            "sender_expected_after"
        ]
    )

    features[
        "recipient_expected_after"
    ] = (
        features[
            RECIPIENT_BALANCE_BEFORE_COLUMN
        ]
        + features[
            AMOUNT_COLUMN
        ]
    )

    features[
        "recipient_balance_error"
    ] = (
        features[
            RECIPIENT_BALANCE_AFTER_COLUMN
        ]
        - features[
            "recipient_expected_after"
        ]
    )

    # Scale-independent transaction ratios.
    sender_before = (
        features[
            SENDER_BALANCE_BEFORE_COLUMN
        ]
        .replace(0, np.nan)
    )

    recipient_before = (
        features[
            RECIPIENT_BALANCE_BEFORE_COLUMN
        ]
        .replace(0, np.nan)
    )

    features[
        "amount_sender_ratio"
    ] = (
        features[
            AMOUNT_COLUMN
        ]
        / sender_before
    )

    features[
        "amount_recipient_ratio"
    ] = (
        features[
            AMOUNT_COLUMN
        ]
        / recipient_before
    )

    features[
        "sender_drained"
    ] = (
        features[
            SENDER_BALANCE_AFTER_COLUMN
        ]
        == 0
    ).astype(int)

    features[
        "recipient_was_zero"
    ] = (
        features[
            RECIPIENT_BALANCE_BEFORE_COLUMN
        ]
        == 0
    ).astype(int)

    # Replace undefined/infinite ratios with finite values.
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    features = features.fillna(
        0.0
    )

    return features


def extract_target(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Extract the binary fraud target."""

    if FRAUD_LABEL_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Missing target column: {FRAUD_LABEL_COLUMN}"
        )

    return (
        dataframe[
            FRAUD_LABEL_COLUMN
        ]
        .astype(int)
    )
