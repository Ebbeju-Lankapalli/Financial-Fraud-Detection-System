"""
Leakage-conscious IEEE-CIS feature definitions.

The initial production benchmark deliberately excludes identifiers,
target-derived fraud-rate features, and aggregate features whose
historical computation has not yet been independently verified.
"""

from __future__ import annotations

TARGET_COLUMN = "is_fraud"

TIME_COLUMN = "transaction_ts"


EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "transaction_id",
    TIME_COLUMN,
    "transaction_dt",

    # Potential target-derived leakage.
    "card1_historical_fraud_rate",
    "email_historical_fraud_rate",
    "is_high_risk_product",

    # Potentially future-aware aggregate engineering.
    "card1_txn_count",
    "card1_avg_amt",
    "email_txn_count",
    "amt_vs_card_avg_ratio",
}


SAFE_CATEGORICAL_COLUMNS = [
    "product_cd",
    "card4",
    "card6",
    "purchaser_email_domain",
    "recipient_email_domain",
    "device_type",
]


def get_safe_feature_columns(
    available_columns: list[str],
) -> list[str]:
    """Return approved model feature columns."""

    return [
        column
        for column in available_columns
        if column not in EXCLUDED_COLUMNS
    ]


def get_categorical_columns(
    feature_columns: list[str],
) -> list[str]:
    """Return categorical columns present in the dataset."""

    return [
        column
        for column in SAFE_CATEGORICAL_COLUMNS
        if column in feature_columns
    ]


def get_numeric_columns(
    feature_columns: list[str],
) -> list[str]:
    """Return non-categorical model features."""

    categorical = set(
        get_categorical_columns(
            feature_columns
        )
    )

    return [
        column
        for column in feature_columns
        if column not in categorical
    ]
