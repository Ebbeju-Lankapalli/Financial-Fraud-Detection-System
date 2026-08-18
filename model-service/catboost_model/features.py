"""
Feature definitions for the reduced IEEE-CIS CatBoost production model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from catboost_model.schemas import (
    CatBoostTransactionRequest,
)

FEATURE_COLUMNS = [
    "card2",
    "card1",
    "addr1",
    "C1",
    "D2",
    "C13",
    "C2",
    "M5_enc",
    "D15",
    "C5",
    "C6",
    "C14",
    "M4_enc",
    "purchaser_email_domain",
    "card5",
    "M6_enc",
    "transaction_amt",
    "log_amt",
    "D10",
    "D1",
]


CATEGORICAL_COLUMNS = [
    "purchaser_email_domain",
]


def build_feature_frame(
    transaction: CatBoostTransactionRequest,
) -> pd.DataFrame:
    """Convert one validated request into CatBoost input format."""

    row = transaction.model_dump()

    if row["log_amt"] is None:
        row["log_amt"] = math.log1p(
            row["transaction_amt"]
        )

    dataframe = pd.DataFrame(
        [
            row
        ],
        columns=FEATURE_COLUMNS,
    )

    for column in CATEGORICAL_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    numeric_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in CATEGORICAL_COLUMNS
    ]

    for column in numeric_columns:
        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return dataframe
