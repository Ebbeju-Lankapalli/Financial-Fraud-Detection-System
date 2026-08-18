"""
Production CatBoost fraud-classification schemas.

These schemas represent the reduced 20-feature IEEE-CIS production model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLabel = Literal[
    "HIGH",
    "LOW",
]


class CatBoostTransactionRequest(BaseModel):
    """Features required by the reduced IEEE-CIS CatBoost model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    card2: float | None = None
    card1: float
    addr1: float | None = None

    C1: float | None = None
    D2: float | None = None
    C13: float | None = None
    C2: float | None = None
    M5_enc: float | None = None
    D15: float | None = None
    C5: float | None = None
    C6: float | None = None
    C14: float | None = None
    M4_enc: float | None = None

    purchaser_email_domain: str | None = None

    card5: float | None = None
    M6_enc: float | None = None

    transaction_amt: float = Field(
        ...,
        ge=0.0,
    )

    log_amt: float | None = None

    D10: float | None = None
    D1: float | None = None


class CatBoostPredictionResponse(BaseModel):
    """Production fraud prediction returned by CatBoost."""

    model_config = ConfigDict(
        extra="forbid",
    )

    risk: RiskLabel

    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    model: str

    feature_count: int = Field(
        ...,
        ge=1,
    )

    decision_source: Literal[
        "catboost_ieee_cis"
    ]

    valid_output: bool = True


class CatBoostModelInfoResponse(BaseModel):
    """Metadata describing the production CatBoost fraud model."""

    model_config = ConfigDict(
        extra="forbid",
    )

    model: str

    model_family: Literal[
        "CatBoostClassifier"
    ]

    dataset: Literal[
        "IEEE-CIS Fraud Detection"
    ]

    feature_count: int

    threshold: float

    labels: list[RiskLabel]

    model_loaded: bool
