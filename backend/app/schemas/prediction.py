"""
Production fraud-analysis response and audit schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

RiskLabel = Literal[
    "HIGH",
    "LOW",
]


class FraudPrediction(BaseModel):
    """Normalized CatBoost prediction received from the model service."""

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


class TransactionAnalysisResponse(BaseModel):
    """Backend response returned after fraud analysis."""

    model_config = ConfigDict(
        extra="forbid",
    )

    analysis_id: str
    created_at: str
    prediction: FraudPrediction


class TransactionAuditRecord(BaseModel):
    """Persistent production fraud-analysis record."""

    model_config = ConfigDict(
        extra="forbid",
    )

    analysis_id: str
    created_at: str

    card2: float | None
    card1: float
    addr1: float | None

    C1: float | None
    D2: float | None
    C13: float | None
    C2: float | None
    M5_enc: float | None
    D15: float | None
    C5: float | None
    C6: float | None
    C14: float | None
    M4_enc: float | None

    purchaser_email_domain: str | None

    card5: float | None
    M6_enc: float | None

    transaction_amt: float
    log_amt: float | None
    D10: float | None
    D1: float | None

    risk: RiskLabel
    fraud_probability: float
    threshold: float
    model: str
    feature_count: int
    decision_source: Literal[
        "catboost_ieee_cis"
    ]
    valid_output: bool
