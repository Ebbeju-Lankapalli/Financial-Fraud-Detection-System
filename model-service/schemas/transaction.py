"""
Pydantic schemas for the fraud model service.

These schemas define the stable request and response contract used by
the model-service API and the main backend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TransactionType = Literal[
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

RiskLabel = Literal[
    "HIGH",
    "LOW",
]


class TransactionRequest(BaseModel):
    """Transaction features required by the fine-tuned fraud model."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    type: TransactionType = Field(
        ...,
        description="Financial transaction type.",
    )

    amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount.",
    )

    oldbalanceOrg: float = Field(
        ...,
        ge=0.0,
        description="Sender balance before the transaction.",
    )

    newbalanceOrig: float = Field(
        ...,
        ge=0.0,
        description="Sender balance after the transaction.",
    )

    oldbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="Recipient balance before the transaction.",
    )

    newbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="Recipient balance after the transaction.",
    )


class PredictionResponse(BaseModel):
    """Fraud-risk prediction returned by the model service."""

    model_config = ConfigDict(
        extra="forbid",
    )

    risk: RiskLabel
    model: str
    adapter: str
    decision_source: Literal["fine_tuned_llm"]
    raw_output: str
    valid_output: bool = True


class BatchPredictionRequest(BaseModel):
    """Request body for batch transaction prediction."""

    model_config = ConfigDict(
        extra="forbid",
    )

    transactions: list[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class BatchPredictionResponse(BaseModel):
    """Predictions returned for a batch request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    count: int = Field(
        ...,
        ge=0,
    )

    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """Metadata describing the active fraud model."""

    model_config = ConfigDict(
        extra="forbid",
    )

    base_model: str
    adapter: str
    training_method: Literal["QLoRA"]
    task: Literal["financial_fraud_risk_classification"]
    labels: list[RiskLabel]
    model_loaded: bool
