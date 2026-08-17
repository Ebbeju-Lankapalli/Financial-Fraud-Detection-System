"""
Fraud-analysis response schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

RiskLabel = Literal[
    "HIGH",
    "LOW",
]


class FraudPrediction(BaseModel):
    """Normalized prediction received from the model service."""

    model_config = ConfigDict(
        extra="forbid",
    )

    risk: RiskLabel
    model: str
    adapter: str
    decision_source: Literal["fine_tuned_llm"]
    raw_output: str
    valid_output: bool


class TransactionAnalysisResponse(BaseModel):
    """Backend response returned after fraud analysis."""

    model_config = ConfigDict(
        extra="forbid",
    )

    prediction: FraudPrediction
