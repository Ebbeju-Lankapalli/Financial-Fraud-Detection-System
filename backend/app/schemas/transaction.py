"""
Transaction schemas exposed by the main backend.
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


class TransactionAnalysisRequest(BaseModel):
    """Transaction submitted for fraud analysis."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    type: TransactionType

    amount: float = Field(
        ...,
        ge=0.0,
    )

    oldbalanceOrg: float = Field(
        ...,
        ge=0.0,
    )

    newbalanceOrig: float = Field(
        ...,
        ge=0.0,
    )

    oldbalanceDest: float = Field(
        ...,
        ge=0.0,
    )

    newbalanceDest: float = Field(
        ...,
        ge=0.0,
    )
