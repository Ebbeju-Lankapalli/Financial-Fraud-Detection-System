"""
Production transaction schemas exposed by the main backend.

The production fraud detector uses the reduced 20-feature IEEE-CIS
CatBoost model.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class TransactionAnalysisRequest(BaseModel):
    """Transaction submitted to the production CatBoost detector."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    card2: float | None = None

    card1: float = Field(
        ...,
        ge=0.0,
    )

    addr1: float | None = Field(
        default=None,
        ge=0.0,
    )

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
