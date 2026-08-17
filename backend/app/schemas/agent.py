"""
Schemas for the fraud investigation agent.
"""

from __future__ import annotations

from app.schemas.prediction import (
    FraudPrediction,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class AgentAnalysisRequest(BaseModel):
    """Transaction and optional investigator question."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    transaction: TransactionAnalysisRequest

    question: str | None = Field(
        default=None,
        max_length=2000,
    )


class AgentAnalysisResponse(BaseModel):
    """Fraud agent response grounded in the model prediction."""

    model_config = ConfigDict(
        extra="forbid",
    )

    analysis_id: str
    created_at: str

    prediction: FraudPrediction

    explanation: str

    recommendations: list[str]

    agent_model: str | None

    llm_used: bool


class AgentStatusResponse(BaseModel):
    """Agent runtime status."""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: str
    groq_configured: bool
    model: str
