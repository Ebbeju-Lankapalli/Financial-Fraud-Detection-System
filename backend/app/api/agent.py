"""
Fraud investigation agent API.
"""

from __future__ import annotations

from app.agents.fraud_agent import (
    FraudAgent,
    FraudAgentError,
)
from app.core.config import settings
from app.schemas.agent import (
    AgentAnalysisRequest,
    AgentAnalysisResponse,
    AgentStatusResponse,
)
from app.services.fraud_model_service import (
    FraudModelServiceError,
)
from fastapi import (
    APIRouter,
    HTTPException,
)

router = APIRouter(
    prefix="/api/agent",
    tags=["agent"],
)

fraud_agent = FraudAgent()


@router.get(
    "",
    response_model=AgentStatusResponse,
)
def agent_status() -> AgentStatusResponse:
    """Return fraud-agent runtime status."""

    return AgentStatusResponse(
        status="ready",
        groq_configured=(
            fraud_agent.groq_configured
        ),
        model=settings.groq_model,
    )


@router.post(
    "/analyze",
    response_model=AgentAnalysisResponse,
)
def analyze_with_agent(
    request: AgentAnalysisRequest,
) -> AgentAnalysisResponse:
    """Analyze a transaction and generate investigation guidance."""

    try:
        return fraud_agent.analyze(
            request
        )

    except FraudModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except FraudAgentError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc
