"""
Transaction API routes.
"""

from __future__ import annotations

from app.repositories.transaction_repository import (
    TransactionRepositoryError,
)
from app.schemas.prediction import (
    TransactionAnalysisResponse,
    TransactionAuditRecord,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)
from app.services.fraud_model_service import (
    FraudModelServiceError,
)
from app.services.transaction_service import (
    TransactionService,
)
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

router = APIRouter(
    prefix="/api/transactions",
    tags=["transactions"],
)

transaction_service = TransactionService()


@router.post(
    "/analyze",
    response_model=TransactionAnalysisResponse,
)
def analyze_transaction(
    transaction: TransactionAnalysisRequest,
) -> TransactionAnalysisResponse:
    """Analyze and persist one transaction."""

    try:
        return transaction_service.analyze(
            transaction
        )

    except FraudModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except TransactionRepositoryError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get(
    "/history",
    response_model=list[TransactionAuditRecord],
)
def transaction_history(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),
) -> list[TransactionAuditRecord]:
    """Return recent transaction analyses."""

    try:
        return transaction_service.list_analyses(
            limit=limit
        )

    except TransactionRepositoryError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get(
    "/{analysis_id}",
    response_model=TransactionAuditRecord,
)
def transaction_analysis(
    analysis_id: str,
) -> TransactionAuditRecord:
    """Return one stored transaction analysis."""

    try:
        result = transaction_service.get_analysis(
            analysis_id
        )

    except TransactionRepositoryError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Transaction analysis not found.",
        )

    return result
