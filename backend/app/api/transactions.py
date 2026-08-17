"""
Transaction API routes.
"""

from __future__ import annotations

from app.schemas.prediction import (
    TransactionAnalysisResponse,
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
    """Analyze one transaction using the QLoRA fraud model."""

    try:
        return transaction_service.analyze(
            transaction
        )

    except FraudModelServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
