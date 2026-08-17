"""
Transaction orchestration service.
"""

from __future__ import annotations

from app.schemas.prediction import (
    TransactionAnalysisResponse,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)
from app.services.fraud_model_service import (
    FraudModelService,
)


class TransactionService:
    """Coordinate transaction analysis through the model service."""

    def __init__(
        self,
        fraud_model_service: FraudModelService | None = None,
    ) -> None:
        self.fraud_model_service = (
            fraud_model_service
            or FraudModelService()
        )

    def analyze(
        self,
        transaction: TransactionAnalysisRequest,
    ) -> TransactionAnalysisResponse:
        """Analyze one transaction for fraud risk."""

        prediction = (
            self.fraud_model_service.predict(
                transaction
            )
        )

        return TransactionAnalysisResponse(
            prediction=prediction,
        )
