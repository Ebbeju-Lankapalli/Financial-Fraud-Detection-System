"""
Transaction orchestration service.
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
)
from uuid import uuid4

from app.repositories.transaction_repository import (
    TransactionRepository,
)
from app.schemas.prediction import (
    TransactionAnalysisResponse,
    TransactionAuditRecord,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)
from app.services.fraud_model_service import (
    FraudModelService,
)


class TransactionService:
    """Coordinate prediction and transaction audit persistence."""

    def __init__(
        self,
        fraud_model_service: FraudModelService | None = None,
        transaction_repository: TransactionRepository | None = None,
    ) -> None:
        self.fraud_model_service = (
            fraud_model_service
            or FraudModelService()
        )

        self.transaction_repository = (
            transaction_repository
            or TransactionRepository()
        )

    def analyze(
        self,
        transaction: TransactionAnalysisRequest,
    ) -> TransactionAnalysisResponse:
        """Analyze and persist one transaction."""

        prediction = (
            self.fraud_model_service.predict(
                transaction
            )
        )

        analysis_id = str(
            uuid4()
        )

        created_at = (
            datetime.now(UTC)
            .isoformat()
        )

        self.transaction_repository.save(
            analysis_id=analysis_id,
            created_at=created_at,
            transaction=transaction,
            prediction=prediction,
        )

        return TransactionAnalysisResponse(
            analysis_id=analysis_id,
            created_at=created_at,
            prediction=prediction,
        )

    def get_analysis(
        self,
        analysis_id: str,
    ) -> TransactionAuditRecord | None:
        """Retrieve a stored analysis."""

        return (
            self.transaction_repository
            .get_by_id(
                analysis_id
            )
        )

    def list_analyses(
        self,
        *,
        limit: int = 50,
    ) -> list[TransactionAuditRecord]:
        """Return recent analyses."""

        return (
            self.transaction_repository
            .list_recent(
                limit=limit
            )
        )
