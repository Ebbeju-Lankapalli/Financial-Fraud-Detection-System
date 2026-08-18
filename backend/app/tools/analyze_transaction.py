"""
Fraud-analysis tool used by the investigation agent.
"""

from __future__ import annotations

from app.schemas.prediction import (
    TransactionAnalysisResponse,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)
from app.services.transaction_service import (
    TransactionService,
)


class AnalyzeTransactionTool:
    """Run the canonical production fraud-analysis pipeline."""

    def __init__(
        self,
        transaction_service: TransactionService | None = None,
    ) -> None:
        self.transaction_service = (
            transaction_service
            or TransactionService()
        )

    def run(
        self,
        transaction: TransactionAnalysisRequest,
    ) -> TransactionAnalysisResponse:
        """Analyze and persist one transaction."""

        return self.transaction_service.analyze(
            transaction
        )
