"""
HTTP client for the production CatBoost fraud model service.
"""

from __future__ import annotations

from typing import Any

import requests
from app.core.config import settings
from app.schemas.prediction import (
    FraudPrediction,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)


class FraudModelServiceError(
    RuntimeError
):
    """Raised when the fraud model service cannot return a prediction."""


class FraudModelService:
    """Call the production IEEE-CIS CatBoost inference endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or settings.model_service_url
        ).rstrip("/")

        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.model_service_timeout_seconds
        )

    def _post(
        self,
        path: str,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one POST request to the model service."""

        url = (
            f"{self.base_url}"
            f"{path}"
        )

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )

        except requests.RequestException as exc:
            raise FraudModelServiceError(
                "Fraud model service is unavailable."
            ) from exc

        if response.status_code != 200:
            raise FraudModelServiceError(
                "Fraud model service returned "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        try:
            body = response.json()

        except ValueError as exc:
            raise FraudModelServiceError(
                "Fraud model service returned invalid JSON."
            ) from exc

        if not isinstance(
            body,
            dict,
        ):
            raise FraudModelServiceError(
                "Fraud model service returned "
                "an unexpected response."
            )

        return body

    def predict(
        self,
        transaction: TransactionAnalysisRequest,
    ) -> FraudPrediction:
        """Request a production CatBoost fraud prediction."""

        body = self._post(
            "/catboost/predict",
            payload=(
                transaction.model_dump()
            ),
        )

        try:
            return FraudPrediction.model_validate(
                body
            )

        except Exception as exc:
            raise FraudModelServiceError(
                "Fraud model service returned "
                "an invalid CatBoost prediction schema."
            ) from exc
