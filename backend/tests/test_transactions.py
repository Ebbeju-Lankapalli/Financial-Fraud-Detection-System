from pathlib import Path

import app.api.dashboard as dashboard_api
import app.api.transactions as transactions_api
from app.main import app
from app.repositories.transaction_repository import (
    TransactionRepository,
)
from app.schemas.prediction import FraudPrediction
from app.services.transaction_service import (
    TransactionService,
)
from fastapi.testclient import TestClient


class FakeFraudModelService:
    """Deterministic model service used by API tests."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, transaction) -> FraudPrediction:
        self.calls += 1

        risk = (
            "HIGH"
            if self.calls <= 2
            else "LOW"
        )

        return FraudPrediction(
            risk=risk,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter=(
                "ebbejulankapalli/"
                "financial-fraud-detector-"
                "qwen2.5-qlora"
            ),
            decision_source="fine_tuned_llm",
            raw_output=risk,
            valid_output=True,
        )


def build_test_client(
    tmp_path: Path,
) -> tuple[
    TestClient,
    TransactionRepository,
]:
    database_path = (
        tmp_path / "transactions.db"
    )

    repository = TransactionRepository(
        database_url=(
            f"sqlite:///{database_path}"
        )
    )

    service = TransactionService(
        fraud_model_service=(
            FakeFraudModelService()
        ),
        transaction_repository=repository,
    )

    transactions_api.transaction_service = (
        service
    )

    dashboard_api.transaction_repository = (
        repository
    )

    return TestClient(app), repository


def transaction_payload() -> dict:
    return {
        "type": "TRANSFER",
        "amount": 85000,
        "oldbalanceOrg": 85000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 85000,
    }


def test_analyze_and_persist_transaction(
    tmp_path: Path,
) -> None:
    client, repository = build_test_client(
        tmp_path
    )

    response = client.post(
        "/api/transactions/analyze",
        json=transaction_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"]["risk"] == "HIGH"
    assert body["analysis_id"]
    assert body["created_at"]

    assert repository.count() == 1


def test_transaction_history_and_lookup(
    tmp_path: Path,
) -> None:
    client, _ = build_test_client(
        tmp_path
    )

    created = client.post(
        "/api/transactions/analyze",
        json=transaction_payload(),
    )

    analysis_id = (
        created.json()["analysis_id"]
    )

    history = client.get(
        "/api/transactions/history"
    )

    assert history.status_code == 200
    assert len(history.json()) == 1

    lookup = client.get(
        f"/api/transactions/{analysis_id}"
    )

    assert lookup.status_code == 200
    assert (
        lookup.json()["analysis_id"]
        == analysis_id
    )


def test_unknown_transaction_returns_404(
    tmp_path: Path,
) -> None:
    client, _ = build_test_client(
        tmp_path
    )

    response = client.get(
        "/api/transactions/does-not-exist"
    )

    assert response.status_code == 404


def test_transaction_validation(
    tmp_path: Path,
) -> None:
    client, _ = build_test_client(
        tmp_path
    )

    payload = transaction_payload()
    payload["amount"] = -100

    response = client.post(
        "/api/transactions/analyze",
        json=payload,
    )

    assert response.status_code == 422
