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
    """Deterministic CatBoost service used by backend API tests."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, transaction) -> FraudPrediction:
        self.calls += 1

        risk = (
            "HIGH"
            if self.calls <= 2
            else "LOW"
        )

        probability = (
            0.92
            if risk == "HIGH"
            else 0.25
        )

        return FraudPrediction(
            risk=risk,
            fraud_probability=probability,
            threshold=0.83,
            model="catboost_reduced_fraud_detector",
            feature_count=20,
            decision_source="catboost_ieee_cis",
            valid_output=True,
        )


def build_test_client(
    tmp_path: Path,
) -> tuple[
    TestClient,
    TransactionRepository,
]:
    database_path = (
        tmp_path
        / "transactions.db"
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
        "card2": 404.0,
        "card1": 13926.0,
        "addr1": 315.0,
        "C1": 1.0,
        "D2": 10.0,
        "C13": 1.0,
        "C2": 1.0,
        "M5_enc": 1.0,
        "D15": 20.0,
        "C5": 0.0,
        "C6": 1.0,
        "C14": 1.0,
        "M4_enc": 0.0,
        "purchaser_email_domain": "gmail.com",
        "card5": 142.0,
        "M6_enc": 1.0,
        "transaction_amt": 250.0,
        "log_amt": None,
        "D10": 12.0,
        "D1": 5.0,
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

    assert (
        body["prediction"][
            "decision_source"
        ]
        == "catboost_ieee_cis"
    )

    assert (
        body["prediction"][
            "threshold"
        ]
        == 0.83
    )

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

    record = history.json()[0]

    assert (
        record["transaction_amt"]
        == 250.0
    )

    assert (
        record["feature_count"]
        == 20
    )

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
    payload["transaction_amt"] = -100

    response = client.post(
        "/api/transactions/analyze",
        json=payload,
    )

    assert response.status_code == 422


def test_database_backend_detection() -> None:
    from app.repositories.transaction_repository import (
        detect_database_backend,
        normalize_postgres_url,
    )

    assert (
        detect_database_backend(
            "sqlite:///./app.db"
        )
        == "sqlite"
    )

    assert (
        detect_database_backend(
            "postgresql://user:pass@host/db"
        )
        == "postgresql"
    )

    assert (
        detect_database_backend(
            "postgres://user:pass@host/db"
        )
        == "postgresql"
    )

    assert (
        normalize_postgres_url(
            "postgres://user:pass@host/db"
        )
        == "postgresql://user:pass@host/db"
    )
