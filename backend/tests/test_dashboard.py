from pathlib import Path

from test_transactions import (
    build_test_client,
    transaction_payload,
)


def test_dashboard_metrics(
    tmp_path: Path,
) -> None:
    client, repository = build_test_client(
        tmp_path
    )

    for _ in range(3):
        response = client.post(
            "/api/transactions/analyze",
            json=transaction_payload(),
        )

        assert response.status_code == 200

    assert repository.count() == 3

    response = client.get(
        "/api/dashboard"
    )

    assert response.status_code == 200

    body = response.json()
    metrics = body["metrics"]

    assert metrics["total_analyses"] == 3
    assert metrics["high_risk_count"] == 2
    assert metrics["low_risk_count"] == 1

    assert (
        metrics["high_risk_percentage"]
        == 66.67
    )

    assert metrics["valid_output_count"] == 3
    assert metrics["invalid_output_count"] == 0

    assert len(
        body["recent_analyses"]
    ) == 3


def test_empty_dashboard(
    tmp_path: Path,
) -> None:
    client, _ = build_test_client(
        tmp_path
    )

    response = client.get(
        "/api/dashboard"
    )

    assert response.status_code == 200

    metrics = response.json()["metrics"]

    assert metrics["total_analyses"] == 0
    assert metrics["high_risk_count"] == 0
    assert metrics["low_risk_count"] == 0

    assert (
        metrics["high_risk_percentage"]
        == 0.0
    )
