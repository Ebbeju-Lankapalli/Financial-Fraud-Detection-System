from pathlib import Path

import pytest
from app.main import app
from app.services.evaluation_service import (
    EvaluationService,
    EvaluationServiceError,
)
from fastapi.testclient import TestClient

client = TestClient(
    app
)


def test_evaluation_service_loads_real_artifacts() -> None:
    service = EvaluationService()

    production = (
        service.get_production_metrics()
    )

    assert (
        production["status"]
        == "active_production_model"
    )

    assert (
        production["feature_count"]
        == 20
    )

    assert round(
        production["threshold"],
        2,
    ) == 0.83

    assert (
        production["decision_source"]
        == "catboost_ieee_cis"
    )

    research = (
        service.get_research_summary()
    )

    assert (
        research["status"]
        == "research_experiment"
    )

    assert (
        research[
            "production_decision_source"
        ]
        is False
    )


def test_evaluation_summary_endpoint() -> None:
    response = client.get(
        "/api/evaluation"
    )

    assert response.status_code == 200

    body = response.json()

    assert "production_model" in body
    assert "production_threshold" in body
    assert "research_model" in body

    production = body[
        "production_model"
    ]

    assert (
        production["status"]
        == "active_production_model"
    )

    assert (
        production["model"]
        == "catboost_reduced_fraud_detector"
    )

    assert (
        production["feature_count"]
        == 20
    )

    assert round(
        production["threshold"],
        2,
    ) == 0.83

    assert (
        production[
            "test_used_for_model_selection"
        ]
        is False
    )

    research = body[
        "research_model"
    ]

    assert (
        research["status"]
        == "research_experiment"
    )

    assert (
        research[
            "production_decision_source"
        ]
        is False
    )


def test_evaluation_comparison_endpoint() -> None:
    response = client.get(
        "/api/evaluation/comparison"
    )

    assert response.status_code == 200

    body = response.json()

    assert "base_model" in body
    assert "fine_tuned_model" in body


def test_production_evaluation_endpoint() -> None:
    response = client.get(
        "/api/evaluation/production"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "active_production_model"
    )

    assert (
        body["decision_source"]
        == "catboost_ieee_cis"
    )

    assert (
        body["feature_count"]
        == 20
    )


def test_production_threshold_endpoint() -> None:
    response = client.get(
        "/api/evaluation/production/threshold"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["selection_dataset"]
        == "validation"
    )

    assert (
        body["selection_metric"]
        == "f1"
    )

    assert round(
        body["selected_threshold"],
        2,
    ) == 0.83

    assert (
        body["test_set_used"]
        is False
    )


def test_research_evaluation_endpoint() -> None:
    response = client.get(
        "/api/evaluation/research"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "research_experiment"
    )

    assert (
        body[
            "production_decision_source"
        ]
        is False
    )


def test_missing_evaluation_artifact_raises_error(
    tmp_path: Path,
) -> None:
    service = EvaluationService(
        project_root=tmp_path
    )

    with pytest.raises(
        EvaluationServiceError
    ):
        service.get_production_metrics()
