from pathlib import Path

from app.main import app
from app.services.evaluation_service import (
    EvaluationService,
    EvaluationServiceError,
)
from fastapi.testclient import TestClient

client = TestClient(app)


def test_evaluation_service_loads_real_artifacts() -> None:
    service = EvaluationService()

    base = service.get_base_metrics()
    finetuned = (
        service.get_finetuned_metrics()
    )
    comparison = service.get_comparison()

    assert (
        base["model"]
        == "Qwen/Qwen2.5-1.5B-Instruct"
    )

    assert (
        finetuned["model"]
        == (
            "ebbejulankapalli/"
            "financial-fraud-detector-"
            "qwen2.5-qlora"
        )
    )

    assert finetuned["accuracy"] == 0.468
    assert finetuned["recall"] == 0.752

    assert (
        comparison["evaluation"][
            "held_out_rows"
        ]
        == 500
    )


def test_evaluation_summary_endpoint() -> None:
    response = client.get(
        "/api/evaluation"
    )

    assert response.status_code == 200

    body = response.json()

    assert "base_model" in body
    assert "fine_tuned_model" in body
    assert "comparison" in body


def test_evaluation_comparison_endpoint() -> None:
    response = client.get(
        "/api/evaluation/comparison"
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["fine_tuned_model"][
            "f1"
        ]
        == 0.5857
    )


def test_missing_evaluation_artifact_raises_error(
    tmp_path: Path,
) -> None:
    service = EvaluationService(
        project_root=tmp_path
    )

    try:
        service.get_base_metrics()

    except EvaluationServiceError as exc:
        assert "not found" in str(exc)

    else:
        raise AssertionError(
            "Missing artifact should fail."
        )
