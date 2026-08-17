"""
Evaluation API routes.
"""

from __future__ import annotations

from typing import Any

from app.services.evaluation_service import (
    EvaluationService,
    EvaluationServiceError,
)
from fastapi import (
    APIRouter,
    HTTPException,
)

router = APIRouter(
    prefix="/api/evaluation",
    tags=["evaluation"],
)

evaluation_service = EvaluationService()


def _handle_service_call(
    callback,
):
    """Convert evaluation service errors into HTTP responses."""

    try:
        return callback()

    except EvaluationServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("")
def evaluation_summary() -> dict[str, Any]:
    """Return all major evaluation results."""

    return _handle_service_call(
        evaluation_service.get_summary
    )


@router.get("/base")
def base_model_evaluation() -> dict[str, Any]:
    """Return base-model evaluation metrics."""

    return _handle_service_call(
        evaluation_service.get_base_metrics
    )


@router.get("/finetuned")
def finetuned_model_evaluation() -> dict[str, Any]:
    """Return fine-tuned model evaluation metrics."""

    return _handle_service_call(
        evaluation_service.get_finetuned_metrics
    )


@router.get("/comparison")
def model_comparison() -> dict[str, Any]:
    """Return the base-vs-fine-tuned comparison."""

    return _handle_service_call(
        evaluation_service.get_comparison
    )
