"""
Evaluation API routes.

Production CatBoost evaluation is exposed separately from the preserved
Qwen/QLoRA research experiment.
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

evaluation_service = (
    EvaluationService()
)


def _handle_service_call(
    callback,
):
    """Convert evaluation artifact errors into HTTP responses."""

    try:
        return callback()

    except EvaluationServiceError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("")
def evaluation_summary() -> dict[str, Any]:
    """Return production-first evaluation summary."""

    return _handle_service_call(
        evaluation_service.get_summary
    )


@router.get("/production")
def production_evaluation() -> dict[str, Any]:
    """Return active CatBoost production-model metrics."""

    return _handle_service_call(
        evaluation_service.get_production_metrics
    )


@router.get("/production/validation")
def production_validation() -> dict[str, Any]:
    """Return production-model validation metrics."""

    return _handle_service_call(
        evaluation_service.get_production_validation
    )


@router.get("/production/threshold")
def production_threshold() -> dict[str, Any]:
    """Return validation-only production threshold selection."""

    return _handle_service_call(
        evaluation_service.get_production_threshold
    )


@router.get("/production/full-reference")
def full_catboost_reference() -> dict[str, Any]:
    """Return the full 61-feature CatBoost reference results."""

    return _handle_service_call(
        evaluation_service.get_full_catboost_reference
    )


@router.get("/research")
def research_evaluation() -> dict[str, Any]:
    """Return preserved QLoRA research results."""

    return _handle_service_call(
        evaluation_service.get_research_summary
    )


# ---------------------------------------------------------------------
# Backward-compatible QLoRA routes
# ---------------------------------------------------------------------


@router.get("/base")
def base_model_evaluation() -> dict[str, Any]:
    """Return preserved base-Qwen metrics."""

    return _handle_service_call(
        evaluation_service.get_base_metrics
    )


@router.get("/finetuned")
def finetuned_model_evaluation() -> dict[str, Any]:
    """Return preserved QLoRA fine-tuned metrics."""

    return _handle_service_call(
        evaluation_service.get_finetuned_metrics
    )


@router.get("/comparison")
def model_comparison() -> dict[str, Any]:
    """Return preserved QLoRA comparison."""

    return _handle_service_call(
        evaluation_service.get_comparison
    )
