"""
Production CatBoost inference service.

This deployment entry point intentionally excludes the preserved
Qwen/QLoRA research runtime so the production container remains small.
"""

from __future__ import annotations

from catboost_model.features import FEATURE_COLUMNS
from catboost_model.loader import (
    FRAUD_THRESHOLD,
    MODEL_NAME,
    is_catboost_model_loaded,
)
from catboost_model.predictor import (
    predict_catboost_transaction,
)
from catboost_model.schemas import (
    CatBoostModelInfoResponse,
    CatBoostPredictionResponse,
    CatBoostTransactionRequest,
)
from fastapi import FastAPI

app = FastAPI(
    title="Financial Fraud Detection — Production Model Service",
    version="1.0.0",
    description=(
        "Production inference service for the reduced "
        "IEEE-CIS CatBoost fraud classifier."
    ),
)


@app.get("/health")
def health() -> dict[str, object]:
    """Return production model-service health information."""

    return {
        "status": "ok",
        "catboost_model_loaded": (
            is_catboost_model_loaded()
        ),
    }


@app.get(
    "/catboost/model/info",
    response_model=CatBoostModelInfoResponse,
)
def catboost_model_info() -> CatBoostModelInfoResponse:
    """Return production CatBoost model metadata."""

    return CatBoostModelInfoResponse(
        model=MODEL_NAME,
        model_family="CatBoostClassifier",
        dataset="IEEE-CIS Fraud Detection",
        feature_count=len(FEATURE_COLUMNS),
        threshold=FRAUD_THRESHOLD,
        labels=["HIGH", "LOW"],
        model_loaded=is_catboost_model_loaded(),
    )


@app.post(
    "/catboost/predict",
    response_model=CatBoostPredictionResponse,
)
def catboost_predict(
    transaction: CatBoostTransactionRequest,
) -> CatBoostPredictionResponse:
    """Generate one production CatBoost fraud prediction."""

    return predict_catboost_transaction(
        transaction
    )
