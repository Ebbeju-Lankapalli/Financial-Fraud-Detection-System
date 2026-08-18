"""
FastAPI application for the Financial Fraud Detection model service.
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
from fastapi import FastAPI, HTTPException
from model.loader import (
    ADAPTER_MODEL_ID,
    BASE_MODEL_ID,
    get_model_device,
    is_model_loaded,
)
from model.predictor import (
    InvalidModelOutputError,
    predict_batch,
    predict_transaction,
)
from schemas.transaction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

app = FastAPI(
    title="Financial Fraud Detection Model Service",
    version="2.0.0",
    description=(
        "Production CatBoost IEEE-CIS fraud classifier "
        "with preserved Qwen2.5 QLoRA inference endpoints."
    ),
)


@app.get("/health")
def health() -> dict[str, object]:
    """Return lightweight service health information."""

    return {
        "status": "ok",
        "qlora_model_loaded": is_model_loaded(),
        "catboost_model_loaded": (
            is_catboost_model_loaded()
        ),
        "qlora_device": get_model_device(),
    }


# =====================================================================
# Production CatBoost endpoints
# =====================================================================


@app.get(
    "/catboost/model/info",
    response_model=CatBoostModelInfoResponse,
)
def catboost_model_info() -> CatBoostModelInfoResponse:
    """Return metadata for the production CatBoost fraud model."""

    return CatBoostModelInfoResponse(
        model=MODEL_NAME,
        model_family="CatBoostClassifier",
        dataset="IEEE-CIS Fraud Detection",
        feature_count=len(
            FEATURE_COLUMNS
        ),
        threshold=FRAUD_THRESHOLD,
        labels=[
            "HIGH",
            "LOW",
        ],
        model_loaded=(
            is_catboost_model_loaded()
        ),
    )


@app.post(
    "/catboost/predict",
    response_model=CatBoostPredictionResponse,
)
def catboost_predict(
    transaction: CatBoostTransactionRequest,
) -> CatBoostPredictionResponse:
    """Return production fraud probability and risk classification."""

    try:
        return predict_catboost_transaction(
            transaction
        )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# =====================================================================
# Preserved QLoRA endpoints
# =====================================================================


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """Return metadata for the preserved QLoRA fraud model."""

    return ModelInfoResponse(
        base_model=BASE_MODEL_ID,
        adapter=ADAPTER_MODEL_ID,
        training_method="QLoRA",
        task="financial_fraud_risk_classification",
        labels=[
            "HIGH",
            "LOW",
        ],
        model_loaded=is_model_loaded(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
) -> PredictionResponse:
    """Predict fraud risk with the preserved QLoRA model."""

    try:
        return predict_transaction(
            transaction
        )

    except InvalidModelOutputError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_many(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict fraud risk for multiple QLoRA transactions."""

    try:
        predictions = predict_batch(
            request.transactions
        )

    except InvalidModelOutputError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return BatchPredictionResponse(
        count=len(predictions),
        predictions=predictions,
    )
