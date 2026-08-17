"""
FastAPI application for the Financial Fraud Detection model service.
"""

from __future__ import annotations

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
    version="1.0.0",
    description=(
        "Inference service for Qwen2.5-1.5B-Instruct "
        "with a QLoRA fraud-detection adapter."
    ),
)


@app.get("/health")
def health() -> dict[str, object]:
    """Return lightweight service health information."""

    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "device": get_model_device(),
    }


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """Return metadata for the active fraud model."""

    return ModelInfoResponse(
        base_model=BASE_MODEL_ID,
        adapter=ADAPTER_MODEL_ID,
        training_method="QLoRA",
        task="financial_fraud_risk_classification",
        labels=["HIGH", "LOW"],
        model_loaded=is_model_loaded(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
) -> PredictionResponse:
    """Predict fraud risk for one transaction."""

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
    """Predict fraud risk for multiple transactions."""

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
