"""
Production prediction logic for the reduced IEEE-CIS CatBoost model.
"""

from __future__ import annotations

from catboost_model.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
)
from catboost_model.loader import (
    FRAUD_THRESHOLD,
    MODEL_NAME,
    load_catboost_model,
)
from catboost_model.schemas import (
    CatBoostPredictionResponse,
    CatBoostTransactionRequest,
)


def predict_catboost_transaction(
    transaction: CatBoostTransactionRequest,
) -> CatBoostPredictionResponse:
    """Return CatBoost fraud probability and risk decision."""

    model = load_catboost_model()

    feature_frame = build_feature_frame(
        transaction
    )

    fraud_probability = float(
        model.predict_proba(
            feature_frame
        )[0][1]
    )

    risk = (
        "HIGH"
        if fraud_probability >= FRAUD_THRESHOLD
        else "LOW"
    )

    return CatBoostPredictionResponse(
        risk=risk,
        fraud_probability=fraud_probability,
        threshold=FRAUD_THRESHOLD,
        model=MODEL_NAME,
        feature_count=len(
            FEATURE_COLUMNS
        ),
        decision_source="catboost_ieee_cis",
        valid_output=True,
    )
