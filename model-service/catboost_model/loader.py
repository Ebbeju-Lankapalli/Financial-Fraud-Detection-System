"""
Lazy loader for the production IEEE-CIS CatBoost fraud model.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from catboost import CatBoostClassifier

MODEL_NAME = "catboost_reduced_fraud_detector"

MODEL_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "artifacts"
    / "ieee_cis"
    / "catboost_reduced_fraud_detector.cbm"
)

FRAUD_THRESHOLD = 0.83


_MODEL_STATE: CatBoostClassifier | None = None
_MODEL_LOCK = Lock()


def load_catboost_model() -> CatBoostClassifier:
    """Load and cache the production CatBoost model."""

    global _MODEL_STATE

    if _MODEL_STATE is not None:
        return _MODEL_STATE

    with _MODEL_LOCK:
        if _MODEL_STATE is not None:
            return _MODEL_STATE

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"CatBoost model not found: {MODEL_PATH}"
            )

        model = CatBoostClassifier()

        model.load_model(
            str(MODEL_PATH)
        )

        _MODEL_STATE = model

        return _MODEL_STATE


def is_catboost_model_loaded() -> bool:
    """Return whether the production model is already loaded."""

    return _MODEL_STATE is not None


def clear_catboost_model_cache() -> None:
    """Clear the cached CatBoost model."""

    global _MODEL_STATE

    _MODEL_STATE = None
