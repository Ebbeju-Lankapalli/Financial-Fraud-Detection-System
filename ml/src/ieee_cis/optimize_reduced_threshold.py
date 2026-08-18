"""
Optimize the decision threshold for the reduced 20-feature CatBoost model.

Threshold selection uses validation data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

VALIDATION_PATH = Path(
    "ml/data/ieee_cis/splits/validation"
)

MODEL_PATH = Path(
    "artifacts/ieee_cis/"
    "catboost_reduced_fraud_detector.cbm"
)

RESULTS_PATH = Path(
    "artifacts/ieee_cis/"
    "reduced_threshold_optimization.json"
)


FEATURES = [
    "card2",
    "card1",
    "addr1",
    "C1",
    "D2",
    "C13",
    "C2",
    "M5_enc",
    "D15",
    "C5",
    "C6",
    "C14",
    "M4_enc",
    "purchaser_email_domain",
    "card5",
    "M6_enc",
    "transaction_amt",
    "log_amt",
    "D10",
    "D1",
]

CATEGORICAL = [
    "purchaser_email_domain",
]


def prepare_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    features = dataframe[
        FEATURES
    ].copy()

    for column in CATEGORICAL:
        features[column] = (
            features[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    numeric = [
        column
        for column in FEATURES
        if column not in CATEGORICAL
    ]

    for column in numeric:
        features[column] = (
            pd.to_numeric(
                features[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return features


def metrics_at_threshold(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


dataset = load_from_disk(
    str(VALIDATION_PATH)
)

dataframe = dataset.to_pandas()

X_validation = prepare_features(
    dataframe
)

y_validation = (
    dataframe["is_fraud"]
    .astype(int)
    .to_numpy()
)

model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)

probabilities = model.predict_proba(
    X_validation
)[:, 1]

thresholds = np.arange(
    0.05,
    0.951,
    0.01,
)

results = [
    metrics_at_threshold(
        y_validation,
        probabilities,
        threshold,
    )
    for threshold in thresholds
]

best = max(
    results,
    key=lambda item: item["f1"],
)

print("=" * 82)
print("REDUCED MODEL — THRESHOLD OPTIMIZATION")
print("=" * 82)

print(
    "Validation rows:",
    f"{len(y_validation):,}",
)

print(
    "Fraud rows:",
    f"{int(y_validation.sum()):,}",
)

print()
print("Best threshold:", f"{best['threshold']:.2f}")

print(
    f"Accuracy:  {best['accuracy']:.4f} "
    f"({best['accuracy']:.2%})"
)

print(
    f"Precision: {best['precision']:.4f} "
    f"({best['precision']:.2%})"
)

print(
    f"Recall:    {best['recall']:.4f} "
    f"({best['recall']:.2%})"
)

print(
    f"F1 Score:  {best['f1']:.4f} "
    f"({best['f1']:.2%})"
)

print()
print("TN:", best["tn"])
print("FP:", best["fp"])
print("FN:", best["fn"])
print("TP:", best["tp"])

print()
print("=" * 82)
print("COMPARISON")
print("=" * 82)

for threshold in [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:
    item = metrics_at_threshold(
        y_validation,
        probabilities,
        threshold,
    )

    print(
        f"{threshold:.2f} | "
        f"P={item['precision']:.4f} | "
        f"R={item['recall']:.4f} | "
        f"F1={item['f1']:.4f} | "
        f"FP={item['fp']} | "
        f"FN={item['fn']}"
    )

output = {
    "model": "reduced_20_feature_catboost",
    "selection_dataset": "validation",
    "selection_metric": "f1",
    "selected_threshold": (
        best["threshold"]
    ),
    "selected_metrics": best,
    "all_thresholds": results,
    "test_set_used": False,
}

with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        indent=2,
    )

print()
print(
    "Saved:",
    RESULTS_PATH,
)

print(
    "Final test used:",
    "NO",
)

print("=" * 82)
print("STEP 7B.9 PASSED")
print("=" * 82)
