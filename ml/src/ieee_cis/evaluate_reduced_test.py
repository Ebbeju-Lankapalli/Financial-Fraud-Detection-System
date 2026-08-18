"""
Final evaluation of the reduced 20-feature IEEE-CIS CatBoost model.

The model architecture and probability threshold were selected using
training and validation data only. The chronological test partition is
used only once for final reporting.
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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TEST_PATH = Path(
    "ml/data/ieee_cis/splits/test"
)

MODEL_PATH = Path(
    "artifacts/ieee_cis/"
    "catboost_reduced_fraud_detector.cbm"
)

THRESHOLD_PATH = Path(
    "artifacts/ieee_cis/"
    "reduced_threshold_optimization.json"
)

RESULTS_PATH = Path(
    "artifacts/ieee_cis/"
    "reduced_final_test_metrics.json"
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
    """Prepare reduced-model test features."""

    missing = (
        set(FEATURES)
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Missing reduced model features: "
            + ", ".join(
                sorted(missing)
            )
        )

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


print("=" * 82)
print("REDUCED CATBOOST — FINAL UNTOUCHED TEST")
print("=" * 82)


with THRESHOLD_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    threshold_data = json.load(
        file
    )

threshold = float(
    threshold_data[
        "selected_threshold"
    ]
)


print()
print(
    "Frozen threshold:",
    f"{threshold:.2f}",
)

print(
    "Threshold selected on:",
    "validation",
)


dataset = load_from_disk(
    str(TEST_PATH)
)

dataframe = dataset.to_pandas()

print()
print(
    "Test rows:",
    f"{len(dataframe):,}",
)

fraud_rows = int(
    dataframe[
        "is_fraud"
    ].sum()
)

print(
    "Fraud rows:",
    f"{fraud_rows:,}",
)


X_test = prepare_features(
    dataframe
)

y_test = (
    dataframe[
        "is_fraud"
    ]
    .astype(int)
    .to_numpy()
)


model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)


print()
print(
    "Generating predictions..."
)


probabilities = model.predict_proba(
    X_test
)[:, 1]


predictions = (
    probabilities >= threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)


tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()
)


results = {
    "model": "CatBoostClassifier",
    "variant": "reduced_20_features",
    "dataset": "chronological_test",
    "threshold": threshold,
    "threshold_selected_on": "validation",
    "test_rows": len(y_test),
    "fraud_rows": int(
        y_test.sum()
    ),
    "feature_count": len(
        FEATURES
    ),
    "accuracy": float(
        accuracy
    ),
    "precision": float(
        precision
    ),
    "recall": float(
        recall
    ),
    "f1": float(
        f1
    ),
    "roc_auc": float(
        roc_auc
    ),
    "pr_auc": float(
        pr_auc
    ),
    "true_negatives": int(
        tn
    ),
    "false_positives": int(
        fp
    ),
    "false_negatives": int(
        fn
    ),
    "true_positives": int(
        tp
    ),
    "test_used_for_model_selection": False,
}


RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )


print()
print("=" * 82)
print("REDUCED MODEL — FINAL TEST RESULTS")
print("=" * 82)

print(
    f"Accuracy:  {accuracy:.4f} "
    f"({accuracy:.2%})"
)

print(
    f"Precision: {precision:.4f} "
    f"({precision:.2%})"
)

print(
    f"Recall:    {recall:.4f} "
    f"({recall:.2%})"
)

print(
    f"F1 Score:  {f1:.4f} "
    f"({f1:.2%})"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)

print()
print("Confusion Matrix")
print("----------------")

print(
    "TN:",
    tn,
)

print(
    "FP:",
    fp,
)

print(
    "FN:",
    fn,
)

print(
    "TP:",
    tp,
)

print()
print(
    "Final test used for selection:",
    "NO",
)

print(
    "Saved:",
    RESULTS_PATH,
)

print("=" * 82)
print("STEP 7B.10 PASSED")
print("=" * 82)
