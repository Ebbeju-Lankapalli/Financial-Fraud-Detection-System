"""
Final evaluation of the frozen IEEE-CIS CatBoost fraud detector.

The model and decision threshold were selected using training and
validation data only. This script evaluates them once on the untouched
chronological test partition.
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

from ml.src.ieee_cis.features import (
    TARGET_COLUMN,
    get_categorical_columns,
    get_safe_feature_columns,
)

TEST_PATH = Path(
    "ml/data/ieee_cis/splits/test"
)

MODEL_PATH = Path(
    "artifacts/ieee_cis/catboost_fraud_detector.cbm"
)

THRESHOLD_PATH = Path(
    "artifacts/ieee_cis/threshold_optimization.json"
)

RESULTS_PATH = Path(
    "artifacts/ieee_cis/final_test_metrics.json"
)


def prepare_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Prepare test features exactly like training."""

    features = dataframe[
        feature_columns
    ].copy()

    for column in categorical_columns:
        features[column] = (
            features[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        features[column] = (
            features[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return features


print("=" * 82)
print("IEEE-CIS — FINAL UNTOUCHED TEST EVALUATION")
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
    "Threshold source:",
    "validation set",
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

print(
    "Fraud rows:",
    f"{int(dataframe[TARGET_COLUMN].sum()):,}",
)

feature_columns = (
    get_safe_feature_columns(
        list(dataframe.columns)
    )
)

categorical_columns = (
    get_categorical_columns(
        feature_columns
    )
)

X_test = prepare_features(
    dataframe,
    feature_columns,
    categorical_columns,
)

y_test = (
    dataframe[
        TARGET_COLUMN
    ]
    .astype(int)
    .to_numpy()
)

model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)

print()
print("Generating final predictions...")

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

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
).ravel()

results = {
    "model": "CatBoostClassifier",
    "dataset": "chronological_test",
    "threshold": threshold,
    "threshold_selected_on": "validation",
    "test_rows": len(y_test),
    "fraud_rows": int(
        y_test.sum()
    ),
    "feature_count": len(feature_columns),
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
print("FINAL TEST RESULTS")
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

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print()
print(
    "Final test set used for selection:",
    "NO",
)

print(
    "Saved:",
    RESULTS_PATH,
)

print("=" * 82)
print("STEP 7B.6 PASSED")
print("=" * 82)
