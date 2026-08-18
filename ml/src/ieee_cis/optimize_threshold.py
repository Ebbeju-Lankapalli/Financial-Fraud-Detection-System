"""
Optimize the CatBoost fraud-classification threshold.

IMPORTANT:
The threshold is selected using ONLY the validation dataset.
The final test dataset must remain untouched until the threshold
and model configuration have been frozen.
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

from .features import (
    get_categorical_columns,
    get_safe_feature_columns,
)

VALIDATION_PATH = Path(
    "ml/data/ieee_cis/splits/validation"
)

MODEL_PATH = Path(
    "artifacts/ieee_cis/catboost_fraud_detector.cbm"
)

RESULTS_PATH = Path(
    "artifacts/ieee_cis/threshold_optimization.json"
)


def prepare_dataframe(dataset):
    """Convert the Hugging Face dataset to pandas."""

    dataframe = dataset.to_pandas()

    return dataframe


def prepare_features(dataframe):
    """Prepare model features consistently."""

    feature_columns = get_safe_feature_columns(
        dataframe.columns.tolist()
    )

    categorical_features = get_categorical_columns(
        feature_columns
    )

    X = dataframe[feature_columns].copy()

    for column in categorical_features:
        X[column] = (
            X[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    numerical_features = [
        column
        for column in feature_columns
        if column not in categorical_features
    ]

    for column in numerical_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate fraud metrics at one probability threshold."""

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
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    print("=" * 82)
    print("IEEE-CIS — VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 82)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CatBoost model not found: {MODEL_PATH}"
        )

    print()
    print("Loading validation dataset...")

    validation_dataset = load_from_disk(
        str(VALIDATION_PATH)
    )

    dataframe = prepare_dataframe(
        validation_dataset
    )

    print(
        "Validation rows:",
        f"{len(dataframe):,}",
    )

    print(
        "Fraud rows:",
        f"{int(dataframe['is_fraud'].sum()):,}",
    )

    X_validation = prepare_features(
        dataframe
    )

    y_validation = (
        dataframe["is_fraud"]
        .astype(int)
        .to_numpy()
    )

    print()
    print("Loading trained CatBoost model...")

    model = CatBoostClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # -------------------------------------------------------
    # Broad threshold search
    # -------------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
    )

    results = [
        calculate_metrics(
            y_validation,
            probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    # Primary selection criterion:
    # maximum F1 on validation data.
    best_result = max(
        results,
        key=lambda result: result["f1"],
    )

    print()
    print("=" * 82)
    print("SELECTED THRESHOLD — MAXIMUM VALIDATION F1")
    print("=" * 82)

    print(
        f"Threshold: {best_result['threshold']:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{best_result['accuracy']:.4f} "
        f"({best_result['accuracy']:.2%})"
    )

    print(
        f"Precision: "
        f"{best_result['precision']:.4f} "
        f"({best_result['precision']:.2%})"
    )

    print(
        f"Recall:    "
        f"{best_result['recall']:.4f} "
        f"({best_result['recall']:.2%})"
    )

    print(
        f"F1 Score:  "
        f"{best_result['f1']:.4f} "
        f"({best_result['f1']:.2%})"
    )

    print()
    print("Confusion Matrix")
    print("----------------")

    print(
        "TN:",
        best_result["true_negatives"],
    )

    print(
        "FP:",
        best_result["false_positives"],
    )

    print(
        "FN:",
        best_result["false_negatives"],
    )

    print(
        "TP:",
        best_result["true_positives"],
    )

    # -------------------------------------------------------
    # Show useful comparison thresholds
    # -------------------------------------------------------

    comparison_thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    print()
    print("=" * 82)
    print("THRESHOLD COMPARISON")
    print("=" * 82)

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FP':>8} "
        f"{'FN':>8}"
    )

    print("-" * 82)

    comparison_results = []

    for threshold in comparison_thresholds:
        result = calculate_metrics(
            y_validation,
            probabilities,
            threshold,
        )

        comparison_results.append(
            result
        )

        print(
            f"{threshold:>10.2f} "
            f"{result['precision']:>11.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['f1']:>10.4f} "
            f"{result['false_positives']:>8} "
            f"{result['false_negatives']:>8}"
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "selection_dataset": "validation",
        "selection_metric": "f1",
        "validation_rows": len(dataframe),
        "validation_fraud_rows": int(
            y_validation.sum()
        ),
        "selected_threshold": (
            best_result["threshold"]
        ),
        "selected_metrics": best_result,
        "comparison_thresholds": (
            comparison_results
        ),
        "all_threshold_results": results,
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
    print("=" * 82)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 82)

    print(
        "Selected threshold:",
        f"{best_result['threshold']:.2f}",
    )

    print(
        "Results:",
        RESULTS_PATH,
    )

    print(
        "Final test set used:",
        "NO",
    )

    print()
    print("Threshold selection is now frozen.")
    print("=" * 82)
    print("STEP 7B.5 PASSED")
    print("=" * 82)


if __name__ == "__main__":
    main()
