"""
Train a leakage-conscious CatBoost fraud classifier on IEEE-CIS.

Training uses the chronological train partition.
Model selection uses only the chronological validation partition.
The final chronological test partition remains untouched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import (
    CatBoostClassifier,
    Pool,
)
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

TRAIN_PATH = Path(
    "ml/data/ieee_cis/splits/train"
)

VALIDATION_PATH = Path(
    "ml/data/ieee_cis/splits/validation"
)

OUTPUT_DIR = Path(
    "artifacts/ieee_cis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    OUTPUT_DIR
    / "catboost_fraud_detector.cbm"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "catboost_validation_metrics.json"
)


def prepare_frame(
    path: Path,
) -> pd.DataFrame:
    """Load one prepared IEEE-CIS split."""

    dataset = load_from_disk(
        str(path)
    )

    return dataset.to_pandas()


def prepare_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Prepare safe CatBoost inputs."""

    features = (
        dataframe[
            feature_columns
        ]
        .copy()
    )

    for column in categorical_columns:
        features[column] = (
            features[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    numeric_columns = [
        column
        for column in feature_columns
        if column
        not in categorical_columns
    ]

    for column in numeric_columns:
        features[column] = (
            pd.to_numeric(
                features[column],
                errors="coerce",
            )
        )

        features[column] = (
            features[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return features


print("=" * 78)
print("IEEE-CIS — CATBOOST FRAUD TRAINING")
print("=" * 78)

train_df = prepare_frame(
    TRAIN_PATH
)

validation_df = prepare_frame(
    VALIDATION_PATH
)

feature_columns = (
    get_safe_feature_columns(
        list(train_df.columns)
    )
)

categorical_columns = (
    get_categorical_columns(
        feature_columns
    )
)

X_train = prepare_features(
    train_df,
    feature_columns,
    categorical_columns,
)

y_train = (
    train_df[
        TARGET_COLUMN
    ]
    .astype(int)
)

X_validation = prepare_features(
    validation_df,
    feature_columns,
    categorical_columns,
)

y_validation = (
    validation_df[
        TARGET_COLUMN
    ]
    .astype(int)
)

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

positive_class_weight = (
    negative_count
    / positive_count
)

print()
print(
    "Train rows:",
    f"{len(X_train):,}",
)

print(
    "Validation rows:",
    f"{len(X_validation):,}",
)

print(
    "Features:",
    len(feature_columns),
)

print(
    "Categorical features:",
    len(categorical_columns),
)

print(
    "Train fraud:",
    positive_count,
)

print(
    "Positive class weight:",
    f"{positive_class_weight:.4f}",
)

train_pool = Pool(
    X_train,
    label=y_train,
    cat_features=(
        categorical_columns
    ),
)

validation_pool = Pool(
    X_validation,
    label=y_validation,
    cat_features=(
        categorical_columns
    ),
)

model = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="PRAUC",
    class_weights=[
        1.0,
        positive_class_weight,
    ],
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
)

print()
print("=" * 78)
print("TRAINING")
print("=" * 78)

start_time = time.time()

model.fit(
    train_pool,
    eval_set=validation_pool,
    use_best_model=True,
    early_stopping_rounds=150,
)

elapsed = (
    time.time()
    - start_time
)

probabilities = (
    model.predict_proba(
        validation_pool
    )[:, 1]
)

predictions = (
    probabilities
    >= 0.50
).astype(int)

accuracy = accuracy_score(
    y_validation,
    predictions,
)

precision = precision_score(
    y_validation,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_validation,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_validation,
    probabilities,
)

pr_auc = average_precision_score(
    y_validation,
    probabilities,
)

tn, fp, fn, tp = (
    confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    ).ravel()
)

results = {
    "model": "CatBoostClassifier",
    "threshold": 0.50,
    "train_rows": len(X_train),
    "validation_rows": len(X_validation),
    "feature_count": len(feature_columns),
    "categorical_feature_count": len(categorical_columns),
    "positive_class_weight": float(
        positive_class_weight
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
    "training_seconds": float(
        elapsed
    ),
    "best_iteration": int(
        model.get_best_iteration()
    ),
}

model.save_model(
    str(MODEL_PATH)
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
print("=" * 78)
print("VALIDATION RESULTS")
print("=" * 78)

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
    "Best iteration:",
    model.get_best_iteration(),
)

print(
    "Training time:",
    f"{elapsed:.2f}s",
)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Metrics:",
    RESULTS_PATH,
)

print("=" * 78)
print("STEP 7B.4 PASSED")
print("=" * 78)
