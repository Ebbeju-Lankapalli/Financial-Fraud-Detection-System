"""
Train a reduced IEEE-CIS CatBoost fraud detector.

The reduced feature set is based on feature importance from the
training data. Model comparison is performed on validation data only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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

TRAIN_PATH = Path(
    "ml/data/ieee_cis/splits/train"
)

VALIDATION_PATH = Path(
    "ml/data/ieee_cis/splits/validation"
)

OUTPUT_DIR = Path(
    "artifacts/ieee_cis"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "catboost_reduced_fraud_detector.cbm"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "catboost_reduced_validation_metrics.json"
)


REDUCED_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "purchaser_email_domain",
]


def load_frame(
    path: Path,
) -> pd.DataFrame:
    """Load one cached temporal split."""

    return (
        load_from_disk(
            str(path)
        )
        .to_pandas()
    )


def prepare_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare reduced CatBoost features."""

    missing = (
        set(REDUCED_FEATURES)
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Missing reduced features: "
            + ", ".join(
                sorted(missing)
            )
        )

    features = dataframe[
        REDUCED_FEATURES
    ].copy()

    for column in CATEGORICAL_FEATURES:
        features[column] = (
            features[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    numeric_columns = [
        column
        for column in REDUCED_FEATURES
        if column
        not in CATEGORICAL_FEATURES
    ]

    for column in numeric_columns:
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


print("=" * 80)
print("IEEE-CIS — REDUCED CATBOOST TRAINING")
print("=" * 80)

train_df = load_frame(
    TRAIN_PATH
)

validation_df = load_frame(
    VALIDATION_PATH
)

X_train = prepare_features(
    train_df
)

X_validation = prepare_features(
    validation_df
)

y_train = (
    train_df["is_fraud"]
    .astype(int)
)

y_validation = (
    validation_df["is_fraud"]
    .astype(int)
)

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

positive_weight = (
    negative_count
    / positive_count
)

print()
print(
    "Training rows:",
    f"{len(X_train):,}",
)

print(
    "Validation rows:",
    f"{len(X_validation):,}",
)

print(
    "Reduced features:",
    len(REDUCED_FEATURES),
)

print(
    "Categorical features:",
    CATEGORICAL_FEATURES,
)

print(
    "Positive class weight:",
    f"{positive_weight:.4f}",
)

train_pool = Pool(
    X_train,
    label=y_train,
    cat_features=CATEGORICAL_FEATURES,
)

validation_pool = Pool(
    X_validation,
    label=y_validation,
    cat_features=CATEGORICAL_FEATURES,
)

model = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="PRAUC",
    class_weights=[
        1.0,
        positive_weight,
    ],
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
)

print()
print("=" * 80)
print("TRAINING")
print("=" * 80)

start = time.time()

model.fit(
    train_pool,
    eval_set=validation_pool,
    use_best_model=True,
    early_stopping_rounds=150,
)

elapsed = time.time() - start

probabilities = (
    model.predict_proba(
        validation_pool
    )[:, 1]
)

predictions = (
    probabilities >= 0.50
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
    "variant": "reduced_20_features",
    "threshold": 0.50,
    "features": REDUCED_FEATURES,
    "feature_count": len(
        REDUCED_FEATURES
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
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "best_iteration": int(
        model.get_best_iteration()
    ),
    "training_seconds": float(
        elapsed
    ),
}

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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
print("=" * 80)
print("REDUCED MODEL VALIDATION RESULTS")
print("=" * 80)

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

print("=" * 80)
print("STEP 7B.8 PASSED")
print("=" * 80)
