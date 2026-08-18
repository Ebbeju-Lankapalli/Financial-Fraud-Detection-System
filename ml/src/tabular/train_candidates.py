"""
Train and compare tabular fraud-detection models.

This script operates only on source_train. The external source_test and
balanced 500-row benchmark remain untouched for final evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.src.tabular.features import (
    build_tabular_features,
    extract_target,
)
from ml.src.tabular.models import (
    build_candidate_models,
)
from ml.src.utils.constants import RANDOM_SEED

TRAIN_PATH = Path(
    "ml/data/tabular/source_train"
)

OUTPUT_DIR = Path(
    "artifacts/tabular"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 76)
print("TABULAR FRAUD MODEL — CANDIDATE TRAINING")
print("=" * 76)

dataset = load_from_disk(
    str(TRAIN_PATH)
)

dataframe = dataset.to_pandas()

fraud_count = int(
    dataframe["isFraud"].sum()
)

print(
    "Source-training rows:",
    f"{len(dataframe):,}",
)

print(
    "Fraud rows:",
    fraud_count,
)

print(
    "Legitimate rows:",
    len(dataframe) - fraud_count,
)

X = build_tabular_features(
    dataframe
)

y = extract_target(
    dataframe
)

print(
    "Feature count:",
    len(X.columns),
)

X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
)

print()
print("Internal split")
print("----------------")

print(
    "Training rows:",
    f"{len(X_train):,}",
)

print(
    "Validation rows:",
    f"{len(X_validation):,}",
)

print(
    "Training fraud:",
    int(y_train.sum()),
)

print(
    "Validation fraud:",
    int(y_validation.sum()),
)

models = build_candidate_models(
    list(X.columns)
)

results = {}

for name, model in models.items():
    print()
    print("=" * 76)
    print("TRAINING:", name)
    print("=" * 76)

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()
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

    elapsed = time.time() - start

    results[name] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "training_seconds": float(
            elapsed
        ),
    }

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
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

    print(
        "Time:",
        f"{elapsed:.2f}s",
    )

    model_path = (
        OUTPUT_DIR
        / f"{name}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        "Saved:",
        model_path,
    )

best_name = max(
    results,
    key=lambda candidate_name: (
        results[candidate_name]["pr_auc"],
        results[candidate_name]["f1"],
    ),
)

results["selection"] = {
    "best_model": best_name,
    "criterion": (
        "Highest validation PR-AUC, "
        "with F1 as tie-breaker"
    ),
}

results_path = (
    OUTPUT_DIR
    / "candidate_results.json"
)

with results_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )

print()
print("=" * 76)
print("CANDIDATE COMPARISON COMPLETE")
print("=" * 76)

for name, metrics in results.items():
    if name == "selection":
        continue

    print()
    print(name)

    print(
        f"  Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"  Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"  F1:        "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:    "
        f"{metrics['pr_auc']:.4f}"
    )

print()
print(
    "Best candidate:",
    best_name,
)

print(
    "Results:",
    results_path,
)

print("=" * 76)
print("STEP 7A.3 PASSED")
print("=" * 76)
