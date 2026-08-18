"""
Inspect CatBoost feature importance for production feature selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import (
    CatBoostClassifier,
    Pool,
)
from datasets import load_from_disk

from ml.src.ieee_cis.features import (
    TARGET_COLUMN,
    get_categorical_columns,
    get_safe_feature_columns,
)

TRAIN_PATH = Path(
    "ml/data/ieee_cis/splits/train"
)

MODEL_PATH = Path(
    "artifacts/ieee_cis/catboost_fraud_detector.cbm"
)

OUTPUT_PATH = Path(
    "artifacts/ieee_cis/feature_importance.json"
)


dataset = load_from_disk(
    str(TRAIN_PATH)
)

dataframe = dataset.to_pandas()

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

X = dataframe[
    feature_columns
].copy()

for column in categorical_columns:
    X[column] = (
        X[column]
        .fillna("__MISSING__")
        .astype(str)
    )

numeric_columns = [
    column
    for column in feature_columns
    if column not in categorical_columns
]

for column in numeric_columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce",
    )

y = (
    dataframe[
        TARGET_COLUMN
    ]
    .astype(int)
)

pool = Pool(
    X,
    label=y,
    cat_features=categorical_columns,
)

model = CatBoostClassifier()

model.load_model(
    str(MODEL_PATH)
)

importance = model.get_feature_importance(
    pool
)

ranking = sorted(
    zip(
        feature_columns,
        importance,
        strict=True,
    ),
    key=lambda item: item[1],
    reverse=True,
)

output = [
    {
        "rank": index,
        "feature": feature,
        "importance": float(value),
    }
    for index, (
        feature,
        value,
    ) in enumerate(
        ranking,
        start=1,
    )
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        indent=2,
    )

print("=" * 76)
print("CATBOOST FEATURE IMPORTANCE")
print("=" * 76)

for item in output[:25]:
    print(
        f"{item['rank']:>2}. "
        f"{item['feature']:<30} "
        f"{item['importance']:>9.4f}"
    )

top_10_share = sum(
    item["importance"]
    for item in output[:10]
)

top_20_share = sum(
    item["importance"]
    for item in output[:20]
)

total_importance = sum(
    item["importance"]
    for item in output
)

print()
print(
    "Top-10 importance share:",
    f"{top_10_share / total_importance:.2%}",
)

print(
    "Top-20 importance share:",
    f"{top_20_share / total_importance:.2%}",
)

print()
print(
    "Saved:",
    OUTPUT_PATH,
)

print("=" * 76)
print("STEP 7B.7 PASSED")
print("=" * 76)
