"""
Prepare leakage-safe chronological IEEE-CIS datasets.
"""

from __future__ import annotations

from pathlib import Path

from datasets import (
    Dataset,
    load_from_disk,
)

from ml.src.ieee_cis.features import (
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    get_categorical_columns,
    get_safe_feature_columns,
)
from ml.src.ieee_cis.split_data import (
    chronological_split,
)

SOURCE_PATH = Path(
    "ml/data/ieee_cis/full_dataset"
)

OUTPUT_ROOT = Path(
    "ml/data/ieee_cis/splits"
)


def print_distribution(
    name,
    dataframe,
):
    """Print fraud distribution for one partition."""

    fraud = int(
        dataframe[
            TARGET_COLUMN
        ].sum()
    )

    total = len(dataframe)

    legitimate = (
        total
        - fraud
    )

    print()
    print(name)
    print("-" * 72)

    print(
        "Rows:",
        f"{total:,}",
    )

    print(
        "Legitimate:",
        f"{legitimate:,}",
    )

    print(
        "Fraud:",
        f"{fraud:,}",
    )

    print(
        "Fraud rate:",
        f"{fraud / total:.4%}",
    )

    print(
        "Start:",
        dataframe[
            "transaction_ts"
        ].min(),
    )

    print(
        "End:",
        dataframe[
            "transaction_ts"
        ].max(),
    )


print("=" * 76)
print("IEEE-CIS — TEMPORAL SPLIT PREPARATION")
print("=" * 76)

dataset = load_from_disk(
    str(SOURCE_PATH)
)

dataframe = (
    dataset.to_pandas()
)

print(
    "Full rows:",
    f"{len(dataframe):,}",
)

split = chronological_split(
    dataframe,
    train_ratio=0.70,
    validation_ratio=0.15,
)

train = split.train
validation = split.validation
test = split.test


# ============================================================
# Verify strict chronological separation
# ============================================================

assert (
    train[
        "transaction_ts"
    ].max()
    <= validation[
        "transaction_ts"
    ].min()
)

assert (
    validation[
        "transaction_ts"
    ].max()
    <= test[
        "transaction_ts"
    ].min()
)

print()
print(
    "Chronological separation: PASS"
)


# ============================================================
# Fraud distributions
# ============================================================

print_distribution(
    "TRAIN",
    train,
)

print_distribution(
    "VALIDATION",
    validation,
)

print_distribution(
    "TEST",
    test,
)


# ============================================================
# Feature definitions
# ============================================================

feature_columns = (
    get_safe_feature_columns(
        list(
            dataframe.columns
        )
    )
)

categorical_columns = (
    get_categorical_columns(
        feature_columns
    )
)

print()
print("=" * 76)
print("SAFE FEATURE SET")
print("=" * 76)

print(
    "Features:",
    len(feature_columns),
)

print(
    "Categorical:",
    len(categorical_columns),
)

print()
print(
    "Excluded leakage-sensitive columns:"
)

for column in sorted(
    EXCLUDED_COLUMNS
):
    print(
        " -",
        column,
    )

print()
print("Included features:")

for column in feature_columns:
    print(
        " -",
        column,
    )


# ============================================================
# Save chronological partitions
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

partitions = {
    "train": train,
    "validation": validation,
    "test": test,
}

for name, frame in (
    partitions.items()
):

    output_path = (
        OUTPUT_ROOT
        / name
    )

    Dataset.from_pandas(
        frame,
        preserve_index=False,
    ).save_to_disk(
        str(output_path)
    )

    print(
        f"Saved {name}:",
        output_path,
    )


print()
print("=" * 76)
print("IEEE-CIS TEMPORAL SPLIT COMPLETE")
print("=" * 76)

print(
    "Random train/test leakage: PREVENTED"
)

print(
    "Historical fraud-rate features: EXCLUDED"
)

print(
    "Potential future-aware aggregates: EXCLUDED"
)

print("=" * 76)
print("STEP 7B.3 PASSED")
print("=" * 76)
