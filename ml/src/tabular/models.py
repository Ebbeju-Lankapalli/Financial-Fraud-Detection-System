"""
Candidate scikit-learn models for fraud detection.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from ml.src.utils.constants import (
    RANDOM_SEED,
    TRANSACTION_TYPE_COLUMN,
)


def build_preprocessor(
    feature_columns: list[str],
) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical features."""

    categorical_columns = [
        TRANSACTION_TYPE_COLUMN
    ]

    numeric_columns = [
        column
        for column in feature_columns
        if column
        not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_candidate_models(
    feature_columns: list[str],
) -> dict[str, Pipeline]:
    """Return reproducible fraud-classification candidates."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        feature_columns
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        feature_columns
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        feature_columns
                    ),
                ),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }
