"""
Evaluation artifact service.

The project contains two distinct model tracks:

1. Production fraud detector
   - Reduced 20-feature CatBoost
   - IEEE-CIS Fraud Detection
   - Chronological train/validation/test evaluation

2. Research fine-tuning experiment
   - Qwen2.5-1.5B-Instruct + QLoRA
   - CiferAI / PaySim-style transaction data

The production CatBoost model is the active application decision source.
The QLoRA results remain available for research transparency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationServiceError(
    RuntimeError
):
    """Raised when evaluation artifacts cannot be loaded."""


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

QLORA_RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

CATBOOST_RESULTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ieee_cis"
)


class EvaluationService:
    """Load production and research evaluation artifacts."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
    ) -> None:
        self.project_root = (
            project_root
            or PROJECT_ROOT
        )

        self.qlora_results_dir = (
            self.project_root
            / "evaluation"
            / "results"
        )

        self.catboost_results_dir = (
            self.project_root
            / "artifacts"
            / "ieee_cis"
        )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        """Load one JSON evaluation artifact."""

        if not path.exists():
            raise EvaluationServiceError(
                f"Evaluation artifact not found: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise EvaluationServiceError(
                f"Failed to load evaluation artifact: {path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise EvaluationServiceError(
                f"Evaluation artifact must contain an object: {path}"
            )

        return data

    # =================================================================
    # Production CatBoost
    # =================================================================

    def get_production_metrics(
        self,
    ) -> dict[str, Any]:
        """Return final untouched-test metrics for the production model."""

        metrics = self._load_json(
            self.catboost_results_dir
            / "reduced_final_test_metrics.json"
        )

        return {
            "status": "active_production_model",
            "model": (
                "catboost_reduced_fraud_detector"
            ),
            "model_family": (
                "CatBoostClassifier"
            ),
            "dataset": (
                "IEEE-CIS Fraud Detection"
            ),
            "feature_count": metrics[
                "feature_count"
            ],
            "threshold": metrics[
                "threshold"
            ],
            "threshold_selected_on": metrics[
                "threshold_selected_on"
            ],
            "evaluation_split": metrics[
                "dataset"
            ],
            "test_rows": metrics[
                "test_rows"
            ],
            "fraud_rows": metrics[
                "fraud_rows"
            ],
            "metrics": {
                "accuracy": metrics[
                    "accuracy"
                ],
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "roc_auc": metrics[
                    "roc_auc"
                ],
                "pr_auc": metrics[
                    "pr_auc"
                ],
            },
            "confusion_matrix": {
                "true_negatives": metrics[
                    "true_negatives"
                ],
                "false_positives": metrics[
                    "false_positives"
                ],
                "false_negatives": metrics[
                    "false_negatives"
                ],
                "true_positives": metrics[
                    "true_positives"
                ],
            },
            "test_used_for_model_selection": (
                metrics[
                    "test_used_for_model_selection"
                ]
            ),
            "decision_source": (
                "catboost_ieee_cis"
            ),
        }

    def get_production_validation(
        self,
    ) -> dict[str, Any]:
        """Return reduced-model validation metrics."""

        return self._load_json(
            self.catboost_results_dir
            / "catboost_reduced_validation_metrics.json"
        )

    def get_production_threshold(
        self,
    ) -> dict[str, Any]:
        """Return validation-only threshold-selection results."""

        data = self._load_json(
            self.catboost_results_dir
            / "reduced_threshold_optimization.json"
        )

        return {
            "selection_dataset": data[
                "selection_dataset"
            ],
            "selection_metric": data[
                "selection_metric"
            ],
            "selected_threshold": data[
                "selected_threshold"
            ],
            "selected_metrics": data[
                "selected_metrics"
            ],
            "test_set_used": data[
                "test_set_used"
            ],
        }

    def get_full_catboost_reference(
        self,
    ) -> dict[str, Any]:
        """Return the 61-feature CatBoost reference-model results."""

        validation = self._load_json(
            self.catboost_results_dir
            / "catboost_validation_metrics.json"
        )

        test = self._load_json(
            self.catboost_results_dir
            / "final_test_metrics.json"
        )

        return {
            "status": "reference_model",
            "feature_count": 61,
            "validation": validation,
            "test": test,
        }

    # =================================================================
    # QLoRA research experiment
    # =================================================================

    def get_base_metrics(
        self,
    ) -> dict[str, Any]:
        """Return preserved Qwen base-model evaluation."""

        return self._load_json(
            self.qlora_results_dir
            / "base_model_metrics.json"
        )

    def get_finetuned_metrics(
        self,
    ) -> dict[str, Any]:
        """Return preserved QLoRA fine-tuned evaluation."""

        return self._load_json(
            self.qlora_results_dir
            / "finetuned_model_metrics.json"
        )

    def get_comparison(
        self,
    ) -> dict[str, Any]:
        """Return preserved base-vs-QLoRA comparison."""

        return self._load_json(
            self.qlora_results_dir
            / "model_comparison.json"
        )

    def get_research_summary(
        self,
    ) -> dict[str, Any]:
        """Return QLoRA research-model information."""

        return {
            "status": "research_experiment",
            "production_decision_source": False,
            "base_model": (
                self.get_base_metrics()
            ),
            "fine_tuned_model": (
                self.get_finetuned_metrics()
            ),
            "comparison": (
                self.get_comparison()
            ),
        }

    # =================================================================
    # Combined API summary
    # =================================================================

    def get_summary(
        self,
    ) -> dict[str, Any]:
        """Return production-first model evaluation summary."""

        return {
            "production_model": (
                self.get_production_metrics()
            ),
            "production_threshold": (
                self.get_production_threshold()
            ),
            "research_model": (
                self.get_research_summary()
            ),
        }
