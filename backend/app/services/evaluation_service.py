"""
Evaluation artifact service.

Loads the persisted model evaluation JSON files generated from the
completed Colab evaluation workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationServiceError(
    RuntimeError
):
    """Raised when evaluation artifacts cannot be loaded."""


class EvaluationService:
    """Read model evaluation artifacts from the evaluation directory."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
    ) -> None:
        if project_root is None:
            project_root = (
                Path(__file__)
                .resolve()
                .parents[3]
            )

        self.project_root = project_root

        self.results_dir = (
            self.project_root
            / "evaluation"
            / "results"
        )

    def _load_json(
        self,
        filename: str,
    ) -> dict[str, Any]:
        """Load one evaluation JSON file."""

        path = (
            self.results_dir
            / filename
        )

        if not path.is_file():
            raise EvaluationServiceError(
                f"Evaluation artifact not found: {path}"
            )

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
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
                "Evaluation artifact must contain "
                "a JSON object."
            )

        return data

    def get_base_metrics(
        self,
    ) -> dict[str, Any]:
        """Return base-model evaluation metrics."""

        return self._load_json(
            "base_model_metrics.json"
        )

    def get_finetuned_metrics(
        self,
    ) -> dict[str, Any]:
        """Return fine-tuned model evaluation metrics."""

        return self._load_json(
            "finetuned_model_metrics.json"
        )

    def get_comparison(
        self,
    ) -> dict[str, Any]:
        """Return the base-vs-fine-tuned comparison."""

        return self._load_json(
            "model_comparison.json"
        )

    def get_summary(
        self,
    ) -> dict[str, Any]:
        """Return all major evaluation artifacts in one response."""

        return {
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
