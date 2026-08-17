"""
Strict parser for fraud-model HIGH/LOW output.

The fine-tuned model is expected to generate one of the two labels.
Unexpected model text is rejected rather than silently converted into
a fraud decision.
"""

from __future__ import annotations

import re

from schemas.transaction import RiskLabel

RISK_PATTERN = re.compile(
    r"\b(HIGH|LOW)\b",
    flags=re.IGNORECASE,
)


class InvalidModelOutputError(ValueError):
    """Raised when the model does not return a valid risk label."""


def normalize_model_output(
    output: str,
) -> str:
    """Normalize raw generated text before classification."""

    if not isinstance(output, str):
        raise TypeError(
            "Model output must be a string."
        )

    return output.strip()


def parse_risk_label(
    output: str,
) -> RiskLabel:
    """
    Extract a HIGH or LOW fraud-risk label.

    Exact labels are preferred. If additional text is generated, a
    standalone HIGH/LOW token is accepted. Ambiguous outputs containing
    both labels are rejected.
    """

    normalized = normalize_model_output(
        output
    )

    if not normalized:
        raise InvalidModelOutputError(
            "Model returned an empty response."
        )

    upper_output = normalized.upper()

    if upper_output == "HIGH":
        return "HIGH"

    if upper_output == "LOW":
        return "LOW"

    matches = [
        match.upper()
        for match in RISK_PATTERN.findall(
            normalized
        )
    ]

    unique_labels = set(
        matches
    )

    if unique_labels == {"HIGH"}:
        return "HIGH"

    if unique_labels == {"LOW"}:
        return "LOW"

    if unique_labels == {
        "HIGH",
        "LOW",
    }:
        raise InvalidModelOutputError(
            "Model output is ambiguous because "
            "it contains both HIGH and LOW."
        )

    raise InvalidModelOutputError(
        "Model output does not contain a "
        "valid HIGH or LOW risk label."
    )
