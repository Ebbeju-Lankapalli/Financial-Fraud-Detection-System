"""
Conversation-format utilities for supervised fine-tuning.

This module converts structured financial transactions into the user/assistant
message format expected by instruction-tuned language models such as Qwen.

The fraud label is converted into a concise HIGH or LOW risk response.
"""

from __future__ import annotations

from collections.abc import Mapping

from datasets import Dataset

from ..utils.constants import (
    AMOUNT_COLUMN,
    CLASS_TO_RISK_LABEL,
    FRAUD_ANALYSIS_INSTRUCTION,
    FRAUD_LABEL_COLUMN,
    RECIPIENT_BALANCE_AFTER_COLUMN,
    RECIPIENT_BALANCE_BEFORE_COLUMN,
    REQUIRED_COLUMNS,
    SENDER_BALANCE_AFTER_COLUMN,
    SENDER_BALANCE_BEFORE_COLUMN,
    TRANSACTION_TYPE_COLUMN,
    VALID_RISK_LABELS,
)


def build_transaction_prompt(
    example: Mapping[str, object],
) -> str:
    """
    Convert one transaction into the exact fraud-analysis prompt.

    Args:
        example:
            Transaction mapping containing all required project fields.

    Returns:
        Human-readable transaction-analysis prompt.

    Raises:
        TypeError:
            If example is not a mapping.

        ValueError:
            If any required column is missing.
    """

    if not isinstance(example, Mapping):
        raise TypeError(
            "example must be a mapping of transaction fields."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in example
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise ValueError(
            f"Transaction is missing required fields: {missing}."
        )

    transaction_type = str(
        example[TRANSACTION_TYPE_COLUMN]
    ).strip().upper()

    amount = float(
        example[AMOUNT_COLUMN]
    )

    sender_before = float(
        example[SENDER_BALANCE_BEFORE_COLUMN]
    )

    sender_after = float(
        example[SENDER_BALANCE_AFTER_COLUMN]
    )

    recipient_before = float(
        example[RECIPIENT_BALANCE_BEFORE_COLUMN]
    )

    recipient_after = float(
        example[RECIPIENT_BALANCE_AFTER_COLUMN]
    )

    return (
        f"{FRAUD_ANALYSIS_INSTRUCTION}\n"
        f"- Type: {transaction_type}\n"
        f"- Amount: ${amount:,.2f}\n"
        f"- Sender Balance Before: ${sender_before:,.2f}\n"
        f"- Sender Balance After: ${sender_after:,.2f}\n"
        f"- Recipient Balance Before: ${recipient_before:,.2f}\n"
        f"- Recipient Balance After: ${recipient_after:,.2f}"
    )


def get_expected_risk_label(
    example: Mapping[str, object],
) -> str:
    """
    Convert a binary fraud label into HIGH or LOW risk.

    Args:
        example:
            Transaction mapping containing the fraud label.

    Returns:
        ``HIGH`` for fraud and ``LOW`` for legitimate transactions.

    Raises:
        TypeError:
            If example is not a mapping.

        ValueError:
            If the fraud label is missing or unsupported.
    """

    if not isinstance(example, Mapping):
        raise TypeError(
            "example must be a mapping of transaction fields."
        )

    if FRAUD_LABEL_COLUMN not in example:
        raise ValueError(
            f"Transaction must contain '{FRAUD_LABEL_COLUMN}'."
        )

    raw_label = example[FRAUD_LABEL_COLUMN]

    if isinstance(raw_label, bool):
        raise TypeError(
            "Fraud label must be 0 or 1, not a boolean."
        )

    try:
        numeric_label = int(raw_label)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Fraud label must be convertible to integer 0 or 1."
        ) from exc

    if numeric_label not in CLASS_TO_RISK_LABEL:
        raise ValueError(
            f"Unsupported fraud label: {numeric_label}."
        )

    return CLASS_TO_RISK_LABEL[numeric_label]


def create_conversation(
    example: Mapping[str, object],
) -> dict[str, list[dict[str, str]]]:
    """
    Convert one transaction into supervised chat-format training data.

    Output structure:

    {
        "messages": [
            {
                "role": "user",
                "content": "<transaction prompt>"
            },
            {
                "role": "assistant",
                "content": "HIGH"
            }
        ]
    }

    Args:
        example:
            Transaction mapping.

    Returns:
        Dictionary containing the two-message conversation.
    """

    prompt = build_transaction_prompt(
        example
    )

    risk_label = get_expected_risk_label(
        example
    )

    if risk_label not in VALID_RISK_LABELS:
        raise ValueError(
            f"Invalid generated risk label: {risk_label}."
        )

    return {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": risk_label,
            },
        ]
    }


def convert_dataset_to_conversations(
    dataset: Dataset,
) -> Dataset:
    """
    Add a ``messages`` column to every transaction.

    The original transaction columns are retained so that the same dataset
    can still be inspected and used during debugging and evaluation.

    Args:
        dataset:
            Balanced Hugging Face Dataset.

    Returns:
        Dataset containing the original columns plus ``messages``.

    Raises:
        TypeError:
            If dataset is not a Hugging Face Dataset.

        ValueError:
            If dataset is empty.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "dataset must be an instance of datasets.Dataset."
        )

    if len(dataset) == 0:
        raise ValueError(
            "dataset must contain at least one row."
        )

    return dataset.map(
        create_conversation,
        desc="Creating training conversations",
    )