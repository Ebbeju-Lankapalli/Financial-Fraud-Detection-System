"""
Prompt construction for financial fraud inference.

Inference must match the transaction representation used during QLoRA
training as closely as possible.
"""

from __future__ import annotations

from schemas.transaction import TransactionRequest


def format_currency(
    value: float,
) -> str:
    """Format a transaction value exactly like the training prompts."""

    return f"${value:,.2f}"


def build_transaction_prompt(
    transaction: TransactionRequest,
) -> str:
    """
    Convert a transaction into the fraud-classification prompt.

    The wording intentionally matches the prompt format used during
    supervised fine-tuning.
    """

    return (
        "Analyze this transaction for fraud risk:\n"
        f"- Type: {transaction.type}\n"
        f"- Amount: {format_currency(transaction.amount)}\n"
        "- Sender Balance Before: "
        f"{format_currency(transaction.oldbalanceOrg)}\n"
        "- Sender Balance After: "
        f"{format_currency(transaction.newbalanceOrig)}\n"
        "- Recipient Balance Before: "
        f"{format_currency(transaction.oldbalanceDest)}\n"
        "- Recipient Balance After: "
        f"{format_currency(transaction.newbalanceDest)}"
    )


def build_chat_messages(
    transaction: TransactionRequest,
) -> list[dict[str, str]]:
    """
    Build the user-only chat messages used for model inference.

    No assistant label is included because HIGH/LOW is what the model
    must generate.
    """

    return [
        {
            "role": "user",
            "content": build_transaction_prompt(
                transaction
            ),
        }
    ]
