"""
Fraud prediction logic for the model service.

This module converts validated transaction data into the exact prompt
format used during QLoRA fine-tuning, runs deterministic generation,
and parses the model output into a strict HIGH/LOW decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from schemas.transaction import (
    PredictionResponse,
    TransactionRequest,
)
from utils.output_parser import (
    InvalidModelOutputError,
    parse_risk_label,
)

from model.loader import (
    ADAPTER_MODEL_ID,
    BASE_MODEL_ID,
    LoadedFraudModel,
    load_fraud_model,
)
from model.prompt_builder import build_chat_messages

DEFAULT_MAX_NEW_TOKENS = 5


@dataclass
class RawPrediction:
    """Internal model-generation result."""

    raw_output: str


def move_inputs_to_model_device(
    inputs,
    model,
):
    """Move tokenizer tensors to the active model device."""

    try:
        device = next(
            model.parameters()
        ).device
    except StopIteration:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


def generate_raw_prediction(
    transaction: TransactionRequest,
    loaded_model: LoadedFraudModel,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> RawPrediction:
    """
    Generate the raw HIGH/LOW response for one transaction.

    Generation is deterministic because this service performs
    classification rather than creative text generation.
    """

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than 0."
        )

    model = loaded_model.model
    tokenizer = loaded_model.tokenizer

    messages = build_chat_messages(
        transaction
    )

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = move_inputs_to_model_device(
        inputs,
        model,
    )

    prompt_length = (
        inputs["input_ids"].shape[-1]
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[
        0,
        prompt_length:,
    ]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return RawPrediction(
        raw_output=raw_output,
    )


def predict_transaction(
    transaction: TransactionRequest,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> PredictionResponse:
    """
    Predict HIGH or LOW fraud risk for one validated transaction.

    Raises:
        InvalidModelOutputError:
            If the fine-tuned model does not return an unambiguous
            HIGH/LOW prediction.
    """

    loaded_model = load_fraud_model()

    raw_prediction = generate_raw_prediction(
        transaction,
        loaded_model,
        max_new_tokens=max_new_tokens,
    )

    risk = parse_risk_label(
        raw_prediction.raw_output
    )

    return PredictionResponse(
        risk=risk,
        model=BASE_MODEL_ID,
        adapter=ADAPTER_MODEL_ID,
        decision_source="fine_tuned_llm",
        raw_output=raw_prediction.raw_output,
        valid_output=True,
    )


def predict_batch(
    transactions: list[TransactionRequest],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[PredictionResponse]:
    """Predict fraud risk for multiple transactions sequentially."""

    if not transactions:
        raise ValueError(
            "transactions cannot be empty."
        )

    predictions = []

    for transaction in transactions:
        predictions.append(
            predict_transaction(
                transaction,
                max_new_tokens=max_new_tokens,
            )
        )

    return predictions


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "InvalidModelOutputError",
    "RawPrediction",
    "generate_raw_prediction",
    "predict_batch",
    "predict_transaction",
]
