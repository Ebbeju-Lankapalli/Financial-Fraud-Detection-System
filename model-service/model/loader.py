"""
Lazy model loader for the financial fraud QLoRA service.

The service loads the Qwen base model and the trained PEFT adapter only
when inference is first requested. This avoids expensive model loading
during module import and makes health checks lightweight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

LOGGER = logging.getLogger(__name__)


BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_MODEL_ID = (
    "ebbejulankapalli/"
    "financial-fraud-detector-qwen2.5-qlora"
)


@dataclass
class LoadedFraudModel:
    """Container holding the active fraud model and tokenizer."""

    model: Any
    tokenizer: Any


_MODEL_STATE: LoadedFraudModel | None = None
_MODEL_LOCK = Lock()


def cuda_available() -> bool:
    """Return whether CUDA is available for inference."""

    return torch.cuda.is_available()


def get_compute_dtype():
    """Return the preferred model compute dtype."""

    if cuda_available():
        return torch.float16

    return torch.float32


def create_quantization_config():
    """
    Create the 4-bit inference configuration.

    4-bit loading is enabled only when CUDA is available because
    bitsandbytes quantized inference is intended for supported GPU
    environments.
    """

    if not cuda_available():
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_tokenizer():
    """Load the tokenizer from the fine-tuned adapter repository."""

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_MODEL_ID,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_base_model():
    """Load the Qwen base model for adapter inference."""

    quantization_config = create_quantization_config()

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": get_compute_dtype(),
    }

    if cuda_available():
        load_kwargs["device_map"] = "auto"

    if quantization_config is not None:
        load_kwargs["quantization_config"] = (
            quantization_config
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        **load_kwargs,
    )

    model.config.use_cache = True

    return model


def load_fraud_model() -> LoadedFraudModel:
    """
    Load the base Qwen model and attach the fine-tuned QLoRA adapter.

    The loaded model is cached globally so subsequent predictions reuse
    the same model instance.
    """

    global _MODEL_STATE

    if _MODEL_STATE is not None:
        return _MODEL_STATE

    with _MODEL_LOCK:
        if _MODEL_STATE is not None:
            return _MODEL_STATE

        LOGGER.info(
            "Loading fraud model: base=%s adapter=%s",
            BASE_MODEL_ID,
            ADAPTER_MODEL_ID,
        )

        tokenizer = load_tokenizer()

        base_model = load_base_model()

        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_MODEL_ID,
        )

        model.eval()

        _MODEL_STATE = LoadedFraudModel(
            model=model,
            tokenizer=tokenizer,
        )

        LOGGER.info(
            "Fraud model loaded successfully."
        )

        return _MODEL_STATE


def is_model_loaded() -> bool:
    """Return whether the fraud model has already been loaded."""

    return _MODEL_STATE is not None


def get_model_device() -> str:
    """Return the current inference device."""

    if not is_model_loaded():
        return (
            "cuda"
            if cuda_available()
            else "cpu"
        )

    assert _MODEL_STATE is not None

    try:
        return str(
            next(
                _MODEL_STATE.model.parameters()
            ).device
        )
    except StopIteration:
        return (
            "cuda"
            if cuda_available()
            else "cpu"
        )


def clear_model_cache() -> None:
    """
    Clear the cached model reference.

    Mainly useful for tests and controlled service shutdown.
    """

    global _MODEL_STATE

    _MODEL_STATE = None

    if cuda_available():
        torch.cuda.empty_cache()
