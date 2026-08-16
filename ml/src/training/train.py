"""
QLoRA supervised fine-tuning engine for the Financial Fraud Detection System.

This module provides the reusable training pipeline used to fine-tune
Qwen2.5-1.5B-Instruct on fraud-risk conversations.

The training workflow is intentionally separated from notebooks so the same
pipeline can be executed from Colab, a GPU workstation, or another training
environment without duplicating training logic.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import (
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ..data.conversation_format import convert_dataset_to_conversations
from ..utils.constants import CONFIG_DIR, RANDOM_SEED
from ..utils.seed import set_global_seed
from .lora_config import create_lora_config
from .quantization import create_quantization_config

DEFAULT_MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
DEFAULT_TRAINING_CONFIG_PATH = CONFIG_DIR / "training_config.yaml"


def load_yaml_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration must contain a YAML mapping: {path}"
        )

    return config


def get_nested_value(
    config: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Read a value safely from a nested configuration mapping."""

    current: Any = config

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def get_base_model_id(
    model_config: dict[str, Any],
) -> str:
    """Return the configured Hugging Face base-model identifier."""

    model_id = get_nested_value(
        model_config,
        "model",
        "base_model_id",
    )

    if not isinstance(model_id, str):
        raise TypeError(
            "model.base_model_id must be a string."
        )

    model_id = model_id.strip()

    if not model_id:
        raise ValueError(
            "model_config.yaml must define model.base_model_id."
        )

    return model_id


def get_trust_remote_code(
    model_config: dict[str, Any],
) -> bool:
    """Return whether remote model implementation code is trusted."""

    return bool(
        get_nested_value(
            model_config,
            "model",
            "trust_remote_code",
            default=False,
        )
    )


def load_tokenizer(
    model_id: str,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
):
    """Load and configure the tokenizer used for Qwen fine-tuning."""

    if not isinstance(model_id, str):
        raise TypeError(
            "model_id must be a string."
        )

    model_id = model_id.strip()

    if not model_id:
        raise ValueError(
            "model_id cannot be empty."
        )

    model_config = load_yaml_config(
        model_config_path
    )

    trust_remote_code = get_trust_remote_code(
        model_config
    )

    use_fast = bool(
        get_nested_value(
            model_config,
            "tokenizer",
            "use_fast",
            default=True,
        )
    )

    padding_side = str(
        get_nested_value(
            model_config,
            "tokenizer",
            "padding_side",
            default="right",
        )
    )

    if padding_side not in {
        "left",
        "right",
    }:
        raise ValueError(
            "tokenizer.padding_side must be 'left' or 'right'."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = padding_side

    return tokenizer


def load_quantized_model(
    model_id: str,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
):
    """Load the causal language model using the configured 4-bit setup."""

    if not isinstance(model_id, str):
        raise TypeError(
            "model_id must be a string."
        )

    model_id = model_id.strip()

    if not model_id:
        raise ValueError(
            "model_id cannot be empty."
        )

    model_config = load_yaml_config(
        model_config_path
    )

    trust_remote_code = get_trust_remote_code(
        model_config
    )

    quantization_config = (
        create_quantization_config(
            model_config_path
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_id,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=(
                trust_remote_code
            ),
        )
    )

    model.config.use_cache = False

    return model


def prepare_lora_model(
    model,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
):
    """Prepare the quantized model and attach trainable LoRA adapters."""

    model = prepare_model_for_kbit_training(
        model
    )

    lora_config = create_lora_config(
        model_config_path
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    for parameter in model.parameters():
        if (
            parameter.requires_grad
            and parameter.dtype != torch.float32
        ):
            parameter.data = parameter.data.to(
                torch.float32
            )

    return model


def format_conversation(
    example: Mapping[str, Any],
    tokenizer,
) -> str:
    """Render one messages example using the model chat template."""

    if not isinstance(example, Mapping):
        raise TypeError(
            "example must be a mapping."
        )

    messages = example.get(
        "messages"
    )

    if not isinstance(messages, list):
        raise TypeError(
            "Training example messages must be a list."
        )

    if not messages:
        raise ValueError(
            "Training example must contain "
            "a non-empty messages list."
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_training_arguments(
    training_config: dict[str, Any],
    output_dir: str | Path,
    training_rows: int,
) -> SFTConfig:
    """Create TRL supervised fine-tuning arguments from YAML settings."""

    if not isinstance(training_config, dict):
        raise TypeError(
            "training_config must be a dictionary."
        )

    if not isinstance(training_rows, int):
        raise TypeError(
            "training_rows must be an integer."
        )

    if training_rows <= 0:
        raise ValueError(
            "training_rows must be greater than 0."
        )

    epochs = float(
        get_nested_value(
            training_config,
            "training",
            "num_train_epochs",
            default=3,
        )
    )

    per_device_batch_size = int(
        get_nested_value(
            training_config,
            "training",
            "per_device_train_batch_size",
            default=4,
        )
    )

    gradient_accumulation_steps = int(
        get_nested_value(
            training_config,
            "training",
            "gradient_accumulation_steps",
            default=2,
        )
    )

    learning_rate = float(
        get_nested_value(
            training_config,
            "training",
            "learning_rate",
            default=2e-4,
        )
    )

    weight_decay = float(
        get_nested_value(
            training_config,
            "training",
            "weight_decay",
            default=0.01,
        )
    )

    warmup_ratio = float(
        get_nested_value(
            training_config,
            "training",
            "warmup_ratio",
            default=0.03,
        )
    )

    lr_scheduler_type = str(
        get_nested_value(
            training_config,
            "training",
            "lr_scheduler_type",
            default="cosine",
        )
    )

    max_grad_norm = float(
        get_nested_value(
            training_config,
            "training",
            "max_grad_norm",
            default=0.3,
        )
    )

    max_length = int(
        get_nested_value(
            training_config,
            "sequence",
            "max_length",
            default=512,
        )
    )

    fp16 = bool(
        get_nested_value(
            training_config,
            "precision",
            "fp16",
            default=True,
        )
    )

    bf16 = bool(
        get_nested_value(
            training_config,
            "precision",
            "bf16",
            default=False,
        )
    )

    gradient_checkpointing = bool(
        get_nested_value(
            training_config,
            "memory",
            "gradient_checkpointing",
            default=True,
        )
    )

    optimizer_name = str(
        get_nested_value(
            training_config,
            "optimizer",
            "name",
            default="paged_adamw_8bit",
        )
    )

    logging_strategy = str(
        get_nested_value(
            training_config,
            "logging",
            "logging_strategy",
            default="steps",
        )
    )

    logging_steps = int(
        get_nested_value(
            training_config,
            "logging",
            "logging_steps",
            default=10,
        )
    )

    report_to = str(
        get_nested_value(
            training_config,
            "logging",
            "report_to",
            default="none",
        )
    )

    save_strategy = str(
        get_nested_value(
            training_config,
            "saving",
            "save_strategy",
            default="epoch",
        )
    )

    save_total_limit = int(
        get_nested_value(
            training_config,
            "saving",
            "save_total_limit",
            default=3,
        )
    )

    packing = bool(
        get_nested_value(
            training_config,
            "dataset",
            "packing",
            default=False,
        )
    )

    remove_unused_columns = bool(
        get_nested_value(
            training_config,
            "dataset",
            "remove_unused_columns",
            default=True,
        )
    )

    if epochs <= 0:
        raise ValueError(
            "num_train_epochs must be greater than 0."
        )

    if per_device_batch_size <= 0:
        raise ValueError(
            "per_device_train_batch_size must be greater than 0."
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be greater than 0."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(
            "warmup_ratio must be between 0 and 1."
        )

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than 0."
        )

    if fp16 and bf16:
        raise ValueError(
            "fp16 and bf16 cannot both be enabled."
        )

    effective_batch_size = (
        per_device_batch_size
        * gradient_accumulation_steps
    )

    steps_per_epoch = (
        training_rows
        + effective_batch_size
        - 1
    ) // effective_batch_size

    total_training_steps = int(
        steps_per_epoch
        * epochs
    )

    warmup_steps = round(
    total_training_steps
    * warmup_ratio
)

    return SFTConfig(
        output_dir=str(
            Path(output_dir)
            .expanduser()
            .resolve()
        ),
        num_train_epochs=epochs,
        per_device_train_batch_size=(
            per_device_batch_size
        ),
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=(
            lr_scheduler_type
        ),
        max_grad_norm=max_grad_norm,
        max_length=max_length,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=(
            gradient_checkpointing
        ),
        optim=optimizer_name,
        logging_strategy=(
            logging_strategy
        ),
        logging_steps=logging_steps,
        report_to=report_to,
        save_strategy=save_strategy,
        save_total_limit=(
            save_total_limit
        ),
        packing=packing,
        remove_unused_columns=(
            remove_unused_columns
        ),
        dataset_text_field="text",
        seed=RANDOM_SEED,
    )


def prepare_training_dataset(
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    """Convert fraud records to conversations and render chat-template text."""

    if not isinstance(dataset, Dataset):
        raise TypeError(
            "Training data must be a Hugging Face Dataset."
        )

    if len(dataset) == 0:
        raise ValueError(
            "Training dataset cannot be empty."
        )

    if "messages" not in dataset.column_names:
        dataset = (
            convert_dataset_to_conversations(
                dataset
            )
        )

    def render_example(
        example: Mapping[str, Any],
    ) -> dict[str, str]:
        return {
            "text": format_conversation(
                example,
                tokenizer,
            )
        }

    return dataset.map(
        render_example,
        desc=(
            "Rendering Qwen "
            "training conversations"
        ),
    )


def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    training_arguments: SFTConfig,
) -> SFTTrainer:
    """Create the TRL supervised fine-tuning trainer."""

    if not isinstance(
        train_dataset,
        Dataset,
    ):
        raise TypeError(
            "train_dataset must be a "
            "Hugging Face Dataset."
        )

    if not isinstance(
        training_arguments,
        SFTConfig,
    ):
        raise TypeError(
            "training_arguments must be "
            "an SFTConfig instance."
        )

    return SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )


def save_training_metadata(
    output_dir: str | Path,
    *,
    model_id: str,
    training_rows: int,
    model_config_path: str | Path,
    training_config_path: str | Path,
) -> Path:
    """Save reproducibility metadata alongside the trained adapter."""

    if not isinstance(training_rows, int):
        raise TypeError(
            "training_rows must be an integer."
        )

    if training_rows <= 0:
        raise ValueError(
            "training_rows must be greater than 0."
        )

    destination = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "base_model": model_id,
        "training_rows": training_rows,
        "random_seed": RANDOM_SEED,
        "training_method": "QLoRA",
        "task": (
            "financial_fraud_"
            "risk_classification"
        ),
        "model_config": str(
            Path(model_config_path)
            .expanduser()
            .resolve()
        ),
        "training_config": str(
            Path(training_config_path)
            .expanduser()
            .resolve()
        ),
    }

    metadata_path = (
        destination
        / "training_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return metadata_path


def train_model(
    train_dataset: Dataset,
    output_dir: str | Path,
    *,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    training_config_path: str | Path = DEFAULT_TRAINING_CONFIG_PATH,
):
    """
    Execute the complete QLoRA supervised fine-tuning workflow.

    The caller supplies the already prepared training partition.
    Data splitting and balancing remain separate from model training,
    preventing accidental train/test leakage.
    """

    if not isinstance(
        train_dataset,
        Dataset,
    ):
        raise TypeError(
            "train_dataset must be a "
            "Hugging Face Dataset."
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "train_dataset cannot be empty."
        )

    set_global_seed(
        RANDOM_SEED
    )

    model_config = load_yaml_config(
        model_config_path
    )

    training_config = load_yaml_config(
        training_config_path
    )

    model_id = get_base_model_id(
        model_config
    )

    tokenizer = load_tokenizer(
        model_id,
        model_config_path,
    )

    prepared_dataset = (
        prepare_training_dataset(
            train_dataset,
            tokenizer,
        )
    )

    model = load_quantized_model(
        model_id,
        model_config_path,
    )

    model = prepare_lora_model(
        model,
        model_config_path,
    )

    training_arguments = (
        build_training_arguments(
            training_config,
            output_dir,
            training_rows=len(
                prepared_dataset
            ),
        )
    )

    trainer = create_trainer(
        model,
        tokenizer,
        prepared_dataset,
        training_arguments,
    )

    trainer.train()

    destination = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(destination)
    )

    tokenizer.save_pretrained(
        str(destination)
    )

    save_training_metadata(
        destination,
        model_id=model_id,
        training_rows=len(
            prepared_dataset
        ),
        model_config_path=(
            model_config_path
        ),
        training_config_path=(
            training_config_path
        ),
    )

    return trainer


def load_adapter_for_inference(
    adapter_path: str | Path,
    *,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
):
    """Load a saved LoRA adapter on the configured quantized base model."""

    adapter_directory = (
        Path(adapter_path)
        .expanduser()
        .resolve()
    )

    if not adapter_directory.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: "
            f"{adapter_directory}"
        )

    model_config = load_yaml_config(
        model_config_path
    )

    model_id = get_base_model_id(
        model_config
    )

    tokenizer = load_tokenizer(
        model_id,
        model_config_path,
    )

    base_model = load_quantized_model(
        model_id,
        model_config_path,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_directory),
    )

    model.eval()

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training environments."""

    parser = argparse.ArgumentParser(
        description=(
            "QLoRA training utilities for the "
            "Financial Fraud Detection System."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/"
            "fraud-qlora-adapter"
        ),
        help=(
            "Directory where trained "
            "adapter artifacts are stored."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Display command-line guidance.

    Dataset construction remains explicit because splitting and balancing
    must happen before training. The GPU notebook calls train_model() using
    the already prepared leakage-safe training partition.
    """

    args = parse_args()

    print(
        "QLoRA training engine "
        "configured successfully."
    )

    print(
        "Adapter output directory: "
        f"{args.output_dir}"
    )

    print(
        "Use train_model(train_dataset, output_dir) "
        "with the prepared leakage-safe "
        "training partition."
    )


if __name__ == "__main__":
    main()