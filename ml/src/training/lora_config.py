"""
LoRA configuration for fraud-model QLoRA fine-tuning.

The base Qwen model remains frozen while small trainable LoRA adapter
matrices are attached to selected attention projection layers.

Only the adapter parameters are updated during supervised fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig

from ..utils.constants import CONFIG_DIR

DEFAULT_MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"


def load_lora_model_config(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load model_config.yaml for LoRA configuration.

    Args:
        config_path:
            Path to the model configuration file.

    Returns:
        Parsed YAML mapping.

    Raises:
        FileNotFoundError:
            If the configuration file does not exist.

        ValueError:
            If the configuration file does not contain a mapping.
    """

    path = Path(config_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            "Model configuration must contain a YAML mapping."
        )

    return config


def create_lora_config(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> LoraConfig:
    """
    Create the PEFT LoRA configuration used by QLoRA training.

    The active configuration is read from model_config.yaml.

    Returns:
        Configured LoraConfig instance.

    Raises:
        ValueError:
            If required LoRA/model settings are missing or invalid.
    """

    config = load_lora_model_config(
        config_path
    )

    model_section = config.get(
        "model"
    )

    lora_section = config.get(
        "lora"
    )

    if not isinstance(
        model_section,
        dict,
    ):
        raise TypeError(
            "Model configuration must contain "
            "a 'model' mapping."
        )

    if not isinstance(
        lora_section,
        dict,
    ):
        raise TypeError(
            "Model configuration must contain "
            "a 'lora' mapping."
        )

    rank = int(
        lora_section["rank"]
    )

    alpha = int(
        lora_section["alpha"]
    )

    dropout = float(
        lora_section["dropout"]
    )

    target_modules = list(
        lora_section[
            "target_modules"
        ]
    )

    if rank <= 0:
        raise ValueError(
            "LoRA rank must be greater than 0."
        )

    if alpha <= 0:
        raise ValueError(
            "LoRA alpha must be greater than 0."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "LoRA dropout must be "
            "between 0 and 1."
        )

    if not target_modules:
        raise ValueError(
            "At least one LoRA target module "
            "must be configured."
        )

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=(
            target_modules
        ),
        lora_dropout=dropout,
        bias=str(
            lora_section.get(
                "bias",
                "none",
            )
        ),
        task_type=str(
            model_section.get(
                "task_type",
                "CAUSAL_LM",
            )
        ),
    )


def get_lora_summary(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> dict[str, object]:
    """
    Return a serializable summary of LoRA settings.

    Args:
        config_path:
            Path to model_config.yaml.

    Returns:
        Dictionary describing the adapter configuration.
    """

    config = load_lora_model_config(
        config_path
    )

    lora = config[
        "lora"
    ]

    return {
        "rank": int(
            lora["rank"]
        ),
        "alpha": int(
            lora["alpha"]
        ),
        "dropout": float(
            lora["dropout"]
        ),
        "bias": str(
            lora["bias"]
        ),
        "target_modules": list(
            lora["target_modules"]
        ),
    }
