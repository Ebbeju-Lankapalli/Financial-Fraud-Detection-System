"""
Quantization configuration for QLoRA fraud-model fine-tuning.

The project loads Qwen2.5-1.5B-Instruct in 4-bit precision using
bitsandbytes NF4 quantization.

The configuration is intentionally created in a reusable function so the
same settings are shared by training notebooks and Python training scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import BitsAndBytesConfig

from ..utils.constants import CONFIG_DIR

DEFAULT_MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"


def load_model_config(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load the YAML model configuration.

    Args:
        config_path:
            Path to model_config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError:
            If the configuration file does not exist.

        ValueError:
            If the configuration file is empty or malformed.
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


def resolve_torch_dtype(
    dtype_name: str,
) -> torch.dtype:
    """
    Convert a configured dtype name into a PyTorch dtype.

    Args:
        dtype_name:
            Supported textual dtype name.

    Returns:
        Corresponding torch.dtype.

    Raises:
        TypeError:
            If dtype_name is not a string.

        ValueError:
            If the dtype is unsupported.
    """

    if not isinstance(dtype_name, str):
        raise TypeError(
            "dtype_name must be a string."
        )

    normalized_dtype = (
        dtype_name
        .strip()
        .lower()
    )

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if normalized_dtype not in dtype_map:
        supported = ", ".join(
            sorted(dtype_map)
        )

        raise ValueError(
            f"Unsupported compute dtype "
            f"'{dtype_name}'. "
            f"Supported values: {supported}."
        )

    return dtype_map[normalized_dtype]


def create_quantization_config(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> BitsAndBytesConfig:
    """
    Create the project's 4-bit BitsAndBytesConfig.

    The configuration is read from model_config.yaml and currently uses:

    - 4-bit model loading
    - NF4 quantization
    - nested/double quantization
    - float16 computation for NVIDIA T4 training

    Args:
        config_path:
            Path to the YAML model configuration.

    Returns:
        Configured BitsAndBytesConfig instance.

    Raises:
        ValueError:
            If the quantization section is missing or disabled.
    """

    config = load_model_config(
        config_path
    )

    quantization = config.get(
        "quantization"
    )

    if not isinstance(
        quantization,
        dict,
    ):
        raise TypeError(
            "Model configuration must contain "
            "a 'quantization' mapping."
        )

    if not quantization.get(
        "enabled",
        False,
    ):
        raise ValueError(
            "4-bit quantization is disabled "
            "in model_config.yaml."
        )

    compute_dtype = resolve_torch_dtype(
        quantization.get(
            "compute_dtype",
            "float16",
        )
    )

    return BitsAndBytesConfig(
        load_in_4bit=bool(
            quantization.get(
                "load_in_4bit",
                True,
            )
        ),
        bnb_4bit_quant_type=str(
            quantization.get(
                "quant_type",
                "nf4",
            )
        ),
        bnb_4bit_use_double_quant=bool(
            quantization.get(
                "use_double_quant",
                True,
            )
        ),
        bnb_4bit_compute_dtype=(
            compute_dtype
        ),
    )


def get_quantization_summary(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> dict[str, object]:
    """
    Return a serializable summary of quantization settings.

    Args:
        config_path:
            Path to model_config.yaml.

    Returns:
        Dictionary describing the active quantization settings.
    """

    config = load_model_config(
        config_path
    )

    quantization = config[
        "quantization"
    ]

    return {
        "enabled": bool(
            quantization["enabled"]
        ),
        "load_in_4bit": bool(
            quantization[
                "load_in_4bit"
            ]
        ),
        "quant_type": str(
            quantization[
                "quant_type"
            ]
        ),
        "use_double_quant": bool(
            quantization[
                "use_double_quant"
            ]
        ),
        "compute_dtype": str(
            quantization[
                "compute_dtype"
            ]
        ),
    }
