"""
Backend application configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the backend API."""

    app_name: str = os.getenv(
        "APP_NAME",
        "Financial Fraud Detection System",
    )

    model_service_url: str = os.getenv(
        "MODEL_SERVICE_URL",
        "http://127.0.0.1:8001",
    )

    model_service_timeout_seconds: float = float(
        os.getenv(
            "MODEL_SERVICE_TIMEOUT_SECONDS",
            "120",
        )
    )

    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./app.db",
    )

    groq_api_key: str | None = os.getenv(
        "GROQ_API_KEY"
    )

    groq_model: str = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )

    groq_timeout_seconds: float = float(
        os.getenv(
            "GROQ_TIMEOUT_SECONDS",
            "60",
        )
    )


settings = Settings()
