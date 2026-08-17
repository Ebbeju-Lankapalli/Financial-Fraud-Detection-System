"""
Main FastAPI application for the Financial Fraud Detection System.
"""

from __future__ import annotations

from app.api.agent import (
    router as agent_router,
)
from app.api.dashboard import (
    router as dashboard_router,
)
from app.api.evaluation import (
    router as evaluation_router,
)
from app.api.transactions import (
    router as transactions_router,
)
from app.core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        settings.cors_origins
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    transactions_router
)

app.include_router(
    agent_router
)

app.include_router(
    dashboard_router
)

app.include_router(
    evaluation_router
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a simple API identity response."""

    return {
        "message": (
            "Financial Fraud Detection System API"
        )
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return backend health status."""

    return {
        "status": "ok",
    }
