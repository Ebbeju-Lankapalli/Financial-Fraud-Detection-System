"""
Dashboard API for fraud-analysis statistics.
"""

from __future__ import annotations

from app.repositories.transaction_repository import (
    TransactionRepository,
    TransactionRepositoryError,
)
from app.schemas.prediction import (
    TransactionAuditRecord,
)
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)
from pydantic import (
    BaseModel,
    ConfigDict,
)

router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
)

transaction_repository = TransactionRepository()


class DashboardMetrics(BaseModel):
    """Aggregate fraud-analysis metrics."""

    model_config = ConfigDict(
        extra="forbid",
    )

    total_analyses: int
    high_risk_count: int
    low_risk_count: int
    high_risk_percentage: float
    valid_output_count: int
    invalid_output_count: int


class DashboardResponse(BaseModel):
    """Dashboard summary returned to the frontend."""

    model_config = ConfigDict(
        extra="forbid",
    )

    metrics: DashboardMetrics
    recent_analyses: list[
        TransactionAuditRecord
    ]


@router.get(
    "",
    response_model=DashboardResponse,
)
def dashboard(
    recent_limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
) -> DashboardResponse:
    """Return aggregate metrics and recent analyses."""

    try:
        aggregates = (
            transaction_repository
            .get_dashboard_metrics()
        )

        recent = (
            transaction_repository
            .list_recent(
                limit=recent_limit
            )
        )

    except TransactionRepositoryError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    total = aggregates[
        "total"
    ]

    high_count = aggregates[
        "high_count"
    ]

    high_percentage = (
        round(
            high_count / total * 100,
            2,
        )
        if total
        else 0.0
    )

    return DashboardResponse(
        metrics=DashboardMetrics(
            total_analyses=total,
            high_risk_count=high_count,
            low_risk_count=aggregates[
                "low_count"
            ],
            high_risk_percentage=(
                high_percentage
            ),
            valid_output_count=aggregates[
                "valid_count"
            ],
            invalid_output_count=aggregates[
                "invalid_count"
            ],
        ),
        recent_analyses=recent,
    )
