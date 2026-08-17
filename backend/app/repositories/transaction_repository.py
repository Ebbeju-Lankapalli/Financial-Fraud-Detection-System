"""
SQLite repository for transaction-analysis audit history.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from app.core.config import settings
from app.schemas.prediction import (
    FraudPrediction,
    TransactionAuditRecord,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)


class TransactionRepositoryError(
    RuntimeError
):
    """Raised when transaction persistence fails."""


def sqlite_path_from_url(
    database_url: str,
) -> Path:
    """Convert a sqlite:/// URL into a filesystem path."""

    prefix = "sqlite:///"

    if not database_url.startswith(prefix):
        raise ValueError(
            "Only sqlite:/// database URLs are supported."
        )

    raw_path = database_url[
        len(prefix):
    ]

    if not raw_path:
        raise ValueError(
            "SQLite database path cannot be empty."
        )

    return Path(raw_path).expanduser()


class TransactionRepository:
    """Persist and retrieve fraud-analysis audit records."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
    ) -> None:
        self.database_url = (
            database_url
            or settings.database_url
        )

        self.database_path = sqlite_path_from_url(
            self.database_url
        )

        self._lock = Lock()

        self._initialize_database()

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """Create a configured SQLite connection."""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize_database(
        self,
    ) -> None:
        """Create the audit table when necessary."""

        query = """
        CREATE TABLE IF NOT EXISTS transaction_analyses (
            analysis_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,

            type TEXT NOT NULL,
            amount REAL NOT NULL,
            oldbalanceOrg REAL NOT NULL,
            newbalanceOrig REAL NOT NULL,
            oldbalanceDest REAL NOT NULL,
            newbalanceDest REAL NOT NULL,

            risk TEXT NOT NULL,
            model TEXT NOT NULL,
            adapter TEXT NOT NULL,
            decision_source TEXT NOT NULL,
            raw_output TEXT NOT NULL,
            valid_output INTEGER NOT NULL
        )
        """

        try:
            with self._connect() as connection:
                connection.execute(
                    query
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise TransactionRepositoryError(
                "Failed to initialize "
                "transaction database."
            ) from exc

    def save(
        self,
        *,
        analysis_id: str,
        created_at: str,
        transaction: TransactionAnalysisRequest,
        prediction: FraudPrediction,
    ) -> TransactionAuditRecord:
        """Persist one analyzed transaction."""

        query = """
        INSERT INTO transaction_analyses (
            analysis_id,
            created_at,
            type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
            risk,
            model,
            adapter,
            decision_source,
            raw_output,
            valid_output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            analysis_id,
            created_at,
            transaction.type,
            transaction.amount,
            transaction.oldbalanceOrg,
            transaction.newbalanceOrig,
            transaction.oldbalanceDest,
            transaction.newbalanceDest,
            prediction.risk,
            prediction.model,
            prediction.adapter,
            prediction.decision_source,
            prediction.raw_output,
            int(
                prediction.valid_output
            ),
        )

        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    query,
                    values,
                )

                connection.commit()

        except sqlite3.Error as exc:
            raise TransactionRepositoryError(
                "Failed to save "
                "transaction analysis."
            ) from exc

        return TransactionAuditRecord(
            analysis_id=analysis_id,
            created_at=created_at,
            **transaction.model_dump(),
            **prediction.model_dump(),
        )

    def get_by_id(
        self,
        analysis_id: str,
    ) -> TransactionAuditRecord | None:
        """Retrieve one audit record by analysis ID."""

        query = """
        SELECT *
        FROM transaction_analyses
        WHERE analysis_id = ?
        """

        try:
            with self._connect() as connection:
                row = connection.execute(
                    query,
                    (analysis_id,),
                ).fetchone()

        except sqlite3.Error as exc:
            raise TransactionRepositoryError(
                "Failed to retrieve "
                "transaction analysis."
            ) from exc

        if row is None:
            return None

        return self._row_to_record(
            row
        )

    def list_recent(
        self,
        *,
        limit: int = 50,
    ) -> list[TransactionAuditRecord]:
        """Return recent transaction analyses."""

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        if limit > 500:
            raise ValueError(
                "limit cannot exceed 500."
            )

        query = """
        SELECT *
        FROM transaction_analyses
        ORDER BY created_at DESC
        LIMIT ?
        """

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    query,
                    (limit,),
                ).fetchall()

        except sqlite3.Error as exc:
            raise TransactionRepositoryError(
                "Failed to list "
                "transaction analyses."
            ) from exc

        return [
            self._row_to_record(
                row
            )
            for row in rows
        ]

    def count(
        self,
    ) -> int:
        """Return total number of stored analyses."""

        query = """
        SELECT COUNT(*) AS total
        FROM transaction_analyses
        """

        try:
            with self._connect() as connection:
                row = connection.execute(
                    query
                ).fetchone()

        except sqlite3.Error as exc:
            raise TransactionRepositoryError(
                "Failed to count "
                "transaction analyses."
            ) from exc

        if row is None:
            return 0

        return int(
            row["total"]
        )

    def get_dashboard_metrics(
        self,
    ) -> dict[str, int]:
        """Return aggregate metrics across all stored analyses."""

        query = """
        SELECT
            COUNT(*) AS total,
            COALESCE(
                SUM(
                    CASE
                        WHEN risk = 'HIGH'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS high_count,
            COALESCE(
                SUM(
                    CASE
                        WHEN risk = 'LOW'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS low_count,
            COALESCE(
                SUM(
                    CASE
                        WHEN valid_output = 1
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS valid_count,
            COALESCE(
                SUM(
                    CASE
                        WHEN valid_output = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS invalid_count
        FROM transaction_analyses
        """

        try:
            with self._connect() as connection:
                row = connection.execute(
                    query
                ).fetchone()

        except sqlite3.Error as exc:
            raise TransactionRepositoryError(
                "Failed to calculate "
                "dashboard metrics."
            ) from exc

        if row is None:
            return {
                "total": 0,
                "high_count": 0,
                "low_count": 0,
                "valid_count": 0,
                "invalid_count": 0,
            }

        return {
            "total": int(
                row["total"]
            ),
            "high_count": int(
                row["high_count"]
            ),
            "low_count": int(
                row["low_count"]
            ),
            "valid_count": int(
                row["valid_count"]
            ),
            "invalid_count": int(
                row["invalid_count"]
            ),
        }


    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> TransactionAuditRecord:
        """Convert a SQLite row to the API schema."""

        return TransactionAuditRecord(
            analysis_id=row["analysis_id"],
            created_at=row["created_at"],
            type=row["type"],
            amount=row["amount"],
            oldbalanceOrg=row["oldbalanceOrg"],
            newbalanceOrig=row["newbalanceOrig"],
            oldbalanceDest=row["oldbalanceDest"],
            newbalanceDest=row["newbalanceDest"],
            risk=row["risk"],
            model=row["model"],
            adapter=row["adapter"],
            decision_source=row[
                "decision_source"
            ],
            raw_output=row["raw_output"],
            valid_output=bool(
                row["valid_output"]
            ),
        )
