"""
SQLite repository for production fraud-analysis audit history.

The current audit table stores the complete IEEE-CIS transaction payload
as JSON together with the CatBoost prediction metadata.

Legacy QLoRA/PaySim audit tables are preserved automatically when detected.
"""

from __future__ import annotations

import json
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

    if not database_url.startswith(
        prefix
    ):
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

    return Path(
        raw_path
    ).expanduser()


class TransactionRepository:
    """Persist and retrieve production fraud-analysis records."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
    ) -> None:
        self.database_url = (
            database_url
            or settings.database_url
        )

        self.database_path = (
            sqlite_path_from_url(
                self.database_url
            )
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

    @staticmethod
    def _table_exists(
        connection: sqlite3.Connection,
        table_name: str,
    ) -> bool:
        """Return whether a SQLite table exists."""

        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (
                table_name,
            ),
        ).fetchone()

        return row is not None

    @staticmethod
    def _table_columns(
        connection: sqlite3.Connection,
        table_name: str,
    ) -> set[str]:
        """Return column names for one SQLite table."""

        rows = connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        return {
            str(
                row["name"]
            )
            for row in rows
        }

    @classmethod
    def _next_legacy_table_name(
        cls,
        connection: sqlite3.Connection,
    ) -> str:
        """Return an unused legacy audit-table name."""

        base = (
            "transaction_analyses_"
            "qlora_legacy"
        )

        if not cls._table_exists(
            connection,
            base,
        ):
            return base

        index = 2

        while cls._table_exists(
            connection,
            f"{base}_{index}",
        ):
            index += 1

        return f"{base}_{index}"

    def _migrate_legacy_table(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """Preserve an existing QLoRA audit table before v2 creation."""

        table_name = (
            "transaction_analyses"
        )

        if not self._table_exists(
            connection,
            table_name,
        ):
            return

        columns = self._table_columns(
            connection,
            table_name,
        )

        current_columns = {
            "analysis_id",
            "created_at",
            "transaction_json",
            "risk",
            "fraud_probability",
            "threshold",
            "model",
            "feature_count",
            "decision_source",
            "valid_output",
        }

        if current_columns.issubset(
            columns
        ):
            return

        legacy_name = (
            self._next_legacy_table_name(
                connection
            )
        )

        connection.execute(
            f"""
            ALTER TABLE
                transaction_analyses
            RENAME TO
                {legacy_name}
            """
        )

    def _initialize_database(
        self,
    ) -> None:
        """Initialize the current audit table safely."""

        create_query = """
        CREATE TABLE IF NOT EXISTS transaction_analyses (
            analysis_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,

            transaction_json TEXT NOT NULL,

            risk TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            threshold REAL NOT NULL,
            model TEXT NOT NULL,
            feature_count INTEGER NOT NULL,
            decision_source TEXT NOT NULL,
            valid_output INTEGER NOT NULL
        )
        """

        try:
            with self._connect() as connection:
                self._migrate_legacy_table(
                    connection
                )

                connection.execute(
                    create_query
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
        """Persist one production fraud analysis."""

        transaction_json = json.dumps(
            transaction.model_dump(),
            separators=(
                ",",
                ":",
            ),
        )

        query = """
        INSERT INTO transaction_analyses (
            analysis_id,
            created_at,
            transaction_json,
            risk,
            fraud_probability,
            threshold,
            model,
            feature_count,
            decision_source,
            valid_output
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            analysis_id,
            created_at,
            transaction_json,
            prediction.risk,
            prediction.fraud_probability,
            prediction.threshold,
            prediction.model,
            prediction.feature_count,
            prediction.decision_source,
            int(
                prediction.valid_output
            ),
        )

        try:
            with (
                self._lock,
                self._connect() as connection,
            ):
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
        """Retrieve one production audit record."""

        query = """
        SELECT *
        FROM transaction_analyses
        WHERE analysis_id = ?
        """

        try:
            with self._connect() as connection:
                row = connection.execute(
                    query,
                    (
                        analysis_id,
                    ),
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
        """Return recent production analyses."""

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
                    (
                        limit,
                    ),
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
        """Return total current production analyses."""

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
        """Return aggregate production fraud metrics."""

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
        """Convert one current SQLite row into the API schema."""

        try:
            transaction = json.loads(
                row[
                    "transaction_json"
                ]
            )

        except (
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise TransactionRepositoryError(
                "Stored transaction payload "
                "is invalid."
            ) from exc

        return TransactionAuditRecord(
            analysis_id=row[
                "analysis_id"
            ],
            created_at=row[
                "created_at"
            ],
            **transaction,
            risk=row[
                "risk"
            ],
            fraud_probability=float(
                row[
                    "fraud_probability"
                ]
            ),
            threshold=float(
                row[
                    "threshold"
                ]
            ),
            model=row[
                "model"
            ],
            feature_count=int(
                row[
                    "feature_count"
                ]
            ),
            decision_source=row[
                "decision_source"
            ],
            valid_output=bool(
                row[
                    "valid_output"
                ]
            ),
        )
