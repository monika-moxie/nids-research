"""Minimal local audit storage for human review in the Phase 8 demonstrator."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_PATH = Path("outputs/deployment-intelligence/analyst_audit.sqlite3")


class AnalystAuditStore:
    """Persist report metadata and analyst feedback without raw flow features."""

    def __init__(self, database_path: Path = DEFAULT_AUDIT_PATH) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyst_audit (
                    report_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    attack_probability REAL NOT NULL,
                    certified_radius REAL NOT NULL,
                    analyst_verdict TEXT,
                    analyst_note TEXT,
                    feedback_at_utc TEXT
                )
                """
            )

    def record_intelligence(self, result: dict[str, Any]) -> None:
        """Store traceable model metadata for a new incident draft."""

        report = result["incident_report"]
        certification = result["certification"]
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO analyst_audit (
                    report_id, created_at_utc, recommended_action, severity,
                    attack_probability, certified_radius
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report["report_id"],
                    report["created_at_utc"],
                    report["recommended_action"],
                    report["severity"],
                    result["attack_probability"],
                    certification["certified_radius"],
                ),
            )

    def record_feedback(self, report_id: str, verdict: str, analyst_note: str, submitted_at_utc: str) -> dict[str, Any]:
        """Attach one analyst verdict to an existing locally generated report."""

        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analyst_audit
                SET analyst_verdict = ?, analyst_note = ?, feedback_at_utc = ?
                WHERE report_id = ?
                """,
                (verdict, analyst_note, submitted_at_utc, report_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(report_id)

        return {
            "report_id": report_id,
            "analyst_verdict": verdict,
            "analyst_note": analyst_note,
            "feedback_at_utc": submitted_at_utc,
        }

    def list_records(self, limit: int) -> list[dict[str, Any]]:
        """Return recent audit metadata for local review, newest first."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT report_id, created_at_utc, recommended_action, severity,
                       attack_probability, certified_radius, analyst_verdict,
                       analyst_note, feedback_at_utc
                FROM analyst_audit
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection
