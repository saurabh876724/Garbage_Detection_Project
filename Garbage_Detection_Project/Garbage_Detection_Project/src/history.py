"""SQLite history of detection events (data/detection_history.db).

Stores one row per logged scene: when, from which source, the derived status,
per-class counts, average confidence, evidence path and the model version that
produced it. Older databases are migrated in place - no rows are dropped.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import config
from src.logging_setup import get_logger

log = get_logger("history")

TABLE = "detection_events"
SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,
    source             TEXT NOT NULL,
    source_type        TEXT NOT NULL DEFAULT 'unknown',
    status             TEXT NOT NULL,
    total_count        INTEGER NOT NULL,
    counts_json        TEXT NOT NULL,
    average_confidence REAL NOT NULL DEFAULT 0,
    evidence_path      TEXT,
    model_version      TEXT,
    model_path         TEXT,
    unique_objects     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON detection_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_status
    ON detection_events (status);
"""
EXPORT_COLUMNS = ("id", "timestamp", "source", "source_type", "status",
                  "total_count", "average_confidence", "unique_objects",
                  "model_version", "evidence_path", "counts")


class HistoryError(RuntimeError):
    """Raised when the history database cannot be used."""


@dataclass
class Event:
    id: int
    timestamp: str
    source: str
    source_type: str
    status: str
    total_count: int
    counts: dict[str, int]
    average_confidence: float
    evidence_path: str | None
    model_version: str | None
    model_path: str | None
    unique_objects: int | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Event":
        try:
            counts = json.loads(row["counts_json"] or "{}")
        except json.JSONDecodeError:
            counts = {}
        return cls(
            id=int(row["id"]), timestamp=row["timestamp"],
            source=row["source"],
            source_type=row["source_type"] or "unknown",
            status=row["status"], total_count=int(row["total_count"]),
            counts=counts,
            average_confidence=float(row["average_confidence"] or 0.0),
            evidence_path=row["evidence_path"],
            model_version=row["model_version"],
            model_path=row["model_path"],
            unique_objects=(int(row["unique_objects"])
                            if row["unique_objects"] is not None else None),
        )

    def as_export_row(self) -> dict:
        row = {k: getattr(self, k) for k in EXPORT_COLUMNS if k != "counts"}
        row["counts"] = "; ".join(f"{k}:{v}" for k, v in self.counts.items()
                                  if v)
        return row


@dataclass
class Filters:
    """Optional narrowing used by query(), export_csv() and analytics."""

    text: str = ""
    source_type: str = ""
    status: str = ""
    date_from: date | None = None
    date_to: date | None = None
    limit: int = 500
    offset: int = 0

    def clause(self) -> tuple[str, list]:
        """Build the WHERE fragment; timestamps are ISO strings so date
        bounds can be compared lexically."""
        conditions, params = [], []
        if self.text:
            conditions.append("(source LIKE ? OR status LIKE ? "
                              "OR counts_json LIKE ?)")
            like = f"%{self.text}%"
            params.extend([like, like, like])
        if self.source_type:
            conditions.append("source_type = ?")
            params.append(self.source_type)
        if self.status:
            conditions.append("status = ?")
            params.append(self.status)
        if self.date_from:
            conditions.append("timestamp >= ?")
            params.append(self.date_from.isoformat())
        if self.date_to:
            conditions.append("timestamp < ?")
            params.append((self.date_to + timedelta(days=1)).isoformat())
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return where, params


class HistoryDB:
    """Thread-safe wrapper around the SQLite history database."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or config.HISTORY_DB)
        self._lock = threading.RLock()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path),
                                        check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            with self._lock:
                self.conn.executescript(SCHEMA)
                self._migrate()
                self.conn.commit()
        except sqlite3.Error as exc:
            log.exception("history database unavailable")
            raise HistoryError(
                f"The history database could not be opened:\n{exc}\n\n"
                f"Path: {self.db_path}\nCheck that the file is not locked by "
                "another running copy of the application.") from exc
        log.info("history database ready: %s", self.db_path)

    # -- schema -------------------------------------------------------------

    def _migrate(self) -> None:
        """Add columns introduced after v1 without touching existing rows."""
        existing = {row["name"] for row in
                    self.conn.execute(f"PRAGMA table_info({TABLE})")}
        defaults = {
            "source_type": "TEXT NOT NULL DEFAULT 'unknown'",
            "average_confidence": "REAL NOT NULL DEFAULT 0",
            "model_version": "TEXT",
            "model_path": "TEXT",
            "unique_objects": "INTEGER",
        }
        for column, definition in defaults.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE {TABLE} "
                                  f"ADD COLUMN {column} {definition}")
                log.info("history migrated: added column %s", column)

    # -- writes -------------------------------------------------------------

    def log_event(self, source: str, status: str, total_count: int,
                  counts: dict[str, int], evidence_path: str | None = None,
                  average_confidence: float = 0.0,
                  source_type: str = "unknown",
                  model_version: str = config.MODEL_VERSION,
                  model_path: str = "",
                  unique_objects: int | None = None,
                  when: datetime | None = None) -> int:
        """Insert one event and return its row id."""
        stamp = (when or datetime.now()).isoformat(timespec="seconds")
        with self._lock:
            cur = self.conn.execute(
                f"INSERT INTO {TABLE} (timestamp, source, source_type, status, "
                "total_count, counts_json, average_confidence, evidence_path, "
                "model_version, model_path, unique_objects) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (stamp, source, source_type, status, int(total_count),
                 json.dumps(counts or {}), float(average_confidence),
                 evidence_path, model_version, model_path, unique_objects))
            self.conn.commit()
            row_id = int(cur.lastrowid)
        log.info("history #%d: %s %s %d objects", row_id, source_type, status,
                 total_count)
        return row_id

    def prune(self, keep: int = config.MAX_HISTORY_ENTRIES) -> int:
        """Delete rows beyond the newest `keep`; returns how many were removed."""
        with self._lock:
            cur = self.conn.execute(
                f"DELETE FROM {TABLE} WHERE id NOT IN "
                f"(SELECT id FROM {TABLE} ORDER BY timestamp DESC, id DESC "
                f"LIMIT ?)", (int(keep),))
            self.conn.commit()
            removed = cur.rowcount
        if removed > 0:
            log.info("history pruned: %d old rows removed", removed)
        return removed

    # -- reads --------------------------------------------------------------

    def query(self, filters: Filters | None = None) -> list[Event]:
        filters = filters or Filters()
        where, params = filters.clause()
        sql = (f"SELECT * FROM {TABLE} {where} "
               "ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?")
        with self._lock:
            rows = self.conn.execute(
                sql, [*params, int(filters.limit), int(filters.offset)]
            ).fetchall()
        return [Event.from_row(row) for row in rows]

    def count(self, filters: Filters | None = None) -> int:
        filters = filters or Filters()
        where, params = filters.clause()
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {TABLE} {where}", params).fetchone()
        return int(row[0])

    def recent_events(self, limit: int = 50) -> list[Event]:
        return self.query(Filters(limit=limit))

    def event_count(self) -> int:
        return self.count()

    def status_totals(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT status, COUNT(*) AS n FROM {TABLE} "
                "GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def class_totals(self) -> dict[str, int]:
        totals = {name: 0 for name in config.GARBAGE_CLASSES}
        with self._lock:
            rows = self.conn.execute(
                f"SELECT counts_json FROM {TABLE}").fetchall()
        for row in rows:
            try:
                counts = json.loads(row["counts_json"] or "{}")
            except json.JSONDecodeError:
                continue
            for label, value in counts.items():
                if label in totals:
                    totals[label] += int(value)
        return totals

    def average_confidence(self) -> float:
        with self._lock:
            row = self.conn.execute(
                f"SELECT AVG(average_confidence) FROM {TABLE} "
                "WHERE total_count > 0").fetchone()
        return float(row[0] or 0.0)

    def events_on(self, day: date) -> list[Event]:
        return self.query(Filters(date_from=day, date_to=day, limit=10_000))

    def daily_counts(self, days: int = 14) -> list[tuple[str, dict[str, int]]]:
        """Events per day for the last N days: [(iso_date, {status: n}), ...]."""
        today = date.today()
        start = today - timedelta(days=days - 1)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT substr(timestamp, 1, 10) AS day, status, "
                f"COUNT(*) AS n FROM {TABLE} "
                "WHERE substr(timestamp, 1, 10) >= ? GROUP BY day, status",
                (start.isoformat(),)).fetchall()
        per_day: dict[str, dict[str, int]] = {}
        for row in rows:
            per_day.setdefault(row["day"], {})[row["status"]] = int(row["n"])
        return [((start + timedelta(days=i)).isoformat(),
                 per_day.get((start + timedelta(days=i)).isoformat(), {}))
                for i in range(days)]

    def source_totals(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT source_type, COUNT(*) AS n FROM {TABLE} "
                "GROUP BY source_type").fetchall()
        return {(row["source_type"] or "unknown"): int(row["n"])
                for row in rows}

    def model_versions(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT model_version, COUNT(*) AS n FROM {TABLE} "
                "GROUP BY model_version").fetchall()
        return {(row["model_version"] or "unknown"): int(row["n"])
                for row in rows}

    # -- export -------------------------------------------------------------

    def export_csv(self, path: str | Path,
                   filters: Filters | None = None) -> Path:
        """Write the filtered events to a CSV file; returns the path."""
        path = Path(path)
        unbounded = replace(filters or Filters(), limit=100_000, offset=0)
        events = self.query(unbounded)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(EXPORT_COLUMNS))
            writer.writeheader()
            for event in events:
                writer.writerow(event.as_export_row())
        log.info("exported %d history rows to %s", len(events), path)
        return path

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "HistoryDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
