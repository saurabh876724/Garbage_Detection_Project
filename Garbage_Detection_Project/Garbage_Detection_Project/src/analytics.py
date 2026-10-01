"""Analytics: aggregates computed from real stored data only.

Everything here reads the history database or the JSON reports written by the
dataset/evaluation scripts - no synthetic numbers are produced. When there is
no data yet, the result says so and the UI renders an empty state.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import config
from src.history import Filters, HistoryDB
from src.logging_setup import get_logger

log = get_logger("analytics")

EVALUATION_REPORT = config.PROJECT_ROOT / "dataset_final" / "reports" / "model_evaluation.json"
DATASET_REPORT = config.PROJECT_ROOT / "dataset_final" / "reports" / "dataset_statistics.json"
AUDIT_REPORT = config.PROJECT_ROOT / "dataset_final" / "reports" / "annotation_audit_report.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def overview(history: HistoryDB) -> dict:
    """Headline numbers for the dashboard metric cards."""
    status = history.status_totals()
    total = sum(status.values())
    dirty = status.get(config.STATUS_DIRTY, 0)
    clean = status.get(config.STATUS_CLEAN, 0)
    review = status.get(config.STATUS_REVIEW, 0)
    classes = history.class_totals()
    top_class = max(classes.items(), key=lambda kv: kv[1]) if total else None

    today = history.query(Filters(date_from=date.today(), date_to=date.today(),
                                  limit=100_000))
    return {
        "total_events": total,
        "today_events": len(today),
        "today_objects": sum(e.total_count for e in today),
        "clean": clean,
        "dirty": dirty,
        "review": review,
        "clean_ratio": round(clean / total, 4) if total else 0.0,
        "dirty_ratio": round(dirty / total, 4) if total else 0.0,
        "total_objects": sum(classes.values()),
        "average_confidence": round(history.average_confidence(), 4),
        "most_common_class": top_class[0] if top_class and top_class[1] else None,
        "most_common_count": top_class[1] if top_class else 0,
        "sources": history.source_totals(),
        "model_versions": history.model_versions(),
        "has_data": total > 0,
    }


def class_distribution(history: HistoryDB) -> dict:
    """Boxes per class, descending - drives the bar chart."""
    totals = history.class_totals()
    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return {"labels": [name for name, _ in ordered],
            "values": [count for _, count in ordered],
            "has_data": any(count for _, count in ordered)}


def status_split(history: HistoryDB) -> dict:
    """CLEAN / DIRTY / REVIEW share - drives the donut chart."""
    totals = history.status_totals()
    labels = [config.STATUS_CLEAN, config.STATUS_DIRTY, config.STATUS_REVIEW]
    values = [totals.get(label, 0) for label in labels]
    return {"labels": labels, "values": values, "has_data": sum(values) > 0}


def daily_activity(history: HistoryDB, days: int = 14) -> dict:
    """Events per day split by status - drives the line chart."""
    rows = history.daily_counts(days)
    series = {status: [] for status in
              (config.STATUS_CLEAN, config.STATUS_DIRTY, config.STATUS_REVIEW)}
    for _, counts in rows:
        for status in series:
            series[status].append(counts.get(status, 0))
    return {"labels": [day for day, _ in rows], "series": series,
            "has_data": any(sum(values) for values in series.values())}


def hourly_activity(history: HistoryDB, days: int = 7) -> dict:
    """When detections happen during the day (0-23), over the last N days."""
    since = datetime.now() - timedelta(days=days)
    buckets = [0] * 24
    for event in history.query(Filters(date_from=since.date(),
                                       date_to=date.today(), limit=100_000)):
        try:
            hour = datetime.fromisoformat(event.timestamp).hour
        except ValueError:
            continue
        buckets[hour] += 1
    return {"labels": [f"{h:02d}:00" for h in range(24)], "values": buckets,
            "days": days, "has_data": sum(buckets) > 0}


def objects_per_event(history: HistoryDB) -> dict:
    """Distribution of how many objects a scene contained."""
    buckets = {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for event in history.query(Filters(limit=100_000)):
        n = event.total_count
        if n == 0:
            buckets["0"] += 1
        elif n <= 2:
            buckets["1-2"] += 1
        elif n <= 5:
            buckets["3-5"] += 1
        elif n <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1
    return {"labels": list(buckets), "values": list(buckets.values()),
            "has_data": sum(buckets.values()) > 0}


def model_metrics() -> dict:
    """Held-out test metrics written by evaluate_model.py (empty if absent)."""
    report = _read_json(EVALUATION_REPORT)
    if not report:
        return {"available": False}
    test = report.get("test", {})
    return {
        "available": True,
        "generated_at": report.get("generated_at", ""),
        "weights": report.get("weights", ""),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1": test.get("f1"),
        "map50": test.get("map50"),
        "map50_95": test.get("map50_95"),
        "fps": test.get("fps"),
        "inference_ms": test.get("inference_ms"),
        "model_size_mb": test.get("model_size_mb"),
        "per_class": test.get("per_class", []),
    }


def dataset_summary() -> dict:
    """Dataset facts from the audit/statistics reports (for the About page)."""
    stats = _read_json(DATASET_REPORT)
    audit = _read_json(AUDIT_REPORT)
    if not stats and not audit:
        return {"available": False}
    return {
        "available": True,
        "total_images": stats.get("total_images"),
        "total_boxes": sum((stats.get("total_boxes") or {}).values()) or None,
        "splits": audit.get("images_per_split"),
        "quarantined_images": stats.get("quarantined_images"),
        "hard_errors": audit.get("hard_errors"),
        "imbalance": stats.get("imbalance"),
        "provenance": stats.get("annotation_method_provenance"),
        "boxes_per_class": stats.get("boxes_per_class"),
    }
