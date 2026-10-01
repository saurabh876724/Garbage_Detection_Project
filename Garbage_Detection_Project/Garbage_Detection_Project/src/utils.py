"""Small shared helpers: measured FPS, timestamps, unique file paths."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
FILENAME_FORMAT = "%Y-%m-%d_%H-%M-%S"


class FpsMeter:
    """Rolling FPS from wall-clock time between frames (never fabricated)."""

    def __init__(self, window: int = 30):
        self._stamps: deque[float] = deque(maxlen=max(2, window))

    def tick(self) -> float:
        """Record a frame and return the smoothed FPS so far."""
        now = time.perf_counter()
        self._stamps.append(now)
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def reset(self) -> None:
        self._stamps.clear()


def now() -> datetime:
    return datetime.now()


def timestamp(dt: datetime | None = None) -> str:
    """Sortable timestamp used in the database and evidence metadata."""
    return (dt or now()).strftime(TIMESTAMP_FORMAT)


def file_timestamp(dt: datetime | None = None) -> str:
    """Filesystem-safe timestamp used in generated file names."""
    return (dt or now()).strftime(FILENAME_FORMAT)


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return a path in `directory` that does not overwrite existing evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def parse_timestamp(value: str) -> datetime | None:
    """Parse a stored timestamp, returning None when unusable."""
    for fmt in (TIMESTAMP_FORMAT, FILENAME_FORMAT, "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None
