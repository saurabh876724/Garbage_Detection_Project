"""Counting: per-frame totals and honest session statistics.

Frame-by-frame detections are NOT unique objects. Unique counts are only
reported when tracking is enabled (see src/tracking.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from src.detector import SceneResult
from src.tracking import UniqueObjectTracker


@dataclass
class SessionCounter:
    """Aggregates one detection session (camera, video or batch run)."""

    source: str = ""
    tracking: bool = False
    frames: int = 0
    dirty_frames: int = 0
    review_frames: int = 0
    clean_frames: int = 0
    detections: int = 0
    confidence_sum: float = 0.0
    class_totals: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in config.GARBAGE_CLASSES})
    peak_objects: int = 0
    _tracker: UniqueObjectTracker = field(default_factory=UniqueObjectTracker)
    _started: float = field(default_factory=time.perf_counter)

    def record(self, result: SceneResult) -> None:
        """Fold one analyzed frame into the running totals."""
        self.frames += 1
        self.detections += result.total
        self.confidence_sum += sum(d.confidence for d in result.detections)
        self.peak_objects = max(self.peak_objects, result.total)
        for name, count in result.counts.items():
            self.class_totals[name] = self.class_totals.get(name, 0) + count
        if result.status == config.STATUS_DIRTY:
            self.dirty_frames += 1
        elif result.status == config.STATUS_REVIEW:
            self.review_frames += 1
        else:
            self.clean_frames += 1
        if self.tracking:
            self._tracker.update(result.detections)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._started

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / self.detections if self.detections else 0.0

    def dominant_status(self) -> str:
        """Session level verdict from the frame statuses actually observed."""
        if self.dirty_frames:
            return config.STATUS_DIRTY
        if self.review_frames:
            return config.STATUS_REVIEW
        return config.STATUS_CLEAN

    def unique_objects(self) -> int | None:
        """Unique tracked objects, or None when tracking was off."""
        return self._tracker.unique_total() if self.tracking else None

    def unique_per_class(self) -> dict[str, int] | None:
        """Unique tracked objects per class, or None when tracking was off."""
        return self._tracker.unique_per_class() if self.tracking else None

    def summary(self) -> dict:
        return {
            "source": self.source,
            "frames": self.frames,
            "dirty_frames": self.dirty_frames,
            "review_frames": self.review_frames,
            "clean_frames": self.clean_frames,
            "detections": self.detections,
            "peak_objects": self.peak_objects,
            "average_confidence": round(self.average_confidence, 4),
            "class_totals": dict(self.class_totals),
            "unique_objects": self.unique_objects(),
            "tracking": self.tracking,
            "status": self.dominant_status(),
            "seconds": round(self.elapsed_seconds, 2),
        }

    def reset(self) -> None:
        self.frames = self.detections = self.dirty_frames = 0
        self.review_frames = self.clean_frames = self.peak_objects = 0
        self.confidence_sum = 0.0
        self.class_totals = {name: 0 for name in config.GARBAGE_CLASSES}
        self._tracker.reset()
        self._started = time.perf_counter()
