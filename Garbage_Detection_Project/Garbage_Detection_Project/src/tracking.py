"""Optional ByteTrack bookkeeping.

Tracking is opt-in. Without it the application reports detections per frame
only; with it, each object keeps a track id so unique objects can be counted
honestly instead of being inferred from frame counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.detector import Detection


@dataclass
class UniqueObjectTracker:
    """Remembers track ids seen so far, per class and overall."""

    seen: dict[str, set[int]] = field(default_factory=dict)

    def update(self, detections: list[Detection]) -> None:
        for det in detections:
            if det.track_id is None:
                continue
            self.seen.setdefault(det.class_name, set()).add(det.track_id)

    @property
    def enabled(self) -> bool:
        return any(self.seen.values())

    def unique_total(self) -> int:
        return sum(len(ids) for ids in self.seen.values())

    def unique_per_class(self) -> dict[str, int]:
        counts = {name: 0 for name in config.GARBAGE_CLASSES}
        for name, ids in self.seen.items():
            counts[name] = len(ids)
        return counts

    def reset(self) -> None:
        self.seen.clear()


def describe(detections: list[Detection], tracker: UniqueObjectTracker | None,
             ) -> dict:
    """Frame counts plus unique counts, with tracking clearly labelled."""
    frame_counts = {name: 0 for name in config.GARBAGE_CLASSES}
    for det in detections:
        frame_counts[det.class_name] += 1
    tracking = tracker is not None and tracker.enabled
    return {
        "frame_objects": len(detections),
        "frame_counts": frame_counts,
        "tracking": tracking,
        "unique_objects": tracker.unique_total() if tracking else None,
        "unique_counts": tracker.unique_per_class() if tracking else None,
    }
