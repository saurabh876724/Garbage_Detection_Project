"""Evidence capture: timestamped JPG (original + annotated) and JSON sidecar.

Files are never overwritten - a numeric suffix is added when a capture lands in
the same second as another one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from src.detector import SceneResult
from src.logging_setup import get_logger
from src.postprocessing import annotate_scene, status_text
from src.utils import file_timestamp, unique_path

log = get_logger("evidence")


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be written."""


@dataclass
class EvidenceBundle:
    """Paths of everything written for one capture."""

    original: Path
    annotated: Path
    metadata: Path

    def as_dict(self) -> dict:
        return {"original": str(self.original),
                "annotated": str(self.annotated),
                "metadata": str(self.metadata)}


class EvidenceRecorder:
    def __init__(self, evidence_dir: Path | None = None,
                 store_original: bool = True):
        self.evidence_dir = Path(evidence_dir or config.EVIDENCE_DIR)
        self.store_original = store_original
        try:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EvidenceError(
                f"Evidence folder is not writable:\n{self.evidence_dir}\n{exc}"
            ) from exc

    def save(self, frame: np.ndarray, result: SceneResult,
             source: str = "unknown", source_type: str = "unknown",
             model_info: dict | None = None, fps: float | None = None,
             unique_objects: int | None = None) -> EvidenceBundle:
        """Write the evidence files for one scene and return their paths."""
        if frame is None or frame.size == 0:
            raise EvidenceError("cannot save evidence from an empty frame")

        moment = datetime.now()
        stem = f"garbage_detection_{file_timestamp(moment)}"
        annotated_path = unique_path(self.evidence_dir, f"{stem}_annotated",
                                     ".jpg")
        annotated = annotate_scene(frame, result, fps=fps,
                                   device=(model_info or {}).get("device", ""))
        if not cv2.imwrite(str(annotated_path), annotated):
            raise EvidenceError(f"could not write {annotated_path}")

        original_path = annotated_path
        if self.store_original:
            original_path = unique_path(self.evidence_dir, stem, ".jpg")
            if not cv2.imwrite(str(original_path), frame):
                log.warning("original frame could not be saved")
                original_path = annotated_path

        model_info = model_info or {}
        metadata_path = annotated_path.with_suffix(".json")
        payload = {
            "timestamp": moment.isoformat(timespec="seconds"),
            "source": source,
            "source_type": source_type,
            "status": result.status,
            "status_detail": status_text(result.status, result),
            "total_garbage_objects": result.total,
            "unique_objects": unique_objects,
            "counts": result.counts,
            "average_confidence": round(result.average_confidence, 4),
            "inference_ms": round(result.inference_ms, 2),
            "fps": round(fps, 2) if fps else None,
            "model": model_info.get("model", config.MODEL_NAME),
            "model_version": model_info.get("version", config.MODEL_VERSION),
            "model_path": model_info.get("weights", ""),
            "device": model_info.get("device", ""),
            "gpu": model_info.get("gpu", ""),
            "confidence_threshold": model_info.get("confidence"),
            "detections": [d.to_dict() for d in result.detections],
            "weak_detections": [d.to_dict() for d in result.weak_detections],
        }
        payload["files"] = {"original": str(original_path),
                            "annotated": str(annotated_path),
                            "metadata": str(metadata_path)}
        try:
            metadata_path.write_text(json.dumps(payload, indent=2),
                                     encoding="utf-8")
        except OSError as exc:
            log.error("evidence metadata could not be written: %s", exc)

        log.info("evidence saved: %s (%s, %d objects)", annotated_path.name,
                 result.status, result.total)
        return EvidenceBundle(original=original_path, annotated=annotated_path,
                              metadata=metadata_path)


def latest_bundles(limit: int = 12) -> list[EvidenceBundle]:
    """Newest evidence bundles on disk, for the dashboard gallery."""
    bundles = []
    sidecars = sorted(config.EVIDENCE_DIR.glob("*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    for sidecar in sidecars:
        payload = read_metadata(sidecar)
        files = payload.get("files", {})
        annotated = Path(files.get("annotated", ""))
        if not annotated.is_file():
            annotated = sidecar.with_suffix(".jpg")
        if not annotated.is_file():
            continue
        original = Path(files["original"]) if files.get("original")             else annotated
        bundles.append(EvidenceBundle(original=original, annotated=annotated,
                                      metadata=sidecar))
        if len(bundles) >= limit:
            break
    return bundles


def read_metadata(path: Path) -> dict:
    """Load one evidence JSON file, returning {} when it is unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("evidence metadata unreadable (%s): %s", path, exc)
        return {}


def frame_from_evidence(bundle: EvidenceBundle) -> np.ndarray | None:
    """Read back a stored evidence image."""
    for path in (bundle.annotated, bundle.original):
        if path.is_file():
            return cv2.imread(str(path))
    return None
