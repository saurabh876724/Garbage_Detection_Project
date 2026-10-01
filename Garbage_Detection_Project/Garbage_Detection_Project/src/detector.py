"""GarbageDetector - YOLOv8 inference, validation and scene status.

The model detects the 10 garbage classes only. "clean" is never a class:
CLEAN / DIRTY / REVIEW are derived from what the detector found in the frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from src.logging_setup import get_logger

log = get_logger("detector")


class ModelError(RuntimeError):
    """Raised when the detection engine cannot be started."""


@dataclass(frozen=True)
class Detection:
    """One validated garbage bounding box in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.class_id < config.NUM_CLASSES:
            raise ValueError(f"class id {self.class_id} outside the 10 "
                             "garbage classes")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"degenerate box {(self.x1, self.y1, self.x2, self.y2)}")

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        """Box area in pixels."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def label(self) -> str:
        """Display text, e.g. 'plastic 94%' or 'plastic #12 94%'."""
        name = (f"{self.class_name} #{self.track_id}"
                if self.track_id is not None else self.class_name)
        return f"{name} {self.confidence * 100:.0f}%"

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "class_name": self.class_name,
                "confidence": round(self.confidence, 4), "box": self.box,
                "center": self.center, "area": self.area,
                "track_id": self.track_id}


@dataclass
class SceneResult:
    """Everything derived from a single frame."""

    detections: list[Detection] = field(default_factory=list)
    weak_detections: list[Detection] = field(default_factory=list)
    status: str = config.STATUS_CLEAN
    counts: dict[str, int] = field(default_factory=dict)
    inference_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.detections)

    @property
    def average_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)

    @property
    def unique_objects(self) -> int | None:
        """Unique tracked objects, or None when tracking is disabled."""
        ids = {d.track_id for d in self.detections if d.track_id is not None}
        return len(ids) if ids else None


def resolve_device(device: str = "auto") -> str:
    """Return the ultralytics device string actually usable on this machine."""
    import torch

    if device in ("auto", "", None):
        return "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise ModelError("CUDA was requested but torch.cuda.is_available() "
                         "is False - run check_gpu.py for details.")
    return device


def device_info(device: str) -> dict[str, str]:
    """Human readable device description for the UI and logs."""
    import torch

    if device == "cpu" or not torch.cuda.is_available():
        return {"device": "CPU", "gpu": "none", "vram": "n/a",
                "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "device": f"CUDA:{device}" if device.isdigit() else device,
        "gpu": props.name,
        "vram": f"{props.total_memory / 1024 ** 3:.2f} GB",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "n/a",
    }


class GarbageDetector:
    """Loads the model once and answers detection queries per frame."""

    def __init__(self, weights: str | Path | None = None,
                 conf_threshold: float = config.CONF_THRESHOLD,
                 iou_threshold: float = config.IOU_THRESHOLD,
                 image_size: int = config.INPUT_SIZE,
                 device: str = "auto",
                 tracking: bool = False):
        self.weights_path = Path(weights or config.default_weights())
        if not self.weights_path.is_file():
            raise ModelError(
                f"Model not found.\n\nExpected: {self.weights_path}\n\n"
                "Train the model first (python train_yolo.py) or point the "
                "settings at an existing best.pt file.")

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.image_size = int(image_size)
        self.tracking = bool(tracking)
        self.device = resolve_device(device)
        self.device_details = device_info(self.device)
        self.inference_ms = 0.0

        self.model = self._load_model()
        self._validate_classes()
        log.info("model ready: %s on %s (%s), conf=%.2f iou=%.2f tracking=%s",
                 self.weights_path.name, self.device_details["device"],
                 self.device_details["gpu"], self.conf_threshold,
                 self.iou_threshold, self.tracking)

    # -- construction -------------------------------------------------------

    def _load_model(self):
        try:
            from ultralytics import YOLO
            return YOLO(str(self.weights_path))
        except Exception as exc:  # corrupt weights, missing dependency
            log.exception("model load failed")
            raise ModelError(
                f"The model could not be loaded:\n{exc}\n\n"
                "Verify that ultralytics is installed and that the weights "
                "file is not corrupted.") from exc

    def _validate_classes(self) -> None:
        names = getattr(self.model, "names", {}) or {}
        model_classes = [names[i] for i in sorted(names)]
        if model_classes != list(config.GARBAGE_CLASSES):
            raise ModelError(
                f"{self.weights_path.name} detects {model_classes}, which does "
                f"not match the project classes {list(config.GARBAGE_CLASSES)}."
                "\n\nThis looks like the old COCO-derived model. Train a new "
                "one with: python train_yolo.py")
        self.class_names = {i: name for i, name in enumerate(model_classes)}

    # -- inference ----------------------------------------------------------

    def _predict(self, frame: np.ndarray) -> list:
        # Inference runs at the REVIEW floor so weak hits can still be
        # reported; the accept threshold is applied afterwards.
        kwargs = dict(conf=min(self.conf_threshold, config.REVIEW_THRESHOLD),
                      iou=self.iou_threshold, imgsz=self.image_size,
                      device=self.device, verbose=False)
        started = time.perf_counter()
        if self.tracking:
            results = self.model.track(source=frame, persist=True,
                                       tracker=config.TRACKER_CONFIG, **kwargs)
        else:
            results = self.model.predict(source=frame, **kwargs)
        self.inference_ms = (time.perf_counter() - started) * 1000.0
        return results or []

    def analyze(self, frame: np.ndarray) -> SceneResult:
        """Run detection and derive counts plus the CLEAN/DIRTY/REVIEW status."""
        if frame is None or frame.size == 0:
            raise ValueError("analyze() needs a non-empty frame")

        detections, weak = [], []
        frame_area = float(frame.shape[0] * frame.shape[1])
        for result in self._predict(frame):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for raw in boxes:
                detection = self._to_detection(raw)
                if detection is None:
                    continue
                if self.is_degenerate(detection, frame_area):
                    # A box spanning (almost) the whole scene is a model
                    # artefact, never a real object: keep it review-only so it
                    # cannot be counted as confident garbage.
                    weak.append(detection)
                elif detection.confidence >= self.conf_threshold:
                    detections.append(detection)
                else:
                    weak.append(detection)

        accepted = detections[:config.MAX_DETECTIONS]
        return SceneResult(
            detections=accepted,
            weak_detections=weak,
            status=self.scene_status(accepted, weak),
            counts=self.counts(accepted),
            inference_ms=self.inference_ms,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Confidence-filtered detections only (no status bookkeeping)."""
        return self.analyze(frame).detections

    def _to_detection(self, raw) -> Detection | None:
        """Convert an ultralytics box into a validated Detection."""
        try:
            class_id = int(raw.cls[0])
            if class_id not in self.class_names:
                log.debug("ignoring unknown class id %s", class_id)
                return None
            confidence = float(raw.conf[0])
            x1, y1, x2, y2 = (int(round(v)) for v in raw.xyxy[0].tolist())
            track_id = None
            if getattr(raw, "id", None) is not None:
                track_id = int(raw.id[0])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            log.warning("skipping malformed detection: %s", exc)
            return None
        try:
            return Detection(class_id, self.class_names[class_id], confidence,
                             x1, y1, x2, y2, track_id)
        except ValueError as exc:
            log.warning("skipping invalid box: %s", exc)
            return None

    # -- derived state ------------------------------------------------------

    @staticmethod
    def counts(detections: list[Detection]) -> dict[str, int]:
        """Per-class counts; every class is present, zero when not detected."""
        counts = {name: 0 for name in config.GARBAGE_CLASSES}
        for det in detections:
            counts[det.class_name] += 1
        return counts

    @staticmethod
    def is_degenerate(detection: "Detection", frame_area: float) -> bool:
        """True when a box covers so much of the frame it cannot be one object."""
        return (bool(frame_area) and
                detection.area / frame_area > config.MAX_BOX_AREA_FRACTION)

    @staticmethod
    def scene_status(detections: list[Detection],
                     weak: list[Detection] | None = None,
                     dirty_threshold: int = config.DIRTY_THRESHOLD) -> str:
        """DIRTY when enough confident garbage, REVIEW when only weak hits."""
        if len(detections) >= dirty_threshold:
            return config.STATUS_DIRTY
        if weak:
            return config.STATUS_REVIEW
        return config.STATUS_CLEAN

    @property
    def fps(self) -> float:
        """Inference-only FPS measured from the last frame."""
        return 1000.0 / self.inference_ms if self.inference_ms > 0 else 0.0

    def info(self) -> dict:
        """Model identity and runtime facts for the UI."""
        return {
            "model": config.MODEL_NAME,
            "version": config.MODEL_VERSION,
            "weights": str(self.weights_path),
            "size_mb": round(self.weights_path.stat().st_size / 1024 ** 2, 2),
            "device": self.device_details["device"],
            "gpu": self.device_details["gpu"],
            "vram": self.device_details["vram"],
            "confidence": self.conf_threshold,
            "iou": self.iou_threshold,
            "image_size": self.image_size,
            "tracking": self.tracking,
            "classes": list(self.class_names.values()),
        }
