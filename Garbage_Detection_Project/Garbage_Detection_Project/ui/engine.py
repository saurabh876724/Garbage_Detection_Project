"""Detection engine and the Qt worker that drives camera/video/image sources.

The model is loaded once (DetectionEngine) and every source runs on a single
QThread so the interface never blocks. Frames are streamed - a video is never
loaded into memory.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

import config
from src.alerts import AlertManager
from src.counting import SessionCounter
from src.detector import GarbageDetector, ModelError, SceneResult
from src.evidence import EvidenceError, EvidenceRecorder
from src.history import HistoryDB
from src.logging_setup import get_logger
from src.postprocessing import annotate_scene
from src.preprocessing import (MediaError, classify_source, load_image,
                               open_capture, video_metadata)
from src.settings import Settings
from src.utils import FpsMeter, file_timestamp

log = get_logger("engine")


class DetectionEngine:
    """Owns the model plus the services every source shares."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.history = HistoryDB()
        self.evidence = EvidenceRecorder()
        self.alerts = AlertManager(
            enabled=settings.alert_enabled,
            count_threshold=settings.alert_threshold,
            cooldown_seconds=settings.alert_cooldown,
            sound_path=settings.alert_sound)
        self.detector = self._build_detector(settings)

    @staticmethod
    def _build_detector(settings: Settings) -> GarbageDetector:
        return GarbageDetector(
            weights=settings.weights_path,
            conf_threshold=settings.confidence,
            iou_threshold=settings.iou,
            image_size=settings.image_size,
            device=settings.device,
            tracking=settings.tracking_enabled)

    def apply_settings(self, settings: Settings) -> bool:
        """Rebuild the detector when model-affecting settings changed."""
        same_model = (settings.weights_path == self.detector.weights_path
                      and settings.tracking_enabled == self.detector.tracking
                      and settings.device == self.settings.device)
        self.settings = settings
        self.alerts.enabled = settings.alert_enabled
        self.alerts.count_threshold = settings.alert_threshold
        self.alerts.cooldown = settings.alert_cooldown
        self.alerts.sound_path = settings.alert_sound
        if same_model:
            self.detector.conf_threshold = settings.confidence
            self.detector.iou_threshold = settings.iou
            self.detector.image_size = settings.image_size
            log.info("thresholds updated: conf=%.2f iou=%.2f imgsz=%d",
                     settings.confidence, settings.iou, settings.image_size)
            return False
        self.detector = self._build_detector(settings)
        log.info("detector rebuilt with %s", settings.weights_path)
        return True

    # -- per frame ----------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> SceneResult:
        return self.detector.analyze(frame)

    def annotate(self, frame: np.ndarray, result: SceneResult,
                 fps: float | None = None) -> np.ndarray:
        return annotate_scene(frame, result, fps=fps,
                              device=self.detector.device_details["device"])

    def save_evidence(self, frame: np.ndarray, result: SceneResult,
                      source: str, source_type: str,
                      fps: float | None = None,
                      unique: int | None = None):
        return self.evidence.save(frame, result, source=source,
                                  source_type=source_type,
                                  model_info=self.detector.info(), fps=fps,
                                  unique_objects=unique)

    def log_event(self, source: str, source_type: str, result: SceneResult,
                  evidence_path: str | None = None,
                  unique: int | None = None) -> int:
        return self.history.log_event(
            source=source, source_type=source_type, status=result.status,
            total_count=result.total, counts=result.counts,
            average_confidence=result.average_confidence,
            evidence_path=evidence_path,
            model_version=config.MODEL_VERSION,
            model_path=str(self.detector.weights_path),
            unique_objects=unique)

    def log_session(self, source: str, source_type: str,
                    session: SessionCounter) -> int:
        return self.history.log_event(
            source=source, source_type=source_type,
            status=session.dominant_status(),
            total_count=session.detections, counts=session.class_totals,
            average_confidence=session.average_confidence,
            model_version=config.MODEL_VERSION,
            model_path=str(self.detector.weights_path),
            unique_objects=session.unique_objects())

    def shutdown(self) -> None:
        try:
            self.history.prune(self.settings.keep_history_entries)
        finally:
            self.history.close()


class DetectionWorker(QThread):
    """Runs one source (camera, video file or image) off the UI thread."""

    frame_ready = Signal(object, object, float)     # frame, SceneResult, fps
    source_started = Signal(str, dict)              # source label, metadata
    source_finished = Signal(dict)                  # session summary
    evidence_saved = Signal(str, str)               # jpg path, status
    progress = Signal(int, int)                     # frame index, total
    error = Signal(str)

    def __init__(self, engine: DetectionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._lock = threading.RLock()
        self._mode: str | None = None
        self._source = None
        self._running = False
        self._paused = False
        self._capture_requested = False
        self._save_video = False
        self._video_output: Path | None = None
        self._last_frame: np.ndarray | None = None
        self._last_result: SceneResult | None = None
        self._fps_meter = FpsMeter()

    # -- control ------------------------------------------------------------

    def start_camera(self, index: int) -> None:
        self._begin("camera", index)

    def start_video(self, path: str | Path, save_output: bool = False) -> None:
        self._save_video = save_output
        self._begin("video", Path(path))

    def analyze_image(self, path: str | Path) -> None:
        self._begin("image", Path(path))

    def _begin(self, mode: str, source) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("a source is already running - stop it first")
            self._mode, self._source = mode, source
            self._running, self._paused = True, False
            self._capture_requested = False
            self._last_frame = self._last_result = None
        self._fps_meter.reset()
        self.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._paused = False

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def request_capture(self) -> None:
        with self._lock:
            self._capture_requested = True

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def mode(self) -> str | None:
        """Active source kind: camera, video, image or None."""
        with self._lock:
            return self._mode

    # -- thread -------------------------------------------------------------

    def run(self) -> None:  # noqa: D102 - QThread entry point
        with self._lock:
            mode, source = self._mode, self._source
        try:
            if mode == "image":
                self._run_image(source)
            elif mode == "camera":
                self._run_camera(source)
            else:
                self._run_video(source)
        except MediaError as exc:
            log.warning("media error: %s", exc)
            self.error.emit(str(exc))
        except ModelError as exc:
            log.error("model error: %s", exc)
            self.error.emit(str(exc))
        except Exception as exc:  # unexpected: log the traceback, tell the UI
            log.exception("detection worker crashed")
            self.error.emit(f"Unexpected error: {exc}\nDetails are in "
                            f"logs/{config.LOG_DIR.name}/app.log")
        finally:
            with self._lock:
                self._running = False
                self._mode = None

    def _run_image(self, path: Path) -> None:
        frame = load_image(path)
        self.source_started.emit(path.name, {"kind": "image",
                                             "path": str(path),
                                             "width": frame.shape[1],
                                             "height": frame.shape[0]})
        result = self.engine.analyze(frame)
        self.engine.alerts.update(result.total, result.status, path.name)
        annotated = self.engine.annotate(frame, result, fps=self.engine.detector.fps)

        evidence_path = None
        if self.engine.settings.evidence_enabled and (
                result.status == config.STATUS_DIRTY or self._consume_capture()):
            try:
                bundle = self.engine.save_evidence(frame, result,
                                                   source=str(path),
                                                   source_type="image")
                evidence_path = str(bundle.annotated)
                self.evidence_saved.emit(evidence_path, result.status)
            except EvidenceError as exc:
                self.error.emit(str(exc))

        self.engine.log_event(str(path), "image", result, evidence_path,
                              result.unique_objects)
        self._publish(annotated, result, self.engine.detector.fps)
        summary = {"kind": "image", "source": path.name, "frames": 1,
                   "status": result.status, "objects": result.total,
                   "counts": result.counts,
                   "average_confidence": result.average_confidence,
                   "evidence": evidence_path,
                   "output": str(path.with_stem(path.stem + "_detected"))}
        out = Path(summary["output"])
        if not cv2.imwrite(str(out), annotated):
            log.warning("could not write annotated image %s", out)
        self.source_finished.emit(summary)

    def _run_camera(self, index: int) -> None:
        capture = open_capture(index)
        meta = video_metadata(capture)
        self.source_started.emit(f"Camera {index}",
                                 {"kind": "camera", "index": index, **meta})
        session = SessionCounter(source=f"camera{index}",
                                 tracking=self.engine.detector.tracking)
        try:
            while self.is_running:
                if self.is_paused:
                    time.sleep(0.05)
                    continue
                ok, frame = capture.read()
                if not ok:
                    self.error.emit("Camera stopped delivering frames. It may "
                                    "have been disconnected or claimed by "
                                    "another application.")
                    break
                self._process_stream_frame(frame, session, f"camera{index}",
                                           "camera")
        finally:
            capture.release()
        self._finish_session(session, f"camera{index}", "camera")

    def _run_video(self, path: Path) -> None:
        capture = open_capture(path)
        meta = video_metadata(capture)
        total = max(0, meta.get("frames", 0))
        self.source_started.emit(path.name, {"kind": "video",
                                             "path": str(path), **meta})
        session = SessionCounter(source=path.name,
                                 tracking=self.engine.detector.tracking)
        writer = self._open_writer(capture, meta) if self._save_video else None
        try:
            while self.is_running:
                if self.is_paused:
                    time.sleep(0.05)
                    continue
                ok, frame = capture.read()
                if not ok:
                    break
                annotated = self._process_stream_frame(
                    frame, session, path.name, "video")
                if writer is not None:
                    writer.write(annotated)
                if total:
                    self.progress.emit(min(session.frames, total), total)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
                log.info("processed video written: %s", self._video_output)
        summary_extra = ({"output": str(self._video_output)}
                         if writer and self._video_output else {})
        self._finish_session(session, path.name, "video", summary_extra)

    def _open_writer(self, capture: cv2.VideoCapture, meta: dict):
        """Create an mp4 writer matching the source resolution and fps."""
        width = int(meta.get("width") or capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(meta.get("height") or capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(meta.get("fps") or 25.0) or 25.0
        config.ensure_dirs()
        self._video_output = (config.EVIDENCE_DIR /
                              f"processed_{file_timestamp()}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self._video_output), fourcc, fps,
                                 (width, height))
        if not writer.isOpened():
            log.warning("video writer could not open - saving disabled")
            return None
        return writer

    def _process_stream_frame(self, frame: np.ndarray, session: SessionCounter,
                              source: str, source_type: str) -> np.ndarray:
        result = self.engine.analyze(frame)
        session.record(result)
        self.engine.alerts.update(result.total, result.status, source)
        fps = self._fps_meter.tick()
        annotated = self.engine.annotate(frame, result, fps=fps)

        with self._lock:
            self._last_frame, self._last_result = frame, result
            capture_requested = self._capture_requested
            self._capture_requested = False
        if capture_requested or (self.engine.settings.evidence_enabled
                                 and result.status == config.STATUS_DIRTY
                                 and session.frames
                                 % config.EVIDENCE_FRAME_INTERVAL == 0):
            self._save_current_evidence(source, source_type, fps)

        self._publish(annotated, result, fps)
        return annotated

    def _save_current_evidence(self, source: str, source_type: str,
                               fps: float) -> None:
        with self._lock:
            frame, result = self._last_frame, self._last_result
        if frame is None or result is None:
            return
        try:
            bundle = self.engine.save_evidence(frame, result, source=source,
                                               source_type=source_type,
                                               fps=fps,
                                               unique=result.unique_objects)
        except EvidenceError as exc:
            self.error.emit(str(exc))
            return
        self.engine.log_event(source, source_type, result,
                              str(bundle.annotated), result.unique_objects)
        self.evidence_saved.emit(str(bundle.annotated), result.status)

    def capture_evidence_now(self) -> str | None:
        """Capture from the UI thread using the newest processed frame."""
        with self._lock:
            frame, result = self._last_frame, self._last_result
        if frame is None or result is None:
            return None
        try:
            bundle = self.engine.save_evidence(frame, result,
                                               source="manual",
                                               source_type="manual")
        except EvidenceError as exc:
            self.error.emit(str(exc))
            return None
        self.engine.log_event("manual capture", "manual", result,
                              str(bundle.annotated), result.unique_objects)
        self.evidence_saved.emit(str(bundle.annotated), result.status)
        return str(bundle.annotated)

    def _consume_capture(self) -> bool:
        with self._lock:
            requested = self._capture_requested
            self._capture_requested = False
        return requested

    def _publish(self, frame: np.ndarray, result: SceneResult,
                 fps: float) -> None:
        self.frame_ready.emit(frame, result, fps)

    def _finish_session(self, session: SessionCounter, source: str,
                        source_type: str, extra: dict | None = None) -> None:
        if session.frames:
            self.engine.log_session(source, source_type, session)
        summary = session.summary()
        summary.update({"kind": source_type, "source": source})
        summary.update(extra or {})
        log.info("session finished: %s", summary)
        self.source_finished.emit(summary)

    def shutdown(self) -> None:
        """Stop the thread and wait for it to finish."""
        self.stop()
        if self.isRunning():
            self.wait(3000)
