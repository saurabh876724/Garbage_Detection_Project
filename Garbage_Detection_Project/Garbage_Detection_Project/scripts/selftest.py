"""Verification suite for the application modules (no trained model needed).

Runs the pieces that can be checked deterministically: settings round-trip,
history schema + migration + filters + CSV export, evidence writing, alert
cooldown, counting (including tracked unique objects), detection validation,
status logic, drawing, analytics aggregates and Qt widget painting.

Usage:
    python scripts/selftest.py            # run everything
    python scripts/selftest.py --keep     # keep the temporary output folder
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run a check, record the outcome, never abort the suite."""
    def wrapper(fn):
        def runner(tmp: Path):
            import inspect
            takes_tmp = bool(inspect.signature(fn).parameters)
            try:
                detail = (fn(tmp) if takes_tmp else fn()) or ""
                RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
            except Exception as exc:
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
                if os.environ.get("SELFTEST_TRACEBACK"):
                    import traceback
                    traceback.print_exc()
        return runner
    return wrapper


def synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    frame[120:320, 180:420] = (60, 140, 200)
    return frame


def synthetic_result(status: str = config.STATUS_DIRTY):
    from src.detector import Detection, SceneResult
    detections = [
        Detection(7, "plastic", 0.94, 180, 120, 420, 320),
        Detection(6, "paper", 0.81, 60, 200, 150, 300),
    ] if status == config.STATUS_DIRTY else []
    weak = [Detection(5, "metal", 0.24, 400, 300, 460, 360)]
    result = SceneResult(detections=detections, weak_detections=weak,
                         counts={name: 0 for name in config.GARBAGE_CLASSES})
    for det in detections:
        result.counts[det.class_name] += 1
    result.status = status
    result.inference_ms = 24.5
    return result


# --- checks -----------------------------------------------------------------

@check("config: directories and defaults")
def check_config():
    config.ensure_dirs()
    assert config.NUM_CLASSES == 10
    assert config.GARBAGE_CLASSES[0] == "battery"
    assert "clean" not in config.GARBAGE_CLASSES
    for path in (config.DATA_DIR, config.EVIDENCE_DIR, config.LOG_DIR):
        assert path.is_dir(), path
    return f"weights -> {config.default_weights().name}"


@check("settings: save/load round-trip and validation")
def check_settings(tmp: Path):
    from src.settings import Settings, load_settings, save_settings, system_info
    path = tmp / "settings.json"
    settings = Settings(confidence=0.62, iou=0.55, camera_index=2,
                        alert_cooldown=45.0, tracking_enabled=True)
    assert save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.confidence == 0.62 and loaded.camera_index == 2
    assert loaded.tracking_enabled is True
    clamped = Settings(confidence=5.0, iou=-1.0, camera_index=-3)
    assert clamped.confidence == 0.95 and clamped.iou == 0.10
    assert clamped.camera_index == 0
    info = system_info()
    assert {"python", "pytorch", "gpu", "ultralytics"} <= set(info)
    return f"conf clamped to {clamped.confidence}"


@check("detection: validation rejects impossible boxes")
def check_detection():
    from src.detector import Detection
    det = Detection(7, "plastic", 0.9, 10, 20, 110, 220)
    assert det.area == 100 * 200 and det.center == (60, 120)
    assert det.center_x == 60 and det.center_y == 120
    assert det.label == "plastic 90%"
    assert Detection(7, "plastic", 0.9, 10, 20, 110, 220, 12).label \
        == "plastic #12 90%"
    for bad in (lambda: Detection(99, "x", 0.5, 0, 0, 10, 10),
                lambda: Detection(0, "battery", 1.5, 0, 0, 10, 10),
                lambda: Detection(0, "battery", 0.5, 10, 10, 10, 20)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid detection was accepted")
    return "class id / confidence / geometry guarded"


@check("status: CLEAN, DIRTY and REVIEW are derived, not predicted")
def check_status():
    from src.detector import Detection, GarbageDetector
    confident = [Detection(7, "plastic", 0.9, 0, 0, 10, 10)]
    weak = [Detection(5, "metal", 0.22, 0, 0, 10, 10)]
    assert GarbageDetector.scene_status([], []) == config.STATUS_CLEAN
    assert GarbageDetector.scene_status(confident, []) == config.STATUS_DIRTY
    assert GarbageDetector.scene_status([], weak) == config.STATUS_REVIEW
    assert GarbageDetector.counts(confident)["plastic"] == 1
    assert GarbageDetector.counts([])["trash"] == 0
    whole = Detection(7, "plastic", 0.9, 0, 0, 640, 480)
    small = Detection(7, "plastic", 0.9, 100, 100, 200, 200)
    assert GarbageDetector.is_degenerate(whole, 640 * 480)
    assert not GarbageDetector.is_degenerate(small, 640 * 480)
    return "3 states verified"


@check("history: schema v2, migration, filters, aggregates, CSV export")
def check_history(tmp: Path):
    from src.history import Event, Filters, HistoryDB
    db_path = tmp / "history.db"

    # v1 database (no average_confidence / model_version columns)
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript("""
        CREATE TABLE detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, source TEXT NOT NULL,
            status TEXT NOT NULL, total_count INTEGER NOT NULL,
            counts_json TEXT NOT NULL, evidence_path TEXT);
    """)
    legacy.execute("INSERT INTO detection_events (timestamp, source, status, "
                   "total_count, counts_json, evidence_path) VALUES "
                   "('2026-01-02T03:04:05', 'old_camera', 'DIRTY', 4, "
                   "'{\"plastic\": 4}', NULL)")
    legacy.commit()
    legacy.close()

    with HistoryDB(db_path) as history:
        columns = {row[1] for row in
                   sqlite3.connect(str(db_path)).execute(
                       "PRAGMA table_info(detection_events)")}
        assert {"average_confidence", "model_version", "source_type",
                "unique_objects"} <= columns, columns

        # Anchor to noon today so the "today" filter is deterministic even
        # when the suite runs shortly after midnight.
        now = datetime.combine(date.today(), dtime(12, 0))
        history.log_event("camera0", config.STATUS_DIRTY, 5,
                          {"plastic": 3, "paper": 2}, "evidence/x.jpg",
                          average_confidence=0.83, source_type="camera",
                          model_version="v1.0", unique_objects=5, when=now)
        history.log_event("img.jpg", config.STATUS_CLEAN, 0, {},
                          average_confidence=0.0, source_type="image",
                          when=now - timedelta(minutes=30))
        history.log_event("clip.mp4", config.STATUS_REVIEW, 0, {},
                          average_confidence=0.22, source_type="video",
                          when=now - timedelta(days=3))

        assert history.event_count() == 4
        dirty = history.query(Filters(status=config.STATUS_DIRTY))
        assert len(dirty) == 2 and all(e.status == "DIRTY" for e in dirty)
        assert isinstance(dirty[0], Event)
        camera = history.query(Filters(source_type="camera"))
        assert len(camera) == 1
        searched = history.query(Filters(text="plastic"))
        assert len(searched) >= 1
        today = history.query(Filters(date_from=date.today(),
                                      date_to=date.today()))
        assert len(today) == 2
        totals = history.class_totals()
        assert totals["plastic"] == 7, totals
        assert history.status_totals()["CLEAN"] == 1
        # rows with total_count > 0: legacy (0.0) and the camera event (0.83)
        assert 0.40 < history.average_confidence() < 0.45,             history.average_confidence()
        assert len(history.daily_counts(7)) == 7
        assert history.source_totals()["video"] == 1

        csv_path = tmp / "export.csv"
        history.export_csv(csv_path, Filters(status=config.STATUS_DIRTY))
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3, lines
        assert "model_version" in lines[0]

        removed = history.prune(2)
        assert removed == 2 and history.event_count() == 2
        return f"migrated v1 row kept, {removed} pruned"


@check("evidence: annotated + original + JSON, never overwritten")
def check_evidence(tmp: Path):
    from src.evidence import EvidenceRecorder, latest_bundles, read_metadata
    frame = synthetic_frame()
    result = synthetic_result()
    recorder = EvidenceRecorder(evidence_dir=tmp / "evidence")
    first = recorder.save(frame, result, source="camera0",
                          source_type="camera",
                          model_info={"device": "CUDA:0", "gpu": "RTX 4050",
                                      "version": config.MODEL_VERSION},
                          fps=27.4, unique_objects=2)
    assert first.annotated.is_file() and first.original.is_file()
    payload = read_metadata(first.metadata)
    assert payload["status"] == config.STATUS_DIRTY
    assert payload["total_garbage_objects"] == 2
    assert payload["counts"]["plastic"] == 1
    assert payload["unique_objects"] == 2
    assert payload["model_version"] == config.MODEL_VERSION
    assert len(payload["detections"][0]["box"]) == 4

    second = recorder.save(frame, result, source="camera0",
                           source_type="camera")
    assert second.annotated != first.annotated, "evidence was overwritten"

    previous = config.EVIDENCE_DIR
    config.EVIDENCE_DIR = tmp / "evidence"
    try:
        bundles = latest_bundles(limit=5)
    finally:
        config.EVIDENCE_DIR = previous
    assert len(bundles) == 2, bundles
    assert bundles[0].metadata.is_file()
    return f"{first.annotated.name}, {second.annotated.name}"


@check("alerts: cooldown prevents per-frame beeping")
def check_alerts():
    from src.alerts import AlertManager
    alerts = AlertManager(enabled=True, count_threshold=2,
                          cooldown_seconds=30.0)
    assert alerts.update(3, config.STATUS_DIRTY, "camera0") is True
    assert alerts.update(5, config.STATUS_DIRTY, "camera0") is False
    state = alerts.state()
    assert state.active and state.triggered_total == 1
    assert 0 < state.cooldown_remaining <= 30.0
    assert alerts.update(0, config.STATUS_CLEAN) is False
    assert alerts.update(3, config.STATUS_REVIEW) is False
    review_alerts = AlertManager(enabled=True, count_threshold=1,
                                 cooldown_seconds=0.0, alert_on_review=True)
    assert review_alerts.update(1, config.STATUS_REVIEW) is True
    disabled = AlertManager(enabled=False, count_threshold=1,
                            cooldown_seconds=0.0)
    assert disabled.update(9, config.STATUS_DIRTY) is False
    return "cooldown + threshold + review behaviour"


@check("counting: session totals and tracked unique objects")
def check_counting():
    from src.counting import SessionCounter
    from src.detector import Detection, SceneResult
    from src.tracking import UniqueObjectTracker, describe

    session = SessionCounter(source="camera0", tracking=True)
    for track_id in (1, 1, 2):
        detections = [Detection(7, "plastic", 0.9, 0, 0, 10, 10, track_id)]
        result = SceneResult(detections=detections, status=config.STATUS_DIRTY,
                             counts={"plastic": 1}, inference_ms=20.0)
        session.record(result)
    clean = SceneResult(detections=[], status=config.STATUS_CLEAN, counts={})
    session.record(clean)

    summary = session.summary()
    assert summary["frames"] == 4
    assert summary["detections"] == 3
    assert summary["unique_objects"] == 2, summary
    assert summary["dirty_frames"] == 3 and summary["clean_frames"] == 1
    assert summary["status"] == config.STATUS_DIRTY

    plain = SessionCounter(source="camera0", tracking=False)
    plain.record(SceneResult(detections=[Detection(4, "glass", 0.7, 0, 0, 5, 5)],
                             status=config.STATUS_DIRTY, counts={"glass": 1}))
    assert plain.summary()["unique_objects"] is None

    tracker = UniqueObjectTracker()
    tracker.update([Detection(7, "plastic", 0.9, 0, 0, 9, 9, 5)])
    info = describe([Detection(7, "plastic", 0.9, 0, 0, 9, 9, 5)], tracker)
    assert info["tracking"] and info["unique_objects"] == 1
    assert info["frame_objects"] == 1
    return "3 frames -> 2 unique objects"


@check("preprocessing: source classification and image loading")
def check_preprocessing(tmp: Path):
    import cv2
    from src.preprocessing import MediaError, classify_source, load_image, valid_frame

    assert classify_source("0") == ("camera", 0)
    assert classify_source(1) == ("camera", 1)
    image = tmp / "sample.jpg"
    assert cv2.imwrite(str(image), synthetic_frame())
    assert classify_source(image)[0] == "image"
    video = tmp / "clip.mp4"
    video.write_bytes(b"\x00")
    assert classify_source(video)[0] == "video"
    frame = load_image(image)
    assert valid_frame(frame) and frame.shape[1] == 640
    assert not valid_frame(None)
    (tmp / "notes.txt").write_text("not an image", encoding="utf-8")
    for bad in (tmp / "missing.jpg", tmp / "notes.txt"):
        try:
            classify_source(bad)
        except MediaError:
            continue
        raise AssertionError(f"accepted invalid source {bad}")
    return "camera / image / video / errors"


@check("postprocessing: drawing keeps frame shape and status colours")
def check_postprocessing():
    from src.postprocessing import annotate_scene, draw_detections, status_text
    frame = synthetic_frame()
    result = synthetic_result()
    annotated = annotate_scene(frame, result, fps=28.3, device="CUDA:0")
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)
    assert draw_detections(frame, result.detections,
                           result.weak_detections).shape == frame.shape
    assert "2 garbage object" in status_text(config.STATUS_DIRTY, result)
    assert "clean" in status_text(config.STATUS_CLEAN, result)
    return "boxes + HUD rendered"


@check("analytics: aggregates come from stored rows only")
def check_analytics(tmp: Path):
    from src import analytics
    from src.history import HistoryDB
    with HistoryDB(tmp / "analytics.db") as history:
        empty = analytics.overview(history)
        assert empty["has_data"] is False and empty["total_events"] == 0
        assert analytics.class_distribution(history)["has_data"] is False

        now = datetime.combine(date.today(), dtime(12, 0))
        history.log_event("camera0", config.STATUS_DIRTY, 3,
                          {"plastic": 2, "glass": 1}, average_confidence=0.77,
                          source_type="camera", when=now)
        history.log_event("img.jpg", config.STATUS_CLEAN, 0, {},
                          source_type="image", when=now - timedelta(hours=1))
        summary = analytics.overview(history)
        assert summary["total_events"] == 2 and summary["today_events"] == 2
        assert summary["dirty"] == 1 and summary["clean"] == 1
        assert summary["most_common_class"] == "plastic"
        distribution = analytics.class_distribution(history)
        assert distribution["labels"][0] == "plastic"
        assert distribution["values"][0] == 2
        split = analytics.status_split(history)
        assert sum(split["values"]) == 2
        assert len(analytics.daily_activity(history, 7)["labels"]) == 7
        assert len(analytics.hourly_activity(history)["values"]) == 24
        assert sum(analytics.objects_per_event(history)["values"]) == 2
    metrics = analytics.model_metrics()
    dataset = analytics.dataset_summary()
    return (f"model report: {'yes' if metrics.get('available') else 'pending'}"
            f", dataset report: "
            f"{'yes' if dataset.get('available') else 'pending'}")


@check("detector guard: wrong class schema is refused")
def check_detector_guard(tmp: Path):
    from src.detector import GarbageDetector, ModelError
    try:
        GarbageDetector(weights=tmp / "does_not_exist.pt", device="cpu")
    except ModelError as exc:
        assert "Model not found" in str(exc)
    else:
        raise AssertionError("missing weights were accepted")

    # An 80-class COCO model must never be usable as the garbage detector.
    coco = config.BASE_WEIGHTS
    if not coco.is_file():
        return "no COCO base weights present to test against"
    try:
        # device="cpu": never touch VRAM while training may be running
        GarbageDetector(weights=coco, device="cpu")
    except ModelError as exc:
        assert "does not match" in str(exc)
        return f"rejected {coco.name} (80 COCO classes)"
    raise AssertionError(f"{coco.name} was accepted by the detector")


@check("ui: widgets, charts and pages build and paint")
def check_ui(tmp: Path):
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ui.charts import BarChart, DonutChart, LineChart
    from ui.components import (Banner, ClassCountTable, InfoTable, MetricCard,
                               StatusBadge, VideoSurface)

    bar = BarChart("classes")
    bar.set_data(list(config.GARBAGE_CLASSES[:4]), [5, 3, 0, 9])
    donut = DonutChart("status")
    donut.set_data(["CLEAN", "DIRTY"], [4, 6])
    line = LineChart("daily")
    line.set_data(["2026-09-21", "2026-09-22", "2026-09-23"],
                  {"DIRTY": [1, 3, 2], "CLEAN": [2, 0, 1]})
    for chart in (bar, donut, line):
        chart.resize(420, 240)
        pixmap = QPixmap(chart.size())
        chart.render(pixmap)
        assert not pixmap.isNull()

    table = ClassCountTable()
    table.set_counts({"plastic": 3, "paper": 1})
    badge = StatusBadge()
    badge.set_status(config.STATUS_DIRTY, "3 objects")
    surface = VideoSurface()
    surface.resize(320, 200)
    surface.set_frame(synthetic_frame())
    assert surface.pixmap() is not None and not surface.pixmap().isNull()
    card = MetricCard("total", "12")
    card.set_value(13)
    info = InfoTable("system")
    info.set_items({"python": "3.11", "gpu": "RTX 4050"})
    banner = Banner()
    banner.show_message("test", "error")
    banner.clear_message()
    for widget in (table, badge, surface, card, info, banner):
        widget.resize(420, 260)
        widget.render(QPixmap(widget.size()))
    assert app is not None
    return "charts + components painted offscreen"


@check("ui: dashboard page modules import cleanly")
def check_ui_modules():
    import importlib
    for module in ("ui.styles", "ui.charts", "ui.components", "ui.engine",
                   "ui.pages", "ui.insights", "ui.dashboard"):
        importlib.import_module(module)
    from ui.dashboard import NAV_ITEMS, AppContext, MainWindow, launch
    assert len(NAV_ITEMS) == 8 and callable(launch)
    assert AppContext and MainWindow
    return f"{len(NAV_ITEMS)} navigation entries"


@check("ui: every dashboard page builds and renders with a stub engine")
def check_pages(tmp: Path):
    """Construct all 8 pages against a fake engine and paint them offscreen."""
    from types import SimpleNamespace

    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    from src.history import HistoryDB
    from src.settings import Settings
    from ui import insights, pages

    app = QApplication.instance() or QApplication([])
    history = HistoryDB(tmp / "pages.db")
    history.log_event("camera0", config.STATUS_DIRTY, 3, {"plastic": 2,
                      "glass": 1}, average_confidence=0.81,
                      source_type="camera")
    context = SimpleNamespace(
        engine=SimpleNamespace(
            history=history,
            detector=SimpleNamespace(info=lambda: {
                "model": config.MODEL_NAME, "version": config.MODEL_VERSION,
                "weights": str(tmp / "best.pt"), "size_mb": 6.2,
                "device": "CUDA:0", "gpu": "RTX 4050 Laptop GPU",
                "vram": "6.00 GB", "confidence": 0.35, "iou": 0.45,
                "image_size": 640, "tracking": False,
                "classes": list(config.GARBAGE_CLASSES)})),
        settings=Settings(),
        last_result=synthetic_result(),
        last_frame=synthetic_frame(),
        mode=None,
        start_camera=lambda index: None,
        start_video=lambda path, save=False: None,
        analyze_image=lambda path: None,
        stop_source=lambda: None,
        toggle_pause=lambda: False,
        capture_evidence=lambda: None,
        set_confidence=lambda value: None)

    built = {}
    for key, factory in (("dashboard", pages.DashboardPage),
                         ("live", pages.LiveCameraPage),
                         ("image", pages.ImagePage),
                         ("video", pages.VideoPage),
                         ("history", insights.HistoryPage),
                         ("analytics", insights.AnalyticsPage),
                         ("settings", insights.SettingsPage),
                         ("about", insights.AboutPage)):
        page = factory(context)
        page.refresh()
        page.resize(1280, 800)
        pixmap = QPixmap(page.size())
        page.render(pixmap)
        assert not pixmap.isNull(), key
        built[key] = page

    built["dashboard"].on_frame(synthetic_frame(), synthetic_result(), 28.4)
    built["live"].on_frame(synthetic_frame(), synthetic_result(), 28.4)
    built["live"].on_finished({"frames": 120, "detections": 40,
                               "peak_objects": 5, "dirty_frames": 30,
                               "review_frames": 4, "average_confidence": 0.72,
                               "unique_objects": None, "seconds": 4.2,
                               "status": config.STATUS_DIRTY})
    built["video"].on_progress(50, 200)
    built["video"].on_finished({"kind": "video", "frames": 200,
                                "detections": 80, "dirty_frames": 60,
                                "review_frames": 2, "peak_objects": 6,
                                "average_confidence": 0.68, "seconds": 8.1,
                                "status": config.STATUS_DIRTY})
    built["image"].on_frame(synthetic_frame(), synthetic_result(), 0.0)
    built["image"].on_finished({"output": str(tmp / "x_detected.jpg")})
    assert built["history"].table.rowCount() == 1
    history.close()
    return f"{len(built)} pages rendered (Qt {app.applicationName() or 'offscreen'})"


@check("logging: rotating file handler writes to logs/")
def check_logging():
    from src.logging_setup import get_logger, setup_logging
    setup_logging()
    logger = get_logger("selftest")
    logger.info("selftest executed")
    log_file = config.LOG_DIR / "app.log"
    assert log_file.is_file()
    assert "selftest executed" in log_file.read_text(encoding="utf-8")
    return log_file.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary folder for inspection")
    args = parser.parse_args()

    print("=" * 66)
    print(" SELFTEST - AI Garbage Detection System")
    print("=" * 66)
    tmp = Path(tempfile.mkdtemp(prefix="garbage_selftest_"))
    try:
        for fn in (check_config, check_settings, check_detection, check_status,
                   check_history, check_evidence, check_alerts, check_counting,
                   check_preprocessing, check_postprocessing, check_analytics,
                   check_detector_guard, check_ui, check_ui_modules,
                   check_pages, check_logging):
            fn(tmp)
    finally:
        if args.keep:
            print(f"temporary files kept in {tmp}")
        else:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("-" * 66)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in RESULTS:
        if name in failed:
            print(f"  FAILED: {name} -> {detail}")
    print("RESULT:", "PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
