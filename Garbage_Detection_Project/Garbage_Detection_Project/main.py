"""Smart Garbage Detection System - command line entry point.

Modes:
  python main.py --gui                    professional dashboard (PySide6)
  python main.py                          webcam live detection
  python main.py --source path/to/img.jpg single image
  python main.py --source path/to/vid.mp4 video file
  python main.py --source 1               webcam index 1

Every session is written to the SQLite history and, when the scene is DIRTY
(or --save-evidence), an evidence JPG + JSON pair is stored under evidence/.
In a preview window: Q quits, C captures evidence, S saves the annotated frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config
from src.counting import SessionCounter
from src.detector import ModelError, SceneResult
from src.evidence import EvidenceError
from src.logging_setup import get_logger, setup_logging
from src.postprocessing import annotate_scene, status_text
from src.preprocessing import MediaError, classify_source, load_image, open_capture
from src.settings import Settings, load_settings
from src.utils import FpsMeter

log = get_logger("main")

WINDOW_TITLE = "AI Garbage Detection - Q quit | C capture | S save frame"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gui", action="store_true",
                        help="launch the dashboard instead of the CLI")
    parser.add_argument("--source", default="",
                        help="webcam index (e.g. 0) or path to image/video "
                             "(default: camera index from settings)")
    parser.add_argument("--weights", default="",
                        help="model weights (default: newest trained best.pt)")
    parser.add_argument("--conf", type=float, default=None,
                        help=f"confidence threshold (default "
                             f"{config.CONF_THRESHOLD})")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU threshold")
    parser.add_argument("--device", default=None,
                        help="auto | cpu | 0 (default: auto)")
    parser.add_argument("--tracking", action="store_true",
                        help="enable ByteTrack and unique-object counting")
    parser.add_argument("--no-alert", action="store_true",
                        help="disable audible alerts")
    parser.add_argument("--save-evidence", action="store_true",
                        help="save evidence even when the scene is CLEAN")
    parser.add_argument("--save-video", action="store_true",
                        help="write an annotated copy of a video file")
    parser.add_argument("--no-display", action="store_true",
                        help="process without a preview window")
    parser.add_argument("--verbose", action="store_true",
                        help="debug logging to console and logs/app.log")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    """Start from the persisted settings and apply CLI overrides."""
    settings = load_settings()
    if args.weights:
        settings.weights = args.weights
    if args.conf is not None:
        settings.confidence = args.conf
    if args.iou is not None:
        settings.iou = args.iou
    if args.device:
        settings.device = args.device
    if args.tracking:
        settings.tracking_enabled = True
    if args.no_alert:
        settings.alert_enabled = False
    return settings


def build_detector(settings: Settings):
    """Create the detector, translating failures into readable messages."""
    from src.detector import GarbageDetector
    try:
        return GarbageDetector(
            weights=settings.weights_path,
            conf_threshold=settings.confidence,
            iou_threshold=settings.iou,
            image_size=settings.image_size,
            device=settings.device,
            tracking=settings.tracking_enabled)
    except ModelError as exc:
        raise SystemExit(f"\n[!] {exc}\n") from exc


def print_scene(result: SceneResult, device: str, fps: float | None = None) -> None:
    line = f"STATUS: {result.status} | objects: {result.total}"
    if fps:
        line += f" | fps: {fps:.1f}"
    line += f" | inference: {result.inference_ms:.0f} ms | device: {device}"
    print(line)
    print(f"        {status_text(result.status, result)}")


def print_counts(counts: dict[str, int],
                 unique: dict[str, int] | None = None) -> None:
    total = sum(counts.values())
    print(f"TOTAL OBJECTS: {total}"
          + (f"   (unique tracked: {sum(unique.values())})" if unique else ""))
    for name, count in counts.items():
        suffix = f"  [{unique.get(name, 0)} unique]" if unique else ""
        marker = "*" if count else " "
        print(f" {marker} {name:12s} {count:3d}{suffix}")


def save_evidence(engine_services, detector, frame, result, source, source_type,
                  fps=None, unique=None):
    """Write evidence and log the event; returns the annotated path."""
    evidence, history = engine_services
    try:
        bundle = evidence.save(frame, result, source=source,
                               source_type=source_type,
                               model_info=detector.info(), fps=fps,
                               unique_objects=unique)
    except EvidenceError as exc:
        print(f"[!] Evidence could not be saved: {exc}")
        log.error("evidence failure: %s", exc)
        return None
    history.log_event(source=source, source_type=source_type,
                      status=result.status, total_count=result.total,
                      counts=result.counts,
                      average_confidence=result.average_confidence,
                      evidence_path=str(bundle.annotated),
                      model_version=config.MODEL_VERSION,
                      model_path=str(detector.weights_path),
                      unique_objects=unique)
    print(f"Evidence saved: {bundle.annotated.name}")
    return str(bundle.annotated)


def run_image(detector, evidence, history, alerts, path: Path,
              force_save: bool) -> int:
    """Detect in a single image, save the annotated copy and log the event."""
    try:
        frame = load_image(path)
    except MediaError as exc:
        print(f"[!] {exc}")
        return 1

    result = detector.analyze(frame)
    alerts.update(result.total, result.status, path.name)
    annotated = annotate_scene(frame, result, device=detector.device_details["device"])

    out_path = path.with_stem(path.stem + "_detected")
    if not cv2.imwrite(str(out_path), annotated):
        print(f"[!] Could not write {out_path}")

    print(f"\nImage: {path.name}")
    print_scene(result, detector.device_details["device"])
    print_counts(result.counts)
    print(f"Annotated image: {out_path}")

    unique = result.unique_objects
    if force_save or result.status != config.STATUS_CLEAN:
        save_evidence((evidence, history), detector, frame, result,
                      str(path), "image", unique=unique)
    else:
        history.log_event(source=str(path), source_type="image",
                          status=result.status, total_count=result.total,
                          counts=result.counts,
                          average_confidence=result.average_confidence,
                          model_version=config.MODEL_VERSION,
                          model_path=str(detector.weights_path),
                          unique_objects=unique)

    cv2.imshow("Result - press any key", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def run_stream(detector, evidence, history, alerts, source, source_type: str,
               label: str, force_save: bool, no_display: bool,
               save_video: bool) -> int:
    """Detect over a camera or video file, frame by frame (streamed)."""
    try:
        capture = open_capture(source)
    except MediaError as exc:
        print(f"[!] {exc}")
        return 1

    from src.preprocessing import video_metadata
    meta = video_metadata(capture)
    total_frames = meta.get("frames", 0) if source_type == "video" else 0

    writer = None
    output_path = None
    if save_video and source_type == "video":
        from src.utils import file_timestamp
        output_path = config.EVIDENCE_DIR / f"processed_{file_timestamp()}.mp4"
        config.ensure_dirs()
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            meta.get("fps") or 25.0, (meta["width"], meta["height"]))
        if not writer.isOpened():
            print("[!] Could not open the video writer - saving disabled")
            writer, output_path = None, None

    print(f"\nDetecting from {label} "
          f"({meta.get('width')}x{meta.get('height')} @ {meta.get('fps', 0):.1f} fps)")
    print("Controls: Q quit | C capture evidence | S save annotated frame")

    session = SessionCounter(source=label, tracking=detector.tracking)
    meter = FpsMeter()
    device = detector.device_details["device"]

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera stopped delivering frames."
                  if source_type == "camera" else "End of video.")
            break

        result = detector.analyze(frame)
        session.record(result)
        alerts.update(result.total, result.status, label)
        fps = meter.tick()
        annotated = annotate_scene(frame, result, fps=fps, device=device)

        if writer is not None:
            writer.write(annotated)

        key = 0
        if not no_display:
            cv2.imshow(WINDOW_TITLE, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                unique = result.unique_objects
                save_evidence((evidence, history), detector, frame, result,
                              label, source_type, fps=fps, unique=unique)
            elif key == ord("s"):
                from src.utils import unique_path
                saved = unique_path(config.EVIDENCE_DIR,
                                    f"frame_{session.frames:06d}", ".jpg")
                cv2.imwrite(str(saved), annotated)
                print(f"Frame saved: {saved.name}")
            elif key == ord("q"):
                print("Stopped by user.")
                break

        if (force_save and result.status == config.STATUS_DIRTY
                and session.frames % config.EVIDENCE_FRAME_INTERVAL == 0):
            save_evidence((evidence, history), detector, frame, result,
                          label, source_type, fps=fps,
                          unique=result.unique_objects)

        if session.frames % 60 == 0:
            print(f"  frame {session.frames}"
                  f"{'/' + str(total_frames) if total_frames else ''} | "
                  f"{result.status} | objects {result.total} | fps {fps:.1f}")

    capture.release()
    if writer is not None:
        writer.release()
        print(f"Processed video: {output_path}")
    cv2.destroyAllWindows()

    if not session.frames:
        print("No frames were processed.")
        return 1

    summary = session.summary()
    unique = session.unique_per_class()
    print(f"\n{'=' * 54}\nSESSION SUMMARY - {label}\n{'=' * 54}")
    print(f"Frames processed : {summary['frames']}")
    print(f"Detections       : {summary['detections']}")
    print(f"Peak objects     : {summary['peak_objects']}")
    print(f"Dirty / Review   : {summary['dirty_frames']} / "
          f"{summary['review_frames']}")
    print(f"Avg confidence   : {summary['average_confidence'] * 100:.1f}%")
    print(f"Elapsed          : {summary['seconds']:.1f} s "
          f"({summary['frames'] / max(summary['seconds'], 1e-6):.1f} fps "
          "including display)")
    if unique is not None:
        print(f"Unique objects   : {summary['unique_objects']} (ByteTrack)")
    else:
        print("Unique objects   : n/a (tracking disabled - frame counts only)")
    print_counts(summary["class_totals"], unique)
    print(f"Overall status   : {summary['status']}")

    history.log_event(source=label, source_type=source_type,
                      status=summary["status"],
                      total_count=summary["detections"],
                      counts=summary["class_totals"],
                      average_confidence=summary["average_confidence"],
                      model_version=config.MODEL_VERSION,
                      model_path=str(detector.weights_path),
                      unique_objects=summary["unique_objects"])
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(level=10 if args.verbose else 20)

    if args.gui:
        from ui.dashboard import launch
        return launch()

    settings = build_settings(args)
    detector = build_detector(settings)

    from src.alerts import AlertManager
    from src.evidence import EvidenceRecorder
    from src.history import HistoryDB

    alerts = AlertManager(enabled=settings.alert_enabled,
                          count_threshold=settings.alert_threshold,
                          cooldown_seconds=settings.alert_cooldown,
                          sound_path=settings.alert_sound)
    evidence = EvidenceRecorder()
    history = HistoryDB()

    info = detector.info()
    print("=" * 60)
    print(" AI GARBAGE DETECTION - Intelligent Waste Monitoring System")
    print("=" * 60)
    print(f"Model   : {info['model']} {info['version']} "
          f"({info['size_mb']} MB)")
    print(f"Weights : {info['weights']}")
    print(f"Device  : {info['device']} - {info['gpu']} ({info['vram']})")
    print(f"Limits  : conf {info['confidence']:.2f} | iou {info['iou']:.2f} "
          f"| imgsz {info['image_size']} | tracking "
          f"{'on' if info['tracking'] else 'off'}")
    print(f"Classes : {', '.join(info['classes'])}")

    source = args.source or str(settings.camera_index)
    try:
        kind, target = classify_source(source)
    except MediaError as exc:
        print(f"[!] {exc}")
        history.close()
        return 1

    try:
        if kind == "image":
            return run_image(detector, evidence, history, alerts, target,
                             args.save_evidence)
        return run_stream(detector, evidence, history, alerts, target, kind,
                          f"camera{target}" if kind == "camera"
                          else Path(str(target)).name,
                          args.save_evidence, args.no_display, args.save_video)
    finally:
        history.prune(settings.keep_history_entries)
        history.close()


if __name__ == "__main__":
    raise SystemExit(main())
