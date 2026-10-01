"""Evaluate a trained YOLOv8 garbage detection model on the val and test splits.

Final project accuracy comes from the TEST split, which was never used for
training or model selection. The val split is reported for comparison only.
Evaluating the train split is deliberately not supported, and weights whose
class schema is not the 10 garbage classes are rejected - the old
COCO-derived model must never be presented as project accuracy.

Also measures real inference speed (no fabricated FPS) and records model size.
Writes dataset_final/reports/model_evaluation.json plus a confusion matrix CSV
per split.

Usage:
    python evaluate_model.py
    python evaluate_model.py --weights runs/detect/garbage_final/weights/best.pt
    python evaluate_model.py --device cpu --benchmark-images 50
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config

REPORTS_DIR = config.PROJECT_ROOT / "dataset_final" / "reports"
DATASET_ROOT = config.PROJECT_ROOT / "dataset_final"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot be trusted or cannot run."""


def newest_weights() -> Path | None:
    candidates = config.trained_weights()
    return candidates[-1] if candidates else None


def f1_score(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def load_model(weights: Path, device: str):
    """Load weights and refuse anything that is not the 10-class detector."""
    from src.detector import GarbageDetector, resolve_device

    resolved = resolve_device(device)
    detector = GarbageDetector(weights=weights, device=resolved)
    return detector.model, detector, resolved


def split_image_count(split: str) -> int:
    folder = DATASET_ROOT / "images" / split
    if not folder.is_dir():
        raise EvaluationError(f"split folder missing: {folder}")
    return sum(1 for p in folder.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def evaluate_split(model, split: str, imgsz: int) -> dict:
    """Run ultralytics validation on one split and shape the metrics."""
    metrics = model.val(data=str(config.DATA_YAML), split=split, imgsz=imgsz,
                        plots=True, verbose=False, project=str(REPORTS_DIR),
                        name=f"eval_{split}", exist_ok=True)

    per_class = []
    for index, name in metrics.names.items():
        precision = float(metrics.box.p[index])
        recall = float(metrics.box.r[index])
        per_class.append({
            "class": name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1_score(precision, recall), 4),
            "ap50": round(float(metrics.box.ap50[index]), 4),
            "ap50_95": round(float(metrics.box.ap[index]), 4),
        })

    mean_p, mean_r = float(metrics.box.mp), float(metrics.box.mr)
    speed = {k: round(float(v), 2) for k, v in (metrics.speed or {}).items()}
    return {
        "split": split,
        "images": split_image_count(split),
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(mean_p, 4),
        "recall": round(mean_r, 4),
        "f1": round(f1_score(mean_p, mean_r), 4),
        "per_class": per_class,
        "speed_ms_reported_by_ultralytics": speed,
        "confusion_matrix_csv": write_confusion_matrix(metrics, split),
    }


def write_confusion_matrix(metrics, split: str) -> str:
    """Rows = ground truth, columns = predictions, last entry = background."""
    labels = list(metrics.names.values()) + ["background"]
    matrix = np.asarray(metrics.confusion_matrix.matrix)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"confusion_matrix_{split}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("gt\\pred," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            fh.write(label + "," + ",".join(str(int(v)) for v in row) + "\n")
    return str(path.relative_to(config.PROJECT_ROOT))


def benchmark(model, detector, split: str, count: int, imgsz: int) -> dict:
    """Measure real per-image inference time on the active device."""
    import cv2

    folder = DATASET_ROOT / "images" / split
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)[:max(count, 10)]
    if not images:
        raise EvaluationError(f"no images to benchmark in {folder}")

    timings: list[float] = []
    for index, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if index < 3:  # warm-up: cuDNN autotune and allocator spin-up
            model.predict(source=frame, imgsz=imgsz, device=detector.device,
                          verbose=False)
            continue
        started = time.perf_counter()
        model.predict(source=frame, imgsz=imgsz, device=detector.device,
                      verbose=False)
        timings.append((time.perf_counter() - started) * 1000.0)

    if not timings:
        raise EvaluationError("benchmark produced no usable frames")
    array = np.asarray(timings)
    return {
        "images": len(timings),
        "device": detector.device_details["device"],
        "gpu": detector.device_details["gpu"],
        "mean_ms": round(float(array.mean()), 2),
        "median_ms": round(float(np.median(array)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "min_ms": round(float(array.min()), 2),
        "max_ms": round(float(array.max()), 2),
        "fps": round(float(1000.0 / array.mean()), 2),
    }


def model_facts(weights: Path, model) -> dict:
    """Size and architecture facts about the evaluated weights."""
    facts = {
        "weights": str(weights),
        "model_size_mb": round(weights.stat().st_size / 1024 ** 2, 2),
        "task": getattr(model, "task", "detect"),
        "names": list((getattr(model, "names", {}) or {}).values()),
    }
    torch_model = getattr(model, "model", None)
    if torch_model is not None and hasattr(torch_model, "parameters"):
        try:
            facts["parameters"] = sum(p.numel() for p in
                                      torch_model.parameters())
        except (AttributeError, TypeError):
            pass
    return facts


def print_split(title: str, result: dict) -> None:
    print(f"\n=== {title} ({result['images']} images) ===")
    print(f"mAP@50      : {result['map50']}")
    print(f"mAP@50-95   : {result['map50_95']}")
    print(f"precision   : {result['precision']}")
    print(f"recall      : {result['recall']}")
    print(f"F1          : {result['f1']}")
    print(f"{'class':12s} {'P':>7s} {'R':>7s} {'F1':>7s} "
          f"{'AP50':>7s} {'AP50-95':>8s}")
    for row in result["per_class"]:
        print(f"{row['class']:12s} {row['precision']:7.3f} "
              f"{row['recall']:7.3f} {row['f1']:7.3f} {row['ap50']:7.3f} "
              f"{row['ap50_95']:8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="",
                        help="path to best.pt (default: newest trained run)")
    parser.add_argument("--imgsz", type=int, default=config.INPUT_SIZE)
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | 0 (default: auto)")
    parser.add_argument("--benchmark-images", type=int, default=100,
                        help="how many test images to time (default 100)")
    parser.add_argument("--skip-val", action="store_true",
                        help="evaluate the test split only")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else newest_weights()
    if not weights or not weights.is_file():
        print("ERROR: no trained weights found.\n"
              "Expected: runs/detect/<run>/weights/best.pt\n"
              "Train first with: python train_yolo.py")
        return 1

    print(f"Evaluating: {weights}")
    try:
        model, detector, device = load_model(weights, args.device)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Device    : {detector.device_details['device']} "
          f"({detector.device_details['gpu']})")

    try:
        test_result = evaluate_split(model, "test", args.imgsz)
        val_result = (None if args.skip_val
                      else evaluate_split(model, "val", args.imgsz))
        speed = benchmark(model, detector, "test", args.benchmark_images,
                          args.imgsz)
    except EvaluationError as exc:
        print(f"ERROR: {exc}")
        return 1

    facts = model_facts(weights, model)
    test_result.update({"inference_ms": speed["mean_ms"], "fps": speed["fps"],
                        "model_size_mb": facts["model_size_mb"],
                        "benchmark": speed})
    if val_result:
        val_result.update({"inference_ms": speed["mean_ms"],
                           "fps": speed["fps"],
                           "model_size_mb": facts["model_size_mb"]})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "device": device,
        "model_name": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "model": facts,
        "note": "TEST split results are the final project accuracy; val is "
                "model-selection only. The train split is never evaluated.",
        "val": val_result,
        "test": test_result,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "model_evaluation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if val_result:
        print_split("VALIDATION (model selection only)", val_result)
    print_split("TEST (final reported accuracy)", test_result)
    print(f"\nInference : {speed['mean_ms']} ms mean "
          f"({speed['median_ms']} ms median, p95 {speed['p95_ms']} ms)")
    print(f"FPS       : {speed['fps']} on {speed['device']} "
          f"({speed['gpu']}, {speed['images']} images)")
    print(f"Model size: {facts['model_size_mb']} MB"
          + (f", {facts['parameters']:,} parameters"
             if "parameters" in facts else ""))
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
