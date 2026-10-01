"""Train YOLOv8 on the final garbage detection dataset (dataset_final).

Pre-flight gate: training refuses to start unless the annotation audit
(scripts/audit_annotations.py) has passed with zero hard errors, so the
invalid-labels mistake that produced the old 0.629 mAP50 model can never
repeat. Override with --force only for deliberate experiments.

Defaults are sized for an RTX 4050 Laptop GPU (6 GB VRAM):
yolov8n, imgsz 640, batch 16, AMP on, early stopping via --patience.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data.yaml"
AUDIT_REPORT = PROJECT_ROOT / "dataset_final" / "reports" / "annotation_audit_report.json"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def default_base_model() -> str:
    local = PROJECT_ROOT / "models" / "yolov8n.pt"
    return str(local) if local.is_file() else "yolov8n.pt"


def preflight(force: bool) -> str:
    """Return an error message, or empty string when ready to train."""
    if not DATA_YAML.is_file():
        return f"data.yaml not found: {DATA_YAML}"
    if not AUDIT_REPORT.is_file():
        return ("annotation audit report not found - run "
                "scripts/audit_annotations.py first")
    report = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    if report.get("hard_errors", 1) != 0 and not force:
        return ("annotation audit reports hard errors - fix the labels "
                "before training (see "
                "dataset_final/reports/annotation_audit_issues.csv). "
                "Use --force to override deliberately.")
    import torch
    if not torch.cuda.is_available():
        return "CUDA unavailable - check_gpu.py must report READY before training"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="",
                        help="base weights (default: models/yolov8n.pt, "
                             "else downloaded yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15,
                        help="early stopping patience (epochs without val "
                             "mAP improvement)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="garbage_final",
                        help="run name under runs/detect/ (auto-incremented)")
    parser.add_argument("--force", action="store_true",
                        help="train even if the audit gate fails")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    error = preflight(args.force)
    if error:
        print("TRAINING BLOCKED:", error)
        return 1
    print(f"Preflight OK - data={DATA_YAML}")

    from ultralytics import YOLO
    model = YOLO(args.model or default_base_model())
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
        workers=args.workers,
        seed=42,
        amp=True,
        cache=False,
        verbose=True,
    )

    weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {weights}")
    print("Evaluate with: python evaluate_model.py --weights "
          f'"{weights}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
