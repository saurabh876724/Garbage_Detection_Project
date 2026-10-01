"""Auto-annotate dataset_final/ with auditable bounding boxes.

Three-tier strategy per image (class ID always comes from the source folder,
which is the ground truth for a classification dataset):

  1. YOLO-World zero-shot detection with garbage-specific prompts.
     Accept boxes whose predicted class matches the folder class (conf >=
     ACCEPT_SAME) or, more strictly, other garbage classes (conf >=
     ACCEPT_OTHER) for multi-object scenes.
  2. Otsu-threshold saliency: single dominant object box (foreground polarity
     chosen by the center pixel, cleaned with morphology, largest connected
     component containing the image center).
  3. GrabCut with a center rectangle prior.

Images where every tier fails get an empty label plus a
`needs_manual_box` flag. `clean` images always receive empty labels; any
garbage prediction inside them is flagged for review.

Every image's method, per-box confidence and flags are recorded in
dataset_final/manifests/annotation_metadata.json so the annotation
provenance is fully auditable. Originals are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset_final"
MANIFEST = DATASET_ROOT / "manifests" / "image_manifest.csv"
METADATA_OUT = DATASET_ROOT / "manifests" / "annotation_metadata.json"
SUMMARY_OUT = DATASET_ROOT / "reports" / "annotation_summary.json"

GARBAGE_CLASSES = ("battery", "biological", "cardboard", "clothes", "glass",
                   "metal", "paper", "plastic", "shoes", "trash")
CLASS_PROMPTS = {
    "battery": "battery",
    "biological": "food waste, rotten food, organic waste",
    "cardboard": "cardboard box, corrugated cardboard",
    "clothes": "clothes, clothing, garment, fabric",
    "glass": "glass bottle, glass jar, broken glass",
    "metal": "metal can, aluminum can, steel can, tin can",
    "paper": "paper, newspaper, stack of paper",
    "plastic": "plastic bottle, plastic bag, plastic container, plastic packaging",
    "shoes": "shoes, sneaker, footwear",
    "trash": "trash bag, garbage bag, litter, waste",
}
PROMPT_ORDER = list(GARBAGE_CLASSES)
ACCEPT_SAME = 0.20
ACCEPT_OTHER = 0.40
MAX_BOXES = 12
PREDICT_BATCH = 32
WORKERS = 12
CHECKPOINT_EVERY_CHUNKS = 4

MIN_AREA_FRACTION = 0.02
MAX_AREA_FRACTION = 0.98
MIN_SIDE_FRACTION = 0.04


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_world_model():
    from ultralytics import YOLO
    weights = PROJECT_ROOT / "models" / "yolov8x-worldv2.pt"
    if not weights.is_file():
        weights = Path("yolov8x-worldv2.pt")  # cwd copy or fresh download
    model = YOLO(str(weights))
    model.set_classes([CLASS_PROMPTS[name] for name in PROMPT_ORDER])
    device = 0 if torch.cuda.is_available() else "cpu"
    return model, device


def _largest_center_component(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return None
    center_label = labels[h // 2, w // 2]
    best, best_area = None, -1
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        score = area * (2 if i == center_label else 1)
        if score > best_area:
            best_area, best = score, (int(x), int(y), int(ww), int(hh))
    return best


def _valid_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    if not (MIN_AREA_FRACTION <= (w * h) / (img_w * img_h) <= MAX_AREA_FRACTION):
        return False
    if w / img_w < MIN_SIDE_FRACTION or h / img_h < MIN_SIDE_FRACTION:
        return False
    cx0, cy0 = 0.25 * img_w, 0.25 * img_h
    cx1, cy1 = 0.75 * img_w, 0.75 * img_h
    return x < cx1 and x + w > cx0 and y < cy1 and y + h > cy0


def otsu_box(path: Path) -> tuple[int, int, int, int] | None:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    h, w = gray.shape
    scale = max(1, max(h, w) // 800)
    if scale > 1:
        small = cv2.resize(gray, (w // scale, h // scale),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sh, sw = small.shape
    fg = bw if bw[sh // 2, sw // 2] == 255 else cv2.bitwise_not(bw)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    box = _largest_center_component(fg)
    if box is None:
        return None
    x, y, bw_, bh_ = box
    x, y, bw_, bh_ = x * scale, y * scale, bw_ * scale, bh_ * scale
    if not _valid_box(x, y, bw_, bh_, w, h):
        return None
    return x, y, bw_, bh_


def grabcut_box(path: Path) -> tuple[int, int, int, int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    # 512 max side: GrabCut costs ~O(pixels); coarse whole-object boxes do
    # not benefit from higher resolution.
    scale = max(1, max(h, w) // 512)
    if scale > 1:
        small = cv2.resize(img, (w // scale, h // scale),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(0.08 * sw), int(0.08 * sh), int(0.84 * sw), int(0.84 * sh))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                  255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    box = _largest_center_component(fg)
    if box is None:
        return None
    x, y, bw_, bh_ = box
    x, y, bw_, bh_ = x * scale, y * scale, bw_ * scale, bh_ * scale
    if not _valid_box(x, y, bw_, bh_, w, h):
        return None
    return x, y, bw_, bh_


def norm_box(x1, y1, x2, y2, img_w, img_h):
    """Normalize pixel corners to YOLO xywhn, clipped to the image bounds.

    Corners (not center/size) are rounded to 6 decimals so the reconstructed
    edges never leave [0, 1] through floating-point rounding.
    """
    x1 = round(min(max(x1, 0.0), img_w) / img_w, 6)
    x2 = round(min(max(x2, 0.0), img_w) / img_w, 6)
    y1 = round(min(max(y1, 0.0), img_h) / img_h, 6)
    y2 = round(min(max(y2, 0.0), img_h) / img_h, 6)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def image_size(path: Path):
    # Default imread flags: applies EXIF orientation, matching what the
    # YOLO predictor and the audit/visual tools see. IMREAD_UNCHANGED would
    # return the raw stored orientation and desynchronize dimensions.
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def extract_detections(result):
    dets = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        dets.append({"cls": cls, "conf": conf, "xyxy": [x1, y1, x2, y2]})
    w_img, h_img = result.orig_shape[1], result.orig_shape[0]
    return dets, (w_img, h_img)


def annotate_record(record, dets, img_shape):
    """Worker: decide boxes for one image, write its label file, return entry."""
    name = record["target_image_name"]
    split = record["split"]
    category = record["source_category"]
    image_path = DATASET_ROOT / "images" / split / name
    label_path = DATASET_ROOT / "labels" / split / f"{Path(name).stem}.txt"
    entry = {"source_category": category, "split": split,
             "boxes": [], "flags": [], "method": None}
    w_img, h_img = img_shape

    if category == "clean":
        for det in dets:
            if det["conf"] >= ACCEPT_OTHER:
                entry["flags"].append("clean_with_garbage")
                entry["boxes"].append(
                    {"cls": det["cls"], "conf": round(det["conf"], 3),
                     "kept": False})
        label_path.write_text("", encoding="utf-8")
        entry["method"] = "negative"
        return name, entry

    folder_cls = GARBAGE_CLASSES.index(category)
    same, other = [], []
    for det in dets:
        record_box = {"cls": det["cls"], "conf": round(det["conf"], 3),
                      "xyxy": [round(v, 1) for v in det["xyxy"]]}
        if det["cls"] == folder_cls and det["conf"] >= ACCEPT_SAME:
            same.append(record_box)
        elif det["cls"] != folder_cls and det["conf"] >= ACCEPT_OTHER:
            other.append(record_box)

    def yolo_boxes(box_list):
        return [{"cls": rb["cls"], "conf": rb["conf"], "kept": True,
                 "xywhn": list(norm_box(*rb["xyxy"], w_img, h_img))}
                for rb in box_list]

    if same and len(same) + len(other) <= MAX_BOXES:
        entry["boxes"] = yolo_boxes(same + other)
        entry["method"] = "yolo_world"
        if other:
            entry["flags"].append("multi_class")
    else:
        box = otsu_box(image_path)
        tier = "otsu"
        if box is None:
            box = grabcut_box(image_path)
            tier = "grabcut"
        size = image_size(image_path)
        if box is not None and size:
            w_size, h_size = size
            bx, by, bw_, bh_ = box
            entry["boxes"] = [{
                "cls": folder_cls, "conf": None, "kept": True,
                "xywhn": list(norm_box(bx, by, bx + bw_, by + bh_,
                                       w_size, h_size))}]
            entry["method"] = tier
        elif other and len(other) <= MAX_BOXES:
            entry["boxes"] = yolo_boxes(other)
            entry["method"] = "yolo_world_other_only"
            entry["flags"].append("other_only_review")
        else:
            entry["flags"].append("needs_manual_box")
        if same and len(same) + len(other) > MAX_BOXES:
            entry["flags"].append("too_many_detections")

    label_path.write_text(
        "".join(f"{b['cls']} {b['xywhn'][0]} {b['xywhn'][1]} "
                f"{b['xywhn'][2]} {b['xywhn'][3]}\n"
                for b in entry["boxes"] if b["kept"]),
        encoding="utf-8")
    return name, entry


FLAG_STATS = {
    "multi_class": "flag_multi_class",
    "other_only_review": "flag_other_only",
    "needs_manual_box": "flag_needs_manual_box",
    "clean_with_garbage": "flag_clean_with_garbage",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="skip images already present in annotation metadata")
    parser.add_argument("--limit", type=int, default=0,
                        help="annotate at most N images (0 = all, for smoke tests)")
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"ERROR: manifest not found: {MANIFEST}")
        return 1
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        manifest = list(csv.DictReader(fh))

    metadata = {}
    if args.resume and METADATA_OUT.is_file():
        metadata = json.loads(METADATA_OUT.read_text(encoding="utf-8"))
        log(f"resume: {len(metadata)} images already annotated")

    todo = [r for r in manifest
            if not (args.resume and r["target_image_name"] in metadata)]
    if args.limit:
        todo = todo[:args.limit]
    log(f"{len(todo)} images to annotate")

    model, device = load_world_model()
    log(f"YOLO-World ready on device={device}")

    stats = Counter()
    t0 = time.time()
    chunks_done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for chunk_start in range(0, len(todo), PREDICT_BATCH):
            chunk = todo[chunk_start:chunk_start + PREDICT_BATCH]
            results = model.predict(
                source=[str(DATASET_ROOT / "images" / r["split"] / r["target_image_name"])
                        for r in chunk],
                conf=min(ACCEPT_SAME, 0.15), imgsz=640, device=device,
                verbose=False, stream=True)

            # Extract tensor data in the main thread; only plain Python and
            # GIL-releasing OpenCV work runs in the worker pool.
            payloads = [(record, *extract_detections(result))
                        for record, result in zip(chunk, results)]

            futures = [pool.submit(annotate_record, record, dets, img_shape)
                       for record, dets, img_shape in payloads]
            for future in futures:
                name, entry = future.result()
                metadata[name] = entry
                stats["total"] += 1
                if entry["source_category"] == "clean":
                    stats["clean"] += 1
                else:
                    stats[f"method_{entry['method']}"] += 1
                for flag in entry["flags"]:
                    if flag in FLAG_STATS:
                        stats[FLAG_STATS[flag]] += 1

            chunks_done += 1
            done = min(chunk_start + PREDICT_BATCH, len(todo))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            log(f"{done}/{len(todo)} images "
                f"({rate:.1f}/s, eta {((len(todo) - done) / rate / 60):.1f} min)")

            if chunks_done % CHECKPOINT_EVERY_CHUNKS == 0:
                METADATA_OUT.write_text(json.dumps(metadata, indent=1),
                                        encoding="utf-8")

    METADATA_OUT.write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "accept_thresholds": {"same_class": ACCEPT_SAME, "other_class": ACCEPT_OTHER},
        "counts": dict(stats),
        "annotated_total": len(metadata),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
