"""Render montage grids of images with their annotation boxes drawn.

Two modes:
  - random stratified sample across classes (for statistical visual audit)
  - all flagged images (needs_manual_box, multi_class, other_only_review,
    clean_with_garbage, too_many_detections)

Boxes are read from the YOLO label files (the training source of truth),
not from annotation metadata, so label-file bugs are caught visually.

Usage:
    python scripts/visual_audit.py [--n-sample 240] [--max-flagged 400] [--seed 4050]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from _common import (DATASET_ROOT, GARBAGE_CLASSES, REPORTS_DIR,
                     load_manifest, load_metadata)

TILE_W, TILE_H = 460, 380
COLS, ROWS = 4, 3
FLAG_KEYS = ("needs_manual_box", "multi_class", "other_only_review",
             "clean_with_garbage", "too_many_detections")
CLASS_COLORS = {
    "battery": (0, 0, 255), "biological": (0, 128, 255),
    "cardboard": (0, 180, 0), "clothes": (255, 0, 200),
    "glass": (255, 255, 0), "metal": (0, 255, 255),
    "paper": (128, 128, 0), "plastic": (255, 0, 0),
    "shoes": (160, 0, 255), "trash": (0, 120, 120),
}


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
        except ValueError:
            continue
    return boxes


def render_tile(image_path: Path, boxes, caption: str) -> np.ndarray:
    canvas = np.full((TILE_H, TILE_W, 3), 60, np.uint8)
    img = cv2.imread(str(image_path))
    if img is not None:
        h, w = img.shape[:2]
        scale = min(TILE_W / w, (TILE_H - 26) / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = cv2.resize(img, (nw, nh))
        for cls, cx, cy, bw, bh in boxes:
            x1 = int(max(0, (cx - bw / 2) * nw))
            y1 = int(max(0, (cy - bh / 2) * nh))
            x2 = int(min(nw - 1, (cx + bw / 2) * nw))
            y2 = int(min(nh - 1, (cy + bh / 2) * nh))
            color = CLASS_COLORS.get(GARBAGE_CLASSES[cls], (0, 255, 0)) \
                if 0 <= cls < len(GARBAGE_CLASSES) else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        canvas[0:nh, 0:nw] = img
    cv2.rectangle(canvas, (0, TILE_H - 24), (TILE_W, TILE_H), (24, 24, 24), -1)
    cv2.putText(canvas, caption[:70], (4, TILE_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_montages(entries, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows, montage_count = [], 0
    for start in range(0, len(entries), COLS * ROWS):
        grid = entries[start:start + COLS * ROWS]
        montage_count += 1
        canvas = np.full((ROWS * TILE_H, COLS * TILE_W, 3), 40, np.uint8)
        for i, entry in enumerate(grid):
            tile = render_tile(entry["image"], entry["boxes"], entry["caption"])
            r, c = divmod(i, COLS)
            canvas[r * TILE_H:(r + 1) * TILE_H,
                   c * TILE_W:(c + 1) * TILE_W] = tile
            index_rows.append({
                "montage": f"{tag}_{montage_count:03d}.jpg",
                "image": entry["image"].name,
                "split": entry["split"],
                "caption": entry["caption"],
            })
        cv2.imwrite(str(out_dir / f"{tag}_{montage_count:03d}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return montage_count, index_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sample", type=int, default=240)
    parser.add_argument("--max-flagged", type=int, default=400)
    parser.add_argument("--seed", type=int, default=4050)
    args = parser.parse_args()

    manifest = load_manifest()
    metadata = load_metadata()
    by_class = defaultdict(list)
    flagged = []
    for row in manifest:
        name = row["target_image_name"]
        meta = metadata.get(name, {})
        label_path = DATASET_ROOT / "labels" / row["split"] / f"{Path(name).stem}.txt"
        entry = {
            "image": DATASET_ROOT / "images" / row["split"] / name,
            "split": row["split"],
            "caption": f"{row['source_category']}/{row['split']} "
                       f"m={meta.get('method', '?')}"
                       + (f" F={','.join(meta.get('flags', []))}"
                          if meta.get("flags") else ""),
            "boxes": read_label(label_path) if label_path.is_file() else [],
        }
        by_class[row["source_category"]].append(entry)
        if any(f in meta.get("flags", []) for f in FLAG_KEYS):
            flagged.append(entry)

    rng = random.Random(args.seed)
    classes = [c for c in by_class if c != "clean"]
    per_class = max(1, args.n_sample // len(classes))
    sample = []
    for cls in classes:
        pool = list(by_class[cls])
        rng.shuffle(pool)
        sample.extend(pool[:per_class])

    out_dir = REPORTS_DIR / "visual_audit"
    count_a, rows_a = build_montages(sample, out_dir, "sample")
    count_b, rows_b = build_montages(flagged[:args.max_flagged], out_dir, "flagged")
    with (out_dir / "montage_index.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["montage", "image", "split",
                                                "caption"])
        writer.writeheader()
        writer.writerows(rows_a + rows_b)

    print(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_montages": count_a,
        "sample_images": len(sample),
        "flagged_montages": count_b,
        "flagged_images_total": len(flagged),
        "flagged_images_rendered": min(len(flagged), args.max_flagged),
        "output_dir": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
