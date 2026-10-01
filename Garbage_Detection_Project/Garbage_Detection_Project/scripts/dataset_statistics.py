"""Dataset statistics and class-imbalance report for dataset_final/.

Reports images per class, annotation boxes per class, split counts,
boxes-per-image, annotation-method provenance, and imbalance ratios.

Usage:
    python scripts/dataset_statistics.py
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

from _common import (DATASET_ROOT, GARBAGE_CLASSES, NEGATIVE_CLASS,
                     REPORTS_DIR, SPLITS, active_manifest, load_metadata)


def read_label_counts(path):
    counts = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                counts[int(parts[0])] += 1
            except ValueError:
                pass
    return counts


def main() -> int:
    active, quarantined, _missing = active_manifest()
    metadata = load_metadata()

    images_per_class = {c: Counter() for c in GARBAGE_CLASSES + (NEGATIVE_CLASS,)}
    boxes_per_class = {c: Counter() for c in GARBAGE_CLASSES}
    boxes_total, images_with_boxes, method_counts, multi_box = Counter(), 0, Counter(), 0

    for row in active:
        category, split, name = (row["source_category"], row["split"],
                                  row["target_image_name"])
        if category == NEGATIVE_CLASS:
            images_per_class[category][split] += 1
            continue
        images_per_class[category][split] += 1
        label_path = DATASET_ROOT / "labels" / split / f"{name.rsplit('.', 1)[0]}.txt"
        if not label_path.is_file():
            continue
        counts = read_label_counts(label_path)
        if counts:
            images_with_boxes += 1
        if sum(counts.values()) > 1:
            multi_box += 1
        for cid, n in counts.items():
            if 0 <= cid < len(GARBAGE_CLASSES):
                boxes_per_class[GARBAGE_CLASSES[cid]][split] += n
                boxes_total[split] += n
        method_counts[metadata.get(name, {}).get("method", "unknown")] += 1

    garbage_images = sum(sum(v.values()) for c, v in images_per_class.items()
                         if c != NEGATIVE_CLASS)
    box_totals = {cls: sum(boxes_per_class[cls].values())
                  for cls in GARBAGE_CLASSES}
    min_cls = min(box_totals, key=box_totals.get)
    max_cls = max(box_totals, key=box_totals.get)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": len(active),
        "garbage_images": garbage_images,
        "negative_images": sum(images_per_class[NEGATIVE_CLASS].values()),
        "quarantined_images": len(quarantined),
        "images_per_class": {
            c: {"train": v["train"], "val": v["val"], "test": v["test"],
                "total": sum(v.values())}
            for c, v in images_per_class.items()},
        "boxes_per_class": {
            c: {"train": v["train"], "val": v["val"], "test": v["test"],
                "total": sum(v.values())}
            for c, v in boxes_per_class.items()},
        "total_boxes": dict(boxes_total),
        "images_with_boxes": images_with_boxes,
        "multi_box_images": multi_box,
        "annotation_method_provenance": dict(method_counts),
        "imbalance": {
            "max_class": {"name": max_cls, "boxes": box_totals[max_cls]},
            "min_class": {"name": min_cls, "boxes": box_totals[min_cls]},
            "max_min_ratio": round(box_totals[max_cls] /
                                   max(1, box_totals[min_cls]), 2),
        },
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dataset_statistics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"total images: {len(active)} "
          f"(garbage {garbage_images}, negative {report['negative_images']}, "
          f"quarantined {len(quarantined)})")
    print(f"total boxes: {sum(boxes_total.values())} across "
          f"{images_with_boxes} images ({multi_box} multi-box)")
    print(f"imbalance max/min: {report['imbalance']['max_min_ratio']} "
          f"({max_cls} vs {min_cls})")
    print("provenance:", json.dumps(report["annotation_method_provenance"]))
    print("boxes per class:")
    for cls in GARBAGE_CLASSES:
        print(f"  {cls:12s} {box_totals[cls]:6d}")
    print(f"full report -> {REPORTS_DIR / 'dataset_statistics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
