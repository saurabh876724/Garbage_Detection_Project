"""Audit dataset_final/ annotations: label validity, coverage, distribution.

Checks per split:
  - missing label file for an image / orphan label without an image
  - malformed lines (wrong field count, non-numeric)
  - invalid class IDs (outside 0-9)
  - invalid boxes (non-normalized or out-of-range coordinates,
    non-positive width/height, box extending outside the image)
  - duplicate image stems within a split
  - garbage images with empty labels (manual-box backlog)
  - class distribution of boxes

Exit code 1 when hard errors exist (training gate), 0 when clean.
Review flags (multi_class etc.) are reported as warnings only.

Usage:
    python scripts/audit_annotations.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _common import (DATASET_ROOT, GARBAGE_CLASSES, NEGATIVE_CLASS,
                     REPORTS_DIR, SPLITS, load_manifest, load_metadata)

HARD = ("missing_label", "orphan_label", "malformed_label",
        "invalid_class_id", "invalid_box", "duplicate_stem")


def parse_label(path: Path):
    rows, issues = [], []
    for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            issues.append((lineno, "malformed_label"))
            continue
        try:
            cid = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            issues.append((lineno, "malformed_label"))
            continue
        # 1e-6 tolerance: labels store 6-decimal values, so reconstructed
        # edges (x +/- w/2) may leave [0, 1] by half a rounding unit.
        ok_box = (all(map(math.isfinite, (x, y, w, h))) and w > 0 and h > 0
                  and -1e-6 <= x - w / 2 and x + w / 2 <= 1 + 1e-6
                  and -1e-6 <= y - h / 2 and y + h / 2 <= 1 + 1e-6
                  and w <= 1 + 1e-6 and h <= 1 + 1e-6)
        if cid not in range(len(GARBAGE_CLASSES)):
            issues.append((lineno, "invalid_class_id"))
            continue
        if not ok_box:
            issues.append((lineno, "invalid_box"))
            continue
        rows.append((cid, x, y, w, h))
    return rows, issues


def main() -> int:
    manifest = load_manifest()
    metadata = load_metadata()
    category_of = {r["target_image_name"]: r["source_category"] for r in manifest}

    issues, counts, empty_labels = [], Counter(), Counter()
    seen_stems = {s: set() for s in SPLITS}
    review_flags = Counter()
    images_per_split = Counter()
    missing_manifest_entries = []

    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        images = sorted(p for p in image_dir.iterdir() if p.is_file())
        labels = sorted(p for p in label_dir.glob("*.txt"))
        images_per_split[split] = len(images)
        label_names = {l.stem for l in labels}

        for image in images:
            stem = image.stem
            if stem in seen_stems[split]:
                issues.append((split, image.name, "duplicate_stem", ""))
            seen_stems[split].add(stem)
            if stem not in label_names:
                issues.append((split, image.name, "missing_label", ""))
                continue
            rows, file_issues = parse_label(label_dir / f"{stem}.txt")
            for lineno, code in file_issues:
                issues.append((split, image.name, code, f"line {lineno}"))
            if not rows:
                empty_labels[split] += 1
                category = category_of.get(image.name)
                if category is None:
                    missing_manifest_entries.append(image.name)
                elif category != NEGATIVE_CLASS:
                    issues.append((split, image.name, "empty_label_garbage",
                                   f"category={category}"))
            counts.update((split, GARBAGE_CLASSES[cid]) for cid, *_ in rows)

        image_names = {i.stem for i in images}
        for label in labels:
            if label.stem not in image_names:
                issues.append((split, label.name, "orphan_label", ""))

    for name, meta in metadata.items():
        for flag in meta.get("flags", []):
            review_flags[flag] += 1

    hard_errors = [i for i in issues if i[2] in HARD]
    empty_garbage = [i for i in issues if i[2] == "empty_label_garbage"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "images_per_split": dict(images_per_split),
        "box_counts_per_class": {
            cls: {s: counts.get((s, cls), 0) for s in SPLITS}
            for cls in GARBAGE_CLASSES},
        "total_boxes": sum(counts.values()),
        "empty_labels": dict(empty_labels),
        "hard_errors": len(hard_errors),
        "garbage_images_without_boxes": len(empty_garbage),
        "review_flags": dict(review_flags),
        "manifest_missing_entries": len(missing_manifest_entries),
        "errors": [dict(zip(("split", "image", "code", "detail"), i))
                   for i in issues],
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "annotation_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (REPORTS_DIR / "annotation_audit_issues.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("split", "image", "code", "detail"))
        writer.writerows(issues)

    summary = {k: report[k] for k in
               ("images_per_split", "total_boxes", "empty_labels",
                "hard_errors", "garbage_images_without_boxes", "review_flags")}
    print(json.dumps(summary, indent=2))
    print(f"full report -> {REPORTS_DIR / 'annotation_audit_report.json'}")

    if hard_errors:
        print("RESULT: FAIL - hard errors present, do not train")
        return 1
    print("RESULT: PASS - no hard errors "
          f"({len(empty_garbage)} garbage images still lack boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
