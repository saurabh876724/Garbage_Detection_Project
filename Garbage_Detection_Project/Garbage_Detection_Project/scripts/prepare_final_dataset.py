"""Create dataset_final/ from the original classification dataset.

Read-only towards the original: images are hardlinked (fallback: copied)
into dataset_final/images/{train,val,test} with collision-safe names,
never moved or deleted. Labels are written later by auto_annotate.py.

Usage:
    python scripts/prepare_final_dataset.py [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "dataset" / "train"
OUTPUT_ROOT = PROJECT_ROOT / "dataset_final"
GARBAGE_CLASSES = ("battery", "biological", "cardboard", "clothes", "glass",
                   "metal", "paper", "plastic", "shoes", "trash")
NEGATIVE_CLASS = "clean"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 4050
FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from split_dataset import assign_splits  # noqa: E402


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def materialize(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        import shutil
        shutil.copy2(source, target)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="allow rebuilding an existing dataset_final")
    args = parser.parse_args()

    if not SOURCE_ROOT.is_dir():
        print(f"ERROR: source dataset not found: {SOURCE_ROOT}")
        return 1
    if OUTPUT_ROOT.exists() and not args.force:
        print(f"ERROR: {OUTPUT_ROOT} already exists; pass --force to rebuild")
        return 1
    if OUTPUT_ROOT.exists() and args.force:
        import shutil
        shutil.rmtree(OUTPUT_ROOT)

    found = sorted(p.name for p in SOURCE_ROOT.iterdir() if p.is_dir())
    missing = [c for c in GARBAGE_CLASSES + (NEGATIVE_CLASS,) if c not in found]
    unexpected = [c for c in found if c not in GARBAGE_CLASSES + (NEGATIVE_CLASS,)]
    if missing:
        print(f"ERROR: missing class folders in source: {missing}")
        return 1
    if unexpected:
        print(f"WARNING: unexpected source folders ignored: {unexpected}")

    classes = [NEGATIVE_CLASS] + list(GARBAGE_CLASSES)
    records, skipped, per_class = [], [], []
    seen_hashes: dict[str, str] = {}  # md5 -> first target name

    for class_index, category in enumerate(classes):
        folder = SOURCE_ROOT / category
        files = sorted(f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS)
        valid = []
        for f in files:
            try:
                with Image.open(f) as im:
                    im.verify()
                md5 = hashlib.md5(f.read_bytes()).hexdigest()
            except Exception as e:
                skipped.append({"source_path": str(f), "source_category": category,
                                "reason": f"unreadable: {e}"})
                continue
            if md5 in seen_hashes:
                skipped.append({"source_path": str(f), "source_category": category,
                                "reason": f"exact duplicate of {seen_hashes[md5]}"})
                continue
            seen_hashes[md5] = f"{category}__{f.stem}"
            valid.append((f, md5))

        assignment = assign_splits(len(valid), SEED + class_index, FRACTIONS)
        for (f, md5), split in zip(valid, assignment):
            safe_name = f"{category}__{sanitize(f.stem)}__{f.suffix.lstrip('.').lower()}{f.suffix}"
            records.append({
                "source_path": str(f),
                "source_category": category,
                "md5": md5,
                "split": split,
                "target_image_name": safe_name,
            })
        per_class.append({
            "source_category": category,
            "files_found": len(files),
            "valid_unique": len(valid),
            "train": assignment.count("train"),
            "val": assignment.count("val"),
            "test": assignment.count("test"),
            "skipped": len(files) - len(valid),
        })

    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    methods = Counter()
    targets = {}
    for r in records:
        target = OUTPUT_ROOT / "images" / r["split"] / r["target_image_name"]
        if target.name in targets and targets[target.name] != r["source_path"]:
            print(f"ERROR: target name collision: {target.name}")
            return 1
        targets[target.name] = r["source_path"]
        methods[materialize(Path(r["source_path"]), target)] += 1

    (OUTPUT_ROOT / "manifests").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "reports").mkdir(exist_ok=True)
    with (OUTPUT_ROOT / "manifests" / "image_manifest.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with (OUTPUT_ROOT / "reports" / "skipped_images.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_path", "source_category", "reason"])
        writer.writeheader()
        writer.writerows(skipped)
    with (OUTPUT_ROOT / "reports" / "class_distribution.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_class[0].keys()))
        writer.writeheader()
        writer.writerows(per_class)

    data_yaml = f"""path: {OUTPUT_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
""" + "".join(f"  {i}: {name}\n" for i, name in enumerate(GARBAGE_CLASSES))
    (OUTPUT_ROOT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    split_counts = Counter(r["split"] for r in records)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "seed": SEED,
        "fractions": FRACTIONS,
        "total_images": len(records),
        "split_counts": dict(split_counts),
        "garbage_images": sum(1 for r in records
                              if r["source_category"] != NEGATIVE_CLASS),
        "negative_images": sum(1 for r in records
                               if r["source_category"] == NEGATIVE_CLASS),
        "skipped": {"count": len(skipped),
                    "unreadable": sum(1 for s in skipped if "unreadable" in s["reason"]),
                    "duplicates": sum(1 for s in skipped if "duplicate" in s["reason"])},
        "materialization": dict(methods),
        "per_class": per_class,
        "next_step": "Run scripts/auto_annotate.py to generate labels.",
    }
    (OUTPUT_ROOT / "reports" / "prepare_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("total_images", "split_counts", "garbage_images",
                       "negative_images", "skipped", "materialization")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
