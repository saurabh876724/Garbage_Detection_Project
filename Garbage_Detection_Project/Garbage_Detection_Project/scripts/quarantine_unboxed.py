"""Move garbage images that have no verified box out of the training splits.

Why: the auto-annotator could not localize an object in a few garbage-folder
images (saliency failed, YOLO-World found nothing above threshold). Leaving
them in the splits trains the model that those scenes are CLEAN, which is a
false negative. They are parked under dataset_final/quarantine/<split>/ so
they can be annotated by hand later and restored.

Nothing is deleted: originals under dataset/ are untouched and the dataset_final
images are hard links, so quarantining only moves links out of the split folders.

Usage:
    python scripts/quarantine_unboxed.py            # report only
    python scripts/quarantine_unboxed.py --apply    # move files
    python scripts/quarantine_unboxed.py --restore  # move everything back
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _common import (DATASET_ROOT, METADATA_PATH, NEGATIVE_CLASS, REPORTS_DIR,
                     SPLITS, load_manifest, load_metadata)

QUARANTINE_ROOT = DATASET_ROOT / "quarantine"
QUARANTINE_MANIFEST = DATASET_ROOT / "manifests" / "quarantine_manifest.csv"
REASON = "no_verified_box"


def split_dirs(split: str):
    return (DATASET_ROOT / "images" / split, DATASET_ROOT / "labels" / split)


def target_dirs(split: str):
    return (QUARANTINE_ROOT / split / "images",
            QUARANTINE_ROOT / split / "labels")


def find_unboxed() -> list[dict]:
    """Garbage-category images whose metadata carries the manual-box flag."""
    category_of = {r["target_image_name"]: r["source_category"]
                   for r in load_manifest()}
    metadata = load_metadata()
    found = []
    for split in SPLITS:
        image_dir, label_dir = split_dirs(split)
        for image in sorted(p for p in image_dir.iterdir() if p.is_file()):
            meta = metadata.get(image.name, {})
            category = category_of.get(image.name, meta.get("source_category"))
            label = label_dir / f"{image.stem}.txt"
            has_boxes = label.is_file() and bool(
                label.read_text(encoding="utf-8").strip())
            if category == NEGATIVE_CLASS or has_boxes:
                continue
            found.append({"name": image.name, "split": split,
                          "category": category, "reason": REASON,
                          "flags": meta.get("flags", []),
                          "method": meta.get("method")})
    return found


def move(item: dict, to_quarantine: bool) -> str:
    split = item["split"]
    src_img, src_lbl = split_dirs(split)
    dst_img, dst_lbl = target_dirs(split)
    if not to_quarantine:
        src_img, dst_img = dst_img, src_img
        src_lbl, dst_lbl = dst_lbl, src_lbl
    stem = Path(item["name"]).stem
    moved = []
    for src_dir, dst_dir, name in ((src_img, dst_img, item["name"]),
                                   (src_lbl, dst_lbl, f"{stem}.txt")):
        src = src_dir / name
        if not src.is_file():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        if dst.exists():
            raise FileExistsError(f"refusing to overwrite {dst}")
        shutil.move(str(src), str(dst))
        moved.append(str(dst))
    return ", ".join(moved)


def restore_list() -> list[dict]:
    if not QUARANTINE_MANIFEST.is_file():
        return []
    with QUARANTINE_MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def apply_moves(items: list[dict], to_quarantine: bool) -> int:
    moved = 0
    for item in items:
        paths = move(item, to_quarantine)
        if paths:
            moved += 1
            print(f"{'quarantined' if to_quarantine else 'restored'}: "
                  f"{item['split']}/{item['name']} -> {paths}")
    return moved


def write_records(items: list[dict], to_quarantine: bool) -> None:
    """Update annotation metadata and the quarantine manifest."""
    metadata = load_metadata()
    names = {i["name"] for i in items}
    for name in names:
        meta = metadata.setdefault(name, {})
        if to_quarantine:
            meta["quarantined"] = True
            meta["quarantine_reason"] = REASON
        else:
            meta.pop("quarantined", None)
            meta.pop("quarantine_reason", None)
    METADATA_PATH.write_text(json.dumps(metadata, indent=1), encoding="utf-8")

    if to_quarantine:
        QUARANTINE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        rows = [{**item, "flags": "|".join(item["flags"]),
                 "quarantined_at": stamp} for item in items]
    elif QUARANTINE_MANIFEST.is_file():
        rows = [r for r in restore_list() if r["name"] not in names]
    else:
        return

    fields = ("name", "split", "category", "reason", "flags", "method",
              "quarantined_at")
    with QUARANTINE_MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="actually move files (default: report only)")
    parser.add_argument("--restore", action="store_true",
                        help="move quarantined images back into their splits")
    args = parser.parse_args()

    if args.restore:
        items = restore_list()
        if not items:
            print("nothing quarantined - no action")
            return 0
        print(f"restoring {len(items)} quarantined images")
        moved = apply_moves(items, to_quarantine=False) if args.apply else 0
        if args.apply:
            write_records(items, to_quarantine=False)
    else:
        items = find_unboxed()
        if not items:
            print("no unboxed garbage images found - dataset is clean")
            return 0
        per_split = Counter(i["split"] for i in items)
        print(f"found {len(items)} garbage images without any verified box")
        print(json.dumps(dict(per_split), indent=2))
        if not args.apply:
            print("dry run - re-run with --apply to move them to "
                  f"{QUARANTINE_ROOT}")
            return 0
        moved = apply_moves(items, to_quarantine=True)
        write_records(items, to_quarantine=True)

    if args.apply:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "action": "restore" if args.restore else "quarantine",
            "moved": moved,
            "items": items,
        }
        REPORTS_DIR.mkdir(exist_ok=True)
        (REPORTS_DIR / "quarantine_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {REPORTS_DIR / 'quarantine_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
