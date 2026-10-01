"""Final pre-training validation of dataset_final/.

Verifies:
  - directory structure (images/labels x train/val/test)
  - every image opens with PIL
  - every image has a label file and vice versa
  - class IDs within 0-9, coordinates normalized and boxes inside the frame
  - no data leakage: no image content (md5) appears in more than one split
  - manifest covers every materialized image

Usage:
    python scripts/validate_dataset.py [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from _common import (DATASET_ROOT, GARBAGE_CLASSES, REPORTS_DIR, SPLITS,
                     active_manifest, load_manifest)

NUM_CLASSES = len(GARBAGE_CLASSES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="verify md5 leakage on a 2000-image sample only")
    args = parser.parse_args()

    errors, warnings = [], []

    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"missing directory for split {split}")
            continue
        images = sorted(p for p in image_dir.iterdir() if p.is_file())
        label_stems = {p.stem for p in label_dir.glob("*.txt")}

        for image in images:
            stem = image.stem
            if stem not in label_stems:
                errors.append(f"{split}/{image.name}: missing label file")
                continue
            try:
                with Image.open(image) as im:
                    im.verify()
            except Exception as e:
                errors.append(f"{split}/{image.name}: unreadable image ({e})")
                continue
            for lineno, line in enumerate(
                    (label_dir / f"{stem}.txt").read_text(
                        encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    errors.append(f"{split}/{stem}.txt:{lineno}: bad field count")
                    continue
                try:
                    cid = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f"{split}/{stem}.txt:{lineno}: non-numeric")
                    continue
                if not 0 <= cid < NUM_CLASSES:
                    errors.append(f"{split}/{stem}.txt:{lineno}: class {cid} out of range")
                # 1e-6 tolerance: labels store 6-decimal values, so
                # reconstructed edges may leave [0, 1] by half a rounding unit.
                if not (0 < w <= 1 + 1e-6 and 0 < h <= 1 + 1e-6
                        and -1e-6 <= x - w / 2 and x + w / 2 <= 1 + 1e-6
                        and -1e-6 <= y - h / 2 and y + h / 2 <= 1 + 1e-6):
                    errors.append(f"{split}/{stem}.txt:{lineno}: box outside frame")

        for stem in label_stems - {i.stem for i in images}:
            errors.append(f"{split}: orphan label {stem}.txt")

    manifest = load_manifest()
    manifest_names = {r["target_image_name"] for r in manifest}
    materialized = {p.name for split in SPLITS
                    for p in (DATASET_ROOT / "images" / split).iterdir()
                    if p.is_file()}
    uncovered = materialized - manifest_names
    if uncovered:
        errors.append(f"{len(uncovered)} images not in manifest "
                      f"(e.g. {sorted(uncovered)[:3]})")

    active, quarantined, missing = active_manifest()
    for row in missing:
        errors.append(f"{row['target_image_name']}: listed in manifest but "
                      f"absent from both {row['split']}/ and quarantine/")

    md5_split = {}
    if args.quick:
        import random
        rng = random.Random(4050)
        sample = rng.sample(sorted(active, key=lambda r: r["target_image_name"]),
                            min(2000, len(active)))
    else:
        sample = active
    for row in sample:
        path = DATASET_ROOT / "images" / row["split"] / row["target_image_name"]
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != row["md5"]:
            errors.append(f"{row['target_image_name']}: content changed "
                          f"since prepare (manifest md5 mismatch)")
        if digest in md5_split and md5_split[digest] != row["split"]:
            errors.append(f"leakage: {row['target_image_name']} shares content "
                          f"with an image in {md5_split[digest]}")
        md5_split[digest] = row["split"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structure": "ok" if not any("missing directory" in e for e in errors) else "broken",
        "images_validated": len(sample) if args.quick else len(materialized),
        "md5_leakage_checked": len(sample),
        "quarantined_excluded": len(quarantined),
        "errors": errors,
        "warnings": warnings,
        "result": "PASS" if not errors else "FAIL",
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("structure", "images_validated", "md5_leakage_checked",
                       "quarantined_excluded", "result")}, indent=2))
    for e in errors[:20]:
        print("ERROR:", e)
    if len(errors) > 20:
        print(f"... and {len(errors) - 20} more errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
