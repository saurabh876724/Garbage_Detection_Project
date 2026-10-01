"""Shared constants and helpers for dataset scripts."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset_final"
MANIFEST_PATH = DATASET_ROOT / "manifests" / "image_manifest.csv"
METADATA_PATH = DATASET_ROOT / "manifests" / "annotation_metadata.json"
QUARANTINE_ROOT = DATASET_ROOT / "quarantine"
REPORTS_DIR = DATASET_ROOT / "reports"

GARBAGE_CLASSES = ("battery", "biological", "cardboard", "clothes", "glass",
                   "metal", "paper", "plastic", "shoes", "trash")
NEGATIVE_CLASS = "clean"
SPLITS = ("train", "val", "test")


def load_manifest():
    import csv
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_metadata():
    import json
    if not METADATA_PATH.is_file():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def active_manifest():
    """Split manifest rows into (active, quarantined, missing).

    `active` rows still live in images/<split>/; `quarantined` rows were parked
    under quarantine/<split>/ by quarantine_unboxed.py; `missing` rows point at
    files that exist in neither place, which is a real integrity error.
    """
    active, quarantined, missing = [], [], []
    for row in load_manifest():
        name, split = row["target_image_name"], row["split"]
        if (DATASET_ROOT / "images" / split / name).is_file():
            active.append(row)
        elif (QUARANTINE_ROOT / split / "images" / name).is_file():
            quarantined.append(row)
        else:
            missing.append(row)
    return active, quarantined, missing
