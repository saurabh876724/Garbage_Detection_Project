"""Repair annotation boxes written before the EXIF-orientation fix.

Two defects are repaired in one auditable pass:

1. Saliency boxes (otsu/grabcut) were normalised against raw stored image
   dimensions while the box itself was computed on the EXIF-rotated view,
   producing out-of-frame boxes for rotated phone photos. The saliency
   chain is recomputed with dimension-consistent reads.
2. Center/width values rounded to 6 decimals could reconstruct edges
   slightly outside [0, 1]. All boxes are re-derived from clipped,
   rounded corners (see auto_annotate.norm_box).

Label files and annotation metadata are rewritten only where content
actually changes. Nothing is deleted; failed saliency recomputation is
recorded honestly as needs_manual_box.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_annotate import (  # noqa: E402
    DATASET_ROOT, GARBAGE_CLASSES, METADATA_OUT, grabcut_box, image_size,
    log, norm_box, otsu_box)

SALIENCY_METHODS = {"otsu", "grabcut", None}


def label_text(boxes) -> str:
    return "".join(f"{b['cls']} {b['xywhn'][0]} {b['xywhn'][1]} "
                   f"{b['xywhn'][2]} {b['xywhn'][3]}\n"
                   for b in boxes if b["kept"])


def repair_entry(item):
    name, entry = item
    split = entry["split"]
    category = entry["source_category"]
    image_path = DATASET_ROOT / "images" / split / name
    label_path = DATASET_ROOT / "labels" / split / f"{Path(name).stem}.txt"
    size = image_size(image_path)
    if size is None:
        return name, entry, False

    changed = False
    if entry["method"] in SALIENCY_METHODS and category != "clean":
        box = otsu_box(image_path)
        tier = "otsu"
        if box is None:
            box = grabcut_box(image_path)
            tier = "grabcut"
        flags = [f for f in entry["flags"] if f != "needs_manual_box"]
        if box is not None:
            bx, by, bw, bh = box
            boxes = [{"cls": GARBAGE_CLASSES.index(category), "conf": None,
                      "kept": True,
                      "xywhn": list(norm_box(bx, by, bx + bw, by + bh,
                                             *size))}]
            method = tier
        else:
            boxes, method = [], None
            if "needs_manual_box" not in flags:
                flags.append("needs_manual_box")
        if boxes != entry["boxes"] or method != entry["method"] \
                or flags != entry["flags"]:
            entry["boxes"], entry["method"], entry["flags"] = \
                boxes, method, flags
            changed = True
    else:
        w_img, h_img = size
        for b in entry["boxes"]:
            if "xyxy" not in b:
                continue
            new = list(norm_box(*b["xyxy"], w_img, h_img))
            if new != b["xywhn"]:
                b["xywhn"] = new
                changed = True

    if changed:
        label_path.write_text(label_text(entry["boxes"]), encoding="utf-8")
    return name, entry, changed


def main() -> int:
    metadata = json.loads(METADATA_OUT.read_text(encoding="utf-8"))
    log(f"repairing {len(metadata)} entries")
    changed = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        for name, entry, did_change in pool.map(
                repair_entry, metadata.items()):
            metadata[name] = entry
            changed += did_change
    METADATA_OUT.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    log(f"repaired {changed} entries; metadata updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
