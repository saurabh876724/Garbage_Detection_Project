"""Dependency-free validation of manual YOLO labels before training."""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ("battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash")
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

def issue(items, split, image, code, detail):
    items.append({"split": split, "image": image, "code": code, "detail": detail})

def parse_label(path, split, image, issues):
    valid = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields: continue
        if len(fields) != 5:
            issue(issues, split, image, "invalid_field_count", f"line {number}"); continue
        try: cid = int(fields[0]); x, y, w, h = map(float, fields[1:])
        except ValueError:
            issue(issues, split, image, "non_numeric_value", f"line {number}"); continue
        valid_box = (cid in range(10) and all(map(math.isfinite, (x,y,w,h))) and w > 0 and h > 0 and x-w/2 >= 0 and x+w/2 <= 1 and y-h/2 >= 0 and y+h/2 <= 1)
        if not valid_box:
            issue(issues, split, image, "invalid_class_or_box", f"line {number}"); continue
        valid.append(cid)
    return valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--require-boxes", action="store_true")
    ap.add_argument("--verify-source-class", action="store_true")
    args = ap.parse_args(); root = args.dataset_root.resolve()
    with (root / "manifests" / "image_manifest.csv").open(encoding="utf-8-sig", newline="") as f: manifest = list(csv.DictReader(f))
    issues, counts, empty = [], Counter(), 0
    expected = set()
    for row in manifest:
        split, name, source = row["split"], row["target_image_name"], row["source_category"]
        image = root / "images" / split / name; label = root / "labels" / split / f"{Path(name).stem}.txt"; expected.add(label.resolve())
        if not image.is_file(): issue(issues, split, name, "missing_image", str(image)); continue
        if not label.is_file(): issue(issues, split, name, "missing_label", str(label)); continue
        ids = parse_label(label, split, name, issues)
        if not ids:
            empty += 1
            if args.require_boxes and source != "clean": issue(issues, split, name, "missing_required_box", "non-clean source image")
        else:
            counts.update(ids)
            if args.verify_source_class and source != "clean" and CLASS_ID[source] not in ids: issue(issues, split, name, "source_class_not_present", source)
    for split in ("train", "val", "test"):
        for label in (root / "labels" / split).glob("*.txt"):
            if label.resolve() not in expected: issue(issues, split, label.name, "orphan_label", "not in manifest")
    reports = root / "reports"; reports.mkdir(exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "images_audited": len(manifest), "empty_labels": empty, "valid_boxes_by_class": {name:counts[i] for i,name in enumerate(CLASSES)}, "errors":len(issues), "strict":args.require_boxes}
    (reports / "annotation_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (reports / "annotation_validation_issues.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("split","image","code","detail")); writer.writeheader(); writer.writerows(issues)
    print(json.dumps(report, indent=2)); return 1 if issues else 0
if __name__ == "__main__": raise SystemExit(main())
