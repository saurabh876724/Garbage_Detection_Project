# Demo Test Data Package

**AI-Based Intelligent Garbage Detection and Monitoring System**

This directory contains **external application test data** — real, freely-usable images and video URLs sourced from Pexels (Pexels License). These files are used exclusively for **application functional testing**, NOT for model training or accuracy evaluation.

> **Important:** The official model metrics come ONLY from the held-out `dataset_final/test` split. Do NOT use these demo files for accuracy claims.

---

## Directory Structure

```
demo_data/
├── images/
│   ├── single_object/     # 10 images — one garbage class per image
│   │   ├── single_battery_01.jpg
│   │   ├── single_biological_01.jpg
│   │   ├── single_cardboard_01.jpg
│   │   ├── single_clothes_01.jpg
│   │   ├── single_glass_01.jpg
│   │   ├── single_metal_01.jpg
│   │   ├── single_paper_01.jpg
│   │   ├── single_plastic_01.jpg
│   │   ├── single_shoes_01.jpg
│   │   └── single_trash_01.jpg
│   ├── multi_object/      # 12 images — multiple garbage classes in one frame
│   │   ├── multi_waste_01.jpg  ← BEST for multi-object test (aerial landfill)
│   │   ├── multi_waste_02.jpg
│   │   ├── ...
│   │   └── multi_waste_12.jpg
│   ├── clean/             # 3 images — no visible garbage
│   │   ├── clean_scene_01.jpg  ← BEST for CLEAN status test
│   │   ├── clean_scene_02.jpg
│   │   └── clean_scene_03.jpg
│   └── difficult/         # 5 images — challenging detection scenarios
│       ├── difficult_waste_01.jpg  ← BEST for difficult-scene test
│       ├── difficult_waste_02.jpg
│       ├── ...
│       └── difficult_waste_05.jpg
├── videos/                # Video URLs recorded; manual download required
│   ├── multi_object/
│   ├── street_waste/
│   ├── recycling/
│   └── difficult/
├── test_manifest.csv      # Full inventory with source URLs, licenses, purposes
├── TESTING_GUIDE.md       # Step-by-step testing instructions
├── README.md              # This file
└── download_report.json   # Machine-readable download summary
```

---

## File Counts

| Category | Images | Videos (URLs) |
|----------|--------|---------------|
| Single object | 10 | — |
| Multi-object | 12 | 2 |
| Clean scenes | 3 | — |
| Difficult scenes | 5 | 1 |
| Street waste | — | 1 |
| Recycling | — | 1 |
| **Total** | **30** | **5** |

---

## Source & License

- **Images:** All downloaded from [Pexels](https://www.pexels.com/license/) under the **Pexels License** — free for commercial and non-commercial use, no attribution required.
- **Videos:** URLs point to Pexels Video pages. Download manually; same license applies.
- Individual asset licenses should be verified on the source page before commercial redistribution.

---

## How to Use

1. **Read** `TESTING_GUIDE.md` for step-by-step test procedures.
2. **Launch** the app: `python main.py --gui`
3. **Test** each feature using the files in this directory.
4. **Record** observations using the template in the testing guide.

See `test_manifest.csv` for the full inventory including source URLs, expected classes, and purpose of each file.

---

## What This Is NOT

- ❌ NOT a training dataset
- ❌ NOT for model accuracy evaluation
- ❌ NOT annotated with YOLO ground-truth labels
- ❌ NOT a replacement for `dataset_final/`

These files exist solely to verify that the **application UI, detection pipeline, video processing, history logging, and analytics** all work correctly with real-world media.

---

## Quick Answers

| Question | Answer |
|----------|--------|
| Best image for multi-object detection? | `images/multi_object/multi_waste_01.jpg` (aerial landfill, many classes) |
| Best image for CLEAN testing? | `images/clean/clean_scene_01.jpg` (clean indoor room) |
| Best image for difficult-scene testing? | `images/difficult/difficult_waste_01.jpg` (small objects at distance) |
| Best video for first Video Detection test? | `videos/multi_object/garbage_video_01.mp4` (person sorting garbage) |
| Best video for multiple objects? | `videos/multi_object/multi_object_video_01.mp4` (recycling conveyor) |
| Files that failed verification? | None — all 30 images downloaded and opened successfully. |
