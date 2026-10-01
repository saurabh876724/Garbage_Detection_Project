# Smart Garbage Detection System

A professional, end-to-end AI garbage detection system built with **Python, YOLOv8, PyTorch and OpenCV**, wrapped in a multi-page **PySide6 (Qt)** dashboard. It detects and counts 10 classes of litter from a live webcam, an image, or a video file; derives a scene status (**CLEAN / DIRTY / REVIEW**); records every event to a SQLite history; stores tamper-evident image+JSON evidence; and raises throttled audible alerts when accumulation crosses a threshold.

> Final-year project. The highest priority is **correct data + correct annotations**. Every bounding box used for training was machine-generated and then audited; images whose boxes could not be verified were **quarantined out of the training set** rather than silently kept or deleted.

---

## Object classes

The detector recognises **10 garbage classes**:

`battery` · `biological` · `cardboard` · `clothes` · `glass` · `metal` · `paper` · `plastic` · `shoes` · `trash`

`clean` is **not** an object class. It is a *derived scene status*: a frame with no confident detections is reported as **CLEAN**, a frame with confident detections is **DIRTY**, and a frame with only low-confidence ("weak") detections is **REVIEW** (flagged for a human, never counted as garbage). The detector refuses to load any model whose class schema is not exactly these 10 classes, so an old COCO (80-class) checkpoint can never be mistaken for this project's model.

---

## Features

- **Three input modes** — live webcam, single image, video file — from one CLI and one GUI.
- **Professional 8-page dashboard** (PySide6): Dashboard, Live Camera, Image, Video, History, Analytics, Settings, About.
- **Derived scene status** with per-class colour-coded boxes, HUD, and label-collision avoidance.
- **Unique-object counting** via optional ByteTrack tracking (reports distinct objects, not per-frame duplicates).
- **Evidence bundles** — original frame + annotated frame + JSON metadata, written with a *never-overwrite* guarantee.
- **SQLite history** (schema v2) with migration from v1, text/source/status/date filters, aggregates and CSV export. Thread-safe.
- **Analytics** — class distribution, status split, daily/hourly activity, objects-per-event, plus model and dataset summaries.
- **Throttled alerts** — configurable count threshold and cooldown so the alarm never beeps per-frame.
- **Real measured FPS** — benchmarked on-device; never fabricated.
- **Rotating file logging** (`logs/app.log`) and a **16-check self-test** that runs without a trained model.

---

## Project layout

```
.
├── main.py                 CLI entry point (and --gui launcher)
├── config.py               central configuration and paths
├── train_yolo.py           YOLOv8 training script
├── evaluate_model.py       final evaluation on val + TEST splits
├── data.yaml               dataset config for ultralytics
├── requirements.txt        runtime dependencies
├── src/                    application core
│   ├── detector.py         Detection / SceneResult / GarbageDetector + schema guard
│   ├── preprocessing.py    source classification, image/video loading
│   ├── postprocessing.py   box + HUD drawing, status text
│   ├── tracking.py         ByteTrack unique-object counting
│   ├── counting.py         per-session statistics
│   ├── history.py          SQLite history (schema v2, migration, filters)
│   ├── evidence.py         never-overwrite evidence bundles
│   ├── alerts.py           throttled audible alerts
│   ├── analytics.py        aggregates for the Analytics page
│   ├── settings.py         persisted runtime settings + system info
│   ├── logging_setup.py    rotating file logging
│   └── utils.py            FPS meter, timestamps, unique paths
├── ui/                     PySide6 dashboard
│   ├── dashboard.py        MainWindow, AppContext, navigation, launch()
│   ├── engine.py           DetectionEngine + QThread DetectionWorker
│   ├── pages.py            Dashboard / Live / Image / Video pages
│   ├── insights.py         History / Analytics / Settings / About pages
│   ├── components.py       cards, badges, video surface, tables
│   ├── charts.py           QPainter bar / donut / line charts (no QtCharts)
│   └── styles.py           dark Qt stylesheet + palette
├── scripts/                data pipeline + self-test
│   ├── prepare_final_dataset.py  build dataset_final (80/10/10, deduped)
│   ├── auto_annotate.py          generate candidate boxes
│   ├── audit_annotations.py      verify every box, flag issues
│   ├── repair_annotations.py     fix repairable annotation problems
│   ├── quarantine_unboxed.py     park unverifiable images out of training
│   ├── validate_dataset.py       integrity checks on the active set
│   ├── dataset_statistics.py     per-class image/box counts
│   ├── visual_audit.py           render sampled annotations for eyeballing
│   └── selftest.py               16-check module verification suite
├── dataset_final/          the audited dataset + reports/
├── evidence/               captured evidence bundles
├── data/                   history DB + settings.json
├── logs/                   rotating application log
└── runs/detect/            ultralytics training runs
```

---

## Requirements

- Windows 10/11 (alerts use `winsound`; the rest is cross-platform).
- Python 3.10+ (developed on 3.11).
- NVIDIA GPU recommended (developed on RTX 4050 Laptop, 6 GB, CUDA 12.8). CPU works but is slow.

`requirements.txt` pins the runtime stack:

```
torch>=2.4        torchvision>=0.19     (CUDA 12.8 wheels via --extra-index-url)
ultralytics>=8.3  opencv-python>=4.9    numpy>=1.26
Pillow>=10.0      PySide6>=6.6
```

---

## Installation (VS Code)

Open the project folder in VS Code, then in the integrated terminal:

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate                # Windows PowerShell / cmd
# source .venv/bin/activate           # macOS / Linux

# 2. install dependencies (CUDA 12.8 build of PyTorch)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. verify the GPU is visible to PyTorch
python check_gpu.py

# 4. run the self-test (no trained model required)
python scripts/selftest.py
```

Expected: `check_gpu.py` reports the CUDA device, and `selftest.py` ends with `16/16 checks passed` / `RESULT: PASS`.

---

## Dataset provenance and integrity

The dataset was built from the original images (never deleted or overwritten), split **80 / 10 / 10** into train / val / test, deduplicated, then annotated and **audited**:

| Fact | Value |
|---|---|
| Catalogued images | **19,929** (19,888 garbage + 41 clean negatives) |
| Audited bounding boxes | **23,562** (train 18,814 · val 2,333 · test 2,415) |
| Hard annotation errors | **0** |
| Garbage images with no box (active set) | **0** |
| Quarantined (unverifiable boxes, excluded from training) | **323** |

`scripts/quarantine_unboxed.py` moved 323 images whose boxes could not be verified into `dataset_final/quarantine/` and updated the manifest — they are **excluded from training** but preserved on disk and restorable (`--restore`). The audit, statistics, validation and quarantine reports live in `dataset_final/reports/`.

> **Important:** an earlier checkpoint trained on COCO-derived labels reached mAP50 ≈ 0.629. That number is **invalid for this project** and is **not** reported as final accuracy. The detector actively rejects that 80-class model. Final accuracy comes only from the model trained on the audited 10-class labels, measured on the **TEST** split.

---

## Training

```bash
python train_yolo.py                       # 60 epochs, imgsz 640, batch 16, device 0
python train_yolo.py --epochs 60 --batch 16 --imgsz 640 --name garbage_final
```

Weights are written to `runs/detect/garbage_final/weights/best.pt`.

## Evaluation (final accuracy)

`evaluate_model.py` runs the trained model on the **val** and **test** splits, writes per-class precision / recall / F1 / AP50 / AP50-95, a confusion-matrix CSV per split, and a real on-device inference benchmark (mean / median / p95 ms and FPS). It never evaluates on the train split.

```bash
python evaluate_model.py                   # uses newest runs/detect/*/weights/best.pt
python evaluate_model.py --weights runs/detect/garbage_final/weights/best.pt
```

Outputs `dataset_final/reports/model_evaluation.json` (consumed by the Analytics page) plus `confusion_matrix_val.csv` and `confusion_matrix_test.csv`.

**The TEST-split numbers in that report are the project's final accuracy.**

---

## Usage

### GUI (recommended)

```bash
python main.py --gui
```

Navigate with the left sidebar: run the live camera, analyse an image or video, browse and export history, view analytics charts, and adjust detection / capture / alert settings (persisted to `data/settings.json`).

### CLI

```bash
python main.py                              # live webcam
python main.py --source 1                   # webcam index 1
python main.py --source path/to/image.jpg   # single image
python main.py --source path/to/video.mp4   # video file
```

Useful flags: `--conf`, `--iou`, `--device`, `--tracking` (unique-object counting), `--save-evidence`, `--save-video`, `--no-alert`, `--no-display`, `--verbose`.

In a preview window: **Q** quits, **C** captures evidence, **S** saves the annotated frame. Every session is written to the SQLite history; DIRTY scenes (or `--save-evidence`) also store an evidence JPG + JSON pair under `evidence/`.

---

## Self-test

```bash
python scripts/selftest.py                  # add --keep to inspect temp files
```

16 deterministic checks cover config, settings, detection validation, derived status, history (schema v2 + migration + filters + CSV), evidence (never-overwrite), alert cooldown, counting/tracking, preprocessing, postprocessing, analytics, the COCO-schema guard, Qt widgets/charts, all 8 dashboard pages, and logging — all without needing a trained model.

---

## Technology stack

Python · YOLOv8 (ultralytics) · PyTorch (CUDA) · OpenCV · NumPy · Pillow · PySide6 (Qt, custom QPainter charts) · SQLite · ByteTrack.
