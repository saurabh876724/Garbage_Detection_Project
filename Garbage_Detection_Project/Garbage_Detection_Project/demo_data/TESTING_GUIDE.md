# Demo Test Data — Testing Guide

**Project:** AI-Based Intelligent Garbage Detection and Monitoring System
**Purpose:** Application functional testing using external, freely-usable media.
**License:** All images are from Pexels (Pexels License — free for commercial/non-commercial use, no attribution required). Videos require manual download from the URLs listed in `test_manifest.csv`.

---

## Quick Start

1. Launch the application: `python main.py --gui`
2. Use the files under `demo_data/` to test each feature.
3. Record your observations (number of boxes, classes, confidence, status).

---

## TEST 1 — Image Detection: Single Object

**Goal:** Verify the detector correctly identifies one object per image.

| Step | Action |
|------|--------|
| 1 | Open **Image Detection** page |
| 2 | Select `demo_data/images/single_object/single_plastic_01.jpg` |
| 3 | Click **DETECT** |

**Expected:**
- Exactly 1 bounding box drawn
- Class label: `plastic`
- Confidence ≥ 0.35 (confirmed) or < 0.35 (review-only dashed outline)
- Status badge: DIRTY (if confirmed) or REVIEW (if weak)

**Repeat with:** `single_battery_01.jpg`, `single_glass_01.jpg`, `single_trash_01.jpg`

**Record:** number of boxes, class name, confidence value, status.

---

## TEST 2 — Image Detection: Multiple Objects (HIGH PRIORITY)

**Goal:** Verify ALL valid detections are returned and displayed simultaneously (not just top-1).

| Step | Action |
|------|--------|
| 1 | Open **Image Detection** page |
| 2 | Select `demo_data/images/multi_object/multi_waste_01.jpg` |
| 3 | Click **DETECT** |

**Expected:**
- **Multiple bounding boxes** drawn on the same frame (≥ 3 expected for this aerial landfill image)
- Each box has its own class label and confidence
- No boxes are silently dropped
- Count panel shows correct per-class totals

**Best multi-object images to try (in order):**
1. `multi_waste_01.jpg` — aerial landfill, many classes visible
2. `multi_waste_08.jpg` — outdoor garbage pile with mattress + various waste
3. `multi_waste_10.jpg` — cluttered alley with bicycles, scrap metal, debris
4. `multi_waste_12.jpg` — city street with overflowing bins + cardboard boxes

**Record for each image:**
- Raw YOLO detections (from log/console if available)
- Filtered detections (after conf threshold)
- Displayed detections (boxes actually drawn)
- Classes detected
- Confidence values of each box

---

## TEST 3 — Image Detection: Clean Scene

**Goal:** Verify the system correctly reports CLEAN status when no garbage is detected.

| Step | Action |
|------|--------|
| 1 | Open **Image Detection** page |
| 2 | Select `demo_data/images/clean/clean_scene_01.jpg` |
| 3 | Click **DETECT** |

**Expected:**
- **Zero** bounding boxes drawn
- Status badge: **CLEAN** (green)
- No false-positive detections

**Repeat with:** `clean_scene_02.jpg`, `clean_scene_03.jpg`

---

## TEST 4 — Image Detection: Difficult Scenes

**Goal:** Test edge cases — small objects, occlusion, poor lighting, crowded backgrounds.

| Step | Action |
|------|--------|
| 1 | Open **Image Detection** page |
| 2 | Select `demo_data/images/difficult/difficult_waste_01.jpg` |
| 3 | Click **DETECT** |

**Try all 5 difficult images and record:**
- Whether any objects were detected at all
- If detected, were they correct classes?
- Were any obvious objects missed (false negatives)?
- Were any non-garbage items falsely flagged (false positives)?

---

## TEST 5 — Video Detection: Multi-Object

**Goal:** Test real-time detection on moving video with multiple objects.

**Prerequisite:** Download videos manually from the URLs in `test_manifest.csv` (see Video section below).

| Step | Action |
|------|--------|
| 1 | Open **Video Detection** page |
| 2 | Select `demo_data/videos/multi_object/garbage_video_01.mp4` |
| 3 | Click **START** |

**Test these operations during playback:**
- **Pause / Resume:** Does detection freeze and resume correctly?
- **Stop:** Does it cleanly stop and show final summary?
- **Save processed video:** Does the output file contain annotated frames?

**Expected:**
- Bounding boxes appear on moving objects
- Multiple boxes visible simultaneously when scene contains multiple items
- Status updates dynamically as objects enter/leave frame

---

## TEST 6 — Live Camera

**Goal:** Real-time detection on webcam feed.

| Step | Action |
|------|--------|
| 1 | Open **Live Camera** page |
| 2 | Click **START CAMERA** |
| 3 | Hold up 1 object → observe single box |
| 4 | Hold up 2–3 objects → observe multiple boxes |
| 5 | Show a clean surface → verify CLEAN status |

**Expected:**
- Real-time bounding boxes at ~30 FPS (on GPU)
- Correct class labels and confidence values
- CLEAN status when no garbage is in frame

---

## TEST 7 — History & Analytics

**Goal:** Verify detections are logged and analytics update correctly.

| Step | Action |
|------|--------|
| 1 | Run Tests 1–3 above |
| 2 | Open **History** page |
| 3 | Verify each detection session appears with correct timestamp, source, and counts |
| 4 | Open **Analytics** page |
| 5 | Verify charts reflect the detections you just made |

---

## Video Downloads (Manual)

Videos cannot be hotlinked directly. Download them manually from these Pexels pages:

| File | URL | Category |
|------|-----|----------|
| `garbage_video_01.mp4` | https://www.pexels.com/video/a-person-sorting-through-garbage-3129657/ | multi_object |
| `multi_object_video_01.mp4` | https://www.pexels.com/video/recycling-conveyor-belt-at-waste-management-facility-6990241/ | multi_object |
| `street_waste_video_01.mp4` | https://www.pexels.com/video/garbage-truck-collecting-waste-from-street-11115607/ | street_waste |
| `recycling_video_01.mp4` | https://www.pexels.com/video/colorful-recycling-bins-for-waste-segregation-17869493/ | recycling |
| `difficult_video_01.mp4` | https://www.pexels.com/video/landfill-site-environmental-pollution-3174347/ | difficult |

Place downloaded `.mp4` files in the corresponding subfolder under `demo_data/videos/`.

---

## Reporting Template

For each test, record:

```
Test #: ___
File:   demo_data/images/.../filename.jpg
Boxes:  N
Classes: {class1: count1, class2: count2, ...}
Confidences: [0.XX, 0.XX, ...]
Status: CLEAN / DIRTY / REVIEW
Notes:  (any unexpected behavior)
```
