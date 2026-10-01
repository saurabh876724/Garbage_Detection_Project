"""Central configuration for the garbage detection application.

Every tunable value lives here (or in the persisted runtime settings written
by the Settings page - see src/settings.py). Nothing downstream should
hardcode thresholds, paths or class names.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Model ------------------------------------------------------------------
GARBAGE_CLASSES = (
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
)
CLASS_NAMES = {i: name for i, name in enumerate(GARBAGE_CLASSES)}
NUM_CLASSES = len(GARBAGE_CLASSES)

MODEL_NAME = "YOLOv8n Garbage Detector"
MODEL_VERSION = "v1.0"

CONF_THRESHOLD = 0.35          # minimum confidence for a detection to count
IOU_THRESHOLD = 0.45           # NMS IoU threshold
INPUT_SIZE = 640               # inference image size
REVIEW_THRESHOLD = 0.20        # below CONF but >= REVIEW -> scene is REVIEW
MAX_DETECTIONS = 100           # safety cap for a single frame
MAX_BOX_AREA_FRACTION = 0.90   # a box covering more of the frame than this is a
                               # degenerate whole-scene artefact, not an object

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
TRAINED_WEIGHTS_GLOB = "runs/detect/*/weights/best.pt"
BASE_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n.pt"
DATA_YAML = PROJECT_ROOT / "data.yaml"


def default_weights() -> Path:
    """Newest trained best.pt under runs/detect/, else the base weights."""
    candidates = sorted(PROJECT_ROOT.glob(TRAINED_WEIGHTS_GLOB),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    return BASE_WEIGHTS


def trained_weights() -> list[Path]:
    """All trained best.pt files, oldest first."""
    return sorted(PROJECT_ROOT.glob(TRAINED_WEIGHTS_GLOB),
                  key=lambda p: p.stat().st_mtime)


# --- Scene status -----------------------------------------------------------
# CLEAN / DIRTY / REVIEW are derived from detections, never a model class.
DIRTY_THRESHOLD = 1            # >= N confident garbage objects means DIRTY
STATUS_CLEAN = "CLEAN"
STATUS_DIRTY = "DIRTY"
STATUS_REVIEW = "REVIEW"

# --- Runtime directories ----------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
EVIDENCE_FRAME_INTERVAL = 300  # auto-evidence every N dirty frames in a stream
LOG_DIR = PROJECT_ROOT / "logs"
DATABASE_DIR = PROJECT_ROOT / "database"

HISTORY_DB = DATA_DIR / "detection_history.db"
SETTINGS_PATH = DATA_DIR / "settings.json"

# --- Capture / camera -------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS_TARGET = 30

# --- Alerts -----------------------------------------------------------------
ALERT_ENABLED = True
ALERT_COUNT_THRESHOLD = 3      # garbage count that triggers an audible alert
ALERT_COOLDOWN_SECONDS = 30.0  # minimum gap between audible alerts
ALERT_SOUND_WAV = ""           # optional .wav file; empty = system beep

# --- Tracking ---------------------------------------------------------------
TRACKING_ENABLED = False       # ByteTrack; unique-object counting needs this
TRACKER_CONFIG = "bytetrack.yaml"

# --- History ----------------------------------------------------------------
MAX_HISTORY_ENTRIES = 5000     # oldest rows are pruned beyond this

# --- User interface ---------------------------------------------------------
WINDOW_TITLE = "AI Garbage Detection - Intelligent Waste Monitoring System"
WINDOW_SIZE = (1360, 860)
PREVIEW_WIDTH = 960            # video widget width in pixels
VIDEO_SAVE_ENABLED = True

# Status colours (BGR for OpenCV overlays, hex for the Qt theme).
STATUS_COLORS_BGR = {
    STATUS_CLEAN: (80, 200, 80),
    STATUS_DIRTY: (60, 60, 230),
    STATUS_REVIEW: (60, 200, 240),
    "INFO": (220, 200, 120),
}
STATUS_COLORS_HEX = {
    STATUS_CLEAN: "#2ecc71",
    STATUS_DIRTY: "#e74c3c",
    STATUS_REVIEW: "#f1c40f",
    "INFO": "#3498db",
}

# Per-class box colours (BGR), cycled by class id for stable visuals.
CLASS_COLORS_BGR = (
    (80, 80, 220), (60, 180, 75), (200, 120, 60), (180, 105, 250),
    (60, 60, 220), (140, 140, 140), (250, 200, 90), (90, 200, 200),
    (160, 90, 160), (110, 160, 60),
)


def ensure_dirs() -> None:
    """Create the runtime directories the application writes to."""
    for path in (DATA_DIR, EVIDENCE_DIR, LOG_DIR, DATABASE_DIR):
        path.mkdir(parents=True, exist_ok=True)
