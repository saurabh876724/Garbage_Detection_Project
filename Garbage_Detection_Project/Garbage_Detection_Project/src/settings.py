"""Persisted runtime settings.

The Settings page edits these values; they survive restarts in
data/settings.json. CLI flags override them for a single run only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

import config
from src.logging_setup import get_logger

log = get_logger("settings")

_DEVICES = ("auto", "cpu", "0")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Settings:
    confidence: float = config.CONF_THRESHOLD
    iou: float = config.IOU_THRESHOLD
    image_size: int = config.INPUT_SIZE
    dirty_threshold: int = config.DIRTY_THRESHOLD
    camera_index: int = config.CAMERA_INDEX
    device: str = "auto"
    weights: str = ""
    tracking_enabled: bool = config.TRACKING_ENABLED
    evidence_enabled: bool = True
    alert_enabled: bool = config.ALERT_ENABLED
    alert_threshold: int = config.ALERT_COUNT_THRESHOLD
    alert_cooldown: float = config.ALERT_COOLDOWN_SECONDS
    alert_sound: str = config.ALERT_SOUND_WAV
    save_processed_video: bool = config.VIDEO_SAVE_ENABLED
    keep_history_entries: int = config.MAX_HISTORY_ENTRIES

    def __post_init__(self) -> None:
        self.confidence = _clamp(float(self.confidence), 0.05, 0.95)
        self.iou = _clamp(float(self.iou), 0.10, 0.90)
        self.image_size = int(self.image_size)
        self.dirty_threshold = max(1, int(self.dirty_threshold))
        self.camera_index = max(0, int(self.camera_index))
        self.alert_threshold = max(1, int(self.alert_threshold))
        self.alert_cooldown = _clamp(float(self.alert_cooldown), 1.0, 3600.0)
        if self.device not in _DEVICES:
            self.device = "auto"

    @property
    def weights_path(self):
        """Configured weights, else the newest trained model."""
        return self.weights or str(config.default_weights())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_settings(path=None) -> Settings:
    """Load settings from disk, falling back to config defaults."""
    path = path or config.SETTINGS_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("settings unreadable (%s) - using defaults", exc)
        return Settings()
    settings = Settings.from_dict(data)
    log.info("settings loaded from %s", path)
    return settings


def save_settings(settings: Settings, path=None) -> bool:
    """Persist settings; returns False (never raises) when the write fails."""
    path = path or config.SETTINGS_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.to_dict(), indent=2),
                        encoding="utf-8")
    except OSError as exc:
        log.error("could not save settings to %s: %s", path, exc)
        return False
    log.info("settings saved to %s", path)
    return True


def system_info() -> dict[str, str]:
    """Versions and hardware facts shown on the Settings/About pages."""
    import platform
    import sys

    import cv2
    import numpy
    import torch
    from ultralytics import __version__ as yolo_version

    gpu = "unavailable"
    vram = ""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = f"{props.total_memory / 1024 ** 3:.2f} GB"

    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} "
                    f"({platform.machine()})",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "n/a",
        "cuda_available": str(torch.cuda.is_available()),
        "gpu": gpu,
        "vram": vram,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "ultralytics": yolo_version,
        "model": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "weights": str(config.default_weights()),
        "classes": ", ".join(config.GARBAGE_CLASSES),
    }
