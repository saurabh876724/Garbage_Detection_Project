"""Input handling: image loading, camera/video capture, frame sanity checks.

cv2.imread applies EXIF orientation, which matches what the model, the audit
scripts and the visual tools see - keep using it rather than IMREAD_UNCHANGED.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config
from src.logging_setup import get_logger

log = get_logger("preprocessing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


class MediaError(RuntimeError):
    """Raised when an image or video cannot be used for detection."""


def classify_source(source: str | int | Path) -> tuple[str, object]:
    """Return ("camera", index) or ("image", path) or ("video", path)."""
    if isinstance(source, int) or (isinstance(source, str)
                                   and source.strip().isdigit()):
        return "camera", int(source)
    path = Path(source)
    if not path.is_file():
        raise MediaError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image", path
    if suffix in VIDEO_EXTENSIONS:
        return "video", path
    raise MediaError(f"unsupported file type '{suffix}': {path.name}")


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR frame."""
    path = Path(path)
    frame = cv2.imread(str(path))
    if frame is None:
        raise MediaError(
            f"could not decode image: {path.name}\n"
            "The file is either corrupted or not a supported image format.")
    if frame.size == 0:
        raise MediaError(f"image is empty: {path.name}")
    log.info("loaded image %s (%dx%d)", path.name, frame.shape[1],
             frame.shape[0])
    return frame


def open_capture(source: int | str | Path) -> cv2.VideoCapture:
    """Open a webcam index or video file, raising MediaError on failure."""
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            raise MediaError(
                f"camera {source} is unavailable.\n"
                "Check that it is connected, not used by another app, and "
                "that Windows camera access is allowed "
                "(Settings > Privacy & security > Camera).")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        log.info("camera %s opened", source)
        return capture

    path = str(source)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise MediaError(f"could not open video: {Path(path).name}\n"
                         "The codec may be unsupported or the file corrupt.")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    log.info("video opened: %s frames, %.1f fps", frames, fps)
    return capture


def video_metadata(capture: cv2.VideoCapture) -> dict:
    """Frame count, fps and resolution of an opened capture."""
    return {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def valid_frame(frame: np.ndarray | None) -> bool:
    """True when the frame is a non-empty 2D/3D uint8 image."""
    return (frame is not None and isinstance(frame, np.ndarray)
            and frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0)


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Downscale for display only; never used to feed the model."""
    if width <= 0 or frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)


def bgr_to_rgb_bytes(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as JPEG bytes for Qt display."""
    ok, buffer = cv2.imencode(".jpg", frame,
                              [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise MediaError("frame could not be encoded for display")
    return buffer.tobytes()
