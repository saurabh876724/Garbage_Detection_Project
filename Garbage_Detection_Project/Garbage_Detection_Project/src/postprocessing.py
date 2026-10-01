"""Drawing detections and the status HUD onto frames.

Boxes are coloured per class so a scene stays readable, and labels are placed
in the first free slot (above, inside, below) to limit overlap.
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from src.detector import Detection, SceneResult

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
BAR_HEIGHT = 40


def _color(class_id: int) -> tuple[int, int, int]:
    return config.CLASS_COLORS_BGR[class_id % len(config.CLASS_COLORS_BGR)]


def _text_size(text: str) -> tuple[int, int]:
    (width, height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE,
                                                THICKNESS)
    return width + 8, height + baseline + 8


def _free_slot(x1: int, y1: int, x2: int, y2: int, w: int, h: int,
               frame_w: int, frame_h: int,
               placed: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    """Top-left corner for a label of size (w, h) that avoids placed labels."""
    candidates = ((x1, y1 - h), (x1, y1), (x1, y2 + 2), (x2 - w, y1 - h))
    for cx, cy in candidates:
        rect = (cx, cy, cx + w, cy + h)
        if rect[0] < 0 or rect[1] < 0 or rect[2] > frame_w or rect[3] > frame_h:
            continue
        if any(not (rect[2] <= o[0] or rect[0] >= o[2]
                    or rect[3] <= o[1] or rect[1] >= o[3]) for o in placed):
            continue
        return cx, cy
    return max(0, min(x1, frame_w - w)), max(0, y1 - h)


def draw_detections(frame: np.ndarray, detections: list[Detection],
                    weak: list[Detection] | None = None) -> np.ndarray:
    """Return a copy of the frame with boxes and labels drawn on it."""
    out = frame.copy()
    height, width = out.shape[:2]
    placed: list[tuple[int, int, int, int]] = []

    for det in detections:
        color = _color(det.class_id)
        x1, y1, x2, y2 = det.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = det.label
        tw, th = _text_size(text)
        tx, ty = _free_slot(x1, y1, x2, y2, tw, th, width, height, placed)
        placed.append((tx, ty, tx + tw, ty + th))
        cv2.rectangle(out, (tx, ty), (tx + tw, ty + th), color, -1)
        cv2.putText(out, text, (tx + 4, ty + th - 6), FONT, FONT_SCALE,
                    (15, 15, 15), THICKNESS, cv2.LINE_AA)

    for det in weak or []:
        # Low-confidence hits are hinted with a thin dashed-style outline so
        # they are visible but never presented as confirmed garbage.
        x1, y1, x2, y2 = det.box
        for start in range(x1, x2, 12):
            cv2.line(out, (start, y1), (min(start + 6, x2), y1),
                     config.STATUS_COLORS_BGR[config.STATUS_REVIEW], 1)
            cv2.line(out, (start, y2), (min(start + 6, x2), y2),
                     config.STATUS_COLORS_BGR[config.STATUS_REVIEW], 1)

    return out


def draw_status_bar(frame: np.ndarray, status: str, total: int,
                    fps: float | None = None, device: str = "",
                    inference_ms: float | None = None) -> np.ndarray:
    """Draw the top HUD bar: status, object count, FPS and device."""
    out = frame.copy()
    width = out.shape[1]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, BAR_HEIGHT), (24, 24, 28), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    color = config.STATUS_COLORS_BGR.get(status,
                                         config.STATUS_COLORS_BGR["INFO"])
    cv2.circle(out, (18, BAR_HEIGHT // 2), 7, color, -1)
    parts = [f"{status}", f"objects: {total}"]
    if fps is not None:
        parts.append(f"fps: {fps:.1f}")
    if inference_ms is not None:
        parts.append(f"inference: {inference_ms:.0f} ms")
    if device:
        parts.append(f"device: {device}")
    cv2.putText(out, "   |   ".join(parts), (34, BAR_HEIGHT // 2 + 6), FONT,
                FONT_SCALE + 0.05, (235, 235, 235), THICKNESS, cv2.LINE_AA)
    return out


def annotate_scene(frame: np.ndarray, result: SceneResult,
                   fps: float | None = None, device: str = "",
                   header: bool = True) -> np.ndarray:
    """Detections plus HUD in one call - what the CLI and UI display."""
    out = draw_detections(frame, result.detections, result.weak_detections)
    if header:
        out = draw_status_bar(out, result.status, result.total, fps, device,
                              result.inference_ms)
    return out


def status_text(status: str, result: SceneResult) -> str:
    """One-line explanation of the verdict, shown under the status badge."""
    if status == config.STATUS_DIRTY:
        top = max(result.counts.items(), key=lambda kv: kv[1])
        return (f"{result.total} garbage object(s) detected "
                f"(most: {top[0]} x{top[1]})")
    if status == config.STATUS_REVIEW:
        return (f"only low-confidence hits (<{config.CONF_THRESHOLD:.2f}) - "
                "scene needs review")
    return "no garbage detected - scene is clean"
