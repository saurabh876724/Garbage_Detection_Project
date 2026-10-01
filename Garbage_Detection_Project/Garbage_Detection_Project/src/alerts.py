"""Alerts: audible warning plus a visual state the UI can render.

An alert fires only when a scene is DIRTY (or REVIEW, if configured) with at
least `count_threshold` objects, and never more often than the cooldown -
otherwise a live camera would beep on every frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import config
from src.logging_setup import get_logger

log = get_logger("alerts")


@dataclass
class AlertState:
    """Snapshot of the alert system for the UI."""

    active: bool
    message: str
    triggered_total: int
    seconds_since_last: float | None
    cooldown_remaining: float


class AlertManager:
    def __init__(self, enabled: bool | None = None,
                 count_threshold: int | None = None,
                 cooldown_seconds: float | None = None,
                 sound_path: str | None = None,
                 alert_on_review: bool = False):
        self.enabled = config.ALERT_ENABLED if enabled is None else enabled
        self.count_threshold = (config.ALERT_COUNT_THRESHOLD
                                if count_threshold is None else count_threshold)
        self.cooldown = (config.ALERT_COOLDOWN_SECONDS
                         if cooldown_seconds is None else cooldown_seconds)
        self.sound_path = (config.ALERT_SOUND_WAV if sound_path is None
                           else sound_path)
        self.alert_on_review = alert_on_review
        self.triggered_total = 0
        self._last_alert: float | None = None
        self._last_message = ""
        self._audio_warned = False

    def update(self, garbage_count: int, status: str = config.STATUS_DIRTY,
               source: str = "") -> bool:
        """Evaluate one scene; returns True when an alert was emitted."""
        if not self.enabled or not self._should_alert(garbage_count, status):
            return False
        now = time.monotonic()
        if self._last_alert is not None and now - self._last_alert < self.cooldown:
            return False
        self._last_alert = now
        self.triggered_total += 1
        self._last_message = (f"{status}: {garbage_count} garbage object(s)"
                              f" detected" + (f" ({source})" if source else ""))
        log.warning("alert #%d - %s", self.triggered_total, self._last_message)
        self._play()
        return True

    def _should_alert(self, count: int, status: str) -> bool:
        if count < self.count_threshold:
            return False
        if status == config.STATUS_DIRTY:
            return True
        return status == config.STATUS_REVIEW and self.alert_on_review

    def state(self) -> AlertState:
        """Current alert state, including remaining cooldown."""
        now = time.monotonic()
        since = None if self._last_alert is None else now - self._last_alert
        remaining = 0.0 if since is None else max(0.0, self.cooldown - since)
        active = remaining > 0 and bool(self._last_message)
        return AlertState(active=active,
                          message=self._last_message if active else "",
                          triggered_total=self.triggered_total,
                          seconds_since_last=since,
                          cooldown_remaining=round(remaining, 1))

    def reset(self) -> None:
        self.triggered_total = 0
        self._last_alert = None
        self._last_message = ""

    def _play(self) -> None:
        """Non-blocking sound; degrades silently when no audio is available."""
        try:
            import winsound
        except ImportError:
            if not self._audio_warned:
                log.info("winsound unavailable - alerts are visual only")
                self._audio_warned = True
            return

        wav = Path(self.sound_path) if self.sound_path else None
        try:
            if wav and wav.is_file():
                winsound.PlaySound(str(wav),
                                   winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except RuntimeError as exc:
            if not self._audio_warned:
                log.warning("alert sound failed (%s) - visual alerts only", exc)
                self._audio_warned = True
