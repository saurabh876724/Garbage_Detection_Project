"""Reusable dashboard widgets: cards, badges, video surface, tables."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QProgressBar, QTableWidget, QTableWidgetItem,
                               QHeaderView, QVBoxLayout, QWidget)

import config
from ui.styles import (ACCENT, BORDER, DIM, ELEVATED, MONO_FAMILY, MUTED,
                       STATUS_COLOR, TEXT, icon_svg)


class Card(QFrame):
    """Rounded panel with an optional title and a content layout."""

    def __init__(self, title: str = "", orientation: str = "vertical",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("SectionTitle")
            outer.addWidget(label)
        self.body = (QVBoxLayout() if orientation == "vertical"
                     else QHBoxLayout())
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget: QWidget) -> None:
        self.body.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)


class MetricCard(Card):
    """Big number with a caption, e.g. TOTAL DETECTIONS / 128."""

    def __init__(self, label: str, value: str = "-", color: str = TEXT,
                 parent=None):
        super().__init__(parent=parent)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        self.value_label.setStyleSheet(f"color: {color};")
        self.caption = QLabel(label.upper())
        self.caption.setObjectName("MetricLabel")
        self.add(self.value_label)
        self.add(self.caption)

    def set_value(self, value: str | int | float, color: str | None = None,
                  caption: str | None = None) -> None:
        self.value_label.setText(str(value))
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        if caption is not None:
            self.caption.setText(caption.upper())


class StatusBadge(QLabel):
    """CLEAN / DIRTY / REVIEW indicator with a one-line explanation."""

    def __init__(self, parent=None):
        super().__init__("IDLE", parent)
        self.setObjectName("StatusBadge")
        self.setAlignment(Qt.AlignCenter)
        self.set_status("INFO", "waiting for a source")

    def set_status(self, status: str, detail: str = "") -> None:
        color = STATUS_COLOR.get(status, STATUS_COLOR["INFO"])
        self.setText(f"● {status}")
        self.setToolTip(detail)
        self.setStyleSheet(
            f"QLabel#StatusBadge {{ background: transparent; "
            f"color: {color}; border: none; "
            f"border-radius: 0; padding: 2px 8px; font-weight: 600; "
            f"font-size: 11px; font-family: {MONO_FAMILY}; "
            f"letter-spacing: 0.5px; }}")


class VideoSurface(QLabel):
    """Scales BGR frames to the widget while keeping the aspect ratio."""

    def __init__(self, placeholder: str = "No source running", parent=None):
        super().__init__(placeholder, parent)
        self.setObjectName("VideoSurface")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 380)
        self.setScaledContents(False)
        self._placeholder = placeholder
        self._show_brackets = True

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self._show_brackets = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))

    def set_message(self, text: str) -> None:
        self._show_brackets = True
        self.clear()
        self.setText(text)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.pixmap() is None or self.pixmap().isNull():
            from PySide6.QtGui import QPainter, QPen
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            margin = 40
            length = 28
            pen = QPen(QColor(DIM), 1.5)
            painter.setPen(pen)
            corners = [
                [(margin, margin + length), (margin, margin), (margin + length, margin)],
                [(w - margin - length, margin), (w - margin, margin), (w - margin, margin + length)],
                [(w - margin, h - margin - length), (w - margin, h - margin), (w - margin - length, h - margin)],
                [(margin + length, h - margin), (margin, h - margin), (margin, h - margin - length)],
            ]
            for pts in corners:
                for i in range(len(pts) - 1):
                    painter.drawLine(*pts[i], *pts[i + 1])
            icon_size = 32
            cx, cy = w // 2, h // 2 - 20
            painter.setPen(QPen(QColor(DIM), 1.5))
            painter.drawRect(cx - icon_size // 2, cy - icon_size // 2,
                             icon_size, icon_size)
            painter.drawLine(cx - 6, cy, cx + 6, cy)
            painter.drawLine(cx, cy - 6, cx, cy + 6)
            painter.end()


class ClassCountTable(QTableWidget):
    """Per-class counts with a share bar - always lists all 10 classes."""

    def __init__(self, parent=None):
        super().__init__(len(config.GARBAGE_CLASSES), 3, parent)
        self.setHorizontalHeaderLabels(("Class", "Objects", "Share"))
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionMode(QTableWidget.NoSelection)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.NoFocus)
        self.set_counts({})

    def set_counts(self, counts: dict[str, int],
                   unique: dict[str, int] | None = None) -> None:
        total = sum(counts.values()) if counts else 0
        for row, name in enumerate(config.GARBAGE_CLASSES):
            value = int(counts.get(name, 0))
            label = QTableWidgetItem(name.capitalize())
            dot = QColor(config.CLASS_COLORS_BGR[row][2],
                         config.CLASS_COLORS_BGR[row][1],
                         config.CLASS_COLORS_BGR[row][0])
            label.setForeground(QBrush(dot))
            self.setItem(row, 0, label)

            text = str(value)
            if unique is not None:
                text = f"{value}  ({unique.get(name, 0)} unique)"
            count_item = QTableWidgetItem(text)
            count_item.setTextAlignment(Qt.AlignCenter)
            count_item.setFont(_mono_font())
            if value == 0:
                count_item.setForeground(QBrush(QColor(DIM)))
            self.setItem(row, 1, count_item)

            share = (value / total * 100) if total else 0.0
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(share)))
            bar.setFormat(f"{share:.0f}%")
            bar.setTextVisible(True)
            bar.setMinimumWidth(48)
            bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {dot.name()}; }}"
                f"QProgressBar {{ font-family: {MONO_FAMILY}; font-size: 9px; }}")
            self.setCellWidget(row, 2, bar)


class InfoTable(Card):
    """Key/value panel used for system, model and dataset information."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent=parent)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(18)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)
        self.add_layout(self.grid)

    def set_items(self, items: dict[str, str]) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for row, (key, value) in enumerate(items.items()):
            key_label = QLabel(key)
            key_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            display_text = str(value) if value not in (None, "") else "-"
            value_label = QLabel(display_text)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            if _looks_technical(display_text):
                value_label.setFont(_mono_font())
                value_label.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
            self.grid.addWidget(key_label, row, 0, Qt.AlignTop)
            self.grid.addWidget(value_label, row, 1, Qt.AlignTop)


class Banner(QLabel):
    """Inline message strip for errors and confirmations."""

    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.hide()

    def show_message(self, text: str, kind: str = "info") -> None:
        colors = {"info": STATUS_COLOR["INFO"], "error": STATUS_COLOR["DIRTY"],
                  "ok": STATUS_COLOR["CLEAN"], "warn": STATUS_COLOR["REVIEW"]}
        color = colors.get(kind, colors["info"])
        self.setText(text)
        self.setStyleSheet(
            f"background: {color}18; color: {color}; "
            f"border: 1px solid {color}44; border-radius: 4px; "
            f"padding: 8px 12px; font-size: 12px;")
        self.show()

    def clear_message(self) -> None:
        self.setText("")
        self.hide()


class KeyValueStrip(QWidget):
    """Horizontal row of small label/value pairs (FPS, device, model...)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        self._values: dict[str, QLabel] = {}
        self.row = layout

    def add_field(self, key: str, value: str = "-") -> None:
        box = QVBoxLayout()
        box.setSpacing(2)
        caption = QLabel(key.upper())
        caption.setObjectName("MetricLabel")
        caption.setStyleSheet(f"color: {MUTED}; font-size: 9px; "
                              f"text-transform: uppercase; letter-spacing: 0.6px;")
        display = QLabel(value)
        display.setFont(_mono_font())
        display.setStyleSheet(f"color: {TEXT}; font-weight: 500; font-size: 13px;")
        box.addWidget(caption)
        box.addWidget(display)
        self._values[key] = display
        self.row.addLayout(box)

    def set_value(self, key: str, value: str | int | float,
                  color: str | None = None) -> None:
        label = self._values.get(key)
        if label is None:
            return
        label.setText(str(value))
        label.setStyleSheet(f"color: {color or TEXT}; font-weight: 500; "
                            f"font-family: {MONO_FAMILY}; font-size: 12px;")

    def finish(self) -> None:
        self.row.addStretch(1)


def _mono_font():
    from PySide6.QtGui import QFont
    f = QFont(MONO_FAMILY.split('"')[1] if '"' in MONO_FAMILY else MONO_FAMILY)
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(10)
    return f


def _looks_technical(text: str) -> bool:
    if not text or text == "-":
        return False
    indicators = ("ms", "fps", "MB", "CUDA", "cpu", "GPU", "v1.", "best.pt",
                  "pt", "Hz", "%", "YOLO", "PyTorch", "OpenCV", "SQLite")
    return any(ind in text for ind in indicators) or text.replace(".", "").replace("-", "").isdigit()
