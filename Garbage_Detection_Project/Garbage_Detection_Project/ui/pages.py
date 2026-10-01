"""Operational pages: dashboard overview, live camera, image and video."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QProgressBar, QPushButton, QSizePolicy,
                               QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import config
from src.detector import SceneResult
from ui.charts import BarChart
from ui.components import (Banner, Card, ClassCountTable, InfoTable,
                           KeyValueStrip, MetricCard, StatusBadge, VideoSurface)
from ui.styles import (ACCENT, BORDER, DIM, ELEVATED, GREEN, MONO_FAMILY,
                       MUTED, RED, STATUS_COLOR, TEXT, YELLOW)


def _mono(size: int = 10) -> QFont:
    f = QFont(MONO_FAMILY.split('"')[1] if '"' in MONO_FAMILY else MONO_FAMILY)
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(size)
    return f


class DashboardPage(QWidget):
    """Landing page: headline metrics, live preview, charts, recent events."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(14)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.card_total = MetricCard("Total detections", "0")
        self.card_today = MetricCard("Today", "0", ACCENT)
        self.card_status = MetricCard("Current status", "-", GREEN)
        self.card_conf = MetricCard("Avg confidence", "-", ACCENT)
        for card in (self.card_total, self.card_today, self.card_status,
                     self.card_conf):
            metrics.addWidget(card)
        root.addLayout(metrics)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        self.preview = Card("Live preview")
        self.surface = VideoSurface("Start a camera, image or video to see "
                                   "detections here")
        self.preview.add(self.surface)
        middle.addWidget(self.preview, stretch=3)

        self.chart = BarChart("Class distribution (current session)")
        self.chart.empty_text = "No detections yet in this session"
        chart_card = Card("")
        chart_card.add(self.chart)
        middle.addWidget(chart_card, stretch=2)
        root.addLayout(middle, stretch=3)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        self.recent = Card("Recent detections")
        self.recent_table = QTableWidget(0, 4)
        self.recent_table.setHorizontalHeaderLabels(
            ("Timestamp", "Source", "Status", "Objects"))
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.recent.add(self.recent_table)
        bottom.addWidget(self.recent, stretch=3)

        self.system = InfoTable("System status")
        bottom.addWidget(self.system, stretch=2)
        root.addLayout(bottom, stretch=2)

    # -- updates ------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read history and detector facts (called on page show)."""
        from src import analytics

        engine = self.context.engine
        summary = analytics.overview(engine.history)
        self.card_total.set_value(summary["total_events"])
        self.card_today.set_value(summary["today_events"])
        self.card_conf.set_value(
            f"{summary['average_confidence'] * 100:.1f}%"
            if summary["total_events"] else "-")

        live = self.context.last_result
        status = live.status if live else "IDLE"
        self.card_status.set_value(status,
                                   STATUS_COLOR.get(status, MUTED))

        info = engine.detector.info()
        self.system.set_items({
            "Model": f"{info['model']} {info['version']}",
            "Weights": Path(info["weights"]).name,
            "Device": f"{info['device']} - {info['gpu']}",
            "VRAM": info["vram"],
            "Confidence": f"{info['confidence']:.2f}",
            "Tracking": "on (ByteTrack)" if info["tracking"] else "off",
            "History rows": str(summary["total_events"]),
        })
        self._load_recent()

    def _load_recent(self) -> None:
        events = self.context.engine.history.recent_events(8)
        self.recent_table.setRowCount(len(events))
        for row, event in enumerate(events):
            cells = (event.timestamp.replace("T", " "), event.source,
                     event.status, str(event.total_count))
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setFont(_mono(9))
                if column == 2:
                    item.setForeground(QBrush(QColor(STATUS_COLOR.get(
                        event.status, STATUS_COLOR["INFO"]))))
                    item.setFont(_mono(9))
                if column == 3:
                    item.setFont(_mono(9))
                    item.setTextAlignment(Qt.AlignCenter)
                self.recent_table.setItem(row, column, item)
        if not events:
            self.recent_table.setRowCount(1)
            empty = QTableWidgetItem("No detections recorded yet")
            empty.setForeground(QBrush(QColor(DIM)))
            self.recent_table.setSpan(0, 0, 1, 4)
            self.recent_table.setItem(0, 0, empty)

    def on_frame(self, frame, result: SceneResult, fps: float) -> None:
        self.surface.set_frame(frame)
        counts = {k: v for k, v in result.counts.items() if v}
        self.chart.set_data(list(counts), list(counts.values()))
        status = result.status
        self.card_status.set_value(status, STATUS_COLOR.get(status, MUTED))
        self.card_conf.set_value(f"{result.average_confidence * 100:.0f}%")


class LiveCameraPage(QWidget):
    """Monitoring view: camera feed, status, counts and evidence capture."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(12)

        header = QHBoxLayout()
        self.badge = StatusBadge()
        self.detail = QLabel("Camera stopped")
        self.detail.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        header.addWidget(self.badge)
        header.addWidget(self.detail, stretch=1)
        left.addLayout(header)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.setObjectName("Primary")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setObjectName("Danger")
        self.btn_stop.setEnabled(False)
        self.btn_capture = QPushButton("Capture Evidence")
        self.btn_capture.setEnabled(False)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_capture)
        controls.addStretch(1)

        controls.addWidget(QLabel("Camera"))
        self.spin_camera = QSpinBox()
        self.spin_camera.setRange(0, 9)
        self.spin_camera.setValue(config.CAMERA_INDEX)
        controls.addWidget(self.spin_camera)

        controls.addWidget(QLabel("Confidence"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(int(config.CONF_THRESHOLD * 100))
        self.slider_conf.setFixedWidth(150)
        self.label_conf = QLabel(f"{config.CONF_THRESHOLD:.2f}")
        self.label_conf.setFont(_mono(10))
        controls.addWidget(self.slider_conf)
        controls.addWidget(self.label_conf)
        left.addLayout(controls)

        self.surface = VideoSurface("Press START CAMERA to begin monitoring")
        self.surface.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.surface, stretch=1)

        self.strip = KeyValueStrip()
        for field in ("fps", "objects", "inference", "device", "unique"):
            self.strip.add_field(field)
        self.strip.finish()
        left.addWidget(self.strip)

        self.banner = Banner()
        left.addWidget(self.banner)
        root.addLayout(left, stretch=4)

        right = QVBoxLayout()
        right.setSpacing(12)
        self.counts = Card("Detections by class")
        self.count_table = ClassCountTable()
        self.counts.add(self.count_table)
        right.addWidget(self.counts)

        self.session = InfoTable("Session")
        right.addWidget(self.session)
        root.addLayout(right, stretch=1)

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_capture.clicked.connect(self.capture)
        self.slider_conf.valueChanged.connect(self._conf_changed)

    # -- actions ------------------------------------------------------------

    def _conf_changed(self, value: int) -> None:
        confidence = value / 100
        self.label_conf.setText(f"{confidence:.2f}")
        self.context.set_confidence(confidence)

    def start(self) -> None:
        try:
            self.context.start_camera(self.spin_camera.value())
        except RuntimeError as exc:
            self.banner.show_message(str(exc), "error")
            return
        self.banner.clear_message()
        self.badge.set_status("INFO", "camera starting")
        self.detail.setText("Camera running")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)

    def stop(self) -> None:
        self.context.stop_source()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.detail.setText("Camera stopped")
        self.badge.set_status("INFO", "camera stopped")

    def capture(self) -> None:
        path = self.context.capture_evidence()
        if path:
            self.banner.show_message(f"Evidence saved: {Path(path).name}", "ok")

    # -- updates ------------------------------------------------------------

    def refresh(self) -> None:
        info = self.context.engine.detector.info()
        self.strip.set_value("device", info["device"])
        self.strip.set_value("unique", "n/a (tracking off)"
                             if not info["tracking"] else "0")
        self.slider_conf.blockSignals(True)
        self.slider_conf.setValue(int(info["confidence"] * 100))
        self.label_conf.setText(f"{info['confidence']:.2f}")
        self.slider_conf.blockSignals(False)
        self.spin_camera.setValue(self.context.settings.camera_index)

    def on_frame(self, frame, result: SceneResult, fps: float) -> None:
        self.surface.set_frame(frame)
        self.badge.set_status(result.status,
                              f"{result.total} object(s) detected")
        self.detail.setText(f"{result.total} object(s) - "
                            f"avg confidence {result.average_confidence * 100:.0f}%")
        self.strip.set_value("fps", f"{fps:.1f}")
        self.strip.set_value("objects", result.total)
        self.strip.set_value("inference", f"{result.inference_ms:.0f} ms")
        unique = result.unique_objects
        self.strip.set_value("unique", unique if unique is not None else "n/a")
        self.count_table.set_counts(result.counts)

    def on_finished(self, summary: dict) -> None:
        self.session.set_items({
            "Frames": summary.get("frames", 0),
            "Detections": summary.get("detections", 0),
            "Peak objects": summary.get("peak_objects", 0),
            "Dirty frames": summary.get("dirty_frames", 0),
            "Review frames": summary.get("review_frames", 0),
            "Avg confidence": f"{summary.get('average_confidence', 0) * 100:.1f}%",
            "Unique objects": summary.get("unique_objects") or "n/a",
            "Duration": f"{summary.get('seconds', 0):.1f} s",
            "Session status": summary.get("status", "-"),
        })


class ImagePage(QWidget):
    """Single-image detection with a saved annotated copy."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.image_path: Path | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(12)
        bar = QHBoxLayout()
        self.btn_open = QPushButton("Upload Image")
        self.btn_open.setObjectName("Primary")
        self.btn_evidence = QPushButton("Save Evidence")
        self.btn_evidence.setEnabled(False)
        self.path_label = QLabel("No image selected")
        self.path_label.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_evidence)
        bar.addWidget(self.path_label, stretch=1)
        left.addLayout(bar)

        self.surface = VideoSurface("Select an image to run detection")
        left.addWidget(self.surface, stretch=1)
        self.banner = Banner()
        left.addWidget(self.banner)
        root.addLayout(left, stretch=4)

        right = QVBoxLayout()
        right.setSpacing(12)
        self.badge = StatusBadge()
        right.addWidget(self.badge)
        self.summary = InfoTable("Detection summary")
        right.addWidget(self.summary)
        self.counts = Card("Objects by class")
        self.count_table = ClassCountTable()
        self.counts.add(self.count_table)
        right.addWidget(self.counts)
        root.addLayout(right, stretch=1)

        self.btn_open.clicked.connect(self.choose_image)
        self.btn_evidence.clicked.connect(self.save_evidence)

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", str(config.PROJECT_ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not path:
            return
        self.image_path = Path(path)
        self.path_label.setText(self.image_path.name)
        self.banner.clear_message()
        self.context.analyze_image(self.image_path)
        self.btn_evidence.setEnabled(True)

    def save_evidence(self) -> None:
        path = self.context.capture_evidence()
        if path:
            self.banner.show_message(f"Evidence saved: {Path(path).name}", "ok")

    def refresh(self) -> None:
        self.badge.set_status("INFO", "waiting for an image")

    def on_frame(self, frame, result: SceneResult, fps: float) -> None:
        if self.context.mode != "image":
            return
        self.surface.set_frame(frame)
        self.badge.set_status(result.status)
        self.count_table.set_counts(result.counts)
        detected = ", ".join(f"{name} x{n}" for name, n in result.counts.items()
                             if n) or "none"
        self.summary.set_items({
            "Status": result.status,
            "Detected objects": result.total,
            "Classes": detected,
            "Average confidence": f"{result.average_confidence * 100:.1f}%",
            "Inference": f"{result.inference_ms:.1f} ms",
            "Weak hits (review)": len(result.weak_detections),
        })

    def on_error(self, message: str) -> None:
        self.banner.show_message(message, "error")

    def on_finished(self, summary: dict) -> None:
        output = summary.get("output")
        if output:
            self.banner.show_message(
                f"Annotated image saved: {Path(output).name}", "ok")


class VideoPage(QWidget):
    """Video file detection with pause/stop and optional processed output."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.video_path: Path | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        bar = QHBoxLayout()
        self.btn_open = QPushButton("Select Video")
        self.btn_open.setObjectName("Primary")
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("Primary")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setObjectName("Secondary")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("Danger")
        for button in (self.btn_start, self.btn_pause, self.btn_stop):
            button.setEnabled(False)
        self.check_save = QCheckBox("Save processed video")
        self.check_save.setChecked(config.VIDEO_SAVE_ENABLED)
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(("0.5x", "1x (real time)", "2x", "Max"))
        self.combo_speed.setCurrentIndex(1)
        bar.addWidget(self.btn_open)
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.check_save)
        bar.addStretch(1)
        bar.addWidget(QLabel("Playback"))
        bar.addWidget(self.combo_speed)
        root.addLayout(bar)

        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(10)
        self.surface = VideoSurface("No video selected\nSelect a video file to begin AI analysis")
        left.addWidget(self.surface, stretch=1)
        self.progress_label = QLabel("0 / 0 frames")
        self.progress_label.setStyleSheet(f"color: {MUTED}; font-family: {MONO_FAMILY}; font-size: 11px;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left.addWidget(self.progress)
        left.addWidget(self.progress_label)
        self.strip = KeyValueStrip()
        for field in ("fps", "objects", "status", "inference"):
            self.strip.add_field(field)
        self.strip.finish()
        left.addWidget(self.strip)
        self.banner = Banner()
        left.addWidget(self.banner)
        body.addLayout(left, stretch=4)

        right = QVBoxLayout()
        right.setSpacing(12)
        self.badge = StatusBadge()
        right.addWidget(self.badge)
        self.counts = Card("Objects by class")
        self.count_table = ClassCountTable()
        self.counts.add(self.count_table)
        right.addWidget(self.counts)
        self.summary = InfoTable("Video summary")
        right.addWidget(self.summary)
        body.addLayout(right, stretch=1)
        root.addLayout(body, stretch=1)

        self.btn_open.clicked.connect(self.choose_video)
        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop)

    # -- actions ------------------------------------------------------------

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video", str(config.PROJECT_ROOT),
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.mpg *.mpeg)")
        if not path:
            return
        self.video_path = Path(path)
        self.banner.show_message(f"Selected: {self.video_path.name}", "info")
        self.btn_start.setEnabled(True)

    def start(self) -> None:
        if not self.video_path:
            return
        try:
            self.context.start_video(self.video_path,
                                     self.check_save.isChecked())
        except RuntimeError as exc:
            self.banner.show_message(str(exc), "error")
            return
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

    def toggle_pause(self) -> None:
        paused = self.context.toggle_pause()
        self.btn_pause.setText("Resume" if paused else "Pause")

    def stop(self) -> None:
        self.context.stop_source()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("Pause")

    # -- updates ------------------------------------------------------------

    def refresh(self) -> None:
        pass

    def on_frame(self, frame, result: SceneResult, fps: float) -> None:
        if self.context.mode != "video":
            return
        self.surface.set_frame(frame)
        self.badge.set_status(result.status)
        self.count_table.set_counts(result.counts)
        self.strip.set_value("fps", f"{fps:.1f}")
        self.strip.set_value("objects", result.total)
        self.strip.set_value("status", result.status,
                             STATUS_COLOR.get(result.status))
        self.strip.set_value("inference", f"{result.inference_ms:.0f} ms")

    def on_progress(self, current: int, total: int) -> None:
        if total:
            self.progress.setValue(int(current / total * 100))
        self.progress_label.setText(f"{current} / {total} frames")

    def on_finished(self, summary: dict) -> None:
        if summary.get("kind") != "video":
            return
        self.stop()
        self.summary.set_items({
            "Frames processed": summary.get("frames", 0),
            "Total detections": summary.get("detections", 0),
            "Dirty frames": summary.get("dirty_frames", 0),
            "Review frames": summary.get("review_frames", 0),
            "Peak objects": summary.get("peak_objects", 0),
            "Avg confidence": f"{summary.get('average_confidence', 0) * 100:.1f}%",
            "Duration": f"{summary.get('seconds', 0):.1f} s",
            "Overall status": summary.get("status", "-"),
            "Processed video": Path(summary["output"]).name
            if summary.get("output") else "not saved",
        })
        if summary.get("output"):
            self.banner.show_message(
                f"Processed video saved: {Path(summary['output']).name}", "ok")

    def on_error(self, message: str) -> None:
        self.banner.show_message(message, "error")
        self.stop()
