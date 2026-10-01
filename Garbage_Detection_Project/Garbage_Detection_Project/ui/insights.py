"""Insight pages: history browser, analytics, settings and about."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import QDate, Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QFont, QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDateEdit,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

import config
from src import analytics
from src.history import Filters
from src.settings import system_info
from ui.charts import BarChart, DonutChart, LineChart
from ui.components import Banner, Card, InfoTable, MetricCard
from ui.styles import (ACCENT, BORDER, DIM, ELEVATED, GREEN, MONO_FAMILY,
                       MUTED, RED, STATUS_COLOR, TEXT, YELLOW)


def _mono(size: int = 10) -> QFont:
    f = QFont(MONO_FAMILY.split('"')[1] if '"' in MONO_FAMILY else MONO_FAMILY)
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(size)
    return f


SOURCE_TYPES = ("all", "camera", "image", "video", "manual")
STATUSES = ("all", config.STATUS_CLEAN, config.STATUS_DIRTY,
            config.STATUS_REVIEW)


class HistoryPage(QWidget):
    """Searchable, filterable detection history with CSV export."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("Search source, status or class...")
        self.combo_source = QComboBox()
        self.combo_source.addItems(SOURCE_TYPES)
        self.combo_status = QComboBox()
        self.combo_status.addItems(STATUSES)
        self.date_from = QDateEdit(QDate.currentDate().addDays(-30))
        self.date_to = QDateEdit(QDate.currentDate())
        for editor in (self.date_from, self.date_to):
            editor.setCalendarPopup(True)
            editor.setDisplayFormat("yyyy-MM-dd")
        self.btn_refresh = QPushButton("Refresh")
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.setObjectName("Primary")
        filters.addWidget(self.edit_search, stretch=2)
        filters.addWidget(QLabel("Source"))
        filters.addWidget(self.combo_source)
        filters.addWidget(QLabel("Status"))
        filters.addWidget(self.combo_status)
        filters.addWidget(QLabel("From"))
        filters.addWidget(self.date_from)
        filters.addWidget(QLabel("To"))
        filters.addWidget(self.date_to)
        filters.addWidget(self.btn_refresh)
        filters.addWidget(self.btn_export)
        root.addLayout(filters)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ("Timestamp", "Source", "Type", "Status", "Objects",
             "Avg conf", "Unique", "Evidence"))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)
        root.addWidget(self.table, stretch=1)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.banner = Banner()
        root.addWidget(self.banner)
        root.addWidget(self.count_label)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_export.clicked.connect(self.export_csv)
        self.edit_search.returnPressed.connect(self.refresh)
        self.table.cellDoubleClicked.connect(self.open_evidence)

    def _filters(self) -> Filters:
        source = self.combo_source.currentText()
        status = self.combo_status.currentText()
        return Filters(
            text=self.edit_search.text().strip(),
            source_type="" if source == "all" else source,
            status="" if status == "all" else status,
            date_from=self.date_from.date().toPython(),
            date_to=self.date_to.date().toPython(),
            limit=1000)

    def refresh(self) -> None:
        history = self.context.engine.history
        events = history.query(self._filters())
        self.table.setRowCount(len(events))
        self._events = events
        for row, event in enumerate(events):
            values = (event.timestamp.replace("T", " "), event.source,
                      event.source_type, event.status, str(event.total_count),
                      f"{event.average_confidence * 100:.0f}%",
                      "-" if event.unique_objects is None
                      else str(event.unique_objects),
                      "open" if event.evidence_path else "-")
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setFont(_mono(9))
                if column == 3:
                    color = STATUS_COLOR.get(event.status, STATUS_COLOR["INFO"])
                    item.setForeground(QBrush(QColor(color)))
                    item.setFont(_mono(9))
                if column in (4, 5, 6):
                    item.setFont(_mono(9))
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 7 and event.evidence_path:
                    item.setForeground(QBrush(QColor(ACCENT)))
                self.table.setItem(row, column, item)
        total = history.count(self._filters())
        self.count_label.setText(
            f"{len(events)} of {total} matching events shown "
            f"(database total: {history.event_count()})")
        self.banner.clear_message()

    def export_csv(self) -> None:
        default = config.DATA_DIR / f"history_export_{date.today().isoformat()}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export history", str(default), "CSV files (*.csv)")
        if not path:
            return
        try:
            written = self.context.engine.history.export_csv(path,
                                                             self._filters())
        except OSError as exc:
            self.banner.show_message(f"Export failed: {exc}", "error")
            return
        self.banner.show_message(f"Exported to {Path(written).name}", "ok")

    def open_evidence(self, row: int, column: int) -> None:
        if column != 7 or row >= len(getattr(self, "_events", [])):
            return
        event = self._events[row]
        if not event.evidence_path:
            return
        path = Path(event.evidence_path)
        if not path.is_file():
            self.banner.show_message(
                f"Evidence file is missing: {path.name}", "warn")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class AnalyticsPage(QWidget):
    """Charts built only from stored history and evaluation reports."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(14)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.card_events = MetricCard("Detection events", "0")
        self.card_objects = MetricCard("Garbage objects", "0", RED)
        self.card_dirty = MetricCard("Dirty ratio", "-", YELLOW)
        self.card_conf = MetricCard("Avg confidence", "-", GREEN)
        for card in (self.card_events, self.card_objects, self.card_dirty,
                     self.card_conf):
            metrics.addWidget(card)
        root.addLayout(metrics)

        row1 = QHBoxLayout()
        row1.setSpacing(12)
        self.class_chart = BarChart("Garbage detected by class")
        self.class_chart.empty_text = "No detections recorded yet"
        self.status_chart = DonutChart("Clean vs dirty vs review")
        self.status_chart.empty_text = "No events recorded yet"
        for widget in (self.class_chart, self.status_chart):
            card = Card("")
            card.add(widget)
            row1.addWidget(card, stretch=1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(12)
        self.daily_chart = LineChart("Detection activity (last 14 days)")
        self.daily_chart.empty_text = "No activity recorded yet"
        self.hourly_chart = BarChart("Activity by hour (last 7 days)")
        self.hourly_chart.empty_text = "No activity recorded yet"
        for widget in (self.daily_chart, self.hourly_chart):
            card = Card("")
            card.add(widget)
            row2.addWidget(card, stretch=1)
        root.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(12)
        self.model_card = InfoTable("Model performance (held-out test set)")
        self.dataset_card = InfoTable("Dataset")
        row3.addWidget(self.model_card, stretch=1)
        row3.addWidget(self.dataset_card, stretch=1)
        root.addLayout(row3)

        self.note = QLabel(
            "Metrics shown here come from evaluate_model.py on the held-out "
            "test split and from the application's own history database. "
            "Nothing is estimated or fabricated.")
        self.note.setStyleSheet(f"color: {DIM}; font-size: 11px;")
        self.note.setWordWrap(True)
        root.addWidget(self.note)

        scroll.setWidget(container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def refresh(self) -> None:
        history = self.context.engine.history
        summary = analytics.overview(history)
        self.card_events.set_value(summary["total_events"])
        self.card_objects.set_value(summary["total_objects"])
        self.card_dirty.set_value(f"{summary['dirty_ratio'] * 100:.0f}%")
        self.card_conf.set_value(
            f"{summary['average_confidence'] * 100:.1f}%"
            if summary["total_events"] else "-")

        distribution = analytics.class_distribution(history)
        self.class_chart.set_data(distribution["labels"],
                                  distribution["values"])
        split = analytics.status_split(history)
        self.status_chart.set_data(
            split["labels"], split["values"],
            [QColor(STATUS_COLOR[config.STATUS_CLEAN]),
             QColor(STATUS_COLOR[config.STATUS_DIRTY]),
             QColor(STATUS_COLOR[config.STATUS_REVIEW])])
        daily = analytics.daily_activity(history)
        self.daily_chart.set_data(daily["labels"], daily["series"])
        hourly = analytics.hourly_activity(history)
        self.hourly_chart.set_data(hourly["labels"], hourly["values"])

        metrics = analytics.model_metrics()
        if metrics.get("available"):
            self.model_card.set_items({
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "mAP@50": f"{metrics['map50']:.4f}",
                "mAP@50:95": f"{metrics['map50_95']:.4f}",
                "Inference": f"{metrics['inference_ms']:.1f} ms",
                "FPS": f"{metrics['fps']:.1f}",
                "Model size": f"{metrics['model_size_mb']:.1f} MB",
                "Weights": Path(metrics["weights"]).name,
                "Evaluated": metrics["generated_at"][:19].replace("T", " "),
            })
        else:
            self.model_card.set_items({
                "Status": "not evaluated yet",
                "How": "run: python evaluate_model.py",
            })

        dataset = analytics.dataset_summary()
        if dataset.get("available"):
            splits = dataset.get("splits") or {}
            imbalance = dataset.get("imbalance") or {}
            self.dataset_card.set_items({
                "Images": dataset.get("total_images"),
                "Boxes": dataset.get("total_boxes"),
                "Train / Val / Test": f"{splits.get('train', 0)} / "
                                      f"{splits.get('val', 0)} / "
                                      f"{splits.get('test', 0)}",
                "Quarantined": dataset.get("quarantined_images"),
                "Audit hard errors": dataset.get("hard_errors"),
                "Imbalance max/min": imbalance.get("max_min_ratio"),
            })
        else:
            self.dataset_card.set_items({"Status": "no dataset report found"})


class SettingsPage(QWidget):
    """Runtime configuration; values persist to data/settings.json."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(14)

        detection = Card("Detection")
        grid = _form()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.label_conf = QLabel()
        self.label_conf.setFont(_mono(10))
        self.slider_conf.valueChanged.connect(
            lambda v: self.label_conf.setText(f"{v / 100:.2f}"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.10, 0.90)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setDecimals(2)
        self.combo_size = QComboBox()
        self.combo_size.addItems(("320", "416", "512", "640", "768", "1024"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(("auto", "0", "cpu"))
        self.combo_weights = QComboBox()
        self.combo_weights.setEditable(True)
        self.combo_weights.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.btn_weights = QPushButton("Browse...")
        self.btn_weights.clicked.connect(self.choose_weights)
        self.check_tracking = QCheckBox("Enable ByteTrack (unique object ids)")
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.combo_weights, stretch=1)
        weights_row.addWidget(self.btn_weights)
        _add_rows(grid, [
            ("Confidence threshold", self.slider_conf, self.label_conf),
            ("IoU threshold", self.spin_iou),
            ("Inference image size", self.combo_size),
            ("Device", self.combo_device),
            ("Model weights", weights_row),
            ("Object tracking", self.check_tracking),
        ])
        detection.add_layout(grid)
        root.addWidget(detection)

        capture = Card("Capture & Storage")
        grid2 = _form()
        self.spin_camera = QSpinBox()
        self.spin_camera.setRange(0, 9)
        self.check_evidence = QCheckBox("Save evidence automatically")
        self.check_save_video = QCheckBox("Offer to save processed video")
        self.spin_history = QSpinBox()
        self.spin_history.setRange(100, 100_000)
        self.spin_history.setSingleStep(500)
        _add_rows(grid2, [
            ("Camera index", self.spin_camera),
            ("Evidence", self.check_evidence),
            ("Processed video", self.check_save_video),
            ("History rows to keep", self.spin_history),
        ])
        capture.add_layout(grid2)
        root.addWidget(capture)

        alerts = Card("Alerts")
        grid3 = _form()
        self.check_alert = QCheckBox("Audible alert when garbage is detected")
        self.spin_alert_count = QSpinBox()
        self.spin_alert_count.setRange(1, 50)
        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(1.0, 3600.0)
        self.spin_cooldown.setSuffix(" s")
        self.btn_sound = QPushButton("Choose .wav (optional)")
        self.btn_sound.clicked.connect(self.choose_sound)
        self.label_sound = QLabel("system beep")
        self.label_sound.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        sound_row = QHBoxLayout()
        sound_row.addWidget(self.btn_sound)
        sound_row.addWidget(self.label_sound, stretch=1)
        _add_rows(grid3, [
            ("Alerts", self.check_alert),
            ("Alert threshold (objects)", self.spin_alert_count),
            ("Alert cooldown", self.spin_cooldown),
            ("Alert sound", sound_row),
        ])
        alerts.add_layout(grid3)
        root.addWidget(alerts)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Apply and Save")
        self.btn_apply.setObjectName("Primary")
        self.btn_defaults = QPushButton("Restore Defaults")
        self.btn_defaults.setObjectName("Secondary")
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_defaults)
        buttons.addStretch(1)
        root.addLayout(buttons)
        self.banner = Banner()
        root.addWidget(self.banner)

        self.system = InfoTable("System information")
        root.addWidget(self.system)

        scroll.setWidget(container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self.btn_apply.clicked.connect(self.apply_settings)
        self.btn_defaults.clicked.connect(self.restore_defaults)

    # -- helpers ------------------------------------------------------------

    def choose_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model weights", str(config.PROJECT_ROOT),
            "PyTorch weights (*.pt)")
        if path:
            self.combo_weights.setCurrentText(path)

    def choose_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select alert sound", str(config.PROJECT_ROOT),
            "Wave audio (*.wav)")
        if path:
            self.label_sound.setText(Path(path).name)
            self.label_sound.setProperty("path", path)

    def refresh(self) -> None:
        settings = self.context.settings
        self.slider_conf.setValue(int(settings.confidence * 100))
        self.label_conf.setText(f"{settings.confidence:.2f}")
        self.spin_iou.setValue(settings.iou)
        self.combo_size.setCurrentText(str(settings.image_size))
        self.combo_device.setCurrentText(settings.device)
        self.combo_weights.clear()
        for weights in config.trained_weights():
            self.combo_weights.addItem(str(weights))
        self.combo_weights.addItem(str(config.BASE_WEIGHTS))
        self.combo_weights.setCurrentText(settings.weights_path)
        self.check_tracking.setChecked(settings.tracking_enabled)
        self.spin_camera.setValue(settings.camera_index)
        self.check_evidence.setChecked(settings.evidence_enabled)
        self.check_save_video.setChecked(settings.save_processed_video)
        self.check_alert.setChecked(settings.alert_enabled)
        self.spin_alert_count.setValue(settings.alert_threshold)
        self.spin_cooldown.setValue(settings.alert_cooldown)
        self.label_sound.setText(Path(settings.alert_sound).name
                                 if settings.alert_sound else "system beep")
        self.label_sound.setProperty("path", settings.alert_sound)
        self.spin_history.setValue(settings.keep_history_entries)
        self.system.set_items(system_info())

    def apply_settings(self) -> None:
        from src.settings import Settings, save_settings

        settings = Settings(
            confidence=self.slider_conf.value() / 100,
            iou=self.spin_iou.value(),
            image_size=int(self.combo_size.currentText()),
            dirty_threshold=self.context.settings.dirty_threshold,
            camera_index=self.spin_camera.value(),
            device=self.combo_device.currentText(),
            weights=self.combo_weights.currentText(),
            tracking_enabled=self.check_tracking.isChecked(),
            evidence_enabled=self.check_evidence.isChecked(),
            alert_enabled=self.check_alert.isChecked(),
            alert_threshold=self.spin_alert_count.value(),
            alert_cooldown=self.spin_cooldown.value(),
            alert_sound=self.label_sound.property("path") or "",
            save_processed_video=self.check_save_video.isChecked(),
            keep_history_entries=self.spin_history.value())
        try:
            rebuilt = self.context.apply_settings(settings)
        except Exception as exc:
            self.banner.show_message(f"Settings not applied: {exc}", "error")
            return
        saved = save_settings(settings)
        self.banner.show_message(
            ("Model reloaded with the new weights." if rebuilt else
             "Settings applied.") + ("" if saved else
                                    " (could not persist to settings.json)"),
            "ok" if saved else "warn")

    def restore_defaults(self) -> None:
        from src.settings import Settings
        self.context.apply_settings(Settings())
        self.refresh()
        self.banner.show_message("Defaults restored (not saved yet)", "info")


class AboutPage(QWidget):
    """Project facts for reviewers: scope, data, model, honesty statement."""

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(14)

        title = QLabel("AI-Based Intelligent Garbage Detection and "
                       "Monitoring System")
        title.setObjectName("Title")
        subtitle = QLabel("Final year project - YOLOv8 object detection, "
                          "PyTorch, OpenCV, PySide6")
        subtitle.setObjectName("Subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        objective = Card("Objective")
        text = QLabel(
            "Detect ten classes of litter in live webcam feeds, images and "
            "video files; report a derived CLEAN / DIRTY / REVIEW scene "
            "status; count objects per class; capture timestamped evidence; "
            "and keep a searchable history with analytics. 'Clean' is never "
            "an object class - it is the absence of confirmed garbage.")
        text.setWordWrap(True)
        text.setStyleSheet(f"font-size: 12px; line-height: 1.5;")
        objective.add(text)
        root.addWidget(objective)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.classes = InfoTable("Detection classes")
        self.pipeline = InfoTable("Data pipeline")
        row.addWidget(self.classes, stretch=1)
        row.addWidget(self.pipeline, stretch=1)
        root.addLayout(row)

        self.integrity = Card("Data and metric integrity")
        statement = QLabel(
            "Labels were produced by a documented hybrid pipeline "
            "(YOLO-World prompts, then saliency/GrabCut fallbacks) and every "
            "box was re-checked by scripts/audit_annotations.py, "
            "scripts/validate_dataset.py and scripts/visual_audit.py. Images "
            "without a verifiable box are quarantined instead of being "
            "trained as clean. The earlier single-class model trained on "
            "COCO-derived labels (mAP50 0.629) is rejected at load time and "
            "is not reported as project accuracy. Reported metrics come only "
            "from evaluate_model.py on the held-out test split.")
        statement.setWordWrap(True)
        statement.setStyleSheet(f"font-size: 12px; line-height: 1.5;")
        self.integrity.add(statement)
        root.addWidget(self.integrity)

        self.stack = InfoTable("Technology stack")
        root.addWidget(self.stack)

        scroll.setWidget(container)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def refresh(self) -> None:
        info = system_info()
        self.classes.set_items({
            f"{index}": name
            for index, name in enumerate(config.GARBAGE_CLASSES)})
        dataset = analytics.dataset_summary()
        splits = dataset.get("splits") or {}
        self.pipeline.set_items({
            "Images (train/val/test)": f"{splits.get('train', '-')} / "
                                       f"{splits.get('val', '-')} / "
                                       f"{splits.get('test', '-')}",
            "Annotation boxes": dataset.get("total_boxes", "-"),
            "Quarantined (no box)": dataset.get("quarantined_images", "-"),
            "Audit hard errors": dataset.get("hard_errors", "-"),
            "Provenance": ", ".join(f"{k}: {v}" for k, v in
                                    (dataset.get("provenance") or {}).items())
            or "-",
        })
        self.stack.set_items({
            "Python": info["python"],
            "PyTorch": f"{info['pytorch']} (CUDA {info['cuda']})",
            "Ultralytics YOLO": info["ultralytics"],
            "OpenCV": info["opencv"],
            "Interface": "PySide6 (Qt for Python)",
            "Storage": "SQLite (history) + JSON/CSV reports",
            "GPU": f"{info['gpu']} ({info['vram']})"
                   if info["cuda_available"] == "True" else "CPU only",
            "Model": f"{info['model']} {info['model_version']}",
            "Weights": Path(info["weights"]).name,
        })


def _form() -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(10)
    form.setLabelAlignment(Qt.AlignLeft)
    return form


def _add_rows(form, rows) -> None:
    for row in rows:
        label, widget = row[0], row[1]
        extra = row[2] if len(row) > 2 else None
        if extra is not None:
            box = QHBoxLayout()
            box.addWidget(widget, stretch=1)
            box.addWidget(extra)
            form.addRow(label, box)
        else:
            form.addRow(label, widget)
