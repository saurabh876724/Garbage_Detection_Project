"""Main window: sidebar navigation, header telemetry and page stack."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QStackedWidget, QVBoxLayout,
                               QWidget)

import config
from src.detector import ModelError, SceneResult
from src.logging_setup import get_logger
from src.settings import Settings, load_settings, save_settings
from ui import insights, pages
from ui.components import StatusBadge
from ui.engine import DetectionEngine, DetectionWorker
from ui.styles import (ACCENT, BORDER, DIM, ELEVATED, GREEN, MONO_FAMILY,
                       MUTED, PANEL, RED, TEXT, DARK_QSS, STATUS_COLOR,
                       icon_svg)

log = get_logger("ui")

NAV_ITEMS = (
    ("Dashboard", "dashboard"),
    ("Live Camera", "live"),
    ("Image Detection", "image"),
    ("Video Detection", "video"),
    ("History", "history"),
    ("Analytics", "analytics"),
    ("Settings", "settings"),
    ("About", "about"),
)


class AppContext:
    """Shared services plus the small API every page uses."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = DetectionEngine(settings)
        self.worker = DetectionWorker(self.engine)
        self.last_result: SceneResult | None = None
        self.last_frame = None

    @property
    def mode(self) -> str | None:
        return self.worker.mode

    def start_camera(self, index: int) -> None:
        self.worker.start_camera(index)

    def start_video(self, path: Path, save_output: bool = False) -> None:
        self.worker.start_video(path, save_output)

    def analyze_image(self, path: Path) -> None:
        self.worker.analyze_image(path)

    def stop_source(self) -> None:
        self.worker.stop()

    def toggle_pause(self) -> bool:
        if self.worker.is_paused:
            self.worker.resume()
            return False
        self.worker.pause()
        return True

    def capture_evidence(self) -> str | None:
        return self.worker.capture_evidence_now()

    def set_confidence(self, value: float) -> None:
        self.settings.confidence = value
        self.engine.detector.conf_threshold = value

    def apply_settings(self, settings: Settings) -> bool:
        """Push new settings into the engine; True when the model reloaded."""
        rebuilt = self.engine.apply_settings(settings)
        self.settings = settings
        return rebuilt

    def shutdown(self) -> None:
        self.worker.shutdown()
        self.engine.shutdown()


def _svg_icon(name: str, color: str = MUTED, size: int = 20) -> QIcon:
    """Create a QIcon from an SVG path in the design system."""
    svg_data = icon_svg(name, color, size)
    if not svg_data.startswith("url("):
        return QIcon()
    data_uri = svg_data[4:-1]
    raw = base64.b64decode(data_uri.split(",")[1])
    pix = QPixmap()
    pix.loadFromData(raw)
    return QIcon(pix)


class MainWindow(QMainWindow):
    def __init__(self, context: AppContext):
        super().__init__()
        self.context = context
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(*config.WINDOW_SIZE)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        right.addWidget(self._build_header())
        self.stack = QStackedWidget()
        self.pages = self._build_pages()
        for page in self.pages.values():
            self.stack.addWidget(page)
        right.addWidget(self.stack, stretch=1)
        layout.addLayout(right, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._connect_worker()
        self.nav.setCurrentRow(1)
        self._on_nav_changed(1)

    # -- construction -------------------------------------------------------

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 16, 0, 12)
        layout.setSpacing(4)

        brand_container = QWidget()
        brand_layout = QVBoxLayout(brand_container)
        brand_layout.setContentsMargins(16, 4, 16, 8)
        brand_layout.setSpacing(2)

        brand = QLabel("AI GARBAGE\nDETECTION")
        brand.setStyleSheet(
            f"color: {ACCENT}; font-size: 14px; font-weight: 700;"
            f"letter-spacing: 0.5px; line-height: 1.3;")
        tagline = QLabel("Intelligent waste monitoring")
        tagline.setObjectName("Subtitle")
        tagline.setWordWrap(True)
        brand_layout.addWidget(brand)
        brand_layout.addWidget(tagline)
        layout.addWidget(brand_container)

        separator = QFrame()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background: {BORDER}; margin: 4px 12px;")
        layout.addWidget(separator)

        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        self.nav.setIconSize(QSize(18, 18))

        NAV_GROUPS = [
            ("OVERVIEW", [("Dashboard", "dashboard")]),
            ("DETECTION", [("Live Camera", "live"),
                           ("Image Detection", "image"),
                           ("Video Detection", "video")]),
            ("INSIGHTS", [("History", "history"),
                          ("Analytics", "analytics")]),
            ("SYSTEM", [("Settings", "settings"),
                        ("About", "about")]),
        ]
        for gi, (group_label, items) in enumerate(NAV_GROUPS):
            if gi > 0:
                spacer = QListWidgetItem("", self.nav)
                spacer.setFlags(Qt.NoItemFlags)
                spacer.setSizeHint(QSize(0, 8))
            header_item = QListWidgetItem(f" {group_label}", self.nav)
            header_item.setFlags(Qt.NoItemFlags)
            header_item.setSizeHint(QSize(0, 22))
            header_item.setForeground(QColor(DIM))
            hdr_font = QFont()
            hdr_font.setPointSize(8)
            hdr_font.setWeight(QFont.DemiBold)
            header_item.setFont(hdr_font)
            for label_text, key in items:
                item = QListWidgetItem(f"  {label_text}", self.nav)
                item.setIcon(_svg_icon(key, MUTED, 18))
                item.setSizeHint(QSize(0, 34))
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        layout.addWidget(self.nav, stretch=1)

        footer = QLabel(f"{config.MODEL_NAME}\n{config.MODEL_VERSION}")
        footer.setStyleSheet(f"color: {DIM}; padding: 8px 16px; font-size: 10px;"
                             f"font-family: {MONO_FAMILY};")
        layout.addWidget(footer)
        return sidebar

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(48)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(0)

        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("Title")
        layout.addWidget(self.page_title)
        layout.addStretch(1)

        self.badge = StatusBadge()
        layout.addWidget(self.badge)
        layout.addSpacing(16)

        info = self.context.engine.detector.info()
        self._status_segments: list[tuple[QLabel, QLabel]] = []
        segments = [
            ("ENGINE", "Ready", GREEN),
            ("DEVICE", f"{info['device']}", GREEN if info.get("cuda_available") != "False" else RED),
            ("GPU", info["gpu"], MUTED),
            ("MODEL", f"{info['model']} {info['version']}", ACCENT),
        ]
        for i, (label_text, value_text, dot_color) in enumerate(segments):
            if i > 0:
                connector = QLabel("─")
                connector.setStyleSheet(f"color: {BORDER}; font-size: 10px;"
                                        f"margin: 0 6px;")
                layout.addWidget(connector)
            seg = QWidget()
            seg_layout = QHBoxLayout(seg)
            seg_layout.setContentsMargins(0, 0, 0, 0)
            seg_layout.setSpacing(5)
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {dot_color}; font-size: 8px;")
            lbl = QLabel(f"{label_text}: {value_text}")
            lbl.setFont(QFont(
                MONO_FAMILY.split('"')[1] if '"' in MONO_FAMILY else MONO_FAMILY,
                9))
            lbl.setStyleSheet(f"color: {MUTED};")
            seg_layout.addWidget(dot)
            seg_layout.addWidget(lbl)
            layout.addWidget(seg)
            self._status_segments.append((dot, lbl))

        return header

    def _build_pages(self) -> dict[str, QWidget]:
        return {
            "dashboard": pages.DashboardPage(self.context),
            "live": pages.LiveCameraPage(self.context),
            "image": pages.ImagePage(self.context),
            "video": pages.VideoPage(self.context),
            "history": insights.HistoryPage(self.context),
            "analytics": insights.AnalyticsPage(self.context),
            "settings": insights.SettingsPage(self.context),
            "about": insights.AboutPage(self.context),
        }

    def _connect_worker(self) -> None:
        worker = self.context.worker
        worker.frame_ready.connect(self._on_frame)
        worker.source_finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.progress.connect(self._on_progress)
        worker.evidence_saved.connect(self._on_evidence)
        worker.source_started.connect(self._on_started)

    # -- navigation ---------------------------------------------------------

    def _on_nav_changed(self, row: int) -> None:
        item = self.nav.item(row)
        if item is None or not (item.flags() & Qt.ItemIsSelectable):
            return
        nav_index = -1
        for i in range(self.nav.count()):
            it = self.nav.item(i)
            if it is not None and (it.flags() & Qt.ItemIsSelectable):
                nav_index += 1
            if i == row:
                break
        if not 0 <= nav_index < len(NAV_ITEMS):
            return
        for i in range(self.nav.count()):
            it = self.nav.item(i)
            if it is None or not (it.flags() & Qt.ItemIsSelectable):
                continue
            idx = -1
            for j in range(i + 1):
                jt = self.nav.item(j)
                if jt is not None and (jt.flags() & Qt.ItemIsSelectable):
                    idx += 1
            if idx < len(NAV_ITEMS):
                color = ACCENT if idx == nav_index else MUTED
                it.setIcon(_svg_icon(NAV_ITEMS[idx][1], color, 18))

        label, key = NAV_ITEMS[nav_index]
        self.page_title.setText(label)
        self.stack.setCurrentWidget(self.pages[key])
        page = self.pages[key]
        if hasattr(page, "refresh"):
            try:
                page.refresh()
            except Exception as exc:
                log.exception("page refresh failed")
                self.statusBar().showMessage(f"Could not refresh {label}: {exc}")

    # -- worker signals -----------------------------------------------------

    def _on_frame(self, frame, result: SceneResult, fps: float) -> None:
        self.context.last_frame = frame
        self.context.last_result = result
        self.badge.set_status(result.status,
                              f"{result.total} object(s) detected")
        self.pages["dashboard"].on_frame(frame, result, fps)
        mode = self.context.mode
        if mode == "camera":
            self.pages["live"].on_frame(frame, result, fps)
        elif mode == "video":
            self.pages["video"].on_frame(frame, result, fps)
        elif mode == "image":
            self.pages["image"].on_frame(frame, result, fps)

    def _on_started(self, label: str, meta: dict) -> None:
        self.statusBar().showMessage(f"Source started: {label} ({meta})")

    def _on_finished(self, summary: dict) -> None:
        kind = summary.get("kind")
        if kind == "camera":
            self.pages["live"].on_finished(summary)
        elif kind == "video":
            self.pages["video"].on_finished(summary)
        elif kind == "image":
            self.pages["image"].on_finished(summary)
        self.statusBar().showMessage(
            f"Session finished: {summary.get('status', '-')} - "
            f"{summary.get('detections', summary.get('objects', 0))} "
            "detection(s)")
        self.pages["dashboard"].refresh()

    def _on_error(self, message: str) -> None:
        log.error("worker error: %s", message)
        self.statusBar().showMessage("Error - see logs/app.log")
        for key in ("live", "image", "video"):
            page = self.pages[key]
            if hasattr(page, "on_error"):
                page.on_error(message)
        QMessageBox.warning(self, "Detection problem", message)

    def _on_progress(self, current: int, total: int) -> None:
        self.pages["video"].on_progress(current, total)

    def _on_evidence(self, path: str, status: str) -> None:
        self.statusBar().showMessage(f"Evidence saved ({status}): "
                                     f"{Path(path).name}")

    # -- lifecycle ----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        save_settings(self.context.settings)
        self.context.shutdown()
        log.info("application closed")
        super().closeEvent(event)


def launch() -> int:
    """Create the Qt application and run the dashboard."""
    config.ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("AI Garbage Detection")
    app.setStyleSheet(DARK_QSS)

    settings = load_settings()
    try:
        context = AppContext(settings)
    except (ModelError, RuntimeError) as exc:
        log.error("engine start failed: %s", exc)
        QMessageBox.critical(
            None, "AI Garbage Detection",
            f"The detection engine could not start:\n\n{exc}\n\n"
            "Train the model first with:  python train_yolo.py\n"
            "Then evaluate it with:      python evaluate_model.py")
        return 1

    window = MainWindow(context)
    window.show()
    log.info("dashboard launched")
    return app.exec()


def main() -> None:  # pragma: no cover - convenience entry point
    raise SystemExit(launch())


if __name__ == "__main__":
    main()
