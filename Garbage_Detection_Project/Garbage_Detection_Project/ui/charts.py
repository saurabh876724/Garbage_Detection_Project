"""Lightweight QPainter charts.

PySide6 ships without QtCharts in this environment, so the three charts the
dashboard needs (bars, donut, lines) are drawn directly. They take plain
label/value data from src.analytics - nothing here invents numbers.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ui.styles import BORDER, CHART_SERIES, DIM, MONO_FAMILY, MUTED, TEXT


def _color(index: int) -> QColor:
    return QColor(CHART_SERIES[index % len(CHART_SERIES)])


def _mono(size: int = 8) -> QFont:
    f = QFont(MONO_FAMILY.split('"')[1] if '"' in MONO_FAMILY else MONO_FAMILY)
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(size)
    return f


class ChartBase(QWidget):
    """Common chrome: title, minimum size and an empty-state message."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.empty_text = "No data yet"
        self.setMinimumHeight(220)

    def has_data(self) -> bool:
        return False

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        top = 0
        if self.title:
            painter.setPen(QColor(TEXT))
            painter.setFont(QFont(painter.font().family(), 11, QFont.DemiBold))
            painter.drawText(QRectF(8, 6, self.width() - 16, 20),
                             Qt.AlignLeft | Qt.AlignVCenter, self.title)
            top = 30
        if not self.has_data():
            painter.setPen(QColor(DIM))
            painter.setFont(QFont(painter.font().family(), 10))
            painter.drawText(QRectF(8, top, self.width() - 16,
                                    self.height() - top - 8),
                             Qt.AlignCenter, self.empty_text)
            painter.end()
            return
        self.paint_chart(painter, QRectF(8, top, self.width() - 16,
                                         self.height() - top - 8))
        painter.end()

    def paint_chart(self, painter: QPainter, area: QRectF) -> None:
        """Implemented by subclasses."""


class BarChart(ChartBase):
    """Vertical bars, one per class, value printed above each bar."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.labels: list[str] = []
        self.values: list[float] = []
        self.suffix = ""

    def set_data(self, labels: list[str], values: list[float],
                 suffix: str = "") -> None:
        self.labels, self.values, self.suffix = list(labels), list(values), suffix
        self.update()

    def has_data(self) -> bool:
        return bool(self.values) and max(self.values) > 0

    def paint_chart(self, painter: QPainter, area: QRectF) -> None:
        peak = max(self.values)
        axis_h = 26
        plot = QRectF(area.left(), area.top(), area.width(),
                      area.height() - axis_h)
        step = plot.width() / max(1, len(self.values))
        bar_w = min(38.0, step * 0.62)

        painter.setFont(_mono(7))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot.bottom() - plot.height() * fraction
            painter.setPen(QPen(QColor(BORDER), 1, Qt.DotLine))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(DIM))
            painter.drawText(QRectF(plot.left() - 2, y - 7, step, 14),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             f"{int(peak * fraction)}")

        for index, value in enumerate(self.values):
            height = plot.height() * (value / peak) if peak else 0.0
            x = plot.left() + step * index + (step - bar_w) / 2
            rect = QRectF(x, plot.bottom() - height, bar_w, height)
            painter.setPen(Qt.NoPen)
            painter.setBrush(_color(index))
            painter.drawRoundedRect(rect, 2, 2)

            painter.setPen(QColor(TEXT))
            painter.setFont(_mono(8))
            painter.drawText(QRectF(x - 10, rect.top() - 17, bar_w + 20, 15),
                             Qt.AlignCenter, f"{value:g}{self.suffix}")

            painter.setPen(QColor(MUTED))
            painter.setFont(QFont(painter.font().family(), 8))
            label = self.labels[index] if index < len(self.labels) else ""
            painter.drawText(QRectF(x - 12, plot.bottom() + 5, bar_w + 24, 18),
                             Qt.AlignCenter, label[:11])


class DonutChart(ChartBase):
    """Share-of-total ring with a legend and the total in the middle."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.labels: list[str] = []
        self.values: list[float] = []
        self.colors: list[QColor] = []
        self.setMinimumHeight(200)

    def set_data(self, labels: list[str], values: list[float],
                 colors: list[QColor] | None = None) -> None:
        self.labels, self.values = list(labels), list(values)
        self.colors = list(colors) if colors else [
            _color(i) for i in range(len(values))]
        self.update()

    def has_data(self) -> bool:
        return sum(self.values) > 0

    def paint_chart(self, painter: QPainter, area: QRectF) -> None:
        total = sum(self.values)
        side = min(area.height(), area.width() * 0.55)
        ring = QRectF(area.left(), area.top() + (area.height() - side) / 2,
                      side, side)
        inset = ring.adjusted(side * 0.19, side * 0.19,
                              -side * 0.19, -side * 0.19)

        angle = 90 * 16
        for index, value in enumerate(self.values):
            if value <= 0:
                continue
            span = int(-360 * 16 * value / total)
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.colors[index % len(self.colors)])
            path = QPainterPath()
            path.arcMoveTo(ring, angle / 16)
            path.arcTo(ring, angle / 16, span / 16)
            path.arcTo(inset, (angle + span) / 16, -span / 16)
            path.closeSubpath()
            painter.drawPath(path)
            angle += span

        painter.setPen(QColor(TEXT))
        painter.setFont(_mono(16))
        painter.drawText(inset, Qt.AlignCenter, f"{int(total)}")
        painter.setPen(QColor(DIM))
        painter.setFont(QFont(painter.font().family(), 8))
        painter.drawText(QRectF(inset.left(), inset.center().y() + 14,
                                inset.width(), 16), Qt.AlignCenter, "events")

        legend_x = ring.right() + 18
        legend_y = ring.top() + max(0.0, (ring.height() - len(self.values) * 22) / 2)
        painter.setFont(QFont(painter.font().family(), 9))
        for index, label in enumerate(self.labels):
            value = self.values[index] if index < len(self.values) else 0
            y = legend_y + index * 22
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.colors[index % len(self.colors)])
            painter.drawRoundedRect(QRectF(legend_x, y + 3, 11, 11), 2, 2)
            share = f"{value / total * 100:.0f}%" if total else "0%"
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(legend_x + 18, y, area.right() - legend_x - 18, 18),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             f"{label}  {int(value)} ({share})")


class LineChart(ChartBase):
    """One or more time series sharing an x axis."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.labels: list[str] = []
        self.series: dict[str, list[float]] = {}

    def set_data(self, labels: list[str], series: dict[str, list[float]]) -> None:
        self.labels = list(labels)
        self.series = {k: list(v) for k, v in series.items()}
        self.update()

    def has_data(self) -> bool:
        return bool(self.labels) and any(
            max(values, default=0) > 0 for values in self.series.values())

    def paint_chart(self, painter: QPainter, area: QRectF) -> None:
        legend_h = 20
        axis_h = 24
        plot = QRectF(area.left() + 4, area.top() + legend_h, area.width() - 8,
                      area.height() - legend_h - axis_h)
        peak = max((max(v, default=0) for v in self.series.values()), default=1)
        peak = max(1.0, float(peak))
        step = plot.width() / max(1, len(self.labels) - 1)

        painter.setFont(_mono(7))
        for fraction in (0.0, 0.5, 1.0):
            y = plot.bottom() - plot.height() * fraction
            painter.setPen(QPen(QColor(BORDER), 1, Qt.DotLine))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(DIM))
            painter.drawText(QRectF(plot.left() - 4, y - 7, 40, 14),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             f"{peak * fraction:.0f}")

        for index, (name, values) in enumerate(self.series.items()):
            color = _color(index)
            points = [QPointF(plot.left() + step * i,
                              plot.bottom() - plot.height() * (v / peak))
                      for i, v in enumerate(values)]
            if not points:
                continue
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            for point in points:
                painter.drawEllipse(point, 2.5, 2.5)

            legend_x = plot.left() + index * 110
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(legend_x, area.top() + 4, 10, 10), 2, 2)
            painter.setPen(QColor(TEXT))
            painter.setFont(QFont(painter.font().family(), 9))
            painter.drawText(QRectF(legend_x + 15, area.top(), 90, 18),
                             Qt.AlignLeft | Qt.AlignVCenter, name)
            painter.setFont(_mono(8))

        painter.setPen(QColor(DIM))
        painter.setFont(QFont(painter.font().family(), 8))
        every = max(1, len(self.labels) // 7)
        for index, label in enumerate(self.labels):
            if index % every:
                continue
            x = plot.left() + step * index
            painter.drawText(QRectF(x - 40, plot.bottom() + 4, 80, 18),
                             Qt.AlignCenter, label[5:])
