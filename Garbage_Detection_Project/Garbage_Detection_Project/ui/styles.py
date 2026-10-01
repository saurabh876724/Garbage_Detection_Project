"""Industrial AI waste-sorting control room theme.

One stylesheet, one colour vocabulary. Widgets never hardcode colours - they
reference the constants here so the whole app can be re-themed in one place.

Design language: industrial sorting line / materials recovery facility /
computer-vision inspection station.  Amber safety accent on graphite surfaces,
monospace technical readouts, hairline borders, restrained palette.
"""
from __future__ import annotations

# --- palette ----------------------------------------------------------------
BG = "#12151A"
PANEL = "#1A1E25"
ELEVATED = "#232830"
CARD = "#1E2229"
CARD_HOVER = "#272D36"
BORDER = "#2C323C"
BORDER_LIGHT = "#363D48"
TEXT = "#E8EAED"
MUTED = "#8B93A1"
DIM = "#5C6370"

ACCENT = "#F5A623"         # safety amber
ACCENT_DIM = "#C4871B"
ACCENT_TINT = "rgba(245, 166, 35, 0.08)"

GREEN = "#3DD68C"
RED = "#E5484D"
YELLOW = "#E8C547"
BLUE = "#58A6FF"

STATUS_COLOR = {
    "CLEAN": GREEN,
    "DIRTY": RED,
    "REVIEW": YELLOW,
    "INFO": BLUE,
}

CHART_SERIES = (ACCENT, GREEN, YELLOW, "#A78BFA", "#22D3EE", RED,
                "#F97316", "#84CC16", "#E879F9", "#94A3B8")

# --- typography -------------------------------------------------------------
FONT_FAMILY = '"IBM Plex Sans", "Inter", "Segoe UI", sans-serif'
MONO_FAMILY = '"IBM Plex Mono", "JetBrains Mono", "Consolas", monospace'

# --- SVG icon paths (Lucide-style, 24x24 viewBox) --------------------------
ICONS = {
    "dashboard": '<path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"/>',
    "live": '<circle cx="12" cy="12" r="3"/><path d="M12 2a10 10 0 0 1 0 20 10 10 0 0 1 0-20"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
    "video": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "history": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "analytics": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "about": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "pause": '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
    "stop": '<rect x="3" y="3" width="18" height="18" rx="2"/>',
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "export": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "empty_image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
    "empty_camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "empty_history": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
}


def icon_svg(name: str, color: str = MUTED, size: int = 20) -> str:
    """Return an inline SVG data-URI for embedding in stylesheets or labels."""
    path = ICONS.get(name, "")
    if not path:
        return ""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
           f'height="{size}" viewBox="0 0 24 24" fill="none" '
           f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
           f'stroke-linejoin="round">{path}</svg>')
    import base64
    encoded = base64.b64encode(svg.encode()).decode()
    return f"url(data:image/svg+xml;base64,{encoded})"


DARK_QSS = f"""
/* === BASE ================================================================ */
* {{
    font-family: {FONT_FAMILY};
    font-size: 13px;
    color: {TEXT};
    outline: none;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ background: transparent; }}

/* === CARDS & SURFACES ==================================================== */
QFrame#Card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QFrame#Sidebar {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
QFrame#Header {{
    background: {PANEL};
    border-bottom: 1px solid {BORDER};
}}

/* === TYPOGRAPHY ========================================================== */
QLabel#Title {{
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.3px;
}}
QLabel#Subtitle {{
    color: {MUTED};
    font-size: 12px;
    font-weight: 400;
}}
QLabel#SectionTitle {{
    font-size: 11px;
    font-weight: 700;
    color: {TEXT};
    text-transform: uppercase;
    letter-spacing: 1.0px;
}}
QLabel#MetricValue {{
    font-family: {MONO_FAMILY};
    font-size: 28px;
    font-weight: 600;
    letter-spacing: -0.5px;
}}
QLabel#MetricLabel {{
    color: {MUTED};
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
QLabel#EmptyState {{
    color: {DIM};
    font-size: 13px;
}}
QLabel#MonoValue {{
    font-family: {MONO_FAMILY};
    font-size: 12px;
    font-weight: 500;
}}
QLabel#VideoSurface {{
    background: #0D0F13;
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {DIM};
}}
QLabel#StatusBadge {{
    border-radius: 4px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
    font-family: {MONO_FAMILY};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* === BUTTONS ============================================================= */
QPushButton {{
    background: {ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 16px;
    font-weight: 500;
    font-size: 12px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {CARD_HOVER};
    border-color: {BORDER_LIGHT};
}}
QPushButton:pressed {{
    background: {PANEL};
}}
QPushButton:disabled {{
    color: {DIM};
    border-color: {BORDER};
    background: {PANEL};
}}
QPushButton#Primary {{
    background: {ACCENT};
    color: #12151A;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#Primary:hover {{
    background: #E09A1E;
    border-color: #E09A1E;
}}
QPushButton#Primary:pressed {{
    background: {ACCENT_DIM};
}}
QPushButton#Danger {{
    background: rgba(229, 72, 77, 0.12);
    color: {RED};
    border: 1px solid rgba(229, 72, 77, 0.3);
}}
QPushButton#Danger:hover {{
    background: rgba(229, 72, 77, 0.22);
    border-color: {RED};
}}
QPushButton#Secondary {{
    background: transparent;
    border: 1px solid {BORDER};
    color: {MUTED};
}}
QPushButton#Secondary:hover {{
    border-color: {MUTED};
    color: {TEXT};
}}

/* === NAVIGATION ========================================================== */
QListWidget#Nav {{
    background: transparent;
    border: none;
    font-size: 13px;
}}
QListWidget#Nav::item {{
    padding: 10px 14px;
    border-radius: 4px;
    margin: 1px 6px;
    color: {MUTED};
    border-left: 3px solid transparent;
}}
QListWidget#Nav::item:hover {{
    background: {ELEVATED};
    color: {TEXT};
}}
QListWidget#Nav::item:selected {{
    background: {ACCENT_TINT};
    color: {ACCENT};
    border-left: 3px solid {ACCENT};
    font-weight: 500;
}}

/* === INPUTS ============================================================== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
    font-size: 12px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    selection-color: #12151A;
}}

/* === CHECKBOXES ========================================================== */
QCheckBox {{
    spacing: 8px;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 3px;
    background: {PANEL};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {MUTED};
}}

/* === SLIDERS ============================================================= */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
    background: {ACCENT};
    border: 2px solid {BG};
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

/* === TABLES ============================================================== */
QTableWidget {{
    background: {PANEL};
    alternate-background-color: #161A21;
    border: 1px solid {BORDER};
    border-radius: 4px;
    gridline-color: {BORDER};
    font-size: 12px;
}}
QHeaderView::section {{
    background: {CARD};
    color: {MUTED};
    padding: 8px 10px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
QTableWidget::item {{
    padding: 6px 10px;
    border-bottom: 1px solid rgba(44, 50, 60, 0.4);
}}
QTableWidget::item:selected {{
    background: {ACCENT_TINT};
    color: {TEXT};
}}
QTableWidget::item:hover {{
    background: {ELEVATED};
}}

/* === SCROLLBARS ========================================================== */
QScrollArea {{ border: none; }}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {BORDER_LIGHT};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 4px;
    min-width: 30px;
}}

/* === PROGRESS BAR ======================================================== */
QProgressBar {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 3px;
    height: 10px;
    text-align: center;
    color: {TEXT};
    font-family: {MONO_FAMILY};
    font-size: 9px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 2px;
}}

/* === MISC ================================================================ */
QMessageBox {{
    background: {CARD};
}}
QToolTip {{
    background: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px 8px;
    font-size: 11px;
    border-radius: 3px;
}}
QStatusBar {{
    background: {PANEL};
    color: {MUTED};
    font-size: 11px;
    border-top: 1px solid {BORDER};
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
    font-weight: 600;
    color: {MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
"""
