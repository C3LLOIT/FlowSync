"""
test_app.py
───────────
PySide6 test bench for the eye_module.

Place this file one level above eye_module/:

    project/
    ├── test_app.py
    └── eye_module/
        ├── eye_tracker.py
        └── ...

Run:
    python test_app.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGroupBox, QSlider,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QDateTime
from PySide6.QtGui  import QColor, QPalette, QPixmap, QImage, QPainter

if TYPE_CHECKING:
    from eye_module.eye_tracker import EyeTracker
    from eye_module.preview     import PreviewModule


# ── Screen size ───────────────────────────────────────────────────────────────

def _get_screen_size() -> tuple[int, int]:
    """MouseController → PySide6 → tkinter → 1920×1080 fallback."""
    try:
        from eye_module.mouse_control import MouseController
        return MouseController().screen_size()
    except Exception:
        pass
    try:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            screen = app.primaryScreen()
            if screen is not None:
                s = screen.size()
                return s.width(), s.height()
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy(); return w, h
    except Exception:
        pass
    import logging
    logging.getLogger(__name__).warning("Screen size unknown — defaulting to 1920×1080")
    return 1920, 1080


# ── Palette ───────────────────────────────────────────────────────────────────

DARK: dict[str, str] = {
    "bg":         "#0f1117",
    "surface":    "#1a1d27",
    "surface2":   "#22263a",
    "border":     "#2e3250",
    "accent":     "#5c7cfa",
    "accent_dim": "#2d3b80",
    "green":      "#37b24d",
    "red":        "#f03e3e",
    "amber":      "#f59f00",
    "text":       "#e8eaf6",
    "text_dim":   "#8b90b0",
    "text_muted": "#4a4f6a",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK["bg"]};
    color: {DARK["text"]};
    font-family: "Segoe UI", "SF Pro Text", "Inter", sans-serif;
    font-size: 13px;
}}
QFrame#panel {{
    background-color: {DARK["surface"]};
    border: 1px solid {DARK["border"]};
    border-radius: 10px;
}}
QGroupBox {{
    background-color: {DARK["surface"]};
    border: 1px solid {DARK["border"]};
    border-radius: 8px;
    font-size: 11px; font-weight: 600;
    color: {DARK["text_dim"]};
    letter-spacing: 0.5px;
    margin-top: 14px; padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; top: -1px; padding: 0 6px;
    background-color: {DARK["surface"]};
}}
QPushButton {{
    background-color: {DARK["surface2"]}; color: {DARK["text"]};
    border: 1px solid {DARK["border"]}; border-radius: 6px;
    padding: 7px 18px; font-size: 13px;
}}
QPushButton:hover   {{ background-color: {DARK["accent_dim"]}; border-color: {DARK["accent"]}; }}
QPushButton:pressed {{ background-color: {DARK["accent"]}; color: #fff; }}
QPushButton:disabled {{ color: {DARK["text_muted"]}; background-color: {DARK["surface"]}; border-color: {DARK["border"]}; }}
QPushButton#btn_primary {{ background-color: {DARK["accent"]}; color: #fff; border: none; font-weight: 600; }}
QPushButton#btn_primary:hover {{ background-color: #7b96ff; }}
QPushButton#btn_primary:disabled {{ background-color: {DARK["accent_dim"]}; color: {DARK["text_muted"]}; }}
QPushButton#btn_danger {{ background-color: {DARK["red"]}; color: #fff; border: none; font-weight: 600; }}
QPushButton#btn_danger:hover {{ background-color: #ff6b6b; }}
QLabel#section_title {{ font-size: 11px; font-weight: 600; color: {DARK["text_dim"]}; letter-spacing: 0.6px; }}
QLabel#value_large  {{ font-size: 20px; font-weight: 700; color: {DARK["text"]}; font-family: "Consolas", monospace; }}
QLabel#log_text     {{ font-family: "Consolas", monospace; font-size: 12px; color: {DARK["text_dim"]}; background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {DARK["surface"]}; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: {DARK["border"]}; border-radius: 3px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSlider::groove:horizontal {{ background: {DARK["surface2"]}; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {DARK["accent"]}; width: 14px; height: 14px; border-radius: 7px; margin: -5px 0; }}
QSlider::sub-page:horizontal {{ background: {DARK["accent"]}; border-radius: 2px; }}
"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TRACKER BRIDGE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

class TrackerBridge(QObject):
    """
    Thin QObject wrapper around EyeTracker + PreviewModule.
    Converts plain-Python callbacks → Qt signals (thread-safe).
    No monkey-patching — EyeTracker.on_frame hook used for preview.
    """

    sig_move          = Signal(float, float)
    sig_blink_click   = Signal()
    sig_dwell_click   = Signal()
    sig_face_found    = Signal()
    sig_face_lost     = Signal()
    sig_error         = Signal(str)
    sig_status        = Signal(str)
    sig_preview_frame = Signal(object)   # numpy BGR frame
    sig_cal_state     = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._tracker: Optional[EyeTracker]   = None
        self._preview: Optional[PreviewModule] = None

    # ── Build ──────────────────────────────────────────────────────────

    def _build(self) -> bool:
        from eye_module.eye_tracker import EyeTracker, ensure_model
        from eye_module.preview     import PreviewModule

        self.sig_status.emit("Checking model …")
        try:
            ensure_model()
        except RuntimeError as exc:
            self.sig_error.emit(str(exc))
            return False

        screen_w, screen_h = _get_screen_size()

        self._preview = PreviewModule(mode="eye", mirror=True)
        self._tracker = EyeTracker(
            screen_w=screen_w,
            screen_h=screen_h,
            load_calibration=True,
        )

        # Wire all callbacks → Qt signals
        self._tracker.on_move        = self.sig_move.emit
        self._tracker.on_blink_click = self.sig_blink_click.emit
        self._tracker.on_dwell_click = self.sig_dwell_click.emit
        self._tracker.on_face_found  = self.sig_face_found.emit
        self._tracker.on_face_lost   = self.sig_face_lost.emit
        self._tracker.on_error       = lambda e: self.sig_error.emit(str(e))

        # Preview hook — EyeTracker calls on_frame(bgr_frame, mp_result)
        # every frame from the tracking thread. We render via PreviewModule
        # and emit the result as a signal.
        preview = self._preview

        def _on_frame(bgr_frame: np.ndarray, result: object) -> None:
            rendered = preview.render(bgr_frame, result)
            self.sig_preview_frame.emit(rendered)

        self._tracker.on_frame = _on_frame
        return True

    # ── Control API ───────────────────────────────────────────────────

    def start(self) -> None:
        if self._tracker is None:
            if not self._build():
                return
        assert self._tracker is not None
        self._tracker.start()
        self.sig_status.emit("Tracking active")

    def stop(self) -> None:
        if self._tracker is not None and self._tracker.is_running:
            self._tracker.stop()
        self.sig_status.emit("Stopped")

    def run_calibration(self) -> None:
        """Called from a worker thread."""
        if self._tracker is None:
            if not self._build():
                return
        assert self._tracker is not None

        import cv2, mediapipe as mp

        cap        = self._tracker._open_camera()
        landmarker = self._tracker._build_landmarker(mp, mode="video")

        self.sig_status.emit("Running calibration …")

        def frame_cb(annotated: np.ndarray) -> None:
            self.sig_preview_frame.emit(annotated)

        try:
            ok = self._tracker._calibration.run(
                landmarker, cap, frame_callback=frame_cb
            )
        finally:
            landmarker.close()
            cap.release()
            self.sig_cal_state.emit("")

        self.sig_status.emit(
            "Calibration complete ✓" if ok else "Calibration aborted"
        )

    def set_preview_enabled(self, enabled: bool) -> None:
        if self._preview is not None:
            self._preview.enabled = enabled

    def set_dwell_enabled(self, enabled: bool) -> None:
        if self._tracker is not None:
            self._tracker.dwell_enabled = enabled

    def set_ema_alpha(self, alpha: float) -> None:
        if self._tracker is not None:
            self._tracker._smoother.alpha = alpha

    @property
    def is_running(self) -> bool:
        return self._tracker is not None and self._tracker.is_running

    @property
    def is_calibrated(self) -> bool:
        return self._tracker is not None and self._tracker.is_calibrated


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  WIDGETS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _panel(parent: Optional[QWidget] = None) -> QFrame:
    f = QFrame(parent); f.setObjectName("panel"); return f

def _label(text: str, obj_name: str = "",
           parent: Optional[QWidget] = None) -> QLabel:
    lbl = QLabel(text, parent)
    if obj_name: lbl.setObjectName(obj_name)
    return lbl


class StatusDot(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._color = QColor(DARK["text_muted"])

    def set_color(self, hex_color: str) -> None:
        self._color = QColor(hex_color); self.update()

    def paintEvent(self, event: object) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(self._color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, self.width(), self.height())


class StatCard(QFrame):
    def __init__(self, title: str, initial: str = "—",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12); lay.setSpacing(4)
        self._title = _label(title.upper(), "section_title")
        self._value = _label(initial, "value_large")
        lay.addWidget(self._title); lay.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class PreviewPanel(QFrame):
    W, H = 320, 240

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFixedSize(self.W, self.H + 36)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(28)
        hdr.setStyleSheet(
            f"background:{DARK['surface2']};"
            f"border-bottom:1px solid {DARK['border']}; border-radius:0px;"
        )
        h = QHBoxLayout(hdr); h.setContentsMargins(10, 0, 10, 0)
        self._hdr_lbl = QLabel("CAMERA PREVIEW"); self._hdr_lbl.setObjectName("section_title")
        self._cal_badge = QLabel(""); self._cal_badge.setStyleSheet("font-size:10px;font-weight:600;")
        h.addWidget(self._hdr_lbl); h.addStretch(); h.addWidget(self._cal_badge)
        lay.addWidget(hdr)

        self._img = QLabel()
        self._img.setFixedSize(self.W, self.H)
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet(f"background:{DARK['bg']};")
        lay.addWidget(self._img)
        self._show_placeholder()

    def update_frame(self, bgr: np.ndarray) -> None:
        import cv2
        resized = cv2.resize(bgr, (self.W, self.H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        self._img.setPixmap(QPixmap.fromImage(img))

    def set_cal_state(self, state: str) -> None:
        cfg: dict[str, tuple[str, str]] = {
            "waiting":  ("Hold still …", DARK["amber"]),
            "stable":   ("Gaze stable",  DARK["green"]),
            "sampling": ("Sampling …",   DARK["accent"]),
            "":         ("",             ""),
        }
        text, color = cfg.get(state, ("", ""))
        self._cal_badge.setText(text)
        self._cal_badge.setStyleSheet(
            f"font-size:10px;font-weight:600;color:{color};" if color else ""
        )

    def _show_placeholder(self) -> None:
        import cv2
        ph = np.zeros((self.H, self.W, 3), dtype=np.uint8); ph[:] = (20, 22, 30)
        msg = "Preview off"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(ph, msg, (self.W//2 - tw//2, self.H//2 + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 75, 100), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._img.setPixmap(QPixmap.fromImage(
            QImage(rgb.data.tobytes(), w, h, ch*w, QImage.Format.Format_RGB888)
        ))


class EventLog(QWidget):
    MAX_LINES = 120

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container.setStyleSheet(f"background-color:{DARK['surface']};")
        self._lay = QVBoxLayout(self._container)
        self._lay.setContentsMargins(12, 8, 12, 8); self._lay.setSpacing(1)
        self._lay.addStretch()
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)
        self._lines: list[QLabel] = []

    def append(self, text: str, color: str = "") -> None:
        ts  = QDateTime.currentDateTime().toString("hh:mm:ss.zzz")
        lbl = _label(f"{ts}  {text}", "log_text")
        if color: lbl.setStyleSheet(f"color:{color};")
        self._lay.insertWidget(self._lay.count() - 1, lbl)
        self._lines.append(lbl)
        if len(self._lines) > self.MAX_LINES:
            self._lines.pop(0).deleteLater()
        sb = self._scroll.verticalScrollBar()
        QTimer.singleShot(10, lambda: sb.setValue(sb.maximum()))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN WINDOW                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Eye Module — Test Bench")
        self.resize(1060, 700); self.setMinimumSize(900, 580)

        self._bridge      = TrackerBridge(self)
        self._click_count = 0
        self._dwell_count = 0
        self._preview_on  = False
        self._pending_frame: Optional[np.ndarray] = None

        self._build_ui()
        self._connect_signals()

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(600)
        self._pulse_timer.timeout.connect(self._pulse_dot)
        self._pulse_state = False

        # Frame-rate limiter: consume tracker frames at 30 fps in the UI
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)
        self._frame_timer.timeout.connect(self._flush_preview_frame)
        self._frame_timer.start()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setContentsMargins(16, 14, 16, 14); rl.setSpacing(12)
        rl.addWidget(self._make_header())
        body = QHBoxLayout(); body.setSpacing(12)
        body.addLayout(self._make_left_column(), stretch=0)
        body.addLayout(self._make_right_column(), stretch=1)
        rl.addLayout(body)

    def _make_header(self) -> QFrame:
        frame = _panel(); lay = QHBoxLayout(frame); lay.setContentsMargins(16, 10, 16, 10)
        title = QLabel("Eye Module  ·  Test Bench")
        title.setStyleSheet(f"font-size:15px;font-weight:700;color:{DARK['text']};")
        lay.addWidget(title); lay.addStretch()
        self._dot = StatusDot()
        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet(f"color:{DARK['text_dim']};font-size:12px;")
        lay.addWidget(self._dot); lay.addSpacing(6); lay.addWidget(self._status_label)
        return frame

    def _make_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout(); col.setSpacing(12)

        ctrl = QGroupBox("Controls"); cl = QVBoxLayout(ctrl)
        cl.setSpacing(8); cl.setContentsMargins(12, 16, 12, 12)
        self._btn_start     = QPushButton("Start Tracking"); self._btn_start.setObjectName("btn_primary")
        self._btn_stop      = QPushButton("Stop Tracking");  self._btn_stop.setObjectName("btn_danger"); self._btn_stop.setEnabled(False)
        self._btn_calibrate = QPushButton("Run Calibration")
        cl.addWidget(self._btn_start); cl.addWidget(self._btn_stop); cl.addWidget(self._btn_calibrate)

        opts = QGroupBox("Options"); ol = QVBoxLayout(opts)
        ol.setSpacing(10); ol.setContentsMargins(12, 16, 12, 12)

        # Preview toggle
        pr = QHBoxLayout(); pl = QLabel("Camera Preview"); pl.setStyleSheet(f"color:{DARK['text_dim']};")
        self._btn_preview = QPushButton("Off"); self._btn_preview.setCheckable(True); self._btn_preview.setFixedWidth(56)
        self._btn_preview.setStyleSheet(
            f"QPushButton{{color:{DARK['text_muted']};}} QPushButton:checked{{color:{DARK['green']};border-color:{DARK['green']};}}"
        )
        pr.addWidget(pl); pr.addStretch(); pr.addWidget(self._btn_preview); ol.addLayout(pr)
        self._div(ol)

        # Dwell toggle
        dr = QHBoxLayout(); dl = QLabel("Dwell-click"); dl.setStyleSheet(f"color:{DARK['text_dim']};")
        self._btn_dwell = QPushButton("Off"); self._btn_dwell.setCheckable(True); self._btn_dwell.setFixedWidth(56)
        self._btn_dwell.setStyleSheet(
            f"QPushButton{{color:{DARK['text_muted']};}} QPushButton:checked{{color:{DARK['green']};border-color:{DARK['green']};}}"
        )
        dr.addWidget(dl); dr.addStretch(); dr.addWidget(self._btn_dwell); ol.addLayout(dr)
        self._div(ol)

        # EMA alpha slider
        al = QLabel("Smoothing (EMA α)"); al.setStyleSheet(f"color:{DARK['text_dim']};")
        ar = QHBoxLayout()
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(5, 80); self._alpha_slider.setValue(25)
        self._alpha_label  = QLabel("0.25"); self._alpha_label.setFixedWidth(34)
        self._alpha_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._alpha_label.setStyleSheet(f"color:{DARK['accent']};font-family:monospace;")
        ar.addWidget(self._alpha_slider); ar.addWidget(self._alpha_label)
        ol.addWidget(al); ol.addLayout(ar)

        col.addWidget(ctrl); col.addWidget(opts); col.addStretch()
        return col

    def _make_right_column(self) -> QVBoxLayout:
        col = QVBoxLayout(); col.setSpacing(12)

        top = QHBoxLayout(); top.setSpacing(12)
        cards = QVBoxLayout(); cards.setSpacing(8)

        r1 = QHBoxLayout(); r1.setSpacing(8)
        self._card_pos      = StatCard("Cursor Position", "—, —")
        self._card_face     = StatCard("Face", "Not detected")
        self._card_calib    = StatCard("Calibration", "None")
        r1.addWidget(self._card_pos); r1.addWidget(self._card_face); r1.addWidget(self._card_calib)

        r2 = QHBoxLayout(); r2.setSpacing(8)
        self._card_clicks   = StatCard("Blink Clicks", "0")
        self._card_dwell_c  = StatCard("Dwell Clicks", "0")
        self._card_calstate = StatCard("Cal. State", "—")
        r2.addWidget(self._card_clicks); r2.addWidget(self._card_dwell_c); r2.addWidget(self._card_calstate)

        cards.addLayout(r1); cards.addLayout(r2)
        top.addLayout(cards, stretch=1)
        self._preview_panel = PreviewPanel()
        top.addWidget(self._preview_panel, stretch=0)
        col.addLayout(top)

        lg = QGroupBox("Event Log"); ll = QVBoxLayout(lg); ll.setContentsMargins(4, 12, 4, 4)
        self._log = EventLog(); self._log.setMinimumHeight(260); ll.addWidget(self._log)
        bc = QPushButton("Clear log"); bc.setFixedWidth(90); bc.clicked.connect(self._clear_log)
        ll.addWidget(bc, alignment=Qt.AlignmentFlag.AlignRight)
        col.addWidget(lg)
        return col

    # ── Signals ───────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_calibrate.clicked.connect(self._on_calibrate)
        self._btn_preview.toggled.connect(self._on_preview_toggled)
        self._btn_dwell.toggled.connect(self._on_dwell_toggled)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)

        self._bridge.sig_move.connect(self._on_move)
        self._bridge.sig_blink_click.connect(self._on_blink_click)
        self._bridge.sig_dwell_click.connect(self._on_dwell_click)
        self._bridge.sig_face_found.connect(self._on_face_found)
        self._bridge.sig_face_lost.connect(self._on_face_lost)
        self._bridge.sig_error.connect(self._on_error)
        self._bridge.sig_status.connect(self._on_status)
        self._bridge.sig_preview_frame.connect(self._on_preview_frame)
        self._bridge.sig_cal_state.connect(self._on_cal_state)

    # ── Handlers ──────────────────────────────────────────────────────

    def _on_start(self) -> None:
        self._log.append("▶ Starting tracker …", DARK["accent"])
        self._set_status("Starting …", DARK["amber"])
        self._bridge.start()
        self._btn_start.setEnabled(False); self._btn_stop.setEnabled(True)
        self._btn_calibrate.setEnabled(False)
        self._pulse_timer.start()

    def _on_stop(self) -> None:
        self._log.append("■ Stopping …", DARK["text_dim"])
        self._bridge.stop()
        self._btn_start.setEnabled(True); self._btn_stop.setEnabled(False)
        self._btn_calibrate.setEnabled(True)
        self._pulse_timer.stop(); self._dot.set_color(DARK["text_muted"])
        self._card_face.set_value("Stopped"); self._card_pos.set_value("—, —")
        self._card_calstate.set_value("—")

    def _on_calibrate(self) -> None:
        self._log.append("⊙ Starting calibration …", DARK["amber"])
        self._card_calstate.set_value("Waiting")
        threading.Thread(target=self._bridge.run_calibration, daemon=True).start()

    def _on_preview_toggled(self, checked: bool) -> None:
        self._preview_on = checked
        self._btn_preview.setText("On" if checked else "Off")
        self._bridge.set_preview_enabled(checked)
        if not checked:
            self._preview_panel._show_placeholder()
        self._log.append(
            f"◉ Preview {'enabled' if checked else 'disabled'}", DARK["text_dim"]
        )

    def _on_dwell_toggled(self, checked: bool) -> None:
        self._btn_dwell.setText("On" if checked else "Off")
        self._bridge.set_dwell_enabled(checked)
        self._log.append(
            f"⊕ Dwell-click {'enabled' if checked else 'disabled'}",
            DARK["green"] if checked else DARK["text_dim"],
        )

    def _on_alpha_changed(self, value: int) -> None:
        alpha = value / 100.0
        self._alpha_label.setText(f"{alpha:.2f}")
        self._bridge.set_ema_alpha(alpha)

    def _on_move(self, x: float, y: float) -> None:
        self._card_pos.set_value(f"{x:.0f}, {y:.0f}")

    def _on_blink_click(self) -> None:
        self._click_count += 1
        self._card_clicks.set_value(str(self._click_count))
        self._log.append(f"🖱  Blink-click  #{self._click_count}", DARK["accent"])

    def _on_dwell_click(self) -> None:
        self._dwell_count += 1
        self._card_dwell_c.set_value(str(self._dwell_count))
        self._log.append(f"🕐  Dwell-click  #{self._dwell_count}", DARK["green"])

    def _on_face_found(self) -> None:
        self._card_face.set_value("Detected ✓")
        self._log.append("● Face detected", DARK["green"])

    def _on_face_lost(self) -> None:
        self._card_face.set_value("Lost"); self._card_pos.set_value("—, —")
        self._log.append("○ Face lost", DARK["amber"])

    def _on_error(self, msg: str) -> None:
        self._set_status("Error", DARK["red"])
        self._log.append(f"✖ {msg}", DARK["red"])

    def _on_status(self, msg: str) -> None:
        self._set_status(msg)
        self._log.append(f"  {msg}", DARK["text_dim"])
        if "complete" in msg.lower() or self._bridge.is_calibrated:
            self._card_calib.set_value("Loaded ✓")

    def _on_preview_frame(self, frame: np.ndarray) -> None:
        if self._preview_on:
            self._pending_frame = frame

    def _flush_preview_frame(self) -> None:
        if self._pending_frame is not None and self._preview_on:
            self._preview_panel.update_frame(self._pending_frame)
            self._pending_frame = None

    def _on_cal_state(self, state: str) -> None:
        labels: dict[str, str] = {
            "waiting": "Waiting", "stable": "Stable ✓",
            "sampling": "Sampling …", "": "—",
        }
        colors: dict[str, str] = {
            "waiting": DARK["amber"], "stable": DARK["green"],
            "sampling": DARK["accent"], "": DARK["text_dim"],
        }
        self._card_calstate.set_value(labels.get(state, "—"))
        self._card_calstate._value.setStyleSheet(
            f"font-size:20px;font-weight:700;color:{colors.get(state, DARK['text'])};"
        )
        self._preview_panel.set_cal_state(state)

    # ── Helpers ───────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = "") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color:{color or DARK['text_dim']};font-size:12px;"
        )

    def _pulse_dot(self) -> None:
        self._pulse_state = not self._pulse_state
        self._dot.set_color(DARK["accent"] if self._pulse_state else DARK["accent_dim"])

    def _clear_log(self) -> None:
        for lbl in self._log._lines: lbl.deleteLater()
        self._log._lines.clear()

    def _div(self, layout: QVBoxLayout) -> None:
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color:{DARK['border']};margin:2px 0;")
        layout.addWidget(line)

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        self._frame_timer.stop(); self._bridge.stop()
        if hasattr(event, "accept"): event.accept()  # type: ignore[union-attr]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main() -> None:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    cr = QPalette.ColorRole
    pal = app.palette()
    pal.setColor(cr.Window,          QColor(DARK["bg"]))
    pal.setColor(cr.WindowText,      QColor(DARK["text"]))
    pal.setColor(cr.Base,            QColor(DARK["surface"]))
    pal.setColor(cr.AlternateBase,   QColor(DARK["surface2"]))
    pal.setColor(cr.Text,            QColor(DARK["text"]))
    pal.setColor(cr.Button,          QColor(DARK["surface2"]))
    pal.setColor(cr.ButtonText,      QColor(DARK["text"]))
    pal.setColor(cr.Highlight,       QColor(DARK["accent"]))
    pal.setColor(cr.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
