"""
test_app.py
───────────
PySide6 test application for the eye_module.

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
    QLabel, QPushButton, QFrame, QScrollArea,
    QGroupBox, QSlider,
)
from PySide6.QtCore import (
    Qt, Signal, QObject, QTimer, QDateTime,
)
from PySide6.QtGui import (
    QColor, QPalette, QPixmap, QImage,
    QPainter,
)

# PySide6 6.x scoped enums — imported explicitly so Pyright resolves them.
from PySide6.QtCore import Qt as _Qt
from PySide6.QtGui  import QPainter as _QPainter, QPalette as _QPalette
from PySide6.QtWidgets import QFrame as _QFrame

# TYPE_CHECKING-only import for EyeTracker — avoids runtime circular import
# and satisfies Pyright's attribute checks on self._tracker.
if TYPE_CHECKING:
    from eye_module.eye_tracker import EyeTracker
    from eye_module.preview     import PreviewModule


# ── Screen size helper ────────────────────────────────────────────────────────

def _get_screen_size() -> tuple[int, int]:
    """
    Return (width, height) of the primary screen.
    Tries autopy → PySide6 QScreen → tkinter → 1920×1080 fallback.
    """
    # 1. autopy
    try:
        import autopy  # type: ignore[import]
        sw, sh = autopy.screen.size()
        return int(sw), int(sh)
    except Exception:
        pass

    # 2. PySide6 QApplication (already a hard dep — safe to use here)
    try:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            screen = app.primaryScreen()
            if screen is not None:
                s = screen.size()
                return s.width(), s.height()
    except Exception:
        pass

    # 3. tkinter (stdlib)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        pass

    # 4. Safe fallback
    import logging
    logging.getLogger(__name__).warning(
        "Could not detect screen size — defaulting to 1920×1080."
    )
    return 1920, 1080


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PALETTE & STYLE                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

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
    font-family: "Segoe UI", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
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
    font-size: 11px;
    font-weight: 600;
    color: {DARK["text_dim"]};
    letter-spacing: 0.5px;
    margin-top: 14px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: -1px;
    padding: 0 6px;
    background-color: {DARK["surface"]};
}}
QPushButton {{
    background-color: {DARK["surface2"]};
    color: {DARK["text"]};
    border: 1px solid {DARK["border"]};
    border-radius: 6px;
    padding: 7px 18px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {DARK["accent_dim"]};
    border-color: {DARK["accent"]};
}}
QPushButton:pressed {{
    background-color: {DARK["accent"]};
    color: #ffffff;
}}
QPushButton:disabled {{
    color: {DARK["text_muted"]};
    background-color: {DARK["surface"]};
    border-color: {DARK["border"]};
}}
QPushButton#btn_primary {{
    background-color: {DARK["accent"]};
    color: #ffffff;
    border: none;
    font-weight: 600;
}}
QPushButton#btn_primary:hover  {{ background-color: #7b96ff; }}
QPushButton#btn_primary:disabled {{
    background-color: {DARK["accent_dim"]};
    color: {DARK["text_muted"]};
}}
QPushButton#btn_danger {{
    background-color: {DARK["red"]};
    color: #ffffff;
    border: none;
    font-weight: 600;
}}
QPushButton#btn_danger:hover {{ background-color: #ff6b6b; }}
QLabel#section_title {{
    font-size: 11px;
    font-weight: 600;
    color: {DARK["text_dim"]};
    letter-spacing: 0.6px;
}}
QLabel#value_large {{
    font-size: 20px;
    font-weight: 700;
    color: {DARK["text"]};
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace;
}}
QLabel#log_text {{
    font-family: "Consolas", "JetBrains Mono", "Courier New", monospace;
    font-size: 12px;
    color: {DARK["text_dim"]};
    background: transparent;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {DARK["surface"]};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {DARK["border"]};
    border-radius: 3px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSlider::groove:horizontal {{
    background: {DARK["surface2"]};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {DARK["accent"]};
    width: 14px; height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}}
QSlider::sub-page:horizontal {{
    background: {DARK["accent"]};
    border-radius: 2px;
}}
"""


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TRACKER BRIDGE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

class TrackerBridge(QObject):
    """
    Wraps EyeTracker + PreviewModule, emits Qt signals to the UI.
    All tracker callbacks fire on the background thread; signals
    cross to the Qt main thread automatically.
    """

    sig_move          = Signal(float, float)
    sig_blink_click   = Signal()
    sig_dwell_click   = Signal()
    sig_face_found    = Signal()
    sig_face_lost     = Signal()
    sig_error         = Signal(str)
    sig_status        = Signal(str)
    sig_preview_frame = Signal(object)   # numpy BGR frame
    sig_cal_state     = Signal(str)      # "waiting"|"stable"|"sampling"|""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Typed as Optional; assigned in _build() before any access.
        self._tracker: Optional[EyeTracker]  = None
        self._preview: Optional[PreviewModule] = None

    # ── Internal build ────────────────────────────────────────────────

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
        self._tracker.on_move        = self._on_tracker_move
        self._tracker.on_blink_click = self.sig_blink_click.emit
        self._tracker.on_dwell_click = self.sig_dwell_click.emit
        self._tracker.on_face_found  = self.sig_face_found.emit
        self._tracker.on_face_lost   = self._on_face_lost
        self._tracker.on_error       = lambda e: self.sig_error.emit(str(e))

        self._patch_tracker_for_preview()
        return True

    def _patch_tracker_for_preview(self) -> None:
        """Replace the tracker's _tracking_loop to also emit preview frames."""
        # Guaranteed non-None — called only from _build after assignment.
        assert self._tracker is not None
        assert self._preview is not None

        bridge  = self
        tracker = self._tracker
        preview = self._preview

        def patched_loop() -> None:
            import cv2
            import mediapipe as mp

            from eye_module.eye_tracker import _import_autopy
            from eye_module.calibration import raw_gaze_point
            from eye_module.config      import SCREEN_MARGIN_PX

            autopy_mod = _import_autopy()   # None if not installed

            cap        = tracker._open_camera()
            landmarker = tracker._build_landmarker(mp, mode="video")

            face_was_visible = False
            fps_hint:  int = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            frame_ts_ms: int = 0

            try:
                while not tracker._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        import time
                        time.sleep(0.05)
                        continue

                    frame_ts_ms += max(1, 1000 // fps_hint)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb
                    )
                    result = landmarker.detect_for_video(mp_image, frame_ts_ms)

                    rendered = preview.render(frame, result)
                    bridge.sig_preview_frame.emit(rendered)

                    if not result.face_landmarks:
                        if face_was_visible:
                            face_was_visible = False
                            tracker._smoother.reset()
                            tracker._blink.reset()
                            tracker._fire(tracker.on_face_lost)
                        continue

                    if not face_was_visible:
                        face_was_visible = True
                        tracker._fire(tracker.on_face_found)

                    lms = result.face_landmarks[0]
                    raw_x, raw_y       = raw_gaze_point(lms)
                    screen_x, screen_y = tracker._calibration.map(raw_x, raw_y)
                    sx, sy             = tracker._smoother.update(screen_x, screen_y)

                    sx = max(SCREEN_MARGIN_PX,
                             min(tracker._screen_w - SCREEN_MARGIN_PX, sx))
                    sy = max(SCREEN_MARGIN_PX,
                             min(tracker._screen_h - SCREEN_MARGIN_PX, sy))

                    if autopy_mod is not None:
                        try:
                            autopy_mod.mouse.move(sx, sy)
                        except Exception:
                            pass

                    tracker._fire(tracker.on_move, sx, sy)

                    with tracker._lock:
                        tracker._blink.update(lms, sx, sy)

            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception(exc)
                tracker._fire(tracker.on_error, exc)
            finally:
                landmarker.close()
                cap.release()

        tracker._tracking_loop = patched_loop  # type: ignore[method-assign]

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

        import cv2
        import mediapipe as mp

        cap        = self._tracker._open_camera()
        landmarker = self._tracker._build_landmarker(mp, mode="video")

        self.sig_status.emit("Running calibration …")

        def frame_cb(annotated_frame: np.ndarray) -> None:
            self.sig_preview_frame.emit(annotated_frame)

        try:
            ok = self._tracker._calibration.run(
                landmarker, cap, frame_callback=frame_cb
            )
        finally:
            landmarker.close()
            cap.release()
            self.sig_cal_state.emit("")

        if ok:
            self.sig_status.emit("Calibration complete ✓")
        else:
            self.sig_status.emit("Calibration aborted")

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

    # ── Internal callbacks (tracker thread) ──────────────────────────

    def _on_tracker_move(self, x: float, y: float) -> None:
        self.sig_move.emit(x, y)

    def _on_face_lost(self) -> None:
        self.sig_face_lost.emit()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  WIDGETS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _panel(parent: Optional[QWidget] = None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("panel")
    return f


def _label(text: str, obj_name: str = "",
           parent: Optional[QWidget] = None) -> QLabel:
    lbl = QLabel(text, parent)
    if obj_name:
        lbl.setObjectName(obj_name)
    return lbl


class StatusDot(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._color = QColor(DARK["text_muted"])

    def set_color(self, hex_color: str) -> None:
        self._color = QColor(hex_color)
        self.update()

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
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(4)
        self._title = _label(title.upper(), "section_title")
        self._value = _label(initial, "value_large")
        lay.addWidget(self._title)
        lay.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class PreviewPanel(QFrame):
    PREVIEW_W = 320
    PREVIEW_H = 240

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFixedSize(self.PREVIEW_W, self.PREVIEW_H + 36)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet(
            f"background: {DARK['surface2']};"
            f"border-bottom: 1px solid {DARK['border']};"
            "border-radius: 0px;"
        )
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(10, 0, 10, 0)
        self._header_label = QLabel("CAMERA PREVIEW")
        self._header_label.setObjectName("section_title")
        self._cal_badge = QLabel("")
        self._cal_badge.setStyleSheet("font-size: 10px; font-weight: 600;")
        h_lay.addWidget(self._header_label)
        h_lay.addStretch()
        h_lay.addWidget(self._cal_badge)
        lay.addWidget(header)

        # Image label
        self._image_label = QLabel()
        self._image_label.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        self._image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self._image_label.setStyleSheet(f"background: {DARK['bg']};")
        lay.addWidget(self._image_label)

        self._show_placeholder()

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        import cv2
        resized = cv2.resize(bgr_frame, (self.PREVIEW_W, self.PREVIEW_H))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data.tobytes(), w, h, ch * w,
                     QImage.Format.Format_RGB888)
        self._image_label.setPixmap(QPixmap.fromImage(img))

    def set_cal_state(self, state: str) -> None:
        labels: dict[str, tuple[str, str]] = {
            "waiting":  ("Hold still …",  DARK["amber"]),
            "stable":   ("Gaze stable",   DARK["green"]),
            "sampling": ("Sampling …",    DARK["accent"]),
            "":         ("",              ""),
        }
        text, color = labels.get(state, ("", ""))
        self._cal_badge.setText(text)
        self._cal_badge.setStyleSheet(
            f"font-size: 10px; font-weight: 600; color: {color};"
            if color else ""
        )

    def _show_placeholder(self) -> None:
        import cv2
        ph = np.zeros((self.PREVIEW_H, self.PREVIEW_W, 3), dtype=np.uint8)
        ph[:] = (20, 22, 30)
        msg = "Preview off"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            ph, msg,
            (self.PREVIEW_W // 2 - tw // 2, self.PREVIEW_H // 2 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 75, 100), 1, cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data.tobytes(), w, h, ch * w,
                     QImage.Format.Format_RGB888)
        self._image_label.setPixmap(QPixmap.fromImage(img))


class EventLog(QWidget):
    MAX_LINES = 120

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._container = QWidget()
        self._container.setStyleSheet(
            f"background-color: {DARK['surface']};"
        )
        self._lay = QVBoxLayout(self._container)
        self._lay.setContentsMargins(12, 8, 12, 8)
        self._lay.setSpacing(1)
        self._lay.addStretch()
        self._scroll.setWidget(self._container)
        outer.addWidget(self._scroll)
        self._lines: list[QLabel] = []

    def append(self, text: str, color: str = "") -> None:
        ts  = QDateTime.currentDateTime().toString("hh:mm:ss.zzz")
        lbl = _label(f"{ts}  {text}", "log_text")
        if color:
            lbl.setStyleSheet(f"color: {color};")
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
        self.resize(1060, 700)
        self.setMinimumSize(900, 580)

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

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(33)   # ~30 fps
        self._frame_timer.timeout.connect(self._flush_preview_frame)
        self._frame_timer.start()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(16, 14, 16, 14)
        root_lay.setSpacing(12)
        root_lay.addWidget(self._make_header())
        body = QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(self._make_left_column(),  stretch=0)
        body.addLayout(self._make_right_column(), stretch=1)
        root_lay.addLayout(body)

    def _make_header(self) -> QFrame:
        frame = _panel()
        lay   = QHBoxLayout(frame)
        lay.setContentsMargins(16, 10, 16, 10)
        title = QLabel("Eye Module  ·  Test Bench")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {DARK['text']};"
        )
        lay.addWidget(title)
        lay.addStretch()
        self._dot = StatusDot()
        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet(
            f"color: {DARK['text_dim']}; font-size: 12px;"
        )
        lay.addWidget(self._dot)
        lay.addSpacing(6)
        lay.addWidget(self._status_label)
        return frame

    def _make_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(12)

        # Controls
        ctrl     = QGroupBox("Controls")
        ctrl_lay = QVBoxLayout(ctrl)
        ctrl_lay.setSpacing(8)
        ctrl_lay.setContentsMargins(12, 16, 12, 12)

        self._btn_start = QPushButton("Start Tracking")
        self._btn_start.setObjectName("btn_primary")
        self._btn_stop  = QPushButton("Stop Tracking")
        self._btn_stop.setObjectName("btn_danger")
        self._btn_stop.setEnabled(False)
        self._btn_calibrate = QPushButton("Run Calibration")

        ctrl_lay.addWidget(self._btn_start)
        ctrl_lay.addWidget(self._btn_stop)
        ctrl_lay.addWidget(self._btn_calibrate)

        # Options
        opts     = QGroupBox("Options")
        opts_lay = QVBoxLayout(opts)
        opts_lay.setSpacing(10)
        opts_lay.setContentsMargins(12, 16, 12, 12)

        # Preview toggle
        prev_row = QHBoxLayout()
        prev_lbl = QLabel("Camera Preview")
        prev_lbl.setStyleSheet(f"color: {DARK['text_dim']};")
        self._btn_preview = QPushButton("Off")
        self._btn_preview.setCheckable(True)
        self._btn_preview.setFixedWidth(56)
        self._btn_preview.setStyleSheet(
            f"QPushButton {{ color: {DARK['text_muted']}; }}"
            f"QPushButton:checked {{ color: {DARK['green']}; "
            f"border-color: {DARK['green']}; }}"
        )
        prev_row.addWidget(prev_lbl)
        prev_row.addStretch()
        prev_row.addWidget(self._btn_preview)
        opts_lay.addLayout(prev_row)
        self._div(opts_lay)

        # Dwell toggle
        dwell_row = QHBoxLayout()
        dwell_lbl = QLabel("Dwell-click")
        dwell_lbl.setStyleSheet(f"color: {DARK['text_dim']};")
        self._btn_dwell = QPushButton("Off")
        self._btn_dwell.setCheckable(True)
        self._btn_dwell.setFixedWidth(56)
        self._btn_dwell.setStyleSheet(
            f"QPushButton {{ color: {DARK['text_muted']}; }}"
            f"QPushButton:checked {{ color: {DARK['green']}; "
            f"border-color: {DARK['green']}; }}"
        )
        dwell_row.addWidget(dwell_lbl)
        dwell_row.addStretch()
        dwell_row.addWidget(self._btn_dwell)
        opts_lay.addLayout(dwell_row)
        self._div(opts_lay)

        # EMA alpha slider
        alpha_lbl = QLabel("Smoothing (EMA α)")
        alpha_lbl.setStyleSheet(f"color: {DARK['text_dim']};")
        alpha_row = QHBoxLayout()
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(5, 80)
        self._alpha_slider.setValue(25)
        self._alpha_label = QLabel("0.25")
        self._alpha_label.setFixedWidth(34)
        self._alpha_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._alpha_label.setStyleSheet(
            f"color: {DARK['accent']}; font-family: monospace;"
        )
        alpha_row.addWidget(self._alpha_slider)
        alpha_row.addWidget(self._alpha_label)
        opts_lay.addWidget(alpha_lbl)
        opts_lay.addLayout(alpha_row)

        col.addWidget(ctrl)
        col.addWidget(opts)
        col.addStretch()
        return col

    def _make_right_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        cards_col = QVBoxLayout()
        cards_col.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self._card_pos      = StatCard("Cursor Position", "—, —")
        self._card_face     = StatCard("Face", "Not detected")
        self._card_calib    = StatCard("Calibration", "None")
        row1.addWidget(self._card_pos)
        row1.addWidget(self._card_face)
        row1.addWidget(self._card_calib)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self._card_clicks   = StatCard("Blink Clicks", "0")
        self._card_dwell_c  = StatCard("Dwell Clicks", "0")
        self._card_calstate = StatCard("Cal. State", "—")
        row2.addWidget(self._card_clicks)
        row2.addWidget(self._card_dwell_c)
        row2.addWidget(self._card_calstate)

        cards_col.addLayout(row1)
        cards_col.addLayout(row2)
        top_row.addLayout(cards_col, stretch=1)

        self._preview_panel = PreviewPanel()
        top_row.addWidget(self._preview_panel, stretch=0)
        col.addLayout(top_row)

        log_group = QGroupBox("Event Log")
        log_lay   = QVBoxLayout(log_group)
        log_lay.setContentsMargins(4, 12, 4, 4)
        self._log = EventLog()
        self._log.setMinimumHeight(260)
        log_lay.addWidget(self._log)
        btn_clear = QPushButton("Clear log")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._clear_log)
        log_lay.addWidget(
            btn_clear, alignment=Qt.AlignmentFlag.AlignRight
        )
        col.addWidget(log_group)
        return col

    # ── Signal wiring ─────────────────────────────────────────────────

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
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_calibrate.setEnabled(False)
        self._pulse_timer.start()

    def _on_stop(self) -> None:
        self._log.append("■ Stopping …", DARK["text_dim"])
        self._bridge.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_calibrate.setEnabled(True)
        self._pulse_timer.stop()
        self._dot.set_color(DARK["text_muted"])
        self._card_face.set_value("Stopped")
        self._card_pos.set_value("—, —")
        self._card_calstate.set_value("—")

    def _on_calibrate(self) -> None:
        self._log.append("⊙ Starting calibration …", DARK["amber"])
        self._card_calstate.set_value("Waiting")
        threading.Thread(
            target=self._bridge.run_calibration, daemon=True
        ).start()

    def _on_preview_toggled(self, checked: bool) -> None:
        self._preview_on = checked
        self._btn_preview.setText("On" if checked else "Off")
        self._bridge.set_preview_enabled(checked)
        if not checked:
            self._preview_panel._show_placeholder()
        self._log.append(
            f"◉ Preview {'enabled' if checked else 'disabled'}",
            DARK["text_dim"],
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
        self._log.append(
            f"🖱  Blink-click  #{self._click_count}", DARK["accent"]
        )

    def _on_dwell_click(self) -> None:
        self._dwell_count += 1
        self._card_dwell_c.set_value(str(self._dwell_count))
        self._log.append(
            f"🕐  Dwell-click  #{self._dwell_count}", DARK["green"]
        )

    def _on_face_found(self) -> None:
        self._card_face.set_value("Detected ✓")
        self._log.append("● Face detected", DARK["green"])

    def _on_face_lost(self) -> None:
        self._card_face.set_value("Lost")
        self._card_pos.set_value("—, —")
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
            "waiting":  "Waiting",
            "stable":   "Stable ✓",
            "sampling": "Sampling …",
            "":         "—",
        }
        colors: dict[str, str] = {
            "waiting":  DARK["amber"],
            "stable":   DARK["green"],
            "sampling": DARK["accent"],
            "":         DARK["text_dim"],
        }
        self._card_calstate.set_value(labels.get(state, "—"))
        self._card_calstate._value.setStyleSheet(
            f"font-size: 20px; font-weight: 700; "
            f"color: {colors.get(state, DARK['text'])};"
        )
        self._preview_panel.set_cal_state(state)

    # ── Helpers ───────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = "") -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color or DARK['text_dim']}; font-size: 12px;"
        )

    def _pulse_dot(self) -> None:
        self._pulse_state = not self._pulse_state
        self._dot.set_color(
            DARK["accent"] if self._pulse_state else DARK["accent_dim"]
        )

    def _clear_log(self) -> None:
        for lbl in self._log._lines:
            lbl.deleteLater()
        self._log._lines.clear()

    def _div(self, layout: QVBoxLayout) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {DARK['border']}; margin: 2px 0;")
        layout.addWidget(line)

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        self._frame_timer.stop()
        self._bridge.stop()
        if hasattr(event, "accept"):
            event.accept()  # type: ignore[union-attr]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main() -> None:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    palette = app.palette()
    cr = QPalette.ColorRole
    palette.setColor(cr.Window,          QColor(DARK["bg"]))
    palette.setColor(cr.WindowText,      QColor(DARK["text"]))
    palette.setColor(cr.Base,            QColor(DARK["surface"]))
    palette.setColor(cr.AlternateBase,   QColor(DARK["surface2"]))
    palette.setColor(cr.Text,            QColor(DARK["text"]))
    palette.setColor(cr.Button,          QColor(DARK["surface2"]))
    palette.setColor(cr.ButtonText,      QColor(DARK["text"]))
    palette.setColor(cr.Highlight,       QColor(DARK["accent"]))
    palette.setColor(cr.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
