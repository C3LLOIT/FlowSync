"""
preview.py
──────────
PreviewModule — live camera feed with landmark overlays.

Purpose
───────
Provides a rendered numpy frame (BGR) of the camera feed annotated with
tracking overlays.  Designed to be embedded into a PySide6 panel via a
QLabel + QPixmap, but has no Qt dependency itself.

Two modes
─────────
``"eye"``
    Overlays:
    - Face bounding box  (green = tracked, amber = partial, red = lost)
    - Left and right iris centre dots

``"hand"``
    Overlays:
    - Per-hand bounding box
    - 21 landmark dots
    - Finger connection lines (matching MediaPipe's HAND_CONNECTIONS)

Usage
─────
    preview = PreviewModule(mode="eye")
    preview.enabled = True        # off by default

    # Feed a raw frame + detection result every loop iteration:
    frame_bgr = preview.render(raw_frame, detection_result)

    # Convert to QPixmap for display in a QLabel:
    from preview import bgr_to_qpixmap
    pixmap = bgr_to_qpixmap(frame_bgr)
    label.setPixmap(pixmap)

Calibration
───────────
Call ``preview.set_calibration_target(tx, ty, phase, progress)`` each
frame during calibration so the target dot is drawn on the live feed.
Call ``preview.clear_calibration_target()`` when calibration ends.

Thread safety
─────────────
``render()`` is called from the tracking thread.  The returned numpy
array is a fresh copy each call — safe to pass to the Qt main thread
via a signal without locking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ── Colour palette (BGR) ──────────────────────────────────────────────────────
_GREEN  = (72,  178, 55)    # face tracked / hand detected
_AMBER  = (0,   159, 245)   # partial detection
_RED    = (60,  62,  240)   # face lost
_BLUE   = (250, 124, 92)    # iris dots
_WHITE  = (230, 234, 240)   # landmark dots
_DIM    = (120, 130, 165)   # connection lines
_ACCENT = (250, 124, 92)    # calibration target

# MediaPipe hand connections — (start_idx, end_idx) pairs
_HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle bar
    (5, 9), (9, 13), (13, 17),
]

# Iris landmark indices (MediaPipe 478-point face mesh)
_IRIS_LEFT_INDICES  = [474, 475, 476, 477]
_IRIS_RIGHT_INDICES = [469, 470, 471, 472]


class PreviewModule:
    """
    Renders an annotated camera preview frame.

    Parameters
    ----------
    mode : str
        ``"eye"`` or ``"hand"``.
    mirror : bool
        Flip the frame horizontally so it feels like a mirror.
        Default ``True``.
    """

    def __init__(self, mode: str = "eye", mirror: bool = True) -> None:
        if mode not in ("eye", "hand"):
            raise ValueError(f"mode must be 'eye' or 'hand', got {mode!r}")

        self._mode   = mode
        self._mirror = mirror

        # Off by default — enabled from the settings page
        self._enabled = False

        # Calibration overlay state
        self._cal_target: Optional[Tuple[int, int, str, float]] = None
        # (screen_tx, screen_ty, phase, progress)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def mode(self) -> str:
        return self._mode

    def set_calibration_target(
        self,
        screen_tx: int,
        screen_ty: int,
        screen_w: int,
        screen_h: int,
        phase: str = "look",
        progress: float = 0.0,
    ) -> None:
        """
        Tell the preview where the current calibration target is so it
        can draw the dot on the live feed.

        Parameters
        ----------
        screen_tx, screen_ty : int
            Target position in screen pixels.
        screen_w, screen_h : int
            Full screen resolution (used to scale to frame coordinates).
        phase : str
            ``"look"`` | ``"waiting"`` | ``"sample"``
        progress : float
            0.0 – 1.0 fill for the progress arc.
        """
        self._cal_screen_size = (screen_w, screen_h)
        self._cal_target = (screen_tx, screen_ty, phase, progress)

    def clear_calibration_target(self) -> None:
        """Remove the calibration dot overlay."""
        self._cal_target = None

    def render(
        self,
        raw_frame: np.ndarray,
        detection_result=None,
    ) -> np.ndarray:
        """
        Build and return the annotated preview frame.

        Parameters
        ----------
        raw_frame : np.ndarray
            BGR frame directly from cv2.VideoCapture.
        detection_result
            MediaPipe detection result object, or ``None`` when
            no detection has been run yet.

            - Eye mode  : ``FaceLandmarkerResult``
            - Hand mode : ``HandLandmarkerResult``

        Returns
        -------
        np.ndarray
            Annotated BGR frame, same dimensions as ``raw_frame``.
            Always a fresh copy — safe to hand off to another thread.
        """
        if not self._enabled:
            # Return a minimal placeholder so the caller always gets a frame
            return self._placeholder(raw_frame)

        frame = raw_frame.copy()
        if self._mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        if self._mode == "eye":
            self._draw_eye_overlays(frame, detection_result, w, h)
        else:
            self._draw_hand_overlays(frame, detection_result, w, h)

        # Calibration target dot on top of everything else
        if self._cal_target is not None:
            self._draw_cal_target(frame, w, h)

        # Status badge (top-left corner)
        self._draw_status_badge(frame, detection_result, w, h)

        return frame

    # ── Eye overlays ──────────────────────────────────────────────────────────

    def _draw_eye_overlays(
        self,
        frame: np.ndarray,
        result,
        w: int,
        h: int,
    ) -> None:
        """Draw face bounding box and iris dots."""

        if result is None or not getattr(result, "face_landmarks", None):
            # No face — draw a "lost" indicator border
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), _RED, 2)
            return

        lms = result.face_landmarks[0]

        # ── Face bounding box ─────────────────────────────────────────────────
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        # Pad slightly
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Choose colour based on whether we have iris landmarks (indices 469–477)
        has_iris = len(lms) > 477
        bbox_color = _GREEN if has_iris else _AMBER
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

        # Corner accents
        corner_len = 12
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), bbox_color, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), bbox_color, 2)

        if not has_iris:
            return

        # ── Iris dots ─────────────────────────────────────────────────────────
        for indices in (_IRIS_LEFT_INDICES, _IRIS_RIGHT_INDICES):
            ix = int(sum(lms[i].x for i in indices) / len(indices) * w)
            iy = int(sum(lms[i].y for i in indices) / len(indices) * h)

            # Mirror x if we flipped the frame
            if self._mirror:
                ix = w - ix

            cv2.circle(frame, (ix, iy), 6, _BLUE, -1)
            cv2.circle(frame, (ix, iy), 9, _BLUE, 1)

    # ── Hand overlays ─────────────────────────────────────────────────────────

    def _draw_hand_overlays(
        self,
        frame: np.ndarray,
        result,
        w: int,
        h: int,
    ) -> None:
        """Draw hand bounding boxes, 21 landmark dots, and connections."""

        if result is None or not getattr(result, "hand_landmarks", None):
            cv2.rectangle(frame, (4, 4), (w - 4, h - 4), _RED, 2)
            return

        for hand_lms in result.hand_landmarks:
            pts = []
            for lm in hand_lms:
                px = int(lm.x * w)
                py = int(lm.y * h)
                if self._mirror:
                    px = w - px
                pts.append((px, py))

            # ── Connection lines ──────────────────────────────────────────────
            for start_idx, end_idx in _HAND_CONNECTIONS:
                if start_idx < len(pts) and end_idx < len(pts):
                    cv2.line(frame, pts[start_idx], pts[end_idx], _DIM, 1, cv2.LINE_AA)

            # ── Landmark dots ─────────────────────────────────────────────────
            for idx, (px, py) in enumerate(pts):
                # Fingertips (4, 8, 12, 16, 20) slightly larger
                r = 5 if idx in (4, 8, 12, 16, 20) else 3
                cv2.circle(frame, (px, py), r, _WHITE, -1)

            # ── Bounding box ──────────────────────────────────────────────────
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                pad = 14
                x1 = max(0, min(xs) - pad)
                y1 = max(0, min(ys) - pad)
                x2 = min(w, max(xs) + pad)
                y2 = min(h, max(ys) + pad)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 1)

    # ── Calibration target overlay ────────────────────────────────────────────

    def _draw_cal_target(
        self,
        frame: np.ndarray,
        fw: int,
        fh: int,
    ) -> None:
        """Draw the calibration dot scaled to the frame dimensions."""
        tx, ty, phase, progress = self._cal_target
        sw, sh = getattr(self, "_cal_screen_size", (fw, fh))

        # Scale from screen → frame coordinates
        px = int(tx * fw / sw)
        py = int(ty * fh / sh)

        if phase == "look":
            color = (80, 255, 80)
        elif phase == "waiting":
            color = (0, 200, 255)
        else:
            color = (250, 124, 92)

        cv2.circle(frame, (px, py), 18, color, 2)
        cv2.circle(frame, (px, py),  6, color, -1)

        if phase == "sample" and progress > 0:
            angle = int(progress * 360)
            cv2.ellipse(
                frame, (px, py), (18, 18),
                -90, 0, angle, (0, 220, 255), 2,
            )

    # ── Status badge ──────────────────────────────────────────────────────────

    def _draw_status_badge(
        self,
        frame: np.ndarray,
        result,
        w: int,
        h: int,
    ) -> None:
        """Small top-left status pill: tracking state + mode."""
        if self._mode == "eye":
            detected = (
                result is not None
                and getattr(result, "face_landmarks", None)
            )
        else:
            detected = (
                result is not None
                and getattr(result, "hand_landmarks", None)
            )

        label  = f"{'● ' if detected else '○ '}{self._mode.upper()}"
        color  = _GREEN if detected else _AMBER

        # Background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pad = 6
        cv2.rectangle(
            frame,
            (8, 8),
            (8 + tw + pad * 2, 8 + th + pad * 2),
            (20, 20, 30), -1,
        )
        cv2.putText(
            frame, label,
            (8 + pad, 8 + th + pad),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    # ── Placeholder (preview disabled) ───────────────────────────────────────

    def _placeholder(self, raw_frame: np.ndarray) -> np.ndarray:
        """
        When preview is disabled, return a dark frame with a centred label.
        This keeps the panel a consistent size in the UI.
        """
        h, w = raw_frame.shape[:2]
        out  = np.zeros((h, w, 3), dtype=np.uint8)
        out[:] = (20, 22, 30)

        msg = "Preview off"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            out, msg,
            (w // 2 - tw // 2, h // 2 + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 75, 100), 1, cv2.LINE_AA,
        )
        return out


# ── Qt conversion helper ──────────────────────────────────────────────────────

def bgr_to_qpixmap(frame: np.ndarray):
    """
    Convert a BGR numpy frame to a ``QPixmap`` for display in a QLabel.

    Import is deferred so this module has no hard PySide6 dependency —
    it works as plain OpenCV even when Qt is not installed.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame from ``PreviewModule.render()``.

    Returns
    -------
    QPixmap
    """
    from PySide6.QtGui import QImage, QPixmap

    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img   = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)
