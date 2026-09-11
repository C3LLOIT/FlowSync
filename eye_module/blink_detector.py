"""
blink_detector.py
─────────────────
Detects two kinds of intentional click gestures:

1. **Blink-click**
   Both eyes closed (EAR < threshold) for ≥ BLINK_HOLD_MS milliseconds
   → fires a left-click and resets the timer.

2. **Dwell-click**  (opt-in, disabled by default)
   Gaze remains within a radius of DWELL_RADIUS_PX pixels for
   ≥ DWELL_HOLD_MS milliseconds without a deliberate eye-close event
   → fires a left-click and resets the dwell counter.

Eye-Aspect-Ratio (EAR)
───────────────────────
    EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)

where p1..p6 are the six key-points around one eye in the order:
  p1=left-corner, p4=right-corner,
  p2/p6=upper/lower outer, p3/p5=upper/lower middle.

An open eye typically has EAR ≈ 0.28–0.35.
EAR < EAR_CLOSE_THRESHOLD (≈ 0.20) → eye is closed.

Both eyes must close simultaneously to trigger a blink-click (reduces
false positives from natural unilateral winks or tracking noise).
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from eye_module.config import (
    BLINK_HOLD_MS,
    DWELL_ENABLED_DEFAULT,
    DWELL_HOLD_MS,
    DWELL_RADIUS_PX,
    EAR_CLOSE_THRESHOLD,
    EAR_LEFT_HORIZONTAL,
    EAR_LEFT_VERTICAL_PAIRS,
    EAR_OPEN_HYSTERESIS,
    EAR_RIGHT_HORIZONTAL,
    EAR_RIGHT_VERTICAL_PAIRS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _landmark_xy(landmarks, index: int) -> Tuple[float, float]:
    """
    Extract the (x, y) of a MediaPipe NormalizedLandmark by index.
    Returns floats in [0, 1] (normalised image coordinates).
    """
    lm = landmarks[index]
    return lm.x, lm.y


def compute_ear(
    landmarks,
    vertical_pairs: List[Tuple[int, int]],
    horizontal: Tuple[int, int],
) -> float:
    """
    Compute the Eye-Aspect-Ratio for one eye.

    Parameters
    ----------
    landmarks
        Full list of MediaPipe NormalizedLandmark objects for one face.
    vertical_pairs
        List of (upper_idx, lower_idx) index pairs for the vertical
        measurements of this eye.
    horizontal
        (left_corner_idx, right_corner_idx) for the horizontal measurement.
    """
    v_sum = sum(
        _dist(_landmark_xy(landmarks, u), _landmark_xy(landmarks, l))
        for u, l in vertical_pairs
    )
    h = _dist(
        _landmark_xy(landmarks, horizontal[0]),
        _landmark_xy(landmarks, horizontal[1]),
    )
    if h < 1e-6:
        return 0.0
    return v_sum / (2.0 * h)


# ── Main class ────────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    Stateful blink-click and dwell-click detector.

    Usage
    -----
    Instantiate once, then call :meth:`update` on every frame.
    Register callables for :attr:`on_blink_click` and
    :attr:`on_dwell_click` to receive notifications.

    Parameters
    ----------
    dwell_enabled : bool
        Whether dwell-click starts active.  The main application can
        toggle this at runtime via :attr:`dwell_enabled`.
    """

    def __init__(self, dwell_enabled: bool = DWELL_ENABLED_DEFAULT) -> None:
        # ── Callbacks ──
        self.on_blink_click: Optional[callable] = None  # () -> None
        self.on_dwell_click: Optional[callable] = None  # () -> None

        # ── Blink-click state ──
        self._dwell_enabled   = dwell_enabled
        self._eyes_closed     = False          # True while EAR < threshold
        self._close_start_ts: Optional[float] = None  # time.monotonic()

        # ── Dwell-click state ──
        self._dwell_anchor: Optional[Tuple[float, float]] = None
        self._dwell_start_ts: Optional[float] = None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def dwell_enabled(self) -> bool:
        return self._dwell_enabled

    @dwell_enabled.setter
    def dwell_enabled(self, value: bool) -> None:
        self._dwell_enabled = value
        if not value:
            self._reset_dwell()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        landmarks,
        screen_x: float,
        screen_y: float,
    ) -> None:
        """
        Process one frame.

        Parameters
        ----------
        landmarks
            The landmark list from MediaPipe
            (``result.face_landmarks[0]``).
        screen_x, screen_y : float
            Current smoothed cursor position in screen pixels.
        """
        ear_right = compute_ear(
            landmarks,
            EAR_RIGHT_VERTICAL_PAIRS,
            EAR_RIGHT_HORIZONTAL,
        )
        ear_left = compute_ear(
            landmarks,
            EAR_LEFT_VERTICAL_PAIRS,
            EAR_LEFT_HORIZONTAL,
        )
        ear_avg = (ear_right + ear_left) / 2.0

        self._process_blink(ear_avg)
        if self._dwell_enabled and not self._eyes_closed:
            self._process_dwell(screen_x, screen_y)

    def reset(self) -> None:
        """Reset all state (call when face tracking is lost)."""
        self._eyes_closed   = False
        self._close_start_ts = None
        self._reset_dwell()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process_blink(self, ear: float) -> None:
        now = time.monotonic()

        if not self._eyes_closed:
            # Transition: open → closed
            if ear < EAR_CLOSE_THRESHOLD:
                self._eyes_closed    = True
                self._close_start_ts = now
        else:
            # Already closed — check hold duration
            if self._close_start_ts is not None:
                held_ms = (now - self._close_start_ts) * 1000.0
                if held_ms >= BLINK_HOLD_MS:
                    self._fire_blink_click()
                    self._close_start_ts = None  # prevent re-trigger until re-open

            # Transition: closed → open (with hysteresis)
            if ear >= EAR_OPEN_HYSTERESIS:
                self._eyes_closed    = False
                self._close_start_ts = None

    def _fire_blink_click(self) -> None:
        if callable(self.on_blink_click):
            try:
                self.on_blink_click()
            except Exception:
                pass  # Never let a callback crash the tracker thread

    def _process_dwell(self, sx: float, sy: float) -> None:
        now = time.monotonic()

        if self._dwell_anchor is None:
            # First dwell sample
            self._dwell_anchor   = (sx, sy)
            self._dwell_start_ts = now
            return

        dist = _dist((sx, sy), self._dwell_anchor)

        if dist > DWELL_RADIUS_PX:
            # Gaze moved — restart dwell
            self._dwell_anchor   = (sx, sy)
            self._dwell_start_ts = now
        else:
            held_ms = (now - self._dwell_start_ts) * 1000.0
            if held_ms >= DWELL_HOLD_MS:
                self._fire_dwell_click()
                # Move anchor to current pos so next dwell needs a fresh move
                self._dwell_anchor   = (sx, sy)
                self._dwell_start_ts = now

    def _fire_dwell_click(self) -> None:
        if callable(self.on_dwell_click):
            try:
                self.on_dwell_click()
            except Exception:
                pass

    def _reset_dwell(self) -> None:
        self._dwell_anchor   = None
        self._dwell_start_ts = None
