"""
calibration.py
──────────────
5-point gaze calibration that maps normalised iris coordinates (camera
space) to actual screen coordinates using an affine transform computed
from the collected samples.

Workflow
────────
1.  ``CalibrationManager.run(landmarker, cap)``
        Displays each target point via OpenCV, collects
        CALIBRATION_SAMPLES_PER_POINT iris samples per point,
        fits the mapping, and saves the result to JSON.

2.  ``CalibrationManager.load()``
        Reads the saved JSON and rebuilds the transform matrix.
        Called at startup if a calibration file already exists.

3.  ``CalibrationManager.map(iris_x, iris_y) → (screen_x, screen_y)``
        Applies the affine transform to a raw iris point.

Affine-transform approach
─────────────────────────
We solve  screen_pt = M · [iris_x, iris_y, 1]ᵀ  in a least-squares
sense using numpy.  This handles scale, rotation, shear, and
translation robustly without requiring the target points to be a
perfect rectangle.

If no calibration is loaded we fall back to a simple proportional
mapping scaled to the screen.

Stability gate (bug-fix)
────────────────────────
Samples are only accepted when the iris position has been stable
(moved less than CALIBRATION_STABILITY_RADIUS normalised units) for
at least CALIBRATION_STABILITY_MIN_FRAMES consecutive frames.  This
prevents the calibration from advancing when the user is not actually
looking at the target point.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from eye_module.config import (
    CALIBRATION_PATH,
    CALIBRATION_SAMPLES_PER_POINT,
    CALIBRATION_SETTLE_MS,
    CALIBRATION_STABILITY_MIN_FRAMES,
    CALIBRATION_STABILITY_RADIUS,
    CALIBRATION_TARGETS,
    IRIS_LEFT_INDICES,
    IRIS_RIGHT_INDICES,
)

log = logging.getLogger(__name__)


# ── Iris helpers ──────────────────────────────────────────────────────────────

def _iris_centre(landmarks, indices: List[int]) -> Tuple[float, float]:
    """Return the centroid of the given landmark indices."""
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def raw_gaze_point(landmarks) -> Tuple[float, float]:
    """
    Average of both iris centres in normalised camera space [0, 1].
    Exported so calibration and the tracker share the same computation.
    """
    rx, ry = _iris_centre(landmarks, IRIS_RIGHT_INDICES)
    lx, ly = _iris_centre(landmarks, IRIS_LEFT_INDICES)
    return (rx + lx) / 2.0, (ry + ly) / 2.0


def _iris_delta(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two normalised iris positions."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── CalibrationManager ────────────────────────────────────────────────────────

class CalibrationManager:
    """
    Manages recording, persisting, and applying the gaze→screen mapping.

    Parameters
    ----------
    screen_w, screen_h : int
        Screen resolution in pixels.
    calibration_path : str | Path
        File path for the JSON calibration data.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        calibration_path: str | Path = CALIBRATION_PATH,
    ) -> None:
        self._sw   = screen_w
        self._sh   = screen_h
        self._path = Path(calibration_path)

        # 2×3 affine matrix (numpy float64) or None if not calibrated
        self._matrix: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self._matrix is not None

    def load(self) -> bool:
        """
        Load calibration from JSON file.

        Returns
        -------
        bool
            ``True`` if loaded successfully, ``False`` otherwise.
        """
        if not self._path.exists():
            log.info("No calibration file found at %s", self._path)
            return False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            mat  = np.array(data["matrix"], dtype=np.float64)
            if mat.shape != (2, 3):
                raise ValueError(f"Expected (2,3) matrix, got {mat.shape}")
            self._matrix = mat
            log.info("Calibration loaded from %s", self._path)
            return True
        except Exception as exc:
            log.warning("Failed to load calibration: %s", exc)
            self._matrix = None
            return False

    def save(self) -> None:
        """Persist the current calibration matrix to JSON."""
        if self._matrix is None:
            raise RuntimeError("No calibration to save.")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"matrix": self._matrix.tolist(), "screen": [self._sw, self._sh]}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Calibration saved to %s", self._path)

    def map(self, iris_x: float, iris_y: float) -> Tuple[float, float]:
        """
        Map a normalised iris point to screen coordinates.

        Falls back to a proportional mapping when not calibrated.
        """
        if self._matrix is None:
            # Fallback: simple proportion — mirrors the camera horizontally
            return (1.0 - iris_x) * self._sw, iris_y * self._sh

        pt = np.array([iris_x, iris_y, 1.0], dtype=np.float64)
        result = self._matrix @ pt
        return float(result[0]), float(result[1])

    def run(
        self,
        landmarker,
        cap: cv2.VideoCapture,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> bool:
        """
        Interactive calibration routine.

        Displays each target on a fullscreen OpenCV window, collects
        CALIBRATION_SAMPLES_PER_POINT iris samples per point (only
        once the gaze is confirmed stable on the target), fits the
        affine transform, and saves to disk.

        Parameters
        ----------
        landmarker
            A live ``mediapipe.tasks.vision.FaceLandmarker`` instance
            (VIDEO mode).
        cap
            An open ``cv2.VideoCapture`` instance.
        frame_callback : callable(np.ndarray) | None
            Optional.  Called on every processed frame with the annotated
            BGR frame (camera feed + target dot overlay).  Used by the
            PreviewModule to display the live feed during calibration.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if aborted (press ``Esc``).
        """
        log.info("Starting calibration …")
        iris_pts:   List[Tuple[float, float]] = []
        screen_pts: List[Tuple[float, float]] = []

        # ── Fullscreen calibration window ─────────────────────────────────────
        win = "Eye Tracker – Calibration"
        cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        import mediapipe as mp

        # Strictly-increasing ms timestamp for MediaPipe VIDEO mode
        frame_ts_ms: int = 0
        fps_hint:    int = int(cap.get(cv2.CAP_PROP_FPS) or 30)

        for point_idx, target_norm in enumerate(CALIBRATION_TARGETS):
            tx = int(target_norm[0] * self._sw)
            ty = int(target_norm[1] * self._sh)

            log.debug(
                "Calibration point %d/%d → screen (%d, %d)",
                point_idx + 1, len(CALIBRATION_TARGETS), tx, ty,
            )

            # ── Settle phase ──────────────────────────────────────────────────
            # Show the target, flush stale camera frames, do NOT collect samples.
            settle_end = time.monotonic() + CALIBRATION_SETTLE_MS / 1000.0
            while time.monotonic() < settle_end:
                canvas = self._draw_target(tx, ty, phase="look")
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) == 27:
                    cv2.destroyWindow(win)
                    return False
                ret, raw = cap.read()
                if ret and frame_callback is not None:
                    annotated = self._annotate_frame(raw, tx, ty, phase="look")
                    frame_callback(annotated)

            # ── Stability gate + sample phase ─────────────────────────────────
            # We only accept a sample when the iris has been still for
            # CALIBRATION_STABILITY_MIN_FRAMES consecutive frames.  If
            # the user looks away, the stable_streak resets and the
            # progress ring stops advancing — giving clear visual feedback.
            samples:       List[Tuple[float, float]] = []
            prev_gaze:     Optional[Tuple[float, float]] = None
            stable_streak: int = 0

            while len(samples) < CALIBRATION_SAMPLES_PER_POINT:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Advance timestamp by one frame-worth of milliseconds
                frame_ts_ms += max(1, 1000 // fps_hint)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                result = landmarker.detect_for_video(mp_image, frame_ts_ms)

                if result.face_landmarks:
                    lms = result.face_landmarks[0]
                    gaze = raw_gaze_point(lms)

                    if prev_gaze is None:
                        # First frame — seed the stability check
                        stable_streak = 1
                    elif _iris_delta(gaze, prev_gaze) <= CALIBRATION_STABILITY_RADIUS:
                        stable_streak += 1
                    else:
                        # Gaze drifted — restart streak, do not collect
                        stable_streak = 0
                        log.debug(
                            "Gaze unstable at point %d (delta=%.4f) — resetting streak",
                            point_idx + 1,
                            _iris_delta(gaze, prev_gaze),
                        )

                    prev_gaze = gaze

                    # Only collect once gaze is confirmed stable
                    if stable_streak >= CALIBRATION_STABILITY_MIN_FRAMES:
                        samples.append(gaze)
                else:
                    # Face lost — reset streak, do not advance
                    stable_streak = 0
                    prev_gaze = None

                # ── Visual feedback ───────────────────────────────────────────
                # Progress ring only fills when stable_streak is met and
                # samples are actually being collected.
                collecting  = stable_streak >= CALIBRATION_STABILITY_MIN_FRAMES
                progress    = len(samples) / CALIBRATION_SAMPLES_PER_POINT
                phase_label = "sample" if collecting else "waiting"

                canvas = self._draw_target(
                    tx, ty,
                    phase=phase_label,
                    progress=progress,
                    stable=collecting,
                )
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) == 27:
                    cv2.destroyWindow(win)
                    return False

                # Feed annotated preview frame to the callback
                if frame_callback is not None:
                    annotated = self._annotate_frame(
                        frame, tx, ty,
                        phase=phase_label,
                        progress=progress,
                    )
                    frame_callback(annotated)

            # ── Average the confirmed samples for this point ──────────────────
            avg_x = sum(s[0] for s in samples) / len(samples)
            avg_y = sum(s[1] for s in samples) / len(samples)
            iris_pts.append((avg_x, avg_y))
            screen_pts.append((float(tx), float(ty)))

            log.debug(
                "Point %d/%d done → iris avg (%.4f, %.4f)",
                point_idx + 1, len(CALIBRATION_TARGETS), avg_x, avg_y,
            )

        cv2.destroyWindow(win)

        # ── Fit affine transform ──────────────────────────────────────────────
        src = np.array(iris_pts,   dtype=np.float64)   # (N, 2)
        dst = np.array(screen_pts, dtype=np.float64)   # (N, 2)

        # Build homogeneous source: [x, y, 1] rows
        A = np.hstack([src, np.ones((len(src), 1))])   # (N, 3)

        # Least-squares: solve A @ M.T = dst  →  M is (2, 3)
        M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
        self._matrix = M.T

        log.info("Calibration complete. Matrix:\n%s", self._matrix)
        self.save()
        return True

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_target(
        self,
        tx: int,
        ty: int,
        phase: str   = "look",
        progress: float = 0.0,
        stable: bool = False,
    ) -> np.ndarray:
        """Render the calibration target on the fullscreen dark canvas."""
        canvas = np.zeros((self._sh, self._sw, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        outer_r = 24
        inner_r = 8

        if phase == "look":
            instruction = "Look at the dot and hold still"
            cv2.circle(canvas, (tx, ty), outer_r, (80, 180, 80), 2)
            cv2.circle(canvas, (tx, ty), inner_r, (80, 255, 80), -1)

        elif phase == "waiting":
            # Stable streak not yet reached — pulse amber to signal "hold still"
            instruction = "Hold your gaze steady on the dot …"
            cv2.circle(canvas, (tx, ty), outer_r, (180, 140, 0), 2)
            cv2.circle(canvas, (tx, ty), inner_r, (220, 180, 0), -1)

        else:  # "sample"
            instruction = "Great — keep looking …"
            cv2.circle(canvas, (tx, ty), outer_r, (60, 60, 200), 2)
            cv2.circle(canvas, (tx, ty), inner_r, (100, 100, 255), -1)
            # Progress arc
            angle = int(progress * 360)
            cv2.ellipse(
                canvas, (tx, ty), (outer_r, outer_r),
                -90, 0, angle, (0, 220, 255), 3,
            )

        # Point counter
        cv2.putText(
            canvas, instruction,
            (self._sw // 2 - 220, self._sh - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA,
        )
        return canvas

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tx: int,
        ty: int,
        phase: str   = "look",
        progress: float = 0.0,
    ) -> np.ndarray:
        """
        Draw the calibration target dot onto a camera frame (for the
        preview panel).  Returns an annotated copy; does not modify in place.
        """
        out = cv2.flip(frame.copy(), 1)  # mirror so it feels natural

        # Scale target position to frame size
        fh, fw = out.shape[:2]
        px = int(tx * fw / self._sw)
        py = int(ty * fh / self._sh)

        if phase == "look":
            color = (80, 255, 80)
        elif phase == "waiting":
            color = (0, 180, 220)
        else:
            color = (255, 100, 100)

        cv2.circle(out, (px, py), 16, color, 2)
        cv2.circle(out, (px, py), 5,  color, -1)

        if phase == "sample" and progress > 0:
            angle = int(progress * 360)
            cv2.ellipse(out, (px, py), (16, 16), -90, 0, angle, (0, 220, 255), 2)

        return out
