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
    EAR_LEFT_HORIZONTAL,
    EAR_RIGHT_HORIZONTAL,
    GAZE_IRIS_RANGE_MAX,
    GAZE_IRIS_RANGE_MIN,
    GAZE_LEFT_EYELID_BOTTOM,
    GAZE_LEFT_EYELID_TOP,
    GAZE_MIN_EYE_HEIGHT,
    GAZE_MIN_EYE_WIDTH,
    GAZE_RIGHT_EYELID_BOTTOM,
    GAZE_RIGHT_EYELID_TOP,
    IRIS_LEFT_INDICES,
    IRIS_RIGHT_INDICES,
)

log = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 — GAZE ACCURACY                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── Data classes ──────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class EyeBounds:
    """
    Bounding box of one eye socket in normalised image space.

    x_left  — outer (temporal) corner x
    x_right — inner (nasal) corner x
    y_top   — average of upper eyelid landmarks
    y_bottom— average of lower eyelid landmarks
    width   — horizontal extent (x_right - x_left, always positive)
    height  — vertical extent  (y_bottom - y_top,  always positive)
    """
    x_left:   float
    x_right:  float
    y_top:    float
    y_bottom: float

    @property
    def width(self) -> float:
        return abs(self.x_right - self.x_left)

    @property
    def height(self) -> float:
        return abs(self.y_bottom - self.y_top)


@dataclass
class GazeDebugInfo:
    """
    Per-frame diagnostic bundle emitted via EyeTracker.on_debug().

    All coordinates are in normalised user-perspective gaze space
    (x=0 left, x=1 right, y=0 top, y=1 bottom) unless noted.

    Fields
    ------
    left_iris         Raw normalised iris centre of the left eye  (camera space)
    right_iris        Raw normalised iris centre of the right eye (camera space)
    left_eye_bounds   EyeBounds for the left eye
    right_eye_bounds  EyeBounds for the right eye
    left_gaze         Normalised gaze from left eye alone  (user-perspective)
    right_gaze        Normalised gaze from right eye alone (user-perspective)
    averaged_gaze     Average of left_gaze + right_gaze
    tracking_valid    False if validation failed (cursor not updated)
    validation_reason Human-readable reason when tracking_valid is False
    """
    left_iris:         Optional[Tuple[float, float]]
    right_iris:        Optional[Tuple[float, float]]
    left_eye_bounds:   Optional[EyeBounds]
    right_eye_bounds:  Optional[EyeBounds]
    left_gaze:         Optional[Tuple[float, float]]
    right_gaze:        Optional[Tuple[float, float]]
    averaged_gaze:     Optional[Tuple[float, float]]
    tracking_valid:    bool
    validation_reason: str = ""


# ── Low-level landmark helpers ────────────────────────────────────────────────

def _lm_mean(landmarks, indices: List[int]) -> Tuple[float, float]:
    """Return (mean_x, mean_y) of the given landmark indices."""
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _iris_centre(landmarks, indices: List[int]) -> Tuple[float, float]:
    """Return the centroid (mean_x, mean_y) of iris ring landmarks."""
    return _lm_mean(landmarks, indices)


def _build_eye_bounds(
    landmarks,
    horizontal: Tuple[int, int],   # (outer_idx, inner_idx)
    eyelid_top: List[int],
    eyelid_bottom: List[int],
) -> EyeBounds:
    """
    Compute the bounding box of one eye socket.

    Horizontal extent: outer corner → inner corner (both on the same
    horizontal axis — avoids the near-zero height bug from previous
    implementation that used only two corner landmarks).

    Vertical extent: mean of the three upper-eyelid landmarks (top)
    and mean of the three lower-eyelid landmarks (bottom).  Averaging
    three points per lid gives a much more stable estimate than any
    single landmark.
    """
    outer_x = landmarks[horizontal[0]].x
    inner_x = landmarks[horizontal[1]].x

    _, top_y    = _lm_mean(landmarks, eyelid_top)
    _, bottom_y = _lm_mean(landmarks, eyelid_bottom)

    # Ensure left < right and top < bottom regardless of camera orientation
    x_left  = min(outer_x, inner_x)
    x_right = max(outer_x, inner_x)
    y_top   = min(top_y, bottom_y)
    y_bot   = max(top_y, bottom_y)

    return EyeBounds(x_left=x_left, x_right=x_right,
                     y_top=y_top, y_bottom=y_bot)


def _normalise_iris(
    iris_x: float,
    iris_y: float,
    bounds: EyeBounds,
) -> Tuple[float, float]:
    """
    Express iris position as a fraction of the eye bounding box.

        norm_x = (iris_x - bounds.x_left)  / bounds.width
        norm_y = (iris_y - bounds.y_top)   / bounds.height

    Returns values in roughly [0, 1].  Values outside that range
    occur at extreme gaze angles and are filtered by the validator.
    """
    # Prevent division by zero when eyes are closed or geometry is distorted
    width = max(bounds.width, 0.001)
    height = max(bounds.height, 0.001)

    norm_x = (iris_x - bounds.x_left)  / width
    norm_y = (iris_y - bounds.y_top)   / height
    return norm_x, norm_y


# ── Validation ────────────────────────────────────────────────────────────────

class GazeValidator:
    """
    Validates per-eye geometry before accepting a gaze sample.

    Checks performed
    ----------------
    1. Eye width  >= GAZE_MIN_EYE_WIDTH   — rejects occluded/out-of-frame eyes
    2. Eye height >= GAZE_MIN_EYE_HEIGHT  — rejects blinks and closed eyes
       (important: this is a geometry check, NOT a blink-threshold check;
       blink detection still runs independently via EAR in BlinkDetector)
    3. Iris x and y within GAZE_IRIS_RANGE_MIN..GAZE_IRIS_RANGE_MAX — rejects
       landmark failures that put the iris wildly outside the eye socket

    All thresholds come from config.py and are tunable without code changes.
    """

    @staticmethod
    def validate_eye(
        iris_x: float,
        iris_y: float,
        bounds: EyeBounds,
        label: str = "eye",
    ) -> Tuple[bool, str]:
        """
        Validate one eye.  Returns (ok, reason_if_not_ok).
        """
        if bounds.width < GAZE_MIN_EYE_WIDTH:
            return False, (
                f"{label} width {bounds.width:.4f} < "
                f"MIN_EYE_WIDTH {GAZE_MIN_EYE_WIDTH}"
            )
        if bounds.height < GAZE_MIN_EYE_HEIGHT:
            return False, (
                f"{label} height {bounds.height:.4f} < "
                f"MIN_EYE_HEIGHT {GAZE_MIN_EYE_HEIGHT}"
            )
        if not (GAZE_IRIS_RANGE_MIN <= iris_x <= GAZE_IRIS_RANGE_MAX):
            return False, (
                f"{label} iris_x {iris_x:.4f} out of range "
                f"[{GAZE_IRIS_RANGE_MIN}, {GAZE_IRIS_RANGE_MAX}]"
            )
        if not (GAZE_IRIS_RANGE_MIN <= iris_y <= GAZE_IRIS_RANGE_MAX):
            return False, (
                f"{label} iris_y {iris_y:.4f} out of range "
                f"[{GAZE_IRIS_RANGE_MIN}, {GAZE_IRIS_RANGE_MAX}]"
            )
        return True, ""


# ── Public gaze API ───────────────────────────────────────────────────────────

def compute_gaze(landmarks) -> GazeDebugInfo:
    """
    Compute gaze from MediaPipe face landmarks, returning a full
    GazeDebugInfo bundle that includes per-eye diagnostics.

    Coordinate contract
    ───────────────────
    averaged_gaze (when tracking_valid=True) is in user-perspective
    canonical coordinates:

        x = 0.0  →  user's left   (cursor should move left)
        x = 1.0  →  user's right  (cursor should move right)
        y = 0.0  →  top of screen
        y = 1.0  →  bottom of screen

    The horizontal mirror (camera sees left↔right flipped relative to
    the user) is corrected HERE so the rest of the pipeline is
    coordinate-system agnostic.

    Returns
    -------
    GazeDebugInfo
        Always returned.  Check .tracking_valid before using .averaged_gaze.
        When tracking_valid is False, .averaged_gaze is None and
        .validation_reason explains why.
    """
    # ── Build eye bounding boxes ──────────────────────────────────────────────
    right_bounds = _build_eye_bounds(
        landmarks,
        horizontal    = EAR_RIGHT_HORIZONTAL,   # (33=outer, 133=inner)
        eyelid_top    = GAZE_RIGHT_EYELID_TOP,
        eyelid_bottom = GAZE_RIGHT_EYELID_BOTTOM,
    )
    left_bounds = _build_eye_bounds(
        landmarks,
        horizontal    = EAR_LEFT_HORIZONTAL,    # (362=inner, 263=outer)
        eyelid_top    = GAZE_LEFT_EYELID_TOP,
        eyelid_bottom = GAZE_LEFT_EYELID_BOTTOM,
    )

    # ── Iris centres (camera space) ───────────────────────────────────────────
    r_iris_x, r_iris_y = _iris_centre(landmarks, IRIS_RIGHT_INDICES)
    l_iris_x, l_iris_y = _iris_centre(landmarks, IRIS_LEFT_INDICES)

    # ── Normalise iris within eye bounding box ────────────────────────────────
    r_norm_x, r_norm_y = _normalise_iris(r_iris_x, r_iris_y, right_bounds)
    l_norm_x, l_norm_y = _normalise_iris(l_iris_x, l_iris_y, left_bounds)

    # ── Validate each eye independently ──────────────────────────────────────
    r_ok, r_reason = GazeValidator.validate_eye(
        r_norm_x, r_norm_y, right_bounds, label="right eye"
    )
    l_ok, l_reason = GazeValidator.validate_eye(
        l_norm_x, l_norm_y, left_bounds,  label="left eye"
    )

    if not r_ok or not l_ok:
        reason = " | ".join(r for r in [r_reason, l_reason] if r)
        return GazeDebugInfo(
            left_iris         = (l_iris_x, l_iris_y),
            right_iris        = (r_iris_x, r_iris_y),
            left_eye_bounds   = left_bounds,
            right_eye_bounds  = right_bounds,
            left_gaze         = None,
            right_gaze        = None,
            averaged_gaze     = None,
            tracking_valid    = False,
            validation_reason = reason,
        )

    # ── Apply canonical coordinate system ────────────────────────────────────
    #
    # MediaPipe camera-space:    x increases LEFT → RIGHT from camera view
    # User-perspective canonical: x=0 is user's left, x=1 is user's right
    #
    # Since the camera sees a mirror of the user:
    #   canonical_x = 1.0 - normalised_x  (flips left↔right)
    #   canonical_y = normalised_y         (top↓bottom is the same)
    #
    # This correction lives here so CalibrationManager, MouseController,
    # and every other consumer sees user-perspective coordinates only.
    r_gaze_x = 1.0 - r_norm_x
    r_gaze_y = r_norm_y
    l_gaze_x = 1.0 - l_norm_x
    l_gaze_y = l_norm_y

    avg_x = (r_gaze_x + l_gaze_x) / 2.0
    avg_y = (r_gaze_y + l_gaze_y) / 2.0

    return GazeDebugInfo(
        left_iris         = (l_iris_x, l_iris_y),
        right_iris        = (r_iris_x, r_iris_y),
        left_eye_bounds   = left_bounds,
        right_eye_bounds  = right_bounds,
        left_gaze         = (l_gaze_x, l_gaze_y),
        right_gaze        = (r_gaze_x, r_gaze_y),
        averaged_gaze     = (avg_x, avg_y),
        tracking_valid    = True,
        validation_reason = "",
    )


def raw_gaze_point(landmarks) -> Optional[Tuple[float, float]]:
    """
    Convenience wrapper around compute_gaze() for the tracking loop.

    Returns the averaged user-perspective gaze point, or None if
    landmark validation failed.  Callers must check for None and
    skip the frame when it is returned.
    """
    info = compute_gaze(landmarks)
    return info.averaged_gaze  # None when tracking_valid is False


def _iris_delta(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two normalised gaze positions."""
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
            # Fallback: linear map from canonical gaze space → screen pixels.
            # raw_gaze_point() already returns user-perspective coordinates:
            #   x=0 → left, x=1 → right, y=0 → top, y=1 → bottom
            # So we map directly — no mirroring needed here.

            # The raw gaze coordinates typically only move in a narrow central range (e.g. 0.35 to 0.65).
            # If we map the full 0-1 range to the screen, the cursor is hypersensitive and will
            # frequently clip to the bottom/right edges of the screen due to minor head movement.
            # Here we apply a basic scale and center transform to make uncalibrated tracking usable.
            scale = 3.0
            scaled_x = (iris_x - 0.5) * scale + 0.5
            scaled_y = (iris_y - 0.5) * scale + 0.5

            # Clamp to [0, 1] so it doesn't overshoot the screen drastically
            scaled_x = max(0.0, min(1.0, scaled_x))
            scaled_y = max(0.0, min(1.0, scaled_y))

            return scaled_x * self._sw, scaled_y * self._sh

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
                    lms  = result.face_landmarks[0]
                    gaze = raw_gaze_point(lms)   # None if validation failed

                    if gaze is None:
                        # Landmark validation failed (blink, occlusion, etc.)
                        # — reset streak but don't penalise as a "face lost"
                        stable_streak = 0
                        prev_gaze     = None
                    elif prev_gaze is None:
                        # First valid frame — seed the stability check
                        stable_streak = 1
                        prev_gaze     = gaze
                    elif _iris_delta(gaze, prev_gaze) <= CALIBRATION_STABILITY_RADIUS:
                        stable_streak += 1
                        prev_gaze      = gaze
                    else:
                        # Gaze drifted — restart streak, do not collect
                        stable_streak = 0
                        prev_gaze     = gaze
                        log.debug(
                            "Gaze unstable at point %d (delta=%.4f) — resetting streak",
                            point_idx + 1,
                            _iris_delta(gaze, prev_gaze),
                        )

                    # Only collect once gaze is confirmed valid and stable
                    if gaze is not None and stable_streak >= CALIBRATION_STABILITY_MIN_FRAMES:
                        samples.append(gaze)
                else:
                    # Face lost — reset streak
                    stable_streak = 0
                    prev_gaze     = None

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
