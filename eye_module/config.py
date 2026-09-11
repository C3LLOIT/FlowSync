"""
config.py
─────────
Central configuration for the eye-tracking module.
All tuneable constants live here so the rest of the code
stays free of magic numbers.
"""

from __future__ import annotations
import os

# ── Paths ────────────────────────────────────────────────────────────────────
MODULE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR        = os.path.join(MODULE_DIR, "models")
MODEL_FILENAME    = "face_landmarker.task"
MODEL_PATH        = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

CALIBRATION_DIR      = os.path.join(MODULE_DIR, "calibration")
CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_PATH     = os.path.join(CALIBRATION_DIR, CALIBRATION_FILENAME)

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0          # Default webcam index
CAMERA_WIDTH    = 640        # Capture resolution width  (px)
CAMERA_HEIGHT   = 480        # Capture resolution height (px)
CAMERA_FPS      = 30         # Target capture frame rate

# ── MediaPipe FaceLandmarker ─────────────────────────────────────────────────
MP_NUM_FACES               = 1      # Track only one face for performance
MP_MIN_FACE_DETECTION_CONF = 0.6    # Confidence threshold for face detection
MP_MIN_FACE_PRESENCE_CONF  = 0.6    # Confidence for face presence
MP_MIN_TRACKING_CONF       = 0.6    # Confidence for landmark tracking

# ── Iris landmark indices (MediaPipe 468-landmark topology) ──────────────────
#
#  Right iris (from subject's perspective):
#    469 – centre, 470 – right edge, 471 – bottom, 472 – left edge
#  Left iris (from subject's perspective):
#    474 – centre, 475 – right edge, 476 – bottom, 477 – left edge
#
IRIS_RIGHT_INDICES = [469, 470, 471, 472]   # right eye iris ring
IRIS_LEFT_INDICES  = [474, 475, 476, 477]   # left eye iris ring
IRIS_RIGHT_CENTER  = 469
IRIS_LEFT_CENTER   = 474

# ── Eye-Aspect-Ratio (EAR) landmarks ────────────────────────────────────────
# Six landmarks per eye used to compute EAR for blink/closure detection.
# Vertical pairs: (top-outer, bottom-outer), (top-mid, bottom-mid),
#                 (top-inner, bottom-inner)
# Horizontal pair: (left-corner, right-corner)
#
#  Right eye (camera-space):
EAR_RIGHT_VERTICAL_PAIRS = [(159, 145), (158, 153), (160, 144)]
EAR_RIGHT_HORIZONTAL     = (33, 133)

#  Left eye (camera-space):
EAR_LEFT_VERTICAL_PAIRS  = [(386, 374), (385, 380), (387, 373)]
EAR_LEFT_HORIZONTAL      = (362, 263)

# ── Gaze / Smoothing ─────────────────────────────────────────────────────────
EMA_ALPHA        = 0.25   # Exponential Moving Average weight (0 < α ≤ 1).
                           # Lower = smoother but slower; higher = snappier.
GAZE_USE_BOTH    = True    # Average both irises for the gaze point
                           # (False → dominant/right-eye only)

# ── Blink-click ──────────────────────────────────────────────────────────────
EAR_CLOSE_THRESHOLD   = 0.20    # EAR below this → eye is closed
EAR_OPEN_HYSTERESIS   = 0.25    # EAR must rise above this to re-open
BLINK_HOLD_MS         = 400     # Milliseconds both eyes must stay closed
                                 # before a left-click is fired

# ── Dwell-click ──────────────────────────────────────────────────────────────
DWELL_ENABLED_DEFAULT = False   # Off until enabled by the main application
DWELL_RADIUS_PX       = 30      # Pixel radius within which gaze must stay
DWELL_HOLD_MS         = 800     # Milliseconds the gaze must dwell before click

# ── Calibration ──────────────────────────────────────────────────────────────
# Calibration targets (as fractions of screen width/height, 0–1)
CALIBRATION_TARGETS = [
    (0.05, 0.05),   # top-left
    (0.95, 0.05),   # top-right
    (0.50, 0.50),   # centre
    (0.05, 0.95),   # bottom-left
    (0.95, 0.95),   # bottom-right
]
CALIBRATION_SAMPLES_PER_POINT  = 30    # Frames averaged per target point
CALIBRATION_SETTLE_MS          = 1500  # Pause after displaying target (ms)

# Gaze stability gate — prevents samples being collected when the user
# is not actually looking at the target.
#
# The iris position (normalised 0–1) must not move more than
# CALIBRATION_STABILITY_RADIUS between consecutive frames, and must
# hold still for at least CALIBRATION_STABILITY_MIN_FRAMES frames
# before any samples are accepted for that point.
CALIBRATION_STABILITY_RADIUS     = 0.015  # max iris delta between frames
CALIBRATION_STABILITY_MIN_FRAMES = 10     # consecutive stable frames needed

# ── Mouse movement ───────────────────────────────────────────────────────────
MOUSE_MOVE_SPEED   = 0.5    # autopy move duration (seconds); 0 = instant
SCREEN_MARGIN_PX   = 5      # Keep cursor this far from screen edges
