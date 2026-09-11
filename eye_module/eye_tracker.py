"""
eye_tracker.py
──────────────
Core EyeTracker class.

Architecture
────────────
                  ┌──────────────────────────────────────────┐
                  │           EyeTracker (public API)         │
                  │                                           │
                  │  start() ──► _tracking_loop() [Thread]   │
                  │               │                           │
                  │    ┌──────────┼──────────┐                │
                  │    │          │          │                │
                  │  Camera   FaceLandmarker  │                │
                  │    │          │          │                │
                  │    └──► frame │          │                │
                  │            │  │          │                │
                  │      EMA Smoother        │                │
                  │            │             │                │
                  │       BlinkDetector ─────┘                │
                  │            │                              │
                  │     autopy.mouse.move / click             │
                  │            │                              │
                  │  Callbacks: on_move, on_blink_click,      │
                  │             on_dwell_click, on_error      │
                  └──────────────────────────────────────────┘

Thread safety
─────────────
All mutable tracker state is protected by ``_lock`` (threading.Lock).
Public setters (``dwell_enabled``, ``calibration_enabled``, etc.) acquire
the lock before mutating state.

Integration note (PySide6)
──────────────────────────
When you integrate into the main application, replace the callback
assignments with Qt signal emissions:

    tracker.on_move        = lambda x, y: self.move_signal.emit(x, y)
    tracker.on_blink_click = lambda:      self.click_signal.emit()
    tracker.on_dwell_click = lambda:      self.dwell_signal.emit()
    tracker.on_error       = lambda e:    self.error_signal.emit(str(e))

The background thread never touches the Qt event loop directly, which
keeps the design safe for Qt's thread-affinity rules.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Deferred heavy imports (keeps import time low) ─────────────────────────--

def _import_autopy():
    try:
        import autopy
        return autopy
    except ImportError as exc:
        raise ImportError(
            "autopy is not installed. Run: pip install autopy"
        ) from exc


def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is not installed. Run: pip install mediapipe"
        ) from exc


# ── Local imports ─────────────────────────────────────────────────────────────
from blink_detector import BlinkDetector
from calibration import CalibrationManager, raw_gaze_point
from config import (
    CALIBRATION_PATH,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    GAZE_USE_BOTH,
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
    MP_MIN_FACE_DETECTION_CONF,
    MP_MIN_FACE_PRESENCE_CONF,
    MP_MIN_TRACKING_CONF,
    MP_NUM_FACES,
    MOUSE_MOVE_SPEED,
    SCREEN_MARGIN_PX,
)
from smoother import EMASmoother


# ── Model bootstrapping ───────────────────────────────────────────────────────

def ensure_model(model_path: str = MODEL_PATH) -> None:
    """
    Download the MediaPipe FaceLandmarker model if it is missing.

    This runs synchronously and should be called before the first
    :class:`EyeTracker` is created.  It only downloads once; subsequent
    calls return immediately.

    Parameters
    ----------
    model_path : str
        Destination path for the ``.task`` file.
    """
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 1_000:
        log.debug("Model already present: %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading FaceLandmarker model from %s …", MODEL_DOWNLOAD_URL)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  Downloading model … {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(path), _progress)
        print()  # newline after progress
        log.info("Model saved to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the MediaPipe model.\n"
            f"Please download it manually from:\n  {MODEL_DOWNLOAD_URL}\n"
            f"and save it to:\n  {model_path}"
        ) from exc


# ── EyeTracker ────────────────────────────────────────────────────────────────

class EyeTracker:
    """
    Eye-tracking controller that moves the system mouse cursor.

    Parameters
    ----------
    screen_w, screen_h : int
        Screen resolution in pixels (used for coordinate mapping and
        clamping).
    camera_index : int
        OpenCV camera index.
    dwell_enabled : bool
        Initial state of dwell-click.  Can be toggled at runtime.
    load_calibration : bool
        Whether to attempt loading an existing calibration file on init.
    model_path : str
        Path to the ``face_landmarker.task`` model file.
    calibration_path : str | Path
        Path for saving / loading calibration JSON.

    Callbacks (set before calling start())
    ────────────────────────────────────────
    on_move(x, y)        Called after every successful cursor move.
    on_blink_click()     Called when a blink-click fires.
    on_dwell_click()     Called when a dwell-click fires.
    on_face_lost()       Called when the face leaves the frame.
    on_face_found()      Called when the face re-enters the frame.
    on_error(exc)        Called if a fatal error occurs in the thread.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        camera_index: int = CAMERA_INDEX,
        dwell_enabled: bool = False,
        load_calibration: bool = True,
        model_path: str = MODEL_PATH,
        calibration_path: str | Path = CALIBRATION_PATH,
    ) -> None:
        self._screen_w   = screen_w
        self._screen_h   = screen_h
        self._cam_index  = camera_index
        self._model_path = model_path

        # ── Callbacks ──
        self.on_move:        Optional[Callable[[float, float], None]] = None
        self.on_blink_click: Optional[Callable[[], None]]             = None
        self.on_dwell_click: Optional[Callable[[], None]]             = None
        self.on_face_lost:   Optional[Callable[[], None]]             = None
        self.on_face_found:  Optional[Callable[[], None]]             = None
        self.on_error:       Optional[Callable[[Exception], None]]    = None

        # ── Sub-components ──
        self._smoother = EMASmoother()
        self._blink    = BlinkDetector(dwell_enabled=dwell_enabled)
        self._calibration = CalibrationManager(
            screen_w, screen_h, calibration_path
        )
        if load_calibration:
            self._calibration.load()

        # ── Wire blink/dwell callbacks through to tracker callbacks ──
        self._blink.on_blink_click = self._on_blink_click_internal
        self._blink.on_dwell_click = self._on_dwell_click_internal

        # ── Thread control ──
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()

        # ── State flags ──
        self._face_visible = False

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def dwell_enabled(self) -> bool:
        return self._blink.dwell_enabled

    @dwell_enabled.setter
    def dwell_enabled(self, value: bool) -> None:
        with self._lock:
            self._blink.dwell_enabled = value
        log.info("Dwell-click %s", "enabled" if value else "disabled")

    @property
    def is_calibrated(self) -> bool:
        return self._calibration.is_calibrated

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the background tracking thread.

        Raises
        ------
        RuntimeError
            If the tracker is already running.
        """
        if self.is_running:
            raise RuntimeError("EyeTracker is already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tracking_loop,
            name="EyeTrackerThread",
            daemon=True,
        )
        self._thread.start()
        log.info("EyeTracker started.")

    def stop(self, timeout: float = 3.0) -> None:
        """
        Signal the tracking thread to stop and wait for it to finish.

        Parameters
        ----------
        timeout : float
            Seconds to wait before returning regardless.
        """
        if not self.is_running:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        self._smoother.reset()
        self._blink.reset()
        log.info("EyeTracker stopped.")

    # ── Calibration convenience ───────────────────────────────────────────────

    def run_calibration(self, cap: Optional[cv2.VideoCapture] = None) -> bool:
        """
        Run the interactive calibration routine.

        Must be called from the main thread (OpenCV windows are not
        thread-safe).  Stops the tracker first if it is running.

        Parameters
        ----------
        cap : cv2.VideoCapture | None
            Camera to use.  If ``None`` a new one is opened and closed
            after calibration.

        Returns
        -------
        bool
            ``True`` on success.
        """
        was_running = self.is_running
        if was_running:
            self.stop()

        mp = _import_mediapipe()
        own_cap = cap is None
        if own_cap:
            cap = self._open_camera()

        try:
            landmarker = self._build_landmarker(mp, mode="video")
            success = self._calibration.run(landmarker, cap)
            landmarker.close()
        finally:
            if own_cap:
                cap.release()

        if success and was_running:
            self.start()
        return success

    # ── Internal: tracking loop ───────────────────────────────────────────────

    def _tracking_loop(self) -> None:
        """Main loop executed on the background thread."""
        mp = _import_mediapipe()
        autopy = _import_autopy()

        cap = None
        landmarker = None
        try:
            cap = self._open_camera()
            landmarker = self._build_landmarker(mp, mode="video")

            frame_index = 0
            face_was_visible = False

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("Camera read failed; retrying …")
                    time.sleep(0.05)
                    continue

                frame_index += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                result = landmarker.detect_for_video(mp_image, frame_index)

                if not result.face_landmarks:
                    # ── No face ────────────────────────────────────────────
                    if face_was_visible:
                        face_was_visible = False
                        self._smoother.reset()
                        self._blink.reset()
                        self._fire(self.on_face_lost)
                    continue

                # ── Face detected ──────────────────────────────────────────
                lms = result.face_landmarks[0]

                if not face_was_visible:
                    face_was_visible = True
                    self._fire(self.on_face_found)

                # 1. Raw gaze in normalised camera space
                raw_x, raw_y = raw_gaze_point(lms)

                # 2. Map to screen coordinates
                screen_x, screen_y = self._calibration.map(raw_x, raw_y)

                # 3. Smooth
                sx, sy = self._smoother.update(screen_x, screen_y)

                # 4. Clamp to screen bounds with margin
                sx = max(SCREEN_MARGIN_PX, min(self._screen_w - SCREEN_MARGIN_PX, sx))
                sy = max(SCREEN_MARGIN_PX, min(self._screen_h - SCREEN_MARGIN_PX, sy))

                # 5. Move the mouse
                try:
                    autopy.mouse.move(sx, sy)
                except Exception as exc:
                    log.debug("autopy.mouse.move error: %s", exc)

                # 6. Notify listeners
                self._fire(self.on_move, sx, sy)

                # 7. Blink / dwell detection
                with self._lock:
                    self._blink.update(lms, sx, sy)

        except Exception as exc:
            log.exception("Fatal error in EyeTracker thread: %s", exc)
            self._fire(self.on_error, exc)
        finally:
            if landmarker is not None:
                landmarker.close()
            if cap is not None:
                cap.release()
            log.debug("Tracking thread exiting.")

    # ── Internal: MediaPipe setup ─────────────────────────────────────────────

    def _build_landmarker(self, mp, mode: str = "video"):
        """
        Construct a ``FaceLandmarker`` in VIDEO (synchronous) mode.
        VIDEO mode lets us call ``detect_for_video`` on each frame and
        get results immediately without callbacks — the simplest and most
        reliable approach for a dedicated tracking thread.
        """
        VisionTaskRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        BaseOptions           = mp.tasks.BaseOptions

        running_mode = (
            VisionTaskRunningMode.VIDEO
            if mode == "video"
            else VisionTaskRunningMode.IMAGE
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=running_mode,
            num_faces=MP_NUM_FACES,
            min_face_detection_confidence=MP_MIN_FACE_DETECTION_CONF,
            min_face_presence_confidence=MP_MIN_FACE_PRESENCE_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
            output_face_blendshapes=False,   # off → lower CPU usage
            output_facial_transformation_matrixes=False,
        )
        return FaceLandmarker.create_from_options(options)

    # ── Internal: camera ──────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cam_index}. "
                "Check your camera connection and index."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        # Lower internal buffer to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        log.info(
            "Camera opened: index=%d, %dx%d @ %d fps",
            self._cam_index, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
        )
        return cap

    # ── Internal: callback relay ──────────────────────────────────────────────

    def _fire(self, callback, *args) -> None:
        """Call a user callback safely, swallowing any exceptions."""
        if callable(callback):
            try:
                callback(*args)
            except Exception as exc:
                log.debug("Callback %s raised: %s", callback, exc)

    def _on_blink_click_internal(self) -> None:
        try:
            import autopy
            autopy.mouse.click()
        except Exception as exc:
            log.debug("autopy click error: %s", exc)
        self._fire(self.on_blink_click)

    def _on_dwell_click_internal(self) -> None:
        try:
            import autopy
            autopy.mouse.click()
        except Exception as exc:
            log.debug("autopy click error: %s", exc)
        self._fire(self.on_dwell_click)
