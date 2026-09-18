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
                  │             on_dwell_click, on_error,     │
                  │             on_face_found, on_face_lost   │
                  │             on_frame (preview hook)       │
                  └──────────────────────────────────────────┘

Thread safety
─────────────
All mutable tracker state is protected by ``_lock`` (threading.Lock).
Public setters acquire the lock before mutating state.

Integration note (PySide6)
──────────────────────────
Replace callbacks with Qt signal emissions:

    tracker.on_move        = lambda x, y: self.move_signal.emit(x, y)
    tracker.on_blink_click = lambda:      self.click_signal.emit()
    tracker.on_frame       = lambda f, r: self.frame_signal.emit(f, r)

The background thread never touches the Qt event loop directly.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Deferred heavy imports ────────────────────────────────────────────────────

def _import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is not installed. Run: pip install mediapipe"
        ) from exc


# ── Local imports ─────────────────────────────────────────────────────────────
from eye_module.blink_detector import BlinkDetector
from eye_module.calibration    import (
    CalibrationManager,
    GazeDebugInfo,
    compute_gaze,
    raw_gaze_point,
)
from eye_module.config import (
    CALIBRATION_PATH,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
    MP_MIN_FACE_DETECTION_CONF,
    MP_MIN_FACE_PRESENCE_CONF,
    MP_MIN_TRACKING_CONF,
    MP_NUM_FACES,
    SCREEN_MARGIN_PX,
)
from eye_module.mouse_control import MouseController
from eye_module.smoother      import EMASmoother


# ── Model bootstrapping ───────────────────────────────────────────────────────

def ensure_model(model_path: str = MODEL_PATH) -> None:
    """
    Download the MediaPipe FaceLandmarker model if it is missing.
    Runs synchronously; subsequent calls return immediately if already present.
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
        print()
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
        Screen resolution in pixels.
    camera_index : int
        OpenCV camera index (default 0).
    dwell_enabled : bool
        Initial dwell-click state. Togglable at runtime.
    load_calibration : bool
        Attempt to load an existing calibration file on init.
    model_path : str
        Path to the face_landmarker.task model file.
    calibration_path : str | Path
        Path for saving / loading calibration JSON.

    Callbacks (assign before calling start())
    ─────────────────────────────────────────
    on_move(x, y)         Called after every successful cursor move.
    on_blink_click()      Called when blink-click fires.
    on_dwell_click()      Called when dwell-click fires.
    on_face_lost()        Called when face leaves the frame.
    on_face_found()       Called when face enters the frame.
    on_error(exc)         Called on fatal thread error.
    on_frame(frame, result)
        Called every frame with the raw BGR numpy frame and the
        MediaPipe FaceLandmarkerResult. Use this for preview rendering.
        Fires even when no face is detected (result.face_landmarks == []).
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        camera_index:     int        = CAMERA_INDEX,
        dwell_enabled:    bool       = False,
        load_calibration: bool       = True,
        model_path:       str        = MODEL_PATH,
        calibration_path: str | Path = CALIBRATION_PATH,
    ) -> None:
        self._screen_w   = screen_w
        self._screen_h   = screen_h
        self._cam_index  = camera_index
        self._model_path = model_path

        # ── Callbacks ──────────────────────────────────────────────────
        self.on_move:        Optional[Callable[[float, float], None]] = None
        self.on_blink_click: Optional[Callable[[], None]]             = None
        self.on_dwell_click: Optional[Callable[[], None]]             = None
        self.on_face_lost:   Optional[Callable[[], None]]             = None
        self.on_face_found:  Optional[Callable[[], None]]             = None
        self.on_error:       Optional[Callable[[Exception], None]]    = None
        # Preview hook — receives (bgr_frame, mediapipe_result) every frame
        self.on_frame: Optional[Callable[[np.ndarray, object], None]] = None

        # Debug hook — receives GazeDebugInfo every frame when debug_enabled=True.
        # Subscribe before calling start(); safely ignored when None.
        self.on_debug: Optional[Callable[[GazeDebugInfo], None]] = None

        # Set to True to emit on_debug every frame (slight CPU overhead).
        self.debug_enabled: bool = False

        # ── Sub-components ─────────────────────────────────────────────
        self._smoother    = EMASmoother()
        self._blink       = BlinkDetector(dwell_enabled=dwell_enabled)
        self._calibration = CalibrationManager(screen_w, screen_h, calibration_path)
        if load_calibration:
            self._calibration.load()

        # Wire blink/dwell callbacks
        self._blink.on_blink_click = self._on_blink_click_internal
        self._blink.on_dwell_click = self._on_dwell_click_internal

        # ── Thread control ─────────────────────────────────────────────
        self._thread:    Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()

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
        """Start the background tracking thread."""
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
        """Signal the tracking thread to stop and wait for it."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._smoother.reset()
        self._blink.reset()
        log.info("EyeTracker stopped.")

    # ── Calibration convenience ───────────────────────────────────────────────

    def run_calibration(self, cap: Optional[cv2.VideoCapture] = None) -> bool:
        """
        Run the interactive calibration routine (call from main thread).
        Stops the tracker first if running; restarts after if it was running.
        """
        was_running = self.is_running
        if was_running:
            self.stop()

        mp      = _import_mediapipe()
        own_cap = cap is None
        if own_cap:
            cap = self._open_camera()

        try:
            landmarker = self._build_landmarker(mp, mode="video")
            success    = self._calibration.run(landmarker, cap)
            landmarker.close()
        finally:
            if own_cap and cap is not None:
                cap.release()

        if success and was_running:
            self.start()
        return success

    # ── Internal: tracking loop ───────────────────────────────────────────────

    def _tracking_loop(self) -> None:
        """Main loop — runs entirely on the background thread."""
        mp    = _import_mediapipe()
        mouse = MouseController()   # cross-platform, no external deps

        cap:        Optional[cv2.VideoCapture] = None
        landmarker = None

        try:
            cap        = self._open_camera()
            landmarker = self._build_landmarker(mp, mode="video")

            face_was_visible = False
            fps_hint:    int = int(cap.get(cv2.CAP_PROP_FPS) or 30)
            frame_ts_ms: int = 0   # strictly-increasing ms timestamp for MediaPipe

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("Camera read failed; retrying …")
                    time.sleep(0.05)
                    continue

                # ── MediaPipe requires strictly-increasing MILLISECOND timestamps ──
                frame_ts_ms += max(1, 1000 // fps_hint)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )
                result = landmarker.detect_for_video(mp_image, frame_ts_ms)

                # ── Preview hook (fires every frame, before face check) ──────────
                self._fire(self.on_frame, frame, result)

                # ── No face ───────────────────────────────────────────────────────
                if not result.face_landmarks:
                    if face_was_visible:
                        face_was_visible = False
                        self._smoother.reset()
                        self._blink.reset()
                        self._fire(self.on_face_lost)
                    continue

                # ── Face detected ─────────────────────────────────────────────────
                lms = result.face_landmarks[0]

                if not face_was_visible:
                    face_was_visible = True
                    self._fire(self.on_face_found)

                # 1. Compute gaze (full debug bundle always computed when
                #    debug_enabled; otherwise use the lightweight wrapper)
                if self.debug_enabled:
                    gaze_info = compute_gaze(lms)
                    self._fire(self.on_debug, gaze_info)
                    gaze_pt = gaze_info.averaged_gaze   # None if invalid
                else:
                    gaze_pt = raw_gaze_point(lms)       # None if invalid

                # 2. Validation gate — skip cursor update on bad landmarks
                #    (blink, occlusion, extreme head pose, tracking failure)
                if gaze_pt is None:
                    # Smoother keeps last position; blink detector still runs
                    with self._lock:
                        if self._smoother.has_state:
                            sx, sy = self._smoother.value  # type: ignore[misc]
                            self._blink.update(lms, sx, sy)
                    continue

                raw_x, raw_y = gaze_pt

                # 3. Map gaze → screen coordinates
                screen_x, screen_y = self._calibration.map(raw_x, raw_y)

                # 4. Smooth
                sx, sy = self._smoother.update(screen_x, screen_y)

                # 5. Clamp to screen bounds
                sx = max(SCREEN_MARGIN_PX, min(self._screen_w - SCREEN_MARGIN_PX, sx))
                sy = max(SCREEN_MARGIN_PX, min(self._screen_h - SCREEN_MARGIN_PX, sy))

                # 6. Move mouse
                mouse.move(sx, sy)

                # 7. Notify move listeners
                self._fire(self.on_move, sx, sy)

                # 8. Blink / dwell detection (always runs on valid frames)
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
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return FaceLandmarker.create_from_options(options)

    # ── Internal: camera ──────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        # On Windows CAP_DSHOW opens significantly faster than the default
        # MSMF backend (~0.3s vs ~3-5s). We try CAP_DSHOW first on all
        # platforms (it is a no-op on Linux/macOS) and fall back once.
        import platform
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._cam_index, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._cam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cam_index}. "
                "Check your camera connection and index."
            )
        # Set resolution and FPS. Do NOT set FOURCC before these — some
        # drivers reset all props when FOURCC is changed, causing extra delay.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
        # Minimal internal buffer: reduces latency and avoids stale frames
        # appearing after a pause (e.g. while MediaPipe is initialising).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warm-up: drain the first few frames so MediaPipe gets a clean image
        # immediately rather than a dark/blurry startup frame.
        for _ in range(3):
            cap.read()
        log.info(
            "Camera opened: index=%d, %dx%d @ %d fps (backend=%s)",
            self._cam_index, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
            "CAP_DSHOW" if backend == cv2.CAP_DSHOW else "default",
        )
        return cap

    # ── Internal: callback relay ──────────────────────────────────────────────

    def _fire(self, callback: object, *args: object) -> None:
        """Call a user callback safely, swallowing exceptions."""
        if callable(callback):
            try:
                callback(*args)  # type: ignore[call-arg]
            except Exception as exc:
                log.debug("Callback %s raised: %s", callback, exc)

    def _on_blink_click_internal(self) -> None:
        MouseController().click()
        self._fire(self.on_blink_click)

    def _on_dwell_click_internal(self) -> None:
        MouseController().click()
        self._fire(self.on_dwell_click)
