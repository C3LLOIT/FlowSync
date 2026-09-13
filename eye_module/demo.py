"""
demo.py
───────
Standalone demo for the eye_module.

Run from the eye_module directory:

    python demo.py [--calibrate] [--dwell]

Flags
─────
  --calibrate   Run the 5-point calibration routine before tracking.
  --dwell       Enable dwell-click (hold gaze for 800 ms to click).
  --alpha FLOAT Override the EMA smoothing factor (default: 0.25).
  --camera INT  Camera index to use (default: 0).

Press  Ctrl+C  to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eye-tracking mouse control demo.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run calibration before tracking.")
    p.add_argument("--dwell",     action="store_true",
                   help="Enable dwell-click.")
    p.add_argument("--alpha",     type=float, default=None,
                   help="EMA alpha (0 < α ≤ 1, default from config).")
    p.add_argument("--camera",    type=int,   default=None,
                   help="Camera index (default from config).")
    return p.parse_args()



def _get_screen_size():
    """Return (width, height) of the primary screen with autopy optional."""
    try:
        import autopy
        sw, sh = autopy.screen.size()
        return int(sw), int(sh)
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy(); return w, h
    except Exception:
        pass
    log.warning("Cannot detect screen size — defaulting to 1920x1080")
    return 1920, 1080


def main() -> None:
    args = parse_args()

    # ── 1. Ensure the MediaPipe model is present ───────────────────────────--
    from eye_module.eye_tracker import ensure_model
    ensure_model()

    # ── 2. Resolve screen size ─────────────────────────────────────────────--
    screen_w, screen_h = _get_screen_size()
    log.info("Screen: %d × %d px", screen_w, screen_h)

    # ── 3. Build tracker ───────────────────────────────────────────────────--
    from eye_module.eye_tracker import EyeTracker
    from eye_module import config

    cam_idx = args.camera if args.camera is not None else config.CAMERA_INDEX

    tracker = EyeTracker(
        screen_w=screen_w,
        screen_h=screen_h,
        camera_index=cam_idx,
        dwell_enabled=args.dwell,
        load_calibration=True,
    )

    # Override alpha if requested
    if args.alpha is not None:
        tracker._smoother.alpha = args.alpha
        log.info("EMA alpha set to %.3f", args.alpha)

    # ── 4. Register callbacks ──────────────────────────────────────────────--
    _last_pos = [0.0, 0.0]

    def on_move(x: float, y: float) -> None:
        _last_pos[0], _last_pos[1] = x, y

    def on_blink_click() -> None:
        log.info("🖱  Blink-click at (%.0f, %.0f)", *_last_pos)

    def on_dwell_click() -> None:
        log.info("🕐  Dwell-click at (%.0f, %.0f)", *_last_pos)

    def on_face_lost() -> None:
        log.warning("Face lost — tracking paused.")

    def on_face_found() -> None:
        log.info("Face detected — tracking resumed.")

    def on_error(exc: Exception) -> None:
        log.error("Tracker error: %s", exc)

    tracker.on_move        = on_move
    tracker.on_blink_click = on_blink_click
    tracker.on_dwell_click = on_dwell_click
    tracker.on_face_lost   = on_face_lost
    tracker.on_face_found  = on_face_found
    tracker.on_error       = on_error

    # ── 5. Calibration (optional) ──────────────────────────────────────────--
    if args.calibrate or not tracker.is_calibrated:
        if not tracker.is_calibrated:
            log.info("No calibration found — running calibration now.")
        log.info("Starting calibration …")
        ok = tracker.run_calibration()
        if not ok:
            log.warning("Calibration aborted.")

    # ── 6. Start tracking ──────────────────────────────────────────────────--
    tracker.start()

    mode_str = "dwell+blink" if args.dwell else "blink-only"
    log.info(
        "Tracking started. Mode: %s | Press Ctrl+C to stop.", mode_str
    )

    # ── 7. Status ticker ──────────────────────────────────────────────────--
    def _status_ticker() -> None:
        while tracker.is_running:
            time.sleep(5)
            if tracker.is_running:
                log.info(
                    "Status: running | cursor≈(%.0f, %.0f) | "
                    "calibrated=%s | dwell=%s",
                    _last_pos[0], _last_pos[1],
                    tracker.is_calibrated, tracker.dwell_enabled,
                )

    ticker = threading.Thread(target=_status_ticker, daemon=True)
    ticker.start()

    # ── 8. Wait for interrupt ─────────────────────────────────────────────--
    try:
        while tracker.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt — stopping …")
    finally:
        tracker.stop()
        log.info("Done.")


if __name__ == "__main__":
    main()
