# eye_module — Gaze-Controlled Mouse

Eye-tracking module that maps iris gaze position to the system mouse cursor.
Built for Python 3.11.9 with MediaPipe, OpenCV, autopy, and PySide6 (integration-ready).

---

## File Structure

```
eye_module/
├── __init__.py          # Public API surface
├── config.py            # All constants and tuneable defaults
├── smoother.py          # EMA smoother (noise → smooth coordinates)
├── blink_detector.py    # EAR blink-click + dwell-click logic
├── calibration.py       # 5-point calibration + JSON persist
├── eye_tracker.py       # Core EyeTracker class (background thread)
├── demo.py              # Standalone test / demo script
├── models/              # MediaPipe .task model (auto-downloaded)
│   └── face_landmarker.task
└── calibration/
    └── calibration.json # Saved after first calibration run
```

---

## Dependencies

```
mediapipe>=0.10.13
opencv-python>=4.8
autopy>=4.0
numpy>=1.24
pyside6>=6.5          # not yet used — ready for integration
```

Install:
```bash
pip install mediapipe opencv-python autopy numpy pyside6
```

---

## Quick Start

```python
import autopy
from eye_module import EyeTracker, ensure_model

# 1. Download model on first run (~30 MB, one-time)
ensure_model()

# 2. Screen size
sw, sh = autopy.screen.size()

# 3. Create tracker
tracker = EyeTracker(screen_w=int(sw), screen_h=int(sh))

# 4. Wire up callbacks
tracker.on_move        = lambda x, y: print(f"Cursor ({x:.0f}, {y:.0f})")
tracker.on_blink_click = lambda: print("Blink click!")
tracker.on_dwell_click = lambda: print("Dwell click!")
tracker.on_face_lost   = lambda: print("Face lost")
tracker.on_face_found  = lambda: print("Face found")
tracker.on_error       = lambda e: print(f"Error: {e}")

# 5. Enable dwell-click (off by default)
tracker.dwell_enabled = True

# 6. Start (non-blocking — runs on a daemon thread)
tracker.start()

# ... your application loop ...

# 7. Stop gracefully
tracker.stop()
```

---

## Demo Script

```bash
cd eye_module

# Basic tracking (blink to click)
python demo.py

# With 5-point calibration
python demo.py --calibrate

# With dwell-click enabled
python demo.py --dwell

# All options
python demo.py --calibrate --dwell --alpha 0.3 --camera 0
```

---

## Calibration

Calibration maps raw iris coordinates (camera space) to screen pixels using
a 5-point affine transform, stored as JSON.

```python
# Run calibration (opens a fullscreen OpenCV window)
tracker.run_calibration()

# Calibration is saved automatically to:
#   eye_module/calibration/calibration.json
# and reloaded on next startup automatically.

# Check if calibrated
if tracker.is_calibrated:
    print("Ready to track")
```

**Calibration window controls:**
- A dot appears at each of 5 positions — look at each dot and hold still.
- A progress ring fills as samples are collected.
- Press `Esc` to abort.

Without calibration a proportional fallback mapping is used (good enough for testing).

---

## Click Gestures

### Blink-click
Hold **both eyes closed** for ≥ 400 ms → left-click fires.
- EAR threshold: 0.20 (close), 0.25 (re-open hysteresis)
- Requires simultaneous closure of both eyes (avoids false positives from winks)

### Dwell-click (disabled by default)
Keep gaze **within 30 px radius** for ≥ 800 ms → left-click fires.
- Automatically re-arms after firing (gaze must move before next dwell)
- Enable via `tracker.dwell_enabled = True` at any time

---

## Configuration (`config.py`)

| Constant | Default | Description |
|---|---|---|
| `EMA_ALPHA` | `0.25` | Smoothing factor (lower = smoother) |
| `EAR_CLOSE_THRESHOLD` | `0.20` | EAR value that marks eyes as closed |
| `BLINK_HOLD_MS` | `400` | Hold duration for blink-click (ms) |
| `DWELL_HOLD_MS` | `800` | Hold duration for dwell-click (ms) |
| `DWELL_RADIUS_PX` | `30` | Gaze stability radius for dwell (px) |
| `CAMERA_INDEX` | `0` | Default webcam index |
| `CAMERA_WIDTH/HEIGHT` | `640×480` | Capture resolution |
| `MOUSE_MOVE_SPEED` | `0.5` | autopy move duration in seconds |

---

## PySide6 Integration (When Ready)

The tracker is designed to slot into PySide6 cleanly.
Replace callbacks with Qt signal emissions:

```python
# In your QObject / QThread wrapper:
from PySide6.QtCore import QObject, Signal

class EyeTrackerBridge(QObject):
    move_signal  = Signal(float, float)
    click_signal = Signal()
    dwell_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, tracker: EyeTracker):
        super().__init__()
        tracker.on_move        = self.move_signal.emit
        tracker.on_blink_click = self.click_signal.emit
        tracker.on_dwell_click = self.dwell_signal.emit
        tracker.on_error       = lambda e: self.error_signal.emit(str(e))
```

The background thread never touches the Qt event loop, so thread-affinity
rules are respected.

---

## Landmark Reference

| Role | Indices |
|---|---|
| Right iris ring | 469, 470, 471, 472 |
| Left iris ring  | 474, 475, 476, 477 |
| Right EAR verts | (159,145), (158,153), (160,144) |
| Left EAR verts  | (386,374), (385,380), (387,373) |

---

## Troubleshooting

**Model not found**
Run `ensure_model()` or download manually from:
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
Save to `eye_module/models/face_landmarker.task`.

**autopy permission error (Linux)**
autopy needs X11 access. Run with a display, or set `DISPLAY=:0`.

**Cursor jitter**
Lower `EMA_ALPHA` in `config.py` (e.g. `0.15`) or run calibration.

**False blink-clicks**
Raise `EAR_CLOSE_THRESHOLD` slightly (e.g. `0.22`) if natural blinks trigger unintended clicks, or increase `BLINK_HOLD_MS`.
