"""
eye_module
──────────
Eye-tracking module for gaze-controlled mouse input.

Public API
──────────
    from eye_module import EyeTracker, ensure_model, CalibrationManager
    from eye_module import PreviewModule, bgr_to_qpixmap
    from eye_module import MouseController
"""

from eye_module.eye_tracker    import EyeTracker, ensure_model
from eye_module.calibration    import CalibrationManager
from eye_module.blink_detector import BlinkDetector, compute_ear
from eye_module.smoother       import EMASmoother
from eye_module.preview        import PreviewModule, bgr_to_qpixmap
from eye_module.mouse_control  import MouseController
from eye_module                import config

__all__ = [
    "EyeTracker",
    "ensure_model",
    "CalibrationManager",
    "BlinkDetector",
    "compute_ear",
    "EMASmoother",
    "PreviewModule",
    "bgr_to_qpixmap",
    "MouseController",
    "config",
]
