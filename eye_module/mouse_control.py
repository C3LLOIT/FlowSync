"""
mouse_control.py
────────────────
Cross-platform mouse control using only the OS stdlib — no autopy,
no pyautogui, no external dependencies.

Windows  : ctypes → user32.SetCursorPos / mouse_event
Linux    : python-xlib if available, else xdotool subprocess
macOS    : Quartz CoreGraphics via ctypes

Why not autopy?
───────────────
autopy requires a Rust toolchain to build from source and frequently
fails to install on Windows 10/11. ctypes.windll is always available
on Windows and calls the exact same Win32 API under the hood.

Public API
──────────
    from eye_module.mouse_control import MouseController
    mc = MouseController()           # auto-detects platform
    mc.move(x, y)                    # absolute screen coordinates (pixels)
    mc.click()                       # left click at current position
    w, h = mc.screen_size()          # primary screen resolution
"""

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Tuple

log = logging.getLogger(__name__)

_SYSTEM = platform.system()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  WINDOWS                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

class _WindowsMouse:
    """Win32 SetCursorPos via ctypes — zero dependencies."""

    def __init__(self) -> None:
        import ctypes
        self._user32 = ctypes.windll.user32  # type: ignore[attr-defined]

    def move(self, x: float, y: float) -> None:
        self._user32.SetCursorPos(int(round(x)), int(round(y)))

    def click(self) -> None:
        # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
        self._user32.mouse_event(0x0002, 0, 0, 0, 0)
        self._user32.mouse_event(0x0004, 0, 0, 0, 0)

    def screen_size(self) -> Tuple[int, int]:
        import ctypes
        # SM_CXSCREEN=0, SM_CYSCREEN=1
        w = self._user32.GetSystemMetrics(0)
        h = self._user32.GetSystemMetrics(1)
        return w, h


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  LINUX                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

class _LinuxMouseXlib:
    """python-xlib based mouse control."""

    def __init__(self) -> None:
        from Xlib import display as xdisplay
        from Xlib.ext import xtest
        self._d    = xdisplay.Display()
        self._root = self._d.screen().root
        self._xtest = xtest

    def move(self, x: float, y: float) -> None:
        self._root.warp_pointer(int(round(x)), int(round(y)))
        self._d.sync()

    def click(self) -> None:
        self._xtest.fake_input(self._d, 4, 1)   # ButtonPress button=1
        self._xtest.fake_input(self._d, 5, 1)   # ButtonRelease button=1
        self._d.sync()

    def screen_size(self) -> Tuple[int, int]:
        info = self._d.screen()
        return info.width_in_pixels, info.height_in_pixels


class _LinuxMouseXdotool:
    """xdotool subprocess fallback (no python deps)."""

    def move(self, x: float, y: float) -> None:
        subprocess.run(
            ["xdotool", "mousemove", str(int(round(x))), str(int(round(y)))],
            check=False, capture_output=True,
        )

    def click(self) -> None:
        subprocess.run(["xdotool", "click", "1"],
                       check=False, capture_output=True)

    def screen_size(self) -> Tuple[int, int]:
        try:
            out = subprocess.check_output(
                ["xdotool", "getdisplaygeometry"], text=True
            ).strip()
            w, h = map(int, out.split())
            return w, h
        except Exception:
            return 1920, 1080


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  macOS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

class _MacMouse:
    """CoreGraphics via ctypes."""

    def __init__(self) -> None:
        import ctypes, ctypes.util
        cg_path = ctypes.util.find_library("CoreGraphics")
        self._cg = ctypes.CDLL(cg_path)  # type: ignore[arg-type]

        # CGEventCreateMouseEvent, CGEventPost, etc.
        self._cg.CGWarpMouseCursorPosition.restype  = None

        class CGPoint(ctypes.Structure):
            _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]
        self._CGPoint = CGPoint

    def move(self, x: float, y: float) -> None:
        pt = self._CGPoint(x=float(x), y=float(y))
        self._cg.CGWarpMouseCursorPosition(pt)
        self._cg.CGAssociateMouseAndMouseCursorPosition(1)

    def click(self) -> None:
        # Use osascript as the simplest no-dep click on macOS
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to click'],
            check=False, capture_output=True,
        )

    def screen_size(self) -> Tuple[int, int]:
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            import re
            m = re.search(r"Resolution: (\d+) x (\d+)", out)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
        return 1920, 1080


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PUBLIC: MouseController                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

class MouseController:
    """
    Cross-platform mouse controller. Auto-detects the best backend.

    Falls back gracefully — if no backend works, move() and click()
    become no-ops with a warning so the rest of the module keeps running.
    """

    def __init__(self) -> None:
        self._backend = self._detect_backend()
        if self._backend is None:
            log.warning(
                "No mouse backend available — cursor control disabled.\n"
                "  Windows: should work automatically via ctypes.\n"
                "  Linux:   pip install python-xlib  OR  apt install xdotool\n"
                "  macOS:   should work automatically via CoreGraphics."
            )

    def _detect_backend(self):
        if _SYSTEM == "Windows":
            try:
                b = _WindowsMouse()
                log.debug("Mouse backend: Win32 (ctypes)")
                return b
            except Exception as e:
                log.warning("Win32 mouse backend failed: %s", e)

        elif _SYSTEM == "Linux":
            try:
                b = _LinuxMouseXlib()
                log.debug("Mouse backend: Xlib")
                return b
            except Exception:
                pass
            try:
                subprocess.run(
                    ["xdotool", "--version"],
                    check=True, capture_output=True,
                )
                b = _LinuxMouseXdotool()
                log.debug("Mouse backend: xdotool")
                return b
            except Exception as e:
                log.warning("Linux mouse backends failed: %s", e)

        elif _SYSTEM == "Darwin":
            try:
                b = _MacMouse()
                log.debug("Mouse backend: CoreGraphics (macOS)")
                return b
            except Exception as e:
                log.warning("macOS mouse backend failed: %s", e)

        return None

    def move(self, x: float, y: float) -> None:
        """Move cursor to absolute screen position (pixels)."""
        if self._backend is not None:
            try:
                self._backend.move(x, y)
            except Exception as e:
                log.debug("mouse.move error: %s", e)

    def click(self) -> None:
        """Fire a left click at the current cursor position."""
        if self._backend is not None:
            try:
                self._backend.click()
            except Exception as e:
                log.debug("mouse.click error: %s", e)

    def screen_size(self) -> Tuple[int, int]:
        """Return (width, height) of the primary screen in pixels."""
        if self._backend is not None:
            try:
                return self._backend.screen_size()
            except Exception as e:
                log.debug("screen_size error: %s", e)
        # PySide6 fallback
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if isinstance(app, QApplication):
                s = app.primaryScreen()
                if s is not None:
                    sz = s.size()
                    return sz.width(), sz.height()
        except Exception:
            pass
        return 1920, 1080


# ── Module-level singleton (lazy init) ───────────────────────────────────────

_controller: MouseController | None = None

def get_mouse() -> MouseController:
    """Return the shared MouseController instance (created on first call)."""
    global _controller
    if _controller is None:
        _controller = MouseController()
    return _controller
