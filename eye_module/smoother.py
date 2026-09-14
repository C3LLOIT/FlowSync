"""
smoother.py
───────────
Exponential Moving Average (EMA) smoother for 2-D gaze coordinates.

Formula
───────
    smoothed_x = α * raw_x + (1 - α) * prev_x
    smoothed_y = α * raw_y + (1 - α) * prev_y

α (alpha) controls the trade-off:
  • Small α  (e.g. 0.10) → very smooth, but lags behind fast movements.
  • Large α  (e.g. 0.50) → responsive, but less stable at rest.
  • α = 1.0  → no smoothing (pass-through).

The smoother is reset whenever tracking is lost so stale history
does not corrupt the first frame after re-acquisition.
"""

from __future__ import annotations

from typing import Optional, Tuple

from eye_module.config import EMA_ALPHA


class EMASmoother:
    """
    Exponential Moving Average smoother for (x, y) coordinates.

    Parameters
    ----------
    alpha : float
        Smoothing factor in the range (0, 1].  Defaults to ``EMA_ALPHA``
        from ``config.py``.
    """

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self._alpha: float = alpha
        # Stored as float once initialised; None only before first update.
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {value!r}")
        self._alpha = value

    @property
    def has_state(self) -> bool:
        """``True`` after the first :meth:`update`, ``False`` after :meth:`reset`."""
        return self._x is not None

    @property
    def value(self) -> Optional[Tuple[float, float]]:
        """Current smoothed value, or ``None`` if no state."""
        if self._x is None or self._y is None:
            return None
        return self._x, self._y

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Feed a raw sample; return the smoothed (x, y).

        On the first call (or after :meth:`reset`) the raw value is
        returned unchanged and stored as the initial state.
        """
        if self._x is None or self._y is None:
            # Bootstrap: accept the first sample as-is.
            # After this branch _x/_y are always float.
            self._x = x
            self._y = y
        else:
            a = self._alpha
            # Both _x and _y are guaranteed float here — narrowed above.
            self._x = a * x + (1.0 - a) * self._x
            self._y = a * y + (1.0 - a) * self._y

        # At this point _x and _y are always float (never None).
        return self._x, self._y

    def reset(self) -> None:
        """
        Discard accumulated history.

        Call this when the tracked face disappears so stale state does
        not bleed into the next detection.
        """
        self._x = None
        self._y = None
