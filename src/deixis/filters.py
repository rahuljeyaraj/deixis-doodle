"""Smoothing filters for tracked points (fingertips, feet).

The One-Euro filter (Casiez, Roussel and Vogel, CHI 2012) is a low-pass filter whose cutoff
rises with speed: heavy smoothing when the point is still (kills jitter) and little lag when
it moves fast. Units here are whatever the caller feeds in (mm for us) and seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deixis.geometry import Point


@dataclass(frozen=True)
class OneEuroParams:
    """Tuning knobs. Defaults are a starting point for a fingertip in mm at 30 fps.

    min_cutoff: cutoff in Hz when still. Lower means smoother but laggier at rest.
    beta: how fast the cutoff rises with speed (Hz per mm/s). Higher means less lag in motion.
    d_cutoff: cutoff in Hz for the speed estimate itself.
    """

    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    """One-Euro filter for a 2D point. The cutoff adapts to the speed magnitude,
    so both axes are smoothed alike.

    Call it with increasing timestamps. Samples with a non-increasing time are ignored
    (the last output is returned). Call `reset()` after a tracking dropout.
    """

    def __init__(self, params: OneEuroParams | None = None) -> None:
        self.params = params or OneEuroParams()
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._x: tuple[float, float] | None = None
        self._dx: tuple[float, float] = (0.0, 0.0)

    @property
    def value(self) -> Point | None:
        return self._x

    @property
    def speed(self) -> float:
        """Filtered speed magnitude in units per second."""
        return math.hypot(*self._dx)

    def __call__(self, t: float, p: Point) -> Point:
        x, y = float(p[0]), float(p[1])
        if self._t is None or self._x is None:
            self._t, self._x, self._dx = t, (x, y), (0.0, 0.0)
            return self._x
        dt = t - self._t
        if dt <= 0:
            return self._x
        p_ = self.params
        raw_dx = ((x - self._x[0]) / dt, (y - self._x[1]) / dt)
        a_d = _alpha(p_.d_cutoff, dt)
        self._dx = (
            a_d * raw_dx[0] + (1 - a_d) * self._dx[0],
            a_d * raw_dx[1] + (1 - a_d) * self._dx[1],
        )
        cutoff = p_.min_cutoff + p_.beta * math.hypot(*self._dx)
        a = _alpha(cutoff, dt)
        self._x = (a * x + (1 - a) * self._x[0], a * y + (1 - a) * self._x[1])
        self._t = t
        return self._x
