"""One-Euro filter: jitter reduction when still, bounded lag when moving."""

import math

import numpy as np

from deixis.filters import OneEuroFilter2D, OneEuroParams


def _run(f, ts, pts):
    return np.array([f(t, p) for t, p in zip(ts, pts, strict=True)])


def test_first_sample_passes_through():
    f = OneEuroFilter2D()
    assert f(0.0, (3.0, 4.0)) == (3.0, 4.0)
    assert f.value == (3.0, 4.0)


def test_reduces_jitter_when_still():
    rng = np.random.default_rng(0)
    ts = np.arange(0, 3, 1 / 30)
    raw = np.array([50.0, 80.0]) + rng.normal(0, 0.8, size=(len(ts), 2))
    out = _run(OneEuroFilter2D(), ts, raw)
    # Compare frame-to-frame motion after the filter settles.
    raw_step = np.linalg.norm(np.diff(raw[30:], axis=0), axis=1).mean()
    out_step = np.linalg.norm(np.diff(out[30:], axis=0), axis=1).mean()
    assert out_step < raw_step / 4
    assert np.linalg.norm(out[30:].mean(axis=0) - [50, 80]) < 0.5


def test_small_lag_when_moving_fast():
    ts = np.arange(0, 2, 1 / 30)
    v = 200.0  # mm/s, a quick hand movement
    raw = np.stack([v * ts, np.zeros_like(ts)], axis=1)
    out = _run(OneEuroFilter2D(OneEuroParams(min_cutoff=1.0, beta=0.05)), ts, raw)
    lag_mm = raw[-1, 0] - out[-1, 0]
    assert 0 <= lag_mm < 15  # a fixed low-pass at 1 Hz would lag about 32 mm


def test_ignores_non_increasing_time_and_resets():
    f = OneEuroFilter2D()
    f(1.0, (0.0, 0.0))
    assert f(1.0, (10.0, 10.0)) == (0.0, 0.0)
    assert f(0.5, (10.0, 10.0)) == (0.0, 0.0)
    f.reset()
    assert f.value is None
    assert f(2.0, (7.0, 7.0)) == (7.0, 7.0)
    assert math.isclose(f.speed, 0.0)
