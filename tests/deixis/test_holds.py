"""Hold detector on synthetic fingertip trajectories with jitter."""

import numpy as np
import pytest

from deixis.filters import OneEuroFilter2D
from deixis.holds import HoldDetector, HoldParams

FPS = 30.0


def _trajectory(segments, fps=FPS):
    """segments: list of (duration_s, start_xy, end_xy). Returns times and points (mm)."""
    ts, pts, t = [], [], 0.0
    for dur, a, b in segments:
        n = max(1, round(dur * fps))
        for i in range(n):
            s = i / n
            ts.append(t)
            pts.append((a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s))
            t += 1 / fps
    return np.array(ts), np.array(pts)


def _feed(det, ts, pts, pointing=None, smooth=True):
    f = OneEuroFilter2D()
    ended = []
    for i, (t, p) in enumerate(zip(ts, pts, strict=True)):
        q = f(t, p) if smooth else tuple(p)
        ok = True if pointing is None else pointing[i]
        h = det.update(float(t), q, ok)
        if h is not None:
            ended.append(h)
    last = det.flush()
    if last is not None:
        ended.append(last)
    return ended


def test_two_holds_with_jitter():
    rng = np.random.default_rng(1)
    ts, pts = _trajectory(
        [
            (0.5, (0, 0), (100, 50)),  # move at ~220 mm/s
            (0.6, (100, 50), (100, 50)),  # hold 1
            (0.4, (100, 50), (40, 120)),  # move
            (0.5, (40, 120), (40, 120)),  # hold 2
            (0.3, (40, 120), (150, 20)),  # move away
        ]
    )
    pts = pts + rng.normal(0, 0.5, pts.shape)  # MediaPipe-like jitter in mm
    holds = _feed(HoldDetector(), ts, pts)
    assert len(holds) == 2
    h1, h2 = holds
    assert h1.mean == pytest.approx((100, 50), abs=2.5)
    assert h2.mean == pytest.approx((40, 120), abs=2.5)
    assert 0.5 < h1.t0 < 0.8 and 0.9 < h1.t1 < 1.2
    assert h1.t1 - h1.t0 >= 0.15
    assert max(h1.spread) < 2.0
    assert h1.frame == "paper" and h1.source == "index_tip"


def test_short_pause_is_not_a_hold():
    ts, pts = _trajectory(
        [
            (0.5, (0, 0), (100, 0)),
            (0.1, (100, 0), (100, 0)),  # 100 ms: too short
            (0.5, (100, 0), (0, 0)),
        ]
    )
    assert _feed(HoldDetector(), ts, pts, smooth=False) == []


def test_slow_drift_below_threshold_is_a_hold_fast_is_not():
    slow_ts, slow = _trajectory([(1.0, (0, 0), (10, 0))])  # 10 mm/s
    assert len(_feed(HoldDetector(), slow_ts, slow, smooth=False)) == 1
    fast_ts, fast = _trajectory([(1.0, (0, 0), (40, 0))])  # 40 mm/s
    assert _feed(HoldDetector(), fast_ts, fast, smooth=False) == []


def test_requires_pointing_pose():
    ts, pts = _trajectory([(1.0, (50, 50), (50, 50))])
    not_pointing = [False] * len(ts)
    assert _feed(HoldDetector(), ts, pts, pointing=not_pointing) == []
    loose = HoldDetector(HoldParams(require_pointing=False))
    assert len(_feed(loose, ts, pts, pointing=not_pointing)) == 1


def test_long_dropout_splits_hold_but_one_frame_does_not():
    ts, pts = _trajectory([(1.0, (50, 50), (50, 50))])
    det = HoldDetector()
    ended = []
    for i, (t, p) in enumerate(zip(ts, pts, strict=True)):
        q = None if i == 15 else tuple(p)  # one dropped frame
        h = det.update(float(t), q)
        if h:
            ended.append(h)
    ended.append(det.flush())
    assert len(ended) == 1 and ended[0].t1 - ended[0].t0 > 0.8

    det = HoldDetector()
    ended = []
    for i, (t, p) in enumerate(zip(ts, pts, strict=True)):
        q = None if 12 <= i < 20 else tuple(p)  # ~270 ms dropout
        h = det.update(float(t), q)
        if h:
            ended.append(h)
    h = det.flush()
    if h:
        ended.append(h)
    assert len(ended) == 2


def test_current_and_window_queries():
    ts, pts = _trajectory([(0.3, (0, 0), (60, 0)), (0.5, (60, 0), (60, 0))])
    det = HoldDetector()
    for t, p in zip(ts, pts, strict=True):
        det.update(float(t), tuple(p))
    cur = det.current()
    assert cur is not None and cur.mean == pytest.approx((60, 0), abs=1e-6)
    assert det.holds(0.0, 10.0) == [cur]
    assert det.holds(0.0, 10.0, include_current=False) == []
    assert det.holds(5.0, 10.0) == []
    det.flush()
    assert det.current() is None
    assert len(det.holds()) == 1
