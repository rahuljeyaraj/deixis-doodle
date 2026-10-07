"""The amber dot must not fly off to a chair leg.

7 Oct, live in rahul's room: "the orange dot is not sticking to my leg, it is jumping away
most of the time". MediaPipe sees one frame at a time and will put a foot on furniture for a
few frames at a stretch. `SteadySpot` drops what a person could not have done.
"""

from __future__ import annotations

import numpy as np

from pointdraw.vacuum.feet import SteadySpot


def walk(n: int = 30, step_mm: float = 20.0, fps: float = 30.0):
    """Someone strolling along x at a believable pace."""
    for i in range(n):
        yield i / fps, np.array([100.0 + i, 200.0]), (1000.0 + i * step_mm, 500.0)


def test_a_steady_walk_is_followed():
    s = SteadySpot()
    last = None
    for t, px, mm in walk():
        _, last, took = s.offer(t, px, mm)
        assert took
    assert s.dropped == 0
    assert last is not None and last[0] > 1000.0


def test_a_jump_across_the_room_is_dropped():
    s = SteadySpot()
    for t, px, mm in walk(10):
        s.offer(t, px, mm)
    held = s._best()[1]
    # the tracker finds a chair three metres away, for three frames
    for i in range(3):
        _, mm, took = s.offer(10 / 30 + i / 30, np.array([900.0, 800.0]), (4000.0, 2500.0))
        assert not took, "a 3 m hop between frames is not a person walking"
        assert mm == held, "the dot should stay where it was, not follow"
    assert s.dropped == 3


def test_it_recovers_if_the_person_really_did_move():
    """Dropping for ever would be worse than jumping: after a gap, believe what you see."""
    s = SteadySpot()
    for t, px, mm in walk(10):
        s.offer(t, px, mm)
    t_late = 10 / 30 + 2.0  # two seconds later, well past recover_s
    _, mm, took = s.offer(t_late, np.array([900.0, 800.0]), (4000.0, 2500.0))
    assert took
    assert mm is not None


def test_nothing_seen_holds_the_last_spot_rather_than_blanking():
    s = SteadySpot()
    for t, px, mm in walk(10):
        s.offer(t, px, mm)
    held = s._best()[1]
    px, mm, took = s.offer(10 / 30, None, None)
    assert not took and mm == held and px is not None


def test_it_says_nothing_before_it_has_seen_anything():
    s = SteadySpot()
    px, mm, took = s.offer(0.0, None, None)
    assert px is None and mm is None and not took


def test_the_spot_is_smoothed_not_just_passed_through():
    """One frame of jitter inside the believable range should not move the dot much."""
    s = SteadySpot()
    for t, px, mm in walk(10, step_mm=0.0):
        s.offer(t, px, mm)
    steady = s._best()[1]
    _, jittered, took = s.offer(10 / 30, np.array([100.0, 200.0]), (1040.0, 500.0))
    assert took
    assert abs(jittered[0] - steady[0]) < 20.0, "a 40 mm wobble should be mostly smoothed away"
