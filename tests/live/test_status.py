"""The live window's banner: why the plotter waits, in plain words (night 2 Oct 2026)."""

import pytest

from pointdraw.controller.api import PlotEvent
from pointdraw.live.status import Banner


def ev(kind: str, tag: str = "cross#1") -> PlotEvent:
    return PlotEvent(0.0, kind, "draw", tag)


@pytest.mark.parametrize(
    ("events", "state", "busy", "clear", "problem", "want"),
    [
        ([], "idle", False, True, None, "Ready"),
        ([], "paused", True, False, None, "Waiting: hand over the paper"),
        ([], "paused", True, True, None, "Starting: hand away, half a second"),
        (["job_start"], "drawing", True, True, None, "Drawing"),
        (["job_start"], "previewing", True, True, None, "Showing where it goes (pen up)"),
        (
            ["job_start", "pause_request", "paused"],
            "paused",
            True,
            False,
            None,
            "Paused: hand over the paper",
        ),
        (["job_start", "paused", "resumed"], "drawing", True, True, None, "Drawing"),
        (
            ["job_start", "pause_request"],
            "paused",
            True,
            False,
            None,
            "Paused: hand over the paper",
        ),
        (["job_start", "job_end"], "paused", True, False, None, "Waiting: hand over the paper"),
        (["job_start", "stop_request", "halted"], "stopped", False, True, None, "Stopped"),
        (
            ["job_start", "error"],
            "error",
            False,
            True,
            None,
            "Plotter problem: check its USB cable, then press R",
        ),
        (
            [],
            "idle",
            False,
            True,
            "write failed",
            "Plotter problem: check its USB cable, then press R",
        ),
        (["job_start", "job_end"], "moving", True, True, None, "Moving"),
    ],
)
def test_banner_says_why(events, state, busy, clear, problem, want):
    b = Banner()
    b.update([ev(k) for k in events])
    assert b.text(state, busy, clear, "LISTEN", problem)[0] == want


def test_banner_names_what_the_app_waits_for():
    b = Banner()
    assert b.text("idle", False, True, "PREVIEW")[0] == "Say yes to draw it, or change it"
    assert b.text("idle", False, True, "CLARIFY")[0] == "Ready: answer the question"


def test_a_paused_job_that_ended_is_not_paused_any_more():
    b = Banner()
    evs = [ev("job_start"), ev("paused")]
    b.update(evs)
    assert b.paused
    evs += [ev("halted")]
    b.update(evs)
    assert b.running is None and not b.paused
