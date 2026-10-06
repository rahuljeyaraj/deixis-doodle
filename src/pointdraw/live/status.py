"""What the plotter is doing, in big plain words for the live window.

On 1 Oct the window said "plotter paused" both when a drawing had not started because a
hand was over the paper and when one had stopped half way for a hand, and nothing when the
app was simply waiting for quiet. `Banner` says which, from the controller's state, the
hands-clear interlock and the plotter's events:

- "Plotter problem: check its USB cable, then press R"  the link failed
- "Stopped"                                              after a stop
- "Paused: hand over the paper"                          a job began, then a hand came
- "Waiting: hand over the paper"                         a job is ready, a hand is in the way
- "Starting: hand away, half a second"                   the hand just left (the 0.5 s hold)
- "Drawing" / "Showing where it goes" / "Moving"         the job running
- "Ready" (with what the app is doing, when it matters)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RED, ORANGE, GREEN, BLUE, GREY = (
    (40, 40, 220),
    (0, 140, 255),
    (60, 180, 60),
    (220, 140, 40),
    (150, 150, 150),
)

APP = {
    "UNDERSTAND": "Ready: thinking about what you said",
    "CLARIFY": "Ready: answer the question",
    "PREVIEW": "Say yes to draw it, or change it",
    "ADJUST": "Say yes to draw it, or change it",
    "WAIT_HANDS_CLEAR": "Waiting: take your hand away to draw",
}


@dataclass
class Banner:
    """Follows `LivePlotter.events` (by index) to know whether a job is under way or paused."""

    seen: int = 0
    running: Any = None  # the tag of the job that started and has not ended
    paused: bool = False

    def update(self, events: list) -> None:
        n = len(events)
        for e in events[self.seen : n]:
            if e.kind == "job_start":
                self.running, self.paused = e.tag, False
            elif e.kind == "paused":
                self.paused = True
            elif e.kind == "resumed":
                self.paused = False
            elif e.kind in ("job_end", "halted", "error", "reconnected"):
                self.running, self.paused = None, False
        self.seen = n

    def text(
        self,
        state: str,
        busy: bool,
        hands_clear: bool,
        app_state: str = "LISTEN",
        problem: str | None = None,
    ) -> tuple[str, tuple[int, int, int]]:
        """The banner's words and colour (BGR) for the plotter's `state` (ControllerStatus)."""
        if state == "error" or problem:
            return "Plotter problem: check its USB cable, then press R", RED
        if state == "stopped" and not busy:
            return "Stopped", RED
        if state == "paused" or (busy and not hands_clear):
            if self.running is not None:  # under way (the "paused" event comes a bit later)
                return "Paused: hand over the paper", ORANGE
            if not hands_clear:
                return "Waiting: hand over the paper", ORANGE
            return "Starting: hand away, half a second", BLUE
        if state == "drawing":
            return "Drawing", GREEN
        if state == "previewing":
            return "Showing where it goes (pen up)", GREEN
        if state == "moving" or busy:
            return "Moving", GREEN
        return APP.get(app_state, "Ready"), GREY
