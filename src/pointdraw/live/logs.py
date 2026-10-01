"""The live run's timeline, written as it happens.

`timeline.jsonl` used to be written when the program ended, so a crash (one start in six
on 1 Oct 2026 ended in a segmentation fault) lost the whole session. `TimelineWriter`
appends the runtime's new events every frame and flushes, so a crash loses at most a frame.

The plotter's own events (`LivePlotter.events`: job_start, job_end, pause_request, paused,
resumed, stop_request, halted, error) go in too, as `{"kind": "plotter", "event": ...}`,
stamped on the same clock as the runtime's events. Lines are in the order written, so a
plotter event can come a frame after a runtime event with a later time.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from pointdraw.controller.api import PlotEvent

PLOTTER = "plotter"  # the kind of a plotter event's line


def plotter_record(e: PlotEvent) -> dict[str, Any]:
    """One timeline line for a plotter event: t, kind "plotter", event, label, tag, detail."""
    rec: dict[str, Any] = {"t": round(e.t, 4), "kind": PLOTTER, "event": e.kind}
    if e.label:
        rec["label"] = e.label
    if e.tag is not None:
        rec["tag"] = e.tag
    for k, v in e.detail.items():
        rec.setdefault(k, v)
    return rec


class TimelineWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f = self.path.open("w", encoding="utf-8")
        self._runtime = 0  # runtime events written so far
        self._plotter = 0  # plotter events written so far

    def write_new(self, events: list[dict[str, Any]]) -> None:
        """Append the events added to the runtime's log since the last call."""
        new = events[self._runtime :]
        self._runtime = len(events)
        self.write(new)

    def write_plotter(self, events: list[PlotEvent]) -> None:
        """Append the plotter events added since the last call (`LivePlotter.events`)."""
        n = len(events)  # the worker may append while we write: take a fixed count
        new = events[self._plotter : n]
        self._plotter = n
        self.write([plotter_record(e) for e in new])

    def write(self, records: list[dict[str, Any]]) -> None:
        for e in records:
            self._f.write(json.dumps(e, default=str) + "\n")
        if records:
            self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> TimelineWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JsonlWriter:
    """Writes records as JSON lines from a background thread, off the camera loop.

    `hands.jsonl` gets every frame's landmarks (about 5 KB a frame at 30 fps); turning that
    into text and writing it took time from the loop that watches for hands.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._f = self.path.open("w", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="jsonl-writer", daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        self._q.put(record)

    def _run(self) -> None:
        while True:
            rec = self._q.get()
            if rec is None:
                break
            self._f.write(json.dumps(rec) + "\n")
            if self._q.empty():
                self._f.flush()
        self._f.close()

    def close(self) -> None:
        if self._thread.is_alive():
            self._q.put(None)
            self._thread.join(10)
