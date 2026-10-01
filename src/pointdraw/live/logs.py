"""The live run's timeline, written as it happens.

`timeline.jsonl` used to be written when the program ended, so a crash (one start in six
on 1 Oct 2026 ended in a segmentation fault) lost the whole session. `TimelineWriter`
appends the runtime's new events every frame and flushes, so a crash loses at most a frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TimelineWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f = self.path.open("w", encoding="utf-8")
        self._runtime = 0  # runtime events written so far

    def write_new(self, events: list[dict[str, Any]]) -> None:
        """Append the events added to the runtime's log since the last call."""
        new = events[self._runtime :]
        self._runtime = len(events)
        self.write(new)

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
