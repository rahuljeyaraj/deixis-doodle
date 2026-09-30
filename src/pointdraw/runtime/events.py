"""A timestamped event log shared by the runtime, the virtual world and its reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventLog:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, t: float, kind: str, **data: Any) -> dict[str, Any]:
        ev = {"t": round(float(t), 3), "kind": kind, **data}
        self.events.append(ev)
        return ev

    def of(self, *kinds: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] in kinds]

    def first(self, kind: str, **match: Any) -> dict[str, Any] | None:
        for e in self.events:
            if e["kind"] == kind and all(e.get(k) == v for k, v in match.items()):
                return e
        return None
