"""Run a logged live session again, offline, with the code as it is now.

`scripts/live_run.py` logs every camera frame's hands (`hands.jsonl`, with landmarks) and the
runtime's events (`timeline.jsonl`: what was heard, the agent's replies, ...). This feeds the
same hands and the same words, at the same times, into a fresh `Runtime` with a simulated
plotter, so a fix to the gesture rules or the runtime can be checked against what really
happened on the Ender, without the Ender.

- Poses are classified again from the landmarks, with today's rules.
- The agent is not called: each utterance gets the reply recorded for the same words (in
  order), else the rule parser. A reply recorded in another context can differ from what the
  model would say now.
- There is no camera image, so marks drawn by hand on the sheet are not seen.
- The plotter's events in the timeline (kind "plotter", logged from 2 Oct) are kept apart in
  `Session.plotter` and not fed in: the replay's own plotter makes its own.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deixis.types import Word
from pointdraw.agent.fallback_parser import parse as fallback_parse
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.schema import ToolCallError, norm_word, parse_call
from pointdraw.controller.config import MachineConfig
from pointdraw.live.logs import PLOTTER
from pointdraw.runtime.events import EventLog
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import Runtime
from pointdraw.virtual.plotter import VirtualPlotter
from pointdraw.vision.hands import HandObservation, HandsFrame, classify_pointing


@dataclass
class Session:
    t0: float  # the camera's first frame, on the shared clock
    frames: list[HandsFrame]  # times on the shared clock
    events: list[dict[str, Any]]  # the original timeline (the runtime's events)
    plotter: list[dict[str, Any]] = field(default_factory=list)  # the plotter's, if logged


class RecordedAgent:
    """The replies recorded in the session, by the words; the rule parser for anything else."""

    def __init__(self, events: list[dict[str, Any]], categories: list[str]) -> None:
        self.categories = categories
        by_utt = {e["id"]: e["text"] for e in events if e["kind"] == "utterance"}
        self.replies: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for e in events:
            if e["kind"] == "tool_calls" and e["utt"] in by_utt:
                self.replies[_key(by_utt[e["utt"]].split())].append(e)
        self.recorded = 0
        self.parsed = 0

    def understand(self, words: list[str], context: Any) -> AgentResult:
        q = self.replies.get(_key(words))
        if q:
            e = q.popleft()
            try:
                calls = [parse_call(c["name"], c["args"]) for c in e["calls"]]
            except ToolCallError:
                calls = []
            self.recorded += 1
            return AgentResult(
                calls=calls,
                speech=e.get("speech"),
                source="cache",  # type: ignore[arg-type]
                model=e.get("model") or "",
                latency_s=float(e.get("latency_s") or 0.0),
            )
        self.parsed += 1
        return AgentResult(calls=fallback_parse(words, context, self.categories), source="fallback")


def _key(words: list[str]) -> str:
    return " ".join(norm_word(w) for w in words)


def load(run_dir: str | Path) -> Session:
    d = Path(run_dir)
    lines = (d / "hands.jsonl").read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    if "t0" not in head:
        raise ValueError(f"{d} has no hands.jsonl header: logged before replay was possible")
    t0 = float(head["t0"])
    frames = []
    for line in lines[1:]:
        r = json.loads(line)
        hands = []
        for h in r["hands"]:
            if "mm" not in h:
                raise ValueError(f"{d}: hands.jsonl has no landmarks")
            mm = np.asarray(h["mm"], dtype=np.float64)
            world = None if h.get("world") is None else np.asarray(h["world"], dtype=np.float64)
            pose = classify_pointing(world if world is not None else mm)
            hands.append(
                HandObservation(h.get("handedness", "Right"), h["score"], mm * 4, mm, world, pose)
            )
        frames.append(HandsFrame(t0 + r["t"], hands, (1920, 1080)))
    timeline = [
        json.loads(x) for x in (d / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [e for e in timeline if e["kind"] != PLOTTER]
    plotter = [e for e in timeline if e["kind"] == PLOTTER]
    return Session(t0, frames, events, plotter)


def replay(
    session: Session, machine: MachineConfig | None = None, tail_s: float = 120.0
) -> tuple[Runtime, Any]:
    """Run the session through a fresh runtime. Returns it and the agent (for its counts)."""
    machine = machine or MachineConfig()
    planner = Planner()
    agent = RecordedAgent(session.events, planner.categories)
    plotter = VirtualPlotter(machine)
    rt = Runtime(agent, plotter, machine, planner, log=EventLog())
    heard = sorted(
        (e for e in session.events if e["kind"] in ("speech_start", "utterance")),
        key=lambda e: e["t"],
    )
    if not session.frames:
        return rt, agent
    t_start = session.frames[0].t
    plotter.t = t_start
    rt.start(t_start)
    i = 0
    for f in session.frames:
        while i < len(heard) and heard[i]["t"] <= f.t:
            e = heard[i]
            i += 1
            if e["kind"] == "speech_start":
                rt.on_speech_start(e["t"])
            else:
                words = [Word(w, a, b) for w, a, b in e["words"]]
                rt.on_utterance(words, e["t"])
        plotter.tick(f.t)
        rt.on_frame(f)
        rt.tick(f.t)
    # After the last frame: no hands, until the runtime and the plotter have finished.
    t, t_end = session.frames[-1].t, session.frames[-1].t + tail_s
    while t < t_end and (rt.pending or rt.proposal or rt.drawing or plotter.busy):
        t += 1 / 23
        plotter.tick(t)
        rt.on_frame(HandsFrame(t, [], (1920, 1080)))
        rt.tick(t)
    return rt, agent


SHOWN = ("utterance", "ignored", "guard", "ask", "answer", "say", "preview", "drawn", "stop")


def summary(log: EventLog) -> dict[str, int]:
    return {
        "heard": sum(1 for e in log.of("utterance") if e["text"]),
        "asks": len(log.of("ask")),
        "drawn": len(log.of("drawn")),
        "guards": len(log.of("guard")) + len(log.of("ignored")),
    }
