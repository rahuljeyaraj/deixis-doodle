"""Run one scene in the virtual world, in simulated time, and score it.

Per camera frame (23 fps): the plotter advances, due sheet events happen (the user draws a
mark), speech starts and finished utterances reach the runtime, the virtual hand's frame
goes to the runtime, and the runtime ticks. Nothing touches hardware: the plotter is
`VirtualPlotter`, or with `RunConfig(plotter="controller")` the real controller behind the
live adapter in simulated time (`lockstep.LockstepPlotter`). No serial port is ever opened.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from pointdraw.controller.config import MachineConfig
from pointdraw.drawing import Library
from pointdraw.runtime.events import EventLog
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import CLARIFY, LISTEN, Runtime, RuntimeParams
from pointdraw.scene.model import SceneObject
from pointdraw.virtual.agents import LiveAgent, MockAgent, ReplyCache
from pointdraw.virtual.hand import FPS, VirtualHand
from pointdraw.virtual.plotter import InkSegment, VirtualPlotter
from pointdraw.virtual.scenefile import SceneSpec
from pointdraw.virtual.vision import MockVision, VirtualCamera, VirtualSheet
from pointdraw.virtual.voice import Spoken, text_mode


@dataclass
class RunConfig:
    ai: str = "mock"  # mock | live
    cosmos: str = "mock"  # mock | live
    voice: str = "text"  # text | tts
    seed: int = 0
    hover_mm: float = 0.0
    shift_s: float = 0.0
    model: str | None = None  # live agent model
    record: bool = False  # live: write replies into the cache
    emergency_parser: bool | None = None  # None: the scene's setting
    use_cache: bool = True  # mock: use recorded replies
    plotter: str = "virtual"  # virtual | controller (the real controller, in lockstep)

    @property
    def variant(self) -> str:
        parts = [f"seed{self.seed}"]
        if self.hover_mm:
            parts.append(f"hover{self.hover_mm:g}")
        if self.shift_s:
            parts.append(f"shift{self.shift_s:+g}")
        if self.emergency_parser is not None:
            parts.append("ep" if self.emergency_parser else "noep")
        if not self.use_cache:
            parts.append("parser")
        if self.plotter != "virtual":
            parts.append(self.plotter)
        return "-".join(parts)


@dataclass
class RunResult:
    spec: SceneSpec
    cfg: RunConfig
    log: EventLog
    runtime: Runtime
    plotter: Any  # VirtualPlotter or LockstepPlotter
    sheet: VirtualSheet
    hand: VirtualHand
    camera: VirtualCamera
    spoken: list[Spoken]
    t_end: float
    wall_s: float
    hand_track: list[tuple[float, Any]] = field(default_factory=list)
    agent: Any = None
    report: dict[str, Any] = field(default_factory=dict)


_LIBRARY: Library | None = None
_CACHE: ReplyCache | None = None


def _library() -> Library:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = Library.load()
    return _LIBRARY


def _cache() -> ReplyCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ReplyCache()
    return _CACHE


def make_agent(cfg: RunConfig, categories: list[str]) -> Any:
    if cfg.ai == "live":
        return LiveAgent(categories, model=cfg.model, cache=_cache(), record=cfg.record)
    return MockAgent(categories, cache=_cache(), use_cache=cfg.use_cache)


def run(
    spec: SceneSpec,
    cfg: RunConfig | None = None,
    machine: MachineConfig | None = None,
    params: RuntimeParams | None = None,
    spoken: list[Spoken] | None = None,
) -> RunResult:
    """Run a scene to the end and return everything that happened (not yet scored)."""
    cfg = cfg or RunConfig()
    machine = machine or MachineConfig()
    wall0 = time.perf_counter()
    ep = spec.plotter.emergency_parser if cfg.emergency_parser is None else cfg.emergency_parser
    if cfg.plotter == "controller":
        from pointdraw.virtual.lockstep import LockstepPlotter

        plotter: Any = LockstepPlotter(machine, emergency_parser=ep)
        plotter.start()
    else:
        plotter = VirtualPlotter(machine, emergency_parser=ep)
    try:
        return _run(spec, cfg, machine, params, spoken, plotter, wall0)
    finally:
        close = getattr(plotter, "close", None)
        if close is not None:
            close()


def _run(
    spec: SceneSpec,
    cfg: RunConfig,
    machine: MachineConfig,
    params: RuntimeParams | None,
    spoken: list[Spoken] | None,
    plotter: Any,
    wall0: float,
) -> RunResult:
    from pointdraw.virtual.checks import score

    paper = machine.paper_box
    origin = (paper.x, paper.y)
    lib = _library()
    lib._next.clear()  # variants cycle per category; start each run from the first one
    planner = Planner(lib)
    sheet = VirtualSheet(seed=cfg.seed)
    camera = VirtualCamera(sheet, plotter, origin, seed=cfg.seed)
    if cfg.cosmos == "live":
        from pointdraw.virtual.vision import LiveVision

        vision: Any = LiveVision()
    else:
        vision = MockVision(sheet)
    agent = make_agent(cfg, planner.categories)
    log = EventLog()

    def on_new_sheet(t: float) -> None:
        sheet.clear(t)

    rt = Runtime(
        agent,
        plotter,
        machine,
        planner,
        camera=camera,
        vision=vision,
        params=params,
        log=log,
        on_new_sheet=on_new_sheet,
    )
    _preload(spec, rt, plotter, sheet)
    hand = VirtualHand(spec.hand, paper, seed=cfg.seed, default_height=cfg.hover_mm)
    if spoken is None:
        if cfg.voice == "tts":
            from pointdraw.virtual.tts import tts_mode

            spoken = tts_mode(spec.lines, shift=cfg.shift_s)
        else:
            spoken = text_mode(spec.lines, shift=cfg.shift_s)
    starts = sorted(spoken, key=lambda s: s.t_start_heard)
    avail = sorted(spoken, key=lambda s: s.t_avail)
    events = sorted(spec.events, key=lambda e: e.at)
    inputs_end = max(
        [s.t_avail for s in spoken] + [hand.end_time()] + [e.at for e in events] + [0.0]
    )

    dt = 1.0 / FPS
    rt.start(0.0)
    i_start = i_avail = i_ev = 0
    k = 0
    t = 0.0
    idle_since: float | None = None
    track: list[tuple[float, Any]] = []
    while True:
        t = k * dt
        k += 1
        plotter.tick(t)
        while i_ev < len(events) and events[i_ev].at <= t:
            ev = events[i_ev]
            i_ev += 1
            if ev.user_mark is not None:
                tm = sheet.add(ev.user_mark, ev.at)
                log.add(ev.at, "user_mark", label=tm.label, box=_box(tm.mark.box_mm))
        while i_start < len(starts) and starts[i_start].t_start_heard <= t:
            rt.on_speech_start(starts[i_start].t_start_heard)
            i_start += 1
        while i_avail < len(avail) and avail[i_avail].t_avail <= t:
            rt.on_utterance(avail[i_avail].words, avail[i_avail].t_avail)
            i_avail += 1
        frame = hand.frame(t)
        track.append((t, hand.state(t)))
        rt.on_frame(frame)
        rt.tick(t)
        quiet = (
            rt.state in (LISTEN, CLARIFY)
            and not rt.pending
            and rt.proposal is None
            and rt.drawing is None
            and not plotter.busy
        )
        if t > inputs_end and quiet:
            idle_since = t if idle_since is None else idle_since
            if t - idle_since >= 1.0:
                break
        else:
            idle_since = None
        if t >= spec.max_time:
            log.add(t, "timeout", max_time=spec.max_time)
            break
    if isinstance(getattr(agent, "cache", None), ReplyCache) and cfg.record:
        agent.cache.save()
    res = RunResult(
        spec,
        cfg,
        log,
        rt,
        plotter,
        sheet,
        hand,
        camera,
        spoken,
        t,
        time.perf_counter() - wall0,
        track,
        agent,
    )
    res.report = score(res)
    return res


def _preload(spec: SceneSpec, rt: Runtime, plotter: Any, sheet: VirtualSheet) -> None:
    """What was on the sheet before the scene: earlier robot drawings and the user's marks."""
    ox, oy = rt.origin
    for item in spec.sheet.robot:
        if item.object:
            cat = rt.planner.resolve(item.object) or item.object
            placed = rt.planner.object(cat, item.at, item.size, id=item.id or rt.scene.new_id(cat))
            kind, label, text = "object", cat, None
        else:
            text = item.label or ""
            placed = rt.planner.label(text, item.at, item.size, id=item.id or f"label_{text}")
            kind, label = "label", None
        rt.scene.add(
            SceneObject(placed.id, "robot", label, placed.box, kind, placed.polylines, text, t=-1.0)
        )
        for pl in placed.polylines:
            for a, b in zip(pl, pl[1:], strict=False):
                plotter.ink.append(
                    InkSegment(-1.0, (a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy), placed.id)
                )
        rt.log.add(
            -1.0, "preloaded", id=placed.id, type=kind, label=label, text=text, box=_box(placed.box)
        )
    for u in spec.sheet.user:
        tm = sheet.add(u, -1.0)
        rt.log.add(-1.0, "user_mark", label=tm.label, box=_box(tm.mark.box_mm))


def _box(b: Any) -> list[float]:
    return [round(b.x, 2), round(b.y, 2), round(b.w, 2), round(b.h, 2)]


def over_paper(state: Any, paper_w: float, paper_h: float, margin: float = 10.0) -> bool:
    """Ground truth: is the virtual hand over the paper? (Fingertip or the hand behind it.)"""
    if not state.present:
        return False
    x, y = state.tip
    # The hand extends ~180 mm toward the user (-y) behind the fingertip.
    return -margin <= x <= paper_w + margin and y >= -margin and y - 180 <= paper_h + margin


def inf_or(v: float | None, default: float = math.inf) -> float:
    return default if v is None else v
