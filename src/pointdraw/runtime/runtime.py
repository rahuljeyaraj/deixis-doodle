"""The agent runtime (plan 4.4): a deterministic state machine around the agent's tool calls.

    LISTEN -> UNDERSTAND -> (CLARIFY) -> PREVIEW <-> ADJUST -> WAIT_HANDS_CLEAR -> DRAW -> LISTEN
    STOP works from any state. NEW_SHEET clears the scene.

Inputs, all stamped on one clock (real or simulated):

- `on_frame(HandsFrame)` every camera frame: holds, size gestures, the hands-clear interlock.
- `on_speech_start(t)` when the voice detector hears speech (blocks the auto-confirm).
- `on_utterance(words, t)` when a transcript with word times is ready.
- `tick(t)`: timers, agent results that are due, and the plotter's progress.

The agent (Nemotron, the recorded-reply cache or the rule parser) only returns typed tool
calls. In simulated time it answers at once and its reply is applied `latency_s` later. A
live agent may instead offer `submit(words, context) -> Future`: the model is then asked in
the background, so camera frames (and the hands-clear interlock) keep flowing, and `tick`
applies the reply when it arrives.

Places, sizes and objects are resolved here by the deixis resolver and the scene model; the
language model never produces coordinates or G-code (plan 4.1).

Confirmation: a request is previewed (the pen traces its box, pen up) and the guess is said
("A kite, about four centimetres?"), then drawn only when the person says "yes" (6 Oct
2026: `auto_confirm` is off). With `auto_confirm` on (the virtual world and replays) it is
also drawn after `confirm_quiet_s` of quiet once the trace is done. Corrections
("bigger", "a bit left", "no, there") during the preview change it and trace it again.
A new, different request during a preview drops that guess; it is never drawn (with
`auto_confirm` on it confirms it instead, since nothing is waiting on a person).

"Stop" is spotted in the transcript before the agent is called, so it doesn't wait for a
model reply.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from deixis.geometry import Box, Point
from deixis.resolver import ResolverParams
from deixis.types import Word
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.prompts import AgentContext
from pointdraw.agent.schema import (
    DeicticObject,
    DeicticPlace,
    RelationPlace,
    ToolCall,
    norm_word,
)
from pointdraw.controller.config import MachineConfig
from pointdraw.drawing import Placed, optimize
from pointdraw.drawing.place import circle_size
from pointdraw.profile import plotter_profile
from pointdraw.runtime.events import EventLog
from pointdraw.runtime.gestures import GestureTracker
from pointdraw.runtime.guess import guess_line, item_guess
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.references import (
    NeedAsk,
    Question,
    References,
    Utterance,
    adjust_offset,
    deictic_words,
)
from pointdraw.scene.ink import segment_ink
from pointdraw.scene.model import SceneModel, SceneObject, union_box

LISTEN, UNDERSTAND, CLARIFY, PREVIEW, ADJUST, WAIT_HANDS_CLEAR, DRAW = (
    "LISTEN",
    "UNDERSTAND",
    "CLARIFY",
    "PREVIEW",
    "ADJUST",
    "WAIT_HANDS_CLEAR",
    "DRAW",
)

STOP_WORDS = {"stop", "halt", "freeze"}
WRITE_VERBS = {"write", "label", "print"}
FILLERS = {"please", "um", "uh", "er", "erm", "okay", "ok", "oh", "now", "wait", "robot"}
CANCEL_ANSWERS = {"cancel", "nevermind", "never", "forget", "nothing"}
# Said on their own, these mean "yes" to a preview or a question, and nothing otherwise. On
# the Ender (1 Oct 2026) a lone "hello", "hmm" or "okay" came back as another place_drawing
# of the last object, drawn in the middle of the page.
ACKS = {"yes", "yeah", "yep", "yup", "sure", "right", "alright", "fine", "hmm", "hm", "mm"}
ACKS |= {"mhm", "hello", "hi", "hey", "so", "well", "and", "i", "mean", "just"}
# Words that let place_drawing repeat an object the person didn't name ("another one").
SIZE_WORDS = {"this", "that", "so", "big", "small", "large", "long", "wide", "tall", "size"}
REPEAT_WORDS = {"another", "again", "same", "more", "copy", "one", "ones"}
# Words that ask to move or resize the preview. Live 2 Oct: "I think" (half heard) came back
# as adjust(dx=-1, scale=0.8) and the cat moved 5 mm off rahul's point; an adjust needs one.
ADJUST_WORDS = {"left", "right", "up", "down", "north", "south", "east", "west", "move"}
ADJUST_WORDS |= {"shift", "higher", "lower", "back", "forward", "closer", "away", "further"}
ADJUST_WORDS |= {"bigger", "smaller", "larger", "wider", "narrower", "taller", "shorter"}
ADJUST_WORDS |= {"grow", "shrink", "half", "double", "twice", "less", "bit", "little"}
ADJUST_WORDS |= {"tiny", "big", "small", "large", "long", "wide", "tall", "size", "more"}
# Words that say a place relative to something, or a size. Live Nemotron (2 Oct, garden scene)
# answered "A sun up here" with "above the house, small": neither was said, and the pointing
# was lost. Without such words, a relation or a size preset in a call was made up.
RELATION_WORDS = {"next", "beside", "near", "left", "right", "above", "over", "under"}
RELATION_WORDS |= {"underneath", "beneath", "below", "between", "inside", "in", "on", "top"}
RELATION_WORDS |= {"by", "around", "behind", "front", "it", "them"}
PRESET_WORDS = {"tiny", "little", "small", "smaller", "medium", "big", "bigger", "large"}
PRESET_WORDS |= {"larger", "huge", "giant", "whole", "page", "full", "half"}
DEICTIC_WORDS = {"here", "there", "this", "that", "these", "those", "no"}
# A call may name the object on preview without saying it ("no, there", "a bit bigger"), but
# only if the utterance carries the correction itself.
CORRECTION_WORDS = DEICTIC_WORDS | ADJUST_WORDS | RELATION_WORDS | PRESET_WORDS | REPEAT_WORDS


class AgentPort(Protocol):
    def understand(self, words: list[str], context: AgentContext) -> AgentResult: ...


class FutureResult(Protocol):
    def done(self) -> bool: ...
    def result(self) -> AgentResult: ...


class CameraPort(Protocol):
    def paper_image(self, t: float) -> tuple[Any, float]:
        """A top-down image of the paper (user at the bottom) and its mm per pixel."""


class VisionPort(Protocol):
    def label(self, image: Any, mark: Any, mm_per_px: float) -> Any: ...  # str, or (str, also)

    def look(self, query: str, image: Any, mm_per_px: float, scene: SceneModel) -> str: ...


class PlotterPort(Protocol):
    """`PlotterController` plus what the runtime needs to follow jobs (VirtualPlotter)."""

    events: list[Any]
    busy: bool

    def preview_box(self, box: Box, tag: Any = None) -> None: ...
    def preview_circle(self, center: Point, size: float, tag: Any = None) -> None: ...
    def draw_polylines(self, polylines: list, speed: str = "normal", tag: Any = None) -> None: ...
    def park(self, tag: Any = None) -> None: ...
    def stop(self) -> None: ...
    def set_hands_clear(self, clear: bool) -> None: ...
    def ink_polylines(self, tag: Any = None) -> list: ...


NOT_UNDERSTOOD = "Sorry, I didn't catch that. Could you say it again?"


@dataclass
class RuntimeParams:
    default_size_mm: float = 40.0  # when no size is spoken or shown
    cross_size_mm: float = 10.0
    label_cap_mm: float = 5.0  # label cap height when no size is given
    label_gap_mm: float = 3.0
    size_wait_s: float = 6.0  # after "this big", how long to wait for the fingers to open
    auto_confirm: bool = False  # off (6 Oct): it draws only on "yes"; the virtual world turns it on
    confirm_quiet_s: float = 1.5  # 2.0 until 2 Oct (scripts/start_latency.py); with auto_confirm
    resolver: ResolverParams = field(default_factory=lambda: plotter_profile().resolver_params())


@dataclass
class Proposal:
    items: list[SceneObject]
    utt: int
    t: float
    previews: set[str] = field(default_factory=set)  # preview job tags still running
    preview_done: float | None = None
    drawing: set[str] = field(default_factory=set)  # draw job tags still running
    confirmed_t: float | None = None
    dirty: bool = False  # changed while a trace was running: trace again when it ends


@dataclass
class Pending:
    t_ready: float
    utt: Utterance
    result: AgentResult | None
    deferred: bool = False  # waiting for the pen to park before looking at the paper
    future: FutureResult | None = None  # a background agent call (result None until done)
    context: AgentContext | None = None


@dataclass
class OpenQuestion:
    q: Question
    utt: Utterance | None
    call: ToolCall | None
    t: float


class Runtime:
    def __init__(
        self,
        agent: AgentPort,
        plotter: PlotterPort,
        machine: MachineConfig,
        planner: Planner | None = None,
        *,
        camera: CameraPort | None = None,
        vision: VisionPort | None = None,
        params: RuntimeParams | None = None,
        log: EventLog | None = None,
        on_new_sheet: Callable[[float], None] | None = None,
    ) -> None:
        self.agent = agent
        self.plotter = plotter
        self.machine = machine
        self.params = params or RuntimeParams()
        paper = machine.paper_box
        self.origin: Point = (paper.x, paper.y)
        d = machine.drawable
        self.drawable = Box(d.x - paper.x, d.y - paper.y, d.w, d.h)  # paper frame
        self.planner = planner or Planner(drawable=self.drawable)
        if self.planner.drawable is None:
            self.planner.drawable = self.drawable
        self.scene = SceneModel(self.drawable)
        self.tracker = GestureTracker(paper)
        self.camera = camera
        self.vision = vision
        self.log = log or EventLog()
        self.on_new_sheet = on_new_sheet
        self.state = LISTEN
        self.t = 0.0
        self.pending: list[Pending] = []
        self.proposal: Proposal | None = None
        self.queue: list[Proposal] = []
        self.drawing: Proposal | None = None
        self.question: OpenQuestion | None = None
        self.speech_active = False
        self.last_heard = -math.inf
        self.utts: list[Utterance] = []
        self.samples: list[Any] = []
        self._ev = 0  # plotter events read so far
        self._clear: bool | None = None
        self.backlog: list[Utterance] = []  # heard while an earlier one was being understood
        self.asked: ToolCall | None = None  # the call of a question just left unanswered
        self._rel: dict[str, tuple[str, str]] = {}  # item id -> (relation, target's label)
        self.size_wait: tuple[float, float] | None = None  # "this big" said at, wait until
        self.plotter_problem: str | None = None  # the plotter's link failed (until plotter_back)
        self._tags = 0

    # ------------------------------------------------------------------ inputs

    def start(self, t: float) -> None:
        self.t = t
        self.log.add(t, "state", state=self.state)
        self.refresh_scene(t)

    def on_frame(self, frame: Any) -> None:
        s = self.tracker.update(frame)
        self.samples.append(s)
        self.plotter.set_hands_clear(self.tracker.hands_clear)
        if self.tracker.hands_clear != self._clear:
            self._clear = self.tracker.hands_clear
            self.log.add(frame.t, "interlock", clear=self._clear)
        h = self.tracker.ended_hold
        if h is not None:
            self.log.add(
                h.t1,
                "hold",
                t0=round(h.t0, 3),
                t1=round(h.t1, 3),
                mean=[round(h.mean[0], 2), round(h.mean[1], 2)],
                spread=round(max(h.spread), 2),
            )
        sp = self.tracker.ended_span
        self._check_size_wait(frame.t, sp)
        if sp is not None:
            self.log.add(
                sp.t1,
                "span",
                t0=round(sp.t0, 3),
                t1=round(sp.t1, 3),
                span=round(sp.span_mm, 2),
                span_kind=sp.kind,
            )

    def on_speech_start(self, t: float) -> None:
        self.speech_active = True
        self.log.add(t, "speech_start")

    def on_utterance(self, words: list[Word], t: float) -> None:
        self.t = max(self.t, t)
        self.speech_active = False
        self.last_heard = t
        fixed = repair_words(list(words), self.planner.categories)
        if [w.text for w in fixed] != [w.text for w in words]:
            self.log.add(
                t,
                "asr_repair",
                heard=" ".join(w.text for w in words),
                fixed=" ".join(w.text for w in fixed),
            )
        utt = Utterance(len(self.utts), fixed, t)
        self.utts.append(utt)
        self.log.add(
            t,
            "utterance",
            id=utt.id,
            text=utt.text,
            words=[[w.text, round(w.t0, 3), round(w.t1, 3)] for w in words],
        )
        if not words:
            return
        if self._is_noise(utt):
            self.log.add(t, "ignored", utt=utt.id, why="only fillers, and nothing to answer")
            return
        if self._is_stop(utt):
            self.log.add(
                t,
                "tool_calls",
                utt=utt.id,
                source="fast_path",
                latency_s=0.0,
                calls=[{"name": "stop", "args": {}}],
            )
            self.do_stop(t, "stop word")
            return
        if self.pending:
            # One utterance at a time: "draw an ear here ... about this big" needs the ear's
            # preview in the context of the second call (live 30 Sep: a race otherwise).
            self.backlog.append(utt)
            self.log.add(t, "waiting", utt=utt.id, behind=[p.utt.id for p in self.pending])
            return
        self._understand(utt, t)

    def _understand(self, utt: Utterance, t: float) -> None:
        if self.question is not None and self._answer(utt, t):
            return
        if not self.plotter.busy:
            self.refresh_scene(t)
        ctx = self.context()
        submit = getattr(self.agent, "submit", None)
        if submit is not None:
            fut = submit(utt.tokens(), ctx)
            self.pending.append(Pending(math.inf, utt, None, future=fut, context=ctx))
        else:
            res = self.agent.understand(utt.tokens(), ctx)
            self._log_calls(t, utt, res, ctx)
            self.pending.append(Pending(t + res.latency_s, utt, res))
        if self.state == LISTEN:
            self._set_state(UNDERSTAND, t)

    def _log_calls(self, t: float, utt: Utterance, res: AgentResult, ctx: AgentContext) -> None:
        self.log.add(
            t,
            "tool_calls",
            utt=utt.id,
            source=res.source,
            model=res.model,
            latency_s=round(res.latency_s, 3),
            calls=[c.to_dict() for c in res.calls],
            speech=res.speech,
            context=ctx.lines(),
        )

    def tick(self, t: float) -> None:
        self.t = max(self.t, t)
        self._plotter_events()
        self._check_size_wait(t)
        for p in self.pending:
            if p.result is None and p.future is not None and p.future.done():
                p.result, p.t_ready = p.future.result(), t
                assert p.context is not None
                self._log_calls(t, p.utt, p.result, p.context)
        due = [p for p in self.pending if p.t_ready <= t]
        for p in due:
            assert p.result is not None
            if self.plotter.busy and _needs_view(p.result.calls):
                # "Put a hand inside that" while the pen is drawing: finish, park, look at the
                # paper (the user may have just drawn something), then resolve.
                if not p.deferred:
                    p.deferred = True
                    self.log.add(t, "deferred", utt=p.utt.id, why="wait for the pen to park")
                continue
            self.pending.remove(p)
            if p.deferred:
                self.refresh_scene(t)
            self._apply(p.utt, p.result, max(p.t_ready, t) if p.deferred else p.t_ready)
        while self.backlog and not self.pending:
            self._understand(self.backlog.pop(0), t)
        self._auto_confirm(t)
        if self.state == UNDERSTAND and not self.pending:
            self._set_state(self._rest_state(), t)

    # ------------------------------------------------------------------ context

    def context(self) -> AgentContext:
        last = self.scene.last()
        prev = None
        if self.proposal is not None and self.proposal.items:
            it = self.proposal.items[-1]
            tool = {"object": "place_drawing", "label": "write_label", "arrow": "draw_connector"}
            prev = f"{tool.get(it.kind, 'place_drawing')}({it.label or it.text or it.kind})"
        ctx = AgentContext(
            scene_labels=self.scene.labels(),
            last_label=last.label if last else None,
            previewing=prev,
        )
        if self.question is not None and hasattr(ctx, "pending_question"):
            ctx.pending_question = self.question.q.text  # type: ignore[attr-defined]
            ctx.options = list(self.question.q.options)  # type: ignore[attr-defined]
        return ctx

    def refresh_scene(self, t: float) -> None:
        """Look at the paper: segment the user's marks (minus the robot's own ink)."""
        if self.camera is None:
            return
        img, mmpp = self.camera.paper_image(t)
        robot = [[self._to_paper(p) for p in pl] for pl in self.plotter.ink_polylines()]
        marks = segment_ink(img, mmpp, robot)

        def labeler(m: Any) -> Any:
            return self.vision.label(img, m, mmpp) if self.vision else None

        added = self.scene.set_user_marks(marks, labeler, t)
        self.log.add(
            t,
            "scene",
            marks=len(self.scene.marks),
            new=[
                {"id": o.id, "label": o.label, "also": o.attrs.get("also"), "box": _box(o.box)}
                for o in added
            ],
        )

    # ------------------------------------------------------------------ agent results

    def _apply(self, utt: Utterance, res: AgentResult, t: float) -> None:
        if not res.calls:
            self.say(t, res.speech or NOT_UNDERSTOOD)
            return
        calls = split_shared_labels(res.calls)
        if calls != res.calls:
            self.log.add(t, "guard", said=utt.text, why="one label text shared by several labels")
        named = [c for c in calls if not self._invented(c, utt)]
        self.asked = None
        if len(named) < len(calls):
            self.log.add(t, "guard", said=utt.text, why="place_drawing of an object not said")
            calls = named
            if not calls:
                self.say(t, NOT_UNDERSTOOD)
                return
        if any(c.name == "adjust" for c in calls):
            if self._resize_to_gesture(utt, t):
                return
            toks = {norm_word(w.text) for w in utt.words}
            if not toks & ADJUST_WORDS:
                self.log.add(t, "guard", said=utt.text, why="adjust with no move or size word")
                calls = [c for c in calls if c.name != "adjust"]
                if not calls:
                    return  # stay quiet: it was not meant for the robot
        calls = [self._grounded(c, utt, t) for c in calls]
        refs = self._refs(utt)
        refs.prepare(calls)
        new: list[SceneObject] = []
        for call in calls:
            try:
                self._handle(call, refs, new, t)
            except NeedAsk as e:
                self._ask(e.q, utt, call, t)
                break
            except Exception as e:  # noqa: BLE001 - one bad call must not stop the app
                self.log.add(t, "error", tool=call.name, why=f"{type(e).__name__}: {e}")
                self.say(t, "Sorry, I couldn't draw that.")
                break
            finally:
                for entry in refs.log:
                    self.log.add(t, "resolve", utt=utt.id, tool=call.name, **entry)
                refs.log.clear()
        if new:
            self._propose(new, utt.id, t)

    def _refs(
        self, utt: Utterance, overrides: dict | None = None, sizes: dict | None = None
    ) -> References:
        holds = self.tracker.holds.holds()
        spans = self.tracker.spans.holds()
        return References(
            utt,
            holds,
            spans,
            self.scene,
            self.params.resolver,
            self.params.default_size_mm,
            overrides,
            sizes,
        )

    def _handle(self, call: ToolCall, refs: References, new: list[SceneObject], t: float) -> None:
        refs.current = call
        name, a = call.name, call.args
        if name == "stop":
            self.do_stop(t, "tool call")
        elif name == "new_sheet":
            self.new_sheet(t)
        elif name == "confirm":
            self.confirm(t)
        elif name == "cancel":
            self.cancel(t)
        elif name == "adjust":
            # "A brain up here, a bit bigger": the adjust is for the item just placed.
            target = new[-1] if new else None
            self.adjust(a.dx, a.dy, a.scale, t, target)  # type: ignore[attr-defined]
        elif name == "ask_user":
            q = Question(a.question, "agent", list(a.options))  # type: ignore[attr-defined]
            self._ask(q, refs.utt, None, t)
        elif name == "look":
            self.look(a.query, t)  # type: ignore[attr-defined]
        elif name == "place_drawing":
            self._place_drawing(call, refs, new, t)
        elif name == "write_label":
            self._write_label(call, refs, new, t)
        elif name == "draw_connector":
            self._connector(call, refs, new, t)
        else:
            self.say(t, f"I can't do '{name}' yet.")

    # ------------------------------------------------------------------ drawing requests

    def _drop_preview_for(self, t: float) -> None:
        """A different request arrived during a preview: drop that guess, never draw it.

        Until 6 Oct this *confirmed* the waiting guess, so one misheard word turned into ink:
        in rahul's run that day "Beaker" drew the car he was still being asked about, and the
        next request drew the label. Asking for something else means he changed his mind, not
        that he said yes. "Yes" and "cancel" are the only ways a guess becomes ink (claim 16).
        """
        prop = self.proposal
        if prop is None:
            return
        if self.params.auto_confirm:
            # Nothing is waiting on a person here: the guess was going to be drawn anyway once
            # the quiet ran out, so a new request does not take it back (the virtual world).
            self.confirm(t, reason="new request")
            return
        for obj in prop.items:
            self.scene.remove(obj)
        self.proposal = None
        self.log.add(t, "dropped", items=[o.id for o in prop.items], why="new request")

    def _place_drawing(self, call: ToolCall, refs: References, new: list, t: float) -> None:
        a = call.args
        cat = self.planner.resolve(a.object)  # type: ignore[attr-defined]
        if cat is None:
            obj = a.object  # type: ignore[attr-defined]
            opts = self.planner.suggestions(obj)
            text = f"I can't draw a {obj} yet." + (
                f" Would a {' or a '.join(opts)} do?" if opts else " What else shall I draw?"
            )
            raise NeedAsk(Question(text, "unknown_object", opts))
        prev = self._previewed_item()
        if (
            prev is not None
            and prev.kind == "object"
            and prev.label != cat
            and not new
            and not self._named(cat, refs.utt)
        ):
            # Live 30 Sep: "about this big" while an ear was previewed came back as a
            # *circle* this big. Nobody said "circle": it means the previewed object.
            self.log.add(t, "guard", said=refs.utt.text, model_object=cat, used=prev.label)
            cat = prev.label or cat
        if prev is not None and prev.label == cat and not new:
            # "no, there" / "about this big" / "put it here": change the preview in place.
            size = refs.size(a.size, prev.size) if a.size else None  # type: ignore[attr-defined]
            if a.place is not None:  # type: ignore[attr-defined]
                s = size or prev.size
                pl = refs.place(a.place, s)  # type: ignore[attr-defined]
                center, where = pl.center, pl.where.model_dump(mode="json")
                s = pl.size or s
            else:
                center, where, s = prev.aim, None, size or prev.size
            again = self.planner.object(cat, center, s, id=prev.id, variant=prev.key_id)
            self._replace(prev, again, t, where)
            return
        if prev is not None or (self.proposal is not None and not new):
            self._drop_preview_for(t)
        default = self.params.cross_size_mm if cat == "cross" else None
        size = refs.size(a.size, default)  # type: ignore[attr-defined]
        pl = refs.place(a.place, size)  # type: ignore[attr-defined]
        size = pl.size or size
        style = getattr(a, "style", None)  # "simple" / "detailed", or the library default
        placed = self.planner.object(cat, pl.center, size, id=self.scene.new_id(cat), style=style)
        rel = getattr(a.place, "relation", None)  # type: ignore[attr-defined]
        if rel in ("next_to", "left_of", "right_of", "above", "below", "on") and pl.target:
            # Re-place with the drawing's real width and height, so the gap is exact.
            ext = (placed.box.w, placed.box.h)
            c = self.scene.relation(rel, pl.target, size, extent=ext)
            placed = self.planner.object(cat, c, size, id=placed.id, variant=placed.key_id)
        obj = self._add_item(
            placed, "object", cat, None, pl.where.model_dump(mode="json"), new, t, target=pl.target
        )
        if rel in ("next_to", "left_of", "right_of", "above", "below", "on") and pl.target:
            # the word the person used ("the house"), else the thing's own label
            said = getattr(getattr(a.place, "target", None), "label", None)  # type: ignore[attr-defined]
            name = said or pl.target.label
            if name:
                self._rel[obj.id] = (rel, str(name).lower().removeprefix("the "))

    def _write_label(self, call: ToolCall, refs: References, new: list, t: float) -> None:
        a = call.args
        text: str = a.text  # type: ignore[attr-defined]
        if self._previewed_item() is not None and not new:
            self._drop_preview_for(t)
        cap = self.params.label_cap_mm
        if a.size is not None:  # type: ignore[attr-defined]
            s = refs.size(a.size)  # type: ignore[attr-defined]
            probe = self.planner.label(text, (0, 0), cap)
            cap = cap * s / max(probe.box.w, 1e-6)
        probe = self.planner.label(text, (0, 0), cap)
        w, h = probe.box.w, probe.box.h
        g = self.params.label_gap_mm
        place = a.place  # type: ignore[attr-defined]
        target: SceneObject | None = None
        where: dict | None = None
        if place is None or isinstance(place, DeicticPlace):
            if place is None:
                p, wh = refs.last_point()
            else:
                p, wh = refs.point_for_word(place.word_index)
            where = wh.model_dump(mode="json")
            if place is None or place.deixis in ("this", "that", "these", "those"):
                target, cands = self.scene.object_at(p)
                if target is None and len(cands) > 1:
                    opts = [c.label or c.id for c in cands]
                    raise NeedAsk(
                        Question(
                            "The " + " or the ".join(opts) + "?", "ambiguous_object", opts, cands
                        )
                    )
            center = (target.center[0], target.box.y - g - h / 2) if target else p
        elif isinstance(place, RelationPlace) and place.relation != "between":
            target = refs.obj(place.target)
            tb = target.box
            cx, cy = tb.center
            center = {
                "below": (cx, tb.y - g - h / 2),
                "next_to": (cx, tb.y - g - h / 2),
                "above": (cx, tb.y2 + g + h / 2),
                "left_of": (tb.x - g - w / 2, cy),
                "right_of": (tb.x2 + g + w / 2, cy),
                "inside": (cx, cy),
                "on": (cx, cy),
            }[place.relation]
            where = {"relation": place.relation, "target": target.id}
        else:
            pl = refs.place(place, w)
            center, where = pl.center, pl.where.model_dump(mode="json")
        placed = self.planner.label(text, center, cap, id=self.scene.new_id(f"label_{text}"))
        self._add_item(placed, "label", None, text, where, new, t, target=target)

    def _connector(self, call: ToolCall, refs: References, new: list, t: float) -> None:
        a = call.args
        if self._previewed_item() is not None and not new:
            self._drop_preview_for(t)
        ends = []
        for ref in (a.from_, a.to):  # type: ignore[attr-defined]
            if isinstance(ref, DeicticObject):
                p, _ = refs.point_for_word(ref.word_index)
                o, cands = self.scene.object_at(p)
                if o is None and len(cands) > 1:
                    opts = [c.label or c.id for c in cands]
                    raise NeedAsk(
                        Question(
                            "The " + " or the ".join(opts) + "?", "ambiguous_object", opts, cands
                        )
                    )
                ends.append((o.center, o.box, o.id) if o else (p, None, None))
            else:
                o = refs.obj(ref)
                ends.append((o.center, o.box, o.id))
        (pa, ba, ia), (pb, bb, ib) = ends
        if (ia is not None and ia == ib) or math.dist(pa, pb) < 2.0:
            # Both words found the same thing (night 2 Oct stress test, S12 seed 7: the
            # second point was lost, both took the first; the arrow code then raised).
            raise NeedAsk(
                Question("From where to where? Point at one, then the other.", "same_ends")
            )
        kind = a.kind  # type: ignore[attr-defined]
        placed = self.planner.arrow(pa, pb, ba, bb, kind, id=self.scene.new_id(kind))
        where = {
            "kind": "path",
            "from": ia or [round(pa[0], 2), round(pa[1], 2)],
            "to": ib or [round(pb[0], 2), round(pb[1], 2)],
        }
        self._add_item(placed, "arrow", None, None, where, new, t, anchor=f"{ia}->{ib}")

    def _add_item(
        self,
        placed: Placed,
        kind: str,
        label: str | None,
        text: str | None,
        where: dict | None,
        new: list,
        t: float,
        target: SceneObject | None = None,
        anchor: str | None = None,
    ) -> SceneObject:
        placed, cut = self.planner.clip_inside(placed)
        obj = SceneObject(
            placed.id,
            "robot",
            label,
            placed.box,
            kind,
            placed.polylines,
            text,
            t,
            pending=True,
            anchor=anchor or (target.id if target else None),
            key_id=placed.key_id,
            size_circle=placed.size_circle,
        )
        self.scene.add(obj)
        new.append(obj)
        self.log.add(
            t,
            "item",
            id=obj.id,
            type=kind,
            label=label,
            text=text,
            box=_box(obj.box),
            center=_pt(obj.center),
            circle=_circle(obj.size_circle),
            where=where,
            cut=cut,
        )
        if cut and not placed.polylines:
            self.say(t, "That's off the page, so I can't draw it there.")
        elif cut:
            self.say(t, "Part of it goes off the page, so I'll leave that part out.")
        return obj

    def _replace(self, old: SceneObject, placed: Placed, t: float, where: dict | None) -> None:
        placed, cut = self.planner.clip_inside(placed)
        old.polylines, old.box, old.key_id = placed.polylines, placed.box, placed.key_id
        old.size_circle = placed.size_circle
        self.log.add(
            t,
            "item",
            id=old.id,
            type=old.kind,
            label=old.label,
            text=old.text,
            box=_box(old.box),
            center=_pt(old.center),
            circle=_circle(old.size_circle),
            where=where,
            cut=cut,
            replaced=True,
        )
        self._preview(t)

    def _named(self, cat: str, utt: Utterance) -> bool:
        """Whether the person said this category (or a synonym of it) in the utterance."""
        toks = [norm_word(w.text) for w in utt.words]
        spans = toks + [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
        return any(self.planner.resolve(x) == cat for x in spans if x)

    def _previewed_item(self) -> SceneObject | None:
        if self.proposal is None or not self.proposal.items:
            return None
        return self.proposal.items[-1]

    # ------------------------------------------------------------------ preview and draw

    def _propose(self, items: list[SceneObject], utt: int, t: float) -> None:
        prop = Proposal(items, utt, t)
        if self.drawing is not None or self.proposal is not None:
            self.queue.append(prop)
            self.log.add(t, "queued", items=[o.id for o in items])
            return
        self.proposal = prop
        self._preview(t)

    def _preview(self, t: float) -> None:
        prop = self.proposal
        if prop is None:
            return
        prop.preview_done = None
        if prop.previews:  # a trace is still running: trace the new box when it ends
            prop.dirty = True
        else:
            self._trace(prop)
        self.log.add(
            t,
            "preview",
            items=[o.id for o in prop.items],
            box=_box(union_box([o.box for o in prop.items])),
            circles=[_circle(o.size_circle) for o in prop.items if o.size_circle],
        )
        self._set_state(PREVIEW, t)
        self.say(t, self._guess(prop))

    def _guess(self, prop: Proposal) -> str:
        """The guess in plain words, "A kite, about four centimetres?" (claim 16)."""
        parts = []
        for o in prop.items:
            rel, target = self._rel.get(o.id, (None, None))
            parts.append(item_guess(o.kind, o.label, o.text, o.size, rel, target))
        return guess_line(parts)

    def _trace(self, prop: Proposal) -> None:
        """Show each item with the pen up: a drawing's circle (the "here" and "this big" it
        was asked for; rahul, 6 Oct), else the box of the words or the arrow."""
        prop.dirty = False
        for obj in prop.items:
            tag = self._tag("preview")
            if obj.size_circle is not None:
                c, d = obj.size_circle
                self.plotter.preview_circle(self._to_bed(c), d, tag=tag)
            else:
                self.plotter.preview_box(self._box_to_bed(obj.box), tag=tag)
            prop.previews.add(tag)

    def adjust(
        self, dx: int, dy: int, scale: float, t: float, target: SceneObject | None = None
    ) -> None:
        obj = target or self._previewed_item()
        if obj is None:
            self.say(t, "There's nothing to adjust right now.")
            return
        self._set_state(ADJUST, t)
        ox, oy = adjust_offset(self.scene, obj.box, dx, dy, scale)  # the step is the core's
        if obj.kind == "object" and obj.size_circle is not None:
            # placed again from its circle, so a part cut off the page comes back when it
            # shrinks, and "here" stays put
            (cx, cy), d = obj.size_circle
            p = self.planner.object(
                obj.label or obj.kind, (cx + ox, cy + oy), d * scale, id=obj.id, variant=obj.key_id
            )
        else:
            p = Placed(obj.id, obj.kind, obj.polylines, obj.box, obj.label)
            p = self.planner.shift(p, ox, oy, scale)
        self.log.add(t, "adjust", id=obj.id, dx=dx, dy=dy, scale=scale)
        if target is not None:  # not previewed yet: just change it
            p, _ = self.planner.clip_inside(p)
            obj.polylines, obj.box, obj.size_circle = p.polylines, p.box, p.size_circle
            self.log.add(
                t,
                "item",
                id=obj.id,
                type=obj.kind,
                label=obj.label,
                text=obj.text,
                box=_box(obj.box),
                center=_pt(obj.center),
                circle=_circle(obj.size_circle),
                where=None,
                cut=False,
                replaced=True,
            )
            return
        self._replace(obj, p, t, None)

    def confirm(self, t: float, reason: str = "confirm") -> None:
        prop = self.proposal
        if prop is None:
            if self.state != DRAW:
                self.say(t, "There's nothing to confirm.")
            return
        self.proposal = None
        prop.confirmed_t = t
        for obj in prop.items:
            obj.pending = False
            obj.t = t
            poly, _ = optimize(obj.polylines)
            bed = [[self._to_bed(p) for p in pl] for pl in poly]
            self.plotter.draw_polylines(bed, "normal", tag=obj.id)
            prop.drawing.add(obj.id)
        park = self._tag("park")
        self.plotter.park(tag=park)
        prop.drawing.add(park)
        self.drawing = prop
        self.log.add(t, "confirm", items=[o.id for o in prop.items], reason=reason)
        self._set_state(WAIT_HANDS_CLEAR, t)
        if self.queue:  # the next proposal is previewed once this one is drawn
            pass

    def cancel(self, t: float) -> None:
        if self.question is not None:
            self.question = None
            self.say(t, "OK, never mind.")
        prop = self.proposal
        if prop is not None:
            for obj in prop.items:
                self.scene.remove(obj)
            self.proposal = None
            self.log.add(t, "cancel", items=[o.id for o in prop.items])
            self.say(t, "Cancelled.")
        self._set_state(self._rest_state(), t)

    def do_stop(self, t: float, why: str) -> None:
        self.plotter.stop()
        dropped = []
        for prop in ([self.proposal] if self.proposal else []) + self.queue:
            for obj in prop.items:
                if obj.pending:
                    self.scene.remove(obj)
                    dropped.append(obj.id)
        self.proposal, self.queue, self.pending, self.question = None, [], [], None
        self.backlog = []
        if self.drawing is not None:
            self.drawing = None
        self.log.add(t, "stop", why=why, dropped=dropped)
        self.say(t, "Stopped.")
        self._set_state(LISTEN, t)

    def _plotter_failed(self, t: float, why: str) -> None:
        """The plotter's link failed (a pulled cable): drop what waited for it, say so once."""
        dropped = []
        for prop in ([self.proposal] if self.proposal else []) + self.queue:
            for obj in prop.items:
                if obj.pending:
                    self.scene.remove(obj)
                    dropped.append(obj.id)
        self.proposal, self.queue, self.drawing = None, [], None
        self.log.add(t, "plotter_problem", why=why, dropped=dropped)
        if self.plotter_problem is None:
            self.say(t, "The plotter isn't answering. Check its cable.")
        self.plotter_problem = why or "link failed"
        self._set_state(self._rest_state(), t)

    def plotter_back(self, t: float) -> None:
        """The link works again (the live app reconnected it)."""
        self.plotter_problem = None
        self.log.add(t, "plotter_back")
        self.say(t, "The plotter is back. It will home before it draws.")

    def new_sheet(self, t: float) -> None:
        if self.plotter.busy:
            self.plotter.stop()
        self.proposal, self.queue, self.drawing, self.question = None, [], None, None
        self.scene.clear()
        if self.on_new_sheet is not None:
            self.on_new_sheet(t)
        self.log.add(t, "new_sheet")
        self.say(t, "New page. Ready when you are.")
        self._set_state(LISTEN, t)

    def look(self, query: str, t: float) -> None:
        if self.vision is None or self.camera is None:
            self.say(t, "I can't see the paper right now.")
            return
        self.say(t, "Let me look.")
        img, mmpp = self.camera.paper_image(t)
        answer = self.vision.look(query, img, mmpp, self.scene)
        self.log.add(t, "look", query=query, answer=answer)
        self.say(t, answer)
        self._set_state(self._rest_state(), t)

    def _auto_confirm(self, t: float) -> None:
        p = self.params
        prop = self.proposal
        if not p.auto_confirm or prop is None or self.state != PREVIEW:
            return
        if prop.preview_done is None or self.pending or self.speech_active:
            return
        if self.question is not None:
            return
        if t - max(prop.preview_done, self.last_heard) >= p.confirm_quiet_s:
            self.confirm(t, reason="quiet")

    def _plotter_events(self) -> None:
        evs = self.plotter.events
        while self._ev < len(evs):
            e = evs[self._ev]
            self._ev += 1
            tag = e.tag
            prop = self.proposal
            if (
                prop is not None
                and tag in prop.previews
                and e.kind in ("job_end", "halted", "error")
            ):
                prop.previews.discard(tag)
                if not prop.previews and prop.dirty and e.kind == "job_end":
                    self._trace(prop)
                elif not prop.previews:
                    prop.preview_done = e.t
                    self.log.add(e.t, "preview_done", items=[o.id for o in prop.items])
            d = self.drawing
            if d is not None and tag in d.drawing:
                if e.kind == "job_start" and self.state == WAIT_HANDS_CLEAR:
                    self._set_state(DRAW, e.t)
                if e.kind == "job_end":
                    d.drawing.discard(tag)
                    obj = next((o for o in d.items if o.id == tag), None)
                    if obj is not None:
                        self._drawn(obj, e.t)
                    if not d.drawing:
                        self.drawing = None
                        self._next(e.t)
            if e.kind == "halted":
                self.drawing = None
            if e.kind == "error":
                self._plotter_failed(e.t, str(e.detail.get("message", "")))

    def _drawn(self, obj: SceneObject, t: float) -> None:
        ink = self.plotter.ink_polylines(obj.id)
        pts = [self._to_paper(p) for pl in ink for p in pl]
        box = None
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            box = Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self.log.add(
            t,
            "drawn",
            id=obj.id,
            type=obj.kind,
            label=obj.label,
            text=obj.text,
            planned_box=_box(obj.box),
            ink_box=_box(box) if box else None,
            ink_center=_pt(box.center) if box else None,
            ink_size=round(circle_size([pts]), 3) if pts else None,  # its circle's diameter
        )

    def _next(self, t: float) -> None:
        if self.queue:
            self.proposal = self.queue.pop(0)
            self._preview(t)
        else:
            self._set_state(self._rest_state(), t)
            self.refresh_scene(t)

    # ------------------------------------------------------------------ questions

    def _ask(self, q: Question, utt: Utterance | None, call: ToolCall | None, t: float) -> None:
        self.question = OpenQuestion(q, utt, call, t)
        self.log.add(t, "ask", question=q.text, reason=q.reason, options=q.options)
        self.say(t, q.text)
        self._set_state(CLARIFY, t)

    def _answer(self, utt: Utterance, t: float) -> bool:
        """Try to answer the open question from this utterance. True if it was handled."""
        oq = self.question
        assert oq is not None
        toks = [norm_word(w.text) for w in utt.words]
        if set(toks) & CANCEL_ANSWERS or toks[:1] == ["no"] and len(toks) == 1:
            self.cancel(t)
            return True
        q = oq.q
        chosen = None
        for opt in q.options:
            if all(p in toks for p in opt.lower().split()) or norm_word(opt) in toks:
                chosen = opt
                break
        if chosen is not None and oq.call is not None:
            self.question = None
            self.log.add(t, "answer", question=q.text, answer=chosen)
            call = oq.call
            overrides: dict = {}
            if q.reason == "unknown_object":
                call = ToolCall(
                    name=call.name, args=call.args.model_copy(update={"object": chosen})
                )
            elif q.candidates:
                obj = next(c for c in q.candidates if (c.label or c.id) == chosen)
                call = _retarget(call, obj)
            self._rerun(oq.utt, call, t, overrides)
            return True
        if q.reason == "no_size" and oq.call is not None and oq.utt is not None:
            # "How big?" ... "this big": the size comes from this answer's gesture, the rest
            # from the request (on the Ender the agent, asked alone, guessed a new object).
            span = self._answer_span(utt)
            if span is not None:
                self._size_answer(oq, span, t)
                return True
            if set(toks) & (SIZE_WORDS - {"this", "that", "so"}):
                # "this big" with no gesture yet: on the Ender the fingers often opened a few
                # seconds after the words. The next steady span answers; else ask again.
                said = utt.words[0].t0 if utt.words else t
                self.size_wait = (said, t + self.params.size_wait_s)
                self.log.add(t, "waiting", why="size gesture after 'this big'")
                return True
        demos = [i for i, w in enumerate(toks) if w in ("here", "there", "this", "that")]
        if q.reason in ("no_gesture", "unsteady", "ambiguous") and demos and oq.call is not None:
            refs = self._refs(utt)
            try:
                _, where = refs.point_for_word(demos[-1])
            except NeedAsk as e:
                self._ask(e.q, oq.utt, oq.call, t)
                return True
            pick = refs.picks[demos[-1]]
            self.question = None
            self.log.add(t, "answer", question=q.text, answer="pointing", where=where.mean_mm)
            assert oq.utt is not None
            idx = _deictic_indices(oq.call) or [-1]
            self._rerun(oq.utt, oq.call, t, {i: pick.hold for i in idx})  # type: ignore[union-attr]
            return True
        if chosen is not None and oq.call is None:
            return False  # an agent question: let the agent read the answer in context
        self.question = None  # not an answer: treat it as a new request
        self.asked = oq.call  # but the agent may read it as one ("How big?" ... "this big")
        return False

    def _resize_to_gesture(self, utt: Utterance, t: float) -> bool:
        """A lone "this big" with the fingers open, while an object is previewed: make it
        that size where it is. Live 2 Oct: "draw a cat here" ... "this big" was sent to the
        agent alone, which answered adjust(scale=0.8) and ignored rahul's 26 mm pinch."""
        prev = self._previewed_item()
        toks = {norm_word(w.text) for w in utt.words}
        if (
            prev is None
            or prev.kind != "object"
            or not toks & (SIZE_WORDS - {"this", "that", "so"})
        ):
            return False
        span = self._answer_span(utt)
        if span is None:
            return False
        self.log.add(
            t,
            "guard",
            said=utt.text,
            why="size gesture: resize the preview",
            size=round(span.span_mm, 1),
        )
        again = self.planner.object(
            prev.label or prev.kind, prev.aim, span.span_mm, id=prev.id, variant=prev.key_id
        )
        self._replace(prev, again, t, None)
        return True

    def _answer_span(self, utt: Utterance) -> Any:
        """The size gesture (a SpanHold) shown with an answer like "this big", if any."""
        refs = self._refs(utt)
        for i, w in enumerate(utt.words):
            if norm_word(w.text) in SIZE_WORDS:
                span = refs.measure(i)
                if span is not None:
                    return span
        return None

    def _size_answer(self, oq: OpenQuestion, span: Any, t: float) -> None:
        """Answer "How big?" with a size gesture: rerun the asked call with that size."""
        assert oq.call is not None and oq.utt is not None
        self.question, self.size_wait = None, None
        self.log.add(t, "answer", question=oq.q.text, answer="size", size=round(span.span_mm, 1))
        _, sizes = deictic_words(oq.call.args)
        self._rerun(oq.utt, oq.call, t, {}, {i: span for i in sizes})

    def _check_size_wait(self, t: float, span: Any = None) -> None:
        if self.size_wait is None:
            return
        oq = self.question
        if oq is None or oq.q.reason != "no_size":
            self.size_wait = None
            return
        said, deadline = self.size_wait
        if span is not None and span.t1 >= said - 1.0:
            self._size_answer(oq, span, t)
        elif t > deadline:
            self.size_wait = None
            self._ask(oq.q, oq.utt, oq.call, t)

    def _rerun(
        self,
        utt: Utterance | None,
        call: ToolCall,
        t: float,
        overrides: dict,
        sizes: dict | None = None,
    ) -> None:
        assert utt is not None
        refs = self._refs(utt, overrides, sizes)
        refs.prepare([call])
        new: list[SceneObject] = []
        try:
            self._handle(call, refs, new, t)
        except NeedAsk as e:
            self._ask(e.q, utt, call, t)
        for entry in refs.log:
            self.log.add(t, "resolve", utt=utt.id, tool=call.name, **entry)
        if new:
            self._propose(new, utt.id, t)
        elif self.state == CLARIFY and self.question is None:
            self._set_state(self._rest_state(), t)

    # ------------------------------------------------------------------ helpers

    def say(self, t: float, text: str) -> None:
        self.log.add(t, "say", text=text)

    def _is_noise(self, utt: Utterance) -> bool:
        """Only fillers and acknowledgements, with no preview or question to answer."""
        if self.proposal is not None or self.question is not None:
            return False
        toks = [norm_word(w.text) for w in utt.words]
        return all(w in FILLERS or w in ACKS or not w for w in toks)

    def _invented(self, call: ToolCall, utt: Utterance) -> bool:
        """place_drawing of an object the person never named (the model repeating the last)."""
        if call.name != "place_drawing":
            return False
        obj = getattr(call.args, "object", "")
        target = self.planner.resolve(obj) or norm_word(obj)
        previewed = [o.label for p in [self.proposal, *self.queue] if p for o in p.items]
        toks = [norm_word(w.text) for w in utt.words]
        if target in previewed:
            # "no, there" / "a bit bigger" name no object but mean the one on preview. The
            # utterance must still carry the correction: on 6 Oct a mangled "a bit bigger"
            # ("Well we went") came back as place_drawing of the crown at the "top left
            # corner" and the crown jumped there, though neither word was said.
            return not set(toks) & CORRECTION_WORDS
        q = self.question
        call0 = q.call if q is not None else self.asked
        asked = getattr(call0.args, "object", None) if call0 is not None else None
        if asked and (self.planner.resolve(asked) or norm_word(asked)) == target:
            return False  # "How big?" ... "this big"
        if any(w in REPEAT_WORDS for w in toks):
            return False
        grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:], strict=False)]
        for g in grams:
            for form in (g, g[:-1] if g.endswith("s") else g, g[:-2] if g.endswith("es") else g):
                if form and (self.planner.resolve(form) or form) == target:
                    return False
        return True

    def _grounded(self, call: ToolCall, utt: Utterance, t: float) -> ToolCall:
        """Undo a place or a size the model made up: "a sun up here" is the pointed place,
        not "above the house"; nobody said "small"."""
        if call.name not in ("place_drawing", "write_label"):
            return call
        from pointdraw.agent.schema import PresetSize

        toks = [norm_word(w.text) for w in utt.words]
        args = call.args
        upd: dict[str, Any] = {}
        place = getattr(args, "place", None)
        demos = [i for i, w in enumerate(toks) if w in ("here", "there")]
        if (
            demos
            and isinstance(place, RelationPlace)
            and not set(toks) & RELATION_WORDS
            and not any(isinstance(o, DeicticObject) for o in (place.target, place.target2))
        ):
            upd["place"] = DeicticPlace(deixis=toks[demos[0]], word_index=demos[0])
        size = getattr(args, "size", None)
        if isinstance(size, PresetSize) and not set(toks) & PRESET_WORDS:
            upd["size"] = None
        if not upd:
            return call
        self.log.add(
            t,
            "guard",
            said=utt.text,
            why="a place or size not said: "
            + ", ".join(f"{k} {getattr(args, k).model_dump()}" for k in upd),
        )
        return ToolCall(name=call.name, args=args.model_copy(update=upd))

    def _is_stop(self, utt: Utterance) -> bool:
        toks = [norm_word(w.text) for w in utt.words]
        toks = [w for w in toks if w and w not in FILLERS]
        return bool(toks) and toks[0] in STOP_WORDS

    def _rest_state(self) -> str:
        if self.question is not None:
            return CLARIFY
        if self.drawing is not None:
            return DRAW if self.state == DRAW else WAIT_HANDS_CLEAR
        if self.proposal is not None:
            return PREVIEW
        if self.pending:
            return UNDERSTAND
        return LISTEN

    def _set_state(self, s: str, t: float) -> None:
        if s != self.state:
            self.log.add(t, "state", state=s, prev=self.state)
            self.state = s

    def _tag(self, base: str) -> str:
        self._tags += 1
        return f"{base}#{self._tags}"

    def _to_bed(self, p: Point) -> Point:
        return (p[0] + self.origin[0], p[1] + self.origin[1])

    def _to_paper(self, p: Point) -> Point:
        return (p[0] - self.origin[0], p[1] - self.origin[1])

    def _box_to_bed(self, b: Box) -> Box:
        return Box(b.x + self.origin[0], b.y + self.origin[1], b.w, b.h)


_DEMONSTRATIVE_WORDS = {"here", "there", "this", "that"}


def split_shared_labels(calls: list[ToolCall]) -> list[ToolCall]:
    """Live 30 Sep: "label this Cosmos and this ASR" once came back as two write_label calls
    both with the text "Cosmos and ASR". If k labels share one text that splits on "and" into
    k parts, give each label its own part, in order."""
    labels = [c for c in calls if c.name == "write_label"]
    if len(labels) < 2 or len({c.args.text for c in labels}) != 1:  # type: ignore[attr-defined]
        return calls
    parts = [x.strip(" ,") for x in re.split(r",?\s+and\s+|,\s*", labels[0].args.text)]  # type: ignore[attr-defined]
    parts = [x for x in parts if x]
    if len(parts) != len(labels):
        return calls
    it = iter(parts)
    return [
        ToolCall(name=c.name, args=c.args.model_copy(update={"text": next(it)}))
        if c.name == "write_label"
        else c
        for c in calls
    ]


def _needs_view(calls: list[ToolCall]) -> bool:
    """True if a call refers to an object on the sheet by pointing or by name."""
    from pointdraw.agent.schema import LabelObject

    def walk(m: Any) -> bool:
        if isinstance(m, (DeicticObject, LabelObject)):
            return True
        if isinstance(m, BaseModel):
            return any(walk(getattr(m, f)) for f in type(m).model_fields)
        return False

    return any(walk(c.args) for c in calls)


#: Homophones of drawable things that Parakeet produced in tts mode (30 Sep), fixed only
#: straight after an article (so "the son" also becomes "the sun": fine for a drawing robot).
_HOMOPHONES = {"son": "sun", "sunn": "sun", "i": "eye", "aye": "eye", "year": "ear"}
_ARTICLES = {"a", "an", "the"}
# "this big" heard as "this week" (rahul's voice, live 1 Oct, twice) or "best bit" (once).
_BIG_SLIPS = {"week", "weak", "wig"}


def repair_words(words: list[Word], categories: list[str]) -> list[Word]:
    """Undo a few known ASR slips on drawable words (Parakeet, tts mode, 30 Sep):

    - "a cross here" heard as "across here": a word that is "a" + a drawable category is
      split, but only before a demonstrative or at the end ("across the page" is English);
    - "an ear" heard as "a near": re-split;
    - "a sun" heard as "a son" (and eye/I, ear/year): homophones right after an article;
    - "this big" heard as "this week" (rahul's voice, 1 Oct): only at the end of what was said,
      or before a filler ("draw it this week, then a tree" keeps its week), and "best bit"
      only as the whole utterance.
    """
    cats = {c for c in categories if " " not in c}
    if [norm_word(w.text) for w in words] == ["best", "bit"]:
        a, b = words
        return [Word("this", a.t0, a.t1, a.confidence), Word("big", b.t0, b.t1, b.confidence)]
    out: list[Word] = []
    i = 0
    while i < len(words):
        w = words[i]
        low = norm_word(w.text)
        prev = norm_word(out[-1].text) if out else ""
        nxt = norm_word(words[i + 1].text) if i + 1 < len(words) else ""
        if low.startswith("a") and low[1:] in cats and (not nxt or nxt in _DEMONSTRATIVE_WORDS):
            mid = w.t0 + (w.t1 - w.t0) * 0.25
            out += [Word("a", w.t0, mid, w.confidence), Word(low[1:], mid, w.t1, w.confidence)]
        elif low == "a" and nxt.startswith("n") and nxt[1:] in cats and nxt[1:][:1] in "aeiou":
            n = words[i + 1]
            out += [Word("an", w.t0, w.t1, w.confidence), Word(nxt[1:], n.t0, n.t1, n.confidence)]
            i += 1
        elif prev in _ARTICLES and low in _HOMOPHONES and _HOMOPHONES[low] in cats:
            out.append(Word(_HOMOPHONES[low], w.t0, w.t1, w.confidence))
        elif (
            prev in ("this", "that", "about")
            and low in _BIG_SLIPS
            and all(norm_word(x.text) in FILLERS for x in words[i + 1 :])
        ):
            out.append(Word("big", w.t0, w.t1, w.confidence))
        else:
            out.append(w)
        i += 1
    return out


def _retarget(call: ToolCall, obj: SceneObject) -> ToolCall:
    """Replace every deictic object reference in a call with the chosen object's label.

    Used after "the house or the tree?" -> "the house"."""
    from pointdraw.agent.schema import LabelObject

    def walk(m: Any) -> Any:
        if isinstance(m, DeicticObject):
            return LabelObject(label=obj.label or obj.id)
        if isinstance(m, DeicticPlace) and call.name == "write_label":
            return RelationPlace(relation="below", target=LabelObject(label=obj.label or obj.id))
        if isinstance(m, BaseModel):
            upd = {f: walk(getattr(m, f)) for f in type(m).model_fields}
            return m.model_copy(update=upd)
        return m

    return ToolCall(name=call.name, args=walk(call.args))


def _deictic_indices(call: ToolCall) -> list[int]:
    from pointdraw.runtime.references import deictic_words

    places, _ = deictic_words(call.args)
    return places


def _box(b: Box | None) -> list[float] | None:
    if b is None:
        return None
    return [round(b.x, 2), round(b.y, 2), round(b.w, 2), round(b.h, 2)]


def _circle(c: tuple[Point, float] | None) -> list[float] | None:
    """A size circle for the log: [x, y, diameter], mm."""
    return None if c is None else [round(c[0][0], 2), round(c[0][1], 2), round(c[1], 2)]


def _pt(p: Point) -> list[float]:
    return [round(p[0], 2), round(p[1], 2)]
