"""The vacuum app's brain: feet and words in, a cleaning box out, after a "yes".

    "Clean here."            a 1 m box around where the person stands
    "Clean from here ...
     ... to here."           the two standing spots are the box's corners
    "Yes." / "No."           send it / drop it
    "A bit left."            move the box one step (250 mm) before saying yes
    "Stop." / "Go home."     at any time

Each frame's feet go in as a `Sample` (`on_sample`), each utterance as timed words
(`on_utterance`), and `tick` moves things on. The agent turns words into the vacuum's tool
calls in a background thread; the deixis core (a `DeixisClient`, over MCP) turns each
"here" into a box on the floor, or a question. The app shows the box (`preview`), asks
"Clean about one metre square here?", and on "yes" waits until the person has stepped out of
the box, then maps the box onto the robot's own map (the calibration, `FloorToMap`, applied
by the core) and sends it to the robot (`VacuumLink.clean_area`). The robot cleans exactly
that box or says why not; nothing here moves a box to make it fit.

"Clean from here ... to here" is echoed as "Clean from there to here?" (no size said). If
that box is narrower than half a metre, it asks "That's a narrow strip. Clean half a metre
wide?" and "yes" widens it about its centre (the robot's server refuses under 200 mm). After
"yes", one line: "OK. Step out, and I'll clean it."

Everything it does goes in `log` (an `EventLog`); everything it says goes to `say`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from deixis.api import Request, RequestError, Response
from deixis.profile import Profile
from deixis.refs import DEMONSTRATIVES, norm_word
from deixis.types import Sample, Word
from deixis.where import Answer, Ask, Where
from pointdraw.agent.nemotron import AgentResult
from pointdraw.runtime.events import EventLog
from pointdraw.vacuum.agent import Context
from pointdraw.vacuum.calibrate import FloorToMap

LISTEN, PREVIEW, WAIT_CLEAR, CLEANING = "LISTEN", "PREVIEW", "WAIT_CLEAR", "CLEANING"


@dataclass
class VacuumParams:
    before_s: float = 6.0  # feet samples sent with an utterance: this long before its first word
    after_s: float = 2.0  # ... and this long after its last
    keep_s: float = 60.0  # samples kept
    clear_margin_mm: float = 150.0  # the person must stand this far outside the box
    clear_s: float = 1.0  # ... for this long before the robot is sent
    gone_s: float = 3.0  # or have left the camera's view (last seen outside) this long
    remind_s: float = 8.0  # ask again to step out after this
    join_s: float = 8.0  # "clean from here" waits this long for "... to here" (a walk apart)
    narrow_mm: float = 500.0  # a two-spot box narrower than this: "Clean half a metre wide?"


@dataclass
class Utterance:
    words: list[Word]
    t: float  # when it arrived

    @property
    def texts(self) -> list[str]:
        return [w.text for w in self.words]

    @property
    def text(self) -> str:
        return " ".join(self.texts)

    @property
    def span(self) -> tuple[float, float]:
        return (self.words[0].t0, self.words[-1].t1) if self.words else (self.t, self.t)


@dataclass
class Preview:
    where: Where  # the box on the floor (frame "floor")
    utt: Utterance
    call: dict[str, Any]  # the clean call it came from
    t: float
    confirmed_t: float | None = None
    reminded_t: float | None = None
    out_since: float | None = None  # the person has stood outside the box since then
    narrow: bool = False  # asked "Clean half a metre wide?"; "yes" widens it first


@dataclass
class OpenQuestion:
    ask: Ask
    utt: Utterance
    calls: list[dict[str, Any]]
    k: int  # the call it is about


def _opens(utt: Utterance) -> bool:
    """'Clean from here' (a pause, a walk): 'from' then a pointing word, and no 'to' after."""
    ws = [norm_word(w) for w in utt.texts]
    if "from" not in ws:
        return False
    after = ws[ws.index("from") + 1 :]
    return any(w in ("here", "there") for w in after) and "to" not in after


def area_text(where: Where) -> str:
    """'about one metre square', 'about 1.3 by 0.9 metres' (for speech)."""
    assert where.box_mm is not None
    w, h = where.box_mm[2] / 1000, where.box_mm[3] / 1000
    if abs(w - h) < 0.1:
        if abs(w - 1) < 0.05:
            return "about one metre square"
        return f"about {w:.1f} metres square"
    return f"about {w:.1f} by {h:.1f} metres"


def two_spot(where: Where) -> bool:
    """A box from "clean from here ... to here" (its two corners), not a box around one spot."""
    return where.kind == "box" and "corners" in where.evidence


def widened(where: Where, side_mm: float) -> Where:
    """The box with each side under `side_mm` widened to it, about the same centre."""
    assert where.box_mm is not None
    x, y, w, h = where.box_mm
    if w < side_mm:
        x, w = x - (side_mm - w) / 2, side_mm
    if h < side_mm:
        y, h = y - (side_mm - h) / 2, side_mm
    ev = {**where.evidence, "widened_to_mm": side_mm}
    return where.model_copy(
        update={"box_mm": (x, y, w, h), "mean_mm": (x + w / 2, y + h / 2), "evidence": ev}
    )


def inside(p: tuple[float, float], box: Sequence[float], margin: float = 0.0) -> bool:
    x, y, w, h = box
    return x - margin <= p[0] <= x + w + margin and y - margin <= p[1] <= y + h + margin


class VacuumApp:
    def __init__(
        self,
        agent: Any,  # .submit(words, Context) -> Future[AgentResult], or .understand(...)
        core: Any,  # a DeixisClient for the vacuum's profile
        link: Any | None,  # a VacuumLink (None: a dry run, nothing is sent)
        floor_to_map: FloorToMap | None,
        profile: Profile,
        *,
        log: EventLog | None = None,
        say: Callable[[str], None] | None = None,
        params: VacuumParams | None = None,
    ) -> None:
        self.agent = agent
        self.core = core
        self.link = link
        self.floor_to_map = floor_to_map
        self.profile = profile
        self.log = log or EventLog()
        self._say = say
        self.p = params or VacuumParams()
        self.samples: deque[Sample] = deque()
        self.state = LISTEN
        self.preview: Preview | None = None
        self.question: OpenQuestion | None = None
        self.pending: deque[tuple[Utterance, Future[AgentResult]]] = deque()
        self.opening: Utterance | None = None  # "clean from here", waiting for "to here"
        self.sent: list[dict[str, Any]] = []  # the boxes sent to the robot (map frame)

    # ------------------------------------------------------------------ inputs

    def on_sample(self, s: Sample) -> None:
        self.samples.append(s)
        while self.samples and self.samples[0].t < s.t - self.p.keep_s:
            self.samples.popleft()

    def on_utterance(self, words: Sequence[Word], t: float) -> None:
        utt = Utterance(list(words), t)
        if not utt.words:
            return
        self.log.add(t, "utterance", text=utt.text, words=[[w.text, w.t0, w.t1] for w in words])
        if self.opening is not None:  # "... to here": one request with "clean from here"
            utt = Utterance(self.opening.words + utt.words, t)
            self.opening = None
            self.log.add(t, "joined", text=utt.text)
        elif _opens(utt):
            self.opening = utt
            self.log.add(t, "waiting", why="'from here' without 'to': walk to the other corner")
            return
        if self._is_answer(utt):
            self._answer(utt, t)
            return
        self._ask_agent(utt, t)

    def _ask_agent(self, utt: Utterance, t: float) -> None:
        ctx = Context(
            previewing=self.state in (PREVIEW, WAIT_CLEAR),
            cleaning=self.state == CLEANING,
            question=self.question.ask.question if self.question else None,
        )
        submit = getattr(self.agent, "submit", None)
        if submit is not None:
            fut = submit(utt.texts, ctx)
        else:
            fut = Future()
            fut.set_result(self.agent.understand(utt.texts, ctx))
        self.pending.append((utt, fut))
        self.tick(t)

    def tick(self, t: float) -> None:
        if self.opening is not None and t - self.opening.t >= self.p.join_s:
            utt, self.opening = self.opening, None  # no "to here" came: as said
            self._ask_agent(utt, t)
        while self.pending and self.pending[0][1].done():
            utt, fut = self.pending.popleft()
            self._handle(utt, fut.result(), t)
        if self.state == WAIT_CLEAR:
            self._wait_clear(t)

    # ------------------------------------------------------------------ talking

    def say(self, t: float, text: str) -> None:
        self.log.add(t, "say", text=text)
        if self._say is not None:
            self._say(text)

    # ------------------------------------------------------------------ the agent's calls

    def _handle(self, utt: Utterance, res: AgentResult, t: float) -> None:
        calls = [c.to_dict() for c in res.calls]
        self.log.add(t, "tool_calls", utt=utt.text, calls=calls, source=res.source,
                     latency_s=round(res.latency_s, 3), errors=res.errors)  # fmt: skip
        if not calls:
            self.say(t, "Sorry, I didn't get that.")
            return
        names = [c["name"] for c in calls]
        if "stop" in names:  # stopping wins over everything else said with it
            self.stop(t)
            return
        for k, c in enumerate(calls):
            name = c["name"]
            if name == "clean":
                self._clean(utt, calls, k, t)
            elif name == "confirm":
                self.confirm(t)
            elif name == "cancel":
                self.cancel(t)
            elif name == "adjust":
                self._adjust(utt, c, t)
            elif name == "dock":
                self.dock(t)
            elif name == "ask_user":
                self.say(t, str(c["args"].get("question", "Can you say that again?")))

    def _resolve(
        self,
        utt: Utterance,
        calls: list[dict[str, Any]],
        k: int | None,
        answers: Sequence[Answer] = (),
        context: dict[str, Any] | None = None,
    ) -> Response:
        t0, t1 = utt.span
        window = [s for s in self.samples if t0 - self.p.before_s <= s.t <= t1 + self.p.after_s]
        req = Request(
            words=utt.words,
            samples=window,
            calls=calls,  # type: ignore[arg-type]
            answers=list(answers),
            speech=(t0, t1),
            only=k,
            **({"context": context} if context else {}),
        )
        return self.core.resolve(req)

    def _clean(self, utt: Utterance, calls: list[dict[str, Any]], k: int, t: float,
               answers: Sequence[Answer] = ()) -> None:  # fmt: skip
        try:
            res = self._resolve(utt, calls, k, answers)
        except (RequestError, ValueError) as e:
            self.log.add(t, "error", where="resolve", error=str(e))
            self.say(t, "Sorry, something went wrong. Please say it again.")
            return
        self.log.add(t, "resolve", status=res.status, log=res.log)
        if res.status == "ask":
            assert res.ask is not None
            self.question = OpenQuestion(res.ask, utt, calls, k)
            self.log.add(t, "ask", reason=res.ask.reason, question=res.ask.question)
            self.say(t, res.ask.question)
            return
        where = res.calls[0].where("place")
        assert where is not None
        self.question = None
        self._show(Preview(where, utt, calls[k], t), t)

    def _show(self, prev: Preview, t: float) -> None:
        """Show the box and say the guess. A two-spot box is echoed ("Clean from there to
        here?"), not sized; under `narrow_mm` across, it asks to widen it first (claim 30a)."""
        self.preview = prev
        self.state = PREVIEW
        w = prev.where
        two = two_spot(w)
        assert w.box_mm is not None
        prev.narrow = two and min(w.box_mm[2], w.box_mm[3]) < self.p.narrow_mm
        self.log.add(t, "preview", box=[round(v) for v in w.box_mm or ()], mean=w.mean_mm,
                     source=w.source, outside=bool(w.evidence.get("outside")),
                     two_spot=two, narrow=prev.narrow)  # fmt: skip
        if prev.narrow:
            self.say(t, "That's a narrow strip. Clean half a metre wide?")
        elif w.evidence.get("outside"):
            what = "from there to here" if two else f"{area_text(w)} there"
            self.say(t, f"That's past the edge of what I can see. Clean {what}?")
        elif two:
            self.say(t, "Clean from there to here?")
        else:
            self.say(t, f"Clean {area_text(w)} here?")

    def _adjust(self, utt: Utterance, call: dict[str, Any], t: float) -> None:
        if self.preview is None or self.state not in (PREVIEW, WAIT_CLEAR):
            self.say(t, self.profile.ask_text("no_previous"))
            return
        prev = {"place": self.preview.where.model_dump(mode="json")}
        res = self._resolve(utt, [call], 0, context={"previous": prev})
        if res.status != "ok":
            assert res.ask is not None
            self.say(t, res.ask.question)
            return
        where = res.calls[0].where("place")
        assert where is not None
        self.log.add(t, "adjust", args=call["args"], box=[round(v) for v in where.box_mm or ()])
        self._show(Preview(where, self.preview.utt, self.preview.call, t), t)

    # ------------------------------------------------------------------ answers

    def _is_answer(self, utt: Utterance) -> bool:
        """'Here.' after "Where should I clean?": an answer, no agent needed."""
        if self.question is None or self.question.ask.reason not in (
            "no_gesture", "unsteady", "ambiguous"
        ):  # fmt: skip
            return False
        ws = [norm_word(w) for w in utt.texts]
        return any(w in ("here", "there") for w in ws) and len(ws) <= 3

    def _answer(self, utt: Utterance, t: float) -> None:
        """Where the person stands now answers the open question about the place."""
        q = self.question
        assert q is not None
        ws = [norm_word(w) for w in utt.texts]
        i = next(i for i, w in enumerate(ws) if w in ("here", "there"))
        word = ws[i] if ws[i] in DEMONSTRATIVES else "here"
        locate = {"name": "locate", "args": {"place": {"deixis": word, "word_index": i}}}
        res = self._resolve(utt, [locate], None)
        if res.status != "ok" or not res.holds:
            assert res.ask is not None
            self.log.add(t, "ask", reason=res.ask.reason, question=res.ask.question)
            self.say(t, res.ask.question)
            return
        self.log.add(t, "answer", question=q.ask.question, hold=res.holds[0].mean)
        answer = Answer(call=q.k, arg=q.ask.arg or "place", hold=res.holds[0])
        self._clean(q.utt, q.calls, q.k, t, [answer])

    # ------------------------------------------------------------------ yes, no, stop

    def confirm(self, t: float) -> None:
        prev = self.preview
        if prev is None or self.state != PREVIEW:
            self.say(t, "There's nothing to start.")
            return
        prev.confirmed_t = t
        if prev.narrow:  # "Clean half a metre wide?" ... "yes"
            prev.where = widened(prev.where, self.p.narrow_mm)
            prev.narrow = False
            self.log.add(t, "widened", box=[round(v) for v in prev.where.box_mm or ()])
        self.state = WAIT_CLEAR
        self.log.add(t, "confirm", box=[round(v) for v in prev.where.box_mm or ()])
        if self._person_in_box(t):
            self.say(t, "OK. Step out, and I'll clean it.")  # one line (claim 30)
            prev.reminded_t = t
        self._wait_clear(t)

    def cancel(self, t: float) -> None:
        if self.preview is not None and self.state in (PREVIEW, WAIT_CLEAR):
            self.preview = None
            self.state = LISTEN
            self.log.add(t, "cancel")
            self.say(t, "OK, cancelled.")
        elif self.question is not None:
            self.question = None
            self.log.add(t, "cancel")
            self.say(t, "OK, never mind.")
        else:
            self.say(t, "There's nothing to cancel.")

    def stop(self, t: float) -> None:
        self.preview, self.question = None, None
        self.state = LISTEN
        reply = self._robot("stop", t)
        self.log.add(t, "stop", reply=reply)
        self.say(t, "Stopped.")

    def dock(self, t: float) -> None:
        self.preview = None
        self.state = LISTEN
        reply = self._robot("dock", t)
        self.log.add(t, "dock", reply=reply)
        self.say(t, "Going back to the dock.")

    def _robot(self, tool: str, t: float, *args: Any) -> dict[str, Any] | None:
        """Call the robot's server; None in a dry run; a failure is logged and said."""
        if self.link is None:
            self.log.add(t, "dry_run", tool=tool)
            return None
        try:
            return getattr(self.link, tool)(*args)
        except Exception as e:  # noqa: BLE001 - the app must keep running
            self.log.add(t, "error", where=tool, error=str(e))
            self.say(t, "I can't reach the robot.")
            return {"ok": False, "why": str(e)}

    # ------------------------------------------------------------------ waiting, then sending

    def _last_seen(self, t: float) -> Sample | None:
        return next((s for s in reversed(self.samples) if s.point is not None), None)

    def _person_in_box(self, t: float) -> bool:
        assert self.preview is not None and self.preview.where.box_mm is not None
        s = self._last_seen(t)
        if s is None or t - s.t > self.p.gone_s:
            return False
        return inside(s.point, self.preview.where.box_mm, self.p.clear_margin_mm)  # type: ignore[arg-type]

    def _wait_clear(self, t: float) -> None:
        prev = self.preview
        assert prev is not None and prev.where.box_mm is not None
        s = self._last_seen(t)
        box = prev.where.box_mm
        in_box = False  # seen inside the box now (a reminder is said only then)
        if s is None:
            clear = True  # nobody seen at all
        elif t - s.t >= self.p.gone_s:  # left the view: fine if last seen outside the box
            clear = not inside(s.point, box, 0.0)  # type: ignore[arg-type]
            in_box = not clear
        else:
            out = not inside(s.point, box, self.p.clear_margin_mm)  # type: ignore[arg-type]
            if not out:
                prev.out_since = None
            elif prev.out_since is None:
                prev.out_since = s.t
            clear = prev.out_since is not None and t - prev.out_since >= self.p.clear_s
            in_box = not out
        if clear:
            self._send(t)
        elif in_box and (prev.reminded_t is None or t - prev.reminded_t >= self.p.remind_s):
            prev.reminded_t = t
            self.log.add(t, "waiting", why="person in the box")
            self.say(t, "Please step out of the box so I can start.")

    def _send(self, t: float) -> None:
        prev = self.preview
        assert prev is not None
        self.preview = None
        self.state = LISTEN
        if self.floor_to_map is None:
            floor_box = [round(v) for v in prev.where.box_mm or ()]
            if self.link is None:  # a dry run before calibration: the floor box is enough
                self.log.add(t, "clean_area", floor_box=floor_box, map_box=None, reply=None)
                self.say(t, "Dry run: the robot would clean there now.")
                return
            self.log.add(t, "error", where="send", error="no floor-to-map calibration")
            self.say(t, "I don't know where the robot's map is yet. Please run the calibration.")
            return
        try:
            on_map = self.core.apply_transform(prev.where, self.floor_to_map.transform, "map")
        except Exception as e:  # noqa: BLE001
            self.log.add(t, "error", where="apply_transform", error=str(e))
            self.say(t, "Sorry, I couldn't place that on the robot's map.")
            return
        box = [round(v) for v in on_map.box_mm or ()]
        reply = self._robot("clean_area", t, on_map)
        self.log.add(t, "clean_area", floor_box=[round(v) for v in prev.where.box_mm or ()],
                     map_box=box, reply=reply)  # fmt: skip
        self.sent.append({"t": t, "where": on_map, "reply": reply})
        if reply is None:
            self.say(t, "Dry run: the robot would clean there now.")
        elif reply.get("ok"):
            self.state = CLEANING
            self.say(t, "Cleaning there now.")
        else:
            self.say(t, f"The robot said no: {reply.get('why', 'no reason given')}.")

    # ------------------------------------------------------------------ for the view

    def banner(self) -> str:
        if self.state == PREVIEW:
            return "Clean here? Say yes or no"
        if self.state == WAIT_CLEAR:
            return "Waiting: step out of the box"
        if self.state == CLEANING:
            return "Cleaning"
        if self.question is not None:
            return self.question.ask.question
        return "Listening"

    def spot(self, t: float) -> tuple[float, float] | None:
        s = self._last_seen(t)
        return None if s is None or t - s.t > 0.5 else s.point  # type: ignore[return-value]


@dataclass
class Spoken:
    """For tests: what `say` was given, in order."""

    lines: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> None:
        self.lines.append(text)
