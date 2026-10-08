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

import math
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
from pointdraw.vacuum.objects import thing_name

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
    send_inset_mm: float = 0.0  # shrink the box sent to the robot by this on every side
    # (7 Oct: tried at 40 mm, turned off by rahul -- the box sent is the box asked for)
    min_clean_s: float = 6.0  # the robot gets this long to start before it is believed
    clean_settle_s: float = 3.0  # ... and must say it has stopped for this long to count
    trail_step_px: float = 3.0  # the path keeps a point only this far from the last one


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
    resized: bool = False  # "smaller"/"a bit left": say the new size, not just the thing


@dataclass
class OpenQuestion:
    ask: Ask
    utt: Utterance
    calls: list[dict[str, Any]]
    k: int  # the call it is about


#: What Parakeet made of "to here" in rahul's voice (live, 7 Oct): "To hear", "Two here".
#: Both only ever mean "to here" to a vacuum, and both are fixed by their neighbour, so
#: "two metres" and "I can't hear you" are left alone.
_TO_SLIPS = {"two", "too"}  # -> "to", only right before a pointing word
_HERE_SLIPS = {"hear", "heir"}  # -> "here", only right after "to" or "from"
_POINTING = {"here", "there", "hear", "heir", "their"}
#: "Clean" as Parakeet writes it in rahul's voice (live, 7 and 8 Oct). Repaired only when the
#: sentence is about a place, so an utterance that really is about claims or screens is left.
_CLEAN_SLIPS = {"claim", "claims", "claimed", "cling", "clings", "cleaned", "cleans", "screen",
                "screens", "clear", "glean", "queen", "kleen"}  # fmt: skip
_PLACE_WORDS = {"here", "there", "from", "around", "round", "near", "beside", "under", "this",
                "that", "spot", "area", "bit", "square", "metre", "meter"}  # fmt: skip
#: What the vacuum answers to when nothing is waiting on an answer. Everything else said in
#: the room is not for it: 31 utterances in one run on 8 Oct, 20 of them other people talking,
#: three of which ("Stop up", "Scrap", "Stop tripping") stopped the robot.
#:
#: A pointing word counts as well as an order, because "clean" comes out as almost anything
#: -- "playing from here", "Clicking from here" (8 Oct) -- and no list of its mishearings
#: will ever be complete. "Here" is the word this whole project is about and it is rare in
#: ordinary talk, so it, not the verb, is what says the robot is being spoken to. The agent
#: still decides what was actually asked for.
_FOR_THE_ROBOT = {
    "clean", "stop", "dock", "vacuum", "hoover", "hover", "home",
    *_CLEAN_SLIPS, *_POINTING,
}  # fmt: skip


def repair_words(words: Sequence[Word]) -> list[Word]:
    """Undo the two ASR slips that break "clean from here ... to here" (live, 7 Oct).

    - "two here" / "too here" -> "to here": only when a pointing word comes next;
    - "to hear" -> "to here": only when "to" or "from" came before;
    - a first word that is "clean" misheard ("claim", "cling", "screen") -> "clean", but only
      when the sentence is about a place.

    Nothing else is touched, and the words keep their times, so the core still finds each
    "here" by its own word index.
    """
    out: list[Word] = []
    place = any(norm_word(w.text) in _PLACE_WORDS for w in words)
    for i, w in enumerate(words):
        low = norm_word(w.text)
        if i == 0 and low in _CLEAN_SLIPS and place:  # "Claim from here" -> "clean from here"
            out.append(Word("clean", w.t0, w.t1, w.confidence))
            continue
        nxt = norm_word(words[i + 1].text) if i + 1 < len(words) else ""
        prev = norm_word(out[-1].text) if out else ""
        if low in _TO_SLIPS and nxt in _POINTING:
            out.append(Word("to", w.t0, w.t1, w.confidence))
        elif low in _HERE_SLIPS and prev in ("to", "from"):
            out.append(Word("here", w.t0, w.t1, w.confidence))
        else:
            out.append(w)
    return out


def _opens(utt: Utterance) -> bool:
    """'Clean from here' (a pause, a walk): 'from' then a pointing word, and no 'to' after."""
    ws = [norm_word(w) for w in utt.texts]
    if "from" not in ws:
        return False
    after = ws[ws.index("from") + 1 :]
    return any(w in ("here", "there") for w in after) and "to" not in after


def inset(where: Where, by_mm: float) -> Where:
    """A box shrunk on every side, for the robot rather than for the person.

    The N50 hugs the line it is given and rides a little past it: measured 7 Oct, it was
    2 mm outside at the 75th percentile, 40 mm at the 95th. Sending a box shrunk by that
    much puts its overshoot on the line the person actually asked for. Nothing is lost at
    the edges -- the robot is 350 mm wide, so its body still sweeps well past its own path.

    Never shrinks a box to less than a third of a side, so a narrow strip stays a strip.
    """
    if not by_mm or where.box_mm is None:
        return where
    x, y, w, h = where.box_mm
    take = min(by_mm, w / 3, h / 3)
    box = (x + take, y + take, w - 2 * take, h - 2 * take)
    return where.model_copy(update={"box_mm": box, "mean_mm": (x + w / 2, y + h / 2)})


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


def _the(label: str) -> str:
    """ "chair" -> "the chair"; a label that already names itself is left alone."""
    low = label.lower()
    return label if low.startswith(("the ", "a ", "an ", "my ")) else f"the {label}"


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
        eyes: Callable[[str], Any] | None = None,  # a name -> a Where on the floor, or None
    ) -> None:
        self.agent = agent
        self.core = core
        self.link = link
        self.floor_to_map = floor_to_map
        self.profile = profile
        self.log = log or EventLog()
        self._say = say
        self._eyes = eyes
        self.p = params or VacuumParams()
        self.samples: deque[Sample] = deque()
        self.trail: deque[tuple[float, float]] = deque(maxlen=3000)  # the lid marker, in pixels
        self.state = LISTEN
        self.preview: Preview | None = None
        self.question: OpenQuestion | None = None
        self.pending: deque[tuple[Utterance, Future[AgentResult]]] = deque()
        self.opening: Utterance | None = None  # "clean from here", waiting for "to here"
        self.sent: list[dict[str, Any]] = []  # the boxes sent to the robot (map frame)
        self.last_floor_box: list[float] | None = None  # the last box sent, for the window
        self.chat_from = 0  # the window's thread starts here in the log (C clears it)
        self._cleaning_since: float | None = None  # when the robot was told to clean
        self._quiet_since: float | None = None  # since when it has not said "cleaning"

    # ------------------------------------------------------------------ inputs

    def on_sample(self, s: Sample) -> None:
        self.samples.append(s)
        while self.samples and self.samples[0].t < s.t - self.p.keep_s:
            self.samples.popleft()

    def on_utterance(self, words: Sequence[Word], t: float) -> None:
        utt = Utterance(repair_words(words), t)
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
        if not self._for_me(utt):
            self.log.add(t, "not_for_me", text=utt.text)
            return
        self._ask_agent(utt, t)

    def _for_me(self, utt: Utterance) -> bool:
        """Is this said to the robot? Only asked when nothing is waiting on an answer.

        With a box on the screen or a question open, anything may be the answer ("yes", "a bit
        left", "smaller"), so everything goes through. The rest of the time the room is full
        of talk that is not for it, and answering it ("Sorry, I didn't get that") is both
        noise on film and, for a word like "Scrap", a stop the robot should never have had.
        """
        if self.state in (PREVIEW, WAIT_CLEAR) or self.question is not None:
            return True
        return any(norm_word(w) in _FOR_THE_ROBOT for w in utt.texts)

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
        if self.state == CLEANING:
            self._finished(t)

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
            if res.ask.reason == "unknown_place" and self._look(utt, calls, k, t):
                return
            self.question = OpenQuestion(res.ask, utt, calls, k)
            self.log.add(t, "ask", reason=res.ask.reason, question=res.ask.question)
            self.say(t, res.ask.question)
            return
        where = res.calls[0].where("place")
        assert where is not None
        self.question = None
        self._show(Preview(where, utt, calls[k], t), t)

    def _look(self, utt: Utterance, calls: list[dict[str, Any]], k: int, t: float) -> bool:
        """A place given by name: ask the eyes to find it in the room. True if they did.

        Cosmos names and finds it, the floor fit places it, and the person still says yes or
        no to the box -- so a thing it gets wrong costs one "no", not a wrong clean.
        """
        name = str((calls[k].get("args") or {}).get("place", {}).get("named") or "").strip()
        if not name or self._eyes is None:
            return False
        self.log.add(t, "look", name=name)
        self.say(t, f"Let me look for {thing_name(name)}.")
        where = self._eyes(name)
        if where is None:
            self.log.add(t, "look_failed", name=name)
            return False
        self.log.add(t, "found", name=name, label=where.label,
                     box=[round(v) for v in where.box_mm or ()])  # fmt: skip
        self.question = None
        self._show(Preview(where, utt, calls[k], t), t)
        return True

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
        if w.source == "vlm":  # found by name, not by where the person stood
            what = _the(w.label or "it")
            size = f"{area_text(w)} " if prev.resized else ""
            self.say(t, f"Clean {size}around {what}?")
        elif prev.narrow:
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
        old = self.preview.where
        if old.source == "vlm":  # still the chair, moved or resized: don't start saying "here"
            where = where.model_copy(update={"source": "vlm", "label": old.label})
        self._show(
            Preview(where, self.preview.utt, self.preview.call, t, resized=old.source == "vlm"), t
        )

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
            # 7 Oct, live: rahul said "okay" twice while it was already cleaning and got
            # "There's nothing to start", which sounds like a refusal. Say what is going on.
            busy = self.state == CLEANING
            self.say(t, "It's already cleaning." if busy else "Nothing's waiting.")
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

    def on_robot_seen(self, t: float, centre_px: Any) -> None:
        """The marker on the robot's lid, where the camera sees it, every frame it is found.

        The path is kept in the picture's own pixels, exactly as `vacuum_map.py` tracks it
        during the calibration drive (rahul, 8 Oct 2026): it is what the camera saw, with no
        transform in between, so nothing the floor fit or the lid's height gets wrong can bend
        it. The camera must not move, which the fixed floor already requires.
        """
        here = (float(centre_px[0]), float(centre_px[1]))
        if self.trail and math.dist(self.trail[-1], here) < self.p.trail_step_px:
            return  # a robot sitting still must not pile up thousands of points
        self.trail.append(here)

    def on_robot_state(self, t: float, state: str | None) -> None:
        """What the robot says it is doing. Only remembers it -- `tick` decides.

        The watcher reports a *change*, so a robot that finishes and falls silent says
        "idle" exactly once. Judging it here meant that one report could land inside the
        settling time and be thrown away, and the app then waited for ever for a second one
        (7 Oct, live, twice).
        """
        if state == "cleaning":
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = t

    def _finished(self, t: float) -> None:
        """Leave CLEANING once the robot has stopped by itself. Called every frame.

        The robot is given `min_clean_s` to get going before anything it says counts (it
        reports the old state while it leaves the dock), and must have been quiet for
        `clean_settle_s`: a single stray "idle" mid-clean must not end the job.
        """
        if self.state != CLEANING or self._cleaning_since is None or self._quiet_since is None:
            return
        if t - self._cleaning_since < self.p.min_clean_s:
            return
        if t - self._quiet_since < self.p.clean_settle_s:
            return
        self.state = LISTEN
        self._cleaning_since = self._quiet_since = None
        self.log.add(t, "finished")
        self.say(t, "Finished.")

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
        self._cleaning_since = self._quiet_since = None
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
                self.last_floor_box = floor_box
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
        on_map = inset(on_map, self.p.send_inset_mm)
        box = [round(v) for v in on_map.box_mm or ()]
        reply = self._robot("clean_area", t, on_map)
        self.log.add(t, "clean_area", floor_box=[round(v) for v in prev.where.box_mm or ()],
                     map_box=box, inset_mm=self.p.send_inset_mm, reply=reply)  # fmt: skip
        self.sent.append({"t": t, "where": on_map, "reply": reply})
        self.last_floor_box = [round(v) for v in prev.where.box_mm or ()]
        if reply is None:
            self.say(t, "Dry run: the robot would clean there now.")
        elif reply.get("ok"):
            self.state = CLEANING
            self._cleaning_since, self._quiet_since = t, None
            self.say(t, "Cleaning there now.")
        else:
            self.say(t, f"The robot said no: {reply.get('why', 'no reason given')}.")

    # ------------------------------------------------------------------ a fresh take

    def clear_chat(self, t: float) -> None:
        """Hide the thread so far. The window only; the log keeps every line."""
        self.log.add(t, "chat_cleared")
        self.chat_from = len(self.log.events)

    def reset(self, t: float, why: str = "r key") -> None:
        """Back to listening with nothing showing, without restarting the app.

        The robot is deliberately left alone (rahul, 8 Oct 2026): if it is cleaning it
        carries on, and "stop" still reaches it from here, because `stop` always sends.
        Between takes this saves the 30 s of loading the camera, the ASR and the pose
        tracker again. The floor fit and the camera are untouched.
        """
        self.log.add(t, "reset", why=why, was=self.state,
                     robot_left_cleaning=self.state == CLEANING)  # fmt: skip
        self.state = LISTEN
        self.preview = self.question = self.opening = None
        self.pending.clear()
        self.sent.clear()
        self.last_floor_box = None
        self.trail.clear()
        self._cleaning_since = self._quiet_since = None
        self.clear_chat(t)

    # ------------------------------------------------------------------ for the view

    def chat(self, limit: int = 6) -> list[tuple[str, str]]:
        """The last few turns as ("you" | "it", text), oldest first, for the window.

        Both sides, unlike the plotter's window, which lists only what it heard: on film the
        thread has to read as a conversation (rahul, 8 Oct 2026).
        """
        out: list[tuple[str, str]] = []
        for e in self.log.events[self.chat_from :]:
            if e["kind"] == "utterance" and e.get("text"):
                out.append(("you", str(e["text"])))
            elif e["kind"] == "say":
                out.append(("it", str(e["text"])))
        return out[-limit:]

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
