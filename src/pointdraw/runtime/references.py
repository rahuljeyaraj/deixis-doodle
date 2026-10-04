"""Turn one utterance's references (from the agent's tool calls) into places, sizes and
scene objects, by asking the deixis core (`deixis.resolve`, over MCP or in-process; see
`pointdraw.core`).

This is a thin wrapper: it builds a `deixis.Request` (the words, the holds and spans, the
scene's objects as things, all the utterance's calls), sends it, and turns the `Response`
back into what the runtime uses (`Place`, sizes in mm, `SceneObject`s, `Question`s). All
the utterance's demonstratives are matched to holds together, in time order, inside the
core, so "a house here and a tree there" gets two different holds. The runtime handles one
call at a time (a call may add a drawing the next one refers to), so each request resolves
one call (`only`) while every call takes part in the matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from deixis.actions import Call, CallError
from deixis.api import Request, RequestError, Response
from deixis.geometry import Point
from deixis.mcp import DeixisClient
from deixis.refs import DEMONSTRATIVES, norm_word
from deixis.resolver import Pick, ResolverParams, SpanHold
from deixis.types import Hold, Word
from deixis.where import Answer, Ask, Size, Where
from pointdraw.agent.schema import (
    DeicticObject,
    DeicticPlace,
    DeicticSize,
    PointedPlace,
)
from pointdraw.core import core_client
from pointdraw.scene.model import SceneModel, SceneObject


@dataclass
class Utterance:
    id: int
    words: list[Word]
    t_avail: float  # when the transcript reached the runtime

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def t0(self) -> float:
        return self.words[0].t0 if self.words else self.t_avail

    @property
    def t1(self) -> float:
        return self.words[-1].t1 if self.words else self.t_avail

    def tokens(self) -> list[str]:
        return [w.text for w in self.words]


@dataclass
class Question:
    """Something the runtime must ask before it can act."""

    text: str
    reason: str  # no_gesture, ambiguous, unsteady, no_size, unknown_object, no_object, ...
    options: list[str] = field(default_factory=list)
    candidates: list[SceneObject] = field(default_factory=list)


class NeedAsk(Exception):  # noqa: N818 - control flow, not an error
    def __init__(self, q: Question) -> None:
        super().__init__(q.text)
        self.q = q


@dataclass
class Place:
    center: Point
    where: Where
    size: float | None = None  # a size the place imposes (empty space, inside)
    target: SceneObject | None = None


def deictic_words(args: Any) -> tuple[list[int], list[int]]:
    """Word indices of deictic place/object references and of deictic sizes."""
    places: list[int] = []
    sizes: list[int] = []

    def walk(o: Any) -> None:
        if isinstance(o, (DeicticPlace, DeicticObject)):
            places.append(o.word_index)
        elif isinstance(o, DeicticSize):
            sizes.append(o.word_index)
        elif isinstance(o, BaseModel):
            for f in type(o).model_fields:
                walk(getattr(o, f))

    walk(args)
    return places, sizes


#: The plotter's tools and their reference arguments (as the profile names them).
_REF_ARGS = {
    "place_drawing": ("place", "size"),
    "draw_connector": ("from", "to"),
    "write_label": ("place", "size"),
}


def _field(arg: str) -> str:
    return "from_" if arg == "from" else arg


def _contains(value: Any, ref: Any) -> bool:
    if value is ref:
        return True
    if isinstance(value, BaseModel):
        return any(_contains(getattr(value, f), ref) for f in type(value).model_fields)
    return False


class References:
    """Resolves the references of one utterance against gestures and the scene."""

    def __init__(
        self,
        utt: Utterance,
        holds: list[Hold],
        spans: list[SpanHold],
        scene: SceneModel,
        params: ResolverParams | None = None,  # unused: the core takes them from the profile
        default_size: float = 40.0,
        overrides: dict[int, Hold] | None = None,
        size_overrides: dict[int, SpanHold] | None = None,
        client: DeixisClient | None = None,
    ) -> None:
        self.utt = utt
        self.holds = holds
        self.spans = spans
        self.scene = scene
        self.default_size = default_size
        self.overrides = overrides or {}  # word index -> hold, from a clarifying answer
        self.size_overrides = size_overrides or {}  # word index -> span, from "this big"
        self.client = client or core_client(scene.profile)
        self.calls: list[Call] = []
        self.current: Call | None = None  # the call being handled (set by the runtime)
        self.picks: dict[int, Pick] = {}  # word index -> the hold used (for answers)
        self.log: list[dict[str, Any]] = []
        self._cache: dict[str, Response] = {}

    def prepare(self, calls: list[Any]) -> None:
        self.calls = list(calls)

    # ------------------------------------------------------------------ asking the core

    def _request(
        self, calls: list[dict[str, Any]], only: int | None, answers: list[Answer]
    ) -> Response:
        req = Request(
            words=self.utt.words,
            holds=self.holds,
            spans=self.spans,
            things=self.scene.things(),
            calls=calls,  # type: ignore[arg-type]
            answers=answers,
            speech=(self.utt.t0, self.utt.t1),
            only=only,
        )
        key = json.dumps(
            {"c": calls, "o": only, "a": [a.model_dump(mode="json") for a in answers]},
            sort_keys=True,
        )
        res = self._cache.get(key)
        if res is None:
            try:
                res = self.client.resolve(req)
            except (CallError, RequestError) as e:
                raise ValueError(f"deixis: {e}") from e
            self._cache[key] = res
            self.log += res.log
        return res

    def _resolve_call(self, k: int, size: float | None = None) -> Response:
        """Resolve call k of the utterance, with the answers this References carries."""
        call = self.calls[k]
        answers = self._answers(k, call)
        if size is not None and call.name in ("place_drawing", "write_label"):
            answers = [a for a in answers if a.arg != "size"]
            answers.append(Answer(call=k, arg="size", size=Size(mm=size, source="default")))
        return self._request([c.to_dict() for c in self.calls], k, answers)

    def _answers(self, k: int, call: Call) -> list[Answer]:
        out: list[Answer] = []
        for arg in _REF_ARGS.get(call.name, ()):
            ref = getattr(call.args, _field(arg), None)
            places, sizes = deictic_words(ref)
            hit = next((i for i in places if i in self.overrides), None)
            if hit is not None:
                out.append(Answer(call=k, arg=arg, hold=self.overrides[hit]))
            elif arg == "place" and -1 in self.overrides and not places:
                out.append(Answer(call=k, arg=arg, hold=self.overrides[-1]))
            span_hit = next((i for i in sizes if i in self.size_overrides), None)
            if arg == "size" and span_hit is not None:
                out.append(Answer(call=k, arg=arg, size=_span_size(self.size_overrides[span_hit])))
        return out

    def _ask(self, ask: Ask) -> NeedAsk:
        cands = [self.scene.get(c.thing) for c in ask.candidates if c.thing is not None]
        return NeedAsk(
            Question(ask.question, ask.reason, list(ask.options), [c for c in cands if c])
        )

    def _arg(self, res: Response, k: int, arg: str) -> Any:
        """The resolved argument, or NeedAsk if the core asked about it (or before it)."""
        if res.status == "ok":
            return res.calls[0].args.get(arg)
        done = next((a for a in res.partial if a.call == k and a.arg == arg), None)
        if done is not None:
            return done.where or done.size
        assert res.ask is not None
        raise self._ask(res.ask)

    def _find(self, ref: Any) -> tuple[int, str]:
        """The call and the argument that hold this reference (the current call first)."""
        order = list(range(len(self.calls)))
        if self.current is not None:
            cur = next((i for i, c in enumerate(self.calls) if c is self.current), None)
            if cur is not None:
                order = [cur] + [i for i in order if i != cur]
        for k in order:
            call = self.calls[k]
            for arg in _REF_ARGS.get(call.name, ()):
                if _contains(getattr(call.args, _field(arg), None), ref):
                    return k, arg
        raise TypeError(f"reference not in this utterance's calls: {ref!r}")

    def _current_index(self) -> int:
        if self.current is None:
            raise TypeError("no current call")
        return next(i for i, c in enumerate(self.calls) if c is self.current)

    # ------------------------------------------------------------------ holds

    def point_for_word(self, word_index: int) -> tuple[Point, Where]:
        found = self._find_word(word_index)
        if found is None:
            return self._locate(word_index)
        k, arg = found
        w = Where.model_validate(self._arg(self._resolve_call(k), k, arg))
        if w.thing is not None and "point" in w.evidence:  # an object: where it was pointed
            p = w.evidence["point"]
            return (p[0], p[1]), w
        assert w.mean_mm is not None
        return w.mean_mm, w

    def _find_word(self, word_index: int) -> tuple[int, str] | None:
        for k, call in enumerate(self.calls):
            for arg in _REF_ARGS.get(call.name, ()):
                places, _ = deictic_words(getattr(call.args, _field(arg), None))
                if word_index in places:
                    return k, arg
        return None

    def _locate(self, word_index: int) -> tuple[Point, Where]:
        """No call uses the word (an answer to a question): ask the core where it points."""
        word = norm_word(self.utt.words[word_index].text)
        deixis = word if word in DEMONSTRATIVES else "here"
        place = {"deixis": deixis, "word_index": word_index}
        answers = []
        if word_index in self.overrides:
            answers.append(Answer(call=0, arg="place", hold=self.overrides[word_index]))
        res = self._request([{"name": "locate", "args": {"place": place}}], None, answers)
        w = Where.model_validate(self._arg(res, 0, "place"))
        if res.holds:
            self.picks[word_index] = Pick(res.holds[0], "locate", self.utt.words[word_index])
        assert w.mean_mm is not None
        return w.mean_mm, w

    def last_point(self) -> tuple[Point, Where]:
        k = self._current_index()
        w = Where.model_validate(self._arg(self._resolve_call(k), k, "place"))
        assert w.mean_mm is not None
        return w.mean_mm, w

    # ------------------------------------------------------------------ objects

    def obj(self, ref: Any) -> SceneObject:
        k, arg = self._find(ref)
        got = self._arg(self._resolve_call(k), k, arg)
        value = getattr(self.calls[k].args, _field(arg))
        if isinstance(got, Size) or (isinstance(got, dict) and "mm" in got):
            thing = Size.model_validate(got).evidence.get("same_as")
        else:
            w = Where.model_validate(got)
            if value is ref:
                thing = w.thing
            elif getattr(value, "target2", None) is ref:
                thing = w.evidence.get("target2")
            else:
                thing = w.evidence.get("target")
        o = self.scene.get(thing) if thing else None
        if o is None:
            raise NeedAsk(Question(self.scene.profile.ask_text("no_object"), "no_object"))
        return o

    # ------------------------------------------------------------------ sizes

    def size(self, ref: Any, default: float | None = None) -> float:
        if ref is None:
            return self.default_size if default is None else default
        if isinstance(ref, DeicticSize) and ref.word_index in self.size_overrides:
            span = self.size_overrides[ref.word_index]
            w = self.utt.words[ref.word_index]
            self.log.append(
                {"size": round(span.span_mm, 2), "size_kind": span.kind, "word": w.text}
            )
            return span.span_mm
        k, arg = self._find(ref)
        s = self._arg(self._resolve_call(k), k, arg)
        return Size.model_validate(s).mm

    # ------------------------------------------------------------------ places

    def place(self, ref: Any, size: float) -> Place:
        if ref is None or isinstance(ref, PointedPlace):
            k, arg = self._current_index(), "place"
        else:
            k, arg = self._find(ref)
        w = Where.model_validate(self._arg(self._resolve_call(k, size), k, arg))
        assert w.mean_mm is not None
        imposed = w.evidence.get("size_mm")
        target = w.evidence.get("target")
        return Place(
            w.mean_mm,
            w,
            float(imposed) if isinstance(imposed, (int, float)) else None,
            self.scene.get(target) if target else None,
        )


def _span_size(span: SpanHold) -> Size:
    return Size(
        mm=span.span_mm,
        source="gesture",
        evidence={"size_kind": span.kind, "span_t": [round(span.t0, 3), round(span.t1, 3)]},
    )
