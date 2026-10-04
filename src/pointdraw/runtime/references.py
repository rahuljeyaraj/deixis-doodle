"""Turn one utterance's references (from the agent's tool calls) into places, sizes and
scene objects, using the timing resolver (`deixis.resolver`) and the scene model.

All k demonstratives of an utterance are matched to holds together, in time order, before
any call is handled, so "a house here and a tree there" gets two different holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from deixis.geometry import Point
from deixis.resolver import (
    Ask,
    Pick,
    ResolverParams,
    SpanHold,
    assign_holds,
    hold_where,
    last_hold,
    pick_span,
    size_holds,
)
from deixis.types import Hold, Word
from deixis.where import Where
from pointdraw.agent.schema import (
    PRESET_MM,
    DeicticObject,
    DeicticPlace,
    DeicticSize,
    EmptySpace,
    LabelObject,
    LastObject,
    NamedPlace,
    PresetSize,
    RelationPlace,
    RelativeSize,
)
from pointdraw.profile import plotter_profile
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


class References:
    """Resolves the references of one utterance against gestures and the scene."""

    def __init__(
        self,
        utt: Utterance,
        holds: list[Hold],
        spans: list[SpanHold],
        scene: SceneModel,
        params: ResolverParams | None = None,
        default_size: float = 40.0,
        overrides: dict[int, Hold] | None = None,
        size_overrides: dict[int, SpanHold] | None = None,
    ) -> None:
        self.utt = utt
        self.holds = holds
        self.spans = spans
        self.scene = scene
        self.p = params or plotter_profile().resolver_params()
        self.default_size = default_size
        self.avoid = size_holds(holds, spans, self.p)
        self.picks: dict[int, Pick | Ask] = {}
        self.overrides = overrides or {}  # word index -> hold, from a clarifying answer
        self.size_overrides = size_overrides or {}  # word index -> span, from "this big"
        self.log: list[dict[str, Any]] = []

    def prepare(self, calls: list[Any]) -> None:
        idx: list[int] = []
        for c in calls:
            places, _ = deictic_words(c.args)
            idx += places
        idx = sorted(set(i for i in idx if i < len(self.utt.words) and i not in self.overrides))
        words = [self.utt.words[i] for i in idx]
        for i, pick in zip(idx, assign_holds(words, self.holds, self.p, self.avoid), strict=True):
            self.picks[i] = pick
        for i, h in self.overrides.items():
            if 0 <= i < len(self.utt.words):
                self.picks[i] = Pick(h, "answer", self.utt.words[i])

    # ------------------------------------------------------------------ holds

    def _pick(self, word_index: int) -> Pick:
        pick = self.picks.get(word_index)
        if pick is None:
            w = self.utt.words[word_index]
            [pick] = assign_holds([w], self.holds, self.p, self.avoid)
            self.picks[word_index] = pick
        if isinstance(pick, Ask):
            raise NeedAsk(Question(pick.question, pick.reason))
        return pick

    def _where(self, pick: Pick) -> Where:
        w = hold_where(pick, "paper", self.p)
        self.log.append({"where": w.model_dump(mode="json")})
        return w

    def point_for_word(self, word_index: int) -> tuple[Point, Where]:
        pick = self._pick(word_index)
        w = self._where(pick)
        assert w.mean_mm is not None
        return w.mean_mm, w

    def last_point(self) -> tuple[Point, Where]:
        if -1 in self.overrides:  # answered "where?" by pointing again
            pick: Pick | Ask = Pick(self.overrides[-1], "answer")
        else:
            pick = last_hold(self.utt.t0, self.utt.t1, self.holds, self.p, self.avoid)
        if isinstance(pick, Ask):
            raise NeedAsk(Question(pick.question, pick.reason))
        w = self._where(pick)
        assert w.mean_mm is not None
        return w.mean_mm, w

    # ------------------------------------------------------------------ objects

    def obj(self, ref: Any) -> SceneObject:
        if isinstance(ref, LastObject):
            o = self.scene.last()
            if o is None:
                raise NeedAsk(Question("There's nothing drawn yet. What do you mean?", "no_object"))
            return o
        if isinstance(ref, LabelObject):
            hits = self.scene.find(ref.label)
            if not hits:
                opts = self.scene.labels()[:4]
                raise NeedAsk(
                    Question(
                        f"I don't see a {ref.label}. Which one do you mean?", "no_object", opts
                    )
                )
            return hits[0]
        if isinstance(ref, DeicticObject):
            p, _ = self.point_for_word(ref.word_index)
            o, cands = self.scene.object_at(p)
            if o is not None:
                return o
            if cands:
                opts = [c.label or c.id for c in cands]
                text = "The " + " or the ".join(opts) + "?"
                raise NeedAsk(Question(text, "ambiguous_object", opts, cands))
            raise NeedAsk(
                Question("I don't see anything there. Which one do you mean?", "no_object")
            )
        raise TypeError(f"not an object reference: {ref!r}")

    # ------------------------------------------------------------------ sizes

    def size(self, ref: Any, default: float | None = None) -> float:
        if ref is None:
            return self.default_size if default is None else default
        if isinstance(ref, PresetSize):
            mm = PRESET_MM.get(ref.preset)
            if mm is None:  # whole page
                d = self.scene.drawable
                return 0.95 * min(d.w, d.h)
            return mm
        if isinstance(ref, RelativeSize):
            o = self.obj(ref.same_as)
            return max(o.box.w, o.box.h)
        if isinstance(ref, DeicticSize):
            w = self.utt.words[ref.word_index]
            span = self.size_overrides.get(ref.word_index)
            if span is None:
                pick = pick_span(w, self.spans, self.p)
                if isinstance(pick, Ask):
                    raise NeedAsk(Question(pick.question, pick.reason))
                span = pick.hold
            assert isinstance(span, SpanHold)
            entry = {"size": round(span.span_mm, 2), "size_kind": span.kind, "word": w.text}
            self.log.append(entry)
            return span.span_mm
        raise TypeError(f"not a size reference: {ref!r}")

    # ------------------------------------------------------------------ places

    def place(self, ref: Any, size: float) -> Place:
        if ref is None:
            c, w = self.last_point()
            return Place(c, w)
        if isinstance(ref, DeicticPlace):
            c, w = self.point_for_word(ref.word_index)
            return Place(c, w)
        if isinstance(ref, RelationPlace):
            t1 = self.obj(ref.target)
            t2 = self.obj(ref.target2) if ref.target2 is not None else None
            new_size = None
            if ref.relation == "inside":
                new_size = min(size, self.scene.inside_size(t1))
            c = self.scene.relation(ref.relation, t1, new_size or size, t2)
            ev = {"relation": ref.relation, "target": t1.id}
            if t2 is not None:
                ev["target2"] = t2.id
            return Place(c, _geo(c, ev), new_size, t1)
        if isinstance(ref, NamedPlace):
            c = self.scene.named(ref.named, size)
            if c is None:
                q = f"Where is '{ref.named}'? Please point at it."
                raise NeedAsk(Question(q, "unknown_place"))
            return Place(c, _geo(c, {"named": ref.named}))
        if isinstance(ref, EmptySpace):
            res = self.scene.empty_space(size)
            if res is None:
                raise NeedAsk(Question("The sheet is full. Shall I start a new page?", "no_room"))
            c, side = res
            return Place(c, _geo(c, {"empty_space": True, "side_mm": round(side, 1)}), side)
        raise TypeError(f"not a place reference: {ref!r}")


def _geo(c: Point, evidence: dict[str, Any]) -> Where:
    return Where(
        frame="paper",
        kind="point",
        mean_mm=(round(c[0], 2), round(c[1], 2)),
        source="geometry",
        evidence=evidence,
    )
