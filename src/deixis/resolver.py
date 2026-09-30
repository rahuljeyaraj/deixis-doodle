"""Deixis resolver v1 (plan 4.3): pair spoken words with pointing holds and size gestures.

Robot-agnostic: this module knows words, holds and size gestures, and returns `Where`
messages or a reason to ask. Scene objects, relations and the plotter live in `pointdraw`.

Timing rules (LR§3.4, all windows are parameters):

- **With a demonstrative** spoken at [w0, w1]: candidates are holds overlapping
  [w0 - 1.5 s, w1 + 0.5 s]. Pick the one whose midpoint is nearest the word's midpoint;
  on a tie, the earlier hold (the gesture usually comes first).
- **k demonstratives** ("from this to that"): take k holds in time order, choosing the
  order-preserving assignment with the smallest total time distance.
- **No demonstrative:** the last hold in [speech_start - 2 s, speech_end].
- **Size ("this big"):** the same window rule over steady size gestures (a pinch span or
  the distance between two index fingertips).
- **Ask, don't guess:** no candidate at all; two candidates in different places with
  likelihoods closer than 2:1; or a hold that wandered too much.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from deixis.geometry import Point
from deixis.types import Hold, Word
from deixis.where import Where

SpanKind = Literal["pinch", "two_hands"]


@dataclass(frozen=True)
class SpanHold:
    """A steady size gesture: the span stayed about the same for a while."""

    t0: float
    t1: float
    span_mm: float
    kind: SpanKind
    center: Point  # mm; midpoint between the two fingertips

    @property
    def mid(self) -> float:
        return (self.t0 + self.t1) / 2


@dataclass(frozen=True)
class ResolverParams:
    """Timing windows and ask thresholds. Seconds and millimetres; tune on recordings."""

    before_s: float = 1.5  # a hold may end this long before the word starts
    after_s: float = 0.5  # ... or start this long after the word ends
    no_word_before_s: float = 2.0  # no demonstrative: holds from speech start minus this
    time_sigma_s: float = 0.6  # likelihood of a hold falls off with time distance
    ambiguity_ratio: float = 2.0  # best / second best below this -> ask
    same_place_mm: float = 10.0  # two candidates closer than this are the same place
    base_sigma_mm: float = 1.5  # added to the hold's spread in the covariance
    max_spread_mm: float = 6.0  # a hold that wandered more than this is too vague
    size_overlap: float = 0.5  # a hold this much covered by a pinch is a size gesture


@dataclass(frozen=True)
class Ask:
    """The resolver can't decide; the runtime should ask the person."""

    reason: Literal["no_gesture", "ambiguous", "unsteady", "no_size"]
    question: str
    candidates: tuple[Hold, ...] = ()
    word: Word | None = None


@dataclass
class Pick:
    """A chosen hold (or span) and how it was chosen, for the evidence record."""

    hold: Hold | SpanHold
    rule: str  # "nearest", "k_in_order", "last_hold"
    word: Word | None = None
    n_candidates: int = 1
    notes: list[str] = field(default_factory=list)


def _wmid(w: Word) -> float:
    return (w.t0 + w.t1) / 2


def window(word: Word, p: ResolverParams) -> tuple[float, float]:
    return (word.t0 - p.before_s, word.t1 + p.after_s)


def _overlaps(h: Hold | SpanHold, t_from: float, t_to: float) -> bool:
    return h.t1 >= t_from and h.t0 <= t_to


def _likelihood(h: Hold | SpanHold, word: Word, p: ResolverParams) -> float:
    d = h.mid - _wmid(word)
    return math.exp(-0.5 * (d / p.time_sigma_s) ** 2)


def _pos(h: Hold | SpanHold) -> Point:
    return h.mean if isinstance(h, Hold) else h.center


def size_holds(holds: Sequence[Hold], spans: Sequence[SpanHold], p: ResolverParams) -> set[Hold]:
    """Holds that are mostly a pinch ("this big") rather than a pointing gesture."""
    out = set()
    for h in holds:
        dur = max(h.t1 - h.t0, 1e-6)
        for s in spans:
            if s.kind != "pinch":
                continue
            ov = min(h.t1, s.t1) - max(h.t0, s.t0)
            if ov / dur >= p.size_overlap:
                out.add(h)
                break
    return out


def _ranked(
    word: Word, cands: Sequence[Hold | SpanHold], avoid: set | None = None
) -> list[Hold | SpanHold]:
    """Nearest midpoint first; ties to the earlier hold; `avoid` ones last."""
    avoid = avoid or set()
    return sorted(cands, key=lambda h: (h in avoid, abs(h.mid - _wmid(word)), h.t0))


def pick_hold(
    word: Word,
    holds: Sequence[Hold],
    p: ResolverParams | None = None,
    avoid: set[Hold] | None = None,
) -> Pick | Ask:
    """The hold that goes with one demonstrative word ("here", "that")."""
    p = p or ResolverParams()
    t_from, t_to = window(word, p)
    cands = [h for h in holds if _overlaps(h, t_from, t_to)]
    if not cands:
        return Ask("no_gesture", "Where do you mean? Point at the paper.", word=word)
    ranked = _ranked(word, cands, avoid)
    usable = [h for h in ranked if h not in (avoid or set())] or ranked
    best = usable[0]
    if len(usable) >= 2:
        second = usable[1]
        l1, l2 = _likelihood(best, word, p), _likelihood(second, word, p)
        far = math.dist(_pos(best), _pos(second)) > p.same_place_mm
        if far and l2 > 0 and l1 / l2 < p.ambiguity_ratio:
            return Ask(
                "ambiguous",
                "Did you mean this place or that one? Please point again.",
                candidates=(best, second),  # type: ignore[arg-type]
                word=word,
            )
    assert isinstance(best, Hold)
    if max(best.spread) > p.max_spread_mm:
        return Ask("unsteady", "I couldn't tell where you pointed. Hold still a moment.", (best,))
    return Pick(best, "nearest", word, len(cands))


def assign_holds(
    words: Sequence[Word],
    holds: Sequence[Hold],
    p: ResolverParams | None = None,
    avoid: set[Hold] | None = None,
) -> list[Pick | Ask]:
    """One hold per demonstrative, in time order ("from this to that").

    With one word this is `pick_hold`. With k words, candidates overlapping the whole
    utterance's window are matched in order, minimising the total time distance.
    """
    p = p or ResolverParams()
    if not words:
        return []
    if len(words) == 1:
        return [pick_hold(words[0], holds, p, avoid)]
    t_from, t_to = words[0].t0 - p.before_s, words[-1].t1 + p.after_s
    cands = sorted(
        (h for h in holds if _overlaps(h, t_from, t_to) and h not in (avoid or set())),
        key=lambda h: h.t0,
    )
    k, n = len(words), len(cands)
    if n < k:
        # Not enough holds for every word: resolve each on its own, without reusing a hold.
        out: list[Pick | Ask] = []
        used: set[Hold] = set(avoid or ())
        for w in words:
            r = pick_hold(w, [h for h in holds if h not in used], p)
            if isinstance(r, Pick):
                used.add(r.hold)  # type: ignore[arg-type]
            out.append(r)
        return out
    # DP over (word i, candidate j): cost of matching words[:i+1] with word i -> cands[j].
    inf = math.inf
    cost = [[inf] * n for _ in range(k)]
    back = [[-1] * n for _ in range(k)]
    for j in range(n):
        cost[0][j] = abs(cands[j].mid - _wmid(words[0]))
    for i in range(1, k):
        best_prev, best_j = inf, -1
        for j in range(n):
            if j - 1 >= 0 and cost[i - 1][j - 1] < best_prev:
                best_prev, best_j = cost[i - 1][j - 1], j - 1
            if best_j >= 0:
                cost[i][j] = best_prev + abs(cands[j].mid - _wmid(words[i]))
                back[i][j] = best_j
    j = min(range(n), key=lambda jj: cost[k - 1][jj])
    chosen = [0] * k
    for i in range(k - 1, -1, -1):
        chosen[i] = j
        j = back[i][j]
    return [Pick(cands[chosen[i]], "k_in_order", words[i], n) for i in range(k)]


def last_hold(
    speech_t0: float,
    speech_t1: float,
    holds: Sequence[Hold],
    p: ResolverParams | None = None,
    avoid: set[Hold] | None = None,
) -> Pick | Ask:
    """No demonstrative was spoken ("a tree" while pointing): the last hold in the window."""
    p = p or ResolverParams()
    cands = [h for h in holds if _overlaps(h, speech_t0 - p.no_word_before_s, speech_t1)]
    usable = [h for h in cands if h not in (avoid or set())] or cands
    if not usable:
        return Ask("no_gesture", "Where should it go? Point at the paper.")
    best = max(usable, key=lambda h: (h.t1, h.t0))
    return Pick(best, "last_hold", None, len(cands))


def pick_span(word: Word, spans: Sequence[SpanHold], p: ResolverParams | None = None) -> Pick | Ask:
    """The size gesture that goes with "this big" (window rule, nearest midpoint)."""
    p = p or ResolverParams()
    t_from, t_to = window(word, p)
    cands = [s for s in spans if _overlaps(s, t_from, t_to)]
    if not cands:
        return Ask("no_size", "How big? Show me with your fingers or hands.", word=word)
    best = _ranked(word, cands)[0]
    return Pick(best, "nearest", word, len(cands))


def hold_where(pick: Pick, frame: str = "paper", p: ResolverParams | None = None) -> Where:
    """A point `Where` from a chosen hold, with its covariance and the evidence."""
    p = p or ResolverParams()
    h = pick.hold
    assert isinstance(h, Hold)
    sx = math.hypot(h.spread[0], p.base_sigma_mm)
    sy = math.hypot(h.spread[1], p.base_sigma_mm)
    ev: dict[str, object] = {
        "rule": pick.rule,
        "hold_t": [round(h.t0, 3), round(h.t1, 3)],
        "candidates": pick.n_candidates,
    }
    if pick.word is not None:
        ev["word"] = pick.word.text
        ev["word_t"] = [round(pick.word.t0, 3), round(pick.word.t1, 3)]
        ev["lead_s"] = round(pick.word.t0 - h.mid, 3)  # > 0: the gesture came first
    return Where(
        frame=frame,
        kind="point",
        mean_mm=(round(h.mean[0], 2), round(h.mean[1], 2)),
        cov_mm2=((sx * sx, 0.0), (0.0, sy * sy)),
        source="gesture",
        evidence=ev,
    )
