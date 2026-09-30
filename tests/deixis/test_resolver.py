"""Resolver v1 timing rules (plan 4.3) on hand-made holds and words."""

from deixis.resolver import (
    Ask,
    Pick,
    SpanHold,
    assign_holds,
    hold_where,
    last_hold,
    pick_hold,
    pick_span,
    size_holds,
)
from deixis.types import Hold, Word


def hold(t0, t1, x=50.0, y=50.0, spread=0.3):
    return Hold(t0, t1, (x, y), (spread, spread))


def test_nearest_hold_in_window():
    here = Word("here", 2.0, 2.3)
    h1, h2 = hold(0.2, 0.6, 10, 10), hold(1.2, 2.1, 80, 80)
    p = pick_hold(here, [h1, h2])
    assert isinstance(p, Pick) and p.hold is h2


def test_hold_before_the_word_counts():
    """The gesture comes first: a hold that ended 1.2 s before the word is still used."""
    here = Word("here", 3.0, 3.3)
    p = pick_hold(here, [hold(0.8, 1.8)])
    assert isinstance(p, Pick)


def test_outside_window_asks():
    here = Word("here", 5.0, 5.3)
    r = pick_hold(here, [hold(0.5, 3.0)])  # ended 2 s before the word
    assert isinstance(r, Ask) and r.reason == "no_gesture"


def test_tie_goes_to_the_earlier_hold_when_same_place():
    here = Word("here", 2.0, 2.2)  # midpoint 2.1
    early, late = hold(1.5, 1.9, 50, 50), hold(2.3, 2.7, 52, 50)
    p = pick_hold(here, [late, early])
    assert isinstance(p, Pick) and p.hold is early


def test_two_places_equally_likely_asks():
    here = Word("here", 2.0, 2.2)
    r = pick_hold(here, [hold(1.5, 1.9, 20, 20), hold(2.3, 2.7, 150, 100)])
    assert isinstance(r, Ask) and r.reason == "ambiguous"


def test_k_demonstratives_take_holds_in_order():
    words = [Word("this", 1.0, 1.2), Word("that", 2.5, 2.7)]
    h1, h2 = hold(0.4, 1.3, 10, 10), hold(2.0, 2.9, 90, 90)
    picks = assign_holds(words, [h2, h1])
    assert [p.hold for p in picks] == [h1, h2]


def test_last_hold_without_demonstrative():
    h1, h2 = hold(0.5, 1.0, 10, 10), hold(1.5, 2.4, 70, 30)
    p = last_hold(1.2, 2.0, [h1, h2])
    assert isinstance(p, Pick) and p.hold is h2
    assert isinstance(last_hold(10.0, 11.0, [h1, h2]), Ask)


def test_size_gesture_and_size_holds():
    this = Word("this", 2.4, 2.6)
    span = SpanHold(2.3, 3.2, 45.0, "pinch", (70, 90))
    p = pick_span(this, [span])
    assert isinstance(p, Pick) and p.hold.span_mm == 45.0
    assert isinstance(pick_span(Word("this", 9, 9.2), [span]), Ask)
    pinch_hold = hold(2.3, 3.2)
    assert size_holds([pinch_hold, hold(0.5, 2.0)], [span], _params()) == {pinch_hold}


def test_where_records_the_evidence():
    here = Word("here", 2.0, 2.3)
    p = pick_hold(here, [hold(1.2, 1.9, 62, 88)])
    w = hold_where(p)
    assert w.kind == "point" and w.mean_mm == (62, 88) and w.source == "gesture"
    assert w.evidence["word"] == "here" and w.evidence["lead_s"] > 0


def _params():
    from deixis.resolver import ResolverParams

    return ResolverParams()
