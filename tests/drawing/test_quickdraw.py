"""Quick, Draw! parsing, features, filters and ranking, on made-up drawings."""

from pointdraw.drawing.quickdraw import (
    RawDrawing,
    ScoreParams,
    count_stray,
    features,
    rank,
    reject_reason,
    to_library_strokes,
)


def drawing(strokes, key="k", recognized=True):
    return RawDrawing("thing", key, tuple(tuple(s) for s in strokes), recognized)


SQUARE = [(0, 0), (255, 0), (255, 200), (0, 200), (0, 0)]
CROSS = [(50, 100), (200, 100)]


def test_ndjson_round_trip():
    obj = {"word": "cat", "key_id": 42, "recognized": True, "drawing": [[[0, 5], [1, 6]]]}
    d = RawDrawing.from_ndjson(obj)
    assert d.key_id == "42"
    assert d.strokes == (((0, 1), (5, 6)),)
    assert RawDrawing.from_ndjson(d.to_ndjson()) == d


def test_features():
    f = features(drawing([SQUARE, CROSS]))
    assert f.n_strokes == 2
    assert f.ink == 255 * 2 + 200 * 2 + 150
    assert f.shortest == 150
    assert f.aspect == 255 / 200
    assert f.n_stray == 0


def test_reject_reasons():
    p = ScoreParams()
    assert reject_reason(features(drawing([SQUARE, CROSS])), p) is None
    assert "strokes" in reject_reason(features(drawing([SQUARE])), p)
    assert reject_reason(features(drawing([SQUARE, [(10, 10), (11, 10)]])), p) == "has a dot"
    thin = [[(0, 0), (255, 0)], [(0, 10), (255, 10)]]
    assert reject_reason(features(drawing(thin)), p) in ("too little ink", "extreme aspect ratio")


def test_stray_strokes():
    inside = [(100, 100), (120, 110)]  # a pupil: inside the outline's bbox
    outside = [(100, 240), (110, 255)]  # a stray tick far below
    body = [(0, 0), (255, 0), (255, 180), (0, 180), (0, 0)]
    assert count_stray([body, inside], 15) == 0
    assert count_stray([body, outside], 15) == 1


def test_rank_prefers_typical_and_drops_unrecognised():
    typical = [drawing([SQUARE, CROSS], key=f"t{i}") for i in range(10)]
    odd = drawing(
        [SQUARE, CROSS, *[[(i * 20, 50), (i * 20 + 10, 60)] for i in range(9)]], key="odd"
    )
    bad = drawing([SQUARE, CROSS], key="bad", recognized=False)
    ranked = rank([odd, bad, *typical], top=5)
    keys = [s.drawing.key_id for s in ranked]
    assert "bad" not in keys and "odd" not in keys
    assert keys == sorted(keys)  # ties break by key_id: deterministic


def test_rank_allows_one_stroke_when_typical():
    clouds = [drawing([SQUARE], key=f"c{i}") for i in range(5)]
    assert len(rank(clouds, top=30)) == 5


def test_to_library_strokes_flips_y_and_moves_to_origin():
    d = drawing([[(10, 20), (30, 60)]])
    assert to_library_strokes(d) == [[[0, 40], [20, 0]]]
