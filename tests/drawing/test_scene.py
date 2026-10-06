"""Scene composition and rendering, using the committed library (offline)."""

import json
from pathlib import Path

import pytest

from deixis.geometry import Box
from pointdraw.drawing import Library, compose, place_object, plan
from pointdraw.drawing.place import circle_size
from pointdraw.drawing.render import A5, render, render_cell

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lib() -> Library:
    return Library.load(ROOT / "library")


def test_place_object_size_and_centre(lib):
    p = place_object("ear", (60, 70), 40, lib, variant=0)
    assert p.category == "ear" and p.key_id
    assert circle_size(p.polylines) == pytest.approx(40)  # its farthest line on the circle
    assert max(p.box.w, p.box.h) <= 40
    assert p.box.center == pytest.approx((60, 70))
    assert p.size_circle == ((60, 70), 40)


@pytest.mark.parametrize("scene", ["hero-senses.json", "garden.json"])
def test_storyboard_scenes_fit_on_paper(lib, scene):
    spec = json.loads((ROOT / "library" / "scenes" / scene).read_text(encoding="utf-8"))
    placed, paper = compose(spec, lib)
    assert paper == A5
    for p in placed:
        assert paper.contains((p.box.x, p.box.y)) and paper.contains((p.box.x2, p.box.y2)), p.id
    polylines, stats = plan(placed)
    assert stats.lifts == len(polylines)
    assert stats.est_time_s < 240  # a storyboard page should plot in a few minutes


def test_arrows_start_and_end_outside_objects(lib):
    spec = {
        "items": [
            {"type": "object", "id": "a", "name": "sun", "at": [40, 74], "size": 30},
            {"type": "object", "id": "b", "name": "moon", "at": [170, 74], "size": 30},
            {"type": "arrow", "from": "a", "to": "b"},
            {"type": "label", "text": "day", "below": "a", "size": 5},
        ]
    }
    placed, _ = compose(spec, lib)
    a, b, arrow, label = placed
    shaft = arrow.polylines[0]
    assert not a.box.contains(shaft[0]) and not b.box.contains(shaft[-1])
    assert label.box.y2 < a.box.y  # under the sun
    assert label.box.center[0] == pytest.approx(a.box.center[0], abs=1)


def test_compose_errors(lib):
    with pytest.raises(KeyError):
        compose({"items": [{"type": "arrow", "from": "nope", "to": [0, 0]}]}, lib)
    with pytest.raises(ValueError):
        compose({"items": [{"type": "spiral"}]}, lib)


def test_primitives_in_specs(lib):
    spec = {
        "paper": [0, 0, 100, 100],
        "items": [
            {"type": "box", "box": [10, 10, 20, 20]},
            {"type": "circle", "at": [50, 50], "r": 10},
            {"type": "line", "from": [0, 90], "to": [100, 90]},
        ],
    }
    placed, paper = compose(spec, lib)
    assert paper == Box(0, 0, 100, 100)
    assert placed[0].box == Box(10, 10, 20, 20)
    assert placed[1].box.w == pytest.approx(20, abs=0.01)


def test_render_png(tmp_path):
    out = tmp_path / "x.png"
    img = render(
        [[(0, 0), (210, 148)], [(5, 5)]], out, px_per_mm=2, margin_mm=5, boxes=[Box(10, 10, 20, 20)]
    )
    assert out.exists()
    assert img.size == (round(220 * 2), round(158 * 2))
    # the diagonal passes through the paper's centre (105, 74) -> pixel (220, 158)
    assert img.getpixel((220, 158))[0] < 100
    # y up: the paper's bottom edge is near the bottom of the image
    assert img.getpixel((220, img.size[1] - 5))[0] > 200  # margin, not paper edge


def test_render_cell():
    img = render_cell([[(0, 0), (10, 10)]], size_px=64)
    assert img.size == (64, 64)
