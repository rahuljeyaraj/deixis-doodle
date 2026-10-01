"""Simple and detailed drawings: SANA variants carry a style (icon or colouring)."""

from __future__ import annotations

import json

import pytest

from pointdraw.drawing.library import Library, style_tag
from pointdraw.drawing.scene import place_object

SQUARE = [[[0, 0], [255, 0], [255, 255], [0, 255], [0, 0]]]
CROSS = [[[0, 0], [255, 255]], [[0, 255], [255, 0]]]


@pytest.fixture
def lib(tmp_path):
    variants = [
        {"key_id": "cat_icon_s1000", "style": "icon", "strokes": SQUARE},
        {"key_id": "cat_icon_s1001", "style": "icon", "strokes": SQUARE},
        {"key_id": "cat_colouring_s1000", "style": "colouring", "strokes": CROSS},
    ]
    (tmp_path / "cat.json").write_text(json.dumps({"variants": variants}))
    (tmp_path / "dog.json").write_text(json.dumps({"variants": [{"key_id": 1, "strokes": SQUARE}]}))
    index = {
        "default_style": "colouring",
        "categories": {
            "cat": {"file": "cat.json", "variants": 3, "synonyms": []},
            "dog": {"file": "dog.json", "variants": 1, "synonyms": []},
        },
        "synonyms": {},
    }
    (tmp_path / "index.json").write_text(json.dumps(index))
    return Library.load(tmp_path)


def test_style_words():
    assert style_tag("simple") == "icon" and style_tag("Detailed") == "colouring"
    assert style_tag(None) is None and style_tag("purple") is None


def test_simple_and_detailed_pick_their_style(lib):
    assert {d.key_id for d in lib.variants("cat", "simple")} == {"cat_icon_s1000", "cat_icon_s1001"}
    assert [d.key_id for d in lib.variants("cat", "detailed")] == ["cat_colouring_s1000"]
    assert len(lib.variants("cat", "any")) == 3
    assert lib.styles("cat") == ["colouring", "icon"]


def test_no_style_word_uses_the_default_and_cycles_within_it(lib):
    assert lib.get("cat").style == "colouring"
    a, b, c = (lib.get("cat", style="simple").key_id for _ in range(3))
    assert a != b and a == c  # "another simple cat" gives the other icon


def test_a_category_without_styles_ignores_the_word(lib):
    assert lib.get("dog", style="detailed").key_id == "1"
    assert lib.styles("dog") == []


def test_place_object_passes_the_style(lib):
    p = place_object("cat", center=(50, 50), size=40, library=lib, style="simple")
    assert p.key_id.startswith("cat_icon")


def test_a_key_id_gets_that_drawing_and_does_not_advance_the_cycle(lib):
    """Re-placing a preview (moved, resized, or re-placed at the exact gap) must keep the same
    drawing: before 1 Oct each placement took the next variant, so the eye placed 'under the
    ear' was measured on one variant and drawn as another, with a different height."""
    first = lib.get("cat", style="simple")
    assert lib.get("cat", variant=first.key_id).key_id == first.key_id
    assert lib.get("cat", variant="cat_colouring_s1000").style == "colouring"
    assert lib.get("cat", style="simple").key_id != first.key_id  # the cycle moved on once
    with pytest.raises(ValueError):
        lib.get("cat", variant="no_such_key")
