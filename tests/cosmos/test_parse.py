"""JSON extraction and coordinate normalisation for Cosmos replies."""

import importlib.util
from pathlib import Path

import pytest

from pointdraw.cosmos.parse import (
    detect_coord_mode,
    extract_json,
    normalise_label,
    parse_detections,
    parse_label,
    strip_reasoning,
)

_spec = importlib.util.spec_from_file_location(
    "replies_22sep", Path(__file__).parent / "data" / "replies_22sep.py"
)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_fenced_grounding_reply_to_pixels():
    dets = parse_detections(R.GROUNDING_PAGE1, 1920, 1080)
    assert [d.label for d in dets] == ["paper", "elephant"]
    x1, y1, x2, y2 = dets[1].box_px
    assert (x1, y1, x2, y2) == pytest.approx((251 * 1.92, 264 * 1.08, 540 * 1.92, 512 * 1.08))
    assert dets[1].center_px == pytest.approx(((x1 + x2) / 2, (y1 + y2) / 2))


def test_three_objects():
    dets = parse_detections(R.GROUNDING_PAGE10, 1920, 1080)
    assert [d.label for d in dets] == ["paper", "flower", "flowerpot"]


def test_describe_reply_with_corner_pairs_and_objects_wrapper():
    dets = parse_detections(R.DESCRIBE_PAGE10, 1000, 1000)
    assert [d.label for d in dets] == ["child's drawing", "table"]
    assert dets[0].box_px == pytest.approx((235, 90, 768, 841))


def test_positions_as_words_give_no_detections():
    assert parse_detections(R.DESCRIBE_PAGE3, 1920, 1080) == []


def test_think_block_and_prose_are_ignored():
    text = (
        "<think>The circle is at the left, roughly [10, 10, 20, 20] maybe.</think>\n"
        'Here you go: [{"label": "circle", "bbox_2d": [100, 200, 300, 400]}] Hope that helps.'
    )
    dets = parse_detections(text, 1000, 1000)
    assert len(dets) == 1
    assert dets[0].box_px == pytest.approx((100, 200, 300, 400))


def test_missing_open_think_tag_and_answer_tags():
    text = 'I see a star.</think>\n<answer>\n{"label": "star"}\n</answer>'
    assert strip_reasoning(text) == '{"label": "star"}'
    assert extract_json(text) == {"label": "star"}


def test_unclosed_think_keeps_the_answer_before_it():
    assert strip_reasoning("[1, 2]\n<think>and now I ramble") == "[1, 2]"


def test_trailing_commas_are_tolerated():
    assert extract_json('```json\n[{"label": "a", "point_2d": [1, 2],},]\n```') == [
        {"label": "a", "point_2d": [1, 2]}
    ]


def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("I cannot see any drawing.")


def test_points_and_bare_lists():
    dets = parse_detections('[{"point_2d": [500, 250], "label": "empty"}]', 800, 600)
    assert dets[0].point_px == pytest.approx((400, 150))
    assert dets[0].box_px is None
    bare = parse_detections("[500, 500]", 800, 600)
    assert bare[0].point_px == pytest.approx((400, 300))


def test_boxes_are_reordered_and_clipped():
    dets = parse_detections('[{"bbox_2d": [900, 800, 100, 1200]}]', 1000, 1000, "norm1000")
    assert dets[0].box_px == (100, 800, 900, 1000)


def test_coordinate_mode_detection():
    assert detect_coord_mode([0.1, 0.5, 0.9], 1920, 1080) == "norm1"
    assert detect_coord_mode([120, 999], 1920, 1080) == "norm1000"
    assert detect_coord_mode([120, 1500], 1920, 1080) == "pixels"
    assert detect_coord_mode([120, 5000], 1920, 1080) == "norm1000"  # out of range: not pixels
    dets = parse_detections('[{"bbox_2d": [0.1, 0.2, 0.3, 0.4]}]', 1000, 500)
    assert dets[0].box_px == pytest.approx((100, 100, 300, 200))
    px = parse_detections('[{"bbox_2d": [100, 200, 1300, 400]}]', 1920, 1080)
    assert px[0].box_px == pytest.approx((100, 200, 1300, 400))


def test_yx_order():
    dets = parse_detections('[{"point_2d": [250, 500]}]', 1000, 1000, "norm1000", "yx")
    assert dets[0].point_px == pytest.approx((500, 250))


@pytest.mark.parametrize(
    "reply, label",
    [
        ('{"label": "Cat", "alternatives": ["dog"]}', "cat"),
        ('```json\n{"label": "a smiley face"}\n```', "smiley face"),
        ("A drawing of a house.", "house"),
        ("<think>hmm</think>\nTree", "tree"),
        ('[{"label": "the sun"}]', "sun"),
    ],
)
def test_parse_label(reply, label):
    assert parse_label(reply) == label


def test_normalise_label():
    assert normalise_label("  'A drawing of a Cat.' ") == "cat"
    assert normalise_label("It's   an  arrow!") == "arrow"
