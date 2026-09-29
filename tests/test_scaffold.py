"""Tests for the shared types every workstream builds on."""

import json

import pytest

from deixis import Box, Hold, Where
from deixis.where import where_json_schema
from pointdraw import clock, layout
from pointdraw.config import Settings


def test_box_helpers():
    b = Box.from_center(100, 50, 40, 20)
    assert b.center == (100, 50)
    assert (b.x, b.y, b.x2, b.y2) == (80, 40, 120, 60)
    assert b.contains((100, 50))
    assert not b.contains((81, 50), margin=5)
    assert b.inset(5) == Box(85, 45, 30, 10)


def test_where_point_box_path():
    p = Where(frame="paper", kind="point", mean_mm=(10, 20), source="gesture")
    assert p.mean_mm == (10, 20)

    b = Where(frame="paper", kind="box", box_mm=(10, 20, 40, 30), source="gesture")
    assert b.mean_mm == (30, 35)

    path = Where(frame="paper", kind="path", path_mm=[(0, 0), (50, 50)], source="object_ref")
    assert len(path.path_mm) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "point"},
        {"kind": "box"},
        {"kind": "box", "box_mm": (0, 0, -1, 5)},
        {"kind": "path", "path_mm": [(0, 0)]},
    ],
)
def test_where_rejects_incomplete(kwargs):
    with pytest.raises(ValueError):
        Where(frame="paper", source="gesture", **kwargs)


def test_where_json_round_trip():
    w = Where(
        frame="floor",
        kind="box",
        box_mm=(1000, 500, 800, 800),
        cov_mm2=((25.0, 0.0), (0.0, 25.0)),
        source="gesture",
        evidence={"word": "here", "word_t": 12.3},
    )
    again = Where.model_validate_json(w.model_dump_json())
    assert again == w
    assert "properties" in json.loads(json.dumps(where_json_schema()))


def test_hold_mid():
    h = Hold(t0=1.0, t1=1.4, mean=(5, 5), spread=(0.5, 0.5))
    assert h.mid == pytest.approx(1.2)


def test_clock_is_monotonic():
    a = clock.now()
    b = clock.now()
    assert b >= a
    assert isinstance(clock.now_ms(), int)


def test_layout_ids_disjoint():
    assert not set(layout.BED_MARKER_IDS) & set(layout.FLOOR_MARKER_IDS)
    assert layout.PAPER_WIDTH_MM > layout.PAPER_HEIGHT_MM


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    s = Settings()
    assert s.nebius_api_key.get_secret_value() == ""
    assert s.nebius_base_url.startswith("https://")
