"""M01 to M09 from logs: matching drawings to printed targets, and the timeline numbers."""

import json

from pointdraw.measure import accuracy, logs
from pointdraw.measure.targets import Target

TARGETS = [Target(1, 50.0, 50.0), Target(2, 100.0, 50.0), Target(3, 150.0, 120.0, 40.0)]
COV = [[4.0, 0.0], [0.0, 4.0]]


def row(i, x, y, size=10.0, mean=None):
    return {
        "id": f"c{i}",
        "label": "cross",
        "ink_xy": None if x is None else [x, y],
        "ink_size": size,
        "where": {"mean_mm": mean or [x, y], "cov_mm2": COV},
    }


def test_crosses_are_matched_and_scored():
    rows = [
        row(1, 52.0, 50.0),  # 2 mm off, pointed right on it
        row(2, 100.0, 53.0, mean=[100.0, 56.0]),  # 3 mm off, pointed 6 mm off: outside 95 %
        row(3, None, None),  # no ink: not counted
        row(4, 300.0, 300.0),  # far from every target: not counted
    ]
    res, s = accuracy.match(rows, TARGETS, "crosses")
    assert [r["counted"] for r in res] == [True, True, False, False]
    assert s["centre_median_mm"] == 2.5 and s["within_4mm"] == 2
    assert s["inside_95"] == 1 and s["inside_95_of"] == 2
    text = "\n".join(accuracy.lines(s))
    assert "M01" in text and "M03 your target inside the 95 % circle: 1 of 2" in text


def test_circles_size_error():
    rows = [row(1, 150.0, 121.0, size=46.0), row(2, 151.0, 120.0, size=50.0)]
    _, s = accuracy.match(rows, TARGETS, "circles")
    assert s["size_of"] == 2 and s["size_within_20pct"] == 1  # +15 % and +25 %
    assert "M02" in "\n".join(accuracy.lines(s))


def write_run(tmp_path, events):
    run = tmp_path / "run"
    run.mkdir()
    (run / "timeline.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return run


def test_timeline_numbers(tmp_path):
    ev = [
        {"t": 0.5, "kind": "hold", "t0": 0.2, "t1": 1.4, "mean": [10, 10], "spread": 1},
        {
            "t": 2.0,
            "kind": "utterance",
            "text": "draw a kite here this big",
            "words": [
                ["draw", 0.1, 0.3],
                ["a", 0.3, 0.4],
                ["kite", 0.4, 0.8],
                ["here", 0.9, 1.1],
                ["this", 1.3, 1.5],
                ["big", 1.5, 1.7],
            ],
        },
        {"t": 3.0, "kind": "item", "id": "kite", "label": "kite"},
        {"t": 3.0, "kind": "preview", "items": ["kite"]},
        {"t": 3.0, "kind": "state", "state": "PREVIEW"},
        {"t": 3.2, "kind": "plotter", "event": "job_start", "label": "preview", "tag": "p1"},
        {"t": 6.0, "kind": "plotter", "event": "job_end", "label": "preview", "tag": "p1"},
        {
            "t": 8.0,
            "kind": "utterance",
            "text": "a bit bigger",
            "words": [["a", 7.0, 7.1], ["bit", 7.1, 7.3], ["bigger", 7.3, 7.7]],
        },
        {"t": 8.5, "kind": "adjust", "id": "kite", "dx": 0, "dy": 0, "scale": 1.25},
        {"t": 8.5, "kind": "preview", "items": ["kite"]},
        {"t": 12.0, "kind": "plotter", "event": "job_start", "label": "draw", "tag": "kite"},
        {"t": 30.0, "kind": "plotter", "event": "job_end", "label": "draw", "tag": "kite"},
        {"t": 31.0, "kind": "plotter", "event": "job_start", "label": "park", "tag": "park#2"},
        {"t": 33.0, "kind": "plotter", "event": "job_end", "label": "park", "tag": "park#2"},
    ]
    res = logs.summary([write_run(tmp_path, ev)])
    m4 = res["M04_last_word_to_guess_s"]
    assert m4["n"] == 2 and abs(m4["min"] - 0.8) < 1e-6  # "big" ends 1.7, trace at 3.2: 1.5
    m5 = res["M05_hand_before_word"]
    assert m5["words"] == 1 and m5["hold_started_first"] == 1  # "this big" isn't pointing
    assert abs(m5["median_lead_s"] - 0.7) < 1e-6
    assert res["M06_draw_time_s"]["kite"]["median"] == 21.0
    assert res["M09_bigger"] == {"tries": 1, "worked": 1}
    assert res["M07_take_length_s"][0]["s"] == 32.9
    assert any(x.startswith("M09") for x in logs.lines(res))
