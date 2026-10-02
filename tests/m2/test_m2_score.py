"""`scripts/m2_score.py`: the M2 scorer, on hand-made timelines and pictures."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from deixis.geometry import Box

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m2 = load_script("m2_score")


def words(text: str, t0: float) -> list:
    return [[w, t0 + 0.3 * i, t0 + 0.3 * i + 0.25] for i, w in enumerate(text.split())]


def where(x: float, y: float) -> dict:
    return {"frame": "paper", "kind": "point", "mean_mm": [x, y], "source": "gesture"}


def cat_timeline() -> list[dict]:
    """ "Draw a cat here, this big" in one go, a 40 mm pinch at (80, 90)."""
    said = words("Draw a cat here this big", 10.0)  # ends at 11.75
    return [
        {"t": 9.0, "kind": "speech_start"},
        {
            "t": 12.2,
            "kind": "utterance",
            "id": 0,
            "text": "Draw a cat here this big",
            "words": said,
        },
        {"t": 13.0, "kind": "tool_calls", "utt": 0, "calls": []},
        {
            "t": 13.0,
            "kind": "item",
            "id": "cat",
            "label": "cat",
            "box": [60, 75, 40, 30],
            "center": [80, 90],
            "where": where(80, 90),
        },
        {"t": 13.0, "kind": "resolve", "utt": 0, "size": 40.0, "size_kind": "pinch"},
        {"t": 13.0, "kind": "resolve", "utt": 0, "where": where(80, 90)},
        {"t": 13.0, "kind": "preview", "items": ["cat"], "box": [60, 75, 40, 30]},
        {
            "t": 13.05,
            "kind": "plotter",
            "event": "job_start",
            "label": "preview",
            "tag": "preview#1",
        },
        {"t": 20.0, "kind": "confirm", "items": ["cat"], "reason": "quiet"},
        {"t": 40.0, "kind": "drawn", "id": "cat", "label": "cat", "planned_box": [60, 75, 40, 30]},
    ]


def test_command_from_one_utterance():
    (c,) = m2.commands(cat_timeline())
    assert c.label == "cat"
    assert c.shown_xy == [80, 90] and c.planned_xy == [80, 90]
    assert c.shown_size == 40.0 and c.size_kind == "pinch"
    assert c.speech_end_t == pytest.approx(11.75)
    assert c.preview_start_t == 13.05  # the plotter's start, not the runtime's event
    assert c.latency_s == pytest.approx(1.3)
    assert c.asks == []


def test_answer_to_how_big_ends_the_speech():
    """ "Draw a cat here" -> "How big?" -> "This big": latency counts from "big"."""
    ev = [
        {
            "t": 11.0,
            "kind": "utterance",
            "id": 0,
            "text": "Draw a cat here",
            "words": words("Draw a cat here", 10.0),
        },
        {"t": 12.0, "kind": "tool_calls", "utt": 0, "calls": []},
        {"t": 12.0, "kind": "ask", "question": "How big?", "reason": "no_size"},
        {
            "t": 16.0,
            "kind": "utterance",
            "id": 1,
            "text": "This big",
            "words": words("This big", 15.0),
        },
        {"t": 16.0, "kind": "answer", "answer": "size"},
        {
            "t": 16.0,
            "kind": "item",
            "id": "cat",
            "label": "cat",
            "box": [60, 75, 40, 30],
            "center": [80, 90],
            "where": where(80, 90),
        },
        {"t": 16.0, "kind": "resolve", "utt": 0, "size": 40.0, "size_kind": "pinch"},
        {"t": 16.0, "kind": "preview", "items": ["cat"], "box": [60, 75, 40, 30]},
        {"t": 30.0, "kind": "drawn", "id": "cat", "label": "cat", "planned_box": [60, 75, 40, 30]},
    ]
    (c,) = m2.commands(ev)
    assert c.said == "This big"
    assert c.speech_end_t == pytest.approx(15.55)
    assert c.latency_s == pytest.approx(0.45)  # no plotter events: the runtime's preview
    assert c.asks == ["How big?"]
    assert c.shown_size == 40.0


def test_moved_to_fit_fails_on_centre():
    ev = cat_timeline()
    item = ev[3]
    item["center"] = [80, 100]  # the app moved it 10 mm so it fits
    item["box"] = [60, 85, 40, 30]
    (c,) = m2.commands(ev)
    c.ink_xy, c.ink_size = [80.2, 100.1], 40.5
    assert c.plot_err == pytest.approx(0.22, abs=0.01)
    assert c.centre_err == pytest.approx(10.1, abs=0.05)
    assert not c.passed


def test_pass_needs_centre_and_size():
    (c,) = m2.commands(cat_timeline())
    c.ink_xy, c.ink_size = [83.0, 92.0], 46.0  # 3.6 mm, +15 %
    assert c.passed
    c.ink_size = 49.0  # +22.5 %
    assert not c.passed


def test_find_ink_box_ignores_specks_and_far_marks():
    s = 0.25
    paper = Box(0, 0, 192, 196)
    H, W = round(paper.h / s), round(paper.w / s)
    mask = np.zeros((H, W), np.uint8)

    def px(x, y):
        return int(x / s), int(H - y / s)

    # a "cat": a body and a head with a gap between (pen lifted), 40 x 30 mm at (60, 75)
    cv2.rectangle(mask, px(60, 75), px(90, 95), 1, 2)
    cv2.circle(mask, px(92, 97), int(8 / s), 1, 2)
    mask[px(55, 70)[1], px(55, 70)[0]] = 1  # a speck just outside
    cv2.line(mask, px(150, 150), px(170, 150), 1, 2)  # a mark far away
    box, n = m2.find_ink(mask, [60, 75, 40, 30], s, 30)
    assert box[0] == pytest.approx(60, abs=0.5)
    assert box[1] == pytest.approx(75, abs=0.5)
    assert box[0] + box[2] == pytest.approx(100, abs=0.5)
    assert box[1] + box[3] == pytest.approx(105, abs=0.5)
    assert n > 100


def test_ink_mask_finds_a_faint_line_under_changed_light():
    rng = np.random.default_rng(0)
    before = np.full((400, 400, 3), 200, np.uint8)
    before = np.clip(before + rng.normal(0, 1.5, before.shape), 0, 255).astype(np.uint8)
    after = (before * 0.9).astype(np.uint8)  # dimmer light
    cv2.line(after, (100, 200), (300, 200), (150, 150, 150), 2)
    m = m2.ink_mask(before, after)
    assert m[198:203, 120:280].any(axis=0).mean() > 0.95
    assert m[:150].sum() == 0


def test_photos_pair_each_drawing(tmp_path: Path):
    ev = cat_timeline() + [
        {"t": 5.0, "kind": "photo", "file": "photos/5000.png"},
        {"t": 45.0, "kind": "photo", "file": "photos/45000.png"},
    ]
    ev = sorted(ev, key=lambda e: e["t"])
    (c,) = m2.commands(ev)
    ph = m2.Photos(tmp_path, ev, Box(0, 0, 192, 196))
    assert ph.before(c, None) == tmp_path / "photos/5000.png"
    assert ph.after(c, None) == tmp_path / "photos/45000.png"


def test_run_paper_from_header(tmp_path: Path):
    (tmp_path / "hands.jsonl").write_text(
        json.dumps({"t0": 1.0, "paper": [28, 24, 192, 196]}) + "\n"
    )
    assert m2.run_paper(tmp_path) == Box(28, 24, 192, 196)
    (tmp_path / "hands.jsonl").write_text(json.dumps({"t": 0.0, "hands": []}) + "\n")
    assert m2.run_paper(tmp_path) is None


def test_summary_median_latency():
    cs = []
    for lat in (1.0, 2.0, 4.0):
        (c,) = m2.commands(cat_timeline())
        c.preview_start_t = c.speech_end_t + lat
        cs.append(c)
    s = m2.summary(cs)
    assert s["median_latency_s"] == pytest.approx(2.0)
    assert s["latency_ok"] and s["passed"] == 0 and s["measured"] == 0
