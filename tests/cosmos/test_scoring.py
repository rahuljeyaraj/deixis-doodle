"""R1 scoring and the evaluation loop, with fake Cosmos responses."""

import json

import cv2
import numpy as np
import pytest

from pointdraw.config import Settings
from pointdraw.cosmos.client import CosmosClient
from pointdraw.cosmos.r1 import crop_px, evaluate, load_dataset
from pointdraw.cosmos.scoring import (
    Trial,
    format_table,
    iou,
    label_matches,
    latency_stats,
    point_ok,
    summarise,
)
from pointdraw.scene.synth import make_sheet

from .fakes import FakeOpenAI


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


@pytest.mark.parametrize(
    "pred, truth, syn, ok",
    [
        ("cat", "cat", [], True),
        ("a black cat", "cat", [], True),
        ("cats", "cat", [], True),
        ("dog", "cat", ["kitty"], False),
        ("kitty", "cat", ["kitty"], True),
        ("smiley face", "smiley face", [], True),
        ("face", "smiley face", [], False),
        ("face", "smiley face", ["face"], True),
        ("", "cat", [], False),
        ("butterflies", "butterfly", [], True),
        ("category", "cat", [], False),
    ],
)
def test_label_matches(pred, truth, syn, ok):
    assert label_matches(pred, truth, syn) is ok


def test_point_ok():
    accept = (100, 0, 200, 100)
    assert point_ok((150, 50), accept)
    assert not point_ok((250, 50), accept)
    assert not point_ok(None, accept)
    assert not point_ok((150, 50), accept, avoid_boxes=[(140, 40, 160, 60)])
    assert not point_ok((165, 50), accept, avoid_boxes=[(140, 40, 160, 60)], margin_px=6)


def test_summary_and_table():
    trials = [
        Trial("label", "a", "cat", True, 1.0),
        Trial("label", "a", "dog", False, 2.0),
        Trial("find", "a", "cat", True, 1.2),
    ]
    s = summarise(trials)
    assert s["label"]["accuracy"] == 0.5 and s["label"]["latency"]["median_s"] == 1.5
    assert "point" not in s
    table = format_table(s)
    assert "(a) label a pointed crop | 2 | 1/2 (50%)" in table
    assert latency_stats([]) == {"n": 0}
    assert latency_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])["p90_s"] == 9


def test_crop_px_pads_with_white():
    img = np.zeros((100, 100, 3), np.uint8)
    c = crop_px(img, (0, 0, 20, 20), pad=2.0, min_side=10)
    assert c.shape == (40, 40, 3)
    assert (c[:10, :10] == 255).all() and (c[25:, 25:] == 0).all()


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(3)
    images = []
    for i in range(2):
        sheet = make_sheet(rng, n_marks=3, n_robot=1)
        name = f"s{i}.png"
        cv2.imwrite(str(tmp_path / name), sheet.image)
        images.append(sheet.ground_truth(name))
    (tmp_path / "labels.json").write_text(json.dumps({"version": 1, "images": images}))
    return tmp_path / "labels.json", images


def test_evaluate_with_a_perfect_fake(dataset):
    """A fake that answers from the ground truth scores 100% on every task."""
    labels_path, images = dataset
    cases = load_dataset(labels_path)
    state = {"case": None}

    def answer(kwargs):
        text = FakeOpenAI.prompt_text(kwargs)
        gt = state["case"]
        w, h = 840, 592
        if "close-up" in text:  # label_crop: the loop visits marks in order
            m = gt["marks"][state["label_i"]]
            state["label_i"] += 1
            return json.dumps({"label": m["label"], "alternatives": []})
        if "bounding box" in text:
            boxes = [
                {
                    "label": m["label"],
                    "bbox_2d": [
                        m["box_px"][0] / w * 1000,
                        m["box_px"][1] / h * 1000,
                        m["box_px"][2] / w * 1000,
                        m["box_px"][3] / h * 1000,
                    ],
                }
                for m in gt["marks"]
                if f"the {m['label']} as a whole" in text
            ]
            return json.dumps(boxes)
        q = next(q for q in gt["point_queries"] if q["query"].lower() in text.lower())
        b = q["accept_box_px"]
        # A spot in the accepted band that avoids every mark box.
        for fx in np.linspace(0.05, 0.95, 19):
            for fy in np.linspace(0.05, 0.95, 19):
                x, y = b[0] + fx * (b[2] - b[0]), b[1] + fy * (b[3] - b[1])
                if point_ok((x, y), b, [m["box_px"] for m in gt["marks"]]):
                    return json.dumps([{"point_2d": [x / w * 1000, y / h * 1000]}])
        return "[]"

    client = CosmosClient(Settings(_env_file=None), openai_client=FakeOpenAI(answer))
    trials = []
    for case, gt in zip(cases, images, strict=True):
        state.update(case=gt, label_i=0)
        trials += evaluate(client, [case])
    s = summarise(trials)
    assert s["label"]["accuracy"] == 1.0
    assert s["find"]["accuracy"] == 1.0
    assert s["point"]["n"] >= 2
    # Some accepted bands are fully covered by another mark; everything else must pass.
    assert s["point"]["correct"] >= s["point"]["n"] - 1


def test_evaluate_counts_errors_as_wrong(dataset):
    labels_path, _ = dataset

    class Broken:
        def label_crop(self, image):
            raise TimeoutError("endpoint asleep")

        find = point_to = lambda self, image, q: (_ for _ in ()).throw(TimeoutError("x"))

    seen = []
    trials = evaluate(Broken(), load_dataset(labels_path), limit=1, on_trial=seen.append)
    assert trials and not any(t.correct for t in trials)
    assert "TimeoutError" in trials[0].detail["error"]
    assert len(seen) == len(trials)
