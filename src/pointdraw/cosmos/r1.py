"""R1 harness: run the three Cosmos tasks over a labelled image set and score them.

Dataset format (the same for synthetic sheets and rahul's phone photos, see data/r1/README.md):

    {"version": 1,
     "images": [{"file": "img01.jpg",
                 "marks": [{"label": "cat", "synonyms": ["kitty"], "box_px": [x1, y1, x2, y2]}],
                 "point_queries": [{"query": "empty space to the right of the cat",
                                    "accept_box_px": [x1, y1, x2, y2]}]}]}

Pixels are in the image as stored (x right, y down). `file` is relative to the labels file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pointdraw.cosmos.scoring import Trial, best_iou, label_matches, point_ok

log = logging.getLogger("pointdraw.cosmos.r1")

TASKS = ("label", "point", "find")


@dataclass
class GtMark:
    label: str
    box_px: tuple[float, float, float, float]
    synonyms: list[str] = field(default_factory=list)


@dataclass
class GtQuery:
    query: str
    accept_box_px: tuple[float, float, float, float]


@dataclass
class Case:
    path: Path
    marks: list[GtMark]
    point_queries: list[GtQuery]


def load_dataset(labels_path: str | Path) -> list[Case]:
    labels_path = Path(labels_path)
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    cases = []
    for item in data.get("images", []):
        marks = [
            GtMark(m["label"], tuple(m["box_px"]), list(m.get("synonyms", [])))
            for m in item.get("marks", [])
        ]
        queries = [
            GtQuery(q["query"], tuple(q["accept_box_px"])) for q in item.get("point_queries", [])
        ]
        cases.append(Case(labels_path.parent / item["file"], marks, queries))
    return cases


def crop_px(
    image: np.ndarray,
    box_px: Sequence[float],
    pad: float = 1.4,
    min_side: int = 96,
    fill: int = 255,
) -> np.ndarray:
    """A square crop centred on a pixel box, `pad` times its larger side, padded with white."""
    x1, y1, x2, y2 = box_px
    side = int(round(max(min_side, pad * max(x2 - x1, y2 - y1))))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    h, w = image.shape[:2]
    out = np.full((side, side) + image.shape[2:], fill, image.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def _read(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def evaluate(
    client: Any,
    cases: Iterable[Case],
    tasks: Sequence[str] = TASKS,
    limit: int | None = None,
    on_trial: Callable[[Trial], None] | None = None,
    image_loader: Callable[[Path], np.ndarray] = _read,
) -> list[Trial]:
    """Run the chosen tasks. `client` needs label_crop, point_to and find (see client.py).

    - label: crop around each mark and ask what it is; correct if the label names the truth.
    - find: ask for "the <label>"; correct if any returned box has IoU >= 0.5 with a mark of
      that label.
    - point: ask each semantic-place query; correct if the first point is inside the accepted
      region and not on any mark's box.
    Errors (timeouts, bad replies) count as wrong and are recorded in `detail`.
    """
    trials: list[Trial] = []

    def add(t: Trial) -> None:
        trials.append(t)
        if on_trial:
            on_trial(t)

    for n, case in enumerate(cases):
        if limit is not None and n >= limit:
            break
        image = image_loader(case.path)
        name = case.path.name
        if "label" in tasks:
            for m in case.marks:
                try:
                    r = client.label_crop(crop_px(image, m.box_px))
                    ok = label_matches(r.label, m.label, m.synonyms)
                    alt_ok = any(label_matches(a, m.label, m.synonyms) for a in r.alternatives)
                    add(
                        Trial(
                            "label",
                            name,
                            m.label,
                            ok,
                            r.reply.latency_s,
                            {"pred": r.label, "alternatives": r.alternatives, "alt_ok": alt_ok},
                        )
                    )
                except Exception as e:  # noqa: BLE001 - any failure is a wrong answer
                    add(Trial("label", name, m.label, False, 0.0, {"error": repr(e)}))
        if "find" in tasks:
            for label in dict.fromkeys(m.label for m in case.marks):
                truths = [m.box_px for m in case.marks if m.label == label]
                try:
                    r = client.find(image, label)
                    score = max(best_iou(r.detections, t) for t in truths)
                    add(
                        Trial(
                            "find",
                            name,
                            label,
                            score >= 0.5,
                            r.reply.latency_s,
                            {"iou": round(score, 3), "n_boxes": len(r.detections)},
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    add(Trial("find", name, label, False, 0.0, {"error": repr(e)}))
        if "point" in tasks:
            avoid = [m.box_px for m in case.marks]
            for q in case.point_queries:
                try:
                    r = client.point_to(image, q.query)
                    pt = r.detections[0].center_px if r.detections else None
                    add(
                        Trial(
                            "point",
                            name,
                            q.query,
                            point_ok(pt, q.accept_box_px, avoid),
                            r.reply.latency_s,
                            {"point": [round(v, 1) for v in pt] if pt else None},
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    add(Trial("point", name, q.query, False, 0.0, {"error": repr(e)}))
    return trials
