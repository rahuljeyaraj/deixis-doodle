"""Scoring for the R1 Cosmos test: label accuracy, box IoU, point placement and latency.

All geometry here is in image pixels (x right, y down), because that is what both the synthetic
ground truth and rahul's hand-labelled photos use (data/r1/README.md).
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pointdraw.cosmos.parse import Detection, normalise_label

BoxPx = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(label: str) -> list[str]:
    return [_singular(w) for w in re.findall(r"[a-z]+", normalise_label(label))]


def label_matches(pred: str, truth: str, synonyms: Iterable[str] = ()) -> bool:
    """True if the predicted label names the truth or a synonym.

    Matches whole words after normalising, so 'a black cat' matches 'cat' and 'smiley face'
    matches 'face' only if 'face' is listed as a synonym. Plurals are folded.
    """
    p = _words(pred)
    if not p:
        return False
    ptxt = " ".join(p)
    for cand in [truth, *synonyms]:
        c = _words(cand)
        if not c:
            continue
        ctxt = " ".join(c)
        if ptxt == ctxt or re.search(rf"\b{re.escape(ctxt)}\b", ptxt):
            return True
    return False


def best_iou(dets: Iterable[Detection], truth: Sequence[float]) -> float:
    return max((iou(d.box_px, truth) for d in dets if d.box_px is not None), default=0.0)


def point_ok(
    point: tuple[float, float] | None,
    accept_box: Sequence[float],
    avoid_boxes: Iterable[Sequence[float]] = (),
    margin_px: float = 0.0,
) -> bool:
    """The point lies in the accepted region and not on (or within margin of) any mark."""
    if point is None:
        return False
    x, y = point
    if not (accept_box[0] <= x <= accept_box[2] and accept_box[1] <= y <= accept_box[3]):
        return False
    for b in avoid_boxes:
        if b[0] - margin_px <= x <= b[2] + margin_px and b[1] - margin_px <= y <= b[3] + margin_px:
            return False
    return True


@dataclass
class Trial:
    task: str  # "label", "point" or "find"
    image: str
    query: str
    correct: bool
    latency_s: float
    detail: dict = field(default_factory=dict)


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    p90 = xs[min(len(xs) - 1, max(0, round(0.9 * (len(xs) - 1))))]
    return {
        "n": len(xs),
        "median_s": round(statistics.median(xs), 3),
        "p90_s": round(p90, 3),
        "max_s": round(xs[-1], 3),
    }


def summarise(trials: Sequence[Trial]) -> dict[str, dict]:
    """Accuracy and latency per task."""
    out: dict[str, dict] = {}
    for task in ("label", "point", "find"):
        ts = [t for t in trials if t.task == task]
        if not ts:
            continue
        n_ok = sum(t.correct for t in ts)
        out[task] = {
            "n": len(ts),
            "correct": n_ok,
            "accuracy": round(n_ok / len(ts), 3),
            "latency": latency_stats([t.latency_s for t in ts]),
        }
    return out


def format_table(summary: dict[str, dict]) -> str:
    """A markdown table for docs/spikes/r1-cosmos.md."""
    names = {
        "label": "(a) label a pointed crop",
        "point": "(b) semantic place",
        "find": "(c) find a named mark",
    }
    rows = [
        "| Task | n | Accuracy | Median latency | p90 latency |",
        "|---|---|---|---|---|",
    ]
    for task, s in summary.items():
        lat = s["latency"]
        med, p90 = lat.get("median_s", "-"), lat.get("p90_s", "-")
        rows.append(
            f"| {names.get(task, task)} | {s['n']} | {s['correct']}/{s['n']} "
            f"({100 * s['accuracy']:.0f}%) | {med} s | {p90} s |"
        )
    return "\n".join(rows)
