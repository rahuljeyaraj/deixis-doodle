"""Quick, Draw! data: parsing, heuristic scoring and candidate ranking.

The simplified dataset stores each drawing as strokes of integer points in a 0..255 box,
y pointing down, already scaled so the larger dimension touches 255. Everything here works
on that raw form; `build_library.py` converts picks to the library's y-up form.

Dataset: Quick, Draw! by Google Creative Lab, CC BY 4.0
(https://github.com/googlecreativelab/quickdraw-dataset).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
)
SIMPLIFIED_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{name}.ndjson"

# Plan section 9.2 (hero demo) first, then scene vocabulary. Every name is an official
# category (fetch_quickdraw.py checks them against categories.txt).
HERO_CATEGORIES = [
    "ear", "eye", "brain", "mouth", "hand", "microphone", "camera", "laptop", "cloud",
]  # fmt: skip
SCENE_CATEGORIES = [
    "house", "tree", "sun", "flower", "bush", "pond", "fence", "bench", "cat", "dog", "bird",
    "fish", "star", "moon", "mountain", "car", "bicycle", "sailboat", "apple", "cup", "hat",
    "crown", "key", "light bulb", "smiley face", "table", "chair", "couch", "bed", "door",
    "floor lamp", "television",
]  # fmt: skip
DEFAULT_CATEGORIES = HERO_CATEGORIES + SCENE_CATEGORIES

RawStroke = list[tuple[int, int]]


@dataclass(frozen=True)
class RawDrawing:
    """One Quick, Draw! drawing in dataset units (0..255, y down)."""

    word: str
    key_id: str
    strokes: tuple[tuple[tuple[int, int], ...], ...]
    recognized: bool = True

    @classmethod
    def from_ndjson(cls, obj: dict) -> RawDrawing:
        strokes = tuple(
            tuple((int(x), int(y)) for x, y in zip(xs, ys, strict=False))
            for xs, ys in obj["drawing"]
        )
        return cls(
            word=obj["word"],
            key_id=str(obj["key_id"]),
            strokes=strokes,
            recognized=bool(obj.get("recognized", False)),
        )

    def to_ndjson(self) -> dict:
        return {
            "word": self.word,
            "key_id": self.key_id,
            "recognized": self.recognized,
            "drawing": [[[p[0] for p in s], [p[1] for p in s]] for s in self.strokes],
        }


@dataclass(frozen=True)
class ScoreParams:
    """Heuristics for "clean, plottable, recognisable". Units are dataset units (0..255)."""

    min_strokes: int = 2
    max_strokes: int = 12
    min_ink: float = 250.0  # total stroke length; less is a scribble or a single tick
    max_ink: float = 3500.0  # more usually means shading or hatching
    dot_length: float = 3.0  # strokes shorter than this are dots: rejected
    tiny_length: float = 12.0  # strokes shorter than this are penalised
    max_aspect: float = 4.0  # w/h or h/w beyond this is rejected


@dataclass(frozen=True)
class Features:
    n_strokes: int
    ink: float
    shortest: float
    n_tiny: int
    aspect: float  # w / h, both at least 1 unit
    n_points: int


@dataclass(frozen=True)
class Scored:
    drawing: RawDrawing
    features: Features
    score: float
    reasons: list[str] = field(default_factory=list)


def stroke_length(stroke: Iterable[tuple[float, float]]) -> float:
    pts = np.asarray(list(stroke), dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def features(d: RawDrawing, tiny_length: float = ScoreParams.tiny_length) -> Features:
    lengths = [stroke_length(s) for s in d.strokes] or [0.0]
    pts = np.array([p for s in d.strokes for p in s] or [(0, 0)], dtype=float)
    w = max(float(pts[:, 0].max() - pts[:, 0].min()), 1.0)
    h = max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0)
    return Features(
        n_strokes=len(d.strokes),
        ink=float(sum(lengths)),
        shortest=float(min(lengths)),
        n_tiny=sum(1 for ln in lengths if ln < tiny_length),
        aspect=w / h,
        n_points=int(len(pts)),
    )


def reject_reason(f: Features, p: ScoreParams) -> str | None:
    """Why a drawing fails the hard filters, or None if it passes."""
    if not p.min_strokes <= f.n_strokes <= p.max_strokes:
        return f"{f.n_strokes} strokes"
    if f.ink < p.min_ink:
        return "too little ink"
    if f.ink > p.max_ink:
        return "too much ink"
    if f.shortest < p.dot_length:
        return "has a dot"
    if max(f.aspect, 1 / f.aspect) > p.max_aspect:
        return "extreme aspect ratio"
    return None


def score(f: Features, typical: tuple[float, float]) -> float:
    """Higher is better. `typical` is the category's median (stroke count, log ink).

    Terms: closeness to the category's typical drawing (so the doodle reads as the object),
    a mild preference for fewer strokes (fewer pen lifts), and penalties for tiny strokes
    and very elongated shapes.
    """
    med_strokes, med_log_ink = typical
    s = 0.0
    s -= abs(math.log(f.ink) - med_log_ink) * 1.5
    s -= abs(f.n_strokes - med_strokes) / max(med_strokes, 1.0)
    s -= 0.03 * f.n_strokes
    s -= 0.25 * f.n_tiny
    s -= 0.3 * max(0.0, abs(math.log(f.aspect)) - math.log(2.0))
    return s


def rank(
    drawings: Iterable[RawDrawing], top: int = 30, params: ScoreParams | None = None
) -> list[Scored]:
    """Score recognised drawings and return the best `top`, best first.

    If fewer than `top` pass the stroke-count filter, it is relaxed to 1..15 so every
    category still gets candidates (a one-stroke moon is fine for a plotter).
    """
    params = params or ScoreParams()
    pool = [(d, features(d, params.tiny_length)) for d in drawings if d.recognized]
    passing = [(d, f) for d, f in pool if reject_reason(f, params) is None]
    if len(passing) < top:
        relaxed = ScoreParams(**{**params.__dict__, "min_strokes": 1, "max_strokes": 15})
        passing = [(d, f) for d, f in pool if reject_reason(f, relaxed) is None]
    if not passing:
        return []
    typical = (
        float(median(f.n_strokes for _, f in passing)),
        float(median(math.log(f.ink) for _, f in passing)),
    )
    scored = [Scored(d, f, score(f, typical)) for d, f in passing]
    scored.sort(key=lambda s: (-s.score, s.drawing.key_id))
    return scored[:top]


def iter_ndjson(path: Path) -> Iterator[RawDrawing]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield RawDrawing.from_ndjson(json.loads(line))


def cache_file(cache_dir: Path, category: str) -> Path:
    return cache_dir / f"{category}.ndjson"


def to_library_strokes(d: RawDrawing) -> list[list[list[int]]]:
    """Flip y (dataset y down -> paper y up) and move the bbox's min corner to (0, 0)."""
    pts = [p for s in d.strokes for p in s]
    x0 = min(p[0] for p in pts)
    y1 = max(p[1] for p in pts)
    return [[[x - x0, y1 - y] for x, y in s] for s in d.strokes]
