"""The virtual sheet, the camera's view of it, and the vision model (mock or live).

- `VirtualSheet`: the user's own marks (with their ground-truth labels and when they were
  drawn) and when the page was last replaced.
- `VirtualCamera`: a top-down image of the paper **as it is now**: the user's marks
  (`scene.synth`, hand-drawn look, several inks) plus every stroke the simulated plotter
  has laid. The runtime runs the real ink segmentation on it, subtracting its own strokes.
  It renders the already-warped paper image; the perspective camera, ArUco markers and the
  warp are covered by the vision tests (`vision/synthetic.py`), not repeated here.
- `MockVision`: Cosmos stand-in. Labels come from the scene file's ground truth.
- `LiveVision`: the real Cosmos endpoint (`CosmosClient`); needs COSMOS_BASE_URL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deixis.geometry import Point
from pointdraw import layout
from pointdraw.scene.ink import UserMark, crop_for_mark
from pointdraw.scene.model import SceneModel, iou
from pointdraw.scene.synth import INKS, SynthMark, make_mark, render_sheet
from pointdraw.virtual.plotter import VirtualPlotter
from pointdraw.virtual.scenefile import UserItem

PAPER_MM = (layout.PAPER_WIDTH_MM, layout.PAPER_HEIGHT_MM)


@dataclass
class TrueMark:
    t: float
    mark: SynthMark
    label: str


@dataclass
class VirtualSheet:
    seed: int = 0
    marks: list[TrueMark] = field(default_factory=list)
    cleared_at: float = -math.inf

    def add(self, item: UserItem, t: float) -> TrueMark:
        rng = np.random.default_rng(self.seed * 1000 + len(self.marks))
        ink = next((i for i in INKS if i[0] == item.ink), INKS[1])
        m = make_mark(item.shape, item.at, item.size, rng, ink=ink, rotate_deg=4.0)
        tm = TrueMark(t, m, item.label or item.shape)
        self.marks.append(tm)
        return tm

    def visible(self, t: float) -> list[TrueMark]:
        return [m for m in self.marks if self.cleared_at < m.t <= t]

    def clear(self, t: float) -> None:
        self.cleared_at = t


class VirtualCamera:
    def __init__(
        self,
        sheet: VirtualSheet,
        plotter: VirtualPlotter,
        origin: Point,
        mm_per_px: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.sheet = sheet
        self.plotter = plotter
        self.origin = origin
        self.mm_per_px = mm_per_px
        self.seed = seed
        self.shots = 0

    def robot_ink(self, t: float) -> list[list[Point]]:
        ox, oy = self.origin
        return [
            [(s.a[0] - ox, s.a[1] - oy), (s.b[0] - ox, s.b[1] - oy)]
            for s in self.plotter.ink
            if self.sheet.cleared_at < s.t <= t
        ]

    def paper_image(self, t: float, mm_per_px: float | None = None) -> tuple[np.ndarray, float]:
        mmpp = mm_per_px or self.mm_per_px
        self.shots += 1
        rng = np.random.default_rng(self.seed + 7919 * self.shots)
        marks = [m.mark for m in self.sheet.visible(t)]
        img = render_sheet(marks, PAPER_MM, mmpp, self.robot_ink(t), rng=rng).image
        return img, mmpp


class MockVision:
    """Cosmos stand-in: the ground-truth label of the mark that best overlaps."""

    def __init__(self, sheet: VirtualSheet, min_iou: float = 0.2) -> None:
        self.sheet = sheet
        self.min_iou = min_iou
        self.calls = 0

    def label(self, image: Any, mark: UserMark, mm_per_px: float) -> str | None:
        self.calls += 1
        best, best_iou = None, 0.0
        for tm in self.sheet.marks:
            if tm.t <= self.sheet.cleared_at:
                continue
            v = iou(tm.mark.box_mm, mark.box_mm)
            if v > best_iou:
                best, best_iou = tm, v
        return best.label if best is not None and best_iou >= self.min_iou else None

    def look(self, query: str, image: Any, mm_per_px: float, scene: SceneModel) -> str:
        self.calls += 1
        mine = [o.label for o in scene.marks if o.label]
        robot = [o.label for o in scene.robot if o.label and o.kind == "object"]
        q = query.lower()
        if any(w in q for w in ("i draw", "i drew", "did i", "my drawing", "i made")):
            if not mine:
                return "I don't see anything you drew."
            return "You drew " + _and([f"a {x}" for x in mine]) + "."
        parts = [f"a {x}" for x in mine] + [f"my {x}" for x in robot]
        return "I see " + _and(parts) + "." if parts else "The page is empty."


class LiveVision:
    """The real Cosmos endpoint. Crops are sent, never the whole camera frame (plan 4.5)."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            from pointdraw.cosmos.client import CosmosClient

            client = CosmosClient(think=False)
        self.client = client
        self.calls = 0
        self.latencies: list[float] = []
        self.replies: list[dict[str, Any]] = []  # raw replies, for the live-run report

    def label(self, image: Any, mark: UserMark, mm_per_px: float) -> str | None:
        crop = crop_for_mark(image, mark, mm_per_px)
        res = self.client.label_crop(crop.image)
        self.calls += 1
        self.latencies.append(res.reply.latency_s)
        self.replies.append({"task": "label", "label": res.label, "raw": res.reply.raw})
        return res.label or None

    def look(self, query: str, image: Any, mm_per_px: float, scene: SceneModel) -> str:
        from pointdraw.cosmos.parse import strip_reasoning
        from pointdraw.cosmos.prompts import look_prompt

        robot = [o.label for o in scene.robot if o.label and o.kind == "object"]
        reply, _, _ = self.client.ask(image, look_prompt(query, robot))
        self.calls += 1
        self.latencies.append(reply.latency_s)
        self.replies.append({"task": "look", "query": query, "raw": reply.raw})
        return strip_reasoning(reply.raw) or "Sorry, I couldn't make that out."


def _and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
