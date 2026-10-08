"""Finding a named thing in the room with Cosmos, so "clean around the chair" has a where.

The vacuum watches feet, not fingers, so a place it is told by name ("the chair", "the bin")
has nowhere to go: the core answers `unknown_place` and the app used to ask the person to
point at it, which this robot cannot see (live, 8 Oct 2026).

Cosmos names and finds; the geometry is still ours, which is the line the whole project
keeps. Only the **bottom edge** of its box is used, because that edge is where the thing
stands on the floor and so is the one part of it the floor fit can place honestly. Its
height is not measured and never guessed at: the footprint is as deep as it is wide, and
the margin round it is what makes the box worth cleaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from deixis.where import Where
from pointdraw.cosmos.client import CosmosClient
from pointdraw.cosmos.parse import Detection, parse_detections
from pointdraw.vacuum.floor import FloorView

log = logging.getLogger(__name__)

ROOM_FIND = (
    "This photo is from a camera high on the wall of a room, looking down over the floor. "
    "Locate the accurate bounding box of {query} as a whole, including its legs or base down "
    "to where it stands on the floor. If there are several, return each one. If there is "
    "none, return []. "
    'Return a json: [{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}}] with coordinates '
    "normalised to 0-1000. Answer directly, without explaining your reasoning."
)


def thing_name(name: str) -> str:
    """The thing in "around the chair": "the chair". What is looked for, and what is said."""
    q = name.strip().rstrip(".?!")
    for lead in ("around ", "round ", "near ", "by ", "beside ", "next to ", "under ", "at "):
        if q.lower().startswith(lead):
            q = q[len(lead) :]
            break
    if not q.lower().startswith(("the ", "a ", "an ", "every ", "all ", "my ")):
        q = "the " + q
    return q


def room_find_prompt(name: str) -> str:
    return ROOM_FIND.format(query=thing_name(name))


def biggest(dets: list[Detection]) -> Detection | None:
    """The largest box. Cosmos returns one thing per name here, but "the chairs" may give
    several and the largest is the one the camera sees best."""
    boxed = [d for d in dets if d.box_px is not None]
    if not boxed:
        return None
    return max(boxed, key=lambda d: (d.box_px[2] - d.box_px[0]) * (d.box_px[3] - d.box_px[1]))  # type: ignore[index]


def footprint_mm(
    view: FloorView, box_px: tuple[float, float, float, float], margin_mm: float
) -> tuple[float, float, float, float]:
    """The floor box to clean around a thing whose picture box is `box_px`.

    The bottom edge's two corners are the only part of the box on the floor, so they are the
    only part put through the floor fit: they give the width, and the thing is taken to be as
    deep as it is wide (a chair, a bin, a box: near enough, and the margin covers the rest).

    That edge is the thing's **near** side, though, so a square centred on it would sit in
    front of the thing rather than on it (8 Oct, the chair). The square is pushed half its
    side away from the camera, and "away" is read off the floor itself: a step up the picture
    from the bottom edge is a step away across the floor, whatever way the camera faces.
    """
    x1, y1, x2, y2 = box_px
    step = max(2.0, 0.03 * (y2 - y1))  # small, so it stays near the floor
    (ax, ay), (bx, by), (ux, uy) = view.to_mm(
        np.array([[x1, y2], [x2, y2], [(x1 + x2) / 2, y2 - step]], float)
    )
    mid = np.array([(ax + bx) / 2, (ay + by) / 2])
    side = float(np.hypot(bx - ax, by - ay))
    away = np.array([ux, uy]) - mid
    norm = float(np.hypot(*away))
    centre = mid + away / norm * (side / 2) if norm > 1e-6 else mid
    half = side / 2 + margin_mm
    return (float(centre[0] - half), float(centre[1] - half), 2 * half, 2 * half)


@dataclass
class Eyes:
    """Cosmos looking for a named thing in the floor camera's picture.

    `margin_mm` is how far out from the thing the robot cleans: a quarter turn of the N50
    (its 350 mm body) each side by default, which is what "around it" has to mean for a robot
    that cannot clean a ring.
    """

    client: CosmosClient
    margin_mm: float = 350.0
    frame: str = "floor"

    def look(self, name: str, image: np.ndarray, view: FloorView) -> Where | None:
        """A box on the floor around `name`, or None if Cosmos does not see it."""
        try:
            reply, w, h = self.client.ask(image, room_find_prompt(name), max_tokens=400)
            dets = parse_detections(
                reply.raw, w, h, self.client.profile.coord_mode, self.client.profile.xy_order
            )
        except Exception as e:  # noqa: BLE001 - a thinking endpoint, a timeout, a bad reply
            log.warning("cosmos could not look for %r: %s", name, e)
            return None
        best = biggest(dets)
        if best is None or best.box_px is None:
            log.info("cosmos did not see %r", name)
            return None
        box = footprint_mm(view, best.box_px, self.margin_mm)
        return Where(
            frame=self.frame,
            kind="box",
            box_mm=box,
            label=best.label or name,
            source="vlm",
            confidence=0.7,  # a name and a near edge, not a measurement
            evidence={
                "vlm": "cosmos",
                "asked": name,
                "box_px": [round(v) for v in best.box_px],
                "margin_mm": self.margin_mm,
            },  # fmt: skip
        )
