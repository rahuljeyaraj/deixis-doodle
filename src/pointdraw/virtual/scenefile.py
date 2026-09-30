"""Scene files (`tests/scenes/*.yaml`): what is said, what the hand does, what is on the
sheet, and what must happen. Format reference: `tests/scenes/README.md`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

SCENES_DIR = Path(__file__).resolve().parents[3] / "tests" / "scenes"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Line(_M):
    at: float  # when speech starts, s
    text: str
    words: list[list[Any]] | None = None  # [[word, t0], [word, t0, t1], ...] to set times
    rate: float = 1.0  # speaking speed factor


class RobotItem(_M):
    """Something the robot drew earlier (known exactly)."""

    object: str | None = None
    label: str | None = None
    at: tuple[float, float]
    size: float = 40.0
    id: str | None = None


class UserItem(_M):
    """A mark the user drew with their own pen (only the camera sees it)."""

    shape: str  # a shape from pointdraw.scene.synth.SHAPES
    at: tuple[float, float]
    size: float = 35.0
    ink: str = "black pen"
    label: str | None = None  # ground truth for the Cosmos mock (default: the shape)


class Sheet(_M):
    robot: list[RobotItem] = Field(default_factory=list)
    user: list[UserItem] = Field(default_factory=list)


class Event(_M):
    at: float
    user_mark: UserItem | None = None


class DrawnExpect(_M):
    kind: str = "object"  # object | label | arrow
    label: str | None = None  # category for objects
    text: str | None = None  # for labels
    center: tuple[float, float] | None = None
    tol_mm: float = 5.0
    size: float | None = None  # larger side of the ink box
    size_tol: float = 0.2  # relative
    below: str | None = None  # a label's object: its label, e.g. "brain"
    inside: str | None = None  # the drawing must lie inside the object/mark with this label
    between: tuple[str, str] | None = None  # centre lies between these two objects
    near: str | None = None  # centre within `near_mm` of this object's box
    near_mm: float = 25.0
    from_: str | None = Field(default=None, alias="from")  # arrows: end objects by label
    to: str | None = None
    empty: bool = False  # must not overlap any other object


class Expect(_M):
    calls: list[str] = Field(default_factory=list)  # tool names, in order (a subsequence)
    drawn: list[DrawnExpect] | None = None  # every drawing, in the order drawn
    asks: int | None = None  # exact number of questions asked
    ask_contains: list[str] = Field(default_factory=list)
    ask_options: list[str] = Field(default_factory=list)  # must be among the options
    states: list[str] = Field(default_factory=list)  # must be visited in this order
    look_contains: list[str] = Field(default_factory=list)
    say_contains: list[str] = Field(default_factory=list)
    stop: bool = False  # a stop must happen, within the limits
    pause: bool = False  # the interlock must pause a drawing, within the limits
    scene_empty: bool = False  # the scene is empty at the end (new sheet)
    min_pen_mm: float | None = None  # at least this much ink laid (e.g. stop mid-drawing)
    max_pen_mm: float | None = None


class Variations(_M):
    seeds: list[int] = Field(default_factory=lambda: [1, 2])
    hover_mm: float | None = 8.0  # a run with the finger this high above the paper
    shifts: list[float] = Field(default_factory=lambda: [-0.2, 0.2])  # speech vs gesture, s


class Plotter(_M):
    emergency_parser: bool = False


class SceneSpec(_M):
    id: str
    title: str
    tests: str = ""
    notes: str = ""
    voice: str = "text"  # the voice modes this scene is meant for: text, tts or both
    lines: list[Line] = Field(default_factory=list)
    hand: list[dict[str, Any]] = Field(default_factory=list)
    sheet: Sheet = Field(default_factory=Sheet)
    events: list[Event] = Field(default_factory=list)
    expect: Expect = Field(default_factory=Expect)
    variations: Variations = Field(default_factory=Variations)
    plotter: Plotter = Field(default_factory=Plotter)
    max_time: float = 90.0
    path: str | None = None


def load(path: str | Path) -> SceneSpec:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    spec = SceneSpec.model_validate(data)
    spec.path = str(p)
    return spec


def all_scenes(folder: str | Path = SCENES_DIR) -> list[SceneSpec]:
    return [load(p) for p in sorted(Path(folder).glob("S*.yaml"))]


def find(name: str, folder: str | Path = SCENES_DIR) -> SceneSpec:
    """A scene by id ("S1", "s01") or path."""
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load(p)
    want = name.upper().lstrip("S").lstrip("0")
    for spec in all_scenes(folder):
        if spec.id.upper().lstrip("S").lstrip("0") == want:
            return spec
    raise FileNotFoundError(f"no scene {name!r} in {folder}")
