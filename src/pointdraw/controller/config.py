"""Machine settings for the Ender 3 pen plotter, with conservative defaults.

Everything is in the bed frame (machine XY, mm; see `pointdraw.layout`). On a stock Ender 3
the origin is the front-left corner of the bed: Y = 0 has the bed pushed back, and Y max pushes
the bed toward the user. Feedrates are mm/min (G-code F), firmware limits are mm/s (M203) and
mm/s^2 (M201/M204).

Tomorrow's bring-up tunes these. Save the tuned values with `scripts/plot.py config --set ...`,
which writes a local override to `.cache/controller.json` (gitignored). Only the keys you set
need to be in that file; the rest come from the defaults here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deixis.geometry import Box
from pointdraw import layout

DEFAULT_CONFIG_PATH = Path(".cache") / "controller.json"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rect(_Model):
    """An axis-aligned rectangle in the bed frame, min corner plus size."""

    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    def box(self) -> Box:
        return Box(self.x, self.y, self.w, self.h)


class Feedrates(_Model):
    """Feedrates in mm/min."""

    slow: float = Field(1000, gt=0)  # drawing while a hand could be near
    normal: float = Field(1800, gt=0)  # drawing
    fast: float = Field(2500, gt=0)  # drawing
    travel: float = Field(3000, gt=0)  # pen-up moves
    preview: float = Field(2000, gt=0)  # pen-up preview trace, slow enough to follow by eye
    z: float = Field(300, gt=0)  # pen lifts; stock Ender 3 Marlin caps Z at 5 mm/s


class MotionLimits(_Model):
    """Firmware limits sent at startup (M201, M203, M204)."""

    max_xy_mm_s: float = Field(60, gt=0)  # M203 X Y
    max_z_mm_s: float = Field(5, gt=0)  # M203 Z
    accel_xy: float = Field(500, gt=0)  # M201 X Y and M204 P (drawing)
    accel_z: float = Field(100, gt=0)  # M201 Z
    accel_travel: float = Field(500, gt=0)  # M204 T


class MachineConfig(_Model):
    # Travel limits: what our G-code may ever address. Stock Ender 3 firmware allows a bit more
    # (235 x 235 x 250); we stay inside the 220 x 220 printable area and never need Z > 20.
    x_min: float = 0.0
    x_max: float = 220.0
    y_min: float = 0.0
    y_max: float = 220.0
    z_min: float = 0.0
    z_max: float = 20.0

    # The paper on the bed. Placeholder: A5 landscape centred in X, toward the front of the bed
    # (low Y), so all of it sits in front of the gantry at park. Measure it at bring-up.
    paper: Rect = Rect(x=5.0, y=10.0, w=layout.PAPER_WIDTH_MM, h=layout.PAPER_HEIGHT_MM)
    margin_mm: float = Field(8.0, ge=0)  # the drawable area is the paper minus this margin

    # Park: pen up, bed toward the user (Y max on a stock Ender 3), carriage to one side.
    park_x: float = 0.0
    park_y: float = 220.0

    pen_up_z: float = 3.0
    pen_down_z: float = 0.0

    feed: Feedrates = Feedrates()
    limits: MotionLimits = MotionLimits()

    # Streaming and pause behaviour.
    max_segment_mm: float = Field(10.0, gt=0)  # split every XY move into pieces this long
    sync_every_mm: float = Field(30.0, ge=0)  # insert M400 after this much motion (0 = never)

    # Job limits.
    max_pen_down_mm: float = Field(10_000.0, gt=0)
    max_pen_lifts: int = Field(300, gt=0)
    max_points: int = Field(20_000, gt=0)
    clamp_tolerance_mm: float = Field(0.05, ge=0, le=0.05)

    servo_enabled: bool = False  # allows M280 (a servo pen lift, later)

    # Serial link.
    baud: int = 115200
    boot_timeout_s: float = Field(10.0, gt=0)
    ack_timeout_s: float = Field(30.0, gt=0)
    window: int = Field(2, ge=1, le=4)  # lines sent but not yet acknowledged

    # Interlock: hands must have been clear this long before motion starts or resumes.
    hands_clear_hold_s: float = Field(0.5, ge=0)

    @property
    def travel_box(self) -> Box:
        return Box(self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min)

    @property
    def paper_box(self) -> Box:
        return self.paper.box()

    @property
    def drawable(self) -> Box:
        return self.paper_box.inset(self.margin_mm)

    @property
    def pen_threshold_z(self) -> float:
        """Z below which the pen counts as down."""
        return (self.pen_up_z + self.pen_down_z) / 2

    def feed_for(self, speed: str) -> float:
        return {"slow": self.feed.slow, "normal": self.feed.normal, "fast": self.feed.fast}[speed]

    @model_validator(mode="after")
    def _check(self) -> MachineConfig:
        if not (self.x_min < self.x_max and self.y_min < self.y_max and self.z_min < self.z_max):
            raise ValueError("travel limits must have min < max")
        d = self.drawable
        if d.w <= 0 or d.h <= 0:
            raise ValueError("margin leaves no drawable area")
        t = self.travel_box
        if not (t.contains((d.x, d.y)) and t.contains((d.x2, d.y2))):
            raise ValueError("drawable area must lie inside the travel limits")
        if not t.contains((self.park_x, self.park_y)):
            raise ValueError("park position must lie inside the travel limits")
        if not self.pen_up_z > self.pen_down_z:
            raise ValueError("pen_up_z must be above pen_down_z")
        for name in ("pen_up_z", "pen_down_z"):
            if not self.z_min <= getattr(self, name) <= self.z_max:
                raise ValueError(f"{name} must lie within z_min..z_max")
        xy_cap = self.limits.max_xy_mm_s * 60
        for name in ("slow", "normal", "fast", "travel", "preview"):
            if getattr(self.feed, name) > xy_cap:
                raise ValueError(f"feed.{name} exceeds limits.max_xy_mm_s")
        if self.feed.z > self.limits.max_z_mm_s * 60:
            raise ValueError("feed.z exceeds limits.max_z_mm_s")
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None = DEFAULT_CONFIG_PATH) -> MachineConfig:
    """Defaults, overlaid with the JSON override at `path` if it exists (None = defaults)."""
    base = MachineConfig().model_dump()
    if path is None or not Path(path).exists():
        return MachineConfig.model_validate(base)
    override = json.loads(Path(path).read_text(encoding="utf-8"))
    return MachineConfig.model_validate(_deep_merge(base, override))


def save_config(config: MachineConfig, path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    """Write the whole config as JSON (creating `.cache/` if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config.model_dump(), indent=2) + "\n", encoding="utf-8")
    return p


def with_changes(config: MachineConfig, changes: dict[str, Any]) -> MachineConfig:
    """A validated copy with dotted keys changed, e.g. {"feed.normal": 2000}."""
    nested: dict[str, Any] = {}
    for key, value in changes.items():
        node = nested
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return MachineConfig.model_validate(_deep_merge(config.model_dump(), nested))
