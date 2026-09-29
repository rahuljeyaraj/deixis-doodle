"""A fake transport with the same interface as the serial link.

It parses the G-code, tracks position and pen state, records pen-down strokes, can run in
(scaled) real time so pause and stop can be tested, and renders a PNG of the bed with the paper
outline and a millimetre grid.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path

from deixis.geometry import Point, Polyline
from pointdraw.clock import now
from pointdraw.controller.config import MachineConfig
from pointdraw.controller.safety import check_line, link_whitelist, parse_line
from pointdraw.controller.transport import LinkError, LinkStopped


class SimulatedPlotter:
    """A `Transport` that draws into memory instead of onto paper."""

    def __init__(
        self,
        config: MachineConfig | None = None,
        *,
        realtime: bool = False,
        speed_factor: float = 1.0,
        echo: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or MachineConfig()
        self.realtime = realtime
        self.speed_factor = speed_factor
        self.echo = echo
        self._allowed = link_whitelist(self.config.servo_enabled)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._stopped = False
        self.connected = False
        self.x = 0.0
        self.y = 0.0
        self.z = self.config.pen_up_z
        self.homed = False
        self.lines: list[str] = []
        self.strokes: list[Polyline] = []
        self.travel: list[tuple[Point, Point]] = []
        self.violations: list[str] = []
        self.quickstop_count = 0
        self.sim_time_s = 0.0
        self._stroke: Polyline | None = None

    # ------------------------------------------------------------------ Transport

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def send(self, line: str, timeout: float | None = None) -> None:
        self._run(line)

    def request(self, line: str, timeout: float | None = None) -> list[str]:
        return self._run(line)

    def wait_idle(self, timeout: float | None = None) -> None:
        return None  # moves run synchronously inside send()

    def quickstop(self) -> None:
        with self._lock:
            self._stopped = True
            self.quickstop_count += 1
            self._stop_evt.set()
            if self.echo:
                self.echo("M410")

    def resync(self, timeout: float | None = None) -> None:
        with self._lock:
            self._stopped = False
            self._stop_evt.clear()

    # ------------------------------------------------------------------ state

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def pen_is_down(self) -> bool:
        return self.z < self.config.pen_threshold_z

    @property
    def pen_down_mm(self) -> float:
        return sum(math.dist(a, b) for s in self.strokes for a, b in zip(s, s[1:], strict=False))

    # ------------------------------------------------------------------ execution

    def _run(self, line: str) -> list[str]:
        check_line(line, allowed=self._allowed)
        if not self.connected:
            raise LinkError("not connected")
        with self._lock:
            if self._stopped:
                raise LinkStopped("stopped; call resync() first")
            self.lines.append(line)
        if self.echo:
            self.echo(line)
        cmd, p = parse_line(line)
        if cmd in ("G0", "G1"):
            self._move(p)
        elif cmd == "G28":
            self._end_stroke()
            axes = [a for a in "XYZ" if a in p] or ["X", "Y", "Z"]
            if "X" in axes:
                self.x = 0.0
            if "Y" in axes:
                self.y = 0.0
            if "Z" in axes:
                self.z = 0.0
            self.homed = True
        elif cmd == "M114":
            return [f"X:{self.x:.2f} Y:{self.y:.2f} Z:{self.z:.2f} E:0.00 Count X:0 Y:0 Z:0"]
        elif cmd == "M115":
            return ["FIRMWARE_NAME:pointdraw simulator PROTOCOL_VERSION:1.0 EXTRUDER_COUNT:0"]
        elif cmd == "M105":
            return ["T:25.00 /0.00 B:25.00 /0.00 @:0 B@:0"]
        return []

    def _move(self, p: dict[str, float | None]) -> None:
        c = self.config
        feed = p.get("F") or 1000.0
        start = (self.x, self.y, self.z)
        target = [
            p["X"] if p.get("X") is not None else self.x,
            p["Y"] if p.get("Y") is not None else self.y,
            p["Z"] if p.get("Z") is not None else self.z,
        ]
        for i, (lo, hi) in enumerate(((c.x_min, c.x_max), (c.y_min, c.y_max), (c.z_min, c.z_max))):
            if not lo <= target[i] <= hi:
                self.violations.append(f"soft endstop: {'XYZ'[i]}{target[i]:g} clamped")
                target[i] = min(max(target[i], lo), hi)
        dist = math.dist(start, target)
        duration = dist / (feed / 60.0) if dist > 0 else 0.0
        frac = 1.0
        if self.realtime and duration > 0:
            wall = duration / self.speed_factor
            t0 = now()
            if self._stop_evt.wait(wall):
                frac = min(1.0, (now() - t0) / wall)
        self.sim_time_s += duration * frac
        end = tuple(s + (t - s) * frac for s, t in zip(start, target, strict=True))
        was_down = self.pen_is_down
        self.x, self.y, self.z = end
        now_down = self.pen_is_down
        if (start[0], start[1]) != (self.x, self.y) and not (was_down and now_down):
            self.travel.append(((start[0], start[1]), (self.x, self.y)))
        if now_down:
            if self._stroke is None or not was_down:
                self._stroke = [(start[0], start[1])] if was_down else [(self.x, self.y)]
                self.strokes.append(self._stroke)
            if (self.x, self.y) != self._stroke[-1]:
                self._stroke.append((self.x, self.y))
        else:
            self._end_stroke()

    def _end_stroke(self) -> None:
        self._stroke = None

    # ------------------------------------------------------------------ rendering

    def render_png(
        self, path: str | Path, *, px_per_mm: float = 4.0, show_travel: bool = True
    ) -> Path:
        """Draw the bed (travel limits), the paper, the drawable area, a mm grid and the strokes.

        The image shows the bed as the user sees it from the front: +x right, +y up (away).
        """
        from PIL import Image, ImageDraw

        c = self.config
        s = px_per_mm
        m = 40  # pixel margin for labels
        w_mm, h_mm = c.x_max - c.x_min, c.y_max - c.y_min
        img = Image.new("RGB", (int(w_mm * s) + 2 * m, int(h_mm * s) + 2 * m), "white")
        d = ImageDraw.Draw(img)

        def px(x: float, y: float) -> tuple[float, float]:
            return (m + (x - c.x_min) * s, m + (c.y_max - y) * s)

        for mm in range(0, int(max(w_mm, h_mm)) + 1, 10):
            major = mm % 50 == 0
            colour = (200, 200, 200) if major else (235, 235, 235)
            if mm <= w_mm:
                d.line([px(c.x_min + mm, c.y_min), px(c.x_min + mm, c.y_max)], fill=colour)
                if major:
                    d.text(
                        (px(c.x_min + mm, c.y_min)[0] - 6, m + h_mm * s + 6),
                        str(int(c.x_min + mm)),
                        fill="grey",
                    )
            if mm <= h_mm:
                d.line([px(c.x_min, c.y_min + mm), px(c.x_max, c.y_min + mm)], fill=colour)
                if major:
                    d.text(
                        (6, px(c.x_min, c.y_min + mm)[1] - 6), str(int(c.y_min + mm)), fill="grey"
                    )
        d.rectangle([px(c.x_min, c.y_max), px(c.x_max, c.y_min)], outline=(120, 120, 120))
        pb, db = c.paper_box, c.drawable
        d.rectangle([px(pb.x, pb.y2), px(pb.x2, pb.y)], outline=(40, 90, 200), width=2)
        d.rectangle([px(db.x, db.y2), px(db.x2, db.y)], outline=(230, 150, 40), width=1)
        if show_travel:
            for a, b in self.travel:
                d.line([px(*a), px(*b)], fill=(240, 150, 150), width=1)
        width = max(1, round(0.5 * s))
        for stroke in self.strokes:
            if len(stroke) == 1:
                x, y = px(*stroke[0])
                d.ellipse([x - width, y - width, x + width, y + width], fill="black")
            else:
                d.line([px(*p) for p in stroke], fill="black", width=width, joint="curve")
        hx, hy = px(self.x, self.y)
        d.ellipse([hx - 4, hy - 4, hx + 4, hy + 4], outline=(200, 0, 0), width=2)
        d.text(
            (m, 12),
            f"sim: {len(self.strokes)} strokes, {self.pen_down_mm:.0f} mm pen-down, "
            f"{self.sim_time_s:.1f} s   blue = paper, orange = drawable, red = pen-up travel",
            fill="black",
        )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        return out
