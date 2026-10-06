"""'This big' is the diameter of a circle round 'here' (rahul, 6 Oct 2026).

The drawing's box is centred on the point and the drawing grows until its farthest line
touches the circle: a square's corners touch it, a circle fills it. Before drawing, the
plotter traces that circle in the air, so you see where and how big; a part off the page
runs along the edge there, and is cut from the drawing, never moved.
"""

import math
import re

import pytest

from deixis.geometry import Box
from deixis.types import Word
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.schema import parse_call
from pointdraw.controller import gcode
from pointdraw.controller.config import MachineConfig
from pointdraw.controller.shapes import circle_points
from pointdraw.drawing.place import bbox, circle_size, fit, reach
from pointdraw.runtime.guess import size_words
from pointdraw.runtime.planner import Planner
from pointdraw.runtime.runtime import Runtime
from pointdraw.virtual.plotter import VirtualPlotter
from pointdraw.vision.hands import HandsFrame

CFG = MachineConfig()
HERE = (100.0, 80.0)


@pytest.fixture(scope="module")
def planner() -> Planner:
    return Planner()


def test_a_square_has_its_corners_on_the_circle(planner):
    p = planner.object("square", HERE, 60)
    assert p.box.w == pytest.approx(60 / math.sqrt(2))
    assert reach(p.polylines, HERE) == pytest.approx(30)
    assert p.size_circle == (HERE, 60)


def test_a_circle_fills_the_circle(planner):
    p = planner.object("circle", HERE, 60)
    assert max(p.box.w, p.box.h) == pytest.approx(60, abs=0.1)
    assert reach(p.polylines, HERE) == pytest.approx(30)


def test_the_cross_is_unchanged(planner):
    """The M01 cross: arms 10 mm across, as before."""
    p = planner.object("cross", HERE, 10)
    assert (p.box.w, p.box.h) == (pytest.approx(10), pytest.approx(10))


@pytest.mark.parametrize("name", ["cat", "house", "tree", "fish"])
def test_a_library_drawing_touches_the_circle_from_inside(planner, name):
    p = planner.object(name, HERE, 50)
    assert p.box.center == pytest.approx(HERE)
    assert reach(p.polylines, HERE) == pytest.approx(25)
    assert circle_size(p.polylines) == pytest.approx(50)
    assert max(p.box.w, p.box.h) <= 50 + 1e-6


def test_fit_keeps_the_aspect():
    wide = [[(0, 0), (100, 0), (100, 40)]]
    b = bbox(fit(wide, (0, 0), 50, smooth_iterations=0))
    assert b.w / b.h == pytest.approx(100 / 40)
    assert math.hypot(b.w, b.h) == pytest.approx(50)  # corners on the circle


def test_a_cut_drawing_keeps_its_circle():
    """Part of it off the page: that part is left out, and the circle stays where it was
    asked for (the preview and a resize use it)."""
    area = Box(0, 0, 200, 150)
    p = Planner(drawable=area).object("circle", (190.0, 75.0), 40)
    cut, was_cut = Planner(drawable=area).clip_inside(p)
    assert was_cut and cut.box.x2 <= 200
    assert cut.size_circle == ((190.0, 75.0), 40)


def test_shift_moves_and_scales_the_circle(planner):
    p = planner.shift(planner.object("cat", HERE, 40), 5, -3, 1.5)
    (cx, cy), d = p.size_circle
    assert (cx, cy, d) == (pytest.approx(105), pytest.approx(77), pytest.approx(60))


# ---------------------------------------------------------------- the pen-up circle


def _xy(lines: list[str]) -> list[tuple[float, float]]:
    out = []
    for ln in lines:
        x, y = re.search(r"X(-?[\d.]+)", ln), re.search(r"Y(-?[\d.]+)", ln)
        if x and y:
            out.append((float(x.group(1)), float(y.group(1))))
    return out


def test_the_preview_traces_the_circle_with_the_pen_up():
    d = CFG.drawable
    c = d.center
    lines = gcode.preview_circle_gcode(c, 40, CFG, start=(c[0], c[1] - 100))
    zs = [float(z) for z in re.findall(r"Z(-?[\d.]+)", " ".join(lines))]
    assert zs and min(zs) >= CFG.pen_up_z  # never down
    pts = _xy(lines)  # the travel there, then the circle (long moves come in short pieces)
    i = next(k for k, p in enumerate(pts) if math.dist(p, c) == pytest.approx(20, abs=0.01))
    assert pts[i] == pytest.approx((c[0], c[1] - 20), abs=0.01)  # starts nearest the head
    assert all(math.dist(p, c) == pytest.approx(20, abs=0.25) for p in pts[i:])
    assert pts[-1] == pytest.approx(pts[i], abs=0.01)  # once round


def test_off_the_page_the_preview_runs_along_the_edge():
    d = CFG.drawable
    pts = circle_points((d.x2 - 5, d.center[1]), 40, d)
    assert all(d.x - 1e-9 <= x <= d.x2 + 1e-9 and d.y <= y <= d.y2 for x, y in pts)
    assert sum(1 for x, _ in pts if x == pytest.approx(d.x2)) > 5


# ---------------------------------------------------------------- the runtime


class Script:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def understand(self, words, context):
        return AgentResult(calls=self.answers.pop(0), source="cache")


class Recording(VirtualPlotter):
    def __init__(self, cfg: MachineConfig) -> None:
        super().__init__(cfg)
        self.circles: list[tuple] = []

    def preview_circle(self, center, size, tag=None):
        self.circles.append((center, size))
        super().preview_circle(center, size, tag)


def run(rt: Runtime, plotter: VirtualPlotter, t0: float, t1: float) -> None:
    t = t0
    while t < t1:
        rt.on_frame(HandsFrame(t, [], (1920, 1080)))
        plotter.tick(t)
        rt.tick(t)
        t += 0.05


def say(rt: Runtime, text: str, t: float) -> None:
    rt.on_utterance(
        [Word(w, t + 0.3 * i, t + 0.3 * i + 0.2) for i, w in enumerate(text.split())], t + 1
    )


def test_the_runtime_previews_the_circle_and_says_its_diameter():
    args = {"object": "square", "place": {"empty_space": True}, "size": {"preset": "big"}}
    call = parse_call("place_drawing", args)
    plotter = Recording(CFG)
    rt = Runtime(Script([call]), plotter, CFG, Planner())
    rt.start(0.0)
    say(rt, "draw a big square", 0.0)
    run(rt, plotter, 1.0, 3.0)
    item = rt.proposal.items[0]
    (cx, cy), d = item.size_circle
    assert plotter.circles == [(rt._to_bed((cx, cy)), d)]
    assert item.box.w == pytest.approx(d / math.sqrt(2), abs=0.01)  # corners on the circle
    words = {e["text"] for e in rt.log.of("say")}
    assert f"A square, {size_words(d)}?" in words
    assert rt.log.of("preview")[-1]["circles"] == [[round(cx, 2), round(cy, 2), round(d, 2)]]


def test_a_bit_bigger_keeps_here_and_grows_the_circle():
    call = parse_call("place_drawing", {"object": "cat", "place": {"empty_space": True}})
    plotter = Recording(CFG)
    rt = Runtime(Script([call]), plotter, CFG, Planner())
    rt.start(0.0)
    say(rt, "draw a cat", 0.0)
    run(rt, plotter, 1.0, 3.0)
    item = rt.proposal.items[0]
    (c0, d0) = item.size_circle
    rt.adjust(0, 0, 1.25, 3.0)
    # the first trace ends, then the new circle is traced; longer since the 80 mm default
    run(rt, plotter, 3.0, 25.0)
    c1, d1 = item.size_circle
    assert d1 == pytest.approx(d0 * 1.25)
    assert c1 == pytest.approx(c0, abs=0.01)
    assert circle_size(item.polylines) == pytest.approx(d1, abs=0.01)
    assert plotter.circles[-1] == (pytest.approx(rt._to_bed(c1)), pytest.approx(d1))
