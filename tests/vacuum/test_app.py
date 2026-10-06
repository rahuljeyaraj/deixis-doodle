"""The vacuum app end to end, without hardware: a pretend person walks and talks, the rule
agent reads the words, the deixis core finds the box (in-process, and over MCP once), and
a recording link stands in for the robot (the real `deixis-vacuum-mcp --fake` once)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from deixis.mcp import DeixisClient
from pointdraw.vacuum.agent import RuleAgent
from pointdraw.vacuum.app import CLEANING, LISTEN, PREVIEW, WAIT_CLEAR, Spoken, VacuumApp
from pointdraw.vacuum.calibrate import FloorToMap
from pointdraw.vacuum.pretend import Person, play
from pointdraw.vacuum.profile import vacuum_profile

PROFILE = vacuum_profile((0.0, 0.0, 3000.0, 2500.0))
TURN = math.radians(30)


def to_map(p):
    """The robot's map: turned 30 degrees, mirrored and shifted from the floor."""
    x, y = p
    c, s = math.cos(TURN), math.sin(TURN)
    return (1850 + c * x + s * y, -950 + s * x - c * y)


class Link:
    """Stands in for the robot's MCP server; records what it was asked."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, Any]] = []

    def clean_area(self, where):
        self.calls.append(("clean_area", where))
        if not self.ok:
            return {"ok": False, "why": "the box's width is 100 mm"}
        x, y, w, h = where.box_mm
        return {"ok": True, "zone": [round(x), round(y), round(x + w), round(y + h)]}

    def stop(self):
        self.calls.append(("stop", None))
        return {"ok": True}

    def dock(self):
        self.calls.append(("dock", None))
        return {"ok": True}


def run(person, link=None, f2m="fit", core=None, app_kw=None):
    core = core or DeixisClient(PROFILE, mode="direct")
    if f2m == "fit":
        floor = [(200, 150), (2600, 300), (2400, 2200), (500, 2000)]
        f2m = FloorToMap.fit([(p, to_map(p)) for p in floor], core)
    spoken = Spoken()
    app = VacuumApp(RuleAgent(PROFILE), core, link, f2m, PROFILE, say=spoken, **(app_kw or {}))
    return app, spoken, play(app, person)


def sent_box(link):
    (name, where), *_ = [c for c in link.calls if c[0] == "clean_area"]
    return where


def test_clean_here_yes_and_step_out():
    spot = (1500.0, 1000.0)
    p = Person((0, 0)).walk(spot).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(1.5).walk((2900, 2400)).stand(2.0)
    link = Link()
    app, spoken, states = run(p, link)
    assert spoken.lines[:2] == ["Clean about one metre square here?",
                                "OK. Step out, and I'll clean it."]  # fmt: skip
    assert spoken.lines[-1] == "Cleaning there now." and app.state == CLEANING
    assert PREVIEW in states and WAIT_CLEAR in states
    w = sent_box(link)
    assert w.frame == "map" and w.kind == "box"
    assert math.dist(w.mean_mm, to_map(spot)) < 40  # the 1 m box around the spot, on the map
    floor_box = app.log.of("clean_area")[0]["floor_box"]
    assert floor_box[2:] == [1000, 1000]
    # Sent only once the person was out of the box (plus its margin) for a second.
    t_sent = app.log.of("clean_area")[0]["t"]
    out = [s for s in p.samples if s.t <= t_sent and s.point is not None]
    assert not (1000 - 150 <= out[-1].point[0] <= 2000 + 150 and
                500 - 150 <= out[-1].point[1] <= 1500 + 150)  # fmt: skip


def test_clean_from_here_to_here():
    a, b = (2200.0, 1600.0), (900.0, 700.0)
    p = Person((3000, 2400)).walk(a).stand(0.5).say("clean from here").stand(1.2)
    p.walk(b).stand(0.4).say("to here").stand(1.5).say("yes").stand(1.5).walk((0, 0)).stand(2)
    link = Link()
    app, spoken, _ = run(p, link)
    assert spoken.lines[:2] == ["Clean from there to here?", "OK. Step out, and I'll clean it."]
    floor_box = app.log.of("clean_area")[0]["floor_box"]
    assert floor_box == pytest.approx([900, 700, 1300, 900], abs=40)


def narrow_strip(answer):
    a, b = (2200.0, 1300.0), (900.0, 1000.0)  # 1.3 m long, 0.3 m across
    p = Person((3000, 2400)).walk(a).stand(0.5).say("clean from here").stand(1.2)
    p.walk(b).stand(0.4).say("to here").stand(1.5).say(answer).stand(1.5)
    p.walk((0, 0)).stand(2)
    link = Link()
    app, spoken, _ = run(p, link)
    return app, spoken, link


def test_a_narrow_strip_asks_and_yes_widens_it():
    app, spoken, link = narrow_strip("yes")
    assert spoken.lines[0] == "That's a narrow strip. Clean half a metre wide?"
    assert spoken.lines[1] == "OK. Step out, and I'll clean it."
    x, y, w, h = app.log.of("clean_area")[0]["floor_box"]
    assert w == pytest.approx(1300, abs=40) and h == pytest.approx(500, abs=1)
    assert y + h / 2 == pytest.approx(1150, abs=30)  # widened about the strip's middle
    assert app.state == CLEANING


def test_no_reminder_while_already_stepping_out():
    a, b = (2200.0, 1600.0), (900.0, 700.0)
    p = Person((3000, 2400)).walk(a).stand(0.5).say("clean from here").stand(1.2)
    p.walk(b).stand(0.4).say("to here").stand(1.5).say("yes").stand(0.5).walk((0, 0)).stand(2)
    app, spoken, _ = run(p, Link())
    assert spoken.lines == ["Clean from there to here?", "Cleaning there now."]


def test_a_narrow_strip_no_cancels():
    app, spoken, link = narrow_strip("no")
    assert spoken.lines[-1] == "OK, cancelled." and link.calls == []


def test_walking_through_asks_then_here_answers():
    spot = (1200.0, 800.0)
    p = Person((0, 0)).walk((1500, 1200), speed=800).say("clean here")
    p.walk((3000, 2400), speed=800)  # still walking while saying it
    p.walk(spot).stand(0.5).say("here").stand(2.0).say("yes").stand(0.5).walk((0, 0)).stand(2)
    link = Link()
    app, spoken, _ = run(p, link)
    assert spoken.lines[0] == "Where? Stand on the spot and say here."
    assert spoken.lines[1] == "Clean about one metre square here?"
    assert math.dist(sent_box(link).mean_mm, to_map(spot)) < 40


def test_a_bit_left_moves_the_box_one_step():
    spot = (1500.0, 1000.0)
    p = Person((0, 0)).walk(spot).stand(0.6).say("clean here").stand(2.0)
    p.say("a bit left").stand(1.5).say("yes").stand(0.5).walk((2900, 2400)).stand(2)
    link = Link()
    app, spoken, _ = run(p, link)
    assert spoken.lines[:2] == ["Clean about one metre square here?"] * 2
    first, moved = app.log.of("preview")
    assert moved["box"][0] == first["box"][0] - 250 and moved["box"][1] == first["box"][1]
    assert math.dist(sent_box(link).mean_mm, to_map((spot[0] - 250, spot[1]))) < 40


def test_no_cancels_and_nothing_is_sent():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("no").stand(1.0).walk((0, 0)).stand(2)
    link = Link()
    app, spoken, _ = run(p, link)
    assert spoken.lines[-1] == "OK, cancelled." and link.calls == [] and app.state == LISTEN


def test_stop_and_go_home():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").walk((2900, 2400)).stand(2.0).say("stop").stand(1.0)
    p.say("go back to the dock").stand(2.5)
    link = Link()
    app, spoken, _ = run(p, link)
    assert [c[0] for c in link.calls] == ["clean_area", "stop", "dock"]
    assert spoken.lines[-2:] == ["Stopped.", "Going back to the dock."]


def test_waits_while_the_person_stays_in_the_box():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(10.0)
    link = Link()
    app, spoken, _ = run(p, link)
    assert link.calls == [] and app.state == WAIT_CLEAR
    assert spoken.lines.count("Please step out of the box so I can start.") == 1  # after 8 s


def test_no_calibration_no_robot_order():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(0.3).walk((2900, 2400)).stand(2)
    link = Link()
    app, spoken, _ = run(p, link, f2m=None)
    assert link.calls == []
    assert spoken.lines[-1] == ("I don't know where the robot's map is yet. "
                                "Please run the calibration.")  # fmt: skip


def test_the_robot_says_no():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(0.3).walk((2900, 2400)).stand(2)
    app, spoken, _ = run(p, Link(ok=False))
    assert spoken.lines[-1] == "The robot said no: the box's width is 100 mm."
    assert app.state == LISTEN


def test_a_dry_run_sends_nothing():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(0.3).walk((2900, 2400)).stand(2)
    app, spoken, _ = run(p, None)
    assert spoken.lines[-1] == "Dry run: the robot would clean there now."
    assert app.log.of("clean_area")[0]["map_box"]


def test_a_dry_run_before_calibration_still_shows_the_box():
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(0.3).walk((2900, 2400)).stand(2)
    app, spoken, _ = run(p, None, f2m=None)
    assert spoken.lines[-1] == "Dry run: the robot would clean there now."
    sent = app.log.of("clean_area")[0]
    assert sent["floor_box"][2:] == [1000, 1000] and sent["map_box"] is None


def test_over_mcp_to_the_pretend_robot():
    """The whole chain over MCP: the core's server, and the vacuum's server (--fake)."""
    from pointdraw.vacuum.link import LinkError, VacuumLink, default_command

    try:
        link = VacuumLink(default_command(fake=True), start_timeout_s=30)
    except (LinkError, FileNotFoundError, OSError) as e:
        pytest.skip(f"deixis-vacuum-mcp is not installed here: {e}")
    core = DeixisClient(PROFILE, mode="mcp", fallback=None)
    try:
        spot = (1500.0, 1000.0)
        p = Person((0, 0)).walk(spot).stand(0.6).say("clean here").stand(2.0)
        p.say("yes").stand(0.3).walk((2900, 2400)).stand(2)
        app, spoken, _ = run(p, link, core=core)
        assert spoken.lines[-1] == "Cleaning there now." and core.over_mcp
        reply = app.log.of("clean_area")[0]["reply"]
        assert reply["ok"] and len(reply["zone"]) == 4
        st = link.status()
        assert st["fake"] and st["state"] in ("cleaning", "idle")
    finally:
        core.close()
        link.close()


def test_the_window_shows_the_guess_dashed_blue_and_you_amber():
    """Claim 39q: the same look as the plotter's window."""
    from pointdraw.live import look
    from pointdraw.vacuum.feet import FeetSeen
    from pointdraw.vacuum.floor import FloorLayout, FloorView
    from pointdraw.vacuum.view import draw

    spot = (1500.0, 1000.0)
    p = Person((0, 0)).walk(spot).stand(0.6).say("clean here").stand(2.0)
    app, _, _ = run(p, Link())
    assert app.state == PREVIEW
    view = FloorView(np.diag([4.0, 4.0, 1.0]), (10, 11, 12, 13), True)  # 4 mm a pixel
    img = np.full((720, 1000, 3), 120, np.uint8)
    feet = FeetSeen(0.0, (np.array([370.0, 250.0]),), np.array([375.0, 250.0]), spot, True)
    vis = draw(img, view, FloorLayout.rectangle(3000, 2500), app, feet, 0.0)
    colours = {tuple(c) for c in vis.reshape(-1, 3)[::3]}
    assert look.GUESS in colours and look.AMBER in colours
