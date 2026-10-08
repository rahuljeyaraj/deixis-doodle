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
    vis = draw(img, view, FloorLayout.rectangle(3000, 2500).area_mm, app, feet, 0.0)
    colours = {tuple(c) for c in vis.reshape(-1, 3)[::3]}
    assert look.GUESS in colours and look.AMBER in colours


class NamedPlaceAgent:
    """Stands in for Nemotron, the only agent that returns a place by name. "Clean ..." gives
    the name; everything else ("yes") goes to the rule agent as usual."""

    def __init__(self, name: str = "around the chair") -> None:
        self.name, self.rules = name, RuleAgent(PROFILE)

    def understand(self, words, ctx):
        from deixis.actions import parse_call
        from pointdraw.agent.nemotron import AgentResult

        if not any(w.lower().startswith("clean") for w in words):
            return self.rules.understand(words, ctx)
        call = parse_call(PROFILE, "clean", {"place": {"named": self.name}})
        return AgentResult(calls=[call], source="nemotron")


def found_box(box=(1000.0, 800.0, 1500.0, 1500.0)):
    """What the eyes hand back: a floor box round the thing, as `objects.Eyes` builds one."""
    from deixis.where import Where

    return Where(frame="floor", kind="box", box_mm=box, label="chair", source="vlm",
                 confidence=0.7, evidence={"vlm": "cosmos"})  # fmt: skip


def test_a_thing_found_by_name_is_previewed_and_sent():
    """rahul, 8 Oct: "clean around the chair". The eyes find it, the person still says yes."""
    p = Person((2800, 2300)).stand(1.0).say("clean around the chair").stand(2.0)
    p.say("yes").stand(3.0)
    link = Link()
    asked: list[str] = []

    def eyes(name):
        asked.append(name)
        return found_box()

    app = VacuumApp(NamedPlaceAgent(), DeixisClient(PROFILE, mode="direct"), link,
                    FloorToMap.fit([(q, to_map(q)) for q in
                                    [(200, 150), (2600, 300), (2400, 2200), (500, 2000)]],
                                   DeixisClient(PROFILE, mode="direct")),
                    PROFILE, say=(spoken := Spoken()), eyes=eyes)  # fmt: skip
    play(app, p)
    assert asked == ["around the chair"]  # the eyes are given it whole
    assert spoken.lines[:2] == ["Let me look for the chair.", "Clean around the chair?"]
    assert app.log.of("found")[0]["label"] == "chair"
    w = sent_box(link)
    assert w.frame == "map" and w.kind == "box"


def test_eyes_that_see_nothing_fall_back_to_asking():
    p = Person((2800, 2300)).stand(1.0).say("clean around the chair").stand(2.0)
    app = VacuumApp(NamedPlaceAgent(), DeixisClient(PROFILE, mode="direct"), Link(), None,
                    PROFILE, say=(spoken := Spoken()), eyes=lambda name: None)  # fmt: skip
    play(app, p)
    assert spoken.lines == [
        "Let me look for the chair.",  # not "look for around the chair"
        "I don't know where 'around the chair' is. Stand next to it and say here.",
    ]
    assert app.log.of("look_failed")[0]["name"] == "around the chair"


def test_with_no_eyes_at_all_it_just_asks():
    p = Person((2800, 2300)).stand(1.0).say("clean around the chair").stand(2.0)
    app = VacuumApp(NamedPlaceAgent(), DeixisClient(PROFILE, mode="direct"), Link(), None,
                    PROFILE, say=(spoken := Spoken()))  # fmt: skip
    play(app, p)
    assert spoken.lines == [
        "I don't know where 'around the chair' is. Stand next to it and say here."
    ]
    assert not app.log.of("look")


def test_a_place_it_does_not_know_says_what_does_work():
    """Live, 8 Oct: "clean around the chair" was answered "Please point at it", which this
    robot cannot see -- it watches feet, not fingers. It now says what does work."""
    said = PROFILE.ask_text("unknown_place", name="around the chair")
    assert said == "I don't know where 'around the chair' is. Stand next to it and say here."
    assert "point" not in said


# ------------------------------------------------------- a fresh take without restarting


def previewing():
    """A person who has asked for a clean and is waiting on the question."""
    p = Person((0, 0)).walk((1500.0, 1000.0)).stand(0.6).say("clean here").stand(2.0)
    return run(p, Link())


def test_the_chat_thread_has_both_sides_newest_last():
    app, _, _ = previewing()
    assert app.chat() == [
        ("you", "clean here"),
        ("it", "Clean about one metre square here?"),
    ]


def test_c_clears_the_chat_but_not_the_log_or_the_guess():
    app, _, _ = previewing()
    app.clear_chat(99.0)
    assert app.chat() == [] and app.state == PREVIEW and app.preview is not None
    assert len(app.log.of("utterance", "say")) == 2  # the log keeps every line
    app.say(100.0, "Still talking.")
    assert app.chat() == [("it", "Still talking.")]


def test_r_resets_to_listening_with_nothing_showing():
    app, _, _ = previewing()
    app.reset(99.0)
    assert app.state == LISTEN
    assert app.preview is None and app.question is None and app.chat() == []
    assert app.last_floor_box is None and not app.sent
    assert app.log.of("reset")[0]["was"] == PREVIEW


def test_a_reset_mid_clean_leaves_the_robot_cleaning_and_stop_still_reaches_it():
    """rahul, 8 Oct 2026: R is for a fresh take, not for the robot."""
    spot = (1500.0, 1000.0)
    p = Person((0, 0)).walk(spot).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(1.5).walk((2900, 2400)).stand(2.0)
    link = Link()
    app, _, _ = run(p, link)
    assert app.state == CLEANING
    app.reset(99.0)
    assert app.state == LISTEN and app.log.of("reset")[0]["robot_left_cleaning"] is True
    assert [c[0] for c in link.calls] == ["clean_area"]  # nothing was sent on the reset
    app.stop(100.0)
    assert [c[0] for c in link.calls] == ["clean_area", "stop"]


def test_a_reset_then_a_new_clean_works():
    spot = (1500.0, 1000.0)
    p = Person((0, 0)).walk(spot).stand(0.6).say("clean here").stand(2.0)
    app, _, _ = run(p, Link())
    app.reset(p.t)
    again = Person(spot, t=p.t + 0.5).stand(0.6).say("clean here").stand(2.0)
    states = play(app, again)
    assert PREVIEW in states and app.chat()[0] == ("you", "clean here")


# ------------------------------------------------------- the path the robot drove


def idle_app(link=None):
    core = DeixisClient(PROFILE, mode="direct")
    floor = [(200, 150), (2600, 300), (2400, 2200), (500, 2000)]
    f2m = FloorToMap.fit([(q, to_map(q)) for q in floor], core)
    return VacuumApp(RuleAgent(PROFILE), core, link, f2m, PROFILE, say=Spoken())


def test_the_path_is_the_marker_where_the_camera_saw_it():
    """Pixels, straight from the camera, as the calibration drive tracks it: no floor fit and
    no lid height in between (rahul, 8 Oct)."""
    app = idle_app()
    for px in [(100.0, 100.0), (200.0, 100.0), (200.0, 300.0)]:
        app.on_robot_seen(0.0, np.array(px))
    assert list(app.trail) == [(100.0, 100.0), (200.0, 100.0), (200.0, 300.0)]


def test_a_robot_sitting_still_does_not_pile_up_points():
    app = idle_app()
    for _ in range(50):
        app.on_robot_seen(0.0, np.array([400.0, 400.0]))
    app.on_robot_seen(0.0, np.array([401.0, 400.0]))  # under the step: still one point
    assert len(app.trail) == 1


def test_the_path_is_dropped_on_a_reset():
    app = idle_app()
    app.on_robot_seen(0.0, np.array([100.0, 100.0]))
    app.reset(1.0)
    assert not app.trail


def test_the_window_draws_the_path_in_green():
    from pointdraw.vacuum.feet import FeetSeen
    from pointdraw.vacuum.floor import FloorLayout, FloorView
    from pointdraw.vacuum.view import draw

    app = idle_app()
    for px in [(100.0, 100.0), (400.0, 100.0), (400.0, 500.0)]:
        app.on_robot_seen(0.0, np.array(px))
    view = FloorView(np.diag([4.0, 4.0, 1.0]), (10, 11, 12, 13), True)
    img = np.full((720, 1000, 3), 120, np.uint8)
    feet = FeetSeen(0.0, (), None, None, False)
    vis = draw(img, view, FloorLayout.rectangle(3000, 2500).area_mm, app, feet, 0.0)
    b, g, r = (vis[..., i].astype(int) for i in range(3))
    assert ((g > r + 30) & (g > b + 30)).sum() > 100  # a green line is there


def test_the_path_is_drawn_half_way_through():
    """rahul, 8 Oct: 50 % opacity. The line must move the picture toward green, not replace
    it, so what is under the path is still visible."""
    from pointdraw.vacuum.view import GREEN, trail

    flat = np.full((200, 200, 3), 100, np.uint8)
    vis = flat.copy()
    trail(vis, [(20.0, 100.0), (180.0, 100.0)], alpha=0.5)
    on_line = vis[100, 100]
    assert tuple(on_line) != GREEN  # not the solid colour
    assert all(abs(int(on_line[i]) - (100 + GREEN[i]) // 2) <= 2 for i in range(3))


# ------------------------------------------------------- talk that is not for the robot


def heard(lines):
    """Say each line at an idle app; returns the app and what it said back."""
    p = Person((1500.0, 1000.0)).stand(0.6)
    for line in lines:
        p.say(line).stand(3.0)  # long lines take a while to be said and then heard
    app, spoken, _ = run(p, Link())
    return app, spoken


@pytest.mark.parametrize(
    "chatter",
    [
        "Yeah",  # all of these were heard on 8 Oct while the app sat idle
        "I think it s a good idea",
        "Same as that",
        "What do we should be selling",
        "Sounds very",
        "Hello",
        "One for the money",
        "Mm hmm",
    ],
)
def test_talk_that_is_not_for_the_robot_gets_no_answer(chatter: str) -> None:
    app, spoken = heard([chatter])
    assert spoken.lines == []
    assert app.log.of("not_for_me")[0]["text"] == chatter


@pytest.mark.parametrize(
    "said, stops",
    [
        ("Scrap", False),  # all three stopped the real robot on 8 Oct
        ("Stop up", True),
        ("Stop tripping", True),
    ],
)
def test_a_word_with_stop_in_it_still_stops_on_purpose(said: str, stops: bool) -> None:
    """A word that only sounds like a command is now ignored. A word that really contains
    "stop" is not: stopping too often costs nothing, and not stopping could cost a lot."""
    p = Person((1500.0, 1000.0)).stand(0.6).say(said).stand(1.5)
    link = Link()
    run(p, link)
    assert [c[0] for c in link.calls if c[0] == "stop"] == (["stop"] if stops else [])


def test_stop_always_gets_through():
    p = Person((1500.0, 1000.0)).stand(0.6).say("stop").stand(1.5)
    link = Link()
    app, spoken, _ = run(p, link)
    assert spoken.lines == ["Stopped."] and ("stop", None) in [(c[0], None) for c in link.calls]


@pytest.mark.parametrize(
    "said",
    [
        "playing from here",  # all of these are "clean from here" as Parakeet wrote it
        "Clicking from here",
        "Cling here",
        "Claim from here",
    ],
)
def test_a_pointing_word_is_enough_to_be_spoken_to(said: str) -> None:
    """The verb comes out as anything; "here" is what says it is for the robot (8 Oct)."""
    app, _ = heard([said])
    assert not app.log.of("not_for_me")


def test_an_answer_is_still_heard_while_a_box_is_showing():
    """With a box up, "okay" must still mean yes -- the gate is only for an idle app."""
    p = Person((0, 0)).walk((1500.0, 1000.0)).stand(0.6).say("clean here").stand(2.0)
    p.say("okay").stand(1.5).walk((2900, 2400)).stand(2.0)
    link = Link()
    app, spoken, _ = run(p, link)
    assert app.state == CLEANING and not app.log.of("not_for_me")


# ------------------------------------------------------- the words Parakeet got wrong


@pytest.mark.parametrize(
    "said, want",
    [
        # live, 7 Oct: both of these broke "clean from here ... to here"
        ("clean from here To hear", "clean from here To here"),
        ("Clean from here Two here", "Clean from here to here"),
        # live, 8 Oct: "clean" is heard as almost anything, and these were lost
        ("Claim from here", "clean from here"),
        ("screen from here", "clean from here"),
        ("Cleaned from here", "clean from here"),
        ("Cling here", "clean here"),
        # left alone: a size, a sentence that really is about hearing, and one that really is
        # about a claim or a screen, with no place in it
        ("clean a two metre square here", "clean a two metre square here"),
        ("I cannot hear you", "I cannot hear you"),
        ("Claim the prize", "Claim the prize"),
        ("screen share", "screen share"),
    ],
)
def test_repair_words(said: str, want: str) -> None:
    from deixis.types import Word
    from pointdraw.vacuum.app import repair_words

    words = [Word(w, i, i + 0.2) for i, w in enumerate(said.split())]
    assert " ".join(w.text for w in repair_words(words)) == want


def test_mishdeard_to_here_still_makes_a_two_corner_box() -> None:
    """ "Two here" is heard instead of "to here": the box still has two corners."""
    from deixis.types import Word
    from pointdraw.vacuum.agent import parse
    from pointdraw.vacuum.app import repair_words

    words = [Word(w, i, i + 0.2) for i, w in enumerate("clean from here Two here".split())]
    calls = parse([w.text for w in repair_words(words)])
    assert calls == [
        {
            "name": "clean",
            "args": {
                "place": {
                    "from": {"deixis": "here", "word_index": 2},
                    "to": {"deixis": "here", "word_index": 4},
                }
            },
        }
    ]
