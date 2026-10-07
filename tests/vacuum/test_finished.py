"""The app must notice when the robot has finished by itself.

7 Oct, live: rahul said "clean here", the robot was sent, and the app sat in CLEANING for
ever. The only ways out were "stop" and "dock", so every later command got "Sorry, I didn't
get that". The robot had already finished and was saying so; nobody was listening.
"""

from __future__ import annotations

from tests.vacuum.test_app import PROFILE, Link, Person, Spoken, play, to_map

from deixis.mcp import DeixisClient
from pointdraw.vacuum.agent import RuleAgent
from pointdraw.vacuum.app import CLEANING, LISTEN, VacuumApp
from pointdraw.vacuum.calibrate import FloorToMap


def cleaning_app() -> tuple[VacuumApp, Spoken, float]:
    """An app that has just sent the robot off to clean, and the time it did."""
    core = DeixisClient(PROFILE, mode="direct")
    floor = [(200, 150), (2600, 300), (2400, 2200), (500, 2000)]
    f2m = FloorToMap.fit([(p, to_map(p)) for p in floor], core)
    spoken = Spoken()
    p = Person((0, 0)).walk((1500.0, 1000.0)).stand(0.6).say("clean here").stand(2.0)
    p.say("yes").stand(1.5).walk((2900, 2400)).stand(2.0)
    app = VacuumApp(RuleAgent(PROFILE), core, Link(), f2m, PROFILE, say=spoken)
    play(app, p)
    assert app.state == CLEANING, app.state
    return app, spoken, app._cleaning_since


def test_it_goes_back_to_listening_when_the_robot_stops():
    app, spoken, t0 = cleaning_app()
    p = app.p
    app.on_robot_state(t0 + p.min_clean_s + 0.1, "idle")
    app.tick(t0 + p.min_clean_s + 0.2)
    assert app.state == CLEANING, "it has not been quiet long enough yet"
    app.tick(t0 + p.min_clean_s + p.clean_settle_s + 0.2)
    assert app.state == LISTEN
    assert spoken.lines[-1] == "Finished."


def test_one_report_is_enough_because_the_watcher_only_sends_changes():
    """The live bug: the robot finishes, says "idle" once inside the grace time, and goes
    quiet. Waiting for a second report meant waiting for ever."""
    app, spoken, t0 = cleaning_app()
    p = app.p
    app.on_robot_state(t0 + 0.5, "idle")  # inside min_clean_s, and the only one ever sent
    app.tick(t0 + 1.0)
    assert app.state == CLEANING
    app.tick(t0 + p.min_clean_s + p.clean_settle_s + 0.5)
    assert app.state == LISTEN
    assert spoken.lines[-1] == "Finished."


def test_a_stray_idle_mid_clean_does_not_end_the_job():
    app, _, t0 = cleaning_app()
    p = app.p
    app.on_robot_state(t0 + p.min_clean_s + 0.1, "idle")
    app.on_robot_state(t0 + p.min_clean_s + 0.5, "cleaning")  # it was only a blip
    app.tick(t0 + p.min_clean_s + p.clean_settle_s + 1.0)
    assert app.state == CLEANING


def test_the_robot_gets_time_to_set_off():
    """A robot still leaving the dock reports the old state; that must not end the job."""
    app, _, t0 = cleaning_app()
    for dt in (0.1, 1.0, 2.0, 3.0):
        app.on_robot_state(t0 + dt, "docked")
        app.tick(t0 + dt)
    assert app.state == CLEANING


def test_stop_clears_the_cleaning_clock():
    app, _, t0 = cleaning_app()
    app.stop(t0 + 1.0)
    assert app.state == LISTEN
    app.on_robot_state(t0 + 100.0, "idle")  # must not speak again
    app.tick(t0 + 200.0)
    assert app.state == LISTEN
