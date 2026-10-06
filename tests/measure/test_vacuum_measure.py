"""M24 and M25 from a vacuum run's timeline, and the robot-position watcher."""

import json

from pointdraw.measure import vacuum
from pointdraw.vacuum.watch import RobotWatch


def write_run(tmp_path, events):
    run = tmp_path / "run"
    run.mkdir()
    (run / "timeline.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return run


def test_box_and_echo(tmp_path):
    box = [1000, -1000, 1500, 1000]  # map mm
    ev = [
        {
            "t": 1.0,
            "kind": "utterance",
            "text": "clean from here to here",
            "words": [["clean", 0.1, 0.4], ["here", 3.0, 3.3]],
        },
        {"t": 4.5, "kind": "preview", "box": [0, 0, 1500, 1000]},
        {"t": 4.5, "kind": "say", "text": "Clean from there to here?"},
        {"t": 4.9, "kind": "spoken", "text": "Clean from there to here?", "t1": 6.0},
        {"t": 10.0, "kind": "clean_area", "map_box": box, "reply": {"ok": True}},
        {"t": 20.0, "kind": "robot", "state": "cleaning", "x": 0, "y": 0},  # leaving the dock
        {"t": 40.0, "kind": "robot", "state": "cleaning", "x": 1100, "y": -900},  # in
        {"t": 41.0, "kind": "robot", "state": "cleaning", "x": 1100, "y": -900},  # same again
        {"t": 45.0, "kind": "robot", "state": "cleaning", "x": 2600, "y": -500},  # 10 cm out
        {"t": 50.0, "kind": "robot", "state": "cleaning", "x": 2400, "y": -100},  # in
        {"t": 60.0, "kind": "robot", "state": "returning", "x": 300, "y": 0},  # going home
    ]
    res = vacuum.summary([write_run(tmp_path, ev)])
    b = res["M24_box"]
    assert b["runs"] == 1 and b["positions"] == 3 and b["inside"] == 2
    assert b["furthest_out_cm"] == 10.0
    m = res["M25_last_word_to_echo_s"]
    assert m["n"] == 1 and m["median"] == 1.6 and m["all_played"]
    text = "\n".join(vacuum.lines(res))
    assert "M24" in text and "67 %" in text and "M25" in text


class Link:
    def __init__(self, replies):
        self.replies = list(replies)

    def status(self):
        return self.replies.pop(0)


def test_watch_keeps_only_changes():
    st = {"state": "cleaning", "robot": {"x": 1, "y": 2, "a": 0, "age_s": 0.5}}
    moved = {"state": "cleaning", "robot": {"x": 5, "y": 2, "a": 0, "age_s": 0.0}}
    w = RobotWatch(Link([st, st, moved]), now=lambda: 100.0)
    for _ in range(3):
        w.poll()
    ev = w.drain()
    assert [(e["x"], e["t"]) for e in ev] == [(1, 99.5), (5, 100.0)]
    assert w.drain() == []
