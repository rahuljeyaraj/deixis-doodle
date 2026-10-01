"""replay_live: a session logged like live_run.py, run again offline."""

import json

import numpy as np

from pointdraw.controller.config import MachineConfig
from pointdraw.live.replay import load, replay, summary
from pointdraw.virtual import scenefile
from pointdraw.virtual.hand import VirtualHand
from pointdraw.virtual.voice import text_mode

T0 = 5000.0  # the camera's first frame on the shared clock


def write_session(tmp_path, scene: str, calls=None):
    """Log a virtual scene in live_run.py's format (hands.jsonl + timeline.jsonl)."""
    cfg = MachineConfig()
    spec = scenefile.find(scene)
    hand = VirtualHand(spec.hand, cfg.paper_box, seed=0)
    with (tmp_path / "hands.jsonl").open("w") as f:
        print(json.dumps({"t0": T0}), file=f)
        for k in range(int((hand.end_time() + 3) * 23)):
            t = k / 23
            hf = hand.frame(t)
            hands = [
                {
                    "score": h.score,
                    "handedness": h.handedness,
                    "mm": np.round(h.landmarks_mm, 2).tolist(),
                    "world": np.round(h.world, 4).tolist(),
                }
                for h in hf.hands
            ]
            print(json.dumps({"t": round(t, 3), "hands": hands}), file=f)
    events = []
    for i, s in enumerate(text_mode(spec.lines)):
        words = [[w.text, T0 + w.t0, T0 + w.t1] for w in s.words]
        events.append({"t": T0 + s.t_start_heard, "kind": "speech_start"})
        events.append(
            {"t": T0 + s.t_avail, "kind": "utterance", "id": i, "text": s.text, "words": words}
        )
        if calls:
            events.append(
                {
                    "t": T0 + s.t_avail,
                    "kind": "tool_calls",
                    "utt": i,
                    "calls": calls[i],
                    "source": "nemotron",
                    "latency_s": 1.0,
                }
            )
    events.sort(key=lambda e: e["t"])
    with (tmp_path / "timeline.jsonl").open("w") as f:
        for e in events:
            print(json.dumps(e), file=f)
    return cfg


def test_replay_draws_where_the_finger_was(tmp_path):
    cfg = write_session(tmp_path, "S27")  # hooked point, the other hand resting
    rt, agent = replay(load(tmp_path), cfg)
    assert agent.parsed == 1 and agent.recorded == 0
    assert summary(rt.log) == {"heard": 1, "asks": 0, "drawn": 1, "guards": 0}
    [d] = rt.log.of("drawn")
    assert d["label"] == "cross"
    assert np.allclose(d["ink_center"], (120, 60), atol=3)


def test_replay_uses_the_recorded_reply(tmp_path):
    star = {"name": "place_drawing", "args": {"object": "star", "place": {"empty_space": True}}}
    cfg = write_session(tmp_path, "S27", calls=[[star]])
    rt, agent = replay(load(tmp_path), cfg)
    assert agent.recorded == 1 and agent.parsed == 0
    # The recorded reply names an object that wasn't said: today's guard drops it.
    assert rt.log.of("guard") and not rt.log.of("drawn")


def test_a_run_without_landmarks_is_refused(tmp_path):
    (tmp_path / "hands.jsonl").write_text('{"t": 0.0, "hands": []}\n')
    (tmp_path / "timeline.jsonl").write_text("")
    try:
        load(tmp_path)
    except ValueError as e:
        assert "header" in str(e)
    else:
        raise AssertionError("expected a ValueError")
