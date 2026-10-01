"""Recovering from a pulled USB cable or microphone (night 2 Oct 2026, item 6)."""

from __future__ import annotations

import time

import numpy as np

from deixis.types import Word
from pointdraw.audio.vad import EnergyVad
from pointdraw.controller.config import MachineConfig
from pointdraw.live.speech import Listener


class _Asr:
    def transcribe(self, samples, rate, t0):
        return [Word("hello", t0, t0 + 0.2)]


class _Cap:
    """A microphone: `fail` raises on open; else `n` blocks of silence, then it stops."""

    def __init__(self, fail: bool = False, n: int = 3, forever: bool = False) -> None:
        self.fail, self.n, self.forever = fail, n, forever

    def __enter__(self):
        if self.fail:
            raise RuntimeError("no microphone found")
        return self

    def __exit__(self, *exc):
        pass

    def blocks(self, timeout: float = 1.0):
        i = 0
        while self.forever or i < self.n:
            i += 1
            time.sleep(0.005)
            yield np.zeros(512, np.float32), i * 0.032
        raise TimeoutError("no audio from the microphone")


def test_listener_reports_a_lost_microphone_once_and_comes_back():
    caps = iter([_Cap(n=3), _Cap(fail=True), _Cap(fail=True), _Cap(forever=True)])
    lis = Listener(_Asr(), EnergyVad(), retry_s=0.01, capture=lambda: next(caps))
    lis.start()
    try:
        got: list[tuple] = []
        t_end = time.perf_counter() + 5
        while time.perf_counter() < t_end and not any(e[0] == "mic_back" for e in got):
            try:
                got.append(lis.events.get(timeout=0.05))
            except Exception:  # noqa: BLE001 - queue.Empty
                pass
        kinds = [e[0] for e in got]
        assert kinds.count("error") == 1  # said once, not at every retry
        assert "TimeoutError" in next(e[1] for e in got if e[0] == "error")
        assert kinds[-1] == "mic_back" and not lis.lost
    finally:
        lis.close()


def test_runtime_drops_what_waited_for_a_failed_plotter_and_says_so_once():
    from pointdraw.agent.nemotron import AgentResult
    from pointdraw.agent.schema import parse_call
    from pointdraw.controller.api import PlotEvent
    from pointdraw.runtime.runtime import Runtime
    from pointdraw.virtual.plotter import VirtualPlotter

    class _Agent:
        def understand(self, words, context):
            call = parse_call("place_drawing", {"object": "cross", "place": {"named": "middle"}})
            return AgentResult(calls=[call], source="cache")

    machine = MachineConfig()
    plotter = VirtualPlotter(machine)
    rt = Runtime(_Agent(), plotter, machine)
    rt.start(0.0)
    words = ["draw", "a", "cross", "in", "the", "middle"]
    rt.on_utterance([Word(w, i * 0.2, i * 0.2 + 0.2) for i, w in enumerate(words)], 1.5)
    rt.tick(1.5)
    assert rt.proposal is not None
    tag = next(iter(rt.proposal.previews))
    plotter.events.append(PlotEvent(1.1, "error", "preview", tag, {"message": "write failed"}))
    plotter.events.append(PlotEvent(1.2, "error", "", None, {"message": "write failed"}))
    rt.tick(1.3)
    assert rt.proposal is None and rt.state == "LISTEN"
    says = [e["text"] for e in rt.log.of("say")]
    assert says.count("The plotter isn't answering. Check its cable.") == 1
    assert rt.plotter_problem == "write failed"
    assert len(rt.log.of("plotter_problem")) == 2
    rt.plotter_back(2.0)
    assert rt.plotter_problem is None and "back" in rt.log.of("say")[-1]["text"]


def test_window_says_what_is_wrong_in_plain_words():
    from tests.live.test_live_run import load_script

    live_run = load_script("live_run")
    assert live_run.problems("idle", None, None) == []
    both = live_run.problems("error", None, "OSError: gone")
    assert both[0].startswith("Plotter problem") and "press R" in both[0]
    assert both[1].startswith("Microphone lost")
    assert live_run.problems("idle", "write failed", None)[0].startswith("Plotter problem")
