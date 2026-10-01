"""The app speaks its replies; the microphone is muted meanwhile (night 2 Oct 2026, item 11)."""

from __future__ import annotations

import time

import numpy as np

from pointdraw.live.speaker import Speaker
from pointdraw.live.speech import Listener


def tone(text: str) -> np.ndarray:
    return np.zeros(int(0.2 * 16000), np.float32)  # 0.2 s of "speech" per reply


def test_mute_covers_each_reply_plus_0_3_s_and_writes_wavs(tmp_path):
    sp = Speaker(tone, play=False, wav_dir=tmp_path, after_s=0.3)
    assert not sp.muted()
    sp.say("Where should it go?")
    sp.say("Stopped.")
    samples: list[tuple[float, bool]] = []
    t_end = time.perf_counter() + 2.0
    while time.perf_counter() < t_end:
        samples.append((time.perf_counter(), sp.muted()))
        time.sleep(0.005)
    sp.close()
    assert [t for t, _, _ in sp.spoken] == ["Where should it go?", "Stopped."]
    (_, a0, a1), (_, b0, b1) = sp.spoken
    assert a1 - a0 >= 0.19 and b1 - b0 >= 0.19 and b0 >= a1  # one at a time
    for t, m in samples:
        if a0 + 0.01 < t < a1 - 0.01 or b0 + 0.01 < t < b1 + 0.29:
            assert m, f"not muted at {t - a0:.2f} s"
        if t > b1 + 0.35:
            assert not m, f"still muted at {t - b1:.2f} s after the last reply"
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [
        "001-where-should-it-go-.wav",
        "002-stopped-.wav",
    ]


class _Loud:
    """A VAD that hears speech in every frame."""

    def prob(self, frame) -> float:
        return 1.0

    def reset(self) -> None:
        pass


class _Asr:
    def transcribe(self, samples, rate, t0):
        from deixis.types import Word

        return [Word("hello", t0, t0 + 0.1)]


class _Mic:
    def __init__(self, n: int) -> None:
        self.n = n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def blocks(self, timeout: float = 1.0):
        for i in range(self.n):
            time.sleep(0.002)
            yield np.full(512, 0.3, np.float32), i * 0.032
        while True:  # then silence, so an utterance can close
            time.sleep(0.002)
            i += 1
            yield np.zeros(512, np.float32), i * 0.032


def test_listener_hears_nothing_while_the_app_speaks():
    muted = [True]
    lis = Listener(_Asr(), _Loud(), capture=lambda: _Mic(200), muted=lambda: muted[0])
    lis.start()
    try:
        time.sleep(0.3)
        assert lis.events.empty() and lis.segmenter.muted  # the robot's voice: dropped
        muted[0] = False
        time.sleep(0.3)
        assert not lis.segmenter.muted
        kinds = []
        while not lis.events.empty():
            kinds.append(lis.events.get()[0])
        assert "speech_start" in kinds  # the person, once the reply is over
    finally:
        lis.close()
