"""Segmenter on synthetic audio: pre-roll, end-of-utterance silence, max length, mute."""

import numpy as np
import pytest

from pointdraw.audio.segmenter import FRAME_S, SAMPLE_RATE, Segmenter, SegmenterParams
from pointdraw.audio.vad import FRAME, EnergyVad


class LoudVad:
    """Speech = RMS above a fixed level; deterministic for tests."""

    def __init__(self, level: float = 0.01) -> None:
        self.level = level
        self.resets = 0

    def prob(self, frame: np.ndarray) -> float:
        return 1.0 if np.sqrt(np.mean(frame**2)) > self.level else 0.0

    def reset(self) -> None:
        self.resets += 1


def signal(spec: list[tuple[str, float]], seed: int = 0) -> np.ndarray:
    """Concatenate ("speech"|"silence", seconds) pieces; speech is a 220 Hz tone plus noise."""
    rng = np.random.default_rng(seed)
    parts = []
    for kind, dur in spec:
        n = int(round(dur * SAMPLE_RATE))
        t = np.arange(n) / SAMPLE_RATE
        noise = rng.normal(0, 1e-4, n)
        parts.append(noise + (0.3 * np.sin(2 * np.pi * 220 * t) if kind == "speech" else 0))
    return np.concatenate(parts).astype(np.float32)


def run(seg: Segmenter, x: np.ndarray, t0: float = 100.0, block: int = 700):
    """Push in odd-sized blocks (not a multiple of the VAD frame) with exact timestamps."""
    out = []
    for k in range(0, len(x), block):
        out += seg.push(x[k : k + block], t0 + k / SAMPLE_RATE)
    return out


def test_one_utterance_with_preroll_and_times():
    x = signal([("silence", 1.0), ("speech", 1.0), ("silence", 1.4)])
    utts = run(Segmenter(LoudVad()), x)
    assert len(utts) == 1
    u = utts[0]
    # Speech starts at 1.0 s; the first speech frame is within one VAD frame of that.
    assert u.t_speech == pytest.approx(101.0, abs=FRAME_S)
    # 300 ms of pre-roll before it, give or take a frame.
    assert u.t_speech - u.t_start == pytest.approx(0.3, abs=FRAME_S)
    # Ends about tail_s (0.2 s) after the speech ends at 2.0 s.
    assert u.t_end == pytest.approx(102.2, abs=2 * FRAME_S)
    assert u.duration == pytest.approx(u.t_end - u.t_start, abs=1e-6)
    assert not u.truncated
    # The samples really are the audio at those times.
    k = int(round((u.t_start - 100.0) * SAMPLE_RATE))
    np.testing.assert_array_equal(u.samples[:FRAME], x[k : k + FRAME])


def test_short_pause_does_not_split_but_long_one_does():
    x = signal(
        [("silence", 0.5), ("speech", 0.6), ("silence", 0.9), ("speech", 0.5), ("silence", 1.4)]
    )
    assert len(run(Segmenter(LoudVad()), x)) == 1
    x = signal(
        [("silence", 0.5), ("speech", 0.6), ("silence", 1.3), ("speech", 0.5), ("silence", 1.4)]
    )
    assert len(run(Segmenter(LoudVad()), x)) == 2


def test_needs_a_full_second_of_silence_to_end():
    x = signal([("silence", 0.5), ("speech", 0.5), ("silence", 0.9)])
    seg = Segmenter(LoudVad())
    assert run(seg, x) == []
    assert seg.in_speech
    assert len(seg.flush()) == 1


def test_max_length_truncates():
    x = signal([("silence", 0.5), ("speech", 12.0), ("silence", 1.4)])
    utts = run(Segmenter(LoudVad()), x)
    assert utts[0].truncated
    assert utts[0].duration <= 10.0 + FRAME_S
    assert len(utts) == 2  # the rest of the speech becomes a second utterance


def test_clicks_are_ignored():
    x = signal([("silence", 0.5), ("speech", 0.03), ("silence", 1.4)])
    assert run(Segmenter(LoudVad()), x) == []


def test_mute_drops_audio_and_unmute_resumes():
    vad = LoudVad()
    seg = Segmenter(vad)
    x = signal([("silence", 0.3), ("speech", 0.8), ("silence", 1.4)])
    seg.mute()
    assert run(seg, x) == []
    assert seg.muted and not seg.in_speech
    seg.unmute()
    assert vad.resets >= 2
    assert len(run(seg, x, t0=200.0)) == 1


def test_mute_mid_utterance_discards_it():
    seg = Segmenter(LoudVad())
    x = signal([("silence", 0.3), ("speech", 0.8)])
    run(seg, x)
    assert seg.in_speech
    seg.mute()
    seg.unmute()
    assert not seg.in_speech
    assert seg.flush() == []


def test_energy_vad_segments_too():
    x = signal([("silence", 1.5), ("speech", 1.0), ("silence", 1.4)])
    utts = run(Segmenter(EnergyVad(), SegmenterParams()), x)
    assert len(utts) == 1
    assert utts[0].t_speech == pytest.approx(101.5, abs=2 * FRAME_S)
