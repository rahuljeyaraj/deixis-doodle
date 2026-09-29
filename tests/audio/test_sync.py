"""Clap/onset detection and audio-video offset matching."""

import numpy as np
import pytest

from pointdraw.audio.sync import detect_onsets, first_sound, last_sound, match_offset

SR = 16000


def claps(times: list[float], dur: float = 4.0, seed: int = 1) -> np.ndarray:
    """Background noise with decaying broadband bursts at `times`."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(dur * SR))
    for t in times:
        k = int(t * SR)
        n = int(0.05 * SR)
        x[k : k + n] += rng.normal(0, 0.5, n) * np.exp(-np.arange(n) / (0.008 * SR))
    return x.astype(np.float32)


def test_detects_claps_to_within_a_millisecond():
    truth = [0.5, 1.23, 2.0, 3.1]
    got = detect_onsets(claps(truth), SR)
    assert len(got) == len(truth)
    for g, t in zip(got, truth, strict=True):
        assert g == pytest.approx(t, abs=0.001)


def test_no_onsets_in_steady_noise():
    rng = np.random.default_rng(0)
    assert detect_onsets(rng.normal(0, 0.01, 2 * SR).astype(np.float32), SR) == []


def test_first_and_last_sound():
    x = np.zeros(2 * SR, np.float32)
    x[int(0.4 * SR) : int(1.1 * SR)] = 0.2
    assert first_sound(x, SR) == pytest.approx(0.4, abs=0.006)
    assert last_sound(x, SR) == pytest.approx(1.1, abs=0.006)


def test_match_offset_with_missing_and_extra_events():
    audio = [0.5, 1.23, 2.0, 3.1]
    video = [0.5 + 0.083, 2.0 + 0.081, 3.1 + 0.085, 3.6]  # one clap missed, one spurious event
    d, n = match_offset(audio, video)
    assert n == 3
    assert d == pytest.approx(0.083, abs=0.002)
