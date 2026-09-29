"""Scoring helpers, WAV I/O, the energy VAD, and the committed R2 ground truth."""

import json
from pathlib import Path

import numpy as np
import pytest

from pointdraw.audio.models import SILERO_VAD_PATH
from pointdraw.audio.scoring import (
    DEMONSTRATIVES,
    align,
    matched_pairs,
    normalize_word,
    tokenize,
    word_error_rate,
)
from pointdraw.audio.vad import FRAME, EnergyVad, SileroVad
from pointdraw.audio.wavio import read_wav, resample_linear, write_wav

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "audio" / "data" / "r2_ground_truth.json"


def test_normalize():
    assert normalize_word("Here,") == "here"
    assert normalize_word("that's") == "that's"
    assert normalize_word("'quoted'") == "quoted"
    assert tokenize("Put the dog here, no, there.") == ["put", "the", "dog", "here", "no", "there"]


def test_align_and_wer():
    ref = tokenize("put a tree here")
    hyp = tokenize("a tree right here")
    a = align(ref, hyp)
    assert (a.substitutions, a.deletions, a.insertions) == (0, 1, 1)
    assert matched_pairs(ref, hyp) == {1: 0, 2: 1, 3: 3}
    assert word_error_rate([ref], [hyp]) == pytest.approx(0.5)
    assert word_error_rate([ref], [ref]) == 0.0


def test_wav_round_trip_and_resample(tmp_path):
    x = (0.5 * np.sin(np.arange(16000) / 10)).astype(np.float32)
    write_wav(tmp_path / "a.wav", x)
    y, sr = read_wav(tmp_path / "a.wav")
    assert sr == 16000
    np.testing.assert_allclose(y, x, atol=1e-4)
    assert len(resample_linear(np.zeros(48000, np.float32), 48000)) == 16000


def test_energy_vad_tracks_noise_floor():
    rng = np.random.default_rng(0)
    vad = EnergyVad()
    quiet = [vad.prob(rng.normal(0, 0.003, FRAME)) for _ in range(60)]
    assert max(quiet[20:]) < 0.5
    loud = vad.prob(0.2 * np.sin(np.arange(FRAME) / 5))
    assert loud > 0.9


@pytest.mark.skipif(not (ROOT / SILERO_VAD_PATH).exists(), reason="Silero model not downloaded")
def test_silero_vad_separates_silence_from_voice_like_audio():
    vad = SileroVad(ROOT / SILERO_VAD_PATH)
    assert vad.prob(np.zeros(FRAME, np.float32)) < 0.2
    with pytest.raises(ValueError):
        vad.prob(np.zeros(100, np.float32))


def test_ground_truth_is_consistent():
    gt = json.loads(GT.read_text(encoding="utf-8"))
    items = gt["items"]
    assert len(items) >= 30
    n_dem = 0
    for it in items:
        words = it["words"]
        assert [w["text"] for w in words] == tokenize(it["text"])
        assert [w["i"] for w in words] == list(range(len(words)))
        t0s = [w["t0"] for w in words]
        assert t0s == sorted(t0s)
        assert all(w["t1"] >= w["t0"] for w in words)
        assert words[-1]["t1"] <= it["duration"] + 1e-6
        for w in words:
            assert w["deictic"] == (w["text"] in DEMONSTRATIVES)
            n_dem += w["deictic"]
    assert n_dem == gt["meta"]["n_demonstratives"]
