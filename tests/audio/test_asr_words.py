"""Turning sherpa-onnx tokens into timed words, and the clock mapping of LocalAsr."""

import json
from pathlib import Path

import numpy as np
import pytest

from pointdraw.audio.asr_local import AsrConfig, LocalAsr, greedy_tokenize, tokens_to_words
from pointdraw.audio.models import ASR_MODELS, DEFAULT_ASR
from pointdraw.audio.wavio import read_wav

ROOT = Path(__file__).resolve().parents[2]


def test_leading_space_tokens():
    # nemotron-speech-streaming-en style: " P" "ut" " the" ...
    toks = [" P", "ut", " the", " do", "g", " he", "re"]
    ts = [0.08, 0.08, 0.16, 0.56, 0.56, 0.72, 0.72]
    w = tokens_to_words(toks, ts, t_end=1.2)
    assert [x[0] for x in w] == ["Put", "the", "dog", "here"]
    assert [x[1] for x in w] == [0.08, 0.16, 0.56, 0.72]
    assert w[0][2] == pytest.approx(0.16)  # ends where the next word starts
    assert w[-1][2] == pytest.approx(0.80)  # last token + one 80 ms frame


def test_bare_space_and_punctuation_tokens():
    # nemotron-3.5 style: bare " " before a word, "." as its own token.
    toks = [" ", "P", "ut", " her", "e", ".", " ", " No", " there"]
    ts = [0.16, 0.16, 0.16, 0.88, 0.96, 1.68, 1.68, 1.76, 2.40]
    w = tokens_to_words(toks, ts)
    assert [x[0] for x in w] == ["Put", "here", "No", "there"]
    assert w[1][1] == 0.88


def test_sentencepiece_marker_and_durations():
    toks = ["▁here", ",", "▁th", "is"]
    ts = [0.96, 1.2, 1.5, 1.6]
    w = tokens_to_words(toks, ts, durations=[0.24, 0.3, 0.16, 0.32])
    assert [x[0] for x in w] == ["here", "this"]
    assert w[0][2] == pytest.approx(1.2)
    assert w[1][2] == pytest.approx(1.92)


def test_punctuation_splits_words():
    w = tokens_to_words([" yes", ",", "no"], [0.1, 0.2, 0.3])
    assert [x[0] for x in w] == ["yes", "no"]


def test_greedy_tokenize():
    vocab = {"▁Ne", "▁N", "m", "o", "tron", "e", "▁Nem"}
    assert greedy_tokenize("Nemotron", vocab) == ["▁Nem", "o", "tron"]
    assert greedy_tokenize("Nexus", vocab) is None


def _default_model_available() -> bool:
    return ASR_MODELS[DEFAULT_ASR].available(ROOT / ".cache" / "models")


@pytest.mark.network  # the model is a ~500 MB download (scripts/asr_fetch_models.py)
@pytest.mark.skipif(not _default_model_available(), reason="ASR model not downloaded")
def test_default_model_on_a_synthetic_utterance(monkeypatch):
    monkeypatch.chdir(ROOT)
    gt = json.loads((ROOT / "tests/audio/data/r2_ground_truth.json").read_text())
    item = next(i for i in gt["items"] if i["text"] == "put a tree here")
    wav = ROOT / ".cache" / "r2" / item["wav"]
    if not wav.exists():
        pytest.skip("run scripts/make_test_utterances.py first")
    x, sr = read_wav(wav)
    x = np.concatenate([np.zeros(int(0.3 * sr), np.float32), x])
    asr = LocalAsr(AsrConfig())
    words = asr.transcribe(x, sr, t_start=1000.0 - 0.3)
    assert [w.text.lower() for w in words] == ["put", "a", "tree", "here"]
    here = words[-1]
    assert here.t0 == pytest.approx(1000.0 + item["words"][3]["t0"], abs=0.15)
