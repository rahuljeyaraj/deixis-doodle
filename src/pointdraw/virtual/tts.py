"""tts voice mode: Windows' built-in voice speaks the scene's lines, then the real audio front
end hears them.

1. Each line is spoken by System.Speech (SAPI) to a 16 kHz WAV
   (`scripts/make_test_utterances.ps1`, cached in `.cache/virtual/tts/`).
2. The WAVs are mixed into one continuous stream at the lines' start times, over a quiet
   noise floor (-60 dBFS). A line can be cut off (`cut`) or buried in noise (`noise_db`).
3. The real `Segmenter` with the Silero VAD (energy VAD if Silero is missing) splits the
   stream into utterances, block by block, as the microphone would.
4. The real ASR (`LocalAsr`, Parakeet TDT 0.6B v2 on the CPU) gives words with times.
   The transcript reaches the runtime when the segmenter closes the utterance plus the
   measured ASR time.

Windows only. Needs the ASR model (`uv run python scripts/asr_fetch_models.py`).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from pointdraw.virtual.scenefile import Line
from pointdraw.virtual.voice import VAD_S, Spoken

ROOT = Path(__file__).resolve().parents[3]
PS1 = ROOT / "scripts" / "make_test_utterances.ps1"
CACHE = ROOT / ".cache" / "virtual" / "tts"
SR = 16000
BLOCK = 512  # 32 ms, the segmenter's frame
VOICE = "Microsoft Zira Desktop"

_ASR = None


def _asr():
    global _ASR
    if _ASR is None:
        from pointdraw.audio.asr_local import LocalAsr

        _ASR = LocalAsr()
    return _ASR


def speak(texts: list[tuple[str, int]], voice: str = VOICE) -> list[np.ndarray]:
    """WAV samples (float32, 16 kHz) for each (text, rate), synthesised once and cached."""
    from pointdraw.audio.wavio import read_wav

    CACHE.mkdir(parents=True, exist_ok=True)
    paths, jobs = [], []
    for text, rate in texts:
        key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
        wav = CACHE / f"{key}.wav"
        paths.append(wav)
        if not wav.exists():
            jobs.append({"id": key, "text": text, "voice": voice, "rate": rate, "wav": str(wav)})
    if jobs:
        with tempfile.TemporaryDirectory() as d:
            jf, of = Path(d) / "jobs.json", Path(d) / "events.json"
            jf.write_text(json.dumps(jobs), encoding="utf-8")
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1),
                 "-Jobs", str(jf), "-Out", str(of)],
                check=True, capture_output=True, timeout=120,
            )  # fmt: skip
    out = []
    for p in paths:
        x, sr = read_wav(p)
        assert sr == SR, f"{p}: {sr} Hz"
        out.append(np.asarray(x, np.float32).reshape(-1))
    return out


def mix(lines: list[Line], shift: float = 0.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """One stream with every line at its start time. Returns samples and the stream's t0."""
    waves = speak([(ln.text, getattr(ln, "tts_rate", 0)) for ln in lines])
    end = max(ln.at + shift + len(w) / SR for ln, w in zip(lines, waves, strict=True)) + 2.0
    rng = np.random.default_rng(seed)
    stream = rng.normal(0.0, 10 ** (-60 / 20), int(end * SR)).astype(np.float32)
    for ln, w in zip(lines, waves, strict=True):
        cut = getattr(ln, "cut", None)
        if cut:
            w = w[: int(len(w) * cut)]
        i0 = int(round((ln.at + shift) * SR))
        seg = stream[i0 : i0 + len(w)]
        seg += w[: len(seg)]
        noise_db = getattr(ln, "noise_db", None)
        if noise_db is not None:
            seg += rng.normal(0.0, 10 ** (noise_db / 20), len(seg)).astype(np.float32)
    return stream, 0.0


def tts_mode(lines: list[Line], shift: float = 0.0, seed: int = 0) -> list[Spoken]:
    """Hear the scene's lines through the real VAD, segmenter and ASR."""
    from pointdraw.audio.segmenter import Segmenter
    from pointdraw.audio.vad import make_vad

    if not lines:
        return []
    stream, t0 = mix(lines, shift, seed)
    seg = Segmenter(make_vad())
    asr = _asr()
    out: list[Spoken] = []
    for i in range(0, len(stream), BLOCK):
        t_block = t0 + i / SR
        for u in seg.push(stream[i : i + BLOCK], t_block):
            t_closed = t_block + BLOCK / SR
            w0 = time.perf_counter()
            words = asr.transcribe(u.samples, u.sample_rate, u.t_start)
            asr_s = time.perf_counter() - w0
            if words:
                out.append(Spoken(words, u.t_speech + VAD_S, t_closed + asr_s))
    for u in seg.flush():
        words = asr.transcribe(u.samples, u.sample_rate, u.t_start)
        if words:
            out.append(Spoken(words, u.t_speech + VAD_S, u.t_end + 0.2))
    return out
