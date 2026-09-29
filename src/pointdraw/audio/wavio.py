"""Load and save audio as float32 mono at 16 kHz (the rate every speech model here uses)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as float32 in [-1, 1], mixing down to mono."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = pcm.astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float32 mono samples as 16-bit PCM."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file (WAV, m4a, mp3, ...) as float32 mono at `sample_rate`.

    WAVs at the right rate are read directly; everything else is decoded with PyAV.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        x, sr = read_wav(path)
        if sr == sample_rate:
            return x
    import av  # only needed for compressed formats

    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):  # flush
            out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Cheap linear resampling, good enough for speech at 44.1/48 kHz -> 16 kHz after capture."""
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_out = np.arange(n_out) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)
