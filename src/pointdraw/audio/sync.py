"""Clap and onset detection, for measuring the audio-to-video offset.

Procedure (for later, with the camera): clap a few times in view of the camera. Detect the claps
in the audio with `detect_onsets`, detect the hands meeting in the video (frame differencing or
MediaPipe), then `match_offset(audio_times, video_times)` gives the offset to add to audio
timestamps so both streams agree on `clock.now()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OnsetParams:
    frame_s: float = 0.005  # analysis hop (and window) length
    history_s: float = 0.1  # background level is the median energy over this window
    rise_db: float = 15.0  # an onset must rise this far above the background
    floor_db: float = -45.0  # and be louder than this (dBFS)
    min_gap_s: float = 0.15  # ignore onsets closer than this to the previous one


def frame_energy_db(x: np.ndarray, sr: int, frame_s: float = 0.005) -> np.ndarray:
    """Mean-square energy per non-overlapping frame, in dBFS."""
    n = max(1, int(round(frame_s * sr)))
    k = len(x) // n
    if k == 0:
        return np.zeros(0)
    e = np.mean(np.square(x[: k * n].reshape(k, n), dtype=np.float64), axis=1)
    return 10 * np.log10(e + 1e-12)


def detect_onsets(x: np.ndarray, sr: int, p: OnsetParams | None = None) -> list[float]:
    """Times (s from the first sample) of sharp energy rises such as claps or plosives.

    Each onset is refined to the first sample in its frame whose magnitude reaches half the
    frame's peak, so the result is accurate to well under a millisecond for a clap.
    """
    p = p or OnsetParams()
    db = frame_energy_db(x, sr, p.frame_s)
    n = max(1, int(round(p.frame_s * sr)))
    hist = max(1, int(round(p.history_s / p.frame_s)))
    onsets: list[float] = []
    last = -np.inf
    for k in range(len(db)):
        bg = np.median(db[max(0, k - hist) : k]) if k > 0 else -120.0
        if db[k] > p.floor_db and db[k] - bg >= p.rise_db:
            seg = np.abs(x[k * n : (k + 1) * n])
            first = int(np.argmax(seg >= 0.5 * seg.max())) if seg.size else 0
            t = (k * n + first) / sr
            if t - last >= p.min_gap_s:
                onsets.append(t)
                last = t
    return onsets


def first_sound(x: np.ndarray, sr: int, rel_db: float = -35.0, frame_s: float = 0.005) -> float:
    """Start time of the first frame within `rel_db` of the loudest frame (None-safe: 0.0)."""
    db = frame_energy_db(x, sr, frame_s)
    if db.size == 0:
        return 0.0
    idx = np.flatnonzero(db >= db.max() + rel_db)
    return float(idx[0] * frame_s) if idx.size else 0.0


def last_sound(x: np.ndarray, sr: int, rel_db: float = -35.0, frame_s: float = 0.005) -> float:
    """End time of the last frame within `rel_db` of the loudest frame."""
    db = frame_energy_db(x, sr, frame_s)
    if db.size == 0:
        return 0.0
    idx = np.flatnonzero(db >= db.max() + rel_db)
    return float((idx[-1] + 1) * frame_s) if idx.size else 0.0


def match_offset(
    a: list[float], b: list[float], tol_s: float = 0.05, max_offset_s: float = 2.0
) -> tuple[float, int]:
    """Offset d such that a_i + d ≈ b_j for as many events as possible.

    Returns (median offset over matched pairs, number matched). Tries every pairwise difference
    within `max_offset_s` as a candidate, so it copes with missed or extra events.
    """
    best: tuple[int, float] = (0, 0.0)
    for ai in a:
        for bj in b:
            d = bj - ai
            if abs(d) > max_offset_s:
                continue
            matched = [min(b, key=lambda t, x=x: abs(t - x - d)) - x for x in a]
            good = [m for m in matched if abs(m - d) <= tol_s]
            if len(good) > best[0]:
                best = (len(good), float(np.median(good)))
    return best[1], best[0]
