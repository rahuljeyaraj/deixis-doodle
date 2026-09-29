"""Cut the microphone stream into utterances with a VAD.

Feed it timestamped blocks of any size; it hands back finished utterances:

    seg = Segmenter(make_vad())
    for samples, t0 in capture.blocks():
        for utt in seg.push(samples, t0):
            words = asr.transcribe(utt.samples, utt.sample_rate, utt.t_start)

Rules (plan 4.4, brief SPEECH): 300 ms of pre-roll before the first speech frame; an utterance
ends after 600 ms of silence; 10 s at most. `mute()` while TTS plays so the robot never hears
itself, `unmute()` afterwards.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pointdraw.audio.vad import FRAME, Vad

SAMPLE_RATE = 16000
FRAME_S = FRAME / SAMPLE_RATE


@dataclass
class SegmenterParams:
    start_threshold: float = 0.5  # speech probability to count a frame as speech
    end_threshold: float = 0.35  # below this a frame counts as silence (hysteresis)
    min_speech_s: float = 0.064  # consecutive speech needed to open an utterance
    preroll_s: float = 0.3
    end_silence_s: float = 0.6
    tail_s: float = 0.2  # silence kept after the last speech frame
    max_s: float = 10.0
    min_utterance_s: float = 0.15  # shorter bursts (clicks) are dropped


@dataclass
class Utterance:
    samples: np.ndarray  # float32 mono at sample_rate, pre-roll included
    t_start: float  # clock time of samples[0]
    t_speech: float  # clock time of the first speech frame
    t_end: float  # clock time just after the last sample
    sample_rate: int = SAMPLE_RATE
    truncated: bool = False  # hit max_s

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass
class _Frame:
    x: np.ndarray
    t: float
    speech: bool = False


@dataclass
class Segmenter:
    vad: Vad
    params: SegmenterParams = field(default_factory=SegmenterParams)

    def __post_init__(self) -> None:
        self._pending = np.zeros(0, np.float32)
        self._pending_t = 0.0
        # Room for the pre-roll plus the speech frames needed to open an utterance.
        n_pre = round(self.params.preroll_s / FRAME_S) + round(self.params.min_speech_s / FRAME_S)
        self._pre: deque[_Frame] = deque(maxlen=n_pre + 1)
        self._utt: list[_Frame] = []
        self._run = 0  # consecutive speech frames (before opening)
        self._silence = 0  # consecutive silent frames (while open)
        self._muted = False

    # -- control -----------------------------------------------------------------------------
    @property
    def in_speech(self) -> bool:
        return bool(self._utt)

    @property
    def muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        """Drop input (and any utterance in progress) until `unmute()`, e.g. while TTS plays."""
        self._muted = True
        self.reset()

    def unmute(self) -> None:
        self._muted = False
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, np.float32)
        self._pre.clear()
        self._utt = []
        self._run = self._silence = 0
        self.vad.reset()

    # -- streaming ---------------------------------------------------------------------------
    def push(self, samples: np.ndarray, t0: float) -> list[Utterance]:
        """Add a block whose first sample is at clock time `t0`; return finished utterances."""
        if self._muted:
            return []
        x = np.asarray(samples, np.float32).reshape(-1)
        if self._pending.size == 0:
            self._pending_t = t0
        self._pending = np.concatenate([self._pending, x])
        out: list[Utterance] = []
        while self._pending.size >= FRAME:
            frame = _Frame(self._pending[:FRAME].copy(), self._pending_t)
            self._pending = self._pending[FRAME:]
            self._pending_t += FRAME_S
            u = self._step(frame)
            if u is not None:
                out.append(u)
        return out

    def flush(self) -> list[Utterance]:
        """End any utterance in progress (e.g. when capture stops)."""
        if not self._utt:
            return []
        u = self._close(truncated=False)
        return [u] if u else []

    def _step(self, f: _Frame) -> Utterance | None:
        p = self.params
        prob = self.vad.prob(f.x)
        if not self._utt:
            f.speech = prob >= p.start_threshold
            self._pre.append(f)
            self._run = self._run + 1 if f.speech else 0
            if self._run * FRAME_S >= p.min_speech_s - 1e-9:
                frames = list(self._pre)
                first_speech = len(frames) - self._run
                keep = max(0, first_speech - round(p.preroll_s / FRAME_S))
                self._utt = frames[keep:]
                self._speech_idx = first_speech - keep
                self._pre.clear()
                self._silence = 0
            return None
        f.speech = prob >= p.end_threshold
        self._utt.append(f)
        self._silence = 0 if f.speech else self._silence + 1
        if self._silence * FRAME_S >= p.end_silence_s - 1e-9:
            return self._close(truncated=False)
        if len(self._utt) * FRAME_S >= p.max_s - 1e-9:
            return self._close(truncated=True)
        return None

    def _close(self, truncated: bool) -> Utterance | None:
        p = self.params
        frames = self._utt
        last_speech = max((i for i, f in enumerate(frames) if f.speech), default=len(frames) - 1)
        end = min(len(frames), last_speech + 1 + round(p.tail_s / FRAME_S))
        frames = frames[:end]
        speech_frames = sum(f.speech for f in frames)
        t_speech = frames[min(self._speech_idx, len(frames) - 1)].t
        self._utt = []
        self._run = self._silence = 0
        if speech_frames * FRAME_S < p.min_utterance_s:
            return None
        x = np.concatenate([f.x for f in frames])
        return Utterance(
            samples=x,
            t_start=frames[0].t,
            t_speech=t_speech,
            t_end=frames[-1].t + FRAME_S,
            truncated=truncated,
        )
