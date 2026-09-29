"""Voice activity detection on 32 ms frames (512 samples at 16 kHz).

Default: Silero VAD run directly with onnxruntime, model at `.cache/models/silero_vad.onnx`
(fetched by `scripts/asr_fetch_models.py`; MIT licence, https://github.com/snakers4/silero-vad).
Both ONNX layouts are handled: the sherpa-onnx release copy (inputs x, h, c) and the official
v5 file (inputs input, state, sr, with a 64-sample context).

Fallback, if the model or onnxruntime is missing: `EnergyVad`, an adaptive energy threshold
over a tracked noise floor. It is fine in a quiet room but triggers on knocks and the plotter's
motors, so use it only to keep working, and prefer Silero for demos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from pointdraw.audio.models import SILERO_VAD_PATH

log = logging.getLogger(__name__)

FRAME = 512  # samples per VAD frame at 16 kHz (32 ms)
SAMPLE_RATE = 16000


class Vad(Protocol):
    def prob(self, frame: np.ndarray) -> float:
        """Speech probability for one frame of FRAME float32 samples."""
        ...

    def reset(self) -> None: ...


class SileroVad:
    """Silero VAD through onnxruntime (4 threads max, 1 is plenty)."""

    def __init__(self, path: str | Path = SILERO_VAD_PATH, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        names = {i.name for i in self._sess.get_inputs()}
        self._v5 = "state" in names
        self.reset()

    def reset(self) -> None:
        if self._v5:
            self._state = np.zeros((2, 1, 128), np.float32)
            self._context = np.zeros(64, np.float32)
        else:
            self._h = np.zeros((2, 1, 64), np.float32)
            self._c = np.zeros((2, 1, 64), np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, np.float32).reshape(-1)
        if x.shape[0] != FRAME:
            raise ValueError(f"VAD frame must be {FRAME} samples, got {x.shape[0]}")
        if self._v5:
            inp = np.concatenate([self._context, x])[None, :]
            out, self._state = self._sess.run(
                None, {"input": inp, "state": self._state, "sr": np.array(SAMPLE_RATE, np.int64)}
            )
            self._context = x[-64:]
        else:
            out, self._h, self._c = self._sess.run(
                None, {"x": x[None, :], "h": self._h, "c": self._c}
            )
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVad:
    """Energy above an adaptive noise floor, mapped to a pseudo-probability."""

    def __init__(self, margin_db: float = 12.0, floor_db: float = -60.0, adapt: float = 0.05):
        self.margin_db = margin_db
        self.initial_floor_db = floor_db
        self.adapt = adapt
        self.reset()

    def reset(self) -> None:
        self.noise_db = self.initial_floor_db

    def prob(self, frame: np.ndarray) -> float:
        e = 10 * np.log10(float(np.mean(np.square(frame, dtype=np.float64))) + 1e-12)
        above = e - self.noise_db
        if above < self.margin_db / 2:  # track the floor only on quiet frames
            self.noise_db += self.adapt * (e - self.noise_db)
        # 0.5 at the margin, saturating either side.
        return float(1 / (1 + np.exp(-(above - self.margin_db) / 3)))


def make_vad(prefer: str = "silero") -> Vad:
    """Silero if its model loads, else the energy fallback (with a warning)."""
    if prefer == "silero":
        try:
            if not Path(SILERO_VAD_PATH).exists():
                raise FileNotFoundError(f"{SILERO_VAD_PATH} (run scripts/asr_fetch_models.py)")
            return SileroVad()
        except Exception as e:  # missing model or onnxruntime
            log.warning("Silero VAD unavailable (%s); using the energy VAD", e)
    return EnergyVad()
