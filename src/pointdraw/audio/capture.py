"""Microphone capture: 16 kHz mono float32 blocks stamped on the shared clock.

    with AudioCapture() as cap:
        for samples, t0 in cap.blocks():   # t0 = clock.now() time of samples[0]
            ...

Time stamping: PortAudio reports, in each callback, when the block's first sample hit the ADC
(`inputBufferAdcTime`) and the stream's current time, both on PortAudio's own clock. We map that
onto `clock.now()` by reading `clock.now()` in the same callback. Some Windows host APIs (MME in
particular) report 0 for the ADC time; then we fall back to "now minus the block length minus
the stream's input latency". `BlockClock` then keeps stamps continuous (sample count / rate)
and only re-anchors on a jump of more than 50 ms, so the stamps don't jitter with callback
scheduling.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from pointdraw import clock
from pointdraw.audio.wavio import SAMPLE_RATE, resample_linear

log = logging.getLogger(__name__)


@dataclass
class BlockClock:
    """Maps audio blocks onto the shared clock, smoothly."""

    sample_rate: int = SAMPLE_RATE
    resync_s: float = 0.05  # re-anchor if the raw estimate drifts further than this
    alpha: float = 0.02  # slow drift tracking between re-anchors
    anchor: float | None = None  # clock time of sample 0
    samples: int = 0
    resyncs: int = 0

    @staticmethod
    def raw_estimate(
        n_frames: int,
        sample_rate: int,
        now: float,
        adc_time: float | None = None,
        current_time: float | None = None,
        input_latency: float = 0.0,
    ) -> float:
        """Best single-callback estimate of the block's first-sample time on the shared clock."""
        if (
            adc_time
            and current_time
            and 0 < adc_time <= current_time
            and current_time - adc_time < 2
        ):
            return now - (current_time - adc_time)
        return now - n_frames / sample_rate - input_latency

    def stamp(self, n_frames: int, raw: float) -> float:
        """Stamp for a block of `n_frames` whose raw first-sample estimate is `raw`."""
        if self.anchor is None:
            self.anchor = raw
        predicted = self.anchor + self.samples / self.sample_rate
        err = raw - predicted
        if abs(err) > self.resync_s:
            self.anchor += err
            self.resyncs += 1
        else:
            self.anchor += self.alpha * err
        t = self.anchor + self.samples / self.sample_rate
        self.samples += n_frames
        return t


def list_devices(kind: str = "input") -> list[dict]:
    """The sound devices of `kind` ("input" or "output") as dicts: index, name, host API,
    channels, default rate."""
    import sounddevice as sd

    field = f"max_{kind}_channels"
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d[field] > 0:
            out.append(
                {
                    "index": i,
                    "name": d["name"],
                    "hostapi": apis[d["hostapi"]]["name"],
                    "channels": d[field],
                    "default_samplerate": d["default_samplerate"],
                }
            )
    return out


def list_input_devices() -> list[dict]:
    """Input devices as dicts: index, name, host API, channels, default rate."""
    return list_devices("input")


def list_output_devices() -> list[dict]:
    """Output devices (headphones, speakers) in the same shape as the inputs."""
    return list_devices("output")


def default_input_device() -> int | None:
    """Index of the default input device, or None if there is no microphone."""
    import sounddevice as sd

    try:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            ins = list_input_devices()
            return ins[0]["index"] if ins else None
        return int(idx)
    except Exception:
        return None


def default_output_device() -> int | None:
    """Index of the default output device (where Windows is playing), or None."""
    import sounddevice as sd

    try:
        idx = sd.default.device[1]
        if idx is None or idx < 0:
            outs = list_output_devices()
            return outs[0]["index"] if outs else None
        return int(idx)
    except Exception:
        return None


def wasapi_twin(default_index: int, devices: list[dict]) -> int | None:
    """The WASAPI entry for the same microphone as `default_index`, if the default is MME.

    PortAudio's default on Windows is the MME device, which reports no ADC timestamps (so
    stamps fall back to callback time). WASAPI reports them. MME truncates names to 31
    characters, so match on the prefix.
    """
    d = next((x for x in devices if x["index"] == default_index), None)
    if d is None or d["hostapi"] != "MME":
        return None
    prefix = d["name"][:25]
    for x in devices:
        if x["hostapi"] == "Windows WASAPI" and x["name"].startswith(prefix):
            return x["index"]
    return None


def find_device(spec: int | str, kind: str = "input") -> dict:
    """The sound device a `--mic` or `--speaker` value names: an index, or any part of a
    name (case is ignored, WASAPI entries first, so an input's stamps carry ADC times).

    Raises ValueError naming every device of that kind when nothing matches. rahul,
    7 Oct 2026: the default microphone was the webcam's, not the one on the desk, and
    there was no way to see which it had taken.
    """
    devices = list_devices(kind)
    what = "microphone" if kind == "input" else "speaker"
    if not devices:
        raise ValueError(f"no {what} found")
    text = str(spec).strip()
    if not text:
        raise ValueError(f"needs an index or part of a name.\n{device_table(kind, devices)}")
    if text.lstrip("-").isdigit():
        i = int(text)
        for d in devices:
            if d["index"] == i:
                return d
        raise ValueError(f"no {kind} device {i}.\n{device_table(kind, devices)}")
    hits = [d for d in devices if text.lower() in d["name"].lower()]
    if not hits:
        raise ValueError(f"no {what} whose name holds {text!r}.\n{device_table(kind, devices)}")
    hits.sort(key=lambda d: d["hostapi"] != "Windows WASAPI")
    return hits[0]


def find_input_device(spec: int | str) -> dict:
    """The microphone a `--mic` value names (see `find_device`)."""
    return find_device(spec, "input")


def find_output_device(spec: int | str) -> dict:
    """The speaker a `--speaker` value names (see `find_device`)."""
    return find_device(spec, "output")


def device_table(kind: str = "input", devices: list[dict] | None = None) -> str:
    """The sound devices as lines for a terminal, `index  name  (host API)`."""
    devices = list_devices(kind) if devices is None else devices
    default = default_input_device() if kind == "input" else default_output_device()
    rows = []
    for d in devices:
        mark = " <- Windows' default" if d["index"] == default else ""
        rows.append(f"  {d['index']:>3}  {d['name']}  ({d['hostapi']}){mark}")
    return "\n".join(rows) or "  (none)"


def input_device_table(devices: list[dict] | None = None) -> str:
    """The microphones as lines for a terminal."""
    return device_table("input", devices)


def preferred_input_device() -> int | None:
    """Default microphone, switched to its WASAPI entry on Windows when there is one."""
    idx = default_input_device()
    if idx is None:
        return None
    try:
        twin = wasapi_twin(idx, list_input_devices())
    except Exception:
        twin = None
    return twin if twin is not None else idx


class AudioCapture:
    """Background sounddevice input stream delivering (samples, t0) blocks through a queue."""

    def __init__(
        self,
        device: int | str | None = None,
        block_s: float = 0.032,
        queue_blocks: int = 500,
    ) -> None:
        self.device = device
        self.block_s = block_s
        self.q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=queue_blocks)
        self.clock = BlockClock()
        self.native_rate = SAMPLE_RATE
        self.overflows = 0
        self.dropped = 0
        self._stream = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:
        now = clock.now()
        if status:
            self.overflows += bool(status.input_overflow)
        adc = getattr(time_info, "inputBufferAdcTime", None)
        cur = getattr(time_info, "currentTime", None)
        latency = float(self._stream.latency) if self._stream is not None else 0.0
        raw = BlockClock.raw_estimate(frames, self.native_rate, now, adc, cur, latency)
        x = indata.mean(axis=1) if indata.ndim == 2 and indata.shape[1] > 1 else indata.reshape(-1)
        x = resample_linear(np.asarray(x, np.float32), self.native_rate, SAMPLE_RATE)
        with self._lock:
            self.clock.sample_rate = SAMPLE_RATE
            t0 = self.clock.stamp(len(x), raw)
        try:
            self.q.put_nowait((x, t0))
        except queue.Full:
            self.dropped += 1

    def start(self) -> AudioCapture:
        import sounddevice as sd

        dev = self.device if self.device is not None else preferred_input_device()
        if dev is None:
            raise RuntimeError("no microphone found")
        info = sd.query_devices(dev, "input")
        for rate in (SAMPLE_RATE, int(info["default_samplerate"])):
            try:
                self.native_rate = rate
                self._stream = sd.InputStream(
                    device=dev,
                    channels=1,
                    samplerate=rate,
                    dtype="float32",
                    blocksize=int(self.block_s * rate),
                    callback=self._callback,
                )
                break
            except sd.PortAudioError as e:
                log.info("device %s refused %d Hz (%s)", dev, rate, e)
                self._stream = None
        if self._stream is None:
            raise RuntimeError(f"could not open input device {dev}")
        self._stream.start()
        log.info("capturing from %s at %d Hz", info["name"], self.native_rate)
        return self

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> AudioCapture:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def blocks(self, timeout: float = 1.0) -> Iterator[tuple[np.ndarray, float]]:
        """Yield blocks until the stream stops; raises if no audio arrives within `timeout`."""
        while self._stream is not None:
            try:
                yield self.q.get(timeout=timeout)
            except queue.Empty:
                if self._stream is None:
                    return
                raise TimeoutError("no audio from the microphone") from None
