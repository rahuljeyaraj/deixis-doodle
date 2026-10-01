"""Words for the runtime, from a background thread: the microphone, or lines typed in.

Both put events on `events` (a queue the main loop drains every frame):

- `("speech_start", t)`: the voice detector heard speech start (holds off auto-confirm).
- `("utterance", words, t)`: the words of one utterance, with times on the shared clock,
  ready at `t`.
- `("error", message, t)`: the input stopped working (the microphone was unplugged).
- `("mic_back", t)`: the microphone works again (`Listener` keeps retrying it).
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from collections.abc import Callable
from typing import Any

from deixis.types import Word
from pointdraw import clock

log = logging.getLogger(__name__)


class Listener:
    """Microphone -> voice detector -> local ASR, in one thread.

    If the microphone fails (unplugged, or no audio for 3 s), it says so once (`error`),
    then tries to open it again every `retry_s` until it works (`mic_back`). On 1 Oct a lost
    microphone stopped speech input for good.
    """

    def __init__(
        self,
        asr: Any,
        vad: Any,
        device: int | str | None = None,
        *,
        retry_s: float = 2.0,
        capture: Callable[[], Any] | None = None,
    ) -> None:
        from pointdraw.audio.segmenter import Segmenter

        self.asr = asr
        self.segmenter = Segmenter(vad)
        self.device = device
        self.retry_s = retry_s
        self._capture = capture or self._open
        self.lost = False
        self.events: queue.Queue[tuple] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="listener", daemon=True)

    def _open(self) -> Any:
        from pointdraw.audio.capture import AudioCapture

        return AudioCapture(device=self.device)

    def start(self) -> Listener:
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._capture() as cap:
                    if self.lost:
                        self.lost = False
                        log.info("microphone back")
                        self.events.put(("mic_back", clock.now()))
                    self._listen(cap)
                if self._stop.is_set():
                    return
                raise RuntimeError("the microphone stream ended")
            except Exception as e:  # noqa: BLE001 - report it once, then keep trying
                if self._stop.is_set():
                    return
                if not self.lost:
                    log.exception("microphone stopped")
                    self.lost = True
                    self.events.put(("error", f"{type(e).__name__}: {e}", clock.now()))
                self.segmenter.reset()
                self._stop.wait(self.retry_s)

    def _listen(self, cap: Any) -> None:
        seg = self.segmenter
        for samples, t0 in cap.blocks(timeout=3.0):
            if self._stop.is_set():
                return
            was = seg.in_speech
            utts = seg.push(samples, t0)
            if seg.in_speech and not was:
                self.events.put(("speech_start", t0))
            for u in utts:
                try:
                    words = self.asr.transcribe(u.samples, u.sample_rate, u.t_start)
                except Exception:  # noqa: BLE001 - one bad utterance must not stop listening
                    log.exception("speech recognition failed on one utterance")
                    continue
                self.events.put(("utterance", words, clock.now()))


class TypedInput:
    """Lines typed in the terminal, as if spoken over the last `spoken_s` seconds.

    For dry runs without a microphone. Point first, then press Enter: the words are timed
    to end when Enter is pressed, so "here" lines up with a hold made just before.
    """

    def __init__(self, spoken_s: float = 1.5) -> None:
        self.spoken_s = spoken_s
        self.events: queue.Queue[tuple] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="typed-input", daemon=True)

    def start(self) -> TypedInput:
        self._thread.start()
        return self

    def close(self) -> None:
        pass  # a daemon thread blocked on stdin; it ends with the program

    def _run(self) -> None:
        for line in sys.stdin:
            t = clock.now()
            words = typed_words(line, t, self.spoken_s)
            if words:
                self.events.put(("speech_start", words[0].t0))
                self.events.put(("utterance", words, t))


def typed_words(line: str, t_end: float, spoken_s: float = 1.5) -> list[Word]:
    """Spread a typed line's words evenly over the `spoken_s` seconds before `t_end`."""
    texts = line.split()
    if not texts:
        return []
    step = spoken_s / len(texts)
    t0 = t_end - spoken_s
    return [Word(w, t0 + i * step, t0 + (i + 1) * step) for i, w in enumerate(texts)]
