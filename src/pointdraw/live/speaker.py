"""The app speaks its replies ("Where should it go?", "Stopped.") in Windows' own voice.

`Runtime.say` only logs. `live_run.py --speak` hands each logged reply to a `Speaker`, which
speaks them one at a time in a background thread, so camera frames keep flowing. The voice is
System.Speech (SAPI), as in `virtual.tts`: each line is made once by PowerShell and cached.

The microphone must not hear the robot: while a reply plays, and for `after_s` (0.3 s) after,
`muted()` is True, and the `Listener` mutes its `Segmenter` (dropping any utterance in
progress) from its own thread. With `play=False` the speech is written to WAV files instead
of played (no sound; the mute still lasts as long as the speech would).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pointdraw import clock

log = logging.getLogger(__name__)
SR = 16000


def windows_voice(text: str) -> np.ndarray:
    """16 kHz float32 samples of `text` in Windows' voice (cached in .cache/virtual/tts/)."""
    from pointdraw.virtual.tts import speak

    return speak([(text, 0)])[0]


class Speaker:
    def __init__(
        self,
        synth: Callable[[str], np.ndarray] = windows_voice,
        *,
        play: bool = True,
        wav_dir: str | Path | None = None,
        after_s: float = 0.3,
        now: Callable[[], float] = clock.now,
    ) -> None:
        self.synth = synth
        self.play = play
        self.wav_dir = None if wav_dir is None else Path(wav_dir)
        self.after_s = after_s
        self.now = now
        self.spoken: list[tuple[str, float, float]] = []  # text, start, end (on `now`)
        self._q: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._speaking = False
        self._until = -math.inf
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if text.strip():
            self._q.put(text)

    def muted(self) -> bool:
        """True while a reply plays and for `after_s` after it."""
        with self._lock:
            return self._speaking or self.now() < self._until

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)
        self._thread.join(5)

    def _run(self) -> None:
        n = 0
        while True:
            text = self._q.get()
            if text is None or self._stop.is_set():
                return
            try:
                x = np.asarray(self.synth(text), np.float32).reshape(-1)
            except Exception:  # noqa: BLE001 - a voice failure must not stop the app
                log.exception("could not make speech for %r", text)
                continue
            n += 1
            if self.wav_dir is not None:
                from pointdraw.audio.wavio import write_wav

                slug = "".join(c if c.isalnum() else "-" for c in text.lower())[:40]
                write_wav(self.wav_dir / f"{n:03d}-{slug}.wav", x, SR)
            with self._lock:
                self._speaking = True
            t0 = self.now()
            try:
                if self.play:
                    import sounddevice as sd

                    sd.play(x, SR)
                    sd.wait()
                else:
                    self._stop.wait(len(x) / SR)  # as long as it would take to say
            except Exception:  # noqa: BLE001
                log.exception("could not play %r", text)
            finally:
                t1 = self.now()
                with self._lock:
                    self._speaking = False
                    self._until = t1 + self.after_s
                self.spoken.append((text, t0, t1))
