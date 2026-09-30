"""Virtual voice. Text mode: words and times straight from the scene file (skips ASR).

The runtime hears an utterance when the real pipeline would have it: after the last word,
the segmenter's end-of-speech silence (0.6 s) plus the ASR time (0.21 s, Parakeet on the
NUC, spike R2). Speech start reaches the runtime a little after the first word (VAD).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from deixis.types import Word
from pointdraw.virtual.scenefile import Line

END_SILENCE_S = 0.6  # audio/segmenter.py default
ASR_S = 0.21  # R2: Parakeet TDT 0.6B v2, per command
VAD_S = 0.1  # speech-start detection delay

_EDGE = re.compile(r"^[^\w']+|[^\w']+$")


@dataclass
class Spoken:
    words: list[Word]
    t_start_heard: float  # VAD fires
    t_avail: float  # the transcript reaches the runtime

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def word_duration(w: str, rate: float = 1.0) -> float:
    return min(0.6, max(0.15, 0.10 + 0.06 * len(w))) / rate


def tokens(text: str) -> list[str]:
    out = []
    for raw in text.split():
        w = _EDGE.sub("", raw)
        if w:
            out.append(w)
    return out


def line_words(line: Line, shift: float = 0.0) -> list[Word]:
    """Words with times for one line; explicit times in the file win."""
    if line.words:
        out = []
        for i, item in enumerate(line.words):
            text, t0 = str(item[0]), float(item[1]) + shift
            if len(item) > 2:
                t1 = float(item[2]) + shift
            elif i + 1 < len(line.words):
                t1 = min(t0 + word_duration(text), float(line.words[i + 1][1]) + shift - 0.02)
            else:
                t1 = t0 + word_duration(text)
            out.append(Word(text, t0, t1))
        return out
    t = line.at + shift
    out = []
    for w in tokens(line.text):
        d = word_duration(w, line.rate)
        out.append(Word(w, t, t + d))
        t += d + 0.06 / line.rate
    return out


def text_mode(lines: list[Line], shift: float = 0.0) -> list[Spoken]:
    out = []
    for line in lines:
        words = line_words(line, shift)
        if not words:
            continue
        out.append(Spoken(words, words[0].t0 + VAD_S, words[-1].t1 + END_SILENCE_S + ASR_S))
    return out
