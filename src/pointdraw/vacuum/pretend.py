"""A pretend person for the vacuum app: feet on the floor and timed words, no camera, no mic.

Used by the tests and by `scripts/vacuum_pretend.py`, which runs the whole chain (the agent,
the deixis core over MCP, the floor-to-map step, the vacuum's server with `--fake`) with
nobody in the room and nothing moving.
"""

from __future__ import annotations

import math

import numpy as np

from deixis.types import Sample, Word

FPS = 30


class Person:
    """Feet on the floor, one sample per frame, with seeded noise; and timed words."""

    def __init__(self, at, t=0.0, seed=0, jitter=15.0):
        self.rng = np.random.default_rng(seed)
        self.at, self.t, self.jitter = at, t, jitter
        self.samples: list[Sample] = []
        self.lines: list[tuple[list[Word], float]] = []

    def _frame(self, p):
        n = self.rng.normal(0, self.jitter, 2) if p is not None else (0, 0)
        pt = None if p is None else (p[0] + n[0], p[1] + n[1])
        self.samples.append(Sample(round(self.t, 4), pt, pt is not None))
        self.t += 1 / FPS

    def walk(self, to, speed=1000.0):
        n = max(1, round(math.dist(self.at, to) / speed * FPS))
        a = self.at
        for k in range(n):
            f = k / n
            self._frame((a[0] + (to[0] - a[0]) * f, a[1] + (to[1] - a[1]) * f))
        self.at = to
        return self

    def stand(self, s):
        for _ in range(round(s * FPS)):
            self._frame(self.at)
        return self

    def gone(self, s):
        for _ in range(round(s * FPS)):
            self._frame(None)
        return self

    def say(self, text, lag=0.4, step=0.3):
        """Say it now (while the next frames run): words from now, heard `lag` after."""
        ws = text.split()
        words = [Word(w, self.t + i * step, self.t + (i + 1) * step) for i, w in enumerate(ws)]
        self.lines.append((words, words[-1].t1 + lag))
        return self


def play(app, person: Person) -> list[str]:
    """Feed the person's frames and words to the app; returns the app's state per frame."""
    lines = sorted(person.lines, key=lambda x: x[1])
    states = []
    for s in person.samples:
        app.on_sample(s)
        while lines and lines[0][1] <= s.t:
            words, t = lines.pop(0)
            app.on_utterance(words, t)
        app.tick(s.t)
        states.append(app.state)
    return states


def scenes() -> dict[str, Person]:
    """The three things the Wed 14 Oct gate tries, on a 3 x 2.5 m floor patch."""
    out = {}
    p = Person((0, 0)).walk((1500, 1000)).stand(0.6).say("clean here").stand(2.0)
    out["clean here"] = p.say("yes").stand(1.5).walk((2900, 2400)).stand(2.0)
    a, b = (2250.0, 1500.0), (750.0, 500.0)  # the film's 1.5 x 1 m box
    p = Person((3000, 2400)).walk(a).stand(0.5).say("clean from here").stand(1.2)
    p.walk(b).stand(0.4).say("to here").stand(1.5).say("yes").stand(1.5)
    out["clean from here to here"] = p.walk((0, 0)).stand(2.0)
    a, b = (2200.0, 1300.0), (900.0, 1000.0)  # 0.3 m across: asks to widen it
    p = Person((3000, 2400)).walk(a).stand(0.5).say("clean from here").stand(1.2)
    p.walk(b).stand(0.4).say("to here").stand(1.5).say("yes").stand(1.5)
    out["narrow strip"] = p.walk((0, 0)).stand(2.0)
    return out
