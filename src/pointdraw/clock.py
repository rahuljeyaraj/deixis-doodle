"""The one clock every timestamp uses.

Audio blocks, camera frames, ASR words and plotter events must be comparable, so all of them
are stamped with `now()`: seconds on a monotonic, high-resolution clock. Never use
time.time() for event timing, because it can jump.
"""

import time


def now() -> float:
    """Seconds on the shared monotonic clock."""
    return time.perf_counter()


def now_ms() -> int:
    """Milliseconds on the shared clock (MediaPipe VIDEO mode wants increasing int ms)."""
    return int(time.perf_counter() * 1000)
