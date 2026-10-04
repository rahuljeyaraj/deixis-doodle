"""The plotter's `deixis` profile: its frame, pointer, timing, sizes, questions and actions.

The file `profiles/plotter-a5.json` is the real one (a copy lives in the core's examples).
The drawable area in it is an example: the runtime sets the real one from the machine
config at start (`plotter_profile(drawable)`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from deixis.geometry import Box
from deixis.profile import Profile, load_profile

PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "plotter-a5.json"


@lru_cache(maxsize=1)
def _base() -> Profile:
    return load_profile(PROFILE_PATH)


def plotter_profile(drawable: Box | None = None) -> Profile:
    """The plotter's profile, with the drawable area (paper frame) if given."""
    p = _base()
    if drawable is None:
        return p
    return p.model_copy(update={"area_mm": (drawable.x, drawable.y, drawable.w, drawable.h)})
