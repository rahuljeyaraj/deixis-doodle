"""The vacuum's `deixis` profile: its floor frame, feet as the pointer, 1 m boxes, questions.

The file `profiles/vacuum-floor.json` is the real one (a copy lives in the core's examples).
Its area is an example; the app sets the real one from the floor layout at start.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from deixis.profile import Profile, load_profile

PROFILE_PATH = Path(__file__).resolve().parents[1] / "profiles" / "vacuum-floor.json"


@lru_cache(maxsize=1)
def _base() -> Profile:
    return load_profile(PROFILE_PATH)


def vacuum_profile(area_mm: tuple[float, float, float, float] | None = None) -> Profile:
    """The vacuum's profile, with the floor area (x, y, w, h) the markers span, if given."""
    p = _base()
    return p if area_mm is None else p.model_copy(update={"area_mm": tuple(area_mm)})
