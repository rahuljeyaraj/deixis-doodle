"""Robot-agnostic deixis resolution: turning "here", "that one" and "this big" into places.

Nothing in this package knows about the plotter. Robot-specific code lives in `pointdraw`.
"""

from deixis.geometry import Box, Point, Polyline
from deixis.types import Hold, Word
from deixis.where import Where

__all__ = ["Box", "Hold", "Point", "Polyline", "Where", "Word"]
