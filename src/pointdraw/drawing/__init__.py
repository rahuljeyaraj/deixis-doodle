"""Drawing library (Quick, Draw!), primitives, single-stroke text and path optimisation.

Typical use, once the resolver has a centre and size in mm:

    from pointdraw.drawing import Library, place_object, optimize
    lib = Library.load()
    tree = place_object("a tree", center=(60, 70), size=40, library=lib)
    polylines, stats = optimize(tree.polylines)
"""

from pointdraw.drawing.library import Drawing, Library, UnknownCategory
from pointdraw.drawing.optimize import PlotParams, Stats, optimize
from pointdraw.drawing.scene import Placed, compose, place_label, place_object, plan

__all__ = [
    "Drawing",
    "Library",
    "Placed",
    "PlotParams",
    "Stats",
    "UnknownCategory",
    "compose",
    "optimize",
    "place_label",
    "place_object",
    "plan",
]
