"""The finger test pauses the plotter only for a hand over the bed, not one elsewhere in view."""

import numpy as np

from deixis.geometry import Box
from pointdraw.m1 import over_bed

BED = Box(0, 0, 220, 220)


def test_hand_over_the_bed_pauses():
    assert over_bed(np.array([[100.0, 100.0], [300.0, 300.0]]), BED)


def test_hand_just_outside_counts_within_the_margin():
    assert over_bed(np.array([[-15.0, 50.0]]), BED)
    assert not over_bed(np.array([[-25.0, 50.0]]), BED)


def test_hand_or_foot_far_away_does_not_pause():
    assert not over_bed(np.array([[480.0, 340.0], [520.0, 370.0]]), BED)


def test_no_mapping_counts_as_over_the_bed():
    assert over_bed(None, BED)
