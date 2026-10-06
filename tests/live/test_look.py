"""The window look (claim 39q): amber for you, dashed blue for the robot's guess."""

import numpy as np

from pointdraw.live import look


def blank(h=200, w=300):
    return np.full((h, w, 3), 255, np.uint8)


def has(vis, colour):
    return bool(np.all(vis == np.array(colour, np.uint8), axis=-1).any())


def test_dot_is_filled_amber_with_a_dark_edge():
    vis = blank()
    look.dot(vis, (100, 100), 9)
    assert tuple(vis[100, 100]) == look.AMBER
    assert has(vis, look.EDGE)


def test_dashed_box_has_gaps():
    vis = blank()
    look.dashed(vis, np.array([(20, 20), (280, 20), (280, 180), (20, 180)]), thick=3)
    row = vis[20, 20:280]
    blue = np.all(row == np.array(look.GUESS, np.uint8), axis=-1)
    assert blue.sum() > 80 and (~blue).sum() > 40  # dashes and gaps along the top edge
    assert np.all(row == 255, axis=-1).any()  # some gaps show the paper


def test_status_bar_text_and_said_line():
    vis = blank(300, 900)
    y = look.status(vis, "Ready: say yes to draw it", said="A kite, about four centimetres?")
    assert y > 60
    assert has(vis[:y], look.WHITE) and tuple(vis[2, 450]) == look.BAR


def test_fill_keeps_the_picture_size():
    vis = blank()
    out = look.fill(vis, [(10, 10), (60, 10), (60, 40), (10, 40)])
    assert out.shape == vis.shape and tuple(out[25, 35]) != (255, 255, 255)
