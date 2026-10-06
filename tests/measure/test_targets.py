"""Target sheets (M01, M02, M03): layouts fit the piece, and the marks are found again in a
picture of the paper wherever the piece lies (turned a little, shifted, printed at 99 %)."""

import math

import cv2
import numpy as np
import pytest

from pointdraw.measure import targets as T


@pytest.mark.parametrize("name", list(T.SHEETS))
def test_layouts_fit_the_piece(name):
    ts = T.layout(name)
    for t in ts:
        r = (t.d or 0) / 2
        assert r + 5 <= t.x <= T.PIECE_W - r - 5 and r + 5 <= t.y <= T.PIECE_H - r - 5
    if T.SHEETS[name]["kind"] == "crosses":
        assert len(ts) == 20 and T.min_spacing(ts) >= 30
    else:
        assert len(ts) == 10
        for i, a in enumerate(ts):  # printed circles at least 4 mm apart
            for b in ts[i + 1 :]:
                assert math.dist((a.x, a.y), (b.x, b.y)) > (a.d + b.d) / 2 + 4


def test_sheet_a_and_b_differ():
    a, b = T.layout("crosses-A"), T.layout("crosses-B")
    assert any(math.dist((p.x, p.y), (q.x, q.y)) > 1 for p, q in zip(a, b, strict=True))


def place(piece: np.ndarray, s: float, at=(12.0, 9.0), turn_deg=4.0, scale=0.99):
    """The piece on a 225 x 200 mm white paper picture: turned, shifted, scaled, blurred."""
    W, H = int(225 / s), int(200 / s)
    h = piece.shape[0]
    a = math.radians(turn_deg)
    pts_mm = np.array([(0, 0), (T.PIECE_W, 0), (0, T.PIECE_H)], float)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    dst_mm = pts_mm @ R.T + np.array(at)
    src_px = np.array([(x / s, h - y / s) for x, y in pts_mm], np.float32)
    dst_px = np.array([(x / s, H - y / s) for x, y in dst_mm], np.float32)
    M = cv2.getAffineTransform(src_px, dst_px)
    out = cv2.warpAffine(piece, M, (W, H), borderValue=255, flags=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    rng = np.random.default_rng(1)
    out = np.clip(out.astype(float) + rng.normal(0, 3, out.shape), 0, 255).astype(np.uint8)
    return out, R, np.array(at)


@pytest.mark.parametrize("name", ["crosses-A", "circles-1"])
def test_marks_are_found_where_the_piece_lies(name):
    s = 0.2
    piece = T.render(name, s)
    paper, R, at = place(piece, s)
    found, info = T.find_targets(paper, s, name)
    assert info["found"] == info["of"]
    assert abs(info["scale"] - 0.99) < 0.01 and abs(info["turn_deg"] - 4.0) < 0.5
    for t, f in zip(T.layout(name), found, strict=True):
        truth = R @ (t.x, t.y) + at
        assert math.dist(truth, (f.x, f.y)) < 0.5, (t.id, truth, (f.x, f.y))
        if t.d:
            assert abs(f.d - t.d * 0.99) < 0.2


def test_upside_down_piece_is_numbered_right():
    s = 0.2
    paper, R, at = place(T.render("crosses-B", s), s, at=(212.0, 192.0), turn_deg=181.0)
    found, info = T.find_targets(paper, s, "crosses-B")
    assert info["found"] == 20
    for t, f in zip(T.layout("crosses-B"), found, strict=True):
        assert math.dist(R @ (t.x, t.y) + at, (f.x, f.y)) < 0.5


def test_inside_95():
    cov = [[4.0, 0.0], [0.0, 4.0]]  # sigma 2 mm: the 95 % circle is 4.9 mm across the radius
    assert T.inside_95((4.5, 0), (0, 0), cov)
    assert not T.inside_95((5.2, 0), (0, 0), cov)
