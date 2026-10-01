"""The live glue, without hardware: the bed-aware camera view, the background agent, typed words."""

import time

import numpy as np
import pytest

from deixis.geometry import Box
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.prompts import AgentContext
from pointdraw.live.agent import BackgroundAgent
from pointdraw.live.speech import typed_words
from pointdraw.live.view import BedView, PaperCamera, warp_to_paper
from pointdraw.vision.aruco import BedHomography

PARK_Y = 220.0
PAPER = Box(30.0, 40.0, 150.0, 120.0)
# A camera at 0.5 mm per pixel, upside down like the real one (bed front at the top).
H = np.array([[0.5, 0.0, 0.0], [0.0, -0.5, 220.0], [0.0, 0.0, 1.0]])


def view(bed_y=None) -> BedView:
    hom = BedHomography(H, (), 0.0, 0.0, float("nan"))
    return BedView(hom, PARK_Y, lambda: bed_y)


def test_parked_view_is_the_calibration():
    v = view(PARK_Y)
    px = np.array([[200.0, 240.0]])
    assert np.allclose(v.to_mm(px), [[100.0, 100.0]])
    assert np.allclose(v.to_px(v.to_mm(px)), px)
    assert np.allclose(view(None).to_mm(px), [[100.0, 100.0]])  # unknown counts as parked


def test_moved_bed_shifts_y_only():
    v = view(PARK_Y - 20.0)  # the head went to Y 200: the bed moved 20 mm
    px = np.array([[200.0, 240.0]])
    assert np.allclose(v.to_mm(px), [[100.0, 80.0]])
    assert np.allclose(v.to_px(np.array([[100.0, 80.0]])), px)


@pytest.mark.parametrize("bed_y", [PARK_Y, PARK_Y - 20.0])
def test_paper_image_is_top_down_user_at_bottom(bed_y):
    v = view(bed_y)
    img = np.full((480, 480, 3), 255, np.uint8)
    target = np.array([[PAPER.x + 20.0, PAPER.y + 30.0]])  # bed mm, now
    cx, cy = v.to_px(target)[0].round().astype(int)
    img[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
    s = 0.5
    out = warp_to_paper(img, v, PAPER, s)
    assert out.shape[:2] == (round(PAPER.h / s), round(PAPER.w / s))
    rows, cols = np.nonzero(out[..., 0] < 128)
    x_mm = (cols.mean() + 0.5) * s  # scene.ink's convention
    y_mm = (out.shape[0] - rows.mean() - 0.5) * s
    assert abs(x_mm - 20.0) < 1.0 and abs(y_mm - 30.0) < 1.0


def test_paper_camera_skips_frames_with_a_hand():
    cam = PaperCamera(view(PARK_Y), PAPER)
    img, mmpp = cam.paper_image(0.0)
    assert img.min() == 255 and mmpp == 0.5  # nothing seen yet: a blank sheet
    clear = np.full((480, 480, 3), 200, np.uint8)
    cam.offer(clear, 1.0, hands_over_paper=False)
    cam.offer(np.zeros_like(clear), 2.0, hands_over_paper=True)
    assert cam.t == 1.0
    img, _ = cam.paper_image(3.0)
    assert img.min() == 200


class _Slow:
    def understand(self, words, context):
        time.sleep(0.2)
        return AgentResult(calls=[], speech=" ".join(words), source="cache")


class _Broken:
    def understand(self, words, context):
        raise RuntimeError("no network")


def test_background_agent_answers_later_and_never_raises():
    ctx = AgentContext(scene_labels=[], last_label=None, previewing=None)
    a = BackgroundAgent(_Slow(), ["tree"])
    t0 = time.perf_counter()
    fut = a.submit(["hello", "there"], ctx)
    assert time.perf_counter() - t0 < 0.1 and not fut.done()
    assert fut.result(timeout=5).speech == "hello there"
    a.close()
    b = BackgroundAgent(_Broken(), ["tree"])
    res = b.submit(["draw", "a", "tree", "here"], ctx).result(timeout=5)
    assert res.source == "fallback" and res.errors == ["no network"]
    assert [c.name for c in res.calls] == ["place_drawing"]
    b.close()


def test_typed_words_end_at_enter():
    ws = typed_words("draw a tree here\n", 10.0, spoken_s=2.0)
    assert [w.text for w in ws] == ["draw", "a", "tree", "here"]
    assert ws[0].t0 == pytest.approx(8.0) and ws[-1].t1 == pytest.approx(10.0)
    assert typed_words("   ", 1.0) == []
