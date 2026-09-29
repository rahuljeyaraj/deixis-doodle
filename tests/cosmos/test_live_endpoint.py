"""Live smoke test against a running Cosmos endpoint. Skipped by default (network marker).

Run with: uv run pytest -m network tests/cosmos/test_live_endpoint.py
Needs COSMOS_BASE_URL and COSMOS_API_KEY (or COSMOS_ENDPOINT_TOKEN) in .env.
"""

import numpy as np
import pytest

from pointdraw.config import get_settings
from pointdraw.cosmos.client import CosmosClient
from pointdraw.cosmos.scoring import best_iou
from pointdraw.scene.ink import box_mm_to_px
from pointdraw.scene.synth import INKS, make_mark, render_sheet

pytestmark = pytest.mark.network


def test_find_and_label_a_circle():
    if not get_settings().cosmos_base_url:
        pytest.skip("COSMOS_BASE_URL not set")
    rng = np.random.default_rng(0)
    ink = next(i for i in INKS if i[0] == "black pen")
    mark = make_mark("circle", (70.0, 80.0), 45.0, rng, ink=ink)
    sheet = render_sheet([mark], rng=rng)
    client = CosmosClient(think=False)
    found = client.find(sheet.image, "circle")
    truth = box_mm_to_px(mark.box_mm, sheet.image.shape[0], sheet.mm_per_px)
    assert best_iou(found.detections, truth) > 0.5, found.reply.raw
    label = client.label_crop(sheet.image)
    assert label.label, label.reply.raw
