"""The whole vacuum chain with a pretend person and a pretend robot: nothing moves.

    uv run python scripts/vacuum_pretend.py                 # the rule agent (no network)
    uv run python scripts/vacuum_pretend.py --ai live       # Nemotron on Token Factory (cents)

A scripted person (`pointdraw.vacuum.pretend`) stands, walks and talks: "clean here", the
film's "clean from here ... to here" (a 1.5 x 1 m box) and a narrow strip. Each goes through
the agent, the deixis core over MCP (`deixis-mcp`), the floor-to-map step, and the vacuum's
own server with `--fake` (`deixis-vacuum-mcp`, in ../deixis-vacuum), exactly as in
`vacuum_run.py` but with no camera, microphone or robot.

Prints what the app said and what the pretend robot was sent, and writes `timeline.jsonl` per
scene and `floor.png` (each scene's box on the floor) to `out/vacuum_pretend/<time>/`.
Run it before the first hardware session: if this fails, the real run will too.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from deixis.mcp import DeixisClient
from pointdraw.live import look
from pointdraw.vacuum.agent import RuleAgent, VacuumAgent
from pointdraw.vacuum.app import VacuumApp
from pointdraw.vacuum.calibrate import FloorToMap
from pointdraw.vacuum.link import LinkError, VacuumLink, default_command
from pointdraw.vacuum.pretend import play, scenes
from pointdraw.vacuum.profile import vacuum_profile

AREA = (0.0, 0.0, 3000.0, 2500.0)  # the floor patch, mm
EXPECT = {
    "clean here": ["Clean about one metre square here?", "OK. Step out, and I'll clean it."],
    "clean from here to here": ["Clean from there to here?", "OK. Step out, and I'll clean it."],
    "narrow strip": [
        "That's a narrow strip. Clean half a metre wide?",
        "OK. Step out, and I'll clean it.",
    ],
}


def to_map(p):
    """A pretend calibration: the robot's map turned 30 degrees and shifted from the floor."""
    a = math.radians(30)
    return (1850 + math.cos(a) * p[0] + math.sin(a) * p[1],
            -950 + math.sin(a) * p[0] - math.cos(a) * p[1])  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ai", choices=["rules", "live"], default="rules")
    ap.add_argument("--core", choices=["mcp", "direct"], default="mcp")
    ap.add_argument("--out", type=Path, default=Path("out/vacuum_pretend"))
    a = ap.parse_args(argv)
    out = a.out / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    profile = vacuum_profile(AREA)
    core = DeixisClient(profile, mode=a.core)
    try:
        link = VacuumLink(default_command(fake=True), start_timeout_s=60)
    except (LinkError, OSError) as e:
        print(f"FAIL: the pretend robot's server did not start: {e}")
        print("Is ../deixis-vacuum/.venv installed? (cd ../deixis-vacuum; uv sync)")
        core.close()
        return 1
    floor = [(200, 150), (2600, 300), (2400, 2200), (500, 2000)]
    f2m = FloorToMap.fit([(p, to_map(p)) for p in floor], core)
    agent = VacuumAgent(profile, core.tools_for_agent()) if a.ai == "live" else RuleAgent(profile)
    ok = True
    boxes = {}
    try:
        for name, person in scenes().items():
            said: list[str] = []
            app = VacuumApp(agent, core, link, f2m, profile, say=said.append)
            play(app, person)
            sent = app.log.of("clean_area")
            reply = sent[0]["reply"] if sent else None
            good = said[:2] == EXPECT[name] and bool(reply and reply.get("ok"))
            ok &= good
            print(f"\n== {name}: {'PASS' if good else 'FAIL'}")
            for s in said:
                print(f"   deixis: {s}")
            if sent:
                print(f"   floor box {sent[0]['floor_box']} -> robot map box {sent[0]['map_box']}")
                print(f"   pretend robot: {json.dumps(reply)}")
                boxes[name] = sent[0]["floor_box"]
            with (out / f"{name.replace(' ', '-')}.jsonl").open("w", encoding="utf-8") as f:
                for e in app.log.events:
                    print(json.dumps(e, default=str), file=f)
        print(f"\nover MCP: {core.over_mcp}; status: {json.dumps(link.status())}")
    finally:
        link.close()
        core.close()
    cv2.imwrite(str(out / "floor.png"), floor_picture(boxes))
    print(f"\n{'ALL PASS' if ok else 'SOMETHING FAILED'}; logs in {out}")
    return 0 if ok else 1


def floor_picture(boxes: dict[str, list[float]], s: float = 5.0) -> np.ndarray:
    """The floor patch from above (near edge at the bottom), each scene's box dashed blue."""
    W, H = int(AREA[2] / s), int(AREA[3] / s)
    img = np.full((H + 40, W, 3), 245, np.uint8)
    for gx in range(0, int(AREA[2]) + 1, 500):
        cv2.line(img, (int(gx / s), 0), (int(gx / s), H), (210, 210, 210), 1)
    for gy in range(0, int(AREA[3]) + 1, 500):
        cv2.line(img, (0, int(H - gy / s)), (W, int(H - gy / s)), (210, 210, 210), 1)
    for i, (name, (x, y, w, h)) in enumerate(boxes.items()):
        pts = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]) / s
        pts[:, 1] = H - pts[:, 1]
        look.dashed(img, pts)
        cv2.putText(img, f"{i + 1}: {name}", (6, 18 + 18 * i), 0, 0.5, look.EDGE, 1)
        cv2.putText(
            img, str(i + 1), (int(pts[3][0]) + 4, int(pts[3][1]) + 16), 0, 0.5, look.EDGE, 1
        )
    cv2.putText(img, "3 x 2.5 m floor patch, 0.5 m grid", (6, H + 28), 0, 0.5, look.EDGE, 1)
    return img


if __name__ == "__main__":
    sys.exit(main())
