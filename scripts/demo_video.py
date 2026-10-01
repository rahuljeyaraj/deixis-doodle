"""A short demo video from the pictures the live app saved on 1 Oct (night 2 Oct, item 15).

    uv run python scripts/demo_video.py
    uv run python scripts/demo_video.py --out out/night/demo-1oct.mp4

`live_run.py` saves a small picture twice a second (`frames/`, 640x360, already turned so
the user's side of the bed is at the bottom) and logs what was heard and done
(`timeline.jsonl`). This cuts a few moments together, sped up, with captions from the
timeline: what rahul said (amber underline, `docs/video/brand.md`), what the app said or
did (blue), and the speed. A title card first, an end card last. 1280x720, 10 fps.

The clips are listed in `CLIPS` (run, start and end in seconds from the run's first
picture, speed). The pictures are 2 per second, so this is a flip book, not footage: a
reshoot list is printed at the end and goes in the night report.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 10
PAPER = (247, 250, 250)  # BGR of #FAFAF7
INK = (26, 26, 26)
AMBER = (36, 165, 245)  # #F5A524 in BGR
BLUE = (224, 127, 47)  # #2F7FE0 in BGR
RED = (77, 72, 229)  # #E5484D in BGR
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"
CREDITS = "Nemotron · local speech · MediaPipe hands · an Ender 3"


@dataclass
class Clip:
    run: str
    start: float
    end: float
    speed: float
    title: str


CLIPS = [
    Clip("20261001-225450", 25.0, 50.5, 2.5, "Point, say it, show the size"),
    Clip("20261001-225450", 54.0, 90.0, 3.5, "A bigger one: thumb and index wide apart"),
    Clip("20261001-223324", 138.5, 148.0, 1.5, "Point, “Draw a cross here”, done"),
    Clip("20261001-223324", 155.5, 177.0, 2.5, "Here ... this big"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


TextItem = tuple[str, tuple[int, int], int, tuple, bool]


def text_img(img: np.ndarray, items: list[TextItem]) -> np.ndarray:
    """Draw (text, (x, y), size, BGR colour, bold) with PIL onto a BGR image."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, xy, size, colour, bold in items:
        d.text(xy, text, font=font(size, bold), fill=(colour[2], colour[1], colour[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def text_w(text: str, size: int, bold: bool = False) -> int:
    return int(font(size, bold).getlength(text))


def card(lines: list[tuple[str, int, tuple, bool]]) -> np.ndarray:
    img = np.full((H, W, 3), PAPER, np.uint8)
    total = sum(s + 18 for _, s, _, _ in lines)
    y = (H - total) // 2
    items = []
    for text, size, colour, bold in lines:
        items.append((text, ((W - text_w(text, size, bold)) // 2, y), size, colour, bold))
        y += size + 18
    return text_img(img, items)


def captions(events: list[dict], t0: float) -> list[tuple[float, float, str, str]]:
    """(start, end, who, text) on the run's picture clock, from the timeline."""
    out = []
    for e in events:
        t = e["t"] - t0
        k = e["kind"]
        if k == "utterance" and e.get("text"):
            out.append((t - 1.0, t + 2.5, "human", f"rahul: \u201c{e['text']}\u201d"))
        elif k == "say":
            out.append((t, t + 3.0, "robot", f"app says: \u201c{e['text']}\u201d"))
        elif k == "preview":
            out.append((t, t + 3.0, "robot", "app: shows where it goes (pen up)"))
        elif k == "confirm":
            out.append((t, t + 2.5, "robot", "app: quiet, so it draws"))
        elif k == "drawn":
            size = (e.get("ink_box") or [0, 0, 0, 0])[2]
            out.append((t - 2.0, t + 2.0, "robot", f"app: drew a cross, {size:.0f} mm"))
        elif k == "span" and e.get("span", 0) > 20:
            out.append((t - 0.5, t + 1.5, "human", f"hands show {e['span']:.0f} mm"))
    return out


def run_t0(run_dir: Path) -> tuple[float, list[dict]]:
    evs = [json.loads(x) for x in (run_dir / "timeline.jsonl").read_text().splitlines()]
    head = json.loads((run_dir / "hands.jsonl").read_text().splitlines()[0])
    return float(head.get("t0", evs[0]["t"])), evs


def clip_frames(c: Clip, live: Path) -> list[np.ndarray]:
    d = live / c.run
    pics = sorted((d / "frames").glob("*.jpg"))
    at = np.array([int(p.stem) / 1000 for p in pics])
    t0, evs = run_t0(d)
    caps = captions(evs, t0)
    frames = []
    n = int((c.end - c.start) / c.speed * FPS)
    cache: dict[int, np.ndarray] = {}
    for i in range(n):
        t = c.start + i * c.speed / FPS
        j = max(0, int(np.searchsorted(at, t, side="right")) - 1)
        if j not in cache:
            cache[j] = cv2.resize(cv2.imread(str(pics[j])), (W, H), interpolation=cv2.INTER_CUBIC)
        img = cache[j].copy()
        # top: the clip's title and the speed
        cv2.rectangle(img, (0, 0), (W, 52), (0, 0, 0), -1)
        items = [(c.title, (24, 8), 30, (255, 255, 255), True)]
        sp = f"x{c.speed:g}  \u00b7  1 Oct, live on the Ender"
        items.append((sp, (W - 24 - text_w(sp, 22), 14), 22, (200, 200, 200), False))
        # bottom: up to two captions, newest last
        now = [cap for cap in caps if cap[0] <= t <= cap[1]][-2:]
        if now:
            overlay = img.copy()
            top = H - 34 - 48 * len(now)
            cv2.rectangle(overlay, (0, top), (W, H), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            for k, (_, _, who, text) in enumerate(now):
                y = top + 14 + 48 * k
                items.append((text, (40, y), 30, (255, 255, 255), False))
                colour = AMBER if who == "human" else BLUE
                cv2.rectangle(img, (24, y + 6), (31, y + 34), colour, -1)
                if who == "human":
                    cv2.line(img, (40, y + 40), (40 + text_w(text, 30), y + 40), colour, 3)
        frames.append(text_img(img, items))
    return frames


def open_writer(path: Path) -> tuple[cv2.VideoWriter, str]:
    for codec in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), FPS, (W, H))
        if w.isOpened():
            return w, codec
        w.release()
    raise RuntimeError("no MP4 encoder")


RESHOOT = [
    "Real video, not 2 pictures a second: record the screen or a phone at 30 fps.",
    "The size gesture up close: thumb and index opening along the paper, then 'this big'.",
    "The pen drawing: the 2-per-second pictures miss most of the stroke.",
    "A clean sheet: the 1 Oct sheet has 16 crosses, five of them stacked in the middle.",
    "The preview trace (pen up), now faster and corner to corner (2 Oct defaults).",
    "A hand coming in while it draws: the new banner 'Paused: hand over the paper'.",
    "A drawing that is not a cross (tree, ear and eye, garden: scenes S30 to S32).",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--live", type=Path, default=Path("out/live"))
    ap.add_argument("--out", type=Path, default=Path("out/night/demo-1oct.mp4"))
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer, codec = open_writer(a.out)
    title = card(
        [
            ("Point and Draw", 64, INK, True),
            ("tell any robot where", 40, INK, False),
            ("first live runs on the plotter, 1 Oct 2026", 28, (91, 100, 114), False),
        ]
    )
    end = card(
        [
            ("Point and Draw", 64, INK, True),
            ("point, say \u201chere\u201d and \u201cthis big\u201d, and it draws", 32, INK, False),
            (
                "Nemotron \u00b7 local speech \u00b7 MediaPipe hands \u00b7 an Ender 3",
                26,
                (91, 100, 114),
                False,
            ),
        ]
    )
    n = 0
    for _ in range(int(2.5 * FPS)):
        writer.write(title)
        n += 1
    for c in CLIPS:
        for f in clip_frames(c, a.live):
            writer.write(f)
            n += 1
    for _ in range(int(3.0 * FPS)):
        writer.write(end)
        n += 1
    writer.release()
    check = cv2.VideoCapture(str(a.out))
    ok = check.isOpened() and int(check.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    check.release()
    print(f"{a.out}: {n} frames, {n / FPS:.1f} s, {codec}, reads back: {ok}")
    print("Worth reshooting properly:")
    for r in RESHOOT:
        print(f"  - {r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
