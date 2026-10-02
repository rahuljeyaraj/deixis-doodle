"""Score a live run for the M2 exit test: did each drawing land where and as big as shown?

    uv run python scripts/m2_score.py out/live/<run>
    uv run python scripts/m2_score.py out/live/<run> --json out/m2/s.json --crops out/m2/c.png

M2 (plan 5.0): "draw a cat here, this big", 8 of 10 correct, centre within 5 mm, size
within ±20 %, median time from the end of speech to the preview 3 s or less, no keyboard.

For each drawing in the run's `timeline.jsonl` (a `drawn` event):
- **Shown place:** the pointing hold the app used (the item's `where`, paper mm).
- **Shown size:** the size the app measured (`resolve` with `size`: a pinch, two hands or a
  spoken preset). Drawings keep their aspect, longest side = size, centred on the point
  (`drawing/place.py`).
- **Real drawing:** the ink that appeared between a parked photo before the preview and one
  after the drawing, warped top-down with the fixed calibration, light normalised, and
  subtracted (as `docs/evidence/m1-2026-10-02/tools/ink.py` did for M1). Only ink near the
  planned box counts. Centre = the centre of the ink's box; size = its longest side.
- **Latency:** end of the last word said before the preview, to the preview starting
  (the plotter's `job_start` for it; the runtime's `preview` event in older runs).

Photos: `live_run.py` saves one full-size photo each time the plotter rests at park with no
hand over the paper (`photo` events). Older runs (1 Oct) have none; then the small window
pictures in `frames/` are used (640 x 360, about 1 mm a pixel), picked when `hands.jsonl` says
the paper was clear and the bed was parked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from deixis.geometry import Box
from pointdraw.controller.config import DEFAULT_CONFIG_PATH, load_config
from pointdraw.live.view import BedView, warp_to_paper
from pointdraw.vision.aruco import DEFAULT_CALIBRATION_PATH, BedCalibration, fixed_bed_homography
from pointdraw.vision.lens import DEFAULT_LENS_PATH, LensModel

CENTRE_MM = 5.0  # M2 limits
SIZE_FRAC = 0.20
LATENCY_S = 3.0
MM_PER_PX = 0.25  # the top-down picture the ink is found in
PARK_S = 8.0  # longest park after a drawing (the head travels at 50 mm/s, pen at clear_z)
CLEAR_S = 1.0  # the paper must be clear this long around a frame used as a photo


@dataclass
class Command:
    id: str
    label: str
    preview_t: float  # the last preview shown before it was drawn
    drawn_t: float
    shown_xy: list[float] | None  # the hold the app used, paper mm
    planned_xy: list[float]  # where the drawing was centred (moved if it did not fit)
    planned_box: list[float]  # x, y, w, h, paper mm
    shown_size: float | None
    size_kind: str | None
    speech_end_t: float | None
    said: str
    preview_start_t: float  # the plotter's job_start, else the preview event
    asks: list[str] = field(default_factory=list)
    before: str | None = None
    after: str | None = None
    ink_xy: list[float] | None = None
    ink_box: list[float] | None = None
    ink_size: float | None = None
    ink_px: int = 0

    @property
    def latency_s(self) -> float | None:
        return None if self.speech_end_t is None else self.preview_start_t - self.speech_end_t

    @property
    def centre_err(self) -> float | None:
        if self.ink_xy is None or self.shown_xy is None:
            return None
        return float(np.hypot(*(np.subtract(self.ink_xy, self.shown_xy))))

    @property
    def plot_err(self) -> float | None:
        """Ink centre to the planned centre: the plotter and camera alone."""
        if self.ink_xy is None:
            return None
        return float(np.hypot(*(np.subtract(self.ink_xy, self.planned_xy))))

    @property
    def size_err(self) -> float | None:
        if self.ink_size is None or not self.shown_size:
            return None
        return self.ink_size / self.shown_size - 1.0

    @property
    def passed(self) -> bool:
        c, s = self.centre_err, self.size_err
        return c is not None and s is not None and c <= CENTRE_MM and abs(s) <= SIZE_FRAC

    def row(self) -> dict:
        d = asdict(self)
        for k in ("latency_s", "centre_err", "plot_err", "size_err", "passed"):
            d[k] = getattr(self, k)
        return d


# ---------------------------------------------------------------- timeline


def load_timeline(run: Path) -> list[dict]:
    out = []
    for line in (run / "timeline.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _kind(e: dict) -> dict:
    """Live runs log plotter events as kind "plotter", event "job_start"; the virtual world
    as kind "plotter.job_start". Use the second."""
    if e["kind"] == "plotter" and "event" in e:
        return {**e, "kind": f"plotter.{e['event']}"}
    return e


def commands(events: list[dict]) -> list[Command]:
    """One `Command` per drawing that was drawn."""
    ordered = sorted(enumerate(events), key=lambda p: (p[1]["t"], p[0]))
    events = [_kind(e) for _, e in ordered]
    out: list[Command] = []
    for i, e in enumerate(events):
        if e["kind"] != "drawn":
            continue
        oid = e["id"]
        items = [x for x in events[:i] if x["kind"] == "item" and x["id"] == oid]
        previews = [x for x in events[:i] if x["kind"] == "preview" and oid in x.get("items", [])]
        if not items or not previews:
            continue
        last = items[-1]
        where = next((x["where"] for x in reversed(items) if x.get("where")), None)
        size, kind = _size(events, items, last)
        pv = previews[-1]
        start = _preview_start(events, pv)
        utt = next(
            (
                x
                for x in reversed(events)
                if x["kind"] == "utterance" and x.get("words") and x["t"] <= pv["t"]
            ),
            None,
        )
        since = out[-1].drawn_t if out else -np.inf  # questions since the last drawing
        asks = [x["question"] for x in events if x["kind"] == "ask" and since < x["t"] <= pv["t"]]
        out.append(
            Command(
                id=oid,
                label=e.get("label") or last.get("label") or "",
                preview_t=pv["t"],
                drawn_t=e["t"],
                shown_xy=where["mean_mm"] if where else None,
                planned_xy=last["center"],
                planned_box=e.get("planned_box") or last["box"],
                shown_size=size,
                size_kind=kind,
                speech_end_t=utt["words"][-1][2] if utt else None,
                said=utt["text"] if utt else "",
                preview_start_t=start,
                asks=asks,
            )
        )
    return out


def _size(events: list[dict], items: list[dict], last: dict) -> tuple[float | None, str | None]:
    """The size the app measured for this item: a `resolve` with `size` logged with one of
    its `item` events (same time). Several items at once: the one whose box fits."""
    times = {x["t"] for x in items}
    found = None
    for x in events:
        if x["kind"] == "resolve" and "size" in x and x["t"] in times:
            same_t = [y for y in events if y["kind"] == "item" and y["t"] == x["t"]]
            if len(same_t) > 1:
                best = min(same_t, key=lambda y: abs(max(y["box"][2:]) - x["size"]))
                if best["id"] != last["id"]:
                    continue
            found = (x["size"], x.get("size_kind"))
    return found or (None, None)


def _preview_start(events: list[dict], pv: dict) -> float:
    """When the plotter began the preview: the first preview job_start after the event."""
    for x in events:
        if x["t"] < pv["t"] - 1e-6:
            continue
        if x["kind"] == "plotter.job_start" and x.get("label") == "preview":
            if x["t"] - pv["t"] < 10.0:
                return x["t"]
            break
    return pv["t"]


# ---------------------------------------------------------------- photos


class Photos:
    """Parked photos of the bed: full size from `photo` events, else the small frames."""

    def __init__(self, run: Path, events: list[dict], paper: Box) -> None:
        self.run = run
        self.paper = paper
        self.shots = [(x["t"], run / x["file"]) for x in events if x["kind"] == "photo"]
        self.frames: list[tuple[float, Path]] = []
        self.clear: list[tuple[float, bool]] = []
        if not self.shots and (run / "frames").exists():
            self._load_frames(events)

    @property
    def small(self) -> bool:
        return not self.shots

    def _load_frames(self, events: list[dict]) -> None:
        lines = (self.run / "hands.jsonl").read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0]) if lines else {}
        if "t0" in head:
            t0 = head["t0"]
        else:  # 1 Oct: the first frame's time is the first interlock event's
            t0 = min(x["t"] for x in events if x["kind"] == "interlock")
        for line in lines:
            r = json.loads(line)
            if "hands" in r:
                ok = r.get("clear") is not False and not _hand_on_paper(r, self.paper)
                self.clear.append((t0 + r["t"], ok))
        for p in sorted((self.run / "frames").glob("*.jpg")):
            self.frames.append((t0 + int(p.stem) / 1000.0, p))

    def _clear_around(self, t: float) -> bool:
        near = [c for tt, c in self.clear if t - CLEAR_S <= tt <= t + CLEAR_S]
        return bool(near) and all(near)

    def before(self, cmd: Command, prev_drawn_t: float | None) -> Path | None:
        lo = -np.inf if prev_drawn_t is None else prev_drawn_t
        if self.shots:
            c = [p for t, p in self.shots if lo <= t < cmd.preview_t]
            return c[-1] if c else None
        lo = -np.inf if prev_drawn_t is None else prev_drawn_t + PARK_S
        c = [p for t, p in self.frames if lo <= t < cmd.preview_t and self._clear_around(t)]
        return c[-1] if c else None

    def after(self, cmd: Command, next_preview_t: float | None) -> Path | None:
        hi = np.inf if next_preview_t is None else next_preview_t
        if self.shots:
            c = [p for t, p in self.shots if cmd.drawn_t <= t < hi]
            return c[0] if c else None
        lo = cmd.drawn_t + PARK_S
        c = [p for t, p in self.frames if lo <= t < hi and self._clear_around(t)]
        return c[-1] if c else None  # the latest: most sure the park is over

    def image(self, path: Path, size: tuple[int, int]) -> np.ndarray:
        img = cv2.imread(str(path))
        if self.small:  # window pictures are turned round and shrunk (live_run.py)
            img = cv2.resize(cv2.rotate(img, cv2.ROTATE_180), size, interpolation=cv2.INTER_CUBIC)
        return img


def _hand_on_paper(r: dict, paper: Box) -> bool:
    """A hand whose index tip is over the paper, or within 40 mm of it (the rest of the hand
    trails behind the tip). Paper mm."""
    for h in r["hands"]:
        x, y = h["tip"]
        if -40 <= x <= paper.w + 40 and -40 <= y <= paper.h + 40:
            return True
    return False


# ---------------------------------------------------------------- ink


def _norm(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    return cv2.GaussianBlur(g / np.maximum(bg, 1) * 200, (5, 5), 0)


def ink_mask(before: np.ndarray, after: np.ndarray, small: bool = False) -> np.ndarray:
    """Pixels darker in `after` than in `before` (top-down paper pictures). The threshold is
    6 noise sigmas (at least 3): on 1 Oct's small pictures a line is 5 to 17 above the paper,
    the paper's noise about 0.5. Full-size pictures lose lone pixels; small ones keep them,
    as a line there is one pixel wide."""
    n = _norm(before) - _norm(after)
    sigma = 1.4826 * float(np.median(np.abs(n - np.median(n))))
    m = (n > max(3.0, 6.0 * sigma)).astype(np.uint8)
    if small:
        return m
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def find_ink(
    mask: np.ndarray, box: list[float], s: float, min_px: int
) -> tuple[list[float], int] | None:
    """The ink near a planned box (paper mm): the box of the blobs that touch the box grown
    by a fifth (and 4 mm). Row 0 is the far edge. Returns (x, y, w, h) in paper mm."""
    H, W = mask.shape
    x, y, w, h = box
    g = 0.2 * max(w, h) + 4.0
    c0, c1 = max(0, int((x - g) / s)), min(W, int((x + w + g) / s) + 1)
    r0, r1 = max(0, int(H - (y + h + g) / s)), min(H, int(H - (y - g) / s) + 1)
    win = np.zeros_like(mask)
    win[r0:r1, c0:c1] = mask[r0:r1, c0:c1]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
    # Specks of noise near the drawing would stretch its box: a piece counts if it is at
    # least 3 mm of one-pixel line, or a tenth of the biggest piece.
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = min(3.0 / s, 0.1 * areas.max()) if len(areas) else 0
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= max(floor, 0.1 * areas.max())]
    if not keep:
        return None
    sel = np.isin(lab, keep)
    rows, cols = np.nonzero(sel)
    if len(cols) < min_px:
        return None
    left, right = cols.min() * s, (cols.max() + 1) * s
    top, bottom = H - rows.min(), H - rows.max() - 1  # in rows from the near edge
    return [left, bottom * s, right - left, (top - bottom) * s], int(len(cols))


# ---------------------------------------------------------------- scoring


def bed_view(size: tuple[int, int]) -> tuple[BedView, Box]:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    lens = LensModel.load_or_none(DEFAULT_LENS_PATH)
    if lens is not None and tuple(lens.image_size) != tuple(size):
        lens = None
    bed = fixed_bed_homography(BedCalibration.load_or_none(DEFAULT_CALIBRATION_PATH), lens=lens)
    return BedView(bed, cfg.park_y, lambda: None), cfg.paper_box


def run_paper(run: Path) -> Box | None:
    """The paper box the run used, from the `hands.jsonl` header (runs after 2 Oct)."""
    p = run / "hands.jsonl"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        head = json.loads(f.readline() or "{}")
    return Box(*head["paper"]) if "paper" in head else None


def score(run: Path, size: tuple[int, int] = (1920, 1080), view=None, paper: Box | None = None):
    """`paper`: the paper box (bed mm) the run used; default from the run, else the config.
    The hold and planned places are in mm from its corner, so a wrong box shifts them all."""
    events = load_timeline(run)
    cmds = commands(events)
    paper = paper or run_paper(run)
    if view is None:
        view, cfg_paper = bed_view(size)
        paper = paper or cfg_paper
    photos = Photos(run, events, paper)
    s = MM_PER_PX * (4 if photos.small else 1)
    min_px = 6 if photos.small else 30
    previews = sorted(x["t"] for x in events if x["kind"] == "preview")
    warped: dict[Path, np.ndarray] = {}

    def top_down(p: Path) -> np.ndarray:
        if p not in warped:
            warped[p] = warp_to_paper(photos.image(p, size), view, paper, s, 0.0)
        return warped[p]

    pics = []
    for k, c in enumerate(cmds):
        prev = cmds[k - 1].drawn_t if k else None
        nxt = next((t for t in previews if t > c.drawn_t), None)
        b, a = photos.before(c, prev), photos.after(c, nxt)
        c.before = str(b.relative_to(run)) if b else None
        c.after = str(a.relative_to(run)) if a else None
        if b is None or a is None:
            continue
        A, B = top_down(b), top_down(a)
        found = find_ink(ink_mask(A, B, photos.small), c.planned_box, s, min_px)
        if found is None:
            continue
        box, n = found
        c.ink_box = [round(v, 2) for v in box]
        c.ink_xy = [round(box[0] + box[2] / 2, 2), round(box[1] + box[3] / 2, 2)]
        c.ink_size = round(max(box[2], box[3]), 2)
        c.ink_px = n
        pics.append(_crop(B, c, s))
    return cmds, pics


def _crop(img: np.ndarray, c: Command, s: float) -> np.ndarray:
    """A close-up: the planned box (blue), the ink box (red), the shown point (green)."""
    H = img.shape[0]
    x, y, w, h = c.planned_box
    g = 0.3 * max(w, h) + 6
    c0, r0 = max(0, int((x - g) / s)), max(0, int(H - (y + h + g) / s))
    c1, r1 = int((x + w + g) / s), int(H - (y - g) / s)
    vis = img.copy()

    def px(xm, ym):
        return int(xm / s), int(H - ym / s)

    cv2.rectangle(vis, px(x, y + h), px(x + w, y), (255, 120, 0), 1)
    if c.ink_box:
        ix, iy, iw, ih = c.ink_box
        cv2.rectangle(vis, px(ix, iy + ih), px(ix + iw, iy), (0, 0, 255), 1)
    if c.shown_xy:
        cv2.circle(vis, px(*c.shown_xy), 4, (0, 170, 0), -1)
    crop = vis[r0:r1, c0:c1]
    crop = cv2.resize(crop, (320, round(320 * crop.shape[0] / max(1, crop.shape[1]))))
    crop = cv2.resize(crop, (320, 320)) if crop.shape[0] != 320 else crop
    ce, se = c.centre_err, c.size_err
    text = f"{c.label} {'' if ce is None else f'{ce:.1f}mm'} {'' if se is None else f'{se:+.0%}'}"
    cv2.putText(crop, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return crop


def summary(cmds: list[Command]) -> dict:
    lat = [c.latency_s for c in cmds if c.latency_s is not None]
    med = statistics.median(lat) if lat else None
    return {
        "commands": len(cmds),
        "measured": sum(c.centre_err is not None for c in cmds),
        "passed": sum(c.passed for c in cmds),
        "median_centre_mm": _med([c.centre_err for c in cmds]),
        "median_size_err": _med([c.size_err for c in cmds]),
        "median_latency_s": med,
        "latency_ok": med is not None and med <= LATENCY_S,
    }


def _med(v: list[float | None]) -> float | None:
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def report(cmds: list[Command], summ: dict) -> str:
    out = []
    for i, c in enumerate(cmds, 1):
        parts = [f"{i:2d}. {c.label:10s} said {c.said!r}"]
        if c.centre_err is None:
            why = "no photo" if c.before is None or c.after is None else "no ink found"
            parts.append(f"   {why}")
        else:
            parts.append(
                f"   centre {c.centre_err:4.1f} mm (plotter {c.plot_err:.1f}),"
                f" size {c.ink_size:.0f} of {c.shown_size or 0:.0f} mm"
                f" ({'?' if c.size_err is None else f'{c.size_err:+.0%}'}, {c.size_kind}),"
                f" {'PASS' if c.passed else 'fail'}"
            )
        lat = c.latency_s
        parts.append(f"   speech end to preview {'?' if lat is None else f'{lat:.2f} s'}")
        if c.asks:
            parts.append(f"   asked: {' / '.join(c.asks)}")
        out.append("\n".join(parts))
    lat = summ["median_latency_s"]
    out.append(
        f"passed {summ['passed']} of {summ['commands']} (measured {summ['measured']});"
        f" median latency {'?' if lat is None else f'{lat:.2f} s'}"
    )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="an out/live/<time> folder")
    ap.add_argument("--json", type=Path, help="write the scores here")
    ap.add_argument("--crops", type=Path, help="write close-ups of each drawing here (PNG)")
    ap.add_argument(
        "--paper",
        help="x,y,w,h of the paper in bed mm, for runs that did not log it (1 Oct: 33,37,187,178)",
    )
    a = ap.parse_args(argv)
    paper = Box(*map(float, a.paper.split(","))) if a.paper else None
    cmds, pics = score(a.run, paper=paper)
    summ = summary(cmds)
    print(report(cmds, summ))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {"run": str(a.run), "summary": summ, "commands": [c.row() for c in cmds]}, indent=1
            ),
            encoding="utf-8",
        )
    if a.crops and pics:
        a.crops.parent.mkdir(parents=True, exist_ok=True)
        while len(pics) % 4:
            pics.append(np.full_like(pics[0], 255))
        grid = np.vstack([np.hstack(pics[k : k + 4]) for k in range(0, len(pics), 4)])
        cv2.imwrite(str(a.crops), grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
