"""Outputs of a virtual run: the final sheet (PNG), a replay (animated GIF) and a report page.

The replay shows the sheet from above in real time: the fingertip (green when pointing,
with the pinch or the two-hand span), each finished hold (orange ring), the preview box
(dashed), the pen (red dot when down) and the ink it lays, with the words and the robot's
replies as captions. Under it, a timeline: words, holds, size gestures and plotter jobs,
with a moving cursor.

Colours follow the dataviz reference palette: blue = speech, orange = pointing holds,
aqua = size gestures, dark ink = the pen. Each timeline lane is labelled, so colour is
never the only cue.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pointdraw.virtual.hand import FPS

BLUE, ORANGE, AQUA = (42, 120, 214), (235, 104, 52), (27, 175, 122)
INK, MUTED, GRID, SURFACE = (11, 11, 11), (82, 81, 78), (228, 228, 224), (252, 252, 251)
PAPER, RED, GREEN = (255, 255, 255), (227, 73, 72), (0, 131, 0)
USER_INK = {
    "pencil": (100, 100, 105),
    "black pen": (30, 30, 30),
    "blue crayon": (60, 140, 200),
    "red crayon": (200, 60, 60),
    "green crayon": (80, 170, 90),
    "orange crayon": (235, 150, 70),
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class _Canvas:
    def __init__(self, res: Any, px_per_mm: float = 2.6, margin: int = 16, strip_h: int = 150):
        self.res = res
        self.s = px_per_mm
        self.m = margin
        p = res.plotter.config.paper
        self.pw, self.ph = p.w, p.h
        self.origin = (p.x, p.y)
        self.w = int(self.pw * self.s) + 2 * margin
        self.sheet_h = int(self.ph * self.s) + 2 * margin + 44
        self.strip_h = strip_h
        self.h = self.sheet_h + strip_h
        self.font = _font(13)
        self.small = _font(11)
        self.t_end = max(res.t_end, 1.0)

    def px(self, x: float, y: float) -> tuple[float, float]:
        """Paper mm -> image px (y up on the page, the user at the bottom)."""
        return (self.m + x * self.s, self.m + 44 + (self.ph - y) * self.s)

    def bed_px(self, x: float, y: float) -> tuple[float, float]:
        return self.px(x - self.origin[0], y - self.origin[1])

    def tx(self, t: float) -> float:
        return 90 + (self.w - 100) * min(1.0, max(0.0, t / self.t_end))


def _dashed_rect(d: ImageDraw.ImageDraw, a: tuple, b: tuple, fill: tuple, dash: int = 6) -> None:
    (x0, y0), (x1, y1) = a, b
    for p, q in (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ):
        n = max(1, int(np.hypot(q[0] - p[0], q[1] - p[1]) / dash))
        for i in range(0, n, 2):
            u0, u1 = i / n, min(1.0, (i + 1) / n)
            d.line(
                [
                    (p[0] + (q[0] - p[0]) * u0, p[1] + (q[1] - p[1]) * u0),
                    (p[0] + (q[0] - p[0]) * u1, p[1] + (q[1] - p[1]) * u1),
                ],
                fill=fill,
                width=2,
            )


def _base(c: _Canvas) -> Image.Image:
    img = Image.new("RGB", (c.w, c.h), SURFACE)
    d = ImageDraw.Draw(img)
    d.rectangle([c.px(0, c.ph), c.px(c.pw, 0)], fill=PAPER, outline=(190, 190, 185))
    for mm in range(10, int(c.pw), 10):
        d.line([c.px(mm, 0), c.px(mm, c.ph)], fill=(243, 243, 240))
    for mm in range(10, int(c.ph), 10):
        d.line([c.px(0, mm), c.px(c.pw, mm)], fill=(243, 243, 240))
    dr = c.res.runtime.drawable
    _dashed_rect(d, c.px(dr.x, dr.y2), c.px(dr.x2, dr.y), (225, 225, 220), 4)
    _timeline_static(c, d)
    return img


def _lanes(c: _Canvas) -> dict[str, int]:
    y0 = c.sheet_h + 10
    return {"said": y0, "holds": y0 + 24, "size": y0 + 48, "pen": y0 + 72, "hand over": y0 + 96}


def _timeline_static(c: _Canvas, d: ImageDraw.ImageDraw) -> None:
    res, lanes = c.res, _lanes(c)
    for name, y in lanes.items():
        d.text((8, y + 2), name, fill=MUTED, font=c.small)
        d.line([(c.tx(0), y + 18), (c.tx(c.t_end), y + 18)], fill=GRID)
    for u in res.log.of("utterance"):
        y = lanes["said"]
        for _text, t0, t1 in u["words"]:
            d.rectangle([c.tx(t0), y + 3, max(c.tx(t1) - 1, c.tx(t0) + 2), y + 15], fill=BLUE)
        if u["words"]:
            d.text((c.tx(u["words"][0][1]), y - 10), u["text"][:40], fill=INK, font=c.small)
    for h in res.log.of("hold"):
        y = lanes["holds"]
        d.rounded_rectangle(
            [c.tx(h["t0"]), y + 3, max(c.tx(h["t1"]), c.tx(h["t0"]) + 3), y + 15],
            radius=3,
            fill=ORANGE,
        )
    for s in res.log.of("span"):
        y = lanes["size"]
        d.rounded_rectangle(
            [c.tx(s["t0"]), y + 3, max(c.tx(s["t1"]), c.tx(s["t0"]) + 3), y + 15],
            radius=3,
            fill=AQUA,
        )
        d.text((c.tx(s["t1"]) + 3, y + 3), f"{s['span']:.0f} mm", fill=INK, font=c.small)
    starts: dict[Any, float] = {}
    for e in res.plotter.events:
        if e.kind == "job_start":
            starts[(e.label, e.tag)] = e.t
        elif e.kind in ("job_end", "halted") and (e.label, e.tag) in starts:
            t0 = starts.pop((e.label, e.tag))
            y = lanes["pen"]
            box = [c.tx(t0), y + 3, max(c.tx(e.t), c.tx(t0) + 2), y + 15]
            if e.label == "draw":
                d.rectangle(box, fill=INK)
            elif e.label == "preview":
                d.rectangle(box, outline=INK, width=1)
            else:
                d.rectangle(box, fill=(200, 200, 196))
    for e in res.plotter.events:
        if e.kind in ("paused", "halted", "stop_request"):
            y = lanes["pen"]
            d.line([c.tx(e.t), y, c.tx(e.t), y + 18], fill=RED, width=2)
    y = lanes["hand over"]
    run0 = None
    for t, st in res.hand_track[:: max(1, int(FPS / 10))]:
        from pointdraw.virtual.world import over_paper

        over = over_paper(st, c.pw, c.ph)
        if over and run0 is None:
            run0 = t
        if not over and run0 is not None:
            d.rectangle([c.tx(run0), y + 6, c.tx(t), y + 12], fill=(200, 200, 196))
            run0 = None
    if run0 is not None:
        d.rectangle([c.tx(run0), y + 6, c.tx(c.t_end), y + 12], fill=(200, 200, 196))
    for sec in range(0, int(c.t_end) + 1, 5):
        d.text((c.tx(sec) - 4, c.h - 16), f"{sec}s", fill=MUTED, font=c.small)


def _state_at(res: Any, t: float) -> str:
    s = "LISTEN"
    for e in res.log.of("state"):
        if e["t"] <= t:
            s = e["state"]
    return s


def _preview_box(res: Any, t: float) -> list[float] | None:
    box = None
    for e in res.log.events:
        if e["t"] > t:
            break
        if e["kind"] == "preview":
            box = e["box"]
        elif e["kind"] in ("confirm", "cancel", "stop", "new_sheet"):
            box = None
        elif e["kind"] == "item" and e.get("replaced") and box is not None:
            box = e["box"]
    return box


def _pen_at(res: Any, t: float) -> tuple[float, float, bool]:
    tr = res.plotter.track
    lo, hi = 0, len(tr) - 1
    if not tr:
        c = res.plotter.config
        return (c.park_x, c.park_y, False)
    if t <= tr[0][0]:
        return tr[0][1], tr[0][2], tr[0][3]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tr[mid][0] <= t:
            lo = mid
        else:
            hi = mid - 1
    a = tr[lo]
    if lo + 1 < len(tr):
        b = tr[lo + 1]
        if b[0] > a[0]:
            u = (t - a[0]) / (b[0] - a[0])
            return a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, b[3] and a[3]
    return a[1], a[2], a[3]


def _caption(res: Any, t: float) -> tuple[str, str]:
    said = ""
    for u in res.log.of("utterance"):
        ws = [w for w in u["words"] if w[1] <= t]
        if ws and t - u["words"][-1][2] < 2.5:
            said = " ".join(w[0] for w in ws)
    reply = ""
    for e in res.log.of("say"):
        if e["t"] <= t < e["t"] + 3.0:
            reply = e["text"]
    return said, reply


def render_gif(res: Any, path: str | Path, fps: float = 10.0, speed: float = 1.0) -> Path:
    """Animated replay of a run, `speed` times real time."""
    c = _Canvas(res)
    base = _base(c)
    ink_layer = base.copy()
    di = ImageDraw.Draw(ink_layer)
    # Earlier robot drawings and user marks present from the start.
    segs = sorted(res.plotter.ink, key=lambda s: s.t)
    marks = sorted(res.sheet.marks, key=lambda m: m.t)
    si = mi = 0
    frames: list[Image.Image] = []
    dt = speed / fps
    t = 0.0
    holds = res.log.of("hold")
    resets = sorted(e["t"] for e in res.log.of("new_sheet"))
    ri = 0
    while t <= res.t_end + 1e-9:
        while ri < len(resets) and resets[ri] <= t:  # a fresh sheet: wipe the ink
            ink_layer = base.copy()
            di = ImageDraw.Draw(ink_layer)
            ri += 1
        while mi < len(marks) and marks[mi].t <= t:
            m = marks[mi]
            col = USER_INK.get(m.mark.ink, INK)
            for line in m.mark.polylines_mm:
                di.line([c.px(*p) for p in line], fill=col, width=2)
            mi += 1
        while si < len(segs) and segs[si].t <= t:
            s = segs[si]
            di.line([c.bed_px(*s.a), c.bed_px(*s.b)], fill=INK, width=2)
            si += 1
        fr = ink_layer.copy()
        d = ImageDraw.Draw(fr)
        box = _preview_box(res, t)
        if box is not None:
            x, y, w, h = box
            _dashed_rect(d, c.px(x, y + h), c.px(x + w, y), BLUE)
        for h in holds:
            if h["t1"] <= t < h["t1"] + 1.5:
                hx, hy = c.px(*h["mean"])
                d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], outline=ORANGE, width=2)
        st = res.hand.state(t)
        if st.present:
            fx, fy = c.px(*st.tip)
            col = GREEN if st.pose in ("point", "pinch") else MUTED
            if st.pinch_mm:
                thx, thy = c.px(st.tip[0] + st.pinch_mm * 0.53, st.tip[1] - st.pinch_mm * 0.85)
                d.line([(fx, fy), (thx, thy)], fill=AQUA, width=2)
            if st.two_mm:
                ox, oy = c.px(st.tip[0] - st.two_mm, st.tip[1])
                d.line([(fx, fy), (ox, oy)], fill=AQUA, width=2)
                d.ellipse([ox - 6, oy - 6, ox + 6, oy + 6], fill=col)
            d.line([(fx, fy), (fx, fy + 60)], fill=(210, 190, 170), width=14)  # the finger
            d.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], fill=col)
        px_, py_, down = _pen_at(res, t)
        hx, hy = c.bed_px(px_, py_)
        if down:
            d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=RED)
        else:
            d.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], outline=RED, width=2)
        said, reply = _caption(res, t)
        d.text((c.m, 6), f"{res.spec.id}  {t:5.1f} s   {_state_at(res, t)}", fill=INK, font=c.font)
        if said:
            d.text((c.m, 24), f'"{said}"', fill=BLUE, font=c.font)
        if reply:
            d.text((c.w // 2, 24), f"robot: {reply}", fill=MUTED, font=c.font)
        cx = c.tx(t)
        d.line([(cx, c.sheet_h + 2), (cx, c.h - 18)], fill=RED, width=1)
        frames.append(fr.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        t += dt
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return out


def render_final(res: Any, path: str | Path, mm_per_px: float = 0.25) -> Path:
    """The sheet at the end, as the virtual camera sees it, with the targets marked."""
    img, mmpp = res.camera.paper_image(res.t_end + 1.0, mm_per_px=mm_per_px)
    rgb = Image.fromarray(img[:, :, ::-1].copy())
    d = ImageDraw.Draw(rgb)
    h = rgb.height

    def px(p: tuple[float, float]) -> tuple[float, float]:
        return (p[0] / mmpp, h - p[1] / mmpp)

    for want in res.spec.expect.drawn or []:
        if want.center is None:
            continue
        x, y = px(want.center)
        r = want.tol_mm / mmpp
        d.ellipse([x - r, y - r, x + r, y + r], outline=GREEN, width=2)
    for e in res.log.of("drawn"):
        if e.get("ink_center"):
            x, y = px(e["ink_center"])
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=RED)
    d.text(
        (8, 6),
        f"{res.spec.id} {res.spec.title}: green = target and tolerance, red = drawn centre",
        fill=(60, 60, 60),
        font=_font(16),
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out)
    return out


def write_report(res: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.report, indent=1, default=str) + "\n", encoding="utf-8")
    return out


def write_timeline(res: Any, path: str | Path) -> Path:
    """Every event of the run as JSON lines (for debugging and later analysis)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in res.log.events:
            f.write(json.dumps(e, default=str) + "\n")
        for e in res.plotter.events:
            f.write(
                json.dumps(
                    {
                        "t": round(e.t, 3),
                        "kind": f"plotter.{e.kind}",
                        "label": e.label,
                        "tag": e.tag,
                        **e.detail,
                    }
                )
                + "\n"
            )
    return out


GIF_ALT = (
    "Replay of the run: the sheet from above, with the finger, the preview box and the pen, "
    "and a timeline of words, holds and pen jobs underneath"
)
FINAL_ALT = "The final sheet as the virtual camera sees it"

PAGE_CSS = """
:root { --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --ok:#008300; --bad:#e34948; --line:#e4e4e0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
  --ok:#3fbf3f; --bad:#e66767; --line:#3a3a38; } }
:root[data-theme="dark"] { --surface-1:#1a1a19; --text-primary:#ffffff;
  --text-secondary:#c3c2b7; --ok:#3fbf3f; --bad:#e66767; --line:#3a3a38; }
body { background:var(--surface-1); color:var(--text-primary);
  font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:980px; padding:16px; }
h1 { font-size:20px; margin:0 0 4px; } .muted { color:var(--text-secondary); }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }
td, th { border-bottom:1px solid var(--line); padding:4px 6px; text-align:left;
  vertical-align:top; }
.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.wrap { overflow-x:auto; }
"""


def write_page(res: Any, path: str | Path, gif: str | None, final_png: str | None) -> Path:
    """A small HTML page: the replay, the final sheet and the checks, next to the files."""
    r = res.report
    verdict = "PASS" if r["pass"] else "FAIL"
    rows = "".join(
        f"<tr><td class='{'ok' if c['ok'] else 'bad'}'>{'ok' if c['ok'] else 'FAIL'}</td>"
        f"<td>{html.escape(c['name'])}</td><td>{html.escape(str(c['detail']))}</td></tr>"
        for c in r["checks"]
    )
    mets = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v))}</td></tr>"
        for k, v in r["metrics"].items()
    )
    said = "".join(f"<li>{html.escape(u)}</li>" for u in r["utterances"])
    replies = "".join(f"<li>{html.escape(s)}</li>" for s in r["says"])
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(r["scene"])} replay</title><style>{PAGE_CSS}</style></head><body>
<h1>{html.escape(r["scene"])}: {html.escape(r["title"])}
 <span class="{"ok" if r["pass"] else "bad"}">{verdict}</span></h1>
<p class="muted">{html.escape(res.spec.tests)} · variant {html.escape(r["variant"])} ·
 agent {html.escape(r["config"]["ai"])} · voice {html.escape(r["config"]["voice"])}</p>
{f'<p><img src="{gif}" alt="{GIF_ALT}"></p>' if gif else ""}
<div class="wrap"><table><tr><th></th><th>Check</th><th>Detail</th></tr>{rows}</table></div>
<h2>What was said</h2><ol>{said}</ol><h2>What the robot said</h2><ul>{replies}</ul>
<div class="wrap"><table><tr><th>Metric</th><th>Value</th></tr>{mets}</table></div>
{f'<p><img src="{final_png}" alt="{FINAL_ALT}"></p>' if final_png else ""}
</body></html>"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out
