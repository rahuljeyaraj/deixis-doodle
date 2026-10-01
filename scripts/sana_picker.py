"""A click-to-pick page for traced SANA drawings (what the pen will draw, not the raw image).

    uv run python scripts/sana_picker.py out/traced_pilot [--scores SCORES.json]

Reads the variant files trace_library.py --job wrote (<dir>/<object>.json) and writes
<dir>/index.html: per object, every traced drawing as pen lines with its stroke count (and
score, if given), best first. Click 3 to 5 per object, then copy the JSON into
library/sana_picks.json and run scripts/build_library.py. Same idea as contact_sheets.py for
the Quick, Draw! candidates.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

HELP = (
    "Picked SANA drawings per category, first = first choice: key_ids from "
    "library/sana/candidates/<name>.json, or 1-based ranks. Then run scripts/build_library.py."
)


def svg(strokes: list) -> str:
    ys = [p[1] for s in strokes for p in s] or [0]
    top = max(ys)
    paths = "".join(
        '<polyline points="' + " ".join(f"{x:.1f},{top - y:.1f}" for x, y in s) + '"/>'
        for s in strokes
    )
    return f'<svg viewBox="-12 -12 280 280">{paths}</svg>'


def load(traced: Path, scores: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(traced.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if "variants" not in doc or "category" not in doc:
            continue
        vs = []
        for v in doc["variants"]:
            sc = scores.get(v["key_id"], {})
            vs.append({**v, "_score": sc.get("score"), "_note": sc.get("note", "")})
        vs.sort(key=lambda v: (-(v["_score"] or 0), len(v["strokes"])))
        out[doc["category"]] = vs
    return out


def page(cats: dict[str, list[dict]], picks: dict[str, list[str]]) -> str:
    sections = []
    for cat, vs in cats.items():
        cells = []
        for v in vs:
            k = v["key_id"]
            tag = f"{len(v['strokes'])} strokes"
            if v["_score"] is not None:
                tag += f" · {v['_score']:g}"
            note = html.escape(v["_note"])
            style = html.escape(v.get("style") or "")
            cells.append(
                f'<figure data-cat="{html.escape(cat)}" data-key="{html.escape(k)}"'
                f'{" class=picked" if k in picks.get(cat, []) else ""} title="{note}">'
                f"{svg(v['strokes'])}<figcaption>{html.escape(k)}<br>{tag}"
                f"{'<br><i>' + style + '</i>' if style else ''}</figcaption></figure>"
            )
        sections.append(
            f'<section><h2 id="{html.escape(cat)}">{html.escape(cat)}</h2>'
            f'<div class="grid">{"".join(cells)}</div></section>'
        )
    order = json.dumps({c: [v["key_id"] for v in vs] for c, vs in cats.items()})
    initial = json.dumps({c: picks.get(c, []) for c in cats})
    toc = " · ".join(f'<a href="#{html.escape(c)}">{html.escape(c)}</a>' for c in cats)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SANA picks</title><style>
body{{font:14px system-ui,sans-serif;margin:16px;background:#f4f4f2;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}}
figure{{margin:0;background:#fff;border:3px solid #ddd;border-radius:6px;cursor:pointer}}
figure.picked{{border-color:#d33}} figcaption{{text-align:center;color:#555;font-size:12px}}
svg{{width:100%;display:block}} polyline{{fill:none;stroke:#222;stroke-width:2.5;
stroke-linejoin:round;stroke-linecap:round}}
#bar{{position:sticky;top:0;background:#f4f4f2;padding:8px 0;z-index:1}}
textarea{{width:100%;height:8em;font:12px monospace}}</style></head><body>
<div id="bar"><b>Click the drawings you like: 3 to 5 per object</b> (first click = first
choice). These are the pen lines the robot will draw.
<button onclick="out()">Show picks</button> <button onclick="copy()">Copy</button>
<div>{toc}</div><textarea id="json" hidden></textarea></div>
{"".join(sections)}
<script>
const ORDER={order}; const picks={initial};
document.querySelectorAll('figure').forEach(f=>f.onclick=()=>{{
  const c=f.dataset.cat,k=f.dataset.key,p=picks[c]||(picks[c]=[]);const i=p.indexOf(k);
  if(i>=0)p.splice(i,1);else p.push(k);f.classList.toggle('picked',i<0);out();}});
function out(){{const t=document.getElementById('json');t.hidden=false;
  const o={{"_help":{json.dumps(HELP)}}};Object.keys(ORDER).forEach(c=>{{
  if((picks[c]||[]).length)o[c]=picks[c];}});t.value=JSON.stringify(o,null,1);}}
function copy(){{out();const t=document.getElementById('json');t.select();
  navigator.clipboard?navigator.clipboard.writeText(t.value):document.execCommand('copy');}}
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("traced", type=Path, help="folder written by trace_library.py --job")
    ap.add_argument("--scores", type=Path, help="scores JSON (score_drawings.py format)")
    ap.add_argument("--picks", type=Path, default=Path("library/sana_picks.json"))
    args = ap.parse_args(argv)
    scores: dict[str, dict] = {}
    if args.scores and args.scores.exists():
        rep = json.loads(args.scores.read_text(encoding="utf-8"))
        scores = {Path(r["file"]).stem: r for r in rep["images"]}
    picks: dict[str, list[str]] = {}
    if args.picks.exists():
        picks = json.loads(args.picks.read_text(encoding="utf-8"))
        picks.pop("_help", None)
    cats = load(args.traced, scores)
    if not cats:
        print(f"no traced variant files in {args.traced}", file=sys.stderr)
        return 2
    out = args.traced / "index.html"
    out.write_text(page(cats, picks), encoding="utf-8")
    print(f"{sum(len(v) for v in cats.values())} drawings, {len(cats)} objects: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
