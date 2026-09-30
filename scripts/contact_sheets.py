"""Score Quick, Draw! drawings, keep the top 30 per category, and render contact sheets.

    uv run python scripts/contact_sheets.py            # all categories
    uv run python scripts/contact_sheets.py --categories ear brain

Inputs: .cache/quickdraw/<name>.ndjson (from fetch_quickdraw.py). Without the cache it
falls back to the committed library/candidates/, so it also works offline.

Outputs:
- library/candidates/<name>.json   the top 30, ranked (committed, small)
- out/contact_sheets/<name>.png    numbered grids
- out/contact_sheets/index.html    every candidate; click to pick, then copy picks.json
- library/picks.json               default picks (top 3) for categories that have none yet;
                                   existing picks are never overwritten (--reset-picks does)
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from pointdraw.drawing.quickdraw import (
    DEFAULT_CATEGORIES,
    cache_file,
    iter_ndjson,
    rank,
)
from pointdraw.drawing.render import render_cell

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "quickdraw"
CANDIDATES = ROOT / "library" / "candidates"
PICKS = ROOT / "library" / "picks.json"
OUT = ROOT / "out" / "contact_sheets"
TOP = 30
DEFAULT_PICKS = 3
COLS = 6
CELL = 170
PICKS_HELP = (
    "Per category, the Quick, Draw! key_ids to put in the library, best first. Pick 3 to 5 "
    "with out/contact_sheets/index.html, or write ranks (1-based numbers from the contact "
    "sheets). Then run scripts/build_library.py."
)


def candidates_path(cat: str) -> Path:
    return CANDIDATES / f"{cat}.json"


def score_category(cat: str) -> list[dict] | None:
    """Rank from the cache and write library/candidates/<cat>.json. None if no cache."""
    src = cache_file(CACHE, cat)
    if not src.exists():
        return None
    ranked = rank(iter_ndjson(src), top=TOP)
    cands = [
        {
            "rank": i + 1,
            "key_id": s.drawing.key_id,
            "score": round(s.score, 4),
            "strokes": s.drawing.to_ndjson()["drawing"],  # dataset units, y down
        }
        for i, s in enumerate(ranked)
    ]
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    doc = {"category": cat, "source": "Quick, Draw! (CC BY 4.0)", "candidates": cands}
    candidates_path(cat).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return cands


def load_candidates(cat: str) -> list[dict] | None:
    p = candidates_path(cat)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["candidates"]


def strokes_of(c: dict) -> list[list[tuple[int, int]]]:
    return [list(zip(xs, ys, strict=False)) for xs, ys in c["strokes"]]


def sheet(cat: str, cands: list[dict], picked: set[str]) -> Image.Image:
    rows = (len(cands) + COLS - 1) // COLS
    head = 34
    img = Image.new("RGB", (COLS * CELL, head + rows * CELL), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.text((8, 10), f"{cat}  (top {len(cands)}; red = current pick)", fill=(0, 0, 0))
    for i, c in enumerate(cands):
        x, y = (i % COLS) * CELL, head + (i // COLS) * CELL
        img.paste(render_cell(strokes_of(c), CELL - 6), (x + 3, y + 3))
        colour = (200, 40, 40) if c["key_id"] in picked else (90, 90, 90)
        d.text((x + 8, y + 6), f"#{c['rank']}", fill=colour)
        if c["key_id"] in picked:
            d.rectangle([x + 2, y + 2, x + CELL - 3, y + CELL - 3], outline=colour, width=3)
    return img


def svg(c: dict) -> str:
    paths = "".join(
        f'<polyline points="{" ".join(f"{x},{y}" for x, y in s)}"/>' for s in strokes_of(c)
    )
    return f'<svg viewBox="-8 -8 271 271">{paths}</svg>'


def html_index(all_cands: dict[str, list[dict]], picks: dict[str, list[str]]) -> str:
    sections = []
    for cat, cands in all_cands.items():
        cells = "".join(
            f'<figure data-cat="{html.escape(cat)}" data-key="{c["key_id"]}"'
            f"{' class=picked' if c['key_id'] in picks.get(cat, []) else ''}>"
            f"{svg(c)}<figcaption>#{c['rank']}</figcaption></figure>"
            for c in cands
        )
        sections.append(
            f'<section><h2 id="{html.escape(cat)}">{html.escape(cat)}</h2>'
            f'<div class="grid">{cells}</div></section>'
        )
    order = json.dumps({cat: [c["key_id"] for c in cs] for cat, cs in all_cands.items()})
    initial = json.dumps({cat: picks.get(cat, []) for cat in all_cands})
    toc = " · ".join(f'<a href="#{html.escape(c)}">{html.escape(c)}</a>' for c in all_cands)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Library picks</title><style>
body{{font:14px system-ui,sans-serif;margin:16px;background:#f4f4f2;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}}
figure{{margin:0;background:#fff;border:3px solid #ddd;border-radius:6px;cursor:pointer}}
figure.picked{{border-color:#d33}} figcaption{{text-align:center;color:#555}}
svg{{width:100%;display:block}} polyline{{fill:none;stroke:#222;stroke-width:4;
stroke-linejoin:round;stroke-linecap:round}}
#bar{{position:sticky;top:0;background:#f4f4f2;padding:8px 0;z-index:1}}
textarea{{width:100%;height:8em;font:12px monospace}}</style></head><body>
<div id="bar"><b>Click drawings to pick 3 to 5 per category</b> (first click = first choice).
<button onclick="out()">Show picks.json</button> <button onclick="copy()">Copy</button>
<div>{toc}</div><textarea id="json" hidden></textarea></div>
{"".join(sections)}
<script>
const ORDER={order}; const picks={initial};
document.querySelectorAll('figure').forEach(f=>f.onclick=()=>{{
  const c=f.dataset.cat,k=f.dataset.key,p=picks[c]||(picks[c]=[]);const i=p.indexOf(k);
  if(i>=0)p.splice(i,1);else p.push(k);f.classList.toggle('picked',i<0);out();}});
function out(){{const t=document.getElementById('json');t.hidden=false;
  const o={{"_help":{json.dumps(PICKS_HELP)}}};Object.keys(ORDER).forEach(c=>o[c]=picks[c]||[]);
  t.value=JSON.stringify(o,null,1);}}
function copy(){{out();const t=document.getElementById('json');t.select();
  navigator.clipboard?navigator.clipboard.writeText(t.value):document.execCommand('copy');}}
</script></body></html>"""


def load_picks() -> dict:
    if PICKS.exists():
        return json.loads(PICKS.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--no-rescore", action="store_true", help="use library/candidates only")
    ap.add_argument("--reset-picks", action="store_true", help="overwrite picks with top 3")
    args = ap.parse_args()

    cats = args.categories
    if cats is None:
        cached = {p.stem for p in CACHE.glob("*.ndjson")} if CACHE.exists() else set()
        committed = {p.stem for p in CANDIDATES.glob("*.json")} if CANDIDATES.exists() else set()
        known = cached | committed
        cats = [c for c in DEFAULT_CATEGORIES if c in known] + sorted(
            known - set(DEFAULT_CATEGORIES)
        )
    picks = load_picks()
    picks.pop("_help", None)
    OUT.mkdir(parents=True, exist_ok=True)
    all_cands: dict[str, list[dict]] = {}
    for cat in cats:
        cands = None if args.no_rescore else score_category(cat)
        cands = cands or load_candidates(cat)
        if not cands:
            print(f"{cat:14s} no data: run scripts/fetch_quickdraw.py", file=sys.stderr)
            continue
        all_cands[cat] = cands
        if args.reset_picks or not picks.get(cat):
            picks[cat] = [c["key_id"] for c in cands[:DEFAULT_PICKS]]
        picked = {str(k) for k in picks[cat]}
        picked |= {cands[k - 1]["key_id"] for k in picks[cat] if isinstance(k, int)}
        sheet(cat, cands, picked).save(OUT / f"{cat}.png")
        print(f"{cat:14s} {len(cands)} candidates, {len(picks[cat])} picked")
    PICKS.write_text(json.dumps({"_help": PICKS_HELP, **picks}, indent=1) + "\n", encoding="utf-8")
    str_picks = {c: [str(k) for k in v] for c, v in picks.items()}
    (OUT / "index.html").write_text(html_index(all_cands, str_picks), encoding="utf-8")
    print(f"Contact sheets: {OUT / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
