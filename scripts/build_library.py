"""Build the committed drawing library from the ranked candidates and library/picks.json.

    uv run python scripts/build_library.py

Reads library/candidates/<name>.json (from contact_sheets.py) and library/picks.json
(key_ids, or 1-based ranks). Writes:
- library/quickdraw/<name>.json   the picked variants, normalised (y up, min corner 0,0)
- library/index.json              categories, variant counts and synonyms
- library/ATTRIBUTION.md          the Quick, Draw! and Hershey credits
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pointdraw.drawing.library import SYNONYMS
from pointdraw.drawing.quickdraw import RawDrawing, to_library_strokes

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "library"
CANDIDATES = LIB / "candidates"
PICKS = LIB / "picks.json"

ATTRIBUTION = """# Attribution

## Drawings: Quick, Draw!

The doodles in `quickdraw/` and `candidates/` come from the
[Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) by
Google Creative Lab, made available under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Changes made: we selected drawings from the "simplified" files (recognised drawings only),
flipped the y axis, moved each drawing's bounding box to the origin, and dropped the
country and timestamp fields. Each variant keeps its original `key_id`. At draw time the
strokes are smoothed (Chaikin corner cutting) and scaled.

## Lettering: Hershey fonts

Labels are drawn with the Hershey vector fonts (Dr A. V. Hershey, US National Bureau of
Standards, 1967), as packaged by [vpype](https://github.com/abey79/vpype) (MIT licence),
whose font data comes from Michael Fogleman's [axi](https://github.com/fogleman/axi) (MIT).
The Hershey fonts are free to use; this credit is given as their usage terms ask.
"""


def main() -> int:
    if not PICKS.exists():
        print("library/picks.json is missing: run scripts/contact_sheets.py first", file=sys.stderr)
        return 1
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    picks.pop("_help", None)
    (LIB / "quickdraw").mkdir(parents=True, exist_ok=True)
    cats: dict[str, dict] = {}
    for cat, chosen in picks.items():
        cpath = CANDIDATES / f"{cat}.json"
        if not cpath.exists():
            print(f"{cat:14s} no candidates, skipped", file=sys.stderr)
            continue
        cands = json.loads(cpath.read_text(encoding="utf-8"))["candidates"]
        by_key = {c["key_id"]: c for c in cands}
        variants = []
        for k in chosen:
            c = cands[k - 1] if isinstance(k, int) and 0 < k <= len(cands) else by_key.get(str(k))
            if c is None:
                print(f"{cat:14s} pick {k!r} not among the candidates, skipped", file=sys.stderr)
                continue
            raw = RawDrawing.from_ndjson(
                {"word": cat, "key_id": c["key_id"], "drawing": c["strokes"]}
            )
            variants.append({"key_id": c["key_id"], "strokes": to_library_strokes(raw)})
        if not variants:
            print(f"{cat:14s} nothing picked, skipped", file=sys.stderr)
            continue
        fname = f"quickdraw/{cat}.json"
        doc = {
            "category": cat,
            "source": "Quick, Draw! by Google Creative Lab",
            "license": "CC BY 4.0",
            "units": "larger side about 255, y up, bbox min corner at (0, 0)",
            "variants": variants,
        }
        (LIB / fname).write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
        cats[cat] = {"file": fname, "variants": len(variants)}
        print(f"{cat:14s} {len(variants)} variants")

    synonyms = {k: v for k, v in sorted(SYNONYMS.items()) if v in cats and k not in cats}
    for cat in cats:
        cats[cat]["synonyms"] = sorted(k for k, v in synonyms.items() if v == cat)
    index = {
        "version": 1,
        "source": "Quick, Draw! by Google Creative Lab (CC BY 4.0); see ATTRIBUTION.md",
        "categories": dict(sorted(cats.items())),
        "synonyms": synonyms,
    }
    (LIB / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    (LIB / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    print(f"{len(cats)} categories in {LIB / 'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
