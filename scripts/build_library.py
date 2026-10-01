"""Build the committed drawing library from the ranked candidates and library/picks.json.

    uv run python scripts/build_library.py

Reads library/candidates/<name>.json (from contact_sheets.py) and library/picks.json
(key_ids, or 1-based ranks). Writes:
- library/quickdraw/<name>.json   the picked variants, normalised (y up, min corner 0,0)
- library/index.json              categories, variant counts and synonyms
- library/ATTRIBUTION.md          the Quick, Draw!, SANA and Hershey credits

Generated line art (docs/virtual-test-plan.md section 6): library/sana/candidates/<name>.json
(traced by scripts/trace_library.py --job) and library/sana_picks.json (key_ids, or 1-based
ranks). A category with SANA picks uses library/sana/<name>.json; its Quick, Draw! file stays
as the index entry's "fallback". Every other category stays Quick, Draw!.
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
SANA_CANDIDATES = LIB / "sana" / "candidates"
SANA_PICKS = LIB / "sana_picks.json"
SANA_MODEL = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
# the style a request gets when it says neither "simple" nor "detailed". rahul liked the
# colouring-book look best (1 Oct 2026), but it draws 2 to 3 times slower than the icons: with
# it as the default, a big house ran past S4's 90 s and the hero script's spoken lines arrived
# mid-drawing. So "simple" (icon) is the default and "a detailed X" gets the colouring page.
DEFAULT_STYLE = "icon"
SANA_LICENCE = (
    "Apache-2.0 (model card, checked 2026-10-01); text encoder Gemma-2-2B-IT: Gemma Terms of Use"
)

SANA_ATTRIBUTION = """## Drawings: generated with SANA

The line drawings in `sana/` were generated ahead of time (never during a demo) with NVIDIA's
[SANA 1600M](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers)
(`Efficient-Large-Model/Sana_1600M_1024px_diffusers`), released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (model card checked on
1 Oct 2026). Its text encoder, Gemma-2-2B-IT, is used under Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). The images were
made on a Nebius AI Cloud Serverless Job, then traced to single pen lines (centre-line
tracing, `scripts/trace_library.py`); Cosmos3-Nano labels ranked them and a person picked
the variants. Each variant keeps its image name (`<object>_s<seed>`) as its `key_id`, with
its prompt and seed.

"""

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

    sana = add_sana(cats)

    synonyms = {k: v for k, v in sorted(SYNONYMS.items()) if v in cats and k not in cats}
    for cat in cats:
        cats[cat]["synonyms"] = sorted(k for k, v in synonyms.items() if v == cat)
    index = {
        "version": 1,
        "source": "Quick, Draw! by Google Creative Lab (CC BY 4.0); see ATTRIBUTION.md"
        + (". Categories with source 'sana': SANA 1600M line art (Apache-2.0)" if sana else ""),
        **({"default_style": DEFAULT_STYLE} if sana else {}),
        "categories": dict(sorted(cats.items())),
        "synonyms": synonyms,
    }
    (LIB / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    attribution = ATTRIBUTION
    if sana:
        attribution = attribution.replace("## Lettering", SANA_ATTRIBUTION + "## Lettering", 1)
    (LIB / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    print(f"{len(cats)} categories in {LIB / 'index.json'}")
    return 0


def add_sana(cats: dict[str, dict]) -> list[str]:
    """Switch categories with SANA picks to library/sana/<name>.json (Quick, Draw! fallback).
    Returns the categories switched."""
    done: list[str] = []
    if not SANA_PICKS.exists():
        return done
    picks = json.loads(SANA_PICKS.read_text(encoding="utf-8"))
    picks.pop("_help", None)
    (LIB / "sana").mkdir(parents=True, exist_ok=True)
    for cat, chosen in picks.items():
        cpath = SANA_CANDIDATES / f"{cat}.json"
        if not cpath.exists():
            print(f"{cat:14s} no SANA candidates, stays Quick, Draw!", file=sys.stderr)
            continue
        cands = json.loads(cpath.read_text(encoding="utf-8"))["variants"]
        by_key = {c["key_id"]: c for c in cands}
        variants = []
        for k in chosen:
            c = cands[k - 1] if isinstance(k, int) and 0 < k <= len(cands) else by_key.get(str(k))
            if c is None:
                print(f"{cat:14s} SANA pick {k!r} not among the candidates", file=sys.stderr)
                continue
            keep = ("key_id", "style", "strokes", "seed", "prompt")
            variants.append({k2: c[k2] for k2 in keep if k2 in c})
        if not variants:
            continue
        fname = f"sana/{cat}.json"
        doc = {
            "category": cat,
            "source": f"generated with {SANA_MODEL}; centre-line traced; picked by hand",
            "license": SANA_LICENCE,
            "units": "larger side about 255, y up, bbox min corner at (0, 0)",
            "variants": variants,
        }
        text = json.dumps(doc, separators=(",", ":")) + "\n"
        (LIB / fname).write_text(text, encoding="utf-8")
        entry = {"file": fname, "variants": len(variants), "source": "sana"}
        styles = sorted({v["style"] for v in variants if v.get("style")})
        if styles:
            entry["styles"] = {st: sum(v.get("style") == st for v in variants) for st in styles}
        if cat in cats:
            entry["fallback"] = cats[cat]["file"]
        cats[cat] = entry
        done.append(cat)
        kept = " (Quick, Draw! kept as fallback)" if "fallback" in entry else ""
        print(f"{cat:14s} {len(variants)} SANA variants{kept}")
    return done


if __name__ == "__main__":
    sys.exit(main())
