"""Download the first few thousand Quick, Draw! drawings per category into .cache/quickdraw/.

Streams the simplified ndjson (the full files are tens to hundreds of MB), reads only the
first --lines lines, keeps drawings with `recognized: true`, then closes the stream.

    uv run python scripts/fetch_quickdraw.py                 # the default ~40 categories
    uv run python scripts/fetch_quickdraw.py --categories cat "light bulb" --force

Dataset: Quick, Draw! by Google Creative Lab, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pointdraw.drawing.quickdraw import (
    CATEGORIES_URL,
    DEFAULT_CATEGORIES,
    SIMPLIFIED_URL,
    cache_file,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "quickdraw"
TIMEOUT_S = 60


def official_categories(force: bool = False) -> set[str]:
    path = CACHE / "categories.txt"
    if force or not path.exists():
        with urllib.request.urlopen(CATEGORIES_URL, timeout=TIMEOUT_S) as r:
            path.write_bytes(r.read())
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def fetch(category: str, lines: int, force: bool) -> tuple[str, int, int]:
    """Returns (category, lines read, recognised kept)."""
    out = cache_file(CACHE, category)
    if out.exists() and not force:
        kept = sum(1 for _ in open(out, encoding="utf-8"))
        return category, -1, kept
    url = SIMPLIFIED_URL.format(name=urllib.parse.quote(category))
    read = kept = 0
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(tmp, "w", encoding="utf-8") as f:
        for raw in r:
            read += 1
            obj = json.loads(raw)
            if obj.get("recognized"):
                obj.pop("countrycode", None)  # not needed; keep the cache minimal
                obj.pop("timestamp", None)
                f.write(json.dumps(obj, separators=(",", ":")) + "\n")
                kept += 1
            if read >= lines:
                break  # leaving the with-block closes the HTTP stream
    tmp.replace(out)
    return category, read, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--lines", type=int, default=3000, help="lines to read per category")
    ap.add_argument("--force", action="store_true", help="download again even if cached")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    official = official_categories(args.force)
    missing = [c for c in args.categories if c not in official]
    if missing:
        print(f"Not official Quick, Draw! categories, skipped: {missing}", file=sys.stderr)
    todo = [c for c in args.categories if c in official]

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, c, args.lines, args.force): c for c in todo}
        for fut in futures:
            cat = futures[fut]
            try:
                _, read, kept = fut.result()
                note = "cached" if read < 0 else f"read {read}"
                print(f"{cat:14s} {note:>10s}  kept {kept} recognised")
            except Exception as e:  # network errors: report and carry on
                failed.append(cat)
                print(f"{cat:14s} FAILED: {e}", file=sys.stderr)
    print(f"{len(todo) - len(failed)}/{len(todo)} categories in {CACHE}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
