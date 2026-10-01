"""Score line drawings with Cosmos: is each one recognised as its object? (plan T2.5)

    # a Job output folder (manifest.json from scripts/nebius/library_job.py)
    uv run python scripts/score_drawings.py --job out/sana_pilot

    # loose images of one object
    uv run python scripts/score_drawings.py out/sana_pilot/ear/*.png --object ear --out ear.json

Needs the Cosmos endpoint (COSMOS_BASE_URL and COSMOS_API_KEY in .env). Writes
<job>/cosmos_scores.json by default; `trace_library.py --job` then ranks its contact sheets by
it. Synonyms come from library/index.json ("hands", "palm" count as "hand").
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from pointdraw.cosmos.drawings import score_images

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def library_synonyms(index: Path = ROOT / "library" / "index.json") -> dict[str, list[str]]:
    if not index.exists():
        return {}
    cats = json.loads(index.read_text(encoding="utf-8"))["categories"]
    return {name: list(c.get("synonyms", [])) for name, c in cats.items()}


def job_items(job: Path) -> list[tuple[Path, str]]:
    data = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    return [(job / e["file"], e["object"]) for e in data["images"]]


def loose_items(paths: list[str], obj: str) -> list[tuple[Path, str]]:
    out = []
    for p in paths:
        for h in sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]:
            q = Path(h)
            files = sorted(q.iterdir()) if q.is_dir() else [q]
            out += [(f, obj) for f in files if f.suffix.lower() in IMAGE_EXT]
    return out


def make_client(args: argparse.Namespace):
    from pointdraw.cosmos.client import CosmosClient

    return CosmosClient(think=False, timeout_s=args.timeout)


def main(argv: list[str] | None = None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*", help="images, folders or globs (with --object)")
    ap.add_argument("--job", type=Path, help="a Job output folder with manifest.json")
    ap.add_argument("--object", help="object name for loose images")
    ap.add_argument("--out", type=Path, help="JSON to write (default <job>/cosmos_scores.json)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per Cosmos call")
    args = ap.parse_args(argv)

    if args.job:
        items = job_items(args.job)
        out = args.out or args.job / "cosmos_scores.json"
    elif args.images and args.object and args.out:
        items = loose_items(args.images, args.object)
        out = args.out
    else:
        ap.error("give --job DIR, or images plus --object and --out")
    if not items:
        print("no images found", file=sys.stderr)
        return 2

    def show(row: dict) -> None:
        extra = f"  ({row['error']})" if "error" in row else ""
        alts = ", ".join(row["alternatives"])
        print(
            f"{Path(row['file']).name}: {row['label'] or '-'!s}"
            f"{' [' + alts + ']' if alts else ''} -> {row['match']}{extra}",
            flush=True,
        )

    report = score_images(client or make_client(args), items, library_synonyms(), show)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print()
    for obj, s in report["objects"].items():
        print(f"{obj:>10}: top label {s['label_hits']}/{s['n']}, any {s['any_hits']}/{s['n']}")
    if report["median_latency_s"] is not None:
        print(f"median latency {report['median_latency_s']:.2f} s")
    print(f"wrote {out}")
    errors = sum(s["errors"] for s in report["objects"].values())
    return 1 if errors == len(items) else 0


if __name__ == "__main__":
    sys.exit(main())
