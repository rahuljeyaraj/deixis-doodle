"""Score line drawings with Cosmos: is each one recognised as its object? (plan T2.5)

    # a Job output folder (manifest.json from scripts/nebius/library_job.py)
    uv run python scripts/score_drawings.py --job out/sana_pilot

    # loose images of one object
    uv run python scripts/score_drawings.py out/sana_pilot/ear/*.png --object ear --out ear.json

Needs the Cosmos endpoint (COSMOS_BASE_URL and COSMOS_API_KEY in .env). Writes
<job>/cosmos_scores.json by default; `trace_library.py --job` then ranks its contact sheets by
it. Synonyms come from library/index.json ("hands", "palm" count as "hand").

The endpoint bills by the second, so: `--workers 4` asks four at once (vLLM batches them;
852 pictures in about 4 to 5 minutes instead of 16), each answer is saved as it comes
(`<out>.partial.jsonl`), and a run that was cut short carries on where it stopped. With a
Job folder it also reports the drawings kept in the library (`library/sana/*.json`, 421 on
6 Oct): measure list M19, "Cosmos names {n} of 421 kept drawings as their object".
`--check` asks about 3 pictures only (a few cents), to see the endpoint answers first.
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


def kept_keys(root: Path = ROOT / "library" / "sana") -> set[str]:
    """The key_ids of the drawings kept in the library (`<object>_<style>_s<seed>`)."""
    out = set()
    for f in root.glob("*.json"):
        for v in json.loads(f.read_text(encoding="utf-8")).get("variants", []):
            out.add(v["key_id"])
    return out


def kept_summary(rows: list[dict], keys: set[str] | None = None) -> dict:
    keys = kept_keys() if keys is None else keys
    mine = [r for r in rows if Path(r["file"]).stem in keys]
    return {
        "n": len(mine),
        "label_hits": sum(r["match"] == "label" for r in mine),
        "any_hits": sum(r["match"] in ("label", "alternative") for r in mine),
        "errors": sum(r["match"] == "error" for r in mine),
    }


def load_partial(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["file"]: r for r in rows if r.get("match") != "error"}


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
    ap.add_argument("--workers", type=int, default=4, help="calls at once")
    ap.add_argument("--check", action="store_true", help="only the first 3 pictures")
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
    if args.check:
        items, out = items[:3], out.with_name(out.stem + "-check.json")
    partial = out.with_name(out.name + ".partial.jsonl")
    done = load_partial(partial)
    if done:
        print(f"carrying on: {len(done)} answers already in {partial}", flush=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    keep = partial.open("a", encoding="utf-8")

    def show(row: dict) -> None:
        print(json.dumps(row), file=keep, flush=True)
        extra = f"  ({row['error']})" if "error" in row else ""
        alts = ", ".join(row["alternatives"])
        print(
            f"{Path(row['file']).name}: {row['label'] or '-'!s}"
            f"{' [' + alts + ']' if alts else ''} -> {row['match']}{extra}",
            flush=True,
        )

    try:
        report = score_images(
            client or make_client(args), items, library_synonyms(), show,
            workers=args.workers, done=done,
        )  # fmt: skip
    finally:
        keep.close()
    if args.job:
        report["kept"] = kept_summary(report["images"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print()
    for obj, s in report["objects"].items():
        print(f"{obj:>10}: top label {s['label_hits']}/{s['n']}, any {s['any_hits']}/{s['n']}")
    if report["median_latency_s"] is not None:
        print(f"median latency {report['median_latency_s']:.2f} s")
    if report.get("kept"):
        k = report["kept"]
        print(
            f"M19 kept drawings: Cosmos named {k['label_hits']} of {k['n']} as their object "
            f"(top label); {k['any_hits']} of {k['n']} with its other labels too"
        )
    print(f"wrote {out}")
    errors = sum(s["errors"] for s in report["objects"].values())
    return 1 if errors == len(items) else 0


if __name__ == "__main__":
    sys.exit(main())
