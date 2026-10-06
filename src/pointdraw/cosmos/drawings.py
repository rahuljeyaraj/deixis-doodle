"""Does Cosmos recognise a library drawing as its object? (plan T2.5)

One `label_crop` call per image. The top label naming the object (or a synonym from
library/index.json) scores 1.0, a match among the alternatives 0.5, anything else 0. Scores
rank each object's candidates before rahul picks; they don't replace his look.

    from pointdraw.cosmos.drawings import score_images
    report = score_images(client, [("out/sana_pilot/ear/ear_s1000.png", "ear"), ...], synonyms)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pointdraw.cosmos.prompts import PROMPT_VERSION
from pointdraw.cosmos.scoring import label_matches

SCORES = {"label": 1.0, "alternative": 0.5, "none": 0.0}


def judge(label: str, alternatives: Iterable[str], obj: str, synonyms: Iterable[str] = ()) -> str:
    """'label', 'alternative' or 'none': where (if anywhere) Cosmos named the object."""
    syn = list(synonyms)
    if label_matches(label, obj, syn):
        return "label"
    if any(label_matches(a, obj, syn) for a in alternatives):
        return "alternative"
    return "none"


def score_images(
    client: Any,
    items: Iterable[tuple[str | Path, str]],
    synonyms: Mapping[str, Iterable[str]] | None = None,
    progress: Callable[[dict], None] | None = None,
    *,
    workers: int = 1,
    done: Mapping[str, dict] | None = None,
) -> dict:
    """Label every (image, object) pair with `client.label_crop` and score the match.

    Returns {"model", "prompt_version", "created", "images": [...], "objects": {...}}, where
    `objects[obj]["ranked"]` lists that object's files best first (score, then input order).
    A failed call is recorded with its error and scores 0, so one bad reply doesn't lose a run.
    `workers` calls run at once (the endpoint batches them: less billed GPU time); `done`
    holds rows from an earlier, cut-short run, by file, and those images aren't asked again.
    """
    synonyms = synonyms or {}
    done = done or {}

    def one(n: int, path: str | Path, obj: str) -> dict:
        row: dict[str, Any] = {"file": str(path), "object": obj, "order": n}
        if str(path) in done and done[str(path)].get("match") != "error":
            return {**done[str(path)], "order": n}
        try:
            r = client.label_crop(Path(path))
            match = judge(r.label, r.alternatives, obj, synonyms.get(obj, ()))
            row.update(
                label=r.label,
                alternatives=r.alternatives,
                match=match,
                score=SCORES[match],
                latency_s=round(r.reply.latency_s, 3),
                raw=r.reply.raw,
            )
        except Exception as e:  # noqa: BLE001 - one failed call must not end the run
            row.update(label="", alternatives=[], match="error", score=0.0, error=repr(e))
        if progress:
            progress(row)
        return row

    jobs = list(enumerate(items))
    if workers <= 1:
        rows = [one(n, p, o) for n, (p, o) in jobs]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(lambda j: one(j[0], *j[1]), jobs))

    objects: dict[str, dict] = {}
    for obj in dict.fromkeys(r["object"] for r in rows):
        mine = [r for r in rows if r["object"] == obj]
        ranked = sorted(mine, key=lambda r: (-r["score"], r["order"]))
        objects[obj] = {
            "n": len(mine),
            "label_hits": sum(r["match"] == "label" for r in mine),
            "any_hits": sum(r["match"] in ("label", "alternative") for r in mine),
            "errors": sum(r["match"] == "error" for r in mine),
            "ranked": [r["file"] for r in ranked],
        }
    lat = sorted(r["latency_s"] for r in rows if "latency_s" in r)
    return {
        "model": getattr(client, "model", None),
        "prompt_version": PROMPT_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "median_latency_s": lat[len(lat) // 2] if lat else None,
        "images": rows,
        "objects": objects,
    }


def rank_of(report: Mapping, file: str | Path) -> tuple[int, dict] | None:
    """(1-based rank within its object, the image's row) for a file in a score report."""
    name = Path(file).name
    for row in report.get("images", []):
        if Path(row["file"]).name == name:
            ranked = [Path(f).name for f in report["objects"][row["object"]]["ranked"]]
            return ranked.index(name) + 1, row
    return None
