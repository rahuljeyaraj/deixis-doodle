"""Export replay cases for the deixis core (core split C16, plan 8.2).

    uv run python scripts/export_deixis_cases.py ../deixis/tests/replay

Runs every virtual scene (recorded agent replies, text voice) and every live session in
out/live/ that can be replayed through today's runtime, records each request the plotter
sends to the core with the answer it got, and writes every distinct one as a case:

    <out>/cases/<source>-<n>.json    {"source", "profile", "request", "response"}
    <out>/profiles/<name>.json       the plotter profiles the cases use (they differ by area)

Holds and spans far from the words are dropped when the answer stays the same without them,
so the files stay small. Only words, fingertip points and boxes: no images, no audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from deixis.api import Request, Response, resolve
from deixis.mcp import DeixisClient
from deixis.profile import Profile
from pointdraw.live.replay import load, replay
from pointdraw.runtime import references
from pointdraw.virtual import scenefile, world
from pointdraw.virtual.world import RunConfig

ROOT = Path(__file__).resolve().parents[1]
BEFORE_S, AFTER_S = 4.0, 3.0  # keep holds and spans this close to the words


class Recorder(DeixisClient):
    """An in-process client that keeps every resolve and its answer."""

    source = ""
    seen: list[tuple[str, Profile, Request, Response]] = []

    def __init__(self, profile: Profile) -> None:
        super().__init__(profile, mode="direct")

    def resolve(self, request: Request | dict[str, Any]) -> Response:
        req = request if isinstance(request, Request) else Request.model_validate(request)
        res = super().resolve(req)
        Recorder.seen.append((Recorder.source, self.profile, req, res))
        return res


def trimmed(req: Request, res: Response, profile: Profile) -> Request:
    """The request without holds and spans far from the words, if the answer is unchanged."""
    if req.speech is None and not req.words:
        return req
    t0, t1 = req.speech if req.speech is not None else (req.words[0].t0, req.words[-1].t1)
    near = req.model_copy(
        update={
            "holds": [h for h in req.holds if h.t1 >= t0 - BEFORE_S and h.t0 <= t1 + AFTER_S],
            "spans": [s for s in req.spans if s.t1 >= t0 - BEFORE_S and s.t0 <= t1 + AFTER_S],
        }
    )
    same = resolve(near, profile).model_dump_json() == res.model_dump_json()
    return near if same else req


def record(live: list[Path]) -> None:
    references.core_client = Recorder  # type: ignore[assignment]
    for spec in scenefile.all_scenes():
        if spec.voice == "tts":
            continue
        Recorder.source = spec.id
        world.run(spec, RunConfig())
    for d in live:
        try:
            session = load(d)
        except (ValueError, FileNotFoundError) as e:
            print(f"skip {d.name}: {e}")
            continue
        Recorder.source = f"live-{d.name}"
        replay(session)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", type=Path, help="the core's tests/replay folder")
    ap.add_argument("--live", type=Path, default=ROOT / "out" / "live", help="live runs")
    a = ap.parse_args(argv)
    live = sorted(p for p in a.live.iterdir() if p.is_dir()) if a.live.exists() else []
    record(live)

    cases_dir, prof_dir = a.out / "cases", a.out / "profiles"
    for d in (cases_dir, prof_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.json"):
            old.unlink()
    profiles: dict[str, str] = {}  # profile JSON -> file name
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for source, profile, req, res in Recorder.seen:
        pj = profile.model_dump_json()
        if pj not in profiles:
            profiles[pj] = f"{profile.name}-{len(profiles) + 1}"
            (prof_dir / f"{profiles[pj]}.json").write_text(
                json.dumps(profile.model_dump(mode="json"), indent=1) + "\n", encoding="utf-8"
            )
        req = trimmed(req, res, profile)
        key = hashlib.sha256((pj + req.model_dump_json()).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        counts[source] = counts.get(source, 0) + 1
        case = {
            "source": source,
            "profile": profiles[pj],
            "request": req.model_dump(mode="json"),
            "response": res.model_dump(mode="json"),
        }
        name = f"{source}-{counts[source]:02d}.json"
        (cases_dir / name).write_text(json.dumps(case, indent=1) + "\n", encoding="utf-8")
    print(f"{len(seen)} cases from {len(counts)} sources ({len(Recorder.seen)} requests), "
          f"{len(profiles)} profiles -> {a.out}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
