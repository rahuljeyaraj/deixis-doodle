"""Rebuild the files a library Job printed into its log (see Sink in library_job.py).

    python3 logs_to_files.py --fetch ~/.nebius/bin/nebius <job id> <job created_at> out/sana_pilot
    python3 logs_to_files.py job.log out/sana_pilot          # from a saved log

The logs API returns at most 1000 lines per request (1 Oct 2026: "limit ... greater than 1000"
is a 400 error), so --fetch reads the log in time slices from the job's creation to now, splits
any slice that comes back full, and overlaps slices by a second (duplicates are harmless).

Lines may carry a "[timestamp] " prefix (`--timestamps`). Each file is checked against the size
and sha256 in its header; a file with a missing chunk is reported and skipped. If the log has
no manifest.json (the job was killed before its end), one is made from the file names.
Standard library only, so it runs in WSL's python3 as well as on Windows.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

PREFIX = re.compile(r"^\[[^\]]*\]\s?")
NAME = re.compile(r"^(?P<obj>.+)_s(?P<seed>\d+)\.png$")


LIMIT = 1000  # lines per logs request (the API's maximum)


def iso(t: int) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_lines(query: Callable[[int, int], list[str]], start: int, end: int, step: int = 30):
    """All log lines from `start` to `end` (epoch seconds). `query(a, b)` returns the lines in
    [a, b], at most LIMIT of them (the newest, as `--tail` does); a full answer is split."""
    out: list[str] = []

    def window(a: int, b: int) -> None:
        got = query(a - 1, b + 1)
        if len(got) < LIMIT:
            out.extend(got)
            return
        if b - a <= 1:
            raise RuntimeError(f"over {LIMIT} log lines in {iso(a)}..{iso(b)}: can't split further")
        m = (a + b) // 2
        window(a, m)
        window(m, b)

    for a in range(start, end + 1, step):
        window(a, min(a + step, end))
    return out


def cli_query(nb: str, job_id: str) -> Callable[[int, int], list[str]]:
    def query(a: int, b: int) -> list[str]:
        cmd = [nb, "ai", "job", "logs", job_id, "--since", iso(a), "--until", iso(b)]
        cmd += ["--tail", str(LIMIT), "--timestamps"]
        for attempt in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return r.stdout.splitlines()
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"logs {iso(a)}..{iso(b)} failed: {r.stderr.strip()[:300]}")

    return query


def parse(lines) -> tuple[dict[int, dict], list[str]]:
    files: dict[int, dict] = {}
    for raw in lines:
        line = PREFIX.sub("", raw.rstrip("\r\n"))
        if line.startswith("@@FILE "):
            _, no, n, size, sha, rel = line.split(" ", 5)
            files[int(no)] = {"n": int(n), "size": int(size), "sha": sha, "rel": rel, "chunks": {}}
        elif line.startswith("@@C "):
            _, no, i, data = line.split(" ", 3)
            f = files.get(int(no))
            if f is not None:
                f["chunks"][int(i)] = data
    problems = []
    for _, f in sorted(files.items()):
        missing = [i for i in range(f["n"]) if i not in f["chunks"]]
        if missing:
            problems.append(f"{f['rel']}: {len(missing)} of {f['n']} chunks missing")
            f["data"] = None
            continue
        data = base64.b64decode("".join(f["chunks"][i] for i in range(f["n"])))
        if len(data) != f["size"] or hashlib.sha256(data).hexdigest() != f["sha"]:
            problems.append(f"{f['rel']}: size or sha256 mismatch")
            f["data"] = None
            continue
        f["data"] = data
    return files, problems


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) == 5 and args[0] == "--fetch":
        nb, job_id, created, out = args[1], args[2], args[3], Path(args[4])
        start = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
        lines = fetch_lines(cli_query(nb, job_id), start, int(time.time()))
        print(f"log: {len(lines)} lines fetched")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "job.log", "w", encoding="utf-8") as fh:
            fh.writelines(line + chr(10) for line in lines)
        files, problems = parse(lines)
    elif len(args) == 2:
        out = Path(args[1])
        with open(args[0], encoding="utf-8", errors="replace") as fh:
            files, problems = parse(fh)
    else:
        print(__doc__)
        return 2
    written = []
    for f in files.values():
        if f["data"] is None:
            continue
        dest = out / f["rel"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f["data"])
        written.append(f["rel"])
    if written and "manifest.json" not in written:
        images = []
        for rel in sorted(r for r in written if r.endswith(".png")):
            m = NAME.match(Path(rel).name)
            if m:
                obj = m["obj"].replace("_", " ")
                images.append({"file": rel, "object": obj, "seed": int(m["seed"])})
        manifest = {"model": None, "rebuilt_from_log": True, "images": images}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        written.append("manifest.json (rebuilt from file names)")
    for p in problems:
        print("PROBLEM", p)
    print(f"{len(written)} files to {out} ({len(problems)} problems)")
    return 1 if problems or not written else 0


if __name__ == "__main__":
    sys.exit(main())
