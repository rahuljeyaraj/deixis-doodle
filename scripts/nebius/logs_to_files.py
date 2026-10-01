"""Rebuild the files a library Job printed into its log (see Sink in library_job.py).

    nebius ai job logs <id> --since 6h > job.log
    python3 logs_to_files.py job.log out/sana_pilot

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
import sys
from pathlib import Path

PREFIX = re.compile(r"^\[[^\]]*\]\s?")
NAME = re.compile(r"^(?P<obj>.+)_s(?P<seed>\d+)\.png$")


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
    if len(args) != 2:
        print(__doc__)
        return 2
    log_path, out = Path(args[0]), Path(args[1])
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        files, problems = parse(fh)
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
