"""Download the local speech models (sherpa-onnx exports) and Silero VAD into .cache/models/.

uv run python scripts/asr_fetch_models.py              # default ASR model + VAD
uv run python scripts/asr_fetch_models.py --all        # every model in the R2 spike
uv run python scripts/asr_fetch_models.py parakeet_v2  # a specific model
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from pointdraw.audio.models import (
    ASR_MODELS,
    DEFAULT_ASR,
    MODELS_DIR,
    SILERO_VAD_PATH,
    SILERO_VAD_URL,
)


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, printing progress; skip if it already exists."""
    if dest.exists():
        print(f"have {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get  {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        next_report = 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"     {done / 1e6:7.1f} / {total / 1e6:.1f} MB", flush=True)
                next_report += 0.2
    tmp.replace(dest)


def fetch_asr(key: str, models_dir: Path) -> None:
    m = ASR_MODELS[key]
    if m.available(models_dir):
        print(f"have {m.dir(models_dir)}")
        return
    archive = models_dir / f"{m.archive}.tar.bz2"
    download(m.url, archive)
    print(f"untar {archive.name}")
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(models_dir, filter="data")
    archive.unlink()
    # Drop the fp32 files some archives also carry; we only use int8.
    for p in m.dir(models_dir).glob("*.onnx"):
        if ".int8." not in p.name:
            p.unlink()
    if not m.available(models_dir):
        sys.exit(f"extracted {m.archive} but expected files are missing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("models", nargs="*", help=f"model keys: {', '.join(ASR_MODELS)}")
    ap.add_argument("--all", action="store_true", help="every ASR model in the registry")
    ap.add_argument("--dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args()

    keys = list(ASR_MODELS) if args.all else (args.models or [DEFAULT_ASR])
    for k in keys:
        if k not in ASR_MODELS:
            sys.exit(f"unknown model {k!r}; choose from {', '.join(ASR_MODELS)}")
    free_gb = shutil.disk_usage(args.dir.resolve().anchor).free / 1e9
    print(f"{free_gb:.0f} GB free; each model needs about 1 GB while unpacking")
    download(SILERO_VAD_URL, args.dir / SILERO_VAD_PATH.name)
    for k in keys:
        fetch_asr(k, args.dir)
    print("done")


if __name__ == "__main__":
    main()
