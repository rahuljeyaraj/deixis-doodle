"""Make the synthetic R2 test set: command-style sentences spoken by Windows' built-in voices.

    uv run python scripts/make_test_utterances.py

Writes 16 kHz mono WAVs to .cache/r2/ (never committed) and the ground truth, with each word's
start time from SAPI's SpeakProgress events, to tests/audio/data/r2_ground_truth.json
(committed). It also checks the event timing against the waveform: the first word's event time
should match the first sound in the file.

Windows only (System.Speech via PowerShell).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from pointdraw.audio.scoring import DEMONSTRATIVES, normalize_word, tokenize
from pointdraw.audio.sync import first_sound, last_sound
from pointdraw.audio.wavio import read_wav

ROOT = Path(__file__).resolve().parents[1]
PS1 = ROOT / "scripts" / "make_test_utterances.ps1"
GROUND_TRUTH = ROOT / "tests" / "audio" / "data" / "r2_ground_truth.json"

SENTENCES = [
    "put a tree here",
    "draw a cat there",
    "about this big",
    "an arrow from this to that",
    "label that one Nemotron",
    "a bit to the left",
    "make it this big",
    "draw a house here and a sun there",
    "put a flower next to that",
    "write Cosmos under this one",
    "move it a bit to the right",
    "draw a line from here to there",
    "a small star here",
    "label this Nebius",
    "make that one bigger",
    "connect this to that",
    "draw a cloud up here about this size",
    "put the dog here, no, there",
    "stop",
    "yes, that one",
    "draw an eye here and an ear there",
    "write brain next to this",
    "put a microphone right here",
    "a big circle around that",
    "draw a fence from here to here",
    "undo that",
    "a bit smaller",
    "put a bird between this and that",
    "new sheet please",
    "draw a laptop here, this wide",
    "label these two sensors",
    "an arrow from the camera to this",
]

VOICES = [("Microsoft David Desktop", "david", 0), ("Microsoft Zira Desktop", "zira", 1)]


def synthesise(out_dir: Path) -> list[dict]:
    jobs = []
    for voice, short, rate in VOICES:
        for i, text in enumerate(SENTENCES):
            uid = f"{short}_{i:02d}"
            jobs.append(
                {
                    "id": uid,
                    "text": text,
                    "voice": voice,
                    "rate": rate,
                    "wav": str((out_dir / f"{uid}.wav").resolve()),
                }
            )
    jobs_path = out_dir / "jobs.json"
    events_path = out_dir / "events.json"
    jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PS1),
        "-Jobs",
        str(jobs_path),
        "-Out",
        str(events_path),
    ]
    subprocess.run(cmd, check=True)
    events = json.loads(events_path.read_text(encoding="utf-8-sig"))
    by_id = {e["id"]: e for e in events}
    return [{**j, "events": by_id[j["id"]]["words"]} for j in jobs]


def build_item(job: dict) -> tuple[dict, float]:
    """Ground truth for one utterance, and the first-word timing check (event - waveform)."""
    x, sr = read_wav(job["wav"])
    duration = len(x) / sr
    speech_end = last_sound(x, sr)
    ref = tokenize(job["text"])
    events = [e for e in job["events"] if normalize_word(e["text"])]
    if [normalize_word(e["text"]) for e in events] != ref:
        raise RuntimeError(f"{job['id']}: SAPI words {events} don't match {ref}")
    words = []
    for k, e in enumerate(events):
        t0 = e["ms"] / 1000
        t1 = events[k + 1]["ms"] / 1000 if k + 1 < len(events) else speech_end
        w = ref[k]
        t0, t1 = round(t0, 4), round(t1, 4)
        words.append({"i": k, "text": w, "t0": t0, "t1": t1, "deictic": w in DEMONSTRATIVES})
    check = words[0]["t0"] - first_sound(x, sr)
    item = {
        "id": job["id"],
        "voice": job["voice"],
        "rate": job["rate"],
        "text": job["text"],
        "wav": Path(job["wav"]).name,
        "duration": round(duration, 4),
        "words": words,
    }
    return item, check


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / ".cache" / "r2")
    args = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("needs Windows (System.Speech)")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = synthesise(args.out)
    items, checks = [], []
    for job in jobs:
        item, check = build_item(job)
        items.append(item)
        checks.append(check * 1000)
    n_deictic = sum(w["deictic"] for it in items for w in it["words"])
    meta = {
        "description": "Synthetic R2 test set: Windows SAPI voices, word starts from "
        "SpeakProgress events. t0/t1 in seconds from the start of the WAV; t1 is the next "
        "word's start (or the end of speech), so it includes any pause.",
        "generator": "scripts/make_test_utterances.py",
        "sample_rate": 16000,
        "first_word_check_ms": {
            "what": "SpeakProgress time of word 0 minus first sound in the waveform (-35 dB)",
            "median": round(statistics.median(checks), 1),
            "min": round(min(checks), 1),
            "max": round(max(checks), 1),
        },
        "n_utterances": len(items),
        "n_demonstratives": n_deictic,
    }
    GROUND_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH.write_text(json.dumps({"meta": meta, "items": items}, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(f"wrote {GROUND_TRUTH.relative_to(ROOT)} and {len(items)} WAVs in {args.out}")


if __name__ == "__main__":
    main()
