"""Does the Cosmos endpoint take video? One 4 fps clip, one request (build list item 5).

    uv run python scripts/cosmos_video_check.py clip                 # free: makes the clip
    uv run python scripts/cosmos_video_check.py ask out/cosmos/clip-4fps.mp4   # one request

`clip` makes a short 4 fps MP4 from a live run's saved window pictures (the 2 Oct M2 run by
default, around rahul's first "draw a cat here, this big"). Those pictures were saved twice a
second at 640 x 360, so each one is shown for two frames: the clip runs in real time. It is
only for checking that the endpoint accepts video; the timing test (M20) uses full-size
frames saved by the live app.

`ask` sends the clip to the endpoint (COSMOS_BASE_URL, COSMOS_API_KEY in .env) as one
`video_url` part plus a short question, and prints the answer, the time, the status and the
token counts. It never prints the key. It writes `<clip>.check.json`. If the server refuses
video, start the endpoint again with `COSMOS_PROFILE=nano-video` (video allowed explicitly)
and ask again. The endpoint must be running: this script never starts one.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import sys
import time
from pathlib import Path

import cv2

RUN = Path("out/live/20261002-154456")
FIRST_WORD_S = 75437.18  # "Draw a cat here this big": first word, last word (timeline clock)
LAST_WORD_S = 75439.66
QUESTION = (
    "This is a top-down video of a pen plotter's paper. A person points with one finger "
    "while speaking. In one sentence: when does the fingertip stop, and where on the paper?"
)


def run_t0(run: Path) -> float:
    with (run / "hands.jsonl").open(encoding="utf-8") as f:
        return float(json.loads(f.readline())["t0"])


def make_clip(run: Path, t_from: float, t_to: float, out: Path, fps: int = 4) -> int:
    """Write the run's saved pictures between t_from and t_to (timeline clock) at `fps`."""
    t0 = run_t0(run)
    pics = sorted((run / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    chosen = [p for p in pics if t_from <= t0 + int(p.stem) / 1000 <= t_to]
    if not chosen:
        raise SystemExit(f"no pictures between {t_from} and {t_to} in {run / 'frames'}")
    first = cv2.imread(str(chosen[0]))
    h, w = first.shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    # MPEG-4 part 2: OpenCV writes it everywhere (H.264 needs openh264 on Windows), and
    # vLLM's video loader reads it
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for p in chosen:
        img = cv2.imread(str(p))
        for _ in range(max(1, round(fps / 2))):  # saved at 2 a second
            writer.write(img)
            n += 1
    writer.release()
    back = cv2.VideoCapture(str(out))
    frames = int(back.get(cv2.CAP_PROP_FRAME_COUNT))
    back.release()
    if frames <= 0:
        raise SystemExit(f"the clip {out} can't be read back")
    return n


def ask(clip: Path, timeout: float = 120.0) -> dict:
    from pointdraw.config import get_settings
    from pointdraw.cosmos.client import _fallback_token, normalise_base_url

    s = get_settings()
    url = normalise_base_url(s.cosmos_base_url)
    key = s.cosmos_api_key.get_secret_value() or _fallback_token()
    if not url or not key:
        raise SystemExit("COSMOS_BASE_URL and COSMOS_API_KEY must be in .env (see scripts/nebius)")
    from openai import APIStatusError, OpenAI

    client = OpenAI(base_url=url, api_key=key, timeout=timeout, max_retries=0)
    data = base64.b64encode(clip.read_bytes()).decode("ascii")
    content = [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}},
        {"type": "text", "text": QUESTION},
    ]
    model = s.cosmos_model or "nvidia/Cosmos3-Nano"
    t = time.perf_counter()
    res: dict = {"when": dt.datetime.now().isoformat(timespec="seconds"), "model": model,
                 "clip": str(clip), "clip_bytes": clip.stat().st_size}  # fmt: skip
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}], max_tokens=200
        )
        res |= {
            "accepted": True,
            "status": 200,
            "seconds": round(time.perf_counter() - t, 2),
            "answer": (r.choices[0].message.content or "").strip(),
            "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
            "completion_tokens": getattr(r.usage, "completion_tokens", None),
        }
    except APIStatusError as e:
        res |= {
            "accepted": False,
            "status": e.status_code,
            "seconds": round(time.perf_counter() - t, 2),
            "error": str(e)[:500],
        }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clip", help="make the 4 fps clip (free)")
    c.add_argument("--run", type=Path, default=RUN)
    c.add_argument("--from", dest="t_from", type=float, default=FIRST_WORD_S - 2.0)
    c.add_argument("--to", dest="t_to", type=float, default=LAST_WORD_S + 2.0)
    c.add_argument("--out", type=Path, default=Path("out/cosmos/clip-4fps.mp4"))
    q = sub.add_parser("ask", help="send the clip to the running endpoint (one request)")
    q.add_argument("clip", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "clip":
        n = make_clip(a.run, a.t_from, a.t_to, a.out)
        print(f"wrote {a.out}: {n} frames at 4 fps ({n / 4:.1f} s), {a.out.stat().st_size} bytes")
        return 0
    res = ask(a.clip)
    out = a.clip.with_name(a.clip.name + ".check.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if res["accepted"]:
        print(f"ACCEPTED in {res['seconds']} s; tokens in {res['prompt_tokens']}, "
              f"out {res['completion_tokens']}")  # fmt: skip
        print(f"answer: {res['answer']}")
    else:
        print(f"REFUSED: status {res['status']}: {res['error']}")
        print("Next: delete the endpoint, create it with COSMOS_PROFILE=nano-video, ask again.")
    print(f"wrote {out}")
    return 0 if res["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
