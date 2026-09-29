"""Live microphone demo: speak, and see each utterance's words with their timestamps.

    uv run python scripts/asr_live.py --list          # input devices
    uv run python scripts/asr_live.py                 # default mic, default model
    uv run python scripts/asr_live.py --device 12 --model nemotron_en --vad energy

Times are shown relative to when the demo started (on the shared clock), plus how long
recognition took after you stopped speaking. Ctrl+C to quit. Exits cleanly without a mic.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

from pointdraw import clock  # noqa: E402
from pointdraw.audio.asr_local import AsrConfig, LocalAsr  # noqa: E402
from pointdraw.audio.capture import AudioCapture, list_input_devices  # noqa: E402
from pointdraw.audio.models import ASR_MODELS, DEFAULT_ASR  # noqa: E402
from pointdraw.audio.scoring import DEMONSTRATIVES, normalize_word  # noqa: E402
from pointdraw.audio.segmenter import Segmenter  # noqa: E402
from pointdraw.audio.vad import make_vad  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--device", default=None, help="input device index or name")
    ap.add_argument("--model", default=DEFAULT_ASR, choices=list(ASR_MODELS))
    ap.add_argument("--vad", default="silero", choices=["silero", "energy"])
    ap.add_argument("--hotwords", action="store_true", help="boost Nemotron/Cosmos/Nebius")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = never)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        devices = list_input_devices()
    except Exception as e:  # PortAudio missing or broken
        print(f"No audio system available ({e}).")
        return 0
    if args.list or not devices:
        if not devices:
            print("No microphone found. Plug one in and try again.")
        for d in devices:
            print(
                f"{d['index']:3d}  {d['name']}  [{d['hostapi']}, {d['default_samplerate']:.0f} Hz]"
            )
        return 0

    try:
        asr = LocalAsr(AsrConfig(model=args.model, use_hotwords=args.hotwords))
    except FileNotFoundError as e:
        print(e)
        return 1
    seg = Segmenter(make_vad(args.vad))
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    t_origin = clock.now()
    print(f"Model {args.model} loaded in {asr.load_s:.1f} s. Speak; Ctrl+C to stop.\n")
    try:
        with AudioCapture(device=device) as cap:
            for samples, t0 in cap.blocks(timeout=3.0):
                for utt in seg.push(samples, t0):
                    t = time.perf_counter()
                    words = asr.transcribe(utt.samples, utt.sample_rate, utt.t_start)
                    dt = time.perf_counter() - t
                    delay = clock.now() - utt.t_end
                    print(
                        f"[{utt.t_speech - t_origin:7.2f} s] {utt.duration:.1f} s audio, "
                        f"ASR {dt * 1000:.0f} ms, words ready {delay * 1000:.0f} ms after the end"
                    )
                    for i, w in enumerate(words):
                        mark = " <- deictic" if normalize_word(w.text) in DEMONSTRATIVES else ""
                        print(
                            f"   {i:2d} {w.text:<14} {w.t0 - t_origin:7.2f} - "
                            f"{w.t1 - t_origin:7.2f}{mark}"
                        )
                    print(" ".join(w.text for w in words) or "(nothing recognised)")
                    print()
                if args.seconds and clock.now() - t_origin > args.seconds:
                    break
            if cap.overflows or cap.dropped:
                print(f"input overflows: {cap.overflows}, dropped blocks: {cap.dropped}")
    except KeyboardInterrupt:
        pass
    except (RuntimeError, TimeoutError) as e:
        print(f"Microphone problem: {e}")
        return 0
    except Exception as e:  # sounddevice.PortAudioError and friends
        print(f"Could not use the microphone: {type(e).__name__}: {e}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
