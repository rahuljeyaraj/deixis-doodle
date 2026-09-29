"""Spike R2 benchmark: word accuracy, demonstrative timing error and latency of local ASR models.

    uv run python scripts/make_test_utterances.py     # once: synthetic WAVs + ground truth
    uv run python scripts/asr_fetch_models.py --all   # once: models
    uv run python scripts/asr_bench.py                # every downloaded model
    uv run python scripts/asr_bench.py --models parakeet_v2 --hotwords

Synthetic set: tests/audio/data/r2_ground_truth.json (WAVs in .cache/r2/).
Real set: rahul's six m4a recordings in C:\\work\\test_nebius_nvidia\\test_data (skipped if absent).
Results are printed as Markdown and saved to .cache/r2/bench_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

from pointdraw.audio.asr_local import AsrConfig, LocalAsr  # noqa: E402
from pointdraw.audio.models import ASR_MODELS  # noqa: E402
from pointdraw.audio.scoring import (  # noqa: E402
    matched_pairs,
    normalize_word,
    tokenize,
    word_error_rate,
)
from pointdraw.audio.wavio import load_audio, read_wav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "tests" / "audio" / "data" / "r2_ground_truth.json"
REAL_DIR = Path(r"C:\work\test_nebius_nvidia\test_data")
# From C:\work\test_nebius_nvidia\RESULTS.md, step 4.
REAL_TRANSCRIPTS = {
    "adullt1.m4a": "draw a dinosaur",
    "adullt2.m4a": "add a rabbit here",
    "adullt3.m4a": "make it bigger",
    "adullt4.m4a": "draw birds in the sky",
    "adullt5.m4a": "add leaves to the tree",
    "adult6.m4a": "draw a lion",
}
BOOSTED = {"nemotron", "cosmos", "nebius"}
STREAM_BLOCK_S = 0.1


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def prepare(
    x: np.ndarray, sr: int, lead_s: float, noise_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Mimic what the segmenter hands the ASR: pre-roll before speech, and a mic noise floor."""
    x = np.concatenate([np.zeros(int(lead_s * sr), np.float32), x])
    if noise_db > -120:
        x = x + rng.normal(0, 10 ** (noise_db / 20), len(x)).astype(np.float32)
    return x


def summarise(errs_s: list[float]) -> dict:
    """Signed errors (hyp - truth, s) -> ms stats of the signed and absolute error."""
    ms = [e * 1000 for e in errs_s]
    a = [abs(e) for e in ms]
    return {
        "n": len(ms),
        "mean_signed": statistics.fmean(ms) if ms else float("nan"),
        "median_signed": statistics.median(ms) if ms else float("nan"),
        "mean_abs": statistics.fmean(a) if a else float("nan"),
        "median_abs": statistics.median(a) if a else float("nan"),
        "p90_abs": pct(a, 90),
        "max_abs": max(a) if a else float("nan"),
    }


def run_model(key: str, args: argparse.Namespace, gt: dict) -> dict:
    cfg = AsrConfig(model=key, num_threads=args.threads, use_hotwords=args.hotwords, lag_s=0.0)
    asr = LocalAsr(cfg)
    items = gt["items"]
    warm, _ = read_wav(ROOT / ".cache" / "r2" / items[0]["wav"])
    asr.transcribe(warm)  # warm-up; not timed

    refs, hyps, rows = [], [], []
    dem_errs: dict[str, list[float]] = {}  # voice -> signed errors (s)
    all_errs: list[float] = []
    dem_records: list[dict] = []
    lat, stream_lat, rtf = [], [], []
    boosted_total = boosted_ok = dem_total = dem_found = 0
    rng = np.random.default_rng(0)
    for it in items:
        x, sr = read_wav(ROOT / ".cache" / "r2" / it["wav"])
        x = prepare(x, sr, args.lead_s, args.noise_db, rng)
        t = time.perf_counter()
        words = asr.transcribe(x, sr, -args.lead_s)  # times relative to the original WAV
        dt = time.perf_counter() - t
        lat.append(dt)
        rtf.append(dt / it["duration"])
        if asr.streaming:
            s = asr.stream(-args.lead_s)
            n = int(STREAM_BLOCK_S * sr)
            for k in range(0, len(x), n):
                s.accept(x[k : k + n], sr)
            t = time.perf_counter()
            s.finish()
            stream_lat.append(time.perf_counter() - t)

        ref = [w["text"] for w in it["words"]]
        hyp = [normalize_word(w.text) for w in words]
        keep = [i for i, h in enumerate(hyp) if h]
        hyp = [hyp[i] for i in keep]
        hw = [words[i] for i in keep]
        refs.append(ref)
        hyps.append(hyp)
        pairs = matched_pairs(ref, hyp)
        voice = it["voice"].split()[1].lower()
        for i, gw in enumerate(it["words"]):
            if gw["text"] in BOOSTED:
                boosted_total += 1
                boosted_ok += i in pairs
            if i not in pairs:
                if gw["deictic"]:
                    dem_total += 1
                continue
            e = hw[pairs[i]].t0 - gw["t0"]
            all_errs.append(e)
            if gw["deictic"]:
                dem_total += 1
                dem_found += 1
                dem_errs.setdefault(voice, []).append(e)
                dem_records.append(
                    {"id": it["id"], "i": i, "word": gw["text"], "err_ms": round(e * 1000)}
                )
        rows.append({"id": it["id"], "ref": " ".join(ref), "hyp": " ".join(hyp)})

    all_dem = [e for v in dem_errs.values() for e in v]
    lag = statistics.median(all_dem) if all_dem else 0.0
    # Lag learned on one voice, applied to the other: an honest estimate after calibration.
    cross: list[float] = []
    voices = list(dem_errs)
    if len(voices) == 2:
        for a, b in (voices, voices[::-1]):
            la = statistics.median(dem_errs[a])
            cross += [e - la for e in dem_errs[b]]

    real = {}
    if REAL_DIR.exists():
        r_refs, r_hyps, r_lat = [], [], []
        for name, text in REAL_TRANSCRIPTS.items():
            p = REAL_DIR / name
            if not p.exists():
                continue
            x = load_audio(p)
            t = time.perf_counter()
            words = asr.transcribe(x, 16000, 0.0)
            r_lat.append(time.perf_counter() - t)
            r_refs.append(tokenize(text))
            r_hyps.append([h for h in (normalize_word(w.text) for w in words) if h])
            real[name] = {"ref": text, "hyp": " ".join(r_hyps[-1]), "dur": len(x) / 16000}
        if r_refs:
            real["_wer"] = word_error_rate(r_refs, r_hyps)
            real["_sentence_acc"] = sum(r == h for r, h in zip(r_refs, r_hyps, strict=True)) / len(
                r_refs
            )
            real["_latency_median_s"] = statistics.median(r_lat)

    res = {
        "model": key,
        "hotwords": args.hotwords,
        "threads": args.threads,
        "lead_s": args.lead_s,
        "noise_db": args.noise_db,
        "load_s": asr.load_s,
        "wer": word_error_rate(refs, hyps),
        "sentence_acc": sum(r == h for r, h in zip(refs, hyps, strict=True)) / len(refs),
        "boosted_words": f"{boosted_ok}/{boosted_total}",
        "demonstratives_found": f"{dem_found}/{dem_total}",
        "dem_timing_ms": summarise(all_dem),
        "dem_timing_lag_corrected_ms": summarise([e - lag for e in all_dem]),
        "dem_timing_cross_voice_corrected_ms": summarise(cross),
        "all_words_timing_ms": summarise(all_errs),
        "lag_s": lag,
        "latency_batch_s": {"median": statistics.median(lat), "p90": pct(lat, 90), "max": max(lat)},
        "rtf_median": statistics.median(rtf),
        "latency_stream_finish_s": (
            {
                "median": statistics.median(stream_lat),
                "p90": pct(stream_lat, 90),
                "max": max(stream_lat),
            }
            if stream_lat
            else None
        ),
        "real": real,
        "errors": [r for r in rows if r["ref"] != r["hyp"]],
        "demonstratives": dem_records,
    }
    return res


def markdown(results: list[dict]) -> str:
    h = (
        "| Model | Hotwords | WER synth | Sent. acc | Boosted words | Dem. found | "
        "Dem. start err median / p90 (ms, raw) | Lag (ms) | After lag corr. median / p90 "
        "(cross-voice) | Batch latency median / p90 (s) | RTF | Stream finish median / p90 (s) "
        "| WER real (6) | Load (s) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in results:
        d, c = r["dem_timing_ms"], r["dem_timing_cross_voice_corrected_ms"]
        s = r["latency_stream_finish_s"]
        sl = f"{s['median']:.2f} / {s['p90']:.2f}" if s else "n/a (offline)"
        real = r["real"].get("_wer")
        h += (
            f"| {r['model']} | {'on' if r['hotwords'] else 'off'} | {r['wer']:.1%} | "
            f"{r['sentence_acc']:.0%} | {r['boosted_words']} | {r['demonstratives_found']} | "
            f"{d['median_abs']:.0f} / {d['p90_abs']:.0f} | {r['lag_s'] * 1000:+.0f} | "
            f"{c['median_abs']:.0f} / {c['p90_abs']:.0f} | "
            f"{r['latency_batch_s']['median']:.2f} / {r['latency_batch_s']['p90']:.2f} | "
            f"{r['rtf_median']:.3f} | {sl} | "
            f"{'n/a' if real is None else f'{real:.1%}'} | {r['load_s']:.1f} |\n"
        )
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--models", nargs="*", help=f"default: every downloaded one of {list(ASR_MODELS)}"
    )
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hotwords", action="store_true", help="beam search with word boosting")
    ap.add_argument("--lead-s", type=float, default=0.3, help="pre-roll added before each clip")
    ap.add_argument("--noise-db", type=float, default=-60, help="noise floor, dBFS (-120 = off)")
    args = ap.parse_args()
    if not GROUND_TRUTH.exists():
        sys.exit("no ground truth; run scripts/make_test_utterances.py")
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    if not (ROOT / ".cache" / "r2" / gt["items"][0]["wav"]).exists():
        sys.exit("no WAVs in .cache/r2; run scripts/make_test_utterances.py")
    keys = args.models or [k for k, m in ASR_MODELS.items() if m.available()]
    results = []
    for k in keys:
        print(f"== {k} (hotwords {'on' if args.hotwords else 'off'})", flush=True)
        try:
            r = run_model(k, args, gt)
        except Exception as e:  # keep going: a failed candidate is a result too
            print(f"   FAILED: {type(e).__name__}: {e}")
            continue
        results.append(r)
        suffix = "_hw" if args.hotwords else ""
        out = ROOT / ".cache" / "r2" / f"bench_{k}{suffix}.json"
        out.write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in r.items() if k != "errors"}, indent=1))
        for e in r["errors"][:12]:
            print(f"   {e['id']}: ref={e['ref']!r} hyp={e['hyp']!r}")
    print()
    print(markdown(results))


if __name__ == "__main__":
    main()
