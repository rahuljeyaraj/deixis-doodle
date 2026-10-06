"""Plotter numbers straight from the live app's `timeline.jsonl` (measure list M04 to M09).

- **M04** last word to the robot's guess: the end of the last word heard before a preview, to
  the plotter starting to trace it (`plotter` job_start "preview").
- **M05** hand before word: for each pointing word ("here", "there", "this", "that", but not
  "this big"), the hold that overlaps it (else the nearest one), with no time window; did the
  hold start before the word, and by how much.
- **M06** draw time per object: the drawing's job_start to the park's job_end after it.
- **M07** a take's length, from the log: the first word to the last park (the phone video is
  the real measure; this is a cross-check).
- **M09** "a bit bigger" during a preview: each utterance with "bigger" (or "larger") heard
  while a guess was showing; it worked if an `adjust` with scale > 1 and a new preview of the
  same item followed within 10 s.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

POINT_WORDS = {"here", "there", "this", "that"}
SIZE_AFTER = {"big", "long", "wide", "tall", "large", "small", "size"}
BIGGER = {"bigger", "larger"}
GUESS_STATES = {"PREVIEW", "ADJUST"}


def load(run: Path) -> list[dict]:
    lines = (Path(run) / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ev = [json.loads(x) for x in lines if x.strip()]
    for e in ev:  # live runs log plotter events as kind "plotter", event "job_start"
        if e["kind"] == "plotter" and "event" in e:
            e["kind"] = f"plotter.{e['event']}"
    return sorted(ev, key=lambda e: e["t"])


def _norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha())


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    return {
        "n": len(s),
        "median": round(statistics.median(s), 3),
        "p90": round(p90, 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


# ---------------------------------------------------------------- M04


def guess_latency(ev: list[dict]) -> list[dict]:
    """Per preview: last word heard before it, to the trace starting."""
    out = []
    utts = [e for e in ev if e["kind"] == "utterance" and e.get("words")]
    for p in (e for e in ev if e["kind"] == "preview"):
        u = [x for x in utts if x["t"] <= p["t"] + 1e-6]
        if not u:
            continue
        start = next(
            (
                x["t"]
                for x in ev
                if x["kind"] == "plotter.job_start"
                and x.get("label") == "preview"
                and p["t"] - 1e-6 <= x["t"] < p["t"] + 10
            ),
            p["t"],
        )
        end = u[-1]["words"][-1][2]
        out.append({"t": p["t"], "said": u[-1]["text"], "s": round(start - end, 3)})
    return out


# ---------------------------------------------------------------- M05


def pointing_words(ev: list[dict]) -> list[dict]:
    out = []
    for u in (e for e in ev if e["kind"] == "utterance" and e.get("words")):
        ws = u["words"]
        for i, (w, t0, t1) in enumerate(ws):
            n = _norm(w)
            if n not in POINT_WORDS:
                continue
            nxt = _norm(ws[i + 1][0]) if i + 1 < len(ws) else ""
            if n in ("this", "that") and nxt in SIZE_AFTER:
                continue  # "this big" shows a size, not a place
            out.append({"word": n, "t0": t0, "t1": t1, "said": u["text"]})
    return out


def hand_before_word(ev: list[dict]) -> list[dict]:
    holds = [e for e in ev if e["kind"] == "hold"]
    out = []
    for w in pointing_words(ev):
        if not holds:
            break

        def overlap(h, w=w):
            return min(h["t1"], w["t1"]) - max(h["t0"], w["t0"])

        def gap(h, w=w):
            return max(h["t0"] - w["t1"], w["t0"] - h["t1"], 0.0)

        over = [h for h in holds if overlap(h) > 0]
        h = max(over, key=overlap) if over else min(holds, key=gap)
        out.append(
            {
                **w,
                "hold": [h["t0"], h["t1"]],
                "overlaps": bool(over),
                "before": h["t0"] < w["t0"],
                "lead_s": round(w["t0"] - h["t0"], 3),
            }
        )
    return out


# ---------------------------------------------------------------- M06, M07


def draw_times(ev: list[dict]) -> list[dict]:
    out = []
    for i, e in enumerate(ev):
        if e["kind"] != "plotter.job_start" or e.get("label") != "draw":
            continue
        park = next(
            (x for x in ev[i:] if x["kind"] == "plotter.job_end" and x.get("label") == "park"),
            None,
        )
        if park is None:
            continue
        label = next(
            (x.get("label") for x in ev if x["kind"] == "item" and x["id"] == e.get("tag")),
            e.get("tag"),
        )
        out.append({"id": e.get("tag"), "label": label, "s": round(park["t"] - e["t"], 2)})
    return out


def take_length(ev: list[dict]) -> float | None:
    words = [w for e in ev if e["kind"] == "utterance" for w in e.get("words") or []]
    parks = [e["t"] for e in ev if e["kind"] == "plotter.job_end" and e.get("label") == "park"]
    if not words or not parks:
        return None
    return round(max(parks) - words[0][1], 2)


# ---------------------------------------------------------------- M09


def bigger_during_guess(ev: list[dict], within_s: float = 10.0) -> list[dict]:
    out = []
    for i, e in enumerate(ev):
        if e["kind"] != "utterance" or not e.get("words"):
            continue
        toks = {_norm(w[0]) for w in e["words"]}
        if not toks & BIGGER:
            continue
        said_at = e["words"][0][1]
        st = _state_at(ev, said_at)
        if st not in GUESS_STATES:
            continue
        later = [x for x in ev[i + 1 :] if x["t"] <= e["t"] + within_s]
        adj = next((x for x in later if x["kind"] == "adjust" and x.get("scale", 1) > 1), None)
        again = adj is not None and any(
            x["kind"] == "preview" and adj["id"] in x.get("items", []) and x["t"] >= adj["t"]
            for x in later
        )
        out.append({"said": e["text"], "t": e["t"], "state": st, "worked": bool(again)})
    return out


def _state_at(ev: list[dict], t: float) -> str:
    st = "LISTEN"
    for e in ev:
        if e["t"] > t:
            break
        if e["kind"] == "state":
            st = e["state"]
    return st


# ---------------------------------------------------------------- all together


def summary(runs: list[Path]) -> dict:
    lat, hbw, draws, big, takes = [], [], [], [], []
    for r in runs:
        ev = load(r)
        lat += guess_latency(ev)
        hbw += hand_before_word(ev)
        draws += draw_times(ev)
        big += bigger_during_guess(ev)
        takes.append({"run": str(r), "s": take_length(ev)})
    n_before = sum(x["before"] for x in hbw)
    by_label: dict[str, list[float]] = {}
    for d in draws:
        by_label.setdefault(str(d["label"]), []).append(d["s"])
    res = {
        "runs": [str(r) for r in runs],
        "M04_last_word_to_guess_s": _stats([x["s"] for x in lat]),
        "M05_hand_before_word": {
            "words": len(hbw),
            "hold_started_first": n_before,
            "share": round(n_before / len(hbw), 3) if hbw else None,
            "median_lead_s": round(statistics.median(x["lead_s"] for x in hbw), 3) if hbw else None,
        },
        "M06_draw_time_s": {k: _stats(v) for k, v in sorted(by_label.items())},
        "M07_take_length_s": takes,
        "M09_bigger": {"tries": len(big), "worked": sum(x["worked"] for x in big)},
        "detail": {"M04": lat, "M05": hbw, "M06": draws, "M09": big},
    }
    return res


def lines(res: dict) -> list[str]:
    """The summary in plain words, one line per number."""
    out = []
    m4 = res["M04_last_word_to_guess_s"]
    if m4["n"]:
        out.append(
            f"M04 last word to the guess: median {m4['median']:.2f} s, "
            f"range {m4['min']:.2f} to {m4['max']:.2f} s, {m4['n']} commands"
        )
    m5 = res["M05_hand_before_word"]
    if m5["words"]:
        out.append(
            f"M05 hand before word: {m5['hold_started_first']} of {m5['words']} pointing words "
            f"({100 * m5['share']:.0f} %), median lead {m5['median_lead_s']:.2f} s"
        )
    for k, v in res["M06_draw_time_s"].items():
        out.append(f"M06 draw time, {k}: median {v['median']:.1f} s ({v['n']} drawn)")
    for t in res["M07_take_length_s"]:
        if t["s"] is not None:
            out.append(f"M07 take length from the log: {t['s']:.1f} s ({Path(t['run']).name})")
    m9 = res["M09_bigger"]
    if m9["tries"]:
        out.append(f"M09 'a bit bigger' during the guess: worked {m9['worked']} of {m9['tries']}")
    return out
