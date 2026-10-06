"""M01, M02, M03: each drawing against the printed target it was meant for.

Input: per drawn command, the ink's centre and size (from `scripts/m2_score.py`'s before and
after photos), the core's `Where` (mean and covariance), and the sheet's targets as found in
the first parked photo (`targets.find_targets`). All in paper mm.

- **M01** (crosses): the drawn cross's centre to the printed "x" nearest it: median, 90th
  percentile. "Where you pointed vs where it drew."
- **M03** (crosses): is the printed "x" inside the 95 % ellipse of the core's `Where`?
- **M02** (circles): the drawn circle's diameter against the printed one: median absolute
  error, and how many within +-20 %.
"""

from __future__ import annotations

import math
import statistics

from pointdraw.measure.targets import Target, inside_95

MATCH_MM = 20.0  # a drawing further than this from every target isn't counted


def _p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


def match(rows: list[dict], targets: list[Target], kind: str) -> tuple[list[dict], dict]:
    """`rows`: dicts with `id`, `label`, `ink_xy`, `ink_size`, `where` (mean_mm, cov_mm2).
    Returns one result per drawing and the summary."""
    out = []
    used: set[int] = set()
    for r in rows:
        if r.get("ink_xy") is None:
            out.append({**_short(r), "counted": False, "why": "no ink found"})
            continue
        t = min(targets, key=lambda t: math.dist(r["ink_xy"], (t.x, t.y)))
        err = math.dist(r["ink_xy"], (t.x, t.y))
        res = {**_short(r), "target": t.id, "target_found": t.found, "err_mm": round(err, 2)}
        if err > MATCH_MM:
            out.append({**res, "counted": False, "why": f"no target within {MATCH_MM:.0f} mm"})
            continue
        if t.id in used:
            res["repeat"] = True  # the same target twice: counted, but said
        used.add(t.id)
        w = r.get("where") or {}
        if w.get("mean_mm") is not None:
            res["pointed_err_mm"] = round(math.dist(w["mean_mm"], (t.x, t.y)), 2)
            if w.get("cov_mm2") is not None:
                res["inside_95"] = inside_95((t.x, t.y), w["mean_mm"], w["cov_mm2"])
        if kind == "circles" and t.d and r.get("ink_size"):
            res["printed_d_mm"] = round(t.d, 2)
            res["drawn_d_mm"] = round(r["ink_size"], 2)
            res["size_err"] = round(r["ink_size"] / t.d - 1, 4)
        out.append({**res, "counted": True})
    return out, summarise(out, kind)


def _short(r: dict) -> dict:
    return {"id": r.get("id"), "label": r.get("label"), "ink_xy": r.get("ink_xy")}


def summarise(res: list[dict], kind: str) -> dict:
    c = [r for r in res if r.get("counted")]
    s: dict = {"kind": kind, "drawn": len(res), "counted": len(c)}
    if not c:
        return s
    errs = [r["err_mm"] for r in c]
    s["centre_median_mm"] = round(statistics.median(errs), 2)
    s["centre_p90_mm"] = round(_p90(errs), 2)
    s["centre_max_mm"] = round(max(errs), 2)
    s["within_4mm"] = sum(e <= 4.0 for e in errs)
    ins = [r["inside_95"] for r in c if "inside_95" in r]
    if ins:
        s["inside_95"] = sum(ins)
        s["inside_95_of"] = len(ins)
    se = [r["size_err"] for r in c if "size_err" in r]
    if se:
        s["size_abs_err_median"] = round(statistics.median(abs(x) for x in se), 4)
        s["size_within_20pct"] = sum(abs(x) <= 0.20 for x in se)
        s["size_of"] = len(se)
    return s


def lines(s: dict) -> list[str]:
    out = []
    if not s.get("counted"):
        return [f"{s['kind']}: nothing to score ({s['drawn']} drawn)"]
    n = s["counted"]
    if s["kind"] == "crosses":
        out.append(
            f"M01 where you pointed vs where it drew: median {s['centre_median_mm']:.1f} mm, "
            f"90th percentile {s['centre_p90_mm']:.1f} mm ({n} crosses; "
            f"{s['within_4mm']} of {n} within 4 mm)"
        )
        if "inside_95" in s:
            out.append(
                f"M03 your target inside the 95 % circle: {s['inside_95']} of {s['inside_95_of']}"
            )
    else:
        if "size_of" in s:
            out.append(
                f"M02 size shown vs drawn: median error {100 * s['size_abs_err_median']:.0f} %, "
                f"{s['size_within_20pct']} of {s['size_of']} within +-20 %"
            )
        out.append(f"   (circle centres: median {s['centre_median_mm']:.1f} mm from the printed)")
    if s["counted"] < s["drawn"]:
        out.append(f"   {s['drawn'] - s['counted']} drawing(s) not counted: see the JSON")
    return out
