"""Score a virtual run against its scene's `expect` block, plus the checks every scene gets.

Pass rules (virtual-test-plan section 4, plan 4.5):
- S1 place error <= 1 mm (virtual, so anything above is a bug in our maths); drawings
  within 5 mm of the intended centre and within +-20 % of the intended size.
- The right tool calls. Questions asked when expected, and only then.
- Always: no motion starts while a hand is over the paper; a pause stops the pen within
  1.7 s and 50 mm; every stroke inside the drawable area; only whitelisted G-code.
- Stop: the plotter halts within 1.7 s and 50 mm of the stop command, and within 3 s of
  the word "stop" being spoken (end of speech + VAD + ASR included).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from deixis.geometry import Box
from pointdraw.controller.safety import check_line
from pointdraw.virtual.scenefile import DrawnExpect

PAUSE_MAX_S = 1.7  # CTRL: pause or stop within ~50 mm, ~1.7 s without EMERGENCY_PARSER
PAUSE_MAX_MM = 50.0
STOP_SPOKEN_MAX_S = 3.0
# A hovering finger is seen shifted by parallax (VIS report: about 135 mm x h / 370 mm), and
# no height correction is built yet, so hover runs use the plan's general 5 mm rule.
HOVER_TOL_MM = 5.0


def _check(out: list, name: str, ok: bool, detail: str = "", **extra: Any) -> None:
    out.append({"name": name, "ok": bool(ok), "detail": detail, **extra})


def _box(v: list[float] | None) -> Box | None:
    return Box(*v) if v else None


def _dist_to_box(p: tuple[float, float], b: Box) -> float:
    dx = max(b.x - p[0], 0.0, p[0] - b.x2)
    dy = max(b.y - p[1], 0.0, p[1] - b.y2)
    return math.hypot(dx, dy)


def score(res: Any) -> dict[str, Any]:
    spec, log, plotter, rt = res.spec, res.log, res.plotter, res.runtime
    ex = spec.expect
    checks: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    tool_events = log.of("tool_calls")
    calls = [c["name"] for e in tool_events for c in e["calls"]]
    sources = sorted({e["source"] for e in tool_events})
    drawn = log.of("drawn")
    asks = log.of("ask")
    states = [e["state"] for e in log.of("state")]

    # --- tool calls (a subsequence, in order) --------------------------------------------
    if ex.calls:
        it = iter(calls)
        ok = all(any(c == want for c in it) for want in ex.calls)
        _check(checks, "calls", ok, f"expected {ex.calls} in order; got {calls}")

    # --- drawings ------------------------------------------------------------------------
    objects = {o.id: o for o in rt.scene.objects}
    if ex.drawn is not None:
        _check(
            checks,
            "drawn_count",
            len(drawn) == len(ex.drawn),
            f"{len(drawn)} drawn, expected {len(ex.drawn)}",
            value=len(drawn),
            limit=len(ex.drawn),
        )
        errors = []
        for i, want in enumerate(ex.drawn):
            if i >= len(drawn):
                break
            got = drawn[i]
            errors.append(
                _score_drawn(checks, i, want, got, drawn, log, objects, res.cfg.hover_mm > 0)
            )
        errs = [e for e in errors if e is not None]
        if errs:
            metrics["center_error_mm"] = {
                "median": round(statistics.median(errs), 3),
                "max": round(max(errs), 3),
            }

    # --- questions ------------------------------------------------------------------------
    if ex.asks is not None:
        _check(
            checks,
            "asks",
            len(asks) == ex.asks,
            f"{len(asks)} asked, expected {ex.asks}: " + "; ".join(a["question"] for a in asks),
            value=len(asks),
            limit=ex.asks,
        )
    for s in ex.ask_contains:
        ok = any(s.lower() in a["question"].lower() for a in asks)
        _check(checks, f"ask_contains:{s}", ok, "; ".join(a["question"] for a in asks))
    for s in ex.ask_options:
        opts = [o.lower() for a in asks for o in a.get("options", [])]
        _check(checks, f"ask_option:{s}", s.lower() in opts, f"options {opts}")

    # --- states ---------------------------------------------------------------------------
    if ex.states:
        it2 = iter(["LISTEN", *states])
        ok = all(any(s == want for s in it2) for want in ex.states)
        _check(checks, "states", ok, f"expected {ex.states} in order; got {states}")

    # --- look and replies ----------------------------------------------------------------
    looks = log.of("look")
    for s in ex.look_contains:
        ok = any(s.lower() in e["answer"].lower() for e in looks)
        _check(checks, f"look_contains:{s}", ok, "; ".join(e["answer"] for e in looks))
    says = [e["text"] for e in log.of("say")]
    for s in ex.say_contains:
        _check(checks, f"say_contains:{s}", any(s.lower() in x.lower() for x in says), str(says))

    # --- scene reset ---------------------------------------------------------------------
    if ex.scene_empty:
        n = len(rt.scene.objects)
        _check(checks, "scene_empty", n == 0 and bool(log.of("new_sheet")), f"{n} objects left")

    pen_mm = sum(math.dist(s.a, s.b) for s in plotter.ink if s.t >= 0)
    metrics["pen_mm"] = round(pen_mm, 1)
    if ex.min_pen_mm is not None:
        _check(
            checks,
            "min_pen_mm",
            pen_mm >= ex.min_pen_mm,
            f"{pen_mm:.0f} mm of ink",
            value=round(pen_mm, 1),
            limit=ex.min_pen_mm,
        )
    if ex.max_pen_mm is not None:
        _check(
            checks,
            "max_pen_mm",
            pen_mm <= ex.max_pen_mm,
            f"{pen_mm:.0f} mm of ink",
            value=round(pen_mm, 1),
            limit=ex.max_pen_mm,
        )

    # --- always: safety ------------------------------------------------------------------
    _interlock_checks(res, checks, metrics, require_pause=ex.pause)
    _stop_checks(res, checks, metrics, require=ex.stop)
    cfg = plotter.config
    d = cfg.drawable
    outside = [
        s for s in plotter.ink if s.t >= 0 and not all(d.contains(p, -0.06) for p in (s.a, s.b))
    ]
    _check(checks, "ink_inside_drawable", not outside, f"{len(outside)} segments outside")
    bad = []
    for line in plotter.sent:
        try:
            check_line(line, allowed=None, config=cfg)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{line}: {e}")
    _check(checks, "gcode_whitelist", not bad, "; ".join(bad[:3]))
    if log.of("timeout"):
        _check(checks, "finished", False, f"hit max_time {spec.max_time} s")

    metrics.update(_latencies(res))
    metrics["sim_s"] = round(res.t_end, 2)
    metrics["wall_s"] = round(res.wall_s, 3)
    metrics["agent_sources"] = sources
    vision = getattr(res.runtime, "vision", None)
    if getattr(vision, "latencies", None):
        lat = sorted(vision.latencies)
        metrics["cosmos_calls"] = vision.calls
        metrics["cosmos_s"] = round(statistics.median(lat), 3)
        metrics["cosmos_max_s"] = round(lat[-1], 3)
    report = {
        "scene": spec.id,
        "title": spec.title,
        "variant": res.cfg.variant,
        "config": {
            "ai": res.cfg.ai,
            "cosmos": res.cfg.cosmos,
            "voice": res.cfg.voice,
            "model": res.cfg.model,
        },
        "pass": all(c["ok"] for c in checks),
        "checks": checks,
        "metrics": metrics,
        "calls": [
            {"utt": e["utt"], "source": e["source"], "calls": e["calls"]} for e in tool_events
        ],
        "utterances": [e["text"] for e in log.of("utterance")],
        "asks": [{"question": a["question"], "options": a.get("options", [])} for a in asks],
        "drawn": [
            {k: e.get(k) for k in ("id", "type", "label", "text", "ink_box", "ink_center")}
            for e in drawn
        ],
        "says": says,
    }
    if getattr(vision, "replies", None):
        report["cosmos"] = {
            "labels": [
                {"id": o["id"], "label": o["label"]}
                for e in log.of("scene")
                for o in e.get("new", [])
            ],
            "replies": vision.replies,
        }
    return report


def _score_drawn(
    checks: list,
    i: int,
    want: DrawnExpect,
    got: dict,
    drawn: list,
    log: Any,
    objects: dict,
    hover: bool = False,
) -> float | None:
    tag = f"drawn[{i}]"
    if want.kind != got["type"]:
        _check(checks, f"{tag}.kind", False, f"{got['type']} != {want.kind}")
    if want.label is not None and got.get("label") != want.label:
        _check(checks, f"{tag}.label", False, f"{got.get('label')} != {want.label}")
    if want.text is not None and (got.get("text") or "").lower() != want.text.lower():
        _check(checks, f"{tag}.text", False, f"{got.get('text')!r} != {want.text!r}")
    box = _box(got.get("ink_box"))
    if box is None:
        _check(checks, f"{tag}.ink", False, "no ink")
        return None
    c = box.center
    err = None
    if want.center is not None:
        err = math.dist(c, want.center)
        tol = max(want.tol_mm, HOVER_TOL_MM) if hover else want.tol_mm
        _check(
            checks,
            f"{tag}.center",
            err <= tol,
            f"{got.get('label') or got.get('text') or got['type']} at "
            f"({c[0]:.1f}, {c[1]:.1f}), want {tuple(want.center)}: {err:.2f} mm"
            + (" (hovering: no parallax correction yet)" if hover else ""),
            value=round(err, 3),
            limit=tol,
        )
    if want.size is not None:
        size = max(box.w, box.h)
        rel = abs(size - want.size) / want.size
        _check(
            checks,
            f"{tag}.size",
            rel <= want.size_tol,
            f"{size:.1f} mm, want {want.size:g} mm (+-{want.size_tol:.0%})",
            value=round(size, 2),
            limit=want.size,
        )

    def find(label: str) -> Box | None:
        for e in log.of("drawn", "preloaded"):
            if (e.get("label") or e.get("text") or "") == label:
                return _box(e.get("ink_box") or e.get("box"))
        # the user's own marks by their scene-file label, not the vision model's word for them
        # (live Cosmos may call the box a "square")
        for e in log.of("user_mark"):
            if e.get("label") == label:
                return _box(e.get("box"))
        for o in objects.values():
            if o.label == label:
                return o.box
        return None

    if want.below is not None:
        b = find(want.below)
        ok = b is not None and box.y2 <= b.y + 1.0 and abs(c[0] - b.center[0]) <= 5.0
        _check(checks, f"{tag}.below", ok, f"label box {box} vs {want.below} {b}")
    if want.inside is not None:
        b = find(want.inside)
        ok = (
            b is not None
            and b.contains((box.x, box.y), -0.5)
            and b.contains((box.x2, box.y2), -0.5)
        )
        _check(checks, f"{tag}.inside", ok, f"{box} inside {want.inside} {b}")
    if want.between is not None:
        a, b = find(want.between[0]), find(want.between[1])
        ok = False
        if a is not None and b is not None:
            mid = ((a.center[0] + b.center[0]) / 2, (a.center[1] + b.center[1]) / 2)
            ok = math.dist(mid, c) <= 5.0
        _check(checks, f"{tag}.between", ok, f"centre {c} between {want.between}")
    if want.near is not None:
        b = find(want.near)
        ok = b is not None and _gap(box, b) <= want.near_mm and not _overlaps(box, b)
        _check(checks, f"{tag}.near", ok, f"{box} near {want.near} {b}")
    if want.from_ is not None or want.to is not None:
        ends = [x for x in (want.from_, want.to) if x]
        ok = True
        for lab in ends:
            b = find(lab)
            ok = ok and b is not None and _dist_to_box(_nearest_end(got, b), b) <= 6.0
        _check(checks, f"{tag}.ends", ok, f"arrow {box} between {ends}")
    if want.empty:
        others = [
            _box(e.get("ink_box") or e.get("box"))
            for e in log.of("drawn", "preloaded")
            if e.get("id") != got.get("id")
        ]
        others += [o.box for o in objects.values() if o.source == "user"]
        hit = [o for o in others if o is not None and _overlaps(box, o)]
        _check(checks, f"{tag}.empty", not hit, f"overlaps {hit}")
    return err


def _nearest_end(got: dict, b: Box) -> tuple[float, float]:
    x, y, w, h = got["ink_box"]
    corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h), (x + w / 2, y + h / 2)]
    return min(corners, key=lambda p: _dist_to_box(p, b))


def _gap(a: Box, b: Box) -> float:
    """Distance between two boxes (0 if they touch or overlap)."""
    dx = max(0.0, b.x - a.x2, a.x - b.x2)
    dy = max(0.0, b.y - a.y2, a.y - b.y2)
    return math.hypot(dx, dy)


def _overlaps(a: Box, b: Box) -> bool:
    return a.x < b.x2 and b.x < a.x2 and a.y < b.y2 and b.y < a.y2


def _gt_over(res: Any, t: float) -> bool:
    from pointdraw.virtual.world import over_paper

    st = res.hand.state(t)
    p = res.plotter.config.paper
    return over_paper(st, p.w, p.h)


def _travel(res: Any, t0: float, t1: float) -> float:
    """XY distance the head moved between two times (from the plotter's track)."""
    pts = [(x, y) for t, x, y, _ in res.plotter.track if t0 <= t <= t1]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))


def _pen_down_time(res: Any, t0: float, t1: float) -> float:
    """Seconds the pen spent moving across the paper (down, XY moving) between two times.

    Pen lifts (Z only, 5 mm/s on a stock Ender 3) are left out: they take the pen away."""
    tr = [s for s in res.plotter.track if t0 <= s[0] <= t1]
    out = 0.0
    for a, b in zip(tr, tr[1:], strict=False):
        if a[3] and b[3] and (a[1], a[2]) != (b[1], b[2]):
            out += b[0] - a[0]
    return out


def _interlock_checks(res: Any, checks: list, metrics: dict, require_pause: bool) -> None:
    evs = res.plotter.events
    starts = [e for e in evs if e.kind in ("job_start", "resumed")]
    bad = [e for e in starts if _gt_over(res, e.t)]
    _check(
        checks,
        "no_motion_with_hand_over_paper",
        not bad,
        "; ".join(f"{e.kind} {e.label} at {e.t:.2f}s" for e in bad),
    )
    # Ink laid while the (true) hand was over the paper, beyond the pause allowance.
    reqs = [e for e in evs if e.kind == "pause_request"]
    paused = [e for e in evs if e.kind == "paused"]
    lat = []
    for r in reqs:
        p = next((e for e in paused if e.t >= r.t), None)
        if p is None:
            continue
        lat.append((p.t - r.t, _travel(res, r.t, p.t), _pen_down_time(res, r.t, p.t)))
    if lat:
        metrics["pause_latency_s"] = round(max(x for x, _, _ in lat), 3)
        metrics["pause_pen_down_s"] = round(max(d for _, _, d in lat), 3)
        metrics["pause_travel_mm"] = round(max(m for _, m, _ in lat), 1)
        ok = all(d <= PAUSE_MAX_S and m <= PAUSE_MAX_MM for _, m, d in lat)
        _check(
            checks,
            "pause_within_limit",
            ok,
            f"worst {metrics['pause_travel_mm']} mm and {metrics['pause_pen_down_s']} s of "
            f"pen-down motion (limit {PAUSE_MAX_MM:g} mm, {PAUSE_MAX_S} s); "
            f"{metrics['pause_latency_s']} s until fully paused, pen lifts included",
        )
    if require_pause:
        resumed = [e for e in evs if e.kind == "resumed"]
        _check(
            checks,
            "paused_and_resumed",
            bool(paused) and bool(resumed),
            f"{len(paused)} pauses, {len(resumed)} resumes",
        )


def _stop_checks(res: Any, checks: list, metrics: dict, require: bool) -> None:
    evs = res.plotter.events
    reqs = [e for e in evs if e.kind == "stop_request"]
    halts = [e for e in evs if e.kind == "halted"]
    if require:
        _check(checks, "stopped", bool(reqs) and bool(halts), f"{len(reqs)} stop requests")
    if not reqs or not halts:
        return
    r = reqs[0]
    h = next((e for e in halts if e.t >= r.t), None)
    if h is None:
        return
    lat = h.t - r.t
    mm = _travel(res, r.t, h.t)
    metrics["stop_latency_s"] = round(lat, 3)
    metrics["stop_travel_mm"] = round(mm, 1)
    _check(
        checks,
        "stop_within_limit",
        lat <= PAUSE_MAX_S and mm <= PAUSE_MAX_MM,
        f"{lat:.2f} s, {mm:.1f} mm after the stop command",
    )
    # From the spoken word "stop" to the halt.
    for u in res.log.of("utterance"):
        words = [w for w in u["words"] if w[0].lower().strip("!.,") in ("stop", "halt", "freeze")]
        if words and u["t"] <= r.t + 1e-6:
            spoken = h.t - words[0][1]
            metrics["stop_from_word_s"] = round(spoken, 3)
            _check(
                checks,
                "stop_from_word",
                spoken <= STOP_SPOKEN_MAX_S,
                f"{spoken:.2f} s from the word to the halt (limit {STOP_SPOKEN_MAX_S} s)",
            )


def _latencies(res: Any) -> dict[str, Any]:
    """Per utterance: end of speech -> transcript -> tool calls -> preview starts -> pen down."""
    log = res.log
    out: dict[str, list[float]] = {"asr_s": [], "agent_s": [], "to_preview_s": [], "to_pen_s": []}
    evs = res.plotter.events
    for u in log.of("utterance"):
        if not u["words"]:
            continue
        end = u["words"][-1][2]
        out["asr_s"].append(u["t"] - end)
        tc = next((e for e in log.of("tool_calls") if e["utt"] == u["id"]), None)
        if tc is not None:
            out["agent_s"].append(tc["latency_s"])
        prev = next(
            (
                e
                for e in evs
                if e.kind == "job_start" and str(e.tag).startswith("preview") and e.t >= u["t"]
            ),
            None,
        )
        if prev is not None:
            out["to_preview_s"].append(prev.t - end)
        draw = next(
            (e for e in evs if e.kind == "job_start" and e.label == "draw" and e.t >= u["t"]), None
        )
        if draw is not None:
            out["to_pen_s"].append(draw.t - end)
    return {k: round(statistics.median(v), 3) for k, v in out.items() if v}
