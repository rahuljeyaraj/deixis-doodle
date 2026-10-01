"""Centre-line trace line-art images into library drawings, and prove the tracer on ground truth.

    # the proof: render every library drawing as thick lines, trace it back, measure
    uv run python scripts/trace_library.py --eval
    uv run python scripts/trace_library.py --eval --widths 4 8 14 --categories ear tree

    # generated images -> one library variant file (key_id = image file name)
    uv run python scripts/trace_library.py out/sana/ear/*.png --category ear \
        --out out/traced/ear.json

    # a whole Job output folder (manifest.json from scripts/nebius/library_job.py)
    uv run python scripts/trace_library.py --job out/sana_pilot --out-dir out/traced

--eval prints a table, writes out/trace_eval.json and contact sheets in out/trace_eval/.
Distances are in mm with each drawing scaled so its larger side is 40 mm (a typical demo size).
Targets: mean <= 0.3 mm, p95 <= 1.0 mm, double-line score < 2 % above the original's own (the
absolute score is printed too; an outline tracer scores about 90 %).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pointdraw.drawing.optimize import merge, path_length
from pointdraw.drawing.trace import (
    TraceResult,
    compare,
    degrade,
    double_line_score,
    ink_mask,
    render_strokes,
    to_library,
    trace,
)

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "library"
OUT = ROOT / "out"

DRAWING_MM = 40.0  # evaluation scale: the drawing's larger side
IMAGE_PX = 1024
MARGIN_PX = 64
TARGETS = {"mean_mm": 0.3, "p95_mm": 1.0, "double": 0.02}
DOUBLE_WINDOW_W = 1.5  # double-line window, in line widths
DOUBLE_MIN_GAP_W = 0.3  # closer than this lies on top of the other ink (a retrace), not beside it
SHEET_CATEGORIES = ["ear", "eye", "brain", "hand", "mouth", "tree", "bicycle", "house"]
UNITS = "larger side about 255, y up, bbox min corner at (0, 0)"


# ---------------------------------------------------------------------------------------------
# Ground-truth evaluation
# ---------------------------------------------------------------------------------------------


def library_drawings(categories: list[str] | None) -> list[tuple[str, int, str, list]]:
    """(category, variant, key_id, strokes) for every variant in library/quickdraw."""
    index = json.loads((LIB / "index.json").read_text(encoding="utf-8"))["categories"]
    out = []
    for cat in sorted(index):
        if categories and cat not in categories:
            continue
        data = json.loads((LIB / index[cat]["file"]).read_text(encoding="utf-8"))
        for i, v in enumerate(data["variants"]):
            out.append((cat, i, str(v["key_id"]), v["strokes"]))
    return out


def edge_trace(img: np.ndarray) -> list[list[tuple[float, float]]]:
    """What an ordinary (outline) tracer gives: the ink's contours. Used only as a baseline
    to show that the double-line score catches double lines."""
    mask = ink_mask(img).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        pts = [(float(x), float(y)) for x, y in c[:, 0, :]]
        if len(pts) > 2:
            out.append(pts + [pts[0]])
    return out


def evaluate_one(strokes: list, width: float, seed: int) -> tuple[dict, dict]:
    """Render, degrade, trace and measure one drawing at one line width."""
    clean, orig_px, _ = render_strokes(strokes, IMAGE_PX, width, MARGIN_PX)
    img = degrade(clean, seed=seed)
    t0 = time.perf_counter()
    res = trace(img)
    ms = (time.perf_counter() - t0) * 1000
    px_per_mm = (IMAGE_PX - 2 * MARGIN_PX) / DRAWING_MM
    orig = [o.tolist() for o in orig_px]
    m = compare(res.strokes, orig, px_per_mm)
    window, gap = DOUBLE_WINDOW_W * width, DOUBLE_MIN_GAP_W * width
    ink_o = sum(path_length(s) for s in orig)
    ink_t = sum(path_length(s) for s in res.strokes)
    lift_tol = 0.1 * px_per_mm  # the planner merges ends within 0.1 mm
    row = {
        **m,
        "double": double_line_score(res.strokes, window, min_gap_px=gap),
        "double_original": double_line_score(orig, window, min_gap_px=gap),
        "double_edge_tracer": double_line_score(edge_trace(img), window, min_gap_px=gap),
        "lifts": len(res.strokes),
        "lifts_original": len(merge(orig, lift_tol)),
        "ink_ratio": ink_t / ink_o if ink_o else math.nan,
        "width_est_px": res.line_width_px,
        "ms": ms,
    }
    # per drawing, double lines count only beyond what the original itself has: where two of its
    # lines run closer than 1.5 line widths, the traced copy rightly has them too
    row["double_excess"] = max(0.0, row["double"] - row["double_original"])
    row["pass"] = bool(
        row["mean_mm"] <= TARGETS["mean_mm"]
        and row["p95_mm"] <= TARGETS["p95_mm"]
        and row["double_excess"] < TARGETS["double"]
    )
    return row, {"img": img, "orig": orig, "res": res}


def summarise(rows: list[dict]) -> dict:
    def avg(k: str) -> float:
        return float(np.mean([r[k] for r in rows]))

    return {
        "drawings": len(rows),
        "width_est_px": avg("width_est_px"),
        "mean_mm": avg("mean_mm"),
        "p95_mm": avg("p95_mm"),
        "worst_p95_mm": float(max(r["p95_mm"] for r in rows)),
        "double": avg("double"),
        "worst_double": float(max(r["double"] for r in rows)),
        "double_original": avg("double_original"),
        "double_excess": avg("double_excess"),
        "double_edge_tracer": avg("double_edge_tracer"),
        "lifts": int(sum(r["lifts"] for r in rows)),
        "lifts_original": int(sum(r["lifts_original"] for r in rows)),
        "ink_ratio": avg("ink_ratio"),
        "min_ink_ratio": float(min(r["ink_ratio"] for r in rows)),
        "ms": avg("ms"),
        "passing": int(sum(r["pass"] for r in rows)),
    }


def passes(s: dict) -> bool:
    return (
        s["mean_mm"] <= TARGETS["mean_mm"]
        and s["p95_mm"] <= TARGETS["p95_mm"]
        and s["double_excess"] < TARGETS["double"]
    )


def print_table(summary: dict[str, dict]) -> None:
    head = (
        f"{'width':>6} {'n':>4} {'w est':>6} {'mean mm':>8} {'p95 mm':>7} {'worst p95':>9} "
        f"{'double':>7} {'worst':>6} {'orig dbl':>8} {'excess':>7} {'edge dbl':>8} "
        f"{'lifts t/o':>10} {'ink':>5} {'min ink':>7} {'ms':>4} {'pass':>8}  verdict"
    )
    print(head)
    print("-" * len(head))
    for w, s in summary.items():
        print(
            f"{w + ' px':>6} {s['drawings']:>4} {s['width_est_px']:>6.1f} {s['mean_mm']:>8.3f} "
            f"{s['p95_mm']:>7.3f} {s['worst_p95_mm']:>9.2f} {s['double'] * 100:>6.2f}% "
            f"{s['worst_double'] * 100:>5.1f}% {s['double_original'] * 100:>7.2f}% "
            f"{s['double_excess'] * 100:>6.2f}% {s['double_edge_tracer'] * 100:>7.1f}% "
            f"{str(s['lifts']) + '/' + str(s['lifts_original']):>10} {s['ink_ratio']:>5.2f} "
            f"{s['min_ink_ratio']:>7.2f} {s['ms']:>4.0f} "
            f"{str(s['passing']) + '/' + str(s['drawings']):>8}  "
            f"{'PASS' if passes(s) else 'FAIL'}"
        )


# ---------------------------------------------------------------------------------------------
# Pictures for a human
# ---------------------------------------------------------------------------------------------

PALETTE = [
    (200, 60, 40),
    (40, 120, 200),
    (40, 160, 80),
    (180, 120, 20),
    (140, 60, 170),
    (20, 150, 150),
    (200, 60, 140),
    (100, 100, 100),
]


def _fit(strokes: list, cell: int, pad: int, frame: tuple[int, int] | None) -> list[np.ndarray]:
    """Map pixel-frame strokes into a cell. `frame` = the source image size, so the original
    and the traced strokes share one mapping."""
    k = (cell - 2 * pad) / max(frame) if frame else 1.0
    return [np.asarray(s, float) * k + pad for s in strokes if len(s)]


def draw_cell(
    strokes: list, cell: int, frame: tuple[int, int], under: list | None = None, colour=True
) -> np.ndarray:
    img = np.full((cell, cell, 3), 255, np.uint8)
    if under:
        for a in _fit(under, cell, 6, frame):
            pts = np.round(a * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (200, 200, 200), 5, cv2.LINE_AA, 4)
    for i, a in enumerate(_fit(strokes, cell, 6, frame)):
        c = PALETTE[i % len(PALETTE)] if colour else (30, 30, 30)
        pts = np.round(a * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, c[::-1], 1, cv2.LINE_AA, 4)
        for p in (a[0], a[-1]):  # stroke ends show where the pen lifts
            cv2.circle(img, (int(p[0]), int(p[1])), 2, c[::-1], -1, cv2.LINE_AA)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(
        img, text, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
    )
    return img


def contact_sheet(items: list[dict], path: Path, cell: int = 240) -> None:
    """Rows: input image | original (thin black) | traced (one colour per stroke, dots at the
    ends) | traced over the original (grey)."""
    rows = []
    for it in items:
        res: TraceResult = it["res"]
        frame = res.image_size
        inp = cv2.cvtColor(
            cv2.resize(it["img"], (cell, cell), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR
        )
        r = it["row"]
        cells = [
            label(inp, it["name"]),
            label(
                draw_cell(it["orig"], cell, frame, colour=False),
                f"original: {r['lifts_original']} strokes",
            ),
            label(draw_cell(res.strokes, cell, frame), f"traced: {r['lifts']} strokes"),
            label(
                draw_cell(res.strokes, cell, frame, under=it["orig"]),
                f"mean {r['mean_mm']:.3f} p95 {r['p95_mm']:.2f} mm, dbl {r['double']:.1%}",
            ),
        ]
        rows.append(
            np.hstack([np.pad(c, ((1, 1), (1, 1), (0, 0)), constant_values=180) for c in cells])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def run_eval(args: argparse.Namespace) -> int:
    cv2.setNumThreads(4)
    drawings = library_drawings(args.categories)
    if not drawings:
        print("no drawings found", file=sys.stderr)
        return 2
    per_width: dict[str, list[dict]] = {}
    sheets: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    for w in args.widths:
        key = f"{w:g}"
        per_width[key] = []
        sheets[key] = []
        for n, (cat, vi, key_id, strokes) in enumerate(drawings):
            row, extra = evaluate_one(strokes, w, seed=n)
            row.update(category=cat, variant=vi, key_id=key_id, width_px=w)
            per_width[key].append(row)
            if cat in args.sheet_categories and vi < 2:
                sheets[key].append({"name": f"{cat} v{vi}, {key} px", "row": row, **extra})
        print(f"  width {key} px: {len(drawings)} drawings traced", file=sys.stderr)
    summary = {w: summarise(rows) for w, rows in per_width.items()}
    print()
    print(
        f"Centre-line tracer vs ground truth: {len(drawings)} library drawings, "
        f"{IMAGE_PX} px images, blur 0.8 px + noise 6/255 + 30 specks; "
        f"distances at {DRAWING_MM:g} mm drawing size"
    )
    print()
    print_table(summary)
    ok = all(passes(s) for s in summary.values())
    print()
    print(
        f"Targets (verdict, on the averages): mean <= {TARGETS['mean_mm']} mm, "
        f"p95 <= {TARGETS['p95_mm']} mm, double-line excess < {TARGETS['double']:.0%}.\n"
        f"double = share of traced ink with other ink parallel to it {DOUBLE_MIN_GAP_W:g} to "
        f"{DOUBLE_WINDOW_W:g} line widths away; excess = traced minus the same score on the "
        f"original (where the original's own lines run that close, the copy rightly does too);"
        f"\nedge dbl = an outline tracer on the same images, for scale."
        f"\nOverall: {'PASS' if ok else 'FAIL'}"
    )
    worst = sorted((r for rows in per_width.values() for r in rows), key=lambda r: -r["p95_mm"])
    print(
        "Worst p95:",
        ", ".join(
            f"{r['category']} v{r['variant']} @{r['width_px']:g}px {r['p95_mm']:.2f} mm"
            for r in worst[:5]
        ),
    )
    worst = sorted((r for rows in per_width.values() for r in rows), key=lambda r: -r["double"])
    print(
        "Worst double-line:",
        ", ".join(
            f"{r['category']} v{r['variant']} @{r['width_px']:g}px {r['double']:.1%}"
            for r in worst[:5]
        ),
    )
    print(f"({time.perf_counter() - t0:.0f} s)")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "image_px": IMAGE_PX,
            "margin_px": MARGIN_PX,
            "drawing_mm": DRAWING_MM,
            "degrade": {"blur_sigma_px": 0.8, "noise_sigma": 6.0, "specks": 30},
            "double_window_line_widths": DOUBLE_WINDOW_W,
            "targets": TARGETS,
        },
        "pass": ok,
        "summary": summary,
        "drawings": per_width,
    }
    args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.json.relative_to(ROOT) if args.json.is_relative_to(ROOT) else args.json}")
    for key, items in sheets.items():
        if items:
            path = args.sheet_dir / f"sheet_w{key}.png"
            contact_sheet(items, path)
            print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# Generated images -> library variant files
# ---------------------------------------------------------------------------------------------

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def expand(paths: list[str]) -> list[Path]:
    """Files from paths, folders and globs (PowerShell doesn't expand * for us)."""
    out: list[Path] = []
    for p in paths:
        hits = glob.glob(p) if any(c in p for c in "*?[") else [p]
        for h in sorted(hits):
            q = Path(h)
            if q.is_dir():
                out += sorted(f for f in q.iterdir() if f.suffix.lower() in IMAGE_EXT)
            elif q.suffix.lower() in IMAGE_EXT:
                out.append(q)
    return out


def load_manifest(path: Path | None) -> dict[str, dict]:
    """Manifest entries by image file name, if a manifest exists."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {Path(e["file"]).name: {**e, "model": data.get("model")} for e in data["images"]}


def round_strokes(strokes: list, nd: int = 1) -> list:
    return [[[round(x, nd), round(y, nd)] for x, y in s] for s in strokes]


def trace_images(
    images: list[Path],
    category: str,
    out: Path,
    manifest: dict[str, dict],
    args,
    scores: dict[str, dict] | None = None,
) -> int:
    """Trace each image into one variant file and a contact sheet. With Cosmos `scores` (by
    file name, from score_drawings.py) the images arrive best first and the sheet says so."""
    scores = scores or {}
    variants, items = [], []
    for p in images:
        res = trace(p)
        lib = to_library(res.strokes)
        w = res.line_width_px
        dbl = (
            double_line_score(res.strokes, DOUBLE_WINDOW_W * w, min_gap_px=DOUBLE_MIN_GAP_W * w)
            if res.strokes
            else 0.0
        )
        meta = manifest.get(p.name, {})
        v = {"key_id": p.stem, "strokes": round_strokes(lib)}
        for k in ("model", "seed", "prompt", "style"):
            if meta.get(k) is not None:
                v[k] = meta[k]
        v["trace"] = {
            "line_width_px": round(w, 1),
            "strokes": len(lib),
            "double_line": round(dbl, 4),
            "notes": res.notes,
        }
        sc = scores.get(p.name)
        if sc is not None:
            v["score"] = {
                k: sc.get(k) for k in ("scorer", "label", "alternatives", "match", "score", "note")
            }
        variants.append(v)
        flag = "  <- check" if (dbl >= TARGETS["double"] or res.notes or len(lib) > 40) else ""
        print(
            f"{p.name}: {len(lib)} strokes, line {w:.1f} px, double-line {dbl:.1%}"
            f"{', ' + '; '.join(res.notes) if res.notes else ''}{flag}"
        )
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        items.append((p.stem, gray, res))
    models = sorted({v["model"] for v in variants if v.get("model")})
    doc = {
        "category": category,
        "source": args.source
        or (
            f"generated with {', '.join(models)}; centre-line traced"
            if models
            else "generated line art; centre-line traced"
        ),
        "license": args.license,
        "units": UNITS,
        "variants": variants,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({len(variants)} variants)")
    if items and not args.no_sheet:
        sheet = (args.sheet_dir or out.parent) / f"{out.stem}_sheet.png"
        cell = 240
        rows = []
        for n, (name, gray, res) in enumerate(items, 1):
            inp = np.full((cell, cell), 255, np.uint8)
            if gray is not None:
                inp = cv2.resize(gray, (cell, cell), interpolation=cv2.INTER_AREA)
            sc = scores.get(f"{name}.png") or scores.get(name)
            heard = ""
            if sc:
                heard = f", {sc.get('scorer', 'cosmos')}: {sc['label'] or '-'} ({sc['match']})"
            rows.append(
                np.hstack(
                    [
                        label(cv2.cvtColor(inp, cv2.COLOR_GRAY2BGR), f"{n}. {name}"),
                        label(
                            draw_cell(res.strokes, cell, res.image_size, colour=False),
                            f"{len(res.strokes)} strokes{heard}",
                        ),
                    ]
                )
            )
        sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet), np.vstack(rows))
        print(f"wrote {sheet}")
    return 0 if variants else 1


def run_job(args: argparse.Namespace) -> int:
    job = args.job
    man_path = job / "manifest.json"
    if not man_path.exists():
        print(f"no manifest.json in {job}", file=sys.stderr)
        return 2
    data = json.loads(man_path.read_text(encoding="utf-8"))
    manifest = load_manifest(man_path)
    by_obj: dict[str, list[Path]] = {}
    for e in data["images"]:
        by_obj.setdefault(e["object"], []).append(job / e["file"])
    scores: dict[str, dict] = {}
    score_path = args.scores or job / "cosmos_scores.json"
    if score_path.exists():
        rep = json.loads(score_path.read_text(encoding="utf-8"))
        who = rep.get("scorer", "cosmos")
        scores = {Path(r["file"]).name: {"scorer": who, **r} for r in rep["images"]}
        for obj, files in by_obj.items():
            ranked = [Path(f).name for f in rep["objects"].get(obj, {}).get("ranked", [])]
            files.sort(key=lambda f: ranked.index(f.name) if f.name in ranked else len(ranked))
        print(f"ranked by {who} scores in {score_path}")
    rc = 0
    for obj, files in sorted(by_obj.items()):
        rc |= trace_images(files, obj, args.out_dir / f"{obj}.json", manifest, args, scores)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="*", help="images, folders or globs to trace")
    ap.add_argument("--eval", action="store_true", help="ground-truth evaluation on library/")
    ap.add_argument("--widths", nargs="+", type=float, default=[4, 8, 14], help="line widths, px")
    ap.add_argument("--categories", nargs="+", help="only these categories (--eval)")
    ap.add_argument("--sheet-categories", nargs="+", default=SHEET_CATEGORIES)
    ap.add_argument("--json", type=Path, default=OUT / "trace_eval.json")
    ap.add_argument("--sheet-dir", type=Path, default=None, help="where contact sheets go")
    ap.add_argument("--category", help="category name for the traced images")
    ap.add_argument("--out", type=Path, help="library variant file to write")
    ap.add_argument("--manifest", type=Path, help="manifest.json from the Job (optional)")
    ap.add_argument("--job", type=Path, help="a Job output folder with manifest.json")
    ap.add_argument("--out-dir", type=Path, default=OUT / "traced", help="for --job")
    ap.add_argument(
        "--scores",
        type=Path,
        help="scores to rank by: score_drawings.py's format, with an optional 'scorer' "
        "(default <job>/cosmos_scores.json, if any)",
    )
    ap.add_argument("--source", help="source text for the variant file")
    ap.add_argument(
        "--license",
        default="generated image; model licence in library/ATTRIBUTION.md",
        help="licence text for the variant file",
    )
    ap.add_argument("--no-sheet", action="store_true", help="skip the contact sheet PNG")
    args = ap.parse_args()

    if args.eval:
        args.sheet_dir = args.sheet_dir or OUT / "trace_eval"
        return run_eval(args)
    if args.job:
        return run_job(args)
    images = expand(args.images)
    if not images or not args.category or not args.out:
        ap.error("give images plus --category and --out, or --job DIR, or --eval")
    man = args.manifest or images[0].parent / "manifest.json"
    if not man.exists():
        man = images[0].parent.parent / "manifest.json"
    return trace_images(images, args.category, args.out, load_manifest(man), args)


if __name__ == "__main__":
    sys.exit(main())
