"""Every detailed (colouring) drawing in the library, timed and numbered, as a PDF.

    uv run python scripts/detailed_catalogue.py
    uv run python scripts/detailed_catalogue.py --size 60 --pen-up-z 3.0

Times use the machine's real feedrates from the saved config, not the library's
conservative defaults. The pictures are sorted fastest first and numbered, so a number
is enough to pick one for the spinner wheel (M06).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pointdraw.controller.config import load_config
from pointdraw.drawing import Library
from pointdraw.drawing.optimize import PlotParams, optimize
from pointdraw.drawing.scene import place_object

ROOT = Path(__file__).resolve().parents[1]
COLS, ROWS = 4, 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=float, default=60.0, help="drawing size in mm")
    ap.add_argument("--pen-up-z", type=float, help="override the config's pen-up height")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "out" / "detailed-catalogue.pdf")
    args = ap.parse_args()

    cfg = load_config()
    up_z = args.pen_up_z if args.pen_up_z is not None else cfg.pen_up_z
    lift_s = 2 * (up_z - cfg.pen_down_z) / cfg.limits.max_z_mm_s
    params = PlotParams(
        draw_mm_s=cfg.feed.normal / 60, travel_mm_s=cfg.feed.travel / 60, lift_s=lift_s
    )
    print(
        f"draw {params.draw_mm_s:.1f} mm/s, travel {params.travel_mm_s:.1f} mm/s, "
        f"pen lift {lift_s:.2f} s (pen-up z {up_z} mm)"
    )

    lib = Library.load()
    items = []
    for cat in lib.categories:
        for d in lib.variants(cat, style="detailed"):
            p = place_object(cat, (0.0, 0.0), args.size, lib, variant=d.key_id, style="detailed")
            pls, st = optimize([pl for pl in p.polylines], params=params)
            items.append((st.est_time_s, cat, d.key_id, pls, st))
    items.sort(key=lambda r: r[0])
    print(f"{len(items)} detailed drawings from {len(lib.categories)} objects")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_page = COLS * ROWS
    picks = []
    with PdfPages(args.out) as pdf:
        for page_start in range(0, len(items), per_page):
            page = items[page_start : page_start + per_page]
            fig, axes = plt.subplots(ROWS, COLS, figsize=(8.27, 11.69))
            fig.suptitle(
                f"Detailed drawings at {args.size:g} mm, fastest first "
                f"(page {page_start // per_page + 1})",
                fontsize=11,
            )
            for ax in axes.flat:
                ax.axis("off")
            for k, (t, cat, key_id, pls, st) in enumerate(page):
                n = page_start + k + 1
                ax = axes.flat[k]
                for pl in pls:
                    ax.plot([q[0] for q in pl], [q[1] for q in pl], "-", lw=0.7, color="black")
                ax.set_aspect("equal")
                ax.set_title(f"{n}. {cat}  —  {t:.0f} s\n{st.lifts} lifts", fontsize=8, pad=3)
                picks.append(
                    {
                        "n": n,
                        "object": cat,
                        "variant": key_id,
                        "seconds": round(t, 1),
                        "lifts": st.lifts,
                    }
                )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    index = args.out.with_suffix(".json")
    index.write_text(json.dumps(picks, indent=1), encoding="utf-8")
    print(f"wrote {args.out}\nwrote {index}")
    print(
        f"fastest {items[0][0]:.0f} s ({items[0][1]}), "
        f"slowest {items[-1][0]:.0f} s ({items[-1][1]})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
