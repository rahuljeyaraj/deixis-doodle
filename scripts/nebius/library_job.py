"""Generate line-art images for the drawing library on a GPU, then stop.

Runs inside a Nebius Serverless Job (see library_job.sh, which injects this file and
library_prompts.yaml into the container and mounts a bucket at /out):

    python library_job.py --prompts library_prompts.yaml --set pilot --out /out/pilot

For every object in the set it makes N images with fixed seeds (seed_base + i), 1024 x 1024,
and writes

    <out>/<object>/<object>_s<seed>.png
    <out>/manifest.json   model, licence note, prompts, parameters, one entry per image

The manifest is rewritten after every image, so a job that is stopped early still leaves a
usable set. Then, on the laptop: `uv run python scripts/trace_library.py --job <folder>`.

Model: SANA 1600M (Efficient-Large-Model/Sana_1600M_1024px_diffusers, Apache-2.0 on its model
card, checked 2026-09-30; its Gemma-2-2B-IT text encoder adds Google's Gemma terms). If SANA
fails to load, FLUX.1 [schnell] (Apache-2.0) is used instead, unless --no-fallback.

Time limits, three layers: this script stops starting new images after --max-minutes; a
watchdog thread kills the process 3 minutes after that if something hangs; the Job itself has
a hard --timeout (library_job.sh).

`--model fake` draws placeholder shapes with Pillow instead (no GPU, no torch): a local test
of the plumbing from here to the tracer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

MODELS = {
    "sana": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "sana-4.8b": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "fake": "fake (Pillow placeholder shapes)",
}
LICENCES = {
    "sana": "Apache-2.0 (model card, checked 2026-09-30); text encoder Gemma-2-2B-IT: Gemma "
    "Terms of Use and Prohibited Use Policy",
    "sana-4.8b": "Apache-2.0 (model card, checked 2026-09-30); text encoder Gemma-2-2B-IT: "
    "Gemma Terms of Use and Prohibited Use Policy",
    "flux": "Apache-2.0 (FLUX.1 [schnell] model card; re-check before use)",
    "fake": "n/a",
}
GRACE_S = 180  # the watchdog kills the process this long after the soft limit


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_prompts(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prompt_for(cfg: dict, obj: str) -> str:
    subject = (cfg.get("subjects") or {}).get(obj) or f"a {obj}"
    return " ".join(cfg["style"].format(subject=subject).split())


def start_watchdog(limit_s: float) -> None:
    """Hard stop: exit the whole process if it is still running `limit_s` after start."""

    def kill() -> None:
        time.sleep(limit_s)
        log(f"WATCHDOG: still running after {limit_s / 60:.0f} min, exiting")
        os._exit(3)

    threading.Thread(target=kill, daemon=True).start()


# ---------------------------------------------------------------------------------------------
# Generators: each returns a function (prompt, negative, seed, params) -> PIL image
# ---------------------------------------------------------------------------------------------


def load_sana(repo: str):
    import torch
    from diffusers import SanaPipeline

    kwargs = {"torch_dtype": torch.bfloat16}
    if "1600M" in repo:  # the model card's recipe: fp16 weights, VAE and text encoder in bf16
        kwargs = {"variant": "fp16", "torch_dtype": torch.float16}
    pipe = SanaPipeline.from_pretrained(repo, **kwargs)
    pipe.to("cuda")
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)

    def generate(prompt: str, negative: str, seed: int, p: dict):
        return pipe(
            prompt=prompt,
            negative_prompt=negative,
            height=p["height"],
            width=p["width"],
            guidance_scale=p["guidance"],
            num_inference_steps=p["steps"],
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    return generate


def load_flux(repo: str):
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    def generate(prompt: str, negative: str, seed: int, p: dict):
        # schnell is distilled: 4 steps, no guidance and no negative prompt
        return pipe(
            prompt=prompt,
            height=p["height"],
            width=p["width"],
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    return generate


def load_fake(_repo: str):
    from PIL import Image, ImageDraw

    def generate(prompt: str, negative: str, seed: int, p: dict):
        rng = random.Random(seed)
        w, h = p["width"], p["height"]
        img = Image.new("L", (w, h), 255)
        d = ImageDraw.Draw(img)
        cx, cy, r = w / 2, h / 2, min(w, h) * 0.3
        k = rng.randint(3, 6)
        ring = [
            (
                cx + r * (1 + 0.15 * math.sin(k * t)) * math.cos(t),
                cy + r * (1 + 0.15 * math.sin(k * t)) * math.sin(t),
            )
            for t in [i * 2 * math.pi / 180 for i in range(181)]
        ]
        width = rng.randint(6, 12)
        d.line(ring, fill=0, width=width, joint="curve")
        d.line([(cx, cy - r * 0.4), (cx, cy + r * 1.6)], fill=0, width=width)
        return img.convert("RGB")

    return generate


LOADERS = {"sana": load_sana, "sana-4.8b": load_sana, "flux": load_flux, "fake": load_fake}


def load_model(name: str, fallback: bool):
    """(model key, generate function). Falls back to FLUX.1 [schnell] if SANA won't load."""
    try:
        t0 = time.time()
        gen = LOADERS[name](MODELS[name])
        log(f"loaded {MODELS[name]} in {time.time() - t0:.0f} s")
        return name, gen
    except Exception as exc:  # any load failure means: try the fallback
        if not fallback or name in ("flux", "fake"):
            raise
        log(f"could not load {MODELS[name]}: {exc!r}; falling back to FLUX.1 [schnell]")
        t0 = time.time()
        gen = load_flux(MODELS["flux"])
        log(f"loaded {MODELS['flux']} in {time.time() - t0:.0f} s")
        return "flux", gen


def write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--prompts", type=Path, default=here / "library_prompts.yaml")
    ap.add_argument("--set", default="pilot", help="a set in the prompts file (pilot, full)")
    ap.add_argument("--out", type=Path, required=True, help="output folder (the bucket mount)")
    ap.add_argument("--objects", nargs="+", help="override the set's object list")
    ap.add_argument("--per-object", type=int, help="override images per object")
    ap.add_argument("--max-minutes", type=float, help="override the set's time limit")
    ap.add_argument("--model", choices=sorted(MODELS), default="sana")
    ap.add_argument("--no-fallback", action="store_true", help="don't fall back to FLUX")
    args = ap.parse_args()

    start = time.time()
    cfg = load_prompts(args.prompts)
    s = cfg["sets"][args.set]
    objects = args.objects or s["objects"]
    per = args.per_object or s["per_object"]
    limit_s = 60 * (args.max_minutes or s["max_minutes"])
    start_watchdog(limit_s + GRACE_S)
    params = cfg["params"]
    negative = " ".join(cfg.get("negative", "").split())
    log(f"set {args.set}: {len(objects)} objects x {per} images, limit {limit_s / 60:.0f} min")

    try:
        import torch

        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    model, generate = load_model(args.model, not args.no_fallback)
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for mod in ("torch", "diffusers", "transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            pass
    log(f"versions: {versions}")
    manifest = {
        "model": MODELS[model],
        "model_key": model,
        "licence": LICENCES[model],
        "versions": versions,
        "set": args.set,
        "style": cfg["style"],
        "negative": negative,
        "params": params,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished": None,
        "stopped_early": False,
        "images": [],
    }
    man_path = args.out / "manifest.json"
    done = 0
    for obj in objects:
        prompt = prompt_for(cfg, obj)
        folder = args.out / obj.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per):
            if time.time() - start > limit_s:
                log(f"time limit reached after {done} images: stopping")
                manifest["stopped_early"] = True
                break
            seed = int(params["seed_base"]) + i
            t0 = time.time()
            img = generate(prompt, negative, seed, params)
            name = f"{obj.replace(' ', '_')}_s{seed}.png"
            img.save(folder / name)
            took = time.time() - t0
            manifest["images"].append(
                {
                    "file": f"{folder.name}/{name}",
                    "object": obj,
                    "seed": seed,
                    "prompt": prompt,
                    "seconds": round(took, 2),
                }
            )
            write_manifest(man_path, manifest)
            done += 1
            log(f"{done:4d} {obj} seed {seed}: {took:.1f} s")
        if manifest["stopped_early"]:
            break
    manifest["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["elapsed_s"] = round(time.time() - start, 1)
    write_manifest(man_path, manifest)
    log(f"done: {done} images in {manifest['elapsed_s']:.0f} s -> {args.out}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
