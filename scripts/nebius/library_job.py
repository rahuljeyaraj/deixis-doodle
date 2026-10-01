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
fails to load, FLUX.1 [schnell] (Apache-2.0) is used instead, unless --no-fallback. FLUX.1
[schnell] is gated on Hugging Face (checked 2026-09-30): the fallback needs HF_TOKEN from an
account that accepted its terms; SANA needs no token.

Time limits, three layers: this script stops starting new images after --max-minutes; a
watchdog thread kills the process 3 minutes after that if something hangs; the Job itself has
a hard --timeout (library_job.sh).

`--model fake` draws placeholder shapes with Pillow instead (no GPU, no torch): a local test
of the plumbing from here to the tracer.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
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
    "flux": "Apache-2.0 (FLUX.1 [schnell] model card, checked 2026-09-30; gated: accept the terms)",
    "fake": "n/a",
}
GRACE_S = 180  # the watchdog kills the process this long after the soft limit


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_prompts(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prompt_for(cfg: dict, obj: str, style: str | None = None) -> str:
    """The prompt for one object: the style template (a named one from `styles`, else
    `style`) with the object's subject text."""
    subject = (cfg.get("subjects") or {}).get(obj) or f"a {obj}"
    template = cfg["styles"][style] if style else cfg["style"]
    return " ".join(template.format(subject=subject).split())


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


# ---------------------------------------------------------------------------------------------
# Getting files out. The job's disk is deleted when it ends. On 1 Oct 2026 every write to the
# bucket mount failed (OSError EIO on close; nothing reached the bucket), so each finished file
# is also printed into the job's log, base64 in short lines with its size and sha256, and
# rebuilt on the laptop by logs_to_files.py (`library_job.sh fetch-logs`). The bucket stays as
# the first choice: after its first failed write it is not tried again.
# ---------------------------------------------------------------------------------------------

LOG_CHUNK = 1000  # base64 characters per log line


def emit_file(rel: str, data: bytes, fileno: int) -> None:
    """Print one file into the log: a header, numbered chunks, an end marker."""
    b64 = base64.b64encode(data).decode("ascii")
    n = (len(b64) + LOG_CHUNK - 1) // LOG_CHUNK
    sha = hashlib.sha256(data).hexdigest()
    lines = [f"@@FILE {fileno} {n} {len(data)} {sha} {rel}"]
    lines += [f"@@C {fileno} {i} {b64[i * LOG_CHUNK : (i + 1) * LOG_CHUNK]}" for i in range(n)]
    lines.append(f"@@END {fileno}")
    for line in lines:
        print(line, flush=True)


def compact_png(img) -> bytes:
    """A small copy for the log: grey, 16 levels, white kept white. Line art survives it; the
    noise a diffusion model leaves in the background (which makes PNGs big) does not."""
    g = img.convert("L").point(lambda v: min(255, (v // 16) * 17))
    buf = io.BytesIO()
    g.save(buf, "PNG", optimize=True)
    return buf.getvalue()


class Sink:
    """Where finished files go: local disk, then the bucket (best effort), then the log."""

    def __init__(self, out: Path, work: Path, to_log: bool = True) -> None:
        self.out, self.work, self.to_log = out, work, to_log
        self.bucket_ok = True
        self.files = 0

    def put(self, rel: str, data: bytes, log_data: bytes | None = None) -> None:
        local = self.work / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        if self.bucket_ok:
            try:
                dest = self.out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:  # plain sequential write; no rename, no seek
                    fh.write(data)
            except OSError as e:
                self.bucket_ok = False
                log(f"bucket write failed ({e!r}): files go to the log only from now on")
        if self.to_log:
            self.files += 1
            emit_file(rel, log_data if log_data is not None else data, self.files)


def manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(manifest, indent=1).encode("utf-8")


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
    ap.add_argument("--work", type=Path, default=Path("/tmp/library_work"), help="local copy")
    ap.add_argument("--no-log-files", action="store_true", help="don't print files into the log")
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
    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log(f"cannot make {args.out} ({e!r}): files go to the log only")
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
    sink = Sink(args.out, args.work, to_log=not args.no_log_files)
    done = 0
    styles = s.get("styles") or [None]  # a style test makes every object in several styles
    for obj in objects:
        folder = obj.replace(" ", "_")
        for style in styles:
            prompt = prompt_for(cfg, obj, style)
            tag = f"_{style}" if style else ""
            for i in range(per):
                if time.time() - start > limit_s:
                    log(f"time limit reached after {done} images: stopping")
                    manifest["stopped_early"] = True
                    break
                seed = int(params["seed_base"]) + i
                t0 = time.time()
                img = generate(prompt, negative, seed, params)
                took = time.time() - t0
                rel = f"{folder}/{folder}{tag}_s{seed}.png"
                buf = io.BytesIO()
                img.save(buf, "PNG")
                sink.put(rel, buf.getvalue(), compact_png(img))
                entry = {"file": rel, "object": obj, "seed": seed, "prompt": prompt}
                if style:
                    entry["style"] = style
                entry["seconds"] = round(took, 2)
                manifest["images"].append(entry)
                done += 1
                log(f"{done:4d} {obj}{' ' + style if style else ''} seed {seed}: {took:.1f} s")
            if manifest["stopped_early"]:
                break
        if manifest["stopped_early"]:
            break
    manifest["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["elapsed_s"] = round(time.time() - start, 1)
    manifest["bucket_ok"] = sink.bucket_ok
    sink.put("manifest.json", manifest_bytes(manifest))
    where = args.out if sink.bucket_ok else "the job log (fetch-logs)"
    log(f"done: {done} images in {manifest['elapsed_s']:.0f} s -> {where}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
