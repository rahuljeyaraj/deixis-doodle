"""OpenAI-compatible client for a Cosmos vLLM endpoint on Nebius Serverless.

Configured through `Settings`: COSMOS_BASE_URL (the endpoint's https URL, with or without /v1),
COSMOS_API_KEY (falls back to COSMOS_ENDPOINT_TOKEN, the older name in rahul's .env) and
COSMOS_MODEL (default nvidia/Cosmos3-Nano).

Privacy (plan 4.5): callers send crops of the paper or bed, never whole camera frames.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pointdraw.config import Settings, get_settings
from pointdraw.cosmos import prompts
from pointdraw.cosmos.models import DEFAULT_MODEL, ModelProfile, profile_for
from pointdraw.cosmos.parse import (
    Detection,
    extract_json,
    normalise_label,
    parse_detections,
    parse_label,
)

log = logging.getLogger("pointdraw.cosmos")

ImageLike = np.ndarray | bytes | str | Path
CALL_LOG = Path(os.environ.get("COSMOS_CALL_LOG", "out/cosmos/calls.jsonl"))


@dataclass
class CosmosReply:
    raw: str
    latency_s: float
    model: str
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class LabelResult:
    label: str
    alternatives: list[str]
    reply: CosmosReply


@dataclass
class DetectResult:
    detections: list[Detection]
    image_size: tuple[int, int]  # width, height of the caller's image (detections use it)
    reply: CosmosReply


def _fallback_token(env_file: str | Path = ".env") -> str:
    """COSMOS_ENDPOINT_TOKEN from the environment or the .env file (never logged)."""
    if os.environ.get("COSMOS_ENDPOINT_TOKEN"):
        return os.environ["COSMOS_ENDPOINT_TOKEN"]
    p = Path(env_file)
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "COSMOS_ENDPOINT_TOKEN":
                return value.strip().strip("\"'")
    return ""


def normalise_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return url
    return url if url.endswith("/v1") else url + "/v1"


def encode_image(image: ImageLike, max_side: int = 1920, quality: int = 90) -> tuple[str, int, int]:
    """JPEG data URL plus the ORIGINAL width and height.

    Replies are normalised (0-1000 or 0-1), so they are mapped onto the original image even when
    it was downscaled before sending. A model that answers in raw pixels would be answering
    for the downscaled image; that case is rare and not handled.
    """
    import cv2

    if isinstance(image, str | Path):
        arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(image)
    elif isinstance(image, bytes):
        arr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("could not decode image bytes")
    else:
        arr = image
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    h, w = arr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        arr = cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", w, h


class CosmosClient:
    """label_crop, point_to and find against one Cosmos endpoint.

    `openai_client` can be injected (tests pass a fake with `.chat.completions.create`).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        openai_client: Any = None,
        think: bool | None = None,
    ) -> None:
        s = settings or get_settings()
        self.model = model or s.cosmos_model or DEFAULT_MODEL
        self.profile: ModelProfile = profile_for(self.model)
        self.timeout_s = timeout_s
        # None: use the model's default. False asks for a direct answer (faster, R1 default).
        self.think = think
        if openai_client is None:
            from openai import OpenAI

            url = normalise_base_url(base_url or s.cosmos_base_url)
            key = api_key or s.cosmos_api_key.get_secret_value() or _fallback_token()
            if not url:
                raise ValueError("COSMOS_BASE_URL is not set (see scripts/nebius/README.md)")
            if not key:
                raise ValueError("COSMOS_API_KEY (or COSMOS_ENDPOINT_TOKEN) is not set")
            openai_client = OpenAI(base_url=url, api_key=key, timeout=timeout_s, max_retries=1)
        # every call to a real endpoint goes in out/cosmos/calls.jsonl (busy time, M21)
        self.call_log = None if openai_client is not None else CALL_LOG
        self._client = openai_client

    # -- low level -------------------------------------------------------------------------

    def ask(
        self, image: ImageLike, text: str, max_side: int = 1920
    ) -> tuple[CosmosReply, int, int]:
        """One image + one text prompt. Returns the reply and the image's original width, height."""
        data_url, w, h = encode_image(image, max_side=max_side)
        if self.think is False:
            text = text + prompts.NO_THINK_SUFFIX
        params: dict[str, Any] = dict(self.profile.sampling)
        extra = dict(params.pop("extra_body", {}))
        if self.think is not None:
            # Honoured by Qwen3-style chat templates; ignored by the others.
            extra["chat_template_kwargs"] = {"enable_thinking": bool(self.think)}
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.profile.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": text},
                    ],
                }
            ],
            extra_body=extra or None,
            timeout=self.timeout_s,
            **params,
        )
        latency = time.perf_counter() - t0
        msg = resp.choices[0].message
        raw = msg.content or ""
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
            }
        self._log_call(t0, latency, usage)
        log.info(
            "cosmos %s %.2fs in=%s out=%s",
            self.model,
            latency,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        return CosmosReply(raw=raw, latency_s=latency, model=self.model, usage=usage), w, h

    def _log_call(self, t0: float, latency: float, usage: dict[str, int]) -> None:
        """One line per call: when it started (UTC), how long, the tokens. No image, no text."""
        if self.call_log is None:
            return
        import json
        from datetime import UTC, datetime, timedelta

        start = datetime.now(UTC) - timedelta(seconds=time.perf_counter() - t0)
        row = {"utc": start.isoformat(timespec="milliseconds"), "s": round(latency, 3),
               "model": self.model, **usage}  # fmt: skip
        try:
            self.call_log.parent.mkdir(parents=True, exist_ok=True)
            with self.call_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            log.warning("could not write %s", self.call_log)

    # -- the three R1 tasks ----------------------------------------------------------------

    def label_crop(self, image: ImageLike) -> LabelResult:
        """(a) What is drawn in this crop? Returns a normalised label ('cat', 'circle')."""
        reply, _, _ = self.ask(image, prompts.label_crop_prompt(), max_side=768)
        label = parse_label(reply.raw)
        alts: list[str] = []
        try:
            data = extract_json(reply.raw)
            if isinstance(data, dict) and isinstance(data.get("alternatives"), list):
                alts = [normalise_label(str(a)) for a in data["alternatives"] if str(a).strip()]
        except ValueError:
            pass
        return LabelResult(label=label, alternatives=alts, reply=reply)

    def point_to(self, image: ImageLike, query: str) -> DetectResult:
        """(b) A semantic place, e.g. 'empty space right of the house'. Points in pixels."""
        reply, w, h = self.ask(image, prompts.point_to_prompt(query))
        dets = self._parse(reply.raw, w, h)
        return DetectResult(detections=dets, image_size=(w, h), reply=reply)

    def find(self, image: ImageLike, query: str) -> DetectResult:
        """(c) Boxes of a named mark, e.g. 'the circle'. Empty query finds every mark."""
        reply, w, h = self.ask(image, prompts.find_prompt(query))
        dets = self._parse(reply.raw, w, h)
        return DetectResult(detections=dets, image_size=(w, h), reply=reply)

    def _parse(self, raw: str, w: int, h: int) -> list[Detection]:
        try:
            return parse_detections(raw, w, h, self.profile.coord_mode, self.profile.xy_order)
        except ValueError:
            log.warning("cosmos reply had no JSON: %r", raw[:200])
            return []
