"""Turn raw Cosmos replies into boxes and points in image pixels.

Vision-language models wrap their JSON in many ways: ```json fences, `<think>...</think>`
reasoning, `<answer>` tags, prose before or after, and several coordinate conventions. This
module is pure Python so it can be tested offline against recorded replies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

CoordMode = Literal["auto", "norm1000", "norm1", "pixels"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


@dataclass(frozen=True)
class Detection:
    """One labelled box (and/or point) in image pixels, origin top-left, y down."""

    label: str
    box_px: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2
    point_px: tuple[float, float] | None = None

    @property
    def center_px(self) -> tuple[float, float] | None:
        if self.point_px is not None:
            return self.point_px
        if self.box_px is not None:
            x1, y1, x2, y2 = self.box_px
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None


def strip_reasoning(text: str) -> str:
    """Remove `<think>` blocks and unwrap `<answer>` tags.

    Handles a missing opening tag (vLLM reasoning parsers sometimes eat it) by keeping only the
    text after the last `</think>`, and an unclosed `<think>` by dropping everything after it
    unless that leaves nothing.
    """
    out = _THINK_RE.sub("", text)
    if "</think>" in out.lower():
        out = re.split(r"</think>", out, flags=re.IGNORECASE)[-1]
    low = out.lower()
    if "<think>" in low:
        head = out[: low.index("<think>")]
        out = head if head.strip() else out[low.index("<think>") + len("<think>") :]
    m = _ANSWER_RE.search(out)
    if m and m.group(1).strip():
        out = m.group(1)
    return out.strip()


def _balanced_spans(text: str) -> list[str]:
    """Every top-level [...] or {...} span, honouring strings, in order of appearance."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = depth > 0
        elif ch in "[{":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "]}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def _loads(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except ValueError:
        return json.loads(_TRAILING_COMMA_RE.sub(r"\1", candidate))


def extract_json(text: str) -> Any:
    """Find the JSON value in a model reply. Raises ValueError if there is none.

    Order of preference: fenced blocks, then the whole reply, then the longest balanced
    bracket span. Reasoning blocks are removed first.
    """
    body = strip_reasoning(text)
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(body)]
    candidates.append(body)
    candidates += sorted(_balanced_spans(body), key=len, reverse=True)
    for cand in candidates:
        if not cand:
            continue
        try:
            return _loads(cand)
        except ValueError:
            continue
    raise ValueError(f"no JSON found in reply: {text[:200]!r}")


def _flatten_numbers(value: Any) -> list[float] | None:
    """[1, 2, 3, 4] or [[1, 2], [3, 4]] or "1, 2, 3, 4" -> a flat list of floats."""
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        return [float(n) for n in nums] if nums else None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, list | tuple):
        out: list[float] = []
        for v in value:
            sub = _flatten_numbers(v)
            if sub is None:
                return None
            out += sub
        return out
    return None


_BOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box", "position", "bounding_box")
_POINT_KEYS = ("point_2d", "point", "points", "coordinates", "xy")
_LABEL_KEYS = ("label", "name", "object", "category", "description")


def _items(data: Any) -> list[dict[str, Any]]:
    """The list of dicts a reply describes, whatever the wrapper."""
    if isinstance(data, dict):
        for key in ("objects", "detections", "results", "items", "points", "boxes", "answer"):
            if isinstance(data.get(key), list):
                return _items(data[key])
        return [data]
    if isinstance(data, list):
        if data and all(isinstance(v, int | float) for v in data):
            return [{"point_2d": data} if len(data) == 2 else {"bbox_2d": data}]
        out: list[dict[str, Any]] = []
        for v in data:
            if isinstance(v, dict):
                out.append(v)
            elif isinstance(v, list):
                out += _items(v)
        return out
    return []


def raw_detections(data: Any) -> list[tuple[str, list[float] | None, list[float] | None]]:
    """(label, box numbers, point numbers) per item, in the model's own coordinates."""
    out = []
    for item in _items(data):
        label = ""
        for k in _LABEL_KEYS:
            if isinstance(item.get(k), str):
                label = item[k].strip()
                break
        box = None
        for k in _BOX_KEYS:
            nums = _flatten_numbers(item.get(k)) if k in item else None
            if nums is not None and len(nums) == 4:
                box = nums
                break
        point = None
        for k in _POINT_KEYS:
            nums = _flatten_numbers(item.get(k)) if k in item else None
            if nums is not None and len(nums) >= 2:
                point = nums[:2]
                break
        if box is None and point is None:
            continue
        out.append((label, box, point))
    return out


def detect_coord_mode(values: list[float], width: int, height: int) -> CoordMode:
    """Guess the coordinate convention from the numbers alone.

    All values in [0, 1] -> normalised 0-1. Any value above 1000 (and within the image) ->
    pixels. Otherwise 0-1000, which is what Cosmos3 and Qwen3-VL-based models emit.
    """
    if not values:
        return "norm1000"
    if max(values) <= 1.0 and min(values) >= 0.0:
        return "norm1"
    if max(values) > 1000.0 and max(values) <= max(width, height) * 1.05:
        return "pixels"
    return "norm1000"


def to_pixels(
    nums: list[float], width: int, height: int, mode: CoordMode, xy_order: str = "xy"
) -> list[float]:
    """Convert alternating x, y numbers to pixels (clipped to the image)."""
    if xy_order == "yx":
        nums = [nums[i + 1] if i % 2 == 0 else nums[i - 1] for i in range(len(nums))]
    if mode == "norm1000":
        sx, sy = width / 1000.0, height / 1000.0
    elif mode == "norm1":
        sx, sy = float(width), float(height)
    else:
        sx = sy = 1.0
    out = []
    for i, v in enumerate(nums):
        if i % 2 == 0:
            out.append(min(max(v * sx, 0.0), float(width)))
        else:
            out.append(min(max(v * sy, 0.0), float(height)))
    return out


def parse_detections(
    text: str,
    width: int,
    height: int,
    coord_mode: CoordMode = "auto",
    xy_order: str = "xy",
) -> list[Detection]:
    """Parse a grounding or pointing reply into `Detection`s in pixels.

    Boxes are re-ordered so x1 <= x2 and y1 <= y2. In "auto" mode the convention is guessed
    once per reply from all its numbers (see `detect_coord_mode`).
    """
    raw = raw_detections(extract_json(text))
    mode = coord_mode
    if mode == "auto":
        allnums = [v for _, b, p in raw for v in (b or []) + (p or [])]
        mode = detect_coord_mode(allnums, width, height)
    out = []
    for label, box, point in raw:
        box_px = None
        point_px = None
        if box is not None:
            x1, y1, x2, y2 = to_pixels(box, width, height, mode, xy_order)
            box_px = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if point is not None:
            px, py = to_pixels(point, width, height, mode, xy_order)
            point_px = (px, py)
        out.append(Detection(label=label, box_px=box_px, point_px=point_px))
    return out


def parse_label(text: str) -> str:
    """The label from a `label_crop` reply: JSON {"label": ...} if present, else the first line."""
    body = strip_reasoning(text)
    try:
        data = extract_json(body)
    except ValueError:
        data = None
    if isinstance(data, dict):
        for k in _LABEL_KEYS:
            if isinstance(data.get(k), str) and data[k].strip():
                return normalise_label(data[k])
    if isinstance(data, list) and data and isinstance(data[0], dict):
        for k in _LABEL_KEYS:
            if isinstance(data[0].get(k), str):
                return normalise_label(data[0][k])
    first = next((ln for ln in body.splitlines() if ln.strip() and "```" not in ln), "")
    return normalise_label(first)


_ARTICLES = re.compile(r"^(?:a|an|the|this is|it is|it's|drawing of|a drawing of)\s+", re.I)


def normalise_label(label: str) -> str:
    """Lower-case, strip punctuation and leading articles: 'A drawing of a Cat.' -> 'cat'."""
    s = label.strip().strip("\"'`").strip().rstrip(".!").lower()
    for _ in range(5):
        s2 = _ARTICLES.sub("", s)
        if s2 == s:
            break
        s = s2
    return re.sub(r"\s+", " ", s).strip()
