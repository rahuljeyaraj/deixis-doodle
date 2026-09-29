"""Scoring for the R3 set: does a list of tool calls match what we expected?

Matching rules (deliberately about meaning, not wording):
- tool names and call order must match;
- ask_user and look: only the tool name counts (the question text is free);
- adjust: only the direction of dx, dy and scale counts;
- deictic references: only word_index counts;
- labels and text: compared case-insensitively, without punctuation or leading articles;
- named places: synonyms folded ("center" = "middle", "top left" = "top left corner");
- a missing size equals {"preset": "medium"}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pointdraw.agent.schema import ToolCall, ToolCallError, parse_call

FREE_TOOLS = {"ask_user", "look"}
_ARTICLES = re.compile(r"^(the|a|an|my)\s+")
_NON_WORD = re.compile(r"[^\w\s']")


@dataclass
class R3Item:
    id: str
    tags: list[str]
    text: str
    words: list[str]
    expected: list[dict[str, Any]]
    alt: list[list[dict[str, Any]]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def expectations(self) -> list[list[dict[str, Any]]]:
        return [self.expected, *self.alt]


def load_items(path: Path) -> list[R3Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(R3Item(**d))
    return items


def _text(s: str) -> str:
    s = _NON_WORD.sub(" ", s.lower())
    s = " ".join(s.split())
    return _ARTICLES.sub("", s)


def _named(s: str) -> str:
    s = _text(s).replace("centre", "middle").replace("center", "middle")
    s = s.replace("upper", "top").replace("lower", "bottom")
    words = [w for w in s.split() if w not in ("of", "the", "page", "sheet", "corner", "in")]
    return " ".join(sorted(words))


def _sign(x: float) -> int:
    return (x > 1e-9) - (x < -1e-9)


def canon(value: Any, key: str = "") -> Any:
    """Canonical form of an argument value for comparison."""
    if isinstance(value, dict):
        if "word_index" in value:
            return ("word", value["word_index"])
        if "label" in value:
            return ("label", _text(value["label"]))
        if "named" in value:
            return ("named", _named(value["named"]))
        return {k: canon(v, k) for k, v in value.items() if k != "deixis"}
    if isinstance(value, str) and key in ("text", "object"):
        return _text(value)
    return value


def canon_call(call: dict[str, Any]) -> tuple[str, Any]:
    name = call["name"]
    args = dict(call.get("args") or {})
    if name in FREE_TOOLS:
        return name, None
    if name == "adjust":
        return name, (
            _sign(args.get("dx", 0)),
            _sign(args.get("dy", 0)),
            _sign(args.get("scale", 1.0) - 1.0),
        )
    if name in ("place_drawing", "write_label"):
        if args.get("size") in (None, {"preset": "medium"}):
            args.pop("size", None)
        args = {k: v for k, v in args.items() if v is not None}
    if name == "draw_connector":
        args.setdefault("kind", "arrow")
    return name, canon(args)


def calls_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    return [canon_call(c) for c in actual] == [canon_call(c) for c in expected]


def item_matches(item: R3Item, actual: list[dict[str, Any]]) -> bool:
    return any(calls_match(actual, exp) for exp in item.expectations)


def names_match(item: R3Item, actual: list[dict[str, Any]]) -> bool:
    got = [c["name"] for c in actual]
    return any(got == [c["name"] for c in exp] for exp in item.expectations)


def expected_calls(item: R3Item) -> list[ToolCall]:
    """Validate the expected calls through the real schema (ask/look get dummy args)."""
    out = []
    for c in item.expected:
        args = dict(c.get("args") or {})
        if c["name"] == "ask_user":
            args = {"question": "?"}
        elif c["name"] == "look":
            args = {"query": "?"}
        try:
            out.append(parse_call(c["name"], args))
        except ToolCallError as e:
            raise ValueError(f"{item.id}: {e}") from e
    return out


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); 0.0 for an empty list."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q / 100 * len(s) + 0.5)) - 1))
    return s[k]
