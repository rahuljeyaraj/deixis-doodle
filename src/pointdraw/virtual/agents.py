"""The agent in the virtual world: mock (recorded replies + the rule parser) or live.

- **mock:** a cache of real Nemotron replies, keyed by the words and the context lines the
  model saw (`tests/scenes/agent_cache.json`, written by live runs with `--record`). Anything
  not in the cache goes to the rule-based fallback parser. Free, offline, deterministic.
  Latency: the recorded one, or `mock_latency_s` (Lightning's p50 from R3) for the parser,
  so timelines look like the real thing.
- **live:** Nemotron on Token Factory (`NemotronAgent`), with the same fallback if it can't
  be reached. Every call is counted.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from pointdraw.agent.fallback_parser import parse as fallback_parse
from pointdraw.agent.nemotron import AgentResult, AgentUnavailable, NemotronAgent
from pointdraw.agent.prompts import PROMPT_VERSION, AgentContext
from pointdraw.agent.schema import ToolCall, ToolCallError, norm_word, parse_call
from pointdraw.virtual.scenefile import SCENES_DIR

CACHE_PATH = SCENES_DIR / "agent_cache.json"


def cache_key(words: list[str], ctx: AgentContext) -> str:
    body = json.dumps(
        {"w": [norm_word(w) for w in words], "c": ctx.lines(), "p": PROMPT_VERSION},
        separators=(",", ":"),
    )
    return hashlib.sha1(body.encode()).hexdigest()[:16]


class ReplyCache:
    def __init__(self, path: Path | str = CACHE_PATH) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))
        self.dirty = False

    def get(self, words: list[str], ctx: AgentContext) -> dict[str, Any] | None:
        return self.entries.get(cache_key(words, ctx))

    def put(self, words: list[str], ctx: AgentContext, res: AgentResult) -> None:
        self.entries[cache_key(words, ctx)] = {
            "words": " ".join(words),
            "context": ctx.lines(),
            "calls": [c.to_dict() for c in res.calls],
            "speech": res.speech,
            "model": res.model,
            "latency_s": round(res.latency_s, 3),
            "recorded": dt.date.today().isoformat(),
        }
        self.dirty = True

    def save(self) -> None:
        if self.dirty:
            text = json.dumps(dict(sorted(self.entries.items())), indent=1, ensure_ascii=False)
            self.path.write_text(text + "\n", encoding="utf-8")
            self.dirty = False


class MockAgent:
    """Recorded replies first, then the rule parser."""

    def __init__(
        self,
        categories: list[str],
        cache: ReplyCache | None = None,
        mock_latency_s: float = 0.6,
        use_cache: bool = True,
    ) -> None:
        self.categories = categories
        self.cache = cache if cache is not None else ReplyCache()
        self.mock_latency_s = mock_latency_s
        self.use_cache = use_cache
        self.hits = 0
        self.misses = 0

    def understand(self, words: list[str], context: AgentContext) -> AgentResult:
        entry = self.cache.get(words, context) if self.use_cache else None
        if entry is not None:
            calls = []
            try:
                calls = [parse_call(c["name"], c["args"]) for c in entry["calls"]]
            except ToolCallError:
                calls = []
            if calls or entry.get("speech"):
                self.hits += 1
                return AgentResult(
                    calls=calls,
                    speech=entry.get("speech"),
                    source="cache",  # type: ignore[arg-type]
                    model=entry.get("model", ""),
                    latency_s=float(entry.get("latency_s", self.mock_latency_s)),
                )
        self.misses += 1
        calls = fallback_parse(words, context, self.categories)
        return AgentResult(calls=calls, source="fallback", latency_s=self.mock_latency_s)


class LiveAgent:
    """Nemotron on Token Factory, recording replies into the cache."""

    def __init__(
        self,
        categories: list[str],
        model: str | None = None,
        cache: ReplyCache | None = None,
        record: bool = True,
    ) -> None:
        self.categories = categories
        self.agent = NemotronAgent(model=model, categories=categories)
        self.cache = cache if cache is not None else ReplyCache()
        self.record = record
        self.calls = 0  # Token Factory requests made (including retries)
        self.failures = 0
        self.usage: dict[str, int] = {}

    def understand(self, words: list[str], context: AgentContext) -> AgentResult:
        try:
            res = self.agent.run(words, context)
        except AgentUnavailable as e:
            self.failures += 1
            calls = fallback_parse(words, context, self.categories)
            return AgentResult(calls=calls, source="fallback", errors=[str(e)])
        self.calls += res.attempts
        for k, v in res.usage.items():
            self.usage[k] = self.usage.get(k, 0) + v
        if self.record and (res.calls or res.speech):
            self.cache.put(words, context, res)
        if not res.calls and not res.speech:
            calls = fallback_parse(words, context, self.categories)
            return AgentResult(calls=calls, source="fallback", model=res.model,
                               latency_s=res.latency_s, errors=res.errors)  # fmt: skip
        return res


def describe(calls: list[ToolCall]) -> list[str]:
    return [str(c) for c in calls]
