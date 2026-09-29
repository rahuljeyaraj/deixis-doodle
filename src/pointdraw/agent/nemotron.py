"""Nemotron on Nebius Token Factory: one chat call per utterance returns typed tool calls.

- OpenAI SDK against Token Factory's OpenAI-compatible API, key from `Settings`.
- Reasoning off (`chat_template_kwargs.enable_thinking = false`): about 1 s per call.
- `tool_choice="auto"` by default. On 30 Sep, "required" made Nano return only the first
  of several calls and made Lightning loop on `look` calls until max_tokens (see
  docs/spikes/r3-nemotron.md). A plain-text reply is retried once with feedback, then
  returned as speech.
- One retry: invalid calls are sent back to the model with the validation errors.
- Every call's latency and token use are logged on the `pointdraw.agent` logger.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pointdraw import clock
from pointdraw.agent.prompts import (
    PROMPT_VERSION,
    AgentContext,
    load_categories,
    system_prompt,
    user_message,
)
from pointdraw.agent.schema import (
    TOOL_ARGS,
    ToolCall,
    ToolCallError,
    check_against_transcript,
    norm_word,
    openai_tools,
    parse_call,
)
from pointdraw.config import Settings, get_settings

log = logging.getLogger("pointdraw.agent")

# R3 (30 Sep): Lightning 52/61 at p50 0.60 s vs Nano 43/61 at 1.00 s, and only Lightning
# returns several calls in one reply. NEMOTRON_MODEL in .env overrides this.
DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"
FALLBACK_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


class AgentUnavailable(RuntimeError):
    """Token Factory can't be used (no key, network error, timeout, HTTP error)."""


@dataclass
class AgentResult:
    """What the agent decided for one utterance."""

    calls: list[ToolCall]
    speech: str | None = None  # plain text the model returned instead of tool calls
    source: Literal["nemotron", "fallback"] = "nemotron"
    model: str = ""
    latency_s: float = 0.0  # total, including a retry
    attempts: int = 1
    errors: list[str] = field(default_factory=list)  # validation errors seen (and fixed?)
    usage: dict[str, int] = field(default_factory=dict)
    guarded: bool = False  # True if guard_label_commands replaced the model's calls
    salvaged: bool = False  # True if calls were read from text like 'cancel()'

    @property
    def ok(self) -> bool:
        return bool(self.calls)


class NemotronAgent:
    """Turns an utterance (list of words) plus context into validated tool calls."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        client: Any = None,
        *,
        thinking: bool = False,
        tool_choice: str = "auto",
        timeout_s: float = 8.0,
        max_retries: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        categories: list[str] | None = None,
        check_categories: bool = True,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.nemotron_model or DEFAULT_MODEL
        self.thinking = thinking
        self.tool_choice = tool_choice
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.categories = categories if categories is not None else load_categories()
        self.check_categories = check_categories
        self.prompt_version = prompt_version
        self.system = system_prompt(self.categories, prompt_version)
        self.tools = openai_tools()
        self._client = client

    # ------------------------------------------------------------------ plumbing

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self.settings.nebius_api_key.get_secret_value()
            if not key:
                raise AgentUnavailable("NEBIUS_API_KEY is not set (see .env.example)")
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.nebius_base_url,
                api_key=key,
                timeout=self.timeout_s,
                max_retries=0,  # we retry ourselves, with feedback
            )
        return self._client

    def _request(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.thinking}},
        }
        try:
            return self.client.chat.completions.create(**kwargs)
        except AgentUnavailable:
            raise
        except Exception as e:  # openai.APIError, httpx errors, timeouts
            raise AgentUnavailable(f"{type(e).__name__}: {e}") from e

    # ------------------------------------------------------------------ main entry

    def run(self, words: list[str], context: AgentContext | None = None) -> AgentResult:
        """One utterance in, validated tool calls out. Raises AgentUnavailable."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_message(words, context)},
        ]
        result = AgentResult(calls=[], model=self.model, attempts=0)
        t_start = clock.now()
        for attempt in range(self.max_retries + 1):
            t0 = clock.now()
            resp = self._request(messages)
            dt = clock.now() - t0
            result.attempts = attempt + 1
            _add_usage(result.usage, getattr(resp, "usage", None))
            msg = resp.choices[0].message
            raw_calls = list(getattr(msg, "tool_calls", None) or [])
            if not raw_calls:
                raw_calls = calls_from_text(getattr(msg, "content", None) or "")
                result.salvaged = result.salvaged or bool(raw_calls)
            calls, errors = self._validate(raw_calls, words)
            log.info(
                "nemotron model=%s attempt=%d latency=%.2fs calls=%d errors=%d",
                self.model,
                attempt + 1,
                dt,
                len(calls),
                len(errors),
            )
            if not raw_calls:
                text = (getattr(msg, "content", None) or "").strip()
                result.speech = text or None
                errors = ["no tool call; reply only with tool calls (use ask_user to talk)"]
            result.errors += errors
            if not errors:
                result.calls, result.speech = calls, None
                break
            if attempt == self.max_retries:
                result.calls = calls  # keep the valid ones
                break
            messages += _feedback_messages(msg, raw_calls, errors)
        result.calls, result.guarded = guard_label_commands(
            words, result.calls, context, self.categories
        )
        result.latency_s = clock.now() - t_start
        return result

    def _validate(self, raw_calls: list[Any], words: list[str]) -> tuple[list[ToolCall], list[str]]:
        calls: list[ToolCall] = []
        errors: list[str] = []
        cats = self.categories if self.check_categories else None
        for rc in raw_calls:
            fn = rc.function
            try:
                call = parse_call(fn.name, fn.arguments)
            except ToolCallError as e:
                errors.append(str(e))
                continue
            problems = check_against_transcript(call, words, cats)
            if problems:
                errors += problems
            else:
                calls.append(call)
        return calls, errors


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _TextCall:
    """Quacks like an OpenAI tool call; built from a call written as text."""

    id: str
    function: _Fn
    type: str = "function"


_TEXT_CALL = re.compile(r"^\s*`*([a-z_]+)\s*\((.*)\)\s*;?`*\s*$")


def calls_from_text(text: str) -> list[Any]:
    """Tool calls the model wrote as text in the few-shot notation, one per line, e.g.
    'cancel()' or 'place_drawing({"object": "cat"})'. Unknown names are ignored."""
    out: list[Any] = []
    for line in text.splitlines():
        m = _TEXT_CALL.match(line)
        if m and m.group(1) in TOOL_ARGS:
            args = m.group(2).strip() or "{}"
            out.append(_TextCall(id=f"text-{len(out)}", function=_Fn(m.group(1), args)))
    return out


_WRITE_VERBS = {"write", "label", "print"}
_GUARDED_TOOLS = {"new_sheet", "cancel", "confirm"}


def guard_label_commands(
    words: list[str],
    calls: list[ToolCall],
    context: AgentContext | None,
    categories: list[str],
) -> tuple[list[ToolCall], bool]:
    """If the person said "write ..." but the model returned new_sheet, cancel or confirm
    (e.g. "write new sheet there"), use the rule parser's reading instead. `stop` is never
    overridden: stopping by mistake is safe, not stopping isn't."""
    from pointdraw.agent.fallback_parser import FILLERS
    from pointdraw.agent.fallback_parser import parse as fallback_parse

    first = next((w for w in map(norm_word, words) if w and w not in FILLERS), "")
    if first in _WRITE_VERBS and any(c.name in _GUARDED_TOOLS for c in calls):
        fixed = fallback_parse(words, context, categories)
        if fixed and not any(c.name in _GUARDED_TOOLS for c in fixed):
            log.warning("guard: replaced %s with %s", calls, fixed)
            return fixed, True
    return calls, False


def _add_usage(total: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            total[k] = total.get(k, 0) + v


def _feedback_messages(msg: Any, raw_calls: list[Any], errors: list[str]) -> list[dict[str, Any]]:
    """The assistant turn as sent, a tool result per call, then the list of errors."""
    out: list[dict[str, Any]] = []
    if raw_calls:
        out.append(
            {
                "role": "assistant",
                "content": getattr(msg, "content", None) or "",
                "tool_calls": [
                    {
                        "id": rc.id,
                        "type": "function",
                        "function": {"name": rc.function.name, "arguments": rc.function.arguments},
                    }
                    for rc in raw_calls
                ],
            }
        )
        for rc in raw_calls:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": rc.id,
                    "content": "not executed: the reply had errors",
                }
            )
    else:
        out.append({"role": "assistant", "content": getattr(msg, "content", None) or ""})
    out.append(
        {
            "role": "user",
            "content": "Your reply had errors, so nothing was done:\n- "
            + "\n- ".join(errors)
            + "\nReply again with the complete, corrected list of tool calls.",
        }
    )
    return out


def understand(
    words: list[str],
    context: AgentContext | None = None,
    agent: NemotronAgent | None = None,
) -> AgentResult:
    """Nemotron if it's reachable and answers with calls, else the rule-based parser."""
    from pointdraw.agent.fallback_parser import parse as fallback_parse

    err = ""
    if agent is not None:
        try:
            res = agent.run(words, context)
            if res.ok:
                return res
            err = "; ".join(res.errors)
        except AgentUnavailable as e:
            err = str(e)
            log.warning("nemotron unavailable, using fallback parser: %s", e)
    t0 = clock.now()
    calls = fallback_parse(words, context)
    return AgentResult(
        calls=calls,
        source="fallback",
        latency_s=clock.now() - t0,
        errors=[err] if err else [],
    )


def dump_calls(calls: list[ToolCall]) -> str:
    return json.dumps([c.to_dict() for c in calls])
