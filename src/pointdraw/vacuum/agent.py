"""The vacuum's agent: what the person said, as the vacuum's own tool calls.

The tools are the vacuum profile's actions (clean, dock, adjust, confirm, cancel, stop,
ask_user), as the deixis core describes them (`tools_for_agent`, over MCP in the app). Places
are references to the words ("here" is word 1), never coordinates: the core finds the place.

`VacuumAgent` asks Nemotron on Nebius Token Factory (one call per utterance, one retry with
the validation errors). `parse` is a small rule parser for the vacuum's few phrases: the
fallback when Nemotron can't be reached, and the agent for `--ai rules` and the tests.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from deixis.actions import Call, CallError, check_call, parse_call
from deixis.profile import Profile
from deixis.refs import DEMONSTRATIVES, norm_word
from pointdraw import clock
from pointdraw.agent.nemotron import DEFAULT_MODEL, AgentResult, AgentUnavailable, calls_from_text
from pointdraw.config import Settings, get_settings

log = logging.getLogger("pointdraw.vacuum.agent")


@dataclass
class Context:
    """What the agent should know about the moment."""

    previewing: bool = False  # a cleaning area is shown and waits for a yes
    cleaning: bool = False  # the robot is out cleaning
    question: str | None = None  # the app's open question, if any

    def lines(self) -> list[str]:
        out = []
        if self.previewing:
            out.append("A cleaning area is shown and waits for the person's yes or no.")
        if self.cleaning:
            out.append("The robot is cleaning.")
        if self.question:
            out.append(f'The robot just asked: "{self.question}"')
        return out


SYSTEM = """You turn what a person says to a robot vacuum into the robot's tool calls.
The person stands on the floor and talks; a camera sees where they stand.

Rules:
- Reply only with tool calls. Several calls are fine. Never reply with plain text.
- Never write coordinates or sizes. A place is a reference to the person's words:
  {"deixis": "here", "word_index": N}, where N is that word's number in the Words list.
- "clean here", "vacuum there", "clean this spot" ->
  clean(place={"deixis": "here", "word_index": N}).
- "clean from here to here", "from here to there" ->
  clean(place={"from": {"deixis": "here", "word_index": A}, "to": {"deixis": "here", "word_index": B}}).
- "clean" with no pointing word, "clean where I am" -> clean(place={"pointed": true}).
- "yes", "ok", "go ahead", "do it" -> confirm(). "no", "cancel", "never mind" -> cancel().
- "no, here" (another place) -> clean with the new place.
- "stop", "halt", "wait", "freeze" -> stop().
- "go home", "go back to the dock", "go and charge" -> dock().
- "a bit left" -> adjust(dx=-1); "a bit right" -> adjust(dx=1); "further away" -> adjust(dy=1);
  "closer" -> adjust(dy=-1); "a lot" means 3 steps; "bigger" -> adjust(scale=1.25);
  "smaller" -> adjust(scale=0.8).
- If you really cannot tell what they want, call ask_user with one short question.

Examples:
Words: 0:clean 1:here
clean({"place": {"deixis": "here", "word_index": 1}})

Words: 0:please 1:clean 2:from 3:here 4:to 5:here
clean({"place": {"from": {"deixis": "here", "word_index": 3}, "to": {"deixis": "here", "word_index": 5}}})

Words: 0:yes 1:go 2:ahead
confirm({})

Words: 0:a 1:bit 2:more 3:to 4:the 5:left
adjust({"dx": -1})

Words: 0:go 1:back 2:to 3:your 4:dock
dock({})"""  # noqa: E501


def indexed(words: Sequence[str]) -> str:
    return " ".join(f"{i}:{w}" for i, w in enumerate(words))


def user_message(words: Sequence[str], context: Context | None = None) -> str:
    lines = (context or Context()).lines()
    lines.append(f'Utterance: "{" ".join(words)}"')
    lines.append("Words: " + indexed(words))
    return "\n".join(lines)


# ------------------------------------------------------------------ the rule parser

STOP = {"stop", "halt", "freeze", "wait", "pause"}
DOCK = {"dock", "home", "charge", "charging", "base", "charger"}
YES = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "correct", "go", "do"}
NO = {"no", "nope", "cancel", "nevermind"}
CLEAN = {"clean", "vacuum", "hoover", "sweep", "mop"}
LEFT, RIGHT = {"left"}, {"right"}
AWAY = {"further", "farther", "away", "forward", "forwards"}
CLOSER = {"closer", "nearer", "toward", "towards", "back"}
BIGGER, SMALLER = {"bigger", "larger", "wider"}, {"smaller", "narrower", "less"}
FILLERS = {"um", "uh", "er", "so", "well", "please", "now", "and", "oh"}
YES_PHRASES = {"thats right", "all right", "alright", "go ahead", "do it", "sounds good", "fine"}


def parse(words: Sequence[str], context: Context | None = None) -> list[dict[str, Any]]:
    """The vacuum's phrases, by rule: raw calls ({"name", "args"}), or [] if not understood."""
    ws = [norm_word(w).replace("'", "") for w in words]
    s = set(ws)
    if s & STOP and not s & CLEAN:
        return [{"name": "stop", "args": {}}]
    if s & DOCK:
        return [{"name": "dock", "args": {}}]
    demos = [i for i, w in enumerate(ws) if w in DEMONSTRATIVES and w != "so"]
    spots = [i for i in demos if ws[i] in ("here", "there")]
    if s & CLEAN or (spots and (s & NO or not s & YES)):
        if len(demos) >= 2 and "from" in ws[: demos[0]] and "to" in ws[demos[0] : demos[1]]:
            a, b = demos[0], demos[1]
            place: dict[str, Any] = {
                "from": {"deixis": ws[a], "word_index": a},
                "to": {"deixis": ws[b], "word_index": b},
            }
        elif demos:
            place = {"deixis": ws[demos[-1]], "word_index": demos[-1]}
        else:
            place = {"pointed": True}
        return [{"name": "clean", "args": {"place": place}}]
    first = next((w for w in ws if w not in FILLERS), "")
    if first in NO or ("never" in s and "mind" in s):
        return [{"name": "cancel", "args": {}}]
    said = " ".join(w for w in ws if w not in FILLERS)
    if said in YES_PHRASES or (said == "right" and context is not None and context.previewing):
        return [{"name": "confirm", "args": {}}]
    steps = 3 if s & {"lot", "much", "far"} else 1
    adj: dict[str, Any] = {}
    if s & LEFT:
        adj["dx"] = -steps
    elif s & RIGHT and first not in YES:
        adj["dx"] = steps
    if s & AWAY:
        adj["dy"] = steps
    elif s & CLOSER:
        adj["dy"] = -steps
    if s & BIGGER:
        adj["scale"] = 1.25
    elif s & SMALLER:
        adj["scale"] = 0.8
    if adj:
        return [{"name": "adjust", "args": adj}]
    if first in YES:
        return [{"name": "confirm", "args": {}}]
    return []


def validate(
    profile: Profile, raw: Sequence[dict[str, Any]], words: Sequence[str]
) -> tuple[list[Call], list[str]]:
    """The raw calls the profile accepts, and the problems with the others (for the model)."""
    calls: list[Call] = []
    errors: list[str] = []
    for r in raw:
        try:
            call = parse_call(profile, r["name"], r.get("args"))
        except CallError as e:
            errors.append(str(e))
            continue
        problems = check_call(call, list(words))
        if problems:
            errors += problems
        else:
            calls.append(call)
    return calls, errors


class RuleAgent:
    """The rule parser, as an agent (`--ai rules`, tests)."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile

    def understand(self, words: Sequence[str], context: Context | None = None) -> AgentResult:
        t0 = clock.now()
        calls, errors = validate(self.profile, parse(words, context), words)
        return AgentResult(calls, source="fallback", latency_s=clock.now() - t0, errors=errors)


@dataclass
class VacuumAgent:
    """Nemotron with the vacuum's tools; the rule parser if it fails."""

    profile: Profile
    tools: list[dict[str, Any]]  # tools_for_agent(profile), from the core
    settings: Settings = field(default_factory=get_settings)
    model: str | None = None
    client: Any = None  # an OpenAI client (tests pass a fake)
    timeout_s: float = 8.0
    max_retries: int = 1

    def _client(self) -> Any:
        if self.client is None:
            key = self.settings.nebius_api_key.get_secret_value()
            if not key:
                raise AgentUnavailable("NEBIUS_API_KEY is not set (see .env.example)")
            from openai import OpenAI

            self.client = OpenAI(
                base_url=self.settings.nebius_base_url,
                api_key=key,
                timeout=self.timeout_s,
                max_retries=0,
            )
        return self.client

    def run(self, words: Sequence[str], context: Context | None = None) -> AgentResult:
        model = self.model or self.settings.nemotron_model or DEFAULT_MODEL
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(words, context)},
        ]
        result = AgentResult(calls=[], model=model, attempts=0)
        t0 = clock.now()
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client().chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=512,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except AgentUnavailable:
                raise
            except Exception as e:  # openai errors, timeouts
                raise AgentUnavailable(f"{type(e).__name__}: {e}") from e
            result.attempts = attempt + 1
            msg = resp.choices[0].message
            raw_calls = list(getattr(msg, "tool_calls", None) or [])
            if not raw_calls:
                raw_calls = calls_from_text(getattr(msg, "content", None) or "")
            raw = [{"name": c.function.name, "args": c.function.arguments} for c in raw_calls]
            calls, errors = validate(self.profile, raw, words)
            if not raw:
                errors = ["no tool call; reply only with tool calls (use ask_user to talk)"]
            result.errors += errors
            if not errors:
                result.calls = calls
                break
            result.calls = calls
            if attempt < self.max_retries:
                messages.append({"role": "assistant", "content": json.dumps(raw)})
                messages.append(
                    {
                        "role": "user",
                        "content": "Your reply had errors, so nothing was done:\n- "
                        + "\n- ".join(errors)
                        + "\nReply again with the complete, corrected list of tool calls.",
                    }
                )
        result.latency_s = clock.now() - t0
        log.info("vacuum agent: %s in %.2f s", [str(c) for c in result.calls], result.latency_s)
        return result

    def understand(self, words: Sequence[str], context: Context | None = None) -> AgentResult:
        """Nemotron if it answers with calls, else the rule parser."""
        err = ""
        try:
            res = self.run(words, context)
            if res.ok:
                return res
            err = "; ".join(res.errors)
        except AgentUnavailable as e:
            err = str(e)
            log.warning("nemotron unavailable, using the rule parser: %s", e)
        res = RuleAgent(self.profile).understand(words, context)
        res.errors = [err, *res.errors] if err else res.errors
        return res


class Background:
    """Runs an agent's `understand` in one worker thread, so the camera loop never waits."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vacuum-agent")

    def submit(self, words: Sequence[str], context: Context | None) -> Future[AgentResult]:
        return self._pool.submit(self._safe, list(words), context)

    def _safe(self, words: list[str], context: Context | None) -> AgentResult:
        try:
            return self.agent.understand(words, context)
        except Exception as e:  # noqa: BLE001 - a reply must always come
            log.exception("vacuum agent failed")
            return AgentResult([], source="fallback", errors=[str(e)])

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
