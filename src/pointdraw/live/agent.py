"""Ask the agent in a background thread (the runtime's `submit` protocol).

Nemotron takes a second or two. Asked on the main loop, that would freeze the camera, and
with it the hands-clear interlock, while the plotter might be moving. Here the call runs in
one worker thread (one request at a time, in order) and the runtime collects the reply on a
later tick. Any failure falls back to the rule parser, so a reply always arrives.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from pointdraw.agent.fallback_parser import parse as fallback_parse
from pointdraw.agent.nemotron import AgentResult
from pointdraw.agent.prompts import AgentContext

log = logging.getLogger(__name__)


class BackgroundAgent:
    def __init__(self, agent: Any, categories: list[str]) -> None:
        self.agent = agent
        self.categories = categories
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")

    def understand(self, words: list[str], context: AgentContext) -> AgentResult:
        return self._safe(words, context)

    def submit(self, words: list[str], context: AgentContext) -> Future[AgentResult]:
        return self._pool.submit(self._safe, list(words), context)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _safe(self, words: list[str], context: AgentContext) -> AgentResult:
        try:
            return self.agent.understand(words, context)
        except Exception as e:  # noqa: BLE001 - the runtime must always get a reply
            log.exception("agent failed; using the rule parser")
            calls = fallback_parse(words, context, self.categories)
            return AgentResult(calls=calls, source="fallback", errors=[str(e)])
