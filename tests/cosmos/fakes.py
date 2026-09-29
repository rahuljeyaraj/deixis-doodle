"""A fake OpenAI client that records requests and returns scripted replies."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


class FakeOpenAI:
    def __init__(self, reply: str | Callable[[dict[str, Any]], str]) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.reply(kwargs) if callable(self.reply) else self.reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=2112, completion_tokens=40),
        )

    @staticmethod
    def prompt_text(kwargs: dict[str, Any]) -> str:
        return kwargs["messages"][0]["content"][1]["text"]
