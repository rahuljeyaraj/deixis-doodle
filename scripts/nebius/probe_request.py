"""Write the probe's request body: the plotter's real Nemotron request for one fixed sentence.

    uv run python scripts/nebius/probe_request.py            # -> out/probe/probe_request.json

The same system prompt, tools, model and settings the live app sends (`NemotronAgent`), for
"Draw a cat here this big" with an empty page, so the probe's times are the app's times.
No key in it. The Job injects it as a file (Nebius allows up to 64 KiB).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointdraw.agent.nemotron import NemotronAgent
from pointdraw.agent.prompts import AgentContext, user_message
from pointdraw.config import Settings

WORDS = ["Draw", "a", "cat", "here", "this", "big"]
LIMIT = 64 * 1024


def body() -> dict:
    a = NemotronAgent(Settings(_env_file=None))
    return {
        "model": a.model,
        "messages": [
            {"role": "system", "content": a.system},
            {"role": "user", "content": user_message(WORDS, AgentContext())},
        ],
        "tools": a.tools,
        "tool_choice": a.tool_choice,
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
        "chat_template_kwargs": {"enable_thinking": a.thinking},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("out/probe/probe_request.json"))
    a = ap.parse_args(argv)
    data = json.dumps(body(), separators=(",", ":")).encode("utf-8")
    if len(data) > LIMIT:
        raise SystemExit(f"{len(data)} bytes: over the 64 KiB inject limit")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(data)
    print(f"wrote {a.out} ({len(data)} bytes, model {json.loads(data)['model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
