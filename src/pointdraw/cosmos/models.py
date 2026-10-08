"""Per-model settings: sampling, reasoning and coordinate conventions.

Sources (checked 2026-09-29): the Nebius serverless-ai-cookbook Cosmos 3 Reasoner template
README (bbox_2d normalised 0-1000; sampling without and with reasoning), the Cosmos3-Nano,
Cosmos3-Edge and Cosmos-Reason2-8B model cards on Hugging Face, and rahul's 22 Sep runs.
See docs/spikes/r1-cosmos.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pointdraw.cosmos.parse import CoordMode

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"

# Greedy. Grounding is a measurement, not a conversation: we want the single most likely
# answer every time, not a varied one.
#
# Until 7 Oct this used Qwen3's *chat* settings (temperature 0.7, top_p 0.8, top_k 20,
# presence_penalty 1.5), copied from the cookbook. On a sheet with three drawings they made
# the model pair labels with the wrong boxes (car <-> elephant) in 3 runs out of 5, and made
# repeated point answers wander by 20 to 30 parts in 1000. At temperature 0 the same calls
# came back correct 10 times out of 10, and repeated points were identical. Qwen's own notes
# warn that a high presence_penalty costs accuracy.
SAMPLING_FAST: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
}
# With reasoning (<think>): the cookbook's second set.
SAMPLING_REASONING: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 0.0,
    "extra_body": {"top_k": 20},
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    coord_mode: CoordMode = "auto"  # "auto" guesses per reply; 0-1000 is the documented norm
    xy_order: str = "xy"
    reasons_by_default: bool = False
    max_tokens: int = 1024
    sampling: dict[str, Any] = field(default_factory=lambda: dict(SAMPLING_FAST))


PROFILES: dict[str, ModelProfile] = {
    # Tested 22 Sep: bbox_2d [x1, y1, x2, y2] normalised 0-1000, no reasoning unless asked.
    "nvidia/Cosmos3-Nano": ModelProfile("nvidia/Cosmos3-Nano", coord_mode="norm1000"),
    # Cookbook: "reasons by default", so allow room for <think> and use the reasoning sampling.
    "nvidia/Cosmos3-Edge": ModelProfile(
        "nvidia/Cosmos3-Edge",
        coord_mode="norm1000",
        reasons_by_default=True,
        max_tokens=4096,
        sampling=dict(SAMPLING_REASONING),
    ),
    # Qwen3-VL based. Sources disagree on 0-1000 vs 0-1, so detect per reply. The model card asks
    # for max_tokens >= 4096 because of its reasoning.
    "nvidia/Cosmos-Reason2-8B": ModelProfile(
        "nvidia/Cosmos-Reason2-8B",
        coord_mode="auto",
        reasons_by_default=True,
        max_tokens=4096,
        sampling=dict(SAMPLING_REASONING),
    ),
}


def profile_for(model: str) -> ModelProfile:
    """The profile for a model id; unknown models get safe defaults with auto coordinates."""
    if model in PROFILES:
        return PROFILES[model]
    for key, prof in PROFILES.items():
        if key.split("/")[-1].lower() in model.lower():
            return ModelProfile(
                model,
                coord_mode=prof.coord_mode,
                xy_order=prof.xy_order,
                reasons_by_default=prof.reasons_by_default,
                max_tokens=prof.max_tokens,
                sampling=dict(prof.sampling),
            )
    return ModelProfile(model)
