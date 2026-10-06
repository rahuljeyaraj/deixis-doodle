"""Typed tool-call arguments for the plotter's agent, and the OpenAI `tools` JSON built from
them.

Invariant (plan 4.1, 4.5): the language model never emits coordinates or G-code. Every
place, size and object is a *reference* (a spoken word index, a relation to a scene object,
a named region, a preset), which the deixis core later turns into a `deixis.Where`.

The reference types, their repairs and the transcript checks live in the core
(`deixis.refs`) and are re-exported here; this module keeps the plotter's own tools
(`place_drawing`, `draw_connector`, ...). Size presets and their synonyms come from the
plotter's profile.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from deixis.actions import Call, CallError, inline_schema, short_errors
from deixis.refs import (
    DEMONSTRATIVES,
    DeicticObject,
    DeicticPlace,
    DeicticSize,
    Demonstrative,
    EmptySpace,
    LabelObject,
    LastObject,
    NamedPlace,
    ObjectRef,
    PlaceRef,
    PointedPlace,
    PresetSize,
    Relation,
    RelationPlace,
    RelativeSize,
    SizeRef,
    TwoPlaces,
    norm_word,
    ref_context,
    transcript_problems,
    with_presets,
)
from pointdraw.profile import plotter_profile

__all__ = [
    "DEMONSTRATIVES",
    "PRESET_MM",
    "DeicticObject",
    "DeicticPlace",
    "DeicticSize",
    "Demonstrative",
    "EmptySpace",
    "LabelObject",
    "LastObject",
    "NamedPlace",
    "ObjectRef",
    "PlaceRef",
    "PointedPlace",
    "PresetSize",
    "Relation",
    "RelationPlace",
    "RelativeSize",
    "SizeRef",
    "TwoPlaces",
    "norm_word",
]

ConnectorKind = Literal["arrow", "line", "double_arrow"]

#: Sizes in mm for each preset: the diameter of the circle the drawing fills (6 Oct; the larger
#: side of its box until then), from the plotter's profile. None means "fit the page"; the
#: resolver computes it.
PRESET_MM: dict[str, float | None] = dict(plotter_profile().sizes.presets)

#: Validation context: presets and their synonyms ("large" -> "big") from the profile.
_CONTEXT = ref_context(plotter_profile())


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------- tool arguments


class PlaceDrawing(_Strict):
    """Draw a library object (a Quick, Draw! category) at a place and size."""

    object: str = Field(min_length=1, description="A drawable category, e.g. 'tree'.")
    place: PlaceRef | None = Field(
        default=None,
        description="Omit if no place was spoken: the resolver then uses the last pointing.",
    )
    size: SizeRef | None = Field(default=None, description="Omit if no size was spoken.")
    style: Literal["simple", "detailed"] | None = Field(
        default=None,
        description="Only if the person asked for a simple or a detailed drawing; else omit.",
    )


class DrawConnector(_Strict):
    """Draw an arrow or line between two objects or pointed places."""

    kind: ConnectorKind = "arrow"
    from_: ObjectRef = Field(alias="from")
    to: ObjectRef


class WriteLabel(_Strict):
    """Write text in single-stroke letters."""

    text: str = Field(min_length=1, description="The words to write, copied verbatim.")
    place: PlaceRef | None = None
    size: SizeRef | None = None


class Adjust(_Strict):
    """Change the object being previewed. Steps, not millimetres."""

    dx: int = Field(default=0, ge=-5, le=5, description="Steps to the user's right (+) or left.")
    dy: int = Field(default=0, ge=-5, le=5, description="Steps away from the user/up (+) or down.")
    scale: float = Field(default=1.0, ge=0.25, le=4.0, description="Size factor, 1 = unchanged.")


class NoArgs(_Strict):
    pass


class AskUser(_Strict):
    """Ask a short question when the request is ambiguous."""

    question: str = Field(min_length=1)
    options: Annotated[
        list[str],
        BeforeValidator(lambda v: v[:4] if isinstance(v, list) else v),
    ] = Field(default_factory=list, max_length=4)


class Look(_Strict):
    """Ask the vision model about the sheet."""

    query: str = Field(min_length=1)


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "place_drawing": PlaceDrawing,
    "draw_connector": DrawConnector,
    "write_label": WriteLabel,
    "adjust": Adjust,
    "confirm": NoArgs,
    "cancel": NoArgs,
    "stop": NoArgs,
    "new_sheet": NoArgs,
    "ask_user": AskUser,
    "look": Look,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "place_drawing": (
        "Draw one object from the drawable list at a place, at a size. Call once per object."
    ),
    "draw_connector": "Draw an arrow or line from one object or pointed spot to another.",
    "write_label": "Write text on the sheet. Copy the user's words for the text verbatim.",
    "adjust": (
        "Move or resize the drawing being previewed: dx/dy in steps (1 = a bit, 3 = a lot; "
        "+dx right, -dx left, +dy up/away, -dy down/toward the user), scale as a factor "
        "(1.25 bigger, 0.8 smaller, 1.5 much bigger, 0.6 much smaller)."
    ),
    "confirm": "The user accepts the preview (yes, ok, do it, go ahead, looks good).",
    "cancel": "The user rejects the preview or wants to undo the current request (no, cancel).",
    "stop": "Stop the robot immediately (stop, halt, wait, freeze).",
    "new_sheet": "The user wants a fresh sheet of paper; clears the scene.",
    "ask_user": (
        "Ask one short question, only when the request is really ambiguous or not drawable. "
        "Give 2 to 4 short options when you can."
    ),
    "look": "Ask the camera/vision model a question about the sheet (what is drawn, where).",
}


#: One validated tool call (`name`, `args` model) and the error a model must fix: the core's.
ToolCall = Call
ToolCallError = CallError


def parse_call(name: str, args: dict[str, Any] | str | None) -> ToolCall:
    """Validate one raw tool call. Raises ToolCallError with a model-readable message."""
    import json

    if name not in TOOL_ARGS:
        raise ToolCallError(f"unknown tool '{name}'. Tools: {', '.join(TOOL_ARGS)}")
    if args is None or args == "":
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise ToolCallError(f"{name}: arguments are not valid JSON ({e.msg})") from e
    if not isinstance(args, dict):
        raise ToolCallError(f"{name}: arguments must be a JSON object")
    try:
        model = TOOL_ARGS[name].model_validate(args, context=_CONTEXT)
    except ValidationError as e:
        raise ToolCallError(f"{name}: {short_errors(e)}") from e
    return ToolCall(name=name, args=model)


# ---------------------------------------------------------------- transcript checks


def check_against_transcript(
    call: ToolCall, words: list[str], categories: list[str] | None = None
) -> list[str]:
    """Problems with a call given the transcript: word indices that don't point at a
    demonstrative, labels that aren't names (both from the core), and objects that aren't
    drawable. Empty list means fine."""
    problems = transcript_problems(call.name, call.args, words)
    if categories and call.name == "place_drawing":
        obj = call.args.object.strip().lower()  # type: ignore[attr-defined]
        if obj not in {c.lower() for c in categories}:
            problems.append(
                f"place_drawing.object: '{call.args.object}' is not drawable; "  # type: ignore[attr-defined]
                "use the closest drawable category, or ask_user"
            )
    return problems


# ---------------------------------------------------------------- tools JSON


#: Reference shapes and fields the core accepts but the plotter's agent never needs. They are
#: left out of the tools JSON so the prompt stays as short as before the core split (the
#: tools JSON is sent with every request; longer prompts are slower).
_UNUSED_SHAPES = ("pointed", "from")
_UNUSED_OBJECT_FIELDS = ("label", "count")


def _lean(node: Any) -> Any:
    if isinstance(node, dict):
        node = {k: _lean(v) for k, v in node.items()}
        for key in ("oneOf", "anyOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                keep = [
                    b
                    for b in branches
                    if not (
                        isinstance(b, dict) and set(b.get("properties", {})) & set(_UNUSED_SHAPES)
                    )
                ]
                node[key] = keep
        props = node.get("properties")
        if isinstance(props, dict) and "deixis" in props and "word_index" in props:
            node["properties"] = {k: v for k, v in props.items() if k not in _UNUSED_OBJECT_FIELDS}
        return node
    if isinstance(node, list):
        return [_lean(x) for x in node]
    return node


def tool_parameters(name: str) -> dict[str, Any]:
    """JSON Schema for one tool's arguments."""
    schema = TOOL_ARGS[name].model_json_schema(by_alias=True)
    params = _lean(with_presets(inline_schema(schema), list(PRESET_MM)))
    params.pop("description", None)
    params.setdefault("properties", {})
    params["type"] = "object"
    return params


def openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """The OpenAI-format `tools` list for chat completions."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": tool_parameters(name),
            },
        }
        for name in names or list(TOOL_ARGS)
    ]
