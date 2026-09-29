"""Typed tool-call arguments for the agent, and the OpenAI `tools` JSON built from them.

Invariant (plan 4.1, 4.5): the language model never emits coordinates or G-code. Every
place, size and object is a *reference* (a spoken word index, a relation to a scene object,
a named region, a preset), which the deixis resolver later turns into a `deixis.Where`.

Reference shapes are told apart by their keys, so the model writes compact JSON:

- PlaceRef:  {"deixis": "here", "word_index": 3} | {"relation": "left_of", "target": ObjectRef,
  "target2": ObjectRef?} | {"named": "top left corner"} | {"empty_space": true}
- SizeRef:   {"word_index": 5} | {"preset": "small"} | {"same_as": ObjectRef}
- ObjectRef: {"deixis": "that", "word_index": 7} | {"label": "house"} | {"last": true}

A few common slips (a bare "empty_space" string, a bare preset word, "large") are repaired
before validation, so the runtime sees one clean shape.
"""

from __future__ import annotations

import copy
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    model_validator,
)

#: Words that can carry a pointing gesture ("so" as in "so big").
DEMONSTRATIVES = ("here", "there", "this", "that", "these", "those", "so")

Demonstrative = Literal["here", "there", "this", "that", "these", "those", "so"]
Relation = Literal["next_to", "left_of", "right_of", "above", "below", "between", "inside", "on"]
SizePreset = Literal["tiny", "small", "medium", "big", "whole_page"]
ConnectorKind = Literal["arrow", "line", "double_arrow"]

#: Sizes in mm for each preset: the larger side of the object's box (plan 4.3).
#: None means "fit the page"; the resolver computes it.
PRESET_MM: dict[str, float | None] = {
    "tiny": 20.0,
    "small": 35.0,
    "medium": 55.0,
    "big": 80.0,
    "whole_page": None,
}

_PRESET_SYNONYMS = {
    "large": "big",
    "huge": "big",
    "giant": "big",
    "little": "small",
    "mini": "tiny",
    "normal": "medium",
    "regular": "medium",
    "full_page": "whole_page",
    "page": "whole_page",
    "whole page": "whole_page",
    "the whole page": "whole_page",
}

_RELATION_SYNONYMS = {
    "next to": "next_to",
    "beside": "next_to",
    "near": "next_to",
    "left of": "left_of",
    "right of": "right_of",
    "over": "above",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "in": "inside",
    "on_top_of": "on",
    "on top of": "on",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------- object references


class DeicticObject(_Strict):
    """The scene object nearest where the person pointed while saying the word."""

    deixis: Demonstrative = Field(description="The demonstrative word as spoken.")
    word_index: int = Field(ge=0, description="Index of that word in the transcript.")


class LabelObject(_Strict):
    """A scene object by its label, e.g. "house" or "the circle I drew"."""

    label: str = Field(min_length=1, description="Label of an object in the scene.")


class LastObject(_Strict):
    """The most recently drawn or previewed object ("it", "that one you just drew")."""

    model_config = ConfigDict(json_schema_extra={"required": ["last"]})

    last: Literal[True] = True


def _object_tag(v: Any) -> str | None:
    if isinstance(v, BaseModel):
        return {DeicticObject: "deictic", LabelObject: "label", LastObject: "last"}.get(type(v))
    if isinstance(v, dict):
        for key, tag in (("deixis", "deictic"), ("label", "label"), ("last", "last")):
            if key in v:
                return tag
    return None


def _repair_object(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("last", "it", "that one", "the last one"):
            return {"last": True}
        return {"label": v.strip()}
    if isinstance(v, dict) and "word_index" in v and "deixis" not in v:
        return {"deixis": "that", **v}
    if isinstance(v, dict) and isinstance(v.get("deixis"), str):
        d = v["deixis"].strip().lower()
        if d in ("it", "last"):
            return {"last": True}
        if d not in DEMONSTRATIVES:
            return {"label": v["deixis"].strip()}  # {"deixis": "sun"} means the label "sun"
    return v


ObjectRef = Annotated[
    Annotated[DeicticObject, Tag("deictic")]
    | Annotated[LabelObject, Tag("label")]
    | Annotated[LastObject, Tag("last")],
    Discriminator(
        _object_tag,
        custom_error_type="object_ref",
        custom_error_message='ObjectRef must have one of the keys "deixis", "label" or "last"',
    ),
    BeforeValidator(_repair_object),
]

# ---------------------------------------------------------------- place references


class DeicticPlace(_Strict):
    """Where the person pointed while saying a demonstrative ("here", "there")."""

    deixis: Demonstrative = Field(description="The demonstrative word as spoken.")
    word_index: int = Field(ge=0, description="Index of that word in the transcript.")


class RelationPlace(_Strict):
    """A place relative to scene objects, computed by geometry in the paper frame."""

    relation: Relation
    target: ObjectRef
    target2: ObjectRef | None = Field(default=None, description="Only for 'between'.")

    @model_validator(mode="after")
    def _between_needs_two(self) -> RelationPlace:
        if self.relation == "between" and self.target2 is None:
            raise ValueError("relation 'between' needs target2")
        if self.relation != "between" and self.target2 is not None:
            raise ValueError("target2 is only used with relation 'between'")
        return self


class NamedPlace(_Strict):
    """A place named in words: a page region ("top left corner", "middle") or a part of
    a scene object that needs vision ("on the roof")."""

    named: str = Field(min_length=1)


class EmptySpace(_Strict):
    """The largest empty area of the sheet."""

    model_config = ConfigDict(json_schema_extra={"required": ["empty_space"]})

    empty_space: Literal[True] = True


def _place_tag(v: Any) -> str | None:
    if isinstance(v, BaseModel):
        return {
            DeicticPlace: "deictic",
            RelationPlace: "relation",
            NamedPlace: "named",
            EmptySpace: "empty",
        }.get(type(v))
    if isinstance(v, dict):
        for key, tag in (
            ("deixis", "deictic"),
            ("relation", "relation"),
            ("named", "named"),
            ("empty_space", "empty"),
        ):
            if key in v:
                return tag
    return None


def _repair_place(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "_")
        if s in ("empty_space", "empty", "anywhere", "somewhere"):
            return {"empty_space": True}
        return {"named": v.strip()}
    if isinstance(v, dict) and "relation" in v and isinstance(v["relation"], str):
        rel = v["relation"].strip().lower()
        v = {**v, "relation": _RELATION_SYNONYMS.get(rel, rel.replace(" ", "_"))}
    if isinstance(v, dict) and v.get("target2", "") is None:
        v = {k: x for k, x in v.items() if k != "target2"}
    return v


PlaceRef = Annotated[
    Annotated[DeicticPlace, Tag("deictic")]
    | Annotated[RelationPlace, Tag("relation")]
    | Annotated[NamedPlace, Tag("named")]
    | Annotated[EmptySpace, Tag("empty")],
    Discriminator(
        _place_tag,
        custom_error_type="place_ref",
        custom_error_message=(
            'PlaceRef must have one of the keys "deixis", "relation", "named" or "empty_space"'
        ),
    ),
    BeforeValidator(_repair_place),
]

# ---------------------------------------------------------------- size references


class DeicticSize(_Strict):
    """A size shown with the hands while saying "this big" (index of "this")."""

    word_index: int = Field(ge=0, description='Index of "this"/"that"/"so" in "this big".')
    deixis: Demonstrative | None = None


class PresetSize(_Strict):
    """A spoken size word."""

    preset: SizePreset


class RelativeSize(_Strict):
    """The same size as another scene object."""

    same_as: ObjectRef


def _size_tag(v: Any) -> str | None:
    if isinstance(v, BaseModel):
        return {DeicticSize: "deictic", PresetSize: "preset", RelativeSize: "relative"}.get(type(v))
    if isinstance(v, dict):
        for key, tag in (("preset", "preset"), ("same_as", "relative"), ("word_index", "deictic")):
            if key in v:
                return tag
    return None


def _norm_preset(s: str) -> str:
    s = s.strip().lower()
    return _PRESET_SYNONYMS.get(s, s.replace(" ", "_"))


def _repair_size(v: Any) -> Any:
    if isinstance(v, str):
        return {"preset": _norm_preset(v)}
    if isinstance(v, dict) and isinstance(v.get("preset"), str):
        return {**v, "preset": _norm_preset(v["preset"])}
    return v


SizeRef = Annotated[
    Annotated[DeicticSize, Tag("deictic")]
    | Annotated[PresetSize, Tag("preset")]
    | Annotated[RelativeSize, Tag("relative")],
    Discriminator(
        _size_tag,
        custom_error_type="size_ref",
        custom_error_message='SizeRef must have one of the keys "word_index", "preset", "same_as"',
    ),
    BeforeValidator(_repair_size),
]

# ---------------------------------------------------------------- tool arguments


class PlaceDrawing(_Strict):
    """Draw a library object (a Quick, Draw! category) at a place and size."""

    object: str = Field(min_length=1, description="A drawable category, e.g. 'tree'.")
    place: PlaceRef | None = Field(
        default=None,
        description="Omit if no place was spoken: the resolver then uses the last pointing.",
    )
    size: SizeRef | None = Field(default=None, description="Omit if no size was spoken.")


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


class ToolCall(BaseModel):
    """One validated tool call. `args` is the tool's argument model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    args: BaseModel

    def args_dict(self) -> dict[str, Any]:
        return self.args.model_dump(by_alias=True, exclude_none=True, mode="json")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args_dict()}

    def __str__(self) -> str:  # compact, for logs
        import json

        return f"{self.name}({json.dumps(self.args_dict(), separators=(',', ':'))})"


class ToolCallError(ValueError):
    """A tool call the model got wrong; the message is written for the model to read."""


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
        model = TOOL_ARGS[name].model_validate(args)
    except ValidationError as e:
        raise ToolCallError(f"{name}: {_short_errors(e)}") from e
    return ToolCall(name=name, args=model)


def _short_errors(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:4]:
        loc = ".".join(str(p) for p in err["loc"] if not str(p).startswith("function-"))
        parts.append(f"{loc or 'args'}: {err['msg']}")
    return "; ".join(parts)


# ---------------------------------------------------------------- transcript checks

_PUNCT = re.compile(r"[^\w']+")


def norm_word(w: str) -> str:
    """Lowercase and strip punctuation: 'Here.' -> 'here'."""
    return _PUNCT.sub("", w.lower())


def _word_refs(obj: Any, path: str = "") -> list[tuple[str, int, str | None]]:
    """All (path, word_index, deixis) pairs inside a validated args model."""
    out: list[tuple[str, int, str | None]] = []
    if isinstance(obj, (DeicticPlace, DeicticObject, DeicticSize)):
        out.append((path, obj.word_index, obj.deixis))
    elif isinstance(obj, BaseModel):
        for field in type(obj).model_fields:
            out += _word_refs(getattr(obj, field), f"{path}.{field}" if path else field)
    return out


def check_against_transcript(
    call: ToolCall, words: list[str], categories: list[str] | None = None
) -> list[str]:
    """Problems with a call given the transcript: word indices that don't point at a
    demonstrative, and objects that aren't drawable. Empty list means fine."""
    problems = []
    normed = [norm_word(w) for w in words]
    for path, idx, deixis in _word_refs(call.args):
        if idx >= len(words):
            problems.append(f"{call.name}.{path}: word_index {idx} is past the last word")
        elif normed[idx] not in DEMONSTRATIVES:
            hint = ""
            if deixis and deixis in normed:
                hint = f"; '{deixis}' is word {normed.index(deixis)}"
            problems.append(
                f"{call.name}.{path}: word {idx} is '{words[idx]}', not a demonstrative{hint}"
            )
        elif deixis and normed[idx] != deixis:
            problems.append(
                f"{call.name}.{path}: word {idx} is '{words[idx]}' but deixis says '{deixis}'"
            )
    if categories and call.name == "place_drawing":
        obj = call.args.object.strip().lower()  # type: ignore[attr-defined]
        if obj not in {c.lower() for c in categories}:
            problems.append(
                f"place_drawing.object: '{call.args.object}' is not drawable; "  # type: ignore[attr-defined]
                "use the closest drawable category, or ask_user"
            )
    return problems


# ---------------------------------------------------------------- tools JSON


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs/$ref and drop titles, so the schema reads cleanly in a prompt."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
                extra = {k: walk(v) for k, v in node.items() if k != "$ref"}
                return {**target, **extra}
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    out = walk(schema)
    # A nullable field is anyOf[X, null]; drop the null branch and keep it optional.
    return _drop_null_branches(out)


def _drop_null_branches(node: Any) -> Any:
    if isinstance(node, dict):
        node = {k: _drop_null_branches(v) for k, v in node.items()}
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and any(b == {"type": "null"} for b in any_of):
            rest = [b for b in any_of if b != {"type": "null"}]
            node.pop("anyOf")
            if node.get("default", "x") is None:
                node.pop("default")
            if len(rest) == 1:
                node = {**rest[0], **node}
            else:
                node["anyOf"] = rest
        return node
    if isinstance(node, list):
        return [_drop_null_branches(x) for x in node]
    return node


def tool_parameters(name: str) -> dict[str, Any]:
    """JSON Schema for one tool's arguments."""
    schema = TOOL_ARGS[name].model_json_schema(by_alias=True)
    params = _inline_refs(schema)
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
