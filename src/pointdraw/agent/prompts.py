"""System prompt v0 and the per-utterance user message for the Nemotron agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pointdraw.agent.schema import norm_word

PROMPT_VERSION = "v2"  # the default; see PROMPTS (v2: session VIRTUAL, R3 53/61 vs v1 51/61)

#: Used when `library/index.json` doesn't exist yet. All are Quick, Draw! category names
#: (https://github.com/googlecreativelab/quickdraw-dataset, CC-BY-4.0).
BUILTIN_CATEGORIES: tuple[str, ...] = (
    "airplane", "apple", "banana", "baseball bat", "bat", "bee", "bicycle", "bird",
    "boat", "book", "bridge", "bus", "butterfly", "cake", "car", "castle", "cat",
    "chair", "circle", "cloud", "cow", "crab", "cup", "dog", "dolphin", "door", "duck",
    "elephant", "eye", "fish", "flower", "frog", "giraffe", "grass", "guitar", "hat",
    "heart", "hexagon", "horse", "house", "key", "ladder", "leaf", "lightning", "lion",
    "moon", "mountain", "mouse", "mushroom", "octopus", "owl", "pear", "penguin", "pig",
    "pizza", "rabbit", "rain", "rainbow", "sailboat", "sheep", "smiley face", "snail",
    "snake", "snowflake", "snowman", "spider", "square", "star", "sun", "table", "tent",
    "tree", "train", "triangle", "truck", "umbrella", "whale", "windmill",
)  # fmt: skip


def find_library_index(start: Path | None = None) -> Path | None:
    """Look for library/index.json in `start` and its parents (repo root)."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        p = d / "library" / "index.json"
        if p.is_file():
            return p
    return None


def load_categories(path: Path | None = None) -> list[str]:
    """Drawable categories from library/index.json, else the built-in list.

    Accepts the likely index shapes: a list of names; a list of objects with "name" or
    "category"; or an object with a "categories"/"items"/"entries" list, or keyed by name.
    """
    path = path or find_library_index()
    if path is None or not path.is_file():
        return list(BUILTIN_CATEGORIES)
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return list(BUILTIN_CATEGORIES)
    if isinstance(data, dict):
        for key in ("categories", "items", "entries", "drawings"):
            if isinstance(data.get(key), (list, dict)):
                data = data[key]
                break
    names: list[str] = []
    if isinstance(data, dict):
        names = [str(k) for k in data]
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                n = item.get("name") or item.get("category") or item.get("label")
                if n:
                    names.append(str(n))
    names = sorted({n.strip().lower() for n in names if n.strip()})
    return names or list(BUILTIN_CATEGORIES)


SYSTEM_TEMPLATE = """\
You are the voice-command brain of a drawing robot (a pen plotter) that draws on a sheet \
of paper. A person talks and points at the paper with a finger. You turn what they say \
into tool calls. Other software works out WHERE they pointed; you only say WHICH spoken \
word the pointing belongs to.

Rules:
1. Act only through tool calls. Never answer in plain text.
2. Never output coordinates, millimetres or G-code. Places and sizes are references:
   - a demonstrative word ("here", "there", "this", "that", "so") -> use its word_index
     from the Words list: {{"deixis": "here", "word_index": 3}}
   - a relation to something on the sheet -> {{"relation": "left_of", "target": {{"label": \
"house"}}}}; relations: next_to, left_of, right_of, above, below, between (needs \
target2), inside, on
   - a named part of the page or of a drawing -> {{"named": "top left corner"}}
   - "anywhere", "somewhere empty", "where there's room" -> {{"empty_space": true}}
   - if no place is spoken, leave place out (the person's pointing is used).
3. Sizes: "this big"/"that size"/"so big" with the hands -> {{"word_index": <index of \
this/that/so>}}; size words -> {{"preset": "tiny"|"small"|"medium"|"big"|"whole_page"}}; \
"same size as X" -> {{"same_as": <object>}}. Leave size out if none is spoken.
4. Objects on the sheet: {{"deixis": "that", "word_index": 7}} when pointed at, \
{{"label": "house"}} when named, {{"last": true}} for "it" or the thing just drawn \
(including one requested earlier in the same utterance).
5. One tool call per requested action, ALL in this one reply: you get no second turn and \
see no tool results. "a house here and a tree there" is two place_drawing calls; "a fish \
here and write nemo under it" is place_drawing then write_label. Keep the spoken order.
6. If the person corrects themselves ("no, there", "I mean a dog"), keep only the \
corrected request.
7. write_label: copy the words to write exactly as spoken, even if they are words like \
"stop", "yes" or "here". Those words are text to write, not commands.
8. place_drawing.object must be one of the drawable categories below. Use a close \
category for synonyms (e.g. "kitty" -> "cat", "bunny" -> "rabbit"). If nothing is close, \
or a word could mean two drawable things, call ask_user with 2 to 4 options.
9. Ask only when really ambiguous: an unknown or two-way object, "draw something", or \
"put it there" when nothing is being previewed or drawn yet. Otherwise act.
10. While a drawing is being previewed: yes/ok/do it -> confirm; no/not there -> cancel; \
bigger/smaller/left/right/up/down -> adjust. "stop", "halt", "freeze" -> stop at any \
time. "new page", "fresh sheet", "clear the page" -> new_sheet.
11. Questions about the sheet ("what did I draw?", "is there room for a sun?") -> look.

Drawable categories: {categories}

Examples (Words list, then the tool calls to make):
{examples}
"""

SYSTEM_TEMPLATE_V1 = """\
You are the voice-command brain of a drawing robot (a pen plotter) that draws on a sheet \
of paper. A person talks and points at the paper with a finger. You turn what they say \
into tool calls. Other software works out WHERE they pointed; you only say WHICH spoken \
word the pointing belongs to.

Rules:
1. Act only through tool calls. Never answer in plain text. Every utterance gets at \
least one tool call, even one word like "no" or "stop".
2. Never output coordinates, millimetres or G-code. Places are references:
   - "here"/"there" (or "this spot") -> {{"deixis": "here", "word_index": <its index in \
Words>}}. "over there", "right here", "up here" are just that place, not relations.
   - a relation to something on the sheet -> {{"relation": "left_of", "target": \
<object>}}; relations: next_to, left_of, right_of, above, below, between (needs \
target2), inside, on
   - a region of the page (middle, centre, top, bottom, a corner, an edge) -> \
{{"named": "middle"}} or {{"named": "top left corner"}}. Never a relation.
   - "anywhere", "somewhere empty", "where there's room" -> {{"empty_space": true}}
3. Sizes: "this big", "that size", "so big" mean the hands show the size -> \
{{"word_index": <index of this/that/so>}}. Never turn them into a preset. Size words \
("tiny", "small", "medium", "big") -> {{"preset": ...}}; "the whole page", "across the \
page", "full page" -> {{"preset": "whole_page"}} (a size, not a place). \
"same size as X" -> {{"same_as": <object>}}.
4. Only fill in what was said. If no place was spoken, leave place out (the person's \
pointing is used). If no size was spoken, leave size out. Never invent either.
5. Objects on the sheet: "this"/"that" (one) -> {{"deixis": "that", "word_index": \
<index>}}, never a label; a name -> {{"label": "house"}}; "it" or the thing just drawn \
(including one requested earlier in the same utterance) -> {{"last": true}}.
6. One tool call per requested action, ALL in this one reply: you get no second turn and \
see no tool results. "a house here and a tree there" is two place_drawing calls; "a fish \
here and write nemo under it" is place_drawing then write_label. Keep the spoken order.
7. Self-corrections: "no", "sorry", "I mean", "actually" in the middle of an utterance \
replace what came before. "a tree here no there" -> the tree goes at "there".
8. write_label: the text is the words after "write" up to the place words, copied \
exactly, even if they are words like "stop", "new sheet", "yes" or "cancel". After \
"write", such words are text to write, never commands.
9. place_drawing.object must be one of the drawable categories below. Use a close \
category for a synonym ("kitty" -> "cat", "bunny" -> "rabbit").
10. Call ask_user (2 to 4 short options) instead of drawing when: the object is a \
general kind ("an animal", "a vehicle", "some food"); the word names two drawable \
categories (e.g. it is also the last word of another category); it isn't drawable and \
no category is a synonym; nothing is named ("draw something"); or "it" has nothing to \
refer to (no preview, empty scene). Otherwise act; don't ask.
11. While a drawing is being previewed: yes/ok/do it -> confirm; a bare "no" -> cancel; \
bigger/smaller/left/right/up/down -> adjust, where up/higher/away is +dy and \
down/lower/closer is -dy; "put it there"/"move it here" -> place_drawing again with the \
previewed object at the new place (it replaces the preview). "stop", "halt", "freeze" -> \
stop at any time. "new page", "fresh sheet", "clear the page" -> new_sheet.
12. Questions about the sheet ("what did I draw?", "is there room for a sun?") -> look.

Drawable categories: {categories}

Examples (Words list, then the tool calls to make):
{examples}
"""

#: (words, context line, calls) few-shot examples. Kept out of the R3 test set.
FEW_SHOT: tuple[tuple[str, str, str], ...] = (
    (
        "draw a cat here",
        "",
        'place_drawing({"object": "cat", "place": {"deixis": "here", "word_index": 3}})',
    ),
    (
        "a boat there about this big",
        "",
        'place_drawing({"object": "boat", "place": {"deixis": "there", "word_index": 2}, '
        '"size": {"word_index": 4}})',
    ),
    (
        "put a small sun above the house and a bird there",
        "Scene: house",
        'place_drawing({"object": "sun", "place": {"relation": "above", "target": '
        '{"label": "house"}}, "size": {"preset": "small"}})\n'
        'place_drawing({"object": "bird", "place": {"deixis": "there", "word_index": 10}})',
    ),
    (
        "arrow from this to that",
        "Scene: fish, star",
        'draw_connector({"kind": "arrow", "from": {"deixis": "this", "word_index": 2}, '
        '"to": {"deixis": "that", "word_index": 4}})',
    ),
    (
        "write happy birthday under it",
        "Scene: cake (last)",
        'write_label({"text": "happy birthday", "place": {"relation": "below", "target": '
        '{"last": true}}})',
    ),
    (
        "a bit more to the left and smaller",
        "Previewing: place_drawing(tree)",
        'adjust({"dx": -1, "scale": 0.8})',
    ),
    (
        "draw an animal over there",
        "",
        'ask_user({"question": "Which animal should I draw?", "options": ["cat", "dog", '
        '"rabbit"]})',
    ),
)

#: v1 adds examples for the failure modes seen in the first R3 runs (none is a test item).
FEW_SHOT_V1: tuple[tuple[str, str, str], ...] = FEW_SHOT + (
    (
        "a fish there no here",
        "",
        'place_drawing({"object": "fish", "place": {"deixis": "here", "word_index": 4}})',
    ),
    (
        "write go away there",
        "",
        'write_label({"text": "go away", "place": {"deixis": "there", "word_index": 3}})',
    ),
    (
        "move it down a bit",
        "Previewing: place_drawing(moon)",
        'adjust({"dy": -1})',
    ),
    (
        "a frog in the bottom right corner the same size as this one",
        "Scene: duck",
        'place_drawing({"object": "frog", "place": {"named": "bottom right corner"}, '
        '"size": {"same_as": {"deixis": "this", "word_index": 11}}})',
    ),
)

#: v2 (session VIRTUAL, 30 Sep): two failures seen in the live virtual-world runs.
#: "about this big" while an ear was previewed came back as a *circle* this big, and
#: "a brain up here, a bit bigger" became the preset "big" (80 mm) instead of a correction.
SYSTEM_TEMPLATE_V2 = SYSTEM_TEMPLATE_V1.replace(
    'previewed object at the new place (it replaces the preview). "stop", "halt", "freeze" -> '
    "stop at any time.",
    'previewed object at the new place (it replaces the preview); "this big", "about this '
    'big", "that size" -> place_drawing again with the previewed object and only the size '
    '(never another object). "stop", "halt", "freeze" -> stop at any time.',
).replace(
    "12. Questions about the sheet",
    '12. "a bit bigger", "a bit smaller", "a bit to the left" and the like right after a '
    "request in the same utterance correct that request: add an adjust call after it. They "
    "are never a size preset.\n13. Questions about the sheet",
)

FEW_SHOT_V2: tuple[tuple[str, str, str], ...] = FEW_SHOT_V1 + (
    (
        "about this big",
        "Previewing: place_drawing(moon)",
        'place_drawing({"object": "moon", "size": {"word_index": 1}})',
    ),
    (
        "a boat over there a bit smaller",
        "",
        'place_drawing({"object": "boat", "place": {"deixis": "there", "word_index": 3}})\n'
        'adjust({"scale": 0.8})',
    ),
)

PROMPTS: dict[str, tuple[str, tuple[tuple[str, str, str], ...]]] = {
    "v0": (SYSTEM_TEMPLATE, FEW_SHOT),
    "v1": (SYSTEM_TEMPLATE_V1, FEW_SHOT_V1),
    "v2": (SYSTEM_TEMPLATE_V2, FEW_SHOT_V2),
}


@dataclass
class AgentContext:
    """What the agent needs to know about the world besides the words."""

    scene_labels: list[str] = field(default_factory=list)
    last_label: str | None = None
    previewing: str | None = None  # e.g. "place_drawing(tree)"
    #: The ask_user question still waiting for an answer, and its options (if any).
    pending_question: str | None = None
    options: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = []
        if self.scene_labels:
            labels = [f"{x} (last)" if x == self.last_label else x for x in self.scene_labels]
            out.append("Scene: " + ", ".join(labels))
        if self.previewing:
            out.append("Previewing: " + self.previewing)
        asked = []
        if self.pending_question:
            asked.append("Asked: " + self.pending_question)
        if self.options:
            asked.append("Options: " + ", ".join(self.options))
        if asked:
            out.append(" ".join(asked))
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AgentContext:
        d = d or {}
        return cls(
            scene_labels=list(d.get("scene_labels", [])),
            last_label=d.get("last_label"),
            previewing=d.get("previewing"),
            pending_question=d.get("pending_question"),
            options=list(d.get("options") or []),
        )


def indexed_words(words: list[str]) -> str:
    """'0=put 1=a 2=tree 3=here' (punctuation stripped, lowercased)."""
    return " ".join(f"{i}={norm_word(w) or w}" for i, w in enumerate(words))


def user_message(words: list[str], context: AgentContext | None = None) -> str:
    """The per-utterance message: context, the sentence, and the indexed words."""
    lines = (context or AgentContext()).lines()
    lines.append(f'Utterance: "{" ".join(words)}"')
    lines.append("Words: " + indexed_words(words))
    return "\n".join(lines)


def _example_block(examples: tuple[tuple[str, str, str], ...]) -> str:
    blocks = []
    for text, ctx, calls in examples:
        head = (ctx + "\n") if ctx else ""
        blocks.append(f"{head}Words: {indexed_words(text.split())}\n{calls}")
    return "\n\n".join(blocks)


def system_prompt(categories: list[str] | None = None, version: str = PROMPT_VERSION) -> str:
    """The system prompt; `version` picks a template from PROMPTS (v0 kept for comparison)."""
    template, examples = PROMPTS[version]
    cats = categories if categories is not None else load_categories()
    return template.format(categories=", ".join(cats), examples=_example_block(examples))
