"""Rule-based parser for the core commands, used when Token Factory can't be reached.

It returns the same `ToolCall` objects as the Nemotron client, so the runtime doesn't
care which one answered. Coverage (plan 4.4): "<object> here", "this big", "arrow from
this to that", "write <text> here", "label that one <text>", yes/no/stop/cancel,
bigger/smaller, left/right/up/down (also trailing: "a brain here a bit bigger"), new sheet,
simple relations ("next to the house"), named places ("in the middle"), empty space,
self-corrections ("a cat, sorry, a dog here"), re-pointing or re-sizing a preview ("no,
there", "this big"), and unknown objects ("a dragon" -> ask_user with close categories).
"""

from __future__ import annotations

import difflib
import re

from pointdraw.agent.prompts import AgentContext, load_categories
from pointdraw.agent.schema import (
    Adjust,
    AskUser,
    DeicticObject,
    DeicticPlace,
    DeicticSize,
    DrawConnector,
    EmptySpace,
    LabelObject,
    LastObject,
    Look,
    NamedPlace,
    NoArgs,
    PlaceDrawing,
    PresetSize,
    RelationPlace,
    RelativeSize,
    ToolCall,
    WriteLabel,
    norm_word,
)

PLACE_WORDS = {"here", "there"}
POINT_WORDS = {"this", "that", "these", "those"}
SIZE_NOUNS = {"big", "size", "large", "tall", "wide", "long", "small", "high"}
SPOT_NOUNS = {"spot", "place", "point", "area", "side", "bit"}
ARTICLES = {"the", "a", "an", "my", "your", "our", "that", "this"}

PRESETS = {
    "tiny": "tiny",
    "mini": "tiny",
    "small": "small",
    "little": "small",
    "medium": "medium",
    "big": "big",
    "large": "big",
    "huge": "big",
    "giant": "big",
}
SYNONYMS = {
    "kitty": "cat",
    "kitten": "cat",
    "puppy": "dog",
    "doggy": "dog",
    "bunny": "rabbit",
    "home": "house",
    "ship": "boat",
    "plane": "airplane",
    "aeroplane": "airplane",
    "smiley": "smiley face",
    # Library names (library/index.json); used only when the name itself isn't drawable.
    "boat": "sailboat",
    "tv": "television",
    "telly": "television",
    "sofa": "couch",
    "lamp": "floor lamp",
    "bulb": "light bulb",
    "lightbulb": "light bulb",
    "mic": "microphone",
}
#: "draw an animal": general kinds get "Which animal?" with drawable members as options.
GENERAL_KINDS: dict[str, tuple[str, ...]] = {
    "animal": ("cat", "dog", "bird", "fish", "rabbit", "cow", "horse", "pig"),
    "vehicle": ("car", "bicycle", "sailboat", "bus", "truck", "train", "airplane", "boat"),
    "food": ("apple", "cake", "pizza", "banana", "pear"),
    "fruit": ("apple", "banana", "pear"),
    "furniture": ("chair", "table", "bed", "couch", "bench"),
    "plant": ("tree", "flower", "bush", "grass", "mushroom"),
    "shape": ("circle", "square", "triangle", "star", "cross", "heart"),
    "building": ("house", "castle", "windmill", "tent"),
}
RELATIONS: list[tuple[tuple[str, ...], str]] = [
    (("to", "the", "left", "of"), "left_of"),
    (("to", "the", "right", "of"), "right_of"),
    (("left", "of"), "left_of"),
    (("right", "of"), "right_of"),
    (("on", "top", "of"), "on"),
    (("next", "to"), "next_to"),
    (("beside",), "next_to"),
    (("near",), "next_to"),
    (("by",), "next_to"),
    (("above",), "above"),
    (("over",), "above"),
    (("below",), "below"),
    (("under",), "below"),
    (("underneath",), "below"),
    (("beneath",), "below"),
    (("between",), "between"),
    (("inside",), "inside"),
    (("in",), "inside"),
    (("on",), "on"),
]
NAMED_PLACES: list[tuple[tuple[str, ...], str]] = [
    (("top", "left"), "top left corner"),
    (("top", "right"), "top right corner"),
    (("bottom", "left"), "bottom left corner"),
    (("bottom", "right"), "bottom right corner"),
    (("upper", "left"), "top left corner"),
    (("upper", "right"), "top right corner"),
    (("lower", "left"), "bottom left corner"),
    (("lower", "right"), "bottom right corner"),
    (("in", "the", "middle"), "middle"),
    (("in", "the", "center"), "middle"),
    (("in", "the", "centre"), "middle"),
    (("at", "the", "top"), "top"),
    (("at", "the", "bottom"), "bottom"),
    (("in", "the", "corner"), "corner"),
]
EMPTY_PHRASES: list[tuple[str, ...]] = [
    ("anywhere",),
    ("somewhere",),
    ("wherever",),
    ("empty", "space"),
    ("empty", "spot"),
    ("empty", "place"),
    ("empty", "area"),
    ("empty", "bit"),
    ("free", "space"),
    ("free", "spot"),
    ("free", "place"),
    ("free", "area"),
    ("blank", "space"),
    ("blank", "spot"),
    ("open", "space"),
    ("where", "there's", "room"),
    ("where", "there", "is", "room"),
]
#: "where there's (some) room/space", "where it fits": "where" and one of these soon after.
ROOM_WORDS = {"room", "space", "fits"}
CONFIRM = {
    ("yes",), ("yeah",), ("yep",), ("yup",), ("ok",), ("okay",), ("sure",), ("perfect",),
    ("great",), ("fine",), ("good",), ("do", "it"), ("go", "ahead"), ("go",), ("draw", "it"),
    ("looks", "good"), ("that's", "good"), ("that's", "fine"), ("that's", "perfect"),
    ("yes", "please"), ("yes", "draw", "it"),
}  # fmt: skip
CANCEL_WORDS = {"cancel", "nevermind", "undo"}
CANCEL_PHRASES = [("never", "mind"), ("forget", "it"), ("not", "that")]
NEGATIONS = {"no", "nope", "nah"}
STOP_WORDS = {"stop", "halt", "freeze"}
WRITE_VERBS = {"write", "label", "print"}
CONNECT_NOUNS = {"arrow": "arrow", "line": "line", "arrows": "double_arrow"}
CORRECTION_MARKERS: list[tuple[str, ...]] = [("sorry",), ("i", "mean"), ("actually",), ("no",)]
FILLERS = {"please", "um", "uh", "er", "erm", "okay", "ok", "so", "now", "and", "then"}
RELATION_HEADS = {
    "of", "to", "above", "below", "under", "over", "beside", "near", "between", "inside",
    "on", "by", "as", "underneath", "beneath", "from", "and", "in",
}  # fmt: skip
QUESTION_STARTS = {"what", "what's", "whats", "is", "are", "where", "where's", "how", "which"}
PRONOUNS = {"it", "him", "her", "them"}

# A correction spoken after the place: "a brain up here a bit bigger".
COMPARATIVES = {"bigger", "larger", "smaller", "tinier", "higher", "lower", "further", "closer"}
DEGREES = {"bit", "little", "lot", "lots", "much", "way", "slightly", "tad", "touch"}
DIRECTIONS = {"left", "right", "up", "down", "away"}
ADJUST_TAIL = COMPARATIVES | DEGREES | DIRECTIONS | {
    "a", "more", "to", "the", "and", "but", "make", "move", "it", "just", "please", "then",
}  # fmt: skip
#: After "little", these mean an amount ("a little bit"), not a size ("a little bird").
AMOUNT_AFTER_LITTLE = {"bit", "more", "less", "to"} | COMPARATIVES | DIRECTIONS

# Unknown objects: the noun after the first article or draw verb ("draw me a dragon").
DRAW_VERBS = {"draw", "add", "make", "put", "sketch", "place"}
NOUN_STARTS = {"a", "an", "another", "some", "the"} | DRAW_VERBS
NOUN_SKIP = set(PRESETS) | {
    "me", "us", "a", "an", "the", "some", "another", "nice", "cute", "pretty", "cool", "funny",
    "very", "really", "different", "other", "new", "second", "red", "blue", "green", "yellow",
    "black", "orange", "purple", "pink", "brown", "white", "grey", "gray",
}  # fmt: skip
NOT_NOUNS = (
    PLACE_WORDS | POINT_WORDS | PRONOUNS | FILLERS | NEGATIONS | RELATION_HEADS
    | QUESTION_STARTS | DIRECTIONS | DEGREES | COMPARATIVES | SPOT_NOUNS | {
        "one", "ones", "thing", "things", "something", "anything", "stuff", "few", "couple",
        "same", "sheet", "page", "paper", "picture", "drawing", "room", "space", "top",
        "bottom", "middle", "center", "centre", "corner", "edge", "at", "for", "with",
        "instead", "again", "more", "less", "too", "there's", "it's", "i", "you", "we",
    }
)  # fmt: skip


def _find(w: list[str], phrase: tuple[str, ...], start: int = 0, end: int | None = None) -> int:
    end = len(w) if end is None else end
    for i in range(start, end - len(phrase) + 1):
        if tuple(w[i : i + len(phrase)]) == phrase:
            return i
    return -1


_JSON_ARG = r'"{key}"\s*:\s*"([^"]+)"'
_KW_ARG = r"\b{key}\s*=\s*['\"]?([^'\",)]+)"


def previewed(previewing: str) -> tuple[str, str] | None:
    """(tool, object or text) of the preview string, e.g. "place_drawing(tree)" ->
    ("place_drawing", "tree"). Also reads str(ToolCall) JSON and object=... forms."""
    s = previewing.strip()
    m = re.match(r"([a-z_]+)\s*\((.*)\)\s*$", s, re.S)
    tool, inner = (m.group(1), m.group(2)) if m else ("place_drawing", s)
    key = "text" if tool == "write_label" else "object"
    for pattern in (_JSON_ARG, _KW_ARG):
        found = re.search(pattern.format(key=key), inner)
        if found:
            return tool, found.group(1).strip()
    value = inner.split(",")[0].strip().strip("'\"{} ")
    return (tool, value) if value and ":" not in value and "=" not in value else None


class _Parser:
    def __init__(self, words: list[str], context: AgentContext | None, categories: list[str]):
        self.raw = words
        self.w = [norm_word(x) for x in words]
        self.ctx = context or AgentContext()
        self.cats = sorted({c.lower() for c in categories}, key=lambda c: -len(c.split()))
        self.scene = {s.lower() for s in self.ctx.scene_labels}

    # ------------------------------------------------------------------ helpers

    def content(self) -> list[str]:
        return [x for x in self.w if x and x not in FILLERS]

    def object_at(self, i: int) -> tuple[str, int] | None:
        """(category, word count) if a drawable category starts at word i."""
        for cat in self.cats:
            parts = tuple(cat.split())
            seg = self.w[i : i + len(parts)]
            if len(seg) < len(parts):
                continue
            if tuple(seg) == parts:
                return cat, len(parts)
            last = seg[-1]
            if last.endswith("s") and tuple([*seg[:-1], last[:-1]]) == parts:
                return cat, len(parts)
        syn = SYNONYMS.get(self.w[i]) if i < len(self.w) else None
        if syn and syn not in self.cats:
            syn = SYNONYMS.get(syn)  # "ship" -> "boat" -> "sailboat"
        if syn and syn in self.cats:
            return syn, 1
        return None

    def object_ref(self, i: int, end: int) -> tuple[object, int] | None:
        """Parse an object reference starting at word i; returns (ref, next index)."""
        while i < end and self.w[i] in {"the", "a", "an", "my", "your"}:
            i += 1
        if i >= end:
            return None
        x = self.w[i]
        if x in ("it", "him", "her", "them"):
            return LastObject(), i + 1
        if x in POINT_WORDS or x in PLACE_WORDS:
            nxt = i + 1
            if nxt < end and self.w[nxt] in ("one", "thing"):
                nxt += 1
            return DeicticObject(deixis=x, word_index=i), nxt
        for label in sorted(self.scene, key=lambda s: -len(s.split())):
            parts = label.split()
            if tuple(self.w[i : i + len(parts)]) == tuple(parts):
                return LabelObject(label=label), i + len(parts)
        obj = self.object_at(i)
        if obj:
            return LabelObject(label=obj[0]), i + obj[1]
        if x in FILLERS or x in NEGATIONS:
            return None
        return LabelObject(label=x), i + 1

    # ------------------------------------------------------------------ slots

    def size_in(self, a: int, b: int) -> tuple[object | None, set[int]]:
        """A size reference in words[a:b], plus the indices it used."""
        for i in range(a, b - 1):
            if self.w[i] in POINT_WORDS | {"so"} and self.w[i + 1] in SIZE_NOUNS:
                return DeicticSize(word_index=i), {i, i + 1}
        j = _find(self.w, ("same", "size", "as"), a, b)
        if j >= 0:
            ref = self.object_ref(j + 3, b)
            if ref:
                return RelativeSize(same_as=ref[0]), set(range(j, ref[1]))
        for phrase in (("the", "whole", "page"), ("whole", "page"), ("full", "page")):
            j = _find(self.w, phrase, a, b)
            if j >= 0:
                return PresetSize(preset="whole_page"), set(range(j, j + len(phrase)))
        for i in range(a, b):
            if self.w[i] in PRESETS:
                if i > a and self.w[i - 1] in POINT_WORDS | {"so"}:
                    continue
                if self.w[i] in ("little", "tiny") and (
                    i + 1 >= b or self.w[i + 1] in AMOUNT_AFTER_LITTLE
                ):
                    continue  # "a little bit", "a tiny bit to the left": an amount
                return PresetSize(preset=PRESETS[self.w[i]]), {i}
        return None, set()

    def place_in(self, a: int, b: int, used: set[int]) -> object | None:
        """A place reference in words[a:b], skipping indices already used for size."""
        for phrase in EMPTY_PHRASES:
            if _find(self.w, phrase, a, b) >= 0:
                return EmptySpace()
        for j in range(a, b):
            if self.w[j] == "where" and ROOM_WORDS & set(self.w[j + 1 : min(b, j + 5)]):
                return EmptySpace()
        for phrase, name in NAMED_PLACES:
            if _find(self.w, phrase, a, b) >= 0:
                return NamedPlace(named=name)
        for phrase, rel in RELATIONS:
            j = _find(self.w, phrase, a, b)
            if j < 0 or j in used:
                continue
            ref = self.object_ref(j + len(phrase), b)
            if ref is None:
                continue
            target, nxt = ref
            if isinstance(target, DeicticObject) and target.deixis in PLACE_WORDS:
                continue  # "over there" is a place word, not a relation
            if (
                rel == "inside"
                and isinstance(target, LabelObject)
                and target.label not in self.scene | set(self.cats)
            ):
                continue  # "in blue" etc.
            if rel == "between":
                if nxt < b and self.w[nxt] == "and":
                    ref2 = self.object_ref(nxt + 1, b)
                    if ref2:
                        return RelationPlace(relation="between", target=target, target2=ref2[0])
                continue
            return RelationPlace(relation=rel, target=target)
        for i in range(a, b):
            if i not in used and self._pointed_place(i, b):
                return DeicticPlace(deixis=self.w[i], word_index=i)
        return None

    def _pointed_place(self, i: int, b: int) -> bool:
        """Word i is a pointed place: "here", "there", "this spot"."""
        if self.w[i] in PLACE_WORDS:
            return True
        return self.w[i] in POINT_WORDS and i + 1 < b and self.w[i + 1] in SPOT_NOUNS

    def repoint_place(self, a: int, b: int, used: set[int]) -> object | None:
        """The new place for a preview: the last pointed place wins ("not there here",
        "there, no, here") and one right after "not" is rejected; else a spoken relation,
        region or empty space."""
        skip = used | {i for i in range(max(a, 1), b) if self.w[i - 1] == "not"}
        for i in range(b - 1, a - 1, -1):
            if i not in skip and self._pointed_place(i, b):
                return DeicticPlace(deixis=self.w[i], word_index=i)
        return self.place_in(a, b, skip)

    # ------------------------------------------------------------------ commands

    def simple_command(self) -> list[ToolCall] | None:
        c = self.content()
        if not c:
            return []
        if c[0] in WRITE_VERBS:
            return None
        if any(x in STOP_WORDS for x in c):
            return [ToolCall(name="stop", args=NoArgs())]
        for noun in ("sheet", "page", "paper"):
            if noun in c:
                i = c.index(noun)
                prev = set(c[max(0, i - 3) : i])
                if prev & {"new", "fresh", "clear", "another", "blank", "next", "clean"}:
                    return [ToolCall(name="new_sheet", args=NoArgs())]
        if " ".join(c) in ("start again", "start over", "clear everything"):
            return [ToolCall(name="new_sheet", args=NoArgs())]
        if set(c) & CANCEL_WORDS or any(_find(c, p) >= 0 for p in CANCEL_PHRASES):
            return [ToolCall(name="cancel", args=NoArgs())]
        if tuple(c) in CONFIRM or (c[0] in {"yes", "yeah", "ok", "okay"} and len(c) <= 3):
            if not self.has_object():
                return [ToolCall(name="confirm", args=NoArgs())]
        if tuple(self.w[:2]) in {("okay", "do"), ("ok", "do")} and len(c) <= 2:
            return [ToolCall(name="confirm", args=NoArgs())]
        if c[0] in NEGATIONS | {"not"} and len(c) <= 3 and not self.has_object():
            # "no" / "not there" reject; "no, there" / "no, this big" re-point (see requests)
            size, used = self.size_in(0, len(self.w))
            if (
                self.adjust_args() is None
                and size is None
                and self.repoint_place(0, len(self.w), used) is None
            ):
                return [ToolCall(name="cancel", args=NoArgs())]
        if c[0] in QUESTION_STARTS:
            return [ToolCall(name="look", args=Look(query=" ".join(self.raw)))]
        return None

    def has_object(self) -> bool:
        return any(self.object_at(i) for i in range(len(self.w)))

    def adjust_args(self, a: int = 0, b: int | None = None) -> Adjust | None:
        w = self.w[a:b]
        ws = set(w)
        strong = bool({"lot", "much", "way", "lots", "loads"} & ws)
        step = 3 if strong else 1
        dx = dy = 0
        scale = 1.0
        if {"bigger", "larger"} & ws:
            scale = 1.5 if strong else 1.25
        elif {"smaller", "tinier"} & ws:
            scale = 0.6 if strong else 0.8
        right_deixis = _find(w, ("right", "there")) >= 0 or _find(w, ("right", "here")) >= 0
        if "left" in ws:
            dx = -step
        elif "right" in ws and not right_deixis:
            dx = step
        if {"up", "higher", "further", "away"} & ws:
            dy = step
        elif {"down", "lower", "closer"} & ws:
            dy = -step
        if dx == dy == 0 and scale == 1.0:
            return None
        return Adjust(dx=dx, dy=dy, scale=scale)

    def write(self) -> list[ToolCall] | None:
        w = self.w
        verbs = [i for i, x in enumerate(w) if x in WRITE_VERBS]
        if not verbs:
            return None
        v = verbs[0]
        before: list[ToolCall] = []
        if v > 0:  # "draw a fish here then write nemo under it"
            before = [
                ToolCall(name="place_drawing", args=PlaceDrawing(object=c, place=p, size=s))
                for c, p, s in self.drawings(0, v)
            ]
        start = v + 1
        if w[v] == "label":
            on_objects = self.labels_on_objects(start)
            if on_objects is not None:
                return [*before, *on_objects]
        for skip in (("the", "words"), ("the", "word"), ("the", "text"), ("that", "says")):
            if tuple(w[start : start + len(skip)]) == skip:
                start += len(skip)
        end = len(w)
        place: object | None = None
        cut = self._trailing_place_start(start, end)
        if cut is not None:
            place = self.place_in(cut, end, set())
            end = cut
        text = self.text(start, end)
        if not text:
            return [ToolCall(name="ask_user", args=AskUser(question="What should I write?"))]
        return [*before, ToolCall(name="write_label", args=WriteLabel(text=text, place=place))]

    def text(self, a: int, b: int) -> str:
        """Words a..b as spoken, without surrounding punctuation."""
        return " ".join(t.strip(" ,.!?\"'") for t in self.raw[a:b]).strip()

    def label_target(self, i: int) -> tuple[object, int] | None:
        """The object a label goes on, at word i: "it", "this"/"that" (+ "one"/"thing"),
        or "the <scene label>". Returns (ObjectRef, index after it)."""
        w = self.w
        if i >= len(w):
            return None
        if w[i] in PRONOUNS:
            return LastObject(), i + 1
        if w[i] in POINT_WORDS:
            if i + 1 < len(w) and w[i + 1] == "says":
                return None  # "label that says hello"
            nxt = i + 2 if i + 1 < len(w) and w[i + 1] in ("one", "thing") else i + 1
            return DeicticObject(deixis=w[i], word_index=i), nxt
        if w[i] == "the":
            for label in sorted(self.scene, key=lambda s: -len(s.split())):
                parts = tuple(label.split())
                if tuple(w[i + 1 : i + 1 + len(parts)]) == parts:
                    return LabelObject(label=label), i + 1 + len(parts)
        return None

    def labels_on_objects(self, start: int) -> list[ToolCall] | None:
        """Labels on objects ("label that one nemotron", "label it magpie", "label this
        cosmos and this asr"): one write_label per object, below it. None if no object
        follows "label"."""
        w = self.w
        first = self.label_target(start)
        if first is None:
            return None
        out: list[ToolCall] = []
        todo: tuple[object, int] | None = first
        while todo is not None:
            target, a = todo
            b, todo = a, None
            while b < len(w):
                if w[b] == "and":
                    nxt = self.label_target(b + 1)
                    if nxt is not None and nxt[1] < len(w):  # it has text of its own
                        todo = nxt
                        break
                b += 1
            text = self.text(a, b)
            if not text:
                return [ToolCall(name="ask_user", args=AskUser(question="What should I write?"))]
            place = RelationPlace(relation="below", target=target)
            out.append(ToolCall(name="write_label", args=WriteLabel(text=text, place=place)))
        return out

    def _trailing_place_start(self, a: int, b: int) -> int | None:
        w = self.w
        if b - 1 > a and w[b - 1] in PLACE_WORDS:
            i = b - 1
            if i - 1 > a and w[i - 1] in ("over", "right", "just", "up"):
                i -= 1
            return i
        best = None
        for phrase, _ in RELATIONS + NAMED_PLACES:
            j = _find(w, phrase, a + 1, b)
            if j >= 0 and self.place_in(j, b, set()) is not None:
                best = j if best is None else min(best, j)
        return best

    def connector(self) -> list[ToolCall] | None:
        w = self.w
        kind = next((CONNECT_NOUNS[x] for x in w if x in CONNECT_NOUNS), None)
        if "connect" in w:
            kind = kind or "line"
        if kind is None:
            return None
        f = _find(w, ("from",))
        if f >= 0:
            t = _find(w, ("to",), f + 1)
            if t < 0:
                return None
            a = self.object_ref(f + 1, t)
            b = self.object_ref(t + 1, len(w))
        else:
            c = w.index("connect") if "connect" in w else _find(w, ("between",))
            if c < 0:
                return None
            joins = [i for i in range(c + 1, len(w)) if w[i] in ("to", "and")]
            if not joins:
                return None
            stop = next((i for i in range(joins[0] + 1, len(w)) if w[i] == "with"), len(w))
            a = self.object_ref(c + 1, joins[0])
            b = self.object_ref(joins[0] + 1, stop)
        if not a or not b:
            return None
        args = DrawConnector.model_validate({"kind": kind, "from": a[0], "to": b[0]})
        return [ToolCall(name="draw_connector", args=args)]

    def drawings(self, a: int, b: int) -> list[tuple[str, object | None, object | None]]:
        """(object, place, size) for each object requested in words[a:b]."""
        hits = []
        i = a
        while i < b:
            obj = self.object_at(i)
            if obj and not self._is_target(i):
                hits.append((i, obj[0], obj[1]))
                i += obj[1]
            else:
                i += 1
        out = []
        for n, (i, cat, ln) in enumerate(hits):
            seg_a = a if n == 0 else self._segment_start(hits[n - 1][0] + hits[n - 1][2], i)
            seg_b = b if n == len(hits) - 1 else self._segment_start(i + ln, hits[n + 1][0])
            size, used = self.size_in(seg_a, seg_b)
            place = self.place_in(seg_a, seg_b, used)
            out.append((cat, place, size))
        return out

    def _segment_start(self, a: int, b: int) -> int:
        for i in range(a, b):
            if self.w[i] in ("and", "then", "also", "plus"):
                return i
        # no conjunction: the next request starts at its article, if any
        for i in range(b - 1, a - 1, -1):
            if self.w[i] in ("a", "an", "another", "the"):
                return i
        return b

    def _is_target(self, i: int) -> bool:
        """True if the word at i is the object of a relation or connector, not a request."""
        for back in range(1, 4):
            j = i - back
            if j < 0:
                break
            if self.w[j] in RELATION_HEADS:
                if self.w[j] == "and":
                    return self._between_before(j)
                return True
            if self.w[j] not in ARTICLES:
                break
        return False

    def _between_before(self, j: int) -> bool:
        return any(self.w[k] == "between" for k in range(max(0, j - 4), j))

    # ------------------------------------------------------------------ main

    def parse(self) -> list[ToolCall]:
        for step in (self.simple_command, self.write, self.connector):
            calls = step()
            if calls is not None:
                return calls
        return self.requests()

    def _correction_cut(self, end: int) -> int:
        w = self.w[:end]
        cut = 0
        for marker in CORRECTION_MARKERS:
            for i in range(1, len(w)):
                if tuple(w[i : i + len(marker)]) == marker:
                    cut = max(cut, i + len(marker))
        return cut

    def _adjust_tail(self) -> int | None:
        """Start of a trailing correction ("... a bit bigger", "... a bit to the left")."""
        w = self.w
        t = len(w)
        while t > 0 and w[t - 1] in ADJUST_TAIL:
            t -= 1
        tail = set(w[t:])
        if 0 < t < len(w) and (tail & COMPARATIVES or (tail & DEGREES and tail & DIRECTIONS)):
            return t
        return None

    def requests(self) -> list[ToolCall]:
        """Drawing requests; a place spoken before a trailing correction ("a brain up here
        a bit bigger") gives place_drawing then adjust."""
        tail = self._adjust_tail()
        if tail is not None:
            calls = self._requests(tail)
            last = calls[-1] if calls else None
            if last and last.name == "place_drawing" and last.args.place is not None:  # type: ignore[attr-defined]
                adj = self.adjust_args(tail)
                if adj is not None:
                    return [*calls, ToolCall(name="adjust", args=adj)]
        return self._requests(len(self.w))

    def _requests(self, end: int) -> list[ToolCall]:
        w = self.w[:end]
        # Self-corrections: parse the last segment, fill missing slots from the one before.
        cut = self._correction_cut(end)
        latest = self.drawings(cut, end)
        earlier = self.drawings(0, cut) if cut else []
        if cut and earlier:
            if not latest:
                size, used = self.size_in(cut, end)
                latest = [(earlier[-1][0], self.place_in(cut, end, used), size)]
            cat, place, size = latest[-1]
            _, e_place, e_size = earlier[-1]
            latest[-1] = (cat, place or e_place, size or e_size)
            if len(latest) == 1 and len(earlier) > 1:
                latest = [*earlier[:-1], latest[-1]]
        for c, _, _ in latest:
            others = [x for x in self.cats if x != c and x.split()[-1] == c]
            if others:  # "bat": the animal or a baseball bat?
                q = AskUser(question=f"Which {c} do you mean?", options=[c, *others][:4])
                return [ToolCall(name="ask_user", args=q)]
        calls = [
            ToolCall(name="place_drawing", args=PlaceDrawing(object=c, place=p, size=s))
            for c, p, s in latest
        ]
        if calls:
            return calls
        unknown = self.unknown_object(end)
        if unknown is not None:
            return [ToolCall(name="ask_user", args=unknown)]
        size, used = self.size_in(0, end)
        place = self.repoint_place(cut, end, used)
        prev = previewed(self.ctx.previewing) if self.ctx.previewing else None
        if prev and (place or size):
            # "no, there", "put it here instead", "this big": the preview moves or resizes;
            # a missing place means the runtime keeps the preview's place.
            tool, what = prev
            if tool == "write_label":
                return [
                    ToolCall(name="write_label", args=WriteLabel(text=what, place=place, size=size))
                ]
            if tool == "place_drawing":
                args = PlaceDrawing(object=what, place=place, size=size)
                return [ToolCall(name="place_drawing", args=args)]
        if (place or size) and {"it", "put", "move"} & set(w):
            q = AskUser(question="What should I draw there?")
            return [ToolCall(name="ask_user", args=q)]
        adj = self.adjust_args(0, end)
        if adj is not None:
            return [ToolCall(name="adjust", args=adj)]
        if {"draw", "put", "add", "make"} & set(w):
            return [ToolCall(name="ask_user", args=AskUser(question="What should I draw?"))]
        return []

    def unknown_object(self, end: int) -> AskUser | None:
        """A question for a thing that isn't drawable: the noun after the first article or
        draw verb ("draw me a dragon", "a unicorn here"), with close categories as options."""
        w = self.w[:end]
        i = next((k for k, x in enumerate(w) if x in NOUN_STARTS), None)
        if i is None:
            return None
        article = w[i] if w[i] in ("a", "an", "some") else ""
        i += 1
        while i < len(w) and w[i] in NOUN_SKIP:
            if w[i] in ("a", "an", "some"):
                article = w[i]
            i += 1
        if i >= len(w) or w[i] in NOT_NOUNS or not w[i].isalpha():
            return None
        if self.object_at(i) or w[i] in self.scene or self._is_target(i):
            return None  # drawable, on the sheet, or the target of "next to the ..."
        noun = w[i]
        kind = GENERAL_KINDS.get(noun) or GENERAL_KINDS.get(noun.rstrip("s"))
        members = [c for c in kind or () if c in self.cats]
        if members:
            base = noun.rstrip("s") if noun not in GENERAL_KINDS else noun
            return AskUser(question=f"Which {base} should I draw?", options=members[:4])
        options: list[str] = []
        for probe in dict.fromkeys((noun, noun[:-1] if noun.endswith("s") else noun)):
            for c in difflib.get_close_matches(probe, self.cats, n=3, cutoff=0.5):
                if c not in options:
                    options.append(c)
        if article in ("a", "an") and w[i - 1] != article:
            article = "an" if noun[0] in "aeiou" else "a"  # "a big owl" -> "an owl"
        thing = f"{article} {noun}".strip()
        if options:
            return AskUser(
                question=f"I can't draw {thing} yet. Which one instead?", options=options[:3]
            )
        return AskUser(question=f"I can't draw {thing} yet. What should I draw instead?")


def parse(
    words: list[str] | str,
    context: AgentContext | None = None,
    categories: list[str] | None = None,
) -> list[ToolCall]:
    """Parse one utterance. Returns [] if nothing was understood."""
    if isinstance(words, str):
        words = words.split()
    cats = categories if categories is not None else load_categories()
    return _Parser(list(words), context, cats).parse()
