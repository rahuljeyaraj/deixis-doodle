"""The robot's guess in a few plain words, said (and logged) when the preview starts.

"A kite, about four centimetres?" / "A tree, next to the house?" The person answers "yes",
or corrects it ("a bit bigger"). Nothing is drawn until "yes" (`RuntimeParams.auto_confirm`
is off).
"""

from __future__ import annotations

from collections.abc import Sequence

NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty"
).split()

RELATIONS = {
    "next_to": "next to the",
    "left_of": "to the left of the",
    "right_of": "to the right of the",
    "above": "above the",
    "below": "below the",
    "on": "on the",
    "inside": "inside the",
}


def number_word(n: int) -> str:
    return NUMBERS[n] if 0 <= n < len(NUMBERS) else str(n)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def size_words(mm: float) -> str:
    """ "about four centimetres" (whole centimetres, at least one)."""
    cm = max(1, round(mm / 10))
    return f"about {number_word(cm)} centimetre{'' if cm == 1 else 's'}"


def item_guess(
    kind: str,
    label: str | None,
    text: str | None,
    size_mm: float,
    relation: str | None = None,
    target: str | None = None,
) -> str:
    """One item: "A kite, about four centimetres", "A tree, next to the house"."""
    if kind == "label":
        return f"The words '{text}'"
    if kind == "arrow":
        return "An arrow"
    name = label or kind
    head = f"{article(name).capitalize()} {name}"
    if relation in RELATIONS and target:
        return f"{head}, {RELATIONS[relation]} {target}"
    return f"{head}, {size_words(size_mm)}"


def guess_line(parts: Sequence[str]) -> str:
    """Join the items and ask: "A kite, about four centimetres?" """
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return " and ".join(parts) + "?"
