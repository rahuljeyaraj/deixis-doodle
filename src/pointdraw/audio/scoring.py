"""Word normalisation, alignment and error rates for judging speech recognition."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words that point: their times are what the deixis resolver needs.
DEMONSTRATIVES = frozenset({"here", "there", "this", "that", "these", "those"})

_STRIP = re.compile(r"[^a-z0-9']+")


def normalize_word(w: str) -> str:
    """Lowercase, drop punctuation (keep apostrophes inside words)."""
    return _STRIP.sub("", w.lower()).strip("'")


def normalize_words(words: list[str]) -> list[str]:
    return [n for n in (normalize_word(w) for w in words) if n]


def tokenize(text: str) -> list[str]:
    return normalize_words(text.split())


@dataclass(frozen=True)
class Alignment:
    """Levenshtein alignment of a reference against a hypothesis."""

    pairs: list[tuple[int | None, int | None]]  # (ref index, hyp index); None = ins/del
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def align(ref: list[str], hyp: list[str]) -> Alignment:
    """Minimum-edit alignment of two normalised word lists."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j - 1] + cost, d[i - 1][j] + 1, d[i][j - 1] + 1)
    pairs: list[tuple[int | None, int | None]] = []
    s = de = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            pairs.append((i - 1, None))
            i -= 1
        else:
            ins += 1
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return Alignment(pairs, s, de, ins)


def word_error_rate(refs: list[list[str]], hyps: list[list[str]]) -> float:
    """Corpus WER: total edits over total reference words."""
    errors = sum(align(r, h).errors for r, h in zip(refs, hyps, strict=True))
    total = sum(len(r) for r in refs)
    return errors / total if total else 0.0


def matched_pairs(ref: list[str], hyp: list[str]) -> dict[int, int]:
    """Reference index -> hypothesis index for words recognised exactly."""
    return {
        i: j
        for i, j in align(ref, hyp).pairs
        if i is not None and j is not None and ref[i] == hyp[j]
    }
