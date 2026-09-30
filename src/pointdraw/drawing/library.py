"""The committed drawing library: load it, resolve names and synonyms, choose variants.

Layout (written by scripts/build_library.py):

    library/index.json               categories, synonyms, source and licence
    library/quickdraw/<name>.json    variants, each a list of strokes

Strokes are in library units: y up, the bbox's min corner at (0, 0), and the larger side
about 255 units long (the Quick, Draw! simplified scale). `place.fit` turns them into mm.
Works offline: nothing here touches the network.
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from deixis.geometry import Polyline

LIBRARY_ENV = "POINTDRAW_LIBRARY"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "library"

# Words people say for library categories. build_library.py writes these into index.json;
# only synonyms whose target category is in the library are kept.
SYNONYMS: dict[str, str] = {
    # hero demo
    "ears": "ear", "eyeball": "eye", "mind": "brain", "lips": "mouth", "hands": "hand",
    "palm": "hand", "mic": "microphone", "microphones": "microphone", "webcam": "camera",
    "notebook": "laptop", "computer": "laptop", "pc": "laptop",
    # scene
    "home": "house", "cottage": "house", "hut": "house", "sunshine": "sun", "shrub": "bush",
    "hedge": "bush", "lake": "pond", "puddle": "pond", "kitty": "cat", "kitten": "cat",
    "puppy": "dog", "doggy": "dog", "birdie": "bird", "hill": "mountain", "mountains": "mountain",
    "automobile": "car", "bike": "bicycle", "boat": "sailboat", "yacht": "sailboat",
    "mug": "cup", "teacup": "cup", "cap": "hat", "tiara": "crown", "bulb": "light bulb",
    "lightbulb": "light bulb", "lamp": "floor lamp", "smiley": "smiley face",
    "smile": "smiley face", "face": "smiley face", "happy face": "smiley face",
    "desk": "table", "seat": "chair", "sofa": "couch", "settee": "couch", "tv": "television",
    "telly": "television", "screen": "television", "gate": "fence", "flowers": "flower",
    "trees": "tree", "stars": "star", "clouds": "cloud", "rose": "flower", "daisy": "flower",
}  # fmt: skip

_ARTICLES = re.compile(r"^(a|an|the|some|another|one|my|this|that)\s+")


def normalise(name: str) -> str:
    """Lower case, single spaces, no leading article, hyphens and underscores as spaces."""
    s = re.sub(r"[-_]+", " ", name.strip().lower())
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    while (t := _ARTICLES.sub("", s)) != s:
        s = t
    return s


def _singular(s: str) -> list[str]:
    out = []
    if s.endswith("ies"):
        out.append(s[:-3] + "y")
    if s.endswith("es"):
        out.append(s[:-2])
    if s.endswith("s") and not s.endswith("ss"):
        out.append(s[:-1])
    return out


def _strokes(raw: list[list[list[float]]]) -> list[Polyline]:
    return [[(float(x), float(y)) for x, y in s] for s in raw]


@dataclass(frozen=True)
class Drawing:
    category: str
    key_id: str
    variant: int
    strokes: list[Polyline]  # library units, y up

    @property
    def size(self) -> tuple[float, float]:
        xs = [p[0] for s in self.strokes for p in s]
        ys = [p[1] for s in self.strokes for p in s]
        return max(xs) - min(xs), max(ys) - min(ys)


class UnknownCategory(KeyError):
    def __init__(self, name: str, suggestions: list[str]):
        super().__init__(name)
        self.name = name
        self.suggestions = suggestions

    def __str__(self) -> str:
        hint = f"; did you mean {', '.join(self.suggestions)}?" if self.suggestions else ""
        return f"no drawing for {self.name!r}{hint}"


class Library:
    def __init__(self, root: Path, index: dict):
        self.root = root
        self.index = index
        self._cats: dict[str, dict] = index["categories"]
        self._syn: dict[str, str] = {
            normalise(k): v for k, v in index.get("synonyms", {}).items() if v in self._cats
        }
        self._cache: dict[str, list[Drawing]] = {}
        self._next: dict[str, int] = {}

    @classmethod
    def load(cls, root: str | Path | None = None) -> Library:
        """Load from `root`, else $POINTDRAW_LIBRARY, else the repo's library/ folder."""
        path = Path(root or os.environ.get(LIBRARY_ENV) or DEFAULT_DIR)
        with open(path / "index.json", encoding="utf-8") as fh:
            return cls(path, json.load(fh))

    @property
    def categories(self) -> list[str]:
        return sorted(self._cats)

    def resolve(self, name: str) -> str | None:
        """Category for a spoken name ("a kitty", "Light-bulbs"), or None."""
        s = normalise(name)
        for cand in [s, *_singular(s)]:
            if cand in self._cats:
                return cand
            if cand in self._syn:
                return self._syn[cand]
        return None

    def __contains__(self, name: str) -> bool:
        return self.resolve(name) is not None

    def variants(self, name: str) -> list[Drawing]:
        cat = self.resolve(name)
        if cat is None:
            pool = self.categories + sorted(self._syn)
            raise UnknownCategory(name, difflib.get_close_matches(normalise(name), pool, n=3))
        if cat not in self._cache:
            with open(self.root / self._cats[cat]["file"], encoding="utf-8") as fh:
                data = json.load(fh)
            self._cache[cat] = [
                Drawing(cat, str(v["key_id"]), i, _strokes(v["strokes"]))
                for i, v in enumerate(data["variants"])
            ]
        return self._cache[cat]

    def get(
        self, name: str, variant: int | str | None = None, rng: random.Random | None = None
    ) -> Drawing:
        """A drawing of `name`.

        variant: an index (wraps around); "random"; or None, which cycles through the
        variants so asking for "another tree" gives a different one.
        """
        vs = self.variants(name)
        cat = vs[0].category
        if variant is None:
            i = self._next.get(cat, 0)
            self._next[cat] = i + 1
        elif variant == "random":
            i = (rng or random).randrange(len(vs))
        elif isinstance(variant, int):
            i = variant
        else:
            raise ValueError(f"bad variant {variant!r}")
        return vs[i % len(vs)]
