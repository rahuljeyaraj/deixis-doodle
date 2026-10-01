"""The committed library loads offline, resolves names and cycles variants."""

import json
import random
from pathlib import Path

import pytest

from pointdraw.drawing.library import Library, UnknownCategory, normalise
from pointdraw.drawing.quickdraw import DEFAULT_CATEGORIES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lib() -> Library:
    return Library.load(ROOT / "library")


def test_every_default_category_is_in_the_library(lib):
    assert set(DEFAULT_CATEGORIES) <= set(lib.categories)
    assert {"ear", "eye", "brain", "mouth", "hand"} <= set(lib.categories)


def test_strokes_are_normalised(lib):
    for cat in lib.categories:
        for d in lib.variants(cat):
            xs = [p[0] for s in d.strokes for p in s]
            ys = [p[1] for s in d.strokes for p in s]
            assert min(xs) == 0 and min(ys) == 0
            # the dataset scales drawings so the larger side is about 255
            assert 245 <= max(max(xs), max(ys)) <= 256
            assert all(len(s) >= 1 for s in d.strokes)


@pytest.mark.parametrize(
    ("spoken", "cat"),
    [
        ("tree", "tree"),
        ("a Tree", "tree"),
        ("the trees", "tree"),
        ("kitty", "cat"),
        ("a kitten", "cat"),
        ("Light-bulb", "light bulb"),
        ("lightbulb", "light bulb"),
        ("TV", "television"),
        ("mic", "microphone"),
        ("some flowers", "flower"),
        ("benches", "bench"),
        ("lips", "mouth"),
        ("smiley face", "smiley face"),
    ],
)
def test_resolve(lib, spoken, cat):
    assert lib.resolve(spoken) == cat
    assert spoken in lib


def test_unknown_name_suggests(lib):
    assert lib.resolve("dragon") is None
    with pytest.raises(UnknownCategory) as e:
        lib.get("tre")
    assert "tree" in e.value.suggestions
    assert "did you mean" in str(e.value)


def test_variants_cycle_and_index(lib):
    fresh = Library.load(ROOT / "library")
    vs = fresh.variants("house")  # the default style's drawings (SANA: two styles per object)
    n = len(vs)
    ids = [d.variant for d in vs]
    seen = [fresh.get("house").variant for _ in range(n + 1)]
    assert seen == ids + ids[:1]
    assert fresh.get("house", variant=n + 1).variant == ids[1]
    assert fresh.get("house", variant="random", rng=random.Random(0)).variant in ids
    with pytest.raises(ValueError):
        fresh.get("house", variant="best")


def test_index_is_consistent(lib):
    idx = json.loads((ROOT / "library" / "index.json").read_text(encoding="utf-8"))
    for syn, cat in idx["synonyms"].items():
        assert cat in idx["categories"], syn
        assert syn not in idx["categories"]
    for cat, meta in idx["categories"].items():
        data = json.loads((ROOT / "library" / meta["file"]).read_text(encoding="utf-8"))
        assert data["category"] == cat
        assert len(data["variants"]) == meta["variants"] >= 1
        if meta.get("source") == "sana":  # generated line art (1 Oct 2026)
            assert data["license"].startswith("Apache-2.0")
            styles = [v.get("style") for v in data["variants"]]
            assert meta["styles"] == {st: styles.count(st) for st in set(styles)}
            if "fallback" in meta:
                assert (ROOT / "library" / meta["fallback"]).exists()
        else:
            assert data["license"] == "CC BY 4.0"
    if any(m.get("source") == "sana" for m in idx["categories"].values()):
        assert idx["default_style"] in ("icon", "colouring")


def test_attribution_present():
    text = (ROOT / "library" / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text and "Quick, Draw!" in text and "Hershey" in text
    assert "SANA" in text and "Apache License 2.0" in text and "Gemma" in text


def test_load_from_env_and_custom_dir(tmp_path, monkeypatch):
    (tmp_path / "quickdraw").mkdir()
    (tmp_path / "quickdraw" / "blob.json").write_text(
        json.dumps(
            {"category": "blob", "variants": [{"key_id": "1", "strokes": [[[0, 0], [255, 10]]]}]}
        )
    )
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "categories": {"blob": {"file": "quickdraw/blob.json", "variants": 1}},
                "synonyms": {"splodge": "blob", "ghost": "missing"},
            }
        )
    )
    monkeypatch.setenv("POINTDRAW_LIBRARY", str(tmp_path))
    lib = Library.load()
    assert lib.categories == ["blob"]
    assert lib.get("splodge").strokes == [[(0.0, 0.0), (255.0, 10.0)]]
    assert lib.resolve("ghost") is None  # synonyms to missing categories are ignored


def test_normalise():
    assert normalise("  The  Floor_Lamp! ") == "floor lamp"
    assert normalise("a an tree") == "tree"
