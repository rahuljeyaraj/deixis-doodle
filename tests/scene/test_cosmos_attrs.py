"""Cosmos's other labels are kept as the mark's attributes (claim 26c), so "the house" finds a
sketch Cosmos called "building" through the core's loose match, with no core change."""

from deixis import scene as core
from deixis.geometry import Box
from pointdraw.scene.ink import UserMark
from pointdraw.scene.model import SceneModel, split_label, thing


def mark(i, x, y):
    return UserMark(i, Box(x, y, 40, 30), 300.0, (x + 20, y + 15), (0, 0, 1, 1))


def test_split_label():
    assert split_label(None) == (None, [])
    assert split_label("house") == ("house", [])
    assert split_label(("building", ["House", "home", "building", "home", " "])) == (
        "building",
        ["house", "home"],
    )


def test_other_labels_are_attributes_the_core_finds():
    scene = SceneModel(Box(0, 0, 220, 196))
    answers = {1: ("building", ["house", "home", "hut"]), 2: "fish"}
    added = scene.set_user_marks([mark(1, 20, 20), mark(2, 120, 100)], lambda m: answers[m.id])
    house = added[0]
    assert house.label == "building" and house.attrs == {"also": "house home hut"}
    assert thing(house).attrs == {"also": "house home hut"}
    found = core.find(scene.things(), "house")
    assert [t.id for t in found] == [house.id]
    assert [t.id for t in core.find(scene.things(), "the fish")] == [added[1].id]
    # a mark seen again keeps its id, label and attributes (Cosmos isn't asked again)
    again = scene.set_user_marks([mark(1, 21, 20)], lambda m: ("x", ["y"]))
    assert again == [] and scene.marks[0].attrs == {"also": "house home hut"}
