"""Cosmos scoring of library drawings (plan T2.5), against a fake OpenAI client."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from pointdraw.config import Settings
from pointdraw.cosmos.client import CosmosClient
from pointdraw.cosmos.drawings import judge, rank_of, score_images

from .fakes import FakeOpenAI

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "score_drawings", ROOT / "scripts" / "score_drawings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png(path: Path, radius: int) -> Path:
    img = np.full((256, 256, 3), 255, np.uint8)
    cv2.circle(img, (128, 128), radius, (0, 0, 0), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def _client(answers: dict[int, str]):
    """Answers by the drawn circle's radius, read back from the image the client sends."""

    def reply(kwargs):
        import base64

        url = kwargs["messages"][0]["content"][0]["image_url"]["url"]
        data = base64.b64decode(url.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(img < 128)
        r = round((xs.max() - xs.min()) / 2 / img.shape[1] * 256)
        best = min(answers, key=lambda k: abs(k - r))
        return answers[best]

    fake = FakeOpenAI(reply)
    return CosmosClient(Settings(_env_file=None), openai_client=fake, think=False), fake


def test_judge_label_alternative_none():
    assert judge("a human ear", [], "ear") == "label"
    assert judge("Ears", [], "ear") == "label"  # plural folded
    assert judge("glove", ["palm", "mitten"], "hand", ["hands", "palm"]) == "alternative"
    assert judge("leaf", ["feather"], "tree", ["trees"]) == "none"
    assert judge("eyeball", [], "eye", ["eyeball"]) == "label"
    assert judge("year", [], "ear") == "none"  # whole words only


def test_score_images_ranks_each_object(tmp_path):
    files = [
        (_png(tmp_path / "ear" / "ear_s1000.png", 40), "ear"),
        (_png(tmp_path / "ear" / "ear_s1001.png", 70), "ear"),
        (_png(tmp_path / "ear" / "ear_s1002.png", 100), "ear"),
        (_png(tmp_path / "hand" / "hand_s1000.png", 100), "hand"),
    ]
    client, fake = _client(
        {
            40: '{"label": "circle", "alternatives": ["ring"]}',
            70: '{"label": "shell", "alternatives": ["ear", "snail"]}',
            100: '```json\n{"label": "a human ear", "alternatives": []}\n```',
        }
    )
    rep = score_images(client, files, {"ear": ["ears"], "hand": ["palm"]})
    assert len(fake.calls) == 4
    # the label_crop prompt, with the no-thinking suffix
    assert "hand-drawn mark" in FakeOpenAI.prompt_text(fake.calls[0])
    rows = rep["images"]
    assert [r["match"] for r in rows] == ["none", "alternative", "label", "none"]
    assert [r["score"] for r in rows] == [0.0, 0.5, 1.0, 0.0]
    ear = rep["objects"]["ear"]
    assert (ear["n"], ear["label_hits"], ear["any_hits"]) == (3, 1, 2)
    assert [Path(f).name for f in ear["ranked"]] == [
        "ear_s1002.png",
        "ear_s1001.png",
        "ear_s1000.png",
    ]
    assert rank_of(rep, "ear_s1001.png")[0] == 2
    assert rep["model"] == "nvidia/Cosmos3-Nano"
    assert rep["median_latency_s"] is not None


def test_a_failed_call_is_recorded_not_fatal(tmp_path):
    files = [(_png(tmp_path / "a.png", 50), "tree"), (_png(tmp_path / "b.png", 90), "tree")]
    n = {"i": 0}

    def reply(kwargs):
        n["i"] += 1
        if n["i"] == 1:
            raise TimeoutError("endpoint went away")
        return '{"label": "tree", "alternatives": []}'

    client = CosmosClient(Settings(_env_file=None), openai_client=FakeOpenAI(reply))
    rep = score_images(client, files)
    assert [r["match"] for r in rep["images"]] == ["error", "label"]
    assert "endpoint went away" in rep["images"][0]["error"]
    assert rep["objects"]["tree"]["errors"] == 1
    assert [Path(f).name for f in rep["objects"]["tree"]["ranked"]] == ["b.png", "a.png"]


def test_script_scores_a_job_folder(tmp_path, capsys):
    job = tmp_path / "job"
    entries = []
    for obj, r in (("ear", 40), ("ear", 100), ("hand", 70)):
        f = f"{obj}/{obj}_s{1000 + len(entries)}.png"
        _png(job / f, r)
        entries.append({"file": f, "object": obj, "seed": 1000 + len(entries)})
    (job / "manifest.json").write_text(json.dumps({"model": "fake", "images": entries}))
    client, _ = _client(
        {
            40: '{"label": "circle", "alternatives": []}',
            70: '{"label": "palm", "alternatives": []}',  # a library synonym of hand
            100: '{"label": "ear", "alternatives": []}',
        }
    )
    mod = _load_script()
    assert mod.main(["--job", str(job)], client=client) == 0
    rep = json.loads((job / "cosmos_scores.json").read_text())
    assert rep["objects"]["ear"]["label_hits"] == 1
    assert rep["objects"]["hand"]["label_hits"] == 1  # 'palm' via library/index.json
    out = capsys.readouterr().out
    assert "ear: top label 1/2" in out and "wrote" in out


def test_script_carries_on_after_a_cut_and_counts_kept(tmp_path, capsys):
    """A run cut short is carried on (no picture asked twice); kept drawings are counted."""
    job = tmp_path / "job"
    entries = []
    for i, r in enumerate((40, 70, 100)):
        f = f"ear/ear_icon_s{1000 + i}.png"
        _png(job / f, r)
        entries.append({"file": f, "object": "ear", "seed": 1000 + i})
    (job / "manifest.json").write_text(json.dumps({"model": "fake", "images": entries}))
    done = {"file": str(job / entries[0]["file"]), "object": "ear", "label": "ear",
            "alternatives": [], "match": "label", "score": 1.0, "latency_s": 1.0}  # fmt: skip
    (job / "cosmos_scores.json.partial.jsonl").write_text(json.dumps(done) + "\n")
    client, fake = _client({70: '{"label": "ear", "alternatives": []}',
                            100: '{"label": "cup", "alternatives": ["ear"]}'})  # fmt: skip
    mod = _load_script()
    mod.kept_keys = lambda: {"ear_icon_s1000", "ear_icon_s1002"}
    assert mod.main(["--job", str(job), "--workers", "2"], client=client) == 0
    assert len(fake.calls) == 2  # the first picture was not asked again
    rep = json.loads((job / "cosmos_scores.json").read_text())
    assert rep["kept"] == {"n": 2, "label_hits": 1, "any_hits": 2, "errors": 0}
    assert "M19 kept drawings: Cosmos named 1 of 2" in capsys.readouterr().out
