"""CosmosClient against a fake OpenAI client: request shape, parsing and configuration."""

import base64

import cv2
import numpy as np
import pytest
from pydantic import SecretStr

from pointdraw.config import Settings
from pointdraw.cosmos.client import CosmosClient, encode_image, normalise_base_url
from pointdraw.cosmos.models import profile_for

from .fakes import FakeOpenAI


def _img(w=640, h=480):
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 40, (40, 40, 40), 3)
    return img


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_find_request_shape_and_parsing():
    fake = FakeOpenAI('```json\n[{"label": "circle", "bbox_2d": [400, 400, 600, 600]}]\n```')
    c = CosmosClient(_settings(), openai_client=fake)
    r = c.find(_img(), "circle")
    call = fake.calls[0]
    assert call["model"] == "nvidia/Cosmos3-Nano"
    assert call["temperature"] == 0.7 and call["top_p"] == 0.8
    assert call["presence_penalty"] == 1.5
    assert call["extra_body"]["top_k"] == 20
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "image_url"  # image before text, as the cookbook recommends
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "the circle" in content[1]["text"]
    assert r.image_size == (640, 480)
    assert r.detections[0].box_px == pytest.approx((256, 192, 384, 288))
    assert r.reply.usage == {"prompt_tokens": 2112, "completion_tokens": 40}
    assert r.reply.latency_s >= 0


def test_label_crop_and_alternatives():
    fake = FakeOpenAI('{"label": "A Circle", "alternatives": ["ring", "letter O"]}')
    r = CosmosClient(_settings(), openai_client=fake).label_crop(_img(200, 200))
    assert r.label == "circle"
    assert r.alternatives == ["ring", "letter o"]


def test_point_to_uses_query():
    fake = FakeOpenAI('[{"point_2d": [750, 500], "label": "empty space"}]')
    r = CosmosClient(_settings(), openai_client=fake).point_to(
        _img(), "empty space right of the house"
    )
    assert "Empty space right of the house." in FakeOpenAI.prompt_text(fake.calls[0])
    assert r.detections[0].point_px == pytest.approx((480, 240))


def test_reply_without_json_gives_no_detections():
    fake = FakeOpenAI("Sorry, I can't find that.")
    r = CosmosClient(_settings(), openai_client=fake).find(_img(), "cat")
    assert r.detections == []


def test_think_flag():
    fake = FakeOpenAI("[]")
    CosmosClient(_settings(), openai_client=fake, think=False).find(_img(), "cat")
    call = fake.calls[0]
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert FakeOpenAI.prompt_text(call).endswith("without explaining your reasoning.")


def test_model_profiles():
    edge = CosmosClient(_settings(cosmos_model="nvidia/Cosmos3-Edge"), openai_client=FakeOpenAI(""))
    assert edge.profile.reasons_by_default and edge.profile.max_tokens == 4096
    assert profile_for("nvidia/Cosmos-Reason2-8B").coord_mode == "auto"
    assert profile_for("my-org/cosmos3-nano-finetune").coord_mode == "norm1000"
    assert profile_for("someone/other-vlm").coord_mode == "auto"


def test_missing_configuration_is_explained(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("COSMOS_ENDPOINT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="COSMOS_BASE_URL"):
        CosmosClient(_settings())
    with pytest.raises(ValueError, match="COSMOS_API_KEY"):
        CosmosClient(_settings(cosmos_base_url="https://example.invalid"))


def test_token_falls_back_to_endpoint_token(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COSMOS_ENDPOINT_TOKEN", "tok-from-env")
    c = CosmosClient(_settings(cosmos_base_url="https://example.invalid/"))
    assert c._client.api_key == "tok-from-env"
    assert str(c._client.base_url).rstrip("/") == "https://example.invalid/v1"
    c2 = CosmosClient(
        _settings(cosmos_base_url="https://x.invalid", cosmos_api_key=SecretStr("k")),
    )
    assert c2._client.api_key == "k"


def test_normalise_base_url():
    assert normalise_base_url("https://a.b/") == "https://a.b/v1"
    assert normalise_base_url("https://a.b/v1/") == "https://a.b/v1"
    assert normalise_base_url("") == ""


def test_encode_image_downscales_but_reports_original_size(tmp_path):
    img = _img(3000, 1500)
    url, w, h = encode_image(img, max_side=1000)
    assert (w, h) == (3000, 1500)
    data = base64.b64decode(url.split(",", 1)[1])
    sent = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert sent.shape[:2] == (500, 1000)
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), _img(100, 50))
    assert encode_image(p)[1:] == (100, 50)
    assert encode_image(p.read_bytes())[1:] == (100, 50)
    with pytest.raises(FileNotFoundError):
        encode_image(tmp_path / "missing.png")
