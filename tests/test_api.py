import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from nano_llm.api.app import create_app
from nano_llm.config import Config
from nano_llm.models.slm import MockSLM


def build_client():
    cfg = Config()
    cfg.api.api_key = ""
    slm = MockSLM(cfg)
    return TestClient(create_app(slm, cfg))


def test_health_and_models():
    client = build_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["backend"] == "mock"

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["object"] == "model"


def test_chat_completion():
    client = build_client()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Xin chào"}], "max_tokens": 8},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"]
    assert payload["usage"]["completion_tokens"] == 8
    assert payload["timings"]["decode_tok_s"] > 0


def test_chat_completion_stream():
    client = build_client()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Chào"}], "max_tokens": 4, "stream": True},
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert "chat.completion.chunk" in response.text


def test_completion_and_dashboard():
    client = build_client()
    response = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 4})
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"]

    dashboard = client.get("/dashboard")
    assert dashboard.status_code == 200
    assert "NanoPi-R6C-LLM" in dashboard.text


def test_api_key_enforced():
    cfg = Config()
    cfg.api.api_key = "secret"
    client = TestClient(create_app(MockSLM(cfg), cfg))

    assert client.get("/v1/models").status_code == 401
    allowed = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert allowed.status_code == 200
