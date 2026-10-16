"""Request compatibility across legacy and reasoning Chat Completions models."""

import pytest

from freecad_ai.llm.client import LLMClient


def _client(model, provider="openai", **kwargs):
    return LLMClient(provider_name=provider, base_url="https://example.invalid",
                     api_key="", model=model, max_tokens=2048, **kwargs)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {
    "name": "inspect_document", "parameters": {"type": "object"}}}]])
@pytest.mark.parametrize("model", [
    "o1", "o1-preview", "o1-mini", "o3", "o3-mini-2025-01-31",
    "o4-mini", "gpt-5", "gpt-5.4-mini", "gpt-6.1-sol",
    "ft:gpt-5:example:custom:id",
])
def test_reasoning_models_use_completion_limit(model, stream, tools):
    body = _client(model)._openai_body([], "Inspect the document", stream, tools)
    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert body["stream"] is stream
    if tools:
        assert body["tools"] == tools


@pytest.mark.parametrize("model", [
    "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.1-mini", "custom-model",
])
def test_legacy_models_keep_existing_parameters(model):
    body = _client(model, temperature=0.7)._openai_body([], "", False)
    assert body["max_tokens"] == 2048
    assert "max_completion_tokens" not in body
    assert body["temperature"] == 0.7


@pytest.mark.parametrize("provider", ["ollama", "openrouter", "custom"])
@pytest.mark.parametrize("model", ["gpt-5", "gpt-6.1-sol", "o3-mini"])
def test_other_providers_are_not_rewritten(provider, model):
    body = _client(model, provider)._openai_body([], "", False)
    assert body["max_tokens"] == 2048
    assert "max_completion_tokens" not in body
    assert "temperature" in body


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-5"), ("openai", "gpt-6.1-sol"),
    ("openai", "gpt-4o"), ("custom", "deployment-alias"),
])
def test_explicit_completion_limit_never_sends_both_parameters(provider, model):
    params = {"max_completion_tokens": 8192, "top_p": 0.9}
    body = _client(model, provider, model_params=params)._openai_body([], "", False)
    assert body["max_completion_tokens"] == 8192
    assert "max_tokens" not in body
    assert body["top_p"] == 0.9
    assert params == {"max_completion_tokens": 8192, "top_p": 0.9}


@pytest.mark.parametrize("model,limit_key", [
    ("gpt-4o", "max_tokens"), ("o3-mini", "max_completion_tokens"),
    ("gpt-6.1-sol", "max_completion_tokens"),
])
@pytest.mark.parametrize("mode", ["connection", "tools", "stream"])
def test_public_request_paths_satisfy_token_contract(monkeypatch, model, limit_key, mode):
    """Evaluate outbound requests against a strict API double, without billing."""
    client = _client(model)
    requests = []

    def check_request(url, headers, body):
        assert url.endswith("/chat/completions")
        assert body[limit_key] == 2048
        assert len({"max_tokens", "max_completion_tokens"} & body.keys()) == 1
        requests.append(body)

    def post(url, headers, body):
        check_request(url, headers, body)
        return {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    def stream(url, headers, body):
        check_request(url, headers, body)
        yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(client, "_http_post", post)
    monkeypatch.setattr(client, "_http_stream", stream)
    if mode == "connection":
        assert client.test_connection() == "hello"
    elif mode == "tools":
        assert client.send_with_tools([], tools=[]).text == "hello"
    else:
        events = list(client.stream_with_tools([], tools=[]))
        assert any(event.text == "hello" for event in events)
    assert len(requests) == 1
