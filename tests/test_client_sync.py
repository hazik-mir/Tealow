"""Tests for TeaLow.client.TeaLow using mocked HTTP responses."""

from __future__ import annotations

import json

import responses

from TeaLow import RateLimitError, RetryExhaustedError, TeaLow
from TeaLow.exceptions import InvalidAPIKeyError, JSONModeError


@responses.activate
def test_openai_send_returns_text() -> None:
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
        status=200,
    )

    ai = TeaLow(model="gpt-4o-mini", api="sk-test-1234567890")
    response = ai.send("Hi!")

    assert str(response) == "Hello there!"
    assert response.provider == "openai"
    assert response.usage.total_tokens == 8
    assert len(ai.history) == 2  # user + assistant


@responses.activate
def test_gemini_send_returns_text() -> None:
    responses.add(
        responses.POST,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": "Bonjour!"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 2,
                "totalTokenCount": 6,
            },
        },
        status=200,
    )

    ai = TeaLow(model="gemini-2.5-flash", api="AIza1234567890abcdef")
    response = ai.send("Say hello in French")

    assert str(response) == "Bonjour!"
    assert response.provider == "gemini"
    assert response.usage.total_tokens == 6


@responses.activate
def test_anthropic_send_returns_text() -> None:
    responses.add(
        responses.POST,
        "https://api.anthropic.com/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Ahoy!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        status=200,
    )

    ai = TeaLow(model="claude-sonnet-4-6", api="sk-ant-1234567890")
    response = ai.send("Talk like a pirate")

    assert str(response) == "Ahoy!"
    assert response.provider == "anthropic"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2


@responses.activate
def test_deepseek_send_returns_text() -> None:
    responses.add(
        responses.POST,
        "https://api.deepseek.com/v1/chat/completions",
        json={
            "model": "deepseek-chat",
            "choices": [
                {"message": {"content": "42"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
        status=200,
    )

    ai = TeaLow(model="deepseek-chat", api="sk-deepseek-1234567890")
    response = ai.send("What is the answer to everything?")

    assert str(response) == "42"
    assert response.provider == "deepseek"


@responses.activate
def test_invalid_api_key_raises() -> None:
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={"error": {"message": "Invalid API key"}},
        status=401,
    )

    ai = TeaLow(model="gpt-4o-mini", api="sk-test-1234567890", max_retries=0)
    try:
        ai.send("Hi!")
        assert False, "expected InvalidAPIKeyError"
    except InvalidAPIKeyError as exc:
        assert exc.status_code == 401


@responses.activate
def test_rate_limit_exhausts_retries() -> None:
    for _ in range(3):
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"error": {"message": "rate limited"}},
            status=429,
        )

    ai = TeaLow(model="gpt-4o-mini", api="sk-test-1234567890", max_retries=2)
    try:
        ai.send("Hi!")
        assert False, "expected RetryExhaustedError"
    except RetryExhaustedError as exc:
        assert isinstance(exc.last_exception, RateLimitError)


@responses.activate
def test_json_mode_validates_valid_json() -> None:
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {"content": json.dumps({"name": "Ada", "age": 30})},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        },
        status=200,
    )

    ai = TeaLow(model="gpt-4o-mini", api="sk-test-1234567890")
    response = ai.send("Give me JSON", json_mode=True)
    parsed = json.loads(response.text)
    assert parsed["name"] == "Ada"


@responses.activate
def test_json_mode_raises_on_invalid_json() -> None:
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "choices": [
                {"message": {"content": "not json"}, "finish_reason": "stop"}
            ],
            "usage": {},
        },
        status=200,
    )

    ai = TeaLow(model="gpt-4o-mini", api="sk-test-1234567890")
    try:
        ai.send("Give me JSON", json_mode=True)
        assert False, "expected JSONModeError"
    except JSONModeError:
        pass


def test_conversation_reset() -> None:
    ai = TeaLow(model="gpt-4o-mini", api="sk-test-1234567890")
    ai.conversation.add_user_message("hi")
    ai.conversation.add_assistant_message("hello")
    assert len(ai.history) == 2
    ai.reset_history()
    assert len(ai.history) == 0


def test_context_manager_closes_session() -> None:
    with TeaLow(model="gpt-4o-mini", api="sk-test-1234567890") as ai:
        assert ai.provider == "openai"
    # closing twice should not raise
    ai.close()
