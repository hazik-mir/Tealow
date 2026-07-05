"""Tests for provider adapter internals: message conversion, image gen."""

from __future__ import annotations

import responses

from TeaLow import TeaLow
from TeaLow.exceptions import ImageGenerationError
from TeaLow.models import Message
from TeaLow.providers.anthropic import AnthropicProvider
from TeaLow.providers.deepseek import DeepSeekProvider
from TeaLow.providers.gemini import GeminiProvider
from TeaLow.providers.openai import OpenAIProvider
from TeaLow.session import HTTPSession


def test_openai_convert_message_with_image_url() -> None:
    message = Message(role="user", content="what is this?", images=[{"url": "http://x/y.png"}])
    converted = OpenAIProvider._convert_message(message)
    assert converted["role"] == "user"
    assert converted["content"][0] == {"type": "text", "text": "what is this?"}
    assert converted["content"][1]["image_url"]["url"] == "http://x/y.png"


def test_openai_convert_message_without_image_is_plain_string() -> None:
    message = Message(role="user", content="hello")
    converted = OpenAIProvider._convert_message(message)
    assert converted == {"role": "user", "content": "hello"}


def test_anthropic_convert_message_with_base64_image() -> None:
    message = Message(
        role="user",
        content="describe",
        images=[{"base64": "abcd1234", "mime_type": "image/jpeg"}],
    )
    converted = AnthropicProvider._convert_message(message)
    assert converted["content"][0]["type"] == "image"
    assert converted["content"][0]["source"]["media_type"] == "image/jpeg"
    assert converted["content"][-1] == {"type": "text", "text": "describe"}


def test_gemini_convert_role_maps_assistant_to_model() -> None:
    provider = GeminiProvider(
        api_key="AIza-test",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        timeout=30.0,
        max_retries=3,
        session=HTTPSession(),
    )
    assert provider._convert_role("assistant") == "model"
    assert provider._convert_role("user") == "user"


@responses.activate
def test_openai_generate_image_returns_urls() -> None:
    responses.add(
        responses.POST,
        "https://api.openai.com/v1/images/generations",
        json={"data": [{"url": "https://example.com/generated.png"}]},
        status=200,
    )
    ai = TeaLow(model="dall-e-3", api="sk-test-1234567890")
    urls = ai.generate_image("a teapot")
    assert urls == ["https://example.com/generated.png"]


def test_deepseek_does_not_support_image_generation() -> None:
    ai = TeaLow(model="deepseek-chat", api="sk-test-1234567890")
    try:
        ai.generate_image("a teapot")
        assert False, "expected ImageGenerationError"
    except ImageGenerationError:
        pass


def test_deepseek_provider_name() -> None:
    provider = DeepSeekProvider(
        api_key="sk-test",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        timeout=30.0,
        max_retries=3,
        session=HTTPSession(),
    )
    assert provider.name == "deepseek"
    assert provider.supports_image_generation() is False
