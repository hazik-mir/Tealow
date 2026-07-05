"""Tests for TeaLow.streaming and provider SSE parsing."""

from __future__ import annotations

from typing import Iterator

import pytest

from TeaLow.exceptions import StreamingError
from TeaLow.models import StreamChunk
from TeaLow.providers.anthropic import AnthropicProvider
from TeaLow.providers.gemini import GeminiProvider
from TeaLow.providers.openai import OpenAIProvider
from TeaLow.session import HTTPSession
from TeaLow.streaming import StreamAccumulator


def _make_openai_provider() -> OpenAIProvider:
    return OpenAIProvider(
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        timeout=30.0,
        max_retries=3,
        session=HTTPSession(),
    )


def _make_anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(
        api_key="sk-ant-test",
        model="claude-sonnet-4-6",
        base_url="https://api.anthropic.com/v1",
        timeout=30.0,
        max_retries=3,
        session=HTTPSession(),
    )


def _make_gemini_provider() -> GeminiProvider:
    return GeminiProvider(
        api_key="AIza-test",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        timeout=30.0,
        max_retries=3,
        session=HTTPSession(),
    )


def test_openai_sse_line_parsing_yields_delta() -> None:
    provider = _make_openai_provider()
    line = '{"model": "gpt-4o-mini", "choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}'
    chunk = provider._parse_sse_line(f"data: {line}")
    assert isinstance(chunk, StreamChunk)
    assert chunk.delta == "Hi"
    assert chunk.is_final is False


def test_openai_sse_line_done_marker_is_final() -> None:
    provider = _make_openai_provider()
    chunk = provider._parse_sse_line("data: [DONE]")
    assert chunk is not None
    assert chunk.is_final is True
    assert chunk.delta == ""


def test_openai_sse_line_ignores_non_data_lines() -> None:
    provider = _make_openai_provider()
    assert provider._parse_sse_line(": keep-alive") is None


def test_anthropic_sse_content_block_delta() -> None:
    provider = _make_anthropic_provider()
    line = '{"type": "content_block_delta", "delta": {"text": "Ahoy"}}'
    chunk = provider._parse_sse_line(f"data: {line}")
    assert chunk is not None
    assert chunk.delta == "Ahoy"


def test_anthropic_sse_message_stop_is_final() -> None:
    provider = _make_anthropic_provider()
    chunk = provider._parse_sse_line('data: {"type": "message_stop"}')
    assert chunk is not None
    assert chunk.is_final is True


def test_gemini_sse_line_parsing() -> None:
    provider = _make_gemini_provider()
    line = (
        '{"candidates": [{"content": {"parts": [{"text": "Bonjour"}]}, '
        '"finishReason": null}]}'
    )
    chunk = provider._parse_sse_line(f"data: {line}")
    assert chunk is not None
    assert chunk.delta == "Bonjour"


def test_stream_accumulator_collects_text() -> None:
    def fake_stream() -> Iterator[StreamChunk]:
        yield StreamChunk(delta="Hello ", model="gpt-4o-mini", provider="openai")
        yield StreamChunk(delta="world", model="gpt-4o-mini", provider="openai")
        yield StreamChunk(
            delta="", model="gpt-4o-mini", provider="openai", finish_reason="stop", is_final=True
        )

    accumulator = StreamAccumulator(model="gpt-4o-mini", provider="openai")
    collected = "".join(chunk.delta for chunk in accumulator.consume(fake_stream()))
    assert collected == "Hello world"

    final = accumulator.finalize()
    assert final.text == "Hello world"
    assert final.finish_reason == "stop"


def test_stream_accumulator_wraps_errors() -> None:
    def failing_stream() -> Iterator[StreamChunk]:
        yield StreamChunk(delta="partial", model="gpt-4o-mini", provider="openai")
        raise ValueError("network dropped")

    accumulator = StreamAccumulator(model="gpt-4o-mini", provider="openai")
    with pytest.raises(StreamingError):
        list(accumulator.consume(failing_stream()))
