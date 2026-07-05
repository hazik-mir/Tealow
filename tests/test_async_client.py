"""Tests for TeaLow.async_client.AsyncTeaLow."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional
from unittest.mock import AsyncMock

import pytest

from TeaLow import AsyncTeaLow


class _FakeHTTPXResponse:
    """Minimal stand-in for httpx.Response used in async tests."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


@pytest.mark.asyncio
async def test_async_send_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    ai = AsyncTeaLow(model="gpt-4o-mini", api="sk-test-1234567890")

    fake_payload = {
        "model": "gpt-4o-mini",
        "choices": [
            {"message": {"content": "Hello async world"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    async def fake_request(*args: Any, **kwargs: Any) -> _FakeHTTPXResponse:
        return _FakeHTTPXResponse(fake_payload)

    monkeypatch.setattr(ai._async_session, "request", fake_request)

    response = await ai.send("Hi!")
    assert str(response) == "Hello async world"
    assert response.provider == "openai"
    assert len(ai.history) == 2

    await ai.close()


@pytest.mark.asyncio
async def test_async_stream_yields_chunks_and_records_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ai = AsyncTeaLow(model="gpt-4o-mini", api="sk-test-1234567890")

    async def fake_stream_lines(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        lines = [
            'data: {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]}',
            'data: {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}',
            "data: [DONE]",
        ]
        for line in lines:
            yield line

    monkeypatch.setattr(ai._async_session, "stream_lines", fake_stream_lines)

    collected = ""
    async for chunk in ai.stream("Hi!"):
        collected += chunk.delta

    assert collected == "Hello"
    assert len(ai.history) == 2
    assert ai.history[-1]["content"] == "Hello"

    await ai.close()


@pytest.mark.asyncio
async def test_async_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    async with AsyncTeaLow(model="claude-sonnet-4-6", api="sk-ant-1234567890") as ai:
        assert ai.provider == "anthropic"
