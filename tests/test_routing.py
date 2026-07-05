"""Tests for automatic provider detection based on model name."""

from __future__ import annotations

import pytest

from TeaLow.client import detect_provider
from TeaLow.constants import Provider
from TeaLow.exceptions import UnsupportedModelError


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-2.5-flash", Provider.GEMINI),
        ("gemini-2.5-pro", Provider.GEMINI),
        ("gpt-4o", Provider.OPENAI),
        ("gpt-4o-mini", Provider.OPENAI),
        ("o3", Provider.OPENAI),
        ("o3-mini", Provider.OPENAI),
        ("dall-e-3", Provider.OPENAI),
        ("claude-sonnet-4-6", Provider.ANTHROPIC),
        ("claude-opus-4-8", Provider.ANTHROPIC),
        ("deepseek-chat", Provider.DEEPSEEK),
        ("deepseek-reasoner", Provider.DEEPSEEK),
        ("GPT-4O", Provider.OPENAI),  # case-insensitive
    ],
)
def test_detect_provider(model: str, expected: Provider) -> None:
    assert detect_provider(model) == expected


def test_detect_provider_unsupported_model_raises() -> None:
    with pytest.raises(UnsupportedModelError):
        detect_provider("some-totally-unknown-model-xyz")
