"""Shared pytest fixtures for the TeaLow test suite."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure provider environment variables never leak between tests."""
    for var in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_api_key() -> str:
    """A syntactically valid (but fake) API key for use in tests."""
    return "sk-test-1234567890abcdef"
