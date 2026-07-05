"""Tests for TeaLow.utils."""

from __future__ import annotations

import pytest

from TeaLow.constants import Provider
from TeaLow.exceptions import MissingAPIKeyError, RetryExhaustedError
from TeaLow.utils import (
    compute_backoff_delay,
    redact,
    resolve_api_key,
    retry_sync,
    validate_api_key_format,
)


def test_resolve_api_key_prefers_explicit_key() -> None:
    key = resolve_api_key(Provider.OPENAI, "explicit-key-123")
    assert key == "explicit-key-123"


def test_resolve_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")
    key = resolve_api_key(Provider.OPENAI, None)
    assert key == "env-key-123"


def test_resolve_api_key_raises_when_missing() -> None:
    with pytest.raises(MissingAPIKeyError):
        resolve_api_key(Provider.OPENAI, None)


def test_validate_api_key_format_rejects_empty() -> None:
    with pytest.raises(MissingAPIKeyError):
        validate_api_key_format(Provider.OPENAI, "")


def test_validate_api_key_format_rejects_too_short() -> None:
    with pytest.raises(MissingAPIKeyError):
        validate_api_key_format(Provider.OPENAI, "short")


def test_validate_api_key_format_accepts_valid_key() -> None:
    validate_api_key_format(Provider.OPENAI, "sk-1234567890abcdef")  # should not raise


def test_compute_backoff_delay_grows_exponentially_without_jitter() -> None:
    d0 = compute_backoff_delay(0, base=1.0, maximum=100.0, jitter=False)
    d1 = compute_backoff_delay(1, base=1.0, maximum=100.0, jitter=False)
    d2 = compute_backoff_delay(2, base=1.0, maximum=100.0, jitter=False)
    assert d0 == 1.0
    assert d1 == 2.0
    assert d2 == 4.0


def test_compute_backoff_delay_respects_maximum() -> None:
    delay = compute_backoff_delay(10, base=1.0, maximum=5.0, jitter=False)
    assert delay == 5.0


def test_redact_short_string() -> None:
    assert redact("abc") == "***"


def test_redact_long_string_keeps_prefix() -> None:
    result = redact("sk-1234567890", keep=4)
    assert result.startswith("sk-1")
    assert result.endswith("*" * (len("sk-1234567890") - 4))


def test_retry_sync_succeeds_first_try() -> None:
    calls = []

    def func() -> str:
        calls.append(1)
        return "ok"

    result = retry_sync(func, max_retries=3)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_sync_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def func() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("transient")
        return "ok"

    result = retry_sync(func, max_retries=5)
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_sync_exhausts_and_raises() -> None:
    def func() -> str:
        raise ValueError("always fails")

    with pytest.raises(RetryExhaustedError) as exc_info:
        retry_sync(func, max_retries=2)
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_exception, ValueError)


def test_retry_sync_respects_should_retry_predicate() -> None:
    def func() -> str:
        raise ValueError("do not retry me")

    with pytest.raises(ValueError):
        retry_sync(func, max_retries=3, should_retry=lambda exc: False)
