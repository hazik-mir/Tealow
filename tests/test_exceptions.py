"""Tests for TeaLow's exception hierarchy."""

from __future__ import annotations

from TeaLow.exceptions import (
    APIResponseError,
    InvalidAPIKeyError,
    RateLimitError,
    RetryExhaustedError,
    TeaLowError,
    TimeoutError as TeaLowTimeoutError,
    UnsupportedModelError,
)


def test_all_exceptions_inherit_from_tealow_error() -> None:
    for cls in (
        InvalidAPIKeyError,
        RateLimitError,
        APIResponseError,
        TeaLowTimeoutError,
        UnsupportedModelError,
    ):
        assert issubclass(cls, TeaLowError)


def test_tealow_error_str_includes_context() -> None:
    err = TeaLowError("boom", provider="openai", status_code=500)
    text = str(err)
    assert "boom" in text
    assert "openai" in text
    assert "500" in text


def test_tealow_error_to_dict() -> None:
    err = TeaLowError("boom", provider="openai", status_code=500, response_body={"a": 1})
    data = err.to_dict()
    assert data["error_type"] == "TeaLowError"
    assert data["message"] == "boom"
    assert data["provider"] == "openai"
    assert data["status_code"] == 500
    assert data["response_body"] == {"a": 1}


def test_rate_limit_error_carries_retry_after() -> None:
    err = RateLimitError("too many requests", provider="openai", retry_after=12.5)
    assert err.retry_after == 12.5


def test_retry_exhausted_error_carries_last_exception() -> None:
    original = ValueError("inner")
    err = RetryExhaustedError("gave up", provider="openai", last_exception=original, attempts=3)
    assert err.last_exception is original
    assert err.attempts == 3
