"""Utility helpers shared across the TeaLow SDK.

Includes logging configuration, environment-variable based API key
resolution, retry/backoff helpers (both synchronous and asynchronous),
and small validation utilities.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Awaitable, Callable, Optional, Tuple, Type, TypeVar

from .constants import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_BACKOFF_MAX,
    DEFAULT_MAX_RETRIES,
    ENV_VAR_NAMES,
    Provider,
)
from .exceptions import MissingAPIKeyError, RetryExhaustedError

T = TypeVar("T")

_LOGGER_NAME = "TeaLow"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger namespaced under the ``TeaLow`` root logger.

    A :class:`logging.NullHandler` is attached to the root ``TeaLow``
    logger by default so that the SDK never emits log records unless
    the host application explicitly configures logging, per Python
    logging best practices for libraries.
    """
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        root.addHandler(logging.NullHandler())
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return root


def configure_logging(level: int = logging.INFO, *, propagate: bool = True) -> None:
    """Convenience function to enable human-readable console logging.

    Args:
        level: Logging level, e.g. ``logging.DEBUG``.
        propagate: Whether log records should propagate to ancestor
            loggers (kept ``True`` by default so root-configured
            handlers still receive records).
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def resolve_api_key(provider: Provider, explicit_key: Optional[str]) -> str:
    """Resolve the API key to use for a given provider.

    Precedence: explicitly supplied key, then the provider-specific
    environment variable (e.g. ``OPENAI_API_KEY``).

    Raises:
        MissingAPIKeyError: If no key is found through either channel.
    """
    if explicit_key:
        key = explicit_key.strip()
        if key:
            return key
    env_var = ENV_VAR_NAMES[provider]
    env_value = os.environ.get(env_var)
    if env_value and env_value.strip():
        return env_value.strip()
    raise MissingAPIKeyError(
        f"No API key provided for provider '{provider.value}'. Pass api= "
        f"explicitly or set the {env_var} environment variable.",
        provider=provider.value,
    )


def validate_api_key_format(provider: Provider, api_key: str) -> None:
    """Perform a lightweight sanity check on an API key's shape.

    This does not guarantee the key is valid (only the provider can
    confirm that), but it catches obvious mistakes such as empty
    strings, whitespace-only values, or accidentally passing a model
    name instead of a key.

    Raises:
        MissingAPIKeyError: If the key is empty or clearly malformed.
    """
    if not api_key or not api_key.strip():
        raise MissingAPIKeyError(
            f"API key for provider '{provider.value}' is empty.",
            provider=provider.value,
        )
    if len(api_key.strip()) < 8:
        raise MissingAPIKeyError(
            f"API key for provider '{provider.value}' looks too short to be valid.",
            provider=provider.value,
        )


def compute_backoff_delay(
    attempt: int,
    *,
    base: float = DEFAULT_BACKOFF_BASE,
    maximum: float = DEFAULT_BACKOFF_MAX,
    jitter: bool = True,
) -> float:
    """Compute an exponential backoff delay for a given retry attempt.

    Args:
        attempt: Zero-indexed retry attempt number.
        base: Base delay in seconds.
        maximum: Upper bound on the computed delay.
        jitter: Whether to add random jitter (up to 50% of the delay)
            to avoid thundering-herd retries.
    """
    delay = min(base * (2**attempt), maximum)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    return delay


def retry_sync(
    func: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    should_retry: Optional[Callable[[BaseException], bool]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    provider: Optional[str] = None,
) -> T:
    """Execute ``func`` with exponential backoff retry semantics.

    Args:
        func: A zero-argument callable to invoke.
        max_retries: Maximum number of retry attempts after the initial call.
        retry_on: Tuple of exception types eligible for retry.
        should_retry: Optional predicate to further filter which
            exceptions should trigger a retry (e.g. based on status code).
        on_retry: Optional callback invoked with ``(attempt, exception, delay)``
            before sleeping.
        provider: Provider name, used only for error messages.

    Returns:
        The return value of ``func`` on success.

    Raises:
        RetryExhaustedError: If all attempts fail.
    """
    last_exception: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_on as exc:  # type: ignore[misc]
            last_exception = exc
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= max_retries:
                break
            delay = compute_backoff_delay(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            time.sleep(delay)
    raise RetryExhaustedError(
        f"All {max_retries + 1} attempt(s) failed.",
        provider=provider,
        last_exception=last_exception,
        attempts=max_retries + 1,
    )


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    should_retry: Optional[Callable[[BaseException], bool]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    provider: Optional[str] = None,
) -> T:
    """Asynchronous counterpart to :func:`retry_sync`."""
    last_exception: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retry_on as exc:  # type: ignore[misc]
            last_exception = exc
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= max_retries:
                break
            delay = compute_backoff_delay(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
    raise RetryExhaustedError(
        f"All {max_retries + 1} attempt(s) failed.",
        provider=provider,
        last_exception=last_exception,
        attempts=max_retries + 1,
    )


def detect_mime_type(filename: str) -> str:
    """Guess a MIME type from a filename extension.

    Falls back to ``application/octet-stream`` for unknown extensions.
    """
    import mimetypes

    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


def encode_file_to_base64(path: str) -> str:
    """Read a file from disk and return its Base64-encoded contents."""
    import base64

    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def redact(value: str, *, keep: int = 4) -> str:
    """Redact a secret string for safe logging, keeping a short prefix."""
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep)


def merge_headers(*header_dicts: Optional[dict]) -> dict:
    """Merge multiple header dictionaries, later ones taking precedence."""
    merged: dict = {}
    for headers in header_dicts:
        if headers:
            merged.update(headers)
    return merged


def is_rate_limit_status(status_code: Optional[int]) -> bool:
    """Return True if the given HTTP status code indicates rate limiting."""
    from .constants import RATE_LIMIT_STATUS_CODE

    return status_code == RATE_LIMIT_STATUS_CODE


def is_retryable_status(status_code: Optional[int]) -> bool:
    """Return True if the given HTTP status code is considered transient."""
    from .constants import RETRYABLE_STATUS_CODES

    return status_code is not None and status_code in RETRYABLE_STATUS_CODES


def any_value(*values: Any) -> Any:
    """Return the first non-``None`` value among the arguments."""
    for value in values:
        if value is not None:
            return value
    return None
